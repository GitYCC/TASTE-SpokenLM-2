import torch
import torchaudio

from taste_speech.taste2.taste2_interface import TASTE2Model


class TASTE2Chatbot:
    """Focused TASTE2 Chatbot implementation using centralized TASTE2Model"""

    def __init__(self, model_dir, stage='sft', fp16=False):
        # Simply instantiate the complete TASTE2Model
        self.model = TASTE2Model(model_dir, stage, fp16=fp16)

        # Expose references needed for chatbot functionality
        self.device = self.model.device
        self.taste_stage1 = self.model.taste_stage1
        self.stage = self.model.stage
        self.frontend = self.model.frontend  # CRITICAL: Needed for detokenize()
        self.audio_extractor = self.model.audio_extractor  # Needed for preprocessing
        self.sample_rate = self.model.sample_rate  # Needed for audio processing

    def taste_tokenize(self, text_token, text_token_len, audio, sample_rate=16000):
        """
        Tokenize function that converts audio to features and generates TASTE embeddings

        Args:
            text_token: Text token tensor
            text_token_len: Text token length tensor
            audio: Audio tensor to be processed

        Yields:
            tuple: (text_token, taste_token_emb) - full tensors (not streaming)
        """

        # Extract audio features using audio_extractor
        # Convert to mono and resample to 16kHz
        if audio.dim() > 1:
            audio_16k = audio.mean(dim=0, keepdim=True)
        else:
            audio_16k = audio.unsqueeze(0)

        if sample_rate != 16000:
            audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio_16k)

        audio_feature, audio_feature_len = self.model.audio_extractor(audio_16k, [audio_16k.shape[-1]])

        # Convert to half precision if model is using fp16
        if self.model.fp16:
            audio_feature = audio_feature.half()

        # Move to correct device
        audio_feature = audio_feature.to(self.model.device)
        print(f"Extracted audio feature shape: {audio_feature.shape}, length: {audio_feature_len}")

        # Get the taste tokenizer from stage1
        taste_tokenizer = self.taste_stage1.taste_tokenizer

        # Tokenize using the TASTE tokenizer
        # Only use autocast if model is in fp16 mode
        if self.model.fp16:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                tokenized = taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
                taste_token = tokenized['quantized_indices']
                taste_token_emb = tokenized['taste_token_emb']  # Shape: [1, seq_len, emb_dim]
        else:
            # Use float32 (no autocast)
            tokenized = taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
            taste_token = tokenized['quantized_indices']
            taste_token_emb = tokenized['taste_token_emb']  # Shape: [1, seq_len, emb_dim]

        print(f"TASTE token embedding shape: {taste_token_emb.shape}")

        # Yield the full text_token and taste_token_emb (no streaming)
        return (text_token, taste_token, taste_token_emb)

    def streaming_generate(self, input_buffer, min_len=3, max_len=20, sampling=25):
        """
        Streaming generate function using SLM that yields (text_token, taste_emb) pairs

        Args:
            text_token: Text token tensor
            text_token_len: Text token length tensor
            taste_token_emb: TASTE token embedding tensor
            min_len: Minimum generation length
            max_len: Maximum generation length
            sampling: Sampling parameter

        Yields:
            tuple: (text_token, taste_emb) pairs from SLM generation
        """


        # Prepare data for SLM inference
        text_token = input_buffer["text_tokens"]
        text_token_len = len(input_buffer["text_tokens"])
        taste_token_emb = input_buffer["taste_tokens"]
        
        data = {
            'text_token': text_token.to(self.device),
            'text_token_len': text_token_len.to(self.device),
            'taste_token_emb': taste_token_emb.to(self.device),
        }

        # Use SLM inference to generate and yield each output
        slm_output_generator = self.model.slm.inference(
            **data,
            min_len=min_len,
            max_len=max_len,
            sampling=sampling,
        )

        print("Starting to yield SLM outputs...")
        output_count = 0
        found_newline = False
        detected_pattern = False
        skip_count = 0
        last_token_was_newline = False

        for output in slm_output_generator:
            output_count += 1

            # Get the token ID from output
            token_id = output['text_token'].item()
            token_str = self.model.tokenizer.decode([token_id])

            # First check: detect newline
            if output_count == 1:
                if token_str == "\n":
                    found_newline = True
                    print("Detected \\n as first token")
                else:
                    # First token is not \n, yield None
                    print(f"First token not \\n: {token_str}, yielding None")

            # Second check: only if newline was found
            elif output_count == 2 and found_newline:
                if token_str == "<|im_end|>":
                    detected_pattern = True
                    skip_count += 5  # Skip next 5 tokens: \n, <|im_end|>, <|im_start|>, role, \n
                    print("Detected consecutive \\n followed by <|im_end|>, will skip next 5 tokens")
                else:
                    # Second token is not <|im_end|>, yield None
                    detected_pattern = False
                    print(f"Second token not <|im_end|>: {token_str}, yielding None")

            if detected_pattern:
                # Skip the pattern tokens
                if skip_count > 0:
                    skip_count -= 1
                    continue

                # Continuously check for \n + <|im_end|> pattern during generation
                if last_token_was_newline and token_str == "<|im_end|>":
                    print("Detected \\n + <|im_end|> during generation, stopping")
                    yield False, False
                    return

                # Track if current token is newline for next iteration
                last_token_was_newline = (token_str == "\n")

                # After skipping, yield normal outputs
                yield output
            elif found_newline:
                skip_count -= 1
                yield True,True
                continue # Still waiting for the second token check
            else:
                yield None, None
                print("Pattern not detected, stopping generation")
                return
            
        

    def taste_detokenize(self, slm_generator, audio_16k, sampling=25):
        """
        Streaming detokenize function that converts SLM output to ready-to-play audio

        Args:
            slm_generator: SLM output generator
            audio_16k: Reference audio tensor for speaker embedding extraction
            sampling: Sampling parameter for bistream inference
            stream_audio: If True, yields audio chunks as they're generated; if False, yields all tokens then final audio

        Yields:
            torch.Tensor: Audio chunks ready to play
        """
        
        '''
        BELOW CODE is just a copy paste version from /mnt/shared/p01/wilz/TASTE-SpokenLM-2/scripts/generate_audio_stream.py sft straming function
        The objective is to use neew input (text,taste) token pairs to generate audio
        TODO: define what is slm_generator in the chatbot repo, audio_16k is also misssing
        slm_generator is a thind proveide (text,taste) token pairs for taste_stage1.inference_bistream
        '''
        print("Starting streaming detokenization to audio using bistream inference + TTS...")

        # Extract speaker embedding from reference audio
        speaker_embedding = self.model.frontend._extract_spk_embedding(audio_16k)
        print(f"Extracted speaker embedding shape: {speaker_embedding.shape}")

        # Prepare prompt (empty for now, can be extended)
        prompt_text = torch.zeros(1, 0, dtype=torch.int32).to(self.device)
        prompt_text_len = torch.tensor([0], dtype=torch.int32).to(self.device)

        # Detect model dtype from model parameters
        model_dtype = next(self.taste_stage1.parameters()).dtype
        print(f"Model dtype detected: {model_dtype}")

        # Ensure prompt_speech_feature has the same dtype as the model
        prompt_speech_feature = torch.zeros(1, 0, 80, dtype=model_dtype).to(self.device)
        prompt_speech_feature_len = torch.tensor([0], dtype=torch.int32).to(self.device)

        # Get s3 token generator from bistream inference
        s3_token_generator = self.taste_stage1.inference_bistream(
            input_generator=slm_generator,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_feature=prompt_speech_feature,
            prompt_speech_feature_len=prompt_speech_feature_len,
            sampling=sampling,
        )

        # Stream audio chunks as s3 tokens come in (chunk-by-chunk)
        print("Starting streaming audio generation...")
        s3_tokens_buffer = []
        chunk_size = 20  # Process every N s3 tokens 

        ''' 
        TODO: In below logic is uisg s3 token genreator to generate s3 token and accumalate s3 token chunk_size = 20 then generate audio
        But we need to yield something out to the chat bot so the chat bot can keep consume new (text,taste) token pairs.
        There's a edge case the s3 token in rabbit mq are all be consumee but left chunk is < 20, the remain s3 need to be generate to audio too 
        So how to detect the edge case and how to solve the edge case is needed
        '''
        for s3_token in s3_token_generator:
            s3_tokens_buffer.append(s3_token)
            print(f"Collected s3 token: {s3_token} (buffer size: {len(s3_tokens_buffer)})")

            # Convert tokens to audio when buffer reaches chunk_size
            if len(s3_tokens_buffer) >= chunk_size:
                s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
                print(f"Converting {len(s3_tokens_buffer)} s3 tokens to audio...")

                # Use TTS to convert s3 tokens to audio
                for audio_chunk in self.model.tts(
                    source_speech_token=s3_tokens_tensor,
                    flow_embedding=speaker_embedding
                ):
                    print(f"Yielding audio chunk: {audio_chunk['tts_speech'].shape}")
                    yield audio_chunk['tts_speech']

                s3_tokens_buffer.clear()

        # Process remaining tokens
        if s3_tokens_buffer:
            s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
            print(f"Converting final {len(s3_tokens_buffer)} s3 tokens to audio...")

            for audio_chunk in self.model.tts(
                source_speech_token=s3_tokens_tensor,
                flow_embedding=speaker_embedding
            ):
                print(f"Yielding final audio chunk: {audio_chunk['tts_speech'].shape}")
                yield audio_chunk['tts_speech']
