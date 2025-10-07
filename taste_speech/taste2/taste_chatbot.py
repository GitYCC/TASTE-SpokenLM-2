import torch
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

    def tokenize(self, text_token, text_token_len, audio_feature, audio_feature_len):
        """
        Streaming tokenize function that yields individual (text_token, taste_emb) pairs

        Args:
            text_token: Text token tensor
            text_token_len: Text token length tensor
            audio_feature: Audio feature tensor
            audio_feature_len: Audio feature length tensor

        Yields:
            tuple: (text_token, taste_emb) pairs for each token position
        """
        print("Starting streaming tokenization...")

        # Get the taste tokenizer from stage1
        taste_tokenizer = self.taste_stage1.taste_tokenizer

        # Tokenize using the TASTE tokenizer to get full embeddings
        tokenized = taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
        taste_token_emb = tokenized['taste_token_emb']  # Shape: [1, seq_len, emb_dim]

        print(f"TASTE token embedding shape: {taste_token_emb.shape}")
        seq_len = text_token.shape[1]

        # Stream individual token pairs
        for i in range(seq_len):
            # Extract individual token and embedding
            single_text_token = text_token[:, i:i+1]  # [1, 1]
            single_taste_emb = taste_token_emb[:, i:i+1, :]  # [1, 1, emb_dim]

            print(f"Yielding token pair {i+1}/{seq_len}: text_token={single_text_token.item()}, taste_emb_shape={single_taste_emb.shape}")

            yield (single_text_token, single_taste_emb)

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
        print(f"Starting streaming SLM generation (stage {self.stage})...")

        if self.stage not in [2, 'sft']:
            raise ValueError(f"Generate function requires stage 2 or sft, but got stage {self.stage}")

        # Prepare data for SLM inference
        text_token = session_buffer["text_tokens"]
        text_token_len = len(session_buffer["text_tokens"])
        taste_token_emb = ["taste_tokens"]
        
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
        for output in slm_output_generator:
            output_count += 1
            print(f"Yielding SLM output #{output_count}: {type(output)}")
            yield output

    def detokenize(self, slm_generator, audio_16k, sampling=25, stream_audio=True):
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

        if stream_audio:
            # Stream audio chunks as s3 tokens come in (chunk-by-chunk)
            print("Starting streaming audio generation...")
            s3_tokens_buffer = []
            chunk_size = 20  # Process every N s3 tokens

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

        else:
            # Collect all s3 tokens first, then convert to audio
            print("Collecting all s3 tokens first...")
            all_s3_tokens = list(s3_token_generator)

            if all_s3_tokens:
                s3_tokens_tensor = torch.tensor(all_s3_tokens, dtype=torch.long).unsqueeze(0)
                print(f"Converting all {len(all_s3_tokens)} s3 tokens to audio...")

                # Use TTS to convert all s3 tokens to audio
                for audio_chunk in self.model.tts(
                    source_speech_token=s3_tokens_tensor,
                    flow_embedding=speaker_embedding
                ):
                    print(f"Yielding complete audio: {audio_chunk['tts_speech'].shape}")
                    yield audio_chunk['tts_speech']