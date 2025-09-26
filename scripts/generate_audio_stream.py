import os
import sys
import argparse
import threading
import uuid
import json
from contextlib import nullcontext
from hyperpyyaml import load_hyperpyyaml

from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
import torch
import torchaudio
import numpy as np
from torch.nn import functional as F

from taste_speech.taste2.cosyvoice.cli.frontend import CosyVoiceFrontEnd
from taste_speech.taste2.cosyvoice.utils.file_utils import logging
from taste_speech.taste2.cosyvoice.utils.common import fade_in_out

def print_green(text):
    """Print text in green color"""
    print(f"\033[92m{text}\033[0m")

def load_audio_data(test_file, output_dir):
    """Unified function to load audio from either arrow file or audio file"""
    assert os.path.exists(test_file)
    
    if test_file.endswith('.arrow'):
        from datasets import Dataset
        print("Loading dataset sample...")
        dataset = Dataset.from_file(test_file)
        sample = dataset[0]
        
        audio_array = sample['mp3']['array']
        sr = sample['mp3']['sampling_rate']
        audio_16k = torch.tensor(audio_array, dtype=torch.float32)
        ground_truth_text = sample['json']['text']
        audio_basename = "audio"
    else:
        print(f"Loading audio file: {test_file}")
        audio, sr = torchaudio.load(test_file)
        audio_16k = audio
        ground_truth_text = None
        audio_basename = os.path.basename(test_file).split('.')[0]
    
    # Convert to mono and resample to 16kHz
    if audio_16k.dim() > 1:
        audio_16k = audio_16k.mean(dim=0, keepdim=True)
    else:
        audio_16k = audio_16k.unsqueeze(0)
    
    if sr != 16000:
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_16k)
    
    print(f"Audio shape: {audio_16k.shape}, Duration: {audio_16k.shape[-1] / 16000:.2f}s")
    
    # Save original audio
    ext = ".mp3" if test_file.endswith('.arrow') else ".wav"
    original_path = os.path.join(output_dir, f"{audio_basename}_original{ext}")
    torchaudio.save(original_path, audio_16k, 16000)
    print(f"Saved original audio to: {original_path}")
    
    return audio_16k, ground_truth_text, audio_basename

class TASTE2Model:
    """Unified TASTE2 Model for both Stage 1 and Stage 2 audio reconstruction"""
    
    def __init__(self, llm, flow, hift, slm=None, fp16=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm, self.flow, self.hift = llm, flow, hift
        self.slm = slm  # Optional SLM for Stage 2
        self.fp16 = fp16
        
        if fp16:
            self.llm.half()
            self.flow.half()
        
        # Cache and streaming parameters
        self.token_hop_len = 25
        self.mel_cache_len = 8
        self.source_cache_len = self.mel_cache_len * 480
        self.speech_window = np.hamming(2 * self.source_cache_len)
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        self.session_data = {}
    
    def load(self, llm_model, flow_model, hift_model, slm_model=None):
        """Load model weights"""
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()
        if slm_model is not None:
            self.slm.load_state_dict(torch.load(slm_model, map_location=self.device), strict=True)
            self.slm.to(self.device).eval()
    
    def _run_inference_job(self, source_token, session_id, **kwargs):
        """Unified inference job for both LLM and VC"""
        if source_token.shape[1] == 0:  # LLM job
            with self.llm_context, torch.cuda.amp.autocast(self.fp16 and not hasattr(self.llm, 'vllm')):
                for token in self.llm.inference(**{k: v.to(self.device) if hasattr(v, 'to') else v for k, v in kwargs.items()}, uuid=session_id):
                    self.session_data[session_id]['tokens'].append(token)
        else:  # VC job
            self.session_data[session_id]['tokens'] = source_token.flatten().tolist()
        self.session_data[session_id]['finished'] = True
    
    def token2wav(self, token, prompt_token, prompt_feat, embedding, session_id, token_offset=0, finalize=True, speed=1.0):
        """Convert tokens to waveform"""
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(
                token=token.to(self.device),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_token=prompt_token.to(self.device),
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_feat=prompt_feat.to(self.device),
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                embedding=embedding.to(self.device),
                streaming=False,
                finalize=finalize
            )
        
        if speed != 1.0 and finalize:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
        
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0))
        return tts_speech
    
    def tts(self, source_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), speed=1.0, **kwargs):
        """Simplified TTS generation for reconstruction"""
        session_id = str(uuid.uuid1())
        
        with self.lock:
            self.session_data[session_id] = {'tokens': [], 'finished': False}
        
        # Run inference job
        job_kwargs = {k: v for k, v in kwargs.items() if k in ['text', 'prompt_text', 'llm_prompt_speech_token', 'llm_embedding']}
        thread = threading.Thread(target=self._run_inference_job, args=(source_speech_token, session_id), kwargs=job_kwargs)
        thread.start()
        thread.join()
        
        # Convert tokens to audio
        tokens = torch.tensor(self.session_data[session_id]['tokens']).unsqueeze(0)
        speech = self.token2wav(
            token=tokens,
            prompt_token=kwargs.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
            prompt_feat=kwargs.get('prompt_speech_feat', torch.zeros(1, 0, 80)),
            embedding=flow_embedding,
            session_id=session_id,
            speed=speed
        )
        
        # Cleanup
        with self.lock:
            self.session_data.pop(session_id)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield {'tts_speech': speech.cpu()}

class TASTE2:
    """Unified TASTE2 class for both Stage 1 and Stage 2 audio reconstruction"""
    
    def __init__(self, model_dir, stage=1, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_dir = model_dir
        self.fp16 = fp16
        self.stage = stage

        # Download model if needed
        if not os.path.exists(model_dir):
            if stage == 1:
                model_dir = snapshot_download(model_dir)
            elif stage == 'sft':
                model_dir = hf_snapshot_download(model_dir)
            else:
                model_dir = hf_snapshot_download(model_dir)

        # Load appropriate config file
        if stage == 'sft':
            config_file = 'taste2_stagesft.yaml'
        else:
            config_file = f'taste2_stage{stage}.yaml'
        hyper_yaml_path = os.path.join(model_dir, config_file)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError(f'{hyper_yaml_path} not found!')
        
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN'),
                    'taste_tokenizer_backbond_path': os.path.join(model_dir, 'distil-whisper'),
                    'qwen_pretrain_path_for_slm': os.path.join(model_dir, 'qwen2-1_5b'),
                    'qwen_pretrain_path_for_slm_7b': os.path.join(model_dir, 'qwen2-7b'),
                }
            )

        # Initialize frontend and extractors
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            os.path.join(model_dir, 'campplus.onnx'),
            os.path.join(model_dir, 'speech_tokenizer_v2.onnx'),
            os.path.join(model_dir, 'spk2info.pt'),
            configs['allowed_special']
        )
        self.audio_extractor = configs['audio_extractor']
        self.sample_rate = configs['sample_rate']
        
        # Handle CUDA availability
        if not torch.cuda.is_available() and (load_jit or load_trt or fp16):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        
        # Initialize model based on stage
        slm = configs.get('slm') if stage in [2, 'sft'] else None
        self.model = TASTE2Model(configs['llm'], configs['flow'], configs['hift'], slm, fp16)

        # Load model weights
        if stage == 'sft':
            self.model.load(
                os.path.join(model_dir, 'llm.pt'),
                os.path.join(model_dir, 'flow.pt'),
                os.path.join(model_dir, 'hift.pt'),
                os.path.join(model_dir, 'slm.pt')
            )
        else:
            self.model.load(
                os.path.join(model_dir, 'llm.pt'),
                os.path.join(model_dir, 'flow.pt'),
                os.path.join(model_dir, 'hift.pt'),
                os.path.join(model_dir, 'slm.pt') if stage == 2 else None
            )
        
        # Optional optimizations
        if load_vllm:
            self.model.load_vllm(os.path.join(model_dir, 'vllm'))
        if load_jit:
            precision = 'fp16' if fp16 else 'fp32'
            self.model.load_jit(os.path.join(model_dir, f'flow.encoder.{precision}.zip'))
        if load_trt:
            precision = 'fp16' if fp16 else 'fp32'
            self.model.load_trt(
                os.path.join(model_dir, f'flow.decoder.estimator.{precision}.mygpu.plan'),
                os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx'),
                trt_concurrent,
                fp16
            )

        self.taste_stage1 = self.model.slm.taste_stage1 if stage in [2, 'sft'] else self.model.llm
        
        del configs

    def _run_asr(self, audio_16k, asr_model_dir="openai/whisper-large-v3"):
        """Run ASR on audio to get text transcription"""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            asr_model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(asr_model_dir)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device
        )
        
        return pipe(audio_16k.squeeze().numpy())["text"]

    def _preprocess(self, audio_16k, asr_model_dir=None, text=None):
        assert asr_model_dir or text, "Either ASR model or text must be provided"

        # Get text from ASR or use provided text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        
        # Extract text and audio tokens
        text_token, text_token_len = self.frontend._extract_text_token(asr_text)
        # Use audio_extractor directly like in generate_stage1.py
        audio_feature, audio_feature_len = self.audio_extractor(audio_16k, [audio_16k.shape[-1]])
        
        # Convert to half precision if model is using fp16
        if self.fp16:
            audio_feature = audio_feature.half()
        
        # Move to correct device
        audio_feature = audio_feature.to(self.device)

        taste_tokenizer = self.taste_stage1.taste_tokenizer
        tokenized = taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
        taste_token_emb = tokenized['taste_token_emb']

        return dict(
            text_token=text_token.to(self.device),
            text_token_len=text_token_len.to(self.device),
            taste_token_emb=taste_token_emb.to(self.device),
        )

    def _postprocess(self, s3_tokens, audio_16k):
        # Convert tokens back to audio
        speaker_embedding = self.frontend._extract_spk_embedding(audio_16k)
        audio_chunks = list(self.model.tts(
            source_speech_token=s3_tokens,
            flow_embedding=speaker_embedding
        ))
        
        return torch.cat([chunk['tts_speech'] for chunk in audio_chunks], dim=1)
    
    def _postprocess_streaming(self, s3_tokens, ref_s3_tokens, audio_16k):
        speaker_embedding = self.frontend._extract_spk_embedding(audio_16k)
        for audio_chunk in self.model.tts(
            flow_prompt_speech_token=ref_s3_tokens,
            source_speech_token=s3_tokens,
            flow_embedding=speaker_embedding,
            stream=True
        ):
            yield audio_chunk['tts_speech']
    
    def _crossfade_audio_chunks(self, audio_chunks, crossfade_samples=400):
        """
        Crossfade audio chunks to eliminate popping sounds at boundaries.
        This maintains the original total duration by applying smooth transitions at chunk boundaries.
        
        Args:
            audio_chunks: List of audio tensors [batch, time]
            crossfade_samples: Number of samples to crossfade (default: 400 samples ≈ 25ms at 16kHz)
        
        Returns:
            Combined audio tensor with smooth transitions and same total length as simple concatenation
        """
        if len(audio_chunks) <= 1:
            return torch.cat(audio_chunks, dim=-1) if audio_chunks else torch.zeros(1, 0)
        
        print(f"Crossfading {len(audio_chunks)} chunks with {crossfade_samples} samples ({crossfade_samples/self.sample_rate*1000:.1f}ms)")
        
        # Calculate expected total length (simple concatenation - no reduction)
        expected_length = sum(chunk.shape[-1] for chunk in audio_chunks)
        print(f"Expected final length: {expected_length} samples ({expected_length/self.sample_rate:.2f}s)")
        
        # Start with the first chunk
        combined_audio = audio_chunks[0].clone()
        
        for i in range(1, len(audio_chunks)):
            current_chunk = audio_chunks[i]
            
            # Ensure we don't crossfade more samples than available
            actual_crossfade = min(crossfade_samples, combined_audio.shape[-1], current_chunk.shape[-1])
            
            if actual_crossfade > 0:
                # Create fade curves
                fade_out = torch.linspace(1.0, 0.0, actual_crossfade).to(combined_audio.device)
                fade_in = torch.linspace(0.0, 1.0, actual_crossfade).to(current_chunk.device)
                
                # Apply crossfade only to the boundary region without changing total length
                # Modify the end of combined_audio and beginning of current_chunk
                combined_audio[..., -actual_crossfade:] = (
                    combined_audio[..., -actual_crossfade:] * fade_out + 
                    current_chunk[..., :actual_crossfade] * fade_in
                )
                
                # Concatenate the crossfaded combined_audio with the remainder of current_chunk
                combined_audio = torch.cat([
                    combined_audio,                          # Combined audio with crossfaded end
                    current_chunk[..., actual_crossfade:]    # Remainder of current chunk
                ], dim=-1)
            else:
                # If no overlap possible, just concatenate
                combined_audio = torch.cat([combined_audio, current_chunk], dim=-1)
        
        print(f"Final combined audio length: {combined_audio.shape[-1]} samples ({combined_audio.shape[-1]/self.sample_rate:.2f}s)")
        
        return combined_audio

        
    @torch.inference_mode()
    def generation_stage1_streaming(self, audio_16k, asr_model_dir=None, text=None):
        """
        Streaming generation for Stage 1 using bistream inference
        
        Args:
            audio_16k: Input audio tensor
            asr_model_dir: ASR model directory for text transcription
            text: Optional text (if provided, ASR will be skipped)
            
        Yields:
            Dict containing audio chunks and metadata
        """
        if self.stage != 1:
            raise ValueError("generation_stage1_streaming can only be called on stage 1 model")
        
        print("=== Starting Stage 1 Streaming Generation ===")
        
        # 1. Retrieve text tokens & taste embeddings using _preprocess()
        print("Preprocessing audio and extracting features...")
        data = self._preprocess(audio_16k, asr_model_dir=asr_model_dir, text=text)
        
        # Get ASR text for logging
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        print(f"ASR Text: {asr_text}")
        
        # 2. Pack every (text token, taste emb) pair and construct a generator
        def create_streaming_input_generator():
            """Create generator of individual (text_token, taste_emb) pairs"""
            text_tokens = data['text_token']  # Shape: [1, seq_len]
            taste_embs = data['taste_token_emb']  # Shape: [1, seq_len, emb_dim]
            
            seq_len = text_tokens.shape[1]
            print(f"Total sequence length: {seq_len} tokens")
            
            # Yield individual token pairs
            for i in range(seq_len):
                # Extract individual token and embedding
                text_token = text_tokens[:, i]  # [1, 1]
                taste_emb = taste_embs[:, i:i+1, :]  # [1, 1, emb_dim]
                
                # Ensure consistent dtype for taste_emb to match model dtype
                model_dtype = next(self.taste_stage1.parameters()).dtype
                if taste_emb.dtype != model_dtype:
                    taste_emb = taste_emb.to(model_dtype)
                
                print(f"Yielding token pair {i+1}/{seq_len}: text_token={text_token.item()}, taste_emb_shape={taste_emb.shape}, taste_emb_dtype={taste_emb.dtype}")
                
                yield text_token, taste_emb
        
        # 3. Call self.taste_stage1.inference_bistream()
        print("\nStarting bistream inference...")
        
        # Debug: Check model parameter dtypes
        print("Checking model parameter dtypes...")
        for name, param in self.taste_stage1.named_parameters():
            print(f"  {name}: {param.dtype}")
            break  # Just show first few parameters
        
        # Prepare prompt (empty for now, can be extended)
        prompt_text = torch.zeros(1, 0, dtype=torch.int32).to(self.device)
        prompt_text_len = torch.tensor([0], dtype=torch.int32).to(self.device)  
        
        # Detect model dtype from model parameters
        model_dtype = next(self.taste_stage1.parameters()).dtype
        print(f"Model dtype detected: {model_dtype}")
        
        # Ensure prompt_speech_feature has the same dtype as the model
        prompt_speech_feature = torch.zeros(1, 0, 80, dtype=model_dtype).to(self.device)
        prompt_speech_feature_len = torch.tensor([0], dtype=torch.int32).to(self.device)
        
        # Create input generator
        input_generator = create_streaming_input_generator()
        
        # Get s3 token generator from bistream inference
        s3_token_generator = self.taste_stage1.inference_bistream(
            input_generator=input_generator,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_feature=prompt_speech_feature,
            prompt_speech_feature_len=prompt_speech_feature_len,
            sampling=25,
        )
        
        # 4. Collect ALL s3 tokens first, then generate overlapping audio sequences
        print("\nCollecting all s3 tokens first...")
        
        all_s3_tokens = []
        s3_tokens_buffer = []
        audio_chunks = []
        chunk_id = 0
        chunk_size = 20  # s3 token window length, cannot exceed 50
        
        # chunk_size / 25 * 24000
        window_size = int(chunk_size / self.model.flow.input_frame_rate * self.sample_rate)
        import time
        start_time = time.time()
        # Collect all s3 tokens
        for s3_token in s3_token_generator:
            all_s3_tokens.append(s3_token)
            s3_tokens_buffer.append(s3_token)
            print(f"Generated s3 token: {s3_token}")
            
            # Process every 5 s3 tokens
            if len(s3_tokens_buffer) >= chunk_size:
                chunk_id += 1
                print(f"\n--- Processing audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens) ---")
                
                # Convert tokens to tensor (all current s3 tokens)
                s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
                ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)],  dtype=torch.long).unsqueeze(0)
                print(f"S3 tokens shape: {s3_tokens_tensor.shape}")
                print(f"Ref S3 tokens shape: {ref_tokens_tensor.shape}")
                
                # Generate audio chunk
                try:
                    # Maybe different _postprocess_streaming, which takes prev s3 tokens as reference
                    for audio_chunk in self._postprocess_streaming(
                        s3_tokens=s3_tokens_tensor,
                        ref_s3_tokens=ref_tokens_tensor,
                        audio_16k=audio_16k
                    ):
                        # audio_chunk = self._postprocess(all_s3_tokens, audio_16k)
                        print(f"Generated audio shape: {audio_chunk.shape}")
                        
                        new_audio_chunk = audio_chunk[:, -window_size:]
                        print(f"Truncated audio shape: {new_audio_chunk.shape}")
                        audio_chunks.append(new_audio_chunk)
                        
                        # Yield chunk result
                        yield {
                            'audio_chunk': new_audio_chunk,
                            'chunk_id': chunk_id,
                            's3_tokens': s3_tokens_buffer,
                            'asr_text': asr_text,
                            'is_final': False
                        }
                        
                    print(f"Audio chunk {chunk_id} generated successfully")
                    print(f"Time: {time.time() - start_time} seconds")
                    
                except Exception as e:
                    print(f"Error generating audio chunk {chunk_id}: {e}")
                    continue
                
                # Clear current s3 token buffer
                s3_tokens_buffer.clear()
                start_time = time.time()
        
        # Process remaining tokens as final chunk with overlap
        if s3_tokens_buffer:
            chunk_id += 1
            print(f"\n--- Processing final audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens) ---")
            
            s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
            ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)],  dtype=torch.long).unsqueeze(0)
            print(f"Final S3 tokens shape: {s3_tokens_tensor.shape}")
            print(f"Ref S3 tokens shape: {ref_tokens_tensor.shape}")
            
            try:
                # audio_chunk = self._postprocess(s3_tokens_tensor, audio_16k)
                for audio_chunk in self._postprocess_streaming(
                    s3_tokens=s3_tokens_tensor,
                    ref_s3_tokens=ref_tokens_tensor,
                    audio_16k=audio_16k
                ):
                    final_window_size = int(window_size / chunk_size * len(s3_tokens_buffer))
                    new_audio_chunk = audio_chunk[:, -final_window_size:]
                    audio_chunks.append(new_audio_chunk)
                    # Yield final chunk
                    yield {
                        'audio_chunk': new_audio_chunk,
                        'chunk_id': chunk_id,
                        's3_tokens': s3_tokens_buffer,
                        'asr_text': asr_text,
                        'is_final': True
                    }
                
                print(f"Final audio chunk {chunk_id} generated successfully")
                
            except Exception as e:
                print(f"Error generating final audio chunk: {e}")
        
        # Generate audio from ALL s3 tokens at once for comparison
        if all_s3_tokens:
            print(f"\n=== Generating audio from ALL {len(all_s3_tokens)} s3 tokens at once ===")
            all_s3_tokens_tensor = torch.tensor(all_s3_tokens, dtype=torch.long).unsqueeze(0)
            print(f"All S3 tokens shape: {all_s3_tokens_tensor.shape}")
            print(f"All S3 tokens: {all_s3_tokens}")
            
            try:
                single_audio = self._postprocess(all_s3_tokens_tensor, audio_16k)
                print(f"Single audio from all tokens generated successfully (shape: {single_audio.shape})")
                print(f"Single audio duration: {single_audio.shape[-1] / self.sample_rate:.2f}s")
                
                # Yield the single audio result
                yield {
                    'single_audio': single_audio,
                    'all_s3_tokens': all_s3_tokens,
                    'total_tokens': len(all_s3_tokens),
                    'asr_text': asr_text,
                    'is_single_audio': True
                }
                
            except Exception as e:
                print(f"Error generating single audio from all tokens: {e}")
                single_audio = None
        else:
            single_audio = None
        
        # Combine all audio chunks with crossfading for smooth transitions
        if audio_chunks:
            print(f"\n=== Combining {len(audio_chunks)} audio chunks with crossfading ===")
            final_audio = self._crossfade_audio_chunks(audio_chunks, crossfade_samples=int(0.01 * self.sample_rate))  # 10ms crossfade
            
            print(f"Final audio shape: {final_audio.shape}")
            print(f"Final audio duration: {final_audio.shape[-1] / self.sample_rate:.2f}s")
            
            yield {
                'final_audio': final_audio,
                'total_chunks': len(audio_chunks),
                'asr_text': asr_text,
                'is_complete': True
            }
        else:
            print("Warning: No audio chunks were generated")

    
    @torch.inference_mode()
    def generation_stage2_streaming(self, audio_16k, asr_model_dir=None, text=None):
        """
        Streaming generation for Stage 2 using bistream inference
        
        Args:
            audio_16k: Input audio tensor
            asr_model_dir: ASR model directory for text transcription
            text: Optional text (if provided, ASR will be skipped)
            
        Yields:
            Dict containing audio chunks and metadata
        """
        if self.stage != 2:
            raise ValueError("generation_stage2_streaming can only be called on stage 2 model")
        
        print("=== Starting Stage 2 Streaming Generation ===")
        
        # 1. Retrieve text tokens & taste embeddings using _preprocess()
        print("Preprocessing audio and extracting features...")
        data = self._preprocess(audio_16k, asr_model_dir=asr_model_dir, text=text)
        
        # Get ASR text for logging
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        print(f"ASR Text: {asr_text}")
        
        # 2. Create SLM output generator (this yields (text_token, taste_emb) pairs)
        print("Starting SLM inference to generate streaming text and taste embeddings...")
        slm_output_generator = self.model.slm.inference(
            **data,
            min_len=3,
            max_len=20,
            sampling=25,
        )
        
        # 3. Call self.taste_stage1.inference_bistream() using SLM generator directly
        print("\nStarting bistream inference with SLM output generator...")
        
        # Debug: Check model parameter dtypes
        print("Checking model parameter dtypes...")
        for name, param in self.taste_stage1.named_parameters():
            print(f"  {name}: {param.dtype}")
            break  # Just show first few parameters
        
        # Prepare prompt (empty for now, can be extended)
        prompt_text = torch.zeros(1, 0, dtype=torch.int32).to(self.device)
        prompt_text_len = torch.tensor([0], dtype=torch.int32).to(self.device)  
        
        # Detect model dtype from model parameters
        model_dtype = next(self.taste_stage1.parameters()).dtype
        print(f"Model dtype detected: {model_dtype}")
        
        # Ensure prompt_speech_feature has the same dtype as the model
        prompt_speech_feature = torch.zeros(1, 0, 80, dtype=model_dtype).to(self.device)
        prompt_speech_feature_len = torch.tensor([0], dtype=torch.int32).to(self.device)
        
        # Get s3 token generator from bistream inference using SLM generator directly
        s3_token_generator = self.taste_stage1.inference_bistream(
            input_generator=slm_output_generator,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_feature=prompt_speech_feature,
            prompt_speech_feature_len=prompt_speech_feature_len,
            sampling=25,
        )
        
        # 4. Collect ALL s3 tokens first, then generate overlapping audio sequences
        print("\nCollecting all s3 tokens first...")
        
        all_s3_tokens = []
        s3_tokens_buffer = []
        audio_chunks = []
        chunk_id = 0
        chunk_size = 20  # s3 token window length, cannot exceed 50
        
        # chunk_size / 25 * 24000
        window_size = int(chunk_size / self.model.flow.input_frame_rate * self.sample_rate)
        import time
        start_time = time.time()
        # Collect all s3 tokens
        for s3_token in s3_token_generator:
            all_s3_tokens.append(s3_token)
            s3_tokens_buffer.append(s3_token)
            print(f"Generated s3 token: {s3_token}")
            
            # Process every 5 s3 tokens
            if len(s3_tokens_buffer) >= chunk_size:
                chunk_id += 1
                print(f"\n--- Processing audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens) ---")
                
                # Convert tokens to tensor (all current s3 tokens)
                s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
                ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)],  dtype=torch.long).unsqueeze(0)
                print(f"S3 tokens shape: {s3_tokens_tensor.shape}")
                print(f"Ref S3 tokens shape: {ref_tokens_tensor.shape}")
                
                # Generate audio chunk
                try:
                    # Maybe different _postprocess_streaming, which takes prev s3 tokens as reference
                    for audio_chunk in self._postprocess_streaming(
                        s3_tokens=s3_tokens_tensor,
                        ref_s3_tokens=ref_tokens_tensor,
                        audio_16k=audio_16k
                    ):
                        # audio_chunk = self._postprocess(all_s3_tokens, audio_16k)
                        print(f"Generated audio shape: {audio_chunk.shape}")
                        
                        new_audio_chunk = audio_chunk[:, -window_size:]
                        print(f"Truncated audio shape: {new_audio_chunk.shape}")
                        audio_chunks.append(new_audio_chunk)
                        
                        # Yield chunk result
                        yield {
                            'audio_chunk': new_audio_chunk,
                            'chunk_id': chunk_id,
                            's3_tokens': s3_tokens_buffer,
                            'asr_text': asr_text,
                            'is_final': False
                        }
                        
                    print(f"Audio chunk {chunk_id} generated successfully")
                    print(f"Time: {time.time() - start_time} seconds")
                    
                except Exception as e:
                    print(f"Error generating audio chunk {chunk_id}: {e}")
                    continue
                
                # Clear current s3 token buffer
                s3_tokens_buffer.clear()
                start_time = time.time()
        
        # Process remaining tokens as final chunk with overlap
        if s3_tokens_buffer:
            chunk_id += 1
            print(f"\n--- Processing final audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens) ---")
            
            s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
            ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)],  dtype=torch.long).unsqueeze(0)
            print(f"Final S3 tokens shape: {s3_tokens_tensor.shape}")
            print(f"Ref S3 tokens shape: {ref_tokens_tensor.shape}")
            
            try:
                # audio_chunk = self._postprocess(s3_tokens_tensor, audio_16k)
                for audio_chunk in self._postprocess_streaming(
                    s3_tokens=s3_tokens_tensor,
                    ref_s3_tokens=ref_tokens_tensor,
                    audio_16k=audio_16k
                ):
                    final_window_size = int(window_size / chunk_size * len(s3_tokens_buffer))
                    new_audio_chunk = audio_chunk[:, -final_window_size:]
                    audio_chunks.append(new_audio_chunk)
                    # Yield final chunk
                    yield {
                        'audio_chunk': new_audio_chunk,
                        'chunk_id': chunk_id,
                        's3_tokens': s3_tokens_buffer,
                        'asr_text': asr_text,
                        'is_final': True
                    }
                
                print(f"Final audio chunk {chunk_id} generated successfully")
                
            except Exception as e:
                print(f"Error generating final audio chunk: {e}")
        
        # Generate audio from ALL s3 tokens at once for comparison
        if all_s3_tokens:
            print(f"\n=== Generating audio from ALL {len(all_s3_tokens)} s3 tokens at once ===")
            all_s3_tokens_tensor = torch.tensor(all_s3_tokens, dtype=torch.long).unsqueeze(0)
            print(f"All S3 tokens shape: {all_s3_tokens_tensor.shape}")
            print(f"All S3 tokens: {all_s3_tokens}")
            
            try:
                single_audio = self._postprocess(all_s3_tokens_tensor, audio_16k)
                print(f"Single audio from all tokens generated successfully (shape: {single_audio.shape})")
                print(f"Single audio duration: {single_audio.shape[-1] / self.sample_rate:.2f}s")
                
                # Yield the single audio result
                yield {
                    'single_audio': single_audio,
                    'all_s3_tokens': all_s3_tokens,
                    'total_tokens': len(all_s3_tokens),
                    'asr_text': asr_text,
                    'is_single_audio': True
                }
                
            except Exception as e:
                print(f"Error generating single audio from all tokens: {e}")
                single_audio = None
        else:
            single_audio = None
        
        # Combine all audio chunks with crossfading for smooth transitions
        if audio_chunks:
            print(f"\n=== Combining {len(audio_chunks)} audio chunks with crossfading ===")
            final_audio = self._crossfade_audio_chunks(audio_chunks, crossfade_samples=int(0.01 * self.sample_rate))  # 10ms crossfade
            
            print(f"Final audio shape: {final_audio.shape}")
            print(f"Final audio duration: {final_audio.shape[-1] / self.sample_rate:.2f}s")
            
            yield {
                'final_audio': final_audio,
                'total_chunks': len(audio_chunks),
                'asr_text': asr_text,
                'is_complete': True
            }
        else:
            print("Warning: No audio chunks were generated")

    def _segment_audio(self, audio_16k, segment_duration=0.5):
        """
        Segment audio into cumulative chunks that include all past audio

        Args:
            audio_16k: Input audio tensor [1, samples]
            segment_duration: Duration increment for each segment in seconds

        Returns:
            List of cumulative audio segments
            - Segment 1: 0 to 0.5s
            - Segment 2: 0 to 1.0s
            - Segment 3: 0 to 1.5s
            - etc.
        """
        segment_samples = int(segment_duration * self.sample_rate)
        audio_length = audio_16k.shape[-1]
        segments = []

        # Create cumulative segments
        current_end = segment_samples
        segment_num = 1

        while current_end <= audio_length:
            # Each segment starts from the beginning and goes to current_end
            segment = audio_16k[:, 0:current_end]
            segments.append(segment)

            print(f"Segment {segment_num}: 0 to {current_end/self.sample_rate:.1f}s (shape: {segment.shape})")

            current_end += segment_samples
            segment_num += 1

        # Add final segment if there's remaining audio
        if current_end - segment_samples < audio_length:
            final_segment = audio_16k[:, 0:audio_length]
            segments.append(final_segment)
            print(f"Final segment {segment_num}: 0 to {audio_length/self.sample_rate:.1f}s (shape: {final_segment.shape})")

        print(f"Created {len(segments)} cumulative segments with {segment_duration}s increments")
        return segments

    @torch.inference_mode()
    def generation_sft_streaming(self, audio_16k, asr_model_dir=None, text=None):
        """
        Streaming generation for SFT (Supervised Fine-Tuning) mode using bistream inference
        Now processes audio in 0.5-second segments, preprocessing each segment individually
        and sending each to SLM inference sequentially.

        Args:
            audio_16k: Input audio tensor
            asr_model_dir: ASR model directory for text transcription
            text: Optional text (if provided, ASR will be skipped)

        Yields:
            Dict containing audio chunks and metadata
        """
        if self.stage != 'sft':
            raise ValueError("generation_sft_streaming can only be called on sft model")

        print("=== Starting Stage SFT Streaming Generation with Audio Segmentation ===")

        # 1. Segment input audio into 0.5-second chunks
        print("Segmenting audio into 0.5-second chunks...")
        audio_segments = self._segment_audio(audio_16k, segment_duration=0.5)

        # Get ASR text for logging (using full audio)
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        print(f"ASR Text: {asr_text}")

        # 2. Find the segment that produces completion pattern and use its SLM generator
        def find_valid_slm_generator():
            """Find the first segment that produces \n + <|im_end|> pattern and return its SLM generator"""
            # Get special tokens from frontend tokenizer
            newline_token = self.frontend.tokenizer.encode("\n",add_special_tokens = False)
            im_end_token = self.frontend.tokenizer.encode('<|im_end|>',add_special_tokens = False)
            im_start_token = self.frontend.tokenizer.encode('<|im_start|>',add_special_tokens = False)
            user_token = self.frontend.tokenizer.encode('user',add_special_tokens = False)

            print(f"Searching for valid segment - newline: {newline_token}, <|im_end|>: {im_end_token}")

            for i, audio_segment in enumerate(audio_segments):
                print(f"\nTesting audio segment {i+1}/{len(audio_segments)}")
                print(f"Segment shape: {audio_segment.shape}, Duration: {audio_segment.shape[-1] / self.sample_rate:.2f}s")

                # Preprocess this segment individually
                print(f"Preprocessing segment {i+1}...")
                segment_data = self._preprocess(audio_segment, asr_model_dir=asr_model_dir, text=text)

                # Add chat format prefix: <|im_start|> + role(user) + \n
                print(f"Adding chat format prefix to segment {i+1}...")

                print(f"Chat tokens - <|im_start|>: {im_start_token}, user: {user_token}, \\n: {newline_token}")

                # Extract single integers from token lists and ensure all tokens are valid
                im_start_val = im_start_token[0] if isinstance(im_start_token, list) and len(im_start_token) > 0 else (im_start_token if im_start_token is not None else 151644)
                user_val = user_token[0] if isinstance(user_token, list) and len(user_token) > 0 else (user_token if user_token is not None else 872)
                newline_val = newline_token[0] if isinstance(newline_token, list) and len(newline_token) > 0 else (newline_token if newline_token is not None else 198)

                print(f"Extracted token values: im_start={im_start_val}, user={user_val}, newline={newline_val}")

                # Create prefix tokens: <|im_start|> + user + \n  (3 tokens)
                prefix_tokens = torch.tensor([[im_start_val, user_val, newline_val]], dtype=segment_data['text_token'].dtype).to(segment_data['text_token'].device)

                # Prepend prefix to text_token
                segment_data['text_token'] = torch.cat([prefix_tokens, segment_data['text_token']], dim=1)

                # Update text_token_len (+3)
                segment_data['text_token_len'] = segment_data['text_token_len'] + 3

                # Pad 3 zeros at front of taste_token_emb
                batch_size, seq_len, emb_dim = segment_data['taste_token_emb'].shape
                zero_padding = torch.zeros(batch_size, 3, emb_dim, dtype=segment_data['taste_token_emb'].dtype).to(segment_data['taste_token_emb'].device)
                segment_data['taste_token_emb'] = torch.cat([zero_padding, segment_data['taste_token_emb']], dim=1)

                print(f"Modified segment {i+1}: text_token shape: {segment_data['text_token'].shape}, taste_emb shape: {segment_data['taste_token_emb'].shape}, text_len: {segment_data['text_token_len']}")

                # Run SLM inference on this preprocessed segment
                print(f"Testing SLM inference on segment {i+1}...")
                test_slm_generator = self.model.slm.inference(
                    **segment_data,
                    min_len=3,
                    max_len=20,
                    sampling=25,
                )

                # Test only the first 2 tokens to see if this generator has completion pattern
                has_completion_pattern = False
                test_outputs = []

                for output_count, output in enumerate(test_slm_generator):
                    test_outputs.append(output)
                    # print_green(f"🎯 SLM OUTPUT #{output_count}: {output} (type: {type(output)})")

                    # Extract text_token from output (assuming it's a tuple (text_token, taste_emb))
                    if isinstance(output, tuple) and len(output) >= 2:
                        text_token, taste_emb = output[0], output[1]
                        print_green(f"   📝 Text Token: {text_token} (type: {type(text_token)})")
                        print_green(f"   🎵 Taste Embedding: shape={taste_emb.shape if hasattr(taste_emb, 'shape') else 'N/A'}")

                        # Check if text_token contains the completion pattern: \n + <|im_end|>
                        if torch.is_tensor(text_token):
                            token_ids = text_token.flatten().tolist()
                        else:
                            token_ids = [text_token] if isinstance(text_token, int) else text_token

                        print_green(f"   🔢 Token IDs: {token_ids}")

                        # Extract expected token values (handle lists)
                        expected_newline = newline_token[0] if isinstance(newline_token, list) and len(newline_token) > 0 else newline_token
                        expected_im_end = im_end_token[0] if isinstance(im_end_token, list) and len(im_end_token) > 0 else im_end_token

                        # For first token: check if it's newline
                        if output_count == 0 and len(token_ids) > 0 and token_ids[0] == expected_newline:
                            print_green(f"✓ Segment {i+1}: First token is \\n ({token_ids[0]})")
                        # For second token: check if it's <|im_end|> (completing the pattern)
                        if output_count == 1 and len(token_ids) > 0 and token_ids[0] == expected_im_end:
                            has_completion_pattern = True
                            print_green(f"✓ Segment {i+1}: Found completion pattern \\n + <|im_end|> in first 2 tokens")
                            break

                    # Only test first 2 tokens
                    if output_count >= 1:  # Stop after 2nd token (0-indexed)
                        break

                if has_completion_pattern:
                    print(f"✓ Segment {i+1}: SELECTED as the SLM generator (contains \\n + <|im_end|>)")
                    # Create a fresh SLM generator for this segment and filter to extract only TEXT
                    raw_slm_generator = self.model.slm.inference(
                        **segment_data,
                        min_len=3,
                        max_len=20,
                        sampling=25,
                    )

                    # Create filtered generator that extracts only TEXT content
                    def create_text_only_generator():
                        """Filter SLM output to extract only TEXT content"""
                        # Get special tokens
                        im_start_token = self.frontend.tokenizer.encode('<|im_start|>')[0]
                        im_end_token = self.frontend.tokenizer.encode('<|im_end|>')[0]
                        newline_token = self.frontend.tokenizer.encode("\n")[0]
                        assistant_token = self.frontend.tokenizer.encode("assistant")[0]

                        print(f"Text filtering tokens - <|im_start|>: {im_start_token}, <|im_end|>: {im_end_token}, \\n: {newline_token}, assistant: {assistant_token}")

                        # State machine to track parsing progress
                        state = "looking_for_im_end"  # \n + <|im_end|> + <|im_start|> + assistant + \n + TEXT + \n + <|im_end|>

                        # Collect all tokens for full sequence logging
                        all_raw_tokens = []
                        yielded_text_tokens = []

                        print_green("=" * 80)
                        print_green("🎯 FULL SLM GENERATOR TOKEN SEQUENCE:")
                        print_green("=" * 80)

                        for output_idx, output in enumerate(raw_slm_generator):
                            if isinstance(output, tuple) and len(output) >= 2:
                                text_token, taste_emb = output[0], output[1]

                                # Extract token ID
                                if torch.is_tensor(text_token):
                                    token_id = text_token.flatten().tolist()[0] if text_token.numel() > 0 else None
                                else:
                                    token_id = text_token if isinstance(text_token, int) else (text_token[0] if isinstance(text_token, list) and text_token else None)

                                # Log every token from SLM generator
                                all_raw_tokens.append(token_id)
                                token_name = "UNKNOWN"
                                if token_id == newline_token:
                                    token_name = "\\n"
                                elif token_id == im_start_token:
                                    token_name = "<|im_start|>"
                                elif token_id == im_end_token:
                                    token_name = "<|im_end|>"
                                elif token_id == assistant_token:
                                    token_name = "assistant"

                                print_green(f"  [{output_idx:02d}] Token ID: {token_id:5d} | Name: {token_name:15s} | State: {state:20s}")

                                # State machine parsing
                                if state == "looking_for_im_end" and token_id == im_end_token:
                                    state = "looking_for_im_start"
                                    print("Found first <|im_end|>, looking for <|im_start|>...")
                                elif state == "looking_for_im_start" and token_id == im_start_token:
                                    state = "looking_for_assistant"
                                    print("Found <|im_start|>, looking for assistant...")
                                elif state == "looking_for_assistant" and token_id == assistant_token:
                                    state = "looking_for_newline"
                                    print("Found assistant, looking for newline...")
                                elif state == "looking_for_newline" and token_id == newline_token:
                                    state = "extracting_text"
                                    print("Found newline, now extracting TEXT content...")
                                elif state == "extracting_text":
                                    # Check if this is the ending newline + <|im_end|>
                                    if token_id == newline_token:
                                        state = "check_final_im_end"
                                        print("Found potential ending newline, checking for final <|im_end|>...")
                                    elif token_id == im_end_token and state == "check_final_im_end":
                                        print("Found final <|im_end|>, TEXT extraction complete.")
                                        break
                                    else:
                                        # This is actual TEXT content - yield it
                                        yielded_text_tokens.append(token_id)
                                        print_green(f"🎯 ✅ YIELDING TEXT TOKEN: {token_id} | Full output: {output}")
                                        yield output
                                        if state == "check_final_im_end":
                                            state = "extracting_text"  # Reset if it wasn't final <|im_end|>

                        # Print final summary
                        print_green("=" * 80)
                        print_green("📊 SLM GENERATOR TOKEN SEQUENCE SUMMARY:")
                        print_green("=" * 80)
                        print_green(f"🎯 Total SLM tokens generated: {len(all_raw_tokens)}")
                        print_green(f"📝 Raw token sequence: {all_raw_tokens}")
                        print_green(f"✅ Text tokens yielded: {len(yielded_text_tokens)}")
                        print_green(f"📝 Yielded token sequence: {yielded_text_tokens}")
                        print_green(f"🔄 Filtered out: {len(all_raw_tokens) - len(yielded_text_tokens)} tokens")
                        print_green("=" * 80)

                    filtered_slm_generator = create_text_only_generator()
                    return filtered_slm_generator, i+1
                else:
                    print(f"✗ Segment {i+1}: No completion pattern found, trying next segment...")

            # If no segment has completion pattern, return None
            print("⚠️  No segment found with completion pattern!")
            return None, -1

        # Find the valid SLM generator
        slm_output_generator, selected_segment = find_valid_slm_generator()

        if slm_output_generator is None:
            print("ERROR: No valid SLM generator found with completion pattern. Cannot proceed.")
            return

        print(f"Using SLM generator from segment {selected_segment} for bistream inference.")
        
        # 3. Call self.taste_stage1.inference_bistream() using SLM generator directly
        print("\nStarting bistream inference with SLM output generator...")
        
        # Debug: Check model parameter dtypes
        print("Checking model parameter dtypes...")
        for name, param in self.taste_stage1.named_parameters():
            print(f"  {name}: {param.dtype}")
            break  # Just show first few parameters
        
        # Prepare prompt (empty for now, can be extended)
        prompt_text = torch.zeros(1, 0, dtype=torch.int32).to(self.device)
        prompt_text_len = torch.tensor([0], dtype=torch.int32).to(self.device)  
        
        # Detect model dtype from model parameters
        model_dtype = next(self.taste_stage1.parameters()).dtype
        print(f"Model dtype detected: {model_dtype}")
        
        # Ensure prompt_speech_feature has the same dtype as the model
        prompt_speech_feature = torch.zeros(1, 0, 80, dtype=model_dtype).to(self.device)
        prompt_speech_feature_len = torch.tensor([0], dtype=torch.int32).to(self.device)
        
        # Get s3 token generator from bistream inference using SLM generator directly
        s3_token_generator = self.taste_stage1.inference_bistream(
            input_generator=slm_output_generator,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_feature=prompt_speech_feature,
            prompt_speech_feature_len=prompt_speech_feature_len,
            sampling=25,
        )
        
        # 4. Collect ALL s3 tokens first, then generate overlapping audio sequences
        print("\nCollecting all s3 tokens first...")
        
        all_s3_tokens = []
        s3_tokens_buffer = []
        audio_chunks = []
        chunk_id = 0
        chunk_size = 20  # s3 token window length, cannot exceed 50
        
        # chunk_size / 25 * 24000
        window_size = int(chunk_size / self.model.flow.input_frame_rate * self.sample_rate)
        import time
        start_time = time.time()
        # Collect all s3 tokens
        for s3_token in s3_token_generator:
            all_s3_tokens.append(s3_token)
            s3_tokens_buffer.append(s3_token)
            print(f"Generated s3 token: {s3_token}")
            
            # Process every 5 s3 tokens
            if len(s3_tokens_buffer) >= chunk_size:
                chunk_id += 1
                print(f"\n--- Processing audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens) ---")
                
                # Convert tokens to tensor (all current s3 tokens)
                s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
                ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)],  dtype=torch.long).unsqueeze(0)
                print(f"S3 tokens shape: {s3_tokens_tensor.shape}")
                print(f"Ref S3 tokens shape: {ref_tokens_tensor.shape}")
                
                # Generate audio chunk
                try:
                    # Maybe different _postprocess_streaming, which takes prev s3 tokens as reference
                    for audio_chunk in self._postprocess_streaming(
                        s3_tokens=s3_tokens_tensor,
                        ref_s3_tokens=ref_tokens_tensor,
                        audio_16k=audio_16k
                    ):
                        # audio_chunk = self._postprocess(all_s3_tokens, audio_16k)
                        print(f"Generated audio shape: {audio_chunk.shape}")
                        
                        new_audio_chunk = audio_chunk[:, -window_size:]
                        print(f"Truncated audio shape: {new_audio_chunk.shape}")
                        audio_chunks.append(new_audio_chunk)
                        
                        # Yield chunk result
                        yield {
                            'audio_chunk': new_audio_chunk,
                            'chunk_id': chunk_id,
                            's3_tokens': s3_tokens_buffer,
                            'asr_text': asr_text,
                            'is_final': False
                        }
                        
                    print(f"Audio chunk {chunk_id} generated successfully")
                    print(f"Time: {time.time() - start_time} seconds")
                    
                except Exception as e:
                    print(f"Error generating audio chunk {chunk_id}: {e}")
                    continue
                
                # Clear current s3 token buffer
                s3_tokens_buffer.clear()
                start_time = time.time()
        
        # Process remaining tokens as final chunk with overlap
        if s3_tokens_buffer:
            chunk_id += 1
            print(f"\n--- Processing final audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens) ---")
            
            s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
            ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)],  dtype=torch.long).unsqueeze(0)
            print(f"Final S3 tokens shape: {s3_tokens_tensor.shape}")
            print(f"Ref S3 tokens shape: {ref_tokens_tensor.shape}")
            
            try:
                # audio_chunk = self._postprocess(s3_tokens_tensor, audio_16k)
                for audio_chunk in self._postprocess_streaming(
                    s3_tokens=s3_tokens_tensor,
                    ref_s3_tokens=ref_tokens_tensor,
                    audio_16k=audio_16k
                ):
                    final_window_size = int(window_size / chunk_size * len(s3_tokens_buffer))
                    new_audio_chunk = audio_chunk[:, -final_window_size:]
                    audio_chunks.append(new_audio_chunk)
                    # Yield final chunk
                    yield {
                        'audio_chunk': new_audio_chunk,
                        'chunk_id': chunk_id,
                        's3_tokens': s3_tokens_buffer,
                        'asr_text': asr_text,
                        'is_final': True
                    }
                
                print(f"Final audio chunk {chunk_id} generated successfully")
                
            except Exception as e:
                print(f"Error generating final audio chunk: {e}")
        
        # Generate audio from ALL s3 tokens at once for comparison
        if all_s3_tokens:
            print(f"\n=== Generating audio from ALL {len(all_s3_tokens)} s3 tokens at once ===")
            all_s3_tokens_tensor = torch.tensor(all_s3_tokens, dtype=torch.long).unsqueeze(0)
            print(f"All S3 tokens shape: {all_s3_tokens_tensor.shape}")
            print(f"All S3 tokens: {all_s3_tokens}")
            
            try:
                single_audio = self._postprocess(all_s3_tokens_tensor, audio_16k)
                print(f"Single audio from all tokens generated successfully (shape: {single_audio.shape})")
                print(f"Single audio duration: {single_audio.shape[-1] / self.sample_rate:.2f}s")
                
                # Yield the single audio result
                yield {
                    'single_audio': single_audio,
                    'all_s3_tokens': all_s3_tokens,
                    'total_tokens': len(all_s3_tokens),
                    'asr_text': asr_text,
                    'is_single_audio': True
                }
                
            except Exception as e:
                print(f"Error generating single audio from all tokens: {e}")
                single_audio = None
        else:
            single_audio = None
        
        # Combine all audio chunks with crossfading for smooth transitions
        if audio_chunks:
            print(f"\n=== Combining {len(audio_chunks)} audio chunks with crossfading ===")
            final_audio = self._crossfade_audio_chunks(audio_chunks, crossfade_samples=int(0.01 * self.sample_rate))  # 10ms crossfade
            
            print(f"Final audio shape: {final_audio.shape}")
            print(f"Final audio duration: {final_audio.shape[-1] / self.sample_rate:.2f}s")
            
            yield {
                'final_audio': final_audio,
                'total_chunks': len(audio_chunks),
                'asr_text': asr_text,
                'is_complete': True
            }
        else:
            print("Warning: No audio chunks were generated")


def process_single_file_streaming(model, audio_file, output_dir, asr_model_dir, stage):
    """Process a single audio file with streaming generation"""
    print(f"\n=== Streaming Processing: {os.path.basename(audio_file)} ===")
    
    # Load audio data
    audio_16k, ground_truth_text, audio_basename = load_audio_data(audio_file, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Starting streaming generation for: {audio_basename}")
    print(f"{'='*60}")
    
    # Run streaming generation based on stage
    try:
        chunk_paths = []
        single_audio_path = None
        generated_text = None
        
        # Choose appropriate streaming method based on stage
        if stage == '1':
            streaming_generator = model.generation_stage1_streaming(
                audio_16k, asr_model_dir=asr_model_dir
            )
        elif stage == '2':
            streaming_generator = model.generation_stage2_streaming(
                audio_16k, asr_model_dir=asr_model_dir
            )
        elif stage == 'sft':
            streaming_generator = model.generation_sft_streaming(
                audio_16k, asr_model_dir=asr_model_dir
            )
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        
        # Process streaming results
        for result in streaming_generator:
            if result.get('is_single_audio'):
                # Save single audio generated from all tokens at once
                single_audio_path = os.path.join(output_dir, f"{audio_basename}_single_from_all_tokens.wav")
                torchaudio.save(single_audio_path, result['single_audio'].cpu(), model.sample_rate)
                print_green(f"Single audio (all tokens) saved to: {single_audio_path}")
                print_green(f"Total tokens used: {result['total_tokens']}")
                print_green(f"All s3 tokens: {result['all_s3_tokens']}")
                
            elif result.get('is_complete'):
                # Save final combined audio (crossfaded chunks)
                final_path = os.path.join(output_dir, f"{audio_basename}_streaming_final_crossfaded.wav")
                torchaudio.save(final_path, result['final_audio'].cpu(), model.sample_rate)
                print_green(f"Final crossfaded streaming audio saved to: {final_path}")
                
                # Extract generated text if available (Stage 2)
                if 'generated_text' in result:
                    generated_text = result['generated_text']
                
                # Save streaming results summary
                summary = {
                    'audio_basename': audio_basename,
                    'asr_text': result['asr_text'],
                    'total_chunks': result['total_chunks'],
                    'chunk_files': chunk_paths,
                    'final_crossfaded_path': final_path,
                    'single_audio_path': single_audio_path,
                    'ground_truth_text': ground_truth_text,
                    'stage': stage,
                    'streaming': True
                }
                
                # Add generated text for stage 2
                if generated_text:
                    summary['generated_text'] = generated_text
                
                summary_path = os.path.join(output_dir, f"{audio_basename}_streaming_summary.json")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                print_green(f"Streaming summary saved to: {summary_path}")
                print_green(f"\n=== Streaming Generation Complete ===\nTotal chunks: {result['total_chunks']}")
                print_green(f"📊 Compare these files:")
                print_green(f"   • Single audio (all tokens): {single_audio_path}")
                print_green(f"   • Crossfaded chunks: {final_path}")
                
                if generated_text:
                    print_green(f"   • Generated text: {generated_text}")
                
                # Return results in same format as non-streaming
                results = {
                    f'stage{stage}_streaming': {
                        'final_audio_path': final_path,
                        'single_audio_path': single_audio_path,
                        'asr_text': result['asr_text'],
                        'total_chunks': result['total_chunks'],
                        'streaming_summary_path': summary_path
                    }
                }
                
                if generated_text:
                    results[f'stage{stage}_streaming']['generated_text'] = generated_text
                
                return {
                    'original_text': ground_truth_text,
                    'audio_basename': audio_basename,
                    'audio_file': audio_file,
                    'results': results
                }
                
            elif result.get('audio_chunk') is not None:
                # Save individual chunk
                chunk_path = os.path.join(output_dir, f"{audio_basename}_chunk_{result['chunk_id']:03d}.wav")
                torchaudio.save(chunk_path, result['audio_chunk'].cpu(), model.sample_rate)
                chunk_paths.append(chunk_path)
                
                print_green(f"Chunk {result['chunk_id']} saved to: {chunk_path}")
                print(f"Chunk contains {len(result['s3_tokens'])} s3 tokens: {result['s3_tokens']}")
    
    except Exception as e:
        print(f"Error during streaming generation for {audio_basename}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='TASTE2 Audio Streaming Generation (Stage 1 & 2)')
    parser.add_argument('--model_dir', type=str, required=True, help='TASTE2 model directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--test_files', type=str, nargs='+', required=True, help='Audio/Arrow file paths (multiple files supported)')
    parser.add_argument('--asr_model_dir', type=str, default="openai/whisper-large-v3", help='ASR model')
    parser.add_argument('--stage', type=str, choices=['1', '2', 'sft'], default='1',
                        help='Which stage to run: 1, 2, or sft (default: 1)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    test_files = args.test_files
    
    print(f"Processing {len(test_files)} audio files...")
    print(f"Output directory: {args.output_dir}")
    print(f"Stage: {args.stage}")
    
    # Initialize model once
    if args.stage == '1':
        print(f"\nInitializing TASTE2 model for Stage 1 (Streaming)...")
        model = TASTE2(args.model_dir, stage=1, fp16=False)
    elif args.stage == '2':
        print(f"\nInitializing TASTE2 model for Stage 2 (Streaming)...")
        model = TASTE2(args.model_dir, stage=2, fp16=False)
    elif args.stage == 'sft':
        print(f"\nInitializing TASTE2 model for SFT (Streaming)...")
        model = TASTE2(args.model_dir, stage='sft', fp16=False)
    
    # Process all files
    all_results = []
    successful_files = 0
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(test_files)}: {os.path.basename(test_file)}")
        print('='*60)
        
        try:
            if not os.path.exists(test_file):
                print(f"Warning: File {test_file} does not exist, skipping...")
                continue
            
            # Process with streaming mode only
            file_results = process_single_file_streaming(
                model, test_file, args.output_dir, args.asr_model_dir, args.stage
            )
            
            if file_results:  # Only add if processing was successful
                all_results.append(file_results)
                successful_files += 1
            
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save batch results summary
    batch_results = {
        'total_files': len(test_files),
        'successful_files': successful_files,
        'failed_files': len(test_files) - successful_files,
        'stage': args.stage,
        'streaming': True,
        'model_dir': args.model_dir,
        'asr_model_dir': args.asr_model_dir,
        'files': all_results
    }
    
    batch_json_path = os.path.join(args.output_dir, "batch_results.json")
    with open(batch_json_path, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    print_green(f"\n{'='*60}")
    print_green(f"BATCH PROCESSING COMPLETE (STREAMING MODE)")
    print_green(f"{'='*60}")
    print_green(f"Total files: {batch_results['total_files']}")
    print_green(f"Successful: {batch_results['successful_files']}")
    print_green(f"Failed: {batch_results['failed_files']}")
    print_green(f"Stage: {args.stage} | Streaming: True")
    print_green(f"Batch results saved to: {batch_json_path}")
    print_green(f"All outputs in: {args.output_dir}")

if __name__ == '__main__':
    main()