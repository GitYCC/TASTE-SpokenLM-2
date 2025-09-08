"""
TASTE2 Model Implementation (Stage 2 SLM only) based on generate_audio_stream.py patterns.

This module provides the TASTE2 model class optimized for Stage 2 SLM pipeline only.
All stage-related conditionals have been removed for simplicity.
"""

import os
import sys
import threading
import uuid
import numpy as np
from contextlib import nullcontext
from typing import Optional, List, Dict, Union, Generator, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.utils import ModelOutput
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download

# Import configuration
from .configuration_taste2 import Taste2Config

# Import CosyVoice components
try:
    from taste_speech.taste2.cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from taste_speech.taste2.cosyvoice.utils.file_utils import logging
    from taste_speech.taste2.cosyvoice.utils.common import fade_in_out
except ImportError:
    # Fallback import path
    try:
        from cosyvoice.cli.frontend import CosyVoiceFrontEnd
        from cosyvoice.utils.file_utils import logging
        from cosyvoice.utils.common import fade_in_out
    except ImportError:
        print("Warning: CosyVoice components not found. Make sure CosyVoice is properly installed.")
        CosyVoiceFrontEnd = None
        logging = None
        fade_in_out = None


class Taste2OutputWithPast(ModelOutput):
    """
    Output type for TASTE2 streaming models.
    """
    audio_chunk: Optional[torch.FloatTensor] = None
    asr_text: Optional[str] = None
    generated_text: Optional[str] = None
    s3_tokens: Optional[torch.LongTensor] = None
    chunk_id: Optional[int] = None
    is_final: Optional[bool] = None
    is_complete: Optional[bool] = None
    is_single_audio: Optional[bool] = None
    final_audio: Optional[torch.FloatTensor] = None
    single_audio: Optional[torch.FloatTensor] = None
    total_tokens: Optional[int] = None
    all_s3_tokens: Optional[List] = None


class Taste2ForCausalLM(PreTrainedModel, GenerationMixin):
    """
    TASTE2 Model for Stage 2 SLM streaming audio generation.
    
    Based on generate_audio_stream.py implementation, this model integrates:
    - LLM component for text-to-speech token generation
    - Flow component for mel spectrogram generation  
    - HiFT component for vocoding mel spectrograms to audio
    - SLM component for advanced speech generation (always loaded)
    - Frontend for text tokenization and audio feature extraction
    - Streaming capabilities with bistream inference
    
    All components are always loaded for the Stage 2 SLM pipeline.
    """
    
    config_class = Taste2Config
    
    def __init__(self, config: Taste2Config):
        super().__init__(config)
        self.config = config
        self._device = torch.device(config.device if hasattr(config, 'device') and config.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components - will be loaded in from_pretrained
        # All components are required for Stage 2 SLM
        self.llm = None
        self.flow = None
        self.hift = None
        self.slm = None       # Always loaded for Stage 2
        self.frontend = None
        self.audio_extractor = None
        self.taste_stage1 = None
        
        # Streaming parameters (from generate_audio_stream.py)
        self.token_hop_len = config.token_hop_len
        self.mel_cache_len = config.mel_cache_len
        self.source_cache_len = config.source_cache_len
        self.speech_window = config.get_speech_window()
        self.chunk_size = config.chunk_size
        
        # Threading and session management (from TASTE2Model)
        self.lock = threading.Lock()
        self.session_data = {}
        
        if torch.cuda.is_available():
            self.llm_context = torch.cuda.stream(torch.cuda.Stream(self._device))
        else:
            self.llm_context = nullcontext()
    
    @property
    def device(self):
        """Return the model device for compatibility."""
        return self._device
    
    @classmethod
    def from_pretrained(cls, model_dir_or_name, **kwargs):
        """
        Load TASTE2 model from pretrained directory (Stage 2 SLM only).
        """
        # Handle model download if needed
        model_dir = model_dir_or_name
        if not os.path.exists(model_dir):
            # Try HuggingFace download for Stage 2 models
            model_dir = hf_snapshot_download(model_dir_or_name)
        
        # Create config with model directory
        config = cls.config_class(
            model_dir=model_dir,
            **kwargs
        )
        
        # Validate model directory structure
        missing_files = config.validate_model_directory()
        if missing_files:
            print(f"Warning: Missing files in {model_dir}: {missing_files[:3]}...")
        
        # Create model instance
        model = cls(config)
        
        # Load model components and parameters
        model._load_model_components()
        
        return model
    
    def _load_model_components(self):
        """Load all model components and their parameters (Stage 2 SLM pipeline)."""
        
        # Load YAML configuration
        config_path = self.config.get_config_file_path()
        if not os.path.exists(config_path):
            raise ValueError(f'{config_path} not found!')
        
        with open(config_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={
                'qwen_pretrain_path': self.config.get_qwen_pretrain_path()
            })
        
        # Initialize frontend
        if CosyVoiceFrontEnd is not None:
            self.frontend = CosyVoiceFrontEnd(
                configs['get_tokenizer'],
                configs['feat_extractor'],
                self.config.get_frontend_file_path('campplus_model'),
                self.config.get_frontend_file_path('speech_tokenizer_model'),
                self.config.get_frontend_file_path('speaker_info'),
                configs['allowed_special']
            )
        else:
            raise ImportError("CosyVoiceFrontEnd not available. Please install CosyVoice.")
        
        # Initialize audio extractor
        self.audio_extractor = configs['audio_extractor']
        
        # Initialize model components from configs (all required for Stage 2)
        self.llm = configs['llm']
        self.flow = configs['flow'] 
        self.hift = configs['hift']
        self.slm = configs['slm']  # Always loaded for Stage 2
        
        # Load model weights from checkpoint files
        self._load_model_weights()
        
        # Set taste_stage1 reference (always use slm.taste_stage1 for Stage 2)
        self.taste_stage1 = self.slm.taste_stage1
        
        # Apply configurations and move to device
        self._configure_models()
        
        # Load optimizations if requested
        self._load_optimizations()
        
        # Clean up configs
        del configs
    
    def _load_model_weights(self):
        """Load model weights from checkpoint files (all required for Stage 2 SLM)."""
        
        # Load LLM weights
        llm_path = self.config.get_checkpoint_path('llm')
        if os.path.exists(llm_path):
            llm_state_dict = torch.load(llm_path, map_location=self._device)
            self.llm.load_state_dict(llm_state_dict, strict=True)
            print(f"Loaded LLM weights from {llm_path}")
        else:
            raise FileNotFoundError(f"Required LLM weights not found at {llm_path}")
        
        # Load Flow weights
        flow_path = self.config.get_checkpoint_path('flow')
        if os.path.exists(flow_path):
            flow_state_dict = torch.load(flow_path, map_location=self._device)
            self.flow.load_state_dict(flow_state_dict, strict=True)
            print(f"Loaded Flow weights from {flow_path}")
        else:
            raise FileNotFoundError(f"Required Flow weights not found at {flow_path}")
        
        # Load HiFT weights (handle potential HifiGAN naming)
        hift_path = self.config.get_checkpoint_path('hift')
        if os.path.exists(hift_path):
            hift_state_dict = torch.load(hift_path, map_location=self._device)
            # Handle HifiGAN generator prefix
            hift_state_dict = {k.replace('generator.', ''): v for k, v in hift_state_dict.items()}
            self.hift.load_state_dict(hift_state_dict, strict=True)
            print(f"Loaded HiFT weights from {hift_path}")
        else:
            raise FileNotFoundError(f"Required HiFT weights not found at {hift_path}")
        
        # Load SLM weights (always required for Stage 2)
        slm_path = self.config.get_checkpoint_path('slm')
        if os.path.exists(slm_path):
            slm_state_dict = torch.load(slm_path, map_location=self._device)
            self.slm.load_state_dict(slm_state_dict, strict=True)
            print(f"Loaded SLM weights from {slm_path}")
        else:
            raise FileNotFoundError(f"Required SLM weights not found at {slm_path}")
    
    def _configure_models(self):
        """Configure models with fp16, device placement, etc."""
        
        # Apply fp16 if requested
        if self.config.fp16:
            if self.llm is not None:
                self.llm.half()
            if self.flow is not None:
                self.flow.half()
        
        # Move models to device and set to eval mode (all required)
        self.llm.to(self._device).eval()
        self.flow.to(self._device).eval()
        self.hift.to(self._device).eval()
        self.slm.to(self._device).eval()
        
        # Handle CUDA availability warnings
        if not torch.cuda.is_available() and (self.config.load_jit or self.config.load_trt or self.config.fp16):
            if logging is not None:
                logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
            else:
                print('Warning: no cuda device, set load_jit/load_trt/fp16 to False')
    
    def _load_optimizations(self):
        """Load optimization modules if requested."""
        optimization_paths = self.config.get_optimization_paths()
        
        # Load VLLM optimization
        if self.config.load_vllm and 'vllm' in optimization_paths:
            if hasattr(self, 'load_vllm'):
                self.load_vllm(optimization_paths['vllm'])
        
        # Load JIT optimization
        if self.config.load_jit and 'jit' in optimization_paths:
            if hasattr(self, 'load_jit'):
                self.load_jit(optimization_paths['jit'])
        
        # Load TensorRT optimization
        if self.config.load_trt and 'trt_plan' in optimization_paths and 'trt_onnx' in optimization_paths:
            if hasattr(self, 'load_trt'):
                self.load_trt(
                    optimization_paths['trt_plan'],
                    optimization_paths['trt_onnx'],
                    self.config.trt_concurrent,
                    self.config.fp16
                )
    
    def _run_asr(self, audio_16k, asr_model_dir="openai/whisper-large-v3"):
        """Run ASR on audio to get text transcription."""
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
        """Preprocess audio and text into model inputs."""
        assert asr_model_dir or text, "Either ASR model or text must be provided"

        # Get text from ASR or use provided text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        
        # Extract text and audio tokens using frontend
        text_token, text_token_len = self.frontend._extract_text_token(asr_text)
        audio_feature, audio_feature_len = self.audio_extractor(audio_16k, [audio_16k.shape[-1]])
        
        # Convert to model dtype if using fp16
        if self.config.fp16:
            audio_feature = audio_feature.half()
        
        # Move to device
        audio_feature = audio_feature.to(self._device)

        # Tokenize using taste_stage1 tokenizer
        taste_tokenizer = self.taste_stage1.taste_tokenizer
        tokenized = taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
        taste_token_emb = tokenized['taste_token_emb']

        return dict(
            text_token=text_token.to(self._device),
            text_token_len=text_token_len.to(self._device),
            taste_token_emb=taste_token_emb.to(self._device),
            asr_text=asr_text,
        )
    
    def _postprocess_streaming(self, s3_tokens, ref_s3_tokens, audio_16k):
        """Convert tokens to audio with streaming support."""
        speaker_embedding = self.frontend._extract_spk_embedding(audio_16k)
        for audio_chunk in self.tts(
            flow_prompt_speech_token=ref_s3_tokens,
            source_speech_token=s3_tokens,
            flow_embedding=speaker_embedding,
            stream=True
        ):
            yield audio_chunk['tts_speech']
    
    def _postprocess(self, s3_tokens, audio_16k):
        """Convert tokens back to audio."""
        speaker_embedding = self.frontend._extract_spk_embedding(audio_16k)
        audio_chunks = list(self.tts(
            source_speech_token=s3_tokens,
            flow_embedding=speaker_embedding
        ))
        
        return torch.cat([chunk['tts_speech'] for chunk in audio_chunks], dim=1)
    
    def _crossfade_audio_chunks(self, audio_chunks, crossfade_samples=None):
        """
        Crossfade audio chunks to eliminate popping sounds.
        """
        if crossfade_samples is None:
            crossfade_samples = self.config.crossfade_samples
            
        if len(audio_chunks) <= 1:
            return torch.cat(audio_chunks, dim=-1) if audio_chunks else torch.zeros(1, 0)
        
        print(f"Crossfading {len(audio_chunks)} chunks with {crossfade_samples} samples ({crossfade_samples/self.config.sample_rate*1000:.1f}ms)")
        
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
                
                # Apply crossfade only to the boundary region
                combined_audio[..., -actual_crossfade:] = (
                    combined_audio[..., -actual_crossfade:] * fade_out + 
                    current_chunk[..., :actual_crossfade] * fade_in
                )
                
                # Concatenate with remainder of current chunk
                combined_audio = torch.cat([
                    combined_audio,
                    current_chunk[..., actual_crossfade:]
                ], dim=-1)
            else:
                # If no overlap possible, just concatenate
                combined_audio = torch.cat([combined_audio, current_chunk], dim=-1)
        
        return combined_audio
    
    def _run_inference_job(self, source_token, session_id, **kwargs):
        """Unified inference job for both LLM and VC."""
        if source_token.shape[1] == 0:  # LLM job
            with self.llm_context, torch.cuda.amp.autocast(self.config.fp16 and not hasattr(self.llm, 'vllm')):
                for token in self.llm.inference(**{k: v.to(self._device) if hasattr(v, 'to') else v for k, v in kwargs.items()}, uuid=session_id):
                    self.session_data[session_id]['tokens'].append(token)
        else:  # VC job
            self.session_data[session_id]['tokens'] = source_token.flatten().tolist()
        self.session_data[session_id]['finished'] = True
    
    def token2wav(self, token, prompt_token, prompt_feat, embedding, session_id, token_offset=0, finalize=True, speed=1.0):
        """Convert tokens to waveform."""
        with torch.cuda.amp.autocast(self.config.fp16):
            tts_mel, _ = self.flow.inference(
                token=token.to(self._device),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self._device),
                prompt_token=prompt_token.to(self._device),
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self._device),
                prompt_feat=prompt_feat.to(self._device),
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self._device),
                embedding=embedding.to(self._device),
                streaming=False,
                finalize=finalize
            )
        
        if speed != 1.0 and finalize:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
        
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0))
        return tts_speech
    
    def tts(self, source_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), speed=1.0, **kwargs):
        """TTS generation for reconstruction."""
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
    
    @torch.inference_mode()
    def tokenize(self, audio_16k, asr_model_dir=None, text=None):
        """
        Tokenization process: Convert audio to text token IDs and taste embeddings.
        
        Args:
            audio_16k (torch.Tensor): Input audio waveform at 16kHz, shape (1, samples)
            asr_model_dir (str, optional): ASR model directory for text extraction
            text (str, optional): Pre-extracted text (alternative to ASR)
            
        Yields:
            Taste2TokenizationOutput: Contains text_token_ids, taste_token_embs, and asr_text
        """
        assert asr_model_dir or text, "Either ASR model or text must be provided"
        print("=== TASTE2 Tokenization Process ===")

        # Get text from ASR or use provided text
        asr_text = text if text else self._run_asr(audio_16k, asr_model_dir)
        print(f"ASR Text: {asr_text}")
        
        # Extract text and audio tokens using frontend
        print("Extracting text tokens...")
        text_token, text_token_len = self.frontend._extract_text_token(asr_text)
        
        print("Extracting audio features...")
        audio_feature, audio_feature_len = self.audio_extractor(audio_16k, [audio_16k.shape[-1]])
        
        # Convert to model dtype if using fp16
        if self.config.fp16:
            audio_feature = audio_feature.half()
        
        # Move to device
        audio_feature = audio_feature.to(self._device)
        text_token = text_token.to(self._device)
        text_token_len = text_token_len.to(self._device)

        # Tokenize using taste_stage1 tokenizer
        print("Generating taste token embeddings...")
        taste_tokenizer = self.taste_stage1.taste_tokenizer
        tokenized = taste_tokenizer(text_token, text_token_len, audio_feature, audio_feature_len)
        taste_token_emb = tokenized['taste_token_emb']
        
        print(f"Tokenization complete: text_tokens={text_token.shape}, taste_embs={taste_token_emb.shape}")
        
        yield Taste2TokenizationOutput(
            text_token_ids=text_token,
            text_token_len=text_token_len,
            taste_token_embs=taste_token_emb,
            asr_text=asr_text,
        )
    
    @torch.inference_mode()
    def slm_generate(self, text_token_ids, text_token_len, taste_token_embs, min_len=3, max_len=20, sampling=25):
        """
        SLM Generation process: Generate new text tokens and taste embeddings.
        
        Args:
            text_token_ids (torch.Tensor): Input text token IDs, shape (1, seq_len)
            text_token_len (torch.Tensor): Length of input text tokens, shape (1,)
            taste_token_embs (torch.Tensor): Input taste token embeddings, shape (1, seq_len, emb_dim)
            min_len (int): Minimum generation length
            max_len (int): Maximum generation length  
            sampling (int): Sampling parameter
            
        Yields:
            Taste2SLMGenerationOutput: Contains generated text tokens and taste embeddings
        """
        print("=== TASTE2 SLM Generation Process ===")
        print(f"Input: text_tokens={text_token_ids.shape}, taste_embs={taste_token_embs.shape}")
        
        # Create SLM output generator (yields (text_token, taste_emb) pairs)
        print("Starting SLM inference to generate streaming text and taste embeddings...")
        slm_output_generator = self.slm.inference(
            text_token=text_token_ids,
            text_token_len=text_token_len,
            taste_token_emb=taste_token_embs,
            min_len=min_len,
            max_len=max_len,
            sampling=sampling,
        )
        
        step = 0
        generated_text_tokens = []
        generated_taste_embs = []
        
        for new_text_token, new_taste_emb in slm_output_generator:
            step += 1
            generated_text_tokens.append(new_text_token)
            generated_taste_embs.append(new_taste_emb)
            
            # Ensure new_taste_emb has shape [1, 1, 896]
            if new_taste_emb.dim() == 2:  # [seq_len, embed_dim] -> [1, seq_len, embed_dim]
                new_taste_emb = new_taste_emb.unsqueeze(0)
            elif new_taste_emb.dim() == 1:  # [embed_dim] -> [1, 1, embed_dim]
                new_taste_emb = new_taste_emb.unsqueeze(0).unsqueeze(0)
            elif new_taste_emb.dim() > 3:  # Squeeze extra dimensions to get [1, 1, embed_dim]
                new_taste_emb = new_taste_emb.squeeze()
                if new_taste_emb.dim() == 2:
                    new_taste_emb = new_taste_emb.unsqueeze(0)
                elif new_taste_emb.dim() == 1:
                    new_taste_emb = new_taste_emb.unsqueeze(0).unsqueeze(0)
            
            print(f"SLM Step {step}: Generated text_token={new_text_token.item() if hasattr(new_text_token, 'item') else new_text_token}, taste_emb_shape={new_taste_emb.shape}")
            
            # Yield individual token and embedding (not accumulated)
            yield Taste2SLMGenerationOutput(
                generated_text_token_ids=new_text_token.unsqueeze(0).to(self._device),  # [1, 1]
                generated_taste_token_embs=new_taste_emb.to(self._device),  # [1, 1, 896] 
                generated_text=str(new_text_token.item() if hasattr(new_text_token, 'item') else new_text_token),
                is_final=False,
                step=step
            )
        
        # Final output
        if generated_text_tokens:
            final_text_tokens = torch.cat([
                text_token_ids,
                torch.cat(generated_text_tokens, dim=0).unsqueeze(0).to(self._device)
            ], dim=1)
            # Process generated taste embeddings - each should be [1, 1, 896]
            processed_final_taste_embs = []
            for emb in generated_taste_embs:
                # Ensure each embedding is [1, 1, 896]
                if emb.dim() == 2:  # [seq_len, embed_dim] -> [1, seq_len, embed_dim]
                    emb = emb.unsqueeze(0)
                elif emb.dim() == 1:  # [embed_dim] -> [1, 1, embed_dim]
                    emb = emb.unsqueeze(0).unsqueeze(0)
                elif emb.dim() > 3:  # Squeeze extra dimensions
                    emb = emb.squeeze()
                    if emb.dim() == 2:
                        emb = emb.unsqueeze(0)
                    elif emb.dim() == 1:
                        emb = emb.unsqueeze(0).unsqueeze(0)
                processed_final_taste_embs.append(emb)
            
            # Concatenate along sequence dimension (dim=1)
            final_generated_taste_tensor = torch.cat(processed_final_taste_embs, dim=1).to(self._device)
            final_taste_embs = torch.cat([
                taste_token_embs,
                final_generated_taste_tensor
            ], dim=1)
            
            final_generated_text = self.frontend.tokenizer.decode(final_text_tokens[0].tolist())
        else:
            final_text_tokens = text_token_ids
            final_taste_embs = taste_token_embs
            final_generated_text = self.frontend.tokenizer.decode(text_token_ids[0].tolist())
        
        print(f"SLM Generation complete: {step} steps, final_text='{final_generated_text}'")
        
        yield Taste2SLMGenerationOutput(
            generated_text_token_ids=final_text_tokens,
            generated_taste_token_embs=final_taste_embs,
            generated_text=final_generated_text,
            is_final=True,
            step=step
        )
    
    @torch.inference_mode()
    def detokenize(self, generated_text_token_ids, generated_taste_token_embs, audio_16k, streaming=True, chunk_size=None):
        """
        Detokenization process: Convert generated text tokens and taste embeddings to audio.
        
        Args:
            generated_text_token_ids (torch.Tensor): Generated text token IDs, shape (1, seq_len)
            generated_taste_token_embs (torch.Tensor): Generated taste embeddings, shape (1, seq_len, emb_dim)
            audio_16k (torch.Tensor): Reference audio for speaker embedding extraction
            streaming (bool): Whether to use streaming generation
            chunk_size (int, optional): Override chunk size for streaming
            
        Yields:
            Taste2DetokenizationOutput: Contains audio chunks and final audio
        """
        print("=== TASTE2 Detokenization Process ===")
        print(f"Input: text_tokens={generated_text_token_ids.shape}, taste_embs={generated_taste_token_embs.shape}")
        
        # Debug: Check if all slices have consistent shape
        seq_len = generated_taste_token_embs.shape[1]
        for i in range(min(5, seq_len)):  # Check first 5 slices
            slice_shape = generated_taste_token_embs[:, i:i+1, :].shape
            print(f"Debug taste_emb slice {i}: {slice_shape}")
        
        # Debug: Check the overall tensor structure
        print(f"Debug overall taste_embs min/max dims: {generated_taste_token_embs.min():.4f}/{generated_taste_token_embs.max():.4f}")
        
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        # Create input generator from the generated tokens and embeddings
        def create_input_generator():
            """Create generator of individual (text_token, taste_emb) pairs"""
            seq_len = generated_text_token_ids.shape[1]
            print(f"Creating input generator with {seq_len} token pairs")
            
            for i in range(seq_len):
                text_token = generated_text_token_ids[:, i:i+1]  # [1, 1] 
                taste_emb = generated_taste_token_embs[:, i:i+1, :]  # [1, 1, emb_dim]
                
                # The bistream inference expects text_token without batch dimension for unsqueeze(0)
                # So we need to provide [1] instead of [1, 1] to get [1, 1] after unsqueeze(0)
                text_token = text_token.squeeze(0)  # [1, 1] -> [1]
                
                print(f"Debug - Step {i}: text_token.shape={text_token.shape}, taste_emb.shape={taste_emb.shape}")
                
                # Ensure consistent dtype
                model_dtype = next(self.taste_stage1.parameters()).dtype
                if taste_emb.dtype != model_dtype:
                    taste_emb = taste_emb.to(model_dtype)
                
                yield text_token, taste_emb
        
        # Prepare bistream inference
        print("Starting bistream inference for speech token generation...")
        
        # Prepare prompt (empty)
        prompt_text = torch.zeros(1, 0, dtype=torch.int32).to(self._device)
        prompt_text_len = torch.tensor([0], dtype=torch.int32).to(self._device)
        
        model_dtype = next(self.taste_stage1.parameters()).dtype
        prompt_speech_feature = torch.zeros(1, 0, 80, dtype=model_dtype).to(self._device)
        prompt_speech_feature_len = torch.tensor([0], dtype=torch.int32).to(self._device)
        
        # Get s3 token generator from bistream inference
        s3_token_generator = self.taste_stage1.inference_bistream(
            input_generator=create_input_generator(),
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_feature=prompt_speech_feature,
            prompt_speech_feature_len=prompt_speech_feature_len,
            sampling=25,
        )
        
        # Process s3 tokens into audio
        all_s3_tokens = []
        s3_tokens_buffer = []
        audio_chunks = []
        chunk_id = 0
        
        if streaming:
            window_size = self.config.get_window_size(self.flow.input_frame_rate)
            print(f"Streaming mode: chunk_size={chunk_size}, window_size={window_size}")
            
            # Process tokens in streaming chunks
            for s3_token in s3_token_generator:
                all_s3_tokens.append(s3_token)
                s3_tokens_buffer.append(s3_token)
                
                # Process every chunk_size s3 tokens
                if len(s3_tokens_buffer) >= chunk_size:
                    chunk_id += 1
                    print(f"Processing audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens)")
                    
                    s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
                    ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)], dtype=torch.long).unsqueeze(0)
                    
                    try:
                        for audio_chunk in self._postprocess_streaming(
                            s3_tokens=s3_tokens_tensor,
                            ref_s3_tokens=ref_tokens_tensor,
                            audio_16k=audio_16k
                        ):
                            new_audio_chunk = audio_chunk[:, -window_size:]
                            audio_chunks.append(new_audio_chunk)
                            
                            yield Taste2DetokenizationOutput(
                                audio_chunk=new_audio_chunk,
                                s3_tokens=s3_tokens_tensor,
                                chunk_id=chunk_id,
                                is_final=False
                            )
                    except Exception as e:
                        print(f"Error generating audio chunk {chunk_id}: {e}")
                        continue
                    
                    s3_tokens_buffer.clear()
            
            # Process remaining tokens
            if s3_tokens_buffer:
                chunk_id += 1
                print(f"Processing final audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens)")
                
                s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
                ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)], dtype=torch.long).unsqueeze(0)
                
                try:
                    for audio_chunk in self._postprocess_streaming(
                        s3_tokens=s3_tokens_tensor,
                        ref_s3_tokens=ref_tokens_tensor,
                        audio_16k=audio_16k
                    ):
                        final_window_size = int(window_size / chunk_size * len(s3_tokens_buffer))
                        new_audio_chunk = audio_chunk[:, -final_window_size:]
                        audio_chunks.append(new_audio_chunk)
                        
                        yield Taste2DetokenizationOutput(
                            audio_chunk=new_audio_chunk,
                            s3_tokens=s3_tokens_tensor,
                            chunk_id=chunk_id,
                            is_final=True
                        )
                except Exception as e:
                    print(f"Error generating final audio chunk: {e}")
        else:
            # Non-streaming mode: collect all tokens first
            print("Non-streaming mode: collecting all s3 tokens...")
            all_s3_tokens = list(s3_token_generator)
        
        # Generate single audio from all tokens
        if all_s3_tokens:
            print(f"Generating single audio from {len(all_s3_tokens)} s3 tokens")
            all_s3_tokens_tensor = torch.tensor(all_s3_tokens, dtype=torch.long).unsqueeze(0)
            
            try:
                single_audio = self._postprocess(all_s3_tokens_tensor, audio_16k)
                
                yield Taste2DetokenizationOutput(
                    single_audio=single_audio,
                    all_s3_tokens=all_s3_tokens,
                    total_tokens=len(all_s3_tokens),
                    is_complete=False
                )
            except Exception as e:
                print(f"Error generating single audio: {e}")
        
        # Combine streaming chunks with crossfading
        if streaming and audio_chunks:
            print(f"Combining {len(audio_chunks)} audio chunks with crossfading")
            final_audio = self._crossfade_audio_chunks(
                audio_chunks, 
                crossfade_samples=int(0.01 * self.config.sample_rate)
            )
            
            yield Taste2DetokenizationOutput(
                final_audio=final_audio,
                total_tokens=len(all_s3_tokens),
                all_s3_tokens=all_s3_tokens,
                is_complete=True
            )

    @torch.inference_mode()
    def forward(self, audio_16k, asr_model_dir=None, text=None, streaming=True, separate_processes=False):
        """
        Main forward pass for Stage 2 SLM streaming generation.
        
        Uses SLM + bistream inference pipeline from generate_audio_stream.py.
        """
        print("=== Starting Stage 2 SLM Streaming Generation ===")
        
        # 1. Preprocess audio and text
        print("Preprocessing audio and extracting features...")
        data = self._preprocess(audio_16k, asr_model_dir=asr_model_dir, text=text)
        asr_text = data['asr_text']
        print(f"ASR Text: {asr_text}")
        
        # 2. Create SLM output generator (yields (text_token, taste_emb) pairs)
        print("Starting SLM inference to generate streaming text and taste embeddings...")
        slm_output_generator = self.slm.inference(
            text_token=data['text_token'],
            text_token_len=data['text_token_len'],
            taste_token_emb=data['taste_token_emb'],
            min_len=3,
            max_len=20,
            sampling=25,
        )
        
        # 3. Call taste_stage1.inference_bistream() using SLM generator directly
        print("Starting bistream inference with SLM output generator...")
        
        # Prepare prompt (empty for now, can be extended)
        prompt_text = torch.zeros(1, 0, dtype=torch.int32).to(self._device)
        prompt_text_len = torch.tensor([0], dtype=torch.int32).to(self._device)
        
        # Detect model dtype from model parameters
        model_dtype = next(self.taste_stage1.parameters()).dtype
        print(f"Model dtype detected: {model_dtype}")
        
        # Ensure prompt_speech_feature has the same dtype as the model
        prompt_speech_feature = torch.zeros(1, 0, 80, dtype=model_dtype).to(self._device)
        prompt_speech_feature_len = torch.tensor([0], dtype=torch.int32).to(self._device)
        
        # Get s3 token generator from bistream inference using SLM generator directly
        s3_token_generator = self.taste_stage1.inference_bistream(
            input_generator=slm_output_generator,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_feature=prompt_speech_feature,
            prompt_speech_feature_len=prompt_speech_feature_len,
            sampling=25,
        )
        
        # 4. Process streaming s3 tokens
        print("Collecting and processing s3 tokens...")
        
        all_s3_tokens = []
        s3_tokens_buffer = []
        audio_chunks = []
        chunk_id = 0
        chunk_size = self.config.chunk_size
        
        window_size = self.config.get_window_size(self.flow.input_frame_rate)
        
        import time
        start_time = time.time()
        
        # Collect all s3 tokens and process in chunks
        for s3_token in s3_token_generator:
            all_s3_tokens.append(s3_token)
            s3_tokens_buffer.append(s3_token)
            print(f"Generated s3 token: {s3_token}")
            
            # Process every chunk_size s3 tokens
            if len(s3_tokens_buffer) >= chunk_size:
                chunk_id += 1
                print(f"\\n--- Processing audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens) ---")
                
                # Convert tokens to tensor
                s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
                ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)], dtype=torch.long).unsqueeze(0)
                
                # Generate audio chunk
                try:
                    for audio_chunk in self._postprocess_streaming(
                        s3_tokens=s3_tokens_tensor,
                        ref_s3_tokens=ref_tokens_tensor,
                        audio_16k=audio_16k
                    ):
                        new_audio_chunk = audio_chunk[:, -window_size:]
                        audio_chunks.append(new_audio_chunk)
                        
                        # Yield chunk result
                        yield Taste2OutputWithPast(
                            audio_chunk=new_audio_chunk,
                            chunk_id=chunk_id,
                            s3_tokens=s3_tokens_tensor,
                            asr_text=asr_text,
                            is_final=False
                        )
                        
                    print(f"Audio chunk {chunk_id} generated successfully")
                    
                except Exception as e:
                    print(f"Error generating audio chunk {chunk_id}: {e}")
                    continue
                
                # Clear current s3 token buffer
                s3_tokens_buffer.clear()
                start_time = time.time()
        
        # Process remaining tokens as final chunk
        if s3_tokens_buffer:
            chunk_id += 1
            print(f"\\n--- Processing final audio chunk {chunk_id} ({len(s3_tokens_buffer)} tokens) ---")
            
            s3_tokens_tensor = torch.tensor(s3_tokens_buffer, dtype=torch.long).unsqueeze(0)
            ref_tokens_tensor = torch.tensor(all_s3_tokens[:-len(s3_tokens_buffer)], dtype=torch.long).unsqueeze(0)
            
            try:
                for audio_chunk in self._postprocess_streaming(
                    s3_tokens=s3_tokens_tensor,
                    ref_s3_tokens=ref_tokens_tensor,
                    audio_16k=audio_16k
                ):
                    final_window_size = int(window_size / chunk_size * len(s3_tokens_buffer))
                    new_audio_chunk = audio_chunk[:, -final_window_size:]
                    audio_chunks.append(new_audio_chunk)
                    
                    # Yield final chunk
                    yield Taste2OutputWithPast(
                        audio_chunk=new_audio_chunk,
                        chunk_id=chunk_id,
                        s3_tokens=s3_tokens_tensor,
                        asr_text=asr_text,
                        is_final=True
                    )
                
                print(f"Final audio chunk {chunk_id} generated successfully")
                
            except Exception as e:
                print(f"Error generating final audio chunk: {e}")
        
        # Generate audio from ALL s3 tokens at once for comparison
        if all_s3_tokens:
            print(f"\\n=== Generating audio from ALL {len(all_s3_tokens)} s3 tokens at once ===")
            all_s3_tokens_tensor = torch.tensor(all_s3_tokens, dtype=torch.long).unsqueeze(0)
            
            try:
                single_audio = self._postprocess(all_s3_tokens_tensor, audio_16k)
                
                # Yield the single audio result
                yield Taste2OutputWithPast(
                    single_audio=single_audio,
                    all_s3_tokens=all_s3_tokens,
                    total_tokens=len(all_s3_tokens),
                    asr_text=asr_text,
                    is_single_audio=True
                )
                
            except Exception as e:
                print(f"Error generating single audio from all tokens: {e}")
        
        # Combine all audio chunks with crossfading
        if audio_chunks:
            print(f"\\n=== Combining {len(audio_chunks)} audio chunks with crossfading ===")
            final_audio = self._crossfade_audio_chunks(
                audio_chunks, 
                crossfade_samples=int(0.01 * self.config.sample_rate)  # 10ms crossfade
            )
            
            yield Taste2OutputWithPast(
                final_audio=final_audio,
                total_chunks=len(audio_chunks),
                asr_text=asr_text,
                is_complete=True
            )
        else:
            print("Warning: No audio chunks were generated")


@dataclass
class Taste2TokenizationOutput:
    """Output from the tokenization process."""
    text_token_ids: torch.Tensor
    text_token_len: torch.Tensor
    taste_token_embs: torch.Tensor
    asr_text: str


@dataclass  
class Taste2SLMGenerationOutput:
    """Output from the SLM generation process."""
    generated_text_token_ids: torch.Tensor
    generated_taste_token_embs: torch.Tensor
    generated_text: str
    is_final: bool
    step: int


@dataclass
class Taste2DetokenizationOutput:
    """Output from the detokenization process."""
    audio_chunk: Optional[torch.FloatTensor] = None
    s3_tokens: Optional[torch.LongTensor] = None
    chunk_id: Optional[int] = None
    is_final: Optional[bool] = None
    single_audio: Optional[torch.FloatTensor] = None
    final_audio: Optional[torch.FloatTensor] = None
    total_tokens: Optional[int] = None
    all_s3_tokens: Optional[List] = None
    is_complete: Optional[bool] = None