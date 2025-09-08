"""
Configuration class for TASTE2 models (Stage 2 SLM only).
"""

import os
import numpy as np
from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class Taste2Config(PretrainedConfig):
    """
    Configuration class for TASTE2 model (Stage 2 SLM pipeline only).
    
    This configuration is specifically designed for Stage 2 with SLM,
    based on generate_audio_stream.py patterns.
    """
    model_type = "taste2"
    
    def __init__(
        self,
        # Core model parameters (Stage 2 SLM only)
        model_dir="/mnt/shared/NTU_TASLM/models/taste2_8B_final",  # Model directory path
        fp16=False,                     # Half precision inference
        
        # Model loading optimizations
        load_jit=False,                 # JIT compilation optimization
        load_trt=False,                 # TensorRT optimization
        load_vllm=False,                # VLLM optimization
        trt_concurrent=1,               # TensorRT concurrency setting
        
        # Audio parameters
        sample_rate=24000,              # Audio sampling rate
        device="cuda",                  # Device ('cuda' or 'cpu')
        
        # Frontend configuration files (expected in model_dir)
        campplus_model="campplus.onnx",
        speech_tokenizer_model="speech_tokenizer_v2.onnx", 
        speaker_info="spk2info.pt",
        
        # Model checkpoint files (expected in model_dir) - All required for Stage 2 SLM
        llm_checkpoint="llm.pt",
        flow_checkpoint="flow.pt", 
        hift_checkpoint="hift.pt",
        slm_checkpoint="slm.pt",        # SLM always required
        
        # YAML configuration file (expected in model_dir)  
        config_file="taste2_stage2.yaml",           # Stage 2 config file
        qwen_pretrain_subdir="CosyVoice-BlankEN",   # Subdirectory for Qwen model
        
        # Streaming parameters (from generate_audio_stream.py)
        token_hop_len=25,               # Token frame rate
        mel_cache_len=8,                # Mel spectrogram cache length
        source_cache_len=None,          # Calculated as mel_cache_len * 480
        chunk_size=20,                  # Streaming chunk size (cannot exceed 50)
        
        # Bistream inference parameters
        window_size_calc=True,          # Calculate window size automatically
        crossfade_samples=400,          # Crossfade samples for smooth transitions (25ms at 16kHz)
        
        # Additional optimizations directories (optional, in model_dir)
        vllm_dir="vllm",
        jit_flow_encoder_template="flow.encoder.{precision}.zip",  # precision = fp16/fp32
        trt_flow_decoder_template="flow.decoder.estimator.{precision}.mygpu.plan",
        trt_flow_onnx="flow.decoder.estimator.fp32.onnx",
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Core settings (Stage 2 SLM fixed)
        self.model_dir = model_dir
        self.fp16 = fp16
        
        # Optimization flags
        self.load_jit = load_jit
        self.load_trt = load_trt
        self.load_vllm = load_vllm
        self.trt_concurrent = trt_concurrent
        
        # Audio settings
        self.sample_rate = sample_rate
        self.device = device
        
        # File paths (relative to model_dir)
        self.campplus_model = campplus_model
        self.speech_tokenizer_model = speech_tokenizer_model
        self.speaker_info = speaker_info
        
        # Checkpoint files (all required for Stage 2 SLM)
        self.llm_checkpoint = llm_checkpoint
        self.flow_checkpoint = flow_checkpoint
        self.hift_checkpoint = hift_checkpoint
        self.slm_checkpoint = slm_checkpoint
        
        # Configuration files
        self.config_file = config_file
        self.qwen_pretrain_subdir = qwen_pretrain_subdir
        
        # Streaming parameters
        self.token_hop_len = token_hop_len
        self.mel_cache_len = mel_cache_len
        self.source_cache_len = source_cache_len if source_cache_len else mel_cache_len * 480
        self.chunk_size = chunk_size
        self.window_size_calc = window_size_calc
        self.crossfade_samples = crossfade_samples
        
        # Optimization directories
        self.vllm_dir = vllm_dir
        self.jit_flow_encoder_template = jit_flow_encoder_template
        self.trt_flow_decoder_template = trt_flow_decoder_template
        self.trt_flow_onnx = trt_flow_onnx
    
    def get_config_file_path(self):
        """Get the YAML config file path for Stage 2."""
        return os.path.join(self.model_dir, self.config_file)
    
    def get_checkpoint_path(self, checkpoint_name):
        """Get the full path for a checkpoint file."""
        checkpoint_file = getattr(self, f"{checkpoint_name}_checkpoint")
        return os.path.join(self.model_dir, checkpoint_file)
    
    def get_frontend_file_path(self, file_type):
        """Get the full path for frontend files (campplus, speech_tokenizer, etc.)."""
        file_name = getattr(self, file_type)
        return os.path.join(self.model_dir, file_name)
    
    def get_qwen_pretrain_path(self):
        """Get the Qwen pretrained model path."""
        return os.path.join(self.model_dir, self.qwen_pretrain_subdir)
    
    def get_precision_string(self):
        """Get precision string for file templates."""
        return "fp16" if self.fp16 else "fp32"
    
    def get_window_size(self, flow_input_frame_rate=25):
        """Calculate window size for streaming (from generate_audio_stream.py)."""
        return int(self.chunk_size / flow_input_frame_rate * self.sample_rate)
    
    def get_speech_window(self):
        """Get hamming window for speech processing."""
        return np.hamming(2 * self.source_cache_len)
    
    def get_optimization_paths(self):
        """Get paths for optimization files if they should be loaded."""
        paths = {}
        
        if self.load_vllm:
            paths['vllm'] = os.path.join(self.model_dir, self.vllm_dir)
        
        if self.load_jit:
            precision = self.get_precision_string()
            jit_file = self.jit_flow_encoder_template.format(precision=precision)
            paths['jit'] = os.path.join(self.model_dir, jit_file)
        
        if self.load_trt:
            precision = self.get_precision_string()
            trt_plan = self.trt_flow_decoder_template.format(precision=precision)
            paths['trt_plan'] = os.path.join(self.model_dir, trt_plan)
            paths['trt_onnx'] = os.path.join(self.model_dir, self.trt_flow_onnx)
        
        return paths
    
    def validate_model_directory(self):
        """Validate that the model directory contains required files for Stage 2 SLM."""
        required_files = []
        
        # YAML config file
        required_files.append(self.get_config_file_path())
        
        # All checkpoint files (required for Stage 2 SLM)
        required_files.append(self.get_checkpoint_path('llm'))
        required_files.append(self.get_checkpoint_path('flow'))
        required_files.append(self.get_checkpoint_path('hift'))
        required_files.append(self.get_checkpoint_path('slm'))  # Always required
        
        # Frontend files
        required_files.append(self.get_frontend_file_path('campplus_model'))
        required_files.append(self.get_frontend_file_path('speech_tokenizer_model'))
        required_files.append(self.get_frontend_file_path('speaker_info'))
        
        # Qwen pretrained directory
        required_files.append(self.get_qwen_pretrain_path())
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        return missing_files


# Register the configuration
CONFIG_MAPPING.register("taste2", Taste2Config)