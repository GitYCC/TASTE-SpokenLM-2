"""
TASTE2 tokenization: Convert audio waveform to TASTE tokens aligned with text.
"""

import torch
import torchaudio
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..modeling_taste2 import Taste2ForCausalLM


def taste_tokenize(
    model: "Taste2ForCausalLM",
    audio: torch.Tensor,
    token_ids: torch.Tensor,
    sampling_rate: int = 16000
) -> torch.Tensor:
    """
    Convert audio waveform to TASTE token embeddings aligned with provided text tokens.
    
    This function performs only TASTE tokenization using the audio and pre-computed text tokens.
    It does NOT perform ASR or text tokenization - those should be done beforehand.
    
    Args:
        model: Taste2ForCausalLM model with tokenization capabilities
        audio: Input audio waveform tensor of shape (1, num_samples) 
        token_ids: Text token IDs tensor of shape (1, seq_len)
        sampling_rate: Input audio sampling rate in Hz (will be resampled to 16000 if different)
    
    Returns:
        torch.Tensor: TASTE token embeddings of shape (1, seq_len, embed_dim)
        
    Raises:
        ValueError: If audio or token_ids have incorrect shapes
        AssertionError: If batch sizes don't match
    """
    
    # Validate inputs
    if not isinstance(audio, torch.Tensor):
        raise TypeError("audio must be a torch.Tensor")
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    
    if audio.ndim != 2:
        raise ValueError("audio must have shape (1, num_samples)")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (1, seq_len)")
    
    # Ensure batch size alignment
    assert audio.size(0) == token_ids.size(0) == 1, "Batch size must be 1 for both audio and token_ids"
    
    device = model.device
    
    # Resample audio to model's expected sampling rate (16000 Hz) if needed
    target_sr = 16000
    if sampling_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        audio = resampler(audio)
    
    # Move inputs to model device
    audio = audio.to(device)
    token_ids = token_ids.to(device)
    
    with torch.no_grad():
        # Extract audio features using the model's audio extractor
        audio_feature, audio_feature_len = model.audio_extractor(audio, [audio.shape[-1]])
        
        # Convert to model dtype if using fp16
        if model.config.fp16:
            audio_feature = audio_feature.half()
        
        # Move to device
        audio_feature = audio_feature.to(device)
        token_len = torch.tensor([token_ids.shape[1]], dtype=torch.int32).to(device)
        
        # Use taste_stage1 tokenizer to get TASTE embeddings
        taste_tokenizer = model.taste_stage1.taste_tokenizer
        tokenized = taste_tokenizer(token_ids, token_len, audio_feature, audio_feature_len)
        taste_token_emb = tokenized['taste_token_emb']
        
        # Ensure output shape matches input token sequence length
        if taste_token_emb.size(1) != token_ids.size(1):
            raise ValueError(
                f"Sequence length mismatch: taste_token_emb has {taste_token_emb.size(1)} tokens "
                f"but token_ids has {token_ids.size(1)} tokens. This indicates an alignment "
                f"problem between audio and text that cannot be automatically corrected."
            )
        
        return taste_token_emb


def taste_tokenize_with_text_tokens(
    model: "Taste2ForCausalLM",
    audio: torch.Tensor,
    text: Optional[str] = None,
    asr_model_dir: Optional[str] = None,
    sampling_rate: int = 16000
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Convert audio waveform to TASTE tokens and return both text tokens and TASTE embeddings.
    
    This is an extended version that returns all tokenization outputs for compatibility
    with downstream processes that need both text tokens and TASTE embeddings.
    
    Args:
        model: Taste2ForCausalLM model with tokenization capabilities
        audio: Input audio waveform tensor of shape (1, num_samples)
        text: Text transcript (optional - will use ASR if not provided)
        asr_model_dir: ASR model directory for text extraction (optional)
        sampling_rate: Input audio sampling rate in Hz (will be resampled to 16000 if different)
    
    Returns:
        tuple containing:
        - text_token_ids: Text token IDs tensor of shape (1, seq_len)
        - text_token_len: Text token length tensor of shape (1,)
        - taste_token_embs: TASTE token embeddings of shape (1, seq_len, embed_dim)
        - asr_text: The transcribed/provided text string
        
    Raises:
        ValueError: If audio has incorrect shape or neither text nor asr_model_dir is provided
        AssertionError: If batch size is not 1
    """
    
    # Validate inputs
    if not isinstance(audio, torch.Tensor):
        raise TypeError("audio must be a torch.Tensor")
    
    if audio.ndim != 2:
        raise ValueError("audio must have shape (1, num_samples)")
    
    if text is None and asr_model_dir is None:
        raise ValueError("Either text or asr_model_dir must be provided")
    
    # Ensure batch size is 1
    assert audio.size(0) == 1, "Batch size must be 1 for audio"
    
    # Resample audio to model's expected sampling rate (16000 Hz) if needed
    target_sr = 16000
    if sampling_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        audio = resampler(audio)
    
    # Use TASTE2 model's tokenization process
    with torch.no_grad():
        # Get tokenization result from TASTE2 model
        tokenization_results = list(model.tokenize(
            audio_16k=audio,
            text=text,
            asr_model_dir=asr_model_dir
        ))
        
        if not tokenization_results:
            raise RuntimeError("TASTE2 tokenization returned no results")
        
        # Extract all outputs
        tokenization_result = tokenization_results[0]
        
        return (
            tokenization_result.text_token_ids,
            tokenization_result.text_token_len,
            tokenization_result.taste_token_embs,
            tokenization_result.asr_text
        )