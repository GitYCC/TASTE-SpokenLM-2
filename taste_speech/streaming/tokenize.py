"""
TASTE tokenization: Convert audio waveform to TASTE tokens aligned with text.
"""

import torch
import torchaudio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..modeling_taste import TasteForCausalLM
    from ..processing_taste import TasteProcessor


def taste_tokenize(
    model: "TasteForCausalLM",
    processor: "TasteProcessor", 
    audio: torch.Tensor,
    token_ids: torch.Tensor,
    word_ids: torch.Tensor,
    sampling_rate: int = 16000
) -> torch.Tensor:
    """
    Convert audio waveform to TASTE token indices aligned with text tokens.
    
    This function extracts vector quantized (VQ) representations from audio
    that are temporally aligned with the provided text token sequence.
    
    Args:
        model: TasteForCausalLM model with audio_tower for VQ extraction
        processor: TasteProcessor with feature extraction capabilities
        audio: Input audio waveform tensor of shape (1, num_samples) 
        token_ids: Text token IDs tensor of shape (1, seq_len)
        word_ids: Word ID tensor for word-level alignment (1, seq_len). Required for joint encoder segmenter.
        sampling_rate: Input audio sampling rate in Hz (will be resampled to 16000 if different)
    
    Returns:
        torch.Tensor: TASTE token indices of shape (1, seq_len, vq_dim)
        
    Raises:
        AssertionError: If batch sizes of audio and token_ids don't match
        ValueError: If audio or token_ids have incorrect shapes
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
    
    # Ensure batch size alignment (critical requirement from PRP)
    assert audio.size(0) == token_ids.size(0) == 1, "Batch size must be 1 for both audio and token_ids"
    
    device = model.device
    dtype = next(model.parameters()).dtype
    
    # Resample audio to model's expected sampling rate (16000 Hz) if needed
    target_sr = 16000
    if sampling_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        audio = resampler(audio)
    
    # Move inputs to model device
    audio = audio.to(device)
    token_ids = token_ids.to(device)
    
    with torch.no_grad():
        # Step 1: Extract audio features using the processor's feature extractor
        # Convert audio to numpy for WhisperFrontend compatibility
        audio_np = audio.cpu().numpy()[0]  # Remove batch dimension for processor
        
        # Use processor's whisper feature extractor
        audio_features, audio_feature_lengths = processor.whisper_feature_extractor(
            torch.tensor([audio_np], dtype=torch.float32), [audio_np.shape[0]]
        )
        
        # Convert to tensors and move to device
        audio_features = audio_features.clone().detach().to(dtype=dtype, device=device)
        audio_feature_lengths = audio_feature_lengths.clone().detach().to(dtype=torch.long, device=device)
        
        # Step 2: Prepare ASR token inputs (use token_ids as ASR tokens for alignment)
        asr_token_ids = token_ids
        asr_token_lengths = torch.tensor([token_ids.shape[1]], dtype=torch.long, device=device)
        
        # Step 3: Use audio_tower to encode and quantize audio features
        # Following the pattern from extract_vq method
        audio_encoded = model.audio_tower(
            asr_token_ids=asr_token_ids,
            asr_token_lengths=asr_token_lengths,
            audio_features=audio_features,
            audio_feature_lengths=audio_feature_lengths,
            asr_word_ids=word_ids,
        )
        
        # Step 4: Extract quantized indices (TASTE tokens)
        if 'quantized_indices' not in audio_encoded:
            raise ValueError("Model audio_tower does not have quantization enabled")
        taste_indices = audio_encoded['quantized_indices']

        # Ensure output shape matches input token sequence length
        if taste_indices.size(1) != token_ids.size(1):
            raise ValueError(
                f"Sequence length mismatch: taste_indices has {taste_indices.size(1)} tokens "
                f"but token_ids has {token_ids.size(1)} tokens. This indicates an alignment "
                f"problem between audio and text that cannot be automatically corrected."
            )
    
    return taste_indices