"""
TASTE detokenization: Convert TASTE tokens to audio waveforms with context continuity.
"""

import torch
import torchaudio
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..modeling_taste import TasteForCausalLM
    from ..processing_taste import TasteProcessor


def _validate_detokenize_inputs(
    speaker_embeds: torch.Tensor,
    asr_token_ids: torch.Tensor,
    asr_word_ids: torch.Tensor,
    asr_taste_ids: torch.Tensor,
    prev_asr_token_ids: Optional[torch.Tensor] = None,
    prev_asr_word_ids: Optional[torch.Tensor] = None,
    prev_asr_taste_ids: Optional[torch.Tensor] = None,
    prev_speech_ids: Optional[torch.Tensor] = None,
) -> None:
    """Validate inputs for taste_detokenize function."""
    
    # Validate required tensors
    if not isinstance(speaker_embeds, torch.Tensor):
        raise TypeError("speaker_embeds must be a torch.Tensor")
    if not isinstance(asr_token_ids, torch.Tensor):
        raise TypeError("asr_token_ids must be a torch.Tensor")
    if not isinstance(asr_word_ids, torch.Tensor):
        raise TypeError("asr_word_ids must be a torch.Tensor")
    if not isinstance(asr_taste_ids, torch.Tensor):
        raise TypeError("asr_taste_ids must be a torch.Tensor")
    
    # Validate shapes
    if speaker_embeds.dim() != 2 or speaker_embeds.size(0) != 1:
        raise ValueError("speaker_embeds must have shape (1, embed_dim)")
    if asr_token_ids.dim() != 2 or asr_token_ids.size(0) != 1:
        raise ValueError("asr_token_ids must have shape (1, seq_len)")
    if asr_word_ids.dim() != 2 or asr_word_ids.size(0) != 1:
        raise ValueError("asr_word_ids must have shape (1, seq_len)")
    if asr_taste_ids.dim() != 3 or asr_taste_ids.size(0) != 1:
        raise ValueError("asr_taste_ids must have shape (1, seq_len, vq_dim)")
    
    # Validate sequence length consistency
    seq_len = asr_token_ids.size(1)
    if asr_word_ids.size(1) != seq_len:
        raise ValueError("asr_token_ids and asr_word_ids must have same sequence length")
    if asr_taste_ids.size(1) != seq_len:
        raise ValueError("asr_token_ids and asr_taste_ids must have same sequence length")
    
    # Validate optional previous tensors if provided
    if prev_asr_token_ids is not None:
        if not isinstance(prev_asr_token_ids, torch.Tensor):
            raise TypeError("prev_asr_token_ids must be a torch.Tensor")
        if prev_asr_token_ids.dim() != 2 or prev_asr_token_ids.size(0) != 1:
            raise ValueError("prev_asr_token_ids must have shape (1, prev_seq_len)")
            
    if prev_asr_word_ids is not None:
        if not isinstance(prev_asr_word_ids, torch.Tensor):
            raise TypeError("prev_asr_word_ids must be a torch.Tensor")
        if prev_asr_word_ids.dim() != 2 or prev_asr_word_ids.size(0) != 1:
            raise ValueError("prev_asr_word_ids must have shape (1, prev_seq_len)")
            
    if prev_asr_taste_ids is not None:
        if not isinstance(prev_asr_taste_ids, torch.Tensor):
            raise TypeError("prev_asr_taste_ids must be a torch.Tensor")
        if prev_asr_taste_ids.dim() != 3 or prev_asr_taste_ids.size(0) != 1:
            raise ValueError("prev_asr_taste_ids must have shape (1, prev_seq_len, vq_dim)")
            
    if prev_speech_ids is not None:
        if not isinstance(prev_speech_ids, torch.Tensor):
            raise TypeError("prev_speech_ids must be a torch.Tensor")
        if prev_speech_ids.dim() != 2 or prev_speech_ids.size(0) != 1:
            raise ValueError("prev_speech_ids must have shape (1, prev_speech_len)")
    
    # Validate consistency between previous tensors
    if prev_asr_token_ids is not None and prev_asr_word_ids is not None:
        if prev_asr_token_ids.size(1) != prev_asr_word_ids.size(1):
            raise ValueError("prev_asr_token_ids and prev_asr_word_ids must have same sequence length")
            
    if prev_asr_token_ids is not None and prev_asr_taste_ids is not None:
        if prev_asr_token_ids.size(1) != prev_asr_taste_ids.size(1):
            raise ValueError("prev_asr_token_ids and prev_asr_taste_ids must have same sequence length")
    
    # Validate VQ dimension consistency
    if prev_asr_taste_ids is not None:
        if prev_asr_taste_ids.size(2) != asr_taste_ids.size(2):
            raise ValueError("prev_asr_taste_ids and asr_taste_ids must have same VQ dimension")


def taste_detokenize(
    model: "TasteForCausalLM",
    processor: "TasteProcessor",
    speaker_embeds: torch.Tensor,
    asr_token_ids: torch.Tensor,
    asr_word_ids: torch.Tensor,
    asr_taste_ids: torch.Tensor,
    prev_asr_token_ids: Optional[torch.Tensor] = None,
    prev_asr_word_ids: Optional[torch.Tensor] = None,
    prev_asr_taste_ids: Optional[torch.Tensor] = None,
    prev_speech_ids: Optional[torch.Tensor] = None,
    prev_audio_ms: int = 0,
    out_sampling_rate: int = 16000,
) -> Dict:
    """
    Convert TASTE tokens to audio waveforms with context-aware synthesis.
    
    Synthesizes audio from TASTE tokens while maintaining continuity with previous
    audio segments. Supports incremental audio generation by properly handling
    context from previous segments.
    
    Args:
        model: TasteForCausalLM model with speech decoding capabilities
        processor: TasteProcessor containing VoiceGenerator
        speaker_embeds: Speaker embedding tensor of shape (1, embed_dim)
        asr_token_ids: Current ASR token sequence of shape (1, seq_len)
        asr_word_ids: Word IDs for current ASR text of shape (1, seq_len)
        asr_taste_ids: Current TASTE tokens of shape (1, seq_len, vq_dim)
        prev_asr_token_ids: Previous ASR token sequence of shape (1, prev_seq_len) (optional)
        prev_asr_word_ids: Word IDs for previous ASR text (optional)
        prev_asr_taste_ids: Previous TASTE tokens of shape (1, prev_seq_len, vq_dim) (optional)
        prev_speech_ids: Previous speech token IDs of shape (1, prev_speech_len) (optional)
        prev_audio_ms: Duration of previous audio in milliseconds (default: 0)
        out_sampling_rate: Target output sampling rate (default: 16000Hz)
        
    Returns:
        Dict containing:
            - audio_waveform: Generated audio tensor of shape (1, T)
            - sampling_rate: Output sampling rate
            - chunk_duration_ms: Duration of current chunk in milliseconds
            - speech_ids: Generated speech token IDs
    """
    
    # Validate all inputs
    _validate_detokenize_inputs(
        speaker_embeds=speaker_embeds,
        asr_token_ids=asr_token_ids,
        asr_word_ids=asr_word_ids,
        asr_taste_ids=asr_taste_ids,
        prev_asr_token_ids=prev_asr_token_ids,
        prev_asr_word_ids=prev_asr_word_ids,
        prev_asr_taste_ids=prev_asr_taste_ids,
        prev_speech_ids=prev_speech_ids,
    )
    
    device = model.device

    with torch.no_grad():
        # Step 1: Build complete text and TASTE token sequences
        if prev_asr_token_ids is not None and prev_asr_taste_ids is not None and prev_asr_token_ids.numel() > 0:
            full_asr_token_ids = torch.cat([prev_asr_token_ids, asr_token_ids], dim=1).to(device)
            full_asr_taste_ids = torch.cat([prev_asr_taste_ids, asr_taste_ids], dim=1).to(device)

            # Adjust current word IDs to continue from previous max + 1
            if prev_asr_word_ids is not None and prev_asr_word_ids.numel() > 0:
                max_prev_word_id = prev_asr_word_ids.max().item()
                min_current_word_id = asr_word_ids.min().item()
                adjusted_asr_word_ids = asr_word_ids - min_current_word_id + max_prev_word_id + 1
                full_asr_word_ids = torch.cat([prev_asr_word_ids, adjusted_asr_word_ids], dim=1)
            else:
                raise Exception
            
        else:
            full_asr_token_ids = asr_token_ids.to(device)
            full_asr_taste_ids = asr_taste_ids.to(device)
            full_asr_word_ids = asr_word_ids.to(device)

        speaker_embeds = speaker_embeds.to(device)
        full_asr_token_lengths = torch.tensor([full_asr_token_ids.shape[1]], device=device, dtype=torch.long)
        
        # Step 3: Get audio unit embeddings from TASTE tokens
        vq_module = model.audio_tower.vq.rvq
        full_audio_unit_embeds, full_audio_unit_lengths = model.spoken_lm.get_audio_embeds_from_taste(
            vq_module, full_asr_token_lengths, full_asr_word_ids,
            asr_taste_indices=full_asr_taste_ids
        )
        full_audio_unit_embeds = full_audio_unit_embeds.to(device)
        full_audio_unit_lengths = full_audio_unit_lengths.to(device)

        # Step 4: Generate speech tokens using extended voice decoder
        speech_decoder_results = model.voice_decoder_generate(
            speaker_embeds=speaker_embeds,
            audio_unit_embeds=full_audio_unit_embeds,
            audio_unit_lengths=full_audio_unit_lengths,
            asr_token_ids=full_asr_token_ids,
            asr_token_lengths=full_asr_token_lengths,
            prev_speech_ids=prev_speech_ids,
        )

        generated_speech_tokens = speech_decoder_results['speech_token_ids']
        generated_speech_lengths = speech_decoder_results['speech_token_lengths']
        
        # Step 5: Generate audio using VoiceGenerator
        generator = processor.get_generator(device=device)
        flow_embedding = speaker_embeds.to(device)

        # Concatenate previous speech tokens with newly generated ones
        if prev_speech_ids is not None and prev_speech_ids.numel() > 0:
            prev_speech_ids = prev_speech_ids.to(device)
            speech_token_ids = torch.cat([prev_speech_ids, generated_speech_tokens], dim=1)
            prev_speech_length = prev_speech_ids.shape[1]
            speech_lengths = torch.tensor([prev_speech_length + generated_speech_lengths.item()], 
                                        device=device, dtype=torch.long)
        else:
            speech_token_ids = generated_speech_tokens
            speech_lengths = generated_speech_lengths
        
        tts_speech, original_sr = generator.inference(
            speech_token_ids=speech_token_ids,
            speech_token_lengths=speech_lengths,
            flow_embedding=flow_embedding
        )
        
        # Step 6: Resample to target sampling rate if needed
        if original_sr != out_sampling_rate:
            # Ensure tts_speech is on the same device as the resampler
            tts_speech = tts_speech.to(device)
            resampler = torchaudio.transforms.Resample(original_sr, out_sampling_rate).to(device)
            tts_speech = resampler(tts_speech)
        
        # Step 7: Extract current chunk by removing previous audio duration
        if prev_audio_ms > 0:
            # Calculate samples corresponding to previous audio duration
            prev_samples = int(prev_audio_ms * out_sampling_rate / 1000)
            # Crop to keep only current chunk
            tts_speech = tts_speech[:, prev_samples:]
        
        # Step 8: Calculate duration of current chunk
        chunk_duration_ms = int(tts_speech.shape[1] * 1000 / out_sampling_rate)
    
    return {
        'audio_waveform': tts_speech,
        'sampling_rate': out_sampling_rate,
        'chunk_duration_ms': chunk_duration_ms,
        'speech_ids': generated_speech_tokens,
    }
