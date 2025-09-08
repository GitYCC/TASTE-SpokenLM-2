"""
TASTE2 detokenization: Convert generated text tokens and taste embeddings to audio waveforms.
"""

import torch
from typing import Dict, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..modeling_taste2 import Taste2ForCausalLM


def taste_detokenize(
    model: "Taste2ForCausalLM",
    generated_text_token_ids: torch.Tensor,
    generated_taste_token_embs: torch.Tensor,
    original_audio_16k: torch.Tensor,
    streaming: bool = True,
    chunk_size: Optional[int] = None
) -> Iterator[Dict]:
    """
    Convert generated text tokens and TASTE embeddings to audio waveforms.
    
    This function performs the TASTE2 detokenization process:
    1. Uses LLM inference to generate s3 tokens from text tokens and TASTE embeddings
    2. Performs streaming TTS using s3 tokens to output audio waveform chunks
    
    Args:
        model: Taste2ForCausalLM model with detokenization capabilities
        generated_text_token_ids: Generated text token IDs of shape (1, seq_len)
        generated_taste_token_embs: Generated TASTE embeddings of shape (1, seq_len, embed_dim)
        original_audio_16k: Original audio for context of shape (1, num_samples)
        streaming: Whether to use streaming mode (default: True)
        chunk_size: Size of chunks for streaming (default: uses model's chunk_size)
        
    Yields:
        Dict containing:
            - audio_waveform: Generated audio tensor of shape (1, T)
            - sampling_rate: Audio sampling rate (24000 Hz)
            - chunk_duration_ms: Duration of audio chunk in milliseconds
            - speech_ids: Generated speech token IDs (s3 tokens)
    """
    
    # Validate inputs
    if not isinstance(generated_text_token_ids, torch.Tensor):
        raise TypeError("generated_text_token_ids must be a torch.Tensor")
    if not isinstance(generated_taste_token_embs, torch.Tensor):
        raise TypeError("generated_taste_token_embs must be a torch.Tensor")
    if not isinstance(original_audio_16k, torch.Tensor):
        raise TypeError("original_audio_16k must be a torch.Tensor")
    
    if generated_text_token_ids.ndim != 2 or generated_text_token_ids.size(0) != 1:
        raise ValueError(f"generated_text_token_ids must have shape (1, seq_len), got {generated_text_token_ids.shape}")
    
    if generated_taste_token_embs.ndim != 3 or generated_taste_token_embs.size(0) != 1:
        raise ValueError(f"generated_taste_token_embs must have shape (1, seq_len, embed_dim), got {generated_taste_token_embs.shape}")
    
    if generated_text_token_ids.size(1) != generated_taste_token_embs.size(1):
        raise ValueError(f"Sequence length mismatch: text_tokens={generated_text_token_ids.size(1)}, taste_embs={generated_taste_token_embs.size(1)}")
    
    if original_audio_16k.ndim != 2 or original_audio_16k.size(0) != 1:
        raise ValueError(f"original_audio_16k must have shape (1, num_samples), got {original_audio_16k.shape}")
    
    device = model.device
    
    # Move inputs to device
    generated_text_token_ids = generated_text_token_ids.to(device)
    generated_taste_token_embs = generated_taste_token_embs.to(device)
    original_audio_16k = original_audio_16k.to(device)
    
    print(f"TASTE2 detokenization: streaming={streaming}")
    print(f"Input shapes: text_tokens={generated_text_token_ids.shape}, taste_embs={generated_taste_token_embs.shape}")
    
    # Use TASTE2 model's detokenization process
    with torch.no_grad():
        try:
            # Get the detokenization iterator from TASTE2 model
            detokenization_results = model.detokenize(
                generated_text_token_ids=generated_text_token_ids,
                generated_taste_token_embs=generated_taste_token_embs,
                audio_16k=original_audio_16k,
                streaming=streaming,
                chunk_size=chunk_size
            )
            
            chunk_count = 0
            
            # Stream the detokenization results
            for detokenization_result in detokenization_results:
                result_dict = {}
                
                # Handle streaming results (audio chunks)
                if hasattr(detokenization_result, 'audio_chunk') and detokenization_result.audio_chunk is not None:
                    chunk_count += 1
                    result_dict = {
                        'audio_waveform': detokenization_result.audio_chunk,
                        'sampling_rate': 24000,  # TASTE2 uses 24kHz
                        'chunk_duration_ms': int(detokenization_result.audio_chunk.shape[1] * 1000 / 24000),
                        'speech_ids': detokenization_result.s3_tokens
                    }
                
                # Handle single audio result (non-streaming)
                elif hasattr(detokenization_result, 'single_audio') and detokenization_result.single_audio is not None:
                    result_dict = {
                        'audio_waveform': detokenization_result.single_audio,
                        'sampling_rate': 24000,
                        'chunk_duration_ms': int(detokenization_result.single_audio.shape[1] * 1000 / 24000),
                        'speech_ids': detokenization_result.all_s3_tokens
                    }
                
                # Skip other results (like final_audio) - only return audio_waveform format
                else:
                    continue
                
                yield result_dict
                
                # Break if complete
                if result_dict.get('is_complete', False):
                    break
            
            print(f"TASTE2 detokenization completed with {chunk_count} chunks")
            
        except Exception as e:
            print(f"Error during TASTE2 detokenization: {e}")
            raise


def taste_detokenize_from_segment(
    model: "Taste2ForCausalLM",
    generated_segment,  # TASTESegment
    original_audio_16k: torch.Tensor,
    streaming: bool = True,
    chunk_size: Optional[int] = None
) -> Iterator[Dict]:
    """
    Convert generated TASTESegment to audio waveforms (convenience function).
    
    This is a convenience function that extracts text_ids and taste_ids from
    a TASTESegment and calls the main taste_detokenize function.
    
    Args:
        model: Taste2ForCausalLM model with detokenization capabilities
        generated_segment: TASTESegment containing generated text_ids and taste_ids
        original_audio_16k: Original audio for context of shape (1, num_samples)
        streaming: Whether to use streaming mode (default: True)
        chunk_size: Size of chunks for streaming (default: uses model's chunk_size)
        
    Yields:
        Dict: Same format as taste_detokenize
    """
    
    if not hasattr(generated_segment, 'text_ids') or generated_segment.text_ids is None:
        raise ValueError("Generated segment must have text_ids")
    
    if not hasattr(generated_segment, 'taste_ids') or generated_segment.taste_ids is None:
        raise ValueError("Generated segment must have taste_ids")
    
    print(f"Detokenizing segment: role='{generated_segment.role}', modality='{generated_segment.modality}'")
    print(f"Segment shapes: text_ids={generated_segment.text_ids.shape}, taste_ids={generated_segment.taste_ids.shape}")
    
    # Use the main detokenization function
    yield from taste_detokenize(
        model=model,
        generated_text_token_ids=generated_segment.text_ids,
        generated_taste_token_embs=generated_segment.taste_ids,
        original_audio_16k=original_audio_16k,
        streaming=streaming,
        chunk_size=chunk_size
    )