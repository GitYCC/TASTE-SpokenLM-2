"""
Streaming generation for TASTE2-SpokenLM using the separated SLM generation process.
"""

import torch
from typing import List, Dict, Iterator, Optional, TYPE_CHECKING

from .segment import TASTESegment, StreamingResult

if TYPE_CHECKING:
    from ..modeling_taste2 import Taste2ForCausalLM


def streaming_generate(
    model: "Taste2ForCausalLM",
    input_segments: List[TASTESegment],
    min_len: int = 3,
    max_len: int = 512,
    sampling: int = 25
) -> Iterator[Dict]:
    """
    Streaming generation using TASTE2 SLM for text tokens and taste embeddings.
    
    This function takes TASTESegment inputs, extracts text_ids and taste_ids,
    and uses the TASTE2 model's SLM generation process to generate new tokens.
    
    Args:
        model: Taste2ForCausalLM model with SLM generation capabilities
        input_segments: List of TASTESegment containing text_ids and taste_ids
        min_len: Minimum generation length (default: 3)
        max_len: Maximum generation length (default: 512)
        sampling: Sampling parameter for generation (default: 25)
        
    Yields:
        Dict: StreamingResult with keys:
            - is_complete: bool indicating if this is the final result
            - completion_reason: str reason for completion ('step' or 'final')
            - segment: TASTESegment containing generated content
    """
    
    if not input_segments:
        raise ValueError("input_segments cannot be empty")
    
    print(f"TASTE2 streaming generation with {len(input_segments)} input segments...")
    
    # Extract text_ids and taste_ids from input segments
    # For simplicity, we'll use the first segment that has both text_ids and taste_ids
    # In practice, you might want to concatenate multiple segments
    
    text_token_ids = None
    taste_token_embs = None
    source_segment = None
    
    for segment in input_segments:
        if segment.text_ids is not None and segment.taste_ids is not None:
            text_token_ids = segment.text_ids  # Shape: (1, seq_len)
            taste_token_embs = segment.taste_ids  # Shape: (1, seq_len, embed_dim)
            source_segment = segment
            break
    
    if text_token_ids is None or taste_token_embs is None:
        raise ValueError("No input segment found with both text_ids and taste_ids")
    
    print(f"Using segment: role='{source_segment.role}', modality='{source_segment.modality}'")
    print(f"Input shapes: text_ids={text_token_ids.shape}, taste_ids={taste_token_embs.shape}")
    
    # Validate input tensor shapes
    if text_token_ids.ndim != 2 or text_token_ids.size(0) != 1:
        raise ValueError(f"text_ids must have shape (1, seq_len), got {text_token_ids.shape}")
    
    if taste_token_embs.ndim != 3 or taste_token_embs.size(0) != 1:
        raise ValueError(f"taste_ids must have shape (1, seq_len, embed_dim), got {taste_token_embs.shape}")
    
    if text_token_ids.size(1) != taste_token_embs.size(1):
        raise ValueError(f"Sequence length mismatch: text_ids={text_token_ids.size(1)}, taste_ids={taste_token_embs.size(1)}")
    
    # Create text_token_len tensor
    text_token_len = torch.tensor([text_token_ids.size(1)], dtype=torch.int32)
    
    device = model.device
    
    # Move inputs to device
    text_token_ids = text_token_ids.to(device)
    text_token_len = text_token_len.to(device)
    taste_token_embs = taste_token_embs.to(device)
    
    print(f"Generation parameters: min_len={min_len}, max_len={max_len}, sampling={sampling}")
    
    # Use TASTE2 model's SLM generation process
    with torch.no_grad():
        try:
            # Get the SLM generation iterator
            slm_generation_results = model.slm_generate(
                text_token_ids=text_token_ids,
                text_token_len=text_token_len,
                taste_token_embs=taste_token_embs,
                min_len=min_len,
                max_len=max_len,
                sampling=sampling
            )
            
            step_count = 0
            
            # Stream the individual generation results
            for generation_result in slm_generation_results:
                step_count += 1
                
                if generation_result.is_final:
                    # Create final TASTESegment with complete generated sequence
                    final_segment = TASTESegment(
                        role='assistant',
                        modality='audio',  # Generated content is audio modality
                        text_ids=generation_result.generated_text_token_ids,  # Shape: (1, total_len)
                        taste_ids=generation_result.generated_taste_token_embs  # Shape: (1, total_len, embed_dim)
                    )
                    
                    yield {
                        'is_complete': True,
                        'completion_reason': 'final',
                        'segment': final_segment
                    }
                    break
                else:
                    # Create intermediate TASTESegment with individual token
                    step_segment = TASTESegment(
                        role='assistant',
                        modality='audio',
                        text_ids=generation_result.generated_text_token_ids,  # Shape: (1, 1)
                        taste_ids=generation_result.generated_taste_token_embs  # Shape: (1, 1, embed_dim)
                    )
                    
                    yield {
                        'is_complete': False,
                        'completion_reason': 'step',
                        'segment': step_segment
                    }
            
            print(f"TASTE2 streaming generation completed after {step_count} steps")
            
        except Exception as e:
            print(f"Error during TASTE2 SLM generation: {e}")
            raise


def streaming_generate_from_tokenized_input(
    model: "Taste2ForCausalLM",
    text_token_ids: torch.Tensor,
    taste_token_embs: torch.Tensor,
    min_len: int = 3,
    max_len: int = 256,
    sampling: int = 25
) -> Iterator[Dict]:
    """
    Streaming generation with direct tensor inputs (for convenience).
    
    This is a convenience function that wraps the tensor inputs in a TASTESegment
    and calls the main streaming_generate function.
    
    Args:
        model: Taste2ForCausalLM model with SLM generation capabilities
        text_token_ids: Text token IDs tensor of shape (1, seq_len)
        taste_token_embs: TASTE token embeddings of shape (1, seq_len, embed_dim)
        min_len: Minimum generation length (default: 3)
        max_len: Maximum generation length (default: 256)
        sampling: Sampling parameter for generation (default: 25)
        
    Yields:
        Dict: StreamingResult with the same format as streaming_generate
    """
    
    # Create a TASTESegment from the tensor inputs
    input_segment = TASTESegment(
        role='user',
        modality='audio',
        text_ids=text_token_ids,
        taste_ids=taste_token_embs
    )
    
    # Use the main streaming generation function
    yield from streaming_generate(
        model=model,
        input_segments=[input_segment],
        min_len=min_len,
        max_len=max_len,
        sampling=sampling
    )