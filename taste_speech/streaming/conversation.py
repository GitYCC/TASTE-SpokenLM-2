"""
Conversation state management for multimodal streaming generation.
"""

import torch
from typing import List, Tuple, TYPE_CHECKING
from .segment import TASTESegment

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


# Special tokens for conversation structure (from PRP specification)
SPECIAL_TOKENS = {
    'segment_start': "<|reserved_special_token_50|>",
    'segment_end': "<|reserved_special_token_51|>",
    'sys_role': "<|reserved_special_token_52|>",
    'user_role': "<|reserved_special_token_53|>",
    'assistant_role': "<|reserved_special_token_54|>",
}


def build_conditional_prompt(
    input_segments: List[TASTESegment], 
    llm_tokenizer: "PreTrainedTokenizer"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build conditional prompt from conversation segments for streaming generation.
    
    Constructs text_ids and taste_ids sequences with proper special tokens
    for role markers and segment boundaries to enable conditional generation.
    
    Args:
        input_segments: List of TASTESegment objects representing conversation history
        llm_tokenizer: Tokenizer for encoding special tokens and text
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (prompt_text_ids, prompt_taste_ids)
            - prompt_text_ids: Text token sequence with special tokens (1, total_seq_len)
            - prompt_taste_ids: Aligned TASTE token sequence (1, total_seq_len, vq_dim)
    """
    
    if not input_segments:
        # Return empty prompts for empty input
        device = torch.device("cpu")
        return (
            torch.empty(1, 0, dtype=torch.long, device=device),
            torch.empty(1, 0, 0, dtype=torch.long, device=device)
        )
    
    # Determine device and VQ dimension from first segment with taste_ids
    device = input_segments[0].text_ids.device
    vq_dim = None
    for segment in input_segments:
        if segment.taste_ids is not None:
            vq_dim = segment.taste_ids.size(2)
            break
    
    # Default VQ dimension if no audio segments found (from PRP: VQ_DIM is 4)
    if vq_dim is None:
        vq_dim = 4
    
    # Encode special tokens
    role_tokens = {}
    for role_key, token_str in SPECIAL_TOKENS.items():
        try:
            # Try to encode the special token
            token_ids = llm_tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) == 1:
                role_tokens[role_key] = token_ids[0]
            else:
                # Fallback: use a placeholder token id (this shouldn't happen with proper special tokens)
                role_tokens[role_key] = llm_tokenizer.unk_token_id if hasattr(llm_tokenizer, 'unk_token_id') else 0
        except:
            # Fallback for any encoding issues
            role_tokens[role_key] = llm_tokenizer.unk_token_id if hasattr(llm_tokenizer, 'unk_token_id') else 0
    
    prompt_text_parts = []
    prompt_taste_parts = []
    
    for segment in input_segments:
        # Add role token based on segment role
        role_key = f"{segment.role}_role"
        if role_key in role_tokens:
            role_token_id = role_tokens[role_key]
            role_text_ids = torch.tensor([[role_token_id]], dtype=torch.long, device=device)
            prompt_text_parts.append(role_text_ids)
            
            # Create corresponding TASTE tokens (use pad values for role tokens)
            role_taste_ids = torch.full((1, 1, vq_dim), -1, dtype=torch.long, device=device)
            prompt_taste_parts.append(role_taste_ids)
        
        # Add segment start token
        start_token_id = role_tokens['segment_start']
        start_text_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
        prompt_text_parts.append(start_text_ids)
        
        start_taste_ids = torch.full((1, 1, vq_dim), -1, dtype=torch.long, device=device)
        prompt_taste_parts.append(start_taste_ids)
        
        # Add segment content (text_ids and taste_ids)
        prompt_text_parts.append(segment.text_ids.to(device))
        
        if segment.taste_ids is not None:
            prompt_taste_parts.append(segment.taste_ids.to(device))
        else:
            # Create pad TASTE tokens for text-only segments
            text_len = segment.text_ids.size(1)
            pad_taste_ids = torch.full((1, text_len, vq_dim), -1, dtype=torch.long, device=device)
            prompt_taste_parts.append(pad_taste_ids)
        
        # Add segment end token
        end_token_id = role_tokens['segment_end'] 
        end_text_ids = torch.tensor([[end_token_id]], dtype=torch.long, device=device)
        prompt_text_parts.append(end_text_ids)
        
        end_taste_ids = torch.full((1, 1, vq_dim), -1, dtype=torch.long, device=device)
        prompt_taste_parts.append(end_taste_ids)
    
    # Concatenate all parts
    if prompt_text_parts:
        prompt_text_ids = torch.cat(prompt_text_parts, dim=1)
        prompt_taste_ids = torch.cat(prompt_taste_parts, dim=1)
    else:
        prompt_text_ids = torch.empty(1, 0, dtype=torch.long, device=device)
        prompt_taste_ids = torch.empty(1, 0, vq_dim, dtype=torch.long, device=device)
    
    return prompt_text_ids, prompt_taste_ids


def check_completion_status(text_id: int, action: str) -> 'CompletionStatus':
    """
    Check if generation should complete based on current token and action.
    
    Args:
        text_id: Current text token ID
        action: Current sampler action
        
    Returns:
        CompletionStatus: Object indicating completion state and reason
    """
    # Only these actions indicate true completion
    completion_reasons = {
        'eos': 'finished',
        'stop': 'finished', 
        'no_speech': 'no_speech',
        'terminate': 'finished',
    }
    
    # segment_end should NOT complete the generation - it just ends a segment
    # The generator should continue or emit the segment but keep generating
    
    is_complete = action in completion_reasons
    reason = completion_reasons.get(action, 'continuing')
    
    # Special case: if action is segment_end, it's not complete but should emit
    if action == 'segment_end':
        reason = 'segment_end'
    
    return CompletionStatus(is_complete, reason)


class CompletionStatus:
    """Simple class to hold completion status information."""
    
    def __init__(self, is_complete: bool, reason: str):
        self.is_complete = is_complete
        self.reason = reason