"""
TASTESegment data structure for managing multimodal conversation segments.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TASTESegment:
    """
    Multimodal conversation segment data structure supporting text and audio modalities.
    
    Args:
        role: Conversation role - 'system', 'user', or 'assistant'
        modality: Content type - 'text' or 'audio'  
        text_ids: Text token IDs tensor of shape (1, seq_len)
        taste_ids: Optional TASTE token indices of shape (1, seq_len, vq_dim)
    """
    role: str
    modality: str  
    text_ids: torch.Tensor
    taste_ids: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Validate segment data consistency."""
        # Validate role
        if self.role not in ['system', 'user', 'assistant']:
            raise ValueError(f"Invalid role '{self.role}'. Must be 'system', 'user', or 'assistant'")
        
        # Validate modality
        if self.modality not in ['text', 'audio']:
            raise ValueError(f"Invalid modality '{self.modality}'. Must be 'text' or 'audio'")
        
        # Validate text_ids
        if not isinstance(self.text_ids, torch.Tensor):
            raise TypeError("text_ids must be a torch.Tensor")
        if self.text_ids.ndim != 2:
            raise ValueError("text_ids must have shape (1, seq_len)")
        if self.text_ids.size(0) != 1:
            raise ValueError("text_ids batch size must be 1")
        
        # Validate taste_ids for audio modality
        if self.modality == 'audio':
            if self.taste_ids is None:
                raise ValueError("Audio segments must have taste_ids")
            if not isinstance(self.taste_ids, torch.Tensor):
                raise TypeError("taste_ids must be a torch.Tensor")
            if self.taste_ids.ndim != 3:
                raise ValueError("taste_ids must have shape (1, seq_len, vq_dim)")
            if self.taste_ids.size(0) != 1:
                raise ValueError("taste_ids batch size must be 1") 
            if self.taste_ids.size(1) != self.text_ids.size(1):
                raise ValueError("taste_ids and text_ids sequence lengths must match")
        
        # For text modality, taste_ids should be None
        elif self.modality == 'text' and self.taste_ids is not None:
            # Allow taste_ids for text segments (for mixed cases), just validate shape
            if not isinstance(self.taste_ids, torch.Tensor):
                raise TypeError("taste_ids must be a torch.Tensor or None")
            if self.taste_ids.ndim != 3:
                raise ValueError("taste_ids must have shape (1, seq_len, vq_dim)")
            if self.taste_ids.size(0) != 1:
                raise ValueError("taste_ids batch size must be 1")
            if self.taste_ids.size(1) != self.text_ids.size(1):
                raise ValueError("taste_ids and text_ids sequence lengths must match")


@dataclass
class StreamingResult:
    """
    Result structure for streaming_generate output.
    
    Args:
        is_complete: Whether generation is complete
        completion_reason: Reason for completion ('no_speech', 'finished', 'segment_end', etc.)
        segment: Generated TASTESegment
    """
    is_complete: bool
    completion_reason: str
    segment: TASTESegment