"""
TASTE-SpokenLM Streaming Support Module

Provides streaming capabilities for real-time conversation generation with multimodal segments.
Supports audio-to-TASTE token conversion, streaming generation, and TASTE token-to-audio synthesis.

Main API Functions:
- taste_tokenize: Convert audio waveform to TASTE tokens aligned with text
- streaming_generate: Iterator-based streaming conversation generation
- taste_detokenize: Convert TASTE tokens to audio waveforms with context continuity

Data Structures:
- TASTESegment: Multimodal conversation segment (text/audio)
- StreamingResult: Streaming generation result container
"""

from .segment import TASTESegment, StreamingResult
from .tokenize import taste_tokenize
from .generator import streaming_generate
from .detokenize import taste_detokenize

__version__ = "0.1.0"

__all__ = [
    # Data structures
    "TASTESegment", 
    "StreamingResult",
    
    # Core streaming functions
    "taste_tokenize",
    "streaming_generate", 
    "taste_detokenize",
]