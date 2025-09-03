
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from .configuration_taste import (
    TasteAudioTowerConfig,
    TasteSpeechDecoderConfig,
    TasteSpokenLMConfig,
    TasteConfig,
)
from .modeling_taste import (
    TasteAudioTower,
    TasteSpeechDecoder,
    TasteSpokenLM,
    TasteForCausalLM,
)
from .processing_taste import (
    TasteProcessor
)
from .modules_taste.inference_audio import VoiceGenerator

# Streaming support module
from . import streaming

AutoConfig.register('taste', TasteConfig)
AutoModelForCausalLM.register(TasteConfig, TasteForCausalLM)
AutoProcessor.register(TasteConfig, TasteProcessor)
