"""Shim module that registers CosyVoice2ForCausalLM with TRT-LLM.

Importing this module triggers the ``@register_auto_model`` and
``@register_input_processor`` decorators in ``cosyvoice2``, making
the custom model discoverable by ``tensorrt_llm.LLM(model=...)``.

This file is imported as a top-level module via sys.path manipulation
(see TASTE2Model._load_trt_llm / TasteS3GenerationLM.load_trt_llm).
"""

from cosyvoice2 import CosyVoice2ForCausalLM, CosyVoice2InputProcessor  # noqa: F401
