"""TASTE2 Stage-3 speech token generation model for TRT-LLM's PyTorch backend.

Adapts TASTE2's modified CosyVoice2 Qwen2-based speech LM
(``TasteS3GenerationLM``) for accelerated inference via TRT-LLM.

In TASTE2, the prefill embedding is fully pre-computed outside the LLM::

    text_emb    = embed_tokens(text_token_ids)
    taste_emb   = taste_tokenizer(text, audio)
    mixed_emb   = weighted_sum(taste_emb, text_emb)   # element-wise blend
    lm_input    = [sos_eos_emb, mixed_emb, task_id_emb]

Because every position holds a pre-computed embedding (not a plain token
lookup), the entire prefill sequence is passed as a single multimodal
embedding block.  All ``input_ids`` positions are OOV placeholders.

Architecture::

    CosyVoiceModel (backbone, inherits QwenModel)
    ------------------------------------------------
    embed_tokens      (151936 -> 896)   only used during warmup / KV estimation
    speech_embedding  (6564 -> 896)     speech-token lookup during decode
    llm_embedding     (2 -> 896)        sos_eos / task_id special tokens
    24x QwenDecoderLayer                (identical to stock Qwen2)
    norm                                (identical to stock Qwen2)

    CosyVoice2ForCausalLM (outer, inherits DecoderModelForCausalLM)
    ----------------------------------------------------------------
    model = CosyVoiceModel              backbone
    llm_decoder  (896 -> 6564, bias)    speech-token output head

Inference (unistream) via TRT-LLM executor:

  Prefill
    The caller pre-computes ``lm_input = [sos_eos, mixed_0..T, task_id]``
    and passes the entire tensor as ``multi_modal_embeddings``.
    ``input_ids`` is all OOV placeholders (one per embedding row).

  Decode
    ``multimodal_params`` is empty.  ``input_ids`` carries the last
    sampled speech-token id (0..6563) which is looked up through
    ``speech_embedding``.

Weight mapping (TASTE2 TasteS3GenerationLM checkpoint -> this model):
  llm.model.model.<X>   ->  model.<X>              (strip ``llm.model.``)
  llm.model.lm_head.*   ->  (skipped)
  llm_decoder.*          ->  llm_decoder.*
  speech_embedding.*     ->  model.speech_embedding.*
  llm_embedding.*        ->  model.llm_embedding.*

Usage with LLM API::

    llm.generate({
        "prompt": "x",                          # routing key (see note below)
        "multi_modal_embeddings": {
            "speech": [lm_input],               # (N, hidden) full prefill embedding
        },
    })

Note: ``"prompt"`` is required by ``LLM._preprocess`` routing — it is the
only branch that processes ``multi_modal_embeddings``.  The value is never
tokenized; it just satisfies the routing check.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_qwen import QwenModel
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModelForCausalLM,
    register_auto_model,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.inputs.data import TextPrompt
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.inputs.registry import (
    BaseMultimodalInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
)
from tensorrt_llm.sampling_params import SamplingParams

DEFAULT_SPEECH_TOKEN_SIZE = 6561
COSYVOICE_PLACEHOLDER_TOKEN = "<|speech_pad|>"


# ======================================================================
#  Input Processor
# ======================================================================

class CosyVoice2InputProcessor(BaseMultimodalInputProcessor):
    """Builds all-placeholder ``input_ids`` + ``multimodal_data`` from
    pre-computed prefill embeddings.

    The caller passes the full pre-fused embedding
    ``[sos_eos, mixed_0..T, task_id]`` via
    ``multi_modal_embeddings["speech"]``.  This processor creates one
    OOV placeholder per embedding row, and the model's forward assigns
    the embeddings directly as ``inputs_embeds``.
    """

    def __init__(self, model_path_or_dir, model_config, tokenizer,
                 trust_remote_code=True):
        self._model_path = model_path_or_dir
        self._config = model_config
        self._tokenizer = tokenizer

    @property
    def processor(self):
        return None

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def dtype(self):
        return getattr(self._config, "torch_dtype", torch.bfloat16)

    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required")
        token_ids = self.tokenizer.encode(
            inputs["prompt"],
            add_special_tokens=sampling_params.add_special_tokens,
        )
        return token_ids, None

    def attach_multimodal_embeddings(
        self,
        inputs: TextPrompt,
        multimodal_embedding: Dict[str, List[torch.Tensor]],
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """Build all-OOV ``input_ids`` + ``multimodal_data``.

        Expected ``multimodal_embedding``::

            {"speech": [tensor of shape (N, hidden_size)]}

        where N = 1 (sos) + T (mixed text/taste) + 1 (task_id).
        """
        if "speech" not in multimodal_embedding:
            raise ValueError(
                "multimodal_embedding must contain 'speech' key"
            )

        mm_tensors = multimodal_embedding["speech"]
        mm_embed = (
            torch.cat(mm_tensors, dim=0)
            if len(mm_tensors) > 1
            else mm_tensors[0]
        )

        placeholder_id = self._config.vocab_size + 1
        fused_ids = [placeholder_id] * mm_embed.shape[0]

        return fused_ids, {
            "multimodal_data": {"multimodal_embedding": mm_embed},
        }


# ======================================================================
#  Backbone: QwenModel + speech / special-token embeddings
# ======================================================================

class CosyVoiceModel(QwenModel):
    """QwenModel extended with TASTE2's speech and special-token embeddings.

    During prefill the entire sequence is pre-computed (weighted-sum of
    text and taste embeddings, bracketed by sos/task_id) and passed as
    multimodal embeddings.  During decode, ``input_ids`` contain
    speech-token IDs looked up through ``speech_embedding``.
    """

    def __init__(self, model_config: ModelConfig[Qwen2Config]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        speech_token_size = getattr(
            config, "speech_token_size", DEFAULT_SPEECH_TOKEN_SIZE
        )
        self.speech_vocab_size = speech_token_size + 3

        self.speech_embedding = Embedding(
            self.speech_vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )
        self.llm_embedding = nn.Embedding(2, config.hidden_size)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_params: List[MultimodalParams] = kwargs.pop(
            "multimodal_params", []
        )

        mm_embeds: List[torch.Tensor] = []
        if multimodal_params:
            num_ctx = attn_metadata.num_contexts
            ctx_params = multimodal_params[:num_ctx]
            mm_embeds = [
                p.multimodal_data["multimodal_embedding"]
                for p in ctx_params
                if (p.multimodal_data
                    and p.multimodal_data.get("multimodal_embedding") is not None)
            ]

        if mm_embeds:
            inputs_embeds = torch.cat(mm_embeds, dim=0).to(
                dtype=self.embed_tokens.weight.dtype,
                device=self.embed_tokens.weight.device,
            )
            input_ids = None
        elif inputs_embeds is None and input_ids is not None:
            if attn_metadata.num_contexts > 0:
                pass
            else:
                inputs_embeds = self.speech_embedding(input_ids)
                input_ids = None

        return super().forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


# ======================================================================
#  CausalLM wrapper
# ======================================================================

@register_input_processor(
    CosyVoice2InputProcessor,
    model_type="cosyvoice2",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"speech": COSYVOICE_PLACEHOLDER_TOKEN},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ),
)
@register_auto_model("CosyVoice2ForCausalLM")
class CosyVoice2ForCausalLM(
    DecoderModelForCausalLM[CosyVoiceModel, Qwen2Config]
):

    def __init__(self, model_config: ModelConfig[Qwen2Config]):
        config = model_config.pretrained_config
        speech_token_size = getattr(
            config, "speech_token_size", DEFAULT_SPEECH_TOKEN_SIZE
        )
        speech_vocab_size = speech_token_size + 3

        super().__init__(
            CosyVoiceModel(model_config),
            config=model_config,
            hidden_size=config.hidden_size,
            vocab_size=speech_vocab_size,
        )

        self.speech_token_size = speech_token_size
        self.speech_vocab_size = speech_vocab_size

        del self.lm_head
        self.llm_decoder = Linear(
            config.hidden_size, speech_vocab_size,
            bias=True, dtype=config.torch_dtype,
        )
        self.epilogue = [self.llm_decoder]

        self.stop_token_ids = [speech_token_size + i for i in range(3)]

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return self.logits_processor.forward(
            output, self.llm_decoder, attn_metadata, return_context_logits
        )

    def load_weights(
        self,
        weights: Dict,
        weight_mapper=None,
        skip_modules: List[str] = [],
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ):
        """Load from a TASTE2 TasteS3GenerationLM checkpoint.

        Remaps keys::

            llm.model.model.layers.N.*      ->  model.layers.N.*      (strip "llm.model.")
            llm.model.model.embed_tokens.*  ->  model.embed_tokens.*  (strip "llm.model.")
            llm.model.model.norm.*          ->  model.norm.*          (strip "llm.model.")
            llm.model.lm_head.*             ->  (skipped)
            llm_decoder.*                   ->  llm_decoder.*         (unchanged)
            speech_embedding.*              ->  model.speech_embedding.*  (add "model.")
            llm_embedding.*                 ->  model.llm_embedding.*    (add "model.")
        """
        remapped: Dict[str, torch.Tensor] = {}
        for name, tensor in weights.items():
            if name.startswith("llm.model."):
                suffix = name[len("llm.model."):]
                if suffix.startswith("lm_head"):
                    continue
                remapped[suffix] = tensor
            elif name.startswith("speech_embedding.") or name.startswith("llm_embedding."):
                remapped["model." + name] = tensor
            else:
                remapped[name] = tensor

        super().load_weights(
            remapped,
            weight_mapper=weight_mapper,
            skip_modules=skip_modules,
            params_map=params_map,
            allow_partial_loading=True,
        )


# ======================================================================
#  Checkpoint export: llm.pt -> HF-style dir for TRT-LLM
# ======================================================================

def export_checkpoint(
    llm_pt_path: str,
    output_dir: str,
    qwen_config_path: Optional[str] = None,
    speech_token_size: int = DEFAULT_SPEECH_TOKEN_SIZE,
):
    """Convert a TASTE2 ``llm.pt`` checkpoint into an HF-style directory
    that TRT-LLM's ``LLM(model=output_dir)`` can load.

    The output directory will contain:
      - ``config.json`` — Qwen2 config with ``CosyVoice2ForCausalLM`` arch
      - ``model.safetensors`` — weights in safetensors format

    Args:
        llm_pt_path: Path to the TASTE2 ``llm.pt`` checkpoint.
        output_dir: Directory to write the HF-style model to.
        qwen_config_path: Optional path to the Qwen2 ``config.json`` from
            the pretrained backbone (``CosyVoice-BlankEN/``).  If ``None``,
            defaults are used (Qwen2-0.5B).
        speech_token_size: Speech vocabulary size (default 6561).
    """
    import json
    import os
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)

    if qwen_config_path and os.path.isfile(qwen_config_path):
        with open(qwen_config_path) as f:
            config = json.load(f)
    else:
        config = {
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": True,
            "use_cache": True,
            "hidden_act": "silu",
            "torch_dtype": "bfloat16",
        }

    config["architectures"] = ["CosyVoice2ForCausalLM"]
    config["model_type"] = "qwen2"
    config["speech_token_size"] = speech_token_size

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    state_dict = torch.load(llm_pt_path, map_location="cpu")

    skip_prefixes = ("taste_tokenizer.", "taste_decoder_mixer.", "llm.model.lm_head.")
    tensors: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        tensors[name] = tensor

    save_file(tensors, os.path.join(output_dir, "model.safetensors"))
    print(f"Exported {len(tensors)} tensors to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export TASTE2 llm.pt checkpoint to HF-style dir for TRT-LLM"
    )
    parser.add_argument("llm_pt", help="Path to llm.pt")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--qwen-config", default=None,
                        help="Path to Qwen2 config.json (optional)")
    parser.add_argument("--speech-token-size", type=int,
                        default=DEFAULT_SPEECH_TOKEN_SIZE)
    args = parser.parse_args()

    export_checkpoint(
        args.llm_pt, args.output_dir,
        qwen_config_path=args.qwen_config,
        speech_token_size=args.speech_token_size,
    )
