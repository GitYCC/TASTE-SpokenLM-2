import logging
from typing import Dict, Optional, Union, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.whisper.modeling_whisper import WhisperEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAudioEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    # process that should be done during data collation
    # TODO: should be moved to other place

    def to(self, device):
        self.device = device
        return super().to(device)
    
    def get_device(self):
        if next(self.parameters(), None) is not None:
            return next(self.parameters()).device
        elif next(self.buffers(), None) is not None:
            return next(self.buffers()).device
        else:
            return 'cpu'

    # Parameter-related functions
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError


class WhisperAudioEncoder(BaseAudioEncoder):
    def __init__(
        self, 
        whisper_config,
        target_hidden_layer: int = 6, # specify which layer to extract. NOTE: zero means to extract the embed feature. Set to -1 to extract all hidden
        unfreeze_hidden_layers_from_last: int = 1,
    ):
        super().__init__()

        self.encoder = WhisperEncoder(whisper_config)

        self.top_layer_index = len(self.encoder.layers)
        if target_hidden_layer == -1:
            target_hidden_layer = self.top_layer_index
        self.target_hidden_layer = target_hidden_layer

        self.encoder._freeze_parameters()
        unfreezed_layer_indexes = set(range(target_hidden_layer - unfreeze_hidden_layers_from_last + 1, target_hidden_layer + 1))
        for i, layer in enumerate(self.encoder.layers):
            if i > target_hidden_layer:
                self.encoder.layers[i] = None  # delete
            if i in unfreezed_layer_indexes:
                self.encoder.layers[i].requires_grad = True

    def _check_input(self, audio_features):
        assert audio_features.size(2) == self.encoder.config.num_mel_bins, audio_features.size()
        return audio_features # dim: [B, L, D]

    def _check_output(self, results):
        assert results['encoded_feats'].size(2) == self.encoder.config.d_model, results['encoded_feats'].size()
        return results # dim: [B, L, D]

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        input_features = self._check_input(audio_features).permute(0, 2, 1)  # (B, C, T)
        output_hidden_states = kwargs.get('output_hidden_states')
        output_attentions = kwargs.get('output_attentions')

        with torch.cuda.amp.autocast():
            expected_seq_length = self.encoder.max_source_positions * self.encoder.conv1.stride[0] * self.encoder.conv2.stride[0]
            if input_features.shape[-1] != expected_seq_length:
                raise ValueError(
                    f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
                )

            inputs_embeds = nn.functional.gelu(self.encoder.conv1(input_features))
            inputs_embeds = nn.functional.gelu(self.encoder.conv2(inputs_embeds))

            inputs_embeds = inputs_embeds.permute(0, 2, 1) # (B, T, C)
            embed_pos = self.encoder.embed_positions.weight

            hidden_states = inputs_embeds + embed_pos
            hidden_states = nn.functional.dropout(hidden_states, p=self.encoder.dropout, training=self.encoder.training)

            encoder_states = () if output_hidden_states else None

            for idx, encoder_layer in enumerate(self.encoder.layers):
                if idx == self.target_hidden_layer:
                    results = {
                        'encoded_feats': hidden_states,
                        'encoded_feat_lengths': audio_feature_lengths // 2, # whisper encoder will down-sample by 2 
                    }
                    return self._check_output(results)

                if  self.target_hidden_layer == self.top_layer_index:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=None,
                        output_attentions=output_attentions,
                    )

                    hidden_states = layer_outputs[0]
                    if output_attentions:
                        raise NotImplementedError
                        # attentions = layer_outputs[1]

            hidden_states = self.layer_norm(hidden_states)
            if self.target_hidden_layer == self.top_layer_index:
                encoder_states = encoder_states + (hidden_states,)
                results = {
                    'encoded_feats': encoder_states,
                    'encoded_feat_lengths': audio_feature_lengths // 2, 
                } # return all encoder states
            else: 
                # return last hidden
                results = {
                    'encoded_feats': hidden_states,
                    'encoded_feat_lengths': audio_feature_lengths // 2, 
                }
        
            return self._check_output(results)
