
from typing import Dict, Optional, Union
import logging

import torch
from torch import nn
import torch.nn.functional as F

from taste_speech.modules_taste.cosyvoice.utils import make_pad_mask

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LocalAveragePoolingSegmenter(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        encoded_feats, # [batch, feat_len, audio_embed]
        encoded_feat_lengths, # [batch]
        asr_token_ids,
        asr_token_lengths,
        asr_token_alignments,
        **kwargs,
    ):
        alignments = (asr_token_alignments * encoded_feat_lengths.unsqueeze(1).unsqueeze(2)).int()
        # alignments: [batch, asr_token_len, start_and_end]

        indexes = torch.arange(encoded_feats.size(1)).repeat(alignments.size(0) * alignments.size(1)) \
            .reshape(alignments.size(0), alignments.size(1), -1).to(encoded_feats.device)
        token_mask = ~ make_pad_mask(asr_token_lengths, asr_token_ids.size(1)).unsqueeze(-1)

        mask = (token_mask & (indexes >= alignments[:, :, 0:1]) &  (indexes <= alignments[:, :, 1:2])).unsqueeze(-1)
        # [batch, asr_token_len, feat_len, 1]
        
        expanded_encoded_feats = encoded_feats.unsqueeze(1).expand(-1, mask.size(1), -1, -1)  
        # [batch, 1, feat_len, audio_embed]

        segmented_feats = ((expanded_encoded_feats * mask).sum(dim=-2) / mask.sum(dim=-2))
        segmented_feats = torch.nan_to_num(segmented_feats)

        segmented_results = {
            'segmented_feats': segmented_feats,  # [batch, asr_token_len, embed]
            'segmented_feat_lengths': asr_token_lengths,
        }
        return segmented_results
