
import torch
import torch.nn as nn
from typing import Tuple

from taste_speech.modules_taste.fusion import TTS_INPUT_FUSION_CLASSES


class TasteDecoderMixer(nn.Module):
    def __init__(
        self,
        class_name='weighted_sum',
    ):
        super().__init__()
        self.mixer = TTS_INPUT_FUSION_CLASSES[class_name]()

    def forward(
        self,
        text_token_emb, 
        taste_token_emb, 
        text_token_len
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_token_encoded: (B, T, C)
            audio_token_len: (B,)
            text_token_encoded: (B, T, C)
            audio_token_len: (B,)
        """
        return self.mixer(
            taste_token_emb,
            text_token_len,
            text_token_emb,
            text_token_len,
        )[0]
