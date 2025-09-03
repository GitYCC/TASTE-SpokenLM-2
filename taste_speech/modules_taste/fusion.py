# This file explore multiple fusion types between the derived audio token and the text token (encoded)
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from taste_speech.modules_taste.cosyvoice.utils import IGNORE_ID


class Concat(nn.Module):
    def __init__(
        self,
        audio_first: bool = True,
        ignore_id: int = IGNORE_ID,
    ):
        super().__init__()
        self.audio_first = audio_first
        self.ignore_id = ignore_id
        # skip init, no submodule
    
    def forward(
        self,
        audio_token_encoded,
        audio_token_len,
        text_token_encoded,
        text_token_len,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_token_encoded: (B, T, C)
            audio_token_len: (B,)
            text_token_encoded: (B, T, C)
        """
        assert audio_token_encoded.shape[-1] == text_token_encoded.shape[-1], "Inconsistency in feature dimension!"
        unpad_audio_token = unpad_sequence(audio_token_encoded, audio_token_len.cpu(), batch_first=True)
        unpad_text_token = unpad_sequence(text_token_encoded, text_token_len.cpu(), batch_first=True)
        
        if self.audio_first:
            audio_text_token_encoded = [torch.concat([unpad_audio_token[i], unpad_text_token[i]], dim=0) for i in range(len(audio_token_len))]
        else:
            audio_text_token_encoded = [torch.concat([unpad_text_token[i], unpad_audio_token[i]], dim=0) for i in range(len(text_token_len))]
        audio_text_token_len = torch.tensor([i.size(0) for i in audio_text_token_encoded], dtype=torch.int32)
        audio_text_token_encoded = pad_sequence(audio_text_token_encoded, batch_first=True, padding_value=self.ignore_id)
        return audio_text_token_encoded, audio_text_token_len


class ConcatWithSEP(nn.Module):
    def __init__(
        self,
        audio_first: bool = True,
        ignore_id: int = IGNORE_ID,
        d: int = 512,
    ):
        super().__init__()
        self.audio_first = audio_first
        self.ignore_id = ignore_id
        self.sep_embed = nn.parameter.Parameter(
            torch.rand((1, d), dtype=torch.float32) * 0.00001
        )
        # skip init, no submodule
    
    def forward(
        self,
        audio_token_encoded,
        audio_token_len,
        text_token_encoded,
        text_token_len,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_token_encoded: (B, T, C)
            audio_token_len: (B,)
            text_token_encoded: (B, T, C)
        """
        assert audio_token_encoded.shape[-1] == text_token_encoded.shape[-1], "Inconsistency in feature dimension!"
        unpad_audio_token = unpad_sequence(audio_token_encoded, audio_token_len.cpu(), batch_first=True)
        unpad_text_token = unpad_sequence(text_token_encoded, text_token_len.cpu(), batch_first=True)
        
        if self.audio_first:
            audio_text_token_encoded = [torch.concat([unpad_audio_token[i], self.sep_embed, unpad_text_token[i]], dim=0) for i in range(len(audio_token_len))]
        else:
            audio_text_token_encoded = [torch.concat([unpad_text_token[i], self.sep_embed, unpad_audio_token[i]], dim=0) for i in range(len(text_token_len))]
        audio_text_token_len = torch.tensor([i.size(0) for i in audio_text_token_encoded], dtype=torch.int32)
        audio_text_token_encoded = pad_sequence(audio_text_token_encoded, batch_first=True, padding_value=self.ignore_id)
        return audio_text_token_encoded, audio_text_token_len


# weighted sum
class WeightedSum(nn.Module):
    def __init__(
        self,
        ignore_id: int = IGNORE_ID,
        normalize: bool = False,
        use_trainable_weight: bool = True,
        use_layer_norm: bool = False,
        weight_init_type: str = 'balance'
    ):
        super().__init__()
        if weight_init_type == 'balance':
            self.weights = nn.Parameter(torch.tensor([1., 1.]), requires_grad=use_trainable_weight)
        elif weight_init_type == 'zero_audio':
            self.weights = nn.Parameter(torch.tensor([-3., 3.]), requires_grad=use_trainable_weight)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            # TODO: Maybe we can add layernorm here. Currently I think it is not necessary. 
            raise NotImplementedError
        self.normalize = normalize
        self.ignore_id = IGNORE_ID
    
    def forward(
        self,
        audio_token_encoded,
        audio_token_len,
        text_token_encoded,
        text_token_len,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_token_encoded: (B, T, C)
            audio_token_len: (B,)
            text_token_encoded: (B, T, C)
            audio_token_len: (B,)
        """
        assert audio_token_encoded.shape[-1] == text_token_encoded.shape[-1], "Inconsistency in feature  dimension!"
        assert audio_token_encoded.shape[-2] == text_token_encoded.shape[-2], "Inconsistency in sequence dimension!"
        assert (audio_token_len == text_token_len).sum().item() == audio_token_len.size(0), "Find length mismatch"
        # unpad_audio_token = unpad_sequence(audio_token_encoded, audio_token_len.cpu(), batch_first=True)
        # unpad_text_token = unpad_sequence(text_token_encoded, text_token_len.cpu(), batch_first=True)
        # use softmax to prevent negative weight
        weights = F.softmax(self.weights, dim=0)
        weights = weights.view(2, 1, 1, 1)

        if self.normalize:
            audio_token_encoded = F.layer_norm(
                audio_token_encoded,
                (audio_token_encoded.size(-1),)
            )
            text_token_encoded = F.layer_norm(
                text_token_encoded,
                (text_token_encoded.size(-1),)
            )
        inputs = torch.stack([audio_token_encoded, text_token_encoded], dim=0)
        audio_text_token_encoded = (weights * inputs).sum(dim=0)
        audio_text_token_len = audio_token_len
        # audio_text_token_len = torch.tensor([i.size(0) for i in audio_text_token_encoded], dtype=torch.int32)
        return audio_text_token_encoded, audio_text_token_len


TTS_INPUT_FUSION_CLASSES = {
    "concat": Concat,
    "concat_with_sep": ConcatWithSEP,
    "weighted_sum": WeightedSum,
}


def test_weighted_sum():
    fusion_layer = WeightedSum(normalize=False)
    x = torch.randn((2, 10, 100))
    x_len = torch.tensor([5, 10])
    y = torch.randn((2, 10, 100))
    y_len = torch.tensor([5, 10])
    z, z_len = fusion_layer(
        x, 
        x_len,
        y,
        y_len
    )
    print(z.shape, z_len)
    print(x[0][0])
    print(y[0][0])
    print(z[0][0])
    print((x[0][0] + y[0][0]) / 2)

if __name__ == "__main__":
    test_weighted_sum()