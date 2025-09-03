
import torch
import torch.nn as nn
from typing import Dict

from taste_speech.modules_taste.audio_joint_encoder_segmenter import WhisperAudioJointEncoderSegmenter
from taste_speech.modules_taste.audio_quantizer import QUANTIZER_CLASSES
from taste_speech.modules_taste.utils import generate_mask_from_length


class TasteTokenizer(nn.Module):
    def __init__(
        self,
        backbond_name_or_path='distil-whisper/distil-large-v3',
        attn_implementation='eager',
        whisper_decoder_embed_dim=1280,
        encoder_input_size=896,
        quantization_on=False,
        kwargs_for_quantizer: Dict = None,
        neck_on=False,
        d_neck=None,
    ):
        super().__init__()
        self.quantization_on = quantization_on

        self.audio_joint_encoder_segmenter = WhisperAudioJointEncoderSegmenter(
            model_name_or_path=backbond_name_or_path,
            attn_implementation=attn_implementation,
            dtype='bfloat16',
            forward_type='asr_attn_pooling',
            make_v_proj_identity=True,
            skip_prefix_idx=0,
            use_custom=True,
            is_word_level=False,
            new_vocab={
                'vocab_size': 151936,
                'padding_idx': 151643,
            }
        )

        self.audio_affine_layer = nn.Linear(whisper_decoder_embed_dim, encoder_input_size)

        if self.quantization_on:
            replaced_kwargs = dict(kwargs_for_quantizer)
            quantizer_class = replaced_kwargs.pop('quantizer_class', 'rvq')
            self.vq = QUANTIZER_CLASSES[quantizer_class](
                **replaced_kwargs,
            )
        elif neck_on:
            self.neck_on = neck_on
            self.neck_proj_in = nn.Linear(encoder_input_size, d_neck)
            self.neck_proj_out = nn.Linear(d_neck, encoder_input_size)

    def load_from_cosyvoice_ckpt(self, pt_path):
        raise NotImplementedError
        # pt_path should be the state_dict of `audio_llm`
        loaded_state_dict = torch.load(pt_path, map_location='cpu')
        converted_state_dict = {}
        for name, param in loaded_state_dict.items():
            if "audio_tokenizer" in name:
                new_name = name.split("audio_tokenizer.")[-1]
                new_name = new_name.replace("audio_quantizer", "vq")
                converted_state_dict[new_name] = param
        self.load_state_dict(converted_state_dict, strict=True) # ensure consistency

    def forward(
            self,
            text_token,
            text_token_len,
            audio_feature,
            audio_feature_len,
            **kwargs,
        ):

        text_token = text_token.detach()
        text_token_len = text_token_len.detach()
        audio_feature = audio_feature.detach()
        audio_feature_len = audio_feature_len.detach()

        encoded_results, segmented_results = self.audio_joint_encoder_segmenter(
            audio_feature, audio_feature_len,
            None, None, None, 
            whisper_text_token=text_token,  # aligned tokenization space
            whisper_text_token_len=text_token_len,  # aligned tokenization space
        )

        taste_token_emb = self.audio_affine_layer(segmented_results['segmented_feats'])
        taste_token_emb_len = segmented_results['segmented_feat_lengths']

        assert (taste_token_emb_len - text_token_len).sum().item() == 0

        if self.quantization_on:
            quantized_results = self.vq(
                taste_token_emb,
                mask=generate_mask_from_length(taste_token_emb_len)
            )
            taste_latent = self.vq.rvq.get_code_from_indices(quantized_results['quantized_indices'])
            taste_token_emb = quantized_results['quantized_feats'] 
        elif self.neck_on:
            taste_latent = self.neck_proj_in(taste_token_emb)
            taste_token_emb = self.neck_proj_out(taste_latent)

        result = {
            'taste_token_emb': taste_token_emb,
            'taste_token_emb_len': taste_token_emb_len,
            'taste_latent': taste_latent,
        }
        if self.quantization_on:
            if self.training:
                result['commit_loss'] = quantized_results['commit_loss']
            result['quantized_indices'] = quantized_results['quantized_indices']

        return result
