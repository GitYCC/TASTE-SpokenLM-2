
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import re

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from einops import reduce
from transformers import WhisperConfig
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM
from transformers import GenerationMixin
from transformers.utils import ModelOutput

from .configuration_taste import TasteConfig, TasteAudioTowerConfig, TasteSpeechDecoderConfig, TasteSpokenLMConfig
from .modules_taste.cosyvoice.encoder import ConformerEncoder as CosyVoiceConformerEncoder
from .modules_taste.cosyvoice.encoder import TransformerEncoder as CosyVoiceTransformerEncoder
from .modules_taste.cosyvoice.label_smoothing_loss import LabelSmoothingLoss
from .modules_taste.cosyvoice.utils import IGNORE_ID, th_accuracy
from .modules_taste.audio_encoder import WhisperAudioEncoder
from .modules_taste.audio_segmenter import LocalAveragePoolingSegmenter
from .modules_taste.audio_quantizer import QUANTIZER_CLASSES
from .modules_taste.audio_joint_encoder_segmenter import WhisperAudioJointEncoderSegmenter
from .modules_taste.bridge import BRIDGE_FUSION_CLASSES, BRIDGE_EXTRACT_CLASSES
from .modules_taste.fusion import TTS_INPUT_FUSION_CLASSES
from .modules_taste.sampler import TasteSampler
from .modules_taste.utils import generate_mask_from_length, debug_print, _find_all_linear_names


class TasteAudioTower(nn.Module):
    def __init__(
        self,
        encoder_input_size: int = 512,
        text_token_size: int = 51866,
        audio_embed_dim: int = 1280, # the size of the audio quantized vector (added)
        quantization_on=False,
        is_joint_encoder_segmenter = False,
        audio_dropout_ratio=0.0,
        kwargs_audio_encoder: Dict = None,
        kwargs_audio_segmenter: Dict = None,
        kwargs_for_joint_encoder_segmenter: Dict = None,
        kwargs_for_quantizer: Dict = None,
    ):
        super().__init__()
        self.quantization_on = quantization_on
        self.is_joint_encoder_segmenter = is_joint_encoder_segmenter

        if kwargs_audio_encoder is None:
            kwargs_audio_encoder = {
                'whisper_config': WhisperConfig(),
                'target_hidden_layer': 6,
                'unfreeze_hidden_layers_from_last': 1,
            }

        if kwargs_audio_segmenter is None:
            kwargs_audio_segmenter = {
            }
        
        if not self.is_joint_encoder_segmenter:
            self.audio_encoder = WhisperAudioEncoder(**kwargs_audio_encoder)
            self.audio_segmenter = LocalAveragePoolingSegmenter(**kwargs_audio_segmenter)
            self.audio_affine_layer = nn.Linear(audio_embed_dim, encoder_input_size)
            self.affine_audio = True
            if kwargs_for_quantizer != None:
                replaced_kwargs = dict(kwargs_for_quantizer)
                quantizer_class = replaced_kwargs.pop('quantizer_class', 'rvq')
                self.vq = QUANTIZER_CLASSES[quantizer_class](
                    **replaced_kwargs,
                )
                self.quantization_on = True   # TODO: remove quantization_on in config
            else:
                self.quantization_on = False
        else:
            if kwargs_for_joint_encoder_segmenter is None:
                kwargs_for_joint_encoder_segmenter = {}
            self.audio_joint_encoder_segmenter = WhisperAudioJointEncoderSegmenter(
                **kwargs_for_joint_encoder_segmenter,
            )
            self.affine_audio = False
            if kwargs_for_quantizer != None:
                replaced_kwargs = dict(kwargs_for_quantizer)
                quantizer_class = replaced_kwargs.pop('quantizer_class', 'rvq')
                self.vq = QUANTIZER_CLASSES[quantizer_class](
                    **replaced_kwargs,
                )
                self.quantization_on = True
            else:
                self.quantization_on = False
        
        self.audio_dropout_ratio = audio_dropout_ratio
        if audio_dropout_ratio > 0.0:
            self.audio_embed_dropout = nn.Dropout(p=audio_dropout_ratio)

        self.add_eos = True

    def load_from_cosyvoice_ckpt(self, pt_path):
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
            asr_token_ids,
            asr_token_lengths,
            audio_features,
            audio_feature_lengths,
            asr_token_alignments=None,
            kwargs_for_encoder=None,
            kwargs_for_segmenter=None,
            kwargs_for_joint_encoder_segmenter=None,
            **kwargs,
        ):
        if kwargs_for_encoder is None:
            kwargs_for_encoder = dict()
        if kwargs_for_segmenter is None:
            kwargs_for_segmenter = dict()

        asr_token_ids = asr_token_ids.detach()
        asr_token_lengths = asr_token_lengths.detach()
        audio_features = audio_features.detach()
        audio_feature_lengths = audio_feature_lengths.detach()

        if asr_token_alignments is not None:
            asr_token_alignments = asr_token_alignments.detach()

        if not self.is_joint_encoder_segmenter:
            encoded_results = self.audio_encoder(
                audio_features, audio_feature_lengths, **kwargs_for_encoder)

            segmented_results = self.audio_segmenter(
                encoded_results['encoded_feats'], encoded_results['encoded_feat_lengths'], 
                asr_token_ids, asr_token_lengths, asr_token_alignments, **kwargs_for_segmenter)
        else:
            words_index = kwargs.get("words_index", None)
            asr_word_ids = kwargs.get("asr_word_ids", None)

            if self.add_eos:
                whisper_text_token = torch.cat(
                    (
                        torch.tensor([[50258, 50259, 50360, 50364] for _ in range(asr_token_ids.size(0))]).to(asr_token_ids.device),
                        asr_token_ids,
                        torch.tensor([[50257] for _ in range(asr_token_ids.size(0))]).to(asr_token_ids.device),
                    ), 
                    dim=1)
                whisper_text_token_len = asr_token_lengths + 5
            else:
                whisper_text_token = torch.cat(
                    (
                        torch.tensor([[50258, 50259, 50360, 50364] for _ in range(asr_token_ids.size(0))]).to(asr_token_ids.device),
                        asr_token_ids
                    ), 
                    dim=1)
                whisper_text_token_len = asr_token_lengths + 4

            encoded_results, segmented_results = self.audio_joint_encoder_segmenter(
                audio_features, audio_feature_lengths,
                None, None, None, 
                whisper_text_token=whisper_text_token,
                whisper_text_token_len=whisper_text_token_len,
                words_index=words_index,
                word_ids=asr_word_ids,
            )
            if self.add_eos:
                segmented_results['segmented_feats'] = segmented_results['segmented_feats'][:,:-1,:]
                segmented_results['segmented_feat_lengths'] -= 1

        if self.affine_audio:
            audio_unit_embeds = self.audio_affine_layer(segmented_results['segmented_feats'])
        else: 
            audio_unit_embeds = segmented_results['segmented_feats']
        audio_unit_lengths = segmented_results['segmented_feat_lengths']

        if self.quantization_on and not kwargs.get('skip_vq_in_audio_encoder', False):
            quantized_results = self.vq(
                audio_unit_embeds,
                mask=generate_mask_from_length(audio_unit_lengths)
            )
            audio_unit_embeds = quantized_results['quantized_feats']

        #randomly empty audio_unit_embeds
        if self.audio_dropout_ratio > 0.0:
            batch_mask = self.audio_embed_dropout(torch.ones((audio_unit_embeds.size(0), )).to(audio_unit_embeds.device))
            std = torch.std(audio_unit_embeds)
            rand_embeds = torch.normal(
                mean=torch.zeros(audio_unit_embeds.size(), dtype=audio_unit_embeds.dtype, device=audio_unit_embeds.device),
                std=std
            )
            audio_unit_embeds = torch.where(
                batch_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, audio_unit_embeds.size(1), audio_unit_embeds.size(2)) > 0.,
                audio_unit_embeds,
                rand_embeds,
            )

        result = {
            'audio_unit_embeds': audio_unit_embeds,
            'audio_unit_lengths': audio_unit_lengths,
        }
        if self.quantization_on and not kwargs.get('skip_vq_in_audio_encoder', False):
            if self.training:
                result['commit_loss'] = quantized_results['commit_loss']
            result['quantized_indices'] = quantized_results['quantized_indices']

        assert (result['audio_unit_lengths'] - asr_token_lengths).sum().item() == 0
        return result


class TasteSpeechDecoder(nn.Module):
    def __init__(
        self,
        encoder_input_size: int = 512,
        audio_encoder_input_size: int = -1, # -1 means it is aligned with `encoder_input_size`
        llm_input_size: int = 1024,
        llm_output_size: int = 1024,
        text_token_size: int = 51866,
        speech_token_size: int = 4096,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        spk_embed_dim: int = 192,
        skip_prefix_idx: int = 0,
        kwargs_cosyvoice_encoder=None,
        kwargs_cosyvoice_audio_token_encoder=None,
        kwargs_cosyvoice_llm=None,
        fuse_encoded_audio_text_type: str = 'weighted_sum',
        fuse_encoded_audio_text_kwargs: Dict = {},
    ):
        super().__init__()

        if kwargs_cosyvoice_encoder is None:
            kwargs_cosyvoice_encoder = {
                "attention_heads": 8,
                "linear_units": 2048,
                "num_blocks": 3,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0,
                "normalize_before": True,
                "input_layer": 'linear',
                "pos_enc_layer_type": 'rel_pos_espnet',
                "selfattention_layer_type": 'rel_selfattn',
                "use_cnn_module": False,
                "macaron_style": False,
                "use_dynamic_chunk": False,
                "use_dynamic_left_chunk": False,
                "static_chunk_size": 1,
            }
        if kwargs_cosyvoice_llm is None:
            kwargs_cosyvoice_llm = {
                "attention_heads": 8,
                "linear_units": 2048,
                "num_blocks": 7,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0,
                "input_layer": 'linear_legacy',
                "pos_enc_layer_type": 'rel_pos_espnet',
                "selfattention_layer_type": 'rel_selfattn',
                "static_chunk_size": 1,
            }
        # temp solution
        if kwargs_cosyvoice_audio_token_encoder is None:
            kwargs_cosyvoice_audio_token_encoder = {
                "attention_heads": 8,
                "linear_units": 2048,
                "num_blocks": 2,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0,
                "normalize_before": True,
                "input_layer": 'linear',
                "pos_enc_layer_type": 'rel_pos_espnet',
                "selfattention_layer_type": 'rel_selfattn',
                "use_cnn_module": False,
                "macaron_style": False,
                "use_dynamic_chunk": False,
                "use_dynamic_left_chunk": False,
                "static_chunk_size": 1,
            }

        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # add skip prefix index
        self.skip_prefix_idx = skip_prefix_idx

        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, encoder_input_size)
        self.text_encoder = CosyVoiceConformerEncoder(
            encoder_input_size,
            output_size=llm_input_size,
            **kwargs_cosyvoice_encoder
        )
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = CosyVoiceTransformerEncoder(
            llm_input_size,
            llm_output_size,
            **kwargs_cosyvoice_llm
        )
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. audio unit encoder
        if audio_encoder_input_size == -1: 
            audio_encoder_input_size = encoder_input_size  # same as input dim of text_encoder
            self.audio_embed_affine_layer = None
        else:
            # construct audio_embed_affine_layer
            self.audio_embed_affine_layer = torch.nn.Linear(audio_encoder_input_size, encoder_input_size)
        self.audio_token_encoder = CosyVoiceConformerEncoder(
            encoder_input_size,
            output_size=llm_input_size,
            **kwargs_cosyvoice_audio_token_encoder
        )
        self.audio_token_encoder_affine_layer = nn.Linear(
            self.audio_token_encoder.output_size(),
            llm_input_size
        )

        # 5. fusion
        self.fuse_encoded_audio_text_module = TTS_INPUT_FUSION_CLASSES[fuse_encoded_audio_text_type](
            **fuse_encoded_audio_text_kwargs
        )

    def load_from_cosyvoice_ckpt(self, pt_path):
        loaded_state_dict = torch.load(pt_path, map_location='cpu')
        converted_state_dict = {}
        for name, param in loaded_state_dict.items():
            if "audio_tokenizer" in name:
                continue
            converted_state_dict[name] = param
        self.load_state_dict(converted_state_dict, strict=True) # ensure consistency


    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            sampling: Union[bool, int, float] = True,
            beam_size: int = 1,
            ignore_eos: bool = True,
    ):
        while True:
            prob, indices = weighted_scores.softmax(dim=-1).topk(sampling)
            top_ids = prob.multinomial(beam_size, replacement=True)
            top_ids = indices[top_ids]
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids

    def encode_text(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def encode_audio(
            self,
            audio: torch.Tensor,
            audio_lengths: torch.Tensor,
    ):
        if self.audio_embed_affine_layer != None:
            audio = self.audio_embed_affine_layer(audio)
        encoder_out, encoder_mask = self.audio_token_encoder(audio, audio_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.audio_token_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(
            self,
            sos_eos_emb,
            speaker_embeds,
            audio_text_token_encoded,
            audio_text_token_len,
            task_id_emb,
            speech_token_embeds=None,
            speech_token_lengths=None,
            padding_value=IGNORE_ID,
            padding_side='right',
        ):
        device = audio_text_token_encoded.device

        unpad_audio_text_token_encoded = unpad_sequence(
            audio_text_token_encoded, audio_text_token_len.cpu(), batch_first=True)
        if speech_token_embeds is not None:
            unpad_speech_token_embeds = unpad_sequence(
                speech_token_embeds, speech_token_lengths.cpu(), batch_first=True)
        batch = len(unpad_audio_text_token_encoded)
        sequences = [
            torch.concat([
                sos_eos_emb.squeeze(dim=0),
                speaker_embeds[i],
                unpad_audio_text_token_encoded[i],
                task_id_emb.squeeze(dim=0),
            ]
            + ([unpad_speech_token_embeds[i]] if speech_token_embeds is not None else [])
            , dim=0) 
            for i in range(batch)
        ]
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.int32)
        if padding_side == 'right':
            sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        elif padding_side == 'left':
            max_length = lengths.max()
            sequences = torch.stack(
                [F.pad(seq, (0, 0, max_length-seq.size(0), 0), 'constant', padding_value)
                 for seq in sequences]
            )
        return sequences.to(device), lengths.to(device)

    def prepare_conditional_embeds(
            self,
            speaker_embeds,
            audio_unit_embeds,
            audio_unit_lengths,
            asr_token_ids,
            asr_token_lengths,
            skip_audio_in_audio_decoder=False,
        ):
        # speaker embedding projection
        speaker_embeds = F.normalize(speaker_embeds, dim=1)
        speaker_embeds = self.spk_embed_affine_layer(speaker_embeds)
        speaker_embeds = speaker_embeds.unsqueeze(1)

        # encode text token
        asr_token_embeds = self.text_embedding(asr_token_ids)
        asr_token_encoded, asr_token_lengths = self.encode_text(asr_token_embeds, asr_token_lengths)

        if skip_audio_in_audio_decoder:
            audio_text_token_encoded, audio_text_token_len = asr_token_encoded, asr_token_lengths
        else:
            # encod audio unit
            audio_unit_encoded, audio_unit_lengths = self.encode_audio(audio_unit_embeds, audio_unit_lengths)

            # fuse audio units and token embeds
            audio_text_token_encoded, audio_text_token_len = self.fuse_encoded_audio_text_module(
                audio_unit_encoded,
                audio_unit_lengths,
                asr_token_encoded,
                asr_token_lengths,
            )

        # other
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        assert (audio_text_token_len - asr_token_lengths).sum().item() == 0
        return (
            sos_eos_emb,
            speaker_embeds, 
            audio_text_token_encoded,
            audio_text_token_len, 
            task_id_emb,
        )

    def forward(
            self,
            speaker_embeds,
            audio_unit_embeds,
            audio_unit_lengths,
            asr_token_ids,
            asr_token_lengths,
            speech_token_ids,
            speech_token_lengths,
            **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:

        # prepare conditional embeds
        (
            sos_eos_emb,
            speaker_embeds, 
            audio_text_token_encoded,
            audio_text_token_len, 
            task_id_emb
        ) \
        = self.prepare_conditional_embeds(
            speaker_embeds,
            audio_unit_embeds,
            audio_unit_lengths,
            asr_token_ids,
            asr_token_lengths,
            skip_audio_in_audio_decoder=kwargs.get('skip_audio_in_audio_decoder', False)
        )

        # encode speech token
        speech_token_embeds = self.speech_embedding(speech_token_ids)

        # prepare lm_input
        lm_input, lm_input_len = self.pad_unpad_sequence(
            sos_eos_emb,
            speaker_embeds, 
            audio_text_token_encoded,
            audio_text_token_len, 
            task_id_emb,
            speech_token_embeds,
            speech_token_lengths,
            padding_side='right'
        )
        
        # prepare lm_target
        lm_target = [
            torch.tensor(
                [IGNORE_ID] * (2 + audio_text_token_len[i]) 
                + speech_token_ids[i, :speech_token_lengths[i]].tolist() 
                + [self.speech_token_size]
            ).to(asr_token_ids.device)
            for i in range(asr_token_ids.size(0))
        ]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)

        # run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len)
        logits = self.llm_decoder(lm_output)
        if self.training:
            loss = self.criterion_ce(logits, lm_target)
        else:
            loss = None

        return {'loss': loss, 'labels': lm_target, 'logits': logits}


class TasteSpokenLM(nn.Module):
    def __init__(
        self,
        text_config,
        k=256,
        d=256,
        sos_id=128000,
        loss_weights='0.05-0.3-0.3-0.2-0.15',
        delay=1,
        delay_level='word',
        audio_embed_conv_mode='fill_forward',
        in_llm_module='weighted_sum',
        out_llm_module='continue_weighted_layer',
        _attn_implementation="flash_attention_2",
        use_lora=False,
        kwargs_for_lora=None,
    ):
        super().__init__()

        self.fuse_for_bridge_in_llm = BRIDGE_FUSION_CLASSES[in_llm_module](
            llm_dim=text_config.hidden_size,
        )

        self.language_model = AutoModelForCausalLM.from_pretrained(
            text_config._name_or_path, attn_implementation=_attn_implementation
        )
        # cast llm to bfloat16
        self.language_model = self.language_model.to(torch.bfloat16)

        self._use_lora = use_lora
        if self._use_lora:
            from peft import LoraConfig, get_peft_model

            # build lora_config
            lora_target_modules = list(kwargs_for_lora['lora_target_modules'] or [])

            if kwargs_for_lora['lora_target_linear']:
                linear_names = _find_all_linear_names(self.language_model)
                print(f"found linear modules: {repr(linear_names)}")
                lora_target_modules = list(set(lora_target_modules + linear_names))

            lora_config = LoraConfig(
                r=kwargs_for_lora['lora_r'],
                lora_alpha=kwargs_for_lora['lora_alpha'],
                target_modules=lora_target_modules,
                layers_to_transform=None,
                lora_dropout=kwargs_for_lora['lora_dropout'],
                fan_in_fan_out=kwargs_for_lora['lora_fan_in_fan_out'],
                modules_to_save=kwargs_for_lora['lora_modules_to_save'] if kwargs_for_lora['lora_modules_to_save'] else None,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.language_model = get_peft_model(self.language_model, lora_config)

        self.extract_for_bridge_out_llm = BRIDGE_EXTRACT_CLASSES[out_llm_module](
            d=d,
            k=k,
            l=4,
            llm_dim=text_config.hidden_size,
            llm_num_hidden_layers=text_config.num_hidden_layers
        )

        self.do_continue_predict = 'continue_' in out_llm_module
        self.do_multihead = 'multi_' in out_llm_module

        self.sos_id = sos_id

        self.k = k
        self.d = d

        self.ce_loss_module = nn.CrossEntropyLoss(reduction="mean", ignore_index=IGNORE_ID)
        self.kl_loss_module = nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.mse_loss_module = nn.MSELoss()

        loss_weights = [float(x) for x in loss_weights.split('-')]
        assert 1.0 - 1e-5 < sum(loss_weights) < 1.0 + 1e-5
        self.loss_weights = loss_weights

        self.delay = delay
        self.delay_level = delay_level

        if self.delay > 0:
            self.pad_text_unit_embed = nn.parameter.Parameter(
                torch.zeros(text_config.hidden_size, dtype=torch.float32)
            )
            self.pad_audio_unit_embed = nn.parameter.Parameter(
                torch.zeros(1280, dtype=torch.float32)
            )

        self.audio_embed_conv_mode = audio_embed_conv_mode
        if self.audio_embed_conv_mode in ['pad', 'embed_a0_only', 'get_codes_from_indices']:
            self.empty_audio_unit_embed = nn.parameter.Parameter(
                torch.zeros(1280, dtype=torch.float32)
            )
        if self.audio_embed_conv_mode == 'embed_a0_only':
            self.a0_embedding = nn.Embedding(k, 1280)
        if self.audio_embed_conv_mode == 'get_codes_from_indices':
            self.code_linear = nn.Linear(d, 1280)

    def register_taste_sampler(self, llm_tokenizer, text_top_p=0.0, taste_top_p=0.0, 
                               text_temperature=1.0, repetition_penalty=1.0):
        self.taste_sampler = TasteSampler(
            self.delay,
            self.delay_level,
            len(llm_tokenizer),
            llm_tokenizer,
            text_top_p=text_top_p,
            taste_top_p=taste_top_p,
            text_temperature=text_temperature,
            repetition_penalty=repetition_penalty,
        )

    def _fill_indices_forward(self, x):
        B, T, D = x.shape
        valid_mask = (x != -1).all(dim=-1)
        indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        valid_indices = torch.where(valid_mask, indices,
            torch.full_like(indices, -1))
        cummax_idx, _ = torch.cummax(valid_indices, dim=1)
        cummax_idx = cummax_idx.long()
        filled = torch.gather(x, 1, cummax_idx.unsqueeze(-1).expand(-1, -1, D))

        return filled

    def encode_audio(
            self,
            llm_indices,
            vq_module,
        ):

        # encod audio token
        if self.audio_embed_conv_mode == 'pad':
            valid_mask = (llm_indices != -1).all(dim=-1)
            zeroed_llm_indices = torch.where(llm_indices >= 0, llm_indices, 0)
            audio_embed = vq_module.get_output_from_indices(zeroed_llm_indices)
            taste_token_embeds = torch.where(
                valid_mask.unsqueeze(-1).repeat(1, 1, audio_embed.size(-1)),
                audio_embed,
                self.empty_audio_unit_embed.unsqueeze(0).unsqueeze(0)
            )
        elif self.audio_embed_conv_mode == 'fill_forward':
            filled_llm_indices = self._fill_indices_forward(llm_indices)
            taste_token_embeds = vq_module.get_output_from_indices(filled_llm_indices)
        elif self.audio_embed_conv_mode == 'embed_a0_only':
            a0_indices = llm_indices[:,:,0]
            valid_mask = (a0_indices != -1)
            zeroed_a0_indices = torch.where(valid_mask, a0_indices, 0)
            audio_embed = self.a0_embedding(zeroed_a0_indices)
            taste_token_embeds = torch.where(
                valid_mask.unsqueeze(-1).repeat(1, 1, audio_embed.size(-1)),
                audio_embed,
                self.empty_audio_unit_embed.unsqueeze(0).unsqueeze(0)
            )
        elif self.audio_embed_conv_mode == 'get_codes_from_indices':
            valid_mask = (llm_indices != -1).all(dim=-1)
            zeroed_llm_indices = torch.where(llm_indices >= 0, llm_indices, 0)
            B, T, _ = zeroed_llm_indices.shape
            codes = self.get_codes_from_indices(zeroed_llm_indices)
            codes_summed = reduce(codes, 'q ... -> ...', 'sum')
            audio_embed = self.code_linear(codes_summed)
            taste_token_embeds = torch.where(
                valid_mask.unsqueeze(-1).repeat(1, 1, audio_embed.size(-1)),
                audio_embed,
                self.empty_audio_unit_embed.unsqueeze(0).unsqueeze(0)
            )

        return taste_token_embeds

    def _prepare_single(self, 
            llm_embed_tokens, vq_module, single_indices, single_token_ids, single_word_ids,
            output_audio_embed=False):

        sos_emb = llm_embed_tokens.weight[self.sos_id].reshape(1, -1)

        # text embeds for fusion
        if self.delay == 0:
            # no delay
            text_embeds = llm_embed_tokens(single_token_ids[:-1].unsqueeze(0))[0]

        elif self.delay > 0:
            # token delay and word delay
            text_embeds = torch.concat(
                [
                    llm_embed_tokens(single_token_ids.unsqueeze(0))[0],
                    self.pad_text_unit_embed.unsqueeze(0).repeat(self.delay, 1)
                ], dim=0)

        # audio embeds for fusion
        if self.delay == 0:
            taste = single_indices[:-1, :]
            audio_embeds = self.encode_audio(taste.unsqueeze(0), vq_module)[0]
            single_taste_labels = single_indices

        elif self.delay > 0 and self.delay_level == 'token':
            post_audio_embeds = self.encode_audio(single_indices.unsqueeze(0), vq_module)[0]
            audio_embeds = torch.concat(
                [
                    self.pad_audio_unit_embed.unsqueeze(0).repeat(self.delay, 1),
                    post_audio_embeds,
                ], dim=0)
            single_taste_labels = F.pad(single_indices, (0, 0, self.delay, 1), "constant", IGNORE_ID)

        elif self.delay > 0 and self.delay_level == 'word':
            device = single_word_ids.device

            keep_start_word_ids = torch.where(
                torch.diff(single_word_ids, prepend=torch.tensor([-1], device=device)) > 0,
                single_word_ids, IGNORE_ID)
            
            shifted = keep_start_word_ids - self.delay
            
            _nonzero = torch.nonzero(shifted == 0)
            if _nonzero.nelement() == 0:
                start_x = None
            else:
                start_x = int(_nonzero)

            word_number = single_word_ids.max() + 1

            if start_x is None:
                length = len(single_token_ids)
                
                full_indices = torch.ones(word_number, 4, dtype=torch.long).to(device) * IGNORE_ID
                for i in range(word_number):
                    full_indices[i, :] = single_indices[int(torch.nonzero(keep_start_word_ids == i)), :]

                audio_embeds = torch.concat(
                    [
                        self.pad_audio_unit_embed.unsqueeze(0).repeat(length + self.delay - word_number, 1),
                        self.encode_audio(full_indices.unsqueeze(0), vq_module)[0],
                    ], dim=0)
                single_taste_labels = torch.concat(
                    [
                        torch.ones(length + self.delay - word_number, 4, device=single_indices.device, dtype=single_indices.dtype) * IGNORE_ID,
                        full_indices,
                        torch.ones(1, 4, device=single_indices.device, dtype=single_indices.dtype) * IGNORE_ID,
                    ], dim=0)
                
            else:
                pre_indices = torch.ones_like(single_indices) * IGNORE_ID
                for i in shifted[shifted >= 0]:
                    i = int(i)
                    new_index = int(torch.nonzero(shifted == i))
                    old_index = int(torch.nonzero(keep_start_word_ids == i))
                    pre_indices[new_index, :] = single_indices[old_index, :]
        
                post_indices = torch.ones(self.delay, 4, dtype=torch.long).to(device) * IGNORE_ID
                
                for i in range(self.delay):
                    post_indices[i, :] = single_indices[
                        int(torch.nonzero(keep_start_word_ids == (word_number - self.delay + i))), :]

                audio_embeds = torch.concat(
                    [
                        self.pad_audio_unit_embed.unsqueeze(0).repeat(start_x, 1),
                        self.encode_audio(pre_indices[start_x:].unsqueeze(0), vq_module)[0],
                        self.encode_audio(post_indices.unsqueeze(0), vq_module)[0],
                    ], dim=0)
                single_taste_labels = torch.concat(
                    [
                        torch.ones(start_x, 4, device=single_indices.device, dtype=single_indices.dtype) * IGNORE_ID,
                        pre_indices[start_x:],
                        post_indices,
                        torch.ones(1, 4, device=single_indices.device, dtype=single_indices.dtype) * IGNORE_ID,
                    ], dim=0)

        # fuse
        assert len(text_embeds) == len(audio_embeds)
        fused_embeds = self.fuse_for_bridge_in_llm(text_embeds.unsqueeze(0), audio_embeds.unsqueeze(0))[0]

        single_input_embeds = torch.concat([sos_emb, fused_embeds], dim=0)

        if output_audio_embed:
            return single_input_embeds, single_taste_labels, audio_embeds
        return single_input_embeds, single_taste_labels

    def prepare_conditional_embeds(
        self,
        llm_indices, 
        llm_token_ids, 
        llm_token_lengths, 
        llm_word_ids,
        llm_embed_tokens,
        vq_module,
    ):
        llm_dtype = next(llm_embed_tokens.parameters()).dtype

        unpad_indices = unpad_sequence(llm_indices, llm_token_lengths.cpu(), batch_first=True)
        unpad_token_ids = unpad_sequence(llm_token_ids, llm_token_lengths.cpu(), batch_first=True)
        unpad_word_ids = unpad_sequence(llm_word_ids, llm_token_lengths.cpu(), batch_first=True)
        
        list_single_inputs_embeds = []
        list_single_taste_labels = []
        for i in range(len(llm_token_ids)):
            single_inputs_embeds, single_taste_labels = self._prepare_single(
                llm_embed_tokens, vq_module, 
                single_indices=unpad_indices[i], 
                single_token_ids=unpad_token_ids[i], 
                single_word_ids=unpad_word_ids[i]
            )

            list_single_inputs_embeds.append(single_inputs_embeds)
            list_single_taste_labels.append(single_taste_labels)

        inputs_embeds = pad_sequence(list_single_inputs_embeds, batch_first=True, padding_value=0.0)
        output_lengths = llm_token_lengths + self.delay + 1 if self.delay > 0 else llm_token_lengths
        attention_mask = generate_mask_from_length(output_lengths)

        inputs_embeds = inputs_embeds.to(llm_dtype)
        taste_labels = pad_sequence(list_single_taste_labels, batch_first=True, padding_value=IGNORE_ID)
        return inputs_embeds, attention_mask, output_lengths, taste_labels

    def _calcuate_loss_text_ce(self, text_logits, text_labels):
        B, T, C = text_logits.shape
        ce_loss = self.ce_loss_module(text_logits.view((B * T, C)), text_labels.view((B * T,)))
        return ce_loss
    
    def _calcuate_loss_text_kl(self, ref_model, llm_token_ids, text_logits, text_labels):
        with torch.no_grad():
            ref_model.eval()
            input_ids_for_ref = F.pad(llm_token_ids, (1, 0), 'constant', self.sos_id)
            target_llm_output = ref_model(
                input_ids=input_ids_for_ref
            )
            target = F.softmax(target_llm_output.logits.detach(), dim=-1)
            target_max_length = target.size(1)
            mask = (text_labels[:, :target_max_length] != IGNORE_ID)

        log_p = F.log_softmax(text_logits, dim=-1)[:, :target_max_length, :]
        kl_div_loss = self.kl_loss_module(log_p[mask], target[mask])
        return kl_div_loss

    def _calcuate_loss_taste_mse(self, vq_module, taste_logits, taste_labels, z=None, mu=None, logvar=None, agg_code=None):
        vq_module.eval()
        valid_mask = (taste_labels != IGNORE_ID).all(dim=-1)

        if agg_code is not None:
            summed_code_target = vq_module.get_code_from_indices(taste_labels[valid_mask]).view(-1, self.d)
            summed_code_pred = agg_code[valid_mask].view(-1, self.d)
            taste_loss = self.mse_loss_module(summed_code_pred, summed_code_target)

        elif z is not None:
            valid_mask = (taste_labels != IGNORE_ID).all(dim=-1)
            summed_code_target = vq_module.get_code_from_indices(taste_labels[valid_mask])

            l_reg = self.mse_loss_module(z[valid_mask], summed_code_target)
            if self.extract_for_bridge_out_llm.b_logvar_is_linear:
                logvar = logvar[valid_mask]
            l_kl = 0.5 * torch.mean(
                torch.mean(torch.exp(logvar) + (mu[valid_mask] - summed_code_target)**2 - 1 - logvar, dim=-1) # we use mean to avoid too diverged value the loss term
            )
            taste_loss = 0.5 * l_reg + 0.5 * l_kl

        else:
            summed_code_target = vq_module.get_code_from_indices(taste_labels[valid_mask])

            predictions = taste_logits.argmax(-1)
            summed_code_pred = vq_module.get_code_from_indices(predictions[valid_mask])
            taste_loss = self.mse_loss_module(summed_code_pred, summed_code_target)

        return taste_loss

    def _calcuate_loss_taste_ce(self, taste_logits, taste_labels):
        taste_loss_list = []
        for i in range(4):
            taste_logits_at_layer = taste_logits[:, :, i, :]
            taste_labels_at_layer = taste_labels[:, :, i]
            B, T, C = taste_logits_at_layer.shape
            taste_loss_at_layer = self.ce_loss_module(
                taste_logits_at_layer.view((B * T, C)),
                taste_labels_at_layer.view((B * T,))
            )
            taste_loss_list.append(taste_loss_at_layer)
        return taste_loss_list

    def forward(
        self,
        llm_indices, 
        llm_token_ids, 
        llm_token_lengths, 
        llm_word_ids,
        vq_module,
        output_loss_at_inference=False,
        **kwargs
    ):
        vq_module.eval()

        if self._use_lora:
            base = self.language_model.base_model.model
        else:
            base = self.language_model

        llm_embed_tokens = base.model.embed_tokens
        llm_backbone = base.model
        lm_head = base.lm_head

        inputs_embeds, attention_mask, output_lengths, taste_labels = self.prepare_conditional_embeds(
            llm_indices, 
            llm_token_ids, 
            llm_token_lengths, 
            llm_word_ids,
            llm_embed_tokens,
            vq_module,
        )

        llm_outputs = llm_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict='pt'
        )

        text_logits = lm_head(llm_outputs.last_hidden_state)
        taste_logits, training_info = self.extract_for_bridge_out_llm(llm_outputs, vq_module)

        text_labels = torch.where(
            attention_mask > 0,
            F.pad(llm_token_ids, (0, self.delay + 1), "constant", IGNORE_ID),
            IGNORE_ID
        )

        if self.training or output_loss_at_inference:
            ref_model = kwargs.get('ref_model', None)
            if ref_model:
                kl_div_loss = self._calcuate_loss_text_kl(ref_model, llm_token_ids, text_logits, text_labels)
                ce_loss = self._calcuate_loss_text_ce(text_logits, text_labels)
                text_loss = 0.9 * kl_div_loss + 0.1 * ce_loss
            else:
                text_loss = self._calcuate_loss_text_ce(text_logits, text_labels)

            if self.do_continue_predict:
                z, mu, logvar = (training_info[n] for n in ('z', 'mu', 'logvar'))
                taste_loss = self._calcuate_loss_taste_mse(vq_module, taste_logits, taste_labels, z=z, mu=mu, logvar=logvar)

            elif self.do_multihead:
                agg_code = training_info['agg_code']
                mse_loss = self._calcuate_loss_taste_mse(vq_module, taste_logits, taste_labels, agg_code=agg_code)
                ce_loss_list = self._calcuate_loss_taste_ce(taste_logits, taste_labels)
                ce_loss = sum(ce_loss_list) / len(ce_loss_list)
                taste_loss = (mse_loss + ce_loss) / 2

            else:
                taste_loss_list = self._calcuate_loss_taste_ce(taste_logits, taste_labels)
                taste_loss = sum(taste_loss_list) / len(taste_loss_list)

            loss = self.loss_weights[0] * text_loss + self.loss_weights[1] * taste_loss

        else:
            loss = None

        return dict(
            loss=loss,
            text_logits=text_logits,
            text_labels=text_labels,
            taste_logits=taste_logits,
            taste_labels=taste_labels,
            output_lengths=output_lengths,
        )

    def get_audio_embeds_from_taste(
            self, vq_module, asr_token_lengths, asr_word_ids, 
            llm_taste_preds=None, llm_taste_logits=None, llm_taste_labels=None,
            asr_taste_indices=None):
        """
        Convert TASTE predictions to audio embeddings for speech synthesis.
        
        This method supports two main workflows:
        1. Direct input: Provide pre-computed asr_taste_indices
        2. LLM output: Provide LLM predictions/logits/labels to compute indices
        
        Args:
            vq_module: Vector quantization module for converting indices to embeddings
            asr_token_lengths (torch.Tensor): Length of each ASR sequence [batch_size]
            asr_word_ids (torch.Tensor): Word-level alignment indices [batch_size, seq_len]
            
            # LLM TASTE outputs (choose one combination):
            llm_taste_preds (torch.Tensor, optional): Pre-computed LLM predictions 
                [batch_size, llm_seq_len, num_taste_tokens]
            llm_taste_logits (torch.Tensor, optional): Raw LLM logits for TASTE tokens
                [batch_size, llm_seq_len, vocab_size] 
            llm_taste_labels (torch.Tensor, optional): Ground truth TASTE labels
                [batch_size, llm_seq_len, num_taste_tokens]
                
            # Direct input (bypasses LLM processing):
            asr_taste_indices (torch.Tensor, optional): Pre-computed ASR-aligned indices
                [batch_size, asr_seq_len, num_taste_tokens]
        
        Parameter Combinations:
            Option 1: asr_taste_indices (direct, most efficient)
            Option 2: llm_taste_preds + asr_word_ids mapping
            Option 3: llm_taste_logits + llm_taste_labels (training mode)
            
        Returns:
            tuple: (audio_unit_embeds, audio_unit_lengths)
                - audio_unit_embeds (torch.Tensor): Audio embeddings [batch_size, asr_seq_len, embed_dim]
                - audio_unit_lengths (torch.Tensor): Actual lengths [batch_size]
                
        Raises:
            AssertionError: If word alignment dimensions don't match
            AttributeError: If all input parameters are None
            
        Note:
            At least one of {asr_taste_indices, llm_taste_preds, (llm_taste_logits + llm_taste_labels)} 
            must be provided.
        """

        if asr_taste_indices is not None:
            device = asr_taste_indices.device
        else:
            device = llm_taste_logits.device if llm_taste_preds is None else llm_taste_preds.device

        if asr_taste_indices is None:
            if llm_taste_preds is None:
                # use `llm_taste_labels`` and `llm_taste_logits`` to get `llm_taste_preds`
                llm_taste_preds = torch.where(
                    llm_taste_labels != IGNORE_ID, 
                    llm_taste_logits.argmax(dim=-1), 
                    IGNORE_ID)

            def _map_to_asr_taste_indices(asr_word_ids, asr_token_lengths, llm_taste_preds):
                unpad_asr_word_ids = unpad_sequence(asr_word_ids, asr_token_lengths.cpu(), batch_first=True)
                sequences = []
                for i, single_asr_word_ids in enumerate(unpad_asr_word_ids):
                    seq_mask = llm_taste_preds[i, :, 0] != IGNORE_ID
                    single_reduced_llm_taste_preds = llm_taste_preds[i, seq_mask, :].long()
                    assert single_reduced_llm_taste_preds.size(0) == int(single_asr_word_ids[-1]) + 1
                    single_asr_taste_indices = torch.index_select(single_reduced_llm_taste_preds, 0, single_asr_word_ids.long())
                    # assert single_audio_embeds.size(0) == int(asr_token_lengths[i])
                    sequences.append(single_asr_taste_indices)
                asr_taste_indices = pad_sequence(sequences, batch_first=True, padding_value=0)
                return asr_taste_indices
            asr_taste_indices = _map_to_asr_taste_indices(asr_word_ids, asr_token_lengths, llm_taste_preds)
        
        asr_seq_mask = generate_mask_from_length(asr_token_lengths)
        audio_unit_embeds = (vq_module.get_output_from_indices(asr_taste_indices) * asr_seq_mask.unsqueeze(-1)).to(device)
        audio_unit_lengths = asr_token_lengths.to(device)
        return audio_unit_embeds, audio_unit_lengths

    @torch.no_grad()
    def generate(
        self,
        vq_module,
        conditional_mode,
        llm_indices=None, 
        llm_token_ids=None, 
        llm_token_lengths=None, 
        llm_word_ids=None,
        extra_words=32,
        **kwargs
    ):
        vq_module.eval()

        assert ((llm_indices is None) or (llm_indices.size(0) == 1)), 'batch size only allow 1 when `spoken_lm.conditional_generate`'
        
        if self._use_lora:
            base = self.language_model.base_model.model
        else:
            base = self.language_model

        llm_embed_tokens = base.model.embed_tokens
        llm_backbone = base.model
        lm_head = base.lm_head
        llm_dtype = next(llm_embed_tokens.parameters()).dtype

        if conditional_mode == 'text':
            has_prefix = False
            stop_id = None
        elif conditional_mode == 'instruct':
            has_prefix = False
            stop_id = kwargs.get('stop_id')
        else:
            has_prefix = (llm_token_ids is not None)
            stop_id = None
        self.taste_sampler.reset(extra_words=extra_words, has_prefix=has_prefix, stop_id=stop_id)

        device = base.device

        pending_audio_embed = None

        if conditional_mode == 'zero':
            inputs_embeds = llm_embed_tokens.weight[self.sos_id].reshape(1, 1, -1)
            input_ids = torch.tensor([[self.sos_id]], device=device)
        elif conditional_mode == 'text':
            inputs_embeds = llm_embed_tokens(llm_token_ids)
            input_ids = llm_token_ids
        elif conditional_mode == 'audio':
            text_input_length = llm_token_lengths[0].item() + 1
            single_inputs_embeds, single_taste_labels, single_audio_embed = self._prepare_single(
                llm_embed_tokens, vq_module, 
                single_indices=llm_indices[0], 
                single_token_ids=llm_token_ids[0], 
                single_word_ids=llm_word_ids[0],
                output_audio_embed=True
            )
            inputs_embeds = single_inputs_embeds[:text_input_length, :].unsqueeze(0).to(dtype=llm_dtype, device=device)
            pending_audio_embed = single_audio_embed[text_input_length - 1:, :]
            input_ids = llm_token_ids
        elif conditional_mode == 'instruct':
            single_inputs_embeds, single_taste_labels, single_audio_embed = self._prepare_single(
                llm_embed_tokens, vq_module, 
                single_indices=llm_indices[0], 
                single_token_ids=llm_token_ids[0], 
                single_word_ids=llm_word_ids[0],
                output_audio_embed=True
            )
            text_input_length = llm_token_lengths[0].item() + 1
            inputs_embeds = single_inputs_embeds[1:text_input_length, :].unsqueeze(0).to(dtype=llm_dtype, device=device)
            input_ids = llm_token_ids[:, 1:]

            instruct_prefix_ids = kwargs.get('instruct_prefix_ids').view(1, -1)
            instruct_suffix_ids = kwargs.get('instruct_suffix_ids').view(1, -1)
            instruct_prefix_embeds = llm_embed_tokens(instruct_prefix_ids)
            instruct_suffix_embeds = llm_embed_tokens(instruct_suffix_ids)

            inputs_embeds = torch.concat([instruct_prefix_embeds, inputs_embeds, instruct_suffix_embeds], dim=1)
            input_ids = torch.concat([instruct_prefix_ids, input_ids, instruct_suffix_ids], dim=1)

        generated_llm_indices, generated_llm_token_ids, generated_llm_token_lengths, generated_llm_word_ids = None, None, None, None

        while True:
            llm_outputs = llm_backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                output_hidden_states=True,
                return_dict='pt'
            )

            text_logits = lm_head(llm_outputs.last_hidden_state)
            taste_logits, _ = self.extract_for_bridge_out_llm(llm_outputs, vq_module)

            text_id, taste_ids, action, taste_action = \
                self.taste_sampler.update(text_logits, taste_logits, input_ids=input_ids)
            input_ids = F.pad(input_ids, (0, 1), 'constant', text_id)

            if action != 'wait_for_taste' and action != 'terminate':
                # update llm_token_ids
                append_llm_token_ids = torch.tensor([[text_id]], device=device, dtype=torch.int64)
                if generated_llm_token_ids is None:
                    generated_llm_token_ids = append_llm_token_ids
                else:
                    generated_llm_token_ids = torch.concat([
                        generated_llm_token_ids,
                        append_llm_token_ids,
                    ], dim=1)

                # update llm_token_lengths
                if generated_llm_token_lengths is None:
                    generated_llm_token_lengths = torch.ones(1, 1, device=device, dtype=torch.int32)
                else:
                    generated_llm_token_lengths += 1

            # update llm_word_ids
            if action == 'continue_at_word_start':
                if generated_llm_word_ids is None:
                    generated_llm_word_ids = torch.zeros(1, 1, device=device, dtype=torch.int32)
                else:
                    generated_llm_word_ids = torch.concat([
                        generated_llm_word_ids,
                        torch.tensor([[generated_llm_word_ids[0, -1].item() + 1]], device=device, dtype=torch.int32)
                    ], dim=1)
            elif action == 'continue_not_at_word_start':
                generated_llm_word_ids = torch.concat([
                    generated_llm_word_ids,
                    torch.tensor([[generated_llm_word_ids[0, -1].item()]], device=device, dtype=torch.int32)
                ], dim=1)

            if taste_action == 'sample':
                # update llm_indices
                if generated_llm_indices is None:
                    generated_llm_indices = taste_ids
                else:
                    generated_llm_indices = torch.concat([
                        generated_llm_indices,
                        taste_ids
                    ], dim=1)

                is_taste_word_start = taste_ids[0, 0, 0].item() != IGNORE_ID
                if is_taste_word_start:
                    last_asr_embed = self.encode_audio(taste_ids, vq_module)
                new_inputs_embeds = self.fuse_for_bridge_in_llm(
                    llm_embed_tokens.weight[text_id].reshape(1, 1, -1),
                    last_asr_embed
                )

            elif taste_action.startswith('use_prefix'):
                if taste_action == 'use_prefix':
                    assert pending_audio_embed is not None and pending_audio_embed.size(0) > 0
                    last_asr_embed = pending_audio_embed[0, :].reshape(1, 1, -1)
                    if pending_audio_embed.size(0) == 1:
                        pending_audio_embed = None
                    else:
                        pending_audio_embed = pending_audio_embed[1:, :]

                new_inputs_embeds = self.fuse_for_bridge_in_llm(
                    llm_embed_tokens.weight[text_id].reshape(1, 1, -1),
                    last_asr_embed
                )

            else:
                new_inputs_embeds = self.fuse_for_bridge_in_llm(
                    llm_embed_tokens.weight[text_id].reshape(1, 1, -1),
                    self.pad_audio_unit_embed.reshape(1, 1, -1)
                )

            inputs_embeds = torch.concat([
                inputs_embeds,
                new_inputs_embeds
            ], dim=1).to(dtype=llm_dtype, device=device)

            
            # end
            if action == 'terminate':
                break

        return (generated_llm_indices, generated_llm_token_ids, generated_llm_token_lengths, generated_llm_word_ids)


class TastePreTrainedModel(PreTrainedModel):
    config_class = TasteConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        # important: this ported version of Taste isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed 
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


@dataclass
class TasteTTSOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    speech_logits: torch.FloatTensor = None
    speech_labels: Optional[torch.LongTensor] = None


@dataclass
class TasteCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    text_logits: torch.FloatTensor = None
    text_labels: Optional[torch.LongTensor] = None
    taste_logits: torch.FloatTensor = None
    taste_labels: Optional[torch.LongTensor] = None
    speech_logits: torch.FloatTensor = None
    speech_labels: Optional[torch.LongTensor] = None


class TasteForCausalLM(TastePreTrainedModel, GenerationMixin):
    config_class = TasteConfig

    def __init__(self, config: TasteConfig):
        super().__init__(config)

        # Audio Tower
        if hasattr(config, 'audio_tower_config'):
            if isinstance(config.audio_tower_config, dict):
                self.audio_tower_config = TasteAudioTowerConfig(**config.audio_tower_config)
            else:
                self.audio_tower_config = config.audio_tower_config

            self.audio_tower = TasteAudioTower(
                encoder_input_size=self.audio_tower_config.encoder_input_size,
                text_token_size=self.audio_tower_config.text_token_size,
                audio_embed_dim=self.audio_tower_config.audio_embed_dim, # the size of the audio quantized vector (added)
                quantization_on=self.audio_tower_config.quantization_on,
                is_joint_encoder_segmenter=self.audio_tower_config.is_joint_encoder_segmenter,
                audio_dropout_ratio=self.audio_tower_config.audio_dropout_ratio,
                kwargs_audio_encoder={
                    'target_hidden_layer': self.audio_tower_config.encoder__target_hidden_layer,
                    'unfreeze_hidden_layers_from_last': self.audio_tower_config.encoder__unfreeze_hidden_layers_from_last,
                    'whisper_config': config.asr_config,
                },
                kwargs_audio_segmenter={
                    'attn_implementation': config._attn_implementation,
                },
                kwargs_for_joint_encoder_segmenter=self.audio_tower_config.kwargs_for_joint_encoder_segmenter,
                kwargs_for_quantizer=self.audio_tower_config.kwargs_for_quantizer,
            )

        # Speech Decoder
        if hasattr(config, 'speech_decoder_config'):
            if isinstance(config.speech_decoder_config, dict):
                self.speech_decoder_config = TasteSpeechDecoderConfig(**config.speech_decoder_config)
            else:
                self.speech_decoder_config = config.speech_decoder_config

            if self.speech_decoder_config.fuse_encoded_audio_text_type == 'concat_with_sep' and 'd' not in self.speech_decoder_config.fuse_encoded_audio_text_kwargs:
                self.speech_decoder_config.fuse_encoded_audio_text_kwargs['d'] = self.speech_decoder_config.llm_input_size

            self.speech_decoder = TasteSpeechDecoder(
                encoder_input_size=self.speech_decoder_config.encoder_input_size,
                audio_encoder_input_size=self.speech_decoder_config.audio_encoder_input_size,
                llm_input_size=self.speech_decoder_config.llm_input_size,
                llm_output_size=self.speech_decoder_config.llm_output_size,
                text_token_size=self.speech_decoder_config.text_token_size,
                speech_token_size=self.speech_decoder_config.speech_token_size,
                length_normalized_loss=self.speech_decoder_config.length_normalized_loss,
                lsm_weight=self.speech_decoder_config.lsm_weight,
                spk_embed_dim=self.speech_decoder_config.spk_embed_dim,
                skip_prefix_idx=self.speech_decoder_config.skip_prefix_idx,
                kwargs_cosyvoice_encoder={
                    "attention_heads": self.speech_decoder_config.encoder__attention_heads,
                    "linear_units": self.speech_decoder_config.encoder__linear_units,
                    "num_blocks": self.speech_decoder_config.encoder__num_blocks,
                    "dropout_rate": self.speech_decoder_config.encoder__dropout_rate,
                    "positional_dropout_rate": self.speech_decoder_config.encoder__positional_dropout_rate,
                    "attention_dropout_rate": self.speech_decoder_config.encoder__attention_dropout_rate,
                    "normalize_before": self.speech_decoder_config.encoder__normalize_before,
                    "input_layer": self.speech_decoder_config.encoder__input_layer,
                    "pos_enc_layer_type": self.speech_decoder_config.encoder__pos_enc_layer_type,
                    "selfattention_layer_type": self.speech_decoder_config.encoder__selfattention_layer_type,
                    "use_cnn_module": self.speech_decoder_config.encoder__use_cnn_module,
                    "macaron_style": self.speech_decoder_config.encoder__macaron_style,
                    "use_dynamic_chunk": self.speech_decoder_config.encoder__use_dynamic_chunk,
                    "use_dynamic_left_chunk": self.speech_decoder_config.encoder__use_dynamic_left_chunk,
                    "static_chunk_size": self.speech_decoder_config.encoder__static_chunk_size,
                },
                kwargs_cosyvoice_llm={
                    "attention_heads": self.speech_decoder_config.llm__attention_heads,
                    "linear_units": self.speech_decoder_config.llm__linear_units,
                    "num_blocks": self.speech_decoder_config.llm__num_blocks,
                    "dropout_rate": self.speech_decoder_config.llm__dropout_rate,
                    "positional_dropout_rate": self.speech_decoder_config.llm__positional_dropout_rate,
                    "attention_dropout_rate": self.speech_decoder_config.llm__attention_dropout_rate,
                    "input_layer": self.speech_decoder_config.llm__input_layer,
                    "pos_enc_layer_type": self.speech_decoder_config.llm__pos_enc_layer_type,
                    "selfattention_layer_type": self.speech_decoder_config.llm__selfattention_layer_type,
                    "static_chunk_size": self.speech_decoder_config.llm__static_chunk_size,
                },
                fuse_encoded_audio_text_type=self.speech_decoder_config.fuse_encoded_audio_text_type,
                fuse_encoded_audio_text_kwargs=self.speech_decoder_config.fuse_encoded_audio_text_kwargs,
            )

        # Spoken LM
        if hasattr(config, 'spoken_lm_config'):
            if isinstance(config.spoken_lm_config, dict):
                self.spoken_lm_config = TasteSpokenLMConfig(**config.spoken_lm_config)
            else:
                self.spoken_lm_config = config.spoken_lm_config

            self.spoken_lm = TasteSpokenLM(
                config.text_config, 
                k=self.audio_tower_config.kwargs_for_quantizer['codebook_size'],
                d=self.audio_tower_config.kwargs_for_quantizer['codebook_dim'],
                sos_id=self.spoken_lm_config.sos_id,
                loss_weights=self.spoken_lm_config.loss_weights,
                delay=self.spoken_lm_config.delay,
                delay_level=self.spoken_lm_config.delay_level,
                audio_embed_conv_mode=self.spoken_lm_config.audio_embed_conv_mode,
                in_llm_module=self.spoken_lm_config.in_llm_module,
                out_llm_module=self.spoken_lm_config.out_llm_module,
                _attn_implementation=config._attn_implementation,
                use_lora=self.spoken_lm_config.use_lora,
                kwargs_for_lora=self.spoken_lm_config.kwargs_for_lora,
            )

        self.weight_commit_loss = 1.0  # TODO: add to config

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

        self._mode = 'SpokenLLM'

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        model._mode = 'SpokenLLM'

        if model.spoken_lm._use_lora:
            llm = model.spoken_lm.language_model.base_model.model
        else:
            llm = model.spoken_lm.language_model

        # copy tie embedding
        use_copy_tie_embedding = False
        if use_copy_tie_embedding:
            embedding_weight = llm.model.embed_tokens.weight.clone().detach()
            llm.lm_head.weight = torch.nn.Parameter(embedding_weight.clone())
            assert llm.lm_head.weight.data_ptr() != llm.model.embed_tokens.weight.data_ptr(), "Weights are still tied!"

        return model

    @classmethod
    def from_pretrained_stage1(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        _skip_audio_in_audio_decoder = kwargs.pop('skip_audio_in_audio_decoder', False)
        _skip_vq_in_audio_encoder = kwargs.pop('skip_vq_in_audio_encoder', False)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        model._mode = 'SpeechAutoEncoder'
        model._skip_audio_in_audio_decoder = _skip_audio_in_audio_decoder
        model._skip_vq_in_audio_encoder = _skip_vq_in_audio_encoder

        return model

    def reload_language_model(self, path):
        self.spoken_lm.language_model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

    def freeze_language_model(self):
        for name, params in self.spoken_lm.language_model.named_parameters():
            params.requires_grad = False

    def freeze_audio_tower(self):
        for name, params in self.audio_tower.named_parameters():
            params.requires_grad = False

    def freeze_speech_decoder(self):
        for name, params in self.speech_decoder.named_parameters():
            params.requires_grad = False

    def apply_lora(self, lora_config):
        self.spoken_lm.apply_lora(lora_config)

    def load_from_cosyvoice_ckpt(self, ckpt_fpath):
        self.audio_tower.load_from_cosyvoice_ckpt(ckpt_fpath)
        self.speech_decoder.load_from_cosyvoice_ckpt(ckpt_fpath)

    def _get_word_start_mapping_matrix(self, source_word_ids, target_word_ids, source_lengths, target_lengths):
        source_mask = generate_mask_from_length(source_lengths)
        target_mask = generate_mask_from_length(target_lengths)
        mapping_matrix = (
            (source_word_ids.unsqueeze(1) == target_word_ids.unsqueeze(2)) 
            & source_mask.unsqueeze(1) 
            & target_mask.unsqueeze(2)
        ).float()
        
        # keep word-start
        word_start_mapping_matrix = (torch.cumsum(mapping_matrix, dim=-1) == 1).float() * mapping_matrix
        word_start_mapping_matrix = (torch.cumsum(word_start_mapping_matrix, dim=-2) == 1).float() * mapping_matrix
        return word_start_mapping_matrix

    def _convert_embeds_between_token_spaces(self, 
            embeds, source_word_ids, target_word_ids, source_lengths, target_lengths):
        
        source_mask = generate_mask_from_length(source_lengths)
        target_mask = generate_mask_from_length(target_lengths)
        mapping_matrix = (
            (source_word_ids.unsqueeze(1) == target_word_ids.unsqueeze(2)) 
            & source_mask.unsqueeze(1) 
            & target_mask.unsqueeze(2)
        ).float()
        
        # keep word-end
        word_end_mapping_matrix = (torch.flip(
            torch.cumsum(
                torch.flip(mapping_matrix, dims=[-1]),
                dim=-1
            ),
            dims=[-1],
        ) == 1).float() * mapping_matrix

        converted_embeds = torch.bmm(word_end_mapping_matrix, embeds)
        return converted_embeds

    def forward(
        self,
        speaker_embeds=None,
        asr_token_ids=None,
        asr_token_lengths=None,
        asr_word_ids=None,
        llm_token_ids=None,
        llm_token_lengths=None,
        llm_word_ids=None,
        audio_features=None,
        audio_feature_lengths=None,
        speech_token_ids=None,
        speech_token_lengths=None,

        llm_indices=None,

        asr_token_alignments=None,
        llm_prev_token_ids=None,
        llm_post_token_ids=None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:

        if self._mode == 'SpokenLLM':
            vq_module = self.audio_tower.vq.rvq
            outputs = self.spoken_lm(
                llm_indices=llm_indices, 
                llm_token_ids=llm_token_ids, 
                llm_token_lengths=llm_token_lengths, 
                llm_word_ids=llm_word_ids,
                vq_module=vq_module,
                ref_model=kwargs.get('ref_model', None),
            )
            do_measure_speech = (speaker_embeds is not None) and (asr_token_ids is not None) and (asr_token_lengths is not None) \
                and (asr_word_ids is not None) and (speech_token_ids is not None) and (speech_token_lengths is not None)
            if do_measure_speech:
                audio_unit_embeds, audio_unit_lengths = self.spoken_lm.get_audio_embeds_from_taste(
                    vq_module=vq_module,
                    llm_taste_logits=outputs['taste_logits'], llm_taste_labels=outputs['taste_labels'],
                    asr_token_lengths=asr_token_lengths, asr_word_ids=asr_word_ids)
                decoded = self.speech_decoder(
                    speaker_embeds=speaker_embeds,
                    asr_token_ids=asr_token_ids,
                    asr_token_lengths=asr_token_lengths,
                    speech_token_ids=speech_token_ids,
                    speech_token_lengths=speech_token_lengths,
                    audio_unit_embeds=audio_unit_embeds,
                    audio_unit_lengths=audio_unit_lengths,
                )
            return TasteCausalLMOutputWithPast(
                loss=outputs['loss'],
                text_logits=outputs['text_logits'],
                text_labels=outputs['text_labels'],
                taste_logits=outputs['taste_logits'],
                taste_labels=outputs['taste_labels'],
                speech_logits=decoded['logits'] if do_measure_speech else None,
                speech_labels=decoded['labels'] if do_measure_speech else None,
            )
        elif self._mode == 'SpeechAutoEncoder':
            audio_encoded = self.audio_tower(
                asr_token_ids=asr_token_ids,
                asr_token_lengths=asr_token_lengths,
                audio_features=audio_features,
                audio_feature_lengths=audio_feature_lengths,
                asr_word_ids=asr_word_ids,
                skip_vq_in_audio_encoder=self._skip_vq_in_audio_encoder,
            )
            decoded = self.speech_decoder(
                speaker_embeds=speaker_embeds,
                asr_token_ids=asr_token_ids,
                asr_token_lengths=asr_token_lengths,
                speech_token_ids=speech_token_ids,
                speech_token_lengths=speech_token_lengths,
                skip_audio_in_audio_decoder=self._skip_audio_in_audio_decoder,
                **audio_encoded,
            )
            return TasteTTSOutputWithPast(
                loss=(
                    decoded['loss'] + self.weight_commit_loss * (audio_encoded.get('commit_loss', 0.0))
                    if decoded['loss'] else None
                ),
                speech_logits=decoded['logits'],
                speech_labels=decoded['labels'],
            )

    def voice_decoder_generate(
            self,
            speaker_embeds,
            audio_unit_embeds,
            audio_unit_lengths,
            asr_token_ids,
            asr_token_lengths,
            prev_speech_ids=None
        ):

        # prepare conditional embeds
        (
            sos_eos_emb,
            speaker_embeds, 
            audio_text_token_encoded,
            audio_text_token_len, 
            task_id_emb
        ) \
        = self.speech_decoder.prepare_conditional_embeds(
            speaker_embeds,
            audio_unit_embeds,
            audio_unit_lengths,
            asr_token_ids,
            asr_token_lengths
        )

        # # prepare lm_input
        # if prev_speech_ids is not None and prev_speech_ids.numel() > 0:
        #     # Convert previous speech IDs to embeddings
        #     prev_speech_embeds = self.speech_decoder.speech_embedding(prev_speech_ids)
        #     # Calculate speech token lengths for previous speech IDs
        #     prev_speech_lengths = torch.tensor([prev_speech_ids.shape[1]], 
        #                                      device=prev_speech_ids.device, dtype=torch.long)
        #     speech_lm_input, speech_lm_input_len = self.speech_decoder.pad_unpad_sequence(
        #         sos_eos_emb,
        #         speaker_embeds, 
        #         audio_text_token_encoded,
        #         audio_text_token_len, 
        #         task_id_emb,
        #         speech_token_embeds=prev_speech_embeds,
        #         speech_token_lengths=prev_speech_lengths,
        #         padding_side='right'
        #     )
        # else:
        #     speech_lm_input, speech_lm_input_len = self.speech_decoder.pad_unpad_sequence(
        #         sos_eos_emb,
        #         speaker_embeds, 
        #         audio_text_token_encoded,
        #         audio_text_token_len, 
        #         task_id_emb,
        #         padding_side='right'
        #     )

        speech_lm_input, speech_lm_input_len = self.speech_decoder.pad_unpad_sequence(
            sos_eos_emb,
            speaker_embeds, 
            audio_text_token_encoded,
            audio_text_token_len, 
            task_id_emb,
            padding_side='right'
        )

        # print('speech_lm_input', speech_lm_input[0,:,0:5])
        # print('speech_lm_input_len', speech_lm_input_len)

        beam_size = 3
        sampling = 15
        max_token_text_ratio = 20
        min_token_text_ratio = 2

        min_len = int(speech_lm_input_len[0] * min_token_text_ratio)
        max_len = int(speech_lm_input_len[0] * max_token_text_ratio)

        device = speech_lm_input.device

        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=device), torch.zeros((0, 0, 0, 0), device=device)

        if prev_speech_ids is not None:
            for top_id in prev_speech_ids[0, :]:
                y_pred, att_cache, cnn_cache = self.speech_decoder.llm.forward_chunk(
                    speech_lm_input, 
                    offset=offset, 
                    required_cache_size=-1, 
                    att_cache=att_cache, 
                    cnn_cache=cnn_cache,
                    att_mask=torch.tril(torch.ones((1, speech_lm_input.shape[1], speech_lm_input.shape[1]), device=device)).to(torch.bool)
                )
                speech_lm_input = self.speech_decoder.speech_embedding.weight[top_id].reshape(1, 1, -1)
                offset += speech_lm_input.size(1)
            max_len -= prev_speech_ids.size(1)

        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.speech_decoder.llm.forward_chunk(
                speech_lm_input, 
                offset=offset, 
                required_cache_size=-1, 
                att_cache=att_cache, 
                cnn_cache=cnn_cache,
                att_mask=torch.tril(torch.ones((1, speech_lm_input.shape[1], speech_lm_input.shape[1]), device=device)).to(torch.bool)
            )
            logp = self.speech_decoder.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.speech_decoder.sampling_ids(
                logp.squeeze(dim=0), 
                sampling, 
                beam_size, 
                ignore_eos=True if i < min_len else False
            )[0].item()
            if top_ids == self.speech_decoder.speech_token_size:
                break
            out_tokens.append(top_ids)
            offset += speech_lm_input.size(1)
            speech_lm_input = self.speech_decoder.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            # print('speech_lm_input', speech_lm_input[0,:,0:5])

        speech_token_ids = torch.tensor([out_tokens], dtype=torch.int32, device=device)
        speech_token_lengths = torch.tensor([len(out_tokens)], dtype=torch.int32, device=device)

        return {
            'speech_token_ids': speech_token_ids,
            'speech_token_lengths': speech_token_lengths,
        }

    @torch.no_grad()
    def scoring(
            self,
            asr_token_ids,
            asr_token_lengths,
            asr_word_ids,
            llm_token_ids,
            llm_token_lengths,
            llm_word_ids,
            audio_features,
            audio_feature_lengths,
            **kwargs,
        ):
        asr_indices, llm_indices = self.extract_vq(
            asr_token_ids,
            asr_token_lengths,
            asr_word_ids,
            llm_token_ids,
            llm_token_lengths,
            llm_word_ids,
            audio_features,
            audio_feature_lengths
        )

        vq_module = self.audio_tower.vq.rvq
        lm_outputs = self.spoken_lm(
            llm_indices=llm_indices, 
            llm_token_ids=llm_token_ids, 
            llm_token_lengths=llm_token_lengths, 
            llm_word_ids=llm_word_ids,
            vq_module=vq_module,
            output_loss_at_inference=True,
        )
        loss = lm_outputs['loss']
        return loss

    
    @torch.no_grad()
    def inference_completion(
        self,
        speaker_embeds,
        conditional_mode,

        llm_tokenizer=None,
        asr_tokenizer=None,
        extra_words=32,
        text_top_p=0.0,
        taste_top_p=0.0,
        text_temperature=1.0,
        repetition_penalty=1.0,
        out_generated_part_only=False,

        asr_token_ids=None,
        asr_token_lengths=None,
        asr_word_ids=None,
        llm_token_ids=None,
        llm_token_lengths=None,
        llm_word_ids=None,
        audio_features=None,
        audio_feature_lengths=None,

        output_text_only=False,
        debug=False,

        **kwargs,
    ):
        assert conditional_mode in ('zero', 'text', 'audio', 'instruct')

        if conditional_mode in ('audio', 'instruct'):
            _, llm_indices = self.extract_vq(
                asr_token_ids,
                asr_token_lengths,
                asr_word_ids,
                llm_token_ids,
                llm_token_lengths,
                llm_word_ids,
                audio_features,
                audio_feature_lengths
            )
        else:
            llm_indices = None

        self.spoken_lm.register_taste_sampler(
            llm_tokenizer=llm_tokenizer,
            text_top_p=text_top_p,
            taste_top_p=taste_top_p,
            text_temperature=text_temperature,
            repetition_penalty=repetition_penalty,
        )

        vq_module = self.audio_tower.vq.rvq
        kwargs_for_spoken_lm_generate = dict(
            llm_indices=llm_indices, 
            llm_token_ids=llm_token_ids, 
            llm_token_lengths=llm_token_lengths, 
            llm_word_ids=llm_word_ids,
            extra_words=extra_words
        )
        if conditional_mode == 'instruct':
            kwargs_for_spoken_lm_generate.update(dict(
                instruct_prefix_ids=kwargs.get('instruct_prefix_ids'),
                instruct_suffix_ids=kwargs.get('instruct_suffix_ids'),
                stop_id=kwargs.get('stop_id'),
            ))
        generated_llm_indices, generated_llm_token_ids, _, generated_llm_word_ids = \
            self.spoken_lm.generate(
                vq_module,
                conditional_mode,
                **kwargs_for_spoken_lm_generate
            )

        if debug:
            debug_print(generated_llm_indices, 'generated_llm_indices')
            debug_print(generated_llm_word_ids, 'generated_llm_word_ids')

        # process on generated part
        generated_text = llm_tokenizer.decode(generated_llm_token_ids[0]).strip()

        if output_text_only:
            return {
                'generated_text': generated_text
            }

        words = [' ' + w for w in re.split(r'\s', generated_text)]

        assert len(words) == generated_llm_word_ids[0, -1].item() + 1, generated_text
        generated_asr_token_ids_list = []
        generated_asr_word_ids_list = []
        for i, word in enumerate(words):
            encoded_ids = asr_tokenizer.encode(word, add_special_tokens=False)
            for asr_token_id in encoded_ids:
                generated_asr_token_ids_list.append(asr_token_id)
                generated_asr_word_ids_list.append(i)

        device = generated_llm_token_ids.device
        generated_asr_token_ids = torch.tensor([generated_asr_token_ids_list], dtype=torch.int64, device=device)
        generated_asr_token_lengths = torch.tensor([len(generated_asr_token_ids_list)], dtype=torch.int32, device=device)
        generated_asr_word_ids = torch.tensor([generated_asr_word_ids_list], dtype=torch.int32, device=device)

        # combine original parts and generated parts
        if out_generated_part_only or conditional_mode in ('zero', 'text', 'instruct'):
            llm_indices = generated_llm_indices
            asr_token_ids = generated_asr_token_ids
            asr_token_lengths = generated_asr_token_lengths
            asr_word_ids = generated_asr_word_ids
        else:
            llm_indices = torch.concat([llm_indices, generated_llm_indices], dim=1)
            asr_token_ids = torch.concat([asr_token_ids, generated_asr_token_ids], dim=1)
            asr_token_lengths = asr_token_lengths + generated_asr_token_lengths
            asr_word_ids = torch.concat([asr_word_ids, asr_word_ids[0][-1] + 1 + generated_asr_word_ids], dim=1)

        audio_unit_embeds, audio_unit_lengths = self.spoken_lm.get_audio_embeds_from_taste(
            vq_module=vq_module,
            llm_taste_preds=llm_indices,
            asr_token_lengths=asr_token_lengths,
            asr_word_ids=asr_word_ids
        )
        results = self.voice_decoder_generate(
            speaker_embeds,
            audio_unit_embeds,
            audio_unit_lengths,
            asr_token_ids,
            asr_token_lengths)
        
        results['generated_text'] = generated_text
        return results

    @torch.no_grad()
    def inference_reconstruction(
        self,
        speaker_embeds,

        asr_token_ids=None,
        asr_token_lengths=None,
        asr_word_ids=None,

        audio_unit_embeds=None,
        audio_unit_lengths=None,

        audio_features=None,
        audio_feature_lengths=None,
        asr_token_alignments=None,

        llm_token_ids=None,
        llm_token_lengths=None,
        llm_word_ids=None,
    ):
        if self._mode == 'SpeechAutoEncoder' and audio_unit_embeds is None:
            audio_encoded = self.audio_tower(
                asr_token_ids=asr_token_ids,
                asr_token_lengths=asr_token_lengths,
                audio_features=audio_features,
                audio_feature_lengths=audio_feature_lengths,
                asr_token_alignments=asr_token_alignments,
                asr_word_ids=asr_word_ids)
            audio_unit_embeds = audio_encoded['audio_unit_embeds']
            audio_unit_lengths = audio_encoded['audio_unit_lengths']

        elif self._mode == 'SpokenLLM':
            asr_indices, llm_indices = self.extract_vq(
                asr_token_ids,
                asr_token_lengths,
                asr_word_ids,
                llm_token_ids,
                llm_token_lengths,
                llm_word_ids,
                audio_features,
                audio_feature_lengths
            )

            vq_module = self.audio_tower.vq.rvq
            lm_outputs = self.spoken_lm(
                llm_indices=llm_indices, 
                llm_token_ids=llm_token_ids, 
                llm_token_lengths=llm_token_lengths, 
                llm_word_ids=llm_word_ids,
                vq_module=vq_module,
            )

            audio_unit_embeds, audio_unit_lengths = self.spoken_lm.get_audio_embeds_from_taste(
                vq_module=vq_module,
                llm_taste_logits=lm_outputs['taste_logits'], llm_taste_labels=lm_outputs['taste_labels'],
                asr_token_lengths=asr_token_lengths, asr_word_ids=asr_word_ids)

        results = self.voice_decoder_generate(
            speaker_embeds,
            audio_unit_embeds,
            audio_unit_lengths,
            asr_token_ids,
            asr_token_lengths)

        return results

    def extract_vq(
        self,
        asr_token_ids,
        asr_token_lengths,
        asr_word_ids,
        llm_token_ids,
        llm_token_lengths,
        llm_word_ids,
        audio_features,
        audio_feature_lengths
    ):
        audio_encoded = self.audio_tower(
            asr_token_ids,
            asr_token_lengths,
            audio_features,
            audio_feature_lengths,
            asr_word_ids=asr_word_ids,
        )
        asr_indices = audio_encoded['quantized_indices']
        start_map = self._get_word_start_mapping_matrix(asr_word_ids, llm_word_ids, asr_token_lengths, llm_token_lengths)
        llm_indices = torch.bmm(start_map, asr_indices.float()) - (start_map.sum(dim=-1, keepdim=True) == 0).float()
        llm_indices = llm_indices.to(asr_indices.dtype)
        return (asr_indices, llm_indices)
