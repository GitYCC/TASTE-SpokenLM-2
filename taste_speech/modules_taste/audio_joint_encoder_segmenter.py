# sometimes the encoder and the segmenter are better to be written together. 

from typing import Dict, Optional, Union, List, Tuple
import logging
import os

import numpy as np
from hyperpyyaml import load_hyperpyyaml
import librosa
import torchaudio
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers import WhisperProcessor, WhisperTokenizer
from transformers.models.whisper.generation_whisper import _median_filter

from .audio_encoder import BaseAudioEncoder
from taste_speech.modules_taste.cosyvoice.model_utils import load_whisper_whole_model


# NOTE: Currently, the model is more like a wrapper to prevent re-loading of the same checkpoint or weights for the encoder and segmenter.
# NOTE: you should use `JointEncoderSegmenterAudioTokenizer` in audio_tokenizer.py if you want to use the modules in this file. 
class BaseAudioJointEncoderSegmenter(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

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
    
    # for the encoder 
    def extract_feature(
        self,
        audio_fpaths: List[str],
    ):
        raise NotImplementedError
    
    def get_audio_encoder(self):
        return self.audio_encoder # should have constructed encoder before calling this
    
    def get_audio_segmenter(self):
        return self.audio_segmenter # should have constructed segmenter before calling this


class WhisperAudioEncoderForJoint(BaseAudioEncoder):
    def __init__(
        self, 
        model_name_or_path: str,
        s3_encoder_ckpt: str = None, # checkpoint path of s3_encoder model 
        target_hidden_layer: int = 6, # specify which layer to extract. NOTE: zero means to extract the embed feature. Set to -1 to extract all hidden
        encoder_model: nn.Module = None, # allow passing a WhisperEncoder down for usage.
        attn_implementation: str = "eager", # possible choices: [eager, sdpa, flash_attention_2]
        dtype: str = "float32",
    ):
        super().__init__()
        if encoder_model == None:
            whole_model, _torch_dtype = load_whisper_whole_model(
                model_name_or_path,
                attn_implementation = attn_implementation,
                dtype = dtype,
            )
            self.encoder = whole_model.get_encoder()
        else:
            self.encoder = encoder_model

        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        if target_hidden_layer != -1 and not return_last_hidden: # -1 means extract all hidden layers. 
            for i, layer in enumerate(self.encoder.layers):
                if i > target_hidden_layer:
                    # print(f"Delete layer {i}")
                    self.encoder.layers[i] = None
                
        # check load s3_encoder_ckpt or no
        if s3_encoder_ckpt != None:
            s3_encoder_dict = get_s3_encoder_dict(self.state_dict(), s3_encoder_ckpt)
            # print(s3_encoder_dict)
            self.load_state_dict(s3_encoder_dict)
        
        self.extractor_hop_length = self.processor.feature_extractor.hop_length  # This is basically 160 for whisper extractor
        self.extractor_max_frames = self.processor.feature_extractor.nb_max_frames  # This is basically 30 * 16000 // 160  
        self.expected_seq_length = self.encoder.max_source_positions * self.encoder.conv1.stride[0] * self.encoder.conv2.stride[0]
        # print(f"WhisperAudioEncoder | expected sequence lengths: {self.expected_seq_length}")
        self.target_hidden_layer = target_hidden_layer
        # print(f"WhisperAudioEncoder | target layer: {self.target_hidden_layer}")
    
    def extract_feature(
        self,
        audio_fpaths: List[str],
        permute: bool = True,
        # pad_to_whisper_input_size: Optional[bool] = None, NOTE: deprecated. This behavior does not align with the original whisper designation
        **cfg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_feat_list, audio_feat_len_list = [], []
        waveforms = []
        for audio_fpath in audio_fpaths:
            waveform, sr = librosa.load(audio_fpath, sr=16_000, mono=True) # target sr is 16_000 for whisper
            assert sr==16_000, "Something went wrong"
            waveforms.append(waveform)
            audio_feat_len_list.append(len(waveform) // self.extractor_hop_length)

        inputs = self.processor(waveforms, sampling_rate=16000, return_tensors='pt', max_length=None) # will automatically pad to 
        audio_feat = inputs["input_features"]
        if permute:
            audio_feat = audio_feat.transpose(-1, -2) # (B, C, T) -> (B, T, C)
        audio_feat_len = torch.tensor(audio_feat_len_list, dtype=torch.int32) # This is for reference only. Whisper encoder accepts same input sizes (=3000) only. 
        return audio_feat, audio_feat_len

    def pad_to_whisper_input_size(self, audio_feat: List[torch.Tensor], padding_value=0.0):
        b, t = len(audio_feat), self.expected_seq_length
        c = audio_feat[0].shape[-1] # each tensor is with shape (T, C)
        padded_tensors = torch.full(
            (b, t, c),
            padding_value,
        )
        for i, tensor in enumerate(audio_feat):
            length = tensor.size(0)
            padded_tensors[i, :length] = tensor
        
        return padded_tensors

    @torch.cuda.amp.autocast()
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # print(audio_features.shape)
        input_features = audio_features.transpose(1, 2) # (B, T, C) -> (B, C, T)

        if input_features.shape[-1] != self.expected_seq_length:
            if input_features.shape[-1] < self.expected_seq_length:
                # already padded but should be extended to fit whisper's input seq length
                p1d = (0, self.expected_seq_length - input_features.shape[-1])
                input_features = F.pad(input_features, p1d, 'constant', 0.0)
            else:
                raise ValueError(
                    f"Whisper expects the mel input features to be of length {self.expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {self.expected_seq_length}."
                )

        inputs_embeds = nn.functional.gelu(self.encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.encoder.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1) # (B, T, C)
        embed_pos = self.encoder.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.encoder.dropout, training=self.encoder.training)

        encoder_states = {} if output_hidden_states else None

        for idx, encoder_layer in enumerate(self.encoder.layers):
            if idx == self.target_hidden_layer:
                results = {
                    'encoded_feats': hidden_states,
                    'encoded_feats_lengths': audio_features_lengths // 2, # whisper encoder will down-sample by 2 
                }
                return results
            elif idx == output_hidden_states:
                encoder_states[f'{idx}'] = hidden_states
            # if self.target_hidden_layer < 0:
            #     encoder_states = encoder_states + (hidden_states,) # forbidden returning all hidden states
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # else:
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

        if self.target_hidden_layer < 0:
            hidden_states = self.encoder.layer_norm(hidden_states)
            encoder_states['last_hidden'] = hidden_states
            results = {
                'encoded_feats': encoder_states,
                'encoded_feats_lengths': audio_features_lengths // 2, 
            } # return all encoder states
        else: 
            # return last hidden
            results = {
                'encoded_feats': hidden_states,
                'encoded_feats_lengths': audio_features_lengths // 2, 
            }
        return results


class WhisperCrossAttentionSegmenterForJoint(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str, # NOTE: We need this for getting the tokenizer.
        decoder_model: nn.Module = None,
        attn_implementation: str = "eager",
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__()
        if decoder_model == None:
            whole_model, _torch_dtype = load_whisper_whole_model(
                model_name_or_path,
                attn_implementation=attn_implementation,
                dtype=dtype,
            )
            self.decoder = whole_model.get_decoder()
        else:
            self.decoder = decoder_model

        self.model_name_or_path = model_name_or_path
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_name_or_path,
        )
    
    def forward(
        self,
        encoded_results: Dict[str, Optional[torch.Tensor]],
        text_token_for_audio: Optional[torch.Tensor],
        text_token_embed_for_audio: Optional[torch.Tensor], 
        text_token_len: Optional[torch.Tensor],
        *args,
        whisper_text_token: Optional[torch.Tensor] = None, # whisper text tokens with decoder prefix
        whisper_text_token_len: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        args:
            audio_features: (B, T, C)
            audio_features_lengths: (B,)
            text_token_embed_for_audio: (B, T, C)
            text_token_len: (B,)
        kwargs:
            words_index: None or {
                'words_begin_index': words_begin_index,
                'words_end_index': words_end_index,
                'words_index_len': words_index_len,
            }
            (refers to llm.audio_llm.py)
        """
        pass


# One may want to use both WhisperEncoder and WhisperDecoder, as AudioEncoder and AudioSegmenter, respectively.
class WhisperAudioJointEncoderSegmenter(BaseAudioJointEncoderSegmenter):
    def __init__(
        self,
        model_name_or_path: str = "",
        target_hidden_layer: int = 6,
        attn_implementation: str = "eager",
        dtype: str = 'float32',
        forward_type = "add_and_norm", # Currently support: ['original', add, add_and_norm, asr_attn_pooling]
        make_v_proj_identity: bool = False, 
        is_word_level: bool = False,
        skip_prefix_idx: Optional[int] = None,
        new_vocab: Dict = None,
        **kwargs,
    ): 
        super().__init__()
        whole_model, torch_dtype = load_whisper_whole_model(
            model_name_or_path,
            attn_implementation = attn_implementation,
            dtype = dtype,
            use_custom = True,
        )
        self.attn_implementation = attn_implementation
        self.config = whole_model.config
        self.torch_dtype = torch_dtype
        encoder = whole_model.get_encoder()
        self.audio_encoder = WhisperAudioEncoderForJoint(
            model_name_or_path, # for loading the preprocessor
            target_hidden_layer = -1, # extract all including target hidden and the last (for the decoder)
            encoder_model = encoder
        )
        decoder = whole_model.get_decoder()
        self.audio_segmenter = WhisperCrossAttentionSegmenterForJoint(
            model_name_or_path,
            decoder_model = decoder,
        )
        self.target_hidden_layer = target_hidden_layer
        # custom module
        # Attempt 1: Add & Norm
        self.forward_type = forward_type
        if self.forward_type == "add_and_norm":
            self.encoder_early_exit_layer_norm = nn.LayerNorm(self.config.d_model) 
        if make_v_proj_identity:
            self._initialize_identity(self.audio_segmenter.decoder.layers[0].encoder_attn.v_proj)
            self._initialize_identity(self.audio_segmenter.decoder.layers[1].encoder_attn.v_proj)
            # print("Initialized cross_attn's v_proj with identity matix.")
        self.is_word_level = is_word_level
        if self.is_word_level:
            # print("Would adopt word-level averaging")
            assert skip_prefix_idx != None, f"To adopt word level averaging, please set `skip_prefix_idx` properly and with cautious."
        self.skip_prefix_idx = skip_prefix_idx

        # Other tokenizer input
        if new_vocab is not None:
            if new_vocab.get("vocab_size", None) and new_vocab.get("padding_idx", None):
                self.audio_segmenter.decoder.embed_tokens = torch.nn.Embedding(new_vocab["vocab_size"], self.audio_segmenter.decoder.embed_tokens.weight.shape[1], padding_idx=new_vocab["padding_idx"])
            elif new_vocab.get("vocab_size", None):
                self.audio_segmenter.decoder.embed_tokens = torch.nn.Embedding(new_vocab["vocab_size"], self.audio_segmenter.decoder.embed_tokens.weight.shape[1])
            else:
                raise ValueError("Plese provide vocab_size in new_vocab")
    
    def _initialize_identity(self, target_linear_layer):
        # initialize the target_linear_layer as identity matrix
        with torch.no_grad():
            target_linear_layer.weight.copy_(torch.eye(target_linear_layer.in_features))
            target_linear_layer.bias.fill_(0.0)

    @torch.cuda.amp.autocast()
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        text_token_for_audio: Optional[torch.Tensor],
        text_token_embed_for_audio: Optional[torch.Tensor], 
        text_token_len: Optional[torch.Tensor],
        *args,
        whisper_text_token: Optional[torch.Tensor] = None, # whisper text tokens with decoder prefix
        whisper_text_token_len: Optional[torch.Tensor] = None,
        words_index: Optional[List[Tuple[int, int, int]]] = None,
        word_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        encoded_results = self.audio_encoder(
            audio_features,
            audio_features_lengths,
            output_hidden_states=self.target_hidden_layer,
        )
        encoded_feats, encoded_feats_len = encoded_results['encoded_feats'], encoded_results['encoded_feats_lengths']
        last_encoder_hidden, target_encoder_hidden = encoded_feats['last_hidden'], encoded_feats[f'{self.target_hidden_layer}']
        # Attempt 1: direct Add & (Norm?)
        if self.forward_type == "original":
            # for debug
            encoded_feats = last_encoder_hidden
            # print(f"Encoder last hidden: {last_encoder_hidden[0][:5,:5]}")
            # print(f"Encoder target hidden: {target_encoder_hidden[0][:5,:5]}")
            _encoder_results_for_test = self.audio_encoder.encoder(
                input_features = audio_features.transpose(1, 2),
                output_hidden_states = True,
                return_dict = True,
            )
            _encoder_last_hidden_for_test = _encoder_results_for_test.last_hidden_state
            _encoder_target_hidden_for_test = _encoder_results_for_test.hidden_states[self.target_hidden_layer]
            # print(f"Encoder last hidden (orig): {_encoder_last_hidden_for_test[0][:5,:5]}")
            # print(f"Encoder target hidden (orig): {_encoder_target_hidden_for_test[0][:5,:5]}")
        elif "add" in self.forward_type:
            encoded_feats = last_encoder_hidden + target_encoder_hidden
            if self.forward_type == "add_and_norm":
                encoded_feats = self.encoder_early_exit_layer_norm(encoded_feats) #(B, T, C)
        elif self.forward_type == "asr_attn_pooling":
            encoded_feats = {
                "states_for_key": last_encoder_hidden,
                "states_for_val": target_encoder_hidden,
            }
        ## decoder forward
        output_attentions = (self.attn_implementation == "eager")
        decoder_outputs = self.audio_segmenter.decoder(
            input_ids = whisper_text_token,
            encoder_hidden_states = encoded_feats,
            output_attentions = output_attentions,
        )

        decoder_last_hidden_state = decoder_outputs.last_hidden_state
        decoder_last_hidden_state_len = whisper_text_token_len

        if self.skip_prefix_idx != None:
            decoder_last_hidden_state = decoder_last_hidden_state[:, self.skip_prefix_idx:, :] # skip the offset of prefix
            # decoder_last_hidden_state_len -= self.skip_prefix_idx # reduce output len based on skip_prefix_idx
            decoder_last_hidden_state_len = decoder_last_hidden_state_len - self.skip_prefix_idx # reduce output len based on skip_prefix_idx. avoid in-place for safety

        if self.is_word_level:
            if words_index == None:
                assert word_ids != None, "joint encoder segmenter is word-level, please pass `words_index` or `word_ids` properly!"
                words_index = self._convert_word_ids_to_words_index(word_ids, decoder_last_hidden_state_len)
            decoder_last_hidden_state = self._averaging_subword_to_word_level(decoder_last_hidden_state, words_index)

            # b, t1, t2 = words_index[-1]
            # _decoder_hidden_slices = decoder_last_hidden_state[b, t1:t2, :]
            # print(_decoder_hidden_slices)
            # print(_decoder_hidden_slices.shape)

        segmented_results = {
            'segmented_feats': decoder_last_hidden_state,
            'segmented_feat_lengths': decoder_last_hidden_state_len,
            'decoder_outputs': decoder_outputs,
        }
        encoded_results['encoded_feats'] = encoded_feats

        return encoded_results, segmented_results
    
    def _averaging_subword_to_word_level(
        self,
        features: Optional[torch.Tensor] = None, # in the shape of (B, T, C)
        words_index: Optional[List[Tuple[int, int, int]]] = None,
        word_ids: Optional[torch.Tensor] = None,
    ):
        bsz, tsz, csz = features.shape
        # iterate through segments with more than one subword 
        averaged_features = features.clone()
        for (b, t1, t2) in words_index:
            if b >= bsz or t1 < 0 or t2 > tsz or t1 >= t2:
                raise ValueError(f"Invalid segment indices {(b, t1, t2)}, {(bsz, tsz, csz)}")
            # extract segment and compute the mean along the time dimension
            segment_mean = features[b, t1:t2, :].mean(dim=0, keepdim=True) # Shape: (1, C)
            # assign the mean value back to features
            averaged_features[b, t1:t2, :] = segment_mean # NOTE: Avoid in-place operation for back-propogation.
        
        return averaged_features

    def _convert_word_ids_to_words_index(
        self,
        word_ids,
        token_lengths,
    ):  
        words_index = []
        for b, (word_id, token_len) in enumerate(zip(word_ids, token_lengths)):
            _, counts = word_id.unique_consecutive(return_counts=True)
            _valid_pooling_lens_mask = counts > 1
            _accumulate_counts = counts.cumsum(-1)
            _valid_lens_mask = _accumulate_counts <= token_len
            mask = _valid_pooling_lens_mask.logical_and_(_valid_lens_mask)
            valid_end_idx = _accumulate_counts[mask]
            _valid_start_idx = _accumulate_counts[:-1][mask[1:]]
            valid_start_idx = torch.zeros_like(valid_end_idx)
            if len(valid_start_idx) > len(_valid_start_idx):
                valid_start_idx[1:] = _valid_start_idx
            else:
                valid_start_idx = _valid_start_idx
            for s, e in zip(valid_start_idx, valid_end_idx):
                words_index.append((b, s.item(), e.item()))
        return words_index

    def _get_alignment_map(
        self,
        generate_outputs,
        alignment_heads,
        time_precision=0.02,
        num_frames=None,
        use_orig=False,
        median_filter_width=7,
    ):  
        if use_orig:
            cross_attentions = []
            for i in range(self.config.decoder_layers):
                cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))
        else:
            # print(generate_outputs.cross_attentions, generate_outputs.cross_attentions[0].shape, len(generate_outputs.cross_attentions))
            cross_attentions = generate_outputs.cross_attentions
        # Select specific cross-attention layers and heads. This is a tensor
        # of shape (batch size, num selected, output length, input length).
        weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
        # print(weights.shape)
        weights = weights.permute([1, 0, 2, 3])

        # print(weights)
        # print(weights.shape)
        # normalize and smoothen the weights
        std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
        mean = torch.mean(weights, dim=-2, keepdim=True)
        weights = (weights - mean) / std
        weights = _median_filter(weights, median_filter_width)

        weights = weights.mean(dim=1)

        matrixes = []
        batch_size = weights.size(0)
        for batch_idx in range(batch_size):
            matrix = weights[batch_idx]
            matrix = matrix.cpu().double().numpy()
            matrixes.append(matrix)
        return matrixes


def draw_attn_map(attn_map, output_fpath, token_ids):
    import matplotlib.pyplot as plt
    x_size, y_size = attn_map.shape
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_map.T, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    plt.xlabel('whisper subwords')
    plt.xticks(ticks=np.arange(x_size), labels=token_ids)

    plt.ylabel('Encoder Last Hidden')
    plt.yticks(ticks=np.arange(y_size))

    plt.colorbar(label='Attention Weight (normalized)')
    plt.savefig(f"{output_fpath}")
