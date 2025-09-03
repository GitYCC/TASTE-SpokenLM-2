
from typing import List, Union, Optional
import re
import os
import logging

import numpy as np
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import torch
import whisper
import onnxruntime
import torch.nn.functional as F
from transformers import pipeline
from transformers import BatchFeature, ProcessorMixin
from transformers import WhisperProcessor, AutoTokenizer
from transformers.utils.hub import cached_file
from torch.nn.utils.rnn import pad_sequence

from .modules_taste.cosyvoice.whisper_frontend import WhisperFrontend
from .configuration_taste import TasteConfig
from .modules_taste.inference_audio import VoiceGenerator


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=False)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech.numpy()


def pad_seq_collate_fn(batch, device=None):
    padded = {}
    for key in batch[0].keys():
        packed_list = [
            x[key][0].clone().detach() if isinstance(x[key][0], torch.Tensor) else torch.tensor(x[key][0]) 
            for x in batch
        ]
        if 'length' in key:
            padded_tensor = torch.tensor(packed_list)
        else:
            padded_tensor = pad_sequence(packed_list, batch_first=True, padding_value=0)

        padded[key] = padded_tensor.to(device) if device is not None else padded_tensor
    return padded


LLAMA_FILES = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']


class TasteProcessor(ProcessorMixin):
    
    attributes = ["audio_processor", "audio_tokenizer", "llm_tokenizer"]
    optional_attributes = [
        "speaker_embed_onnx_session", 
        "speech_token_onnx_session",
        "asr_pipeline", 

        "asr_on", 
        "extract_speaker_embed_on",
        "extract_speech_token_on",
    ]

    audio_processor_class = "WhisperProcessor"
    audio_tokenizer_class = "AutoTokenizer"
    llm_tokenizer_class = "AutoTokenizer"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        path_llama_tokenizer_dir = kwargs.pop('path_llama_tokenizer_dir', None)
        path_cosyvoice_dir = kwargs.pop('path_cosyvoice_dir', None)

        config = TasteConfig.from_pretrained(
            pretrained_model_name_or_path, cache_dir=cache_dir, force_download=force_download,
            local_files_only=local_files_only, token=token, revision=revision, **kwargs)

        if not path_llama_tokenizer_dir:
            for filename in LLAMA_FILES:
                path = f'llama_tokenizer/{filename}'
                file = cached_file(pretrained_model_name_or_path, path)
            path_llama_tokenizer_dir = os.path.dirname(file)   # config.text_config.name_or_path

        asr_model_name_or_path = config.asr_config.name_or_path

        audio_processor = WhisperProcessor.from_pretrained(asr_model_name_or_path)
        audio_tokenizer = AutoTokenizer.from_pretrained(asr_model_name_or_path)
        llm_tokenizer = AutoTokenizer.from_pretrained(path_llama_tokenizer_dir)

        if kwargs.get('asr_on', True):
            asr_pipeline = cls._build_asr_pipeline(asr_model_name_or_path)
        else:
            asr_pipeline = None

        if kwargs.get('extract_speaker_embed_on', True):
            if path_cosyvoice_dir:
                path_speaker_embed_onnx_session = f'{path_cosyvoice_dir}/speaker_embed.onnx'
            else:
                path_speaker_embed_onnx_session = cached_file(pretrained_model_name_or_path, 'cosyvoice/speaker_embed.onnx')
            speaker_embed_onnx_session = cls._build_onnx_session(path_speaker_embed_onnx_session)
        else:
            speaker_embed_onnx_session = None

        if kwargs.get('extract_speech_token_on', True):
            if path_cosyvoice_dir:
                path_speech_token_onnx_session = f'{path_cosyvoice_dir}/speech_tokenizer_v1.onnx'
            else:
                path_speech_token_onnx_session = cached_file(pretrained_model_name_or_path, 'cosyvoice/speech_tokenizer_v1.onnx')
            speech_token_onnx_session = cls._build_onnx_session(path_speech_token_onnx_session)
        else:
            speech_token_onnx_session = None

        if not path_cosyvoice_dir:
            file = cached_file(pretrained_model_name_or_path, 'cosyvoice/voice_generator.pth')
            path_cosyvoice_dir = os.path.dirname(file)

        processor = TasteProcessor(
            audio_processor, audio_tokenizer, llm_tokenizer,
            asr_pipeline=asr_pipeline,
            speaker_embed_onnx_session=speaker_embed_onnx_session, 
            speech_token_onnx_session=speech_token_onnx_session,
            path_llama_tokenizer_dir=path_llama_tokenizer_dir,
            path_cosyvoice_dir=path_cosyvoice_dir,
            **kwargs
        )
        return processor

    def __init__(
        self,
        audio_processor=None,
        audio_tokenizer=None,
        llm_tokenizer=None,

        speaker_embed_onnx_session=None,
        speech_token_onnx_session=None,
        asr_pipeline=None,

        asr_on=True,
        extract_speaker_embed_on=True,
        extract_speech_token_on=True,

        path_llama_tokenizer_dir='',
        path_cosyvoice_dir='',
    ):
        super().__init__(audio_processor, audio_tokenizer, llm_tokenizer, 
            asr_on=asr_on,
            extract_speech_token_on=extract_speech_token_on,
            extract_speaker_embed_on=extract_speaker_embed_on,
            asr_pipeline=asr_pipeline,
            speaker_embed_onnx_session=speaker_embed_onnx_session,
            speech_token_onnx_session=speech_token_onnx_session,
        )

        self.whisper_feature_extractor = WhisperFrontend(
            whisper_model="large-v3",
            do_pad_trim=True,
            permute=True,
        )

        self._path_llama_tokenizer_dir = path_llama_tokenizer_dir
        self._path_cosyvoice_dir = path_cosyvoice_dir

    def process_text(self, words=None, text=None):
        if words is None:
            words = [' ' + w for w in re.split(r'\s', text.strip())]

        asr_token_ids = []
        asr_word_ids = []
        llm_token_ids = []
        llm_word_ids = []
        asr_tokens_per_word = []
        for i, word in enumerate(words):
            encoded_ids = self.audio_tokenizer.encode(word, add_special_tokens=False)
            for asr_token_id in encoded_ids:
                asr_token_ids.append(asr_token_id)
                asr_word_ids.append(i)
            asr_tokens_per_word.append(len(encoded_ids))

            for llm_token_id in self.llm_tokenizer.encode(word, add_special_tokens=False):
                llm_token_ids.append(llm_token_id)
                llm_word_ids.append(i)

        text_info = {
            'words': [words],
            'text': [''.join(words)]
        }

        ids_for_text = {
            'asr_token_ids': torch.tensor([asr_token_ids], dtype=torch.int64),
            'asr_token_lengths': torch.tensor([len(asr_token_ids)], dtype=torch.int32),
            'asr_word_ids': torch.tensor([asr_word_ids], dtype=torch.int32),
            'llm_token_ids': torch.tensor([llm_token_ids], dtype=torch.int64),
            'llm_token_lengths': torch.tensor([len(llm_token_ids)], dtype=torch.int32),
            'llm_word_ids': torch.tensor([llm_word_ids], dtype=torch.int32),
        }
        return text_info, ids_for_text

    def __call__(
        self,
        audio=None,
        sampling_rate=None,
        text=None,
        ref_audio_list=None,
        **kwargs,
    ) -> BatchFeature:

        data = {}

        if isinstance(audio, str):
            audio = load_wav(audio, target_sr=sampling_rate)

        if isinstance(ref_audio_list[0], str):
            ref_audio_list = [load_wav(ref_audio, target_sr=sampling_rate)
                              for ref_audio in ref_audio_list]

        assert len(audio.shape) == 1
        assert len(ref_audio_list[0].shape) == 1
        assert sampling_rate == 16000

        if self.extract_speaker_embed_on:
            speaker_embed = self._get_speaker_embed(self.speaker_embed_onnx_session, ref_audio_list)
            data.update(
                {
                    'speaker_embeds': torch.tensor([speaker_embed], dtype=torch.float32)
                }
            )
        if self.extract_speech_token_on:
            speech_token = self._get_speech_token(self.speech_token_onnx_session, audio)
            data.update(
                {
                    'speech_token_ids': torch.tensor([speech_token], dtype=torch.int64),
                    'speech_token_lengths': torch.tensor([len(speech_token)], dtype=torch.int32)
                }
            )
        
        audio_features, audio_feature_lengths = self.whisper_feature_extractor(
            torch.tensor(np.array([audio]), dtype=torch.float32), [audio.shape[0]])
        data.update({
            'audio_features': torch.tensor(audio_features, dtype=torch.float32),
            'audio_feature_lengths': torch.tensor(audio_feature_lengths, dtype=torch.int32)
        })

        words = None
        text = None
        if self.asr_on:
            result = self.asr_pipeline(
                {'raw': audio, 'sampling_rate': sampling_rate},
                return_timestamps=None,
                generate_kwargs={
                    'language': 'english',
                    'forced_decoder_ids': None,
                    'task': 'transcribe'
                },
                batch_size=1,
            )
            text = result['text']
        elif text:
            text = re.sub(r'\s', ' ', text)
        else:
            raise ValueError("`text` is needed")

        text_info, ids_for_text = self.process_text(words=words, text=text)
        data.update(ids_for_text)
        if kwargs.pop('output_text_info', False):
            data.update(text_info)

        return BatchFeature(data=data)

    def get_generator(self, device='cpu'):
        generator = VoiceGenerator()
        path_generator = f'{self._path_cosyvoice_dir}/voice_generator.pth'
        generator.load_state_dict(torch.load(path_generator, weights_only=True))
        return generator.to(device)

    @classmethod
    def _build_onnx_session(cls, onnx_path):
        option = onnxruntime.SessionOptions()
        option.inter_op_num_threads = 1
        option.intra_op_num_threads = 1
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL # TODO: Should I keep this?
        providers = ["CPUExecutionProvider"]
        onnx_session = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)
        return onnx_session
        
    def _get_speaker_embed(self, session, audio_list):
        embed_list = []
        for audio in audio_list:
            feat = kaldi.fbank(
                torch.tensor(np.array([audio])),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            embed = session.run(
                None, 
                {session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
            )[0].flatten().tolist()
            embed_list.append(embed)
        speaker_embed = F.normalize(torch.tensor(embed_list).mean(dim=0), dim=0).numpy()
        return speaker_embed

    def _get_speech_token(self, session, audio):
        if audio.shape[0] / 16000 > 30:
            logging.warning('do not support extract speech token for audio longer than 30s')
            speech_token = []
        else:
            feat = whisper.log_mel_spectrogram(np.array([audio]), n_mels=128)
            speech_token = session.run(None, 
                {
                    session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                    session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
                }
            )[0].flatten().tolist()
        return speech_token

    @classmethod
    def _build_asr_pipeline(cls, path):
        asr_pipeline = pipeline(
            'automatic-speech-recognition',
            model=path,
            torch_dtype=torch.float16,
            device=0,
            chunk_length_s=30,
        )
        return asr_pipeline
