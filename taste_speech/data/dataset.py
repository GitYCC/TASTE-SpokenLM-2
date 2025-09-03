from functools import partial
import logging
import re
from glob import glob
import os
import random

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, WhisperProcessor
import torch
import tqdm
import torchaudio

from taste_speech.modules_taste.cosyvoice.whisper_frontend import WhisperFrontend

REQUIRED_COLUMNS = [
    'speaker_embeds',
    'asr_token_ids',
    'asr_token_lengths',
    'asr_word_ids',
    'llm_token_ids',
    'llm_token_lengths',
    'llm_word_ids',
    'audio_features',
    'audio_feature_lengths',
    'speech_token_ids',
    'speech_token_lengths',
    # 'asr_token_alignments',
]



def process_one_sample(
    sample,
    target_sr=16_000,
    resampler_dict=None,
    whisper_processor=None,
    llm_tokenizer=None,
    whisper_feature_extractor=None,
    **kwargs,
):  
    audio_array = sample['mp3']['array']
    orig_sr = sample['mp3']['sampling_rate']
    text = sample['json']['text']
    _s3_token = sample['s3_token']
    spk_emb = sample['spk_emb']

    # currently only from emilia *.arrow is supported
    speech_pt = torch.tensor(audio_array, dtype=torch.float32)
    if speech_pt.dim() == 1:
        speech_pt = speech_pt.unsqueeze(0) # unsqueeze to match the dim of torchaudio resampling
    assert resampler_dict != None, "Please set resampler dict for faster resampling."
    resampler = resampler_dict.get((orig_sr, target_sr), None)
    if resampler == None:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampler_dict[(orig_sr, target_sr)] = resampler
    waveform = resampler(speech_pt).mean(0).squeeze(0).numpy()
    # extract feature
    
    audio_feats, audio_feats_len = whisper_feature_extractor(
            torch.tensor([waveform], dtype=torch.float32), [waveform.shape[0]])

    audio_feats = torch.tensor(audio_feats, dtype=torch.float32)
    audio_feats_len = torch.tensor([audio_feats_len], dtype=torch.int32)
    # TODO: revise to word-level
    whisper_tokenizer = whisper_processor.tokenizer
    text = text.strip()

    words = [' ' + w for w in re.split(r'\s', text)]
        
    # Remove the first word's space prefix
    words[0] = words[0].lstrip()

    asr_token_ids = []
    asr_word_ids = []
    llm_token_ids = []
    llm_word_ids = []
    for i, word in enumerate(words):
        encoded_ids = whisper_tokenizer.encode(word, add_special_tokens=False)
        for asr_token_id in encoded_ids:
            asr_token_ids.append(asr_token_id)
            asr_word_ids.append(i)

        for llm_token_id in llm_tokenizer.encode(word, add_special_tokens=False):
            llm_token_ids.append(llm_token_id)
            llm_word_ids.append(i)

    words_with_space = [wrd if i==0 else f" {wrd}" for i, wrd in enumerate(text.split(' '))]
    word_level_token_ids = whisper_tokenizer(words_with_space, add_special_tokens=False).input_ids

    # extract s3 token and spk emb
    
    speech_token = torch.tensor([_s3_token])
    speech_token_len = torch.tensor([len(_s3_token)], dtype=torch.int32)
    spk_embeds = F.normalize(torch.tensor([spk_emb], dtype=torch.float32), dim=1)
    new_sample = {
        "speaker_embeds": spk_embeds,
        "audio_features": audio_feats,
        "audio_feature_lengths": audio_feats_len,
        "asr_token_ids": torch.tensor([asr_token_ids], dtype=torch.int64),
        "asr_token_lengths": torch.tensor([len(asr_token_ids)], dtype=torch.int32),
        "asr_word_ids": torch.tensor([asr_word_ids], dtype=torch.int32),
        "speech_token_ids": speech_token,
        "speech_token_lengths": speech_token_len,
        'llm_token_ids': torch.tensor([llm_token_ids], dtype=torch.int64),
        'llm_token_lengths': torch.tensor([len(llm_token_ids)], dtype=torch.int32),
        'llm_word_ids': torch.tensor([llm_word_ids], dtype=torch.int32),
    }
    return new_sample


def load_from_arrows(arrow_fpath_list, whisper_processor_fpath="", llm_tokenizer_fpath="", streaming=False, num_proc=64):
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(2)
    ds_of_arrows = concatenate_datasets(
        [
            Dataset.from_file(_arrow_fpath) for _arrow_fpath in tqdm.tqdm(arrow_fpath_list, desc="concatenating...")
        ]
    )
    if streaming:
        ds_of_arrows = ds_of_arrows.to_iterable_dataset()
        # num_proc=1

    resampler_dict = {}
    # prepare_one_sample_function
    whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_fpath)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_fpath)
    whisper_feature_extractor = WhisperFrontend(
        whisper_model="large-v3",
        do_pad_trim=True,
        permute=True,
    )
    _process_one_sample = partial(process_one_sample, resampler_dict=resampler_dict, whisper_processor=whisper_processor,
                                llm_tokenizer=llm_tokenizer, whisper_feature_extractor=whisper_feature_extractor)
    kwargs = {}
    if not streaming: 
        kwargs['num_proc'] = num_proc
    else:
        if num_proc > 1:
            logging.info(f"Cannot set num_proc > 1 when using streaming dataset!")
    ds_processed_arrows = ds_of_arrows.map(
        _process_one_sample,
        **kwargs,
    ).select_columns(
        REQUIRED_COLUMNS
    )
    return ds_processed_arrows


def pad_seq_collate_fn(batch, device=None):
    padded = {}
    # print(batch)
    for key in batch[0].keys():
        if batch[0][key] == None:
            padded[key] = None
            continue
        if 'length' in key or isinstance(batch[0][key][0], torch.Tensor):
            decorator = lambda x: x
        else:
            decorator = torch.tensor
        packed_list = [decorator(x[key][0]) for x in batch]
        if 'length' in key:
            padded_tensor = torch.tensor(packed_list, dtype=torch.int32)
        else:
            padded_tensor = pad_sequence(packed_list, batch_first=True, padding_value=0)

        padded[key] = padded_tensor.to(device) if device is not None else padded_tensor
    return padded


# def prepare_dataset(data_list_fpath, whisper_processor_fpath="", llm_tokenizer_fpath=""):
#     arrow_file_fpaths = []
#     with open(data_list_fpath, 'r') as fr:
#         for l in fr:
#             _arrow_fpath = l.split(' ')[0]
#             arrow_file_fpaths.append(_arrow_fpath)
    
#     ds_of_arrows = concatenate_datasets(
#         [
#             Dataset.from_file(_arrow_fpath) for _arrow_fpath in tqdm.tqdm(arrow_file_fpaths, desc="concatenating...")
#         ]
#     )
#     whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_fpath)
#     llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_fpath)
#     whisper_feature_extractor = WhisperFrontend(
#         whisper_model="large-v3",
#         do_pad_trim=True,
#         permute=True,
#     )
#     def _transform(batch):
#         sample = {k: v[0] for k, v in batch.items()}
#         return process_one_sample(sample,
#             resampler_dict=dict(), whisper_processor=whisper_processor,
#             llm_tokenizer=llm_tokenizer, whisper_feature_extractor=whisper_feature_extractor
#         )

#     ds_of_arrows.with_transform(_transform)
#     return ds_of_arrows


class TasteStage1Dataset(Dataset):
    def __init__(self, data_list_dir, whisper_processor_fpath, llm_tokenizer_fpath, selected_cols=None, limit_data=None):
        arrow_file_fpaths = [os.path.abspath(_arrow_fpath) for _arrow_fpath in glob(f'{data_list_dir}/*arrow')]

        emilia_datasets = []
        librispeech_datasets = []
        for _arrow_fpath in tqdm.tqdm(arrow_file_fpaths, desc="concatenating..."):
            filename = _arrow_fpath.split('/')[-1]
            if filename.startswith('cache'):
                continue
            ds = Dataset.from_file(_arrow_fpath)      
            
            if filename.startswith('emilia'):
                emilia_datasets.append(ds)
            elif filename.startswith('librispeech'):
                librispeech_datasets.append(ds)

        self.emilia_ds_of_arrows = None
        self.librispeech_ds_of_arrows = None
        random_indexes = []
        if emilia_datasets:
            self.emilia_ds_of_arrows = concatenate_datasets(emilia_datasets)
            random_indexes += [('e', i) for i in range(len(self.emilia_ds_of_arrows))]
        if librispeech_datasets:
            self.librispeech_ds_of_arrows = concatenate_datasets(librispeech_datasets)
            random_indexes += [('l', i) for i in range(len(self.librispeech_ds_of_arrows))]

        random.seed(42)
        random.shuffle(random_indexes)
        if limit_data is not None:
            random_indexes = random_indexes[:limit_data]
        self.random_indexes = random_indexes

        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_processor_fpath)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_fpath)
        self.whisper_feature_extractor = WhisperFrontend(
            whisper_model="large-v3",
            do_pad_trim=True,
            permute=True,
        )
        self.selected_cols = selected_cols

    def __getitem__(self, index):
        if isinstance(index, int):
            index = [index]
        if self.selected_cols is None:
            outputs = {k: [] for k in REQUIRED_COLUMNS}
        else:
            outputs = {k: [] for k in self.selected_cols}
        for idx in index:
            which, idx = self.random_indexes[idx]
            if which == 'e':
                sample = self.emilia_ds_of_arrows[idx]
            elif which == 'l':
                sample = self.librispeech_ds_of_arrows[idx]

            output = process_one_sample(
                    sample,
                    resampler_dict=dict(), whisper_processor=self.whisper_processor,
                    llm_tokenizer=self.llm_tokenizer, whisper_feature_extractor=self.whisper_feature_extractor
                )
            for k, v in output.items():
                if (self.selected_cols is None) or (k in self.selected_cols):
                    outputs[k].append(v)
        return outputs

    def __len__(self):
        return len(self.random_indexes)

    # def shuffle(self, seed=None):
    #     if seed is not None:
    #         random.seed(seed)  # Set seed for reproducibility
    #     self.ds_of_arrows.shuffle()

# def prepare_dataloader(data_list_fpath, whisper_processor_fpath="", llm_tokenizer_fpath="", streaming=False, num_workers=16):
#     ds_processed = prepare_dataset(data_list_fpath, whisper_processor_fpath, llm_tokenizer_fpath, streaming=streaming)

#     dataloader_kwargs = {
#         "batch_size": 16, 
#         "pin_memory": True, 
#         "collate_fn": pad_seq_collate_fn,
#     }
#     if not streaming:
#         # add num_worker for faster loading speed in non-streaming mode
#         dataloader_kwargs['num_workers'] = num_workers
#         dataloader_kwargs['prefetch_factor'] = 4

#     dataloader = DataLoader(
#         ds_processed, 
#         **dataloader_kwargs,
#     )
#     return dataloader




if __name__ == '__main__':
    whisper_processor_fpath = 'openai/whisper-large-v3'
    llm_tokenizer_fpath = '/media/ycc/Llama-3.2-3B/'
    dev_split_list = "/media/ycc/rtslm/CosyVoice/examples/emilia/taste/data/dev.data.list"

    dataset = TasteStage1Dataset(
        dev_split_list, 
        whisper_processor_fpath=whisper_processor_fpath, 
        llm_tokenizer_fpath=llm_tokenizer_fpath)

    print(dataset[2]['speech_token_lengths'].size())
