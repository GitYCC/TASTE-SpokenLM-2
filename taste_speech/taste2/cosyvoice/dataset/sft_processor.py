# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random

import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pyworld as pw


AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def sft_parquet_opener(data):
    """ Open parquet files containing SFT conversation data with message structure.
        
        Expected parquet format:
        [
          {
            'idx': 'conversation_id',
            'meta': {
              'scenario': '...',
              ...
            },
            'message': [
              {
                'role': 'system',
                'text': 'SYSTEM PROMPT'
              },
              {
                'role': 'user', 
                'text': '....',
                'audio': ~byte~,
                'timestamp_range': [~milsec~, ~milsec~]
              },
              {
                'role': 'assistant',
                'text': '....',
                'audio': ~byte~, 
                'timestamp_range': [~milsec~, ~milsec~]
              }
            ]
          }
        ]

        Args:
            data(Iterable[str]): url or local file list containing 'src' field
            mode: training mode

        Returns:
            Iterable[{idx, meta, message}]: conversation data ready for SFT processing
    """
    # print("parquet open",data)
    
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            parquet_file = pq.ParquetFile(url)
            
            # Check if this is SFT format by examining the actual schema structure
            # For nested parquet, we need to look at the top-level field names in the schema
            schema_field_names = [field.name for field in parquet_file.schema_arrow]
            
            if 'message' not in schema_field_names or 'idx' not in schema_field_names:
                logging.warning(f'Parquet file {url} does not appear to be in SFT format. '
                              f'Expected top-level fields: idx, meta, message. Found: {schema_field_names}')
                continue
                
            for batch in parquet_file.iter_batches(batch_size=64):
                df = batch.to_pandas()
                for i in range(len(df)):
                    row_data = dict(df.loc[i])
                    
                    # Extract conversation data with expected structure
                    messages = row_data.get('message', [])
                    
                    # Convert ndarray to list if needed (pandas sometimes loads as ndarray)
                    if hasattr(messages, 'tolist'):
                        messages = messages.tolist()
                    elif not isinstance(messages, list):
                        messages = list(messages) if messages is not None else []
                    
                    conversation = {
                        'idx': row_data.get('idx', f'conversation_{i}'),
                        'meta': row_data.get('meta', {}),
                        'message': messages,
                        'src': url,
                        'row_index': i
                    }
                    
                    # Validate message structure
                    if isinstance(conversation['message'], list) and len(conversation['message']) > 0:
                        # Validate message format
                        valid_conversation = True
                        for msg_idx, msg in enumerate(conversation['message']):
                            if not isinstance(msg, dict):
                                logging.warning(f'Invalid message {msg_idx} in conversation {conversation["idx"]}: not a dict')
                                valid_conversation = False
                                break
                            if 'role' not in msg:
                                logging.warning(f'Invalid message {msg_idx} in conversation {conversation["idx"]}: missing role')
                                valid_conversation = False
                                break
                            # Ensure audio field is properly handled (can be None or bytes)
                            if 'audio' in msg and msg['audio'] is not None:
                                if not isinstance(msg['audio'], bytes):
                                    logging.debug(f'Message {msg_idx} in conversation {conversation["idx"]}: audio field is not bytes (type: {type(msg["audio"])})')
                        
                        if valid_conversation:
                            yield {**sample, **conversation}
                    else:
                        logging.warning(f'Empty or invalid message list in {url}, row {i}, conversation: {conversation.get("idx", "unknown")}')
                        
        except Exception as ex:
            logging.warning('Failed to open SFT parquet {}, ex info {}'.format(url, ex))
            raise RuntimeError(f'Failed to process parquet file {url}: {ex}') from ex


def expand_sft_conversations(data, mode='train'):
    """ Expand SFT conversation format into individual training samples with multi-segment structure.
        Each conversation with multiple turns is expanded into multiple samples,
        where each sample contains the full conversation context in parallel arrays.
        
        Args:
            data: Iterable[{idx, meta, message: [{'role', 'text', 'audio'?, 'timestamp_range'?}]}]
            mode: training mode
            
        Returns:
            Iterable[{utt, text, audio_data, text_list, text_roles, audio_data_list, 
                     audio_timestamps, current_role, meta, conversation_id}]
    """
    for sample in data:
        try:
            # Skip if not SFT conversation format
            if 'message' not in sample or not isinstance(sample['message'], list):
                continue
                
            messages = sample['message']
            if len(messages) == 0:
                continue
            
            # Build conversation arrays progressively
            text_list = []
            text_roles = []
            audio_data_list = []
            audio_timestamps = []
            
            # Check if conversation has any audio at all
            has_audio = any(msg.get('audio') is not None for msg in messages)
            
            if not has_audio:
                # Skip conversations with no audio
                continue
                
            for turn_idx, message in enumerate(messages):
                # Validate message structure
                if not isinstance(message, dict) or 'role' not in message:
                    logging.warning(f'Invalid message structure in {sample.get("idx", "unknown")}, turn {turn_idx}')
                    continue
                
                # Extract message info
                role = message['role']
                text = message.get('text', '')
                audio = message.get('audio', None)
                timestamp = message.get('timestamp_range', None)
                
                # Add to conversation arrays
                text_list.append(text)
                text_roles.append(role)
                audio_data_list.append(audio)  # Can be None for text-only turns
                audio_timestamps.append(timestamp)
            
            # Create ONE training sample per conversation with ALL turns
            conversation_sample = {
                'utt': f"{sample.get('idx', 'unknown')}_conversation",
                
                # Multi-segment structure (ALL conversation turns)
                'text_list': text_list,
                'text_roles': text_roles, 
                'audio_data_list': audio_data_list,  # [None, audio1, audio2, ...]
                'audio_timestamps': audio_timestamps,
                
                # Metadata
                'meta': sample.get('meta', {}),
                'conversation_id': sample.get('idx', 'unknown'),
                'original_sample': sample
            }
            
            # Preserve speech_token if exists in original
            if 'speech_token' in sample:
                conversation_sample['speech_token'] = sample['speech_token']
                
            yield {**conversation_sample}
                    
        except Exception as ex:
            logging.warning(f'Failed to process SFT conversation {sample.get("idx", "unknown")}: {ex}')
            # continue
            raise RuntimeError(f'Failed to expand_sft: {ex}') from ex
        

def format_and_concatenate_conversation(data, get_tokenizer, mode='train'):
    """
    Format text_list and text_list_tokens with <|im_start|> role ... <|im_end|> structure
    
    and concatenate into single sequences with message dimension tracking
    
    Args:
        data: Iterable containing conversation information
        tokenizer: Tokenizer to convert special tokens to token IDs
    
    Returns:
        Iterable with updated data containing:
            - text: Single formatted text string 
            - text_token: Single token ID list
            - token_message_ids: List indicating which message each token belongs to (-1 for special tokens)
            - speech_feat: Concatenated speech features
    """
    tokenizer = get_tokenizer()
    for sample in data:
        # Special tokens
        im_start_token = "<|im_start|>"
        im_end_token = "<|im_end|>"
        newline_token = "\n"
        
        # Get token IDs for special tokens
        im_start_id = tokenizer.encode(im_start_token, add_special_tokens=False)
        im_end_id = tokenizer.encode(im_end_token, add_special_tokens=False)
        newline_id = tokenizer.encode(newline_token, add_special_tokens=False)
        
        # Initialize concatenated sequences
        concatenated_text = ""
        concatenated_tokens = []
        token_message_ids = []
        concatenated_speech_feat = []
        # audio_feat_per_message = []  # Collect audio features for each message
        
        # Process each text segment and its corresponding role
        for message_idx, (text, role) in enumerate(zip(sample['text_list'], sample['text_roles'])):
            # Format text with role tags
            formatted_segment = f"{im_start_token}{role}\n{text}\n{im_end_token}"
            concatenated_text += formatted_segment
            
            # Get corresponding tokens for this text segment
            content_tokens = sample['text_list_tokens'][message_idx]
            
            # Get role token IDs
            role_ids = tokenizer.encode(role, add_special_tokens=False)
            
            # Add tokens with message dimension tracking
            # {<|im_start|> , <|im_start|> , role, /n }tokens -> message ID = -1
            # List of token groups and corresponding message IDs
            token_groups = [
                (im_start_id, -1),
                (role_ids, -1),
                (newline_id, -1),
                (content_tokens, message_idx),
                (newline_id, -1),
                (im_end_id, -1)
            ]
            # Iterate over the groups and extend tokens and message IDs
            for tokens, message_id in token_groups:
                concatenated_tokens.extend(tokens)
                token_message_ids.extend([message_id] * len(tokens))
            
            # Concatenate speech features if available
            if 'speech_feat_list' in sample and message_idx < len(sample['speech_feat_list']):
                speech_feat = sample['speech_feat_list'][message_idx]
                if speech_feat is not None:
                    concatenated_speech_feat.append(speech_feat)
            
            # # Collect audio features for each message
            # if 'audio_feature_list' in sample and message_idx < len(sample['audio_feature_list']):
            #     audio_feat = sample['audio_feature_list'][message_idx]
            #     audio_feat_per_message.append(audio_feat)  # Can be None
            # else:
            #     audio_feat_per_message.append(None)
        
        # Update the sample with corrected field names
        sample['text'] = concatenated_text  # Changed from concatenated_text
        sample['text_token'] = concatenated_tokens  # Changed from concatenated_tokens  
        sample['token_message_ids'] = token_message_ids
        
        # Concatenate speech features if any were found
        if concatenated_speech_feat:
            sample['speech_feat'] = torch.cat(concatenated_speech_feat, dim=0)  # Concatenate along time dimension
        
        # Use audio_feature_list to create [M, T, D] dimension data (no padding here)
        if 'audio_feature_list' in sample:
            # audio_feature_list is already [M] list of [T, D] tensors
            sample['audio_feature'] = sample['audio_feature_list']  # List of [T, D] tensors
            
            # Create [M] dimension lengths
            audio_feat_lengths = []
            for feat in sample['audio_feature_list']:
                if feat is not None:
                    audio_feat_lengths.append(feat.size(0))  # T dimension
                else:
                    audio_feat_lengths.append(0)  # No audio for this message
            
            sample['audio_feature_lens'] = audio_feat_lengths  # [M] lengths
        
        yield sample


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
    """ Filter sample according to feature and label length
        Inplace operation.
        Handles  SFT format (multi-segment audio_data_list)

        Args::
            data: Iterable[{key, wav, label, sample_rate}] or SFT format with audio_data_list
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            SFT format with speech_list
    """
    for sample in data:
        # SFT format: process all audio segments in conversation
        speech_list = []
        sample_rate = None
        
        for audio_data in sample['audio_data_list']:
            if audio_data is not None:
                speech, sr = torchaudio.load(BytesIO(audio_data))
                speech = speech.mean(dim=0, keepdim=True)
                speech_list.append(speech)
                if sample_rate is None:
                    sample_rate = sr
            else:
                speech_list.append(None)
        
        sample['speech_list'] = speech_list
        sample['sample_rate'] = sample_rate
        
        # # For SFT format, apply filtering based on current turn audio in speech_list
        # if current_speech is None:
        #     continue
            
        # num_frames = current_speech.size(1) / sample_rate * 100
        # if num_frames < min_length or num_frames > max_length:
        #     continue
        # if len(sample.get('text_token', [])) < token_min_length or len(sample.get('text_token', [])) > token_max_length:
        #     continue
        # if len(sample.get('speech_token', [])) == 0:
        #     continue
        # if 'reject_speech_token' in sample and len(sample['reject_speech_token']) == 0:
        #     continue
        # if num_frames != 0:
        #     text_token_len = len(sample.get('text_token', []))
        #     if text_token_len / num_frames < min_output_input_ratio or text_token_len / num_frames > max_output_input_ratio:
        #         continue                 
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
    """ Resample data.
        Inplace operation.
        Handles both pretrain format (single audio) and SFT format (multi-segment speech_list)

        Args:
            data:  SFT format with speech_list
            resample_rate: target resample rate

        Returns:
            SFT format with resampled speech_list
    """
    for sample in data:
        assert 'sample_rate' in sample
        sample_rate = sample['sample_rate']
        
        # Check sample rate threshold for filtering
        if sample_rate < min_sample_rate:
            continue
            
        # SFT format: process all speech segments in conversation only
        resampled_speech_list = []
        
        for speech in sample['speech_list']:
            if speech is not None:
                if sample_rate != resample_rate:
                    resampled_speech = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=resample_rate)(speech)
                else:
                    resampled_speech = speech
                
                # Normalize if needed
                max_val = resampled_speech.abs().max()
                if max_val > 1:
                    resampled_speech /= max_val
                
                resampled_speech_list.append(resampled_speech)
            else:
                resampled_speech_list.append(None)
        
        sample['speech_list'] = resampled_speech_list
        sample['sample_rate'] = resample_rate
                
        yield sample


# NOTE: This is the audio feature extraction process for our `audio branch`!
def extract_audio(data, audio_extractor, target_sample_rate=16_000, **kwargs):
    """ Extract audio for audio branch
        Handles  SFT format (multi-segment speech_list)
        
        Args:
            data:  SFT format with speech_list

        Returns:
            SFT format with audio_feature_list
    """
    audio_extractor.eval()
    with torch.no_grad():
        for sample in data:
            assert 'sample_rate' in sample

            # SFT format: process all speech segments in conversation only
            audio_feature_list = []
            audio_feature_len_list = []
            
            for speech in sample['speech_list']:
                if speech is not None:
                    waveform, orig_sample_rate = speech, sample['sample_rate']
                    if orig_sample_rate != target_sample_rate:
                        waveform = torchaudio.transforms.Resample(
                            orig_freq=orig_sample_rate, new_freq=target_sample_rate)(waveform).mean(0)
                    waveform_length = [waveform.shape[-1]]
                    feat, feat_len = audio_extractor(waveform.view(1,-1), waveform_length, **kwargs)
                    audio_feature_list.append(feat.squeeze(dim=0))
                    audio_feature_len_list.append(feat_len)
                else:
                    audio_feature_list.append(None)
                    audio_feature_len_list.append(None)
            
            sample['audio_feature_list'] = audio_feature_list
            sample['audio_feature_len_list'] = audio_feature_len_list
                
            yield sample


def compute_fbank(data,
                  feat_extractor,
                  token_mel_ratio=0,
                  mode='train'):
    """ Extract fbank
        Handles SFT format (multi-segment speech_list)

        Args:
            data: SFT format with speech_list

        Returns:
            SFT format with speech_feat_list
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'utt' in sample
        
        # SFT format: process all speech segments in conversation only
        speech_feat_list = []
        
        for idx, speech in enumerate(sample['speech_list']):
            if speech is not None:
                feat = feat_extractor(speech).squeeze(dim=0).transpose(0, 1)
                
                # Apply token_mel_ratio alignment only to current turn for SFT training
                if token_mel_ratio != 0:
                    # For SFT, we need speech_token from text_list_tokens current turn
                    if 'text_list_tokens' in sample and idx < len(sample['text_list_tokens']):
                        current_text_tokens = sample['text_list_tokens'][idx]
                        if current_text_tokens:  # make sure tokens exist
                            token_len = int(min(feat.shape[0] / token_mel_ratio, len(current_text_tokens)))
                            feat = feat[:token_mel_ratio * token_len]
                            # Update the current turn tokens in text_list_tokens
                            sample['text_list_tokens'][idx] = current_text_tokens[:token_len]
                
                speech_feat_list.append(feat)
            else:
                speech_feat_list.append(None)
        
        sample['speech_feat_list'] = speech_feat_list
            
        yield sample


def tokenize(data, get_tokenizer, allowed_special, random_lstrip=True, mode='train'):
    """ Decode text to chars or BPE
        Inplace operation
        Handles SFT format (multi-segment text_list)

        Args:
            data: SFT format with text_list

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    for sample in data:
        # SFT format: tokenize conversation context only
        sample['text_list_tokens'] = []
        for text_segment in sample['text_list']:
            if text_segment:  # skip empty text
                text = text_segment
                if random_lstrip and random.random() < 0.5:
                    text = text.lstrip()
                tokens = tokenizer.encode(text, allowed_special=allowed_special)
                sample['text_list_tokens'].append(tokens)
            else:
                sample['text_list_tokens'].append([])
        
        yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['speech_feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        new_sample_frames = sample['speech_feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding_sft(data, mode='train', gan=False, dpo=False):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor([x['speech_feat'].size(1) for x in sample],
                                       dtype=torch.int32)
        order = torch.argsort(speech_feat_len, descending=True)
        sft_training = True
        utts = [sample[i]['utt'] for i in order]
        text = [sample[i]['text'] for i in order]
        text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
        # Handle [M, T, D] audio features - pad to [B, M, T, D]
        audio_feature_lists = [sample[i]['audio_feature'] for i in order]  # List of lists of [T, D] tensors
        audio_feature_lens_lists = [sample[i]['audio_feature_lens'] for i in order]  # List of lists of lengths
        
        # Pad audio_feature_list audio_feature_len to [B, M, T, D] and [B, M]
        max_messages = max(len(af_list) for af_list in audio_feature_lists) if audio_feature_lists else 0
        if max_messages > 0:
            max_time = max(af.size(0) for af_list in audio_feature_lists for af in af_list if af is not None)
            feature_dim = next(af.size(1) for af_list in audio_feature_lists for af in af_list if af is not None)
            
            # Create padded tensors - use half (float16) to match conv layer parameters
            audio_feature = torch.zeros(len(audio_feature_lists), max_messages, max_time, feature_dim, dtype=torch.half)
            audio_feature_lens = torch.zeros(len(audio_feature_lists), max_messages, dtype=torch.int32)
            
            for batch_idx, (af_list, lens_list) in enumerate(zip(audio_feature_lists, audio_feature_lens_lists)):
                for msg_idx, (af, length) in enumerate(zip(af_list, lens_list)):
                    if af is not None:
                        T, D = af.size()
                        audio_feature[batch_idx, msg_idx, :T, :] = af
                        audio_feature_lens[batch_idx, msg_idx] = length
        
        token_message_ids = [torch.tensor(sample[i]['token_message_ids']) for i in order]
        token_message_ids = pad_sequence(token_message_ids, batch_first=True, padding_value=-1)
        # No more audio_message_ids since we have explicit message dimension

        batch = {
            "utts": utts,
            "text": text,
            "text_token": text_token,  # [B, L]
            "text_token_len": text_token_len,  # [B]
            'audio_feature': audio_feature,  # [B, M, T, D] 
            'audio_feature_lens': audio_feature_lens,  # [B, M]
            'token_message_ids': token_message_ids,  # [B, L]
            'sft_training': sft_training,
        }
        if dpo is True:
            reject_speech_token = [torch.tensor(sample[i]['reject_speech_token']) for i in order]
            reject_speech_token_len = torch.tensor([i.size(0) for i in reject_speech_token], dtype=torch.int32)
            reject_speech_token = pad_sequence(reject_speech_token,
                                               batch_first=True,
                                               padding_value=0)
            batch['reject_speech_token'] = reject_speech_token
            batch['reject_speech_token_len'] = reject_speech_token_len

        yield batch