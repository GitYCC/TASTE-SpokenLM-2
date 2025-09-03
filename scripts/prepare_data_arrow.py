
""" Example Usage
cpu:

python scripts/prepare_data_arrow.py --model speech_tokenizer_v2_25hz \
                               --arrow_list arrow_files.txt \
                               --device "cpu" \
                               --output_dir "./output" \
                               --batch_size 32

gpu:

torchrun --nproc_per_node=8 --nnodes=1 \
     --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    scripts/prepare_data_arrow.py --model speech_tokenizer_v2_25hz \
                            --arrow_list arrow_files.txt \
                            --device "cuda" \
                            --output_dir "./output" \
                            --batch_size 32

"""

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import datasets
import io
from io import BytesIO

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchaudio
from tqdm import tqdm

import s3tokenizer  # need to install from `pip install s3tokenizer`

# Setup logging
logger = logging.getLogger(__name__)


def extract_file_info(filename):
    """Extract file info from both emilia and librispeech patterns"""
    # Emilia pattern: emilia-dataset-train-XXXXX-of-XXXXX-taste.arrow
    emilia_pattern = r'emilia-dataset-train-(\d+)-of-\d+-taste\.arrow'
    emilia_match = re.search(emilia_pattern, filename)
    if emilia_match:
        return "emilia", emilia_match.group(1)
    
    # Librispeech patterns: 
    # Train: librispeech-train-{clean|other}-{100|360|500}-data-XXXXX-of-XXXXX.arrow
    # Test/Dev: librispeech-{test|dev}-{clean|other}-data-XXXXX-of-XXXXX.arrow
    librispeech_pattern = r'librispeech-(?:train-(?:clean|other)-\d+|(?:test|dev)-(?:clean|other))-data-(\d+)-of-\d+\.arrow'
    librispeech_match = re.search(librispeech_pattern, filename)
    if librispeech_match:
        return "librispeech", filename.replace('librispeech-', '').replace('.arrow', '')

    raise Exception




def audio_to_bytes(audio_array, audio_sampling_rate, format='mp3'):
    """
    將音訊陣列轉換成 byte 檔案
    
    Args:
        audio_array: numpy array 或 list，音訊資料
        audio_sampling_rate: int，採樣率
        format: str，輸出格式 ('wav', 'mp3', 'flac' 等)
    
    Returns:
        bytes: 音訊檔案的 byte 資料
    """
    
    # 1. 確保 audio_array 是 numpy array
    if not isinstance(audio_array, np.ndarray):
        audio_array = np.array(audio_array)
    
    # 2. 轉換成 torch tensor
    # torchaudio 期望的格式是 (channels, samples)
    if len(audio_array.shape) == 1:
        # 單聲道：從 (samples,) 轉換成 (1, samples)
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
    else:
        # 多聲道：確保是 (channels, samples) 格式
        if audio_array.shape[0] > audio_array.shape[1]:
            # 如果是 (samples, channels)，需要轉置
            audio_array = audio_array.T
        audio_tensor = torch.from_numpy(audio_array)
    
    # 3. 確保資料類型是 float32
    audio_tensor = audio_tensor.float()
    
    # 4. 處理多聲道音訊：使用平均來處理雙聲道
    if audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
    
    # # 5. 正規化音訊資料到 [-1, 1] 範圍（如果需要的話）
    # if audio_tensor.abs().max() > 1.0:
    #     audio_tensor = audio_tensor / audio_tensor.abs().max()
    
    # 5. 使用 BytesIO 創建記憶體中的檔案
    buffer = io.BytesIO()
    
    # 6. 使用 torchaudio.save 將音訊儲存到 buffer
    torchaudio.save(
        buffer, 
        audio_tensor, 
        audio_sampling_rate, 
        format=format
    )
    
    # 7. 取得 byte 資料
    buffer.seek(0)
    audio_bytes = buffer.getvalue()
    
    return audio_tensor, audio_sampling_rate, audio_bytes


class ArrowAudioDataset(Dataset):
    
    def __init__(self, arrow_file, filename):        
        # Load arrow file using datasets library
        dataset = datasets.Dataset.from_file(arrow_file)

        self.dataset = dataset
        self.filename = filename
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        
        # Get audio data from mp3 column
        # The datasets library will handle audio decoding automatically
        audio_item = self.dataset[idx]['mp3']

        audio_tensor, audio_sampling_rate, audio_bytes = audio_to_bytes(audio_item['array'], audio_item['sampling_rate'])


        if audio_sampling_rate != 16000:
            audio_tensor_for_mel = torchaudio.transforms.Resample(audio_sampling_rate, 16000)(audio_tensor)[0]
        else:
            audio_tensor_for_mel = audio_tensor
        mel = s3tokenizer.log_mel_spectrogram(audio_tensor_for_mel)

        spk_id = self.dataset[idx]['json'].get('speaker', '')
        text = self.dataset[idx]['json']['text']
        key = f'{self.filename}--{idx:09d}'

        return key, mel, text, audio_bytes, spk_id


def collate_fn(batch):
    keys = [item[0] for item in batch]
    mels = [item[1] for item in batch]
    texts = [item[2] for item in batch]
    audio_items = [item[3] for item in batch]
    spk_ids = [item[4] for item in batch]
    
    mels, mels_lens = s3tokenizer.padding(mels)
    return keys, mels, mels_lens, texts, audio_items, spk_ids


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    print('Inference on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def get_args():
    parser = argparse.ArgumentParser(description='extract speech code from arrow files')
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        choices=[
                            "speech_tokenizer_v1", "speech_tokenizer_v1_25hz",
                            "speech_tokenizer_v2_25hz"
                        ],
                        help='model version')
    parser.add_argument('--arrow_list',
                        required=True,
                        type=str,
                        help='text file containing list of arrow file paths')
    parser.add_argument('--device',
                        required=True,
                        type=str,
                        choices=["cuda", "cpu"],
                        help='device for inference')
    parser.add_argument('--output_dir',
                        required=True,
                        type=str,
                        help='dir to save result')
    parser.add_argument('--batch_size',
                        required=True,
                        type=int,
                        help='batch size (per-device) for inference')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='workers for dataloader')
    parser.add_argument('--prefetch',
                        type=int,
                        default=5,
                        help='prefetch for dataloader')
    args = parser.parse_args()
    return args


def process_single_arrow(arrow_file, model, device, args, world_size, local_rank, rank):
    """Process a single arrow file"""
    # Create output filename based on arrow file name
    arrow_basename = os.path.basename(arrow_file)

    dataset_type, file_number = extract_file_info(arrow_basename)
    output_filename = f'{dataset_type}-{file_number}'
    if world_size > 1:
        # In distributed mode, append rank to avoid conflicts
        output_filename = f"{output_filename}_rank{rank}"
    output_path = os.path.join(args.output_dir, output_filename + '.parquet')

    dataset = ArrowAudioDataset(arrow_file, output_filename)
    
    if args.device == "cuda":
        sampler = DistributedSampler(dataset,
                                   num_replicas=world_size,
                                   rank=rank)
    else:
        sampler = None
    
    dataloader = DataLoader(dataset,
                          batch_size=args.batch_size,
                          sampler=sampler,
                          shuffle=False,
                          num_workers=args.num_workers,
                          prefetch_factor=args.prefetch,
                          collate_fn=collate_fn)
    
    total_steps = len(dataset)
    
    if rank == 0:
        arrow_name = os.path.basename(arrow_file).replace('.arrow', '').replace('.parquet', '')
        progress_bar = tqdm(total=total_steps, 
                          desc=f"Processing {arrow_name}", 
                          unit="wavs")
    
    
    
    
    all_keys = []
    all_audio_items = []
    all_texts = []
    all_spk_ids = []
    all_speech_tokens = []
    
    for keys, mels, mels_lens, texts, audio_items, spk_ids in dataloader:
        codes, codes_lens = model(mels.to(device), mels_lens.to(device))
        
        # Convert codes to CPU and add to lists
        codes_cpu = codes.cpu().numpy()
        codes_lens_cpu = codes_lens.cpu().numpy()
        
        # Process each item in the batch
        for i, (key, text, audio_item, spk_id) in enumerate(zip(keys, texts, audio_items, spk_ids)):
            # Extract speech tokens for this item (trim to actual length)
            actual_len = codes_lens_cpu[i]
            speech_tokens = codes_cpu[i][:actual_len].tolist()
            
            all_keys.append(key)
            all_texts.append(text)
            all_audio_items.append(audio_item)
            all_spk_ids.append(spk_id)
            all_speech_tokens.append(speech_tokens)
        
        if rank == 0:
            progress_bar.update(len(keys))
    
    if rank == 0:
        progress_bar.close()
    
    # Save data only if we have processed some samples
    if len(all_keys) > 0:
        df = pd.DataFrame()
        df['utt'] = all_keys
        df['audio_data'] = all_audio_items
        df['text'] = all_texts
        df['spk'] = all_spk_ids
        df['speech_token'] = all_speech_tokens
        df.to_parquet(output_path, index=False)
        
        if rank == 0:
            print(f"Saved {len(all_keys)} samples to {output_path}")
    
    return output_path if len(all_keys) > 0 else None


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we're in distributed mode (torchrun sets these environment variables)
    is_distributed = os.environ.get('WORLD_SIZE') is not None
    
    if args.device == "cuda" and is_distributed:
        assert (torch.cuda.is_available())
        world_size, local_rank, rank = init_distributed()
    else:
        world_size, local_rank, rank = 1, 0, 0
    
    device = torch.device(args.device)
    model = s3tokenizer.load_model(args.model).to(device)
    
    if args.device == "cuda" and is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])
    
    # Read arrow file list
    with open(args.arrow_list, 'r', encoding='utf-8') as f:
        arrow_files = [line.strip() for line in f if line.strip()]
    
    if rank == 0:
        print(f"Found {len(arrow_files)} arrow files to process")
    
    # Process each arrow file
    processed_files = []
    for arrow_file in arrow_files:
        if not os.path.exists(arrow_file):
            if rank == 0:
                print(f"Warning: Arrow file {arrow_file} does not exist, skipping...")
            continue
            
        if rank == 0:
            print(f"Processing arrow file: {arrow_file}")
        
        output_path = process_single_arrow(arrow_file, model, device, args, 
                                         world_size, local_rank, rank)
        processed_files.append(output_path)
        
        if args.device == "cuda" and is_distributed:
            dist.barrier()  # Sync after each file
    
    if rank == 0:
        print(f"Completed processing {len(processed_files)} arrow files")
        print(f"Output files saved in: {args.output_dir}")
    
    if args.device == "cuda" and is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()