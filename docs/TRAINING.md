# CosyVoice 訓練指南

本文檔詳細介紹 CosyVoice 的訓練流程，包括資料準備、預處理、模型訓練和評估等各個環節。

## 目錄

- [概述](#概述)
- [資料準備](#資料準備)
- [資料前處理](#資料前處理)
- [訓練配置](#訓練配置)
- [模型訓練](#模型訓練)
- [訓練監控](#訓練監控)
- [模型評估](#模型評估)
- [調優策略](#調優策略)
- [常見問題](#常見問題)

## 概述

CosyVoice 採用三階段訓練架構：

1. **LLM 階段**: 語義建模，學習文本到語音 token 的映射
2. **Flow 階段**: 聲學建模，生成 Mel 頻譜特徵
3. **HiFi-GAN 階段**: 聲碼器訓練，生成最終音頻波形

### 訓練數據要求

**基本要求：**
- 採樣率：22.05kHz 或 16kHz
- 格式：WAV（推薦）、FLAC、MP3
- 質量：清晰無噪音，音量適中
- 長度：3-30 秒（推薦 5-15 秒）

**標註格式：**
```
audio_id|speaker_id|text|audio_path|duration
```

## 資料準備

### 資料集組織結構

```
dataset/
├── audio/
│   ├── speaker1/
│   │   ├── audio_001.wav
│   │   ├── audio_002.wav
│   │   └── ...
│   ├── speaker2/
│   │   └── ...
├── metadata/
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
└── speaker_embeddings/
    ├── speaker1.npy
    ├── speaker2.npy
    └── ...
```

### 標註文件格式

**train.txt 範例：**
```
spk1_001|speaker1|你好，歡迎使用語音合成系統。|audio/speaker1/audio_001.wav|4.2
spk1_002|speaker1|今天天氣很好。|audio/speaker1/audio_002.wav|2.8
spk2_001|speaker2|Hello, welcome to our TTS system.|audio/speaker2/audio_001.wav|3.5
```

### 資料集準備腳本

```python
#!/usr/bin/env python3
# prepare_dataset.py

import os
import json
import librosa
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class DatasetPreprocessor:
    def __init__(self, input_dir, output_dir, target_sr=22050):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        
        # 創建輸出目錄
        (self.output_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
    def validate_audio(self, audio_path):
        """驗證音頻文件有效性"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            # 基本檢查
            if duration < 1.0 or duration > 30.0:
                return False, f"Duration {duration:.2f}s out of range"
            
            if sr not in [16000, 22050, 44100, 48000]:
                return False, f"Unsupported sample rate: {sr}"
                
            # 音頻質量檢查
            if y.max() < 0.01:  # 音量太低
                return False, "Audio volume too low"
                
            if (y > 0.99).sum() > len(y) * 0.01:  # 削峰檢查
                return False, "Audio clipping detected"
            
            return True, None
            
        except Exception as e:
            return False, f"Audio loading error: {str(e)}"
    
    def normalize_text(self, text):
        """文本正規化"""
        import re
        
        # 數字轉換
        number_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        
        for digit, word in number_map.items():
            text = text.replace(digit, word)
        
        # 標點符號正規化
        text = text.replace('?', '？')
        text = text.replace('!', '！')
        text = text.replace(',', '，')
        text = text.replace('.', '。')
        
        # 去除多餘空格
        text = re.sub(r'\s+', '', text)
        
        return text.strip()
    
    def process_dataset(self, metadata_file):
        """處理數據集"""
        results = []
        errors = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Processing dataset"):
            parts = line.strip().split('|')
            if len(parts) < 4:
                errors.append(f"Invalid line format: {line}")
                continue
                
            audio_id = parts[0]
            speaker_id = parts[1] if len(parts) > 1 else "unknown"
            text = parts[2] if len(parts) > 2 else ""
            audio_path = parts[3] if len(parts) > 3 else parts[0] + ".wav"
            
            # 完整音頻路徑
            full_audio_path = self.input_dir / audio_path
            
            if not full_audio_path.exists():
                errors.append(f"Audio file not found: {full_audio_path}")
                continue
            
            # 驗證音頻
            is_valid, error_msg = self.validate_audio(full_audio_path)
            if not is_valid:
                errors.append(f"{audio_id}: {error_msg}")
                continue
            
            # 文本正規化
            normalized_text = self.normalize_text(text)
            if not normalized_text:
                errors.append(f"{audio_id}: Empty text after normalization")
                continue
            
            # 獲取音頻時長
            y, sr = librosa.load(full_audio_path, sr=None)
            duration = len(y) / sr
            
            # 重採樣並保存
            if sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            
            # 輸出路徑
            output_audio_dir = self.output_dir / "audio" / speaker_id
            output_audio_dir.mkdir(parents=True, exist_ok=True)
            output_audio_path = output_audio_dir / f"{audio_id}.wav"
            
            # 保存處理後的音頻
            librosa.output.write_wav(str(output_audio_path), y, self.target_sr)
            
            # 相對路徑
            rel_audio_path = f"audio/{speaker_id}/{audio_id}.wav"
            
            results.append({
                'audio_id': audio_id,
                'speaker_id': speaker_id,
                'text': normalized_text,
                'audio_path': rel_audio_path,
                'duration': duration
            })
        
        return results, errors
    
    def split_dataset(self, results, train_ratio=0.8, dev_ratio=0.1):
        """分割數據集"""
        import random
        
        # 按說話人分組
        speaker_groups = {}
        for item in results:
            speaker_id = item['speaker_id']
            if speaker_id not in speaker_groups:
                speaker_groups[speaker_id] = []
            speaker_groups[speaker_id].append(item)
        
        train_data, dev_data, test_data = [], [], []
        
        for speaker_id, items in speaker_groups.items():
            random.shuffle(items)
            n_total = len(items)
            n_train = int(n_total * train_ratio)
            n_dev = int(n_total * dev_ratio)
            
            train_data.extend(items[:n_train])
            dev_data.extend(items[n_train:n_train + n_dev])
            test_data.extend(items[n_train + n_dev:])
        
        return train_data, dev_data, test_data
    
    def save_metadata(self, data, filename):
        """保存標註文件"""
        metadata_path = self.output_dir / "metadata" / filename
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for item in data:
                line = f"{item['audio_id']}|{item['speaker_id']}|{item['text']}|{item['audio_path']}|{item['duration']:.2f}\n"
                f.write(line)
        
        print(f"Saved {len(data)} samples to {metadata_path}")

def main():
    preprocessor = DatasetPreprocessor(
        input_dir="raw_dataset",
        output_dir="processed_dataset",
        target_sr=22050
    )
    
    # 處理數據集
    results, errors = preprocessor.process_dataset("raw_dataset/metadata.txt")
    
    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors[:10]:  # 顯示前10個錯誤
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print(f"Successfully processed {len(results)} samples")
    
    # 分割數據集
    train_data, dev_data, test_data = preprocessor.split_dataset(results)
    
    # 保存標註文件
    preprocessor.save_metadata(train_data, "train.txt")
    preprocessor.save_metadata(dev_data, "dev.txt")
    preprocessor.save_metadata(test_data, "test.txt")
    
    # 統計信息
    total_duration = sum(item['duration'] for item in results)
    speaker_count = len(set(item['speaker_id'] for item in results))
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(results)}")
    print(f"  Total duration: {total_duration / 3600:.2f} hours")
    print(f"  Number of speakers: {speaker_count}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Dev samples: {len(dev_data)}")
    print(f"  Test samples: {len(test_data)}")

if __name__ == "__main__":
    main()
```

## 資料前處理

### 音頻前處理流水線

```python
# audio_preprocessing.py

import librosa
import numpy as np
import torch
import torchaudio
from scipy import signal

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Mel 濾波器
        self.mel_filter = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate // 2
        )
    
    def load_and_preprocess(self, audio_path):
        """加載和預處理音頻"""
        # 加載音頻
        waveform, sr = torchaudio.load(audio_path)
        
        # 重採樣
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 轉為單聲道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def extract_mel_spectrogram(self, waveform):
        """提取 Mel 頻譜"""
        mel_spec = self.mel_filter(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)  # 對數變換
        return mel_spec.squeeze(0).T  # [T, n_mels]
    
    def extract_f0(self, waveform, method='dio'):
        """提取基頻 F0"""
        audio_np = waveform.numpy().flatten()
        
        if method == 'dio':
            import pyworld as pw
            f0, _ = pw.dio(
                audio_np.astype(np.float64), 
                self.sample_rate, 
                frame_period=self.hop_length / self.sample_rate * 1000
            )
            f0 = pw.stonemask(audio_np.astype(np.float64), f0, _, self.sample_rate)
        
        elif method == 'harvest':
            import pyworld as pw
            f0, _ = pw.harvest(
                audio_np.astype(np.float64), 
                self.sample_rate,
                frame_period=self.hop_length / self.sample_rate * 1000
            )
        
        else:  # 簡化版本，使用 librosa
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_np,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            f0 = np.nan_to_num(f0)  # 將 NaN 替換為 0
        
        return torch.from_numpy(f0).float()
    
    def voice_activity_detection(self, waveform, frame_length=2048, hop_length=512):
        """語音活動檢測"""
        audio_np = waveform.numpy().flatten()
        
        # 計算短時能量
        energy = librosa.feature.rms(
            y=audio_np, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # 自適應閾值
        threshold = np.mean(energy) * 0.1
        vad = energy > threshold
        
        return torch.from_numpy(vad).bool()
    
    def normalize_audio(self, waveform, method='peak'):
        """音頻歸一化"""
        if method == 'peak':
            # 峰值歸一化
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val * 0.95
        
        elif method == 'rms':
            # RMS 歸一化
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > 0:
                waveform = waveform / rms * 0.1
        
        return waveform
    
    def trim_silence(self, waveform, threshold=0.01):
        """去除靜音"""
        audio_np = waveform.numpy().flatten()
        
        # 計算絕對值
        abs_audio = np.abs(audio_np)
        
        # 找到非靜音區域
        non_silent = abs_audio > threshold
        
        if non_silent.any():
            start_idx = np.where(non_silent)[0][0]
            end_idx = np.where(non_silent)[0][-1] + 1
            
            return waveform[:, start_idx:end_idx]
        else:
            return waveform

    def augment_audio(self, waveform, augment_config=None):
        """音頻數據增強"""
        if augment_config is None:
            return waveform
        
        audio = waveform.clone()
        
        # 添加噪音
        if augment_config.get('add_noise', False):
            noise_level = augment_config.get('noise_level', 0.005)
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
        
        # 音量調節
        if augment_config.get('volume_perturbation', False):
            volume_range = augment_config.get('volume_range', (0.8, 1.2))
            volume_factor = torch.uniform(*volume_range)
            audio = audio * volume_factor
        
        # 時間拉伸
        if augment_config.get('time_stretch', False):
            stretch_range = augment_config.get('stretch_range', (0.9, 1.1))
            stretch_factor = torch.uniform(*stretch_range)
            
            # 使用 torchaudio 進行時間拉伸
            audio_stretched = torchaudio.functional.time_stretch(
                audio.unsqueeze(0), 
                stretch_factor, 
                n_freq=self.n_fft // 2 + 1
            )
            audio = audio_stretched.squeeze(0)
        
        return audio

# 批次處理腳本
def batch_preprocess_audio(input_dir, output_dir, metadata_file):
    """批次處理音頻文件"""
    preprocessor = AudioPreprocessor()
    
    with open(metadata_file, 'r') as f:
        lines = f.readlines()
    
    results = []
    
    for line in tqdm(lines):
        parts = line.strip().split('|')
        audio_id = parts[0]
        audio_path = os.path.join(input_dir, parts[3])
        
        try:
            # 加載和預處理音頻
            waveform = preprocessor.load_and_preprocess(audio_path)
            
            # 歸一化和去靜音
            waveform = preprocessor.normalize_audio(waveform)
            waveform = preprocessor.trim_silence(waveform)
            
            # 提取特徵
            mel_spec = preprocessor.extract_mel_spectrogram(waveform)
            f0 = preprocessor.extract_f0(waveform)
            
            # 保存特徵
            feature_dir = os.path.join(output_dir, "features")
            os.makedirs(feature_dir, exist_ok=True)
            
            np.save(
                os.path.join(feature_dir, f"{audio_id}_mel.npy"),
                mel_spec.numpy()
            )
            np.save(
                os.path.join(feature_dir, f"{audio_id}_f0.npy"),
                f0.numpy()
            )
            
            results.append({
                'audio_id': audio_id,
                'mel_path': f"features/{audio_id}_mel.npy",
                'f0_path': f"features/{audio_id}_f0.npy",
                'duration': mel_spec.shape[0] * preprocessor.hop_length / preprocessor.sample_rate
            })
            
        except Exception as e:
            print(f"Error processing {audio_id}: {str(e)}")
    
    return results
```

### 文本前處理

```python
# text_preprocessing.py

import re
import jieba
from pypinyin import lazy_pinyin, Style
from g2p_en import G2p

class TextPreprocessor:
    def __init__(self, language='zh'):
        self.language = language
        
        if language == 'en':
            self.g2p = G2p()
        elif language == 'zh':
            # 載入自定義詞典
            jieba.load_userdict('custom_dict.txt')  # 如果有的話
    
    def normalize_chinese_text(self, text):
        """中文文本正規化"""
        # 數字轉換
        digit_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        
        for digit, word in digit_map.items():
            text = text.replace(digit, word)
        
        # 處理常見縮寫和符號
        text = text.replace('%', '百分之')
        text = text.replace('℃', '攝氏度')
        text = text.replace('°', '度')
        text = text.replace('&', '和')
        
        # 標點符號正規化
        punctuation_map = {
            '.': '。', ',': '，', '?': '？', '!': '！',
            ':': '：', ';': '；', '"': '"', "'": '''
        }
        
        for eng, chn in punctuation_map.items():
            text = text.replace(eng, chn)
        
        # 去除多餘空格
        text = re.sub(r'\s+', '', text)
        
        return text.strip()
    
    def normalize_english_text(self, text):
        """英文文本正規化"""
        # 縮寫展開
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # 數字處理
        text = re.sub(r'\b(\d+)\b', lambda m: self.number_to_words(int(m.group(1))), text)
        
        # 標準化標點
        text = re.sub(r'\s+', ' ', text)  # 多個空格變一個
        text = text.strip()
        
        return text
    
    def number_to_words(self, num):
        """數字轉文字 (簡化版)"""
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
                "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if num == 0:
            return "zero"
        elif num < 10:
            return ones[num]
        elif num < 20:
            return teens[num - 10]
        elif num < 100:
            return tens[num // 10] + ("" if num % 10 == 0 else " " + ones[num % 10])
        else:
            # 簡化處理，實際應該更完整
            return str(num)
    
    def text_to_phoneme(self, text):
        """文本轉音素"""
        if self.language == 'zh':
            # 中文轉拼音
            phonemes = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
            return ' '.join(phonemes)
        
        elif self.language == 'en':
            # 英文轉音素
            phonemes = self.g2p(text)
            return ' '.join(phonemes)
        
        return text
    
    def segment_text(self, text, max_length=200):
        """文本分段"""
        if self.language == 'zh':
            # 中文按標點符號分段
            sentences = re.split(r'[。！？；]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        elif self.language == 'en':
            # 英文按句號分段
            sentences = re.split(r'[.!?;]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # 進一步分段如果太長
        final_segments = []
        for sentence in sentences:
            if len(sentence) <= max_length:
                final_segments.append(sentence)
            else:
                # 按逗號進一步分割
                sub_segments = re.split(r'[，,]', sentence)
                current_segment = ""
                
                for sub in sub_segments:
                    if len(current_segment + sub) <= max_length:
                        current_segment += sub + ("，" if self.language == 'zh' else ", ")
                    else:
                        if current_segment:
                            final_segments.append(current_segment.strip("，, "))
                        current_segment = sub + ("，" if self.language == 'zh' else ", ")
                
                if current_segment:
                    final_segments.append(current_segment.strip("，, "))
        
        return final_segments
    
    def extract_linguistic_features(self, text):
        """提取語言學特徵"""
        features = {}
        
        if self.language == 'zh':
            # 中文特徵
            features['word_count'] = len(jieba.lcut(text))
            features['char_count'] = len([c for c in text if c.isalnum()])
            features['punctuation_count'] = len([c for c in text if not c.isalnum() and not c.isspace()])
            
            # 詞性標註
            import jieba.posseg as pseg
            pos_tags = [word.flag for word in pseg.cut(text)]
            features['pos_distribution'] = {tag: pos_tags.count(tag) for tag in set(pos_tags)}
        
        elif self.language == 'en':
            # 英文特徵
            words = text.split()
            features['word_count'] = len(words)
            features['char_count'] = len([c for c in text if c.isalnum()])
            features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
            
            # 簡單的語調標記
            features['question_mark'] = text.count('?')
            features['exclamation_mark'] = text.count('!')
        
        return features

# 批次文本處理
def batch_process_text(metadata_file, output_file, language='zh'):
    """批次處理文本"""
    preprocessor = TextPreprocessor(language=language)
    
    results = []
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Processing text"):
        parts = line.strip().split('|')
        audio_id = parts[0]
        original_text = parts[2]
        
        try:
            # 文本正規化
            if language == 'zh':
                normalized_text = preprocessor.normalize_chinese_text(original_text)
            else:
                normalized_text = preprocessor.normalize_english_text(original_text)
            
            # 轉音素
            phonemes = preprocessor.text_to_phoneme(normalized_text)
            
            # 提取語言學特徵
            features = preprocessor.extract_linguistic_features(normalized_text)
            
            results.append({
                'audio_id': audio_id,
                'original_text': original_text,
                'normalized_text': normalized_text,
                'phonemes': phonemes,
                'features': features
            })
            
        except Exception as e:
            print(f"Error processing text for {audio_id}: {str(e)}")
    
    # 保存結果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(results)} text samples")
    return results
```

### 說話人嵌入提取

```python
# speaker_embedding.py

import torch
import torchaudio
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier

class SpeakerEmbeddingExtractor:
    def __init__(self, method='resemblyzer'):
        self.method = method
        
        if method == 'resemblyzer':
            self.encoder = VoiceEncoder()
        elif method == 'speechbrain':
            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
        elif method == 'wespeaker':
            # WeSpeaker 實現
            from wespeaker import load_model
            self.encoder = load_model('english')
    
    def extract_embedding(self, audio_path_or_waveform):
        """提取說話人嵌入向量"""
        
        if isinstance(audio_path_or_waveform, str):
            # 從文件路徑讀取
            if self.method == 'resemblyzer':
                wav = preprocess_wav(audio_path_or_waveform)
                embedding = self.encoder.embed_utterance(wav)
            
            elif self.method == 'speechbrain':
                signal, sr = torchaudio.load(audio_path_or_waveform)
                embedding = self.encoder.encode_batch(signal.unsqueeze(0))
                embedding = embedding.squeeze().detach().numpy()
            
        else:
            # 從波形數據提取
            if self.method == 'resemblyzer':
                wav = audio_path_or_waveform.numpy().flatten()
                if len(wav.shape) > 1:
                    wav = wav[0]  # 取第一個通道
                embedding = self.encoder.embed_utterance(wav)
            
            elif self.method == 'speechbrain':
                signal = audio_path_or_waveform
                if len(signal.shape) == 1:
                    signal = signal.unsqueeze(0)
                embedding = self.encoder.encode_batch(signal.unsqueeze(0))
                embedding = embedding.squeeze().detach().numpy()
        
        return embedding
    
    def compute_speaker_embeddings(self, audio_files, output_dir):
        """批次計算說話人嵌入"""
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings = {}
        
        for audio_id, audio_path in tqdm(audio_files.items(), desc="Extracting embeddings"):
            try:
                embedding = self.extract_embedding(audio_path)
                embeddings[audio_id] = embedding
                
                # 保存單個嵌入
                np.save(os.path.join(output_dir, f"{audio_id}.npy"), embedding)
                
            except Exception as e:
                print(f"Error extracting embedding for {audio_id}: {str(e)}")
        
        # 保存所有嵌入
        np.savez(os.path.join(output_dir, "all_embeddings.npz"), **embeddings)
        
        return embeddings
    
    def cluster_speakers(self, embeddings, n_clusters=None, method='kmeans'):
        """說話人聚類"""
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score
        
        # 準備數據
        embedding_matrix = np.array(list(embeddings.values()))
        audio_ids = list(embeddings.keys())
        
        if method == 'kmeans':
            if n_clusters is None:
                # 自動確定聚類數量
                silhouette_scores = []
                k_range = range(2, min(20, len(embeddings) // 2))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    cluster_labels = kmeans.fit_predict(embedding_matrix)
                    score = silhouette_score(embedding_matrix, cluster_labels)
                    silhouette_scores.append(score)
                
                n_clusters = k_range[np.argmax(silhouette_scores)]
                print(f"Optimal number of clusters: {n_clusters}")
            
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(embedding_matrix)
        
        elif method == 'dbscan':
            clustering = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clustering.fit_predict(embedding_matrix)
        
        # 組織結果
        speaker_clusters = {}
        for audio_id, cluster_id in zip(audio_ids, cluster_labels):
            if cluster_id not in speaker_clusters:
                speaker_clusters[cluster_id] = []
            speaker_clusters[cluster_id].append(audio_id)
        
        return speaker_clusters
    
    def compute_speaker_stats(self, embeddings_dir, metadata_file):
        """計算說話人統計信息"""
        # 讀取元數據
        with open(metadata_file, 'r') as f:
            lines = f.readlines()
        
        speaker_info = {}
        
        for line in lines:
            parts = line.strip().split('|')
            audio_id = parts[0]
            speaker_id = parts[1]
            duration = float(parts[4]) if len(parts) > 4 else 0
            
            if speaker_id not in speaker_info:
                speaker_info[speaker_id] = {
                    'audio_files': [],
                    'total_duration': 0,
                    'embeddings': []
                }
            
            speaker_info[speaker_id]['audio_files'].append(audio_id)
            speaker_info[speaker_id]['total_duration'] += duration
            
            # 載入嵌入
            embedding_path = os.path.join(embeddings_dir, f"{audio_id}.npy")
            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
                speaker_info[speaker_id]['embeddings'].append(embedding)
        
        # 計算統計信息
        stats = {}
        for speaker_id, info in speaker_info.items():
            embeddings = np.array(info['embeddings'])
            
            stats[speaker_id] = {
                'num_utterances': len(info['audio_files']),
                'total_duration': info['total_duration'],
                'avg_duration': info['total_duration'] / len(info['audio_files']),
                'embedding_mean': np.mean(embeddings, axis=0),
                'embedding_std': np.std(embeddings, axis=0),
                'embedding_consistency': 1.0 / (1.0 + np.mean(np.std(embeddings, axis=0)))
            }
        
        return stats

# 使用範例
def extract_speaker_embeddings_batch(metadata_file, audio_dir, output_dir):
    """批次提取說話人嵌入"""
    extractor = SpeakerEmbeddingExtractor(method='resemblyzer')
    
    # 讀取元數據
    audio_files = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            audio_id = parts[0]
            audio_path = os.path.join(audio_dir, parts[3])
            audio_files[audio_id] = audio_path
    
    # 提取嵌入
    embeddings = extractor.compute_speaker_embeddings(audio_files, output_dir)
    
    # 計算統計信息
    stats = extractor.compute_speaker_stats(output_dir, metadata_file)
    
    # 保存統計信息
    with open(os.path.join(output_dir, 'speaker_stats.json'), 'w') as f:
        # 將numpy數組轉換為列表以便JSON序列化
        json_stats = {}
        for speaker_id, stat in stats.items():
            json_stats[speaker_id] = {
                'num_utterances': stat['num_utterances'],
                'total_duration': stat['total_duration'],
                'avg_duration': stat['avg_duration'],
                'embedding_consistency': stat['embedding_consistency']
            }
        json.dump(json_stats, f, indent=2)
    
    print(f"Extracted embeddings for {len(embeddings)} utterances")
    print(f"Found {len(stats)} unique speakers")
    
    return embeddings, stats
```

## 訓練配置

### 預訓練配置

**基礎配置檔案 (pretrain.yaml):**
```yaml
# 預訓練配置
stage: pretrain

# 模型配置
model_config:
  # LLM 配置
  llm:
    text_encoder_input_size: 512
    llm_input_size: 1024
    llm_output_size: 1024
    text_token_size: 51866
    speech_token_size: 4096
    num_layers: 14
    num_heads: 16
    hidden_size: 4096
    dropout: 0.1
    
  # Flow 配置  
  flow:
    input_size: 512
    output_size: 80
    input_frame_rate: 50
    num_layers: 6
    hidden_size: 512
    
  # HiFi-GAN 配置
  hifigan:
    in_channels: 80
    base_channels: 512
    nb_harmonics: 8
    upsample_rates: [8, 8]
    upsample_kernel_sizes: [16, 16]

# 訓練配置
train_config:
  # 優化器
  optimizer:
    type: AdamW
    lr: 1e-3
    betas: [0.9, 0.98]
    eps: 1e-9
    weight_decay: 0.0
  
  # 學習率調度
  scheduler:
    type: WarmupLR
    warmup_steps: 2500
    warmup_init_lr: 0.0
  
  # 訓練參數
  max_epochs: 200
  gradient_clip: 5.0
  accumulate_grad_batches: 2
  check_val_every_n_epoch: 5
  save_top_k: 5
  
  # 損失函數權重
  loss_weights:
    llm_loss: 1.0
    flow_loss: 1.0
    
# 數據配置
data_config:
  train_data: "data/train.txt"
  dev_data: "data/dev.txt"
  batch_size: 16
  num_workers: 8
  max_frames_in_batch: 2000
  shuffle: true
  
  # 數據增強
  augmentation:
    enable: true
    noise_level: 0.005
    volume_range: [0.8, 1.2]
    time_stretch_range: [0.9, 1.1]

# 硬體配置
hardware:
  gpus: [0, 1, 2, 3]
  precision: 16
  strategy: ddp
```

**SFT 微調配置 (finetune.yaml):**
```yaml
# 繼承預訓練配置
inherit_from: pretrain.yaml

# 覆蓋特定配置
stage: finetune

# 調整學習率
train_config:
  optimizer:
    lr: 1e-5  # 更小的學習率
  
  scheduler:
    type: ConstantLR  # 常數學習率
  
  max_epochs: 50  # 較少的訓練輪數

# 數據配置
data_config:
  train_data: "data/sft_train.txt"  # SFT 數據
  batch_size: 8  # 較小的批次大小
  
  # 使用說話人嵌入
  use_speaker_embedding: true
  speaker_embedding_dir: "data/speaker_embeddings"
```

### 配置管理系統

```python
# config_manager.py

import yaml
import json
import os
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig

class ConfigManager:
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.config = self.load_config(base_config_path)
    
    def load_config(self, config_path: str) -> DictConfig:
        """載入配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 處理繼承
        if 'inherit_from' in config_dict:
            parent_path = config_dict.pop('inherit_from')
            parent_config = self.load_config(parent_path)
            
            # 合併配置
            config = OmegaConf.merge(parent_config, config_dict)
        else:
            config = OmegaConf.create(config_dict)
        
        return config
    
    def get(self, key: str, default=None):
        """獲取配置值"""
        return OmegaConf.select(self.config, key, default=default)
    
    def set(self, key: str, value: Any):
        """設置配置值"""
        OmegaConf.set(self.config, key, value)
    
    def update_from_dict(self, update_dict: Dict):
        """從字典更新配置"""
        self.config = OmegaConf.merge(self.config, update_dict)
    
    def save_config(self, output_path: str):
        """保存配置到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(OmegaConf.to_yaml(self.config), f)
    
    def validate_config(self):
        """驗證配置完整性"""
        required_sections = ['model_config', 'train_config', 'data_config']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # 驗證模型配置
        model_config = self.config.model_config
        required_model_keys = ['llm', 'flow', 'hifigan']
        
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"Missing model config: {key}")
        
        # 驗證訓練配置
        train_config = self.config.train_config
        if train_config.optimizer.lr <= 0:
            raise ValueError("Learning rate must be positive")
        
        if train_config.max_epochs <= 0:
            raise ValueError("Max epochs must be positive")
        
        # 驗證數據配置
        data_config = self.config.data_config
        if not os.path.exists(data_config.train_data):
            raise ValueError(f"Training data not found: {data_config.train_data}")
    
    def setup_experiment(self, experiment_name: str, output_dir: str):
        """設置實驗環境"""
        exp_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(exp_dir, 'config.yaml')
        self.save_config(config_path)
        
        # 設置日誌目錄
        log_dir = os.path.join(exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 設置檢查點目錄
        ckpt_dir = os.path.join(exp_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 更新配置中的路徑
        self.set('experiment.name', experiment_name)
        self.set('experiment.output_dir', exp_dir)
        self.set('experiment.log_dir', log_dir)
        self.set('experiment.checkpoint_dir', ckpt_dir)
        
        return exp_dir

# 配置驗證和自動調整
class ConfigValidator:
    def __init__(self):
        self.gpu_memory_map = {
            'RTX 3090': 24 * 1024,      # 24GB
            'RTX 4090': 24 * 1024,      # 24GB  
            'V100': 32 * 1024,          # 32GB
            'A100': 80 * 1024,          # 80GB
            'RTX 3080': 10 * 1024,      # 10GB
            'RTX 3070': 8 * 1024,       # 8GB
        }
    
    def auto_adjust_batch_size(self, config: DictConfig, gpu_type: str = None):
        """根據 GPU 自動調整批次大小"""
        if gpu_type and gpu_type in self.gpu_memory_map:
            gpu_memory = self.gpu_memory_map[gpu_type]
            
            if gpu_memory >= 24 * 1024:  # 24GB+
                recommended_batch_size = 16
                recommended_max_frames = 2000
            elif gpu_memory >= 16 * 1024:  # 16GB+
                recommended_batch_size = 8
                recommended_max_frames = 1400
            elif gpu_memory >= 8 * 1024:   # 8GB+
                recommended_batch_size = 4
                recommended_max_frames = 800
            else:  # < 8GB
                recommended_batch_size = 2
                recommended_max_frames = 400
            
            config.data_config.batch_size = recommended_batch_size
            config.data_config.max_frames_in_batch = recommended_max_frames
            
            print(f"Adjusted for {gpu_type}:")
            print(f"  Batch size: {recommended_batch_size}")
            print(f"  Max frames: {recommended_max_frames}")
    
    def estimate_training_time(self, config: DictConfig, dataset_size: int):
        """估算訓練時間"""
        batch_size = config.data_config.batch_size
        num_gpus = len(config.hardware.gpus) if 'gpus' in config.hardware else 1
        max_epochs = config.train_config.max_epochs
        
        # 估算每個 epoch 的步數
        steps_per_epoch = dataset_size // (batch_size * num_gpus)
        total_steps = steps_per_epoch * max_epochs
        
        # 估算每步時間（秒）
        # 這個需要根據實際硬體和模型大小調整
        seconds_per_step = 2.0  # 假設值
        
        total_seconds = total_steps * seconds_per_step
        total_hours = total_seconds / 3600
        
        return {
            'steps_per_epoch': steps_per_epoch,
            'total_steps': total_steps,
            'estimated_hours': total_hours,
            'estimated_days': total_hours / 24
        }
```

## 模型訓練

### 訓練腳本

```python
# train.py

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import hydra
from omegaconf import DictConfig

from cosyvoice.models.cosyvoice_model import CosyVoiceModel
from cosyvoice.data.datamodule import CosyVoiceDataModule
from config_manager import ConfigManager

class CosyVoiceTrainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_logging()
        self.setup_callbacks()
    
    def setup_logging(self):
        """設置日誌記錄"""
        self.loggers = []
        
        # TensorBoard 日誌
        if self.config.get('logging.tensorboard.enable', True):
            tb_logger = TensorBoardLogger(
                save_dir=self.config.experiment.log_dir,
                name="tensorboard",
                version=self.config.experiment.name
            )
            self.loggers.append(tb_logger)
        
        # Weights & Biases 日誌
        if self.config.get('logging.wandb.enable', False):
            wandb_logger = WandbLogger(
                project=self.config.get('logging.wandb.project', 'cosyvoice'),
                name=self.config.experiment.name,
                save_dir=self.config.experiment.log_dir
            )
            self.loggers.append(wandb_logger)
    
    def setup_callbacks(self):
        """設置訓練回調"""
        self.callbacks = []
        
        # 模型檢查點
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.experiment.checkpoint_dir,
            filename='{epoch}-{step}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=self.config.train_config.get('save_top_k', 5),
            save_last=True,
            every_n_epochs=1
        )
        self.callbacks.append(checkpoint_callback)
        
        # 早停
        if self.config.train_config.get('early_stopping.enable', False):
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.config.train_config.early_stopping.patience,
                mode='min',
                min_delta=self.config.train_config.early_stopping.get('min_delta', 0.001)
            )
            self.callbacks.append(early_stop_callback)
        
        # 學習率監控
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.callbacks.append(lr_monitor)
    
    def create_model(self):
        """創建模型"""
        model = CosyVoiceModel(self.config)
        return model
    
    def create_datamodule(self):
        """創建數據模塊"""
        datamodule = CosyVoiceDataModule(self.config)
        return datamodule
    
    def train(self, resume_from_checkpoint=None):
        """開始訓練"""
        # 創建模型和數據
        model = self.create_model()
        datamodule = self.create_datamodule()
        
        # 配置訓練器
        trainer = pl.Trainer(
            max_epochs=self.config.train_config.max_epochs,
            gpus=self.config.hardware.get('gpus', 1),
            precision=self.config.hardware.get('precision', 32),
            strategy=self.config.hardware.get('strategy', 'ddp'),
            accumulate_grad_batches=self.config.train_config.get('accumulate_grad_batches', 1),
            gradient_clip_val=self.config.train_config.get('gradient_clip', 5.0),
            check_val_every_n_epoch=self.config.train_config.get('check_val_every_n_epoch', 1),
            callbacks=self.callbacks,
            logger=self.loggers,
            resume_from_checkpoint=resume_from_checkpoint,
            enable_progress_bar=True,
            log_every_n_steps=self.config.train_config.get('log_interval', 100)
        )
        
        # 開始訓練
        trainer.fit(model, datamodule=datamodule)
        
        # 返回最佳模型路徑
        best_model_path = trainer.checkpoint_callback.best_model_path
        return best_model_path

@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # 設置隨機種子
    pl.seed_everything(cfg.get('seed', 42))
    
    # 配置管理
    config_manager = ConfigManager(cfg)
    config_manager.validate_config()
    
    # 設置實驗
    exp_dir = config_manager.setup_experiment(
        experiment_name=cfg.experiment_name,
        output_dir=cfg.output_dir
    )
    
    # 創建訓練器
    trainer = CosyVoiceTrainer(config_manager.config)
    
    # 開始訓練
    print(f"Starting training: {cfg.experiment_name}")
    print(f"Output directory: {exp_dir}")
    
    best_model_path = trainer.train(
        resume_from_checkpoint=cfg.get('resume_from_checkpoint', None)
    )
    
    print(f"Training completed!")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main()
```

### 數據模塊實現

```python
# cosyvoice/data/datamodule.py

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchaudio
from omegaconf import DictConfig

class CosyVoiceDataset(Dataset):
    def __init__(self, metadata_file, config: DictConfig, stage='train'):
        self.config = config
        self.stage = stage
        self.data_samples = self.load_metadata(metadata_file)
        
        # 數據增強配置
        self.use_augmentation = (
            stage == 'train' and 
            config.data_config.get('augmentation.enable', False)
        )
        
        if self.use_augmentation:
            self.setup_augmentation()
    
    def load_metadata(self, metadata_file):
        """載入元數據"""
        samples = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    samples.append({
                        'audio_id': parts[0],
                        'speaker_id': parts[1],
                        'text': parts[2],
                        'audio_path': parts[3],
                        'duration': float(parts[4]) if len(parts) > 4 else 0.0
                    })
        
        return samples
    
    def setup_augmentation(self):
        """設置數據增強"""
        self.augmentation_config = self.config.data_config.augmentation
        
        # 噪音增強
        self.noise_level = self.augmentation_config.get('noise_level', 0.005)
        
        # 音量增強
        volume_range = self.augmentation_config.get('volume_range', [0.8, 1.2])
        self.volume_min, self.volume_max = volume_range
        
        # 時間拉伸
        stretch_range = self.augmentation_config.get('time_stretch_range', [0.9, 1.1])
        self.stretch_min, self.stretch_max = stretch_range
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # 載入音頻
        audio_path = sample['audio_path']
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 轉為單聲道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重採樣
        target_sr = self.config.model_config.get('sample_rate', 22050)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # 數據增強
        if self.use_augmentation:
            waveform = self.apply_augmentation(waveform)
        
        # 提取特徵
        features = self.extract_features(waveform, sample)
        
        # 處理文本
        text_features = self.process_text(sample['text'])
        
        return {
            'audio_id': sample['audio_id'],
            'speaker_id': sample['speaker_id'],
            'waveform': waveform,
            'text': sample['text'],
            'duration': sample['duration'],
            **features,
            **text_features
        }
    
    def apply_augmentation(self, waveform):
        """應用數據增強"""
        # 添加噪音
        if torch.rand(1) < 0.3:  # 30% 概率
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise
        
        # 音量調節
        if torch.rand(1) < 0.5:  # 50% 概率
            volume_factor = torch.uniform(self.volume_min, self.volume_max, (1,))
            waveform = waveform * volume_factor
        
        # 限制幅值
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform
    
    def extract_features(self, waveform, sample):
        """提取音頻特徵"""
        features = {}
        
        # Mel 頻譜
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.model_config.get('sample_rate', 22050),
            n_fft=self.config.model_config.get('n_fft', 1024),
            hop_length=self.config.model_config.get('hop_length', 256),
            n_mels=self.config.model_config.get('n_mels', 80),
            f_min=0,
            f_max=None
        )
        
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)
        features['mel_spec'] = mel_spec.squeeze(0).T  # [T, n_mels]
        
        # 說話人嵌入
        if self.config.data_config.get('use_speaker_embedding', False):
            speaker_embedding_dir = self.config.data_config.speaker_embedding_dir
            embedding_path = os.path.join(
                speaker_embedding_dir, 
                f"{sample['audio_id']}.npy"
            )
            
            if os.path.exists(embedding_path):
                speaker_embedding = torch.from_numpy(np.load(embedding_path)).float()
            else:
                # 使用默認嵌入或計算新的
                speaker_embedding = torch.zeros(192)  # 默認維度
            
            features['speaker_embedding'] = speaker_embedding
        
        return features
    
    def process_text(self, text):
        """處理文本"""
        # 這裡應該包含分詞、音素轉換等處理
        # 為簡化，返回基本信息
        
        return {
            'text_length': len(text),
            'processed_text': text  # 實際應該是處理後的 token 序列
        }

class CosyVoiceDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        """設置數據集"""
        if stage == 'fit' or stage is None:
            self.train_dataset = CosyVoiceDataset(
                self.config.data_config.train_data,
                self.config,
                stage='train'
            )
            
            self.val_dataset = CosyVoiceDataset(
                self.config.data_config.dev_data,
                self.config,
                stage='val'
            )
        
        if stage == 'test':
            self.test_dataset = CosyVoiceDataset(
                self.config.data_config.get('test_data', self.config.data_config.dev_data),
                self.config,
                stage='test'
            )
    
    def collate_fn(self, batch):
        """批次數據整理"""
        # 按時長排序
        batch = sorted(batch, key=lambda x: x['duration'], reverse=True)
        
        # 動態批次大小
        max_frames = self.config.data_config.get('max_frames_in_batch', 2000)
        actual_batch = []
        total_frames = 0
        
        for sample in batch:
            frames = sample['mel_spec'].shape[0]
            if total_frames + frames <= max_frames or len(actual_batch) == 0:
                actual_batch.append(sample)
                total_frames += frames
            else:
                break
        
        # 填充和對齊
        batch_size = len(actual_batch)
        max_length = max(sample['mel_spec'].shape[0] for sample in actual_batch)
        
        # 準備批次張量
        waveforms = []
        mel_specs = []
        speaker_embeddings = []
        text_lengths = []
        
        for sample in actual_batch:
            # 波形填充
            waveform = sample['waveform']
            waveforms.append(waveform)
            
            # Mel 頻譜填充
            mel_spec = sample['mel_spec']
            padded_mel = torch.zeros(max_length, mel_spec.shape[1])
            padded_mel[:mel_spec.shape[0]] = mel_spec
            mel_specs.append(padded_mel)
            
            # 說話人嵌入
            if 'speaker_embedding' in sample:
                speaker_embeddings.append(sample['speaker_embedding'])
            
            text_lengths.append(sample['text_length'])
        
        result = {
            'batch_size': batch_size,
            'waveforms': waveforms,  # 列表，長度可能不同
            'mel_specs': torch.stack(mel_specs),  # [B, T, n_mels]
            'text_lengths': torch.tensor(text_lengths),
            'audio_ids': [sample['audio_id'] for sample in actual_batch],
            'speaker_ids': [sample['speaker_id'] for sample in actual_batch],
            'texts': [sample['text'] for sample in actual_batch]
        }
        
        if speaker_embeddings:
            result['speaker_embeddings'] = torch.stack(speaker_embeddings)
        
        return result
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data_config.batch_size,
            shuffle=True,
            num_workers=self.config.data_config.get('num_workers', 4),
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data_config.batch_size,
            shuffle=False,
            num_workers=self.config.data_config.get('num_workers', 4),
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=False
        )
```

### 模型實現

```python
# cosyvoice/models/cosyvoice_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import Dict, Any, Optional

from cosyvoice.models.llm import TransformerLM
from cosyvoice.models.flow import MaskedDiffWithXvec  
from cosyvoice.models.hifigan import HiFiGan

class CosyVoiceModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # 創建子模型
        self.llm = self.create_llm()
        self.flow = self.create_flow()
        self.hifigan = self.create_hifigan()
        
        # 損失權重
        self.llm_loss_weight = config.train_config.loss_weights.get('llm_loss', 1.0)
        self.flow_loss_weight = config.train_config.loss_weights.get('flow_loss', 1.0)
        self.hifigan_loss_weight = config.train_config.loss_weights.get('hifigan_loss', 1.0)
        
        # 訓練階段
        self.training_stage = config.get('stage', 'pretrain')
    
    def create_llm(self):
        """創建 LLM 模型"""
        llm_config = self.config.model_config.llm
        
        llm = TransformerLM(
            text_encoder_input_size=llm_config.text_encoder_input_size,
            llm_input_size=llm_config.llm_input_size,
            llm_output_size=llm_config.llm_output_size,
            text_token_size=llm_config.text_token_size,
            speech_token_size=llm_config.speech_token_size,
            num_layers=llm_config.num_layers,
            num_heads=llm_config.num_heads,
            hidden_size=llm_config.hidden_size,
            dropout=llm_config.dropout
        )
        
        return llm
    
    def create_flow(self):
        """創建 Flow 模型"""
        flow_config = self.config.model_config.flow
        
        flow = MaskedDiffWithXvec(
            input_size=flow_config.input_size,
            output_size=flow_config.output_size,
            input_frame_rate=flow_config.input_frame_rate,
            num_layers=flow_config.num_layers,
            hidden_size=flow_config.hidden_size
        )
        
        return flow
    
    def create_hifigan(self):
        """創建 HiFi-GAN 模型"""
        hifigan_config = self.config.model_config.hifigan
        
        hifigan = HiFiGan(
            in_channels=hifigan_config.in_channels,
            base_channels=hifigan_config.base_channels,
            nb_harmonics=hifigan_config.nb_harmonics,
            upsample_rates=hifigan_config.upsample_rates,
            upsample_kernel_sizes=hifigan_config.upsample_kernel_sizes
        )
        
        return hifigan
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """前向傳播"""
        outputs = {}
        
        if self.training_stage in ['pretrain', 'finetune']:
            # LLM 前向傳播
            llm_outputs = self.llm(
                text=batch['texts'],
                speaker_embeddings=batch.get('speaker_embeddings')
            )
            outputs['llm'] = llm_outputs
            
            # Flow 前向傳播
            flow_outputs = self.flow(
                llm_features=llm_outputs['speech_tokens'],
                mel_targets=batch['mel_specs'],
                speaker_embeddings=batch.get('speaker_embeddings')
            )
            outputs['flow'] = flow_outputs
        
        if self.training_stage == 'hifigan':
            # HiFi-GAN 前向傳播
            hifigan_outputs = self.hifigan(
                mel_specs=batch['mel_specs']
            )
            outputs['hifigan'] = hifigan_outputs
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """計算損失"""
        losses = {}
        total_loss = 0
        
        if 'llm' in outputs:
            llm_loss = self.compute_llm_loss(outputs['llm'], batch)
            losses['llm_loss'] = llm_loss
            total_loss += self.llm_loss_weight * llm_loss
        
        if 'flow' in outputs:
            flow_loss = self.compute_flow_loss(outputs['flow'], batch)
            losses['flow_loss'] = flow_loss
            total_loss += self.flow_loss_weight * flow_loss
        
        if 'hifigan' in outputs:
            hifigan_losses = self.compute_hifigan_loss(outputs['hifigan'], batch)
            losses.update(hifigan_losses)
            
            # HiFi-GAN 總損失
            hifigan_total = sum(hifigan_losses.values())
            total_loss += self.hifigan_loss_weight * hifigan_total
        
        losses['total_loss'] = total_loss
        return losses
    
    def compute_llm_loss(self, llm_outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """計算 LLM 損失"""
        # 實現 LLM 的語言建模損失
        logits = llm_outputs['logits']
        targets = batch['speech_token_targets']  # 需要從數據中提供
        
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss
    
    def compute_flow_loss(self, flow_outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """計算 Flow 損失"""
        # 實現 Flow Matching 損失
        predicted_mel = flow_outputs['predicted_mel']
        target_mel = batch['mel_specs']
        
        # L1 損失
        l1_loss = nn.L1Loss()(predicted_mel, target_mel)
        
        # 可以添加其他損失項
        return l1_loss
    
    def compute_hifigan_loss(self, hifigan_outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """計算 HiFi-GAN 損失"""
        losses = {}
        
        # 生成器損失
        if 'generator_loss' in hifigan_outputs:
            losses['gen_loss'] = hifigan_outputs['generator_loss']
        
        # 判別器損失
        if 'discriminator_loss' in hifigan_outputs:
            losses['disc_loss'] = hifigan_outputs['discriminator_loss']
        
        # Mel 損失
        if 'mel_loss' in hifigan_outputs:
            losses['mel_loss'] = hifigan_outputs['mel_loss']
        
        # 特徵匹配損失
        if 'feature_loss' in hifigan_outputs:
            losses['feat_loss'] = hifigan_outputs['feature_loss']
        
        return losses
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """訓練步驟"""
        outputs = self(batch)
        losses = self.compute_loss(outputs, batch)
        
        # 記錄損失
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """驗證步驟"""
        outputs = self(batch)
        losses = self.compute_loss(outputs, batch)
        
        # 記錄損失
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
        
        return losses
    
    def configure_optimizers(self):
        """配置優化器和學習率調度器"""
        optimizer_config = self.config.train_config.optimizer
        
        # 創建優化器
        if optimizer_config.type == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=optimizer_config.lr,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay
            )
        elif optimizer_config.type == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=optimizer_config.lr,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_config.type}")
        
        # 創建學習率調度器
        scheduler_config = self.config.train_config.scheduler
        
        if scheduler_config.type == 'WarmupLR':
            def lr_lambda(step):
                if step < scheduler_config.warmup_steps:
                    return step / scheduler_config.warmup_steps
                return 1.0
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }
        
        elif scheduler_config.type == 'ConstantLR':
            return optimizer
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_config.type}")
```

### 訓練啟動腳本

```bash
#!/bin/bash
# run_training.sh

# 設置環境變數
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/third_party/Matcha-TTS"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4

# 訓練配置
EXPERIMENT_NAME="cosyvoice_pretrain_$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="configs/pretrain.yaml"
OUTPUT_DIR="experiments"

# 創建輸出目錄
mkdir -p $OUTPUT_DIR

# 啟動訓練
python train.py \
    experiment_name=$EXPERIMENT_NAME \
    config_path=$CONFIG_PATH \
    output_dir=$OUTPUT_DIR \
    hardware.gpus=[0,1,2,3] \
    data_config.batch_size=16 \
    train_config.max_epochs=200

echo "Training started: $EXPERIMENT_NAME"
echo "Monitor progress: tensorboard --logdir $OUTPUT_DIR/$EXPERIMENT_NAME/logs"
```

## 訓練監控

### 監控系統設置

```python
# monitoring/training_monitor.py

import os
import json
import time
import wandb
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import torch
import numpy as np
from pathlib import Path

class TrainingMonitor:
    def __init__(self, experiment_dir: str, config: Dict):
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.metrics_history = []
        self.setup_monitoring()
    
    def setup_monitoring(self):
        """設置監控系統"""
        # 創建監控目錄
        self.monitor_dir = self.experiment_dir / "monitoring"
        self.monitor_dir.mkdir(exist_ok=True)
        
        # 設置日誌文件
        self.metrics_file = self.monitor_dir / "metrics.jsonl"
        self.plots_dir = self.monitor_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # 初始化 Weights & Biases
        if self.config.get('wandb', {}).get('enable', False):
            wandb.init(
                project=self.config['wandb']['project'],
                name=self.config['experiment_name'],
                config=self.config
            )
    
    def log_metrics(self, metrics: Dict[str, float], step: int, epoch: int = None):
        """記錄訓練指標"""
        timestamp = time.time()
        
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'epoch': epoch,
            'metrics': metrics
        }
        
        # 保存到本地文件
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # 記錄到 wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
        
        self.metrics_history.append(log_entry)
        
        # 定期生成圖表
        if step % self.config.get('plot_interval', 1000) == 0:
            self.generate_plots()
    
    def log_audio_samples(self, audio_samples: Dict[str, torch.Tensor], step: int):
        """記錄音頻樣本"""
        sample_dir = self.monitor_dir / "audio_samples" / f"step_{step}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_id, audio in audio_samples.items():
            # 保存音頻文件
            audio_file = sample_dir / f"{sample_id}.wav"
            torchaudio.save(str(audio_file), audio.cpu(), 22050)
            
            # 記錄到 wandb
            if wandb.run is not None:
                wandb.log({
                    f"audio/{sample_id}": wandb.Audio(
                        audio.cpu().numpy(), 
                        sample_rate=22050,
                        caption=f"Step {step} - {sample_id}"
                    )
                }, step=step)
    
    def log_spectrograms(self, spectrograms: Dict[str, torch.Tensor], step: int):
        """記錄頻譜圖"""
        spec_dir = self.monitor_dir / "spectrograms" / f"step_{step}"
        spec_dir.mkdir(parents=True, exist_ok=True)
        
        for spec_id, spec in spectrograms.items():
            # 生成頻譜圖
            plt.figure(figsize=(12, 6))
            plt.imshow(spec.cpu().numpy().T, aspect='auto', origin='lower')
            plt.colorbar()
            plt.title(f"{spec_id} - Step {step}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            
            # 保存圖片
            spec_file = spec_dir / f"{spec_id}.png"
            plt.savefig(str(spec_file), dpi=150, bbox_inches='tight')
            plt.close()
            
            # 記錄到 wandb
            if wandb.run is not None:
                wandb.log({
                    f"spectrogram/{spec_id}": wandb.Image(str(spec_file))
                }, step=step)
    
    def generate_plots(self):
        """生成訓練曲線圖"""
        if len(self.metrics_history) < 10:
            return
        
        # 提取數據
        steps = [entry['step'] for entry in self.metrics_history]
        
        # 收集所有指標名稱
        all_metrics = set()
        for entry in self.metrics_history:
            all_metrics.update(entry['metrics'].keys())
        
        # 為每個指標生成圖表
        for metric_name in all_metrics:
            values = []
            metric_steps = []
            
            for entry in self.metrics_history:
                if metric_name in entry['metrics']:
                    values.append(entry['metrics'][metric_name])
                    metric_steps.append(entry['step'])
            
            if len(values) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(metric_steps, values, label=metric_name)
                plt.xlabel('Steps')
                plt.ylabel(metric_name)
                plt.title(f'Training Progress: {metric_name}')
                plt.legend()
                plt.grid(True)
                
                plot_file = self.plots_dir / f"{metric_name.replace('/', '_')}.png"
                plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
                plt.close()
    
    def monitor_system_resources(self):
        """監控系統資源使用"""
        import psutil
        import GPUtil
        
        # CPU 和內存
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU 使用情況
        gpu_stats = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_stats.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                })
        except:
            pass
        
        system_metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'gpu_stats': gpu_stats
        }
        
        return system_metrics
    
    def check_training_health(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """檢查訓練健康狀況"""
        health_status = {
            'status': 'healthy',
            'warnings': [],
            'errors': []
        }
        
        # 檢查損失是否異常
        if 'train_total_loss' in current_metrics:
            loss = current_metrics['train_total_loss']
            
            if loss > 100:  # 損失過大
                health_status['warnings'].append(f"Training loss is very high: {loss:.4f}")
            
            if np.isnan(loss) or np.isinf(loss):
                health_status['errors'].append("Training loss is NaN or Inf")
                health_status['status'] = 'error'
        
        # 檢查梯度範數
        if 'grad_norm' in current_metrics:
            grad_norm = current_metrics['grad_norm']
            
            if grad_norm > 10:
                health_status['warnings'].append(f"Gradient norm is high: {grad_norm:.4f}")
            
            if grad_norm == 0:
                health_status['warnings'].append("Gradient norm is zero - possible gradient vanishing")
        
        # 檢查學習率
        if 'lr' in current_metrics:
            lr = current_metrics['lr']
            
            if lr < 1e-8:
                health_status['warnings'].append(f"Learning rate is very low: {lr:.2e}")
        
        return health_status
    
    def generate_training_report(self):
        """生成訓練報告"""
        report = {
            'experiment_name': self.config.get('experiment_name', 'unknown'),
            'total_steps': len(self.metrics_history),
            'training_duration': self._calculate_training_duration(),
            'best_metrics': self._find_best_metrics(),
            'final_metrics': self.metrics_history[-1]['metrics'] if self.metrics_history else {},
            'system_info': self._get_system_info()
        }
        
        # 保存報告
        report_file = self.experiment_dir / "training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _calculate_training_duration(self):
        """計算訓練時長"""
        if len(self.metrics_history) < 2:
            return 0
        
        start_time = self.metrics_history[0]['timestamp']
        end_time = self.metrics_history[-1]['timestamp']
        duration_seconds = end_time - start_time
        
        return {
            'seconds': duration_seconds,
            'hours': duration_seconds / 3600,
            'days': duration_seconds / (3600 * 24)
        }
    
    def _find_best_metrics(self):
        """找到最佳指標"""
        if not self.metrics_history:
            return {}
        
        best_metrics = {}
        
        # 找到最低的損失
        loss_metrics = ['val_total_loss', 'val_llm_loss', 'val_flow_loss']
        for metric in loss_metrics:
            values = [entry['metrics'].get(metric) for entry in self.metrics_history 
                     if entry['metrics'].get(metric) is not None]
            if values:
                min_val = min(values)
                min_idx = values.index(min_val)
                best_metrics[f'best_{metric}'] = {
                    'value': min_val,
                    'step': self.metrics_history[min_idx]['step']
                }
        
        return best_metrics
    
    def _get_system_info(self):
        """獲取系統信息"""
        import platform
        import torch
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) 
                         for i in range(torch.cuda.device_count())]
        }
```

### 實時監控儀表板

```python
# monitoring/dashboard.py

import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torchaudio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TrainingDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="CosyVoice Training Dashboard",
            page_icon="🎵",
            layout="wide"
        )
        
    def load_metrics(self, experiment_dir: str):
        """載入訓練指標"""
        metrics_file = Path(experiment_dir) / "monitoring" / "metrics.jsonl"
        
        if not metrics_file.exists():
            return pd.DataFrame()
        
        metrics_data = []
        with open(metrics_file, 'r') as f:
            for line in f:
                metrics_data.append(json.loads(line))
        
        return pd.DataFrame(metrics_data)
    
    def display_overview(self, df: pd.DataFrame):
        """顯示概覽信息"""
        if df.empty:
            st.warning("No training data found.")
            return
        
        # 基本統計
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Steps", len(df))
        
        with col2:
            if 'epoch' in df.columns:
                max_epoch = df['epoch'].max()
                st.metric("Max Epoch", max_epoch if pd.notna(max_epoch) else 0)
        
        with col3:
            latest_metrics = df.iloc[-1]['metrics']
            if 'train_total_loss' in latest_metrics:
                st.metric("Latest Loss", f"{latest_metrics['train_total_loss']:.4f}")
        
        with col4:
            # 計算訓練時長
            if len(df) > 1:
                duration = df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']
                hours = duration / 3600
                st.metric("Training Hours", f"{hours:.1f}")
    
    def plot_training_curves(self, df: pd.DataFrame):
        """繪製訓練曲線"""
        if df.empty:
            return
        
        # 展開 metrics 列
        metrics_df = pd.json_normalize(df['metrics'])
        metrics_df['step'] = df['step'].values
        
        # 選擇要顯示的指標
        available_metrics = [col for col in metrics_df.columns if col != 'step']
        selected_metrics = st.multiselect(
            "Select metrics to display:",
            available_metrics,
            default=[m for m in ['train_total_loss', 'val_total_loss'] if m in available_metrics][:4]
        )
        
        if not selected_metrics:
            return
        
        # 創建圖表
        fig = make_subplots(
            rows=len(selected_metrics), cols=1,
            subplot_titles=selected_metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(selected_metrics):
            if metric in metrics_df.columns:
                valid_data = metrics_df.dropna(subset=[metric])
                fig.add_trace(
                    go.Scatter(
                        x=valid_data['step'],
                        y=valid_data[metric],
                        name=metric,
                        mode='lines'
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(height=300*len(selected_metrics), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_audio_samples(self, experiment_dir: str):
        """顯示音頻樣本"""
        audio_dir = Path(experiment_dir) / "monitoring" / "audio_samples"
        
        if not audio_dir.exists():
            st.info("No audio samples found.")
            return
        
        # 獲取所有步數目錄
        step_dirs = sorted([d for d in audio_dir.iterdir() if d.is_dir()], 
                          key=lambda x: int(x.name.split('_')[1]))
        
        if not step_dirs:
            return
        
        # 選擇步數
        selected_step_dir = st.selectbox(
            "Select training step:",
            step_dirs,
            format_func=lambda x: f"Step {x.name.split('_')[1]}"
        )
        
        # 顯示該步數的音頻文件
        audio_files = list(selected_step_dir.glob("*.wav"))
        
        for audio_file in audio_files:
            st.subheader(audio_file.stem)
            
            # 播放音頻
            audio_data, sample_rate = torchaudio.load(audio_file)
            st.audio(str(audio_file), format='audio/wav')
            
            # 顯示波形
            plt.figure(figsize=(12, 4))
            plt.plot(audio_data[0].numpy())
            plt.title(f"Waveform: {audio_file.stem}")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            st.pyplot(plt)
            plt.close()
    
    def display_spectrograms(self, experiment_dir: str):
        """顯示頻譜圖"""
        spec_dir = Path(experiment_dir) / "monitoring" / "spectrograms"
        
        if not spec_dir.exists():
            st.info("No spectrograms found.")
            return
        
        # 獲取所有步數目錄
        step_dirs = sorted([d for d in spec_dir.iterdir() if d.is_dir()], 
                          key=lambda x: int(x.name.split('_')[1]))
        
        if not step_dirs:
            return
        
        # 選擇步數
        selected_step_dir = st.selectbox(
            "Select training step for spectrograms:",
            step_dirs,
            format_func=lambda x: f"Step {x.name.split('_')[1]}"
        )
        
        # 顯示頻譜圖
        spec_files = list(selected_step_dir.glob("*.png"))
        
        cols = st.columns(2)
        for i, spec_file in enumerate(spec_files):
            with cols[i % 2]:
                st.image(str(spec_file), caption=spec_file.stem, use_column_width=True)
    
    def display_system_resources(self, df: pd.DataFrame):
        """顯示系統資源使用情況"""
        # 這需要實際的系統監控數據
        st.subheader("System Resources")
        st.info("System resource monitoring not implemented in this example.")
    
    def run_dashboard(self, experiment_dir: str):
        """運行儀表板"""
        st.title("🎵 CosyVoice Training Dashboard")
        
        # 載入數據
        df = self.load_metrics(experiment_dir)
        
        # 側邊欄
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["Overview", "Training Curves", "Audio Samples", "Spectrograms", "System Resources"]
        )
        
        # 顯示對應頁面
        if page == "Overview":
            st.header("Training Overview")
            self.display_overview(df)
            
        elif page == "Training Curves":
            st.header("Training Curves")
            self.plot_training_curves(df)
            
        elif page == "Audio Samples":
            st.header("Audio Samples")
            self.display_audio_samples(experiment_dir)
            
        elif page == "Spectrograms":
            st.header("Spectrograms")
            self.display_spectrograms(experiment_dir)
            
        elif page == "System Resources":
            st.header("System Resources")
            self.display_system_resources(df)
        
        # 自動刷新
        if st.sidebar.button("Refresh Data"):
            st.experimental_rerun()

# 啟動儀表板
if __name__ == "__main__":
    dashboard = TrainingDashboard()
    
    # 從命令行參數獲取實驗目錄
    import sys
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    else:
        experiment_dir = st.text_input("Enter experiment directory path:")
    
    if experiment_dir:
        dashboard.run_dashboard(experiment_dir)
```

## 模型評估

### 評估指標實現

```python
# evaluation/metrics.py

import torch
import numpy as np
import librosa
from scipy import signal
from pesq import pesq
from pystoi import stoi
import torchaudio
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class TTSEvaluationMetrics:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def mel_cepstral_distortion(self, generated: torch.Tensor, target: torch.Tensor) -> float:
        """計算 Mel-Cepstral Distortion (MCD)"""
        # 轉換到 numpy
        gen_audio = generated.cpu().numpy().flatten()
        tgt_audio = target.cpu().numpy().flatten()
        
        # 確保長度相同
        min_len = min(len(gen_audio), len(tgt_audio))
        gen_audio = gen_audio[:min_len]
        tgt_audio = tgt_audio[:min_len]
        
        # 計算 MFCC
        gen_mfcc = librosa.feature.mfcc(
            y=gen_audio, sr=self.sample_rate, n_mfcc=13
        )
        tgt_mfcc = librosa.feature.mfcc(
            y=tgt_audio, sr=self.sample_rate, n_mfcc=13
        )
        
        # 確保幀數相同
        min_frames = min(gen_mfcc.shape[1], tgt_mfcc.shape[1])
        gen_mfcc = gen_mfcc[:, :min_frames]
        tgt_mfcc = tgt_mfcc[:, :min_frames]
        
        # 計算歐幾里得距離
        mcd = np.mean(np.sqrt(np.sum((gen_mfcc - tgt_mfcc) ** 2, axis=0)))
        mcd = mcd * (10 / np.log(10))  # 轉換為 dB
        
        return float(mcd)
    
    def fundamental_frequency_error(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """計算基頻相關錯誤指標"""
        gen_audio = generated.cpu().numpy().flatten()
        tgt_audio = target.cpu().numpy().flatten()
        
        # 提取 F0
        def extract_f0(audio):
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            return f0, voiced_flag
        
        gen_f0, gen_voiced = extract_f0(gen_audio)
        tgt_f0, tgt_voiced = extract_f0(tgt_audio)
        
        # 對齊長度
        min_len = min(len(gen_f0), len(tgt_f0))
        gen_f0 = gen_f0[:min_len]
        tgt_f0 = tgt_f0[:min_len]
        gen_voiced = gen_voiced[:min_len]
        tgt_voiced = tgt_voiced[:min_len]
        
        # 只考慮有聲段
        voiced_mask = tgt_voiced & gen_voiced & ~np.isnan(gen_f0) & ~np.isnan(tgt_f0)
        
        if np.sum(voiced_mask) == 0:
            return {'f0_rmse': float('inf'), 'f0_corr': 0.0, 'voiced_error': 1.0}
        
        gen_f0_voiced = gen_f0[voiced_mask]
        tgt_f0_voiced = tgt_f0[voiced_mask]
        
        # F0 RMSE
        f0_rmse = np.sqrt(np.mean((gen_f0_voiced - tgt_f0_voiced) ** 2))
        
        # F0 相關係數
        f0_corr = np.corrcoef(gen_f0_voiced, tgt_f0_voiced)[0, 1]
        if np.isnan(f0_corr):
            f0_corr = 0.0
        
        # 有聲/無聲錯誤率
        voiced_error = np.mean(gen_voiced != tgt_voiced)
        
        return {
            'f0_rmse': float(f0_rmse),
            'f0_corr': float(f0_corr),
            'voiced_error': float(voiced_error)
        }
    
    def perceptual_evaluation(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """感知評估指標 (PESQ, STOI)"""
        gen_audio = generated.cpu().numpy().flatten()
        tgt_audio = target.cpu().numpy().flatten()
        
        # 確保採樣率為 16kHz (PESQ 要求)
        if self.sample_rate != 16000:
            gen_audio = librosa.resample(gen_audio, orig_sr=self.sample_rate, target_sr=16000)
            tgt_audio = librosa.resample(tgt_audio, orig_sr=self.sample_rate, target_sr=16000)
            sr_for_pesq = 16000
        else:
            sr_for_pesq = self.sample_rate
        
        # 確保長度相同
        min_len = min(len(gen_audio), len(tgt_audio))
        gen_audio = gen_audio[:min_len]
        tgt_audio = tgt_audio[:min_len]
        
        # 歸一化
        gen_audio = gen_audio / np.max(np.abs(gen_audio) + 1e-8)
        tgt_audio = tgt_audio / np.max(np.abs(tgt_audio) + 1e-8)
        
        metrics = {}
        
        try:
            # PESQ (Perceptual Evaluation of Speech Quality)
            pesq_score = pesq(sr_for_pesq, tgt_audio, gen_audio, 'wb')
            metrics['pesq'] = float(pesq_score)
        except:
            metrics['pesq'] = 0.0
        
        try:
            # STOI (Short-Time Objective Intelligibility)
            stoi_score = stoi(tgt_audio, gen_audio, sr_for_pesq, extended=False)
            metrics['stoi'] = float(stoi_score)
        except:
            metrics['stoi'] = 0.0
        
        return metrics
    
    def spectral_convergence(self, generated: torch.Tensor, target: torch.Tensor) -> float:
        """頻譜收斂性"""
        # 計算 STFT
        gen_stft = torch.stft(generated.squeeze(), n_fft=1024, hop_length=256, 
                             return_complex=True)
        tgt_stft = torch.stft(target.squeeze(), n_fft=1024, hop_length=256, 
                             return_complex=True)
        
        # 對齊長度
        min_time = min(gen_stft.shape[-1], tgt_stft.shape[-1])
        gen_stft = gen_stft[..., :min_time]
        tgt_stft = tgt_stft[..., :min_time]
        
        # 計算幅度
        gen_mag = torch.abs(gen_stft)
        tgt_mag = torch.abs(tgt_stft)
        
        # 頻譜收斂性
        sc = torch.norm(tgt_mag - gen_mag, p='fro') / torch.norm(tgt_mag, p='fro')
        
        return float(sc.item())
    
    def log_spectral_distance(self, generated: torch.Tensor, target: torch.Tensor) -> float:
        """對數頻譜距離"""
        # 計算功率譜
        gen_psd = torch.abs(torch.stft(generated.squeeze(), n_fft=1024, hop_length=256, return_complex=True)) ** 2
        tgt_psd = torch.abs(torch.stft(target.squeeze(), n_fft=1024, hop_length=256, return_complex=True)) ** 2
        
        # 對齊長度
        min_time = min(gen_psd.shape[-1], tgt_psd.shape[-1])
        gen_psd = gen_psd[..., :min_time]
        tgt_psd = tgt_psd[..., :min_time]
        
        # 避免 log(0)
        gen_psd = gen_psd + 1e-8
        tgt_psd = tgt_psd + 1e-8
        
        # 計算對數頻譜距離
        lsd = torch.mean(
            torch.sqrt(torch.mean((torch.log(gen_psd) - torch.log(tgt_psd)) ** 2, dim=-2))
        )
        
        return float(lsd.item())
    
    def evaluate_batch(self, generated_batch: torch.Tensor, 
                      target_batch: torch.Tensor) -> Dict[str, float]:
        """批量評估"""
        batch_size = generated_batch.shape[0]
        
        all_metrics = {
            'mcd': [],
            'f0_rmse': [],
            'f0_corr': [],
            'voiced_error': [],
            'pesq': [],
            'stoi': [],
            'spectral_convergence': [],
            'log_spectral_distance': []
        }
        
        for i in range(batch_size):
            gen_audio = generated_batch[i]
            tgt_audio = target_batch[i]
            
            # MCD
            try:
                mcd = self.mel_cepstral_distortion(gen_audio, tgt_audio)
                all_metrics['mcd'].append(mcd)
            except:
                pass
            
            # F0 相關指標
            try:
                f0_metrics = self.fundamental_frequency_error(gen_audio, tgt_audio)
                all_metrics['f0_rmse'].append(f0_metrics['f0_rmse'])
                all_metrics['f0_corr'].append(f0_metrics['f0_corr'])
                all_metrics['voiced_error'].append(f0_metrics['voiced_error'])
            except:
                pass
            
            # 感知指標
            try:
                perceptual_metrics = self.perceptual_evaluation(gen_audio, tgt_audio)
                all_metrics['pesq'].append(perceptual_metrics['pesq'])
                all_metrics['stoi'].append(perceptual_metrics['stoi'])
            except:
                pass
            
            # 頻譜指標
            try:
                sc = self.spectral_convergence(gen_audio, tgt_audio)
                all_metrics['spectral_convergence'].append(sc)
            except:
                pass
            
            try:
                lsd = self.log_spectral_distance(gen_audio, tgt_audio)
                all_metrics['log_spectral_distance'].append(lsd)
            except:
                pass
        
        # 計算平均值
        averaged_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                averaged_metrics[metric_name] = np.mean(values)
                averaged_metrics[f'{metric_name}_std'] = np.std(values)
        
        return averaged_metrics

class SpeakerSimilarityMetrics:
    def __init__(self):
        # 載入說話人編碼器
        from resemblyzer import VoiceEncoder
        self.speaker_encoder = VoiceEncoder()
    
    def compute_speaker_similarity(self, generated: torch.Tensor, 
                                  reference: torch.Tensor) -> float:
        """計算說話人相似度"""
        # 轉換為 numpy
        gen_audio = generated.cpu().numpy().flatten()
        ref_audio = reference.cpu().numpy().flatten()
        
        # 提取說話人嵌入
        gen_embedding = self.speaker_encoder.embed_utterance(gen_audio)
        ref_embedding = self.speaker_encoder.embed_utterance(ref_audio)
        
        # 計算餘弦相似度
        similarity = np.dot(gen_embedding, ref_embedding) / (
            np.linalg.norm(gen_embedding) * np.linalg.norm(ref_embedding)
        )
        
        return float(similarity)
    
    def evaluate_speaker_consistency(self, generated_batch: List[torch.Tensor], 
                                   speaker_ids: List[str]) -> Dict[str, float]:
        """評估說話人一致性"""
        # 按說話人分組
        speaker_groups = {}
        for i, (audio, spk_id) in enumerate(zip(generated_batch, speaker_ids)):
            if spk_id not in speaker_groups:
                speaker_groups[spk_id] = []
            speaker_groups[spk_id].append(audio)
        
        consistency_scores = []
        
        for spk_id, audios in speaker_groups.items():
            if len(audios) < 2:
                continue
            
            # 計算該說話人所有音頻對的相似度
            similarities = []
            for i in range(len(audios)):
                for j in range(i + 1, len(audios)):
                    sim = self.compute_speaker_similarity(audios[i], audios[j])
                    similarities.append(sim)
            
            if similarities:
                consistency_scores.append(np.mean(similarities))
        
        return {
            'speaker_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
            'speaker_consistency_std': np.std(consistency_scores) if consistency_scores else 0.0
        }
```

### 評估腳本

```python
# evaluation/evaluate.py

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import json

from cosyvoice.cli.cosyvoice import CosyVoice
from evaluation.metrics import TTSEvaluationMetrics, SpeakerSimilarityMetrics

class ModelEvaluator:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = CosyVoice(model_path, device=device)
        self.tts_metrics = TTSEvaluationMetrics()
        self.speaker_metrics = SpeakerSimilarityMetrics()
    
    def evaluate_dataset(self, test_data_file: str, output_dir: str) -> Dict:
        """評估整個測試集"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入測試數據
        test_samples = []
        with open(test_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    test_samples.append({
                        'audio_id': parts[0],
                        'speaker_id': parts[1],
                        'text': parts[2],
                        'audio_path': parts[3]
                    })
        
        print(f"Evaluating {len(test_samples)} samples...")
        
        all_results = []
        generated_audios = []
        target_audios = []
        speaker_ids = []
        
        for sample in tqdm(test_samples, desc="Generating speech"):
            try:
                # 生成語音
                generated_audio = self.generate_speech(
                    text=sample['text'],
                    speaker_id=sample['speaker_id'],
                    reference_audio=sample.get('reference_audio')
                )
                
                # 載入目標音頻
                target_audio, sr = torchaudio.load(sample['audio_path'])
                if sr != 22050:
                    resampler = torchaudio.transforms.Resample(sr, 22050)
                    target_audio = resampler(target_audio)
                
                # 保存生成的音頻
                gen_audio_path = output_dir / f"{sample['audio_id']}_generated.wav"
                torchaudio.save(str(gen_audio_path), generated_audio, 22050)
                
                # 計算指標
                sample_metrics = self.tts_metrics.evaluate_batch(
                    generated_audio.unsqueeze(0),
                    target_audio
                )
                
                # 說話人相似度
                if 'reference_audio' in sample:
                    ref_audio, ref_sr = torchaudio.load(sample['reference_audio'])
                    if ref_sr != 22050:
                        ref_resampler = torchaudio.transforms.Resample(ref_sr, 22050)
                        ref_audio = ref_resampler(ref_audio)
                    
                    speaker_similarity = self.speaker_metrics.compute_speaker_similarity(
                        generated_audio, ref_audio
                    )
                    sample_metrics['speaker_similarity'] = speaker_similarity
                
                sample_metrics['audio_id'] = sample['audio_id']
                sample_metrics['speaker_id'] = sample['speaker_id']
                all_results.append(sample_metrics)
                
                generated_audios.append(generated_audio)
                target_audios.append(target_audio)
                speaker_ids.append(sample['speaker_id'])
                
            except Exception as e:
                print(f"Error processing {sample['audio_id']}: {str(e)}")
                continue
        
        # 計算總體統計
        overall_metrics = self.compute_overall_statistics(all_results)
        
        # 說話人一致性
        consistency_metrics = self.speaker_metrics.evaluate_speaker_consistency(
            generated_audios, speaker_ids
        )
        overall_metrics.update(consistency_metrics)
        
        # 保存結果
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'overall_metrics': overall_metrics,
                'per_sample_metrics': all_results
            }, f, indent=2)
        
        # 保存 CSV 報告
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "detailed_results.csv", index=False)
        
        return overall_metrics
    
    def generate_speech(self, text: str, speaker_id: str, 
                       reference_audio: str = None) -> torch.Tensor:
        """生成語音"""
        if reference_audio:
            # 零樣本生成
            ref_audio, sr = torchaudio.load(reference_audio)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                ref_audio = resampler(ref_audio)
            
            for i, j in enumerate(self.model.inference_zero_shot(
                tts_text=text,
                prompt_text="這是參考語音。",  # 可以自定義
                prompt_speech_16k=ref_audio
            )):
                return j['tts_speech']
        
        else:
            # SFT 生成
            for i, j in enumerate(self.model.inference_sft(
                tts_text=text,
                spk_id=speaker_id
            )):
                return j['tts_speech']
    
    def compute_overall_statistics(self, results: List[Dict]) -> Dict:
        """計算總體統計"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        statistics = {}
        for column in df.columns:
            if column in ['audio_id', 'speaker_id']:
                continue
            
            if df[column].dtype in ['float64', 'int64']:
                statistics[f'{column}_mean'] = float(df[column].mean())
                statistics[f'{column}_std'] = float(df[column].std())
                statistics[f'{column}_median'] = float(df[column].median())
                statistics[f'{column}_min'] = float(df[column].min())
                statistics[f'{column}_max'] = float(df[column].max())
        
        return statistics
    
    def evaluate_single_sample(self, text: str, speaker_id: str, 
                              target_audio_path: str, 
                              reference_audio_path: str = None) -> Dict:
        """評估單個樣本"""
        # 生成語音
        generated_audio = self.generate_speech(text, speaker_id, reference_audio_path)
        
        # 載入目標音頻
        target_audio, sr = torchaudio.load(target_audio_path)
        if sr != 22050:
            resampler = torchaudio.transforms.Resample(sr, 22050)
            target_audio = resampler(target_audio)
        
        # 計算指標
        metrics = self.tts_metrics.evaluate_batch(
            generated_audio.unsqueeze(0),
            target_audio
        )
        
        # 說話人相似度
        if reference_audio_path:
            ref_audio, ref_sr = torchaudio.load(reference_audio_path)
            if ref_sr != 22050:
                ref_resampler = torchaudio.transforms.Resample(ref_sr, 22050)
                ref_audio = ref_resampler(ref_audio)
            
            speaker_similarity = self.speaker_metrics.compute_speaker_similarity(
                generated_audio, ref_audio
            )
            metrics['speaker_similarity'] = speaker_similarity
        
        return metrics, generated_audio

def main():
    parser = argparse.ArgumentParser(description='Evaluate CosyVoice model')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--test_data', required=True, help='Test data file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--device', default='cuda', help='Device to use for inference')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model_path, args.device)
    
    print("Starting evaluation...")
    results = evaluator.evaluate_dataset(args.test_data, args.output_dir)
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nDetailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
```

## 調優策略

### 學習率調優

```python
# optimization/lr_tuning.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Dict, Tuple

class LearningRateFinder:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    def find_lr(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """學習率範圍測試"""
        # 保存原始狀態
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        
        # 設置學習率調度器
        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        
        lrs = []
        losses = []
        best_loss = float('inf')
        
        self.model.train()
        
        for i, batch in enumerate(train_loader):
            if i >= num_iter:
                break
            
            # 記錄當前學習率
            current_lr = self.optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            
            # 前向傳播
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(batch)
            losses_dict = self.model.compute_loss(outputs, batch)
            loss = losses_dict['total_loss']
            
            # 記錄損失
            losses.append(loss.item())
            
            # 反向傳播
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 更新學習率
            lr_scheduler.step()
            
            # 早停條件
            if loss.item() < best_loss:
                best_loss = loss.item()
            elif loss.item() > 4 * best_loss:
                break
        
        # 恢復原始狀態
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
        return lrs, losses
    
    def plot_lr_find(self, lrs, losses, skip_start=10, skip_end=5):
        """繪製學習率尋找結果"""
        if skip_start > len(lrs):
            skip_start = 0
        if skip_end > len(lrs):
            skip_end = 0
            
        lrs = lrs[skip_start:-skip_end if skip_end > 0 else len(lrs)]
        losses = losses[skip_start:-skip_end if skip_end > 0 else len(losses)]
        
        plt.figure(figsize=(12, 6))
        
        # 損失 vs 學習率
        plt.subplot(1, 2, 1)
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate vs Loss')
        plt.grid(True)
        
        # 損失變化率 vs 學習率
        plt.subplot(1, 2, 2)
        losses_smooth = self.smooth(losses, beta=0.98)
        derivatives = np.gradient(losses_smooth)
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss Derivative')
        plt.title('Learning Rate vs Loss Derivative')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 建議的學習率
        min_grad_idx = np.argmin(derivatives)
        suggested_lr = lrs[min_grad_idx]
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        
        return suggested_lr
    
    def smooth(self, losses, beta=0.98):
        """平滑損失曲線"""
        avg_loss = 0
        smoothed = []
        for i, loss in enumerate(losses):
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed.append(avg_loss / (1 - beta ** (i + 1)))
        return smoothed

class ExponentialLR(_LRScheduler):
    """指數學習率調度器"""
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

class CosineAnnealingWarmRestarts(_LRScheduler):
    """餘弦退火重啟調度器"""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i *= self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(np.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
```

### 數據增強策略

```python
# augmentation/advanced_augmentation.py

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Tuple
import random

class AdvancedAudioAugmentation:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def time_masking(self, waveform: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """時間遮罩"""
        seq_len = waveform.shape[-1]
        mask_len = int(seq_len * mask_ratio)
        
        if mask_len > 0:
            start_idx = random.randint(0, seq_len - mask_len)
            waveform_masked = waveform.clone()
            waveform_masked[..., start_idx:start_idx + mask_len] = 0
            return waveform_masked
        
        return waveform
    
    def pitch_shift(self, waveform: torch.Tensor, n_steps: float) -> torch.Tensor:
        """音調變換"""
        try:
            # 使用 phase vocoder 進行音調變換
            stretch_factor = 2 ** (-n_steps / 12)  # 半音計算
            
            # STFT
            stft = torch.stft(waveform.squeeze(), n_fft=2048, hop_length=512, 
                             return_complex=True)
            
            # 時間拉伸
            time_steps = torch.arange(0, stft.shape[-1], stretch_factor)
            time_steps = torch.clamp(time_steps, 0, stft.shape[-1] - 1).long()
            
            stretched_stft = stft[..., time_steps]
            
            # ISTFT
            stretched_waveform = torch.istft(stretched_stft, n_fft=2048, hop_length=512)
            
            # 重採樣回原始長度
            if len(stretched_waveform) != len(waveform.squeeze()):
                stretched_waveform = torch.nn.functional.interpolate(
                    stretched_waveform.unsqueeze(0).unsqueeze(0),
                    size=len(waveform.squeeze()),
                    mode='linear'
                ).squeeze()
            
            return stretched_waveform.unsqueeze(0) if waveform.dim() > 1 else stretched_waveform
            
        except:
            return waveform
    
    def formant_shift(self, waveform: torch.Tensor, shift_factor: float = 1.1) -> torch.Tensor:
        """共振峰變換"""
        # 簡化的共振峰變換實現
        stft = torch.stft(waveform.squeeze(), n_fft=2048, hop_length=512, 
                         return_complex=True)
        
        # 頻率軸變換
        freq_bins = stft.shape[-2]
        new_freq_indices = torch.arange(freq_bins) / shift_factor
        new_freq_indices = torch.clamp(new_freq_indices, 0, freq_bins - 1)
        
        # 線性插值
        shifted_stft = torch.nn.functional.interpolate(
            stft.abs().unsqueeze(0).unsqueeze(0),
            size=(int(freq_bins), stft.shape[-1]),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # 保持原始相位
        shifted_stft = shifted_stft * torch.exp(1j * stft.angle())
        
        # ISTFT
        shifted_waveform = torch.istft(shifted_stft, n_fft=2048, hop_length=512)
        
        return shifted_waveform.unsqueeze(0) if waveform.dim() > 1 else shifted_waveform
    
    def add_reverb(self, waveform: torch.Tensor, room_size: float = 0.3) -> torch.Tensor:
        """添加混響效果"""
        # 簡化的混響實現
        delay_samples = int(self.sample_rate * room_size * 0.01)  # 延遲樣本數
        decay_factor = 0.3
        
        reverb_waveform = waveform.clone()
        
        for i in range(3):  # 多次反射
            delay = delay_samples * (i + 1)
            if delay < len(waveform.squeeze()):
                delayed_signal = torch.nn.functional.pad(
                    waveform * (decay_factor ** (i + 1)), 
                    (delay, 0)
                )[:len(waveform.squeeze())]
                
                reverb_waveform += delayed_signal
        
        # 歸一化
        reverb_waveform = reverb_waveform / reverb_waveform.abs().max() * 0.9
        
        return reverb_waveform
    
    def add_background_noise(self, waveform: torch.Tensor, noise_type: str = 'white', 
                           snr_db: float = 20) -> torch.Tensor:
        """添加背景噪音"""
        signal_power = torch.mean(waveform ** 2)
        
        if noise_type == 'white':
            noise = torch.randn_like(waveform)
        elif noise_type == 'pink':
            # 簡化的粉紅噪音
            noise = torch.randn_like(waveform)
            # 低通濾波近似粉紅噪音
            noise = torch.nn.functional.conv1d(
                noise.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 5).to(waveform.device) / 5,
                padding=2
            ).squeeze()
        else:
            noise = torch.randn_like(waveform)
        
        noise_power = torch.mean(noise ** 2)
        
        # 根據 SNR 調整噪音強度
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        
        return waveform + noise * noise_scale
    
    def speed_perturbation(self, waveform: torch.Tensor, speed_factor: float) -> torch.Tensor:
        """語速擾動"""
        if speed_factor == 1.0:
            return waveform
        
        # 重採樣實現速度變化
        new_sample_rate = int(self.sample_rate * speed_factor)
        
        # 重採樣
        resampler = torchaudio.transforms.Resample(
            self.sample_rate, new_sample_rate
        )
        speed_changed = resampler(waveform)
        
        # 重採樣回原始採樣率
        resampler_back = torchaudio.transforms.Resample(
            new_sample_rate, self.sample_rate
        )
        result = resampler_back(speed_changed)
        
        return result

class MelSpectrogramAugmentation:
    def __init__(self, n_mels=80):
        self.n_mels = n_mels
    
    def freq_masking(self, mel_spec: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """頻率遮罩"""
        n_mels = mel_spec.shape[-1]
        mask_size = int(n_mels * mask_ratio)
        
        if mask_size > 0:
            start_mel = random.randint(0, n_mels - mask_size)
            mel_spec_masked = mel_spec.clone()
            mel_spec_masked[..., start_mel:start_mel + mask_size] = mel_spec.min()
            return mel_spec_masked
        
        return mel_spec
    
    def time_masking(self, mel_spec: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """時間遮罩"""
        time_steps = mel_spec.shape[-2]
        mask_size = int(time_steps * mask_ratio)
        
        if mask_size > 0:
            start_time = random.randint(0, time_steps - mask_size)
            mel_spec_masked = mel_spec.clone()
            mel_spec_masked[..., start_time:start_time + mask_size, :] = mel_spec.min()
            return mel_spec_masked
        
        return mel_spec
    
    def mixup(self, mel_spec1: torch.Tensor, mel_spec2: torch.Tensor, 
             alpha: float = 0.2) -> torch.Tensor:
        """頻譜混合"""
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        
        # 對齊時間維度
        min_time = min(mel_spec1.shape[-2], mel_spec2.shape[-2])
        mel_spec1 = mel_spec1[..., :min_time, :]
        mel_spec2 = mel_spec2[..., :min_time, :]
        
        mixed_spec = lam * mel_spec1 + (1 - lam) * mel_spec2
        
        return mixed_spec

class TextAugmentation:
    def __init__(self, language='zh'):
        self.language = language
        
    def synonym_replacement(self, text: str, replace_prob: float = 0.1) -> str:
        """同義詞替換"""
        if self.language == 'zh':
            # 中文同義詞替換（簡化版）
            synonyms = {
                '很': ['非常', '十分', '相當'],
                '好': ['棒', '不錯', '優秀'],
                '說': ['講', '表示', '提到'],
                '大': ['巨大', '龐大', '大型']
            }
        else:
            # 英文同義詞替換（簡化版）
            synonyms = {
                'good': ['great', 'excellent', 'wonderful'],
                'bad': ['awful', 'terrible', 'horrible'],
                'big': ['large', 'huge', 'enormous'],
                'small': ['tiny', 'little', 'mini']
            }
        
        words = text.split() if self.language == 'en' else list(text)
        
        for i, word in enumerate(words):
            if random.random() < replace_prob and word in synonyms:
                words[i] = random.choice(synonyms[word])
        
        return ' '.join(words) if self.language == 'en' else ''.join(words)
    
    def back_translation(self, text: str) -> str:
        """回譯（需要翻譯API）"""
        # 這裡需要實際的翻譯服務
        # 示例：中文 -> 英文 -> 中文
        return text  # 占位符
    
    def paraphrasing(self, text: str) -> str:
        """句子改寫（需要語言模型）"""
        # 這裡需要實際的改寫模型
        return text  # 占位符
```

### 超參數優化

```python
# optimization/hyperparameter_optimization.py

import optuna
import torch
import pytorch_lightning as pl
from typing import Dict, Any
import json
from pathlib import Path

class HyperparameterOptimizer:
    def __init__(self, base_config: Dict, train_datamodule, val_datamodule):
        self.base_config = base_config
        self.train_datamodule = train_datamodule
        self.val_datamodule = val_datamodule
        
    def objective(self, trial):
        """Optuna 優化目標函數"""
        # 建議超參數
        config = self.base_config.copy()
        
        # 學習率
        config['train_config']['optimizer']['lr'] = trial.suggest_float(
            'lr', 1e-5, 1e-2, log=True
        )
        
        # 批次大小
        config['data_config']['batch_size'] = trial.suggest_categorical(
            'batch_size', [4, 8, 16, 32]
        )
        
        # 梯度累積
        config['train_config']['accumulate_grad_batches'] = trial.suggest_categorical(
            'accumulate_grad_batches', [1, 2, 4]
        )
        
        # 模型參數
        config['model_config']['llm']['dropout'] = trial.suggest_float(
            'dropout', 0.0, 0.3
        )
        
        config['model_config']['llm']['num_layers'] = trial.suggest_int(
            'num_layers', 8, 16
        )
        
        config['model_config']['llm']['num_heads'] = trial.suggest_categorical(
            'num_heads', [8, 12, 16, 20]
        )
        
        # 損失權重
        config['train_config']['loss_weights']['llm_loss'] = trial.suggest_float(
            'llm_loss_weight', 0.5, 2.0
        )
        
        config['train_config']['loss_weights']['flow_loss'] = trial.suggest_float(
            'flow_loss_weight', 0.5, 2.0
        )
        
        # 創建模型和訓練器
        from cosyvoice.models.cosyvoice_model import CosyVoiceModel
        
        model = CosyVoiceModel(config)
        
        # 早停和檢查點
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor='val_total_loss',
                patience=5,
                mode='min'
            )
        ]
        
        trainer = pl.Trainer(
            max_epochs=20,  # 較少的 epoch 用於快速評估
            callbacks=callbacks,
            enable_progress_bar=False,
            logger=False,
            gpus=1 if torch.cuda.is_available() else 0
        )
        
        try:
            trainer.fit(model, self.train_datamodule, self.val_datamodule)
            
            # 返回驗證損失
            val_loss = trainer.callback_metrics.get('val_total_loss', float('inf'))
            
            # 如果訓練失敗或損失異常
            if torch.isnan(val_loss) or torch.isinf(val_loss):
                return float('inf')
            
            return val_loss.item()
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    def optimize(self, n_trials: int = 50, study_name: str = "cosyvoice_hp_opt"):
        """執行超參數優化"""
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        # 輸出結果
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.4f}")
        print("Best params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # 保存結果
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'all_trials': []
        }
        
        for trial in study.trials:
            results['all_trials'].append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })
        
        with open(f'{study_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return study.best_params
    
    def visualize_optimization(self, study_name: str):
        """可視化優化結果"""
        import optuna.visualization as vis
        import plotly.io as pio
        
        # 載入研究結果
        study = optuna.load_study(study_name=study_name)
        
        # 參數重要性
        fig1 = vis.plot_param_importances(study)
        pio.write_html(fig1, f"{study_name}_param_importances.html")
        
        # 優化歷史
        fig2 = vis.plot_optimization_history(study)
        pio.write_html(fig2, f"{study_name}_optimization_history.html")
        
        # 參數關係
        fig3 = vis.plot_parallel_coordinate(study)
        pio.write_html(fig3, f"{study_name}_parallel_coordinate.html")
        
        print(f"可視化圖表已保存到 {study_name}_*.html")

class ModelComparison:
    """模型性能比較"""
    def __init__(self):
        self.results = {}
    
    def add_model(self, model_name: str, metrics: Dict[str, float]):
        """添加模型結果"""
        self.results[model_name] = metrics
    
    def compare_models(self) -> Dict[str, Any]:
        """比較模型性能"""
        if len(self.results) < 2:
            raise ValueError("Need at least 2 models to compare")
        
        comparison = {}
        
        # 獲取所有指標名稱
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())
        
        for metric in all_metrics:
            metric_values = {}
            for model_name, metrics in self.results.items():
                if metric in metrics:
                    metric_values[model_name] = metrics[metric]
            
            if len(metric_values) > 1:
                # 找出最佳模型
                if 'loss' in metric.lower() or 'error' in metric.lower():
                    # 越小越好
                    best_model = min(metric_values.keys(), key=lambda x: metric_values[x])
                else:
                    # 越大越好
                    best_model = max(metric_values.keys(), key=lambda x: metric_values[x])
                
                comparison[metric] = {
                    'values': metric_values,
                    'best_model': best_model,
                    'best_value': metric_values[best_model]
                }
        
        return comparison
    
    def generate_report(self, output_file: str):
        """生成比較報告"""
        comparison = self.compare_models()
        
        report = {
            'models': list(self.results.keys()),
            'comparison': comparison,
            'summary': {}
        }
        
        # 統計每個模型獲勝次數
        win_counts = {model: 0 for model in self.results.keys()}
        for metric_info in comparison.values():
            best_model = metric_info['best_model']
            win_counts[best_model] += 1
        
        report['summary']['win_counts'] = win_counts
        report['summary']['overall_best'] = max(win_counts.keys(), key=lambda x: win_counts[x])
        
        # 保存報告
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
```

## 常見問題

### 訓練相關問題

**Q1: 訓練損失不下降怎麼辦？**

A: 可能的原因和解決方案：

1. **學習率過大或過小**
   ```python
   # 使用學習率尋找器
   lr_finder = LearningRateFinder(model, optimizer)
   lrs, losses = lr_finder.find_lr(train_loader)
   suggested_lr = lr_finder.plot_lr_find(lrs, losses)
   ```

2. **梯度消失或爆炸**
   ```python
   # 檢查梯度範數
   total_norm = 0
   for p in model.parameters():
       if p.grad is not None:
           param_norm = p.grad.data.norm(2)
           total_norm += param_norm.item() ** 2
   total_norm = total_norm ** (1. / 2)
   print(f"Gradient norm: {total_norm}")
   
   # 調整梯度裁剪
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **數據問題**
   ```python
   # 檢查數據分布
   def check_data_distribution(dataloader):
       all_texts = []
       all_durations = []
       
       for batch in dataloader:
           all_texts.extend(batch['texts'])
           all_durations.extend([len(text) for text in batch['texts']])
       
       print(f"Text length - Mean: {np.mean(all_durations):.2f}, Std: {np.std(all_durations):.2f}")
       print(f"Max length: {max(all_durations)}, Min length: {min(all_durations)}")
   ```

**Q2: GPU 內存不足怎麼解決？**

A: 內存優化策略：

```python
# 1. 梯度檢查點
model.gradient_checkpointing_enable()

# 2. 混合精度訓練
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(batch)
    loss = compute_loss(outputs, batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 3. 動態批次大小
def get_optimal_batch_size():
    available_memory = torch.cuda.get_device_properties(0).total_memory
    used_memory = torch.cuda.memory_allocated(0)
    free_memory = available_memory - used_memory
    
    if free_memory > 16e9:  # 16GB
        return 16
    elif free_memory > 8e9:   # 8GB
        return 8
    else:
        return 4
```

**Q3: 訓練速度太慢怎麼優化？**

A: 速度優化方法：

```python
# 1. 數據加載優化
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,  # 增加進程數
    pin_memory=True,  # 固定內存
    prefetch_factor=4,  # 預取因子
    persistent_workers=True  # 持久化工作進程
)

# 2. JIT 編譯
model = torch.jit.script(model)

# 3. 模型並行
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# 4. 使用更快的優化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.98),
    eps=1e-9,
    fused=True  # 使用融合 AdamW
)
```

### 模型效果相關問題

**Q4: 生成的語音品質不好怎麼改善？**

A: 品質改善策略：

```python
# 1. 調整採樣參數
generation_config = {
    'top_p': 0.8,          # 降低隨機性
    'top_k': 25,           # 限制候選詞
    'temperature': 0.8,     # 降低溫度
    'flow_steps': 32,      # 增加推理步數
    'cfg_scale': 0.9       # 增強引導信號
}

# 2. 後處理改善
def post_process_audio(audio):
    # 音量正規化
    audio = audio / audio.abs().max() * 0.95
    
    # 去除DC偏移
    audio = audio - audio.mean()
    
    # 輕微低通濾波
    audio = torchaudio.functional.lowpass_biquad(
        audio, sample_rate=22050, cutoff_freq=8000
    )
    
    return audio
```

**Q5: 如何改善多語言效果？**

A: 多語言優化：

```python
# 1. 語言特定的文本處理
def language_specific_processing(text, language):
    if language == 'zh':
        # 中文數字正規化
        text = normalize_chinese_numbers(text)
    elif language == 'en':
        # 英文縮寫展開
        text = expand_contractions(text)
    elif language == 'ja':
        # 日文假名處理
        text = normalize_japanese_text(text)
    
    return text

# 2. 語言平衡的批次採樣
class LanguageBalancedSampler:
    def __init__(self, dataset, languages):
        self.dataset = dataset
        self.languages = languages
        self.indices_by_lang = self.group_by_language()
    
    def group_by_language(self):
        indices = {lang: [] for lang in self.languages}
        for i, sample in enumerate(self.dataset):
            lang = detect_language(sample['text'])
            if lang in indices:
                indices[lang].append(i)
        return indices
```

這完成了 CosyVoice 訓練指南的所有內容，涵蓋了從數據準備到模型評估和調優的完整流程，為訓練高品質的語音合成模型提供了詳細的技術指導。