# CosyVoice API 參考

## CosyVoice 類別

### 初始化
```python
CosyVoice(model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1)
```

**參數說明**:
- `model_dir` (str): 模型目錄路徑，或 ModelScope 模型 ID
- `load_jit` (bool): 是否載入 JIT 優化模型，預設 False
- `load_trt` (bool): 是否使用 TensorRT 加速，預設 False
- `fp16` (bool): 是否使用半精度推理，預設 False
- `trt_concurrent` (int): TensorRT 併發數量，預設 1

**使用範例**:
```python
# 基本使用
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

# 開啟 FP16 加速
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M', fp16=True)

# 使用 ModelScope ID
cosyvoice = CosyVoice('iic/CosyVoice-300M')
```

### 主要方法

#### inference_sft()
使用預訓練音色進行語音合成。

```python
inference_sft(tts_text, spk_id, stream=False, speed=1.0, text_frontend=True)
```

**參數**:
- `tts_text` (str): 要合成的文字
- `spk_id` (str): 說話人 ID，可用 `list_available_spks()` 查看
- `stream` (bool): 是否使用串流模式，預設 False
- `speed` (float): 語速控制，預設 1.0
- `text_frontend` (bool): 是否使用文字前端處理，預設 True

**返回值**: 生成器，產生包含 `tts_speech` 的字典

**範例**:
```python
for i, j in enumerate(cosyvoice.inference_sft('你好，世界', '中文女')):
    torchaudio.save(f'sft_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### inference_zero_shot()
使用參考語音進行零樣本音色克隆。

```python
inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)
```

**參數**:
- `tts_text` (str): 要合成的文字
- `prompt_text` (str): 參考語音對應的文字
- `prompt_speech_16k` (torch.Tensor): 16kHz 的參考語音
- `zero_shot_spk_id` (str): 已儲存的零樣本說話人 ID，可略
- 其他參數同 `inference_sft()`

**範例**:
```python
prompt_speech = load_wav('./prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('你好', '希望你好', prompt_speech)):
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### inference_cross_lingual()
跨語言語音合成，使用一種語言的音色合成另一種語言。

```python
inference_cross_lingual(tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)
```

**參數**:
- `tts_text` (str): 要合成的文字（可包含語言標記如 `<|en|>`）
- `prompt_speech_16k` (torch.Tensor): 參考語音
- 其他參數同上

#### inference_instruct()
使用自然語言指令控制語音特性。

```python
inference_instruct(tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True)
```

**參數**:
- `tts_text` (str): 要合成的文字
- `spk_id` (str): 說話人 ID
- `instruct_text` (str): 指令文字，如 '用四川話說這句話'
- 其他參數同上

**範例**:
```python
for i, j in enumerate(cosyvoice.inference_instruct('你好', '中文男', '用四川話說')):
    torchaudio.save(f'instruct_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### inference_vc()
語音轉換，將一段語音轉換為另一種音色。

```python
inference_vc(source_speech_16k, prompt_speech_16k, stream=False, speed=1.0)
```

**參數**:
- `source_speech_16k` (torch.Tensor): 源語音
- `prompt_speech_16k` (torch.Tensor): 目標音色參考語音
- 其他參數同上

### 其他方法

#### list_available_spks()
取得所有可用的預訓練說話人 ID。

```python
spks = cosyvoice.list_available_spks()
print(spks)  # ['中文女', '中文男', '英文女', ...]
```

#### add_zero_shot_spk()
新增一個零樣本說話人並儲存。

```python
add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)
```

**參數**:
- `prompt_text` (str): 參考語音對應文字
- `prompt_speech_16k` (torch.Tensor): 參考語音
- `zero_shot_spk_id` (str): 新說話人的 ID

**返回值**: bool，成功返回 True

#### save_spkinfo()
儲存說話人資訊到模型目錄。

```python
cosyvoice.save_spkinfo()
```

## CosyVoice2 類別

CosyVoice2 繼承自 CosyVoice，增強了性能和功能。

### 初始化
```python
CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1)
```

**新增參數**:
- `load_vllm` (bool): 是否使用 vLLM 加速，需要安裝 vllm==v0.9.0

### 特有方法

#### inference_instruct2()
CosyVoice2 版本的指令式合成，支援更多控制功能。

```python
inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True)
```

**範例**:
```python
prompt_speech = load_wav('./prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_instruct2('你好', '用四川話說', prompt_speech)):
    torchaudio.save(f'instruct2_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### 特殊控制符號

CosyVoice2 支援更多的控制符號：
- `[laughter]`: 笑聲
- `[breath]`: 呼吸聲
- `<strong></strong>`: 強調
- `<laughter></laughter>`: 笑聲段落

**範例**:
```python
text = '在他講述那個荒誥故事的過程中，他突然[laughter]停下來，因為他自己也被逗笑了[laughter]。'
for i, j in enumerate(cosyvoice.inference_cross_lingual(text, prompt_speech)):
    torchaudio.save(f'control_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### vLLM 支援

使用 vLLM 可以显著提升推理速度：

```python
# 需要先安裝 vllm==v0.9.0
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_vllm=True)

# 使用方式相同
for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech)):
    torchaudio.save(f'vllm_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

## 前端處理 API

### CosyVoiceFrontEnd
前端處理模組，負責文字和語音的預處理。

```python
CosyVoiceFrontEnd(get_tokenizer, feat_extractor, campplus_model, speech_tokenizer_model, spk2info='', allowed_special='all')
```

**主要方法**:

#### text_normalize()
文字標準化處理。

```python
text_normalize(text, split=True, text_frontend=True)
```

#### frontend_sft()
為 SFT 推理準備輸入數據。

```python
frontend_sft(text, spk_id)
```

#### frontend_zero_shot()
為零樣本推理準備輸入數據。

```python
frontend_zero_shot(text, prompt_text, prompt_speech, sample_rate, spk_id='')
```

## 工具函數

### 檔案工具 (file_utils)

#### load_wav()
載入音頻檔案。

```python
from cosyvoice.utils.file_utils import load_wav
audio = load_wav('path/to/audio.wav', target_sample_rate=16000)
```

#### convert_onnx_to_trt()
將 ONNX 模型轉換為 TensorRT 模型。

### 常用工具 (common)

#### set_all_random_seed()
設定所有隨機種子。

```python
from cosyvoice.utils.common import set_all_random_seed
set_all_random_seed(42)
```

#### fade_in_out()
為語音添加淡入淡出效果。

### 前端工具 (frontend_utils)

#### contains_chinese()
檢查文字是否包含中文。

#### replace_blank()
更換空格和特殊字元。

#### spell_out_number()
將數字轉換為文字。

## 錯誤處理

### 常見異常

#### ModelNotFoundError
模型檔案找不到或損壞。

**解決方案**:
- 檢查模型目錄是否存在
- 重新下載模型
- 確認檔案完整性

#### OutOfMemoryError
顯存或內存不足。

**解決方案**:
- 使用 `fp16=True` 減少內存使用
- 減少批次大小
- 使用更小的模型

#### ValueError: Invalid speaker ID
說話人 ID 不存在。

**解決方案**:
```python
# 檢查可用說話人
available_spks = cosyvoice.list_available_spks()
print(f'可用說話人: {available_spks}')
```

### 異常處理範例

```python
try:
    cosyvoice = CosyVoice('pretrained_models/CosyVoice2-0.5B')
    for i, j in enumerate(cosyvoice.inference_sft('測試文字', '中文女')):
        torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
except FileNotFoundError:
    print('模型檔案找不到，請檢查模型目錄')
except torch.cuda.OutOfMemoryError:
    print('顯存不足，嘗試使用 fp16 或更小的模型')
except Exception as e:
    print(f'發生未知錯誤: {e}')
```