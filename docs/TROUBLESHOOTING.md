# CosyVoice 故障排除

## 安裝問題

### 依賴衝突

**PyTorch 版本衝突:**
```bash
# 錯誤訊息: RuntimeError: The NVIDIA driver on your system is too old
# 解決方案: 更新 CUDA 和 PyTorch 版本
pip uninstall torch torchaudio
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 檢查 CUDA 相容性
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

**HuggingFace 依賴衝突:**
```bash
# 錯誤: ImportError: cannot import name 'AutoTokenizer'
# 解決方案: 更新 transformers 库
pip install transformers>=4.21.0
pip install tokenizers>=0.13.0

# 強制重新安裝
pip install --upgrade --force-reinstall transformers tokenizers
```

**Matcha-TTS 相關問題:**
```bash
# 錯誤: ModuleNotFoundError: No module named 'matcha'
# 解決方案: 正確設置 Python 路徑
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/third_party/Matcha-TTS"

# 檢查子模組是否正確初始化
git submodule update --init --recursive
cd third_party/Matcha-TTS && pip install -e .
```

### 模型下載問題

**ModelScope 下載失敗:**
```python
# 錯誤: ConnectionError: Failed to download
# 解決方案 1: 使用鏡像站
from modelscope import snapshot_download
import os

os.environ['MODELSCOPE_CACHE'] = '/path/to/cache'
os.environ['MODELSCOPE_HUB_ENDPOINT'] = 'https://modelscope.cn'  # 使用國內鏡像

snapshot_download('iic/CosyVoice-300M', 
                  local_dir='pretrained_models/CosyVoice-300M',
                  revision='master')
```

**網絡連接問題:**
```bash
# 使用代理下載
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# 或使用 Git LFS 手動下載
git lfs install
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git

# 如果 LFS 下載失敗，手動下載大文件
wget https://modelscope.cn/api/v1/models/iic/CosyVoice-300M/repo/files/download/cosyvoice.pt
```

**模型文件損壞:**
```bash
# 檢查模型文件完整性
md5sum pretrained_models/CosyVoice-300M/*.pt

# 重新下載損壞的文件
rm -rf pretrained_models/CosyVoice-300M
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
```

### 第三方模組問題

**Matcha-TTS 安裝問題:**
```bash
# 錯誤: ImportError: No module named 'matcha.models'
# 解決方案:
cd third_party/Matcha-TTS
pip install -e .

# 檢查是否安裝成功
python -c "import matcha; print('Matcha-TTS installed successfully')"
```

**WeTextProcessing 問題:**
```bash
# 錯誤: No module named 'WeTextProcessing'
# 解決方案: 手動安裝
git clone https://github.com/wenet-e2e/WeTextProcessing.git
cd WeTextProcessing
python setup.py install

# 或使用 pip 安裝
pip install WeTextProcessing
```

**onnxruntime 相容性問題:**
```bash
# GPU 版本衝突
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.15.1  # 選擇相容的版本

# CPU 版本
pip install onnxruntime==1.15.1
```

## 執行時錯誤

### 記憶體不足

**GPU 記憶體不足:**
```python
# 錯誤: CUDA out of memory
# 解決方案 1: 清理 GPU 記憶體
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

# 解決方案 2: 啟用混合精度
model = CosyVoice('pretrained_models/CosyVoice-300M', fp16=True)

# 解決方案 3: 減小批處理大小
# 在配置文件中修改 max_frames_in_batch
max_frames_in_batch: 1000  # 從 2000 減少到 1000

# 解決方案 4: 啟用梯度檢查點
model.llm.gradient_checkpointing_enable()
model.flow.gradient_checkpointing_enable()
```

**CPU 記憶體不足:**
```bash
# 增加虛擬記憶體
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**內存泉漏問題:**
```python
# 在推理循環中添加內存清理
def inference_with_cleanup():
    try:
        # TTS 推理程序
        result = cosyvoice.inference_sft(...)
        return result
    finally:
        # 清理記憶體
        torch.cuda.empty_cache()
        gc.collect()
```

### CUDA 相關問題

**CUDA 驅動版本不相容:**
```bash
# 檢查 CUDA 版本
nvidia-smi
nvcc --version

# 檢查 PyTorch CUDA 支援
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 安裝相容的 PyTorch 版本
# 為 CUDA 11.8
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 為 CUDA 12.1
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

**GPU 不可用問題:**
```python
# 檢查 GPU 狀態
import torch

if not torch.cuda.is_available():
    print("CUDA not available, possible causes:")
    print("1. NVIDIA driver not installed")
    print("2. CUDA runtime not installed")
    print("3. PyTorch not compiled with CUDA support")
    
# 強制使用 CPU
model = CosyVoice('pretrained_models/CosyVoice-300M', device='cpu')
```

**多 GPU 錯誤:**
```bash
# 錯誤: RuntimeError: Expected all tensors to be on the same device
# 解決方案: 指定單個 GPU
export CUDA_VISIBLE_DEVICES=0  # 只使用第一個 GPU

# 或在 Python 中設定
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

**TensorRT 問題:**
```bash
# TensorRT 安裝
pip install tensorrt==8.6.1  # 選擇相容版本

# 檢查 TensorRT 是否可用
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

# 如果 TensorRT 不可用，禁用 TensorRT 加速
model = CosyVoice('pretrained_models/CosyVoice-300M', load_trt=False)
```

### 模型載入錯誤

**模型文件遺失:**
```python
# 錯誤: FileNotFoundError: [Errno 2] No such file or directory: 'pretrained_models/CosyVoice-300M/cosyvoice.yaml'
# 解決方案: 檢查模型文件結構
import os

model_dir = 'pretrained_models/CosyVoice-300M'
required_files = ['cosyvoice.yaml', 'llm.pt', 'flow.pt', 'hift.pt']

for file in required_files:
    file_path = os.path.join(model_dir, file)
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        print("Please re-download the model")

# 重新下載模型
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M', local_dir=model_dir, allow_file_pattern=['*.pt', '*.yaml', '*.json'])
```

**模型版本不相容:**
```python
# 錯誤: RuntimeError: Error(s) in loading state_dict
# 解決方案: 檢查模型版本相容性
try:
    model = CosyVoice('pretrained_models/CosyVoice-300M')
except Exception as e:
    print(f"Model loading error: {e}")
    print("Try using a different model version:")
    
    # 嘗試不同的模型版本
    alternative_models = [
        'pretrained_models/CosyVoice-300M-SFT',
        'pretrained_models/CosyVoice2-0.5B'
    ]
    
    for alt_model in alternative_models:
        if os.path.exists(alt_model):
            try:
                model = CosyVoice(alt_model)
                print(f"Successfully loaded: {alt_model}")
                break
            except:
                continue
```

**配置文件錯誤:**
```yaml
# 錯誤: yaml.constructor.ConstructorError
# 解決方案: 檢查配置文件格式
# 在 cosyvoice.yaml 中檢查是否有不支援的標籤

# 常見問題: !ref 標籤錯誤
# 確保所有引用的變數都已定義
sample_rate: 22050  # 必須在 !ref <sample_rate> 之前定義

# 如果仍然有問題，使用默認配置
```

**模型初始化失敗:**
```python
# 錯誤: RuntimeError: CUDA error: device-side assert triggered
# 解決方案: 使用 CPU 模式排查
model = CosyVoice('pretrained_models/CosyVoice-300M', device='cpu')

# 或減少模型大小
model = CosyVoice('pretrained_models/CosyVoice-300M', 
                  fp16=True,  # 使用半精度
                  load_jit=False,  # 禁用 JIT
                  load_trt=False)  # 禁用 TensorRT
```

## 音訊品質問題

### 語音品質低劣

**提高採樣參數:**
```python
# 調節採樣策略
for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text=text,
    prompt_text=prompt_text,
    prompt_speech_16k=prompt_speech_16k,
    top_p=0.7,      # 降低 top_p 獲得更穩定的輸出
    top_k=20,       # 降低 top_k
    temperature=0.8  # 降低溫度
)):
    result = j['tts_speech']

# RAS 參數調優
ras_params = {
    'win_size': 15,  # 增大窗口大小
    'tau_r': 0.05    # 更強的重複抑制
}
```

**Flow 推理參數優化:**
```python
# 增加 Flow 推理步數提高語音品質
flow_params = {
    'inference_steps': 32,    # 從默認 22 增加到 32
    'cfg_scale': 0.9,         # 增強引導信號
    'solver': 'heun'          # 使用更精確的求解器
}

# 應用到推理
model.flow.update_config(flow_params)
```

**提高參考音訊品質:**
```python
# 清理參考音訊
import torchaudio
from torchaudio.transforms import Resample, Vol

# 音頻預處理
def preprocess_reference_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 正規化音量
    waveform = waveform / waveform.abs().max() * 0.8
    
    # 重採樣到 16kHz
    if sample_rate != 16000:
        resampler = Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # 去除靜音
    non_silent_indices = torch.where(waveform.abs() > 0.01)[1]
    if len(non_silent_indices) > 0:
        start_idx = non_silent_indices[0].item()
        end_idx = non_silent_indices[-1].item()
        waveform = waveform[:, start_idx:end_idx]
    
    return waveform

# 使用預處理的參考音頻
prompt_speech_16k = preprocess_reference_audio('reference.wav')
```

### 發音錯誤

**中文發音問題:**
```python
# 使用正確的分詞器
from cosyvoice.utils.frontend_utils import get_tokenizer

# 確保使用中文分詞器
tokenizer = get_tokenizer(language='zh')

# 檢查文本預處理
def preprocess_chinese_text(text):
    # 正規化數字
    text = text.replace('1', '一')
    text = text.replace('2', '二')
    text = text.replace('3', '三')
    # ... 繼續其他數字
    
    # 處理标點符號
    text = text.replace('?', '？')
    text = text.replace('!', '！')
    
    return text

# 使用範例
processed_text = preprocess_chinese_text("你好123，今天天氣怎麼樣?")
result = cosyvoice.inference_sft(tts_text=processed_text, spk_id='zh-CN-female-1')
```

**英文發音問題:**
```python
# 使用正確的發音字典
from g2p_en import G2p

g2p = G2p()
phonemes = g2p("Hello world")  # 轉換為音素
print(phonemes)  # ['HH', 'AH0', 'L', 'OW1', ' ', 'W', 'ER1', 'L', 'D']

# 手動指定發音
text_with_phonemes = "Hello <phoneme>HH AH0 L OW1</phoneme> world"
```

**多語言混合問題:**
```python
# 分別處理不同語言片段
def process_multilingual_text(text):
    import re
    
    # 使用正則表達式分離中英文
    chinese_pattern = r'[\u4e00-\u9fff]+'
    english_pattern = r'[a-zA-Z]+'
    
    # 為不同語言添加標記
    processed_text = re.sub(chinese_pattern, r'<zh>\g<0></zh>', text)
    processed_text = re.sub(english_pattern, r'<en>\g<0></en>', processed_text)
    
    return processed_text

# 使用示例
mixed_text = "你好 hello 世界 world"
processed = process_multilingual_text(mixed_text)
print(processed)  # <zh>你好</zh> <en>hello</en> <zh>世界</zh> <en>world</en>
```

### 音訊失真

**HiFi-GAN 參數調優:**
```python
# 調節声碼器參數減少失真
vocoder_params = {
    'audio_limit': 0.95,      # 降低音頻限幅以防止削峰
    'nb_harmonics': 8,        # 保持足夠的諜波數量
    'nsf_alpha': 0.1,         # NSF alpha 參數
    'nsf_sigma': 0.003        # NSF sigma 參數
}

# 更新声碼器配置
model.hifigan.generator.update_config(vocoder_params)
```

**後處理減少失真:**
```python
import torchaudio
from torchaudio.transforms import Vol, Resample

def post_process_audio(audio_tensor, target_sr=22050):
    # 1. 音量正規化
    max_val = audio_tensor.abs().max()
    if max_val > 0.95:
        audio_tensor = audio_tensor / max_val * 0.9
    
    # 2. 去除DC偏置
    audio_tensor = audio_tensor - audio_tensor.mean()
    
    # 3. 輕微濾波去除高頻噪音
    audio_tensor = torchaudio.functional.lowpass_biquad(
        audio_tensor, target_sr, cutoff_freq=8000
    )
    
    # 4. 淺入淺出效果
    fade_len = min(1000, len(audio_tensor[0]) // 10)
    audio_tensor = torchaudio.functional.fade(
        audio_tensor, fade_in_len=fade_len, fade_out_len=fade_len
    )
    
    return audio_tensor

# 使用示例
for i, j in enumerate(cosyvoice.inference_sft(tts_text=text, spk_id=speaker)):
    clean_audio = post_process_audio(j['tts_speech'])
    torchaudio.save(f'clean_output_{i}.wav', clean_audio, 22050)
```

**噪音抑制:**
```python
# 使用噪音抑制算法
from scipy.signal import wiener
import numpy as np

def denoise_audio(audio_np):
    # Wiener 濾波器去噪
    denoised = wiener(audio_np, mysize=5)
    
    # 轉換回 PyTorch tensor
    return torch.from_numpy(denoised).float()

# 應用去噪
audio_np = j['tts_speech'].numpy()
denoised_audio = denoise_audio(audio_np[0])  # 取第一個通道
denoised_tensor = denoised_audio.unsqueeze(0)  # 恢復維度
```

## 效能問題

### 推理速度慢

**啟用所有加速優化:**
```python
# 最大化效能初始化
model = CosyVoice(
    'pretrained_models/CosyVoice-300M',
    load_jit=True,      # JIT 編譯
    load_trt=True,      # TensorRT 加速
    fp16=True,          # 半精度
    device='cuda',      # 使用 GPU
    trt_concurrent=4    # TensorRT 並發數
)

# 啟用其他加速選項
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

**減少推理步數:**
```python
# 調節 Flow 推理參數提高速度
fast_inference_config = {
    'flow_steps': 16,        # 從 22 降低到 16
    'cfg_scale': 0.5,        # 降低 CFG 強度
    'top_k': 15,            # 減少採樣候選
    'temperature': 0.9       # 輕微降低溫度
}

# 應用到推理
result = cosyvoice.inference_sft(
    tts_text=text,
    spk_id=speaker,
    **fast_inference_config
)
```

**批處理優化:**
```python
# 批量處理多個請求
def batch_inference(texts, speaker_ids, batch_size=4):
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_speakers = speaker_ids[i:i+batch_size]
        
        # 批量處理
        batch_results = model.batch_inference_sft(
            batch_texts, batch_speakers
        )
        results.extend(batch_results)
    
    return results

# 使用示例
texts = ["你好", "再見", "謝謝"]
speakers = ["zh-CN-female-1"] * 3
batch_results = batch_inference(texts, speakers)
```

**記憶體和運算優化:**
```python
# 啟用記憶體優化
with torch.no_grad():  # 禁用梯度計算
    with torch.cuda.amp.autocast():  # 自動混合精度
        result = cosyvoice.inference_sft(tts_text=text, spk_id=speaker)

# 啟用 Flash Attention
model.llm.config.use_flash_attention = True
model.flow.config.use_flash_attention = True
```

### 高延遲問題

**流式推理實現:**
```python
# 流式語音合成
class StreamingTTS:
    def __init__(self, model_path):
        self.model = CosyVoice(model_path, streaming=True)
        self.chunk_size = 1024  # 音頻塊大小
    
    def stream_inference(self, text, speaker_id):
        """流式推理生成器"""
        for chunk in self.model.stream_inference_sft(
            tts_text=text,
            spk_id=speaker_id,
            chunk_size=self.chunk_size
        ):
            yield chunk['audio_chunk']
    
    def get_first_chunk_latency(self, text, speaker_id):
        """測量首個音頻塊延遲"""
        import time
        start_time = time.time()
        
        for chunk in self.stream_inference(text, speaker_id):
            first_chunk_time = time.time() - start_time
            print(f"首個音頻塊延遲: {first_chunk_time*1000:.2f}ms")
            return chunk, first_chunk_time

# 使用示例
streaming_tts = StreamingTTS('pretrained_models/CosyVoice-300M')
for chunk in streaming_tts.stream_inference("你好世界", "zh-CN-female-1"):
    # 即時播放或保存音頻塊
    process_audio_chunk(chunk)
```

**預加載和緩存:**
```python
# 模型預加載
class PreloadedTTS:
    def __init__(self, model_path):
        # 預加載所有組件
        self.model = CosyVoice(model_path)
        self._warmup_model()
        self.cache = {}  # 結果緩存
    
    def _warmup_model(self):
        """模型預熱"""
        dummy_text = "測試文本"
        dummy_speaker = "zh-CN-female-1"
        
        # 運行一次推理來初始化所有緩存
        _ = list(self.model.inference_sft(
            tts_text=dummy_text, 
            spk_id=dummy_speaker
        ))
        print("模型預熱完成")
    
    def inference_with_cache(self, text, speaker_id):
        cache_key = f"{text}_{speaker_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = list(self.model.inference_sft(
            tts_text=text,
            spk_id=speaker_id
        ))
        
        # 緩存結果方便下次使用
        self.cache[cache_key] = result
        return result

preloaded_tts = PreloadedTTS('pretrained_models/CosyVoice-300M')
```

**異步處理架構:**
```python
import asyncio
import aiohttp
from queue import Queue
import threading

class AsyncTTSServer:
    def __init__(self, model_path, max_concurrent=4):
        self.model = CosyVoice(model_path)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_queue = Queue()
        
    async def async_inference(self, text, speaker_id, request_id):
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            # 在線程池中異步執行 TTS
            result = await loop.run_in_executor(
                None,
                self._sync_inference,
                text,
                speaker_id
            )
            
            return {
                'request_id': request_id,
                'audio': result,
                'status': 'completed'
            }
    
    def _sync_inference(self, text, speaker_id):
        """ 同步 TTS 推理函數 """
        for i, j in enumerate(self.model.inference_sft(
            tts_text=text,
            spk_id=speaker_id
        )):
            return j['tts_speech']

# FastAPI 服務使用異步處理
from fastapi import FastAPI

app = FastAPI()
tts_server = AsyncTTSServer('pretrained_models/CosyVoice-300M')

@app.post("/async_tts")
async def async_tts_endpoint(text: str, speaker: str, request_id: str):
    result = await tts_server.async_inference(text, speaker, request_id)
    return result
```

### 資源占用過高

**GPU 記憶體優化:**
```python
# 記憶體池管理
class GPUMemoryManager:
    def __init__(self):
        self.memory_pool = torch.cuda.memory.MemoryPool()
        torch.cuda.set_memory_pool(self.memory_pool)
    
    def optimize_memory_usage(self):
        # 清理未使用的緩存
        torch.cuda.empty_cache()
        
        # 設定記憶體分片策略
        torch.cuda.memory._set_allocator_settings('expandable_segments:True')
        
    def get_memory_stats(self):
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'cached': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }

# 使用記憶體管理器
mem_manager = GPUMemoryManager()
mem_manager.optimize_memory_usage()

# 在推理前後清理記憶體
def inference_with_memory_cleanup(model, text, speaker):
    try:
        torch.cuda.empty_cache()  # 推理前清理
        result = list(model.inference_sft(tts_text=text, spk_id=speaker))
        return result
    finally:
        torch.cuda.empty_cache()  # 推理後清理
        gc.collect()
```

**CPU 使用優化:**
```python
# 限制 CPU 線程數
import os

# 設定 OpenMP 線程數
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

# PyTorch 線程數
torch.set_num_threads(4)

# 監控 CPU 使用率
import psutil

def monitor_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    if cpu_percent > 90:
        print(f"警告: CPU 使用率過高: {cpu_percent}%")
    
    if memory_percent > 85:
        print(f"警告: 記憶體使用率過高: {memory_percent}%")
    
    return {'cpu': cpu_percent, 'memory': memory_percent}
```

**模型量化和稀疏化:**
```python
# 模型量化
from torch.quantization import quantize_dynamic

def quantize_model(model):
    # 動態量化
    quantized_model = quantize_dynamic(
        model.llm,  # 量化 LLM 部分
        {torch.nn.Linear},  # 量化線性層
        dtype=torch.qint8
    )
    
    model.llm = quantized_model
    return model

# 模型稀疏化
def apply_sparsity(model, sparsity_level=0.3):
    import torch.nn.utils.prune as prune
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=sparsity_level)
            prune.remove(module, 'weight')  # 永久化修剪
    
    return model

# 使用量化模型
model = CosyVoice('pretrained_models/CosyVoice-300M')
model = quantize_model(model)
model = apply_sparsity(model, 0.3)
```

**資源監控和限制:**
```python
# 資源使用限制
import resource

def set_resource_limits():
    # 限制記憶體使用 (8GB)
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, -1))
    
    # 限制 CPU 時間
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))  # 5分鐘

def resource_usage_monitor(func):
    """ 資源使用監控裝飾器 """
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        print(f"\n資源使用統計:")
        print(f"執行時間: {end_time - start_time:.2f}秒")
        print(f"記憶體增量: {end_memory - start_memory:.2f}MB")
        print(f"當前記憶體: {end_memory:.2f}MB")
        
        return result
    return wrapper

# 使用資源監控
@resource_usage_monitor
def monitored_inference(text, speaker):
    return list(model.inference_sft(tts_text=text, spk_id=speaker))

set_resource_limits()
result = monitored_inference("你好世界", "zh-CN-female-1")
```

## 配置問題

### 配置檔案錯誤

**YAML 解析錯誤:**
```bash
# 錯誤: yaml.constructor.ConstructorError: could not determine a constructor
# 原因: YAML 文件中使用了不支持的標籤

# 檢查配置文件
# 常見問題標籤: !new, !ref, !apply
# 確保使用正確的 YAML 解析器

pip install omegaconf==2.3.0  # 使用相容的版本
```

**參數不匹配錯誤:**
```python
# 錯誤: KeyError: 'text_token_size' 或類似錯誤
# 原因: 不同模型版本的配置參數不同

# CosyVoice-300M 和 CosyVoice-300M-25Hz 的區別
COSYVOICE_300M_CONFIG = {
    'text_token_size': 51866,
    'input_frame_rate': 50
}

COSYVOICE_300M_25HZ_CONFIG = {
    'text_token_size': 60515,  # 不同的詞彙表大小
    'input_frame_rate': 25     # 不同的幀率
}

# 使用正確的配置
if '25Hz' in model_path:
    config.update(COSYVOICE_300M_25HZ_CONFIG)
else:
    config.update(COSYVOICE_300M_CONFIG)
```

**路徑配置錯誤:**
```python
# 錯誤: FileNotFoundError: pretrained_models/...
# 解決方案: 檢查和修正路徑配置

import os
from pathlib import Path

def validate_model_paths(model_dir):
    model_path = Path(model_dir)
    
    required_files = {
        'config': model_path / 'cosyvoice.yaml',
        'llm': model_path / 'llm.pt',
        'flow': model_path / 'flow.pt', 
        'hifigan': model_path / 'hift.pt'
    }
    
    missing_files = []
    for name, file_path in required_files.items():
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"缺少文件: {missing_files}")
        return False
    
    return True

# 使用驗證
if validate_model_paths('pretrained_models/CosyVoice-300M'):
    model = CosyVoice('pretrained_models/CosyVoice-300M')
else:
    print("請重新下載模型")
```

### 參數調優

**語音品質調優:**
```python
# 高品質設定
high_quality_params = {
    # 採樣參數
    'top_p': 0.7,           # 降低隨機性
    'top_k': 20,            # 減少候選
    'temperature': 0.8,      # 降低溫度
    
    # RAS 參數
    'ras_window_size': 15,   # 增大窗口
    'ras_tau': 0.05,        # 更強的重複抑制
    
    # Flow 參數  
    'flow_steps': 32,       # 增加推理步數
    'cfg_scale': 0.9,       # 增強引導信號
    'solver': 'heun'        # 更精確的求解器
}

# 高速度設定
fast_params = {
    'top_p': 0.9,
    'top_k': 15,
    'temperature': 0.9,
    'flow_steps': 16,       # 減少推理步數
    'cfg_scale': 0.5,       # 降低引導強度
    'solver': 'euler'       # 更快的求解器
}

# 平衡設定
balanced_params = {
    'top_p': 0.8,
    'top_k': 25,
    'temperature': 0.85,
    'flow_steps': 22,       # 默認值
    'cfg_scale': 0.7,
    'solver': 'euler'
}

# 應用參數
def apply_quality_preset(model, preset='balanced'):
    presets = {
        'high_quality': high_quality_params,
        'fast': fast_params,
        'balanced': balanced_params
    }
    
    params = presets.get(preset, balanced_params)
    model.update_generation_config(params)
    print(f"已應用 {preset} 預設")
    return model
```

**不同語言的調優:**
```python
# 中文優化參數
CHINESE_OPTIMIZED = {
    'temperature': 0.8,      # 中文適中的溫度
    'top_p': 0.7,
    'ras_window_size': 12,   # 中文字元特性
    'flow_steps': 25
}

# 英文優化參數
ENGLISH_OPTIMIZED = {
    'temperature': 0.9,      # 英文可以略高
    'top_p': 0.8,
    'ras_window_size': 10,   # 英文單詞特性
    'flow_steps': 22
}

# 日文優化參數
JAPANESE_OPTIMIZED = {
    'temperature': 0.75,     # 日文需要更穩定
    'top_p': 0.65,
    'ras_window_size': 8,    # 日文音素特性
    'flow_steps': 28
}

def get_language_params(text):
    import re
    
    # 簡單語言檢測
    if re.search(r'[\u4e00-\u9fff]', text):  # 中文
        return CHINESE_OPTIMIZED
    elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):  # 日文
        return JAPANESE_OPTIMIZED
    else:  # 英文或其他
        return ENGLISH_OPTIMIZED
```

**動態參數調節:**
```python
class AdaptiveParameterTuner:
    def __init__(self):
        self.performance_history = []
        self.quality_history = []
    
    def evaluate_output_quality(self, audio_tensor):
        """ 簡單的音頻品質評估 """
        # 計算信噪比
        signal_power = torch.mean(audio_tensor ** 2)
        
        # 檢測是否有失真或異常
        clipping_ratio = torch.mean((torch.abs(audio_tensor) > 0.95).float())
        
        # 簡單評分 (0-1)
        quality_score = signal_power.item() * (1 - clipping_ratio.item())
        return min(1.0, max(0.0, quality_score))
    
    def adaptive_tune(self, current_params, quality_score, inference_time):
        """ 根據品質和速度調節參數 """
        new_params = current_params.copy()
        
        # 如果品質低，提高品質參數
        if quality_score < 0.7:
            new_params['flow_steps'] = min(40, new_params['flow_steps'] + 2)
            new_params['cfg_scale'] = min(1.0, new_params['cfg_scale'] + 0.1)
            new_params['temperature'] = max(0.6, new_params['temperature'] - 0.1)
        
        # 如果速度太慢，降低計算量
        elif inference_time > 10.0:  # 超過 10 秒
            new_params['flow_steps'] = max(12, new_params['flow_steps'] - 2)
            new_params['cfg_scale'] = max(0.3, new_params['cfg_scale'] - 0.1)
        
        return new_params

# 使用自適應調節
tuner = AdaptiveParameterTuner()
current_params = balanced_params.copy()

for text in test_texts:
    start_time = time.time()
    
    result = model.inference_sft(tts_text=text, spk_id=speaker, **current_params)
    
    inference_time = time.time() - start_time
    quality_score = tuner.evaluate_output_quality(result[0]['tts_speech'])
    
    # 調節下次推理的參數
    current_params = tuner.adaptive_tune(current_params, quality_score, inference_time)
    
    print(f"品質: {quality_score:.3f}, 時間: {inference_time:.2f}s")
```

## 多語言支援問題

### 特定語言問題

**中文方言問題:**
```python
# 常見方言支援
DIALECT_MAPPING = {
    '粤語': 'zh-yue',
    '閩南語': 'zh-min',
    '客家語': 'zh-hak',
    '吳語': 'zh-wuu',
    '湘語': 'zh-xiang'
}

# 方言文本預處理
def preprocess_dialect_text(text, dialect):
    if dialect == 'zh-yue':  # 粤語
        # 粤語常用詞彙正規化
        text = text.replace('嘅', '的')
        text = text.replace('係', '是')
        text = text.replace('咁', '了')
    elif dialect == 'zh-min':  # 閩南語
        # 閩南語特殊處理
        text = text.replace('在遐', '那裡')
        text = text.replace('佇', '他')
    
    return text

# 使用方言模式
def dialect_inference(text, dialect_code):
    processed_text = preprocess_dialect_text(text, dialect_code)
    
    # 選擇對應的語音模型
    speaker_mapping = {
        'zh-yue': 'zh-HK-female-1',  # 港式粤語
        'zh-min': 'zh-TW-female-1',  # 台灣閩南語
        'zh-CN': 'zh-CN-female-1'    # 標準中文
    }
    
    speaker = speaker_mapping.get(dialect_code, 'zh-CN-female-1')
    
    return cosyvoice.inference_sft(
        tts_text=processed_text,
        spk_id=speaker
    )
```

**英文發音問題:**
```python
# 英文音素轉換
from g2p_en import G2p
import re

g2p = G2p()

def improve_english_pronunciation(text):
    # 常見發音問題修正
    corrections = {
        'often': 'ofen',      # 静音 t
        'listen': 'lisen',    # 静音 t
        'castle': 'casl',     # 静音 t
        'debris': 'debree',   # 法語外來詞
    }
    
    for word, phonetic in corrections.items():
        text = re.sub(rf'\b{word}\b', phonetic, text, flags=re.IGNORECASE)
    
    # 數字讀法
    number_words = {
        '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '0': 'zero'
    }
    
    for digit, word in number_words.items():
        text = text.replace(digit, word)
    
    return text

# 音素標記輔助
def add_phoneme_hints(text):
    """ 為困難單詞添加音素提示 """
    difficult_words = {
        'through': 'th-r-oo',
        'thought': 'th-aw-t', 
        'rough': 'r-uh-f',
        'cough': 'k-aw-f'
    }
    
    for word, phoneme in difficult_words.items():
        pattern = rf'\b{word}\b'
        replacement = f'{word}[{phoneme}]'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text
```

**日文特殊處理:**
```python
# 日文假名和漢字處理
def process_japanese_text(text):
    import jaconv
    
    # 全角/半角轉換
    text = jaconv.z2h(text, kana=False, ascii=True, digit=True)
    
    # 片假名轉平假名（可選）
    # text = jaconv.kata2hira(text)
    
    # 數字讀法轉換
    japanese_numbers = {
        '0': 'ざろ', '1': 'いち', '2': 'に', '3': 'さん', '4': 'よん',
        '5': 'ご', '6': 'ろく', '7': 'なな', '8': 'はち', '9': 'きゅう'
    }
    
    for digit, reading in japanese_numbers.items():
        text = text.replace(digit, reading)
    
    return text

# 日文重音記號處理
def handle_japanese_accent(text):
    """ 處理日文重音記號 """
    # 常見重音模式
    accent_patterns = {
        'ありがとう': 'ありがとう。',  # 感謝
        'こんにちは': 'こんにちは。',  # 問候
        'さようなら': 'さようなら。'   # 告別
    }
    
    for original, accented in accent_patterns.items():
        text = text.replace(original, accented)
    
    return text
```

**多語言混合問題:**
```python
# 語言檢測和分段
import langdetect
from langdetect import detect_langs

def segment_multilingual_text(text):
    """ 將混合語言文本分段 """
    import re
    
    # 使用正則表達式分離不同語言
    segments = []
    
    # 中文區段
    chinese_pattern = r'[\u4e00-\u9fff\uff0c\u3002\uff01\uff1f]+'
    # 英文區段  
    english_pattern = r'[a-zA-Z\s\.,!?]+'
    # 日文區段
    japanese_pattern = r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3002\uff01\uff1f]+'
    
    # 找出所有匹配區段
    for match in re.finditer(chinese_pattern, text):
        segments.append(('zh', match.group(), match.span()))
    
    for match in re.finditer(english_pattern, text):
        # 檢查是否與中文重疊
        if not any(match.start() >= s[2][0] and match.end() <= s[2][1] for s in segments):
            segments.append(('en', match.group(), match.span()))
    
    return sorted(segments, key=lambda x: x[2][0])

def process_mixed_language_text(text):
    """ 處理混合語言文本 """
    segments = segment_multilingual_text(text)
    processed_segments = []
    
    for lang, content, span in segments:
        if lang == 'zh':
            # 中文處理
            processed = preprocess_chinese_text(content)
            processed_segments.append(f'<zh>{processed}</zh>')
        elif lang == 'en':
            # 英文處理
            processed = improve_english_pronunciation(content)
            processed_segments.append(f'<en>{processed}</en>')
        else:
            processed_segments.append(content)
    
    return ' '.join(processed_segments)

# 使用示例
mixed_text = "你好，我叫John Smith，nice to meet you。"
processed_text = process_mixed_language_text(mixed_text)
print(processed_text)  # <zh>你好，我叫</zh> <en>John Smith, nice to meet you</en> <zh>。</zh>
```

### 跨語言合成問題

**語言模型選擇:**
```python
# 跨語言模型選擇策略
CROSS_LINGUAL_MODELS = {
    'zh->en': {
        'model': 'pretrained_models/CosyVoice-300M',
        'speaker': 'en-US-female-1',
        'params': {
            'top_p': 0.8,
            'temperature': 0.9,
            'flow_steps': 25
        }
    },
    'en->zh': {
        'model': 'pretrained_models/CosyVoice-300M',
        'speaker': 'zh-CN-female-1', 
        'params': {
            'top_p': 0.7,
            'temperature': 0.8,
            'flow_steps': 28
        }
    },
    'zh->ja': {
        'model': 'pretrained_models/CosyVoice-300M',
        'speaker': 'ja-JP-female-1',
        'params': {
            'top_p': 0.65,
            'temperature': 0.75,
            'flow_steps': 30
        }
    }
}

def cross_lingual_synthesis(text, source_lang, target_lang, reference_audio=None):
    """ 跨語言語音合成 """
    model_key = f'{source_lang}->{target_lang}'
    
    if model_key not in CROSS_LINGUAL_MODELS:
        raise ValueError(f"不支援的語言組合: {model_key}")
    
    config = CROSS_LINGUAL_MODELS[model_key]
    model = CosyVoice(config['model'])
    
    if reference_audio is not None:
        # 使用參考音頻的零樣本模式
        result = model.inference_cross_lingual(
            tts_text=text,
            prompt_speech_16k=reference_audio,
            **config['params']
        )
    else:
        # 使用預設語音
        result = model.inference_sft(
            tts_text=text,
            spk_id=config['speaker'],
            **config['params']
        )
    
    return result
```

**參考音頻預處理:**
```python
# 跨語言參考音頻預處理
def preprocess_cross_lingual_reference(reference_path, target_lang):
    """ 針對跨語言合成預處理參考音頻 """
    import torchaudio
    from torchaudio.transforms import Resample, Vol
    
    # 讀取音頻
    waveform, sr = torchaudio.load(reference_path)
    
    # 重採樣到 16kHz
    if sr != 16000:
        resampler = Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # 音量正規化
    waveform = waveform / waveform.abs().max() * 0.8
    
    # 根據目標語言調節音頻長度
    target_length = {
        'zh': 16000 * 3,  # 中文 3 秒
        'en': 16000 * 4,  # 英文 4 秒  
        'ja': 16000 * 2.5 # 日文 2.5 秒
    }
    
    length = target_length.get(target_lang, 16000 * 3)
    
    if waveform.shape[1] > length:
        # 截取中間部分
        start_idx = (waveform.shape[1] - length) // 2
        waveform = waveform[:, start_idx:start_idx + length]
    elif waveform.shape[1] < length:
        # 填充到目標長度
        pad_length = length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    
    return waveform

# 使用示例
reference_audio = preprocess_cross_lingual_reference(
    'reference_speaker.wav', 
    target_lang='en'
)

result = cross_lingual_synthesis(
    text="Hello, how are you?",
    source_lang='zh',
    target_lang='en', 
    reference_audio=reference_audio
)
```

**語音風格保持:**
```python
# 語音風格特徵提取
def extract_speaker_features(reference_audio, source_lang):
    """ 提取語音風格特徵 """
    # 使用語音編碼器提取特徵
    with torch.no_grad():
        # 這裡需要使用語音編碼器的實際實現
        speaker_embedding = model.extract_speaker_embedding(reference_audio)
    
    return speaker_embedding

def maintain_speaker_consistency(text, reference_audio, source_lang, target_lang):
    """ 保持語音風格一致性 """
    # 提取語音特徵
    speaker_features = extract_speaker_features(reference_audio, source_lang)
    
    # 調節參數以保持風格
    consistency_params = {
        'speaker_embedding': speaker_features,
        'style_weight': 0.8,  # 風格權重
        'content_weight': 0.6  # 內容權重
    }
    
    result = model.inference_cross_lingual(
        tts_text=text,
        prompt_speech_16k=reference_audio,
        **consistency_params
    )
    
    return result

# 語音品質驗證
def validate_cross_lingual_quality(generated_audio, target_lang):
    """ 驗證跨語言合成品質 """
    quality_metrics = {}
    
    # 計算基本指標
    audio_np = generated_audio.numpy().flatten()
    
    # 1. 音量穩定性
    volume_std = np.std(audio_np)
    quality_metrics['volume_stability'] = 1.0 / (1.0 + volume_std)
    
    # 2. 頻譜平衡性
    from scipy import signal
    f, psd = signal.welch(audio_np, fs=22050)
    spectral_balance = np.mean(psd[100:1000]) / np.mean(psd[1000:5000])
    quality_metrics['spectral_balance'] = min(1.0, spectral_balance)
    
    # 3. 語言特性檢測(簡化版)
    if target_lang == 'en':
        # 英文特徵: 更多高頻成分
        high_freq_energy = np.mean(psd[2000:8000])
        quality_metrics['language_match'] = min(1.0, high_freq_energy * 10)
    elif target_lang == 'zh':
        # 中文特徵: 聲調變化
        pitch_variation = np.std(audio_np[::100])  # 簡化的聲調變化
        quality_metrics['language_match'] = min(1.0, pitch_variation * 5)
    
    # 總體評分
    overall_quality = np.mean(list(quality_metrics.values()))
    quality_metrics['overall'] = overall_quality
    
    return quality_metrics
```

## 整合問題

### API 整合問題

**RESTful API 整合:**
```python
# 常見的 HTTP 錯誤處理
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class TTSAPIClient:
    def __init__(self, base_url, timeout=30, max_retries=3):
        self.base_url = base_url
        self.session = requests.Session()
        
        # 設置重試策略
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.timeout = timeout
    
    def synthesize_speech(self, text, speaker_id, **kwargs):
        """ TTS API 請求封裝 """
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/tts/sft",
                data={
                    'text': text,
                    'speaker': speaker_id,
                    **kwargs
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.content  # 音頻数據
            
        except requests.exceptions.Timeout:
            raise Exception("請求超時，請檢查服務器狀態")
        except requests.exceptions.ConnectionError:
            raise Exception("網絡連接錯誤，請檢查網絡設定")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                raise Exception("請求參數錯誤")
            elif e.response.status_code == 429:
                raise Exception("請求頻率過高，請稍後再試")
            elif e.response.status_code == 500:
                raise Exception("服務器內部錯誤")
            else:
                raise Exception(f"HTTP 錯誤: {e.response.status_code}")
    
    def health_check(self):
        """ 健康檢查 """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

# 使用示例
api_client = TTSAPIClient("http://localhost:7860")

if api_client.health_check():
    audio_data = api_client.synthesize_speech(
        text="你好世界",
        speaker_id="zh-CN-female-1"
    )
    with open("output.wav", "wb") as f:
        f.write(audio_data)
else:
    print("TTS 服務不可用")
```

**WebSocket 整合:**
```python
# 即時語音合成 WebSocket 客戶端
import asyncio
import websockets
import json
import base64

class WebSocketTTSClient:
    def __init__(self, uri):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        """ 連接 WebSocket 服務 """
        try:
            self.websocket = await websockets.connect(self.uri)
            print("WebSocket 連接成功")
        except Exception as e:
            print(f"WebSocket 連接失敗: {e}")
            raise
    
    async def synthesize_streaming(self, text, speaker_id):
        """ 流式語音合成 """
        if not self.websocket:
            await self.connect()
        
        # 發送請求
        request = {
            "action": "synthesize",
            "text": text,
            "speaker_id": speaker_id,
            "stream": True
        }
        
        await self.websocket.send(json.dumps(request))
        
        # 接收流式数據
        audio_chunks = []
        
        async for message in self.websocket:
            data = json.loads(message)
            
            if data["type"] == "audio_chunk":
                # 解碼音頻数據
                audio_chunk = base64.b64decode(data["audio"])
                audio_chunks.append(audio_chunk)
                yield audio_chunk
            
            elif data["type"] == "completed":
                print("合成完成")
                break
            
            elif data["type"] == "error":
                raise Exception(f"TTS 錯誤: {data['message']}")
    
    async def close(self):
        """ 關閉連接 """
        if self.websocket:
            await self.websocket.close()

# 使用示例
async def streaming_tts_example():
    client = WebSocketTTSClient("ws://localhost:8080/ws/tts")
    
    try:
        audio_data = b''
        async for chunk in client.synthesize_streaming(
            text="這是一個流式語音合成的測試",
            speaker_id="zh-CN-female-1"
        ):
            audio_data += chunk
            # 即時播放或處理音頻塊
        
        # 保存完整音頻
        with open("streaming_output.wav", "wb") as f:
            f.write(audio_data)
    
    finally:
        await client.close()

# 運行示例
# asyncio.run(streaming_tts_example())
```

**gRPC 客戶端整合:**
```python
# gRPC 客戶端實現
import grpc
import tts_pb2
import tts_pb2_grpc
from concurrent import futures

class GRPCTTSClient:
    def __init__(self, server_address):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = tts_pb2_grpc.TTSServiceStub(self.channel)
    
    def synthesize_zero_shot(self, text, prompt_text, prompt_audio_path):
        """ 零樣本語音合成 """
        try:
            # 讀取參考音頻
            with open(prompt_audio_path, 'rb') as f:
                prompt_audio_data = f.read()
            
            # 創建請求
            request = tts_pb2.ZeroShotRequest(
                text=text,
                prompt_text=prompt_text,
                prompt_audio=prompt_audio_data
            )
            
            # 發送請求
            response = self.stub.ZeroShotTTS(request)
            
            return response.audio_data
        
        except grpc.RpcError as e:
            print(f"gRPC 錯誤: {e.code()} - {e.details()}")
            
            # 根據錯誤類型處理
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise Exception("服動不可用，請檢查服務器狀態")
            elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise Exception("請求超時")
            elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise Exception("請求參數無效")
            else:
                raise Exception(f"gRPC 錯誤: {e.details()}")
    
    def synthesize_streaming(self, text, speaker_id, chunk_size=1024):
        """ 流式語音合成 """
        try:
            request = tts_pb2.StreamingRequest(
                text=text,
                speaker_id=speaker_id,
                chunk_size=chunk_size
            )
            
            # 流式請求
            for chunk in self.stub.StreamingTTS(request):
                yield chunk.chunk_data
                if chunk.is_last:
                    break
        
        except grpc.RpcError as e:
            print(f"流式 gRPC 錯誤: {e.details()}")
            raise
    
    def close(self):
        """ 關閉連接 """
        self.channel.close()

# 使用示例
client = GRPCTTSClient('localhost:50051')

try:
    # 零樣本合成
    audio_data = client.synthesize_zero_shot(
        text="你好，我是人工智能助手",
        prompt_text="歡迎使用我的服務",
        prompt_audio_path="reference.wav"
    )
    
    with open("grpc_output.wav", "wb") as f:
        f.write(audio_data)
    
    # 流式合成
    streaming_audio = b''
    for chunk in client.synthesize_streaming(
        text="流式語音合成測試",
        speaker_id="zh-CN-female-1"
    ):
        streaming_audio += chunk
    
    with open("grpc_streaming.wav", "wb") as f:
        f.write(streaming_audio)

finally:
    client.close()
```

### 部署環境問題

**Docker 容器問題:**
```dockerfile
# 常見 Docker 問題與解決方案

# 1. GPU 不可用問題
# 確保使用 nvidia-docker 和正確的 runtime
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 設置 NVIDIA runtime
# docker run --gpus all --runtime=nvidia

# 2. 記憶體不足
# 增加共享記憶體
# docker run --shm-size=8g

# 3. 模型文件下載問題
RUN apt-get update && apt-get install -y git-lfs

# 使用 multi-stage build 減少容器大小
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as runtime
COPY --from=builder /opt/cosyvoice /opt/cosyvoice
```

```bash
# Docker 部署排錯指令

# 檢查 GPU 支援
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi

# 檢查容器資源
docker stats cosyvoice-container

# 查看容器日誌
docker logs -f cosyvoice-container

# 進入容器調試
docker exec -it cosyvoice-container /bin/bash
```

**Kubernetes 部署問題:**
```yaml
# K8s 常見問題解決

# 1. GPU 節點選擇器
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100  # GPU 節點
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      
      containers:
      - name: cosyvoice
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
        
        # 2. 持久化儲存配置
        volumeMounts:
        - name: model-storage
          mountPath: /app/pretrained_models
        
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: cosyvoice-models-pvc

---
# PVC 配置
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: cosyvoice-models-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
```

**雲端部署問題:**
```python
# AWS ECS/Fargate 部署問題

# 1. GPU 支援限制
# Fargate 不支援 GPU，需要使用 EC2 啟動類型

# 2. 內存限制
# 調節 ECS task definition
task_definition = {
    "family": "cosyvoice-task",
    "memory": "16384",  # 16GB
    "cpu": "4096",     # 4 vCPU
    "requiresCompatibilities": ["EC2"],
    "placementConstraints": [
        {
            "type": "memberOf",
            "expression": "attribute:ecs.instance-type =~ p3.*"  # GPU 實例
        }
    ]
}

# 3. 模型文件傳輸
# 使用 EFS 或 S3 儲存模型
volumes = [
    {
        "name": "cosyvoice-models",
        "efsVolumeConfiguration": {
            "fileSystemId": "fs-xxxxxxxxx",
            "rootDirectory": "/models"
        }
    }
]

# Google Cloud Run GPU 支援
# gcloud run deploy cosyvoice \
#   --image gcr.io/project/cosyvoice \
#   --gpu 1 \
#   --gpu-type nvidia-tesla-t4 \
#   --memory 8Gi \
#   --cpu 2 \
#   --max-instances 5
```

**環境變數配置:**
```bash
# 產環設定清單

# 基本配置
export COSYVOICE_MODEL_DIR="/app/models"
export COSYVOICE_CACHE_DIR="/tmp/cosyvoice_cache"
export COSYVOICE_LOG_LEVEL="INFO"
export COSYVOICE_DEVICE="cuda"

# GPU 配置
export CUDA_VISIBLE_DEVICES="0"
export NVIDIA_VISIBLE_DEVICES="all"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility"

# 效能優化
export COSYVOICE_ENABLE_JIT="true"
export COSYVOICE_ENABLE_TRT="true"
export COSYVOICE_FP16="true"
export COSYVOICE_BATCH_SIZE="4"

# Python 優化
export PYTHONUNBUFFERED="1"
export PYTHONDONTWRITEBYTECODE="1"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"

# 網絡配置
export HTTP_PROXY="http://proxy.company.com:8080"
export HTTPS_PROXY="http://proxy.company.com:8080"
export NO_PROXY="localhost,127.0.0.1,.local"

# 健康檢查腳本
#!/bin/bash
# health_check.sh
set -e

# 檢查服務可用性
curl -f http://localhost:7860/health || exit 1

# 檢查 GPU 狀態
if [ "$COSYVOICE_DEVICE" = "cuda" ]; then
    nvidia-smi > /dev/null || exit 1
fi

# 檢查磁碟空間
DISK_USAGE=$(df /tmp | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "警告: 磁碟空間不足"
    exit 1
fi

echo "健康檢查通過"
```

**網絡和安全性問題:**
```python
# HTTPS 和 SSL 配置
from fastapi import FastAPI, HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
import ssl

app = FastAPI()

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # 限制來源
    allow_credentials=True,
    allow_methods=["POST"],  # 只允許 POST
    allow_headers=["*"],
)

# HTTPS 重定向
app.add_middleware(HTTPSRedirectMiddleware)

# SSL 證書配置
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('/path/to/cert.pem', '/path/to/key.pem')

# 啟動 HTTPS 服務
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=443,
        ssl_context=ssl_context
    )
```

## 常見錯誤碼

**CosyVoice 常見錯誤碼:**

| 錯誤碼 | 描述 | 解決方案 |
|---------|------|----------|
| **COSYVOICE_001** | 模型文件遺失 | 重新下載模型文件 |
| **COSYVOICE_002** | CUDA 記憶體不足 | 減小批處理大小或使用 FP16 |
| **COSYVOICE_003** | 配置文件解析錯誤 | 檢查 YAML 格式和參數 |
| **COSYVOICE_004** | 語音編碼器初始化失敗 | 檢查分詞器配置 |
| **COSYVOICE_005** | Flow 模型載入錯誤 | 檢查模型版本相容性 |
| **COSYVOICE_006** | HiFi-GAN 生成器錯誤 | 重新下載 HiFi-GAN 模型 |
| **COSYVOICE_007** | 音頻重採樣錯誤 | 檢查輸入音頻格式 |
| **COSYVOICE_008** | 跨語言合成失敗 | 檢查語言模型支援 |

**系統級錯誤碼:**

| 錯誤碼 | 描述 | 解決方案 |
|---------|------|----------|
| **SYS_001** | CUDA 驅動不相容 | 更新 NVIDIA 驅動程式 |
| **SYS_002** | Python 版本不支援 | 使用 Python 3.8+ |
| **SYS_003** | PyTorch 版本衝突 | 安裝相容版本 PyTorch |
| **SYS_004** | 系統記憶體不足 | 增加物理記憶體或虛擬記憶體 |
| **SYS_005** | 磁碟空間不足 | 清理臨時文件和緩存 |

**API 錯誤碼:**

| HTTP 狀態碼 | 描述 | 解決方案 |
|-------------|------|----------|
| **400** | 請求參數錯誤 | 檢查輸入參數格式 |
| **413** | 請求實體過大 | 縮短文本或減小音頻文件 |
| **429** | 請求頻率過高 | 實施請求限制或加大服務器資源 |
| **500** | 服務器內部錯誤 | 檢查服務器日誌和資源 |
| **503** | 服務不可用 | 檢查服務健康狀態 |

**排錯指令大全:**
```bash
# 1. 檢查系統資源
free -h                    # 記憶體使用
df -h                      # 磁碟空間
nvidia-smi                 # GPU 狀態
top -p $(pgrep python)     # Python 進程資源

# 2. 檢查 Python 環境
python --version
pip list | grep torch
python -c "import torch; print(torch.cuda.is_available())"

# 3. 檢查模型文件
ls -la pretrained_models/CosyVoice-300M/
md5sum pretrained_models/CosyVoice-300M/*.pt

# 4. 檢查日誌
tail -f /var/log/cosyvoice.log
journalctl -u cosyvoice-service -f

# 5. 網絡連接測試
curl -X POST http://localhost:7860/api/v1/health
telnet localhost 7860

# 6. GPU 資源監控
watch -n 1 nvidia-smi
nvtop  # 如果已安裝
```

## 取得支援

### 社群支援

**官方資源:**
- **GitHub 倉庫**: https://github.com/FunAudioLLM/CosyVoice
- **技術論文**: https://arxiv.org/abs/2407.05407
- **模型下載**: https://modelscope.cn/models/iic/CosyVoice-300M
- **在線體驗**: https://modelscope.cn/studios/iic/CosyVoice-300M

**常用支援渠道:**
1. **GitHub Issues**: 用於回報 Bug 和功能請求
2. **GitHub Discussions**: 用於技術討論和經驗分享
3. **ModelScope 社區**: 中文支援和討論
4. **技術部落格**: 相關技術文章和教程

**提問技巧:**
- 提供完整的錯誤訊息和堆棧追蹤
- 描述重現步驟和環境信息
- 包含配置文件和模型版本信息
- 使用中英文描述問題，方便更多人幫助

### 問題回報

**回報準備清單:**

1. **環境信息**
   ```bash
   # 收集系統信息
   uname -a                  # 作業系統
   python --version          # Python 版本
   pip list | grep torch     # PyTorch 版本
   nvidia-smi                # GPU 信息
   ```

2. **錯誤訊息**
   - 完整的錯誤堆棧追蹤
   - 錯誤發生的歸切時間和頻率
   - 相關的警告訊息

3. **重現步驟**
   ```python
   # 提供最小化的重現代碼
   from cosyvoice.cli.cosyvoice import CosyVoice
   
   model = CosyVoice('pretrained_models/CosyVoice-300M')
   
   # 這裡會發生錯誤
   result = model.inference_sft(
       tts_text="測試文本",
       spk_id="zh-CN-female-1"
   )
   ```

4. **配置信息**
   - 模型版本和來源
   - 使用的配置文件
   - 特殊的參數設定

**問題回報模板:**

```markdown
## Bug Report

### Environment
- OS: Ubuntu 20.04
- Python: 3.8.10
- PyTorch: 2.1.0+cu118
- CUDA: 11.8
- CosyVoice Version: latest
- Model: CosyVoice-300M

### Problem Description
[簡潔描述問題]

### Steps to Reproduce
1. 加載模型: CosyVoice('pretrained_models/CosyVoice-300M')
2. 執行推理: model.inference_sft(...)
3. 出現錯誤

### Error Message
```
[貼上完整的錯誤堆棧追蹤]
```

### Expected Behavior
[描述期望的正常行為]

### Additional Context
[任何其他相關信息]

### Configuration Files
```yaml
# cosyvoice.yaml 相關配置
```

### System Resources
- GPU Memory: 12GB
- RAM: 32GB
- Available Disk Space: 100GB
```

**回報渠道選擇:**

- **Bug**: GitHub Issues
- **功能請求**: GitHub Discussions
- **使用問題**: GitHub Discussions 或 ModelScope 社區
- **文檔問題**: GitHub Issues
- **性能問題**: GitHub Issues 並標記 "performance" 標籤