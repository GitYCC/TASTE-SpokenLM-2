# CosyVoice 使用範例

本文档提供了 CosyVoice 的详细使用示例，涵盖基础用法、高级配置和实际应用场景。

## 基本使用

### 环境初始化

所有示例都需要先进行环境初始化：

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
import torchaudio
import torch
```

### CosyVoice2 范例（推荐）

#### Zero-shot 语音克隆

```python
# 加载 CosyVoice2 模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', 
                      load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# 加载 prompt 音频
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 设置随机种子以确保结果可复现
set_all_random_seed(42)

# Zero-shot 推理
tts_text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
prompt_text = "希望你以后能够做的比我还好呦。"

# 非流式推理
for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text, prompt_text, prompt_speech_16k, stream=False)):
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
    print(f"Generated audio saved as zero_shot_{i}.wav")

# 保存 zero_shot speaker 用于后续使用
if cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, 'my_zero_shot_spk'):
    print("Zero-shot speaker added successfully")
    
    # 使用保存的 speaker
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        tts_text, '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
        torchaudio.save(f'zero_shot_cached_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
    
    # 保存 speaker 信息到文件
    cosyvoice.save_spkinfo()
```

#### 跨语言合成

```python
# 使用 CosyVoice2 进行跨语言合成
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

# 中文音频，英文合成
chinese_prompt = load_wav('./asset/zero_shot_prompt.wav', 16000)
english_text = "Hello, this is a cross-lingual synthesis example using Chinese voice."

for i, j in enumerate(cosyvoice.inference_cross_lingual(
    english_text, chinese_prompt, stream=False)):
    torchaudio.save(f'cross_lingual_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
    print(f"Cross-lingual audio saved as cross_lingual_{i}.wav")

# 支持语言标记的跨语言合成
mixed_text = "<|zh|>你好<|en|>Hello<|zh|>世界<|en|>World"
for i, j in enumerate(cosyvoice.inference_cross_lingual(
    mixed_text, chinese_prompt, stream=False)):
    torchaudio.save(f'multilingual_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### 指令式合成

```python
# 使用自然语言控制语音合成
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

tts_text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 使用不同的指令控制
instructions = [
    "用四川话说这句话",
    "用温柔的声音说",
    "用激动的语气说",
    "说得慢一点",
    "用低沉的声音说"
]

for idx, instruct_text in enumerate(instructions):
    for i, j in enumerate(cosyvoice.inference_instruct2(
        tts_text, instruct_text, prompt_speech_16k, stream=False)):
        filename = f'instruct_{idx}_{instruct_text[:4]}_{i}.wav'
        torchaudio.save(filename, j['tts_speech'], cosyvoice.sample_rate)
        print(f"Instructed audio saved as {filename}")
```

#### 串流推理

```python
# 流式推理示例
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 流式推理，适用于实时应用
audio_chunks = []
for i, j in enumerate(cosyvoice.inference_zero_shot(
    "这是一个流式推理的示例，音频会被分块生成和处理。",
    "希望你以后能够做的比我还好呦。",
    prompt_speech_16k, 
    stream=True  # 启用流式模式
)):
    # 实时处理每个音频块
    audio_chunk = j['tts_speech']
    audio_chunks.append(audio_chunk)
    print(f"Received audio chunk {i}, shape: {audio_chunk.shape}")
    
    # 在实际应用中，这里可以立即播放或传输音频块
    # 例如: play_audio(audio_chunk) 或 stream_to_client(audio_chunk)

# 合并所有音频块
full_audio = torch.cat(audio_chunks, dim=1)
torchaudio.save('streaming_result.wav', full_audio, cosyvoice.sample_rate)
```

#### 生成器输入示例

```python
# 使用生成器作为输入，适用于与 LLM 结合
def text_generator():
    """模拟 LLM 逐步生成文本"""
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 使用生成器进行推理
for i, j in enumerate(cosyvoice.inference_zero_shot(
    text_generator(),  # 传入生成器
    "希望你以后能够做的比我还好呦。", 
    prompt_speech_16k, 
    stream=False
)):
    torchaudio.save(f'generator_input_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### CosyVoice 1.0 范例

#### SFT 推理

```python
# 加载 SFT 模型
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')

# 查看可用的预训练音色
available_spks = cosyvoice.list_available_spks()
print("Available speakers:", available_spks)

# SFT 推理示例
tts_text = "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"

for spk in available_spks[:3]:  # 使用前3个音色
    for i, j in enumerate(cosyvoice.inference_sft(
        tts_text, spk, stream=False, speed=1.0)):
        filename = f'sft_{spk}_{i}.wav'
        torchaudio.save(filename, j['tts_speech'], cosyvoice.sample_rate)
        print(f"SFT audio with speaker {spk} saved as {filename}")

# 调整语速
for speed in [0.8, 1.0, 1.2, 1.5]:
    for i, j in enumerate(cosyvoice.inference_sft(
        tts_text, "中文女", stream=False, speed=speed)):
        filename = f'sft_speed_{speed}_{i}.wav'
        torchaudio.save(filename, j['tts_speech'], cosyvoice.sample_rate)
        print(f"SFT audio with speed {speed} saved as {filename}")
```

#### Zero-shot 和跨语言（CosyVoice 1.0）

```python
# 加载基础模型用于 zero-shot 和跨语言
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

# Zero-shot 推理
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
tts_text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
prompt_text = "希望你以后能够做的比我还好呦。"

for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text, prompt_text, prompt_speech_16k, stream=False)):
    torchaudio.save(f'v1_zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# 跨语言合成（需要不同语言的 prompt 和目标文本）
cross_prompt = load_wav('./asset/cross_lingual_prompt.wav', 16000)
english_text = '<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.'

for i, j in enumerate(cosyvoice.inference_cross_lingual(
    english_text, cross_prompt, stream=False)):
    torchaudio.save(f'v1_cross_lingual_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### 语音转换

```python
# 语音转换示例
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

# 加载源音频和目标音色 prompt
source_speech_16k = load_wav('./asset/cross_lingual_prompt.wav', 16000)
target_prompt_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 执行语音转换
for i, j in enumerate(cosyvoice.inference_vc(
    source_speech_16k, target_prompt_16k, stream=False)):
    torchaudio.save(f'voice_conversion_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
    print(f"Voice conversion result saved as voice_conversion_{i}.wav")
```

#### 指令式控制（CosyVoice 1.0）

```python
# 加载指令模型
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')

# 支持的控制标签: <laughter></laughter>, <strong></strong>, [laughter], [breath]
tts_text = "在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。"
spk_id = "中文男"
instruct_text = "Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness."

for i, j in enumerate(cosyvoice.inference_instruct(
    tts_text, spk_id, instruct_text, stream=False)):
    torchaudio.save(f'v1_instruct_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
    print(f"Instructed audio saved as v1_instruct_{i}.wav")

# 使用情感标签
emotion_text = "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。"
for i, j in enumerate(cosyvoice.inference_instruct(
    emotion_text, "中文女", "说话时带有愉快的情绪", stream=False)):
    torchaudio.save(f'emotion_control_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

## 进阶使用

### 自定义配置

#### 模型加载配置

```python
# 高级模型加载选项
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=True,     # 启用 JIT 编译加速
    load_trt=True,     # 启用 TensorRT 加速
    load_vllm=True,    # 启用 vLLM 加速
    fp16=True,         # 使用半精度浮点数
    trt_concurrent=4   # TensorRT 并发数
)

# 自定义采样配置
from cosyvoice.utils.common import set_all_random_seed

# 设置不同的随机种子获得不同结果
for seed in [42, 123, 456]:
    set_all_random_seed(seed)
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        "每次使用不同的种子会产生不同的语音效果。",
        "希望你以后能够做的比我还好呦。",
        prompt_speech_16k, 
        stream=False
    )):
        torchaudio.save(f'seed_{seed}_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

#### vLLM 加速配置

```python
# 使用 vLLM 加速（需要安装 vllm==v0.9.0）
import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

# 注册 vLLM 模型
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# 启用 vLLM 加速的模型加载
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B', 
    load_jit=True, 
    load_trt=True, 
    load_vllm=True, 
    fp16=True
)

# vLLM 模式下的批量推理
from tqdm import tqdm

prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i in tqdm(range(10)):
    set_all_random_seed(i)
    for _, _ in enumerate(cosyvoice.inference_zero_shot(
        '这是使用vLLM加速的批量推理示例。',
        '希望你以后能够做的比我还好呦。',
        prompt_speech_16k, 
        stream=False
    )):
        continue  # 仅计算推理时间
```

### 批次处理

```python
def batch_synthesis(texts, output_dir="batch_output"):
    """批量语音合成函数"""
    import os
    from pathlib import Path
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    
    results = []
    for idx, text in enumerate(texts):
        try:
            set_all_random_seed(42)  # 保持一致性
            for i, j in enumerate(cosyvoice.inference_zero_shot(
                text, "希望你以后能够做的比我还好呦。", prompt_speech_16k, stream=False)):
                
                output_path = os.path.join(output_dir, f'batch_{idx:03d}_{i}.wav')
                torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
                
                results.append({
                    'text': text,
                    'output_path': output_path,
                    'duration': j['tts_speech'].shape[1] / cosyvoice.sample_rate
                })
                
                print(f"Processed {idx+1}/{len(texts)}: {output_path}")
                
        except Exception as e:
            print(f"Error processing text {idx}: {e}")
            results.append({
                'text': text,
                'output_path': None,
                'error': str(e)
            })
    
    return results

# 使用示例
texts = [
    "这是第一段要合成的文本。",
    "这是第二段要合成的文本，内容稍有不同。",
    "第三段文本包含了更多的信息和细节。",
    "最后一段文本用于测试批量处理的效果。"
]

results = batch_synthesis(texts)

# 打印处理结果统计
successful = len([r for r in results if 'error' not in r])
print(f"Successfully processed {successful}/{len(texts)} texts")
```

### 性能优化

```python
# 性能优化技巧
import time
import torch

def benchmark_inference():
    """推理性能基准测试"""
    # 不同配置的性能测试
    configs = [
        {'load_jit': False, 'load_trt': False, 'fp16': False, 'name': 'baseline'},
        {'load_jit': True, 'load_trt': False, 'fp16': False, 'name': 'jit'},
        {'load_jit': True, 'load_trt': False, 'fp16': True, 'name': 'jit_fp16'},
        {'load_jit': True, 'load_trt': True, 'fp16': True, 'name': 'jit_trt_fp16'}
    ]
    
    test_text = "这是一个用于性能测试的示例文本。"
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    
    results = []
    
    for config in configs:
        print(f"Testing config: {config['name']}")
        
        # 加载模型
        start_time = time.time()
        cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', **{k: v for k, v in config.items() if k != 'name'})
        load_time = time.time() - start_time
        
        # 预热
        for _ in cosyvoice.inference_zero_shot(test_text, "测试", prompt_speech_16k, stream=False):
            break
        
        # 测试推理时间
        inference_times = []
        for _ in range(5):
            start_time = time.time()
            for _ in cosyvoice.inference_zero_shot(test_text, "测试", prompt_speech_16k, stream=False):
                break
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        results.append({
            'config': config['name'],
            'load_time': load_time,
            'avg_inference_time': avg_inference_time,
            'gpu_memory': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        })
        
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Avg inference time: {avg_inference_time:.2f}s")
        print(f"  GPU memory: {results[-1]['gpu_memory']:.1f} MB")
        
        # 清理内存
        del cosyvoice
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

# 运行基准测试
# benchmark_results = benchmark_inference()
```

## 整合范例

### Web 服务整合

#### FastAPI 客户端示例

```python
import requests
import io
import torchaudio
from pathlib import Path

class CosyVoiceClient:
    """CosyVoice FastAPI 客户端"""
    
    def __init__(self, server_url="http://localhost:50000"):
        self.server_url = server_url
    
    def inference_sft(self, tts_text, spk_id, output_path=None):
        """SFT 推理"""
        url = f"{self.server_url}/inference_sft"
        data = {'tts_text': tts_text, 'spk_id': spk_id}
        
        response = requests.post(url, data=data, stream=True)
        if response.status_code == 200:
            audio_data = b''.join(response.iter_content(chunk_size=16000))
            audio_tensor = torch.from_numpy(np.frombuffer(audio_data, dtype=np.int16)).unsqueeze(0)
            
            if output_path:
                torchaudio.save(output_path, audio_tensor.float() / 32768.0, 22050)
            
            return audio_tensor
        else:
            raise Exception(f"Request failed: {response.status_code}")
    
    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav_path, output_path=None):
        """Zero-shot 推理"""
        url = f"{self.server_url}/inference_zero_shot"
        
        data = {'tts_text': tts_text, 'prompt_text': prompt_text}
        files = {'prompt_wav': open(prompt_wav_path, 'rb')}
        
        response = requests.post(url, data=data, files=files, stream=True)
        if response.status_code == 200:
            audio_data = b''.join(response.iter_content(chunk_size=16000))
            audio_tensor = torch.from_numpy(np.frombuffer(audio_data, dtype=np.int16)).unsqueeze(0)
            
            if output_path:
                torchaudio.save(output_path, audio_tensor.float() / 32768.0, 22050)
            
            return audio_tensor
        else:
            raise Exception(f"Request failed: {response.status_code}")

# 使用示例
client = CosyVoiceClient("http://localhost:50000")

# SFT 推理
result = client.inference_sft(
    "你好，这是通过 Web 服务合成的语音。",
    "中文女",
    "web_sft_result.wav"
)

# Zero-shot 推理
result = client.inference_zero_shot(
    "这是通过 Web 服务进行的零样本语音克隆。",
    "希望你以后能够做的比我还好呦。",
    "./asset/zero_shot_prompt.wav",
    "web_zero_shot_result.wav"
)
```

#### gRPC 客户端示例

```python
import grpc
import numpy as np
import torch
import torchaudio

# 假设已生成 protobuf 文件
# import cosyvoice_pb2
# import cosyvoice_pb2_grpc

class CosyVoiceGRPCClient:
    """CosyVoice gRPC 客户端"""
    
    def __init__(self, server_address="localhost:50000"):
        self.channel = grpc.insecure_channel(server_address)
        # self.stub = cosyvoice_pb2_grpc.CosyVoiceStub(self.channel)
    
    def inference_zero_shot(self, tts_text, prompt_text, prompt_audio_path):
        """gRPC Zero-shot 推理"""
        # 读取 prompt 音频
        audio_data, sr = torchaudio.load(prompt_audio_path)
        if sr != 16000:
            audio_data = torchaudio.functional.resample(audio_data, sr, 16000)
        
        # 转换为 int16 格式
        audio_int16 = (audio_data.numpy().flatten() * 32768).astype(np.int16)
        
        # 构建请求
        # request = cosyvoice_pb2.Request()
        # request.zero_shot_request.tts_text = tts_text
        # request.zero_shot_request.prompt_text = prompt_text
        # request.zero_shot_request.prompt_audio = audio_int16.tobytes()
        
        # 发送请求并接收响应
        # responses = self.stub.Inference(request)
        
        audio_chunks = []
        # for response in responses:
        #     audio_chunk = np.frombuffer(response.tts_audio, dtype=np.int16)
        #     audio_chunks.append(audio_chunk)
        
        # 合并音频块
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            return torch.from_numpy(full_audio).unsqueeze(0).float() / 32768.0
        
        return None

# 使用示例（需要先启动 gRPC 服务器）
# client = CosyVoiceGRPCClient()
# result = client.inference_zero_shot(
#     "这是通过 gRPC 服务合成的语音。",
#     "希望你以后能够做的比我还好呦。",
#     "./asset/zero_shot_prompt.wav"
# )
```

### 实时应用

#### 实时语音合成服务

```python
import asyncio
import websockets
import json
import base64
import io

class RealtimeTTSServer:
    """实时 TTS WebSocket 服务器"""
    
    def __init__(self):
        self.cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
        self.prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    
    async def handle_client(self, websocket, path):
        """处理客户端连接"""
        print(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'synthesize':
                    text = data['text']
                    mode = data.get('mode', 'zero_shot')
                    
                    # 执行语音合成
                    audio_chunks = []
                    if mode == 'zero_shot':
                        for i, j in enumerate(self.cosyvoice.inference_zero_shot(
                            text, "希望你以后能够做的比我还好呦。", 
                            self.prompt_speech_16k, stream=True)):
                            
                            # 将音频编码为 base64
                            audio_bytes = (j['tts_speech'].numpy() * 32768).astype(np.int16).tobytes()
                            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                            
                            # 发送音频块
                            response = {
                                'type': 'audio_chunk',
                                'chunk_id': i,
                                'audio_data': audio_b64,
                                'sample_rate': self.cosyvoice.sample_rate
                            }
                            await websocket.send(json.dumps(response))
                    
                    # 发送完成信号
                    await websocket.send(json.dumps({'type': 'synthesis_complete'}))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"Error handling client: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))

# 启动实时服务器
# server = RealtimeTTSServer()
# start_server = websockets.serve(server.handle_client, "localhost", 8765)
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()
```

#### 实时客户端示例

```python
import asyncio
import websockets
import json
import base64
import numpy as np
import pyaudio

class RealtimeTTSClient:
    """实时 TTS 客户端"""
    
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.audio = pyaudio.PyAudio()
        self.stream = None
    
    def setup_audio_output(self, sample_rate=22050):
        """设置音频输出"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024
        )
    
    async def synthesize_and_play(self, text):
        """合成并播放语音"""
        async with websockets.connect(self.server_url) as websocket:
            # 发送合成请求
            request = {
                'type': 'synthesize',
                'text': text,
                'mode': 'zero_shot'
            }
            await websocket.send(json.dumps(request))
            
            # 接收并播放音频块
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'audio_chunk':
                    # 解码音频数据
                    audio_bytes = base64.b64decode(data['audio_data'])
                    
                    # 设置音频输出（首次）
                    if self.stream is None:
                        self.setup_audio_output(data['sample_rate'])
                    
                    # 播放音频块
                    self.stream.write(audio_bytes)
                    
                elif data['type'] == 'synthesis_complete':
                    print("语音合成完成")
                    break
                    
                elif data['type'] == 'error':
                    print(f"合成错误: {data['message']}")
                    break
    
    def cleanup(self):
        """清理资源"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

# 使用示例
# client = RealtimeTTSClient()
# asyncio.run(client.synthesize_and_play("这是实时语音合成的示例。"))
# client.cleanup()
```

## 常见使用场景

### 多语言内容生成

```python
def multilingual_content_generator():
    """多语言内容生成器"""
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
    
    # 多语言内容
    contents = [
        {"text": "欢迎来到多语言语音合成演示。", "lang": "zh", "prompt": "./asset/zero_shot_prompt.wav"},
        {"text": "Welcome to the multilingual speech synthesis demo.", "lang": "en", "prompt": "./asset/cross_lingual_prompt.wav"},
        {"text": "多言語音声合成デモへようこそ。", "lang": "ja", "prompt": "./asset/zero_shot_prompt.wav"},
        {"text": "다국어 음성 합성 데모에 오신 것을 환영합니다.", "lang": "ko", "prompt": "./asset/zero_shot_prompt.wav"},
    ]
    
    for idx, content in enumerate(contents):
        prompt_audio = load_wav(content["prompt"], 16000)
        
        # 跨语言合成
        for i, j in enumerate(cosyvoice.inference_cross_lingual(
            content["text"], prompt_audio, stream=False)):
            
            output_path = f'multilingual_{content["lang"]}_{idx}_{i}.wav'
            torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
            print(f"Generated {content['lang']} audio: {output_path}")

# 运行多语言生成
# multilingual_content_generator()
```

### 语音助理应用

```python
class VoiceAssistant:
    """语音助理示例"""
    
    def __init__(self):
        self.cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
        self.assistant_prompt = load_wav('./asset/zero_shot_prompt.wav', 16000)
        
        # 预定义回复模板
        self.responses = {
            'greeting': "您好！我是您的语音助理，请问有什么可以帮助您的吗？",
            'time': "现在时间是 {time}",
            'weather': "今天天气 {weather}，温度 {temperature} 度",
            'goodbye': "再见！祝您愉快！",
            'unknown': "抱歉，我没有理解您的意思，请您再说一遍。"
        }
    
    def generate_response(self, response_type, **kwargs):
        """生成语音回复"""
        if response_type in self.responses:
            text = self.responses[response_type].format(**kwargs)
        else:
            text = kwargs.get('custom_text', self.responses['unknown'])
        
        # 使用指令控制生成更自然的助理语音
        instruction = "用温和、友善的语气说话，就像一个专业的助理"
        
        audio_result = None
        for i, j in enumerate(self.cosyvoice.inference_instruct2(
            text, instruction, self.assistant_prompt, stream=False)):
            audio_result = j['tts_speech']
            break
        
        return text, audio_result
    
    def process_command(self, command):
        """处理用户命令"""
        command = command.lower()
        
        if "你好" in command or "hello" in command:
            return self.generate_response('greeting')
        elif "时间" in command:
            import datetime
            current_time = datetime.datetime.now().strftime("%H:%M")
            return self.generate_response('time', time=current_time)
        elif "天气" in command:
            return self.generate_response('weather', weather="晴朗", temperature="25")
        elif "再见" in command or "goodbye" in command:
            return self.generate_response('goodbye')
        else:
            return self.generate_response('unknown')

# 使用示例
assistant = VoiceAssistant()

# 模拟用户命令
commands = ["你好", "现在几点了", "今天天气怎么样", "再见"]

for idx, cmd in enumerate(commands):
    print(f"用户: {cmd}")
    response_text, audio = assistant.process_command(cmd)
    print(f"助理: {response_text}")
    
    if audio is not None:
        torchaudio.save(f'assistant_response_{idx}.wav', audio, assistant.cosyvoice.sample_rate)
        print(f"语音保存为: assistant_response_{idx}.wav")
```

### 有声书制作

```python
class AudiobookGenerator:
    """有声书生成器"""
    
    def __init__(self):
        self.cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
        self.available_voices = self.cosyvoice.list_available_spks()
        
        # 角色语音映射
        self.character_voices = {
            'narrator': '中文女',  # 旁白
            'male_character': '中文男',  # 男性角色
            'female_character': '中文女',  # 女性角色
            'child_character': '中文女'  # 儿童角色（使用女声）
        }
    
    def parse_script(self, script_text):
        """解析脚本，识别角色和对话"""
        lines = script_text.strip().split('\n')
        parsed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 识别角色对话格式：【角色名】对话内容
            if line.startswith('【') and '】' in line:
                char_end = line.find('】')
                character = line[1:char_end]
                dialogue = line[char_end+1:].strip()
                
                # 映射角色到语音
                if character == '旁白':
                    voice = self.character_voices['narrator']
                elif character in ['小明', '父亲', '老师']:
                    voice = self.character_voices['male_character']
                elif character in ['小红', '母亲', '老师']:
                    voice = self.character_voices['female_character']
                else:
                    voice = self.character_voices['narrator']  # 默认旁白
                    
                parsed_lines.append({
                    'character': character,
                    'text': dialogue,
                    'voice': voice
                })
            else:
                # 普通旁白
                parsed_lines.append({
                    'character': '旁白',
                    'text': line,
                    'voice': self.character_voices['narrator']
                })
        
        return parsed_lines
    
    def generate_audiobook(self, script_text, output_dir="audiobook", speed=1.0):
        """生成有声书"""
        from pathlib import Path
        import os
        
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 解析脚本
        parsed_lines = self.parse_script(script_text)
        
        audio_files = []
        all_audio = []
        
        for idx, line in enumerate(parsed_lines):
            print(f"正在生成第 {idx+1}/{len(parsed_lines)} 段: {line['character']} - {line['text'][:20]}...")
            
            try:
                # 生成语音
                for i, j in enumerate(self.cosyvoice.inference_sft(
                    line['text'], line['voice'], stream=False, speed=speed)):
                    
                    # 保存单独的音频文件
                    segment_path = os.path.join(output_dir, f'segment_{idx:03d}_{line["character"]}.wav')
                    torchaudio.save(segment_path, j['tts_speech'], self.cosyvoice.sample_rate)
                    
                    audio_files.append({
                        'path': segment_path,
                        'character': line['character'],
                        'text': line['text']
                    })
                    
                    # 收集用于合并的音频
                    all_audio.append(j['tts_speech'])
                    
                    # 添加段落间停顿（0.5秒）
                    silence = torch.zeros(1, int(self.cosyvoice.sample_rate * 0.5))
                    all_audio.append(silence)
                    
                    break  # 只取第一个结果
                    
            except Exception as e:
                print(f"生成第 {idx} 段时发生错误: {e}")
                continue
        
        # 合并所有音频
        if all_audio:
            combined_audio = torch.cat(all_audio, dim=1)
            combined_path = os.path.join(output_dir, 'complete_audiobook.wav')
            torchaudio.save(combined_path, combined_audio, self.cosyvoice.sample_rate)
            print(f"完整有声书已保存: {combined_path}")
        
        # 生成播放列表
        playlist_path = os.path.join(output_dir, 'playlist.txt')
        with open(playlist_path, 'w', encoding='utf-8') as f:
            for audio_file in audio_files:
                f.write(f"{audio_file['path']}\t{audio_file['character']}\t{audio_file['text'][:50]}\n")
        
        return audio_files, combined_path

# 使用示例
script = """
【旁白】在一个阳光明媚的早晨，小明走在上学的路上。
【小明】今天天气真好啊！
【旁白】突然，他遇到了他的朋友小红。
【小红】小明，早上好！我们一起去学校吧。
【小明】好的，我们走吧！
【旁白】两个好朋友一起高高兴兴地走向学校。
"""

# 生成有声书
# generator = AudiobookGenerator()
# audio_files, complete_path = generator.generate_audiobook(script, "my_audiobook", speed=1.0)
# print(f"生成了 {len(audio_files)} 个音频段落")
# print(f"完整有声书路径: {complete_path}")
```

这些示例展示了 CosyVoice 的各种使用方式，从基础的语音合成到复杂的应用场景。每个示例都包含了详细的代码和说明，可以帮助用户快速上手并应用到实际项目中。