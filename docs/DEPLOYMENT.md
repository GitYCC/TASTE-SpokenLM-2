# CosyVoice 部署指南

## 環境準備

### 系統需求

**硬體需求：**
- **GPU**: 建議 NVIDIA GPU（12GB+ VRAM），支援 CUDA 11.8+
- **CPU**: 多核心 CPU，建議 16+ cores
- **記憶體**: 32GB+ RAM 用於訓練，16GB+ 用於推理
- **儲存**: SSD 硬碟，至少 100GB 可用空間

**軟體需求：**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ / ROCm（AMD GPU）
- Docker（容器化部署）
- Git LFS（大型模型文件）

### 依賴安裝

```bash
# 1. 克隆項目
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# 2. 安裝第三方依賴
git submodule update --init --recursive

# 3. 創建虛擬環境
conda create -n cosyvoice python=3.8
conda activate cosyvoice

# 4. 安裝 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. 安裝項目依賴
pip install -r requirements.txt

# 6. 設置 Python 路徑
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/third_party/Matcha-TTS"
```

### 模型下載

**使用 ModelScope（推薦）:**
```python
from modelscope import snapshot_download

# CosyVoice-300M
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')

# CosyVoice2-0.5B
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

# CosyVoice-300M-SFT
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')

# CosyVoice-300M-Instruct
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
```

**手動下載:**
```bash
# 使用 Git LFS
git lfs install
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
```

## 本地部署

### 基本部署

```bash
# 1. 啟動 Python 環境
conda activate cosyvoice

# 2. 設置環境變數
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/third_party/Matcha-TTS"
export COSYVOICE_MODEL_DIR="$(pwd)/pretrained_models"
export CUDA_VISIBLE_DEVICES=0

# 3. 基本推理測試
cd examples
python inference.py --model_dir ../pretrained_models/CosyVoice-300M \
                   --text "Hello, this is CosyVoice speaking." \
                   --output_dir ./output
```

**Python API 使用:**
```python
from cosyvoice.cli.cosyvoice import CosyVoice

# 初始化模型
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

# 零樣本語音合成
for i, j in enumerate(cosyvoice.inference_zero_shot(
    tts_text="你好，我是通義生成式語音大模型。", 
    prompt_text="希望你以後能夠做的比我更好呦。", 
    prompt_speech_16k=prompt_speech_16k
)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], 22050)
```

### Web UI 部署

**Gradio Web UI:**
```bash
# 安裝 Gradio
pip install gradio

# 啟動 Web UI
python webui.py --model_dir pretrained_models/CosyVoice-300M --port 7860
```

**自定義 Web UI:**
```python
import gradio as gr
from cosyvoice.cli.cosyvoice import CosyVoice

# 初始化模型
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

def tts_fn(text, speaker_wav):
    """TTS 函數"""
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        tts_text=text,
        prompt_text="參考音訊",
        prompt_speech_16k=speaker_wav
    )):
        return j['tts_speech'].numpy(), 22050

# 創建界面
iface = gr.Interface(
    fn=tts_fn,
    inputs=[
        gr.Textbox(label="輸入文本"),
        gr.Audio(label="參考音訊", type="numpy")
    ],
    outputs=gr.Audio(label="合成語音"),
    title="CosyVoice TTS"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
```

## 容器化部署

### Docker 部署

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 設置環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace:/workspace/third_party/Matcha-TTS

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    python3 python3-pip git git-lfs \
    libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /workspace

# 複製項目文件
COPY . .

# 安裝 Python 依賴
RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt

# 初始化子模組
RUN git submodule update --init --recursive

# 設置入口點
EXPOSE 7860
CMD ["python3", "webui.py", "--model_dir", "pretrained_models/CosyVoice-300M", "--port", "7860", "--host", "0.0.0.0"]
```

**構建和運行:**
```bash
# 構建映像
docker build -t cosyvoice:latest .

# 運行容器
docker run --gpus all -p 7860:7860 \
           -v $(pwd)/pretrained_models:/workspace/pretrained_models \
           cosyvoice:latest
```

### Docker Compose 部署

```yaml
version: '3.8'

services:
  cosyvoice:
    build: .
    container_name: cosyvoice-app
    ports:
      - "7860:7860"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - COSYVOICE_MODEL_DIR=/workspace/pretrained_models
    volumes:
      - ./pretrained_models:/workspace/pretrained_models
      - ./output:/workspace/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Nginx 反向代理（可選）
  nginx:
    image: nginx:alpine
    container_name: cosyvoice-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - cosyvoice
    restart: unless-stopped
```

**使用方式:**
```bash
# 啟動服務
docker-compose up -d

# 查看日誌
docker-compose logs -f cosyvoice

# 停止服務
docker-compose down
```

## 服務化部署

### FastAPI 服務

**API 服務實現:**
```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import torch
import torchaudio
import io
import base64
from cosyvoice.cli.cosyvoice import CosyVoice

app = FastAPI(title="CosyVoice TTS API")

# 全局模型實例
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

@app.post("/api/v1/tts/zero_shot")
async def zero_shot_tts(
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...)
):
    """零樣本語音合成"""
    # 讀取參考音訊
    audio_data = await prompt_audio.read()
    prompt_speech_16k, sr = torchaudio.load(io.BytesIO(audio_data))
    
    # 重採樣到 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        prompt_speech_16k = resampler(prompt_speech_16k)
    
    # 生成語音
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        tts_text=text,
        prompt_text=prompt_text,
        prompt_speech_16k=prompt_speech_16k
    )):
        # 將音訊轉換為字節流
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, j['tts_speech'], 22050, format='wav')
        audio_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(audio_buffer.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )

@app.post("/api/v1/tts/sft")
async def sft_tts(text: str = Form(...), speaker: str = Form(...)):
    """SFT 模式語音合成"""
    for i, j in enumerate(cosyvoice.inference_sft(
        tts_text=text,
        spk_id=speaker
    )):
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, j['tts_speech'], 22050, format='wav')
        audio_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(audio_buffer.read()),
            media_type="audio/wav"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**啟動服務:**
```bash
# 安裝 FastAPI
pip install fastapi uvicorn python-multipart

# 啟動服務
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### gRPC 服務

**Proto 定義 (tts.proto):**
```protobuf
syntax = "proto3";

package cosyvoice;

service TTSService {
  rpc ZeroShotTTS(ZeroShotRequest) returns (AudioResponse);
  rpc SFTTTS(SFTRequest) returns (AudioResponse);
  rpc StreamingTTS(StreamingRequest) returns (stream AudioChunk);
}

message ZeroShotRequest {
  string text = 1;
  string prompt_text = 2;
  bytes prompt_audio = 3;
}

message SFTRequest {
  string text = 1;
  string speaker_id = 2;
}

message StreamingRequest {
  string text = 1;
  string speaker_id = 2;
  int32 chunk_size = 3;
}

message AudioResponse {
  bytes audio_data = 1;
  int32 sample_rate = 2;
  string format = 3;
}

message AudioChunk {
  bytes chunk_data = 1;
  bool is_last = 2;
}
```

**gRPC 服務實現:**
```python
import grpc
from concurrent import futures
import tts_pb2_grpc
import tts_pb2
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio
import io

class TTSServicer(tts_pb2_grpc.TTSServiceServicer):
    def __init__(self):
        self.cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
    
    def ZeroShotTTS(self, request, context):
        # 解析參考音訊
        prompt_speech_16k, sr = torchaudio.load(io.BytesIO(request.prompt_audio))
        
        # 生成語音
        for i, j in enumerate(self.cosyvoice.inference_zero_shot(
            tts_text=request.text,
            prompt_text=request.prompt_text,
            prompt_speech_16k=prompt_speech_16k
        )):
            # 序列化音訊
            audio_buffer = io.BytesIO()
            torchaudio.save(audio_buffer, j['tts_speech'], 22050, format='wav')
            
            return tts_pb2.AudioResponse(
                audio_data=audio_buffer.getvalue(),
                sample_rate=22050,
                format='wav'
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tts_pb2_grpc.add_TTSServiceServicer_to_server(TTSServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

### Triton 部署

**TensorRT-LLM 優化:**
```python
# 模型轉換為 TensorRT
from cosyvoice.utils.tensorrt import convert_to_tensorrt

# 轉換 LLM 部分
llm_engine = convert_to_tensorrt(
    model_path='pretrained_models/CosyVoice-300M/llm.pt',
    max_batch_size=8,
    max_input_len=200,
    max_output_len=1024,
    dtype='float16'
)

# 轉換 Flow 部分
flow_engine = convert_to_tensorrt(
    model_path='pretrained_models/CosyVoice-300M/flow.pt',
    max_batch_size=8,
    dtype='float16'
)
```

**Triton 模型配置:**
```
# model_repository/cosyvoice_llm/config.pbtxt
name: "cosyvoice_llm"
platform: "tensorrt_llm"
max_batch_size: 8

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [-1]
  }
]

output [
  {
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [-1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

**啟動 Triton Server:**
```bash
# 使用 Docker 啟動
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.12-trtllm-python-py3 \
  tritonserver --model-repository=/models
```

## 雲端部署

### 雲端平台部署選項

**AWS 部署:**
```yaml
# AWS ECS 任務定義
family: cosyvoice-task
networkMode: awsvpc
requiresCompatibilities:
  - FARGATE
cpu: 2048
memory: 8192

containerDefinitions:
  - name: cosyvoice-container
    image: your-account.dkr.ecr.region.amazonaws.com/cosyvoice:latest
    portMappings:
      - containerPort: 7860
        protocol: tcp
    environment:
      - name: CUDA_VISIBLE_DEVICES
        value: "0"
    logConfiguration:
      logDriver: awslogs
      options:
        awslogs-group: /ecs/cosyvoice
        awslogs-region: us-west-2
        awslogs-stream-prefix: ecs
```

**Google Cloud 部署:**
```yaml
# Cloud Run 服務配置
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: cosyvoice-service
  annotations:
    run.googleapis.com/gpu-type: nvidia-tesla-t4
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: '5'
        run.googleapis.com/cpu: '2'
        run.googleapis.com/memory: 8Gi
    spec:
      containers:
      - image: gcr.io/your-project/cosyvoice:latest
        ports:
        - containerPort: 7860
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          limits:
            nvidia.com/gpu: 1
```

**Azure 部署:**
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-03-01",
      "name": "cosyvoice-container-group",
      "location": "East US",
      "properties": {
        "containers": [
          {
            "name": "cosyvoice-container",
            "properties": {
              "image": "your-registry.azurecr.io/cosyvoice:latest",
              "resources": {
                "requests": {
                  "cpu": 2,
                  "memoryInGb": 8,
                  "gpu": {
                    "count": 1,
                    "sku": "V100"
                  }
                }
              },
              "ports": [
                {
                  "port": 7860,
                  "protocol": "TCP"
                }
              ]
            }
          }
        ],
        "osType": "Linux",
        "restartPolicy": "Always"
      }
    }
  ]
}
```

### 擴展和負載均衡

**Kubernetes 部署:**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cosyvoice-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cosyvoice
  template:
    metadata:
      labels:
        app: cosyvoice
    spec:
      containers:
      - name: cosyvoice
        image: cosyvoice:latest
        ports:
        - containerPort: 7860
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
---
apiVersion: v1
kind: Service
metadata:
  name: cosyvoice-service
spec:
  selector:
    app: cosyvoice
  ports:
  - port: 80
    targetPort: 7860
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cosyvoice-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cosyvoice-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Nginx 負載均衡:**
```nginx
upstream cosyvoice_backend {
    least_conn;
    server cosyvoice-1:7860 max_fails=3 fail_timeout=30s;
    server cosyvoice-2:7860 max_fails=3 fail_timeout=30s;
    server cosyvoice-3:7860 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name tts.example.com;
    
    location / {
        proxy_pass http://cosyvoice_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_timeout 300s;
        proxy_read_timeout 300s;
        client_max_body_size 100M;
    }
    
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

## 效能優化

### GPU 加速配置

**GPU 配置優化:**
```python
# GPU 設定
import torch
import os

# 設置 GPU 設備
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU 記憶體優化
torch.backends.cudnn.benchmark = True  # 啟用 cuDNN 自動調優
torch.backends.cuda.matmul.allow_tf32 = True  # 允許 TF32

# 模型初始化
cosyvoice = CosyVoice(
    model_dir='pretrained_models/CosyVoice-300M',
    load_jit=True,    # JIT 編譯加速
    load_trt=True,    # TensorRT 加速
    fp16=True,        # 半精度推理
    device=device
)
```

**多 GPU 配置:**
```python
# 數據並行
if torch.cuda.device_count() > 1:
    cosyvoice.llm = torch.nn.DataParallel(cosyvoice.llm)
    cosyvoice.flow = torch.nn.DataParallel(cosyvoice.flow)
    cosyvoice.hifigan = torch.nn.DataParallel(cosyvoice.hifigan)
    print(f"使用 {torch.cuda.device_count()} 個 GPU")
```

**GPU 監控腳本:**
```bash
#!/bin/bash
# gpu_monitor.sh
while true; do
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
                --format=csv,noheader,nounits \
                >> gpu_usage.log
    sleep 10
done
```

### 記憶體優化

**記憶體管理策略:**
```python
# 記憶體優化配置
import torch
import gc

# 啟用梯度檢查點
torch.backends.cudnn.benchmark = True

# 模型配置
config = {
    'gradient_checkpointing': True,  # 減少記憶體使用
    'use_cache': False,             # 禁用 KV 緩存以節省記憶體
    'low_memory': True,             # 低記憶體模式
    'offload_to_cpu': True,         # CPU 卸載
}

# 定期清理記憶體
def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()

# 批處理大小自適應
def adaptive_batch_size(base_batch_size=4):
    available_memory = torch.cuda.get_device_properties(0).total_memory
    used_memory = torch.cuda.memory_allocated(0)
    free_memory = available_memory - used_memory
    
    if free_memory < 2e9:  # 2GB
        return max(1, base_batch_size // 2)
    elif free_memory < 4e9:  # 4GB
        return base_batch_size
    else:
        return min(8, base_batch_size * 2)
```

**記憶體監控:**
```python
# 記憶體使用監控
import psutil
import torch

def monitor_memory():
    # CPU 記憶體
    cpu_percent = psutil.virtual_memory().percent
    cpu_available = psutil.virtual_memory().available / (1024**3)  # GB
    
    # GPU 記憶體
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"CPU: {cpu_percent:.1f}% used, {cpu_available:.1f}GB available")
        print(f"GPU: {gpu_allocated:.1f}GB allocated, {gpu_cached:.1f}GB cached, {gpu_total:.1f}GB total")
    
    return {
        'cpu_percent': cpu_percent,
        'cpu_available_gb': cpu_available,
        'gpu_allocated_gb': gpu_allocated if torch.cuda.is_available() else 0,
        'gpu_cached_gb': gpu_cached if torch.cuda.is_available() else 0
    }
```

### 並發處理

**異步處理實現:**
```python
import asyncio
import concurrent.futures
from queue import Queue
import threading

class ConcurrentTTSServer:
    def __init__(self, model_path, max_workers=4, queue_size=100):
        self.cosyvoice = CosyVoice(model_path)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.request_queue = Queue(maxsize=queue_size)
        self.processing_workers = []
        
    async def process_request(self, text, speaker_info):
        """異步處理 TTS 請求"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            self._generate_speech,
            text,
            speaker_info
        )
        return await future
    
    def _generate_speech(self, text, speaker_info):
        """實際的語音生成函數"""
        with torch.no_grad():  # 推理時不計算梯度
            for i, j in enumerate(self.cosyvoice.inference_sft(
                tts_text=text,
                spk_id=speaker_info['speaker_id']
            )):
                return j['tts_speech']
    
    async def batch_process(self, requests):
        """批量處理請求"""
        tasks = []
        for req in requests:
            task = asyncio.create_task(
                self.process_request(req['text'], req['speaker_info'])
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

**請求排隊系統:**
```python
import redis
import json
from celery import Celery

# Celery 配置
app = Celery('cosyvoice_tasks', broker='redis://localhost:6379')

@app.task
def tts_task(text, speaker_id, task_id):
    """後台 TTS 任務"""
    cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
    
    try:
        # 生成語音
        result = []
        for i, j in enumerate(cosyvoice.inference_sft(
            tts_text=text,
            spk_id=speaker_id
        )):
            # 保存到臨時文件
            filename = f"output_{task_id}_{i}.wav"
            torchaudio.save(filename, j['tts_speech'], 22050)
            result.append(filename)
        
        return {'status': 'success', 'files': result}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# 啟動 Worker
# celery -A tts_worker worker --loglevel=info --concurrency=4
```

**負載平衡器:**
```python
import random
from typing import List, Dict

class TTSLoadBalancer:
    def __init__(self, workers: List[Dict]):
        self.workers = workers
        self.worker_status = {w['id']: {'load': 0, 'available': True} for w in workers}
    
    def get_best_worker(self):
        """選擇負載最小的可用 worker"""
        available_workers = [
            (worker_id, status) for worker_id, status in self.worker_status.items()
            if status['available']
        ]
        
        if not available_workers:
            return None
        
        # 選擇負載最小的 worker
        best_worker = min(available_workers, key=lambda x: x[1]['load'])
        return best_worker[0]
    
    def assign_task(self, task_data):
        """分配任務到最佳 worker"""
        worker_id = self.get_best_worker()
        if worker_id:
            self.worker_status[worker_id]['load'] += 1
            # 發送任務到對應的 worker
            return self._send_to_worker(worker_id, task_data)
        else:
            return {'error': 'No available workers'}
    
    def task_completed(self, worker_id):
        """標記任務完成，減少 worker 負載"""
        if worker_id in self.worker_status:
            self.worker_status[worker_id]['load'] = max(0, self.worker_status[worker_id]['load'] - 1)
```

## 監控和維護

### 日誌配置

**Python 日誌配置:**
```python
import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

# 設置日誌格式
class TTSFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'filename': record.filename,
            'line': record.lineno
        }
        
        # 添加異常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加自定義字段
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        return json.dumps(log_entry, ensure_ascii=False)

# 配置日誌記錄器
def setup_logging():
    # 創建根記錄器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 文件處理器
    file_handler = RotatingFileHandler(
        'logs/cosyvoice.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    file_handler.setFormatter(TTSFormatter())
    
    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# 請求日誌記錄
def log_request(request_id, text, speaker_id, start_time, end_time, status):
    logger = logging.getLogger('tts.requests')
    duration = end_time - start_time
    
    extra = {
        'request_id': request_id,
        'text_length': len(text),
        'speaker_id': speaker_id,
        'duration_ms': duration * 1000,
        'status': status
    }
    
    logger.info(f"TTS request completed", extra=extra)
```

**Docker 日誌配置:**
```yaml
# docker-compose.yml 日誌配置
version: '3.8'
services:
  cosyvoice:
    # ...
    logging:
      driver: "json-file"
      options:
        max-size: "200m"
        max-file: "10"
        labels: "service=cosyvoice"
    # ...
```

### 效能監控

**Prometheus 監控:**
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import psutil
import torch

# 定義監控指標
REQUEST_COUNT = Counter(
    'tts_requests_total',
    'Total TTS requests',
    ['method', 'status']
)

REQUEST_DURATION = Histogram(
    'tts_request_duration_seconds',
    'TTS request duration',
    ['method']
)

CURRENT_REQUESTS = Gauge(
    'tts_current_requests',
    'Current active TTS requests'
)

GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['device']
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage'
)

class MetricsCollector:
    def __init__(self):
        self.start_metrics_server()
        self.update_system_metrics()
    
    def start_metrics_server(self, port=8001):
        """啟動 Prometheus 指標服務器"""
        start_http_server(port)
    
    def record_request(self, method, status, duration):
        """記錄請求指標"""
        REQUEST_COUNT.labels(method=method, status=status).inc()
        REQUEST_DURATION.labels(method=method).observe(duration)
    
    def update_system_metrics(self):
        """更新系統指標"""
        # CPU 使用率
        CPU_USAGE.set(psutil.cpu_percent())
        
        # GPU 記憶體使用
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                GPU_MEMORY_USAGE.labels(device=f'cuda:{i}').set(allocated)

# 裝飾器用於自動記錄請求
def monitor_request(method_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            CURRENT_REQUESTS.inc()
            
            try:
                result = func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                CURRENT_REQUESTS.dec()
                metrics_collector.record_request(method_name, status, duration)
        
        return wrapper
    return decorator

# 使用示例
metrics_collector = MetricsCollector()

@monitor_request('zero_shot_tts')
def zero_shot_tts(text, prompt_text, prompt_audio):
    # TTS 邏輯
    pass
```

**Grafana 儀表板配置:**
```json
{
  "dashboard": {
    "title": "CosyVoice TTS Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(tts_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Request Duration",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(tts_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "gpu_memory_usage_bytes / 1024 / 1024 / 1024"
          }
        ]
      }
    ]
  }
}
```

### 故障恢復

**健康檢查機制:**
```python
# health_check.py
import asyncio
import aiohttp
import logging
from datetime import datetime

class HealthChecker:
    def __init__(self, services, check_interval=30):
        self.services = services  # 服務列表
        self.check_interval = check_interval
        self.logger = logging.getLogger('health_checker')
        self.failed_services = set()
    
    async def check_service_health(self, service):
        """檢查單個服務健康狀態"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{service['url']}/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        if service['id'] in self.failed_services:
                            self.failed_services.remove(service['id'])
                            self.logger.info(f"Service {service['id']} recovered")
                        return True
                    else:
                        raise Exception(f"Health check failed with status {response.status}")
        
        except Exception as e:
            self.logger.error(f"Health check failed for {service['id']}: {str(e)}")
            self.failed_services.add(service['id'])
            return False
    
    async def auto_restart_service(self, service):
        """自動重啟失敗的服務"""
        try:
            # 發送重啟命令
            restart_command = service.get('restart_command')
            if restart_command:
                import subprocess
                subprocess.run(restart_command, shell=True, check=True)
                self.logger.info(f"Service {service['id']} restarted")
                
                # 等待服務啟動
                await asyncio.sleep(30)
                
                # 重新檢查健康狀態
                return await self.check_service_health(service)
        
        except Exception as e:
            self.logger.error(f"Failed to restart service {service['id']}: {str(e)}")
            return False
    
    async def run_health_checks(self):
        """運行健康檢查循環"""
        while True:
            try:
                for service in self.services:
                    is_healthy = await self.check_service_health(service)
                    
                    if not is_healthy and service.get('auto_restart', False):
                        self.logger.warning(f"Attempting to restart {service['id']}")
                        await self.auto_restart_service(service)
                
                await asyncio.sleep(self.check_interval)
            
            except Exception as e:
                self.logger.error(f"Health check loop error: {str(e)}")
                await asyncio.sleep(self.check_interval)

# 配置服務列表
services = [
    {
        'id': 'cosyvoice-1',
        'url': 'http://localhost:7860',
        'auto_restart': True,
        'restart_command': 'docker restart cosyvoice-container-1'
    },
    {
        'id': 'cosyvoice-2',
        'url': 'http://localhost:7861',
        'auto_restart': True,
        'restart_command': 'docker restart cosyvoice-container-2'
    }
]

# 啟動健康檢查
hc = HealthChecker(services)
asyncio.run(hc.run_health_checks())
```

**Kubernetes 故障恢復:**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cosyvoice-deployment
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: cosyvoice
        # ...
        livenessProbe:
          httpGet:
            path: /health
            port: 7860
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 7860
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "2"
```

**自動擴縮容配置:**
```bash
#!/bin/bash
# auto_scaling.sh

# 監控 CPU 和內存使用率
while true; do
    CPU_USAGE=$(docker stats --no-stream --format "table {{.CPUPerc}}" cosyvoice-app | tail -n +2 | sed 's/%//')
    MEM_USAGE=$(docker stats --no-stream --format "table {{.MemPerc}}" cosyvoice-app | tail -n +2 | sed 's/%//')
    
    # 如果 CPU 使用率超過 80% 或內存使用率超過 85%
    if (( $(echo "$CPU_USAGE > 80" | bc -l) )) || (( $(echo "$MEM_USAGE > 85" | bc -l) )); then
        echo "High resource usage detected. CPU: ${CPU_USAGE}%, MEM: ${MEM_USAGE}%"
        
        # 檢查是否需要擴展
        CURRENT_REPLICAS=$(docker ps --filter "name=cosyvoice" --format "table {{.Names}}" | wc -l)
        if [ $CURRENT_REPLICAS -lt 5 ]; then
            echo "Scaling up services..."
            docker-compose up -d --scale cosyvoice=$((CURRENT_REPLICAS + 1))
        fi
    
    # 如果使用率很低，考慮縮容
    elif (( $(echo "$CPU_USAGE < 30" | bc -l) )) && (( $(echo "$MEM_USAGE < 40" | bc -l) )); then
        CURRENT_REPLICAS=$(docker ps --filter "name=cosyvoice" --format "table {{.Names}}" | wc -l)
        if [ $CURRENT_REPLICAS -gt 1 ]; then
            echo "Low resource usage detected. Scaling down..."
            docker-compose up -d --scale cosyvoice=$((CURRENT_REPLICAS - 1))
        fi
    fi
    
    sleep 60  # 每分鐘檢查一次
done
```