# TASTE-SpokenLM-2 Docker Setup

Containerized environment for TASTE-SpokenLM-2, based on NVIDIA TensorRT-LLM with full GPU acceleration support.

---

## 🚀 Quick Start

### Three Steps to Get Started

```bash
# 1. Build the image
cd /workspace/TASTE-SpokenLM-2
bash docker/build.sh

# 2. Run the container
bash docker/run.sh

# 3. Inside container - run test
bash scripts/run_generate_sft.sh
```

### Common Commands

**Check GPU:**
```bash
nvidia-smi
```

**Custom generation:**
```bash
python scripts/generate_audio.py \
    --model_dir /workspace/models/TASTE2-8B-EN-SFT \
    --audio_dir audio_samples/stage1_EN \
    --output_dir results/sft/custom \
    --stage sft
```

**View results:**
```bash
ls -lh results/sft/test/
cat results/sft/test/batch_results.json
```

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Build & Run](#build--run)
- [Environment Details](#environment-details)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Advanced Options](#advanced-options)
- [Important Paths](#important-paths)
- [Resource Requirements](#resource-requirements)

---

## 📦 Prerequisites

### Required
- Docker 20.10+
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Docker runtime (required for GPU support)

### Installing NVIDIA Docker Runtime

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker service
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 🔨 Build & Run

### Method 1: Using Scripts (Recommended)

```bash
# Build
bash docker/build.sh

# Run
bash docker/run.sh
```

### Method 2: Manual Docker Commands

**Build image:**
```bash
cd /workspace/TASTE-SpokenLM-2
docker build -f docker/Dockerfile -t taste-spokenlm-2:latest .
```

**Run container (with GPU):**
```bash
docker run --gpus all -it --rm \
    --name taste-spokenlm-2-dev \
    --shm-size=8g \
    -v $(pwd):/workspace/TASTE-SpokenLM-2 \
    -v ~/.cache:/root/.cache \
    -w /workspace/TASTE-SpokenLM-2 \
    taste-spokenlm-2:latest \
    /bin/bash
```

**Run container (CPU only):**
```bash
docker run -it --rm \
    --shm-size=8g \
    -v $(pwd):/workspace/TASTE-SpokenLM-2 \
    -v ~/.cache:/root/.cache \
    -w /workspace/TASTE-SpokenLM-2 \
    taste-spokenlm-2:latest \
    /bin/bash
```

### Inside the Container

Once inside the container, you can run:

```bash
# Run SFT generation script
bash scripts/run_generate_sft.sh

# Or run Python script directly
python scripts/generate_audio.py \
    --model_dir /workspace/models/TASTE2-8B-EN-SFT \
    --audio_dir audio_samples/stage1_EN \
    --output_dir results/sft/test \
    --stage sft
```

---

## 📊 Environment Details

### Base Image
- **Base**: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc0`
- **CUDA**: 12.1
- **TensorRT-LLM**: Pre-installed and optimized

### Python Dependencies

**Core Libraries:**
- PyTorch 2.3.1 (CUDA 12.1)
- TorchAudio 2.3.1
- Transformers 4.51.3
- Lightning 2.2.4

**Audio Processing:**
- OpenAI Whisper 20231117
- Librosa 0.10.2
- PyWorld 0.3.4
- Conformer 0.3.2
- SoundFile 0.12.1

**Model Infrastructure:**
- ModelScope 1.34.0
- HyperPyYAML 1.2.2
- Hydra-core 1.3.2
- Diffusers 0.29.0
- ONNX Runtime 1.24.1

**Utilities:**
- einops, einx
- torchmetrics
- omegaconf
- gdown, wget

### System Packages
- ffmpeg, sox
- git, git-lfs
- Build tools and compilers

### Environment Variables
- `PYTHONUNBUFFERED=1`

### Special Configuration
- cuDNN compatibility layer (8→9 automatic symlink)
- Optimized for GPU and CPU execution

---

## ✅ Verification

After building, verify the setup:

```bash
# 1. Check image exists
docker images | grep taste-spokenlm-2

# 2. Start container
bash docker/run.sh

# 3. Inside container - verify Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from taste_speech import *; print('✓ TASTE modules loaded')"

# 4. Check GPU
nvidia-smi

# 5. Run test script
bash scripts/run_generate_sft.sh
```

---

## 🔧 Troubleshooting

### Container Won't Start

```bash
# Check if image exists
docker images | grep taste-spokenlm-2

# Rebuild if needed
bash docker/build.sh

# Check logs
docker logs taste-spokenlm-2-dev
```

### GPU Not Detected

**Problem**: Cannot access GPU inside container

**Solution**:
1. Check NVIDIA Docker runtime:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
2. Verify GPU is visible on host: `nvidia-smi`
3. Check Docker daemon configuration includes nvidia runtime
4. If runtime not installed:
   ```bash
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### Out of Memory (OOM)

**Problem**: Memory overflow during execution

**Solution**:
1. Increase shared memory: `--shm-size=16g`
2. Reduce batch size in scripts
3. Use smaller model
4. Example with more memory:
   ```bash
   docker run --gpus all -it --rm --shm-size=16g \
       -v $(pwd):/workspace/TASTE-SpokenLM-2 \
       taste-spokenlm-2:latest /bin/bash
   ```

### cuDNN Version Issues

Dockerfile automatically handles cuDNN 8/9 compatibility. No manual intervention needed.

---

## 🔧 Advanced Options

### Build Options

**Custom tag:**
```bash
docker build -f docker/Dockerfile -t taste-spokenlm-2:v1.0 .
```

**Clean rebuild (no cache):**
```bash
docker build --no-cache -f docker/Dockerfile -t taste-spokenlm-2:latest .
```

**Specific platform:**
```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t taste-spokenlm-2:latest .
```

### Run Options

**Increase shared memory:**
```bash
docker run --gpus all -it --rm --shm-size=16g \
    -v $(pwd):/workspace/TASTE-SpokenLM-2 \
    taste-spokenlm-2:latest /bin/bash
```

**Run in background:**
```bash
docker run --gpus all -d \
    --name taste-spokenlm-2-dev \
    --shm-size=8g \
    -v $(pwd):/workspace/TASTE-SpokenLM-2 \
    -v ~/.cache:/root/.cache \
    taste-spokenlm-2:latest \
    tail -f /dev/null

# Attach to container
docker exec -it taste-spokenlm-2-dev /bin/bash
```

**Multiple terminal access:**
```bash
# In another terminal
docker exec -it taste-spokenlm-2-dev /bin/bash
```

---

## 📁 Important Paths

- **Project root**: `/workspace/TASTE-SpokenLM-2`
- **Model cache**: `/root/.cache/modelscope` (mounted from host `~/.cache`)
- **Audio samples**: `/workspace/TASTE-SpokenLM-2/audio_samples`
- **Results**: `/workspace/TASTE-SpokenLM-2/results`

---

## 📊 Resource Requirements

### Minimum
- 8GB RAM
- 20GB disk space
- CUDA GPU (optional)

### Recommended
- 16GB+ RAM
- 50GB+ disk space
- NVIDIA GPU with 12GB+ VRAM
- Fast SSD storage

### Build & Runtime
- **Build time**: 10-15 minutes (depending on internet speed)
- **Image size**: 15-20GB (includes all dependencies)
- **First run**: Downloads models (~8-10GB, cached in `~/.cache`)
- **Model cache**: Cached on host, shared across containers

---

## 💡 Usage Tips

- **Persistent cache**: Model downloads cached in host `~/.cache`, shared across containers
- **Live code updates**: Project directory mounted, code changes take effect immediately
- **Auto cleanup**: Container removed on exit (`--rm` flag)
- **Development friendly**: Multiple terminals can attach to running container

---

## 🎯 Advantages Over Manual Setup

Docker setup compared to manual installation:

1. **GPU Support**: Uses CUDA-enabled PyTorch instead of CPU-only version
2. **Reproducibility**: All versions pinned, consistent environment
3. **Isolation**: No conflicts with system packages
4. **Portability**: Works on any system with Docker + NVIDIA runtime
5. **Cleanliness**: No leftover files or environment pollution

---

## 📝 Notes

- Container runs as root by default
- Changes to mounted volumes persist after container exit
- Container auto-removed on exit with `--rm` flag
- Host and container share GPU drivers

---

## 🆘 Getting Help

If you encounter issues:
1. Check logs: `docker logs taste-spokenlm-2-dev`
2. Check disk space: `df -h`
3. Verify GPU: `nvidia-smi`
4. Clean rebuild: `docker build --no-cache ...`

---

**Last Updated**: 2026-02-06
**Based on**: NVIDIA TensorRT-LLM 1.3.0rc0
