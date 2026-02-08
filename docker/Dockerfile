FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc8

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    sox \
    libsox-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the project
COPY .. /workspace/TASTE-SpokenLM-2/

# Install Python dependencies in order to avoid conflicts
# Core dependencies first
RUN pip install --no-cache-dir \
    hyperpyyaml==1.2.2 \
    modelscope==1.34.0 \
    numpy==1.26.4 \
    'ruamel.yaml<0.19'

# Install torchaudio (--no-deps to preserve base image's torch/torchvision)
# Must match PyTorch version in base image (2.9.x) and use cu130 index
RUN pip install --no-cache-dir --no-deps \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130

# Install torchcodec (required by torchaudio.load)
RUN pip install --no-cache-dir \
    torchcodec==0.9.1 \
    --index-url https://download.pytorch.org/whl/cu130

# # Uninstall incompatible torchao if present
# RUN pip uninstall -y torchao || true

# # Install transformers and related packages
# RUN pip install --no-cache-dir \
#     transformers==4.51.3 \
#     tokenizers==0.21.4 \
#     safetensors==0.7.0

# Install audio processing libraries
RUN pip install --no-cache-dir \
    einx==0.3.0 \
    einops==0.8.1 \
    conformer==0.3.2 \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    pyworld==0.3.4 \
    openai-whisper==20250625 \
    onnxruntime==1.24.1

# Install deep learning frameworks
RUN pip install --no-cache-dir \
    lightning==2.2.4 \
    'fsspec<2025.0,>=2022.5.0' \
    torchmetrics==1.8.2

# Remove any pre-installed flash-attn from the base image (linked against a
# different PyTorch ABI), then rebuild from source against PyTorch 2.3.1.
# RUN pip uninstall -y flash-attn flash_attn || true
# RUN pip install --no-cache-dir ninja packaging && \
#     pip install --no-cache-dir flash-attn --no-build-isolation --force-reinstall

# Install utility packages
RUN pip install --no-cache-dir \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    gdown==5.1.0 \
    wget==3.2 \
    wetext==0.0.4 \
    diffusers==0.29.0

# Ensure torch ecosystem CUDA versions are consistent after all installs.
# Later packages (openai-whisper, diffusers, etc.) may pull in a torchvision
# built for a different CUDA, so we force-reinstall matching versions here.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
    torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu130

# Create cuDNN compatibility symlink if needed
RUN if [ ! -f /lib/x86_64-linux-gnu/libcudnn.so.8 ] && [ -f /lib/x86_64-linux-gnu/libcudnn.so.9 ]; then \
    ln -s /lib/x86_64-linux-gnu/libcudnn.so.9 /lib/x86_64-linux-gnu/libcudnn.so.8 && \
    ldconfig; \
    fi

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash &&echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Set working directory to project root
WORKDIR /workspace/TASTE-SpokenLM-2

# Install the project in editable mode
RUN pip install -e /workspace/TASTE-SpokenLM-2

# Default command
CMD ["/bin/bash"]
