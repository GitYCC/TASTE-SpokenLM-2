FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ARG VENV_NAME="taste2"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing
# CTranslate2 (used by faster-whisper / ASR) is built against CUDA 12 and
# dlopen()s libcublas.so.12 at runtime.  Install the CUDA 12 cuBLAS alongside
# the native CUDA 13 toolkit so both stacks work.
RUN apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev tmux protobuf-compiler libopenmpi-dev libcublas-12-8 && \
    apt-get clean && \
    git lfs install

# Node.js 20.x (Ubuntu 22.04 ships v12 which lacks optional-chaining syntax)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# ==================================================================
# conda install and conda forge channel as default
# ------------------------------------------------------------------
# Install miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH=/opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict
# ------------------------------------------------------------------
# ~conda
# ==================================================================
RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace

RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5

# Install the project
COPY . /workspace/TASTE-SpokenLM-2/

# setuptools first (TRT-LLM needs >=70)
RUN conda activate ${VENV} && pip install -U "setuptools>=70,<80"

# TRT-LLM (pulls torch cu128 + transformers + triton + many others)
RUN conda activate ${VENV} && pip install tensorrt-llm==1.2.0

# Force-replace the cu128 torch that TRT-LLM installed with the cu130 build.
# --force-reinstall --no-deps is required because pip considers 2.9.1 == 2.9.1+cu130
# and would skip the reinstall otherwise.
RUN conda activate ${VENV} && \
    pip install --force-reinstall --no-deps \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu130

# Remaining taste2 dependencies (versions updated for CUDA 13 / TRT-LLM compat)
RUN conda activate ${VENV} && \
    pip install -r /workspace/TASTE-SpokenLM-2/docker/cuda13x/requirements.txt

RUN conda activate ${VENV} && pip install /workspace/TASTE-SpokenLM-2
