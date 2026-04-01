FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME="taste2"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing
RUN apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev tmux nodejs npm protobuf-compiler && \
    apt-get clean && \
    git lfs install

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

RUN conda activate ${VENV} && \
    pip install -U "setuptools>=70,<71" && \
    pip install -r /workspace/TASTE-SpokenLM-2/docker/cuda124/requirements.txt

# Install TensorRT-LLM (upgrades torch to 2.9+ and pulls CUDA 13 deps)
RUN conda activate ${VENV} && pip install tensorrt-llm==1.2.0

# TRT-LLM's pip deps (nvidia-cuda-runtime-cu13 etc.) register /usr/local/cuda-13.0
# as an alternative with higher priority (130 > 124), but cuda-13.0 only ships
# libraries -- no nvcc or toolkit. Pin the symlink back to 12.4 so FlashInfer
# and other JIT-compiling packages can find nvcc.
RUN update-alternatives --set cuda /usr/local/cuda-12.4

RUN conda activate ${VENV} && pip install /workspace/TASTE-SpokenLM-2

