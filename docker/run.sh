#!/bin/bash
set -e

# TASTE-SpokenLM-2 Docker Run Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="taste-spokenlm-2:latest"
CONTAINER_NAME="taste-spokenlm-2-dev"

# Check GPU availability
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
    echo "✓ GPU support enabled"
else
    GPU_FLAG=""
    echo "⚠ GPU not available, running CPU-only"
fi

echo "Starting container: ${CONTAINER_NAME}"
echo ""

docker run \
    ${GPU_FLAG} \
    -it \
    --rm \
    --name "${CONTAINER_NAME}" \
    --shm-size=8g \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v ${PROJECT_ROOT}/../:/workspace \
    -v "${HOME}/.cache:/root/.cache" \
    -w /workspace/TASTE-SpokenLM-2 \
    "${IMAGE_NAME}" \
    /bin/bash
