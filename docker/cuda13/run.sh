#!/bin/bash
set -e

# TASTE-SpokenLM-2 Docker Run Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MOUNT_ROOT="$(dirname $(dirname $(dirname "$SCRIPT_DIR")))"
IMAGE_NAME="taste2-cu13:latest"
CONTAINER_NAME="taste2-cu13-dev"

echo "Starting container: ${CONTAINER_NAME}"
echo ""

docker run \
    --gpus all \
    -it \
    --rm \
    --name "${CONTAINER_NAME}" \
    --shm-size=8g \
    -p 8000:8000 \
    -p 3000:3000 \
    -p 16686:16686 \
    -v ${MOUNT_ROOT}:/mount \
    -v "${HOME}/.cache:/root/.cache" \
    -w /mount/TASTE-Voice-Bot \
    "${IMAGE_NAME}" \
    bash
