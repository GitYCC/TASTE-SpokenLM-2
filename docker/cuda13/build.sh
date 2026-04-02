#!/bin/bash
set -e

# TASTE-SpokenLM-2 Docker Build Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname $(dirname $SCRIPT_DIR))"
IMAGE_NAME="taste2-cu13:latest"

echo "Building Docker image: ${IMAGE_NAME}"
echo "Project root: ${PROJECT_ROOT}"
echo ""

cd "$PROJECT_ROOT"
docker build -f $SCRIPT_DIR/Dockerfile -t "${IMAGE_NAME}" .
