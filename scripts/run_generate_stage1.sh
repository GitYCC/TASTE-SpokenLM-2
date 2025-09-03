#!/bin/bash

# TASTE2 Audio Generation Script - Stage 1 Batch Processing
# Usage: ./run_generate_stage1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MODEL_DIR="/mnt/shared/NTU_TASLM/models/taste2_8B_final"
OUTPUT_DIR="./results/stage1/taste2_8B_final"
ASR_MODEL="openai/whisper-large-v3"

# Test files
TEST_FILES=(
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/ex01_happy_00209.wav"
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/ex04_sad_00311.wav"
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/hifi-tts-dev-clean-speaker6097/001.wav"
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/hifi-tts-dev-clean-speaker6097/004.wav"
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/hifi-tts-dev-clean-speaker6097/012.wav"
)

echo "Running TASTE2 Stage 1 Audio Generation..."
echo "Model: $(basename "$MODEL_DIR")"
echo "Test files: ${#TEST_FILES[@]}"

mkdir -p "$OUTPUT_DIR"

# Run batch processing with all files at once
echo "Starting batch processing..."
python "$SCRIPT_DIR/generate_audio.py" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --test_files "${TEST_FILES[@]}" \
    --asr_model_dir "$ASR_MODEL" \
    --stage 1

if [ $? -eq 0 ]; then
    echo "✓ Batch processing completed successfully!"
else
    echo "✗ Batch processing failed"
fi