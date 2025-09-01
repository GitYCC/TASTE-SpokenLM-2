#!/bin/bash

# TASTE2 Audio Generation Script - Stage 2 Batch Processing
# Usage: ./run_generate_stage2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MODEL_DIR="/mnt/shared/NTU_TASLM/yc/TASTE-SpokenLM-2/training/checkpoints/taste2-stage2-pretrain-400k-1B-60k"
OUTPUT_DIR="./reconstruction_stage2/taste2-stage2-pretrain-400k-1B-60k"
ASR_MODEL="openai/whisper-large-v3"

# Test files
TEST_FILES=(
    "/mnt/shared/NTU_TASLM/yc/TASTE-SpokenLM/examples/orig/conditional/cond-gen_1188_133604_000062_000001_cond.wav"
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/ex01_happy_00209.wav"
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/ex04_sad_00311.wav"
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/hifi-tts-dev-clean-speaker6097/004.wav"
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/hifi-tts-dev-clean-speaker6097/012.wav"
    # "/mnt/shared/NTU_TASLM/dienruei/data/emilia-en/data/test/emilia-dataset-train-02207-of-04908-taste.arrow"
)

echo "Running TASTE2 Stage 2 Audio Generation..."
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
    --stage 2

if [ $? -eq 0 ]; then
    echo "✓ Batch processing completed successfully!"
else
    echo "✗ Batch processing failed"
fi
