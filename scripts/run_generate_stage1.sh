#!/bin/bash

# TASTE2 Audio Generation Script - Stage 1 Batch Processing
# Usage: ./run_generate_stage1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MODEL_DIR="/mnt/shared/NTU_TASLM/yc/models/TASTE2_8B_ZH"
OUTPUT_DIR="./results/stage1/TASTE2_ZH/zh"

# for en
ASR_MODEL="openai/whisper-large-v3"
# for zh
# ASR_MODEL="MediaTek-Research/Breeze-ASR-25"

# Test files
TEST_FILES=(
    "./audio_samples/stage1_EN/121-127105-0007.flac"
    "./audio_samples/stage1_EN/1995-1837-0007.flac"
    "./audio_samples/stage1_EN/2830-3980-0032.flac"
    "./audio_samples/stage1_EN/4446-2273-0032.flac"
    "./audio_samples/stage1_EN/8224-274384-0000.flac"
    "./audio_samples/stage1_EN/ex01_happy_00209.wav"
    "./audio_samples/stage1_EN/ex04_00350_sad_happy.wav"
    "./audio_samples/stage1_EN/ex04_sad_00311.wav"
    # for zh
    # "./audio_samples/stage1_ZH/hylee_intrainifa.wav"
    # "./audio_samples/stage1_ZH/jensen.m4a"
    # "./audio_samples/stage1_ZH/sometime.mp3"
    # "./audio_samples/stage1_ZH/taiwan_area.wav"
    # "./audio_samples/stage1_ZH/water_margin.wav"
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