#!/bin/bash

# TASTE2 Audio Generation Script - Stage 2 Batch Processing
# Usage: ./run_generate_stage2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MODEL_DIR="/mount/models/TASTE2-8B-EN"
OUTPUT_DIR="./results/stage2/TASTE2_8B_EN"
# for en
ASR_MODEL="openai/whisper-large-v3"
# for zh
# ASR_MODEL="MediaTek-Research/Breeze-ASR-25"

# Test files
TEST_FILES=(
    "./audio_samples/stage2_EN/cont_en_001.wav"
    "./audio_samples/stage2_EN/cont_en_002.wav"
    "./audio_samples/stage2_EN/cont_en_003.wav"
    "./audio_samples/stage2_EN/cont_en_004.wav"
    "./audio_samples/stage2_EN/cont_en_005.wav"
    # for zh
    # "./audio_samples/stage2_ZH/hylee_intrainifa_cut.m4a"
    # "./audio_samples/stage2_ZH/jensen_cut.m4a"
    # "./audio_samples/stage2_ZH/taiwan_area_cut.m4a"
    # "./audio_samples/stage2_ZH/water_margin_cut.m4a"
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
