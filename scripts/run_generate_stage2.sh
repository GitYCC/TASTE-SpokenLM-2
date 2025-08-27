#!/bin/bash

set -e
# TASTE2 Stage 1 Audio Reconstruction Script
# Usage: ./run_generate_stage1.sh

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_MODEL_DIR="openai/whisper-large-v3"

# Define model directories
MODEL_DIRS=(
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM-2/training/checkpoints/taste2-stage2-pretrain-310k-tiny"
)

# Define corresponding output directories
OUTPUT_DIRS=(
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM-2/reconstruction_stage2/taste2-stage2-pretrain-310k-tiny"
)

# Define test files
TEST_FILES=(
    # "/mnt/shared/NTU_TASLM/dienruei/data/emilia-en/data/test/emilia-dataset-train-02207-of-04908-taste.arrow"
    # "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/ex01_happy_00209.wav"
    # "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/ex04_sad_00311.wav"
    "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/hifi-tts-dev-clean-speaker6097/001.wav"
    # "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/hifi-tts-dev-clean-speaker6097/004.wav"
    # "/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/hifi-tts-dev-clean-speaker6097/012.wav"
)

echo "Running TASTE2 Stage 2 Audio Reconstruction..."
echo "Models: ${#MODEL_DIRS[@]}"
echo "Test files: ${#TEST_FILES[@]}"
echo "Total jobs: $((${#MODEL_DIRS[@]} * ${#TEST_FILES[@]}))"
echo ""

job_count=0
total_jobs=$((${#MODEL_DIRS[@]} * ${#TEST_FILES[@]}))

# Loop through each model
for i in "${!MODEL_DIRS[@]}"; do
    MODEL_DIR="${MODEL_DIRS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
    
    echo "Processing model: $(basename "$MODEL_DIR")"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Loop through each test file
    for TEST_FILE in "${TEST_FILES[@]}"; do
        job_count=$((job_count + 1))
        
        echo "[$job_count/$total_jobs] Running reconstruction..."
        echo "  Model: $(basename "$MODEL_DIR")"
        echo "  Output: $OUTPUT_DIR"
        echo "  Test File: $(basename "$TEST_FILE")"
        
        # Run the Python script
        python "$SCRIPT_DIR/generate_stage2.py" \
            --model_dir "$MODEL_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --test_file "$TEST_FILE" \
            --asr_model_dir "$ASR_MODEL_DIR"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Completed successfully"
        else
            echo "  ✗ Failed"
        fi
        echo ""
    done
done

echo "All reconstructions completed!"