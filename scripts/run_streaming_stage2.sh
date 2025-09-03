#!/bin/bash

# TASTE2 Stage 2 Streaming Generation Test Script
# This script tests the inference_bistream functionality with SLM

set -e  # Exit on any error

# Configuration
MODEL_DIR="/mnt/shared/NTU_TASLM/models/taste2_8B_final"
OUTPUT_DIR="./reconstruction_stage2_streaming/taste2_8B_final"
# TEST_FILE="/mnt/shared/NTU_TASLM/dienruei/TASTE-SpokenLM/examples/orig/hifi-tts-dev-clean-speaker6097/004.wav"
TEST_FILE="/mnt/shared/NTU_TASLM/yc/TASTE-SpokenLM/examples/orig/conditional/cond-gen_1188_133604_000062_000001_cond.wav"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "TASTE2 Stage 2 Streaming Generation Test"
echo "========================================"
echo "Model Directory: $MODEL_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Test File: $TEST_FILE"
echo "Script Directory: $SCRIPT_DIR"
echo ""

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory does not exist: $MODEL_DIR"
    exit 1
fi

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file does not exist: $TEST_FILE"
    exit 1
fi

# Create output directory
echo "📁 Creating output directory..."
mkdir -p "$OUTPUT_DIR"

# Change to parent directory (TASTE-SpokenLM-2 root)
cd "$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Stage 2 streaming generation..."
echo "   This will test the SLM + inference_bistream() functionality"
echo "   and print out all s3 tokens for verification."
echo ""

# Run the streaming generation using streaming interface
python ./scripts/generate_audio_stream.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --test_files "$TEST_FILE" \
    --asr_model_dir "openai/whisper-large-v3" \
    --stage 2

echo ""
echo "✅ Streaming generation test completed!"
echo "📂 Check the output directory for:"
echo "   - Individual audio chunks: *_chunk_*.wav"
echo "   - Final crossfaded audio: *_streaming_final_crossfaded.wav"
echo "   - Single audio (all tokens): *_single_from_all_tokens.wav"
echo "   - Streaming summary: *_streaming_summary.json (includes generated text)"
echo "   - Batch results: batch_results.json"
echo "   - Original audio: *_original.*"
echo ""
echo "Output location: $(realpath "$OUTPUT_DIR")"