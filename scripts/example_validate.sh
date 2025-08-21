#!/bin/bash

# Example script for running model validation
# This script only runs validation (cv function), no training
# Can be run from any directory

# Set paths (absolute paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Example parameters - adjust these for your setup
MODEL="llm"  # or "flow", "hift", "hifigan" 
CONFIG_FILE="/mnt/shared/NTU_TASLM/yc/TASTE-SpokenLM-2/training/pretrained_models/CosyVoice2-0.5B/cosyvoice2.yaml"
CV_DATA_FILE="/mnt/shared/NTU_TASLM/yc/prepared_dataset/dev.data.list"  # Adjust path to your validation data
CHECKPOINT_PATH="/mnt/shared/NTU_TASLM/yc/TASTE-SpokenLM-2/training/pretrained_models/CosyVoice2-0.5B/llm.pt"  # Path to your trained model checkpoint
OUTPUT_DIR="./training/validation_results"
TENSORBOARD_DIR="./training/tensorboard/validation"

# Optional parameters
QWEN_PRETRAIN_PATH="/mnt/shared/NTU_TASLM/yc/TASTE-SpokenLM-2/training/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN/"

# GPU and batch size configuration
NUM_GPUS=8          # Set to number of GPUs you want to use
BATCH_SIZE=512        # Batch size per GPU for validation

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TENSORBOARD_DIR"

echo "Starting model validation..."
echo "Model: $MODEL"
echo "Config: $CONFIG_FILE"
echo "CV data: $CV_DATA_FILE"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"

# Set environment variables
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single-GPU validation..."
    export RANK=0
    export WORLD_SIZE=1
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500
    
    # Single GPU validation
    python "$SCRIPT_DIR/validate_model.py" \
        --train_engine torch_ddp \
        --model "$MODEL" \
        --config "$CONFIG_FILE" \
        --cv_data "$CV_DATA_FILE" \
        --checkpoint "$CHECKPOINT_PATH" \
        --model_dir "$OUTPUT_DIR" \
        --tensorboard_dir "$TENSORBOARD_DIR" \
        --qwen_pretrain_path "$QWEN_PRETRAIN_PATH" \
        --batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --num_workers 4 \
        --prefetch 100 \
        --pin_memory
else
    echo "Running multi-GPU validation with $NUM_GPUS GPUs..."
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500
    
    # Multi-GPU validation using torchrun
    torchrun --nproc_per_node="$NUM_GPUS" "$SCRIPT_DIR/validate_model.py" \
        --train_engine torch_ddp \
        --model "$MODEL" \
        --config "$CONFIG_FILE" \
        --cv_data "$CV_DATA_FILE" \
        --checkpoint "$CHECKPOINT_PATH" \
        --model_dir "$OUTPUT_DIR" \
        --tensorboard_dir "$TENSORBOARD_DIR" \
        --qwen_pretrain_path "$QWEN_PRETRAIN_PATH" \
        --batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --num_workers 4 \
        --prefetch 100 \
        --pin_memory
fi

echo "Model validation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "TensorBoard logs: $TENSORBOARD_DIR"