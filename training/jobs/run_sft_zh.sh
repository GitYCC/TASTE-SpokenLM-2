#!/bin/bash

export PYTHONIOENCODING=UTF-8;
# NOTE: PYTHONNOUSERSITE must stay commented out!
# Conda env is read-only, so packages install to ~/.local/ which must be accessible

# Deactivate any active conda environment
conda deactivate 2>/dev/null || true

# Source conda initialization
source /opt/conda/etc/profile.d/conda.sh

# Activate cosyvoice environment
conda activate cosyvoice

# Install required packages to user site-packages (conda env is read-only)
echo "========================================="
echo "Installing required packages to user site-packages..."
echo "========================================="

# CRITICAL: Install numpy<2 FIRST to maintain binary compatibility
# The conda env has onnxruntime/matplotlib compiled against NumPy 1.x
# Installing NumPy 2.x breaks everything!
echo "Step 1: Installing numpy<2 (required for binary compatibility)..."
pip install --user 'numpy<2,>=1.20' --force-reinstall || {
    echo "ERROR: Failed to install numpy"
    exit 1
}

# Install transformers with version constraints to avoid dependency conflicts
# Constrain fsspec and packaging for lightning 2.2.4 compatibility
echo "Step 2: Installing transformers with compatibility constraints..."
pip install --user --upgrade \
    'transformers==4.49.0' \
    'fsspec[http]>=2022.5.0,<2025.0' \
    'packaging>=20.0,<25.0' || {
    echo "ERROR: Failed to install transformers"
    exit 1
}

# Install peft and other dependencies
echo "Step 3: Installing peft, einx, wandb..."
pip install --user 'peft==0.17.0' einx wandb || {
    echo "ERROR: Failed to install peft/einx/wandb"
    exit 1
}

# Verify critical imports work
echo "========================================="
echo "Verifying package installation..."
echo "========================================="

python -c "import numpy; print('✓ numpy:', numpy.__version__); assert numpy.__version__.startswith('1.'), 'ERROR: NumPy 2.x detected!'" || exit 1

python -c "import transformers; print('✓ transformers:', transformers.__version__, 'at', transformers.__file__)" || exit 1

python -c "import peft; print('✓ peft:', peft.__version__)" || exit 1

python -c "import einx; print('✓ einx imported successfully')" || exit 1

python -c "import onnxruntime; print('✓ onnxruntime:', onnxruntime.__version__)" || {
    echo "WARNING: onnxruntime check failed (may not be critical)"
}

echo "========================================="
echo "✓ All packages verified successfully!"
echo "========================================="

# Install the project in editable mode
echo "========================================="
echo "Installing project in editable mode..."
echo "========================================="
cd /home/chenwils/TASTE-SpokenLM-2
pip install -e .

# Login to wandb (will prompt for API key if not authenticated)
echo "========================================="
echo "Checking wandb authentication..."
echo "========================================="
which wandb && wandb login || echo "WandB not found or already authenticated"

echo "========================================="
echo "Package installation complete!"
echo "========================================="

#!/bin/bash

export PYTHONIOENCODING=UTF-8;

# train slm[]
yaml_name="sft_zh"

export CUDA_VISIBLE_DEVICES="0" #"0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1987
dist_backend="nccl"
num_workers=2
prefetch=80  # Balanced: 20 batches in memory (2 workers × 10)
train_engine=torch_ddp
stage2_checkpoint=/home/chenwils/TASTE_models/TASTE2-8B-ZH

# Set to "true" to merge Stage 2 LoRA into base and train fresh LoRA for SFT
# Set to "false" to continue training Stage 2 LoRA (default behavior)
merge_lora_before_sft=true

echo "Run train. SFT training SLM"
if [ $train_engine == 'deepspeed' ]; then
  echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
fi

# Create log directory if it doesn't exist
mkdir -p training/logs

# Build command with optional merge flag
merge_flag=""
if [ "$merge_lora_before_sft" = "true" ]; then
  merge_flag="--merge_lora_before_sft"
  echo "INFO: merge_lora_before_sft=true - Will merge Stage 2 LoRA and train fresh LoRA for SFT"
else
  echo "INFO: merge_lora_before_sft=false - Will continue training Stage 2 LoRA (default)"
fi

# Run training with full output logging
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1235" \
  training/train.py \
  --train_engine $train_engine \
  --config training/conf/$yaml_name.yaml \
  --train_data training/datalist/train_1.sft.data.list \
  --cv_data training/datalist/dev_1.sft.data.list \
  --model slm \
  --checkpoint $stage2_checkpoint/slm.pt \
  --model_dir training/exp/$yaml_name/ \
  --tensorboard_dir training/tensorboard/$yaml_name/ \
  --ddp.dist_backend $dist_backend \
  --num_workers ${num_workers} \
  --prefetch ${prefetch} \
  --pin_memory \
  --deepspeed_config training/conf/ds_stage2.json \
  --deepspeed.save_states model+optimizer \
  2>&1 | tee training/logs/${yaml_name}_$(date +%Y%m%d_%H%M%S).log