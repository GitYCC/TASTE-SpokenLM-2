#!/bin/bash

export PYTHONIOENCODING=UTF-8;

# Deactivate any active conda environment
conda deactivate 2>/dev/null || true

# Source conda initialization
source /opt/conda/etc/profile.d/conda.sh

# Activate cosyvoice environment
conda activate cosyvoice

# Install required packages if not already installed
echo "Installing required packages (einx, peft, wandb)..."
pip install einx peft wandb
pip install 'transformers==4.49.0' 'peft==0.17.0'

# Install the project in editable mode
echo "Installing project in editable mode..."
cd /home/chenwils/TASTE-SpokenLM-2
pip install -e .

# Login to wandb (will prompt for API key if not authenticated)
echo "Checking wandb authentication..."
wandb login || echo "WandB already authenticated"

# train slm
yaml_name="sft"

export CUDA_VISIBLE_DEVICES="0" #"0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1987
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
stage2_checkpoint=/home/chenwils/TASTE_models/TASTE2-8B-EN

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
  $merge_flag \
  2>&1 | tee training/logs/${yaml_name}_$(date +%Y%m%d_%H%M%S).log