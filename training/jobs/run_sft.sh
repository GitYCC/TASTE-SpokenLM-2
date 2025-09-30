#!/bin/bash

export PYTHONIOENCODING=UTF-8;

# train slm
yaml_name="sft"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1987
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
stage2_checkpoint=/mnt/shared/p01/yc/models/TASTE2-8B-EN

echo "Run train. SFT training SLM"
if [ $train_engine == 'deepspeed' ]; then
  echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
fi

# Create log directory if it doesn't exist
mkdir -p training/logs

# Run training with full output logging
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1235" \
  training/train.py \
  --train_engine $train_engine \
  --config training/conf/$yaml_name.yaml \
  --train_data /mnt/shared/p01/wilz/TASTE-SpokenLM-2/train.data.list \
  --cv_data /mnt/shared/p01/wilz/TASTE-SpokenLM-2/dev.data.list\
  --qwen_pretrain_path $stage1_checkpoint/CosyVoice-BlankEN \
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