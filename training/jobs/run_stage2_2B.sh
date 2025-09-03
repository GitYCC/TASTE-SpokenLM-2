#!/bin/bash

export PYTHONIOENCODING=UTF-8;

# train slm
yaml_name="taste2_stage2_2B"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1988
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

echo "Run train. Training SLM for stage 2"
if [ $train_engine == 'deepspeed' ]; then
  echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
fi

# Create log directory if it doesn't exist
mkdir -p training/logs

# Run training with full output logging
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1236" \
  training/train.py \
  --train_engine $train_engine \
  --config training/conf/$yaml_name.yaml \
  --train_data /mnt/shared/NTU_TASLM/yc/prepared_dataset/train.data.list \
  --cv_data /mnt/shared/NTU_TASLM/yc/prepared_dataset/dev.data.list \
  --qwen_pretrain_path $stage1_checkpoint/CosyVoice-BlankEN \
  --model slm \
  --checkpoint '' \
  --model_dir training/exp/$yaml_name/ \
  --tensorboard_dir training/tensorboard/$yaml_name/ \
  --ddp.dist_backend $dist_backend \
  --num_workers ${num_workers} \
  --prefetch ${prefetch} \
  --pin_memory \
  --deepspeed_config training/conf/ds_stage2.json \
  --deepspeed.save_states model+optimizer \
  2>&1 | tee training/logs/${yaml_name}_$(date +%Y%m%d_%H%M%S).log