#!/bin/bash

export PYTHONIOENCODING=UTF-8;

# train llm
yaml_name="taste2_stage1_vq"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
pretrained_model_dir=training/pretrained_models/Taste2-Stage1-Init/

echo "Run train. We only support llm traning for now"
if [ $train_engine == 'deepspeed' ]; then
  echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
fi

torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
  training/train.py \
  --train_engine $train_engine \
  --config training/conf/$yaml_name.yaml \
  --train_data /mnt/shared/NTU_TASLM/yc/prepared_dataset/train.data.list \
  --cv_data /mnt/shared/NTU_TASLM/yc/prepared_dataset/dev.data.list \
  --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
  --model llm \
  --checkpoint $pretrained_model_dir/llm.pt \
  --model_dir training/exp/$yaml_name/ \
  --tensorboard_dir training/tensorboard/$yaml_name/ \
  --ddp.dist_backend $dist_backend \
  --num_workers ${num_workers} \
  --prefetch ${prefetch} \
  --pin_memory \
  --deepspeed_config training/conf/ds_stage2.json \
  --deepspeed.save_states model+optimizer
