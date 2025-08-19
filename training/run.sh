#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
source training/path.sh || exit 1;

stage=-1
stop_stage=3

data_url=www.openslr.org/resources/60
data_dir=./training/data/raw/
pretrained_model_dir=./training/pretrained_models/CosyVoice2-0.5B

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "Data Download"
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    training/local/download_and_untar.sh ${data_dir} ${data_url} ${part}
  done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    mkdir -p training/data/$x
    python training/local/prepare_data.py --src_dir $data_dir/LibriTTS/$x --des_dir training/data/$x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in training/data/$x dir"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    training/tools/extract_embedding.py --dir training/data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in training/data/$x dir"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    training/tools/extract_speech_token.py --dir training/data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    mkdir -p training/data/$x/parquet
    training/tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir training/data/$x \
      --des_dir training/data/$x/parquet
  done
fi

# train llm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat training/data/{train-clean-100,train-clean-360,train-other-500}/parquet/data.list > training/data/train.data.list
  cat training/data/{dev-clean,dev-other}/parquet/data.list > training/data/dev.data.list
  # NOTE will update llm/hift training later

  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    CosyVoice/cosyvoice/bin/train.py \
    --train_engine $train_engine \
    --config training/conf/cosyvoice2.yaml \
    --train_data training/data/train.data.list \
    --cv_data training/data/dev.data.list \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --model llm \
    --checkpoint $pretrained_model_dir/llm.pt \
    --model_dir training/exp/ \
    --tensorboard_dir training/tensorboard/ \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --use_amp \
    --deepspeed_config training/conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer

fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  decode_checkpoint=training/exp/${model}.pt
  echo "do model average and final checkpoint is $decode_checkpoint"
  python CosyVoice/cosyvoice/bin/average_model.py \
    --dst_model $decode_checkpoint \
    --src_path training/exp/  \
    --num ${average_num} \
    --val_best
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python CosyVoice/cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python CosyVoice/cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi