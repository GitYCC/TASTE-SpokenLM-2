# TASTE SpokenLM-2 Training Manual

This manual provides comprehensive instructions for training the TASTE SpokenLM-2 model, which consists of two main training stages and additional variants.

## Overview

TASTE SpokenLM-2 is a two-stage training pipeline:

- **Stage 1 (`llm`)**: Train the audio tokenizer and audio detokenizer components. This stage focuses on learning speech-to-discrete-token mapping and discrete-token-to-speech reconstruction capabilities.
- **Stage 2 (`slm`)**: Train the central Spoken Language Model component. This stage trains the core language model that processes both text and speech tokens for end-to-end spoken language understanding and generation.

### Training Variants

#### Stage 1 Variants:
- `taste2_stage1_vq`: Standard Stage 1 with vector quantization
- `taste2_stage1_textonly`: Text-only training variant

#### Stage 2 Variants:
- `taste2_stage2_8B_lora`: Stage 2 with 8B model using LoRA
- `taste2_stage2_2B_lora`: Stage 2 with 2B model using LoRA  

## Prerequisites

### Required Packages

The training scripts depend on several packages:
- `torch` (with CUDA support)
- `torchaudio`
- `deepspeed` (for distributed training)
- `hyperpyyaml` (for configuration management)
- `wandb` (for experiment tracking)
- `pyarrow` (for parquet data loading)
- `pyworld` (for audio processing)

### Pretrained Models

You need to download and place the following pretrained models in `training/pretrained_models/`:

#### Required for Stage 1:
- **CosyVoice-BlankEN**: Place in `training/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN/`
- **Whisper Large-v3**: Will be downloaded automatically to `training/pretrained_models/distil-large-v3/`
- **Stage 1 Initial Model**: Place in `training/pretrained_models/Taste2-Stage1-Init/`

#### Required for Stage 2:
- **Qwen2-7B**: For 8B model variants, place in your specified path (e.g., `/mnt/shared/NTU_TASLM/yc/TASTE-SpokenLM-2/training/pretrained_models/Qwen2-7B`)
- **Stage 1 Checkpoint**: Trained Stage 1 model checkpoint

## Directory Structure

```
training/
├── train.py                    # Main training script
├── conf/                       # Configuration files
│   ├── taste2_stage1_vq.yaml
│   ├── taste2_stage2_8B_lora.yaml
│   ├── ds_stage2.json          # DeepSpeed configuration
│   └── ...
├── jobs/                       # Training job scripts
│   ├── run_stage1_vq.sh
│   ├── run_stage2_8B_lora.sh
│   └── ...
├── docs/                       # Documentation
│   ├── TRAINING.md             # This file
│   └── WANDB_INTEGRATION.md    # Weights & Biases guide
├── exp/                        # Experiment outputs
├── logs/                       # Training logs
├── pretrained_models/          # Pretrained model storage
└── tensorboard/                # TensorBoard logs
```

## Data Preparation

### Dataset Format

The training data must be in Parquet format with the following columns:

- `utt`: Unique utterance identifier
- `audio_data`: Raw audio bytes data
- `text_token`: Tokenized text (list of token IDs)
- `speech_token`: Speech tokens (if applicable)
- Additional metadata as needed

### Data Location

Update the data paths in your training scripts:
- **Training Data**: Set `--train_data` to your training data list file
- **Validation Data**: Set `--cv_data` to your validation data list file

Example data list file format:
```
/path/to/train_data_part1.parquet
/path/to/train_data_part2.parquet
...
```

## Stage 1: LLM Training

Stage 1 trains the Language Learning Model with speech tokenization capabilities.

### Configuration

Key parameters in Stage 1 configs (`training/conf/taste2_stage1_*.yaml`):

```yaml
# Model parameters
llm_input_size: 896
llm_output_size: 896
sample_rate: 24000
token_frame_rate: 25

# Training parameters
lr: 5e-4                    # Learning rate
max_frames_in_batch: 6000   # Batch size in frames
accum_grad: 1               # Gradient accumulation steps
save_per_step: 10000        # Save checkpoint every N steps

# Quantization (VQ vs NoVQ)
quantization_on: true       # Set to false for NoVQ variant
```

### Running Stage 1

#### Standard VQ Training
```bash
cd training
bash jobs/run_stage1_vq.sh
```

#### Key Arguments in Stage 1 Scripts:

- `--model llm`: Train the LLM component
- `--config`: Configuration file path
- `--train_data`: Training data list file
- `--cv_data`: Cross-validation data list file  
- `--qwen_pretrain_path`: Path to pretrained Qwen model
- `--checkpoint`: Initial checkpoint (if continuing training)
- `--model_dir`: Output directory for trained models
- `--train_engine`: `torch_ddp` or `deepspeed`

### Stage 1 Checkpoints

Stage 1 training will save checkpoints in:
```
training/exp/taste2_stage1_*/
├── epoch_X_step_Y.pt       # Regular checkpoints
├── init.pt                 # Initial checkpoint
└── config.yaml             # Saved configuration
```

## Stage 2: SLM Training

Stage 2 trains the Speech Language Model for end-to-end spoken language modeling.

### Configuration

Key parameters in Stage 2 configs (`training/conf/taste2_stage2_*.yaml`):

```yaml
# Model parameters
slm_input_size: 3584        # SLM input dimension
slm_output_size: 3584       # SLM output dimension
qwen_pretrain_path_for_slm: '/path/to/Qwen2-7B'

# Training parameters  
lr: 5e-5                    # Lower learning rate for Stage 2
max_frames_in_batch: 1000   # Smaller batch size
accum_grad: 4               # Higher gradient accumulation

# LoRA configuration (for LoRA variants)
use_lora: true
lora_config:
    lora_r: 64
    lora_alpha: 128
    lora_dropout: 0.05
```

### Running Stage 2

#### 8B Model with LoRA
```bash
cd training
bash jobs/run_stage2_8B_lora.sh
```

#### Standard 2B Model
```bash
cd training
bash jobs/run_stage2.sh
```

### Key Arguments in Stage 2 Scripts:

- `--model slm`: Train the SLM component

### Stage 2 Prerequisites

Before running Stage 2:
1. Complete Stage 1 training
2. Update `path_reload_taste_stage1` in the Stage 2 config to point to your Stage 1 checkpoint
3. Ensure Stage 1 checkpoint path is accessible

## Training Configurations


### Parameter Freezing

Both stages support parameter freezing via `freeze_params` configuration:

```yaml
freeze_params:
    enabled: true
    patterns:
        - "taste_tokenizer\\.audio_joint_encoder_segmenter\\.audio_encoder\\..+"  # Stage 1
        - "taste_stage1\\..+"  # Stage 2
```

## Monitoring and Logging

### Weights & Biases Integration

Both stages support Weights & Biases logging:

```yaml
train_conf:
    use_wandb: true
    wandb_project: "stage1-training"  # or "stage2-training" 
    wandb_entity: "TASTE"
    wandb_run_name: "my_experiment_name"
    wandb_tags: "stage1"  # or "stage2"
```

### TensorBoard Logging

TensorBoard logs are saved to:
```
training/tensorboard/taste2_stage1_*/
training/tensorboard/taste2_stage2_*/
```

View with:
```bash
tensorboard --logdir training/tensorboard/
```

### Training Logs

Detailed training logs are saved to:
```
training/logs/taste2_stage*_YYYYMMDD_HHMMSS.log
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
- Reduce `max_frames_in_batch` in config
- Increase `accum_grad` to maintain effective batch size  


#### Loss Not Decreasing
- Verify data quality and preprocessing
- Check learning rate (may need adjustment)
- Ensure proper checkpoint loading
- Monitor gradient norms for vanishing/exploding gradients


## Advanced Topics

### Custom Data Processing

To add custom data processors:

1. Define your processor function in `taste_speech/taste2/cosyvoice/dataset/processor.py`
2. Add it to the data pipeline in your config:

```yaml
my_processor: !name:my_module.my_processor_function
    param1: value1

data_pipeline: [
    !ref <parquet_opener>,
    !ref <my_processor>,
    # ... other processors
]
```

### Mixed Precision Training

Enable automatic mixed precision with:
```bash
--use_amp
```

Add to your training script arguments.


### Checkpoint Management

#### Resuming Training
```bash
--checkpoint /path/to/checkpoint.pt
```

### Multi-Node Training

For multi-node distributed training:

1. Set up shared filesystem accessible from all nodes
2. Update `--nnodes` and network configuration in torchrun
3. Ensure consistent CUDA_VISIBLE_DEVICES across nodes
4. Use appropriate network backend (`nccl` for GPU communication)

### Performance Optimization

#### Data Loading
- Increase `num_workers` for faster data loading
- Adjust `prefetch` parameter for data prefetching
- Use `pin_memory` for faster GPU transfer

#### Memory Optimization
- Use DeepSpeed ZeRO for large models
- Enable gradient checkpointing
- Optimize batch size vs. accumulation steps balance

#### Training Speed
- Optimize data pipeline (remove unnecessary processors)
- Use appropriate precision (fp16/bf16 when supported)
