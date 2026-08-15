# TASTE-SpokenLM-2

**TASTE2: Text-Aligned Speech Modeling and Deployment toward Full-Duplex Voice Interaction**

This repository contains the training and inference code for the TASTE2 spoken language model. It extends the original TASTE method into an incremental dialogue stack for full-duplex voice interaction — processing live speech input, deciding when to speak or yield the floor, and stopping instantly on user interruption, while preserving the pretrained language model's linguistic ability and acoustic fidelity.

<p>
  <a href="https://gitycc.github.io/TASTE2-Homepage/"><img src="https://img.shields.io/badge/Homepage-TASTE2-0A66C2?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Homepage"></a>
  <a href="https://gitycc.github.io/TASTE2-Homepage/assets/taste2-paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-B31B1B?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper"></a>
  <a href="https://github.com/GitYCC/TASTE-Voice-Bot"><img src="https://img.shields.io/badge/VoiceBot-Deployment-181717?style=for-the-badge&logo=github&logoColor=white" alt="VoiceBot Deployment"></a>
  <a href="https://huggingface.co/collections/YC-Chen/taste2"><img src="https://img.shields.io/badge/Model-Checkpoints-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Model Checkpoints"></a>
  <a href="https://huggingface.co/datasets/wilzzzz/paralinguistic_dialogues"><img src="https://img.shields.io/badge/Dataset-paralinguistic__dialogues-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Dataset"></a>
</p>

## Architecture

Each text token is aligned with one continuous audio latent, keeping the text sequence length unchanged while letting acoustic information flow through the entire speech dialogue stack — avoiding interleaving of heterogeneous token streams.

<p align="center">
  <img src="https://gitycc.github.io/TASTE2-Homepage/assets/taste2-model.png" alt="TASTE2 architecture with a Speech Tokenizer, Spoken Language Model, and Speech Detokenizer." width="800">
</p>

1. **Speech Tokenizer (Shared Vocabulary)**: a unified text token vocabulary that removes word-level averaging and language-dependent segmentation
2. **Spoken Language Model (Aligned Prediction)**: the language model predicts one audio latent per text token without extending sequence length
3. **Speech Detokenizer (Streaming Synthesis)**: an incremental speech detokenizer produces S3 units, synthesized into audio via CosyVoice2

See the [paper](https://gitycc.github.io/TASTE2-Homepage/assets/taste2-paper.pdf) for full methodology and experiments.

## Repository Layout

```
taste_speech/       # core model package (audio encoder/quantizer/segmenter, CosyVoice modules, TASTE2 SLM, etc.)
training/           # Stage 1 (audio tokenizer/detokenizer) and Stage 2 (Spoken LM) training code & configs
scripts/            # inference, data processing, validation, and export (ONNX / TensorRT-LLM) scripts
docs/               # training manual and data format specs
docker/             # containerized training/inference environments (cuda124, cuda13)
audio_samples/      # sample audio for inference testing
results/            # inference output results
```

## Setup

A containerized environment with all training/inference dependencies is recommended:

```bash
bash docker/cuda13/build.sh   # build image (CUDA 13 + TensorRT-LLM)
bash docker/cuda13/run.sh     # start container
```

`docker/cuda124/` provides a CUDA 12.4 alternative. For local development, follow the standard convention of creating a virtualenv with `uv venv .venv` and installing `docker/*/requirements.txt`.

## Usage

### Training

Two stages, detailed in [training/docs/TRAINING.md](training/docs/TRAINING.md):

- **Stage 1**: train the speech tokenizer / detokenizer (`training/jobs/run_stage1_*.sh`)
- **Stage 2**: train the core Spoken Language Model (`training/jobs/run_stage2_*.sh`), supporting 1B/2B/8B scales and LoRA variants

### Inference (speech generation)

```bash
bash scripts/run_generate_stage2.sh
```

This wraps `scripts/generate_audio.py`, which supports `--stage {1,2,sft}` and can batch-process multiple input audio files into synthesized speech.

### Model Export

- `scripts/export_flow_estimator_onnx.py` + `scripts/build_flow_estimator_onnx.sh`: export the flow estimator to ONNX
- `scripts/export_cosyvoice_model.py` + `scripts/build_cosyvoice_trtllm.sh`: export and build the TensorRT-LLM engine

## Results

| Metric | Result |
|---|---|
| LLaMA-Questions accuracy | 56.3% (retains 87.6% of the text-reference accuracy) |
| Response latency after user interruption | 0.060s |
| Full-Duplex-Bench v1.0 | Best on 4 of 5 tasks, tied-best on 1 (727 samples) |
| First-audio latency (TensorRT-optimized) | 2.701s (22% faster than unoptimized) |

## Authors

Yi-Chang Chen, Chun Wei Chen, Dien-Ruei Wu, Jie Lin, Hung-yi Lee, Da-Shan Shiu
(MediaTek Research / National Taiwan University)
