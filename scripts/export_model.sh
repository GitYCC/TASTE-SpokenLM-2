#!/bin/bash

# Simple script to export CosyVoice model
# Usage: ./export_model.sh OUTPUT_DIR

export PYTHONIOENCODING=UTF-8
export PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS

python3 scripts/export_cosyvoice_model.py