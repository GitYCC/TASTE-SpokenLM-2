#!/usr/bin/env python3
"""
Script to copy 'llm' weights from taste2_8B_final/llm.pt to taste2_stage2_8B_lora/slm.pt
as 'taste_stage1' key and save the updated file.
"""

import torch
import os

def copy_llm_weights(source_path, dest_path, output_path):
    print(f"Loading source model from: {source_path}")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    source_checkpoint = torch.load(source_path, map_location='cpu')
    print(f"Loaded source model with {len(source_checkpoint)} keys")
    
    print(f"Loading destination model from: {dest_path}")
    if not os.path.exists(dest_path):
        raise FileNotFoundError(f"Destination file not found: {dest_path}")
    
    dest_checkpoint = torch.load(dest_path, map_location='cpu')
    print(f"Loaded destination model with {len(dest_checkpoint)} keys")
    
    # Start with destination checkpoint to preserve existing keys
    new_checkpoint = dest_checkpoint.copy()
    
    # Add source weights with taste_stage1. prefix
    for key, weight in source_checkpoint.items():
        new_key = f"taste_stage1.{key}"
        new_checkpoint[new_key] = weight
        print(f"Copying: {key} -> {new_key}")
    
    print(f"\nSaving updated model with {len(new_checkpoint)} keys to: {output_path}")
    torch.save(new_checkpoint, output_path)
    print("Successfully saved updated model to new file")

if __name__ == "__main__":
    source_path = "/mnt/shared/NTU_TASLM/yc/models/TASTE2_8B_ZH/llm.pt"
    dest_path = "/mnt/shared/NTU_TASLM/yc/models/TASTE2_8B_EN/slm.pt"
    output_path = "/mnt/shared/NTU_TASLM/yc/models/TASTE2_8B_ZH/slm_initialized.pt"

    copy_llm_weights(source_path, dest_path, output_path)
