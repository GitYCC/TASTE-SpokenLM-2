
#!/usr/bin/env python3
"""
Filter PyTorch checkpoint files by removing training metadata and extracting model state dict.

This script loads a PyTorch checkpoint file, extracts the model state dictionary while
filtering out training-related metadata (epoch, step, optimizer state, etc.), and saves
the cleaned checkpoint to a new file.
"""

import argparse
import sys
import torch
from pathlib import Path


def filter_checkpoint(input_path: str, output_path: str) -> None:
    """
    Filter a PyTorch checkpoint file to extract only the model state dict.
    
    Args:
        input_path: Path to the input checkpoint file
        output_path: Path where the filtered checkpoint will be saved
    """
    try:
        print(f"Loading checkpoint from: {input_path}")
        llm_checkpoint = torch.load(input_path, map_location='cpu')
        
        if isinstance(llm_checkpoint, dict) and 'model_state_dict' in llm_checkpoint:
            llm_state_dict = llm_checkpoint['model_state_dict']
            print("Extracted 'model_state_dict' from checkpoint")
        elif isinstance(llm_checkpoint, dict):
            # Filter out training metadata keys
            metadata_keys = ['epoch', 'step', 'optimizer_state_dict', 'scheduler_state_dict', 
                           'lr_scheduler_state_dict', 'scaler_state_dict', 'loss', 'best_loss']
            llm_state_dict = {k: v for k, v in llm_checkpoint.items() 
                            if k not in metadata_keys}
            print(f"Filtered out training metadata keys: {[k for k in llm_checkpoint.keys() if k in metadata_keys]}")
        else:
            llm_state_dict = llm_checkpoint
            print("Checkpoint is already a state dict")
        
        print(f"Saving filtered checkpoint to: {output_path}")
        torch.save(llm_state_dict, output_path)
        print(f"Successfully saved filtered checkpoint with {len(llm_state_dict)} keys")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing checkpoint: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Filter PyTorch checkpoint files")
    parser.add_argument("input_path", help="Path to the input checkpoint file (.pt)")
    parser.add_argument("output_path", nargs='?', help="Path for the filtered checkpoint file (optional)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # Generate output path if not provided
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_path.with_stem(f"{input_path.stem}_filtered")
    
    # Validate input file
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not input_path.suffix == '.pt':
        print("Warning: Input file does not have .pt extension")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    filter_checkpoint(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
