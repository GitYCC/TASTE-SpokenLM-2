#!/usr/bin/env python3

import torch
import argparse
import sys

def print_pt_keys(file_path):
    """Print all keys in a PyTorch checkpoint file with tensor dtype and size."""
    try:
        checkpoint = torch.load(file_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print(f"Keys in {file_path}:")
            for key in sorted(checkpoint.keys()):
                value = checkpoint[key]
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.dtype}, size={tuple(value.shape)}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        else:
            print(f"File {file_path} contains a {type(checkpoint)} object, not a dictionary")
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Print all keys in PyTorch checkpoint files')
    parser.add_argument('files', nargs='+', help='PyTorch checkpoint files (.pt, .pth)')
    
    args = parser.parse_args()
    
    exit_code = 0
    for file_path in args.files:
        if len(args.files) > 1:
            print(f"\n{'='*50}")
        exit_code |= print_pt_keys(file_path)
    
    return exit_code

if __name__ == '__main__':
    sys.exit(main())