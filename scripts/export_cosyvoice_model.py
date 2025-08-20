#!/usr/bin/env python3

import os
import argparse
import shutil
import torch
import glob
from hyperpyyaml import load_hyperpyyaml

def load_config(config_path):
    """Load configuration file with overrides to only instantiate LLM model"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    with open(config_path, 'r') as f:
        configs = load_hyperpyyaml(f)
    
    if 'llm' not in configs or configs['llm'] is None:
        raise ValueError("LLM component not found in configuration")
    
    return configs



def generate_model_report(llm_path, output_dir):
    """Generate a detailed report of the LLM model structure"""
    report_path = os.path.join(output_dir, 'llm_model_report.txt')
    
    if not os.path.exists(llm_path):
        with open(report_path, 'w') as f:
            f.write(f"Error: Model file {llm_path} not found\n")
        return report_path
    
    try:
        state_dict = torch.load(llm_path, map_location='cpu')
        
        with open(report_path, 'w') as f:
            f.write(f"=== LLM Model Report for {llm_path} ===\n\n")
            f.write(f"Total parameters: {len(state_dict)} tensors\n\n")
            
            for i, (key, tensor) in enumerate(state_dict.items(), 1):
                f.write(f"{i:3d}. Key: {key}\n")
                f.write(f"     Shape: {list(tensor.shape)}\n")
                f.write(f"     Size: {tensor.numel():,} elements\n")
                f.write(f"     Dtype: {tensor.dtype}\n")
                
                # Show brief numerical values (batch idx=1, length < 3)
                if tensor.numel() > 0:
                    flat_tensor = tensor.flatten()
                    if len(flat_tensor) >= 3:
                        sample_vals = flat_tensor[:3].tolist()
                        f.write(f"     Sample values: [{sample_vals[0]:.6f}, {sample_vals[1]:.6f}, {sample_vals[2]:.6f}]\n")
                    else:
                        sample_vals = flat_tensor.tolist()
                        formatted_vals = [f"{val:.6f}" for val in sample_vals]
                        f.write(f"     All values: [{', '.join(formatted_vals)}]\n")
                else:
                    f.write("     Sample values: [empty tensor]\n")
                f.write("\n")
            
            f.write("=== End of Report ===\n")
        
        print(f"Model report saved to: {report_path}")
        return report_path
        
    except Exception as e:
        with open(report_path, 'w') as f:
            f.write(f"Error reading model file: {e}\n")
        print(f"Error reading model file: {e}")
        return report_path
    

def reload_llm_pt(model, old_llm_pt=None):
    """Reload LLM weights and return success status"""
    if old_llm_pt is not None:
        if os.path.exists(old_llm_pt):
            try:
                state_dict = torch.load(old_llm_pt, map_location='cpu')
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print(f"✓ Successfully loaded LLM weights from: {old_llm_pt}")
                missing_keys = [x for x in missing_keys if not(x.startswith('taste_tokenizer.') or x.startswith('taste_decoder_mixer'))]
                if missing_keys:
                    print(f"  Warning: Missing keys: {len(missing_keys)} keys")
                    print(f"  Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"  Warning: Unexpected keys: {len(unexpected_keys)} keys")
                return True
            except Exception as e:
                print(f"✗ Failed to load LLM weights from {old_llm_pt}: {e}")
                return False
        else:
            print(f"✗ LLM weights file not found: {old_llm_pt}")
            return False
    else:
        print("⚠ No old_llm_pt specified, skipping LLM weight reload")
        return None


def copy_source_files(source_dir, output_dir):
    """Copy all files from source directory except *.yaml and llm.pt"""
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} not found")
        return
    
    # Get all files and directories in source
    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path from source directory
        rel_path = os.path.relpath(root, source_dir)
        
        # Create corresponding directory in output
        if rel_path != '.':
            output_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)
        else:
            output_subdir = output_dir
        
        # Copy files (excluding *.yaml and llm.pt)
        for file in files:
            if file.endswith('.yaml') or file == 'llm.pt':
                print(f"Skipping: {os.path.join(rel_path, file) if rel_path != '.' else file}")
                continue
            
            source_file = os.path.join(root, file)
            output_file = os.path.join(output_subdir, file)
            
            try:
                shutil.copy2(source_file, output_file)
                print(f"Copied: {os.path.join(rel_path, file) if rel_path != '.' else file}")
            except Exception as e:
                print(f"Error copying {source_file}: {e}")


def save_model_components(configs, output_dir, device='cpu'):
    """Save the LLM model component as .pt file"""
    llm_model = configs['llm'].to(device)
    llm_model.eval()
    
    llm_path = os.path.join(output_dir, 'llm.pt')
    torch.save(llm_model.state_dict(), llm_path)
    
    return llm_path


def main():
    parser = argparse.ArgumentParser(description='Export TASTE model')
    parser.add_argument('--config', default='training/conf/taste2_stage1.yaml',
                       help='Path to TASTE configuration file')
    parser.add_argument('--output', '-o', default='training/pretrained_models/Taste2-Stage1-Init/',
                       help='Output directory')
    parser.add_argument('--source', default='training/pretrained_models/CosyVoice2-0.5B/',
                       help='source directory')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    output_dir = os.path.abspath(args.output)
    
    os.makedirs(output_dir, exist_ok=True)
    
    configs = load_config(config_path)

    reload_llm_pt(configs['llm'], old_llm_pt=args.source + '/llm.pt')
    
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    
    llm_path = save_model_components(configs, output_dir, device)
    
    # Copy all other files from source directory (except *.yaml and llm.pt)
    copy_source_files(args.source, output_dir)
    
    shutil.copy2(config_path, os.path.join(output_dir, 'taste_stage1.yaml'))
    
    print(f"Export completed: {output_dir}")
    
    # Generate detailed model report
    generate_model_report(llm_path, output_dir)

if __name__ == '__main__':
    main()