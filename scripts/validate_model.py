#!/usr/bin/env python3
# Model validation script using CosyVoice cv function
# Only performs validation, no training

from __future__ import print_function
import argparse
import datetime
import logging
import os
import sys
import torch
import torch.distributed as dist
import deepspeed
from copy import deepcopy

from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

from taste_speech.taste2.cosyvoice.utils.losses import DPOLoss
from taste_speech.taste2.cosyvoice.utils.executor import Executor
from taste_speech.taste2.cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_summarywriter,
    wrap_cuda_model, check_modify_and_save_config,
    apply_parameter_freezing)


def get_args():
    parser = argparse.ArgumentParser(description='model validation using cv function')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model to validate')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--cv_data', required=True, help='validation data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model to validate')
    parser.add_argument('--model_dir', required=True, help='output dir for validation results')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision')
    parser.add_argument('--batch_size',
                        default=None,
                        type=int,
                        help='Override batch size for validation')
    parser.add_argument('--num_gpus',
                        default=None,
                        type=int,
                        help='Number of GPUs to use for validation')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    logging.info('Starting model validation...')
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # GAN train has some special initialization logic
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    
    try:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, 'qwen_pretrain_path': args.qwen_pretrain_path})
    except Exception:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=override_dict)
    
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))

    # Override batch size if specified
    if args.batch_size is not None:
        if 'dataset_conf' not in configs:
            configs['dataset_conf'] = {}
        if 'cv_conf' not in configs:
            configs['cv_conf'] = {}
        configs['dataset_conf']['batch_size'] = args.batch_size
        configs['cv_conf']['batch_size'] = args.batch_size
        logging.info(f'Overriding batch size to {args.batch_size}')

    # Set up multi-GPU environment if num_gpus specified
    if args.num_gpus is not None and args.num_gpus > 1:
        import os
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(args.num_gpus)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
        logging.info(f'Setting up multi-GPU validation with {args.num_gpus} GPUs')

    # Init env for ddp
    init_distributed(args)

    # Get dataset & dataloader (use cv_data as both train and cv for initialization)
    args.train_data = args.cv_data  # Use cv_data as placeholder for train_data
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)

    # Do some sanity checks and save config
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args, configs)

    # Load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo
    model = configs[args.model]
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint {args.checkpoint} does not exist!')
    
    logging.info(f'Loading checkpoint: {args.checkpoint}')
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    
    # Get epoch and step from checkpoint if available
    epoch = state_dict.get('epoch', 0)
    step = state_dict.get('step', 0)

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Apply parameter freezing
    freeze_config = configs.get('freeze_params', {})
    freeze_stats = apply_parameter_freezing(model, freeze_config)

    # DPO related (if needed)
    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        ref_model = wrap_cuda_model(args, ref_model)
    else:
        ref_model, dpo_loss = None, None

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.epoch = epoch
    executor.step = step

    # Create info_dict for validation
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = step
    info_dict['epoch'] = epoch
    info_dict['lr'] = 0.0  # Set learning rate to 0 for validation-only runs
    
    logging.info(f'Starting validation on checkpoint from epoch {epoch}, step {step}')

    # Run validation only (the cv function from executor)
    dist.barrier()
    executor.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    # Clean up loggers
    if writer is not None and hasattr(writer, 'finish'):
        writer.finish()

    logging.info('Model validation completed!')


if __name__ == '__main__':
    main()


"""
Usage Examples:

1. Single GPU validation with custom batch size:
   python validate_model.py --model llm --config config.yaml --cv_data data.list 
   --checkpoint model.pt --model_dir ./output --batch_size 8 --num_gpus 1

2. Multi-GPU validation:
   torchrun --nproc_per_node=4 validate_model.py --model llm --config config.yaml 
   --cv_data data.list --checkpoint model.pt --model_dir ./output --batch_size 4 --num_gpus 4

3. Using the shell script:
   # Edit NUM_GPUS and BATCH_SIZE variables in example_validate.sh
   bash scripts/example_validate.sh

Command Line Arguments:
  --batch_size: Override batch size for validation (per GPU)
  --num_gpus: Number of GPUs to use for validation
  --model: Model type (llm, flow, hift, hifigan)
  --config: Configuration file path
  --cv_data: Validation data file path
  --checkpoint: Model checkpoint path
  --model_dir: Output directory for validation results

Environment Variables (auto-set by script):
  WORLD_SIZE: Total number of processes (usually equals num_gpus)
  MASTER_ADDR: Master node address (localhost for single node)
  MASTER_PORT: Master node port (29500 by default)
  RANK: Process rank (set automatically by torchrun)

Note: The script automatically handles single vs multi-GPU setup based on num_gpus parameter.
"""