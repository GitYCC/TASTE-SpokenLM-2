# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import torch
import json
import re
import datetime
import yaml

import deepspeed
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

try:
    import wandb
    WANDB_AVAILABLE = True
    logging.info(f"WandB import successful. Version: {wandb.__version__}")
except ImportError as e:
    logging.warning(f"WandB import failed with ImportError: {e}")
    WANDB_AVAILABLE = False
except Exception as e:
    logging.warning(f"WandB import failed with unexpected error: {type(e).__name__}: {e}")
    WANDB_AVAILABLE = False

from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

from taste_speech.taste2.cosyvoice.dataset.dataset import Dataset
from taste_speech.taste2.cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR


class UnifiedLogger:
    """Unified logger that supports both TensorBoard and wandb
    
    TensorBoard is always enabled, wandb is optional additional logging.
    """
    
    def __init__(self, tensorboard_dir, use_wandb=False, wandb_config=None, training_config=None):
        # Always initialize TensorBoard
        self.tensorboard_writer = SummaryWriter(tensorboard_dir)

        # Initialize wandb only if requested and available
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        if self.use_wandb:
            try:
                if wandb_config is None:
                    wandb_config = {}

                logging.info(f"Attempting to initialize WandB with config: {wandb_config}")

                # Check if wandb is already initialized
                if not wandb.run:
                    wandb.init(**wandb_config)
                    logging.info(f"WandB initialized successfully. Run name: {wandb.run.name}, Run URL: {wandb.run.url}")

                    # Log training configuration to wandb if provided
                    if training_config is not None and wandb.run:
                        wandb.config.update(training_config)
                else:
                    logging.info(f"WandB already initialized. Run name: {wandb.run.name}")
            except Exception as e:
                logging.warning(f"Failed to initialize WandB: {e}. Falling back to TensorBoard only.")
                self.use_wandb = False
            
    def add_scalar(self, tag, value, step):
        """Log scalar values to TensorBoard (always) and wandb (if enabled)"""
        # Always log to TensorBoard
        self.tensorboard_writer.add_scalar(tag, value, step)
        
        # Optionally log to wandb
        if self.use_wandb and wandb.run:
            # Convert TensorBoard tag format to wandb format
            wandb_key = tag.replace('/', '_')
            wandb.log({wandb_key: value}, step=step)
    
    def add_scalars(self, tag_dict, step):
        """Log multiple scalar values"""
        for tag, value in tag_dict.items():
            self.add_scalar(tag, value, step)
    
    def finish(self):
        """Clean up loggers"""
        # Always close TensorBoard
        self.tensorboard_writer.close()
        
        # Optionally close wandb
        if self.use_wandb and wandb.run:
            wandb.finish()


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine == 'torch_ddp':
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    else:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs, gan, dpo):
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']
    train_dataset = Dataset(args.train_data, data_pipeline=data_pipeline, mode='train', gan=gan, dpo=dpo, shuffle=True, partition=True)
    cv_dataset = Dataset(args.cv_data, data_pipeline=data_pipeline, mode='train', gan=gan, dpo=dpo, shuffle=False, partition=False)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def check_modify_and_save_config(args, configs):
    if args.train_engine == "torch_ddp":
        configs['train_conf']["dtype"] = 'bf16' if args.use_amp is True else 'fp32'
    else:
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs['train_conf']["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs['train_conf']["dtype"] = "bf16"
        else:
            configs['train_conf']["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs['train_conf']['save_per_step'] = int(configs['train_conf']['save_per_step'] *
                                                     configs['train_conf']['accum_grad'] / ds_configs["gradient_accumulation_steps"])
        configs['train_conf']['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        configs['train_conf']['grad_clip'] = ds_configs["gradient_clipping"]
        configs['train_conf']['log_interval'] = ds_configs["steps_per_print"]
    return configs


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
    return model


def init_optimizer_and_scheduler(args, configs, model, gan):
    if gan is False:
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        # use deepspeed optimizer for speedup
        if args.train_engine == "deepspeed":
            def scheduler(opt):
                return scheduler_type(opt, **configs['train_conf']['scheduler_conf'])
            model, optimizer, _, scheduler = deepspeed.initialize(
                args=args,
                model=model,
                optimizer=None,
                lr_scheduler=scheduler,
                model_parameters=model.parameters())

        optimizer_d, scheduler_d = None, None

    else:
        # currently we wrap generator and discriminator in one model, so we cannot use deepspeed
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        if configs['train_conf']['optim_d'] == 'adam':
            optimizer_d = optim.Adam(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim_d'] == 'adamw':
            optimizer_d = optim.AdamW(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler_d'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler_d = WarmupLR(optimizer_d, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler_d'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler_d = NoamHoldAnnealing(optimizer_d, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler_d = ConstantLR(optimizer_d)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])
    return model, optimizer, scheduler, optimizer_d, scheduler_d


def init_summarywriter(args, configs=None):
    """Initialize unified logger with TensorBoard and wandb support
    
    TensorBoard is always enabled, wandb configuration comes from config file.
    """
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Get wandb settings from config file
        use_wandb = False
        wandb_config = None
        training_config = None
        
        if configs and 'train_conf' in configs:
            train_conf = configs['train_conf']
            use_wandb = train_conf.get('use_wandb', False)
            
            if use_wandb:
                # Get wandb configuration from config file
                wandb_project = train_conf.get('wandb_project', 'cosyvoice')
                wandb_entity = train_conf.get('wandb_entity', None)
                wandb_run_name = train_conf.get('wandb_run_name', None)
                wandb_tags = train_conf.get('wandb_tags', '')
                
                wandb_config = {
                    'project': wandb_project,
                    'entity': wandb_entity,
                    'name': wandb_run_name,
                    'tags': wandb_tags.split(',') if wandb_tags else None,
                    'save_code': True,
                    'dir': args.model_dir
                }
                # Remove None values
                wandb_config = {k: v for k, v in wandb_config.items() if v is not None}
                
                # Prepare training configuration for wandb
                training_config = {
                    'model_type': args.model,
                    'train_engine': args.train_engine,
                    'use_amp': args.use_amp,
                    'dpo': args.dpo,
                    'num_workers': args.num_workers,
                    'prefetch': args.prefetch,
                    'pin_memory': args.pin_memory,
                }
                # Add training config
                training_config.update(train_conf)
        
        # Initialize unified logger (TensorBoard always enabled, wandb optional)
        print(f"DEBUG: About to initialize UnifiedLogger with use_wandb={use_wandb}, wandb_config={wandb_config}")
        logging.info(f"DEBUG: About to initialize UnifiedLogger with use_wandb={use_wandb}, WANDB_AVAILABLE={WANDB_AVAILABLE}")
        writer = UnifiedLogger(
            tensorboard_dir=args.tensorboard_dir,  # Always provided
            use_wandb=use_wandb,
            wandb_config=wandb_config,
            training_config=training_config
        )
        print(f"DEBUG: UnifiedLogger initialized. writer.use_wandb={writer.use_wandb}")
        logging.info(f"DEBUG: UnifiedLogger initialized. writer.use_wandb={writer.use_wandb}")
    
    return writer


def save_model(model, model_name, info_dict):
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:
            torch.save({**model.module.state_dict(), 'epoch': info_dict['epoch'], 'step': info_dict['step']}, save_model_path)
    else:
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir,
                                  tag=model_name,
                                  client_state=info_dict)
    if rank == 0:
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(rank, save_model_path))


def cosyvoice_join(group_join, info_dict):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    if info_dict["batch_idx"] != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(group=group_join,
                                   timeout=group_join.options._timeout)
            return False
        except RuntimeError as e:
            logging.info("Detected uneven workload distribution: {}\n".format(e) +
                         "Break current worker to manually join all workers, " +
                         "world_size {}, current rank {}, current local_rank {}\n".
                         format(world_size, rank, local_rank))
            return True
    else:
        return False


def batch_forward(model, batch, scaler, info_dict, ref_model=None, dpo_loss=None):
    device = int(os.environ.get('LOCAL_RANK', 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    if info_dict['train_engine'] == 'torch_ddp':
        autocast = torch.cuda.amp.autocast(enabled=scaler is not None, dtype=dtype)
    else:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)

    with autocast:
        info_dict['loss_dict'] = model(batch, device)
        if ref_model is not None and dpo_loss is not None:
            chosen_logps = info_dict['loss_dict']["chosen_logps"]
            rejected_logps = info_dict['loss_dict']["rejected_logps"]
            sft_loss = info_dict['loss_dict']['loss']
            with torch.no_grad():
                ref_loss_dict = ref_model(batch, device)
            reference_chosen_logps = ref_loss_dict["chosen_logps"]
            reference_rejected_logps = ref_loss_dict["rejected_logps"]
            preference_loss, chosen_reward, reject_reward = dpo_loss(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            dpo_acc = (chosen_reward > reject_reward).float().mean()
            info_dict['loss_dict']["loss"] = preference_loss + sft_loss
            info_dict['loss_dict']["sft_loss"] = sft_loss
            info_dict['loss_dict']["dpo_loss"] = preference_loss
            info_dict['loss_dict']["dpo_acc"] = dpo_acc
            info_dict['loss_dict']["chosen_reward"] = chosen_reward.mean()
            info_dict['loss_dict']["reject_reward"] = reject_reward.mean()
    return info_dict


def batch_backward(model, scaler, info_dict):
    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(info_dict['loss_dict']['loss'])
    else:
        scaled_loss = info_dict['loss_dict']['loss'] / info_dict['accum_grad']
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    info_dict['loss_dict']['loss'] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):
    grad_norm = 0.0
    if info_dict['train_engine'] == "deepspeed":
        info_dict["is_gradient_accumulation_boundary"] = model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()
    elif (info_dict['batch_idx'] + 1) % info_dict["accum_grad"] == 0:
        # Use mixed precision training
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            # We don't check grad here since that if the gradient
            # has inf/nan values, scaler.step will skip
            # optimizer.step().
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                logging.warning('get infinite grad_norm, check your code/data if it appears frequently')
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                logging.warning('get infinite grad_norm, check your code/data if it appears frequently')
        optimizer.zero_grad()
        scheduler.step()
    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm
    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = int(os.environ.get('RANK', 0))

    # only rank 0 write to tensorboard to avoid multi-process write
    if writer is not None:
        if (info_dict['train_engine'] == 'deepspeed' and info_dict['is_gradient_accumulation_boundary'] is True) or \
           (info_dict['train_engine'] == 'torch_ddp' and (info_dict['batch_idx'] + 1) % info_dict['accum_grad'] == 0):
            for k in ['epoch', 'lr', 'grad_norm']:
                writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0:
        log_str = '{} Batch {}/{} '.format(tag, epoch, batch_idx + 1)
        for name, value in loss_dict.items():
            log_str += '{} {:.6f} '.format(name, value)
        if tag == "TRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f}'.format(
                info_dict["lr"], info_dict['grad_norm'])
        log_str += ' rank {}'.format(rank)
        logging.debug(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = int(os.environ.get('RANK', 0))
    logging.info(
        'Epoch {} Step {} CV info lr {} {} rank {}'.format(
            epoch, step + 1, lr, rank, ' '.join(['{} {}'.format(k, v) for k, v in loss_dict.items()])))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)


def apply_parameter_freezing(model, freeze_config):
    """Apply parameter freezing based on regex patterns
    
    Args:
        model: PyTorch model to apply freezing to
        freeze_config: Dictionary containing freeze configuration with keys:
            - enabled: bool, whether to enable parameter freezing
            - patterns: list of regex patterns to match parameter names
    
    Returns:
        dict: Statistics about frozen parameters
    """
    if not freeze_config or not freeze_config.get('enabled', False):
        logging.info("Parameter freezing is disabled")
        return {'frozen_count': 0, 'total_count': 0, 'frozen_ratio': 0.0}
    
    patterns = freeze_config.get('patterns', [])
    if not patterns:
        logging.warning("No freeze patterns provided, skipping parameter freezing")
        return {'frozen_count': 0, 'total_count': 0, 'frozen_ratio': 0.0}
    
    # Compile regex patterns for efficiency
    compiled_patterns = []
    for pattern in patterns:
        try:
            compiled_patterns.append(re.compile(pattern))
            logging.info(f"Compiled freeze pattern: {pattern}")
        except re.error as e:
            logging.error(f"Invalid regex pattern '{pattern}': {e}")
            continue
    
    if not compiled_patterns:
        logging.warning("No valid regex patterns found, skipping parameter freezing")
        return {'frozen_count': 0, 'total_count': 0, 'frozen_ratio': 0.0}
    
    frozen_count = 0
    total_params = 0
    frozen_param_names = []
    
    # Apply freezing to parameters
    for name, param in model.named_parameters():
        total_params += 1
        
        # Check if parameter name matches any pattern
        for pattern in compiled_patterns:
            if pattern.search(name):
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
                    frozen_param_names.append(name)
                    logging.debug(f"Frozen parameter: {name}")
                break
    
    frozen_ratio = frozen_count / total_params if total_params > 0 else 0.0
    
    # Log summary
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:  # Only log from rank 0 to avoid duplicate logs
        logging.info(f"Parameter freezing summary:")
        logging.info(f"  Total parameters: {total_params}")
        logging.info(f"  Frozen parameters: {frozen_count}")
        logging.info(f"  Frozen ratio: {frozen_ratio:.2%}")
        logging.info(f"  Freeze patterns used: {patterns}")
        
        if frozen_param_names:
            logging.info(f"  First 10 frozen parameters: {frozen_param_names[:10]}")
            if len(frozen_param_names) > 10:
                logging.info(f"  ... and {len(frozen_param_names) - 10} more")
    
    return {
        'frozen_count': frozen_count,
        'total_count': total_params,
        'frozen_ratio': frozen_ratio,
        'frozen_param_names': frozen_param_names
    }
