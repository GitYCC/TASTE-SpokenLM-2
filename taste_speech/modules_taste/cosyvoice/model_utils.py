from typing import Dict, Optional, Union, List
import torch
import logging
from transformers import WhisperModel
# use custom is required
from .customized_whisper import WhisperModel as CustomWhisperModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def filter_state_dict_by_key(state_dict, target_key: str):
    return {k:v for k, v in state_dict.items() if target_key in k}

def get_partial_state_dict_by_keys(state_dict: Dict, target_keys: List[str]):
    if len(target_keys) == 0:
        logging.warning(f"Get partial state_dict by keys, but the keys is empty. Will return an empty dict!")
    partial_state_dict = {}
    for target_key_str in target_keys: # a target_key_str can be like: text_encoder.encoders, text_encoder.embed.out.0
        filtered_state_dict = filter_state_dict_by_key(state_dict, target_key_str)
        partial_state_dict.update(filtered_state_dict)
    return partial_state_dict

def load_state_dict_by_partial_list(
    model: torch.nn.Module,
    target_state_dict: Dict,
    load_partial_list: List[str] = []    
) -> Optional[List[str]]:
    if len(load_partial_list) == 0:
        logger.warning(f"Called load parameters by partial list, but the load_partial_list is empty. Try to load with full state dict instead.")
        model.load_state_dict(target_state_dict, strict=True)
        return None # NOTE: fully loaded, set the partial_loaded_list to None. 
    # traverse through the partial dict to load part of the state dict only
    logger.info(f"Load model partially, list={self.load_partial_list}.")
    partial_state_dict = get_partial_state_dict_by_keys(target_state_dict, self.load_partial_list)
    logger.info(f"Load partial state dict: keys={partial_state_dict.keys()}")
    model.load_state_dict(partial_state_dict, strict=False)
    partial_loaded_list = load_partial_list
    return partial_loaded_list

def freeze_parameters_by_partial_list(
    model: torch.nn.Module, 
    freeze_partial_list: List[str] = []
) -> Optional[List[str]]:
    # NOTE: Currently 'unfreeze automatically' is not supported!
    if len(freeze_partial_list) == 0:
        logger.warning("Freeze_paramters_by_partial_list, but nothing is going to be freeze!")
        return None
    logger.info(f"Attempt to freeze model partially, list={freeze_partial_list}.")
    partial_state_dict = get_partial_state_dict_by_keys(model.state_dict(), freeze_partial_list)
    param_names_to_freeze = list(partial_state_dict.keys())
    logger.info(f"Get partial state dict by the partial list: keys={param_names_to_freeze[:10]}...")
    for name, param in model.named_parameters():
        if name in param_names_to_freeze:
            param.requires_grad = False
            logger.debug(f"{name} is frozen")
    partial_frozen_list = freeze_partial_list
    return partial_frozen_list

def get_nesty_module_by_key(orig_module: torch.nn.Module, target_key: str):
    target_module = orig_module
    for key in target_key.split('.'):
        target_module = getattr(target_module, key)
    return target_module

def load_whisper_whole_model(
    model_name_or_path: str = "",
    attn_implementation: str = "eager", # select from ['eager', 'sdpa', 'flash_attention_2']
    dtype: str = "float32", # select from ['float32', 'float16', 'bfloat16']
    use_custom: bool = False,
    **kwargs,
):
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else: 
        torch_dtype = torch.float32
    if use_custom:
        whole_model = CustomWhisperModel.from_pretrained(
            model_name_or_path,
            torch_dtype = torch_dtype,
            attn_implementation = attn_implementation,
            **kwargs,
        )
        # print("Use customized whisper!")
    else:
        whole_model = WhisperModel.from_pretrained(
            model_name_or_path,
            torch_dtype = torch_dtype,
            attn_implementation = attn_implementation,
            **kwargs,
        )
    return whole_model, torch_dtype

def get_s3_encoder_dict(
    whisper_encoder_dict: Dict,
    s3_encoder_ckpt: str = "",
):
    pretrained_weights = torch.load(s3_encoder_ckpt)

    for name, param in pretrained_weights.items():
        if name in whisper_encoder_dict:
            whisper_encoder_dict[name].copy_(param)
        else:
            print(f"Skipping {name} as it doesn't exist in the whispher encode's state_dict")

    return whisper_encoder_dict

