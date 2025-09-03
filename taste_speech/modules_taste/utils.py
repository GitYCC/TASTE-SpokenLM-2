
import torch


def generate_mask_from_length(lengths):
    batch = lengths.size(0)
    max_length = lengths.max()
    return torch.arange(max_length, device=lengths.device).repeat(batch, 1) < lengths.unsqueeze(-1)


def debug_print(x, name):
    if isinstance(x, torch.Tensor):
        print(f'{name}, {x.size()}, {x}')
    else:
        print(f'{name}, {x}')


def _find_all_linear_names(model):
    cls = (torch.nn.Linear, )
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    output_embedding = "lm_head"
    if output_embedding in lora_module_names:  # needed for 16-bit
        lora_module_names.remove(output_embedding)

    return list(lora_module_names)
