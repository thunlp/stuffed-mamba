from typing import Union
from pathlib import Path

import torch

from .configuration_mamba import MambaConfig
from .modeling_mamba import MambaModel, MambaForCausalLM, MambaState


def convert_state_dict(state_dict: dict):
    new_state_dict = {}
    for key in state_dict:
        new_key = "" + key
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def load_pretrained(model: MambaForCausalLM, path: Union[str, Path]):
    state_dict = convert_state_dict(torch.load(path))
    missing_keys, unexpected_keys = model.model.load_state_dict(state_dict)
    print("========= missing keys ================")
    print(missing_keys)
    print("========= unexpected keys =============")
    print(unexpected_keys)
    print("=======================================")
    model.tie_lm_head()
    model.to(dtype=model.config.get_dtype())
