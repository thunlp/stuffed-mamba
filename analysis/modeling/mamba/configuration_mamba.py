from dataclasses import dataclass, fields, asdict
from typing import Union
import math

import torch


@dataclass
class MambaConfig:
    '''
    There are the default params for 130M official checkpoint.
    '''
    dtype: str = 'bf16'
    d_model: int = 768  # D
    n_layers: int = 24  # L
    vocab_size: int = 50280  # V
    pad_vocab_size_multiple: int = 8
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    scan_impl: str = 'pscan_torch'  # "pscan_cuda": The pscan from Tri Dao. "pscan_torch": PyTorch implementation of pscan. "seq_scan": Use sequential scan.

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )

    def to_mamba_config(self) -> "MambaConfig":
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {
            k: v for k, v in asdict(self).items() if k in mamba_config_fields
        }
        return MambaConfig(**filtered_dict)

    def get_dtype(self):
        if self.dtype in ['bf16', 'bfloat16']:
            return torch.bfloat16
        elif self.dtype in ['fp16', 'float16']:
            return torch.float16
        elif self.dtype in ['fp32', 'float32']:
            return torch.float32
        else:
            raise ValueError(f"Invalid dtype string: {self.dtype}")
