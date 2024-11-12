from dataclasses import dataclass
import torch


@dataclass
class RWKVConfig:
    vocab_side: int = 65536
    hidden_size: int = 768
    num_hidden_layers: int = 24
    head_size: int = 64
    tie_word_embeddings: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 0
    layer_norm_epsilon: float = 1e-5
    rescale_every: int = 6
    dtype: str = 'bfloat16'

    def get_dtype(self):
        if self.dtype in ['bf16', 'bfloat16']:
            return torch.bfloat16
        elif self.dtype in ['fp16', 'float16']:
            return torch.float16
        elif self.dtype in ['fp32', 'float32']:
            return torch.float32
        else:
            raise ValueError(f"Invalid dtype string: {self.dtype}")
