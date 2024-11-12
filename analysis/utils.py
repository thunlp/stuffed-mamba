import json

from typing import Optional
from accelerate import Accelerator

from torch import Tensor, nn


def load_json(path):
    return json.load(open(path, 'r', encoding='utf8'))


def dump_json(data, path, indent=4, **kwargs):
    json.dump(data, open(path, 'w', encoding='utf8'), ensure_ascii=False, indent=indent, **kwargs)


def get_param_count(model) -> int:
    """Return the number of parameters of a model"""
    total_params = 0
    for param_name, param in model.state_dict().items():
        # print(f"{param_name:>40s} \t {param.numel()}")
        total_params += param.numel()
    return total_params


def get_non_embed_param_count(model) -> int:
    """Return the number of parameters of a model"""
    total_params = 0
    for param_name, param in model.state_dict().items():
        if (".emb" in param_name) or (".lm_head" in param_name):
            continue
        # print(f"{param_name:>40s} \t {tuple(param.shape)} \t {param.numel()}")
        total_params += param.numel()
    return total_params


def inspect_params(model: nn.Module, accelerator: Optional[Accelerator] = None, param_filter=None):
    if param_filter is None:
        param_filter = lambda x: True  # noqa

    print_fn = print if accelerator is None else accelerator.print
    names = [n for n, p in model.named_parameters()]
    max_len = max(len(n) for n in names)
    for n, p in model.named_parameters():
        assert isinstance(p, Tensor)
        if not param_filter(n):
            continue
        print_fn(
            f"{n:>{max_len}s} \t "
            f"{str(tuple(p.shape)):>9s} \t "
            f"{p.mean().item():.4f} \t "
            f"{p.std().item():.4f} \t ",
            # f"{p.min().item():.4f} \t "
            # f"{p.max().item():.4f} \t ",
            end="",
        )
        if p.grad is not None:
            g = p.grad
            print_fn(
                f"{g.mean().item():.4f} \t "
                f"{g.std().item():.4f} \t ",
                # f"{g.min().item():.4f} \t "
                # f"{g.max().item():.4f} \t ",
                end="",
            )
        print_fn()


class Logger:
    def __init__(self, accelerator: Optional[Accelerator] = None):
        self.accelerator = accelerator

    def log(self, msg, rank=0, **kwargs):
        if not isinstance(msg, str):
            msg = str(msg)
        if self.accelerator is None or self.accelerator.process_index == rank:
            print(f"[{self.accelerator.process_index}] " + msg, **kwargs)

    def log_all_rank(self, msg, **kwargs):
        if not isinstance(msg, str):
            msg = str(msg)
        print(f"[{self.accelerator.process_index}] " + msg, **kwargs)

    def log_params(self, model: nn.Module, param_filter=None):
        inspect_params(model, self.accelerator, param_filter)

    def log_epoch(self, epoch, rank=0):
        if self.accelerator is None or self.accelerator.process_index == rank:
            print(f"[{rank}] " + "Epoch {epoch}")

    def log_step(self, step, rank=0):
        if self.accelerator is None or self.accelerator.process_index == rank:
            print(f"Step {step}")

    def log_grads(self, model, rank=0):
        if self.accelerator is None or self.accelerator.process_index == rank:
            for n, p in model.named_parameters():
                if p.grad is not None:
                    print(f"{n}: {p.grad.mean().item():.4f}")

    def log_memory(self, rank=0):
        if self.accelerator is None or self.accelerator.process_index == rank:
            import torch

            print(f"[{rank}] " + f"Memory: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
