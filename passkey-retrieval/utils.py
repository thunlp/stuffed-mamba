from torch import nn
import jsonlines
from pathlib import Path


def get_non_embed_param_count(mod: nn.Module):
    cnt = 0
    for n, p in mod.named_parameters():
        if 'embed' not in n:
            cnt += p.numel()
    return cnt


def get_param_count(mod: nn.Module):
    return sum(p.numel() for p in mod.parameters())



def iter_jsonl(path: Path):
    with jsonlines.open(path, "r") as fin:
        for line in fin:
            yield line


def dump_jsonl(data: list, path: Path):
    with jsonlines.open(path, "w") as fout:
        for line in data:
            fout.write(line)
