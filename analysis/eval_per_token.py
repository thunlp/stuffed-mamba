from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import fla
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from get_prompt import get_long_prompt
from tap import Tap

from utils import get_non_embed_param_count, get_param_count


class Args(Tap):
    model_name: Literal['mamba', 'mamba2', 'rwkv6', 'hgrn2'] = 'mamba2'
    pretrained_path: str = '/path/to/model'
    tok_path: str = '/path/to/model'
    prompt_name: str = 'newlines'
    device: str = 'cuda'
    max_len: int = 1024
    overwrite: int = 0
    train_len: int = 4
    xmax: int = 1024
    ymax: int = 60
    xlog: int = 0
    ylog: int = 0
    ppl: int = 1

    # The input is chunked before feeding to the model to avoid OOM.
    chunk_size: int = 128

    # The loss is averaged over a window of `bucket_size` tokens.
    bucket_size: int = 8192

    # Whether to use sliding window, -1 means no sliding window.
    window_size: int = -1

    verbose: int = 0


def get_model(pretrained_path, tok_path=None, device='cuda', dtype=torch.bfloat16, model_name: str = ''):
    if tok_path is None:
        tok_path = pretrained_path
    if model_name == 'mamba2' or "mamba2" in str(pretrained_path):
        from modeling.mamba2.modeling_mamba2_dao import Mamba2ForCausalLM
        print(f"Loading tokenizer from {tok_path}")
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        print(f"Loading model from {pretrained_path}...")
        model = Mamba2ForCausalLM.from_pretrained(
            pretrained_path,
            device=device,
            dtype=dtype,
        )
    elif 'rwkv6' in str(pretrained_path):
        print(f"Loading tokenizer from {tok_path}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=True)
        print(f"Loading model from {pretrained_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            trust_remote_code=True,
        ).to(device=device, dtype=dtype)
    elif 'hgrn2' in str(pretrained_path):
        print(f"Loading tokenizer from {tok_path}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=True)
        print(f"Loading model from {pretrained_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            trust_remote_code=True,
        ).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model type: {pretrained_path}")
    return model, tokenizer


def compute_per_token_loss(args: Args, model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids: Tensor = inputs.input_ids.to(device=args.device)
    attention_mask: Tensor = inputs.attention_mask.to(device=args.device)
    print("Input ids shape:", input_ids.shape)

    # Run one forward pass and compute per token loss
    with torch.no_grad():
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        if 'mamba2' in args.pretrained_path:
            chunks = torch.split(input_ids, args.chunk_size * 1024, dim=1)  # (bsz, n_chunks, chunk_size)
            n_chunks = len(chunks)
            all_loss = []
            states = None
            for chunk_i, chunk in enumerate(chunks):
                print(f"{chunk_i}/{n_chunks}, {chunk.shape = }")
                outputs = model(
                    input_ids=chunk,
                    attention_mask=attention_mask,
                    states=states,
                )
                states = outputs.states
                logits = outputs.logits  # (bsz, C, V)
                # Compute per token loss
                labels = chunk[:, 1:]  # (bsz, C - 1)
                preds = logits[:, :-1].transpose(1, 2)  # (bsz, V, C - 1)
                loss = loss_fn(preds, labels)  # (bsz, C - 1)
                all_loss.append(loss)

            loss = torch.cat(all_loss, dim=1)  # (bsz, T - 1)
            per_token_loss = loss.float().cpu()
            print(per_token_loss.shape)
        elif 'rwkv6' in args.pretrained_path:
            chunks = torch.split(input_ids, args.chunk_size * 1024, dim=1)  # (bsz, n_chunks, chunk_size)
            n_chunks = len(chunks)
            all_loss = []
            states = None
            for chunk_i, chunk in enumerate(chunks):
                print(f"{chunk_i}/{n_chunks}, {chunk.shape = }")
                outputs = model(
                    input_ids=chunk,
                    attention_mask=attention_mask,
                    state=states,
                )
                states = outputs.state
                logits = outputs.logits  # (bsz, C, V)
                # Compute per token loss
                labels = chunk[:, 1:]  # (bsz, C - 1)
                preds = logits[:, :-1].transpose(1, 2)  # (bsz, V, C - 1)
                loss = loss_fn(preds, labels)  # (bsz, C - 1)
                all_loss.append(loss)

            loss = torch.cat(all_loss, dim=1)  # (bsz, T - 1)
            per_token_loss = loss.float().cpu()
            print(per_token_loss.shape)
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits = outputs.logits
        elif 'hgrn2' in args.pretrained_path:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Compute per token loss
            labels = input_ids[:, 1:]  # (bsz, C - 1)
            preds = logits[:, :-1].transpose(1, 2)  # (bsz, V, C - 1)
            loss = loss_fn(preds, labels)  # (bsz, C - 1)
            per_token_loss = loss.float().cpu()
            print(per_token_loss.shape)
        else:
            raise ValueError(f"Unknown model type: {args.pretrained_path}")

        print(f"Per token loss: {per_token_loss}")
    return per_token_loss


def format_number(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num // 2**30}B"  # Billions
    elif num >= 1_000_000:
        return f"{num // 2**20}M"  # Millions
    elif num >= 1_000:
        return f"{num // 2**10}K"  # Thousands
    else:
        return str(num)  # Return the number as is for values below 1000


def main():
    args = Args().parse_args()

    args.pretrained_path = args.pretrained_path.rstrip('/')
    run_name = args.pretrained_path.replace('/', '--')
    output_dir = Path('result_per_token') / run_name / args.prompt_name
    output_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(output_dir / 'args.json'))

    cache_path = output_dir / 'per_token_loss.pt'
    print(f"Cache path: {cache_path}")
    if cache_path.exists() and not args.overwrite:
        print(f"Loading cached result from {cache_path}")
        with open(cache_path, 'rb') as f:
            per_token_loss = torch.load(f, map_location=args.device)
            if isinstance(per_token_loss, tuple) and len(per_token_loss) == 2:
                per_token_loss, _ = per_token_loss
        print("Per token loss:", per_token_loss)
    else:
        print("Loading tokenizer and model")
        model, tokenizer = get_model(
            args.pretrained_path,
            args.tok_path,
            device=args.device,
            model_name=args.model_name,
        )
        print("========== finish loading =========")
        n_params = get_param_count(model)
        n_non_embed_params = get_non_embed_param_count(model)
        print(f"Param count: {n_params:,}, non-embedding: {n_non_embed_params:,}")
        print("=======================================================")

        prompt = get_long_prompt(args.prompt_name)
        tokens = tokenizer.tokenize(prompt)
        print(f"Prompt length: {len(tokens)}")
        if len(tokens) > args.max_len * 1024:
            print(f"Prompt length {len(tokens)} > max_len {args.max_len * 1024}, truncating")
            ratio = (args.max_len * 1024) / len(tokens)
            print(f"Ratio: {ratio}")
            prompt = prompt[:int(len(prompt) * min(1, 1.5 * ratio)) + 4096]
        else:
            # We need to repeat the prompt to ensure the output length is at least max_len
            n_reps = (args.max_len * 1024 - 1) // len(tokens) + 1
            prompt = prompt * n_reps

        if args.verbose:
            print("======= prompt =========")
            print(prompt[:1000])
            print('--------------------')
            print(prompt[-1000:])
            print("========================")
        per_token_loss = compute_per_token_loss(args, model, tokenizer, prompt)
        print(f"Caching result to: {cache_path}")
        torch.save(per_token_loss, cache_path)

    # Average over batch size
    assert isinstance(per_token_loss, Tensor)
    per_token_loss = per_token_loss.mean(dim=0)  # (T - 1)

    # Bucket average
    buckets = torch.split(per_token_loss, args.bucket_size)  # (n_buckets, bucket_size)
    ys = torch.stack([bucket.mean() for bucket in buckets]).cpu()  # (n_buckets)
    # The first and last bucket have large variations, so we discard it.
    ys = ys[1:-1]
    xs = torch.arange(1, len(ys) + 1) * args.bucket_size + args.bucket_size // 2

    # positions = list(range(len(per_bucket_loss)))
    plt.figure(figsize=(2.3, 2.3))
    if args.ppl:
        ys = torch.exp(ys)
        plt.ylabel('Perplexity')
        plt.ylim(0, args.ymax)
    else:
        plt.ylabel('Loss')
    plt.plot(xs, ys)
    plt.axvline(x=args.train_len * 1024, color='r', linestyle='--')
    plt.xlim((args.bucket_size, args.xmax * 1024))

    xticks = [args.train_len * 1024, args.xmax * 512, args.xmax * 1024]
    xticklabels = [format_number(x) for x in xticks]
    plt.xticks(xticks, xticklabels)

    plt.xlabel(r'Token position ($t$)')
    # plt.xlim(0, 30000)
    # plt.ylim(2.2, 2.8)
    if args.ylog:
        plt.yscale('log')
    if args.xlog:
        plt.xscale('log')
    plt.grid(True)

    plt.tight_layout()

    dst_path = output_dir / 'per_token_loss.pdf'
    print(f"Saving plot to {dst_path}")
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    


if __name__ == "__main__":
    main()
