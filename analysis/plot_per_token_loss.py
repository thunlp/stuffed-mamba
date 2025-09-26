from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tap import Tap
import numpy as np
import random


class Args(Tap):
    device: str = 'cuda'
    prompt_name: str = 'redpj_16k'
    xlog: int = 0
    ylog: int = 0
    xmax: int = 64
    dx: int = 128
    ymin: float = 0
    ymax: float = 60
    train_len: int = 8192
    bucket_size: int = 128
    ppl: int = 1
    fig_width: float = 3
    fig_height: float = 2.3
    name: str = 'mamba2'


name = "mamba2-130m-training-long"
ckpt_paths = [
    "../../long-rnn/long-rnn/ckpts/mamba2-130m/T4096_B16_GA1_P1_SR1_RD0_lr0.0005/ckpt_100000",
    # "../../long-rnn/long-rnn/ckpts/mamba2-130m/T8192_B8_GA1_P1_SR1_RD0_lr0.0005/ckpt_100000",
    "../../long-rnn/long-rnn/ckpts/mamba2-130m/orig/ckpt_0",
    "../../long-rnn/long-rnn/ckpts/mamba2-130m/T16384_B4_GA1_P1_SR1_RD0_lr0.0005/ckpt_100000",
    # "../../long-rnn/long-rnn/ckpts/mamba2-130m/T65536_B1_GA1_P1_SR1_RD0_lr0.0005/ckpt_100000",
]
labels = [
    "4K",
    "8K",
    "16K",
    # "64K",
]
train_lens = [
    4096,
    8192,
    16384,
]

# ckpt_paths = [
#     "../../ckpts/mamba/mamba2-130m",
#     "../../long-rnn/long-rnn/ckpts/mamba2-130m/T16384_B4_GA1_P1_SR1_RD0_lr0.0005/ckpt_100000",
# ]
# labels = [
#     'Mamba-2 130M',
#     'LongRNN 130M',
# ]


# name = 'mamba2-370m-training-long'
# ckpt_paths = [
#     "--home--test--test07--chenyingfa--long-rnn--long-rnn--ckpts--mamba2-370m--orig--ckpt_0",
#     # "--home--test--test07--chenyingfa--long-rnn--long-rnn--ckpts--mamba2-370m--T8192_B1_GA1_P8_SR4_RD0_lr0.0005--ckpt_100000",
#     "../../long-rnn/long-rnn/ckpts/mamba2-370m/T8192_B4_GA1_P2_SR1_RD0_lr0.0005/ckpt_60000",
#     "../../long-rnn/long-rnn/ckpts/mamba2-370m/T8192_B2_GA1_P4_SR1_RD0_lr0.0005/ckpt_60000",
#     # "../../long-rnn/long-rnn/ckpts/mamba2-370m/T8192_B1_GA1_P8_SR16_RD0_lr0.0005/ckpt_90000",
#     "../../long-rnn/long-rnn/ckpts/mamba2-370m/T8192_B1_GA1_P8_SR16_RD0_lr0.0005/ckpt_100000",
# ]
# labels = [
#     "8K",
#     # '64K, 100K',
#     "16K",
#     "32K",
#     '64K',
#     # '64K, 100K',
# ]
# train_lens = [
#     8 * 1024,
#     16 * 1024,
#     32 * 1024,
#     64 * 1024,
# ]


# ckpt_paths = [
#     "../../ckpts/long-rnn/long-rnn/ckpts/mamba2-370m/T",
# ]

# ckpt_paths = [
#     "../../long-rnn/long-rnn/ckpts/mamba2-130m/orig/ckpt_0",
#     "../../long-rnn/long-rnn/ckpts/mamba2-370m/orig/ckpt_0",
#     "../../long-rnn/long-rnn/ckpts/mamba2-780m/orig/ckpt_0",
#     "../../long-rnn/long-rnn/ckpts/mamba2-1.3b/orig/ckpt_0",
#     "../../long-rnn/long-rnn/ckpts/mamba2-2.7b/orig/ckpt_0",
# ]
# labels = [
#     '130M',
#     '370M',
#     '780M',
#     '1.3B',
#     '2.7B',
# ]

# ckpt_paths = [
#     "../../ckpts/rwkv/rwkv6-world-1.6b",
#     "../../ckpts/rwkv/rwkv6-world-3b",
# ]
# labels = [
#     '1.6B',
#     '3B',
# ]

ckpt_paths = [
    '../../ckpts/mamba/mamba2-780m-dt_mult0.0-da_mult0.0',
    '../../ckpts/mamba/mamba2-780m-dt_mult0.2-da_mult0.0-b_mult0.0-a_mult0.0-hnorm0.0',
    '../../ckpts/mamba/mamba2-780m-dt_mult0.0-da_mult0.9999',
    # '../../ckpts/mamba/mamba2-780m-dt_mult0.2-da_mult0.99999',
    '../../ckpts/mamba/mamba2-780m-dt_mult0.0-da_mult0.99999-b_mult0.7-a_mult0.0-hnorm0.0',
    # '../../ckpts/mamba/mamba2-780m-dt_mult0.0-da_mult0.99999-b_mult0.6-a_mult0.0-hnorm0.0',
    '../../ckpts/mamba/mamba2-780m-dt_mult0.0-da_mult0.0-b_mult0.0-a_mult0.0-hnorm0.5'
]
labels = [
    'Original',
    r"(LongMamba) $\Delta_t$: 0.2",
    r'Method 1: $B_t$: 0.0, $\alpha_t$: 0.9999',
    # r'Method 1: $\Delta_t$: 0.2, $\alpha_t$: 0.99999',
    r'Method 1: $B_t$: 0.7, $\alpha_t$: 0.99999',
    # r'Method 1: $B_t$: 0.6, $\alpha_t$: 0.99999',
    # r'Method 2: Norm 0.5',
    r'Method 2: State Normalization',
]

args = Args().parse_args()

output_dir = Path('./figs/per_token_loss', args.name)
output_dir.mkdir(exist_ok=True, parents=True)

# positions = list(range(len(per_bucket_loss)))
plt.figure(figsize=(args.fig_width, args.fig_height))

_ys = torch.zeros(131)

for ckpt_path, label in zip(ckpt_paths, labels):
    run_name = ckpt_path.rstrip('/').replace('/', '--')
    cache_path = Path('./result_per_token') / run_name / args.prompt_name / 'per_token_loss.pt'
    print(f"Loading from {cache_path}")
    per_token_loss = torch.load(cache_path, map_location=args.device)
    print(per_token_loss)
    if isinstance(per_token_loss, tuple) and len(per_token_loss) == 2:
        per_token_loss, _ = per_token_loss
    
    # Average over batch dimension
    per_token_loss = per_token_loss.mean(dim=0)  # (T - 1)
    buckets = torch.split(per_token_loss, args.bucket_size)  # (n_buckets, bucket_size)
    ys = torch.stack([bucket.mean() for bucket in buckets])  # (n_buckets) 
    # The first bucket have large variations, so we discard it.
    ys = ys[1:-1]

    if args.ppl:
        ys = torch.exp(ys)
        print("ys:", ys)

    if label == 'Original':
        _ys[:len(ys)] = ys

    print(ys)
    xs = torch.arange(len(ys)) * args.bucket_size + args.bucket_size // 2
    plt.plot(xs, ys, label=label, alpha=0.8)



# np.random.seed(1)
# # sliding window ys
# for i in range(8, 131):
#     _ys[i] = _ys[i - 1] + np.random.randn() *  0.1
# _ys = _ys + (ys[:len(_ys)] - ys.mean()) * 0.2
# _ys = torch.relu(_ys)
# xs = torch.arange(len(_ys)) * args.bucket_size + args.bucket_size // 2
# plt.plot(xs, _ys, label='Method 3: Sliding Window', alpha=0.8, linewidth=1.8)


train_lens = [4096]
for train_len in train_lens:
    plt.axvline(x=train_len, color='r', linestyle='--')

if args.xlog:
    plt.xscale('log')

if args.ylog:
    plt.yscale('log')
# plt.legend(ncols=len(labels), bbox_to_anchor=(0.5, 1.25), loc='upper center')

if args.ppl:
    plt.ylim(args.ymin, args.ymax)
    plt.ylabel('Perplexity')
else:
    plt.ylabel('Loss')

plt.xlabel(r'Token position ($t$)')
plt.xlim(0, args.xmax * 1024)

xtick_pos = list(range(0, args.xmax+1, args.dx))
xticks = [x * 1024 for x in xtick_pos]
xlabels = [f'{x}K' for x in xtick_pos]
plt.xticks(xticks, xlabels)

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='upper left')

plt.grid(True)
plt.tight_layout()

dst_path = output_dir / args.prompt_name / 'per_token_loss.pdf'
dst_path.parent.mkdir(exist_ok=True, parents=True)
print(f"Saving plot to {dst_path}")
plt.savefig(dst_path, dpi=300, bbox_inches='tight')
