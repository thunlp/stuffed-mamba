from pathlib import Path
from torch import Tensor
from typing import List
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer

from modeling.mamba2.modeling_mamba2_torch import Mamba2ForCausalLM
from get_prompt import get_long_prompt
from arguments import Args


def get_tensor_stats(t):
    mean = torch.mean(t)
    var = torch.var(t)
    median = torch.median(t)
    mn = torch.min(t)
    mx = torch.max(t)
    return mean, var, mn, mx, median


@torch.no_grad()
def get_decays(model: Mamba2ForCausalLM, input_ids: Tensor, chunk_size: int = 128):
    cur_state = None
    all_decays = []
    seqlen = len(input_ids)
    print(f"Seq len: {seqlen}")
    for i in range(0, seqlen, chunk_size):
        this_inputs = input_ids[i:i + chunk_size].unsqueeze(0)
        output = model(this_inputs, states=cur_state, return_decays=True)
        decays = output['decays']
        cur_state = output['states']
        all_decays.append(decays)
    return all_decays


def plot_stats(
    layer_to_dA: Tensor,
    layer_indices: List[int],
    dst_path: Path,
    bucket_size: int = 128,
    # max_len: int = 16 * 1024,
    max_len: int = 128,
):
    '''
    dA: (L, T, nheads)
    '''
    plt.figure(figsize=(3.5, 2.5))
    n_buckets = len(layer_to_dA[0]) // bucket_size
    # Plot decay and cumulative prod for each layer
    for layer_i in layer_indices:
        print(f"Plotting for layer {layer_i}")
        dA = layer_to_dA[layer_i]  # (T, nheads)
        dA = dA.mean(dim=1)  # (T)
        dA = dA[:max_len]  # (T)
        
        buckets = torch.split(dA, bucket_size)
        ys = torch.stack([bucket.float().mean() for bucket in buckets])
        xs = torch.arange(len(ys)) * bucket_size
        line1, = plt.plot(xs, ys, label=f"Layer {layer_i}", alpha=0.6)
        # color1 = line1.get_color()
        # plt.plot(xs, ds, '--', color=color1, label=f"L{layer_i} Product", alpha=0.6)
    plt.axvline(x=train_len, color='r', linestyle='--')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlim(-10, max_len)
    plt.xlabel(r'Token position ($t$)')
    plt.ylabel(r'Decay ($\alpha_t$)')
    
    plt.tight_layout()
    print(f"Saving to {dst_path}")
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')


def plot_heads(
    layer_dA: Tensor,
    dst_path: Path,
    head_indices: List[int],
    bucket_size: int = 512,
    max_len: int = 12 * 1024,
):
    """
    dA: (T, nheads)
    """
    n_chunks: int = len(layer_dA)
    plt.figure(figsize=(2.5, 2.2))
    for head_i in head_indices:
        buckets = torch.split(layer_dA[:, head_i], bucket_size)  # (n_buckets, bucket_size, nheads)
        ys = torch.stack([bucket.float().mean(dim=0) for bucket in buckets])  # (nheads)
        # ys = [torch.mean(layer_dA[i: i + bucket_size, head_i]) for i in range(0, max_len, bucket_size)]
        xs = torch.arange(len(ys)) * bucket_size
        plt.plot(xs, ys, label=f'Head {head_i}', alpha=0.6)
    plt.axvline(x=train_len, color='r', linestyle='--')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel(r"Token position ($t$)")
    plt.ylabel(r"Decay ($\alpha_t$)")
    plt.ylim(0.9, 1.0)
    print(f"Saving to {dst_path}")
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.clf()


def plot_seq_mean(
    seq_mean,
    dst_path: Path,
):
    '''
    seq_mean: (L, nheads)
    '''
    L, H = seq_mean.shape
    ys = []
    xs = []
    for i in range(L):
        for j in range(H):
            ys.append(seq_mean[i, j])
            xs.append(i)
    plt.figure(figsize=(5, 3))
    plt.xlabel('Layer index')
    plt.ylabel(r'Decay ($\alpha_t$)')
    plt.scatter(xs, ys, alpha=0.4)
    plt.grid(True)
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.clf()


def main():
    args = Args().parse_args()  
    model_size = '370m'
    global train_len
    train_len = args.train_len
    model_path = Path(args.model_path)

    run_name = args.model_path.replace('/', '--')
    tok_path = args.tok_path
    figs_dir = Path("./figs/dA") / run_name / args.prompt_name
    cache_dir = Path('./cache/dA') / run_name / args.prompt_name
    
    dtype = torch.float32
    n_layers = {
        '130m': 24,
        '370m': 48,
        '780m': 48,
    }[model_size]
    chunk_size = 512
    bucket_size = 512

    cache_dir.mkdir(exist_ok=True, parents=True)
    decays_path = cache_dir / f'decay-coeffs.pt'
    # states_path = Path(f'states-{model_size}.pt')
    if not decays_path.exists():
        print('====================================')
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        print("Loading model...")
        model = Mamba2ForCausalLM.from_pretrained(
            str(model_path),
            device=args.device,
        ).to(dtype=dtype)

        # print(model)

        print("Tokenizing prompt...")
        prompt = get_long_prompt()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device=args.device)[0]
        print("Getting states...")
        truncate_len = input_ids.shape[0] // chunk_size * chunk_size
        print(f"Original length: {len(input_ids)}, truncated length: {truncate_len}")
        input_ids = input_ids[:truncate_len]
        decays = get_decays(model, input_ids, chunk_size=chunk_size, n_layers=n_layers)
        decays_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Saving stats to {decays_path}")
        torch.save(decays, decays_path)
    else:
        print(f"Loading states and stats from {decays_path}...")
        decays = torch.load(decays_path)

    # Move to CPU
    n_layers = len(decays[0])
    n_chunks = len(decays)
    layer_to_dA = []
    print(f"# chunks: {n_chunks}")
    for layer_i in range(n_layers):
        chunks = []
        for chunk_i in range(n_chunks):
            
            layer_coeffs = decays[chunk_i][layer_i]
            dt = layer_coeffs['dt'][0]  # (T, nheads)
            A = layer_coeffs['A']  # (nheads)
            if chunk_i == 0:
                print('A', layer_i, A)
            dA = torch.exp(dt * A)  # (T, nheads)
            chunks.append(dA)
        layer_to_dA.append(torch.cat(chunks))  # (T, nheads)

    layer_to_dA = torch.stack(layer_to_dA).cpu().float()  # (L, T, nheads)
    
    print(layer_to_dA.shape)
    seq_mean = layer_to_dA.mean(dim=1)  # (L, nheads)
    print(seq_mean.shape)
    torch.save(seq_mean, cache_dir / 'seq_mean.pt')
    plot_seq_mean(seq_mean, figs_dir / f'seq_mean.{args.file_ext}')

    figs_dir.mkdir(exist_ok=True, parents=True)
    nheads = len(layer_to_dA[0][0])

    def plot_all_layers():
        layer_chunk_size = 8
        for layer_lo in range(0, n_layers, layer_chunk_size):
            layer_indices = list(range(layer_lo, layer_lo + layer_chunk_size))
            print(f"Plotting for layers: {layer_lo} - {layer_lo + layer_chunk_size}")
            plot_stats(
                layer_to_dA,
                layer_indices,
                figs_dir / f"Layer{layer_lo}-{layer_lo + layer_chunk_size}.{args.file_ext}",
                max_len=args.max_len,
                bucket_size=bucket_size,
            )
        
    plot_all_layers()

    def plot_all_heads():
        head_chunk_size = 8
        print(f'# heads: {nheads}')
        for target_layer in range(n_layers):
            print(f"Plotting the individual heads for layer {target_layer}")
            for head_lo in range(0, nheads, head_chunk_size):
                head_indices = list(range(head_lo, min(head_lo + head_chunk_size, nheads)))
                print(f"Plotting for heads: {head_indices}")
                dst_dir = figs_dir / f'layer-{target_layer}'
                dst_dir.mkdir(exist_ok=True, parents=True)
                dst_path = dst_dir / f'Head-{head_lo}-{head_lo + head_chunk_size}.{args.file_ext}'
                plot_heads(
                    layer_to_dA[target_layer],
                    head_indices=head_indices,
                    dst_path=dst_path,
                    max_len=args.max_len,
                )

        head_chunk_size = 8
        print(f'# heads: {nheads}')
        for target_layer in [10]:
            print(f"Plotting the individual heads for layer {target_layer}")
            for head_lo in range(0, nheads, head_chunk_size):
                head_indices = list(range(head_lo, head_lo + head_chunk_size))
                print(f"Plotting for heads: {head_indices}")
                dst_dir = figs_dir / f'layer-{target_layer}'
                dst_dir.mkdir(exist_ok=True, parents=True)
                dst_path = dst_dir / f'Head-{head_lo}-{head_lo + head_chunk_size}.{args.file_ext}'
                plot_heads(
                    layer_to_dA[target_layer],
                    head_indices=head_indices,
                    dst_path=dst_path,
                    max_len=args.max_len,
                )

    plot_all_heads()
    
    li = 38
    print("Head, Decay Mean, Decay Variance")
    for hi in range(nheads):
        vals = layer_to_dA[li, :, hi]
        print(vals)
        print(hi, vals.mean(), vals.var())
    

    # Plot product curve
    plt.figure(figsize=(2.5, 2.2))
    for hi in range(0, 8):
        vals = layer_to_dA[li, :, hi]  # (T)
        ys = torch.cumprod(vals, dim=0)  # (T)
        xs = torch.arange(len(vals))
        plt.plot(xs, ys, alpha=0.5, label=f'{hi}')
    plt.axvline(x=train_len, color='r', linestyle='--')
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # plt.ylabel(r"$\alpha_{1:t}$")
    # plt.title(r"$\alpha_{1:t}$")
    plt.xlabel(r"$t$")
    plt.xlim(0, 30000)
    plt.tight_layout()

    dst_path = figs_dir / f'prods-L{li}-H0-8.{args.file_ext}'
    print(f"Saving to {dst_path}")
    plt.savefig(dst_path, bbox_inches='tight')
    plt.clf()

    hi = 2
    vals = layer_to_dA[li, :, hi]
    xs = torch.arange(len(vals))
    plt.figure(figsize=(4, 4))
    plt.plot(xs, vals, alpha=0.5)
    plt.axvline(x=train_len, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(figs_dir / 'all_alpha.pdf', bbox_inches='tight')
    plt.clf()
    
    with open('vals.txt', 'w') as f:
        for val in vals:
            f.write(str(float(val.numpy())))
            f.write('\n')


if __name__ == '__main__':
    main()
