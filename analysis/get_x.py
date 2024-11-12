from pathlib import Path
from typing import List

from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from arguments import Args
from get_prompt import get_long_prompt
from modeling.mamba2.modeling_mamba2_torch import Mamba2ForCausalLM


def get_tensor_stats(t):
    mean = torch.mean(t)
    var = torch.var(t)
    median = torch.median(t)
    mn = torch.min(t)
    mx = torch.max(t)
    return mean, var, mn, mx, median


@torch.no_grad()
def get_decays(model: Mamba2ForCausalLM, input_ids: Tensor, chunk_size: int = 128, n_layers: int = 48):
    cur_state = None
    # print(input_ids)
    ssm_stats = {i: [] for i in range(n_layers)}
    conv_stats = {i: [] for i in range(n_layers)}
    all_decays = []
    seqlen = len(input_ids)
    print(f"Seq len: {seqlen}")
    for i in range(0, seqlen, chunk_size):
        this_inputs = input_ids[i:i + chunk_size].unsqueeze(0)
        # print(f"Chunk {i} - {i + chunk_size}")
        output = model(this_inputs, states=cur_state, return_decays=True)
        decays = output['decays']
        cur_state = output['states']
        all_decays.append(decays)
    return all_decays


def smooth(
    x: np.ndarray,
    window_size: int = 4,
):
    length = x.shape[0]
    y = x.copy()
    for i in range(length):
        y[i] = np.mean(x[i - window_size : i])
    return y


def plot_stats(
    V: Tensor,
    layer_indices: List[int],
    dst_path: Path,
    bucket_size: int = 1024,
    # max_len: int = 16 * 1024,
    max_len: int = 128,
):
    '''
    V: (L, n_chunks, nheads, N)
    '''
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.2))
    axs[0].set_title('Mean')
    axs[1].set_title('Variance')
    for layer_i in layer_indices:
        print(f"Plotting for layer {layer_i}")
        layer_v = V[layer_i, :max_len // bucket_size]  # (n_chunks, nheads, N)
        layer_v = layer_v.mean(dim=1)  # (n_chunks, N)
        print(layer_v.shape)

        mean = layer_v.mean(dim=1)  # (n_chunks)
        var = layer_v.var(dim=1)  # (n_chunks)

        for ax, ys in zip(axs, [mean, var]):
            # buckets = torch.split(ys, bucket_size)
            # ys = torch.stack([bucket.float().mean() for bucket in buckets])
            xs = torch.arange(len(ys)) * bucket_size
            
            ax.plot(xs, ys, label=f"{layer_i}", alpha=0.6)
            ax.set_xlabel(r'($t$)')
            ax.axvline(x=train_len, color='r', linestyle='--')
    axs[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    
    print(f"Saving to {dst_path}")
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')


def plot_heads(
    layer_V: Tensor,
    dst_path: Path,
    head_indices: List[int],
    bucket_size: int = 1024,
    max_len: int = 12 * 1024,
    add_legend: bool = False,
):
    """
    V: (T, nheads, P)
    """
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.2))
    axs[0].set_title('Mean')
    axs[1].set_title('Var')
    for head_i in head_indices:
        print(f"Plotting for hea {head_i}")
        V = layer_V[:max_len, head_i]  # (P)

        mean = V.mean(dim=1)
        var = V.var(dim=1)

        for ax, ys in zip(axs, [mean, var]):
            # buckets = torch.split(ys, bucket_size)  # (n_buckets, bucket_size, means)
            # ys = torch.stack([bucket.float().mean() for bucket in buckets])
            xs = torch.arange(len(ys)) * bucket_size
            
            ax.plot(xs, ys, label=f"{head_i}", alpha=0.6)
            ax.set_xlabel(r'$t$')
            ax.axvline(x=train_len, color='r', linestyle='--')
    
    plt.tight_layout()
    if add_legend:
        axs[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    print(f"Saving to {dst_path}")
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.clf()


def main():
    args = Args().parse_args()
    print("=========== args ===========")
    print(args)
    print("============================")
    global train_len
    train_len = args.train_len
    max_len = args.max_len
    model_path = Path(args.model_path)
    
    run_name = args.model_path.replace('/', '--')
    file_ext = args.file_ext
    device = args.device
    dtype = torch.float32
    bucket_size = args.bucket_size
    figs_dir = Path("./figs/x") / run_name / args.prompt_name
    cache_dir = Path('./cache/x') / run_name / args.prompt_name

    cache_dir.mkdir(exist_ok=True, parents=True)
    decays_path = cache_dir / 'V.pt'
    # states_path = Path(f'states-{model_size}.pt')
    if not decays_path.exists():
        print('====================================')
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.tok_path, trust_remote_code=True)
        print(f"Loading model from {model_path}")
        model = Mamba2ForCausalLM.from_pretrained(str(model_path), device=device).to(dtype=dtype)

        n_layers = model.config.n_layer

        print("Tokenizing prompt...")
        prompt = get_long_prompt()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device=device)[0]
        print("Getting states...")
        truncate_len = input_ids.shape[0] // bucket_size * bucket_size
        print(f"Original length: {len(input_ids)}, truncated length: {truncate_len}")
        input_ids = input_ids[:truncate_len]
        decays = get_decays(model, input_ids, chunk_size=bucket_size, n_layers=n_layers)
        decays_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Saving stats to {decays_path}")
        torch.save(decays, decays_path)
    else:
        print(f"Loading states and stats from {decays_path}...")
        decays = torch.load(decays_path)

    # Move to CPU
    n_layers = len(decays[0])
    n_chunks = len(decays)
    X = []
    print(f"# chunks: {n_chunks}")
    for layer_i in range(n_layers):
        chunk_x = []
        for chunk_i in range(n_chunks):
            layer_coeffs = decays[chunk_i][layer_i]
            # dt = layer_coeffs['dt'][0].unsqueeze(2)  # (chunk_size, nheads, 1)
            x = layer_coeffs['x'][0]  # (chunk_size, nheads, P)
            x = x.mean(dim=0)  # (nheads, P)
            chunk_x.append(x)
        X.append(torch.stack(chunk_x))  # [(n_chunks, nheads, P), ]
    
    X = torch.stack(X).cpu()  # (L, n_chunks, nheads, P)
    print(X.shape)

    figs_dir.mkdir(exist_ok=True, parents=True)
    nheads = X.shape[2]

    def plot_all_layers():
        layer_chunk_size = 8
        for layer_lo in range(0, n_layers, layer_chunk_size):
            layer_indices = list(range(layer_lo, layer_lo + layer_chunk_size))
            print(f"Plotting for layers: {layer_lo} - {layer_lo + layer_chunk_size}")
            plot_stats(
                X,
                layer_indices,
                figs_dir / f"Layer{layer_lo}-{layer_lo + layer_chunk_size}.{file_ext}",
                max_len=max_len,
                bucket_size=bucket_size,
            )
        
    plot_all_layers()

    def plot_all_heads():
        head_chunk_size = 8
        print(f'# heads: {nheads}')
        # for target_layer in range(n_layers):
        for target_layer in [38]:
            print(f"Plotting the individual heads for layer {target_layer}")
            for head_lo in range(0, nheads, head_chunk_size):
                head_indices = list(range(head_lo, head_lo + head_chunk_size))
                print(f"Plotting for heads: {head_indices}")
                dst_dir = figs_dir / f'layer-{target_layer}'
                dst_dir.mkdir(exist_ok=True, parents=True)
                dst_path = dst_dir / f'Head-{head_lo}-{head_lo + head_chunk_size}.{file_ext}'
                plot_heads(
                    X[target_layer],
                    head_indices=head_indices,
                    dst_path=dst_path,
                    max_len=max_len,
                    bucket_size=bucket_size,
                )

    plot_all_heads()
    
    # plot x[22]
    target_layer = 38
    target_head = 2
    target_pi = 11
    T = X.shape[1]
    ys = X[target_layer, :max_len, target_head, target_pi].float()  # (T)
    xs = torch.arange(len(ys))

    plt.figure(figsize=(3, 3))
    plt.plot(xs, ys)
    plt.xlabel(r"Token position ($t$)")
    plt.ylabel(r"$x_t[8]$")
    plt.axvline(x=train_len, color='r', linestyle='--')    
    plt.tight_layout()
    plt.savefig(figs_dir / f"L{target_layer}_H{target_head}_p{target_pi}.{file_ext}", bbox_inches='tight')
    
    bucket_size = 512
    for i in range(0, max_len, bucket_size):
        chunk = ys[i : i + bucket_size]
        print(i, i + bucket_size, chunk.mean(), chunk.var())
    
    with open('vals.txt', 'w') as f:
        for y in ys.tolist():
            f.write(str(y))
            f.write('\n')
    

if __name__ == '__main__':
    main()
