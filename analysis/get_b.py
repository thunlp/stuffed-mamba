from pathlib import Path
from typing import List

from torch import Tensor
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from arguments import Args
from modeling.mamba2.modeling_mamba2_torch import Mamba2ForCausalLM
from get_prompt import get_long_prompt


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


def plot_stats(
    B: Tensor,
    layer_indices: List[int],
    dst_path: Path,
    bucket_size: int = 128,
    # max_len: int = 16 * 1024,
    max_len: int = 128,
):
    '''
    B: (L, T, N)
    '''
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.5))
    axs[0].set_title('Mean')
    axs[1].set_title('Variance')
    for layer_i in layer_indices:
        print(f"Plotting for layer {layer_i}")
        layer_B = B[layer_i, :max_len]  # (T, N)

        mean = layer_B.mean(dim=1)  # (T)
        var = layer_B.var(dim=1)  # (T)

        for ax, ys in zip(axs, [mean, var]):
            buckets = torch.split(ys, bucket_size)
            ys = torch.stack([bucket.float().mean() for bucket in buckets])
            xs = torch.arange(len(ys)) * bucket_size
            
            ax.plot(xs, ys, label=f"Layer {layer_i}", alpha=0.6)
            ax.set_xlabel(r'Token position ($t$)')
            ax.axvline(x=train_len, color='r', linestyle='--')
    axs[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # plt.xlim(-10, max_len)
    
    plt.tight_layout()
    
    print(f"Saving to {dst_path}")
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')


def plot_heads(
    layer_K: Tensor,
    dst_path: Path,
    channel_indices: List[int],
    bucket_size: int = 128,
    max_len: int = 12 * 1024,
    add_legend: bool = False,
):
    """
    B: (T, nheads, N)
    """
    plt.figure(figsize=(1.8, 2.2))
    for channel_i in channel_indices:
        print(f"Plotting for channel {channel_i}")
        B = layer_K[:max_len, channel_i]  # (max_len,)
        buckets = torch.split(B, bucket_size)
        ys = torch.stack([bucket.mean() for bucket in buckets])
        xs = torch.arange(len(ys)) * bucket_size
        plt.plot(xs, ys, label=f"{channel_i}", alpha=0.6)

    plt.axvline(x=train_len, color='r', linestyle='--')
    if add_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel(r"$t$")
    # plt.ylabel(r"$B_t$")
    plt.tight_layout()
    
    print(f"Saving to {dst_path}")
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.clf()


def main():
    args = Args().parse_args()
    print("=========== args ===========")
    print(args)
    print("============================")
    model_size = '780m'
    global train_len
    train_len = args.train_len
    max_len = args.max_len

    model_path = Path(args.model_path)
    run_name = args.model_path.replace('/', '--')
    figs_dir = Path("./figs/b") / run_name / args.prompt_name
    cache_dir = Path('./cache/b') / run_name / args.prompt_name
    file_ext = 'pdf'
    prompt = "The capital of China is"
    device = 'cuda'
    dtype = torch.float32
    n_layers = {
        '130m': 24,
        '370m': 48,
        '780m': 48,
    }[model_size]
    chunk_size = 512

    def inspect_conv():
        ni = 98
        li = 38
        model = Mamba2ForCausalLM.from_pretrained(
            str(model_path), device=device).to(dtype=dtype)
        layer = model.backbone.layers[li].mixer
        conv = layer.conv1d.weight.clone().detach().cpu()
        # print(conv)
        print(conv.shape)
        channel_conv = conv[2048 + ni, 0]
        x_conv = conv[:2048]
        B_conv = conv[-256:-128]
        C_conv = conv[-128:]
        print(channel_conv)
        means = B_conv.mean(dim=(1, 2))
        vars_ = B_conv.var(dim=(1, 2))
        
        plt.figure(figsize=(5, 2.5))
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        xs = torch.arange(len(means))
        axs[0].set_title('Mean')
        axs[1].set_title('Variance')
        axs[0].plot(xs, means, alpha=0.6)
        axs[1].plot(xs, vars_, alpha=0.6)
        for ax in axs:
            ax.set_xlabel(r"Channel Index")
            # ax.axvline(x=2048, color='r', linestyle='--')
            ax.set_xlim(90, 100)
        plt.tight_layout()
        plt.savefig(f"conv_values_L{li}.pdf", bbox_inches='tight')
        
    inspect_conv()

    cache_dir.mkdir(exist_ok=True, parents=True)
    decays_path = cache_dir / f'B.pt'
    # states_path = Path(f'states-{model_size}.pt')
    if not decays_path.exists():
        print('====================================')
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.tok_path, trust_remote_code=True)
        print(f"Loading model from {model_path}")
        model = Mamba2ForCausalLM.from_pretrained(
            str(model_path), device=device).to(dtype=dtype)

        print(model)

        print("Tokenizing prompt...")
        prompt = get_long_prompt(args.prompt_name)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device=device)[0]
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
    dt = []
    A = []
    
    n_chunks = len(decays)
    layer_to_K = []
    print(f"# chunks: {n_chunks}")
    for layer_i in range(n_layers):
        chunk_Ks = []
        for chunk_i in range(n_chunks):
            layer_coeffs = decays[chunk_i][layer_i]
            # dt = layer_coeffs['dt'][0]  # (chunk_size, nheads)
            B = layer_coeffs['B'][0]  # (chunk_size, N)
            # B = einsum(dt, B, 'c h, c n -> c h n')  # (chunk_size, nheads, N)
            chunk_Ks.append(B)
        layer_to_K.append(torch.cat(chunk_Ks))  # [(T, N)]
    
    B = torch.stack(layer_to_K).cpu().float()  # (L, T, N)
    print(B.shape)
    figs_dir.mkdir(exist_ok=True, parents=True)

    def plot_all_layers():
        layer_chunk_size = 8
        for layer_lo in range(0, n_layers, layer_chunk_size):
            layer_indices = list(range(layer_lo, layer_lo + layer_chunk_size))
            print(f"Plotting for layers: {layer_lo} - {layer_lo + layer_chunk_size}")
            plot_stats(
                B,
                layer_indices,
                figs_dir / f"Layer{layer_lo}-{layer_lo + layer_chunk_size}.{file_ext}",
                # figs_dir / f"Layer-30.{file_ext}",
                max_len=max_len,
            )

    plot_all_layers()

    def plot_all_channels():
        head_chunk_size = 8
        nchannels = B.shape[2]
        print(f'# channels: {nchannels}')
        # for target_layer in range(n_layers):
        for target_layer in [38]:

            print(f"Plotting the individual channels for layer {target_layer}")
            for head_lo in range(0, nchannels, head_chunk_size):
                head_indices = list(range(head_lo, min(head_lo + head_chunk_size, nchannels)))
                print(f"Plotting for heads: {head_indices}")
                dst_dir = figs_dir / f'layer-{target_layer}'
                dst_dir.mkdir(exist_ok=True, parents=True)
                dst_path = dst_dir / f'channels-{head_lo}-{head_lo + head_chunk_size}.{file_ext}'
                plot_heads(
                    B[target_layer],
                    channel_indices=head_indices,
                    dst_path=dst_path,
                    max_len=max_len,
                )

    plot_all_channels()

    li = 38
    ni = 98
    pi = 11
    hi = 2

    vals = B[li, :, ni]  # (T)
    xs = torch.arange(len(vals))
    plt.figure(figsize=(8, 4))
    plt.plot(xs, vals, linewidth=0.7, alpha=0.5)
    plt.savefig(figs_dir / f'L{li}_H{hi}_N{ni}.{file_ext}')
        
    chunk_size = 128
    for i in range(0, len(vals), chunk_size):
        chunk = vals[i : i + chunk_size]
        print(i, i + chunk_size, chunk.mean(), chunk.var())



if __name__ == '__main__':
    main()
