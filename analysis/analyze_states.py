'''
For plotting mean and var of the states as a function of token position.
'''
from pathlib import Path
from typing import List

from torch import Tensor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
figsize = (6, 3)
bbox_to_anchor = (1.0, 0.5)

import torch
from modeling.mamba2.modeling_mamba2_torch import Mamba2ForCausalLM
from transformers import AutoTokenizer
from get_prompt import get_long_prompt
from tap import Tap


def get_tensor_stats(t):
    mean = torch.mean(t)
    var = torch.var(t)
    norm = torch.sqrt(torch.mean(t * t))  # L2-norm
    median = torch.median(t)
    # mn = torch.min(t)
    # mx = torch.max(t)
    return mean, var, norm, median


@torch.no_grad()
def get_stats(
    model: Mamba2ForCausalLM,
    input_ids: Tensor,
    n_layers: int = 48,
    bucket_size: int = 128,
    return_states: bool = True,
):
    print(f"Getting stats on prompt with {input_ids.shape[0]} tokens...")
    cur_state = None
    truncate_len = input_ids.shape[0] // bucket_size * bucket_size
    print(f"Original length: {len(input_ids)}, truncated length: {truncate_len}")
    input_ids = input_ids[:truncate_len]

    ssm_stats = {i: [] for i in range(n_layers)}
    conv_stats = {i: [] for i in range(n_layers)}
    states = []
    seqlen = len(input_ids)

    li = 38
    h_lo = 0
    h_hi = 8
    target_heads = list(range(h_lo, h_hi))
    file_ext = 'pdf'
    # hi = 5
    # ni = 91
    # pi = 8
    stats = []  # [(P, N),] * n_buckets * len(target_heads)
    print(f"Seq len: {seqlen}")
    for i in range(0, seqlen, bucket_size):
        this_inputs = input_ids[i:i + bucket_size].unsqueeze(0)
        print(f"Processing chunk [{i}, {i + bucket_size}]")
        output = model(this_inputs, states=cur_state)
        cur_state = output['states']
        if return_states:
            states.append([tuple(cur_state[li][j].clone() for j in range(2)) for li in range(n_layers)])

        ssm_state = cur_state[li][1][0]  # (H, P, N)
        nheads = ssm_state.shape[0]
        for hi in target_heads:
            stats.append(get_tensor_stats(ssm_state[hi]))


        # for li in range(n_layers):
        #     conv_state = cur_state[li][0]
        #     ssm_state = cur_state[li][1]  # (B, H, P, N)
        #     conv_stats[li].append(get_tensor_stats(conv_state))
        #     ssm_stats[li].append(get_tensor_stats(ssm_state))
        torch.mps.empty_cache()

    cache_path = f'L{li}_H{h_lo}-{h_hi}_stats.pt'
    print(f"Saving stats to {cache_path}")
    torch.save(stats, cache_path)

    metric_names = ['Mean', 'Variance']
    fig, axs = plt.subplots(1, len(metric_names), figsize=(len(metric_names) * 2.5, 2.5))
    n_chunks = len(stats[0])
    print(f"# chunks: {n_chunks}")
    for i, ax in enumerate(axs):
        for layer_i in [li]:
            metrics = [stats[layer_i][k][i] for k in range(n_chunks)]
            xs = torch.arange(len(metrics)) * bucket_size
            ax.plot(xs, metrics, label=f"Layer {layer_i}", alpha=0.6)
        ax.set_title(metric_names[i])
        ax.axvline(x=train_len, color='r', linestyle='--')
        ax.set_xlabel(r'Token position ($t$)')
        ax.set_xticks([x * 1024 for x in [10, 20, 30]])
        ax.set_xticklabels([f"{x}K" for x in [10, 20, 30]])

    axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()
    dst_path = f'L{li}_H{h_lo}-{h_hi}_stats.{file_ext}'
    plt.tight_layout()
    print(f"Saving to {dst_path}")
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')

    return states, conv_stats, ssm_stats


def plot_stats(stats, layer_indices: List[int], dst_path: Path, bucket_size: int = 128):
    '''
    stats: (L, C, 5)
    '''
    metric_names = ['Mean', 'Variance']
    fig, axs = plt.subplots(1, len(metric_names), figsize=(len(metric_names) * 2.5, 2.5))
    n_chunks = len(stats[0])
    print(f"# chunks: {n_chunks}")
    for i, ax in enumerate(axs):
        for layer_i in layer_indices:
            metrics = [stats[layer_i][k][i] for k in range(n_chunks)]
            xs = torch.arange(len(metrics)) * bucket_size
            ax.plot(xs, metrics, label=f"Layer {layer_i}", alpha=0.6)
        ax.set_title(metric_names[i])
        ax.axvline(x=train_len, color='r', linestyle='--')
        ax.set_xlabel(r'Token position ($t$)')
        ax.set_xticks([x * 1024 for x in [10, 20, 30]])
        ax.set_xticklabels([f"{x}K" for x in [10, 20, 30]])
    axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()
    plt.tight_layout()
    print(f"Saving to {dst_path}")
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')


def plot_heads(states, dst_dir: Path, target_layer: int, bucket_size: int = 128, file_ext: str = 'pdf'):
    """
    states: (L, C, H, P, N)
    """
    n_layers = states.shape[0]
    n_chunks: int = states.shape[1]
    nheads: int = states.shape[2]
    print(f"# layers: {n_layers}")
    print(f"# chunks: {n_chunks}")
    print(f"# heads: {nheads}")
    metric_names = ['Mean', 'Variance']

    head_stats = []  # {0 ~ nheads: [(metric0, ), ...]}
    for h in range(nheads):
        metrics = []
        for c in range(n_chunks):
            this_state = states[target_layer, c, h]
            metrics.append(list(get_tensor_stats(this_state)))
        metrics = torch.tensor(metrics)
        head_stats.append(metrics)
    head_stats = torch.stack(head_stats)
    print(head_stats.shape)

    head_chunk_size = 8
    for lo in range(0, nheads, head_chunk_size):
        fig, axs = plt.subplots(1, len(metric_names), figsize=(len(metric_names) * 2.5, 2.5))
        dst_path = dst_dir / f"{lo}-{lo + head_chunk_size}.{file_ext}"
        head_indices = list(range(lo, lo + head_chunk_size))
        for i, ax in enumerate(axs):
            for h in head_indices:
                metrics = [head_stats[h, c, i] for c in range(n_chunks)]
                xs = torch.arange(n_chunks) * bucket_size
                ax.plot(xs, metrics, label=f'Head {h}', alpha=0.6)
            ax.set_title(metric_names[i])
            ax.axvline(x=train_len, color='r', linestyle='--')
            ax.set_xlabel(r"Token position ($t$)")
            ax.set_xticks([x * 1024 for x in [10, 20, 30]])
            ax.set_xticklabels([f"{x}K" for x in [10, 20, 30]])

        axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # axs[-1].legend(loc='center left')
        dst_dir.mkdir(exist_ok=True, parents=True)
        # axs[-1].set_ylim(0, 2)

        plt.tight_layout()

        print(f"Saving to {dst_path}")
        plt.savefig(dst_path, dpi=300, bbox_inches='tight')
        plt.clf()


class Args(Tap):
    # pretrained_name = f'mamba2-780m/T8192_B1_GA2_P8_SR16_RD0_lr0.0002/ckpt_40000'
    # pretrained_name = f'mamba2-130m/orig/ckpt_0'
    # pretrained_name = f'mamba2-130m/T16384_B4_GA1_P1_SR1_RD0_lr0.0005/ckpt_100000'
    # pretrained_name = f"mamba2-512-8/T8192_B8_GA1_P1_SR1_RD0_lr0.0005/ckpt_350000"
    # pretrained_name = f'mamba2-130m/T4096_B16_GA1_P1_SR1_RD0_lr0.0005/ckpt_100000'
    # pretrained_name = f'mamba2-130m/T2048_B32_GA1_P1_SR1_RD0_lr0.0005/ckpt_100000'
    pretrained_name = f'mamba2-370m/orig/ckpt_0'
    # pretrained_name = f'mamba2-370m/T8192_B1_GA1_P8_SR16_RD0_lr0.0005/ckpt_100000'
    # pretrained_name = f'mamba2-780m/orig/ckpt_0'

    # prompt_name: str = 'capital'
    prompt_name: str = 'nextlines'
    train_len = 8 * 1024
    device: str = 'cuda'
    max_len: int = 35 * 1024
    chunk_size: int = 128
    file_ext: str = 'pdf'
    ckpt_dir: str = '../ckpts'
    tok_path: str = '../tokenizers/mamba-tok'
    bucket_size: int = 128
    get_states: int = 1
    overwrite: int = 0


def main():
    args = Args().parse_args()
    print("======== args ========")
    print(args)
    print("======================")

    global train_len
    train_len = args.train_len

    run_name = args.pretrained_name.replace('/', '--')
    ckpt_dir = Path(args.ckpt_dir)
    figs_dir = Path("./figs/states") / run_name / args.prompt_name
    cache_dir = Path('./cache/states') / run_name / args.prompt_name
    dtype = torch.float32

    figs_dir.mkdir(exist_ok=True, parents=True)
    cache_dir.mkdir(exist_ok=True, parents=True)
    stats_path = cache_dir / 'stats.pt'
    states_path = cache_dir / 'states.pt'
    if args.overwrite or (not stats_path.exists() or not states_path.exists()):
        print("====================================")
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.tok_path, trust_remote_code=True)
        print("Loading model...")
        model = Mamba2ForCausalLM.from_pretrained(ckpt_dir / args.pretrained_name, device=args.device).to(dtype=dtype)
        
        n_layers = model.config.n_layer
        prompt = get_long_prompt(args.prompt_name)
        print("Tokenizing prompt...")
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device=args.device)[0]
        print("Getting states...")
        states, conv_stats, ssm_stats = get_stats(
            model,
            input_ids,
            n_layers=n_layers,
            bucket_size=args.bucket_size,
            return_states=args.get_states,
        )
        stats = (conv_stats, ssm_stats)
        print(f"Saving stats to {stats_path}")
        torch.save(stats, stats_path)
        print(f"Saving states to {states_path}")
        torch.save(states, states_path)
    else:
        print(f"Loading states and stats from {stats_path} and {states_path}...")
        stats = torch.load(stats_path)
        states = torch.load(states_path)

    n_layers = len(stats[0])

    # Stack and Move to CPU
    new_stats = []
    for i in range(2):
        for layer_i, layer_stats in stats[i].items():
            all_chunk_stats = []
            for chunk_i, chunk_stats in enumerate(layer_stats):
                all_chunk_stats.append(torch.tensor(list(chunk_stats), dtype=torch.float))
            stats[i][layer_i] = torch.stack(all_chunk_stats, dim=0)
        this_stats = [stats[i][j] for j in range(len(stats[i]))]
        new_stats.append(torch.stack(this_stats))

    figs_dir.mkdir(exist_ok=True, parents=True)
    conv_stats, ssm_stats = new_stats
    print(ssm_stats.shape, conv_stats.shape)
    conv_stats = conv_stats[:, :args.max_len // args.bucket_size]
    ssm_stats = ssm_stats[:, :args.max_len // args.bucket_size]
    print(ssm_stats.shape, conv_stats.shape)
    
    def plot_all_layers():
        layer_chunk_size = 8
        for lo in range(0, n_layers, layer_chunk_size):
            layer_indices = list(range(lo, lo + layer_chunk_size))
            conv_dir = figs_dir / 'conv'
            ssm_dir = figs_dir / 'ssm'
            conv_dir.mkdir(exist_ok=True)
            ssm_dir.mkdir(exist_ok=True)
            print(f"Plotting for layers: {layer_indices}")
            plot_stats(
                conv_stats,
                layer_indices,
                conv_dir / f"{layer_indices[0]}-{layer_indices[-1]}.{args.file_ext}",
                bucket_size=args.bucket_size,
            )
            plot_stats(
                ssm_stats,
                layer_indices,
                ssm_dir / f"{layer_indices[0]}-{layer_indices[-1]}.{args.file_ext}",
                bucket_size=args.bucket_size,
            )

    # plot_all_layers()

    # exit()

    # states: (C, L, 2, B, H, P, N)
    n_chunks = len(states)
    n_layers = len(states[0])
    layer_to_states = []
    for li in range(n_layers):
        layer_state = torch.stack([states[ci][li][1].squeeze() for ci in range(n_chunks)])  # (C, nheads, P, N)
        # print(layer_state.shape)
        layer_to_states.append(layer_state)
    ssm_states = torch.stack(layer_to_states)  # (L, C, nheads, P, N)
    print(ssm_states.shape)
    ssm_states = ssm_states[:, :args.max_len // args.chunk_size]

    def plot_all_heads():
        # for target_layer in [20]:
        for target_layer in range(n_layers):
            dst_dir = figs_dir / f'heads-{target_layer}'
            plot_heads(ssm_states, dst_dir, target_layer, chunk_size=args.chunk_size)

    # plot_all_heads()

    target_head = 2
    target_layer = 38
    
    # target_head = 11
    # target_layer = 20
    state_elems_dir = figs_dir / f'state-elems-L{target_layer}-H{target_head}'
    state_elems_dir.mkdir(exist_ok=True, parents=True)
    head_states = ssm_states[target_layer, :, target_head].float().cpu()  # (C, P, N)
    print(head_states.shape)

    def plot_all_vectors():
        # P = 64
        for ni in range(128):
            pi_chunk = 8
            p_min = 0
            p_max = 64
            for p_lo in range(p_min, p_max, pi_chunk):
                plt.figure(figsize=(4, 3))
                for pi in range(p_lo, p_lo + pi_chunk):
                    ys = head_states[:, pi, ni]  # (C)
                    xs = torch.arange(len(ys)) * args.chunk_size
                    plt.plot(xs, ys, label=f"pi = {pi}", alpha=0.5)
                plt.axvline(x=train_len, color='r', linestyle='--')
                plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
                output_dir = state_elems_dir / 'each-elem'
                output_dir.mkdir(exist_ok=True)
                dst_path = output_dir / f'n{ni}_p-{p_lo}-{p_lo + pi_chunk}.{args.file_ext}'
                print(f"Saving to {dst_path}")
                plt.savefig(dst_path, bbox_inches='tight')
                plt.clf()

    # plot_all_vectors()
    
    def plot_channel_distr():
        n_chunks = head_states.shape[0]
        output_dir = state_elems_dir / 'channel-distr'

        colors = ['orange', 'green']
        labels = [r"$t=20K$", r"$t=8K$"]
        chunk_indices = [156, 64]

        # Combine the data to get the global min and max
        all_vals = np.concatenate([head_states[ci].view(-1).numpy() for ci in chunk_indices])
        
        # Define the bins based on the global range
        min_val, max_val = all_vals.min(), all_vals.max()
        bins = np.linspace(min_val, max_val, 40)  # Create 50 bins with equal width across the range

        plt.figure(figsize=(2.8, 2.2))
        for ci, label, color in zip(chunk_indices, labels, colors):
            vals = head_states[ci].view(-1)
            plt.hist(vals.numpy(), bins=bins, label=label, color=color, edgecolor=None)
        plt.xlabel('Value')
        plt.ylabel('Count')

        plt.yscale('log')
        plt.yticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])

        plt.legend()
        plt.tight_layout()
        dst_path = output_dir / f'_t-8K-and-20K.{args.file_ext}'

        
        print(f"Saving to {dst_path}")
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(dst_path, bbox_inches='tight')
        plt.clf()
        exit()

        for ci in range(n_chunks):
            plt.figure(figsize=(2.2, 2.2))
            vals = head_states[ci].view(-1)
            plt.hist(vals.numpy(), bins=50)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.yscale('log')
            plt.tight_layout()
            dst_path = output_dir / f'C{ci}.{file_ext}'
            
            print(f"Saving to {dst_path}")
            output_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(dst_path, bbox_inches='tight')
            plt.clf()
            
    plot_channel_distr()
    exit()
    
    def plot_heatmaps():
        output_dir = state_elems_dir / 'heatmaps'
        output_dir.mkdir(exist_ok=True, parents=True)
        for ci in range(head_states.shape[0]):
            plt.clf()
            plt.imshow(head_states[ci], interpolation='none', cmap='coolwarm')
            plt.colorbar(shrink=0.6)
            dst_path = output_dir / f'c{ci}.png'
            print(f"Saving to {dst_path}")
            plt.savefig(dst_path, bbox_inches='tight', dpi=300)

    plot_heatmaps()
    exit()

    def plot_pi(pi=8):
        # pi = 20
        plt.figure(figsize=(2.4, 2.4))
        # for ni in range(8):
        #     ys = head_states[:, pi, ni]  # (C)
        #     xs = [i * chunk_size for i in range(len(ys))]
        #     plt.plot(xs, ys, label=f"{ni}", alpha=0.5)
        ys = head_states[:, pi].var(axis=1)  # (C)
        xs = np.arange(len(ys)) * chunk_size  # (C)
        plt.plot(xs, ys)
        plt.axvline(x=train_len, color='r', linestyle='--')
        # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        dst_path = state_elems_dir / f'p{pi}_var.{file_ext}'
        print(f"Saving to {dst_path}")
        plt.savefig(dst_path, bbox_inches='tight')

        vecs = head_states[:, pi]  # (C, N)
        sims = []
        angles = []
        for i in range(len(vecs) - 1):
            cosine_angle = np.dot(vecs[i], vecs[i + 1]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i + 1]))
            angle = np.arccos(cosine_angle)
            sims.append(cosine_angle)
            angles.append(angle)

        plt.clf()
        plt.plot(xs[:-1], sims)
        plt.tight_layout()
        plt.axvline(x=train_len, color='r', linestyle='--')
        dst_path = state_elems_dir / f'p{pi}_cossims.{file_ext}'
        plt.savefig(dst_path, bbox_inches='tight')

        plt.clf()
        plt.plot(xs[:-1], sims)
        plt.tight_layout()
        plt.axvline(x=train_len, color='r', linestyle='--')
        dst_path = state_elems_dir / f'p{pi}_angles.{file_ext}'
        plt.savefig(dst_path, bbox_inches='tight')

    plot_pi()

    def plot_ni(ni=91):
        plt.figure(figsize=(2.4, 2.4))
        for pi in range(12, 20):
            ys = head_states[:, pi, ni]  # (C)
            xs = torch.arange(len(ys)) * chunk_size
            plt.plot(xs, ys, label=f"pi={pi}", alpha=0.5)
        # ys = head_states[:, :, ni].var(axis=1)  # (C, P)
        # xs = np.arange(len(ys)) * chunk_size  # (C)
        # plt.plot(xs, ys)
        plt.axvline(x=train_len, color='r', linestyle='--')
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        dst_path = state_elems_dir / f'n{ni}_var.{file_ext}'
        print(f"Saving to {dst_path}")
        plt.savefig(dst_path, bbox_inches='tight')

        vecs = head_states[:, pi]  # (C, N)
        sims = []
        angles = []
        for i in range(len(vecs) - 1):
            cosine_angle = np.dot(vecs[i], vecs[i + 1]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i + 1]))
            angle = np.arccos(cosine_angle)
            sims.append(cosine_angle)
            angles.append(angle)

        plt.clf()
        plt.plot(xs[:-1], sims)
        plt.tight_layout()
        plt.axvline(x=train_len, color='r', linestyle='--')
        dst_path = state_elems_dir / f'n{ni}_cossims.{file_ext}'
        plt.savefig(dst_path, bbox_inches='tight')

        plt.clf()
        plt.plot(xs[:-1], sims)
        plt.tight_layout()
        plt.axvline(x=train_len, color='r', linestyle='--')
        dst_path = state_elems_dir / f'n{ni}_angles.{file_ext}'
        plt.savefig(dst_path, bbox_inches='tight')

    plot_ni()
    
    def plot_elem(li=20, hi=11, pi=17, ni=91):
        vals = ssm_states[li, :, hi, pi, ni].float().cpu()  # (T)
        xs = torch.arange(len(vals)) * chunk_size
        plt.figure(figsize=(2.4, 2.4))
        plt.plot(xs, vals)
        plt.axvline(x=train_len, color='r', linestyle='--')
        # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        dst_path = state_elems_dir / f'L{li}_H{hi}_N{ni}_P{pi}.{file_ext}'
        print(f"Saving to {dst_path}")
        plt.savefig(dst_path, bbox_inches='tight')

    plot_elem()




if __name__ == "__main__":
    main()
