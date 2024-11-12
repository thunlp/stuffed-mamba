from pathlib import Path
from typing import List

from torch import Tensor
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
def get_decays(
    model: Mamba2ForCausalLM,
    input_ids: Tensor,
    chunk_size: int = 128,
    n_layers: int = 48,
):
    cur_state = None
    all_decays = []
    seqlen = len(input_ids)
    print(f"Seq len: {seqlen}")
    for i in range(0, seqlen, chunk_size):
        this_inputs = input_ids[i : i + chunk_size].unsqueeze(0)
        # print(f"Chunk {i} - {i + chunk_size}")
        output = model(this_inputs, states=cur_state, return_decays=True)
        decays = output["decays"]
        cur_state = output['states']
        all_decays.append(decays)
    return all_decays


def plot_stats(
    layer_to_dt,
    layer_indices: List[int],
    dst_path: Path,
    bucket_size: int = 128,
    # max_len: int = 16 * 1024,
    max_len: int = 128,
):
    """
    dA: (L, T, nheads)
    """
    plt.figure(figsize=(3.5, 2.5))
    # Plot decay and cumulative prod for each layer
    for layer_i in layer_indices:
        print(f"Plotting for layer {layer_i}")
        dt = layer_to_dt[layer_i, :max_len]  # (L, T, nheads)
        dt = dt.mean(dim=1)  # (T)
        # ds = np.array([np.prod(ys[i:max_len]) for i in range(max_len)])

        # Get the smooths decay and coeff for each bucket.
        buckets = torch.split(dt, bucket_size)
        ys = [bucket.mean() for bucket in buckets]
        # ds = [ds[i] for i in range(0, max_len, bucket_size)]
        xs = torch.arange(len(ys)) * bucket_size
        plt.plot(xs, ys, label=f"Layer {layer_i}", alpha=0.6)
        # color1 = line1.get_color()
        # plt.plot(xs, ds, '--', color=color1, label=f"L{layer_i} Product", alpha=0.6)
    plt.axvline(x=train_len, color="r", linestyle="--")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    # plt.xlim(-10, max_len)
    plt.xlabel(r"Token position ($t$)")
    # plt.ylabel(r"$\Delta_t$")

    plt.tight_layout()
    print(f"Saving to {dst_path}")
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(dst_path, dpi=300, bbox_inches="tight")
    plt.clf()


def plot_heads(
    layer_dt: Tensor,
    dst_path: Path,
    head_indices: List[int],
    bucket_size: int = 128,
    max_len: int = 12 * 1024,
    add_legend: bool = False,
):
    """
    dA: (T, nheads)
    """
    if add_legend:
        plt.figure(figsize=(2.5, 2.2))
    else:
        plt.figure(figsize=(1.8, 2.2))
    for head_i in head_indices:
        dt = layer_dt[:, head_i]  # (T)
        buckets = torch.split(dt, bucket_size)
        ys = torch.stack([bucket.mean() for bucket in buckets])
        xs = torch.arange(len(ys)) * bucket_size
        plt.plot(xs, ys, label=f"{head_i}", alpha=0.6)
    plt.axvline(x=train_len, color="r", linestyle="--")
    if add_legend:
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.xlabel(r"$t$")
    # plt.ylabel(r"$\Delta_t$")
    plt.title(r"$\Delta_t$")
    # plt.ylim(0, 0.01)
    plt.tight_layout()
    print(f"Saving to {dst_path}")
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(dst_path, dpi=300, bbox_inches="tight")
    plt.clf()


def main():
    args = Args().parse_args()
    global train_len
    train_len = args.train_len
    # pretrained_name = f"mamba2-130m/orig/ckpt_0"
    # pretrained_name = f'../ckpts/mamba2-370m/T8192_B1_GA1_P8_SR16_RD0_lr0.0005/ckpt_100000'
    
    ckpt_dir = Path(args.ckpt_dir)
    run_name = args.pretrained_name.replace("/", "--")
    tok_path = args.tok_path
    figs_dir = Path("./figs/dt") / run_name / args.prompt_name
    cache_dir = Path("./cache/dt") / run_name / args.prompt_name
    file_ext = "pdf"
    prompt = "The capital of China is"
    device = "cuda"
    dtype = torch.float32
    chunk_size = 512

    cache_dir.mkdir(exist_ok=True, parents=True)
    decays_path = cache_dir / "dt.pt"
    # states_path = Path(f'states-{model_size}.pt')
    if not decays_path.exists():
        print("====================================")
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        print("Loading model...")
        model = Mamba2ForCausalLM.from_pretrained(
            ckpt_dir / args.pretrained_name, device=device
        ).to(dtype=dtype)

        n_layers = len(model.backbone.layers)

        print("Tokenizing prompt...")
        prompt = get_long_prompt()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device=device)[0]
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

    n_chunks = len(decays)
    layer_to_dt = []
    print(f"# chunks: {n_chunks}")
    for layer_i in range(n_layers):
        chunks = []
        for chunk_i in range(n_chunks):

            layer_coeffs = decays[chunk_i][layer_i]
            dt = layer_coeffs["dt"][0]  # (T, nheads)
            chunks.append(dt)
        layer_to_dt.append(torch.cat(chunks))  # [(T, nheads)]

    layer_to_dt = torch.stack(layer_to_dt).cpu().float()  # (L, T, nheads)
    # print(layer_to_dt)
    print(layer_to_dt.shape)

    figs_dir.mkdir(exist_ok=True, parents=True)
    nheads = len(layer_to_dt[0][0])

    def plot_all_layers():
        layer_chunk_size = 8
        for layer_lo in range(0, n_layers, layer_chunk_size):
            layer_indices = list(range(layer_lo, layer_lo + layer_chunk_size))
            print(f"Plotting for layers: {layer_lo} - {layer_lo + layer_chunk_size}")
            plot_stats(
                layer_to_dt,
                layer_indices,
                figs_dir
                / f"Layer{layer_lo}-{layer_lo + layer_chunk_size}.{file_ext}",
                max_len=args.max_len,
            )

    plot_all_layers()

    def plot_all_heads():
        head_chunk_size = 8
        print(f"# heads: {nheads}")
        # for target_layer in range(n_layers):
        for target_layer in [38]:
            print(f"Plotting the individual heads for layer {target_layer}")
            for head_lo in range(0, nheads, head_chunk_size):
                head_indices = list(range(head_lo, min(head_lo + head_chunk_size, nheads)))
                print(f"Plotting for heads: {head_indices}")
                dst_dir = figs_dir / f"layer-{target_layer}"
                dst_dir.mkdir(exist_ok=True, parents=True)
                dst_path = (
                    dst_dir
                    / f"Head-{head_lo}-{head_lo + head_chunk_size}.{file_ext}"
                )
                plot_heads(
                    layer_to_dt[target_layer],
                    head_indices=head_indices,
                    dst_path=dst_path,
                    max_len=args.max_len,
                )

    plot_all_heads()
    
    li = 38
    hi = 2
    for hi in range(nheads):
        vals = layer_to_dt[li, 8*1024:, hi]
        print("head", hi, vals.mean(), vals.var())

    vars_ = []
    means = []
    T = layer_to_dt.shape[1]
    chunk_size = 128
    for t_lo in range(0, T, chunk_size):
        vals = layer_to_dt[li, t_lo:t_lo + chunk_size, hi]  # (chunk_size)
        var = vals.var()
        mean = vals.mean()
        vars_.append(var)
        means.append(mean)
        print(t_lo, t_lo + chunk_size, mean, var)
    plt.figure(figsize=(4, 4))
    xs = torch.arange(len(means))
    plt.plot(xs, vars_)
    plt.axvline(x=train_len, linestyle='--', color='r')
    plt.plot(xs)
    plt.savefig(figs_dir / f'L{li}_H{hi}_dt_chunks.{file_ext}')


if __name__ == "__main__":
    main()
