from pathlib import Path
import torch
import matplotlib.pyplot as plt



def plot_seq_mean(
    seq_mean,
    label: str,
    marker=None,
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

    plt.scatter(xs, ys, alpha=0.3, label=label, marker=marker)

plt.figure(figsize=(4, 3))
plt.xlabel('Layer index')
plt.ylabel(r'Decay ($\alpha_t$)')

runs = [
    'mamba2-130m--T2048_B32_GA1_P1_SR1_RD0_lr0.0005--ckpt_100000',
    'mamba2-130m--T4096_B16_GA1_P1_SR1_RD0_lr0.0005--ckpt_100000',
    'mamba2-130m--orig--ckpt_0',
    'mamba2-130m--T16384_B4_GA1_P1_SR1_RD0_lr0.0005--ckpt_100000',
]
labels = [
    '2K', '4K', '8K', '16K'
]

label_to_marker = {
    '2K': 's',
    '4K': 'o',
    '8K': '^',
    '16K': 'v',
    '64K': 'v',
}

runs = [
    'mamba2-370m--orig--ckpt_0',
    'mamba2-370m--T8192_B1_GA1_P8_SR16_RD0_lr0.0005--ckpt_100000',
]
labels = ['8K', '64K']
cache_dir = Path('./cache/dA')

layer_means = {label: [] for label in labels}
plt.figure(figsize=(5, 4))
for run, label in zip(runs, labels):
    cache_path = cache_dir / run / 'nextlines' / 'seq_mean.pt'
    print(cache_path)
    seq_mean = torch.load(cache_path)
    print(label, seq_mean.mean(), seq_mean.var(), seq_mean.min(), seq_mean.max())
    for layer_i in range(24):
        m = seq_mean[layer_i]
        print(label, layer_i, m.mean(), m.var(), m.min(), m.max())
        layer_means[label].append(m.mean())
    plot_seq_mean(seq_mean, label=label, marker=label_to_marker[label])
    

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.legend()
output_dir = Path('./figs/dA/370m')
output_dir.mkdir(exist_ok=True, parents=True)
dst_path = output_dir / 'seq_mean.pdf'
plt.ylabel(r"Decay ($\alpha_t$)")
plt.xlabel("Layer index")
print(f"Saving to {dst_path}")
plt.tight_layout()
plt.savefig(dst_path, dpi=300, bbox_inches='tight')
plt.clf()

dst_path = output_dir / 'seq_layer_mean.pdf'
plt.figure(figsize=(5, 2.5))
markers = ['s', 'o', '^', 'v']
for label, means in layer_means.items():
    xs = torch.arange(len(means))
    plt.scatter(xs, means, label=label, marker=markers.pop(), alpha=0.6)
plt.ylabel(r"Decay ($\alpha_t$)")
plt.xlabel("Layer index")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
print(f"Saving to {dst_path}")
plt.savefig(dst_path, bbox_inches='tight')
