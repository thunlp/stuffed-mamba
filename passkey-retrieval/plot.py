from pathlib import Path
from tap import Tap
import jsonlines
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Args(Tap):
    output_dir = './figs'
    model_name: str = '*'
    run_name: str = "*"
    ckpt_name: str = 'ckpt_*'
    task_name: str = '*'
    max_len: int = 64
    min_len: int = 1
    step: int = 1  # the length step / k
    n_rows: int = 10  # the number of grid along the y axis
    n_x_labels: int = 8  # the number of x labels
    n_y_labels: int = 2  # the number of y labels"
    figs_dir: str = 'figs'
    ylog: int = 0
    train_len: int = 8


def is_correct(data: dict[str, str]) -> bool:
    return data['ground_truth'] in data['prediction']


def load_result(preds: list[dict], y_position: list[float]):
    n_pos = len(y_position)
    pos_to_cnt = {i: 0 for i in range(n_pos)}
    pos_to_correct_cnt = {i: 0 for i in range(n_pos)}
    n_lines = 0
    for pred in preds:
        n_lines += 1
        eg_len = pred['relative_postion']
        i = 0
        while i < len(y_position) and eg_len > y_position[i]:
            i += 1
        pos_to_cnt[i] += 1
        # print(pred)
        pos_to_correct_cnt[i] += int(is_correct(pred))
        
    return pos_to_correct_cnt, pos_to_cnt, n_lines


def create_heatmap(
    results: np.ndarray,
    x_lens: list[int],
    output_path: Path,
    n_cols: int,
    n_rows: int,
    step: int,
    fontsize: int = 12,
):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(results)

    plt.figure(dpi=300, figsize=(2.2, 2.2))

    ax = sns.heatmap(
        data=results,
        # cmap=plt.get_cmap("Greens"),  # type: ignore
        cmap='RdYlGn',
        annot=False,
        fmt=".2f",
        cbar=False,
        vmin=-0.2,
        vmax=1.2,
        linewidths=0,
    )
    if args.ylog:
        xticks_chunk_size = 2
        xlabels = []
        xticks = []
        for i, x_len in enumerate(x_lens):
            if i % xticks_chunk_size == 0:
                xlabels.append(f"{x_len}K")
                xticks.append(i + 0.5)
    else:
        tick_positions = [1] + [l for l in x_lens if is_power_of_two(l) and l > 2]
        xlabels = [f"{l}K" for l in tick_positions]
        xticks = [l - 0.5 for l in tick_positions]
    
    y_ticks = list(range(0, n_rows + 1, n_rows // args.n_y_labels))
    ylabels = [f"{100 * i / n_rows:.0f}" for i in range(0, n_rows + 1, n_rows // args.n_y_labels)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(ylabels, fontsize=fontsize - 1)
    ax.tick_params(axis='y', rotation=90)
    plt.xticks(ticks=xticks, labels=xlabels, rotation=45, ha='center')
    plt.xlabel("Context Len.", fontdict={"fontsize": fontsize})

    # plt.title("Passkey", fontdict={"fontsize": fontsize})
    print(f"Saving figure to {output_path}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.ylabel("Ans. Depth (%)", fontdict={"fontsize": fontsize})
    plt.tight_layout()
    plt.savefig(output_path.parent / f'{output_path.stem}_ylabel.pdf', dpi=300, bbox_inches="tight")
    plt.clf()


def plot_acc_curve(all_result: np.ndarray, x_lens: list, dst_path: Path, step: int, max_len: int, lengths=[0.25, 0.5, 1, 2, 4, 8]):
    n_rows, n_cols = all_result.shape
    len_to_acc = {length: [] for length in lengths}
    # xs = []
    plt.figure(figsize=(2.4, 2), dpi=300, tight_layout=True)
    for col in range(n_cols):
        # cur_len = 2 ** col
        # cur_len = (col + 1) * step
        # xs.append(cur_len)
        cur_len = x_lens[col]
        for length in len_to_acc:
            last_k_prop = length / cur_len
            last_k_rows = int(last_k_prop * n_rows)
            if last_k_rows == 0 or last_k_rows > n_rows:
                len_to_acc[length].append(None)
            else:
                last_k_result = all_result[-last_k_rows:, col]
                acc = sum(last_k_result) / len(last_k_result)
                len_to_acc[length].append(acc)
        # cur_len *= 2
    print(len_to_acc)
    for length, acc in len_to_acc.items():
        plt.plot(x_lens, acc, label=f"{length}K", marker='o')
    plt.ylabel("Accuracy")
    plt.xlabel("Input Length (K)")
    # plt.ylim(0.1, 1.1)
    plt.xscale('log')
    plt.axvline(x=train_len, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dst_path.parent / f"{dst_path.stem}_legend.pdf", dpi=300, bbox_inches='tight')
    plt.clf()
    
        
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def plot_one_run(args, run_dir: Path, file_ext: str = 'pdf'):
    print(f"Plotting for {run_dir}")
    # step = args.step
    if args.ylog:
        x_lens = [length for length in range(1, args.max_len + 1) if is_power_of_two(length)]
    else:
        x_lens = [i for i in range(args.min_len, args.max_len + 1) if i % args.step == 0]
    if x_lens[-1] != args.max_len:
        x_lens += [args.max_len]

    print(f"{x_lens = }")

    id2len = {}
    for row_idx, x_len in enumerate(x_lens):
        id2len[row_idx] = x_len
    len2id = {v: k for k, v in id2len.items()}
    print(f"{id2len = }")

    n_rows = args.n_rows
    n_cols = len(x_lens)
    all_result = np.zeros((n_rows, n_cols))
    y_positions = [x / n_rows for x in range(1, n_rows+1, 1)]

    for this_len in x_lens:
        if (run_dir / f"preds_{this_len}k.jsonl").exists():
            preds_path = run_dir / f"preds_{this_len}k.jsonl"
        else:
            preds_path = run_dir / f"preds_{this_len * 1024}.jsonl"
        print(preds_path)
        if not preds_path.exists():
            continue

        # max_col = max(max_col, col)
        # eg_lens = get_lens(data_path)
        print(f"loading data from: {preds_path}")
        preds = list(jsonlines.open(preds_path, 'r'))
        pos_to_correct_cnt, pos_to_cnt, n_preds = load_result(preds, y_positions)

        col_idx = len2id[this_len]
        for row_idx, n_examples in pos_to_cnt.items():
            if n_examples > 0:
                acc = pos_to_correct_cnt[row_idx] / n_examples
                # print(f"{row_idx = }, {col_idx = }, {acc = }, {n_examples = }, {pos_to_correct_cnt[row_idx] = }")
                all_result[row_idx][col_idx] = acc


    acc_curve_lens = [l for l in range(max(1, args.max_len // 16), args.max_len // 2 + 1) if is_power_of_two(l)]

    create_heatmap(
        results=all_result,
        x_lens=x_lens,
        output_path=run_dir / f'plot.{file_ext}',
        n_cols=n_cols,
        n_rows=n_rows,
        step=args.step,
    )

    plot_acc_curve(
        all_result,
        x_lens=x_lens,
        dst_path=run_dir / f"acc_curve.{file_ext}",
        step=args.step,
        max_len=args.max_len,
        lengths=acc_curve_lens,
    )


def main(args: Args):
    print(args)
    global train_len
    train_len = args.train_len
    output_dir = Path(args.output_dir)
    model_size_dirs = sorted(output_dir.glob(args.model_name))
    for model_size_dir in model_size_dirs:
        print(model_size_dir)
        if not model_size_dir.is_dir():
            continue
        for run_dir in sorted(model_size_dir.glob(args.run_name)):
            if not run_dir.is_dir():
                continue
            print(f"run_dir: {run_dir}")
            for ckpt_dir in sorted(run_dir.glob(args.ckpt_name)):
                if not ckpt_dir.is_dir():
                    continue
                print(f"ckpt_dir: {ckpt_dir}")
                for task_dir in sorted(ckpt_dir.glob(args.task_name)):
                    if not task_dir.is_dir():
                        continue
                    
                    if len(list(task_dir.glob('preds_*.jsonl'))) == 0:
                        continue

                    # if not ckpt_dir.is_dir() or not (ckpt_dir / f'preds_passkey_{args.max_length}k.jsonl').exists():
                    #     continue
                    print("################################")
                    print(f"Plotting for {task_dir}")
                    print("################################")
                    plot_one_run(args, task_dir)


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
