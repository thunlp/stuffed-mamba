from tap import Tap


class Arg(Tap):
    output_dir: str = "./output"
    "Where to dump the prediction results."

    model_path: str = '/path/to/model'
    "The path of the model (in HuggingFace (HF) style). If specified, it"
    " will try to load the model from the specified path, else, it wll default to the official HF path."

    tok_path: str = '/path/to/tokenizer'

    model_name: str = "mamba2"
    "For `compute_scores.py` only, specify which model you want to compute the score for."

    device: str = "cuda"
    dtype: str = "bf16"
    verbose: int = 0
    use_bos: int = 0
    add_prefix: int = 0

    gpu_id: int = 0
    n_gpus: int = 8
    min_len: int = 1
    max_len: int = 128
    width: int = 4
    n_depths: int = 10
    multi_gpu: int = 0
    n_samples: int = 200


def parse_args() -> Arg:
    p = Arg()
    args = p.parse_args()
    assert args.model_name in [
        "rwkv",
        "mamba",
        "rwkv5",
        "mamba2",
        "hgrn2",  # TODO: support this.
    ]
    return args
