from pathlib import Path
import time

import jsonlines
from torch import Tensor
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from safetensors.torch import load_file

from arguments import parse_args
from rwkv5 import Rwkv5Config, Rwkv5ForCausalLM
from utils import get_non_embed_param_count, get_param_count, dump_jsonl

from passkey import PasskeyDataGenerator


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def get_pred(
    model: Rwkv5ForCausalLM,
    tok: PreTrainedTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
    use_bos: bool = False,
    add_prefix: bool = False,
    device: str = 'cuda:0',
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    # print("Truncating...")
    # input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if add_prefix:
        input_text = (
            "The grass is green. The sky is blue. The sun is yellow."
            " Here we go. There and back again.\n" * 1200
        ) + input_text
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:100])
        print("...")
        print(input_text[-100:])
        print("=====================================")
    if use_bos:
        input_ids = tok("<|endoftext|>" + input_text, return_tensors="pt")["input_ids"].to(device)
    else:
        input_ids = tok(input_text, return_tensors="pt")["input_ids"].to(device)

    # input_ids = [tok.bos_token] + tok.encode(input_text)
    # input_ids = torch.Tensor(input_ids, dtype=torch.int64).cuda()

    out: Tensor = model.generate(input_ids, max_new_tokens=max_tokens)
    output = tok.batch_decode(out)[0][len(input_text):]
    
    print("Chunked generation:", output)
    return output


def convert_state_dict(sd):
    new_sd = {}
    for k in sd:
        old_k = k
        if k != 'head.weight':
            k = 'rwkv.' + k
        replacements = {
            'ln0': 'pre_ln',
            'emb.': 'embeddings.',
            'att.': 'attention.',
            'ffn.': 'feed_forward.',
            '.time_mix_k': '.time_mix_key',
            '.time_mix_v': '.time_mix_value',
            '.time_mix_r': '.time_mix_receptance',
            '.time_mix_g': '.time_mix_gate',
        }
        for a, b in replacements.items():
            k = k.replace(a, b)
        new_sd[k] = sd[old_k]
    return new_sd


def get_config(model_path: str):
    if '-7B' in model_path:
        config = Rwkv5Config(hidden_size=4096, num_hidden_layers=32)
    if '-3B' in model_path:
        config = Rwkv5Config(hidden_size=2560, num_hidden_layers=32)
    elif '-1B5' in model_path:
        config = Rwkv5Config(hidden_size=2048, num_hidden_layers=24)
    elif '-0.4B' in model_path:
        config = Rwkv5Config(hidden_size=1024, num_hidden_layers=24)
    else:
        raise ValueError
    return config


def load_model(
    model_path: str,
    tok_path: str,
):
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        tok_path, trust_remote_code=True)  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token
    print("Instantiating model")
    start_time = time.time()
    print(f"Getting config for {model_path}...")
    config = get_config(Path(model_path).name)
    print(config)
    print(f"Instantiating model from config...")
    model = Rwkv5ForCausalLM(config)
    print(f"Loading checkpoint from {model_path}")
    if model_path.endswith('.safetensors'):
        ckpt = convert_state_dict(load_file(model_path))
    else:
        print(f"Loading PyTorch state dict from {model_path}")
        ckpt = torch.load(model_path)
        print("Converting state dict keys...")
        ckpt = convert_state_dict(ckpt)
    model.load_state_dict(ckpt)
    # model = model.cuda()
    print("Time taken:", round(time.time() - start_time))
    return model, tokenizer  # type: ignore


def evaluate(
    args,
    model,
    tok,
    n_samples: int,
    data_len: int = 1024,
    max_new_tokens: int = 5,
    device: str = 'cuda',
):
    data_generator = PasskeyDataGenerator(tok)
    result_dir = Path(args.output_dir) / 'passkey'
    result_dir.mkdir(exist_ok=True, parents=True)

    output_path = (
        result_dir / f"preds_{data_len}.jsonl"
    )
    preds = []
    if output_path.exists():
        with jsonlines.open(output_path, "r") as fin:
            for line in fin:
                preds.append(line)

        # Continue from existing output
        start_idx = len(preds)
        stop_idx = n_samples

    print("==== Evaluation ====")
    print(f"# examples: {n_samples}")
    print(f"Start index: {start_idx}")
    print(f"Stop index: {stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_new_tokens}")
    for idx in range(start_idx, stop_idx):
        cur_depth = (idx + 0.5) / stop_idx
        prompt, ans = data_generator.get_one(
            context_len=data_len,
            answer_pos=cur_depth,
        )
        print(f"====== Example {idx} ======")
        pred = get_pred(
            model=model,
            tok=tok,
            input_text=prompt,
            max_tokens=max_new_tokens,
            verbose=bool(args.verbose),
            use_bos=bool(args.use_bos),
            add_prefix=bool(args.add_prefix),
            device=f'cuda:{args.gpu_id}',
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": str(idx),
                "prediction": pred,
                "ground_truth": ans,
                "relative_postion": cur_depth,
            }
        )
        print(f"Dumping to {output_path}")
        dump_jsonl(preds, output_path)


def main():
    args = parse_args()
    print("==========")
    print(args)
    print("==========")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(output_dir / 'args.json'))

    model, tok = load_model(model_path=args.model_path, tok_path=args.tok_path)
    device = f"cuda:{args.gpu_id}"
    print(f"Moving model to device: {device}")
    model = model.to(device=device)  # type: ignore

    print("=========================================================")
    print(f"Param count: {get_param_count(model)}")
    print(f"Non-embedding param count: {get_non_embed_param_count(model)}")
    print("=========================================================")

    all_lens = [length for length in range(args.min_len, args.max_len + 1)]
    for len_id, length in enumerate(all_lens):
        if args.multi_gpu:
            if len_id % args.n_gpus == args.gpu_id:
                print(f"Evaluating on: {length}")
                evaluate(
                    args,
                    model=model,
                    tok=tok,
                    data_len=length * 1024,
                    n_samples=args.n_samples,
                    device=device,
                )
        else:
            print(f"Evaluating on: {length}")
            evaluate(
                args,
                model=model,
                tok=tok,
                data_len=length * 1024,
                n_samples=args.n_samples,
                device=device,
            )
if __name__ == "__main__":
    main()
