from pathlib import Path
import time
from typing import Tuple
import fla
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import jsonlines
import torch

from passkey import PasskeyDataGenerator
from arguments import parse_args
from utils import (
    get_non_embed_param_count,
    get_param_count,
    dump_jsonl,
)


@torch.no_grad()
def get_pred(
    model: AutoModelForCausalLM,
    tok: PreTrainedTokenizer,
    input_text: str,
    max_new_tokens: int,
    verbose: bool = False,
    use_bos: bool = False,
    add_prefix: bool = False,
    device: str = "cuda",
) -> str:
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
        inputs = tok("<|endoftext|>" + input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
    else:
        inputs = tok(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    output = tok.batch_decode(out)[0][len(input_text) :]
    return output


def load_model(
    model_path: str = "/home/test/test07/chenyingfa/ckpts/hgrn2/hgrn2-1.3b",
    tok_path: str = "/home/test/test07/chenyingfa/ckpts/hgrn2/hgrn2-1.3b",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading model from: {model_path}")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model loading time: {time.time() - start_time:.1f} sec")
    return model, tokenizer  # type: ignore


def evaluate(
    args,
    model,
    tok,
    n_samples: int,
    data_len: int = 1024,
    max_new_tokens: int = 5,
    device: str = "cuda",
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
            max_new_tokens=max_new_tokens,
            verbose=bool(args.verbose),
            use_bos=bool(args.use_bos),
            add_prefix=bool(args.add_prefix),
            device=device,
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "idx": str(idx),
                "prediction": pred,
                "ground_truth": ans,
                "relative_postion": cur_depth,
            }
        )
        dump_jsonl(preds, output_path)


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def main():
    args = parse_args()
    print("==== args ====")
    print(args)
    print("==============")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(output_dir / "args.json"))

    model, tok = load_model(model_path=args.model_path, tok_path=args.tok_path)
    # model, tok = load_model()
    device = args.device
    print(f"Moving model to device: {device}")
    model = model.to(device=device)  # type: ignore

    print("=========================================================")
    print(f"Param count: {get_param_count(model)}")
    print(f"Non-embedding param count: {get_non_embed_param_count(model)}")
    print("=========================================================")

    all_lens = [length for length in range(args.min_len, args.max_len + 1)]
    # all_lens = [length for length in range(args.min_len, args.max_len + 1) if is_power_of_two(length)]
    for len_id, length in enumerate(all_lens):
        if args.multi_gpu:
            if len_id % args.n_gpus == args.gpu_id:
                print(f"Evaluating on: {length}")
                device = f"cuda:{args.gpu_id}"
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
            device = args.device
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
