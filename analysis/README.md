# Analysis of Hidden Activations in Mamba, RWKV etc.

For computing the per-token perplexity and analyzing Mamba-2 hidden representation for better understanding state collapse.

## Data Preparation

For experiments on synthetic prompts (such as the "newlines" prompt in the paper), no data preparation is needed, the function for generating the prompt is implemented in `get_prompt.py`. For instance, to reproduce the experiments with the "newlines" prompt, simple pass in `--prompt_name newlines`. It is also very easy to add new synthetic prompts by modifying the `get_long_prompt` function in `get_prompt.py`.

For experiments using text from pretraining corpus, you have to first download the data, then, you need to add the logic for loading the data in `get_prompt.py`. We have provided the implementation for loading RedPajama-16K (a version of RedPajama when documents shorter than 16K are discarded), SlimPajama, and Book3.

Finally, when running the experiments as described in the following section, you need to specify the data_path using `--data_path`.

## Model Preparation

## Perplexity as a Function of Time

Run:

```shell
python eval_per_token.py --prompt_name newlines
```

## Analyzing the Activations.

This directory contains multiple scripts, each corresponding to the analysis of a certain component/part of the model.

- `analyze_states.py`: For getting and plotting the mean and variance of Mamba-2 as a function of time.
- `get_dA.py`
- `get_dt.py`
- `get_x.py`
- `get_b.py`

while specifying the tokenizer and model checkpoint paths with `--tok_path` and `--model_path`. More command-line options are specified in `arguments.py`.

These code will also cache the intermediate result to `./cache`, and generate the final plot to `./figs`.
