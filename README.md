# Stuffed Mamba: State Collapse and State Capacity of RNN-Based Long-Context Modeling

This is the official code for the paper [Stuffed Mamba: State Collapse and State Capacity of RNN-Based Long-Context Modeling](https://arxiv.org/abs/2410.07145). 

## Introduction

We examine two practical concerns when applying state-of-the-art RNNs to long contexts:

1. The inability to extrapolate to inputs longer than the training length.
2. The upper bound of memory capacity.

Addressing the first concern, we first investigate state collapse (SC), a phenomenon that causes severe performance degradation on sequence lengths not encountered during training. With controlled experiments, we attribute this to overfitting due to the recurrent state being overparameterized for the training length. For the second concern, we train a series of Mamba-2 models on long documents to empirically estimate the recurrent state capacity in language modeling and passkey retrieval.

![](readme-images/fig1.png)

![](readme-images/fig12.png)

## Code Overview

- The `analysis` directory contains the source code for the analysis experiments (Section 4 in the paper). It also contains the code for computing perplexity as a function of context length.
- The `passkey-retrieval` directory contains the source code for evaluating Mamba-2, RWKV-5, and RWKV-6 in passkey retrieval.

For how to run the code to reproduce the results in the paper, checkout the README file inside the above directories.

## Setup

The code is tested with:

- PyTorch 2.3
- Python 3.12
- Transformers 4.45.2

Other required libraries are listed in `requirements.txt`.

For running HGRN-2, you also need to install FlashLinearAttention from: https://github.com/sustcsonglin/flash-linear-attention.

## TODOs

- The checkpoints of Mamba-2 trained on longer sequences will be released soon.
- Add code for the experiments with Mamba-1, HGRN-2.
- Add code for training Mamba-2.

## Acknowledgements

A large part of the code is copied from the following sources:

- <https://www.github.com/OpenBMB/InfiniteBench>: For passkey evaluation.
- <https://github.com/state-spaces/mamba>: For the Mamba-2 implementation.
- The HuggingFace Transformers library: For RWKV implementation.

## Citation

To cite us, please use the following BibTeX.

```bibtex
@article{chen2024stuffed-mamba,
    title={Stuffed Mamba: State Collapse and State Capacity of RNN-Based Long-Context Modeling},
    author={Yingfa Chen and Xinrong Zhang and Shengding Hu and Xu Han and Zhiyuan Liu and Maosong Sun},
    year={2024},
    eprint={2410.07145},
    url={https://arxiv.org/abs/2410.07145}, 
}
```
