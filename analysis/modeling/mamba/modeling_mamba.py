from typing import Tuple, List, Optional, Union
import json

import torch
import torch.nn as nn

# from torch.utils.checkpoint import checkpoint
from .checkpoint import checkpoint
from torch import Tensor, LongTensor
import torch.nn.functional as F

from .mamba_torch import MambaLayer, RMSNorm
from .configuration_mamba import MambaConfig

"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""


class MambaState:
    def __init__(self, ssm_state: Tensor, conv_state: Tensor):
        self.ssm_state = ssm_state
        self.conv_state = conv_state

    @staticmethod
    def create(L: int, B: int, D: int, E: int, N: int, d_conv: int, device, dtype):
        result = MambaState.empty(L, B, D, E, N, d_conv, device, dtype)
        result.ssm_state[:] = 0
        result.conv_state[:] = 0
        return result

    @staticmethod
    def empty(L: int, B: int, D: int, E: int, N: int, d_conv: int, device, dtype):
        ssm_state = torch.empty(L, B, E * D, N, device=device, dtype=dtype)
        conv_state = torch.empty(L, B, E * D, d_conv - 1, device=device, dtype=dtype)
        return MambaState(ssm_state, conv_state)

    def __getitem__(self, layer: int):
        return (self.ssm_state[layer], self.conv_state[layer])

    def __setitem__(self, layer: int, val):
        self.ssm_state[layer] = val[0]
        self.conv_state[layer] = val[1]

    def detach(self):
        self.ssm_state.detach()
        self.conv_state.detach()


class MambaModel(nn.Module):
    def __init__(self, config: MambaConfig, chunk_size: int = 1024, device="cuda"):
        """
        Args:
        - chunk_size (int): The input is split into chunks, and fed sequentially to the model with
            activation checkpointing. This is for lowering the memory usage at the cost of
            training speed.
        """
        super().__init__()

        self.config = config
        self.device = device
        self.chunk_size = chunk_size

        self.embeddings = nn.Embedding(
            config.vocab_size, config.d_model, dtype=config.get_dtype()
        )
        layers = [MambaLayer(config, i) for i in range(config.n_layers)]
        self.layers = nn.ModuleList(layers)
        self.norm_f = RMSNorm(config.d_model)

    def forward(
        self,
        input_ids: LongTensor,
        states: Optional[MambaState] = None,
        act_checkpointing: str = "",
    ):
        """
        input_ids (Tensor): (B T D)
        """
        print("[MambaModel.forward] input_ids")
        print(input_ids)
        x = self.embeddings(input_ids)
        print("[MambaModel.forward] input embeddings")
        print(x)
        if states is None:
            states = MambaState.create(
                self.config.n_layers,
                x.size(0),
                self.config.d_model,
                self.config.expand_factor,
                self.config.d_state,
                self.config.d_conv,
                x.device,
                x.dtype,
            )

        new_states = MambaState.empty(
            self.config.n_layers,
            x.size(0),
            self.config.d_model,
            self.config.expand_factor,
            self.config.d_state,
            self.config.d_conv,
            x.device,
            x.dtype,
        )
        for layer_idx, layer in enumerate(self.layers):
            layer_state = states[layer_idx]
            if act_checkpointing == "chunk":
                # Split x along the T dimension in chunks of size chunk_size, then apply activation checkpointing on each chunk.
                # This is done to save memory, as the activation checkpointing will save the activations of the layers
                # and only recomputes them when needed.
                x_chunks = x.chunk(x.size(1) // self.chunk_size, dim=1)
                y_chunks = []
                chunk_state = states[layer_idx]
                for chunk_idx, x_chunk in enumerate(x_chunks):
                    y_chunk, chunk_state = checkpoint(layer, x_chunk, chunk_state, chunk_idx, use_reentrant=True)  # type: ignore
                    # mem_used = torch.cuda.memory_allocated() / 2 ** 20
                    # print(f"[MambaModel.forward] {layer_idx = } | {chunk_idx = } | {mem_used = }")
                    y_chunks.append(y_chunk)
                x = torch.cat(y_chunks, dim=1)
                # mem_used = torch.cuda.memory_allocated() / 2 ** 20
                # print(f"[MambaModel.forward] {layer_idx = } | {chunk_idx = } | {mem_used = }")
                # exit()
            elif act_checkpointing == "layer":
                x, new_state = checkpoint(layer, x, layer_state, use_reentrant=True)  # type: ignore
            else:
                x, new_state = layer(x, layer_state)

            new_states[layer_idx] = new_state

        exit()
        x = self.norm_f(x)
        return x, new_states

    def step(self, input_ids: LongTensor, states: Optional[MambaState] = None):
        """
        Args:
            x: (B, L, D)
            caches: [cache(layer) for all layers], cache : (h, inputs)

        Returns:
            y: (B, L, D)
            caches: [cache(layer) for all layers], cache : (h, inputs)
        """
        x = self.embeddings(input_ids)
        if states is None:
            states = MambaState.create(
                self.config.n_layers,
                x.size(0),
                self.config.d_model,
                self.config.expand_factor,
                self.config.d_state,
                self.config.d_conv,
                x.device,
                x.dtype,
            )

        for i, layer in enumerate(self.layers):
            layer_state = states[i]
            x, new_state = layer.step(x, layer_state)
            states[i] = layer_state
        x = self.norm_f(x)
        return x, states


class MambaForCausalLM(nn.Module):
    def __init__(
        self, config: MambaConfig, device="cuda", chunk_size: int = 128, **kwargs
    ):
        super().__init__()
        self.config = config

        self.model = MambaModel(
            self.config, device=device, chunk_size=chunk_size, **kwargs
        )
        self.lm_head = nn.Linear(
            self.config.d_model,
            self.config.vocab_size,
            bias=False,
            dtype=config.get_dtype(),
        )
        self.tie_lm_head()

    def tie_lm_head(self):
        self.lm_head.weight = self.model.embeddings.weight

    def forward(
        self,
        input_ids: LongTensor,
        labels: Optional[Tensor] = None,
        states: Optional[MambaState] = None,
        act_checkpointing: str = "",
    ) -> Tuple[Union[Tensor, None], Tensor, MambaState]:
        """
        Args:
            input_ids: (B, L)
            logits: (B, L, vocab_size)
        Returns (loss, logits, new_cache)
        """
        x, new_states = self.model(
            input_ids=input_ids,
            states=states,
            act_checkpointing=act_checkpointing,
        )
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(logits.device)
            labels = torch.cat(
                (labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)),
                1,
            )
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return loss, logits, new_states

    def step(self, input_ids: LongTensor, states: Optional[MambaState] = None):
        """
        Args:
            token: (B)
            caches: [cache(layer) for all layers], cache : (h, inputs)
        Returns:
            logits: (B, vocab_size)
            caches: [cache(layer) for all layers], cache : (h, inputs)
        """
        x, new_cache = self.model.step(input_ids, states)
        logits = self.lm_head(x)
        return logits, new_cache

    # TODO: Use parallel scan for pre-filling.
    def generate(
        self,
        tokenizer,
        input_ids: Optional[Tensor] = None,  # type: ignore
        prompt: Optional[str] = None,
        num_tokens: int = 50,
        batch_size: int = 1,
        sample: bool = True,
        top_k: int = 40,
        temperature: float = 1.0,
    ):
        self.eval()

        if prompt is not None:
            assert input_ids is None
            input_ids: Tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(
                next(self.parameters()).device
            )  # (1, num_tokens)
            input_ids = input_ids.repeat(batch_size, 1)
        else:
            assert input_ids is not None

        # caches is a list of cache, one per layer
        # cache is composed of: the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
        caches = [
            (
                None,
                torch.zeros(
                    batch_size,
                    self.config.d_inner,
                    self.config.d_conv - 1,
                    device=input_ids.device,
                    dtype=self.config.get_dtype(),
                ),
            )
            for _ in range(self.config.n_layers)
        ]

        for i in range(input_ids.size(1) + num_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                # (batch_size, vocab_size), caches
                next_token_logits, caches = self.step(input_ids[:, i], caches)  # type: ignore

            # sample (no sampling when the prompt is being processed)
            if i + 1 >= input_ids.size(1):
                probs = F.softmax(
                    next_token_logits / temperature, dim=-1
                )  # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(
                        probs, k=top_k
                    )  # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)  # type: ignore

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(
                        1
                    )  # (batch_size)
                else:
                    next_token = torch.argmax(probs, dim=-1)  # (batch_size)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        outputs = [tokenizer.decode(output.tolist()) for output in input_ids]

        self.train()

        if batch_size == 1:
            return outputs[0]
        else:
            return outputs


# adapted from https://github.com/johnma2006/mamba-minimal
def from_pretrained(name: str):
    """
    Returns a model loaded with pretrained weights pulled from HuggingFace.

    Note:
    This only work with the state-spaces/mamba-XXX model family, because
    there is a pytorch_model.bin file in the HF repo.
    This is not the case of typical model saved on HF (like the state-spaces/mamba-XXX-hf model family).
    To load the state dict of such models, I think the only way is to load
    the model into a AutoModelForCausalLM, and then
    pass the state_dict to a MambaLM. I see no other way around
    unfortunately (this is how it's done in jamba.py)

    Args:
        name: As of now, supports
            * 'state-spaces/mamba-2.8b-slimpj'
            * 'state-spaces/mamba-2.8b'
            * 'state-spaces/mamba-1.4b'
            * 'state-spaces/mamba-790m'
            * 'state-spaces/mamba-370m'
            * 'state-spaces/mamba-130m'

    Returns:
        model: a Mamba model configured with the proper parameters and initialized with the proper weights
    """

    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    def load_config_hf(model_name):
        resolved_archive_file = cached_file(
            model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
        )
        return json.load(open(resolved_archive_file))  # type: ignore

    def load_state_dict_hf(model_name):
        resolved_archive_file = cached_file(
            model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
        )
        return torch.load(
            resolved_archive_file, weights_only=True, map_location="cpu", mmap=True  # type: ignore
        )

    # copy config data
    config_data = load_config_hf(name)
    config = MambaConfig(
        d_model=config_data["d_model"],
        n_layers=config_data["n_layer"],
        vocab_size=config_data["vocab_size"],
    )

    model = MambaForCausalLM(config)

    # copy weights
    state_dict = load_state_dict_hf(name)

    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("backbone.", "model.")
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)

    return model
