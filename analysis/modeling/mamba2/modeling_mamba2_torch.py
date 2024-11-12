"""
A minimal, single-file implementation of the Mamba-2 model in PyTorch.
"""

import json
from dataclasses import dataclass
from typing import NamedTuple, cast, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn

Device = Union[str, torch.device, None]


@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    prefill_chunk_size: int = 512  # prefill context is split and processed in chunks
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None, dtype=torch.bfloat16):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device, dtype=dtype,
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device, dtype=dtype,
            ),
        )
        
    def __getitem__(self, idx):
        if idx == 0:
            return self.conv_state
        if idx == 1:
            return self.ssm_state
        raise IndexError


class Mamba2ForCausalLM(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None, dtype=torch.bfloat16):
        super().__init__()
        self.config = args
        self.args = args
        self.device = device
        self.dtype = dtype

        layers = [
            nn.ModuleDict(
                dict(
                    mixer=Mamba2Layer(args, layer_idx=layer_i, device=device, dtype=dtype),
                    norm=RMSNorm(args.d_model, device=device),
                )
            )
            for layer_i in range(args.n_layer)
        ]
        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(
                    args.vocab_size, args.d_model, device=device, dtype=dtype),
                layers=nn.ModuleList(layers),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device, dtype=dtype,
        )
        self.lm_head.weight = self.backbone.embedding.weight

    def set_dt_mult(self, dt_mult: float):
        for layer in self.backbone.layers:
            assert isinstance(layer.mixer, Mamba2Layer)
            layer.mixer.set_dt_mult(dt_mult)

    def set_da_mult(self, da_mult: float):
        for layer in self.backbone.layers:
            assert isinstance(layer.mixer, Mamba2Layer)
            layer.mixer.set_da_mult(da_mult)

    def set_b_mult(self, b_mult: float):
        for layer in self.backbone.layers:
            assert isinstance(layer.mixer, Mamba2Layer)
            layer.mixer.set_b_mult(b_mult)

    def set_a_mult(self, a_mult: float):
        for layer in self.backbone.layers:
            assert isinstance(layer.mixer, Mamba2Layer)
            layer.mixer.set_a_mult(a_mult)

    @staticmethod
    def from_pretrained(huggingface_model_id: str, device: Device = None):
        '''
        NOTE: Device must be passed here, or you have to set `self.device` manually
        from outside, which is ugly.
        '''
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        config_path = cached_file(huggingface_model_id, CONFIG_NAME)
        assert config_path, "Failed to get huggingface config file"
        state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME)
        assert state_dict_path, "Failed to get huggingface state dict file"

        config = json.load(open(config_path))
        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config["pad_vocab_size_multiple"],
        )

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, weights_only=False, map_location=map_location, mmap=True
        )
        model = Mamba2ForCausalLM(args, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(
        self,
        input_ids: Tensor,
        states: Union[list[InferenceCache], list[None], None] = None,
        recurrent_mode: bool = False,
        return_decays: bool = False,
    ) -> tuple[LongTensor, list[InferenceCache]]:
        """
        Arguments
            input_ids: (batch, seqlen) tokens from `EleutherAI/gpt-neox-20b` tokenizer
            h: hidden states for inference step. If present the constant-time
               (wrt sequence length) inference path will be taken, input_ids
               should have shape (batch, 1) containing the next batch of prompt
               token.

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated inference cache after processing `input_ids`
        """
        seqlen = input_ids.shape[1]

        if states is None:
            states = [None for _ in range(self.args.n_layer)]

        x = self.backbone.embedding(input_ids)
        all_decays = []
        for i, layer in enumerate(self.backbone.layers):
            layer_outputs = layer.mixer(
                layer.norm(x),
                states=states[i],
                recurrent_mode=recurrent_mode,
                return_decays=return_decays,
            )
            y = layer_outputs['output']
            states[i] = layer_outputs['states']
            x = y + x

            if return_decays:
                layer_decays = {
                    key: layer_outputs['decays'][key] for key in ['dt', 'A', 'B', 'C', 'x']
                }
                all_decays.append(layer_decays)

        x = self.backbone.norm_f(x)
        logits = self.lm_head(x)
        return {
            'logits': logits[:, :seqlen],
            'states': cast(list[InferenceCache], states),
            'decays': all_decays,
        }

    def prefill(
        self,
        input_ids: Tensor,
        states: Union[list[InferenceCache], list[None], None] = None,
    ) -> list[InferenceCache]:
        '''
        Same as forward, but does chunking on excessively long inputs. Also,
        does not return logits.
        
        input_ids: (batch_size, seqlen)
        '''
        seqlen = input_ids.shape[1]

        if seqlen > self.args.prefill_chunk_size:
            # When the context is too long, we need to split them into chunks, or
            # PyTorch will complain about it.
            for chunk_lo in range(0, seqlen, self.args.prefill_chunk_size):
                chunk_hi = chunk_lo + self.args.prefill_chunk_size
                # print(f"Prefilling chunk: [{chunk_lo} : {chunk_hi}]")
                chunk_ids = input_ids[:, chunk_lo : chunk_hi]
                _, states = self.forward(
                    chunk_ids,
                    states=states,
                    recurrent_mode=False,
                )
            return cast(list[InferenceCache], states)

        if states is None:
            states = [None for _ in range(self.args.n_layer)]

        x = self.backbone.embedding(input_ids)
        for i, layer in enumerate(self.backbone.layers):
            y, states[i] = layer.mixer(layer.norm(x), states[i])
            x = y + x

        return {
            'states': cast(list[InferenceCache], states),
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Tensor:
        """
        input_ids: (batch_size, seqlen)
        """
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)  # (B, T)

        # print(f"[Mamba2ForCausalLM.generate] {input_ids.shape = }")
        context, last_token = input_ids[:, :-1], input_ids[:, -1:]  # (B, T-1), (B, 1)
        batch_size, seqlen = context.shape

        # Process prompt
        # The input sequence to forward (non-inference path) must have length multiple that of chunk_size.
        # We split out excess tokens so that n_chunked tokens can be processed by one forward call and
        # process the rest in multiple inference steps.
        mul64_seqlen = (seqlen // self.args.chunk_size) * self.args.chunk_size
        if mul64_seqlen > 0:
            states = self.prefill(context[:, :mul64_seqlen], None)
        else:
            states = [
                InferenceCache.alloc(batch_size, self.args, device=self.device, dtype=self.dtype)
                for _ in range(self.args.n_layer)
            ]
        for i in range(mul64_seqlen, seqlen):
            _, states = self(context[:, i : i + 1], states=states, recurrent_mode=True)

        # Generate
        outputs = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out, states = self(last_token, states, recurrent_mode=True)  # (B, 1, V), ...
            logits = out[:, -1]  # (B, V)
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]  # (B, V)
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                # TODO: check if this is correct
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True
                )  # (B, V), (B, V)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > 0.5  # (B)
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                    :, :-1
                ].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)  # (B, V)
            next_token = torch.multinomial(probs, num_samples=1)  # (B)
            if torch.all(next_token == eos_token_id):
                break
            last_token = next_token
            # yield cast(int, next_token.item()), states
            outputs.append(last_token.squeeze(1))  # (N, B)

        outputs = torch.stack(outputs)  # (N, B)
        outputs = outputs.transpose(1, 0)
        return outputs


class Mamba2Layer(nn.Module):
    def __init__(
        self,
        args: Mamba2Config,
        layer_idx: int,
        device: Device = None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = dtype
        self.layer_idx = layer_idx

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device, dtype=dtype)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device, dtype=dtype))
        self.D = nn.Parameter(torch.empty(args.nheads, device=device, dtype=dtype))
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(
            args.d_inner, args.d_model, bias=False, device=device, dtype=dtype)

        self.dt_mult = None
        self.da_mult = None
        self.b_mult = None
        self.a_mult = None

    def set_dt_mult(self, dt_mult: float):
        self.dt_mult = dt_mult

    def set_da_mult(self, da_mult: float):
        self.da_mult = da_mult

    def set_b_mult(self, b_mult: float):
        self.b_mult = b_mult

    def set_a_mult(self, a_mult: float):
        self.a_mult = a_mult

    def forward(
        self,
        u: Tensor,
        states: Union[InferenceCache, None] = None,
        recurrent_mode: bool = False,
        return_decays: bool = False,
    ) -> dict:
        """
        Parallel forward for fast training and prefilling.

        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        if states is None:
            states = InferenceCache.alloc(
                u.shape[0],
                self.args,
                device=self.device,
                dtype=self.dtype,
            )

        if recurrent_mode:
            return self.forward_recurrent(
                u,
                states,
                return_decays=return_decays,
            )

        A = -torch.exp(self.A_log.to(torch.float32))  # (nheads,)

        if self.a_mult is not None:
            A = A * self.a_mult

        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias).to(torch.float32)  # (batch, seqlen, nheads)

        if self.dt_mult is not None:
            dt = dt * self.dt_mult

        da_log = A * dt
        
        if self.da_mult is not None:
            da_log = da_log + torch.log(torch.tensor(self.da_mult, device=self.device))

        # Pad or truncate xBC seqlen to d_conv
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )
        xBC = self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        xBC = silu(xBC)  # (batch, seqlen, d_inner + 2 * d_state))
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )

        if self.b_mult is not None:
            B = B * self.b_mult

        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            da_log,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            initial_states=states[1].unsqueeze(1),
            device=self.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = y.to(u.dtype)
        y = self.norm(y, z)
        y = self.out_proj(y)

        states = InferenceCache(conv_state, ssm_state)
        outputs = {
            'output': y,
            'states': states,
        }
        
        # if self.layer_idx == 20:
        #     print('dt', dt[0, :, 11])
        #     print('A', A)
        #     print('B', B[0])
        #     print('x', x[0, :, 11, 10])
        
        if return_decays:
            outputs['decays'] = {
                'dt': dt,  # (B, T, H)
                'A': A,  # (H)
                'B': B,  # (B, T, N)
                'C': C,  # (B, T, N)
                'x': x,  # (B, T, H, P)
            }
        return outputs

    def forward_recurrent(
        self,
        u: Tensor,
        states: InferenceCache,
        return_decays: bool = False,
    ) -> tuple[Tensor, InferenceCache]:
        """
        Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        A = -torch.exp(self.A_log)  # (nheads,)

        if self.a_mult is not None:
            A = A * self.a_mult

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias).to(torch.float32)  # (batch, nheads)

        if self.dt_mult is not None:
            dt = dt * self.dt_mult

        # Advance convolution input
        states[0].copy_(torch.roll(states[0], shifts=-1, dims=-1))
        states[0][:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            states[0] * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )

        if self.b_mult is not None:
            B = B * self.b_mult

        # SSM step
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        states[1].copy_(states[1] * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", states[1], C.to(dtype=states[1].dtype))
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        outputs = {
            'output': y.unsqueeze(1),
            'states': states,
        }
        if return_decays:
            outputs['decays'] = {
                'dt': dt,  # (B, T, H)
                'A': A,  # (H)
                'B': B,  # (B, T, N)
                'C': C,  # (B, T, N)
                'x': x,  # (B, T, H, P)
            }
        return outputs


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None, dtype=torch.bfloat16):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    C = C.to(dtype=states.dtype)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Define this manually since torch's version doesn't seem to work on MPS."""
    return x * F.sigmoid(x)
