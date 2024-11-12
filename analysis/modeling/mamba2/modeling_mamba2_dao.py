# Copyright (c) 2024, Tri Dao, Albert Gu.
import json
import os
import math
from typing import Optional, List, Dict
from functools import partial
import copy
import time
from collections import namedtuple
from dataclasses import dataclass, field, asdict

import torch
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat

causal_conv1d_fn, causal_conv1d_update = None, None
selective_state_update = None

from .ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from .utils.generation import GenerationMixin, InferenceParams
from .utils.hf import load_config_hf, load_state_dict_hf
from .ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from .ops.triton.ssd_combined import mamba_chunk_scan_combined
from .ops.triton.ssd_combined import mamba_chunk_scan
from .ssd_minimal import ssd_minimal_discrete
from .ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn


@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = False
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


class MambaState:
    '''
    self.state: {layer_index: (conv_state, ssm_state)}
    ssm_state: (B, nheads, P, N)
    conv_state: (B, ...)
    '''
    def __init__(self, state: dict):
        self.state = state

    def __getitem__(self, layer: int):
        return self.state[layer]

    def __setitem__(self, layer: int, val):
        self.state[layer] = val

    def detach(self):
        for li in self.state:
            conv_state, ssm_state = self.state[li]
            self.state[li] = (conv_state.detach(), ssm_state.detach())

    def subtract(self, other: "MambaState"):
        '''
        Subtract the SSM states of the other MambaState from the current one.
        The conv states are not subtracted.
        '''
        new_state = {}
        for li in self.state:
            conv_state, ssm_state = self.state[li]
            other_ssm_state = other.state[li][1]
            new_ssm_state = ssm_state - other_ssm_state
            new_state[li] = (conv_state, new_ssm_state)
        return MambaState(new_state)

    def scale_ssm(self, scale: Tensor):
        '''
        Multiply the SSM states of the other MambaState from the current one.
        The conv states are not multiplied.

        scale: (L, B, nheads)
        '''
        new_state = {}
        for li in self.state:
            conv_state, ssm_state = self.state[li]
            new_ssm_state = torch.einsum('bhpn,bh->bhpn', ssm_state, scale[li])
            new_state[li] = (conv_state, new_ssm_state)
        return MambaState(new_state)



class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        states=None,
        inference_params=None,
        return_comps: bool = False,
        **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )  # type: ignore
        hidden_states, conv_state, ssm_state, comp = self.mixer(
            hidden_states,
            states=states,
            inference_params=inference_params,
            return_comps=return_comps,
            **mixer_kwargs,
        )

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )  # type: ignore
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual, conv_state, ssm_state, comp

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size,
            max_seqlen,
            dtype=dtype,
            **kwargs,
        )


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        conv_init=None,
        expand: int = 2,
        headdim: int = 64,
        d_ssm: Optional[int] = None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups: int = 1,
        A_init_range: tuple = (1, 16),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: tuple = (0.0, float("inf")),
        bias: bool = False,
        conv_bias: bool = True,
        # Fused kernel and sharding options
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
        layer_idx: Optional[int] = None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel: bool = True,
        use_B_norm: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.ssd_impl = 'triton'
        
        self.use_B_norm = use_B_norm
        if use_B_norm:
            self.B_norm = RMSNorm(self.ngroups * self.d_state)
            self.C_norm = RMSNorm(self.ngroups * self.d_state)

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model, d_in_proj * self.world_size, bias=bias,
                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,  # type: ignore
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

        self.dt_mult = None

    def set_dt_mult(self, dt_mult: float):
        self.dt_mult = dt_mult

    def forward(
        self,
        u: Tensor,
        seqlen: int = None,
        seq_idx=None,
        inference_params: Optional[InferenceParams] = None,
        states=None,
        return_comps: bool = False,
    ):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        states: the previous states, this is used instead of inference_params when I want to
            do parallel processing + backpropagation, because setting inference_params
            will call `self.step`, which only process one token.
        """
        # print(f"[Mamba2.forward]")
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                # assert torch.no_grad
                out, _, _ = self.step(u, conv_state, ssm_state)
                # But we are still returning conv_state and ssm_state so that we can 
                # use our MambaState class instead.
                return out, conv_state, ssm_state
            inplace_state_update = True
        elif states is not None:
            conv_state, ssm_state = states
            inplace_state_update = False

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if False and self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                if inplace_state_update:
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                    new_conv_state = None
                else:
                    new_conv_state = F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))  # Update state (B D W)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = rearrange(xBC, 'b l d -> b d l')
                xBC = self.act(self.conv1d(xBC))  # (B, L, self.d_ssm + 2 * ngroups * d_state)
                xBC = rearrange(xBC, 'b d l -> b l d')
                xBC = xBC[:, :-(self.d_conv - 1)]
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            
            if self.use_B_norm:
                B = self.B_norm(B)
                C = self.C_norm(C)

            if self.ssd_impl == 'triton':
                y = mamba_chunk_scan_combined(
                    rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                    dt,
                    A,
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    seq_idx=seq_idx,
                    initial_states=ssm_state,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                )
            elif self.ssd_impl == 'torch':
                raise NotImplementedError
                # 
                # x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
                # B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
                # C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
                # D = rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D
                # dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
                # print('================')
                # print(f'{x.shape = }')
                # print(f'{B.shape = }')
                # print(f'{C.shape = }')
                # print(f'{dt.shape = }')
                # print(f'{A.shape = }')
                # print("==================")
                # breakpoint()
                # y = self.ssd(x, dt, A, B, C, D)
            else:
                raise ValueError(f"Invalid SSD impl: {self.ssd_impl}")
            if ssm_state is not None:
                assert isinstance(y, tuple)
                y, last_state = y
                if inplace_state_update:
                    ssm_state.copy_(last_state)
                    new_ssm_state = None
                else:
                    new_ssm_state = last_state
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)  # type: ignore
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        if return_comps:
            comps = {
                'dt': F.softplus(dt + self.dt_bias.to(dtype=dt.dtype)),
                'A': A,
                'B': B,
                'C': C,
                'x': x,
            }
            return out, new_conv_state, new_ssm_state, comps
        return out, new_conv_state, new_ssm_state, None

    def ssd(self, x, dt, A, B, C, D, chunk_size: int = 64):
        '''
        Argument:
            x: (batch, seqlen, nheads, headdim)
            dt: (batch, seqlen, nheads)
            A: (nheads)
            B: (batch, seqlen, ngroups, dstate)
            C: (batch, seqlen, ngroups, dstate)
            D: (nheads, headdim) or (nheads,)
            z: (batch, seqlen, nheads, headdim)
        Return:
            out: (batch, seqlen, nheads, headdim)
        '''
        batch, seqlen, nheads, headdim = x.shape
        d_state = B.shape[-1]
        if seqlen % chunk_size != 0:
            dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
        # (B, T, H)
        dA = dt * rearrange(A, 'h -> h 1 1')
        y = ssd_minimal_discrete(x, dA, B, C, block_len=chunk_size)
        return y

    def step(
        self,
        hidden_states: Tensor,
        conv_state: Tensor,
        ssm_state: Tensor,
    ):
        # print(f"[Mamba2.step] {hidden_states.shape = }, {conv_state is None} {ssm_state is None}")
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        in_proj_time = time.time()
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )
        # print("in proj time:", time.time() - in_proj_time)

        # Conv step
        conv_time = time.time()
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        # print("conv time:", time.time() - conv_time)

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        ssm_time = time.time()
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        # print("ssm time:", time.time() - ssm_time)
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        # print("ssm_cfg: ", ssm_cfg)
        ssm_layer = ssm_cfg.pop("layer", "Mamba2")
        # print("ssm_layer: ", ssm_layer)
        if ssm_layer not in ["Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx: Optional[int] = None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        chunk_size: int = 128,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.chunk_size = chunk_size

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids: Tensor,
        inference_params=None,
        grad_ckpt: str = 'none',
        states=None,
        return_comps: bool = False,
        **mixer_kwargs,
    ):
        hidden_states = self.embedding(input_ids)
        residual = None
        # print(f"[MixerModel.forward] inference_params is None: {inference_params is None}")
        # print(f"[MixerModel.forward] states is None: {states is None}")
        if inference_params is not None:
            # During inference, states are passing using inference_params
            # print(f"got inference_params is not None")
            assert states is None
        else:
            # During training, we pass our custom `MambaState` for initializing states.
            if states is None:
                states = self.allocate_inference_cache(input_ids.shape[0], -1)  # max_seqlen is not used.

        if return_comps:
            all_comps: List[Dict[str, Tensor]] = []

        for layer_idx, layer in enumerate(self.layers):
            # mem_used = torch.cuda.memory_allocated() / 2 ** 20
            # print(f"[MambaModel.forward] {layer_idx = } | {mem_used = }")
            layer_state = states[layer_idx] if states is not None else None
            if grad_ckpt is None or grad_ckpt == 'none':
                outputs = layer(
                    hidden_states,
                    residual,
                    states=layer_state,
                    inference_params=inference_params,
                    return_comps=return_comps,
                )
                hidden_states, residual, conv_state, ssm_state, comps = outputs
                if return_comps:
                    all_comps.append(comps)
            elif grad_ckpt == 'layer':
                # The order of arguments must be the same as `Block.forward`.
                # `checkpoint` does not support kwargs.
                outputs = checkpoint(
                    layer,
                    hidden_states,
                    residual,
                    layer_state,
                    inference_params,
                    return_comps,
                    use_reentrant=True,
                )  # type: ignore
                hidden_states, residual, conv_state, ssm_state, comps = outputs
                if return_comps:
                    all_comps.append(comps)
            elif grad_ckpt == 'chunk':
                # Split x along the T dimension in chunks of size chunk_size, then apply activation checkpointing on each chunk.
                # This is done to save memory, as the activation checkpointing will save the activations of the layers
                # and only recomputes them when needed.
                x_chunks = hidden_states.chunk(hidden_states.size(1) // self.chunk_size, dim=1)
                residuals = [None for _ in range(len(x_chunks))]
                y_chunks = []
                chunk_state = layer_state
                for chunk_idx, x_chunk in enumerate(x_chunks):
                    y_chunk, residuals[chunk_idx], chunk_conv_state, chunk_ssm_state = checkpoint(
                        layer,
                        x_chunk,
                        residuals[chunk_idx],
                        chunk_state,
                        inference_params,
                        use_reentrant=False,
                    )  # type: ignore
                    mem_used = torch.cuda.memory_allocated() / 2 ** 20
                    # print(f"[MambaModel.forward] {layer_idx = } | {chunk_idx = } | {mem_used = }")
                    # print(y_chunk)
                    y_chunks.append(y_chunk)
                    chunk_state = (chunk_conv_state, chunk_ssm_state)
                hidden_states = torch.cat(y_chunks, dim=1)
                mem_used = torch.cuda.memory_allocated() / 2 ** 20
                # print(f"[MambaModel.forward] {layer_idx = } | {chunk_idx = } | {mem_used = }")
                # exit()
                conv_state = chunk_conv_state
                ssm_state = chunk_ssm_state
                comps = None
            else:
                raise ValueError("Invalid `grad_ckpt` found.")
            
            if states is not None:
                states[layer_idx] = (conv_state, ssm_state)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        if return_comps:
            return hidden_states, states, all_comps
        else:
            return hidden_states, states, None


class Mamba2ForCausalLM(nn.Module, GenerationMixin):
    def __init__(
        self,
        config,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,  # type: ignore
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
        self,
        input_ids: Tensor,
        position_ids=None,
        inference_params=None,
        num_last_tokens: int = 0,
        states=None,
        grad_ckpt: str = 'none',
        return_comps: bool = False,
        **mixer_kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if inference_params is not None:
            assert states is None
            states = None
        else:
            if states is not None:
                if isinstance(states, MambaState):
                    states = states.state
                elif isinstance(states, dict):
                    pass
                else:
                    raise TypeError(f"`states` has an invalid type: {type(states)}")
        hidden_states, last_states, comps = self.backbone(
            input_ids, 
            inference_params=inference_params,
            states=states, 
            grad_ckpt=grad_ckpt,
            return_comps=return_comps,
            **mixer_kwargs,
        )
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "states", "comps"])
        new_states = MambaState(last_states)
        return CausalLMOutput(logits=lm_logits, states=new_states, comps=comps)

    @classmethod
    def from_config(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)
