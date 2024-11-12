from typing import Tuple, Optional
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

from .ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
# from .utils import RMSNorm
from .ssd_torch import ssd_minimal_discrete


class SSDMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.d_ssm = self.d_inner
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

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

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, state: Optional[Tuple[Tensor]] = None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        if state is not None:
            is_generating = state[2]
            if is_generating:
                self.step(u, state=state)
            ssd_state = state[0]
            conv_state = state[1]  # (d_conv - 1, d_model)

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        # initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # (B, L, d_inner)
        # (B, L, d_inner + 2 * ngroups * N), ngroups=1 because they use MVA
        # (B, L, nheads)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        if conv_state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = rearrange(xBC, "b l d -> b d l")
            conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
        assert self.activation in ["silu", "swish"]

        print(f'[SSDMixer] {xBC.shape = }')
        # xBC = torch.cat([xBC], dim=1)  # (B, L + d_conv - 1, ED + 2GN)
        new_conv_state = xBC[:, -(self.d_conv - 1):]

        # 1D Convolution
        xBC = rearrange(xBC, 'b l d -> b d l')
        xBC = self.conv1d(xBC)[:, :, :-(self.d_conv - 1)]  # (B, L, ED + 2GN)
        xBC = self.act(xBC)
        # (B, L, d_ssm + 2GN)
        xBC = rearrange(xBC, 'b d l -> b l d')

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        # X: (B, L, D)  # Values, D = n_value_heads * dim_v.
        # B: (B, L, H=1 * N)  # one head by default.
        # C: (B, L, H=1 * N)
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y, new_ssd_state = ssd_minimal_discrete(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt * A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            block_len=self.chunk_size,
            # D=self.D,
            # z=None,
            # seq_idx=seq_idx,
            initial_states=ssd_state,
            # **dt_limit_kwargs,
        )
        # (h, b l h p)
        y = y + self.D * x  # (b l h p)
        y = rearrange(y, "b l h p -> b l (h p)")  # (B, L, D)

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        new_state = (new_ssd_state, new_conv_state, True)
        return out, new_state

    def step(self, hidden_states: Tensor, state=None):
        '''
        Args:
            hidden_states: (B, D)
        '''
        # assert len(hidden_states.shape) == 2, "Hidden states must have shape (B, D)"
        batch, seqlen, dim = hidden_states.shape

        ssd_state = state[0]
        conv_state = state[1]  # (d_conv - 1, d_model)

        zxbcdt = self.in_proj(hidden_states)  # (B, L, d_in_proj)
        # initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # (B, d_inner)
        # (B, d_inner + 2 * ngroups * N), ngroups=1 because they use MVA
        # (B, nheads)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        assert self.activation in ["silu", "swish"]
        print(f'[SSDMixer] {xBC.shape = }')
        xBC = torch.cat([conv_state, xBC], dim=1)  # (B, L + d_conv - 1, ED + 2GN)

        # The last 3 vectors are a new 1D conv state.
        new_conv_state = xBC[:, -(self.d_conv - 1):]

        # 1D Convolution
        xBC = rearrange(xBC, 'b l d -> b d l')
        xBC = self.conv1d(xBC)[:, :, :-(self.d_conv - 1)]  # (B, L, ED + 2GN)
        xBC = self.act(xBC)
        # (B, L, d_ssm + 2GN)
        xBC = rearrange(xBC, 'b d l -> b l d')

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
        # Discretize A and B
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads), decay

        print(f'[SSDMixer] {x.shape = }')
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)  # Added to the Hopfield network
        ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
        y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        if not self.rmsnorm:
            y = y * self.act(z)  # (B D)

        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
