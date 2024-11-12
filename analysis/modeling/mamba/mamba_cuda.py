from typing import Optional
import math

import bmtrain as bmt
import torch.nn.functional as F
from einops import rearrange, repeat
import torch
from torch import nn, Tensor

from .modeling_utils import Linear


try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from .modeling_utils import LayerNorm


class Mamba(bmt.DistributedModule):
    """
    Copied from `Mamba` from the official code of Mamba in the file `modules/mamba_simple.py`.

    A class for the Mamba part of every layer in the Mamba model, which is:
        up proj. + conv1d + act_fn + SSM + gate + down proj.
    """

    def __init__(
        self,
        d_model: int,  # dimension of the hidden representations between layers
        d_state: int = 16,  # N
        d_conv: int = 4,  # Size of kernel in 1d conv.
        # expand_factor: int = 2,  # E in the paper, d_inner (D in the paper) = E * d_model
        d_inner: Optional[int] = None,
        dt_rank: str = "auto",  # rank of △
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init="random",
        dt_scale: float = 0.02,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,  # Fused kernel options
        layer_idx: Optional[int] = None,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        tp: int = 0,
        init_std: float = 0.02,
        replace_conv_with_attn: bool = False,  # TODO: Implement this
        use_ssm: bool = True,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.use_ssm = use_ssm
        # self.expand = expand_factor
        # self.d_inner = int(self.expand * self.d_model)
        if d_inner is None:
            self.d_inner = 2 * d_model
        else:
            self.d_inner = d_inner
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        assert isinstance(self.dt_rank, int)
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj = Linear(
            dim_in=self.d_model,
            dim_out=self.d_inner * 2,
            dtype=dtype,
            init_std=init_std,
        )

        # conv1d = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=self.d_inner,
        #     padding=d_conv - 1,
        #     **factory_kwargs,
        # )
        # self.conv1d_weight = bmt.DistributedParameter(conv1d.weight)
        # self.conv1d_bias = bmt.DistributedParameter(conv1d.bias)

        self.conv1d_weight = bmt.DistributedParameter(
            torch.empty((self.d_inner, 1, d_conv), dtype=dtype),
            init_method=bmt.ParameterInitializer(
                torch.nn.init.normal_, mean=0.0, std=init_std
            ),
        )
        self.conv1d_bias = bmt.DistributedParameter(
            torch.empty((self.d_inner,), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.zeros_),
        )
        self.d_conv = d_conv

        self.activation = "silu"
        # self.act = nn.SiLU()
        self.act = F.silu

        if self.use_ssm:
            self.x_proj = Linear(
                dim_in=self.d_inner,
                dim_out=self.dt_rank + self.d_state * 2,
                dtype=dtype,
                bias=bias,
                init_std=init_std,
                tp=tp,
            )
            # self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            # self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            # if dt_init == "constant":
            #     nn.init.constant_(self.dt_proj.weight, dt_init_std)
            # elif dt_init == "random":
            #     nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            # else:
            #     raise NotImplementedError
            self.dt_proj = Linear(
                dim_in=self.dt_rank,
                dim_out=self.d_inner,
                dtype=dtype,
                bias=True,
                init_std=dt_init_std,
                tp=tp,
            )

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            self.dt_proj.bias._no_reinit = True

            # S4D real initialization
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_log = torch.log(A)  # Keep A_log in fp32
            # self.A_log = nn.Parameter(A_log)
            self.A_log = bmt.DistributedParameter(A_log)
            self.A_log._no_weight_decay = True

            # D "skip" parameter
            # self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D = bmt.DistributedParameter(torch.ones(self.d_inner, device=device))
            self.D._no_weight_decay = True

        self.out_proj = Linear(
            dim_in=self.d_inner,
            dim_out=self.d_model,
            dtype=dtype,
            bias=bias,
            init_std=init_std,
            tp=tp,
        )
        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states: Tensor, inference_params=None):
        """
        Args:
            hidden_states: (B, L, D)
        Returns:
            The same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        if self.use_ssm:
            A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
            # In the backward pass we write dx and dz next to each other to avoid torch.cat
            if (
                self.use_fast_path and inference_params is None
            ):  # Doesn't support outputting the states
                assert self.dt_proj.bias is not None
                out = mamba_inner_fn(
                    xz,
                    self.conv1d_weight,
                    self.conv1d_bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )  # type: ignore
            else:
                x, z = xz.chunk(2, dim=1)
                # Compute short convolution
                if conv_state is not None:
                    conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
                if causal_conv1d_fn is None:
                    # x = self.act(self.conv1d(x)[..., :seqlen])
                    x = self.act(
                        F.conv1d(
                            x,
                            self.conv1d_weight,
                            self.conv1d_bias,
                            padding=self.d_conv - 1,
                        )[..., :seqlen]
                    )
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x,
                        rearrange(self.conv1d_weight, "d 1 w -> d w"),
                        self.conv1d_bias,
                        self.activation,
                    )

                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
                x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(
                    x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
                )
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                assert self.dt_proj.bias is not None
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )  # type: ignore
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b d l -> b l d")
                out = self.out_proj(y)
        else:
            x, z = xz.chunk(2, dim=1)  # (B, L, D)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            # x = self.act(self.conv1d(x)[..., :seqlen])
            x = self.act(
                F.conv1d(
                    x, self.conv1d_weight, self.conv1d_bias, padding=self.d_conv - 1
                )[..., :seqlen]
            )
            out = self.out_proj(x)
        return out

    def step(self, hidden_states: Tensor, conv_state: Tensor, ssm_state: Tensor):
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self,
        inference_params,
        batch_size: int,
        initialize_states: bool = False,
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            # batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class MambaBlock(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        activate_fn: str,
        dtype=torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        layer_id: int = 0,
        num_layers: int = 0,
        # Mamba hyperparameters
        mamba_config: dict = {},
        residual_in_fp32: bool = False
    ):
        super().__init__()
        self.layernorm_before_mamba = LayerNorm(dim_model, eps=eps, dtype=dtype)
        self.mamba = Mamba(
            d_model=dim_model,
            d_state=mamba_config.get("d_state", 16),
            d_conv=mamba_config.get("d_conv", 4),
            d_inner=mamba_config.get("d_inner", 1920),
            layer_idx=layer_id,
            device=torch.device("cuda"),
            dtype=dtype,
            use_ssm=mamba_config.get("use_ssm", True),
        )
        self.scale = scale
        self.scale_depth = scale_depth
        self.num_layers = num_layers
        self.residual_in_fp32 = residual_in_fp32

        if dropout_p:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.dropout = nn.Identity()

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.layernorm_before_mamba(hidden_states)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mamba(hidden_states)

        # if self.scale_depth > 0:
        #     hidden_states = hidden_states + x.view_as(hidden_states) * (
        #         self.scale_depth / math.sqrt(self.num_layers)
        #     )  # https://arxiv.org/pdf/2310.02244.pdf
        # else:
        hidden_states = residual + hidden_states.view_as(residual)

        return hidden_states
