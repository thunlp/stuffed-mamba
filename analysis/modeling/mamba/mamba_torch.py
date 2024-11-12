import math
import sys
from typing import Tuple, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, einsum

from .pscan import pscan
from .pscan_triton_seq import triton_selective_scan_sequential
from .configuration_mamba import MambaConfig


"""

This file closely follows the mamba_simple.py from the official Mamba implementation,
and the mamba-minimal by @johnma2006.

The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison.
Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection:
    ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, T, D) and its outputs y is
    also (B, T, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, T, 2*ED) (where E is usually 2) and split it into x and z, each (B, T, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


class MambaLayer(nn.Module):
    """
    Norm + Mamba Mixer + Residual
    """

    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mixer = MambaMixer(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cache: Tuple[Tensor, Tensor],
        chunk_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
        - x: (B, T, D)
        - cache: (h, conv_cache)
        Returns:
        - output: (B, T, D)
        """

        res = x
        print("[MambaLayer.forward] before self.norm:")
        print(x)
        x = self.norm(x)
        print("[MambaLayer.forward] after self.norm:")
        print(x)
        y, cache = self.mixer(x, cache)
        y = y + res

        return y, cache

    def step(self, x: Tensor, cache: Tuple[Tensor, Tensor]):
        """
        Args:
        - x: (B, D)
        - cache: (h, conv_cache)
            - h: (B, ED, N)
            - conv_cache: (B, ED, d_conv-1)
        Returns:
        - y: (B, D)
        - cache: (h, conv_cache)
        """
        res = x
        x = self.norm(x)
        y, cache = self.mixer.step(x, cache)
        y = y + res
        return y, cache


class MambaMixer(nn.Module):
    """
    Up proj + conv1d + SSM + Down proj
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(
            config.d_model,
            2 * config.d_inner,
            bias=config.bias,
            dtype=config.get_dtype(),
        )

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
            dtype=config.get_dtype(),
        )

        # projects x to input-dependent delta, B, C
        assert isinstance(config.dt_rank, int)
        self.x_proj = nn.Linear(
            config.d_inner,
            config.dt_rank + 2 * config.d_state,
            bias=False,
            dtype=config.get_dtype(),
        )

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(
            config.dt_rank,
            config.d_inner,
            bias=True,
            dtype=config.get_dtype(),
        )

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt)
        )  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(
            torch.log(A).to(dtype=config.get_dtype())
        )  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True  # type: ignore

        self.D = nn.Parameter(torch.ones(config.d_inner, dtype=config.get_dtype()))
        self.D._no_weight_decay = True  # type: ignore

        # projects block output from ED back to D
        self.out_proj = nn.Linear(
            config.d_inner,
            config.d_model,
            bias=config.bias,
            dtype=config.get_dtype(),
        )

        # used in Jamba
        if config.inner_layernorms:
            self.dt_layernorm = RMSNorm(config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if config.scan_impl == "pscan_cuda":
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore

                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print(
                    "Trying to use the CUDA implementation of parallel scan (`scan_impl = 'pscan_cuda'`),"
                    " but failed to import mamba_ssm. Falling back to the PyTorch implementation."
                )

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(
        self,
        x: Tensor,
        state: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
        - x: (B, T, D)
        Returns:
        - y, cache
        """
        print("======== mixer input =======")
        print(x)
        _, T, _ = x.shape
        h: Tensor = state[0]  # (B, ED, N)
        conv_cache = state[1]  # (B, ED, d_conv - 1)

        xz = self.in_proj(x)  # (B, T, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, T, ED), (B, T, ED)
        # return self.out_proj(x), (h, conv_cache)  # ==== 6 MB

        # L is the third dimension because that's how 1D conv works in PyTorch.
        x = rearrange(x, "b l d -> b d l")  # (B, ED, L)
        x = torch.cat([conv_cache, x], dim=2)  # (B, ED, L + d_conv-1)

        # Depthwise convolution over time, with a short filter
        D_conv = self.config.d_conv
        new_conv_cache = x[:, :, -(D_conv - 1) :]
        x = self.conv1d(x)[:, :, D_conv - 1 : -(D_conv - 1)]
        x = rearrange(x, "b d l -> b l d")  # (B, T, ED)

        print("========== conv output =========")
        print(x)

        # return self.out_proj(x), (h, x[:, :, -10:])  # ==== 18 MB
        # return self.out_proj(x), (h, conv_cache)  # ==== 6 MB
        # return self.out_proj(x), (h, new_conv_cache.detach())  # ==== 18 MB
        # return self.out_proj(x), (h, new_conv_cache.clone())  # ==== 6 MB
        # return self.out_proj(x), (h, new_conv_cache)  # ==== 18 MB

        # (B, T, ED)
        x = F.silu(x)
        # z is not used by the PyTorch implementation, but it is used by the CUDA implementation
        # (B, T, ED)
        print("x =======================")
        print(x)
        y, h = self.ssm(x, z, h=h)
        print("y =======================")
        print(y)
        exit()

        if self.config.scan_impl == "pscan_cuda":
            output = self.out_proj(y)  # (B, T, D)
            return output

        # z branch
        z = F.silu(z)

        # (B, T, ED)
        output = y * z
        # print(output.dtype)
        output = self.out_proj(output)  # (B, T, D)

        new_cache = (h.clone(), new_conv_cache.clone())
        return output, new_cache

    def ssm(
        self,
        x: Tensor,
        z: Tensor,
        h: Optional[Tensor] = None,
        output_last_h: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x: (B, T, ED)
            z: (B, T, ED)
        Returns:
            y: (B, T, ED)
        """
        # return x, h_init
        assert isinstance(self.config.dt_rank, int)

        n = self.config.d_state
        # A = -torch.exp(self.A_log.float())  # (ED, N)
        # D = self.D.float()
        A = -torch.exp(self.A_log)  # (ED, N)
        D = self.D  # (ED)

        deltaBC = self.x_proj(x)  # (B, T, dt_rank+2*N)
        # (B, T, dt_rank), (B, T, N), (B, T, N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, n, n], dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)

        # choose which selective_scan function to use, according to config
        if self.config.scan_impl == "pscan_cuda":
            delta = self.dt_proj.weight @ delta.transpose(
                1, 2
            )  # (ED, dt_rank) @ (B, T, dt_rank) -> (B, ED, L)
            # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
            # the rest will be applied later (fused if using cuda)
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            assert (
                output_last_h is False
            ), "CUDA implementation does not support return last h."

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(
                x,
                delta,
                A,
                B,
                C,
                D,
                z=z,
                delta_softplus=True,
                delta_bias=self.dt_proj.bias.float(),
            )
            y = y.transpose(1, 2)  # (B, T, ED)  # type: ignore

        else:
            delta = F.softplus(self.dt_proj(delta))  # (B, T, D)

            if self.config.scan_impl == "pscan_torch":
                y, h_last = self.selective_scan(x, delta, A, B, C, D, h=h)
            elif self.config.scan_impl == 'seq_triton':
                y, h_last = self.selective_scan_triton_seq(x, delta, A, B, C, D, h=h)
            elif self.config.scan_impl == "seq":
                y, h_last = self.selective_scan_seq(x, delta, A, B, C, D, h=h)
            elif self.config.scan_impl == "cumsum":
                y, h_last = self.selective_scan_seq_sum(x, delta, A, B, C, D, h=h)
            else:
                raise ValueError(f"Unknown scan_impl in config: {self.config.scan_impl}")

        if output_last_h:
            return y, h_last
        else:
            return y

    def selective_scan_triton_seq(
        self,
        x: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # print(x)
        # print(delta)
        # print(A)
        # print(B)
        # print(C)
        # print(D)
        # print(h)
        # exit()
        orig_dtype = x.dtype
        x, delta, A, B, C, D, h = map(lambda t: t.float(), [x, delta, A, B, C, D, h])
        y, h = triton_selective_scan_sequential(
            u=x,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=D,
            initial_state=h,
        )
        return y.to(orig_dtype), h.to(orig_dtype)

    def selective_scan(
        self,
        x: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, ED)
            Δ: (B, T, ED)
            A: (ED, N)
            B: (B, T, N)
            C: (B, T, N)
            D: (ED)
        Returns:
            y: (B, T, ED)
        """
        # print("[selective_scan]")
        # dA = delta.unsqueeze(-1) * A
        dA = einsum(delta, A, "b t d, d n -> b t d n")
        # return x, dA[:, -1]  # 210 MB
        # return x, dA[:, -1].clone()  # ==== 18 MB
        dA = torch.exp(dA)  # (B, T, ED, N)

        # dB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, T, ED, N)
        # BX = dB * (x.unsqueeze(-1))  # (B, T, ED, N)
        BX = einsum(delta, x, B, "b t d, b t d, b t n -> b t d n")

        if h is None:
            raise NotImplementedError
            H: Tensor = pscan(dA, BX)  # (B, T, ED, N)  # type: ignore
        else:
            # prepend h to BX.
            BX = torch.cat([h.unsqueeze(1), BX], dim=1)  # (B, T+1, ED, N)
            dA = torch.cat(
                [h.unsqueeze(1), dA], dim=1
            )  # (B, T+1, ED, N), A[:, 0] has no effect on the result.
            H: Tensor = pscan(dA, BX)  # type: ignore
            H = H[:, 1:]

        y = einsum(C, H, "b t n, b t d n -> b t d")
        y = y + D * x
        return y, H[:, -1]

    def selective_scan_seq(self, x, delta, A, B, C, D, h=None):
        """
        Args:
        - x: (B, T, ED)
        - Δ: (B, T, ED)
        - A: (ED, N)
        - B: (B, T, N)
        - C: (B, T, N)
        - D: (ED)
        - h: (B, ED, N)
        Returns:
        - y : (B, T, ED)
        """

        _, T, _ = x.shape

        dA = torch.exp(einsum(delta, A, "b t d, d n -> b t d n"))
        dBx = einsum(delta, B, x, "b t d, b t n, b t d -> b t d n")

        if h is None:
            h = torch.zeros(
                x.size(0), self.config.d_inner, self.config.d_state, device=dA.device
            )  # (B, ED, N)
        ys = []
        for t in range(0, T):
            h = dA[:, t] * h + dBx[:, t]  # (B, ED, N)
            yt = einsum(h, C[:, t], "b d n, b n -> b d")  # (B, ED)
            ys.append(yt)
        y = torch.stack(ys, dim=1)  # (B, T, ED)
        y = y + D * x
        return y, h

    def selective_scan_seq_sum(
        self,
        x: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        The sequential summation implementation of Mamba's scan. This uses a trick where
            prod(exp(Delta * A)) = exp(sum(Delta * A))
        to make us of PyTorch's built-in cumsum, which is much faster than a for loop.

        Args:
        - x: (B, T, ED)
        - delta: (B, T, ED)
        - A: (ED, N)
        - B: (B, T, N)
        - C: (B, T, N)
        - D: (B, ED)
        - h: (B, ED, N)
        Returns (y, new_cache)
        - y: (B, T, ED)
        - h: (B, ED, N)
        """
        dA = einsum(delta, A, "b t d, d n -> b t d n")
        BX = einsum(delta, x, B, "b t d, b t d, b t n -> b t d n")

        # print("[selective_scan_seq_sum]", dA.shape, BX.shape)
        # T = dA.shape[1]
        # inv_idx = torch.arange(T - 1, -1, -1, device=x.device, dtype=torch.long)
        # dA = dA.index_select(1, inv_idx)
        # dA_cumsum = dA.cumsum(1).exp().index_select(1, inv_idx)
        dA_cumsum = dA.flip(1).cumsum(1).exp().flip(1)  # (B, T, ED, N)
        if h is None:
            raise NotImplementedError
        else:
            # Prepend h to BX to simulate passing in a state h_0.
            # BX = torch.cat([h.unsqueeze(1), BX], dim=1)  # (B, T+1, ED, N)
            # dA_cumsum = torch.cat([h.unsqueeze(1), dA_cumsum], dim=1)  # (B, T+1, ED, N), A[:, 0] has no effect on the result.
            H: Tensor = dA_cumsum * BX  # (B, T+1, ED, N)
            return dA_cumsum[..., 0], H[:, -1]
            H = H.cumsum(1) / (dA_cumsum + 1e-12)
            return dA_cumsum[..., 0], H[:, -1]
            # H = H[:, 1:]  # (B, T, ED, N)
            y = einsum(H, C, "b t d n, b t n -> b t d")
        y = y + D * x
        return y, H[:, -1]

    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    We just have to keep in cache, for each layer, two things:
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is
        a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the
        updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list
        of cache object. (See mamba_lm.py)
    """

    def step(
        self,
        x: Tensor,
        state: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x : (B, D)
            cache : (h, inputs)
            h : (B, ED, N)
            inputs : (B, ED, d_conv-1)

        Returns:
            y : (B, D)
            cache : (h, inputs)
        """
        h = state[0]  # (B, ED, N)
        conv_state = state[1]  # (B, ED, d_conv-1)

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, ED), (B, ED)

        # L is the third dimension because that's how 1D conv works in PyTorch.
        x = rearrange(x, "b d -> b d 1")  # (B, ED, 1)
        print(conv_state.shape)
        print(x.shape)
        x = torch.cat([conv_state, x], dim=2)  # (B, ED, L + d_conv - 1)
        new_conv_cache = x[:, :, 1:]  # (B, ED, d_conv-1)

        # Depthwise convolution over time, with a short filter
        x = self.conv1d(x)[:, :, self.config.d_conv - 1]  # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)

        new_cache = (h, new_conv_cache)
        return output, new_cache

    def ssm_step(
        self,
        x: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, ED)
            h: (B, ED, N)

        Returns:
            y: (B, ED)
            h: (B, ED, N)
        """
        A = -torch.exp(self.A_log)  # (ED, N)
        D = self.D

        deltaBC = self.x_proj(x)  # (B, dt_rank + 2 * N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],  # type: ignore
            dim=-1,
        )  # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        BX = einsum(deltaB, x, "b d n, b d -> b d n")  # (B, ED, N)

        if h is None:
            h = torch.zeros(
                x.size(0),
                self.config.d_inner,
                self.config.d_state,
                device=deltaA.device,
                dtype=x.dtype,
            )  # (B, ED, N)

        h = deltaA * h + BX  # (B, ED, N)
        # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)
        # y = (h @ C.unsqueeze(-1)).squeeze(2)
        y = einsum(h, C, "b d n, b n -> b d")
        y = y + D * x
        return y, h


# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        old_dtype = x.dtype
        x = x.to(torch.float32)
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )
        return output.to(old_dtype)
