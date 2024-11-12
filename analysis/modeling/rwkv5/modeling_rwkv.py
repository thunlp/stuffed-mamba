"""
Copied from RWKV-infctx-trainer on GitHub:
https://github.com/RWKV/RWKV-infctx-trainer
"""

import gc
import os
from pathlib import Path
from typing import Optional, List, Tuple, Union

global RWKV_JIT_ON, RWKV_TORCH_COMPILE, RWKV_NO_CUDA

from torch import nn, Tensor
import torch
from torch.utils.checkpoint import checkpoint

from .module.CoreDependencies import (
    JITModClass,
    TCompileBaseline,
    JITModMethod,
    IS_TORCH_2_1_COMPATIBLE,
)
from .module.channel_mixer import RWKV_ChannelMix
from .module.time_mixer import RWKV_TimeMix

# Isolating out known operations that **does not work** with torch.compile
# and wrapping them within a torch._dynamo.disable, this is required to get
# the baseline torc.compile to work


class BlockState:
    def __init__(
        self,
        time_mix_state: tuple[Tensor, Tensor],
        channel_mix_state: Tensor,
    ):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:
    def __init__(self, shift_states: Tensor, wkv_states: Tensor):
        self.wkv_states = wkv_states  # (N, B, H, DH, DH)
        self.shift_states = shift_states  # (N, 2, B, C)

    # @ TCompileMax (no difference)
    @staticmethod
    def create(N: int, B: int, C: int, n_head: int, head_size: int, device, dtype):
        result = BlockStateList.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    # @ TCompileMax (no difference)
    @staticmethod
    def empty(N: int, B: int, C: int, n_head: int, head_size: int, device, dtype):
        # @TODO: confirm if dtype can be changed from .flaot to dtype=dtype (when bf16)
        wkv_states = torch.empty(
            (N, B, n_head, head_size, head_size),
            # wkv_states = torch.empty((N, B, 1, n_head, head_size, head_size),
            device=device,
            #  dtype=dtype)
            dtype=torch.float,
        )
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            (self.shift_states[layer, 0], self.wkv_states[layer]),
            (self.shift_states[layer, 1]),
        )

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state[0]
        self.wkv_states[layer] = state.time_mix_state[1]
        self.shift_states[layer, 1] = state.channel_mix_state

    def detach(self):
        self.shift_states.detach()
        self.wkv_states.detach()


class Block(JITModClass):
    def __init__(
        self,
        layer_id: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
        head_size: int,
        dropout: float,
        dim_att: int,
        dim_ffn: int,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        else:
            self.ln0 = nn.Identity()

        self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)

        # Setup droupout at block level
        self.dropout = dropout
        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()

    @JITModMethod
    @TCompileBaseline
    def forward(self, x, last_state: BlockState):
        x = self.ln0(x)

        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )
        x = self.drop0(x + att_out)

        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )
        x = self.drop1(x + ffn_out)

        return x, BlockState(att_state, ffn_state)


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y, factor, currentMask):
        # Currently (8th July 2023), save_for_backward, causes an issue with
        # pytorch.compile (see: https://github.com/pytorch/pytorch/blob/e600505e3209eaf539e8bc99870ea55236cefbf5/torch/_dynamo/variables/higher_order_ops.py#L735)
        #
        # Due to L2Wrap being a major hotspot, we should monitor this for future support.
        # so that once its resolved, we can include the L2Wrap step in the torch.compile path
        #
        # See also:
        # - checkpointed_step
        ctx.save_for_backward(y, factor, currentMask)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y, factor, currentMask = ctx.saved_tensors

        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)

        # We ensure the mask is reshaped accordingly, and apply it against gy
        gy = gy * currentMask.reshape(
            gy.shape[0], gy.shape[1], 1
        )  # currentMask[:, None][None, :]
        return (grad_output, gy, None, None)


# ## ---
# Static optimized functions
# ## ---


class RWKV5(nn.Module):
    """
    Copied from RWKV-infctx-trainer, only support training for now.
    It uses CUDA to speed up, so it's fairly fast, but still slower than the official implementation
    because this supports passing in an initial state.
    """
    def __init__(
        self,
        # Model file path to load from
        pretrained_path: Union[str, Path],
        # Model size settings, which we either
        # "auto detect", or use the user specified settings
        d_model: int = -1,
        n_layer: int = -1,
        vocab_size: int = -1,
        # Context length size for the model
        ctx_len: int = 128*1024,
        # Context length schedule
        ctx_len_cutoffs: List[int] = [],
        ctx_len_warmup_steps: List[int] = [],
        # Dropout rate
        dropout: float = 0.0,
        # Adam optimizer settings
        beta1: float = 0.9,
        beta2: float = 0.99,
        adam_eps: float = 1.0e-08,
        weight_decay: float = 0.001,
        warmup_steps: int = -1,
        # Backprop settings
        grad_cp: bool = True,
        bptt_learning: bool = True,
        bptt_learning_range: int = -1,
        bptt_truncated_learning: bool = True,
        layerwise_lr: bool = True,
        dim_att: Optional[int] = None,
        dim_ffn: Optional[int] = None,
        substep_cuda_cache_clear: bool = False,
        substep_logging: bool = False,
        torch_set_float32_matmul_precision: str = "high",
    ):

        # Lets save everything in one shot
        # (this is used for wandb logging)
        self.setup_args = locals()
        del self.setup_args["self"]
        del self.setup_args["__class__"]

        # Setup the parent class
        super().__init__()

        # Load the model, unless its the special ".//<#|=@%!$init_model$!%@=|#>//." path
        # which is reserved to be used with the `init_model.py`
        #
        # We intentionally used several filesystem illegal characters, to ensure it
        # is not accidentally used by the user for a real file
        model_weights = None
        model_keys = None
        if pretrained_path != ".//<#|=@%!$init_model$!%@=|#>//.":
            # Check if the load_model path exists, and is a file
            if not os.path.isfile(pretrained_path):
                raise ValueError(f"load_model file '{pretrained_path}' does not exist")

            # Load the model weights
            if IS_TORCH_2_1_COMPATIBLE:
                model_weights = torch.load(
                    pretrained_path, map_location="cpu", weights_only=True, mmap=True
                )
            else:
                model_weights = torch.load(pretrained_path, map_location="cpu")

            # Get the model keys
            model_keys = list(model_weights.keys())

        assert model_keys is not None
        assert model_weights is not None
        # Lets compute the model various sizes, if they are not provided
        if n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if "blocks." in x:
                    block_id = int(x.split(".")[1])
                    max_block_id = max(max_block_id, block_id)
            n_layer = max_block_id + 1

        if d_model < 0:
            d_model = model_weights["head.weight"].shape[1]

        if vocab_size < 0:
            vocab_size = model_weights["head.weight"].shape[0]

        # Save the various other params for later
        self.ctx_len = ctx_len
        self.ctx_len_cutoffs = ctx_len_cutoffs
        self.ctx_len_warmup_steps = ctx_len_warmup_steps
        self.n_embd = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.layerwise_lr = layerwise_lr
        self.grad_cp = grad_cp
        self.dropout = dropout
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.bptt_learning = bptt_learning
        self.bptt_learning_range = bptt_learning_range
        self.bptt_truncated_learning = bptt_truncated_learning

        dim_att = dim_att or d_model
        dim_ffn = dim_ffn or int((d_model * 3.5) // 32 * 32)
        self.dim_att = dim_att
        self.dim_ffn = dim_ffn

        # Compute the RWKV-v5 n_head / headsize
        head_size = 64
        self.head_size = head_size
        self.head_size_divisor = 8

        n_head = dim_att // head_size
        self.n_head = n_head
        assert (
            dim_att % n_head == 0
        ), f"dim_att must be divisible by head_size ({self.head_size})"

        # Validate various sizes
        assert d_model % 32 == 0, "d_model must be divisible by 32"
        assert dim_att % 32 == 0, "dim_att must be divisible by 32"
        assert dim_ffn % 32 == 0, "dim_ffn must be divisible by 32"

        # Matmu precision check
        if torch_set_float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(torch_set_float32_matmul_precision)
        self.emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList(
            [
                Block(i, n_layer, d_model, n_head, head_size, dropout, dim_att, dim_ffn)
                for i in range(n_layer)
            ]
        )

        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Dropout handling
        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)

        # load the state, and GC the original cpu copy
        if model_weights != None:
            self.load_state_dict(model_weights)
            del model_weights
            gc.collect()

    def tie_weights(self):
        # RWKV-5 does not share input and output embeddings
        pass

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor,
        states: Optional[BlockStateList] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, BlockStateList]:
        B, T = input_ids.size()
        assert T <= self.ctx_len, "The input length exceeds the pre-set ctx_len"

        x = self.emb(input_ids)

        if self.dropout > 0.0:
            x = self.drop0(x)

        # `states` can be None, when we are performing direct inference
        if states is None:
            states = BlockStateList.create(
                self.n_layer,
                B,
                self.n_embd,
                self.n_head,
                self.head_size,
                x.device,
                x.dtype,
            )
        else:
            states = states

        new_states = BlockStateList.empty(
            self.n_layer, B, self.n_embd, self.n_head, self.head_size, x.device, x.dtype
        )
        # Avoid using the zip operation, as torch.compile throws an exception on it
        # with `zip not reconized as a valid function`
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            state = states[i]
            if self.grad_cp:
                x, new_state = checkpoint(block, x, state)  # type: ignore
            else:
                x, new_state = block(x, state)
            new_states[i] = new_state

        # Final layernorm and head output
        x = self.ln_out(x)
        logits = self.head(x)

        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(logits.device)
            labels = torch.cat(
                (labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)),
                1,
            )
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return loss, logits, new_states
