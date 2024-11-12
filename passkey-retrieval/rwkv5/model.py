########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import types
from typing import Optional

from torch.nn import functional as F
from transformers import AutoTokenizer
import numpy as np
import torch

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method


np.set_printoptions(precision=4, suppress=True, linewidth=200)

########################################################################################################

# THIS IS NOW UPDATED TO SUPPORT LATEST RWKV-5 WORLD v2 MODELS


class RWKV5(MyModule):
    def __init__(self, args: types.SimpleNamespace):
        super().__init__()
        self.args = args
        self.eval()  # set torch to inference mode

        w = torch.load(args.pretrained_path, map_location="cpu")
        for k in w.keys():
            w[k] = w[k].float()  # convert to f32 type
            if ".time_" in k:
                w[k] = w[k].squeeze()
            if ".time_decay" in k:
                w[k] = torch.exp(-torch.exp(w[k])).unsqueeze(-1)
            if ".time_faaaa" in k:
                w[k] = w[k].unsqueeze(-1)

        self.n_head = w["blocks.0.att.time_decay"].shape[0]
        self.head_size = w["blocks.0.ln1.weight"].shape[0] // self.n_head

        self.w = types.SimpleNamespace()  # set self.w from w
        self.w.blocks = {}
        for (
            k
        ) in (
            w.keys()
        ):  # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split(".")
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here:
                        here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p):
                        setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @MyFunction
    def channel_mixing(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        i0 = (2 + self.head_size) * i + 0
        xk = x * time_mix_k + state[i0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[i0] * (1 - time_mix_r)
        state[i0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))  # square relu, primer paper
        return r * (vw @ k)

    @MyFunction
    def time_mixing(
        self,
        x,
        state,
        i: int,
        time_mix_k,
        time_mix_v,
        time_mix_r,
        time_mix_g,
        time_first,
        time_decay,
        kw,
        vw,
        rw,
        gw,
        ow,
        ln_w,
        ln_b,
    ):
        H = self.n_head
        S = self.head_size

        i1 = (2 + S) * i + 1
        xk = x * time_mix_k + state[i1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[i1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[i1] * (1 - time_mix_r)
        xg = x * time_mix_g + state[i1] * (1 - time_mix_g)
        state[i1] = x

        r = (rw @ xr).view(H, 1, S)
        k = (kw @ xk).view(H, S, 1)
        v = (vw @ xv).view(H, 1, S)
        g = F.silu(gw @ xg)

        s = state[(2 + S) * i + 2 : (2 + S) * (i + 1), :].reshape(H, S, S)

        x = torch.zeros(H, S)
        a = k @ v
        x = r @ (time_first * a + s)
        s = a + time_decay * s

        state[(2 + S) * i + 2 : (2 + S) * (i + 1), :] = s.reshape(S, -1)
        x = x.flatten()

        x = (
            F.group_norm(
                x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5
            ).squeeze(0)
            * g
        )  # same as gn(x/8, eps=1e-5)
        return ow @ x

    def forward(self, token, state: Optional[torch.Tensor]):
        with torch.no_grad():
            if state is None:
                state = torch.zeros(
                    self.args.n_layer * (2 + self.head_size), self.args.n_embd
                )

            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(
                    self.layer_norm(x, self.w.blocks[i].ln1),
                    state,
                    i,
                    att.time_mix_k,
                    att.time_mix_v,
                    att.time_mix_r,
                    att.time_mix_g,
                    att.time_faaaa,
                    att.time_decay,
                    att.key.weight,
                    att.value.weight,
                    att.receptance.weight,
                    att.gate.weight,
                    att.output.weight,
                    att.ln_x.weight,
                    att.ln_x.bias,
                )
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(
                    self.layer_norm(x, self.w.blocks[i].ln2),
                    state,
                    i,
                    ffn.time_mix_k,
                    ffn.time_mix_r,
                    ffn.key.weight,
                    ffn.value.weight,
                    ffn.receptance.weight,
                )

            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state

    def generate(self, input_ids, max_new_tokens: int = 20, temperature: float = 1.0, top_p: float = 0.1):
        # print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
        init_state = None
        init_out, init_state = self.forward(input_ids, state=init_state)

        all_tokens = []
        out, state = init_out.clone(), init_state.clone()
        for i in range(max_new_tokens):
            token = sample_logits(out, temperature, top_p)
            all_tokens += [token]
            out, state = self.forward(token, state)
        return all_tokens


def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out


def from_pretrained(path):
    args = types.SimpleNamespace()
    args.pretrained_path = path
    args.n_layer = 24
    args.n_embd = 1024
    args.vocab_size = 65536

    print(f"Loading checkpoint from: {path}")
    model = RWKV5(args).cuda()
    return model


def main():
    tok_name = 'RWKV/v5-EagleX-v2-7B-HF'
    print(f"Loading tokenizer from: {tok_name}")
    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)

    prompt = 'My name is'
    input_ids = tok(prompt, return_tensors='pt', max_length=128, pad_to_multiple_of=128, padding='left').input_ids.cuda()
    print(input_ids)

    path = '../../RWKV-5-World-0.4B-v2-20231113-ctx4096.pt'

    args = types.SimpleNamespace()
    args.pretrained_path = path
    args.n_layer = 24
    args.n_embd = 1024
    args.vocab_size = 65536

    print(f"Loading checkpoint from: {path}")
    model = RWKV5(args).cuda()

    prompt = 'An increasing sequence: one, two, three'
    input_ids = tok(prompt, return_tensors='pt')
    output_ids = model.generate(input_ids, max_new_tokens=20)
    output_text = tok.batch_decode(output_ids)[0][len(prompt):]
    print(output_text)


if __name__ == "__main__":
    main()
