"""Write a checkpoint with Astra's direction applied at a chosen strength: W' = W - alpha u u^T W on the 64
down_proj columns per layer that Astra edited (layers 15 and 19), computed in float32 and rounded to bfloat16 like
the authors' evaluation.

    python make_alpha_ckpt.py <base.safetensors> <astra_u.npz> <alpha> <out.safetensors> [--check <edited.safetensors>]

--check compares the result with another checkpoint's edited tensors (use alpha 6 against Astra's own p50c file to
confirm the recipe reproduces it).
"""
import argparse

import numpy as np
import torch
from safetensors.torch import load_file, save_file

LAYERS = (15, 19)


def name(L):
    return f"model.language_model.layers.{L}.mlp.down_proj.weight"


def main(a):
    sd = load_file(a.base)
    z = np.load(a.astra_u)
    for L in LAYERS:
        W = sd[name(L)]
        u = torch.tensor(z[f"u{L}"], dtype=torch.float32)
        u = u / u.norm()
        cols = torch.tensor(z[f"cols{L}"], dtype=torch.long)
        w = W[:, cols].float()
        new = W.clone()
        new[:, cols] = (w - a.alpha * torch.outer(u, u @ w)).to(W.dtype)
        sd[name(L)] = new
        before, after = u @ w, u @ new[:, cols].float()
        print(f"layer {L}: component along u after/before {float((after / before).mean()):+.3f} "
              f"(expected {1 - a.alpha:+.3f})")
    if a.check:
        ref = load_file(a.check)
        for L in LAYERS:
            same = (sd[name(L)] == ref[name(L)]).float().mean().item()
            diff = (sd[name(L)].float() - ref[name(L)].float()).abs().max().item()
            print(f"layer {L}: {same:.4%} of entries identical to {a.check}; largest difference {diff:.2e}")
    save_file(sd, a.out, metadata={"format": "pt"})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("astra_u")
    ap.add_argument("alpha", type=float)
    ap.add_argument("out")
    ap.add_argument("--check")
    main(ap.parse_args())
