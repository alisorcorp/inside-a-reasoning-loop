"""Recover the direction Astra removed, from the weight diff alone.

The p50c edit changed 64 columns of the MLP down-projection in layers 15 and 19. If it really is "subtract the
loop direction u", every changed column moved along u, so the diff is rank one and its top left-singular vector is u.

    python astra_direction.py <base.safetensors> <edited.safetensors> <out.npz>
"""
import sys

import numpy as np
from safetensors.torch import load_file as tload

NAMES = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in (15, 19)]


def main(base, edited, out):
    a = tload(base)
    b = tload(edited)
    res = {}
    for n in NAMES:
        w0 = a[n].float().numpy()
        w1 = b[n].float().numpy()
        d = w1 - w0                                  # [1024 hidden, 3584 intermediate]
        cols = np.flatnonzero(np.abs(d).sum(0) > 0)
        U, S, Vt = np.linalg.svd(d[:, cols], full_matrices=False)
        u = U[:, 0]
        # sign: the edit removes the component along u, so the changed columns' projection on u shrinks
        before = u @ w0[:, cols]
        after = u @ w1[:, cols]
        if np.abs(after).mean() > np.abs(before).mean():
            u = -u
        L = n.split(".")[3]
        res[f"u{L}"] = u
        res[f"cols{L}"] = cols
        print(f"layer {L}: {len(cols)} columns changed; singular values {np.round(S[:4], 4)} "
              f"(rank-1 share {S[0]**2 / (S**2).sum():.4f}); |u.W_col| mean before {np.abs(before).mean():.4f} "
              f"after {np.abs(after).mean():.4f}; ratio after/before {np.mean(after / before):.3f}")
    print(f"cos(u15, u19) = {float(res['u15'] @ res['u19']):.4f}")
    np.savez(out, **res)


if __name__ == "__main__":
    main(*sys.argv[1:4])
