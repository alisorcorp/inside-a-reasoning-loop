"""Compare an activation-derived loop direction with Astra's, and work out which way Astra's edit pushes.

    python analyze.py <capture.npz> <astra_u.npz> <base model.safetensors> <out_dir>

Prints, per layer: how well the loop-minus-finish mean difference of the MLP output separates the two kinds of
context (leave-one-out AUC, so the direction is never fitted on the sample it scores), its cosine with Astra's
directions, and the AUC of Astra's own direction. Then, for layers 15 and 19: the 64 edited neurons' summed write
along u in each kind of context before the edit (after the edit it is -5 times that). Writes my candidate edit
(direction and 64 columns per chosen layer) to <out_dir>/mine_edit.npz in the format capture.py --edit reads.
"""
import os
import sys

import numpy as np
from safetensors.torch import load_file

TOPK = 64


def auc(pos, neg):
    """P(score of a random positive > score of a random negative)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def loo_auc(X, y):
    """Leave-one-out: score each sample on the mean-difference direction fitted without it."""
    s = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        d = X[m & (y == 1)].mean(0) - X[m & (y == 0)].mean(0)
        s[i] = X[i] @ d / np.linalg.norm(d)
    return auc(s[y == 1], s[y == 0])


def main(cap, astra, base, out_dir):
    z = np.load(cap)
    a = np.load(astra)
    y = z["label"].astype(int)
    mlp, res, act = z["mlp"], z["res"], z["act"]
    u = {15: a["u15"], 19: a["u19"]}
    nl = mlp.shape[1]
    print(f"{(y == 1).sum()} loop contexts, {(y == 0).sum()} finish contexts; "
          f"mean logp of the last tokens: loop {z['logp'][y == 1].mean():.3f}, finish {z['logp'][y == 0].mean():.3f}")
    print("layer  LOO-AUC(mlp)  cos(d,u15)  cos(d,u19)  AUC(u15 on mlp)  AUC(u19 on mlp)  LOO-AUC(resid)")
    dirs = {}
    for L in range(nl):
        X = mlp[:, L]
        d = X[y == 1].mean(0) - X[y == 0].mean(0)
        d /= np.linalg.norm(d)
        dirs[L] = d
        p15, p19 = X @ u[15], X @ u[19]
        print(f"{L:5d}  {loo_auc(X, y):12.3f}  {d @ u[15]:10.3f}  {d @ u[19]:10.3f}  "
              f"{auc(p15[y == 1], p15[y == 0]):15.3f}  {auc(p19[y == 1], p19[y == 0]):15.3f}  "
              f"{loo_auc(res[:, L], y):14.3f}")

    W = load_file(base)
    print("\nAstra's 64 neurons per layer, their summed write along u (u as stored; the lens reads +u as "
          "'results', -u as 'Actually / ?**'):")
    for L in (15, 19):
        w = W[f"model.language_model.layers.{L}.mlp.down_proj.weight"].float().numpy()   # [1024, 3584]
        cols = a[f"cols{L}"]
        c = act[:, L][:, cols] @ (u[L] @ w[:, cols])                                     # per sample
        allc = act[:, L] @ (u[L] @ w)
        print(f"  layer {L}: loop {c[y == 1].mean():+.4f}  finish {c[y == 0].mean():+.4f}  "
              f"(whole MLP: loop {allc[y == 1].mean():+.4f}, finish {allc[y == 0].mean():+.4f}); "
              f"after the edit the 64 write -5x these")

    # my edit: the two layers whose MLP output separates best, direction = mean difference there,
    # columns = the 64 neurons whose loop-minus-finish activation change writes most along it
    scores = {L: loo_auc(mlp[:, L], y) for L in range(4, nl - 2)}
    best = sorted(scores, key=scores.get, reverse=True)[:2]
    edit = {"layers": np.array(sorted(best))}
    for L in sorted(best):
        w = W[f"model.language_model.layers.{L}.mlp.down_proj.weight"].float().numpy()
        dact = act[y == 1, L].mean(0) - act[y == 0, L].mean(0)
        contrib = dact * (dirs[L] @ w)
        cols = np.argsort(-contrib)[:TOPK]
        edit[f"u{L}"] = dirs[L].astype(np.float32)
        edit[f"cols{L}"] = np.sort(cols)
        near = {k: len(set(cols) & set(a[f"cols{k}"])) for k in (15, 19) if k == L}
        print(f"my layer {L}: LOO-AUC {scores[L]:.3f}, cos with u15 {dirs[L] @ u[15]:+.3f}, u19 {dirs[L] @ u[19]:+.3f}; "
              f"columns shared with Astra's at this layer: {near or 'n/a'}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "mine_edit.npz"), **edit)
    np.savez(os.path.join(out_dir, "mean_diff_dirs.npz"), **{f"d{L}": dirs[L] for L in dirs})
    astra_edit = {"layers": np.array([15, 19])}
    for L in (15, 19):
        astra_edit[f"u{L}"] = u[L].astype(np.float32)
        astra_edit[f"cols{L}"] = a[f"cols{L}"]
    np.savez(os.path.join(out_dir, "astra_edit.npz"), **astra_edit)


if __name__ == "__main__":
    main(*sys.argv[1:5])
