"""Directions from decision points (capture_decision.py), compared with Astra's, plus a candidate edit of my own.

    python analyze_decision.py <base_hf> <lens.pt> <capture_decision.npz> <astra_u.npz> <out_dir>

For each layer's MLP output: d = mean(loop-wait) - mean(close), how well it separates the two (leave-one-out AUC),
its cosine with Astra's u15 / u19, and its "Wait over </think>" push: the logit of Wait minus the logit of </think>
after transporting d through the Jacobian lens. Layers are picked by that push alone (no use of Astra's direction).
My edit uses the two top layers, d as the direction, and the 64 down_proj columns with the largest |d . W_col|.
"""
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from analyze import auc, loo_auc

TOPK = 64


def lens_logits(norm, head, J, L, v):
    with torch.no_grad():
        return head(norm((torch.tensor(v, dtype=torch.float32) @ J[L].T)[None]))[0]


def main(base, lens_path, cap, astra, out_dir):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32).eval()
    norm, head = model.model.language_model.norm, model.lm_head
    J = {int(k): v.float() for k, v in torch.load(lens_path, map_location="cpu", weights_only=True)["J"].items()}
    wait = [tok.encode(w, add_special_tokens=False)[0] for w in ("Wait", " Wait")]
    close = tok.convert_tokens_to_ids("</think>")
    z = np.load(cap)
    a = np.load(astra)
    u = {15: a["u15"], 19: a["u19"]}
    A, B, C = z["loop_wait_mlp"], z["close_mlp"], z["prod_wait_mlp"]
    print(f"samples: loop-wait {len(A)}, close {len(B)}, productive-wait {len(C)}")
    for g in ("loop_wait", "close", "prod_wait"):
        print(f"  {g}: mean log p(Wait) {z[g + '_lp_wait'].mean():.2f}, log p(</think>) {z[g + '_lp_close'].mean():.2f}")
    print("layer  AUC(loop|close)  AUC(loop|prod)  cos(d,u15)  cos(d,u19)  AUC u15  AUC u19  Wait-</think> push")
    push, dirs = {}, {}
    for L in range(A.shape[1]):
        X = np.concatenate([A[:, L], B[:, L]])
        y = np.r_[np.ones(len(A)), np.zeros(len(B))].astype(int)
        d = A[:, L].mean(0) - B[:, L].mean(0)
        d /= np.linalg.norm(d)
        dirs[L] = d
        X2 = np.concatenate([A[:, L], C[:, L]])
        y2 = np.r_[np.ones(len(A)), np.zeros(len(C))].astype(int)
        p15, p19 = X @ u[15], X @ u[19]
        if L in J:
            lg = lens_logits(norm, head, J, L, d)
            push[L] = float(torch.logsumexp(lg[wait], 0) - lg[close])
        print(f"{L:5d}  {loo_auc(X, y):15.3f}  {loo_auc(X2, y2):14.3f}  {d @ u[15]:10.3f}  {d @ u[19]:10.3f}  "
              f"{auc(p15[y == 1], p15[y == 0]):7.3f}  {auc(p19[y == 1], p19[y == 0]):7.3f}  {push.get(L, float('nan')):8.2f}")
    for L in (15, 19):
        for s, name in ((1, "+d (loop side)"), (-1, "-d (close side)")):
            lg = lens_logits(norm, head, J, L, s * dirs[L])
            print(f"lens, layer {L} {name}: {[tok.decode([i]) for i in lg.topk(12).indices.tolist()]}")
    best = sorted(push, key=push.get, reverse=True)[:2]
    print(f"layers with the strongest Wait-over-</think> push: {sorted(best)} "
          f"(pushes {[round(push[L], 2) for L in sorted(best)]})")
    W = load_file(os.path.join(base, "model.safetensors-00001-of-00001.safetensors"))
    edit = {"layers": np.array(sorted(best))}
    for L in sorted(best):
        w = W[f"model.language_model.layers.{L}.mlp.down_proj.weight"].float().numpy()
        cols = np.sort(np.argsort(-np.abs(dirs[L] @ w))[:TOPK])
        edit[f"u{L}"] = dirs[L].astype(np.float32)
        edit[f"cols{L}"] = cols
        shared = len(set(cols) & set(a[f"cols{L}"])) if L in (15, 19) else None
        lg = lens_logits(norm, head, J, L, dirs[L])
        print(f"my layer {L}: cos with u15 {dirs[L] @ u[15]:+.3f}, u19 {dirs[L] @ u[19]:+.3f}; columns shared with "
              f"Astra's: {shared}; lens of +d: {[tok.decode([i]) for i in lg.topk(10).indices.tolist()]}")
    np.savez(os.path.join(out_dir, "mine_decision_edit.npz"), **edit)
    np.savez(os.path.join(out_dir, "decision_dirs.npz"), **{f"d{L}": dirs[L] for L in dirs})


if __name__ == "__main__":
    main(*sys.argv[1:6])
