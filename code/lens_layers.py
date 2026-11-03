"""Read each layer's loop-minus-finish MLP direction through the Jacobian lens, and rank layers by how strongly the
direction says the loop's own opener ("Wait"), which the base model's looping text repeats.

    python lens_layers.py <base_hf> <lens.pt> <mean_diff_dirs.npz>
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

PROBES = ["Wait", " Wait", " wait", "Actually", " Actually"]


def main(base, lens_path, npz):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32).eval()
    norm, head = model.model.language_model.norm, model.lm_head
    J = {int(k): v.float() for k, v in torch.load(lens_path, map_location="cpu", weights_only=True)["J"].items()}
    z = np.load(npz)
    probe_ids = [tok.encode(p, add_special_tokens=False) for p in PROBES]
    probe_ids = [p[0] for p in probe_ids if len(p) == 1]
    rows = []
    for L in sorted(J):
        d = torch.tensor(z[f"d{L}"], dtype=torch.float32)
        with torch.no_grad():
            logits = head(norm((d @ J[L].T)[None]))[0]
        ranks = (logits[None, :] > logits[probe_ids][:, None]).sum(1)
        best = int(ranks.min())
        top = [tok.decode([i]) for i in logits.topk(8).indices.tolist()]
        rows.append((best, L, top))
        print(f"layer {L:2d}  best rank of {PROBES}: {best:6d}   top: {top}")
    print("\nlayers by how high the loop opener ranks:", [L for _, L, _ in sorted(rows)[:6]])


if __name__ == "__main__":
    main(*sys.argv[1:4])
