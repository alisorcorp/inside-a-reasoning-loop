"""Read residual-stream directions through Anthropic's Jacobian lens (pre-fitted by Neuronpedia for Qwen3.5-0.8B).

lens_l(h) = lm_head(final_norm(J_l @ h)), as in jlens.lens.JacobianLens.apply / jlens.hf unembed. For each direction
it prints the top tokens for +h and -h (a direction's sign is not known from a weight diff), with the plain logit
lens (no J) for comparison.

    python lens_read.py <base_hf> <lens.pt> <directions.npz> <key>:<layer> [<key>:<layer> ...]
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

TOP = 20


def main(base, lens_path, npz, specs):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32).eval()
    norm = model.model.language_model.norm
    head = model.lm_head
    J = {int(k): v.float() for k, v in torch.load(lens_path, map_location="cpu", weights_only=True)["J"].items()}
    z = np.load(npz)

    def top(h):
        with torch.no_grad():
            logits = head(norm(h[None]))[0]
        return [tok.decode([i]) for i in logits.topk(TOP).indices.tolist()]

    for spec in specs:
        key, layer = spec.split(":")
        u = torch.tensor(z[key], dtype=torch.float32)
        u = u / u.norm()
        for sign, name in ((1, "+"), (-1, "-")):
            h = sign * u
            print(f"{name}{key} @ layer {layer}")
            print("  J-lens:     ", top(h @ J[int(layer)].T))
            print("  logit lens: ", top(h))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
