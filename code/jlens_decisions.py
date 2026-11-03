"""What is on the model's mind at the moment it decides to loop, versus the moment it decides to stop?

For N looping generations, takes the last position inside the loop whose next token is "Wait"; for N finished
generations, the position whose next token is </think>. At each, reads the residual stream after several layers
through the Jacobian lens and averages the lens log-probabilities over samples. Prints, per layer, the top tokens
for each group and the tokens most raised in loop decisions relative to stop decisions. With --edit/--alpha the
same texts are read under edited weights (capture.apply_edit), to see what the edit changes in the workspace.

    python jlens_decisions.py <base_hf> <lens.pt> <exp8_math> <gsm8k_test.jsonl> <out.npz> [--n 8] [--edit e --alpha 6]
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from capture import apply_edit, contexts

LAYERS = (9, 12, 15, 17, 19, 21)
TOP = 15
LOOP_REGION_CHARS = 6000


def main(a):
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16).eval()
    lm = model.model.language_model
    if a.edit:
        apply_edit(lm, a.edit, a.alpha)
    norm, head = lm.norm, model.lm_head
    J = {int(k): v.float() for k, v in torch.load(a.lens, map_location="cpu", weights_only=True)["J"].items()}
    wait_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("Wait", " Wait")}
    loop, finish = contexts(a.exp8, a.test, 40, a.seed)

    def prompt(q):
        return tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put "
                                         "your final answer within \\boxed{}."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True)

    def read(text, pick):
        x = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        p = pick(x[0])
        if p is None:
            return None
        with torch.no_grad():
            hs = model(input_ids=x[:, :p + 1], output_hidden_states=True, logits_to_keep=1).hidden_states
            return {L: torch.log_softmax(head(norm((hs[L + 1][0, p].float() @ J[L].T)[None].to(head.weight.dtype)))
                                         [0].float(), -1) for L in LAYERS}

    groups = {"loop": [], "stop": []}
    for pid, q, text in loop:
        if len(groups["loop"]) >= a.n:
            break
        p = prompt(q)
        start = len(tok(p + text[:-LOOP_REGION_CHARS], add_special_tokens=False).input_ids)

        def last_wait(ids):
            js = [j for j in range(start, len(ids) - 1) if int(ids[j + 1]) in wait_ids]
            return js[-1] if js else None
        r = read(p + text, last_wait)
        if r:
            groups["loop"].append(r)
            print(f"loop {pid}", flush=True)
    for pid, q, text in finish[:a.n]:
        groups["stop"].append(read(prompt(q) + text + "</think>", lambda ids: len(ids) - 2))
        print(f"stop {pid}", flush=True)

    out = {}
    for L in LAYERS:
        m = {g: torch.stack([r[L] for r in rows]).mean(0) for g, rows in groups.items()}
        out[f"loop{L}"], out[f"stop{L}"] = m["loop"].numpy(), m["stop"].numpy()
        dec = lambda ids: [tok.decode([i]) for i in ids]  # noqa: E731
        print(f"\n== layer {L}")
        print("  loop decision:", dec(m["loop"].topk(TOP).indices.tolist()))
        print("  stop decision:", dec(m["stop"].topk(TOP).indices.tolist()))
        print("  raised in loop vs stop:", dec((m["loop"] - m["stop"]).topk(TOP).indices.tolist()))
        print("  raised in stop vs loop:", dec((m["stop"] - m["loop"]).topk(TOP).indices.tolist()))
    np.savez(a.out, **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("lens")
    ap.add_argument("exp8")
    ap.add_argument("test")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--edit")
    ap.add_argument("--alpha", type=float, default=1.0)
    main(ap.parse_args())
