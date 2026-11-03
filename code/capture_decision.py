"""Activations at decision points rather than averaged over a window.

  loop-wait:  in a looping generation (cut 2,000 chars after loop onset), every position whose next token opens a
              new cycle with "Wait" inside the loop region; averaged per sample
  close:      in a finished generation, the position whose next token is </think>
  prod-wait:  in a finished generation, every position whose next token is "Wait" (a check that went somewhere);
              averaged per sample

Stores the MLP output and the residual stream after every layer at those positions, plus the model's own
log-probabilities of "Wait" and of </think> there.

    python capture_decision.py <base_hf> <exp8_math> <gsm8k_test.jsonl> <out.npz> [--n 40] [--edit e.npz --alpha 6]

With --edit, the weights are edited first (capture.apply_edit: W' = W - alpha u u^T W on the listed columns), for
dose tests.
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from capture import apply_edit, contexts

LOOP_REGION_CHARS = 6000


def main(a):
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16).eval()
    lm = model.model.language_model
    if a.edit:
        apply_edit(lm, a.edit, a.alpha)
    wait_ids = {i for w in ("Wait", " Wait") for i in tok.encode(w, add_special_tokens=False)[:1]}
    close_id = tok.convert_tokens_to_ids("</think>")
    assert isinstance(close_id, int) and close_id != tok.unk_token_id, close_id
    loop, finish = contexts(a.exp8, a.test, a.n, a.seed)

    sel = []
    grab = {}
    for i, layer in enumerate(lm.layers):
        layer.mlp.register_forward_hook(lambda m, inp, out, i=i: grab.__setitem__(i, out[0, sel].float()))

    def run(prompt_text, positions_fn):
        x = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        pos = positions_fn(x[0])
        if not pos:
            return None
        sel[:] = pos
        with torch.no_grad():
            out = model(input_ids=x, output_hidden_states=True, logits_to_keep=torch.tensor(pos))
        lp = torch.log_softmax(out.logits[0].float(), -1)
        return {
            "mlp": torch.stack([grab[i].mean(0) for i in range(len(lm.layers))]).numpy(),
            "res": torch.stack([h[0, pos].float().mean(0) for h in out.hidden_states[1:]]).numpy(),
            "lp_wait": float(torch.logsumexp(lp[:, sorted(wait_ids)], -1).mean()),
            "lp_close": float(lp[:, close_id].mean()),
            "n": len(pos),
        }

    def prompt(q):
        return tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put "
                                         "your final answer within \\boxed{}."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True)

    groups = {"loop_wait": [], "close": [], "prod_wait": []}
    t0 = time.time()
    for pid, q, text in loop:
        p = prompt(q)
        start = len(tok(p + text[:-LOOP_REGION_CHARS], add_special_tokens=False).input_ids)
        r = run(p + text, lambda ids: [j for j in range(start, len(ids) - 1) if int(ids[j + 1]) in wait_ids])
        if r:
            groups["loop_wait"].append(r)
        print(f"loop {pid}: {r and r['n']} wait decisions ({time.time() - t0:.0f}s)", flush=True)
    for pid, q, text in finish:
        p = prompt(q)
        r = run(p + text + "</think>", lambda ids: [len(ids) - 2])
        groups["close"].append(r)
        r2 = run(p + text, lambda ids: [j for j in range(16, len(ids) - 1) if int(ids[j + 1]) in wait_ids])
        if r2:
            groups["prod_wait"].append(r2)
        print(f"finish {pid}: close lp {r['lp_close']:.2f}, {r2 and r2['n']} productive waits "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = {}
    for g, rows in groups.items():
        for k in ("mlp", "res"):
            out[f"{g}_{k}"] = np.array([r[k] for r in rows])
        for k in ("lp_wait", "lp_close", "n"):
            out[f"{g}_{k}"] = np.array([r[k] for r in rows])
    np.savez(a.out, **out)
    for g, rows in groups.items():
        print(f"{g}: {len(rows)} samples, mean log p(Wait) {np.mean([r['lp_wait'] for r in rows]):.2f}, "
              f"mean log p(</think>) {np.mean([r['lp_close'] for r in rows]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("exp8")
    ap.add_argument("test")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--edit")
    ap.add_argument("--alpha", type=float, default=1.0)
    main(ap.parse_args())
