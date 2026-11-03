"""Single-example J-lens readouts for the figures: the token before an in-loop "Wait" (base and edited weights) and
the token before </think> in a finished generation.

Layers 13, 15, 17, 19, 21 of 24 sit at 54/62/71/79/88% depth, the rows Anthropic's figures use. "final" is the
model's own next-token prediction. Writes JSON for the page.

    python fig_data.py <base_hf> <lens.pt> <exp8_math> <gsm8k_test.jsonl> <astra_edit.npz> <out.json> [--n 6]
"""
import argparse
import json

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from capture import apply_edit, contexts

LAYERS = (13, 15, 17, 19, 21)
TOP = 6
LOOP_REGION_CHARS = 6000


def main(a):
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16).eval()
    lm = model.model.language_model
    norm, head = lm.norm, model.lm_head
    J = {int(k): v.float() for k, v in torch.load(a.lens, map_location="cpu", weights_only=True)["J"].items()}
    wait_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Wait", " Wait")})
    close_id = tok.convert_tokens_to_ids("</think>")
    loop, finish = contexts(a.exp8, a.test, 40, a.seed)

    def prompt(q):
        return tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put "
                                         "your final answer within \\boxed{}."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True)

    def read(ids, p):
        with torch.no_grad():
            out = model(input_ids=ids[None, :p + 1], output_hidden_states=True, logits_to_keep=1)
        rows = {}
        for L in LAYERS:
            h = (out.hidden_states[L + 1][0, p].float() @ J[L].T)[None].to(head.weight.dtype)
            lp = torch.log_softmax(head(norm(h))[0].float(), -1)
            rows[str(L)] = [tok.decode([i]) for i in lp.topk(TOP).indices.tolist()]
        fp = torch.softmax(out.logits[0, -1].float(), -1)
        top = fp.topk(TOP)
        rows["final"] = [[tok.decode([i]), round(float(v), 4)] for i, v in zip(top.indices.tolist(), top.values.tolist())]
        rows["p_wait"] = float(fp[wait_ids].sum())
        rows["p_close"] = float(fp[close_id])
        return rows

    cases = []
    for pid, q, text in loop:
        if len([c for c in cases if c["kind"] == "loop"]) >= a.n:
            break
        pr = prompt(q)
        ids = tok(pr + text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        start = len(tok(pr + text[:-LOOP_REGION_CHARS], add_special_tokens=False).input_ids)
        js = [j for j in range(start, len(ids) - 1) if int(ids[j + 1]) in wait_ids]
        if not js:
            continue
        p = js[-1]
        cases.append({"kind": "loop", "id": pid, "pos": p, "token": tok.decode(ids[p:p + 1]),
                      "before": tok.decode(ids[max(0, p - 70):p]), "after": tok.decode(ids[p + 1:p + 30]),
                      "base": read(ids, p), "_ids": ids})
        print(f"loop {pid} token {cases[-1]['token']!r}", flush=True)
    for pid, q, text in finish[:2]:
        ids = tok(prompt(q) + text + "</think>", return_tensors="pt", add_special_tokens=False).input_ids[0]
        p = len(ids) - 2
        cases.append({"kind": "stop", "id": pid, "pos": p, "token": tok.decode(ids[p:p + 1]),
                      "before": tok.decode(ids[max(0, p - 70):p]), "after": "</think>",
                      "base": read(ids, p), "_ids": ids})
        print(f"stop {pid} token {cases[-1]['token']!r}", flush=True)

    apply_edit(lm, a.edit, 6.0)
    for c in cases:
        c["edited"] = read(c["_ids"], c["pos"])
        del c["_ids"]
    json.dump({"layers": LAYERS, "n_layers": len(lm.layers), "cases": cases}, open(a.out, "w"), ensure_ascii=False,
              indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("base", "lens", "exp8", "test", "edit", "out"):
        ap.add_argument(k)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    main(ap.parse_args())
