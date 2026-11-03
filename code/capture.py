"""Teacher-force the base model on its own released GSM8K generations and record activations in two kinds of context.

  loop:   a generation that never closed its thinking, cut 2,000 chars after the loop detector fires (so the last
          tokens are inside the loop)
  finish: a generation that did close its thinking, cut just before </think> (the model is wrapping up)

For the last POS tokens of each context it stores, averaged over those tokens: the residual stream after every layer,
every layer's MLP output, and every layer's MLP hidden activation (the input of down_proj); plus the mean log-prob
of those tokens. With --edit it first applies an edit file (from make_edit.py) to the weights, for dose tests.

    python capture.py <base_hf> <exp8_math> <gsm8k_test.jsonl> <out.npz> [--n 40] [--edit edit.npz --alpha 6]
"""
import argparse
import glob
import gzip
import json
import random
import time
import zlib

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

POS = 384
WIN, STEP, MIN_POS, THR, K = 4000, 500, 6000, 0.06, 4


def loop_onset(text):
    run = 0
    for end in range(MIN_POS, len(text) + 1, STEP):
        b = text[end - WIN:end].encode()
        run = run + 1 if len(zlib.compress(b, 6)) / len(b) < THR else 0
        if run >= K:
            return end
    return None


def contexts(exp8, test, n, seed):
    questions = [json.loads(line)["question"] for line in open(test)]
    loop, finish = [], []
    for f in sorted(glob.glob(f"{exp8}/eval/unedited/parts/gsm8k_s*/gsm8k_first_samples.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            q = questions[int(r["id"].rsplit("_", 1)[1])]
            t = r["text"]
            if "</think>" in t:
                finish.append((r["id"], q, t.split("</think>", 1)[0]))
            else:
                at = loop_onset(t)
                if at:
                    loop.append((r["id"], q, t[:at + 2000]))
    rng = random.Random(seed)
    rng.shuffle(loop)
    rng.shuffle(finish)
    return loop[:n], finish[:n]


def apply_edit(lm, path, alpha):
    """W' = W - alpha * u u^T W on the listed columns of each listed down_proj (u unit-norm)."""
    z = np.load(path)
    for L in z["layers"]:
        W = lm.layers[int(L)].mlp.down_proj.weight
        u = torch.tensor(z[f"u{L}"], dtype=torch.float32)
        cols = torch.tensor(z[f"cols{L}"], dtype=torch.long)
        w = W.data[:, cols].float()
        W.data[:, cols] = (w - alpha * torch.outer(u, u @ w)).to(W.dtype)


def main(a):
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16).eval()
    lm = model.model.language_model
    if a.edit:
        apply_edit(lm, a.edit, a.alpha)
    nl = len(lm.layers)
    loop, finish = contexts(a.exp8, a.test, a.n, a.seed)
    rows = [(1, *c) for c in loop] + [(0, *c) for c in finish]

    grab = {}
    hooks = []
    for i, layer in enumerate(lm.layers):
        hooks.append(layer.mlp.register_forward_hook(
            lambda m, inp, out, i=i: grab.__setitem__(("mlp", i), out[0, -POS:].float().mean(0))))
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            lambda m, inp, out, i=i: grab.__setitem__(("act", i), inp[0][0, -POS:].float().mean(0))))

    res, mlp, act, lp, label, ids, ntok = [], [], [], [], [], [], []
    t0 = time.time()
    for k, (y, pid, q, text) in enumerate(rows):
        prompt = tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put "
                                           "your final answer within \\boxed{}."}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if k == 0:
            assert prompt.endswith("<think>\n"), repr(prompt[-40:])
        x = tok(prompt + text, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            out = model(input_ids=x, output_hidden_states=True, logits_to_keep=POS + 1)
        logits = out.logits[0, :-1].float()
        tgt = x[0, -POS:]
        lp.append(float(torch.log_softmax(logits, -1).gather(1, tgt[:, None]).mean()))
        res.append(torch.stack([h[0, -POS:].float().mean(0) for h in out.hidden_states[1:]]).numpy())
        mlp.append(torch.stack([grab[("mlp", i)] for i in range(nl)]).numpy())
        act.append(torch.stack([grab[("act", i)] for i in range(nl)]).numpy())
        label.append(y)
        ids.append(pid)
        ntok.append(x.shape[1])
        print(f"{k + 1}/{len(rows)} {'loop' if y else 'finish'} {pid} {x.shape[1]} tok logp {lp[-1]:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    np.savez(a.out, res=np.array(res), mlp=np.array(mlp), act=np.array(act), logp=np.array(lp),
             label=np.array(label), ids=np.array(ids), ntok=np.array(ntok))


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
