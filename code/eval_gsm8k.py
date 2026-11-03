"""GSM8K eval against a llama-server, with the authors' prompt and sampling, plus early loop abort.

    python eval_gsm8k.py <test.jsonl> <out.jsonl> --port 8301 --n 40 --par 8 --max-tokens 16384

Prompt and grading follow exp8_math/code/mathqa.py: problem + "\\nPlease reason step by step, and put your final
answer within \\boxed{}.", thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5; the answer is
the last \\boxed{} after </think>, checked with math-verify against the gold number.

Loop abort (calibrated on the authors' released generations by calibrate_loop.py): once past 6,000 chars, if the
last 4,000 chars zlib-compress below 6% of their size at 4 consecutive checks 500 chars apart, the stream is closed
and the sample is recorded as a loop. On their outputs this flagged 85% of never-finishing samples (median ~22k
chars in) and wrongly flagged 1% of finished ones. A loop counts as wrong, like a cut at the token limit.

Resumable: problems already in <out.jsonl> are skipped.
"""
import argparse
import asyncio
import json
import os
import random
import re
import time
import zlib

import httpx
from math_verify import parse, verify

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
WIN, STEP, MIN_POS, THR, K = 4000, 500, 6000, 0.06, 4


def load_problems(path, n, seed):
    rows = [json.loads(line) for line in open(path)]
    for i, r in enumerate(rows):
        r["id"] = f"gsm8k_test_{i}"
        r["gold"] = r["answer"].split("####")[-1].strip().replace(",", "")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def grade(text, gold):
    if "</think>" not in text:
        return None, False
    pred = last_boxed(text.split("</think>", 1)[1])
    if pred is None:
        return None, False
    try:
        return pred, bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return pred, False


def looping(text, state):
    """Update the consecutive-low-ratio counter at each STEP boundary; True once it reaches K."""
    while state["next"] <= len(text):
        end = state["next"]
        b = text[end - WIN:end].encode()
        state["run"] = state["run"] + 1 if len(zlib.compress(b, 6)) / len(b) < THR else 0
        state["next"] += STEP
        if state["run"] >= K:
            return True
    return False


async def run_one(client, url, prob, max_tokens):
    body = {
        "messages": [{"role": "user", "content": prob["question"] + "\n" + INSTRUCTION}],
        "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "presence_penalty": 1.5, "repeat_last_n": max_tokens + 1024, "stream": True,
    }
    text, chunks, status, t0 = "", 0, "finished", time.time()
    state = {"next": MIN_POS, "run": 0}
    async with client.stream("POST", url, json=body, timeout=None) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            ch = d["choices"][0] if d.get("choices") else None
            if ch is None:
                continue
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece:
                text += piece
                chunks += 1
            if ch.get("finish_reason") == "length":
                status = "cut"
            if looping(text, state):
                status = "loop"
                break   # closing the stream makes llama-server cancel the slot
    pred, ok = grade(text, prob["gold"])
    return {"id": prob["id"], "gold": prob["gold"], "pred": pred, "correct": ok, "status": status,
            "tokens": chunks, "secs": round(time.time() - t0, 1), "text": text}


async def main(a):
    probs = load_problems(a.test, a.n, a.seed)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["id"] for line in open(a.out)}
    todo = [p for p in probs if p["id"] not in done]
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    sem = asyncio.Semaphore(a.par)
    lock = asyncio.Lock()
    async with httpx.AsyncClient() as client:
        async def task(p):
            async with sem:
                r = await run_one(client, url, p, a.max_tokens)
            async with lock:
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"{r['id']}: {r['status']:8s} {r['tokens']:6d} tok  correct={r['correct']}", flush=True)
        await asyncio.gather(*[task(p) for p in todo])
    rows = [json.loads(line) for line in open(a.out)]
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    by = {s: sum(r["status"] == s for r in rows) for s in ("finished", "loop", "cut")}
    toks = sum(r["tokens"] for r in rows)
    print(f"SUMMARY {a.out}: n={n} acc={acc:.3f} {by} tokens={toks}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=8301)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--par", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16384)
    asyncio.run(main(ap.parse_args()))
