"""Summarize the GSM8K eval runs in a results folder, with paired counts on the shared problems.

    python summarize_eval.py <results dir>
"""
import json
import statistics
import sys

RUNS = ("base-bf16", "a1-bf16", "p50c-bf16", "base-Q4_K_M", "p50c-Q4_K_M")
PAIRS = (("base-bf16", "a1-bf16"), ("a1-bf16", "p50c-bf16"), ("base-bf16", "p50c-bf16"),
         ("p50c-bf16", "p50c-Q4_K_M"), ("base-bf16", "base-Q4_K_M"))


def main(d):
    R = {}
    for n in RUNS:
        rows = [json.loads(line) for line in open(f"{d}/{n}.jsonl")]
        R[n] = {r["id"]: r for r in rows}
    ids = sorted(R[RUNS[0]])
    for n, r in R.items():
        rows = list(r.values())
        fin = [x for x in rows if x["status"] == "finished"]
        med = statistics.median(x["tokens"] for x in fin) if fin else None
        print(f"{n:12s} correct {sum(x['correct'] for x in rows)}/{len(rows)}  finished {len(fin)}  "
              f"correct when finished {sum(x['correct'] for x in fin)}/{len(fin)}  median tokens when finished {med}  "
              f"loops {sum(x['status'] == 'loop' for x in rows)}  cut at 16k {sum(x['status'] == 'cut' for x in rows)}")
    for a, b in PAIRS:
        A, B = R[a], R[b]
        print(f"{b} right where {a} wrong: {sum(B[i]['correct'] and not A[i]['correct'] for i in ids)};  "
              f"{a} right where {b} wrong: {sum(A[i]['correct'] and not B[i]['correct'] for i in ids)}")


if __name__ == "__main__":
    main(sys.argv[1])
