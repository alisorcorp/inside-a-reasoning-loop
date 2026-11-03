"""Calibrate a compression-based loop detector on the authors' released generations.

A sample "looped" if it never closed its thinking (no </think>) - in their eval that means it ran to 32,768 tokens.
For each threshold we report: how many looping samples the detector flags, how early (in chars), and how many
finished samples it would have wrongly killed.

    python calibrate_loop.py <exp8_math dir>
"""
import glob
import gzip
import json
import sys
import zlib

WIN = 4000      # chars examined
STEP = 500      # check every STEP chars
MIN_POS = 6000  # never flag before this many chars


def ratio(s):
    b = s.encode()
    return len(zlib.compress(b, 6)) / max(1, len(b))


def first_flag(text, thr, k=1):
    """Position where the last WIN chars have compressed below thr at k consecutive checks, or None."""
    run = 0
    for end in range(MIN_POS, len(text) + 1, STEP):
        run = run + 1 if ratio(text[end - WIN:end]) < thr else 0
        if run >= k:
            return end
    return None


def load(d, run, bench):
    out = []
    for f in sorted(glob.glob(f"{d}/eval/{run}/parts/{bench}_s*/{bench}_first_samples.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            out.append(r["text"])
    return out


def main(d):
    texts = []
    for run in ("unedited", "p50c"):
        for bench in ("gsm8k", "math500"):
            texts += load(d, run, bench)
    looped = [t for t in texts if "</think>" not in t]
    done = [t for t in texts if "</think>" in t]
    print(f"{len(texts)} samples: {len(looped)} never closed thinking, {len(done)} closed")
    for thr, k in ((0.06, 1), (0.06, 4), (0.05, 4), (0.08, 4), (0.08, 8)):
        flags = [first_flag(t, thr, k) for t in looped]
        hit = sorted(f for f in flags if f)
        fp = sum(1 for t in done if first_flag(t.split("</think>")[0], thr, k))
        med = hit[len(hit) // 2] if hit else None
        print(f"thr {thr:.2f} x{k}: flags {len(hit)}/{len(looped)} looped (median at {med} chars), "
              f"false alarms {fp}/{len(done)} closed")


if __name__ == "__main__":
    main(sys.argv[1])
