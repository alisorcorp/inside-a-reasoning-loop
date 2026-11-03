#!/bin/sh
# Evaluate each named GGUF in turn on chip: start its server, run the GSM8K client, stop the server.
#   campaign.sh <name> [<name> ...]     name = gguf/qwen35-0.8b-<name>.gguf, results in results/<name>.jsonl
set -u
W=$HOME/wsp
PORT=8301
SLOTS=8
CTX=17408
N=${N:-40}
mkdir -p "$W/results"
for name in "$@"; do
  gguf=$W/gguf/qwen35-0.8b-$name.gguf
  [ -f "$gguf" ] || { echo "missing $gguf"; continue; }
  systemctl --user stop wsp-serve 2>/dev/null
  systemd-run --user --unit wsp-serve --collect "$W/run/serve.sh" "$gguf" $PORT $SLOTS $CTX >/dev/null
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
  echo "== $name $(date +%H:%M:%S)"
  "$W/venv/bin/python" -I "$W/mine/eval_gsm8k.py" "$W/data/gsm8k_test.jsonl" "$W/results/$name.jsonl" \
    --port $PORT --n "$N" --par $SLOTS --max-tokens 16384 2>&1 | tail -1
  systemctl --user stop wsp-serve
done
echo "== done $(date +%H:%M:%S)"
