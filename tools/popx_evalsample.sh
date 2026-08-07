#!/bin/sh
# CompressARC comparison run (ledger 2026-08-07 pre-registration): the
# pre-registered N=120 ARC-1 public-eval sample (tools/evalsample120.json,
# seed 20260807), gate-winning arm configuration, per-task isolation.
# $1 = comma ckpt list, $2 = out, $3 = agree-lambda, $4 = offset, $5 = count
set -e
OFF=${4:-0}; CNT=${5:-999}
V=$(python -c "import json; ts=json.load(open('tools/evalsample120.json'))['sample']; print(' '.join(ts[$OFF:$OFF+$CNT]))")
for T in $V; do
  python tools/eval_popx.py --ckpts "$1" --tasks "$T" --steps 600 --out "$2" \
    --agree-lambda "${3:-0}" \
    || { sleep 10; python tools/eval_popx.py --ckpts "$1" --tasks "$T" --steps 600 \
    --out "$2" --agree-lambda "${3:-0}" || echo "TASK-FAILED: $T"; }
  sleep 5
done
echo "POPX EVALSAMPLE DONE: $2 offset=$OFF count=$CNT"
