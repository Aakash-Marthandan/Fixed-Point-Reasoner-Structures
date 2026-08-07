#!/bin/sh
# Cross-bulk population gate on val-hard: $1 = comma ckpt list, $2 = out,
# $3 = agree-lambda (0 for arm X), $4 = task offset, $5 = task count
# (offset/count let two lanes split one arm). Per-task process isolation +
# settle + one retry (the 2026-08-06 stability pattern).
set -e
OFF=${4:-0}; CNT=${5:-999}
V=$(python -c "import json; ts=json.load(open('tools/valhard.json'))['valhard']; print(' '.join(ts[$OFF:$OFF+$CNT]))")
for T in $V; do
  python tools/eval_popx.py --ckpts "$1" --tasks "$T" --steps 600 --out "$2" \
    --agree-lambda "${3:-0}" \
    || { sleep 10; python tools/eval_popx.py --ckpts "$1" --tasks "$T" --steps 600 \
    --out "$2" --agree-lambda "${3:-0}" || echo "TASK-FAILED: $T"; }
  sleep 5
done
echo "POPX VALHARD DONE: $2 offset=$OFF count=$CNT"
