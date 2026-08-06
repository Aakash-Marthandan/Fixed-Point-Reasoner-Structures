#!/bin/sh
# Population gate on val-hard: $1 = ckpt, $2 = out. Per-task PROCESS ISOLATION
# (2026-08-06: multi-task processes die around task ~19-27 — graph
# accumulation; the resumable results file makes per-task processes free).
set -e
V=$(python -c "import json; print(' '.join(json.load(open('tools/valhard.json'))['valhard']))")
for T in $V; do
  python tools/eval_pop.py --ckpt "$1" --tasks "$T" --steps 600 --out "$2" \
    || echo "TASK-FAILED: $T"
done
echo "POP VALHARD DONE: $2"
