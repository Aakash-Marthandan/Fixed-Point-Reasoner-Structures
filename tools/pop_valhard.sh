#!/bin/sh
# Population gate on val-hard: $1 = ckpt, $2 = out. Per-task process isolation
# + inter-task settle + one retry (2026-08-06: process-churn race crashed
# heavy-batch tasks — exactly the frontier ones).
set -e
V=$(python -c "import json; print(' '.join(json.load(open('tools/valhard.json'))['valhard']))")
for T in $V; do
  python tools/eval_pop.py --ckpt "$1" --tasks "$T" --steps 600 --out "$2" \
    || { sleep 10; python tools/eval_pop.py --ckpt "$1" --tasks "$T" --steps 600 --out "$2" \
    || echo "TASK-FAILED: $T"; }
  sleep 5
done
echo "POP VALHARD DONE: $2"
