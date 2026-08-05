#!/bin/sh
# Population gate on val-hard: $1 = ckpt, $2 = out.
# PER-TASK PROCESS ISOLATION (2026-08-06): a libtpu segfault at task
# boundaries reproduces on TPU only (clean on CPU at identical params);
# rather than chase the runtime, each task gets its own process — a crash
# costs ONE task (resumable rows make retries free) and the loop continues.
IDS=$(python -c "import json; print(' '.join(json.load(open('tools/valhard.json'))['valhard']))")
for t in $IDS; do
  python tools/eval_pop.py --ckpt "$1" --tasks "$t" --steps 600 --out "$2" \
    || echo "TASK-FAILED: $t"
done
echo "POP VALHARD DONE: $2"
