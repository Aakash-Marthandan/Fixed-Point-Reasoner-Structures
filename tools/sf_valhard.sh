#!/bin/sh
# Single-fit val-hard gate (arm A): $1 = ckpt, $2 = out. The stable TPU path
# while the population harness's TPU-only segfault is debugged (2026-08-06).
set -e
V=$(python -c "import json; print(','.join(json.load(open('tools/valhard.json'))['valhard']))")
python tools/eval_dev30.py --ckpt "$1" --arms A --steps 600 --alt-size \
  --save-preds --tasks "$V" --out "$2"
echo "SF VALHARD DONE: $2"
