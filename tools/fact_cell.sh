#!/bin/sh
# One factorial cell: $1 = checkpoint path, $2 = out dir, $3 = steps (default 600).
# Protocol (wave-2, 2026-08-05): val-40, arm A, alt-size attempts.
set -e
STEPS=${3:-600}
V=$(python -c "import json; print(','.join(json.load(open('tools/val40.json'))['val40']))")
python tools/eval_dev30.py --ckpt "$1" --arms A --steps "$STEPS" --alt-size --save-preds \
  --tasks "$V" --out "$2"
echo "FACT CELL DONE: $2"
