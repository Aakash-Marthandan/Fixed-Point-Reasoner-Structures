#!/bin/sh
# Eval-5 val-40 gate (ledger 2026-08-05): layered two-prior protocol,
# 600-step fits for gate ordering; the dev-30 shot runs at the registered 2000.
set -e
V=$(python -c "import json; print(','.join(json.load(open('tools/val40.json'))['val40']))")
python tools/eval_dev30.py --ckpt runs/pretrain3_d24/ckpt_latest.pkl \
  --ckpt2 runs/pretrain3_d24obj/ckpt_latest.pkl \
  --steps 600 --save-preds --tasks "$V" --out runs/eval5_gate
echo "EVAL5 GATE DONE"
