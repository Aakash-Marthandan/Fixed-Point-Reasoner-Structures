#!/bin/sh
# EVAL-5: the ONE dev-30 shot (ledger 2026-08-05 registration).
# Layered two-prior protocol at the registered 2000-step fits, seed 0.
set -e
python tools/eval_dev30.py --ckpt runs/pretrain3_d24/ckpt_latest.pkl \
  --ckpt2 runs/pretrain3_d24obj/ckpt_latest.pkl \
  --steps 2000 --save-preds --out runs/eval5
echo "EVAL5 DEV30 DONE"
