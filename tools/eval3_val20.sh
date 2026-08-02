#!/bin/sh
# Ledger 2026-08-02 (eval-3 registration): corpus-side validation of the new
# attempt rule (earliest-val-exact + MDL) and the B2 arm before the ONE
# dev-30 shot. Val-20 tasks, s2000/b0 — the winning cell from the budget grid.
set -e
CKPT=runs/pretrain2/ckpt_latest.pkl
V=$(python -c "import json; print(','.join(json.load(open('runs/pretrain2/config.json'))['val_tasks']))")
echo "val tasks: $V"
python tools/eval_dev30.py --ckpt "$CKPT" --arms A,B,B2 --tasks "$V" \
  --steps 2000 --out runs/ablate_sel
echo "VAL20 SEL CHAIN DONE"
