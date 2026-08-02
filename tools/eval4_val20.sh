#!/bin/sh
# Ledger 2026-08-02 (eval-4 registration): corpus-side gate for inference-side
# D4 orbit voting (attempt 1). Fitting identical to eval-3; prediction only.
set -e
CKPT=runs/pretrain2/ckpt_latest.pkl
V=$(python -c "import json; print(','.join(json.load(open('runs/pretrain2/config.json'))['val_tasks']))")
echo "val tasks: $V"
python tools/eval_dev30.py --ckpt "$CKPT" --arms A,B --tasks "$V" \
  --steps 2000 --vote --out runs/ablate_vote
echo "VAL20 VOTE CHAIN DONE"
