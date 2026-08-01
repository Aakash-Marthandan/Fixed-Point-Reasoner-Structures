#!/bin/sh
# Ledger 2026-08-02: val-20 TTT-protocol ablation (CORPUS-side, holdout-safe).
# Grid: arms {A,B} x steps {600, 2000} x beta_TTT {0, 1e-5}, cold-start e_t
# (table mean), scored on the val tasks' pretraining-held-out queries.
# Runs remotely via dispatcher; each cell dir is resumable independently.
set -e
CKPT=runs/pretrain2/ckpt_latest.pkl
V=$(python -c "import json; print(','.join(json.load(open('runs/pretrain2/config.json'))['val_tasks']))")
echo "val tasks: $V"
for STEPS in 600 2000; do
  for BETA in 0 1e-5; do
    OUT=runs/ablate_ttt/s${STEPS}_b${BETA}
    echo "=== cell steps=$STEPS beta=$BETA -> $OUT"
    python tools/eval_dev30.py --ckpt "$CKPT" --arms A,B --tasks "$V" \
      --steps "$STEPS" --beta "$BETA" --beta-nl "$BETA" --out "$OUT"
  done
done
echo "ABLATION DONE"
