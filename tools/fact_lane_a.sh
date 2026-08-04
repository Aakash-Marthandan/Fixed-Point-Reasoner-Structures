#!/bin/sh
# Factorial eval leg, lane a: d16+C17 bulks (final 20k + pre-drop 10k).
set -e
V=$(python -c "import json; print(','.join(json.load(open('tools/val40.json'))['val40']))")
python tools/eval_dev30.py --ckpt runs/pretrain3_d16obj/ckpt_latest.pkl \
  --arms A --steps 600 --save-preds --tasks "$V" --out runs/fact2_d16obj20k
python tools/eval_dev30.py --ckpt runs/pretrain3_d16obj/ckpt_010000.pkl \
  --arms A --steps 600 --save-preds --tasks "$V" --out runs/fact2_d16obj10k
echo "LANE-A FACTORIAL LEG DONE"
