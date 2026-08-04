#!/bin/sh
# Factorial eval leg, lane b: base d16 (pretrain-2, uploaded) + d24 bulk.
set -e
V=$(python -c "import json; print(','.join(json.load(open('tools/val40.json'))['val40']))")
python tools/eval_dev30.py --ckpt runs/pretrain2/ckpt_latest.pkl \
  --arms A,B --steps 2000 --save-preds --tasks "$V" --out runs/fact_base16
python tools/eval_dev30.py --ckpt runs/pretrain3_d24/ckpt_latest.pkl \
  --arms A,B --steps 2000 --save-preds --tasks "$V" --out runs/fact_d24
echo "LANE-B FACTORIAL LEG DONE"
