#!/bin/sh
# Factorial eval leg, lane c: d24+C17 bulk (B=48-pretrained, HBM-forced).
set -e
V=$(python -c "import json; print(','.join(json.load(open('tools/val40.json'))['val40']))")
python tools/eval_dev30.py --ckpt runs/pretrain3_d24obj/ckpt_latest.pkl \
  --arms A --steps 600 --save-preds --tasks "$V" --out runs/fact2_d24obj
echo "LANE-C FACTORIAL LEG DONE"
