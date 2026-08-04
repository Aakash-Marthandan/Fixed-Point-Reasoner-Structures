#!/bin/sh
# Ledger 2026-08-02 (theory session): factorial pretrains, val-40 holdout.
# Cells: {d16+C17} / {d24} / {d24+C17}; base d16 = pretrain-2 (known).
# Sequential on one chip; each dir resumable independently.
set -e
V=tools/val40.json
python tools/pretrain.py --d 16 --obj --val-ids-file "$V" --n-val 40 --out runs/pretrain3_d16obj
python tools/pretrain.py --d 24       --val-ids-file "$V" --n-val 40 --out runs/pretrain3_d24
python tools/pretrain.py --d 24 --obj --val-ids-file "$V" --n-val 40 --out runs/pretrain3_d24obj
echo "FACTORIAL PRETRAINS DONE"
