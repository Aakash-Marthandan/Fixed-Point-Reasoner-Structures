#!/bin/bash
# The PI-approved extended Mac-only run (2026-09-08 23:30 IST): the released TRM ARC-1 checkpoint's FULL-augmentation vote on the
# 40-task subset (their pass@2 protocol; ≈ 9.5 h at 0.9 s/example), sequenced behind the in-flight retain -> draws chain (PID wait).
cd /Users/aakash/Projects/HRRN/runs/field_ckpts/harness || exit 1
while kill -0 39421 2>/dev/null; do sleep 30; done      # the retain->draws waiter (pid-based, never a pattern)
echo "VOTE-START $(date -u +%FT%TZ)"
../venv/bin/python eval_arc_theirs.py --select aug --tasks 40 --subset-seed 20260908 --n-aug 0 --batch 32 --dtype bf16 --device mps --partial-every 600 --out arc_out/vote40_bf16 > arc_logs/vote40_bf16.log 2>&1
echo "VOTE-DONE rc=$? $(date -u +%FT%TZ)"
