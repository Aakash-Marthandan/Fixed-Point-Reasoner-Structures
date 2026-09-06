#!/bin/bash
cd /Users/aakash/Projects/HRRN/runs/field_ckpts; P=venv/bin/python; R=harness/run_field.py
while kill -0 75992 2>/dev/null; do sleep 120; done
run() { echo "=== $(date -u +%FT%TZ) START $*" ; $P $R "$@" 2>&1 | grep --line-buffered -v -i 'warning' ; echo "=== $(date -u +%FT%TZ) END $*"; }
run --model trmc --mode cold --set scan20k --D 64 --device mps --batch 512 --out out/trmc_cold_scan20k_D64
run --model trmc --mode draws --set sub5k --D 16 --k 8 --device mps --batch 512 --out out/trmc_draws_sub5k_D16_k8
echo "=== GPU QUEUE 3 COMPLETE $(date -u +%FT%TZ)"
