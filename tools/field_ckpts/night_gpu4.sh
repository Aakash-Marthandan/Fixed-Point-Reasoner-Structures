#!/bin/bash
cd /Users/aakash/Projects/HRRN/runs/field_ckpts; P=venv/bin/python; R=harness/run_field.py
while kill -0 38011 2>/dev/null; do sleep 120; done
run() { echo "=== $(date -u +%FT%TZ) START $*" ; $P $R "$@" 2>&1 | grep --line-buffered -v -i 'warning' ; echo "=== $(date -u +%FT%TZ) END $*"; }
for M in hrm trm trmc; do run --model $M --mode initrad --set strat512 --D 16 --device mps --batch 512 --out out/${M}_initrad_strat512_D16; done
run --model eqr --mode initrad --set strat512 --D 16 --noise 0.0 --device mps --batch 512 --out out/eqr_initrad_strat512_D16_n0
run --model hrm --mode draws --set sub5k --limit 1000 --D 64 --k 2 --device mps --batch 512 --out out/hrm_draws_sub1k_D64_k2
echo "=== GPU QUEUE 4 COMPLETE $(date -u +%FT%TZ)"
