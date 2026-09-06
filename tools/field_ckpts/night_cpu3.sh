#!/bin/bash
cd /Users/aakash/Projects/HRRN/runs/field_ckpts; P=venv/bin/python; R=harness/run_field.py
run() { echo "=== $(date -u +%FT%TZ) START $*" ; $P $R "$@" 2>&1 | grep --line-buffered -v -i 'warning' ; echo "=== $(date -u +%FT%TZ) END $*"; }
M=trmc
run --model $M --mode dyn --set strat256 --D 64 --device cpu --batch 256 --out out/${M}_dyn_strat256_D64
run --model $M --mode retain --set strat512 --D 8 --device cpu --batch 512 --out out/${M}_retain_strat512_D8
run --model $M --mode jac --set strat84 --D 64 --device cpu --batch 84 --out out/${M}_jac_strat84_D64
run --model $M --mode cold --set strat512 --D 16 --device cpu --batch 512 --out out/${M}_cold_strat512_D16
run --model $M --mode prefix --set strat512 --D 16 --device cpu --batch 512 --out out/${M}_prefix_strat512_D16
run --model $M --mode sym --set strat512 --D 16 --device cpu --batch 512 --out out/${M}_sym_strat512_D16
echo "=== CPU QUEUE 3 COMPLETE $(date -u +%FT%TZ)"
