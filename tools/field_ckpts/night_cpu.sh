#!/bin/bash
# CPU queue (sequential; runs beside the Metal queue). Logs: logs/cpu.log
cd /Users/aakash/Projects/HRRN/runs/field_ckpts; mkdir -p logs out; P=venv/bin/python; R=harness/run_field.py
run() { echo "=== $(date -u +%FT%TZ) START $*" ; $P $R "$@" 2>&1 | grep -v -i 'warning' ; echo "=== $(date -u +%FT%TZ) END $*"; }
echo "=== $(date -u +%FT%TZ) START verify_port"; $P harness/verify_port.py 2>&1 | grep -v -i warning; echo "=== END verify_port"
for M in trm hrm; do
  run --model $M --mode dyn --set strat256 --D 64 --device cpu --batch 256 --out out/${M}_dyn_strat256_D64
  run --model $M --mode retain --set strat512 --D 8 --device cpu --batch 512 --out out/${M}_retain_strat512_D8
  run --model $M --mode jac --set strat84 --D 64 --device cpu --batch 84 --out out/${M}_jac_strat84_D64
  run --model $M --mode prefix --set strat512 --D 16 --device cpu --batch 512 --out out/${M}_prefix_strat512_D16
done
run --model eqr --mode dyn --set strat256 --D 64 --noise 0.5 --device cpu --batch 256 --out out/eqr_dyn_strat256_D64_n05
run --model eqr --mode dyn --set strat256 --D 64 --noise 0.0 --device cpu --batch 256 --out out/eqr_dyn_strat256_D64_n0
run --model eqr --mode retain --set strat512 --D 8 --noise 0.0 --device cpu --batch 512 --out out/eqr_retain_strat512_D8_n0
run --model eqr --mode retain --set strat512 --D 8 --noise 0.5 --device cpu --batch 512 --out out/eqr_retain_strat512_D8_n05
run --model eqr --mode jac --set strat84 --D 64 --noise 0.0 --device cpu --batch 84 --out out/eqr_jac_strat84_D64_n0
run --model eqr --mode prefix --set strat512 --D 16 --noise 0.5 --device cpu --batch 512 --out out/eqr_prefix_strat512_D16_n05
run --model eqr --mode train --D 16 --noise 0.5 --device cpu --batch 500 --out out/eqr_train1k_D16_n05
for M in trm hrm; do run --model $M --mode sym --set strat512 --D 16 --device cpu --batch 512 --out out/${M}_sym_strat512_D16; done
run --model eqr --mode sym --set strat512 --D 16 --noise 0.5 --device cpu --batch 512 --out out/eqr_sym_strat512_D16_n05
run --model trm --mode cold --set strat512 --D 16 --device cpu --batch 512 --out out/trm_cold_strat512_D16
run --model hrm --mode cold --set strat512 --D 16 --device cpu --batch 512 --out out/hrm_cold_strat512_D16
run --model eqr --mode cold --set strat512 --D 16 --noise 0.5 --device cpu --batch 512 --out out/eqr_cold_strat512_D16_n05
echo "=== CPU QUEUE COMPLETE $(date -u +%FT%TZ)"
