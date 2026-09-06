#!/bin/bash
# Metal queue (sequential). Every job skips itself if its done.json exists. Logs: logs/gpu.log
cd /Users/aakash/Projects/HRRN/runs/field_ckpts; mkdir -p logs out; P=venv/bin/python; R=harness/run_field.py
run() { echo "=== $(date -u +%FT%TZ) START $*" ; $P $R "$@" 2>&1 | grep -v -i 'warning' ; echo "=== $(date -u +%FT%TZ) END $*"; }
run --model trm --mode cold --set scan20k --D 64 --device mps --batch 512 --out out/trm_cold_scan20k_D64
run --model eqr --mode cold --set scan20k --D 64 --noise 0.5 --device mps --batch 512 --out out/eqr_cold_scan20k_D64_n05
run --model hrm --mode cold --set scan20k --D 64 --device mps --batch 512 --out out/hrm_cold_scan20k_D64
run --model eqr --mode cold --set scan20k --D 64 --noise 0.0 --device mps --batch 512 --out out/eqr_cold_scan20k_D64_n0
run --model trm --mode draws --set sub5k --D 16 --k 8 --device mps --batch 512 --out out/trm_draws_sub5k_D16_k8
run --model eqr --mode draws --set sub5k --D 16 --k 8 --noise 0.5 --device mps --batch 512 --out out/eqr_draws_sub5k_D16_k8_n05
run --model hrm --mode draws --set sub5k --D 16 --k 8 --device mps --batch 512 --out out/hrm_draws_sub5k_D16_k8
run --model trm --mode cold --set scan20k --D 16 --dtype bf16 --device mps --batch 512 --out out/trm_cold_scan20k_D16_bf16
run --model eqr --mode draws --set sub5k --D 16 --k 8 --noise 0.0 --device mps --batch 512 --out out/eqr_draws_sub5k_D16_k8_n0
echo "=== GPU QUEUE COMPLETE $(date -u +%FT%TZ)"
