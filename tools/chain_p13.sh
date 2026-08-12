#!/bin/bash
# Ledger: pretrain-13 pilot chain (registration 2026-08-12) — three arms
# SEQUENTIAL on the v6e-8 pod: C (scale-pure d64/T6 @ 53,333, priced knee)
# -> A (+ RI rows + coupled a1/a2 + free-bits floors) -> B (A + NI).
# Per arm: DP-8 pretrain, then lad (val-hard) and ladrg (rg gate) batteries
# CONCURRENTLY on two pinned chips. ladrt + samp = wave 2 (registered).
# Sentinels per stage; set -e halts the chain on any failure so a broken
# arm never silently poisons the next.
# Usage (remote, via dispatcher run --detach):
#   bash tools/chain_p13.sh VH_TASKS_CSV RG_TASKS_CSV
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
VH_TASKS=$1
RG_TASKS=$2
COMMON="--equilibrium --d 64 --T 6 --steps 53333 --anchor-p 0.3
        --beta-flux 3e-5 --beta-flux-nl 1e-5
        --rearc --conceptarc --orbit 4 --dp --seed 0"
PIN0="env TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0"
PIN1="env TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=1"

run_arm () {
  NAME=$1; shift
  echo "=== ARM $NAME pretrain $(date -u +%H:%M) ==="
  # shellcheck disable=SC2086
  python3 tools/pretrain.py --out "runs/pretrain13_$NAME" $COMMON "$@" \
    && echo "ARM-$NAME-PRETRAIN-OK"
  echo "=== ARM $NAME batteries (2 pinned chips) $(date -u +%H:%M) ==="
  $PIN0 python3 tools/probe_ladder.py \
    --ckpt "runs/pretrain13_$NAME/ckpt_latest.pkl" --tasks "$VH_TASKS" \
    --out "runs/lad_p13$NAME" > "runs/lad_p13$NAME.log" 2>&1 &
  P_LAD=$!
  $PIN1 python3 tools/probe_ladder.py \
    --ckpt "runs/pretrain13_$NAME/ckpt_latest.pkl" --tasks "$RG_TASKS" \
    --out "runs/ladrg_p13$NAME" > "runs/ladrg_p13$NAME.log" 2>&1 &
  P_RG=$!
  wait $P_LAD && echo "ARM-$NAME-LAD-OK"
  wait $P_RG && echo "ARM-$NAME-LADRG-OK"
}

run_arm C
run_arm A --ri-p 0.15 --eq-coupled --flux-floors 350,75,50,15,30
run_arm B --ri-p 0.15 --eq-coupled --flux-floors 350,75,50,15,30 --ni-sigma 0.01
echo "CHAIN-P13-COMPLETE"
