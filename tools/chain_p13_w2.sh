#!/bin/bash
# Ledger: pretrain-13 WAVE-2 chain (registration 2026-08-12, amended after
# the pod preemption): arm B rerun + FULL A-decomposition (single-toggle
# cells: RI-only / coupling-only / floors-only vs a C53 rerun) + the 80k
# steps-law decider + the RI multi-init readout (Dri vs C53, matched scale).
# PER-ARM DURABILITY (the preemption lesson): after each arm's batteries,
# its checkpoint + battery dirs are staged to GCS — a preemption never eats
# more than the arm in flight. Rescue failure is non-fatal (sentinel says).
# Usage (remote): bash tools/chain_p13_w2.sh VH_TASKS_CSV RG_TASKS_CSV
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
VH=$1; RG=$2
GCS=gs://qhrrn2-rescue/p13w2
COMMON="--equilibrium --d 64 --T 6 --anchor-p 0.3
        --beta-flux 3e-5 --beta-flux-nl 1e-5
        --rearc --conceptarc --orbit 4 --dp --seed 0"
PIN0="env TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0"
PIN1="env TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=1"

rescue_arm () {
  NAME=$1
  tar czf "/tmp/p13_$NAME.tgz" "runs/pretrain13_$NAME" \
    "runs/lad_p13$NAME" "runs/ladrg_p13$NAME" 2>/dev/null || true
  if gsutil -q cp "/tmp/p13_$NAME.tgz" "$GCS/$NAME.tgz"; then
    echo "RESCUE-$NAME-OK"
  else
    echo "RESCUE-$NAME-FAILED"   # non-fatal: local pulls remain possible
  fi
}

run_arm () {
  NAME=$1; STEPS=$2; shift 2
  echo "=== ARM $NAME pretrain ($STEPS steps) $(date -u +%H:%M) ==="
  # shellcheck disable=SC2086
  python3 tools/pretrain.py --out "runs/pretrain13_$NAME" $COMMON \
    --steps "$STEPS" "$@" && echo "ARM-$NAME-PRETRAIN-OK"
  # ckpt staged IMMEDIATELY (2nd preemption of the night beat arm B to the
  # post-battery rescue): the expensive artifact is durable before batteries
  gsutil -q cp "runs/pretrain13_$NAME/ckpt_latest.pkl" \
    "$GCS/${NAME}_ckpt.pkl" && echo "CKPT-STAGE-$NAME-OK" \
    || echo "CKPT-STAGE-$NAME-FAILED"
  echo "=== ARM $NAME batteries $(date -u +%H:%M) ==="
  $PIN0 python3 tools/probe_ladder.py \
    --ckpt "runs/pretrain13_$NAME/ckpt_latest.pkl" --tasks "$VH" \
    --out "runs/lad_p13$NAME" > "runs/lad_p13$NAME.log" 2>&1 &
  P1=$!
  $PIN1 python3 tools/probe_ladder.py \
    --ckpt "runs/pretrain13_$NAME/ckpt_latest.pkl" --tasks "$RG" \
    --out "runs/ladrg_p13$NAME" > "runs/ladrg_p13$NAME.log" 2>&1 &
  P2=$!
  wait $P1 && echo "ARM-$NAME-LAD-OK"
  wait $P2 && echo "ARM-$NAME-LADRG-OK"
  rescue_arm "$NAME"
}

run_arm B      53333 --ri-p 0.15 --eq-coupled --flux-floors 350,75,50,15,30 --ni-sigma 0.01
run_arm Dri    53333 --ri-p 0.15
run_arm C53    53333
run_arm Dcoup  53333 --eq-coupled
run_arm Dfloor 53333 --flux-floors 350,75,50,15,30
run_arm C80    80000

echo "=== RI multi-init readout (Dri vs C53, matched d64) $(date -u +%H:%M) ==="
$PIN0 python3 tools/probe_sample.py \
  --ckpt runs/pretrain13_Dri/ckpt_latest.pkl --tasks "$VH" \
  --out runs/samp_p13Dri_mi --k 16 --temps 0.0 --init random \
  > runs/samp_Dri.log 2>&1 &
P1=$!
$PIN1 python3 tools/probe_sample.py \
  --ckpt runs/pretrain13_C53/ckpt_latest.pkl --tasks "$VH" \
  --out runs/samp_p13C53_mi --k 16 --temps 0.0 --init random \
  > runs/samp_C53.log 2>&1 &
P2=$!
wait $P1 && wait $P2 && echo "SAMP-RI-READOUT-OK"
tar czf /tmp/p13_samp.tgz runs/samp_p13Dri_mi runs/samp_p13C53_mi \
  runs/samp_*.log 2>/dev/null || true
gsutil -q cp /tmp/p13_samp.tgz "$GCS/samp.tgz" \
  && echo "RESCUE-SAMP-OK" || echo "RESCUE-SAMP-FAILED"
echo "CHAIN-P13W2-COMPLETE"
