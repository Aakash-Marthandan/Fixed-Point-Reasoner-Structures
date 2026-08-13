#!/bin/bash
# Ledger: pretrain-13 WAVE-2 chain, PREMIUM-EFFICIENCY restructure
# (2026-08-13, PI directive: on-demand approved — "squeeze experimentation
# efficiently"). Changes vs chain_p13_w2.sh:
#   PHASE 1: all six pretrains BACK-TO-BACK (DP-8, zero idle chips; live
#            5-min GCS sync + post-pretrain ckpt stage per arm — kept even
#            on-demand: host failures exist).
#   PHASE 2: ONE consolidated battery phase — every (arm x set) battery +
#            the RI multi-init readout, 8-way chip-pinned in waves (the
#            interleaved design idled 6 chips for ~25 min per arm; this
#            recovers ~1.8h of pod time).
#   PHASE 3 (optional by flag file): convert-phase TTT retest — plain vs
#            --kl vs --ewc on the record substrate, 8-way. Runs only if
#            tools/.p13w3_phase3 exists at launch (decided at launch time).
#   PHASE 4: final full rescue; the LOCAL side tears down on completion
#            notification (billing tightness); DMS remains the backstop.
# Usage: bash tools/chain_p13_w3.sh VH_TASKS_CSV RG_TASKS_CSV
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
VH=$1; RG=$2
GCS=gs://qhrrn2-rescue/p13w2
COMMON="--equilibrium --d 64 --T 6 --anchor-p 0.3
        --beta-flux 3e-5 --beta-flux-nl 1e-5
        --rearc --conceptarc --orbit 4 --dp --seed 0"

pretrain_arm () {
  NAME=$1; STEPS=$2; shift 2
  echo "=== PRETRAIN $NAME ($STEPS steps) $(date -u +%H:%M) ==="
  mkdir -p "runs/pretrain13_$NAME"
  if gsutil -q cp "$GCS/${NAME}_ckpt_live.pkl" \
      "runs/pretrain13_$NAME/ckpt_latest.pkl" 2>/dev/null; then
    echo "RESUME-$NAME-FROM-GCS"
  fi
  ( while true; do sleep 300; gsutil -q cp \
      "runs/pretrain13_$NAME/ckpt_latest.pkl" \
      "$GCS/${NAME}_ckpt_live.pkl" 2>/dev/null || true; done ) &
  SYNC_PID=$!
  # shellcheck disable=SC2086
  python3 tools/pretrain.py --out "runs/pretrain13_$NAME" $COMMON \
    --steps "$STEPS" "$@" && echo "PRETRAIN-$NAME-OK"
  kill "$SYNC_PID" 2>/dev/null || true
  gsutil -q cp "runs/pretrain13_$NAME/ckpt_latest.pkl" \
    "$GCS/${NAME}_ckpt.pkl" && echo "CKPT-STAGE-$NAME-OK" \
    || echo "CKPT-STAGE-$NAME-FAILED"
}

# wave scheduler: run queued "TAG|CMD" jobs 8-way chip-pinned
run_waves () {
  local -a QUEUE=("$@")
  local i=0
  while [ $i -lt ${#QUEUE[@]} ]; do
    pids=()
    for c in 0 1 2 3 4 5 6 7; do
      [ $((i)) -ge ${#QUEUE[@]} ] && break
      IFS='|' read -r TAG CMD <<< "${QUEUE[$i]}"
      echo ">>> wave job chip$c: $TAG"
      env TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 \
          TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest \
          bash -c "$CMD" > "runs/wave_${TAG}.log" 2>&1 &
      pids+=("$!")
      i=$((i+1))
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    echo "wave done rc=$rc $(date -u +%H:%M)"
  done
}

# ---- PHASE 1: six pretrains, zero idle ----
pretrain_arm B      53333 --ri-p 0.15 --eq-coupled --flux-floors 350,75,50,15,30 --ni-sigma 0.01
pretrain_arm Dri    53333 --ri-p 0.15
pretrain_arm C53    53333
pretrain_arm Dcoup  53333 --eq-coupled
pretrain_arm Dfloor 53333 --flux-floors 350,75,50,15,30
pretrain_arm C80    80000

# ---- PHASE 2: consolidated batteries + RI readout, 8-way waves ----
echo "=== PHASE 2 batteries $(date -u +%H:%M) ==="
Q=()
for A in B Dri C53 Dcoup Dfloor C80; do
  Q+=("lad_$A|python3 tools/probe_ladder.py --ckpt runs/pretrain13_$A/ckpt_latest.pkl --tasks $VH --out runs/lad_p13$A")
  Q+=("ladrg_$A|python3 tools/probe_ladder.py --ckpt runs/pretrain13_$A/ckpt_latest.pkl --tasks $RG --out runs/ladrg_p13$A")
done
Q+=("sampDri|python3 tools/probe_sample.py --ckpt runs/pretrain13_Dri/ckpt_latest.pkl --tasks $VH --out runs/samp_p13Dri_mi --k 16 --temps 0.0 --init random")
Q+=("sampC53|python3 tools/probe_sample.py --ckpt runs/pretrain13_C53/ckpt_latest.pkl --tasks $VH --out runs/samp_p13C53_mi --k 16 --temps 0.0 --init random")
run_waves "${Q[@]}"
echo "PHASE2-OK"
tar czf /tmp/p13_batteries.tgz runs/lad_p13* runs/ladrg_p13* runs/samp_p13* \
  runs/wave_*.log 2>/dev/null || true
gsutil -q cp /tmp/p13_batteries.tgz "$GCS/batteries.tgz" \
  && echo "RESCUE-BATTERIES-OK" || echo "RESCUE-BATTERIES-FAILED"

# ---- PHASE 3 (optional): convert-phase TTT retest on the record substrate ----
if [ -f tools/.p13w3_phase3 ]; then
  echo "=== PHASE 3 TTT retest $(date -u +%H:%M) ==="
  Q3=()
  for MODE in plain kl ewc; do
    FLAG=""
    [ "$MODE" = "kl" ] && FLAG="--kl 1.0"
    [ "$MODE" = "ewc" ] && FLAG="--ewc 1.0"
    Q3+=("ttt_$MODE|python3 tools/probe_lora.py --ckpt runs/pretrain12_48c_40k/ckpt_latest.pkl --tasks $VH --out runs/ttt_${MODE}_p1248c $FLAG")
  done
  run_waves "${Q3[@]}"
  echo "PHASE3-OK"
  tar czf /tmp/p13_ttt.tgz runs/ttt_*_p1248c runs/wave_ttt*.log 2>/dev/null || true
  gsutil -q cp /tmp/p13_ttt.tgz "$GCS/ttt.tgz" \
    && echo "RESCUE-TTT-OK" || echo "RESCUE-TTT-FAILED"
fi

echo "CHAIN-P13W3-COMPLETE"
