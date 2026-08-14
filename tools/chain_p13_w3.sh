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
    # NCHIPS env (default 8): v6e-4 hosts run 4-wide waves (2026-08-14 —
    # the residue is all single-chip jobs; slice width only sets wave width)
    for c in $(seq 0 $(( ${NCHIPS:-8} - 1 ))); do
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
    # batch-across-spots mode (PI directive 2026-08-13): stage partial
    # battery results after EVERY wave — probes resume per-task from
    # results.jsonl, so a mid-phase-2 preemption costs one wave at most
    tar czf /tmp/p13_partial.tgz runs/lad_p13* runs/ladrg_p13* \
      runs/samp_p13* runs/ttt_* 2>/dev/null || true
    gsutil -q cp /tmp/p13_partial.tgz "$GCS/partial_results.tgz" \
      2>/dev/null || true
  done
}

# ---- PHASE 1: pretrains, zero idle ----
# P13_ARMS env (default all six) parametrizes the arm set — the two-pod
# split plan (v5e-8 on-demand fallback, 2026-08-13) runs {B Dri C53} and
# {Dcoup Dfloor C80} on separate pods.
ARMS=${P13_ARMS:-"B Dri C53 Dcoup Dfloor C80"}
# batch-mode resume: restore any staged partial battery results so phase-2
# probes skip completed tasks (their per-task results.jsonl resume logic)
if gsutil -q cp "$GCS/partial_results.tgz" /tmp/pr.tgz 2>/dev/null; then
  tar xzf /tmp/pr.tgz -C . 2>/dev/null && echo "RESUME-PARTIAL-RESULTS"
fi
if gsutil -q cp "$GCS/ttt_live.tgz" /tmp/tl.tgz 2>/dev/null; then
  tar xzf /tmp/tl.tgz -C . 2>/dev/null && echo "RESUME-TTT-LIVE"
fi
arm_args () {
  case $1 in
    B)      echo "53333 --ri-p 0.15 --eq-coupled --flux-floors 350,75,50,15,30 --ni-sigma 0.01" ;;
    Dri)    echo "53333 --ri-p 0.15" ;;
    C53)    echo "53333" ;;
    Dcoup)  echo "53333 --eq-coupled" ;;
    Dfloor) echo "53333 --flux-floors 350,75,50,15,30" ;;
    C80)    echo "80000" ;;
    *)      echo "UNKNOWN-ARM $1" >&2; return 1 ;;
  esac
}
for A in $ARMS; do
  # shellcheck disable=SC2046
  pretrain_arm "$A" $(arm_args "$A")
done

# ---- PHASE 2: consolidated batteries + RI readout, 8-way waves ----
echo "=== PHASE 2 batteries $(date -u +%H:%M) ==="
Q=()
for A in $ARMS; do
  Q+=("lad_$A|python3 tools/probe_ladder.py --ckpt runs/pretrain13_$A/ckpt_latest.pkl --tasks $VH --out runs/lad_p13$A")
  Q+=("ladrg_$A|python3 tools/probe_ladder.py --ckpt runs/pretrain13_$A/ckpt_latest.pkl --tasks $RG --out runs/ladrg_p13$A")
done
case " $ARMS " in *" Dri "*)
  case " $ARMS " in *" C53 "*)
    Q+=("sampDri|python3 tools/probe_sample.py --ckpt runs/pretrain13_Dri/ckpt_latest.pkl --tasks $VH --out runs/samp_p13Dri_mi --k 16 --temps 0.0 --init random")
    Q+=("sampC53|python3 tools/probe_sample.py --ckpt runs/pretrain13_C53/ckpt_latest.pkl --tasks $VH --out runs/samp_p13C53_mi --k 16 --temps 0.0 --init random")
  ;; esac
;; esac
run_waves "${Q[@]}"
echo "PHASE2-OK"
tar czf /tmp/p13_batteries.tgz runs/lad_p13* runs/ladrg_p13* runs/samp_p13* \
  runs/wave_*.log 2>/dev/null || true
gsutil -q cp /tmp/p13_batteries.tgz "$GCS/batteries.tgz" \
  && echo "RESCUE-BATTERIES-OK" || echo "RESCUE-BATTERIES-FAILED"

# ---- PHASE 3 (optional): convert-phase TTT retest on the record substrate ----
# gated on BOTH the repo flag file and P13_PHASE3=1 (two-pod split: exactly
# one pod runs it)
if [ -f tools/.p13w3_phase3 ] && [ "${P13_PHASE3:-0}" = "1" ]; then
  echo "=== PHASE 3 TTT retest $(date -u +%H:%M) ==="
  Q3=()
  for MODE in plain kl ewc; do
    FLAG=""
    [ "$MODE" = "kl" ] && FLAG="--kl 1.0"
    [ "$MODE" = "ewc" ] && FLAG="--ewc 1.0"
    Q3+=("ttt_$MODE|python3 tools/probe_lora.py --ckpt runs/pretrain12_48c_40k/ckpt_latest.pkl --tasks $VH --out runs/ttt_${MODE}_p1248c $FLAG")
  done
  # phase-3 is ONE wave — between-wave staging never fires, so a live
  # 5-min stager covers it (mid-wave preemption cost: <=5 min of TTT rows;
  # probes resume per-task from the restored results.jsonl)
  ( while true; do sleep 300; tar czf /tmp/ttt_live.tgz runs/ttt_* \
      2>/dev/null && gsutil -q cp /tmp/ttt_live.tgz "$GCS/ttt_live.tgz" \
      2>/dev/null || true; done ) &
  TSYNC=$!
  run_waves "${Q3[@]}"
  kill "$TSYNC" 2>/dev/null || true
  echo "PHASE3-OK"
  tar czf /tmp/p13_ttt.tgz runs/ttt_*_p1248c runs/wave_ttt*.log 2>/dev/null || true
  gsutil -q cp /tmp/p13_ttt.tgz "$GCS/ttt.tgz" \
    && echo "RESCUE-TTT-OK" || echo "RESCUE-TTT-FAILED"
fi

echo "CHAIN-P13W3-COMPLETE"
