#!/bin/bash
# Ledger: SPRINT S2 (2026-08-21 course correction #3) — Sudoku-Extreme on ONE
# v6e-8 pod, pod.sh-compatible (sourced values: R_TAG R_D R_STEPS R0_ARMS;
# extra env SX_NPZ SX_AUG SX_T_LIST SX_K_INIT; the trailing VH RG RB RT args
# that pod.sh appends are ignored). THE SERIES AS AN ABLATION LADDER — each
# arm is one mechanism of HRM->TRM->EqR->FPRM on our substrate, read through
# the instruments:
#   S0 base   equilibrium core, anchors p.3, knee-priced, T6   (the control)
#   S1 +RI .15        EqR's randomized-init training rows (H-37 lever)
#   S2 +NI .01        EqR's per-step training noise
#   S3 T12            training depth (remat)  [2026-08-21: T12 without remat OOMs HBM at
#                     33.0G>31.2G — P11-EXT lesson; S4 already carried --remat]
#   S4 T24            training depth (remat)
#   S5 plain          beta=0 (Law 4 on CSP; the throat of a solving substrate)
#   S6 +digit-aug     explicit digit permutation (S9 covers it: prediction = no change)
#   S7 RI+NI+T12      the combined EqR recipe
# Three phases, each a chip-pinned wave of 8 parallel single-chip jobs:
#   PHASE1 pretrain (STRICT: 1k base x SX_AUG group-augmented copies)
#   PHASE2 eval: stratified 512 @ t in SX_T_STRAT, k=SX_K_INIT (instrument set)
#                + FULL test @ t in SX_T_FULL, k=0 (the benchmark number)
#   PHASE3 probe_sudoku instrument suite on the stratified 512 @ t=64
# Durability: per-arm live ckpt sync (5 min) + .done markers + per-phase GCS
# staging; every step resumes. Sentinel CHAIN-SPORT2-COMPLETE; final object
# $FINAL_OBJ in $GCS (pod.sh reads both).
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
R_TAG=${R_TAG:-sport2}; R_D=${R_D:-16}; R_STEPS=${R_STEPS:-20000}
ARMS=${R0_ARMS:-"S0 S1 S2 S3 S4 S5 S6 S7"}
GCS=${GCS:-gs://qhrrn2-rescue/sport2}; FINAL_OBJ=${FINAL_OBJ:-sport2_final.tgz}
SX_NPZ=${SX_NPZ:-sudoku_extreme_seed0.npz}; SX_AUG=${SX_AUG:-100}
SX_T_STRAT=${SX_T_STRAT:-"6 64 256"}; SX_T_FULL=${SX_T_FULL:-"6 64"}; SX_K_INIT=${SX_K_INIT:-16}; SX_STRAT=${SX_STRAT:-512}
# cost rule (2026-08-21 sizing): a full-test pass costs ~t x 423k row-steps — t=256 is
# ~3-6 chip-hours per arm, so wave 1 runs the FULL test at t in SX_T_FULL (EqR's D=64
# axis) and t=256 on the stratified instrument set only; full t=256 = best arm, wave 2.
NPZ=data/sudoku_extreme/$SX_NPZ
mkdir -p runs data/sudoku_extreme
echo "=== SPORT2 START $(date -u +%FT%TZ) arms=$ARMS d=$R_D steps=$R_STEPS aug=$SX_AUG npz=$SX_NPZ t_full=[$SX_T_FULL] t_strat=[$SX_T_STRAT] k=$SX_K_INIT ==="
[ -f "$NPZ" ] || gsutil -q cp "$GCS/$SX_NPZ" "$NPZ" || { echo "NPZ-MISSING $SX_NPZ"; exit 2; }

pin () {  # pin CHIP -- cmd...   (shard_run.sh recipe, validated by probe_chips.py)
  local c=$1; shift
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c \
  JAX_DEFAULT_MATMUL_PRECISION=highest "$@"
}
COMMON="--equilibrium --d $R_D --steps $R_STEPS --anchor-p 0.3 --sudoku-extreme $NPZ --sudoku-aug $SX_AUG --n-val 64 --seed 0"
arm_flags () {
  case $1 in
    S0) echo "--T 6  --beta-flux 3e-5 --beta-flux-nl 1e-5";;
    S1) echo "--T 6  --beta-flux 3e-5 --beta-flux-nl 1e-5 --ri-p 0.15";;
    S2) echo "--T 6  --beta-flux 3e-5 --beta-flux-nl 1e-5 --ni-sigma 0.01";;
    S3) echo "--T 12 --beta-flux 3e-5 --beta-flux-nl 1e-5 --remat";;
    S4) echo "--T 24 --beta-flux 3e-5 --beta-flux-nl 1e-5 --remat";;
    S5) echo "--T 6  --beta-flux 0 --beta-flux-nl 0";;
    S6) echo "--T 6  --beta-flux 3e-5 --beta-flux-nl 1e-5 --sudoku-digit-aug";;
    S7) echo "--T 12 --beta-flux 3e-5 --beta-flux-nl 1e-5 --ri-p 0.15 --ni-sigma 0.01 --remat";;
    S*s1) echo "$(arm_flags "${1%s1}") --seed 1";;   # seed-1 replicate of any arm
    S*s2) echo "$(arm_flags "${1%s2}") --seed 2";;
    *) echo "UNKNOWN-ARM $1" >&2; return 1;;
  esac
}
arm_T () { arm_flags "$1" | grep -oE -- "--T [0-9]+" | awk '{print $2}'; }

# ---------- PHASE 1: pretrain wave(s) ----------
echo "PHASE1: pretrain $(date -u +%H:%M)"
set -- $ARMS
while [ $# -gt 0 ]; do
  wave=("${@:1:8}"); shift $(( $# < 8 ? $# : 8 ))
  pids=(); c=0
  for arm in "${wave[@]}"; do
    D=runs/pretrain${R_TAG}_$arm; mkdir -p "$D"
    if [ -f "$D/.done" ]; then echo "SKIP-$arm (done)"; continue; fi
    gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" 2>/dev/null && { touch "$D/.done"; echo "SKIP-$arm (GCS complete)"; continue; }
    gsutil -q cp "$GCS/${arm}_ckpt_live.pkl" "$D/ckpt_latest.pkl" 2>/dev/null && echo "RESUME-$arm from live ckpt"
    echo "=== PRETRAIN $arm $(date -u +%H:%M) === chip $c flags: $(arm_flags "$arm")"
    ( ( while true; do sleep 300; gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt_live.pkl" 2>/dev/null || true; done ) & SY=$!
      # shellcheck disable=SC2086
      pin $c python3 tools/pretrain.py --out "$D" $COMMON $(arm_flags "$arm") > "runs/wave_pre_$arm.log" 2>&1 \
        && { touch "$D/.done"; gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt.pkl"; gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics.jsonl"; echo "PRETRAIN-$arm-OK $(date -u +%H:%M)"; } \
        || echo "PRETRAIN-$arm-FAILED rc=$? (see runs/wave_pre_$arm.log)"
      kill $SY 2>/dev/null || true ) &
    pids+=($!); c=$((c+1))
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done   # empty-array-safe under set -u (bash 3.2)
  echo "wave done $(date -u +%H:%M)"
done
echo "PHASE1-OK $(date -u +%H:%M)"

# ---------- PHASE 2: evals (instrument subsample + full test) ----------
echo "PHASE2: eval $(date -u +%H:%M)"
set -- $ARMS
while [ $# -gt 0 ]; do
  wave=("${@:1:8}"); shift $(( $# < 8 ? $# : 8 ))
  pids=(); c=0
  for arm in "${wave[@]}"; do
    D=runs/pretrain${R_TAG}_$arm; [ -f "$D/.done" ] || { echo "EVAL-SKIP-$arm (no ckpt)"; continue; }
    O=runs/sxeval_p${R_TAG}$arm; mkdir -p "$O"
    [ -n "$(ls "$O" 2>/dev/null)" ] || { gsutil -q cp "$GCS/${arm}_eval.tgz" "/tmp/sxeval_$arm.tgz" 2>/dev/null && tar xzf "/tmp/sxeval_$arm.tgz" && echo "RESTORE-$arm evals from GCS"; }
    ( for t in $SX_T_STRAT; do
        [ -f "$O/strat_t$t/summary_all.json" ] || pin $c python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" \
            --split test --stratified "$SX_STRAT" --t-total "$t" --k-init "$SX_K_INIT" --out "$O/strat_t$t" > "runs/wave_ev_${arm}_s$t.log" 2>&1
      done
      for t in $SX_T_FULL; do
        [ -f "$O/full_t$t/summary_all.json" ] || pin $c python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" \
            --split test --t-total "$t" --k-init 0 --out "$O/full_t$t" > "runs/wave_ev_${arm}_f$t.log" 2>&1
      done
      tar czf "/tmp/sxeval_$arm.tgz" "$O" && gsutil -q cp "/tmp/sxeval_$arm.tgz" "$GCS/${arm}_eval.tgz" && echo "EVAL-$arm-OK $(date -u +%H:%M)" ) &
    pids+=($!); c=$((c+1))
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done   # empty-array-safe under set -u (bash 3.2)
done
echo "PHASE2-OK $(date -u +%H:%M)"

# ---------- PHASE 3: instrument suite on the stratified subsample ----------
echo "PHASE3: probes $(date -u +%H:%M)"
set -- $ARMS
while [ $# -gt 0 ]; do
  wave=("${@:1:8}"); shift $(( $# < 8 ? $# : 8 ))
  pids=(); c=0
  for arm in "${wave[@]}"; do
    D=runs/pretrain${R_TAG}_$arm; [ -f "$D/.done" ] || continue
    O=runs/sudprobe_p${R_TAG}$arm
    [ -f "$O/results.jsonl" ] || { gsutil -q cp "$GCS/${arm}_probe.tgz" "/tmp/sxprobe_$arm.tgz" 2>/dev/null && tar xzf "/tmp/sxprobe_$arm.tgz" && echo "RESTORE-$arm probe from GCS"; }
    ( pin $c python3 tools/probe_sudoku.py --ckpt "$D/ckpt_latest.pkl" --pairs-file "$NPZ" --split test \
        --stratified "$SX_STRAT" --t-total 64 --k-init 16 --out "$O" > "runs/wave_pr_$arm.log" 2>&1 \
        && { tar czf "/tmp/sxprobe_$arm.tgz" "$O" && gsutil -q cp "/tmp/sxprobe_$arm.tgz" "$GCS/${arm}_probe.tgz"; echo "PROBE-$arm-OK $(date -u +%H:%M)"; } \
        || echo "PROBE-$arm-FAILED" ) &
    pids+=($!); c=$((c+1))
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done   # empty-array-safe under set -u (bash 3.2)
done
echo "PHASE3-OK $(date -u +%H:%M)"

# ---------- completion GUARD (2026-08-21 incident: a `pod.sh stop` killed the workers but this
# script raced through empty phases and emitted the sentinel + final object -> the supervisor
# read COMPLETE and tore the pod down mid-campaign). The sentinel is earned only when EVERY arm
# has its pretrain .done AND its full-test summaries AND its probe rows; otherwise exit 1 (the
# supervisor relaunches -> resume) and NO final object is written. ----------
missing=""
for arm in $ARMS; do
  D=runs/pretrain${R_TAG}_$arm; O=runs/sxeval_p${R_TAG}$arm; Pq=runs/sudprobe_p${R_TAG}$arm
  [ -f "$D/.done" ] || missing="$missing $arm:pretrain"
  for t in $SX_T_FULL; do [ -f "$O/full_t$t/summary_all.json" ] || missing="$missing $arm:full_t$t"; done
  [ -s "$Pq/results.jsonl" ] || missing="$missing $arm:probe"
done
if [ -n "$missing" ]; then
  echo "CHAIN-SPORT2-INCOMPLETE missing:$missing $(date -u +%FT%TZ)"; exit 1
fi
# ---------- final object + sentinel ----------
tar czf /tmp/$FINAL_OBJ runs/pretrain${R_TAG}_*/ckpt_latest.pkl runs/pretrain${R_TAG}_*/metrics.jsonl runs/*_p${R_TAG}* runs/wave_*.log 2>/dev/null
gsutil -q cp /tmp/$FINAL_OBJ "$GCS/$FINAL_OBJ" && echo "RESCUE-OK"
echo "CHAIN-SPORT2-COMPLETE $(date -u +%FT%TZ)"
