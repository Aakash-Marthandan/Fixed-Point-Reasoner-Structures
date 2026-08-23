#!/bin/bash
# Ledger: SPRINT S2 WAVE 2 (2026-08-22 launch registration) — Sudoku-Extreme on
# ONE spot pod (v6e-16 = 2 workers x 8 chips, or the v6e-8 fallback = 1 worker),
# pod.sh-compatible. Goal metric: Sudoku-Extreme full-test exact (cold) + the
# labeled verify-and-vote breadth number. Arms = the wave-1 verdict's levers on
# the PLAIN base + the PRICE x SCALE surface (PI 2026-08-22: price effects may be
# scale-dependent — measure width/budget/dose, not one cell):
#   W1   plain T6 d16 @50k            (budget; M0 at t=64 on this arm)
#   W13  plain T6 d16 @100k           (budget axis, 3rd point)
#   W2   plain T12 d16 @30k (+remat)  (training depth on the equilibrium base)
#   W3   plain T12 +RI .15 +NI .01    (the EqR recipe on the plain base; breadth hits)
#   W4   plain T6 d32 @20k (+remat)   (width; paired with S5)
#   W8   priced T6 d32 @20k (+remat)  (price x width — H-44)
#   W9   priced T6 d16 @50k           (price x budget — H-44)
#   W6   beta 3e-6 T6 d16 @20k        (price DOSE — H-43's registered test)
#   W5   GEN: generator pretrain 20k (gen npz) -> 1k finetune 20k (--init-from)
#   W7   plain T6 d16 @20k, layout box4  (the registered box-aligned CONTROL)
#   W1s1 W4s1  seed-1 replicates of W1 and W4 (seed spread at both widths)
#   scan:S5:64:256 scan:S4:64:256 scan:S7:64:256 scan:S5:6:256
#        = BREADTH SCAN on banked wave-1 ckpts (strat-512, k=256, nested k-curve)
# JOBS run as per-chip queues (round-robin over the worker's 8 chips, each chip
# sequential, NO global barrier): a training arm's pipeline = pretrain -> evals
# (strat t6/64/256 k16; full t6/t64 k0; val t64; retention t8 solution-init)
# -> probe (origin layout only) -> per-arm GCS upload. Then PHASE4 per worker:
# best-of-my-arms (full t64 exact, mechanical) -> 20k-puzzle seeded subsample at
# t=64 with k=128, sharded over the 8 chips. Then the COMPLETION GUARD: the
# worker checks GCS for EVERY arm's artifacts (all workers); the LAST worker to
# finish builds the final object + emits the global sentinel; otherwise it emits
# CHAIN-SPORT2W2-WORKER-DONE and exits 0 (pod.sh reads both). Everything is
# arm-keyed and resumable, so a v6e-16 -> v6e-8 fallback loses nothing.
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
R_TAG=${R_TAG:-sport2w2}
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
GCS=${GCS:-gs://qhrrn2-rescue/sport2w2}; FINAL_OBJ=${FINAL_OBJ:-sport2w2_final.tgz}
GCS_W1=${GCS_W1:-gs://qhrrn2-rescue/sport2}; W1_TAG=${W1_TAG:-sport2}
SX_NPZ=${SX_NPZ:-sudoku_extreme_seed0.npz}; GEN_NPZ=${GEN_NPZ:-sudoku_gen_g22_seed0.npz}
SX_AUG=${SX_AUG:-100}; SX_T_STRAT=${SX_T_STRAT:-"6 64 256"}; SX_T_FULL=${SX_T_FULL:-"6 64"}
SX_K_INIT=${SX_K_INIT:-16}; SX_STRAT=${SX_STRAT:-512}
SX_SUB=${SX_SUB:-20000}; SX_SUB_K=${SX_SUB_K:-128}; SX_RET_T=${SX_RET_T:-8}
# chips on THIS host: v6e-8 = 1 host x 8 chips; v6e-16 = 4 hosts x 4 chips (learned live 2026-08-22:
# the first launch pinned chips 0-7 on 4-chip hosts and half the jobs died at startup). Detect from
# /dev/vfio (one numeric entry per chip); fall back to 8.
if [ -z "${NCHIP:-}" ]; then NCHIP=$(ls /dev/vfio 2>/dev/null | grep -cE '^[0-9]+$'); fi
[ "${NCHIP:-0}" -ge 1 ] 2>/dev/null || NCHIP=8
SENT=CHAIN-SPORT2W2
NPZ=data/sudoku_extreme/$SX_NPZ; GNPZ=data/sudoku_extreme/$GEN_NPZ
mkdir -p runs data/sudoku_extreme

# ---- job lists: ARMS_W0, ARMS_W1, ... (explicit per-worker lists from campaign.env);
# with NW workers, worker w runs ARMS_W$w; with fewer workers than lists the lists
# are concatenated in order (the v6e-8 fallback runs everything on one worker).
ALL_JOBS=""; MY_JOBS=""
for i in 0 1 2 3 4 5 6 7; do
  v=$(eval "echo \${ARMS_W$i:-}")
  [ -n "$v" ] || continue
  ALL_JOBS="$ALL_JOBS $v"
  if [ "$NW" -ge 2 ]; then [ "$i" -eq "$W" ] && MY_JOBS="$MY_JOBS $v"
  else MY_JOBS="$MY_JOBS $v"; fi
done
[ -n "$ALL_JOBS" ] || ALL_JOBS=${R0_ARMS:-}
[ "$NW" -ge 2 ] || MY_JOBS=$ALL_JOBS
ALL_ARMS=""; for j in $ALL_JOBS; do case $j in scan:*) ;; *) ALL_ARMS="$ALL_ARMS $j";; esac; done
echo "=== SPORT2W2 START $(date -u +%FT%TZ) worker=$W/$NW chips=$NCHIP my_jobs=[$MY_JOBS] all_arms=[$ALL_ARMS] ==="
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/$SX_NPZ" "$NPZ" || { echo "NPZ-MISSING $SX_NPZ"; exit 2; }
[ -f "$GNPZ" ] || gsutil -q cp "$GCS/$GEN_NPZ" "$GNPZ" || echo "GEN-NPZ-MISSING $GEN_NPZ (GEN arm will fail)"

pin () {  # pin CHIP -- cmd...   (shard_run.sh recipe, validated by probe_chips.py)
  local c=$1; shift
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c \
  JAX_DEFAULT_MATMUL_PRECISION=highest "$@"
}
COMMON="--equilibrium --anchor-p 0.3 --sudoku-extreme $NPZ --sudoku-aug $SX_AUG --n-val 64 --seed 0"
base_arm () { echo "$1" | sed -E 's/s[12]$//'; }
arm_flags () {   # the registered per-arm flags (d / T / steps / price / mechanism)
  case $1 in
    W1)  echo "--d 16 --T 6 --steps 50000 --beta-flux 0 --beta-flux-nl 0";;
    W13) echo "--d 16 --T 6 --steps 100000 --beta-flux 0 --beta-flux-nl 0";;
    W2)  echo "--d 16 --T 12 --steps 30000 --beta-flux 0 --beta-flux-nl 0 --remat";;
    W3)  echo "--d 16 --T 12 --steps 30000 --beta-flux 0 --beta-flux-nl 0 --ri-p 0.15 --ni-sigma 0.01 --remat";;
    W4)  echo "--d 32 --T 6 --steps 20000 --beta-flux 0 --beta-flux-nl 0 --remat";;
    W8)  echo "--d 32 --T 6 --steps 20000 --beta-flux 3e-5 --beta-flux-nl 1e-5 --remat";;
    W9)  echo "--d 16 --T 6 --steps 50000 --beta-flux 3e-5 --beta-flux-nl 1e-5";;
    W6)  echo "--d 16 --T 6 --steps 20000 --beta-flux 3e-6 --beta-flux-nl 1e-6";;
    W5)  echo "--d 16 --T 6 --steps 20000 --beta-flux 0 --beta-flux-nl 0";;   # finetune stage (init-from W5gen)
    W5gen) echo "--d 16 --T 6 --steps 20000 --beta-flux 0 --beta-flux-nl 0";; # generator stage (gen npz, aug 9)
    W7)  echo "--d 16 --T 6 --steps 20000 --beta-flux 0 --beta-flux-nl 0 --sudoku-layout box4";;
    W*s1) echo "$(arm_flags "${1%s1}") --seed 1";;
    W*s2) echo "$(arm_flags "${1%s2}") --seed 2";;
    *) echo "UNKNOWN-ARM $1" >&2; return 1;;
  esac
}
arm_layout () { case "$(arm_flags "$1")" in *box4*) echo box4;; *) echo origin;; esac; }
arm_is_gen () { [ "$(base_arm "$1")" = W5 ]; }

# ---------- per-arm pipeline (runs on ONE chip, resumable at every step) ----------
pretrain_one () {   # ARM CHIP OUTDIR NPZ AUG [extra flags]  -> 0 ok
  local arm=$1 c=$2 D=$3 npz=$4 aug=$5; shift 5
  mkdir -p "$D"
  [ -f "$D/.done" ] && { echo "SKIP-$arm (done)"; return 0; }
  gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" 2>/dev/null && { touch "$D/.done"; echo "SKIP-$arm (GCS complete)"; return 0; }
  gsutil -q cp "$GCS/${arm}_ckpt_live.pkl" "$D/ckpt_latest.pkl" 2>/dev/null && echo "RESUME-$arm from live ckpt"
  echo "=== PRETRAIN $arm $(date -u +%H:%M) === chip $c flags: $(arm_flags "$arm") $*"
  ( while true; do sleep 300; gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt_live.pkl" 2>/dev/null || true; done ) & local SY=$!
  # shellcheck disable=SC2086
  pin "$c" python3 tools/pretrain.py --out "$D" --equilibrium --anchor-p 0.3 --sudoku-extreme "$npz" \
      --sudoku-aug "$aug" --n-val 64 --seed 0 $(arm_flags "$arm") "$@" > "runs/wave_pre_$arm.log" 2>&1
  local rc=$?; pkill -P $SY 2>/dev/null; kill $SY 2>/dev/null || true   # stop the live-sync loop AND its sleeping child (no orphans holding fds)
  if [ $rc -eq 0 ]; then
    touch "$D/.done"; gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt.pkl"
    gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics.jsonl"; echo "PRETRAIN-$arm-OK $(date -u +%H:%M)"
  else echo "PRETRAIN-$arm-FAILED rc=$rc (see runs/wave_pre_$arm.log)"; fi
  return $rc
}
eval_one () {      # ARM CHIP  (all evals of one trained arm; each file skip-if-present)
  local arm=$1 c=$2 D=runs/pretrain${R_TAG}_$1 O=runs/sxeval_p${R_TAG}$1 t
  mkdir -p "$O"
  [ -n "$(ls "$O" 2>/dev/null)" ] || { gsutil -q cp "$GCS/${arm}_eval.tgz" "/tmp/sxeval_$arm.tgz" 2>/dev/null && tar xzf "/tmp/sxeval_$arm.tgz" && echo "RESTORE-$arm evals from GCS"; }
  for t in $SX_T_STRAT; do
    [ -f "$O/strat_t$t/summary_all.json" ] || pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" \
        --split test --stratified "$SX_STRAT" --t-total "$t" --k-init "$SX_K_INIT" --out "$O/strat_t$t" > "runs/wave_ev_${arm}_s$t.log" 2>&1 || echo "EVAL-$arm-strat$t-FAILED"
  done
  [ -f "$O/val_t64/summary_all.json" ] || pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" \
      --split val --t-total 64 --k-init 0 --out "$O/val_t64" > "runs/wave_ev_${arm}_v64.log" 2>&1 || echo "EVAL-$arm-val-FAILED"
  [ -f "$O/ret_t$SX_RET_T/summary_all.json" ] || pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" \
      --split test --stratified "$SX_STRAT" --t-total "$SX_RET_T" --k-init 0 --init solution --out "$O/ret_t$SX_RET_T" > "runs/wave_ev_${arm}_r.log" 2>&1 || echo "EVAL-$arm-ret-FAILED"
  for t in $SX_T_FULL; do
    [ -f "$O/full_t$t/summary_all.json" ] || pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" \
        --split test --t-total "$t" --k-init 0 --out "$O/full_t$t" > "runs/wave_ev_${arm}_f$t.log" 2>&1 || echo "EVAL-$arm-full$t-FAILED"
  done
  tar czf "/tmp/sxeval_$arm.tgz" "$O" && gsutil -q cp "/tmp/sxeval_$arm.tgz" "$GCS/${arm}_eval.tgz" && echo "EVAL-$arm-OK $(date -u +%H:%M)"
}
probe_one () {     # ARM CHIP
  local arm=$1 c=$2 D=runs/pretrain${R_TAG}_$1 O=runs/sudprobe_p${R_TAG}$1
  if [ "$(arm_layout "$arm")" != origin ]; then echo "PROBE-SKIP-$arm (layout $(arm_layout "$arm"); evaluator carries retention/multi-init)"; return 0; fi
  [ -s "$O/results.jsonl" ] || { gsutil -q cp "$GCS/${arm}_probe.tgz" "/tmp/sxprobe_$arm.tgz" 2>/dev/null && tar xzf "/tmp/sxprobe_$arm.tgz" && echo "RESTORE-$arm probe from GCS"; }
  if pin "$c" python3 tools/probe_sudoku.py --ckpt "$D/ckpt_latest.pkl" --pairs-file "$NPZ" --split test \
        --stratified "$SX_STRAT" --t-total 64 --k-init 16 --out "$O" > "runs/wave_pr_$arm.log" 2>&1; then
    tar czf "/tmp/sxprobe_$arm.tgz" "$O" && gsutil -q cp "/tmp/sxprobe_$arm.tgz" "$GCS/${arm}_probe.tgz"; echo "PROBE-$arm-OK $(date -u +%H:%M)"
  else echo "PROBE-$arm-FAILED"; fi
}
arm_pipeline () {  # ARM CHIP
  local arm=$1 c=$2 D=runs/pretrain${R_TAG}_$1
  if arm_is_gen "$arm"; then
    local G="runs/pretrain${R_TAG}_${arm}gen"
    pretrain_one "${arm}gen" "$c" "$G" "$GNPZ" 9 || { echo "GEN-STAGE-A-FAILED $arm"; return 1; }
    pretrain_one "$arm" "$c" "$D" "$NPZ" "$SX_AUG" --init-from "$G/ckpt_latest.pkl" || return 1
  else
    pretrain_one "$arm" "$c" "$D" "$NPZ" "$SX_AUG" || return 1
  fi
  eval_one "$arm" "$c"
  probe_one "$arm" "$c"
}
scan_one () {      # scan:ARM:T:K CHIP — breadth scan on a banked WAVE-1 ckpt
  local spec=$1 c=$2; IFS=: read -r _ arm t k <<< "$spec"
  local D=runs/pretrain${W1_TAG}_$arm O=runs/sxbreadth_${arm}_t${t}_k${k} obj=breadth_${arm}_t${t}_k${k}.tgz
  gsutil -q stat "$GCS/$obj" 2>/dev/null && { echo "SKIP-scan-$arm-t$t-k$k (GCS complete)"; return 0; }
  mkdir -p "$D"; [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS_W1/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" || { echo "SCAN-$arm-NOCKPT"; return 1; }
  echo "=== SCAN $arm t=$t k=$k $(date -u +%H:%M) === chip $c"
  if [ -f "$O/summary_all.json" ] || pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" \
        --split test --stratified "$SX_STRAT" --t-total "$t" --k-init "$k" --out "$O" > "runs/wave_scan_${arm}_t${t}_k${k}.log" 2>&1; then
    tar czf "/tmp/$obj" "$O" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "SCAN-$arm-t$t-k$k-OK $(date -u +%H:%M)"
  else echo "SCAN-$arm-t$t-k$k-FAILED"; fi
}
chip_queue () {    # CHIP job job ...  (sequential on one chip)
  local c=$1; shift
  for job in "$@"; do
    case $job in scan:*) scan_one "$job" "$c";; *) arm_pipeline "$job" "$c";; esac
  done
}

# ---------- PHASE 1-3: per-chip queues, round-robin, no barrier ----------
echo "PHASE1-3: per-chip queues $(date -u +%H:%M)"
i=0
for job in $MY_JOBS; do
  c=$((i % NCHIP)); eval "Q_$c=\"\${Q_$c:-} $job\""; i=$((i+1))
done
pids=()
for c in $(seq 0 $((NCHIP-1))); do
  q=$(eval "echo \${Q_$c:-}")
  [ -n "$q" ] || continue
  echo "chip $c queue: $q"
  # shellcheck disable=SC2086
  ( chip_queue "$c" $q ) & pids+=($!)
done
for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
echo "QUEUES-DONE worker=$W $(date -u +%H:%M)"

# ---------- PHASE 4: best-of-my-arms breadth number (20k subsample, k=128, sharded) ----------
best=""; bestacc=-1
for arm in $MY_JOBS; do
  case $arm in scan:*) continue;; esac
  f=runs/sxeval_p${R_TAG}$arm/full_t64/summary_all.json
  [ -f "$f" ] || continue
  acc=$(python3 -c "import json;print(json.load(open('$f'))['exact_acc'])" 2>/dev/null) || continue
  if python3 -c "import sys; sys.exit(0 if float('$acc') > float('$bestacc') else 1)"; then best=$arm; bestacc=$acc; fi
done
if [ -n "$best" ]; then
  O=runs/sxbreadth20k_p${R_TAG}$best; obj=breadth20k_${best}.tgz
  if gsutil -q stat "$GCS/$obj" 2>/dev/null; then echo "SKIP-PHASE4 ($best, GCS complete)"
  else
    echo "PHASE4: best arm $best (full t64 exact $bestacc) -> $SX_SUB-puzzle subsample k=$SX_SUB_K t=64, $NCHIP shards $(date -u +%H:%M)"
    mkdir -p "$O"; pids=()
    for c in $(seq 0 $((NCHIP-1))); do
      [ -f "$O/summary_s$c.json" ] && continue
      ( pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "runs/pretrain${R_TAG}_$best/ckpt_latest.pkl" --npz "$NPZ" \
          --split test --subsample "$SX_SUB" --t-total 64 --k-init "$SX_SUB_K" --shard "$c/$NCHIP" --out "$O" > "runs/wave_b20k_${best}_s$c.log" 2>&1 \
          || echo "PHASE4-shard$c-FAILED" ) & pids+=($!)
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
    if python3 tools/eval_sudoku_extreme.py --merge "$O" > "runs/wave_b20k_${best}_merge.log" 2>&1 && [ -f "$O/summary_all.json" ]; then
      tar czf "/tmp/$obj" "$O" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "PHASE4-OK $best $(date -u +%H:%M)"
    else echo "PHASE4-MERGE-FAILED $best"; fi
  fi
else echo "PHASE4-SKIP (no evaluated arm on this worker)"; fi

# ---------- COMPLETION GUARD (global, arm-keyed, worker-agnostic) ----------
# A job is complete when its GCS artifacts exist: training arm = ckpt + eval.tgz
# (+ probe.tgz unless layout != origin); scan = breadth tgz. The last worker to
# find everything present pulls the others' artifacts, builds the final object
# and emits the global sentinel. Else WORKER-DONE (pod.sh waits for the rest).
missing=""
for job in $ALL_JOBS; do
  case $job in
    scan:*) IFS=: read -r _ arm t k <<< "$job"; gsutil -q stat "$GCS/breadth_${arm}_t${t}_k${k}.tgz" 2>/dev/null || missing="$missing $job";;
    *) gsutil -q stat "$GCS/${job}_ckpt.pkl" 2>/dev/null || missing="$missing $job:ckpt"
       gsutil -q stat "$GCS/${job}_eval.tgz" 2>/dev/null || missing="$missing $job:eval"
       [ "$(arm_layout "$job")" != origin ] || gsutil -q stat "$GCS/${job}_probe.tgz" 2>/dev/null || missing="$missing $job:probe";;
  esac
done
if [ -n "$missing" ]; then echo "$SENT-WORKER-DONE worker=$W missing(other workers or failed):$missing $(date -u +%FT%TZ)"; exit 0; fi
# everything present: pull what this worker does not hold locally, then final object
for job in $ALL_JOBS; do
  case $job in
    scan:*) IFS=: read -r _ arm t k <<< "$job"; [ -f "runs/sxbreadth_${arm}_t${t}_k${k}/summary_all.json" ] || { gsutil -q cp "$GCS/breadth_${arm}_t${t}_k${k}.tgz" /tmp/_b.tgz && tar xzf /tmp/_b.tgz; };;
    *) D=runs/pretrain${R_TAG}_$job; mkdir -p "$D"
       [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS/${job}_ckpt.pkl" "$D/ckpt_latest.pkl"
       [ -f "$D/metrics.jsonl" ] || gsutil -q cp "$GCS/${job}_metrics.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
       [ -d "runs/sxeval_p${R_TAG}$job" ] || { gsutil -q cp "$GCS/${job}_eval.tgz" /tmp/_e.tgz && tar xzf /tmp/_e.tgz; }
       [ "$(arm_layout "$job")" != origin ] || [ -d "runs/sudprobe_p${R_TAG}$job" ] || { gsutil -q cp "$GCS/${job}_probe.tgz" /tmp/_p.tgz && tar xzf /tmp/_p.tgz; };;
  esac
done
for obj in $(gsutil ls "$GCS/breadth20k_*.tgz" 2>/dev/null); do b=$(basename "$obj" .tgz); [ -d "runs/sxbreadth20k_p${R_TAG}${b#breadth20k_}" ] || { gsutil -q cp "$obj" /tmp/_k.tgz && tar xzf /tmp/_k.tgz; }; done
tar czf /tmp/$FINAL_OBJ runs/pretrain${R_TAG}_*/ckpt_latest.pkl runs/pretrain${R_TAG}_*/metrics.jsonl runs/*_p${R_TAG}* runs/sxbreadth_* runs/wave_*.log 2>/dev/null
gsutil -q cp /tmp/$FINAL_OBJ "$GCS/$FINAL_OBJ" && echo "RESCUE-OK"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
# ---------- cloud-side SELF-TEARDOWN backstop (2026-08-22, PI offline: "finish and tear down on its own") ----------
# The supervisor on the Mac is the primary teardown; if it is dead/asleep this node would bill until the
# watchdog deadline. With everything banked (final object uploaded synchronously above), the finalizing
# worker deletes its own node. Gated on SELF_TEARDOWN=1 + SELF_POD/SELF_ZONE (set by pod.sh's launch line).
if [ "${SELF_TEARDOWN:-0}" = 1 ] && [ -n "${SELF_POD:-}" ] && [ -n "${SELF_ZONE:-}" ]; then
  echo "SELF-TEARDOWN: deleting $SELF_POD in $SELF_ZONE (all artifacts banked) $(date -u +%FT%TZ)"
  sleep 20   # let the log line land in any in-flight supervisor poll
  gcloud compute tpus tpu-vm delete "$SELF_POD" --zone "$SELF_ZONE" --quiet >/dev/null 2>&1 && echo "SELF-TEARDOWN-ISSUED" || echo "SELF-TEARDOWN-FAILED (supervisor/watchdog will tear down)"
fi
