#!/bin/bash
# Ledger: PHASE B RUNG 2 (2026-08-27 launch registration) — d96 FULL-WIDTH
# (--width-scale 6; 2.11M params) on ONE spot pod (v6e-16 = 4x4 or v6e-8 = 1x8;
# chips self-detected; SAME launch contract as chain_sportB.sh — this file is
# its parameterized derivative; every mechanism below is the rung-1-proven one).
# Arms (WIDTH the variable; recipes per the registration):
#   C1   carrier s0:  plain T12 FPA k4 @50k     (B2-recipe at d96)
#   C1s1 carrier s1:  the funnel-noise pair rides the CARRIER this rung
#   C2   priced T12 @50k (knee 3e-5/1e-5)       (P-A's cell)
#   C3   plain T6 FPA k4 @20k                   (wide-funnel INSURANCE)
#   C4   priced T12 @50k at beta/3 (1e-5/3.3e-6) (H-47 / P-B's cell)
# Rung-2 instrument deltas (all registered): MONITOR every 2000; screens on
# THREE ckpts per arm (vb + m1 + m2; 50k arms m1=25000 m2=40000, 20k arms
# m1=10000 m2=15000); probes with the EXTENDED eps-ladder (flag; probe default
# unchanged); eval bank quantum --batch 128 (batch-invariance pinned by
# tests/test_eval_bank_resume.py::test_batch_size_invariance); PHASE4-DEPTH =
# winner full-test COLD t=256 (labeled row); D3 demo = OPTIONAL claim task
# (unverified majority vote on banked B2-d64 + S5-d16 ckpts) that can NEVER
# block completion. O2: JAX persistent compile cache synced via GCS.
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
R_TAG=${R_TAG:-sportBr2}; export R_TAG
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
GCS=${GCS:-gs://qhrrn2-rescue/sportBr2}; FINAL_OBJ=${FINAL_OBJ:-sportBr2_final.tgz}
GCS_W1=${GCS_W1:-gs://qhrrn2-rescue/sport2}
GCS_R1=${GCS_R1:-gs://qhrrn2-rescue/sportB}
SX_NPZ=${SX_NPZ:-sudoku_extreme_seed0.npz}
SX_AUG=${SX_AUG:-100}; SX_T_STRAT=${SX_T_STRAT:-"6 64 256"}
SX_K_INIT=${SX_K_INIT:-16}; SX_STRAT=${SX_STRAT:-512}
SX_SUB=${SX_SUB:-20000}; SX_SUB_K=${SX_SUB_K:-128}; SX_RET_T=${SX_RET_T:-8}
SCREEN_K=${SCREEN_K:-256}
MON_EVERY=${MON_EVERY:-2000}
WS=${WS:-6}; RD=${RD:-96}
EVAL_BATCH=${EVAL_BATCH:-128}
EPS_RUNGS=${EPS_RUNGS:-"0.05 0.1 0.2 0.4 0.6 0.8"}
if [ -z "${NCHIP:-}" ]; then NCHIP=$(ls /dev/vfio 2>/dev/null | grep -cE '^[0-9]+$'); fi
[ "${NCHIP:-0}" -ge 1 ] 2>/dev/null || NCHIP=8
SENT=CHAIN-SPORTBR2
NPZ=data/sudoku_extreme/$SX_NPZ
mkdir -p runs data/sudoku_extreme

# ---------- O2: persistent XLA compile cache (GCS-synced; non-fatal everywhere) ----------
export JAX_COMPILATION_CACHE_DIR="$PWD/jax_cache"
mkdir -p "$PWD/jax_cache"
gsutil -q cp "$GCS/jax_cache.tgz" /tmp/jc.tgz 2>/dev/null && tar xzf /tmp/jc.tgz 2>/dev/null && echo "COMPILE-CACHE restored ($(ls jax_cache | wc -l | tr -d ' ') entries)"
cache_push () { tar czf /tmp/jc.tgz jax_cache 2>/dev/null && gsutil -q cp /tmp/jc.tgz "$GCS/jax_cache.tgz" 2>/dev/null && echo "COMPILE-CACHE pushed ($(ls jax_cache | wc -l | tr -d ' ') entries)"; }

ALL_JOBS=""; MY_JOBS=""
for i in 0 1 2 3 4 5 6 7; do
  v=$(eval "echo \${ARMS_W$i:-}")
  [ -n "$v" ] || continue
  ALL_JOBS="$ALL_JOBS $v"
  if [ "$NW" -ge 2 ]; then [ "$i" -eq "$W" ] && MY_JOBS="$MY_JOBS $v"
  else MY_JOBS="$MY_JOBS $v"; fi
done
[ "$NW" -ge 2 ] || MY_JOBS=$ALL_JOBS
ALL_ARMS=$ALL_JOBS
PRIMARY=${PRIMARY:-"C1 C1s1 C2 C3"}
CARRIER_FULLS=${CARRIER_FULLS:-"C1 C1s1"}
echo "=== SPORTBR2 START $(date -u +%FT%TZ) worker=$W/$NW chips=$NCHIP my_arms=[$MY_JOBS] all_arms=[$ALL_ARMS] d=$RD ws=$WS ==="
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/$SX_NPZ" "$NPZ" || { echo "NPZ-MISSING $SX_NPZ"; exit 2; }

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest "$@"; }
is_primary () { case " $PRIMARY " in *" $1 "*) return 0;; *) return 1;; esac; }
is_carrier () { case " $CARRIER_FULLS " in *" $1 "*) return 0;; *) return 1;; esac; }
arm_flags () {   # WIDTH-ONLY scaling of the registered recipes
  case $1 in
    # C1 LR-RETRY (PI-authorized 2026-08-28 ~00:50 IST): NaN onset ~1.5k steps
    # after the 18:21Z preemption-resume (the B1-d64 rng-re-split signature);
    # ONE labeled half-lr relaunch from the clean 10k ckpt — the registered
    # contingency. The analysis pass reads C1 with this label. If C1 NaNs
    # again at half-lr: STOP the arm (no lr iteration; C1s1 carries the pair).
    C1) echo "--d $RD --width-scale $WS --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0 --lr 5e-4";;
    # C1s1 LR-RETRY (autonomous-mode call 2026-08-28 ~01:45 IST, same registered
    # contingency): NaN by step 8300 (clean at 6150) — the SECOND carrier seed to
    # diverge at the registered lr at d96, while priced C2 and T6 C3 stay clean.
    # ONE labeled half-lr relaunch from the clean 5k grid ckpt; if C1s1 NaNs
    # again at half-lr: STOP the arm. Both carrier seeds now run lr 5e-4 —
    # symmetric treatment; the analysis pass reads the pair with this label.
    C1s1) echo "--d $RD --width-scale $WS --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0 --seed 1 --lr 5e-4";;
    C2) echo "--d $RD --width-scale $WS --T 12 --steps 50000 --beta-flux 3e-5 --beta-flux-nl 1e-5";;
    C3) echo "--d $RD --width-scale $WS --T 6 --steps 20000 --beta-flux 0 --beta-flux-nl 0 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0";;
    C4) echo "--d $RD --width-scale $WS --T 12 --steps 50000 --beta-flux 1e-5 --beta-flux-nl 3.3e-6";;
    C*s1) echo "$(arm_flags "${1%s1}") --seed 1";;
    C*s2) echo "$(arm_flags "${1%s2}") --seed 2";;
    *) echo "UNKNOWN-ARM $1" >&2; return 1;;
  esac
}
arm_steps () { case $1 in C3*) echo 20000;; *) echo 50000;; esac; }
mstep () {  # ARM {m1|m2} -> the registered fixed screen step for the arm's length
  local st; st=$(arm_steps "$1")
  if [ "$st" = 50000 ]; then [ "$2" = m1 ] && echo 25000 || echo 40000
  else [ "$2" = m1 ] && echo 10000 || echo 15000; fi
}

sync_loop () {  # ARM DIR — every 5 min bank ckpt_latest + metrics + banked ckpts
  local arm=$1 D=$2
  while true; do sleep 300
    # NaN guard (2026-08-28, after BOTH carrier seeds NaN'd on 08-27): NEVER
    # push non-finite state to the live bank — yesterday the poisoned
    # {arm}_ckpt_live needed manual GCS surgery twice. On a non-finite newest
    # loss: emit ONE marker (detection ≤5 min), halt this arm's pretrain (the
    # run is dead science-wise; the registered contingency decides what's
    # next), and stop syncing so GCS keeps the last CLEAN state.
    if [ -f "$D/metrics.jsonl" ]; then
      lastrow=$(tail -1 "$D/metrics.jsonl")
      case "$(grep -oE '"loss": [^,}]+' <<<"$lastrow" | awk '{print tolower($2)}')" in
        *nan*|*inf*)
          echo "PRETRAIN-$arm-NAN-HALTED $(grep -oE '"step": [0-9]+' <<<"$lastrow") (live bank preserved at last clean sync; registered contingency decides)"
          pkill -f "pretrain[.]py" 2>/dev/null
          return 0;;
      esac
    fi
    gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt_live.pkl" 2>/dev/null || true
    gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics_live.jsonl" 2>/dev/null || true
    for f in "$D"/ckpt_0*.pkl; do [ -f "$f" ] && gsutil -q cp -n "$f" "$GCS/${arm}_$(basename "$f")" 2>/dev/null; done
  done
}
run_pretrain () {  # ARM DIR EXTRA -> rc  (DP over the whole worker host; per-host confinement on multi-host — 93a79d4)
  local arm=$1 D=$2 extra=$3
  local conf=""
  [ "$NW" -ge 2 ] && conf="TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3"
  # shellcheck disable=SC2086
  env $conf JAX_DEFAULT_MATMUL_PRECISION=highest python3 tools/pretrain.py --out "$D" --equilibrium --anchor-p 0.3 \
      --sudoku-extreme "$NPZ" --sudoku-aug "$SX_AUG" --n-val 64 --seed 0 --dp \
      --monitor-every "$MON_EVERY" $(arm_flags "$arm") $extra > "runs/wave_pre_$arm.log" 2>&1
}
pretrain_one () {   # ARM -> 0 ok  (SKIP/RESUME semantics identical to rung 1)
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  mkdir -p "$D"
  [ -f "$D/.done" ] && { echo "SKIP-$arm (done)"; return 0; }
  if gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" 2>/dev/null; then
    gsutil -q cp "$GCS/${arm}_metrics.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
    gsutil -q cp "$GCS/${arm}_val_best.txt" "$D/val_best.txt" 2>/dev/null || true
    for f in $(gsutil ls "$GCS/${arm}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${arm}_}" ] || gsutil -q cp "$f" "$D/${b#${arm}_}"; done
    touch "$D/.done"; echo "SKIP-$arm (GCS complete)"; return 0
  fi
  if gsutil -q cp "$GCS/${arm}_ckpt_live.pkl" "$D/ckpt_latest.pkl" 2>/dev/null; then
    gsutil -q cp "$GCS/${arm}_metrics_live.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
    for f in $(gsutil ls "$GCS/${arm}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${arm}_}" ] || gsutil -q cp "$f" "$D/${b#${arm}_}"; done
    echo "RESUME-$arm from live ckpt (+metrics, +banked ckpts)"
  fi
  echo "=== PRETRAIN $arm $(date -u +%H:%M) === DP x$NCHIP flags: $(arm_flags "$arm")"
  sync_loop "$arm" "$D" & local SY=$!
  run_pretrain "$arm" "$D" ""
  local rc=$?
  if [ $rc -ne 0 ] && grep -qE "RESOURCE_EXHAUSTED|out of memory|OOM" "runs/wave_pre_$arm.log"; then
    echo "PRETRAIN-$arm-REMAT-RETRY (HBM OOM; numerics-equivalent per test_eq_remat_matches_no_remat)"
    run_pretrain "$arm" "$D" "--remat"; rc=$?
  fi
  pkill -P $SY 2>/dev/null; kill $SY 2>/dev/null || true
  if [ $rc -eq 0 ]; then
    lastloss=$(grep -oE '"loss": [0-9.eE+-]+' "$D/metrics.jsonl" 2>/dev/null | tail -1 | awk '{print $2}')
    python3 -c "import sys; l=float('${lastloss:-9}'); sys.exit(0 if l==l and l < 3.0 else 1)" \
      || echo "PRETRAIN-$arm-DIVERGED last_loss=${lastloss:-nan} (report to PI; registered contingency = ONE labeled relaunch at half lr, manual — the B1-d64 precedent)"
    touch "$D/.done"; gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt.pkl"; gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics.jsonl"
    for f in "$D"/ckpt_0*.pkl; do [ -f "$f" ] && gsutil -q cp -n "$f" "$GCS/${arm}_$(basename "$f")" 2>/dev/null; done
    python3 tools/select_ckpt.py "$D" > "$D/val_best.txt" 2>"runs/wave_sel_$arm.log" \
      && { echo "VALBEST-$arm $(cat "$D/val_best.txt")"; gsutil -q cp "$D/val_best.txt" "$GCS/${arm}_val_best.txt"; }
    echo "PRETRAIN-$arm-OK $(date -u +%H:%M)"
  else echo "PRETRAIN-$arm-FAILED rc=$rc (see runs/wave_pre_$arm.log)"; fi
  return $rc
}
eval_cheap () {   # ARM — strat t6/64/256 k16, val t64, ret_t8, retfm_t8
  local arm=$1 D=runs/pretrain${R_TAG}_$1 O=runs/sxeval_p${R_TAG}$1 t i=0 pids=()
  mkdir -p "$O"
  [ -n "$(ls "$O" 2>/dev/null)" ] || { gsutil -q cp "$GCS/${arm}_evalcheap.tgz" "/tmp/sxec_$arm.tgz" 2>/dev/null && tar xzf "/tmp/sxec_$arm.tgz" && echo "RESTORE-$arm cheap evals from GCS"; }
  ec_one () { local kind=$1 c=$2; shift 2
    [ -f "$O/$kind/summary_all.json" ] && return 0
    # Busy-retry (2026-08-28 fix, autonomous mode): on a 4-chip worker the six
    # cheap evals pin-collide (i%4 → ret_t8/retfm_t8 land on chips still running
    # strat evals) and the loser died instantly ("Couldn't open iommu group") —
    # the C3 ret/retfm failure. Same retry contract as sharded_eval; the 8-chip
    # shape (no collision) is byte-equivalent on the first try.
    local try
    for try in $(seq 1 60); do
      pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" --out "$O/$kind" --batch "$EVAL_BATCH" "$@" \
        > "runs/wave_ev_${arm}_$kind.log" 2>&1 && return 0
      if grep -qE "resource busy|Couldn't open iommu group" "runs/wave_ev_${arm}_$kind.log"; then
        [ "$try" -eq 1 ] && echo "EVAL-WAIT $arm $kind (chip busy; retrying)"
        sleep "${SHARD_RETRY_SLEEP:-120}"; continue
      fi
      echo "EVAL-$arm-$kind-FAILED"; return 1
    done
    echo "EVAL-$arm-$kind-FAILED (retries exhausted)"; return 1
  }
  for t in $SX_T_STRAT; do ec_one "strat_t$t" $((i % NCHIP)) --split test --stratified "$SX_STRAT" --t-total "$t" --k-init "$SX_K_INIT" & pids+=($!); i=$((i+1)); done
  ec_one "val_t64" $((i % NCHIP)) --split val --t-total 64 --k-init 0 & pids+=($!); i=$((i+1))
  ec_one "ret_t$SX_RET_T" $((i % NCHIP)) --split test --stratified "$SX_STRAT" --t-total "$SX_RET_T" --k-init 0 --init solution & pids+=($!); i=$((i+1))
  ec_one "retfm_t$SX_RET_T" $((i % NCHIP)) --split test --stratified "$SX_STRAT" --t-total "$SX_RET_T" --k-init 0 --init solution --final-map-only & pids+=($!)
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
  tar czf "/tmp/sxec_$arm.tgz" "$O" && gsutil -q cp "/tmp/sxec_$arm.tgz" "$GCS/${arm}_evalcheap.tgz"
  echo "EVALCHEAP-$arm-OK $(date -u +%H:%M)"
}

# ---------- sharded helper (rung-1-proven; batch quantum 128) ----------
partial_sync () {
  local O=$1 base; base=$(basename "$O")
  while true; do sleep "${PARTIAL_SYNC_SLEEP:-300}"
    for f in "$O"/partial_*.npz; do [ -f "$f" ] && gsutil -q cp "$f" "$GCS/partials/${base}_$(basename "$f")" 2>/dev/null; done
  done
}
partial_restore () {
  local O=$1 base f b; base=$(basename "$O")
  for f in $(gsutil ls "$GCS/partials/${base}_partial_*.npz" 2>/dev/null); do
    b=$(basename "$f"); b=${b#${base}_}
    [ -f "$O/$b" ] || gsutil -q cp "$f" "$O/$b" 2>/dev/null
  done
}
sharded_eval () {  # OUT CKPT [extra flags...] -> merged summary_all.json in OUT
  local O=$1 CK=$2; shift 2
  [ -f "$O/summary_all.json" ] && return 0
  mkdir -p "$O"; partial_restore "$O"
  partial_sync "$O" & local PS=$!
  local pids=() c
  for c in $(seq 0 $((NCHIP-1))); do
    [ -f "$O/summary_s$c.json" ] && continue
    ( for try in $(seq 1 60); do
        pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --shard "$c/$NCHIP" --out "$O" --bank-every 300 --batch "$EVAL_BATCH" "$@" > "$O/shard_s$c.log" 2>&1 && break
        if grep -qE "resource busy|Couldn't open iommu group" "$O/shard_s$c.log"; then [ "$try" -eq 1 ] && echo "SHARD-WAIT $O s$c (chip busy; retrying)"; sleep "${SHARD_RETRY_SLEEP:-120}"; continue; fi
        echo "SHARD-FAILED $O s$c"; break
      done ) & pids+=($!)
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
  pkill -P $PS 2>/dev/null; kill $PS 2>/dev/null || true
  JAX_PLATFORMS=cpu python3 tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  [ -f "$O/summary_all.json" ]
}

# ---------- PHASE1 ----------
# Early cache_push (during-ride tweak, 2026-08-27; deploys at a relaunch
# boundary): the 12:24Z preemption lost the whole d96 compile because the only
# PHASE1 push ran after ALL arms. This one-shot pusher banks the cache as soon
# as the entry count is nonzero and STABLE across one poll (compile settled) —
# a mid-PHASE1 churn then recompiles nothing. Killed at QUEUES-DONE on every
# path; the end-of-PHASE1 push below still refreshes the final state.
( prev=-1; for i in $(seq 1 24); do sleep "${CACHE_SETTLE_SLEEP:-300}"
    n=$(ls jax_cache 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -gt 0 ] && [ "$n" -eq "$prev" ] && { cache_push; break; }
    prev=$n
  done ) & CPUSH=$!
for arm in $MY_JOBS; do
  if pretrain_one "$arm"; then eval_cheap "$arm"; fi
done
cache_push
kill "$CPUSH" 2>/dev/null || true
echo "QUEUES-DONE worker=$W $(date -u +%H:%M)"

# ---------- PHASE2: GLOBAL claim queue ----------
vb_step () { cut -d' ' -f1 "runs/pretrain${R_TAG}_$1/val_best.txt" 2>/dev/null; }
need_arm_local () {
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  mkdir -p "$D"
  [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" || return 1
  [ -f "$D/val_best.txt" ] || gsutil -q cp "$GCS/${arm}_val_best.txt" "$D/val_best.txt" 2>/dev/null || true
  for f in $(gsutil ls "$GCS/${arm}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${arm}_}" ] || gsutil -q cp "$f" "$D/${b#${arm}_}"; done
  return 0
}
task_obj () {
  case $1 in
    scr:*) IFS=: read -r _ a ck <<< "$1"; echo "screen_${a}_${ck}_k${SCREEN_K}.tgz";;
    full:*) IFS=: read -r _ a kind <<< "$1"; echo "full_${a}_${kind}.tgz";;
    probes4) echo "probes4.tgz";;
    p4depth) echo "depth_t256.tgz";;
    d3demo:*) echo "d3demo_${1#d3demo:}.tgz";;
  esac
}
task_ready () {
  case $1 in
    probes4) local a; for a in $PRIMARY; do gsutil -q stat "$GCS/${a}_ckpt.pkl" 2>/dev/null || return 1; done; return 0;;
    p4depth) gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null;;
    d3demo:*) return 0;;
    *) IFS=: read -r _ a _ <<< "$1"; gsutil -q stat "$GCS/${a}_ckpt.pkl" 2>/dev/null;;
  esac
}
screen_ck () {  # ARM KIND -> "STEP PATH" (empty = skip: the fixed step coincides with vb)
  local arm=$1 kind=$2 D=runs/pretrain${R_TAG}_$1 step vstep
  vstep=$(vb_step "$arm")                    # val_best.txt field 1 is zero-padded
  if [ "$kind" = vb ]; then step=$vstep
  else step=$(printf '%06d' "$(mstep "$arm" "$kind")")
       [ "$step" = "${vstep:-x}" ] && { echo ""; return 0; }
  fi
  [ -n "$step" ] || { echo ""; return 0; }
  local CK="$D/ckpt_$step.pkl"
  local latest_step
  latest_step=$(python3 -c "import pickle;print(f\"{pickle.load(open('$D/ckpt_latest.pkl','rb'))['step']:06d}\")" 2>/dev/null)
  [ -n "$latest_step" ] && [ "$step" = "$latest_step" ] && CK="$D/ckpt_latest.pkl"
  echo "$step $CK"
}
run_task () {
  local t=$1 obj; obj=$(task_obj "$t")
  case $t in
    scr:*)
      IFS=: read -r _ arm ck <<< "$t"; need_arm_local "$arm" || return 1
      read -r step CK <<< "$(screen_ck "$arm" "$ck")"
      [ -n "$step" ] || { echo "SCREEN-$arm-$ck-SKIP (step coincides with vb)"; gsutil -q cp /dev/null "$GCS/$obj" 2>/dev/null || true; return 0; }
      [ -f "$CK" ] || { echo "SCREEN-$arm-$ck-NOCKPT step=$step"; return 1; }
      local O=runs/sxscreen_p${R_TAG}${arm}_${ck}
      if sharded_eval "$O" "$CK" --split test --stratified "$SX_STRAT" --t-total 64 --k-init "$SCREEN_K"; then
        echo "$step" > "$O/step.txt"; tar czf "/tmp/$obj" "$O" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "SCREEN-$arm-$ck-OK step=$step $(date -u +%H:%M)"
      else echo "SCREEN-$arm-$ck-FAILED"; return 1; fi;;
    full:*)
      IFS=: read -r _ arm kind <<< "$t"; need_arm_local "$arm" || return 1
      local D=runs/pretrain${R_TAG}_$arm O=runs/sxeval_p${R_TAG}$arm CK tt Odir
      case $kind in
        t6)  CK=$D/ckpt_latest.pkl; tt=6;  Odir=$O/full_t6;;
        t64) CK=$D/ckpt_latest.pkl; tt=64; Odir=$O/full_t64;;
        vb)  local step; step=$(vb_step "$arm")
             [ -n "$step" ] && [ "$step" != "$(printf '%06d' "$(python3 -c "import pickle;print(pickle.load(open('$D/ckpt_latest.pkl','rb'))['step'])")")" ] \
               || { echo "FULLVB-$arm-SKIP (final=best)"; gsutil -q cp /dev/null "$GCS/$obj" 2>/dev/null || true; return 0; }
             CK=$D/ckpt_$step.pkl; tt=64; Odir=$O/full_t64_valbest;;
      esac
      if sharded_eval "$Odir" "$CK" --split test --t-total "$tt" --k-init 0; then
        tar czf "/tmp/$obj" "$Odir" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "FULL-$arm-$kind-OK $(date -u +%H:%M)"
      else echo "FULL-$arm-$kind-FAILED"; return 1; fi;;
    probes4)
      local a c=0 pids=()
      for a in $PRIMARY; do need_arm_local "$a" || return 1; done
      for a in $PRIMARY; do
        local PD=runs/sudprobe_p${R_TAG}$a
        [ -s "$PD/results.jsonl" ] || { pin "$c" python3 tools/probe_sudoku.py --ckpt "runs/pretrain${R_TAG}_$a/ckpt_latest.pkl" --pairs-file "$NPZ" --split test \
            --stratified "$SX_STRAT" --t-total 64 --k-init 16 --eps-rungs "$EPS_RUNGS" --out "$PD" > "runs/wave_pr_$a.log" 2>&1 || echo "PROBE-$a-FAILED"; } & pids+=($!)
        c=$(( (c+1) % NCHIP ))
      done
      for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
      tar czf "/tmp/$obj" runs/sudprobe_p${R_TAG}* 2>/dev/null && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "PROBES4-OK $(date -u +%H:%M)";;
    p4depth)
      # the inference-depth rider: winner full-test COLD t=256 (labeled row)
      local wn; wn=$(gsutil -q cp "$GCS/p4winner.txt" - 2>/dev/null | head -1)
      [ -n "$wn" ] || { echo "P4DEPTH-WAIT (winner marker absent)"; return 1; }
      need_arm_local "$wn" || return 1
      local D=runs/pretrain${R_TAG}_$wn step CK
      step=$(vb_step "$wn"); CK="$D/ckpt_$step.pkl"; [ -f "$CK" ] || CK="$D/ckpt_latest.pkl"
      if sharded_eval "runs/sxdepth_p${R_TAG}${wn}_t256" "$CK" --split test --t-total 256 --k-init 0; then
        tar czf "/tmp/$obj" "runs/sxdepth_p${R_TAG}${wn}_t256" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "P4DEPTH-OK $wn t256 $(date -u +%H:%M)"
      else echo "P4DEPTH-FAILED"; return 1; fi;;
    d3demo:*)
      # OPTIONAL (never blocks completion): unverified majority-vote demo on banked ckpts
      local which=${t#d3demo:} SRC CKL
      if [ "$which" = b2d64 ]; then SRC="$GCS_R1/B2_ckpt.pkl"; else SRC="$GCS_W1/S5_ckpt.pkl"; fi
      CKL="runs/_d3demo_${which}.pkl"
      [ -f "$CKL" ] || gsutil -q cp "$SRC" "$CKL" || { echo "D3DEMO-$which-NOCKPT"; return 1; }
      local O=runs/sxd3demo_${which}
      [ -f "$O/summary_all.json" ] || pin $((RANDOM % NCHIP)) python3 tools/eval_sudoku_extreme.py --ckpt "$CKL" --npz "$NPZ" --out "$O" \
          --split test --stratified "$SX_STRAT" --t-total 64 --k-init 128 --vote-unverified --batch "$EVAL_BATCH" > "runs/wave_d3_$which.log" 2>&1 \
        || { echo "D3DEMO-$which-FAILED (informational; never blocks)"; return 1; }
      tar czf "/tmp/$obj" "$O" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "D3DEMO-$which-OK $(date -u +%H:%M)";;
  esac
}
SCREEN_TASKS=""
for a in $ALL_ARMS; do SCREEN_TASKS="$SCREEN_TASKS scr:$a:vb scr:$a:m1 scr:$a:m2"; done
FULL_TASKS=""
for a in $ALL_ARMS; do
  FULL_TASKS="$FULL_TASKS full:$a:t64"
  is_carrier "$a" && FULL_TASKS="$FULL_TASKS full:$a:t6 full:$a:vb"
done
# p4depth is NOT in the PHASE2 set — its readiness (breadth20k banked) is only
# produced by PHASE4, which runs after PHASE2: putting it here deadlocks the
# queue to pass-exhaustion (CAUGHT BY THE HARNESS, S1, 2026-08-27). It gets its
# own claim-run after the PHASE4 block; the completion guard still requires it.
TASKS="$FULL_TASKS $SCREEN_TASKS probes4"
OPTIONAL_TASKS="d3demo:b2d64 d3demo:s5d16"
CLAIM_TTL=${CLAIM_TTL:-9000}
for pass in $(seq 1 200); do
  pending=0
  for t in $TASKS $OPTIONAL_TASKS; do
    obj=$(task_obj "$t"); claim="claim_${obj%.tgz}"
    gsutil -q stat "$GCS/$obj" 2>/dev/null && continue
    optional=0; case " $OPTIONAL_TASKS " in *" $t "*) optional=1;; esac
    task_ready "$t" || { [ "$optional" -eq 0 ] && pending=1; continue; }
    if gsutil -q stat "$GCS/$claim" 2>/dev/null; then
      cts=$(gsutil -q cp "$GCS/$claim" - 2>/dev/null | head -1)
      if [ -n "$cts" ] && [ $(( $(date -u +%s) - cts )) -lt "$CLAIM_TTL" ] 2>/dev/null; then [ "$optional" -eq 0 ] && pending=1; continue; fi
      echo "CLAIM-STALE $t (age > ${CLAIM_TTL}s) — taking over"
    fi
    date -u +%s | gsutil -q cp - "$GCS/$claim" 2>/dev/null || true
    if run_task "$t"; then :; else [ "$optional" -eq 0 ] && pending=1; fi
    gsutil -q rm "$GCS/$claim" 2>/dev/null || true
  done
  [ "$pending" -eq 0 ] && break
  sleep "${PHASE2_SLEEP:-120}"
done
echo "PHASE2-DONE worker=$W $(date -u +%H:%M)"

# ---------- PHASE4 (cooperative, ALL workers x ALL chips = NSH-way; rung-1-proven) ----------
NSH=$((NCHIP * NW))
p4_winner () {
  # Winner-divergence fix (2026-08-29, autonomous mode): the marker in GCS is
  # AUTHORITATIVE once written — a re-entrant worker with incompletely-hydrated
  # local screens must never recompute a different winner (caught live 19:5xZ:
  # one relaunched worker's silent pull failure dropped C3 from its local view
  # and it scanned C1s1 into the pinned partition slots; zero summaries banked,
  # partials quarantined). Recompute only when no marker exists.
  local mw
  if mw=$(gsutil -q cp "$GCS/p4winner.txt" - 2>/dev/null | head -1) && [ -n "$mw" ]; then
    echo "$mw"; return 0
  fi
  for t2 in $SCREEN_TASKS; do IFS=: read -r _ a2 ck2 <<< "$t2"
    obj2=$(task_obj "$t2"); d2=runs/sxscreen_p${R_TAG}${a2}_${ck2}
    [ -d "$d2" ] || { gsutil -q cp "$GCS/$obj2" /tmp/_s.tgz 2>/dev/null && [ -s /tmp/_s.tgz ] && tar xzf /tmp/_s.tgz 2>/dev/null || true; }
  done
  for a2 in $ALL_ARMS; do
    [ -f "runs/sxeval_p${R_TAG}$a2/retfm_t8/summary_all.json" ] || \
      { gsutil -q cp "$GCS/${a2}_evalcheap.tgz" /tmp/_ec.tgz 2>/dev/null && tar xzf /tmp/_ec.tgz 2>/dev/null || true; }
  done
  python3 - <<'PY'
import json, os
from pathlib import Path
tag = os.environ.get("R_TAG", "sportBr2")
best, bestv = "", -1.0
for d in Path("runs").glob(f"sxscreen_p{tag}*_vb"):
    arm = d.name.replace(f"sxscreen_p{tag}", "")[:-3]
    try:
        sj = json.loads((d / "summary_all.json").read_text())
        rj = json.loads(Path(f"runs/sxeval_p{tag}{arm}/retfm_t8/summary_all.json").read_text())
    except Exception: continue
    if rj.get("exact_acc", 0) < 0.5: continue
    v = sj.get("vote_at_k", {}).get("256", sj.get("exact_acc_vote", 0)) or 0
    if v > bestv: best, bestv = arm, v
print(best or "-")
PY
}
if ! gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null; then
  for w8 in $(seq 1 "${P4_WAIT_PASSES:-90}"); do all_scr=1
    for t2 in $SCREEN_TASKS; do gsutil -q stat "$GCS/$(task_obj "$t2")" 2>/dev/null || all_scr=0; done
    [ "$all_scr" -eq 1 ] && break; sleep "${P4_POLL_SLEEP:-60}"; done
  WINNER=$(p4_winner | tail -1)
  if [ "$WINNER" != "-" ] && [ -n "$WINNER" ]; then
    echo "$WINNER" | gsutil -q cp - "$GCS/p4winner.txt" 2>/dev/null || true
    need_arm_local "$WINNER"
    D=runs/pretrain${R_TAG}_$WINNER; step=$(vb_step "$WINNER"); CK="$D/ckpt_$step.pkl"; [ -f "$CK" ] || CK="$D/ckpt_latest.pkl"
    O=runs/sxbreadth20k_p${R_TAG}${WINNER}; mkdir -p "$O"; partial_restore "$O"
    # PARTITION PIN (pre-mortem catch 2026-08-27): NSH is pinned in GCS on first
    # entry and REUSED by every resume — a node-shape change mid-PHASE4 (16->8
    # or 8->16) must NOT recompute the partition or banked /K shards would mix
    # partitions (the b589334/hollow-merge class, now shape-proofed).
    if NSHP=$(gsutil -q cp "$GCS/p4/NSH.txt" - 2>/dev/null | head -1) && [ -n "$NSHP" ]; then
      NSH=$NSHP; echo "P4 partition pinned: $NSH-way (from GCS)"
    else
      echo "$NSH" | gsutil -q cp - "$GCS/p4/NSH.txt" 2>/dev/null || true
      echo "P4 partition pinned: $NSH-way (fresh)"
    fi
    echo "PHASE4-COOP: winner $WINNER (vb) — $NSH-way, per-shard claims (any worker takes any unbanked shard) $(date -u +%H:%M)"
    partial_sync "$O" & PS4=$!
    pids=()
    slot=0
    for K in $(seq 0 $((NSH-1))); do
      gsutil -q stat "$GCS/p4/summary_s$K.json" 2>/dev/null && continue
      sc="claim_p4_s$K"
      if gsutil -q stat "$GCS/$sc" 2>/dev/null; then
        cts=$(gsutil -q cp "$GCS/$sc" - 2>/dev/null | head -1)
        [ -n "$cts" ] && [ $(( $(date -u +%s) - cts )) -lt "$CLAIM_TTL" ] 2>/dev/null && continue
        echo "CLAIM-STALE p4 s$K — taking over"
      fi
      date -u +%s | gsutil -q cp - "$GCS/$sc" 2>/dev/null || true
      c=$((slot % NCHIP)); slot=$((slot+1))
      ( for try in $(seq 1 60); do
          pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --shard "$K/$NSH" --out "$O" --bank-every 300 --batch "$EVAL_BATCH" \
              --split test --subsample "$SX_SUB" --t-total 64 --k-init "$SX_SUB_K" > "$O/shard_s$K.log" 2>&1 \
            && { gsutil -q cp "$O/summary_s$K.json" "$GCS/p4/summary_s$K.json"; gsutil -q cp "$O/records_s$K.npz" "$GCS/p4/records_s$K.npz"; echo "P4-SHARD-s$K-OK $(date -u +%H:%M)"; break; }
          if grep -qE "resource busy|Couldn't open iommu group" "$O/shard_s$K.log"; then sleep "${SHARD_RETRY_SLEEP:-120}"; continue; fi
          echo "P4-SHARD-s$K-FAILED"; break
        done; gsutil -q rm "$GCS/$sc" 2>/dev/null || true ) & pids+=($!)
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
    pkill -P $PS4 2>/dev/null; kill $PS4 2>/dev/null || true
    for w8 in $(seq 1 "${P4_WAIT_PASSES2:-120}"); do
      n=$(gsutil ls "$GCS/p4/summary_s*.json" 2>/dev/null | wc -l | tr -d ' ')
      [ "$n" -ge "$NSH" ] && break; sleep "${P4_POLL_SLEEP:-60}"
    done
    if [ "$n" -ge "$NSH" ] && ! gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null; then
      for f in $(gsutil ls "$GCS/p4/summary_s*.json" "$GCS/p4/records_s*.npz" 2>/dev/null); do
        b=$(basename "$f"); [ -f "$O/$b" ] || gsutil -q cp "$f" "$O/$b"
      done
      JAX_PLATFORMS=cpu python3 tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
      if [ -f "$O/summary_all.json" ] && ! python3 -c "import json,sys;sys.exit(0 if json.load(open('$O/summary_all.json'))['n']==$SX_SUB else 1)"; then
        echo "P4-MERGE-N-BAD (n != $SX_SUB) — refusing to bank (partition integrity gate)"; rm -f "$O/summary_all.json"
      fi
      if [ -f "$O/summary_all.json" ]; then
        mv2=$(python3 -c "import json;from pathlib import Path
try:
    m=json.loads(Path('runs/sxscreen_p${R_TAG}${WINNER}_m1/summary_all.json').read_text())
    v=json.loads(Path('runs/sxscreen_p${R_TAG}${WINNER}_vb/summary_all.json').read_text())
    print(abs((v.get('vote_at_k',{}).get('256',0) or 0)-(m.get('vote_at_k',{}).get('256',0) or 0)))
except Exception: print(0)")
        if python3 -c "import sys; sys.exit(0 if float('$mv2') >= 0.05 else 1)"; then
          ms=$(mstep "$WINNER" m1)
          if [ -n "$ms" ] && [ -f "$D/ckpt_$(printf '%06d' "$ms").pkl" ]; then
            echo "PHASE4-MID: |vb-m1| screens differ ${mv2} — scanning m1 ckpt $ms"
            sharded_eval "runs/sxbreadth20k_p${R_TAG}${WINNER}_mid" "$D/ckpt_$(printf '%06d' "$ms").pkl" --split test --subsample "$SX_SUB" --t-total 64 --k-init "$SX_SUB_K" || echo "PHASE4-MID-FAILED"
          fi
        fi
        tar czf /tmp/breadth20k.tgz runs/sxbreadth20k_p${R_TAG}* 2>/dev/null && gsutil -q cp /tmp/breadth20k.tgz "$GCS/breadth20k.tgz" && echo "PHASE4-OK $WINNER (coop ${NSH}-way) $(date -u +%H:%M)"
      else echo "PHASE4-MERGE-FAILED"; fi
    fi
  else echo "PHASE4-SKIP (no screened arm passed the guard)"; fi
fi

# ---------- P4DEPTH (after PHASE4: winner full-test COLD t=256; claim-guarded) ----------
if gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null && ! gsutil -q stat "$GCS/depth_t256.tgz" 2>/dev/null; then
  for dpass in $(seq 1 "${P4DEPTH_PASSES:-30}"); do
    gsutil -q stat "$GCS/depth_t256.tgz" 2>/dev/null && break
    claim="claim_depth_t256"
    if gsutil -q stat "$GCS/$claim" 2>/dev/null; then
      cts=$(gsutil -q cp "$GCS/$claim" - 2>/dev/null | head -1)
      if [ -n "$cts" ] && [ $(( $(date -u +%s) - cts )) -lt "$CLAIM_TTL" ] 2>/dev/null; then sleep "${P4_POLL_SLEEP:-60}"; continue; fi
      echo "CLAIM-STALE p4depth — taking over"
    fi
    date -u +%s | gsutil -q cp - "$GCS/$claim" 2>/dev/null || true
    run_task p4depth || true
    gsutil -q rm "$GCS/$claim" 2>/dev/null || true
  done
fi

# ---------- COMPLETION GUARD (global; hydrates from GCS — no hollow final) ----------
missing=""
for a in $ALL_ARMS; do
  gsutil -q stat "$GCS/${a}_ckpt.pkl" 2>/dev/null || missing="$missing $a:ckpt"
  gsutil -q stat "$GCS/${a}_evalcheap.tgz" 2>/dev/null || missing="$missing $a:evalcheap"
done
for t in $TASKS p4depth; do gsutil -q stat "$GCS/$(task_obj "$t")" 2>/dev/null || missing="$missing $t"; done
gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null || missing="$missing phase4"
if [ -n "$missing" ]; then echo "$SENT-WORKER-DONE worker=$W missing(other workers or failed):$missing $(date -u +%FT%TZ)"; exit 0; fi
for a in $ALL_ARMS; do
  need_arm_local "$a" || true
  D=runs/pretrain${R_TAG}_$a
  [ -f "$D/metrics.jsonl" ] || gsutil -q cp "$GCS/${a}_metrics.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
  [ -d "runs/sxeval_p${R_TAG}$a" ] || { gsutil -q cp "$GCS/${a}_evalcheap.tgz" /tmp/_e.tgz && tar xzf /tmp/_e.tgz; }
done
for t in $TASKS p4depth $OPTIONAL_TASKS; do
  obj=$(task_obj "$t")
  gsutil -q cp "$GCS/$obj" /tmp/_t.tgz 2>/dev/null && [ -s /tmp/_t.tgz ] && tar xzf /tmp/_t.tgz 2>/dev/null || true
done
gsutil -q cp "$GCS/breadth20k.tgz" /tmp/_b.tgz 2>/dev/null && tar xzf /tmp/_b.tgz 2>/dev/null || true
cache_push
tar czf /tmp/$FINAL_OBJ runs/pretrain${R_TAG}_*/ckpt_*.pkl runs/pretrain${R_TAG}_*/metrics.jsonl runs/pretrain${R_TAG}_*/val_best.txt runs/pretrain${R_TAG}_*/config.json runs/*_p${R_TAG}* runs/sxd3demo_* runs/wave_*.log 2>/dev/null
gsutil -q cp /tmp/$FINAL_OBJ "$GCS/$FINAL_OBJ" && echo "RESCUE-OK"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
if [ "${SELF_TEARDOWN:-0}" = 1 ] && [ -n "${SELF_POD:-}" ] && [ -n "${SELF_ZONE:-}" ]; then
  echo "SELF-TEARDOWN: deleting $SELF_POD in $SELF_ZONE (all artifacts banked) $(date -u +%FT%TZ)"; sleep 20
  gcloud compute tpus tpu-vm delete "$SELF_POD" --zone "$SELF_ZONE" --quiet >/dev/null 2>&1 && echo "SELF-TEARDOWN-ISSUED" || echo "SELF-TEARDOWN-FAILED (supervisor/watchdog will tear down)"
fi
