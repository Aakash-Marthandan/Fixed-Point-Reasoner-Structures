#!/bin/bash
# Ledger: PHASE B RUNG 1 (2026-08-24 launch registration) — the parity ladder's
# first rung: d64 FULL-WIDTH (--width-scale 4; n_bulk 959,482) on ONE spot pod
# (v6e-16 = 4 hosts x 4 chips, or v6e-8 = 1 x 8; chip count self-detected),
# pod.sh-compatible (same launch contract as chain_sport3a.sh). Goal metrics:
# 20k vote@128 @t64 (EqR like-for-like; bands B-M1 .50 / B-M2 .85 / B-M3 .95)
# + full-test cold (M-bands). Arms = the wave-3a reference recipes scaled in
# WIDTH ONLY (one variable; steps/T/aug/lr per their d16/d32 reference cells):
#   B1  A2-recipe:  plain T12 RI.5 NI.01 @50k   (the wave-3a recipe decision)
#   B2  A3-recipe:  plain T12 FPA k4    @50k    (seed-stable stabilizer alternate)
#   B3  A5-recipe:  priced T12          @50k    (stability/compression twin)
#   B4  S5-recipe:  plain T6            @20k    (the wide-funnel carrier) + B4s1 seed 1
#   B5  W8-recipe:  priced T6           @20k    (priced x wide-funnel at width)
# Pretraining runs DP over the worker's whole host (--dp; batch 64 sharded;
# gradient-equivalence tested), NO remat (auto-RETRY once with --remat on HBM
# OOM — numerics-equivalent per test_eq_remat_matches_no_remat). Monitor every
# 5k + banked ckpts every 5k (wave-3a cadence: cross-wave comparability).
# PHASE2 = a GLOBAL ready-first CLAIM QUEUE over all eval work (no per-worker
# poles; chips stay busy — PI utilization policy 2026-08-23):
#   screens: strat-512 k=256 @t64 on the VAL-SELECTED and one MID banked ckpt
#            per arm (H-46: does the funnel narrow with training at width?)
#   fulls:   B1-B4 full-test t6 + t64 (final) + t64 (val-best, when it differs);
#            B4s1/B5 full-test t64 (final) only (registered protocol trim)
#   probes4: probe suite on B1-B4 in parallel (one chip each; s1/B5 skipped —
#            the evaluator carries ret/retfm/mi at strat level)
#   PHASE4:  screen winner (argmax screen-vb vote@256, retfm>=.5 guard) ->
#            20k-subsample k=128 @t64 (protocol-matched to S5's banked
#            confirmation); + the MID ckpt too iff |vb-mid| vote@256 >= 5pp
# COMPLETION GUARD -> sportB_final.tgz + CHAIN-SPORTB-COMPLETE (+ self-teardown).
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
R_TAG=${R_TAG:-sportB}; export R_TAG
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
GCS=${GCS:-gs://qhrrn2-rescue/sportB}; FINAL_OBJ=${FINAL_OBJ:-sportB_final.tgz}
GCS_W1=${GCS_W1:-gs://qhrrn2-rescue/sport2}
SX_NPZ=${SX_NPZ:-sudoku_extreme_seed0.npz}
SX_AUG=${SX_AUG:-100}; SX_T_STRAT=${SX_T_STRAT:-"6 64 256"}
SX_K_INIT=${SX_K_INIT:-16}; SX_STRAT=${SX_STRAT:-512}
SX_SUB=${SX_SUB:-20000}; SX_SUB_K=${SX_SUB_K:-128}; SX_RET_T=${SX_RET_T:-8}
SCREEN_K=${SCREEN_K:-256}
MON_EVERY=${MON_EVERY:-5000}
WS=${WS:-4}; RD=${RD:-64}
if [ -z "${NCHIP:-}" ]; then NCHIP=$(ls /dev/vfio 2>/dev/null | grep -cE '^[0-9]+$'); fi
[ "${NCHIP:-0}" -ge 1 ] 2>/dev/null || NCHIP=8
SENT=CHAIN-SPORTB
NPZ=data/sudoku_extreme/$SX_NPZ
mkdir -p runs data/sudoku_extreme

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
PRIMARY=${PRIMARY:-"B1 B2 B3 B4"}
echo "=== SPORTB START $(date -u +%FT%TZ) worker=$W/$NW chips=$NCHIP my_arms=[$MY_JOBS] all_arms=[$ALL_ARMS] d=$RD ws=$WS ==="
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/$SX_NPZ" "$NPZ" || { echo "NPZ-MISSING $SX_NPZ"; exit 2; }

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest "$@"; }
is_primary () { case " $PRIMARY " in *" $1 "*) return 0;; *) return 1;; esac; }
arm_flags () {   # WIDTH-ONLY scaling of the registered reference recipes (seed via s1/s2 suffix)
  case $1 in
    B1) echo "--d $RD --width-scale $WS --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --ri-p 0.5 --ni-sigma 0.01";;
    B2) echo "--d $RD --width-scale $WS --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0";;
    B3) echo "--d $RD --width-scale $WS --T 12 --steps 50000 --beta-flux 3e-5 --beta-flux-nl 1e-5";;
    B4) echo "--d $RD --width-scale $WS --T 6 --steps 20000 --beta-flux 0 --beta-flux-nl 0";;
    B5) echo "--d $RD --width-scale $WS --T 6 --steps 20000 --beta-flux 3e-5 --beta-flux-nl 1e-5";;
    B*s1) echo "$(arm_flags "${1%s1}") --seed 1";;
    B*s2) echo "$(arm_flags "${1%s2}") --seed 2";;
    *) echo "UNKNOWN-ARM $1" >&2; return 1;;
  esac
}

sync_loop () {  # ARM DIR — every 5 min bank ckpt_latest + metrics + banked ckpts
  local arm=$1 D=$2
  while true; do sleep 300
    gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt_live.pkl" 2>/dev/null || true
    gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics_live.jsonl" 2>/dev/null || true
    for f in "$D"/ckpt_0*.pkl; do [ -f "$f" ] && gsutil -q cp -n "$f" "$GCS/${arm}_$(basename "$f")" 2>/dev/null; done
  done
}
run_pretrain () {  # ARM DIR EXTRA -> rc  (DP over the whole worker host)
  local arm=$1 D=$2 extra=$3
  # On a MULTI-HOST slice each worker's pretrain must be confined to ITS OWN host
  # (a 2x2 single-process system), else the 4 unconfined processes try to form the
  # global 16-chip system and pmap's local replica groups clash with global device
  # ids (RET_CHECK device_id, seen at launch 2026-08-24 12:19Z). Same mechanism as
  # pin(), widened to the host's 4 chips. Single-host (v6e-8, NW=1): no confinement
  # (global == local; the P11-EXT proven path).
  local conf=""
  [ "$NW" -ge 2 ] && conf="TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3"
  # shellcheck disable=SC2086
  env $conf JAX_DEFAULT_MATMUL_PRECISION=highest python3 tools/pretrain.py --out "$D" --equilibrium --anchor-p 0.3 \
      --sudoku-extreme "$NPZ" --sudoku-aug "$SX_AUG" --n-val 64 --seed 0 --dp \
      --monitor-every "$MON_EVERY" $(arm_flags "$arm") $extra > "runs/wave_pre_$arm.log" 2>&1
}
pretrain_one () {   # ARM -> 0 ok
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
      || echo "PRETRAIN-$arm-DIVERGED last_loss=${lastloss:-nan} (report to PI; registered contingency = ONE relaunch at lr 5e-4, manual)"
    touch "$D/.done"; gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt.pkl"; gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics.jsonl"
    for f in "$D"/ckpt_0*.pkl; do [ -f "$f" ] && gsutil -q cp -n "$f" "$GCS/${arm}_$(basename "$f")" 2>/dev/null; done
    python3 tools/select_ckpt.py "$D" > "$D/val_best.txt" 2>"runs/wave_sel_$arm.log" \
      && { echo "VALBEST-$arm $(cat "$D/val_best.txt")"; gsutil -q cp "$D/val_best.txt" "$GCS/${arm}_val_best.txt"; }
    echo "PRETRAIN-$arm-OK $(date -u +%H:%M)"
  else echo "PRETRAIN-$arm-FAILED rc=$rc (see runs/wave_pre_$arm.log)"; fi
  return $rc
}
eval_cheap () {   # ARM — strat t6/64/256 k16, val t64, ret_t8, retfm_t8; parallel over this worker's chips
  local arm=$1 D=runs/pretrain${R_TAG}_$1 O=runs/sxeval_p${R_TAG}$1 t i=0 pids=()
  mkdir -p "$O"
  [ -n "$(ls "$O" 2>/dev/null)" ] || { gsutil -q cp "$GCS/${arm}_evalcheap.tgz" "/tmp/sxec_$arm.tgz" 2>/dev/null && tar xzf "/tmp/sxec_$arm.tgz" && echo "RESTORE-$arm cheap evals from GCS"; }
  ec_one () { local kind=$1 c=$2; shift 2
    [ -f "$O/$kind/summary_all.json" ] && return 0
    pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" --out "$O/$kind" "$@" \
      > "runs/wave_ev_${arm}_$kind.log" 2>&1 || echo "EVAL-$arm-$kind-FAILED"
  }
  for t in $SX_T_STRAT; do ec_one "strat_t$t" $((i % NCHIP)) --split test --stratified "$SX_STRAT" --t-total "$t" --k-init "$SX_K_INIT" & pids+=($!); i=$((i+1)); done
  ec_one "val_t64" $((i % NCHIP)) --split val --t-total 64 --k-init 0 & pids+=($!); i=$((i+1))
  ec_one "ret_t$SX_RET_T" $((i % NCHIP)) --split test --stratified "$SX_STRAT" --t-total "$SX_RET_T" --k-init 0 --init solution & pids+=($!); i=$((i+1))
  ec_one "retfm_t$SX_RET_T" $((i % NCHIP)) --split test --stratified "$SX_STRAT" --t-total "$SX_RET_T" --k-init 0 --init solution --final-map-only & pids+=($!)
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
  tar czf "/tmp/sxec_$arm.tgz" "$O" && gsutil -q cp "/tmp/sxec_$arm.tgz" "$GCS/${arm}_evalcheap.tgz"
  echo "EVALCHEAP-$arm-OK $(date -u +%H:%M)"
}

# ---------- sharded helper (all chips of this worker) ----------
sharded_eval () {  # OUT CKPT [extra flags...] -> merged summary_all.json in OUT
  local O=$1 CK=$2; shift 2
  [ -f "$O/summary_all.json" ] && return 0
  mkdir -p "$O"; local pids=() c
  for c in $(seq 0 $((NCHIP-1))); do
    [ -f "$O/summary_s$c.json" ] && continue
    ( for try in $(seq 1 60); do
        pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --shard "$c/$NCHIP" --out "$O" "$@" > "$O/shard_s$c.log" 2>&1 && break
        if grep -qE "resource busy|Couldn't open iommu group" "$O/shard_s$c.log"; then [ "$try" -eq 1 ] && echo "SHARD-WAIT $O s$c (chip busy; retrying)"; sleep "${SHARD_RETRY_SLEEP:-120}"; continue; fi
        echo "SHARD-FAILED $O s$c"; break
      done ) & pids+=($!)
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
  JAX_PLATFORMS=cpu python3 tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  [ -f "$O/summary_all.json" ]
}

# ---------- PHASE1: this worker's arms (sequential; DP over all its chips) ----------
for arm in $MY_JOBS; do
  if pretrain_one "$arm"; then eval_cheap "$arm"; fi
done
echo "QUEUES-DONE worker=$W $(date -u +%H:%M)"

# ---------- PHASE2: GLOBAL ready-first claim queue over all eval work ----------
vb_step () { cut -d' ' -f1 "runs/pretrain${R_TAG}_$1/val_best.txt" 2>/dev/null; }
mid_step () {  # banked step closest to half the val-best step (5k grid, min 5000); empty if == vb
  python3 - "$1" <<'PY'
import sys
try: vb = int(open(f"runs/pretrain{__import__('os').environ.get('R_TAG','sportB')}_{sys.argv[1]}/val_best.txt").read().split()[0])
except Exception: sys.exit(0)
mid = max(5000, round(vb / 2 / 5000) * 5000)
print(f"{mid:06d}" if mid != vb else "")
PY
}
need_arm_local () {  # pull an arm's dir (ckpt + banked + val_best) from GCS if absent
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  mkdir -p "$D"
  [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" || return 1
  [ -f "$D/val_best.txt" ] || gsutil -q cp "$GCS/${arm}_val_best.txt" "$D/val_best.txt" 2>/dev/null || true
  for f in $(gsutil ls "$GCS/${arm}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${arm}_}" ] || gsutil -q cp "$f" "$D/${b#${arm}_}"; done
  return 0
}
task_obj () {  # canonical GCS object name for a finished task
  case $1 in
    scr:*) IFS=: read -r _ a ck <<< "$1"; echo "screen_${a}_${ck}_k${SCREEN_K}.tgz";;
    full:*) IFS=: read -r _ a kind <<< "$1"; echo "full_${a}_${kind}.tgz";;
    probes4) echo "probes4.tgz";;
    phase4) echo "breadth20k.tgz";;
  esac
}
task_ready () {  # arm ckpt banked in GCS?  (phase4 additionally needs every screen)
  case $1 in
    probes4) local a; for a in $PRIMARY; do gsutil -q stat "$GCS/${a}_ckpt.pkl" 2>/dev/null || return 1; done; return 0;;
    phase4)  local t; for t in $SCREEN_TASKS; do gsutil -q stat "$GCS/$(task_obj "$t")" 2>/dev/null || return 1; done; return 0;;
    *) IFS=: read -r _ a _ <<< "$1"; gsutil -q stat "$GCS/${a}_ckpt.pkl" 2>/dev/null;;
  esac
}
run_task () {
  local t=$1 obj; obj=$(task_obj "$t")
  case $t in
    scr:*)
      IFS=: read -r _ arm ck <<< "$t"; need_arm_local "$arm" || return 1
      local D=runs/pretrain${R_TAG}_$arm CK step lbl=$ck
      if [ "$ck" = vb ]; then step=$(vb_step "$arm"); else step=$(mid_step "$arm"); fi
      [ -n "$step" ] || { echo "SCREEN-$arm-$ck-SKIP (no distinct step)"; gsutil -q cp /dev/null "$GCS/$obj" 2>/dev/null || true; return 0; }
      CK="$D/ckpt_$step.pkl"; [ "$step" = "$(printf '%06d' "$(python3 -c "import pickle;print(pickle.load(open('$D/ckpt_latest.pkl','rb'))['step'])")")" ] && CK="$D/ckpt_latest.pkl"
      [ -f "$CK" ] || { echo "SCREEN-$arm-$ck-NOCKPT step=$step"; return 1; }
      local O=runs/sxscreen_p${R_TAG}${arm}_${lbl}
      if sharded_eval "$O" "$CK" --split test --stratified "$SX_STRAT" --t-total 64 --k-init "$SCREEN_K"; then
        echo "$step" > "$O/step.txt"; tar czf "/tmp/$obj" "$O" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "SCREEN-$arm-$ck-OK step=$step $(date -u +%H:%M)"
      else echo "SCREEN-$arm-$ck-FAILED"; return 1; fi;;
    full:*)
      IFS=: read -r _ arm kind <<< "$t"; need_arm_local "$arm" || return 1
      local D=runs/pretrain${R_TAG}_$arm O=runs/sxeval_p${R_TAG}$arm CK tt
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
            --stratified "$SX_STRAT" --t-total 64 --k-init 16 --out "$PD" > "runs/wave_pr_$a.log" 2>&1 || echo "PROBE-$a-FAILED"; } & pids+=($!)
        c=$(( (c+1) % NCHIP ))
      done
      for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
      tar czf "/tmp/$obj" runs/sudprobe_p${R_TAG}* 2>/dev/null && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "PROBES4-OK $(date -u +%H:%M)";;
    phase4)
      for t2 in $SCREEN_TASKS; do IFS=: read -r _ a ck <<< "$t2"
        obj2=$(task_obj "$t2"); d2=runs/sxscreen_p${R_TAG}${a}_${ck}
        [ -d "$d2" ] || { gsutil -q cp "$GCS/$obj2" /tmp/_s.tgz 2>/dev/null && tar xzf /tmp/_s.tgz 2>/dev/null || true; }
      done
      read -r WINNER WINCK ALSO <<< "$(python3 - <<'PY'
import json, os
from pathlib import Path
tag = os.environ.get("R_TAG", "sportB")
best, bestv = "", -1.0
for d in Path("runs").glob(f"sxscreen_p{tag}*_vb"):
    arm = d.name.replace(f"sxscreen_p{tag}", "")[:-3]
    s = d / "summary_all.json"
    r = Path(f"runs/sxeval_p{tag}{arm}/retfm_t8/summary_all.json")
    try:
        sj = json.loads(s.read_text()); rj = json.loads(r.read_text())
    except Exception: continue
    if rj.get("exact_acc", 0) < 0.5: continue
    v = sj.get("vote_at_k", {}).get("256", sj.get("exact_acc_vote", 0)) or 0
    if v > bestv: best, bestv = arm, v
also = ""
if best:
    try:
        m = json.loads(Path(f"runs/sxscreen_p{tag}{best}_mid/summary_all.json").read_text())
        vm = m.get("vote_at_k", {}).get("256", m.get("exact_acc_vote", 0)) or 0
        if abs(bestv - vm) >= 0.05: also = "mid"
    except Exception: pass
print(best or "-", "vb", also or "-")
PY
)"
      [ "$WINNER" != "-" ] || { echo "PHASE4-SKIP (no screened arm passed the guard)"; gsutil -q cp /dev/null "$GCS/$obj" 2>/dev/null || true; return 0; }
      need_arm_local "$WINNER" || return 1
      local D=runs/pretrain${R_TAG}_$WINNER step CK
      step=$(vb_step "$WINNER"); CK="$D/ckpt_$step.pkl"; [ -f "$CK" ] || CK="$D/ckpt_latest.pkl"
      echo "PHASE4: screen winner $WINNER (vb) -> ${SX_SUB} k=$SX_SUB_K t=64 $(date -u +%H:%M)"
      local O=runs/sxbreadth20k_p${R_TAG}${WINNER}
      sharded_eval "$O" "$CK" --split test --subsample "$SX_SUB" --t-total 64 --k-init "$SX_SUB_K" || { echo "PHASE4-FAILED"; return 1; }
      if [ "$ALSO" = mid ]; then
        local ms; ms=$(mid_step "$WINNER")
        [ -n "$ms" ] && [ -f "$D/ckpt_$ms.pkl" ] && { echo "PHASE4-MID: |vb-mid| screen delta >= 5pp -> scanning mid ckpt $ms (H-46 full-test-grade)"
          sharded_eval "runs/sxbreadth20k_p${R_TAG}${WINNER}_mid" "$D/ckpt_$ms.pkl" --split test --subsample "$SX_SUB" --t-total 64 --k-init "$SX_SUB_K" || echo "PHASE4-MID-FAILED"; }
      fi
      tar czf "/tmp/$obj" runs/sxbreadth20k_p${R_TAG}* 2>/dev/null && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "PHASE4-OK $WINNER $(date -u +%H:%M)";;
  esac
}
SCREEN_TASKS=""; FULL_TASKS=""
for a in $ALL_ARMS; do SCREEN_TASKS="$SCREEN_TASKS scr:$a:vb scr:$a:mid"; done
for a in $ALL_ARMS; do
  if is_primary "$a"; then FULL_TASKS="$FULL_TASKS full:$a:t64 full:$a:t6 full:$a:vb"
  else FULL_TASKS="$FULL_TASKS full:$a:t64"; fi
done
TASKS="$FULL_TASKS $SCREEN_TASKS probes4 phase4"
for pass in $(seq 1 200); do
  pending=0
  for t in $TASKS; do
    obj=$(task_obj "$t"); claim="claim_${obj%.tgz}"
    gsutil -q stat "$GCS/$obj" 2>/dev/null && continue
    task_ready "$t" || { pending=1; continue; }
    if gsutil -q stat "$GCS/$claim" 2>/dev/null; then pending=1; continue; fi
    echo "worker $W" | gsutil -q cp - "$GCS/$claim" 2>/dev/null || true
    run_task "$t" || pending=1
    gsutil -q rm "$GCS/$claim" 2>/dev/null || true
  done
  [ "$pending" -eq 0 ] && break
  sleep "${PHASE2_SLEEP:-120}"
done
echo "PHASE2-DONE worker=$W $(date -u +%H:%M)"

# ---------- COMPLETION GUARD (global) ----------
missing=""
for a in $ALL_ARMS; do
  gsutil -q stat "$GCS/${a}_ckpt.pkl" 2>/dev/null || missing="$missing $a:ckpt"
  gsutil -q stat "$GCS/${a}_evalcheap.tgz" 2>/dev/null || missing="$missing $a:evalcheap"
done
for t in $TASKS; do gsutil -q stat "$GCS/$(task_obj "$t")" 2>/dev/null || missing="$missing $t"; done
if [ -n "$missing" ]; then echo "$SENT-WORKER-DONE worker=$W missing(other workers or failed):$missing $(date -u +%FT%TZ)"; exit 0; fi
for a in $ALL_ARMS; do
  need_arm_local "$a" || true
  D=runs/pretrain${R_TAG}_$a
  [ -f "$D/metrics.jsonl" ] || gsutil -q cp "$GCS/${a}_metrics.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
  [ -d "runs/sxeval_p${R_TAG}$a" ] || { gsutil -q cp "$GCS/${a}_evalcheap.tgz" /tmp/_e.tgz && tar xzf /tmp/_e.tgz; }
done
for t in $TASKS; do
  obj=$(task_obj "$t")
  gsutil -q cp "$GCS/$obj" /tmp/_t.tgz 2>/dev/null && [ -s /tmp/_t.tgz ] && tar xzf /tmp/_t.tgz 2>/dev/null || true
done
tar czf /tmp/$FINAL_OBJ runs/pretrain${R_TAG}_*/ckpt_*.pkl runs/pretrain${R_TAG}_*/metrics.jsonl runs/pretrain${R_TAG}_*/val_best.txt runs/pretrain${R_TAG}_*/config.json runs/*_p${R_TAG}* runs/wave_*.log 2>/dev/null
gsutil -q cp /tmp/$FINAL_OBJ "$GCS/$FINAL_OBJ" && echo "RESCUE-OK"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
if [ "${SELF_TEARDOWN:-0}" = 1 ] && [ -n "${SELF_POD:-}" ] && [ -n "${SELF_ZONE:-}" ]; then
  echo "SELF-TEARDOWN: deleting $SELF_POD in $SELF_ZONE (all artifacts banked) $(date -u +%FT%TZ)"; sleep 20
  gcloud compute tpus tpu-vm delete "$SELF_POD" --zone "$SELF_ZONE" --quiet >/dev/null 2>&1 && echo "SELF-TEARDOWN-ISSUED" || echo "SELF-TEARDOWN-FAILED (supervisor/watchdog will tear down)"
fi
