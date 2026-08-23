#!/bin/bash
# Ledger: SPRINT S2 WAVE 3a (2026-08-23 launch registration) — "fix, confirm,
# instrument" on ONE spot pod (v6e-8 = 8 chips, or v6e-16 = 4 hosts x 4 chips; the
# chip count is self-detected), pod.sh-compatible (same launch contract as
# chain_sport2w2.sh). Goal metric: Sudoku-Extreme full-test exact (cold) + the
# labeled verify-and-vote breadth number (the EqR-comparable protocol is t=64,
# k=128). Built on the wave-2 verdict (H-45 contractivity collapse; H-44 price as
# a contractive regulariser; S5's strat-512 vote@128 = 70.7 %):
#   PHASE0 (all chips, sharded): BREADTH CONFIRMATION on the banked wave-1 S5 map —
#          bscan:S5:20000:128  bscan:S5:20000:256  (20k seeded subsample @t64)
#          bscan:S5:strat:1024 (strat-512 @t64, saturation)
#   ARMS (one per chip; pretrain with the TRAJECTORY MONITOR every 5k steps and
#          banked ckpts every 5k; then strat t6/64/256 k16, val t64, ret_t8 (schedule),
#          retfm_t8 (FINAL-MAP retention, batched), probe (origin, non-coupled only)):
#     A2  plain T12 + RI .5 + NI .01 @50k     (the EqR recipe on the best cold base)
#     A3  plain T12 + FPA(k4, eps .2) @50k    (fixed-point anchor: the contractivity fix)
#     A4  plain T12 eq_coupled @50k           (FPRM contractive residual)
#     A5  priced T12 @50k                     (H-44 ceiling, depth)
#     A6  priced T6 d32 @50k                  (H-44 ceiling, width)
#     A7  plain T12 @50k  + A7s1 (seed 1)     (W2's recipe continued: does it collapse?)
#     A8  plain T6 aug500 + wd 1e-3 @50k      (conventional-regulariser control)
#   PHASE-F (all chips, sharded per arm): FULL-TEST evals t6 + t64 on the FINAL ckpt,
#          and t64 on the VAL-SELECTED ckpt (argmax val@t64 over banked ckpts, from
#          the monitor rows) when it differs — no single-chip eval poles.
#   PHASE4 (all chips): best arm (FINAL full-t64 exact) -> 20k subsample k=128 @t64.
#   COMPLETION GUARD -> final object + CHAIN-SPORT3A-COMPLETE (+ self-teardown).
# Live-sync every 5 min banks ckpt_latest + metrics.jsonl (+ banked ckpts) per arm,
# so the trajectory survives preemptions (wave-2 lesson).
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
R_TAG=${R_TAG:-sport3a}
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
GCS=${GCS:-gs://qhrrn2-rescue/sport3a}; FINAL_OBJ=${FINAL_OBJ:-sport3a_final.tgz}
GCS_W1=${GCS_W1:-gs://qhrrn2-rescue/sport2}; W1_TAG=${W1_TAG:-sport2}
SX_NPZ=${SX_NPZ:-sudoku_extreme_seed0.npz}
SX_AUG=${SX_AUG:-100}; SX_T_STRAT=${SX_T_STRAT:-"6 64 256"}; SX_T_FULL=${SX_T_FULL:-"6 64"}
SX_K_INIT=${SX_K_INIT:-16}; SX_STRAT=${SX_STRAT:-512}
SX_SUB=${SX_SUB:-20000}; SX_SUB_K=${SX_SUB_K:-128}; SX_RET_T=${SX_RET_T:-8}
MON_EVERY=${MON_EVERY:-5000}
if [ -z "${NCHIP:-}" ]; then NCHIP=$(ls /dev/vfio 2>/dev/null | grep -cE '^[0-9]+$'); fi
[ "${NCHIP:-0}" -ge 1 ] 2>/dev/null || NCHIP=8
SENT=CHAIN-SPORT3A
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
[ -n "$ALL_JOBS" ] || ALL_JOBS=${R0_ARMS:-}
[ "$NW" -ge 2 ] || MY_JOBS=$ALL_JOBS
ALL_ARMS=""; for j in $ALL_JOBS; do case $j in bscan:*) ;; *) ALL_ARMS="$ALL_ARMS $j";; esac; done
BSCANS=${BSCANS:-"bscan:S5:20000:128 bscan:S5:20000:256 bscan:S5:strat:1024"}
echo "=== SPORT3A START $(date -u +%FT%TZ) worker=$W/$NW chips=$NCHIP my_jobs=[$MY_JOBS] all_arms=[$ALL_ARMS] bscans=[$BSCANS] ==="
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/$SX_NPZ" "$NPZ" || { echo "NPZ-MISSING $SX_NPZ"; exit 2; }

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest "$@"; }
base_arm () { echo "$1" | sed -E 's/s[12]$//'; }
arm_flags () {   # the registered per-arm flags (everything else = the wave-2 base: equilibrium, anchors .3, aug 100, B64, seed 0)
  case $1 in
    A2)  echo "--d 16 --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --ri-p 0.5 --ni-sigma 0.01 --remat";;
    A3)  echo "--d 16 --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0 --remat";;
    A4)  echo "--d 16 --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --eq-coupled --remat";;
    A5)  echo "--d 16 --T 12 --steps 50000 --beta-flux 3e-5 --beta-flux-nl 1e-5 --remat";;
    A6)  echo "--d 32 --T 6 --steps 50000 --beta-flux 3e-5 --beta-flux-nl 1e-5 --remat";;
    A7)  echo "--d 16 --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --remat";;
    A8)  echo "--d 16 --T 6 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --sudoku-aug 500 --wd 1e-3";;
    A*s1) echo "$(arm_flags "${1%s1}") --seed 1";;
    A*s2) echo "$(arm_flags "${1%s2}") --seed 2";;
    *) echo "UNKNOWN-ARM $1" >&2; return 1;;
  esac
}
arm_probe_ok () { case "$(arm_flags "$1")" in *box4*|*eq-coupled*) return 1;; *) return 0;; esac; }

sync_loop () {  # ARM DIR — every 5 min bank ckpt_latest + metrics + banked ckpts (trajectory survives preemption)
  local arm=$1 D=$2
  while true; do sleep 300
    gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt_live.pkl" 2>/dev/null || true
    gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics_live.jsonl" 2>/dev/null || true
    for f in "$D"/ckpt_0*.pkl; do [ -f "$f" ] && gsutil -q cp -n "$f" "$GCS/${arm}_$(basename "$f")" 2>/dev/null; done
  done
}
pretrain_one () {   # ARM CHIP -> 0 ok
  local arm=$1 c=$2 D=runs/pretrain${R_TAG}_$1
  mkdir -p "$D"
  [ -f "$D/.done" ] && { echo "SKIP-$arm (done)"; return 0; }
  if gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" 2>/dev/null; then
    gsutil -q cp "$GCS/${arm}_metrics.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
    for f in $(gsutil ls "$GCS/${arm}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${arm}_}" ] || gsutil -q cp "$f" "$D/${b#${arm}_}"; done
    touch "$D/.done"; echo "SKIP-$arm (GCS complete)"; return 0
  fi
  if gsutil -q cp "$GCS/${arm}_ckpt_live.pkl" "$D/ckpt_latest.pkl" 2>/dev/null; then
    gsutil -q cp "$GCS/${arm}_metrics_live.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
    for f in $(gsutil ls "$GCS/${arm}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${arm}_}" ] || gsutil -q cp "$f" "$D/${b#${arm}_}"; done
    echo "RESUME-$arm from live ckpt (+metrics, +banked ckpts)"
  fi
  echo "=== PRETRAIN $arm $(date -u +%H:%M) === chip $c flags: $(arm_flags "$arm")"
  sync_loop "$arm" "$D" & local SY=$!
  # shellcheck disable=SC2086
  pin "$c" python3 tools/pretrain.py --out "$D" --equilibrium --anchor-p 0.3 --sudoku-extreme "$NPZ" \
      --sudoku-aug "$SX_AUG" --n-val 64 --seed 0 --monitor-every "$MON_EVERY" $(arm_flags "$arm") > "runs/wave_pre_$arm.log" 2>&1
  local rc=$?; pkill -P $SY 2>/dev/null; kill $SY 2>/dev/null || true
  if [ $rc -eq 0 ]; then
    touch "$D/.done"; gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt.pkl"; gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics.jsonl"
    for f in "$D"/ckpt_0*.pkl; do [ -f "$f" ] && gsutil -q cp -n "$f" "$GCS/${arm}_$(basename "$f")" 2>/dev/null; done
    python3 tools/select_ckpt.py "$D" > "$D/val_best.txt" 2>"runs/wave_sel_$arm.log" && echo "VALBEST-$arm $(cat "$D/val_best.txt")"
    echo "PRETRAIN-$arm-OK $(date -u +%H:%M)"
  else echo "PRETRAIN-$arm-FAILED rc=$rc (see runs/wave_pre_$arm.log)"; fi
  return $rc
}
eval_cheap () {   # ARM CHIP — strat t6/64/256 k16, val t64, ret_t8 (schedule), retfm_t8 (final map); full-test evals are PHASE-F
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
  [ -f "$O/retfm_t$SX_RET_T/summary_all.json" ] || pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" \
      --split test --stratified "$SX_STRAT" --t-total "$SX_RET_T" --k-init 0 --init solution --final-map-only --out "$O/retfm_t$SX_RET_T" > "runs/wave_ev_${arm}_rf.log" 2>&1 || echo "EVAL-$arm-retfm-FAILED"
  echo "EVALCHEAP-$arm-OK $(date -u +%H:%M)"
}
probe_one () {
  local arm=$1 c=$2 D=runs/pretrain${R_TAG}_$1 O=runs/sudprobe_p${R_TAG}$1
  if ! arm_probe_ok "$arm"; then echo "PROBE-SKIP-$arm (layout/coupled: evaluator carries retention/final-map/multi-init)"; return 0; fi
  [ -s "$O/results.jsonl" ] || { gsutil -q cp "$GCS/${arm}_probe.tgz" "/tmp/sxprobe_$arm.tgz" 2>/dev/null && tar xzf "/tmp/sxprobe_$arm.tgz" && echo "RESTORE-$arm probe from GCS"; }
  if pin "$c" python3 tools/probe_sudoku.py --ckpt "$D/ckpt_latest.pkl" --pairs-file "$NPZ" --split test \
        --stratified "$SX_STRAT" --t-total 64 --k-init 16 --out "$O" > "runs/wave_pr_$arm.log" 2>&1; then
    tar czf "/tmp/sxprobe_$arm.tgz" "$O" && gsutil -q cp "/tmp/sxprobe_$arm.tgz" "$GCS/${arm}_probe.tgz"; echo "PROBE-$arm-OK $(date -u +%H:%M)"
  else echo "PROBE-$arm-FAILED"; fi
}
arm_pipeline () { local arm=$1 c=$2; pretrain_one "$arm" "$c" || return 1; eval_cheap "$arm" "$c"; probe_one "$arm" "$c"; }

# ---------- sharded helpers (all chips of this worker) ----------
sharded_eval () {  # OUT CKPT [extra evaluator flags...] -> merged summary_all.json in OUT (skip if present)
  local O=$1 CK=$2; shift 2
  [ -f "$O/summary_all.json" ] && return 0
  mkdir -p "$O"; local pids=() c
  for c in $(seq 0 $((NCHIP-1))); do
    [ -f "$O/summary_s$c.json" ] && continue
    ( pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --shard "$c/$NCHIP" --out "$O" "$@" > "$O/shard_s$c.log" 2>&1 || echo "SHARD-FAILED $O s$c" ) & pids+=($!)
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
  JAX_PLATFORMS=cpu python3 tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  [ -f "$O/summary_all.json" ]
}
bscan_one () {   # bscan:ARM:SEL:K   (SEL = 20000 | strat)  on a banked WAVE-1 ckpt, sharded over all chips
  local spec=$1; IFS=: read -r _ arm sel k <<< "$spec"
  local D=runs/pretrain${W1_TAG}_$arm O obj
  if [ "$sel" = strat ]; then O=runs/sxbreadth_${arm}_t64_k${k}; obj=breadth_${arm}_t64_k${k}.tgz; else O=runs/sxbreadth${sel}_${arm}_k${k}; obj=breadth${sel}_${arm}_k${k}.tgz; fi
  gsutil -q stat "$GCS/$obj" 2>/dev/null && { echo "SKIP-$spec (GCS complete)"; return 0; }
  mkdir -p "$D"; [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS_W1/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" || { echo "BSCAN-$arm-NOCKPT"; return 1; }
  echo "=== BSCAN $spec $(date -u +%H:%M) === $NCHIP shards"
  local ok
  if [ "$sel" = strat ]; then sharded_eval "$O" "$D/ckpt_latest.pkl" --split test --stratified "$SX_STRAT" --t-total 64 --k-init "$k"; ok=$?
  else sharded_eval "$O" "$D/ckpt_latest.pkl" --split test --subsample "$sel" --t-total 64 --k-init "$k"; ok=$?; fi
  if [ $ok -eq 0 ]; then tar czf "/tmp/$obj" "$O" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "BSCAN-$spec-OK $(date -u +%H:%M)"; else echo "BSCAN-$spec-FAILED"; fi
}

# ---------- PHASE0: breadth confirmation (sharded, all chips) ----------
if [ "$NW" -lt 2 ] || [ "$W" -eq 0 ]; then for b in $BSCANS; do bscan_one "$b"; done; fi
echo "PHASE0-DONE $(date -u +%H:%M)"

# ---------- PHASE1-3: per-chip arm pipelines (round-robin) ----------
i=0
for job in $MY_JOBS; do case $job in bscan:*) continue;; esac; c=$((i % NCHIP)); eval "Q_$c=\"\${Q_$c:-} $job\""; i=$((i+1)); done
pids=()
for c in $(seq 0 $((NCHIP-1))); do
  q=$(eval "echo \${Q_$c:-}"); [ -n "$q" ] || continue
  echo "chip $c queue: $q"
  # shellcheck disable=SC2086
  ( for job in $q; do arm_pipeline "$job" "$c"; done ) & pids+=($!)
done
for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
echo "QUEUES-DONE worker=$W $(date -u +%H:%M)"

# ---------- PHASE-F: full-test evals, sharded, per arm (final ckpt t6+t64; val-best ckpt t64) ----------
for arm in $MY_JOBS; do
  case $arm in bscan:*) continue;; esac
  D=runs/pretrain${R_TAG}_$arm; O=runs/sxeval_p${R_TAG}$arm
  [ -f "$D/.done" ] || { echo "PHASEF-SKIP-$arm (pretrain not done)"; continue; }
  for t in $SX_T_FULL; do
    [ -f "$O/full_t$t/summary_all.json" ] || { sharded_eval "$O/full_t$t" "$D/ckpt_latest.pkl" --split test --t-total "$t" --k-init 0 && echo "FULL-$arm-t$t-OK $(date -u +%H:%M)" || echo "FULL-$arm-t$t-FAILED"; }
  done
  vb=$(cut -d' ' -f1 "$D/val_best.txt" 2>/dev/null || true)
  if [ -n "$vb" ] && [ -f "$D/ckpt_$vb.pkl" ] && [ "$vb" != "$(printf '%06d' "$(python3 -c "import pickle;print(pickle.load(open('$D/ckpt_latest.pkl','rb'))['step'])")")" ]; then
    [ -f "$O/full_t64_valbest/summary_all.json" ] || { sharded_eval "$O/full_t64_valbest" "$D/ckpt_$vb.pkl" --split test --t-total 64 --k-init 0 && echo "FULLVB-$arm-t64-OK step=$vb $(date -u +%H:%M)" || echo "FULLVB-$arm-FAILED"; }
  fi
  tar czf "/tmp/sxeval_$arm.tgz" "$O" && gsutil -q cp "/tmp/sxeval_$arm.tgz" "$GCS/${arm}_eval.tgz" && echo "EVAL-$arm-OK $(date -u +%H:%M)"
done

# ---------- PHASE4: best arm (FINAL full-t64 exact) -> 20k subsample k=128 ----------
best=""; bestacc=-1
for arm in $MY_JOBS; do case $arm in bscan:*) continue;; esac
  f=runs/sxeval_p${R_TAG}$arm/full_t64/summary_all.json; [ -f "$f" ] || continue
  acc=$(python3 -c "import json;print(json.load(open('$f'))['exact_acc'])" 2>/dev/null) || continue
  python3 -c "import sys; sys.exit(0 if float('$acc') > float('$bestacc') else 1)" && { best=$arm; bestacc=$acc; }
done
if [ -n "$best" ]; then
  O=runs/sxbreadth20k_p${R_TAG}$best; obj=breadth20k_${best}.tgz
  if gsutil -q stat "$GCS/$obj" 2>/dev/null; then echo "SKIP-PHASE4 ($best, GCS complete)"
  else echo "PHASE4: best arm $best (full t64 exact $bestacc) -> $SX_SUB-puzzle subsample k=$SX_SUB_K t=64 $(date -u +%H:%M)"
    if sharded_eval "$O" "runs/pretrain${R_TAG}_$best/ckpt_latest.pkl" --split test --subsample "$SX_SUB" --t-total 64 --k-init "$SX_SUB_K"; then
      tar czf "/tmp/$obj" "$O" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "PHASE4-OK $best $(date -u +%H:%M)"
    else echo "PHASE4-FAILED $best"; fi
  fi
else echo "PHASE4-SKIP (no evaluated arm on this worker)"; fi

# ---------- COMPLETION GUARD (global, arm-keyed) ----------
missing=""
for b in $BSCANS; do IFS=: read -r _ arm sel k <<< "$b"; if [ "$sel" = strat ]; then obj=breadth_${arm}_t64_k${k}.tgz; else obj=breadth${sel}_${arm}_k${k}.tgz; fi; gsutil -q stat "$GCS/$obj" 2>/dev/null || missing="$missing $b"; done
for job in $ALL_ARMS; do
  gsutil -q stat "$GCS/${job}_ckpt.pkl" 2>/dev/null || missing="$missing $job:ckpt"
  gsutil -q stat "$GCS/${job}_eval.tgz" 2>/dev/null || missing="$missing $job:eval"
  ! arm_probe_ok "$job" || gsutil -q stat "$GCS/${job}_probe.tgz" 2>/dev/null || missing="$missing $job:probe"
done
if [ -n "$missing" ]; then echo "$SENT-WORKER-DONE worker=$W missing(other workers or failed):$missing $(date -u +%FT%TZ)"; exit 0; fi
for job in $ALL_ARMS; do
  D=runs/pretrain${R_TAG}_$job; mkdir -p "$D"
  [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS/${job}_ckpt.pkl" "$D/ckpt_latest.pkl"
  [ -f "$D/metrics.jsonl" ] || gsutil -q cp "$GCS/${job}_metrics.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
  for f in $(gsutil ls "$GCS/${job}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${job}_}" ] || gsutil -q cp "$f" "$D/${b#${job}_}"; done
  [ -d "runs/sxeval_p${R_TAG}$job" ] || { gsutil -q cp "$GCS/${job}_eval.tgz" /tmp/_e.tgz && tar xzf /tmp/_e.tgz; }
  ! arm_probe_ok "$job" || [ -d "runs/sudprobe_p${R_TAG}$job" ] || { gsutil -q cp "$GCS/${job}_probe.tgz" /tmp/_p.tgz && tar xzf /tmp/_p.tgz; }
done
for obj in $(gsutil ls "$GCS/breadth*.tgz" 2>/dev/null); do gsutil -q cp "$obj" /tmp/_b.tgz && tar xzf /tmp/_b.tgz 2>/dev/null; done
tar czf /tmp/$FINAL_OBJ runs/pretrain${R_TAG}_*/ckpt_*.pkl runs/pretrain${R_TAG}_*/metrics.jsonl runs/pretrain${R_TAG}_*/val_best.txt runs/pretrain${R_TAG}_*/config.json runs/*_p${R_TAG}* runs/sxbreadth* runs/wave_*.log 2>/dev/null
gsutil -q cp /tmp/$FINAL_OBJ "$GCS/$FINAL_OBJ" && echo "RESCUE-OK"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
if [ "${SELF_TEARDOWN:-0}" = 1 ] && [ -n "${SELF_POD:-}" ] && [ -n "${SELF_ZONE:-}" ]; then
  echo "SELF-TEARDOWN: deleting $SELF_POD in $SELF_ZONE (all artifacts banked) $(date -u +%FT%TZ)"; sleep 20
  gcloud compute tpus tpu-vm delete "$SELF_POD" --zone "$SELF_ZONE" --quiet >/dev/null 2>&1 && echo "SELF-TEARDOWN-ISSUED" || echo "SELF-TEARDOWN-FAILED (supervisor/watchdog will tear down)"
fi
