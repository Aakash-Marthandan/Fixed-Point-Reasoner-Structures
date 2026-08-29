#!/bin/bash
# Ledger: PHASE B RUNG 2B (2026-08-29 launch registration) — d96 follow-up round
# on ONE spot pod (v6e-16 = 4x4 Mumbai-first per PI, or v6e-8 = 1x8 fallback);
# derivative of the PROVEN chain_sportBr2.sh (every mechanism rung-1/2-proven).
# Arms (registration = the authority):
#   D1  T6  FPA @50k s0 lr1e-3       — winner extended (B-M2 shot + H-45-anchor)
#   D2  T6  FPA @20k s1 lr1e-3       — C3's exact recipe: matched noise pair
#   D3  T12 FPA @50k s0 lr5e-4 bnl1e-6 — H-48 primary (ONE variable from C1)
#   D4  T12 FPA @50k s1 lr1e-3 bnl1e-6 — registered-lr restoration probe
# Rung-2b deltas (all registered): ONE-SHOT arms — a NaN halt AMPUTATES in-chain
# (last finite grid -> labeled final, truncated metrics, STOPPED.txt, evals
# proceed; NO retries); screens at per-arm STEP LISTS (D1 x5 + vb; D2 x2 + vb;
# D3/D4 x2 + vb; vb screens ALWAYS run — no coincide-skip class); screens run
# EAGERLY from banked 5k-grid ckpts while other arms still pretrain; PHASE4 is
# GATED (D1-vb vs pinned C3 .8848 minus the measured D2 noise; gate marker
# banked either way — gates compute, never validity); claim TTL 14400s > the
# measured d96 shard wall, claims owner-stamped, P4 shard claims CAPPED at
# NCHIP/pass with spawn echoes; screen objects are SIZE-CHECKED (a zero-byte
# screen self-heals unless it is a stopped-arm impossible step — the C4_vb
# class closed); p4depth OPTIONAL (never blocks). MONITOR 2000; eval batch 128.
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
R_TAG=${R_TAG:-sportBr2b}; export R_TAG
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
GCS=${GCS:-gs://qhrrn2-rescue/sportBr2b}; FINAL_OBJ=${FINAL_OBJ:-sportBr2b_final.tgz}
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
SENT=CHAIN-SPORTBR2B
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
PRIMARY=${PRIMARY:-"D1 D3"}
CARRIER_FULLS=${CARRIER_FULLS:-""}  # 2b: no t6 fulls; vb fulls on every arm
echo "=== SPORTBR2B START $(date -u +%FT%TZ) worker=$W/$NW chips=$NCHIP my_arms=[$MY_JOBS] all_arms=[$ALL_ARMS] d=$RD ws=$WS ==="
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/$SX_NPZ" "$NPZ" || { echo "NPZ-MISSING $SX_NPZ"; exit 2; }

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest "$@"; }
is_primary () { case " $PRIMARY " in *" $1 "*) return 0;; *) return 1;; esac; }
is_carrier () { case " $CARRIER_FULLS " in *" $1 "*) return 0;; *) return 1;; esac; }
arm_flags () {   # rung-2b registered recipes (2026-08-29 registration = the authority)
  case $1 in
    # D1: the rung-2 WINNER's recipe trained to 50k FRESH (not a continuation of
    # C3's ckpt — that would splice a decayed lr schedule = a second variable).
    D1) echo "--d $RD --width-scale $WS --T 6 --steps 50000 --beta-flux 0 --beta-flux-nl 0 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0";;
    # D2: C3's EXACT recipe at seed 1 — the matched funnel/cold noise pair.
    # Explicit pin; inherits nothing (the C1s1-pin lesson).
    D2) echo "--d $RD --width-scale $WS --T 6 --steps 20000 --beta-flux 0 --beta-flux-nl 0 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0 --seed 1";;
    # D3: H-48 primary — ONE variable from C1's final config (the bnl dose).
    # ONE-SHOT: a NaN = amputate + STOPPED, no retry (registered).
    D3) echo "--d $RD --width-scale $WS --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 1e-6 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0 --lr 5e-4";;
    # D4: registered-lr restoration probe at the same dose (seed 1; the seed
    # confound is named in the registration — seeds were not protective 0/4).
    D4) echo "--d $RD --width-scale $WS --T 12 --steps 50000 --beta-flux 0 --beta-flux-nl 1e-6 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0 --seed 1";;
    # C3X (REGISTERED ADDENDUM, PI-directed 2026-08-29 ~16:05Z, wave-3a pattern —
    # registered pre-data BEFORE any 2b result was read): the C3-continuation —
    # init from the banked rung-2 WINNER ckpt (sportBr2/C3_ckpt.pkl, 20k) and
    # train +30k steps at CONSTANT floor lr 3e-5 (lr == lr_end -> flat after
    # warmup; fresh optimizer per --init-from semantics, labeled). Free (no
    # dose): floor lr = the safest measured regime (all five NaNs at >=5e-4);
    # ONE-SHOT like every 2b arm. Screens at +10k/+20k/vb (total-equivalent
    # 30k/40k/50k). Rides the measured w2 idle window; full in the pole tail.
    C3X) echo "--d $RD --width-scale $WS --T 6 --steps 30000 --beta-flux 0 --beta-flux-nl 0 --fpa-k 4 --fpa-eps 0.2 --fpa-w 1.0 --lr 3e-5 --init-from runs/init_C3X.pkl";;
    *) echo "UNKNOWN-ARM $1" >&2; return 1;;
  esac
}
arm_steps () { case $1 in D2) echo 20000;; C3X) echo 30000;; *) echo 50000;; esac }
screen_steps () {  # ARM -> the registered fixed screen steps (vb rides separately, ALWAYS runs)
  case $1 in
    D1) echo "010000 015000 020000 025000 040000";;
    D2) echo "010000 015000";;
    D3|D4) echo "025000 040000";;
    C3X) echo "010000 020000";;
  esac
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
          echo "PRETRAIN-$arm-NAN-HALTED $(grep -oE '"step": [0-9]+' <<<"$lastrow") (live bank preserved at last clean sync; ONE-SHOT rule: amputation follows, no retry)"
          printf '%s\n' "$lastrow" > "$D/.nanhalt"
          pkill -f "pretrain[.]py" 2>/dev/null
          return 0;;
      esac
    fi
    gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt_live.pkl" 2>/dev/null || true
    gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics_live.jsonl" 2>/dev/null || true
    for f in "$D"/ckpt_0*.pkl; do [ -f "$f" ] && gsutil -q cp -n "$f" "$GCS/${arm}_$(basename "$f")" 2>/dev/null; done
  done
}
amputate_arm () {  # ARM DIR — ONE-SHOT NaN stop -> labeled final (the proven rung-2
  # manual amputation, automated + harness-verified: last FINITE grid ckpt ->
  # ckpt_latest, metrics truncated to <= that step, STOPPED.txt label; evals
  # then run on the final; impossible screens bank zero-byte legit-skips).
  local arm=$1 D=$2 S
  S=$(python3 - "$D" <<'AMPPY'
import json, math, pickle, shutil, sys
from pathlib import Path
D = Path(sys.argv[1])
def finite(o):
    if isinstance(o, dict): return all(finite(v) for v in o.values())
    if isinstance(o, (list, tuple)): return all(finite(v) for v in o)
    if isinstance(o, float): return math.isfinite(o)
    try:
        import numpy as np
        if hasattr(o, "dtype") or hasattr(o, "shape"):
            a = np.asarray(o)
            return bool(np.isfinite(a).all()) if a.dtype.kind in "fc" else True
    except Exception: pass
    return True
best = None
for ck in sorted(D.glob("ckpt_0*.pkl"), reverse=True):
    try: c = pickle.load(open(ck, "rb"))
    except Exception: continue
    if finite(c): best = (c["step"], ck); break
if not best: print(-1); raise SystemExit
step, ck = best
rows = []
for line in (D / "metrics.jsonl").read_text().splitlines():
    line = line.strip()
    if not line: continue
    try: r = json.loads(line)
    except Exception: continue
    s = r["monitor"]["step"] if "monitor" in r else r.get("step")
    if s is not None and s <= step: rows.append(line)
(D / "metrics.jsonl").write_text("\n".join(rows) + "\n")
shutil.copy(ck, D / "ckpt_latest.pkl")
print(step)
AMPPY
)
  { [ -n "$S" ] && [ "$S" != "-1" ]; } || { echo "AMPUTATE-$arm-FAILED (no finite grid ckpt)"; return 1; }
  printf 'STOPPED final step %s (NaN halt; one-shot rule, no retries; registered 2026-08-29)\n' "$S" > "$D/STOPPED.txt"
  gsutil -q cp "$D/STOPPED.txt" "$GCS/${arm}_STOPPED.txt" 2>/dev/null || true
  rm -f "$D/.nanhalt"
  echo "AMPUTATE-$arm-OK final=$S (labeled STOPPED; evals proceed on the final)"
  return 0
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
    gsutil -q cp "$GCS/${arm}_STOPPED.txt" "$D/STOPPED.txt" 2>/dev/null || true
    for f in $(gsutil ls "$GCS/${arm}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${arm}_}" ] || gsutil -q cp "$f" "$D/${b#${arm}_}"; done
    touch "$D/.done"; echo "SKIP-$arm (GCS complete)"; return 0
  fi
  if gsutil -q cp "$GCS/${arm}_ckpt_live.pkl" "$D/ckpt_latest.pkl" 2>/dev/null; then
    gsutil -q cp "$GCS/${arm}_metrics_live.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
    for f in $(gsutil ls "$GCS/${arm}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${arm}_}" ] || gsutil -q cp "$f" "$D/${b#${arm}_}"; done
    echo "RESUME-$arm from live ckpt (+metrics, +banked ckpts)"
  fi
  if [ "$arm" = C3X ] && [ ! -f runs/init_C3X.pkl ]; then
    gsutil -q cp "${C3X_INIT_SRC:-gs://qhrrn2-rescue/sportBr2/C3_ckpt.pkl}" runs/init_C3X.pkl \
      || { echo "PRETRAIN-C3X-FAILED (init ckpt fetch)"; return 1; }
    echo "C3X-INIT fetched ($(wc -c < runs/init_C3X.pkl | tr -d ' ') bytes from rung-2 C3 20k)"
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
  if [ $rc -ne 0 ]; then
    tailbad=$(tail -1 "$D/metrics.jsonl" 2>/dev/null | grep -ciE '"loss": *(nan|-?inf|NaN|-?Infinity)')
    if [ -f "$D/.nanhalt" ] || [ "${tailbad:-0}" -ge 1 ]; then
      amputate_arm "$arm" "$D" && rc=0
    fi
  fi
  if [ $rc -eq 0 ]; then
    lastloss=$(grep -oE '"loss": [0-9.eE+-]+' "$D/metrics.jsonl" 2>/dev/null | tail -1 | awk '{print $2}')
    python3 -c "import sys; l=float('${lastloss:-9}'); sys.exit(0 if l==l and l < 3.0 else 1)" \
      || echo "PRETRAIN-$arm-DIVERGED last_loss=${lastloss:-nan} (report to PI; registered contingency = ONE labeled relaunch at half lr, manual — the B1-d64 precedent)"
    touch "$D/.done"; gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt.pkl"; gsutil -q cp "$D/metrics.jsonl" "$GCS/${arm}_metrics.jsonl"
    [ -f "$D/STOPPED.txt" ] && gsutil -q cp "$D/STOPPED.txt" "$GCS/${arm}_STOPPED.txt" 2>/dev/null
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
  [ -f "$D/STOPPED.txt" ] || gsutil -q cp "$GCS/${arm}_STOPPED.txt" "$D/STOPPED.txt" 2>/dev/null || true
  for f in $(gsutil ls "$GCS/${arm}_ckpt_0*.pkl" 2>/dev/null); do b=$(basename "$f"); [ -f "$D/${b#${arm}_}" ] || gsutil -q cp "$f" "$D/${b#${arm}_}"; done
  return 0
}
task_obj () {
  case $1 in
    scr:*) IFS=: read -r _ a ck <<< "$1"; echo "screen_${a}_${ck}_k${SCREEN_K}.tgz";;
    full:*) IFS=: read -r _ a kind <<< "$1"; echo "full_${a}_${kind}.tgz";;
    probes4) echo "probes4.tgz";;
    probe:*) echo "probe_${1#probe:}.tgz";;
    p4depth) echo "depth_t256.tgz";;
  esac
}
arm_stop_step () {  # ARM -> stopped step or empty (GCS label is the truth)
  gsutil -q cp "$GCS/${1}_STOPPED.txt" - 2>/dev/null | grep -oE 'step [0-9]+' | awk '{print $2}'
}
task_ready () {
  case $1 in
    probes4) local a; for a in $PRIMARY; do gsutil -q stat "$GCS/${a}_ckpt.pkl" 2>/dev/null || return 1; done; return 0;;
    p4depth) gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null;;
    probe:*) gsutil -q stat "$GCS/${1#probe:}_ckpt.pkl" 2>/dev/null;;
    scr:*:s*)  # EAGER: a fixed-step screen is ready as soon as its 5k-grid ckpt
               # banks (runs while other arms still pretrain), or when the arm
               # completed/STOPPED (stopped-impossible steps bank legit skips).
      local a2 k2; IFS=: read -r _ a2 k2 <<< "$1"
      gsutil -q stat "$GCS/${a2}_ckpt_${k2#s}.pkl" 2>/dev/null && return 0
      gsutil -q stat "$GCS/${a2}_ckpt.pkl" 2>/dev/null && return 0
      gsutil -q stat "$GCS/${a2}_STOPPED.txt" 2>/dev/null && return 0
      return 1;;
    *) IFS=: read -r _ a _ <<< "$1"; gsutil -q stat "$GCS/${a}_ckpt.pkl" 2>/dev/null;;
  esac
}
scr_obj_ok () {  # OBJ ARM KIND -> 0 iff banked AND valid (nonzero, or a LEGIT zero:
  # stopped arm + step past its final — the C4_vb zero-byte class SELF-HEALS)
  local obj=$1 arm=$2 kind=$3
  gsutil -q stat "$GCS/$obj" 2>/dev/null || return 1
  gsutil -q cp "$GCS/$obj" /tmp/_zchk 2>/dev/null || return 1
  [ -s /tmp/_zchk ] && return 0
  [ "$kind" = vb ] && return 1
  local st; st=$(arm_stop_step "$arm")
  [ -n "$st" ] && [ "$((10#${kind#s}))" -gt "$st" ] 2>/dev/null && return 0
  return 1
}
screen_ck () {  # ARM KIND -> "STEP PATH" (empty = LEGIT skip: stopped arm, impossible step).
  # 2b: vb screens ALWAYS run (no coincide-skip class — a duplicate screen costs
  # minutes; the zero-byte skip class shrinks to stopped-impossible steps only).
  local arm=$1 kind=$2 D=runs/pretrain${R_TAG}_$1 step
  if [ "$kind" = vb ]; then step=$(vb_step "$arm")
  else
    step=${kind#s}
    if [ -f "$D/STOPPED.txt" ]; then
      local st; st=$(grep -oE 'step [0-9]+' "$D/STOPPED.txt" | awk '{print $2}')
      [ -n "$st" ] && [ "$((10#$step))" -gt "$st" ] && { echo ""; return 0; }
    fi
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
      IFS=: read -r _ arm ck <<< "$t"
      local D2=runs/pretrain${R_TAG}_$arm; mkdir -p "$D2"
      if [ "$ck" = vb ]; then need_arm_local "$arm" || return 1
      else  # EAGER fixed-step screen: needs only its grid ckpt (+ STOPPED label if any)
        [ -f "$D2/STOPPED.txt" ] || gsutil -q cp "$GCS/${arm}_STOPPED.txt" "$D2/STOPPED.txt" 2>/dev/null || true
        [ -f "$D2/ckpt_${ck#s}.pkl" ] || gsutil -q cp "$GCS/${arm}_ckpt_${ck#s}.pkl" "$D2/ckpt_${ck#s}.pkl" 2>/dev/null || true
      fi
      read -r step CK <<< "$(screen_ck "$arm" "$ck")"
      [ -n "$step" ] || { echo "SCREEN-$arm-$ck-SKIP (stopped arm; step impossible — legit zero-byte)"; gsutil -q cp /dev/null "$GCS/$obj" 2>/dev/null || true; return 0; }
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
    probe:*)
      # OPTIONAL single-arm probe (D4) — never blocks; skipped naturally if the arm is STOPPED
      local pa=${t#probe:}
      need_arm_local "$pa" || return 1
      local PD2=runs/sudprobe_p${R_TAG}$pa
      [ -s "$PD2/results.jsonl" ] || pin $((RANDOM % NCHIP)) python3 tools/probe_sudoku.py --ckpt "runs/pretrain${R_TAG}_$pa/ckpt_latest.pkl" --pairs-file "$NPZ" --split test \
          --stratified "$SX_STRAT" --t-total 64 --k-init 16 --eps-rungs "$EPS_RUNGS" --out "$PD2" > "runs/wave_pr_$pa.log" 2>&1 \
        || { echo "PROBE-$pa-FAILED (optional; never blocks)"; return 1; }
      tar czf "/tmp/$obj" "$PD2" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "PROBE-$pa-OK $(date -u +%H:%M)";;
  esac
}
SCREEN_TASKS=""
for a in $ALL_ARMS; do
  SCREEN_TASKS="$SCREEN_TASKS scr:$a:vb"
  for s in $(screen_steps "$a"); do SCREEN_TASKS="$SCREEN_TASKS scr:$a:s$s"; done
done
FULL_TASKS=""
for a in $ALL_ARMS; do FULL_TASKS="$FULL_TASKS full:$a:t64 full:$a:vb"; done
# p4depth is NOT in the PHASE2 set — its readiness (breadth20k banked) is only
# produced by PHASE4, which runs after PHASE2: putting it here deadlocks the
# queue to pass-exhaustion (CAUGHT BY THE HARNESS, S1, 2026-08-27). It gets its
# own claim-run after the PHASE4 block; the completion guard still requires it.
TASKS="$FULL_TASKS $SCREEN_TASKS probes4"
OPTIONAL_TASKS="probe:D4"
# CLAIM TTL 14400s > the measured d96 shard wall (~10.2ks): a live claim can
# never be taken over (the rung-2 TTL-takeover class closed); claims carry an
# owner stamp (informational) and are parsed first-field-only.
CLAIM_TTL=${CLAIM_TTL:-14400}
for pass in $(seq 1 200); do
  pending=0
  for t in $TASKS $OPTIONAL_TASKS; do
    obj=$(task_obj "$t"); claim="claim_${obj%.tgz}"
    case $t in
      scr:*)  # size-aware: a zero-byte screen object is accepted ONLY if legit
              # (stopped-impossible step) — an illegitimate zero self-heals here
        IFS=: read -r _ za zk <<< "$t"
        scr_obj_ok "$obj" "$za" "$zk" && continue
        if gsutil -q stat "$GCS/$obj" 2>/dev/null; then
          echo "SCREEN-OBJ-INVALID $t (zero-byte, not a legit skip) — re-running (the C4_vb class)"
          gsutil -q rm "$GCS/$obj" 2>/dev/null || true
        fi;;
      *) gsutil -q stat "$GCS/$obj" 2>/dev/null && continue;;
    esac
    optional=0; case " $OPTIONAL_TASKS " in *" $t "*) optional=1;; esac
    task_ready "$t" || { [ "$optional" -eq 0 ] && pending=1; continue; }
    if gsutil -q stat "$GCS/$claim" 2>/dev/null; then
      cts=$(gsutil -q cp "$GCS/$claim" - 2>/dev/null | awk 'NR==1{print $1}')
      if [ -n "$cts" ] && [ $(( $(date -u +%s) - cts )) -lt "$CLAIM_TTL" ] 2>/dev/null; then [ "$optional" -eq 0 ] && pending=1; continue; fi
      echo "CLAIM-STALE $t (age > ${CLAIM_TTL}s) — taking over"
    fi
    printf '%s w%s\n' "$(date -u +%s)" "$W" | gsutil -q cp - "$GCS/$claim" 2>/dev/null || true
    if run_task "$t"; then :; else [ "$optional" -eq 0 ] && pending=1; fi
    gsutil -q rm "$GCS/$claim" 2>/dev/null || true
  done
  [ "$pending" -eq 0 ] && break
  sleep "${PHASE2_SLEEP:-120}"
done
echo "PHASE2-DONE worker=$W $(date -u +%H:%M)"

# ---------- PHASE4 (GATED: D1-only 20k scan; coop shards CAPPED per pass) ----------
NSH=$((NCHIP * NW))
D96_C3_VB=${D96_C3_VB:-0.8848}   # pinned rung-2 constant — gates COMPUTE only; the
                                 # analyzer recomputes every verdict from data
p4_gate () {  # marker-authoritative: first worker to compute writes p4gate.txt
  local g
  if g=$(gsutil -q cp "$GCS/p4gate.txt" - 2>/dev/null | head -1) && [ -n "$g" ]; then echo "$g"; return 0; fi
  local d1 d2
  d1=$(gsutil -q cp "$GCS/screen_D1_vb_k${SCREEN_K}.tgz" /tmp/_g1.tgz 2>/dev/null && tar xzf /tmp/_g1.tgz 2>/dev/null; \
       python3 -c "import json;print(json.load(open('runs/sxscreen_p${R_TAG}D1_vb/summary_all.json')).get('vote_at_k',{}).get('256',0))" 2>/dev/null)
  d2=$(gsutil -q cp "$GCS/screen_D2_vb_k${SCREEN_K}.tgz" /tmp/_g2.tgz 2>/dev/null && tar xzf /tmp/_g2.tgz 2>/dev/null; \
       python3 -c "import json;print(json.load(open('runs/sxscreen_p${R_TAG}D2_vb/summary_all.json')).get('vote_at_k',{}).get('256',0))" 2>/dev/null)
  [ -n "$d1" ] && [ -n "$d2" ] || { echo "WAIT"; return 0; }
  g=$(python3 -c "
d1, d2, ref = float('$d1'), float('$d2'), float('$D96_C3_VB')
fn = max(abs(ref - d2), 0.02)
print(('PASS' if d1 >= ref - fn else 'FAIL') + f' d1={d1:.4f} d2={d2:.4f} ref={ref} fn2b={fn:.4f}')")
  echo "$g" | gsutil -q cp - "$GCS/p4gate.txt" 2>/dev/null || true
  echo "$g"
}
if ! gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null; then
  for w8 in $(seq 1 "${P4_WAIT_PASSES:-90}"); do all_scr=1
    for t2 in "scr:D1:vb" "scr:D2:vb"; do
      IFS=: read -r _ ga gk <<< "$t2"
      scr_obj_ok "$(task_obj "$t2")" "$ga" "$gk" || all_scr=0
    done
    [ "$all_scr" -eq 1 ] && break; sleep "${P4_POLL_SLEEP:-60}"; done
  GATE=$(p4_gate | tail -1)
  echo "P4-GATE: $GATE"
  case $GATE in
    PASS*)
    WINNER=D1
    echo "$WINNER" | gsutil -q cp - "$GCS/p4winner.txt" 2>/dev/null || true
    need_arm_local "$WINNER"
    D=runs/pretrain${R_TAG}_$WINNER; step=$(vb_step "$WINNER"); CK="$D/ckpt_$step.pkl"; [ -f "$CK" ] || CK="$D/ckpt_latest.pkl"
    O=runs/sxbreadth20k_p${R_TAG}${WINNER}; mkdir -p "$O"; partial_restore "$O"
    # PARTITION PIN (rung-2-proven): NSH pinned in GCS on first entry, reused by
    # every resume — a node-shape change mid-PHASE4 must not mix partitions.
    if NSHP=$(gsutil -q cp "$GCS/p4/NSH.txt" - 2>/dev/null | head -1) && [ -n "$NSHP" ]; then
      NSH=$NSHP; echo "P4 partition pinned: $NSH-way (from GCS)"
    else
      echo "$NSH" | gsutil -q cp - "$GCS/p4/NSH.txt" 2>/dev/null || true
      echo "P4 partition pinned: $NSH-way (fresh)"
    fi
    echo "PHASE4-COOP: winner $WINNER (vb) — $NSH-way, per-shard claims CAPPED at $NCHIP/pass (the rung-2 claim-race fix) $(date -u +%H:%M)"
    partial_sync "$O" & PS4=$!
    for p4pass in $(seq 1 "${P4_CLAIM_PASSES:-60}"); do
      unbanked=0; for K in $(seq 0 $((NSH-1))); do gsutil -q stat "$GCS/p4/summary_s$K.json" 2>/dev/null || unbanked=$((unbanked+1)); done
      [ "$unbanked" -eq 0 ] && break
      pids=(); slot=0; claimed=0
      for K in $(seq 0 $((NSH-1))); do
        [ "$claimed" -ge "$NCHIP" ] && break   # per-pass cap: never claim more than this worker can run NOW
        gsutil -q stat "$GCS/p4/summary_s$K.json" 2>/dev/null && continue
        sc="claim_p4_s$K"
        if gsutil -q stat "$GCS/$sc" 2>/dev/null; then
          cts=$(gsutil -q cp "$GCS/$sc" - 2>/dev/null | awk 'NR==1{print $1}')
          [ -n "$cts" ] && [ $(( $(date -u +%s) - cts )) -lt "$CLAIM_TTL" ] 2>/dev/null && continue
          echo "CLAIM-STALE p4 s$K — taking over"
        fi
        printf '%s w%s\n' "$(date -u +%s)" "$W" | gsutil -q cp - "$GCS/$sc" 2>/dev/null || true
        c=$((slot % NCHIP)); slot=$((slot+1)); claimed=$((claimed+1))
        echo "P4-SPAWN s$K -> chip $c (w$W, pass $p4pass)"
        ( for try in $(seq 1 60); do
            pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --shard "$K/$NSH" --out "$O" --bank-every 300 --batch "$EVAL_BATCH" \
                --split test --subsample "$SX_SUB" --t-total 64 --k-init "$SX_SUB_K" > "$O/shard_s$K.log" 2>&1 \
              && { gsutil -q cp "$O/summary_s$K.json" "$GCS/p4/summary_s$K.json"; gsutil -q cp "$O/records_s$K.npz" "$GCS/p4/records_s$K.npz"; echo "P4-SHARD-s$K-OK $(date -u +%H:%M)"; break; }
            if grep -qE "resource busy|Couldn't open iommu group" "$O/shard_s$K.log"; then sleep "${SHARD_RETRY_SLEEP:-120}"; continue; fi
            echo "P4-SHARD-s$K-FAILED"; break
          done; gsutil -q rm "$GCS/$sc" 2>/dev/null || true ) & pids+=($!)
      done
      echo "P4-INFLIGHT w$W pass=$p4pass claimed=$claimed unbanked=$unbanked"
      for pp in ${pids[@]+"${pids[@]}"}; do wait "$pp" || true; done
      [ "$claimed" -eq 0 ] && sleep "${P4_POLL_SLEEP:-60}"   # others own the rest — poll
    done
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
        tar czf /tmp/breadth20k.tgz runs/sxbreadth20k_p${R_TAG}* 2>/dev/null && gsutil -q cp /tmp/breadth20k.tgz "$GCS/breadth20k.tgz" && echo "PHASE4-OK $WINNER (coop ${NSH}-way) $(date -u +%H:%M)"
      else echo "PHASE4-MERGE-FAILED"; fi
    fi
    ;;
    FAIL*) echo "P4-GATE-FAIL (D1 funnel within/below the C3 noise band — scan not decision-bearing; gate marker banked)";;
    *) echo "P4-GATE-WAIT (vb screens not yet banked — a later worker/pass computes it)";;
  esac
fi

# ---------- P4DEPTH (OPTIONAL rider: t256 on the scanned arm; never blocks) ----------
if gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null && ! gsutil -q stat "$GCS/depth_t256.tgz" 2>/dev/null; then
  for dpass in $(seq 1 "${P4DEPTH_PASSES:-30}"); do
    gsutil -q stat "$GCS/depth_t256.tgz" 2>/dev/null && break
    claim="claim_depth_t256"
    if gsutil -q stat "$GCS/$claim" 2>/dev/null; then
      cts=$(gsutil -q cp "$GCS/$claim" - 2>/dev/null | awk 'NR==1{print $1}')
      if [ -n "$cts" ] && [ $(( $(date -u +%s) - cts )) -lt "$CLAIM_TTL" ] 2>/dev/null; then sleep "${P4_POLL_SLEEP:-60}"; continue; fi
      echo "CLAIM-STALE p4depth — taking over"
    fi
    printf '%s w%s\n' "$(date -u +%s)" "$W" | gsutil -q cp - "$GCS/$claim" 2>/dev/null || true
    run_task p4depth || true
    gsutil -q rm "$GCS/$claim" 2>/dev/null || true
  done
fi

# ---------- COMPLETION GUARD (global; hydrates from GCS — no hollow final;
# size-checks every screen object [the C4_vb zero-byte class]; requires
# breadth20k OR a banked FAIL gate; optional tasks never block) ----------
missing=""
for a in $ALL_ARMS; do
  gsutil -q stat "$GCS/${a}_ckpt.pkl" 2>/dev/null || missing="$missing $a:ckpt"
  gsutil -q stat "$GCS/${a}_evalcheap.tgz" 2>/dev/null || missing="$missing $a:evalcheap"
done
for t in $TASKS; do
  case $t in
    scr:*) IFS=: read -r _ ga gk <<< "$t"; scr_obj_ok "$(task_obj "$t")" "$ga" "$gk" || missing="$missing $t";;
    *) gsutil -q stat "$GCS/$(task_obj "$t")" 2>/dev/null || missing="$missing $t";;
  esac
done
if ! gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null; then
  GATEQ=$(gsutil -q cp "$GCS/p4gate.txt" - 2>/dev/null | head -1)
  case $GATEQ in FAIL*) : ;; *) missing="$missing phase4(gate=${GATEQ:-unset})";; esac
fi
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
gsutil -q cp "$GCS/p4gate.txt" runs/p4gate.txt 2>/dev/null || true
cache_push
tar czf /tmp/$FINAL_OBJ runs/pretrain${R_TAG}_*/ckpt_*.pkl runs/pretrain${R_TAG}_*/metrics.jsonl runs/pretrain${R_TAG}_*/val_best.txt runs/pretrain${R_TAG}_*/config.json runs/pretrain${R_TAG}_*/STOPPED.txt runs/p4gate.txt runs/*_p${R_TAG}* runs/wave_*.log 2>/dev/null
gsutil -q cp /tmp/$FINAL_OBJ "$GCS/$FINAL_OBJ" && echo "RESCUE-OK"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
if [ "${SELF_TEARDOWN:-0}" = 1 ] && [ -n "${SELF_POD:-}" ] && [ -n "${SELF_ZONE:-}" ]; then
  echo "SELF-TEARDOWN: deleting $SELF_POD in $SELF_ZONE (all artifacts banked) $(date -u +%FT%TZ)"; sleep 20
  gcloud compute tpus tpu-vm delete "$SELF_POD" --zone "$SELF_ZONE" --quiet >/dev/null 2>&1 && echo "SELF-TEARDOWN-ISSUED" || echo "SELF-TEARDOWN-FAILED (supervisor/watchdog will tear down)"
fi
