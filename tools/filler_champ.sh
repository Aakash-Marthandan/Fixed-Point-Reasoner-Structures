#!/bin/bash
# IDLE-WORKER FILLER for the champion night (2026-09-08; the PI: "as any worker finishes its jobs, they can pick up
# these instead of idling"). Eval-only, idempotent, and SEPARATE from the running chain: its own dirs (runs/filler_*),
# its own markers ($GCS/filler/*), the chain's code untouched; named filler_* so the supervisor's relaunch pkill
# (tools/chain_*.sh | tools/eval_*.py) never kills the queue — only its evaluator shards, which resume from their
# 300 s partials on the next attempt. Never runs a job unless THIS worker is idle: every arm of MY_ARMS has its
# ARM_OK and no trainer / evaluator process is alive at the job's start.
# Jobs per arm, in FILLER_ARMS order, each only once that arm's ARM_OK exists (its selected grid is final):
#   k128    : the 5,000-puzzle x k128 t64 scan at --batch 128 (the registered scan is k32; matched restarts vs EqR's 128)
#   d64full : the D64 row on the FULL 422,786-puzzle test set (the registered row is on 100k)
# The grid = the one the arm's battery used (the full_vsel_t16 summary's ckpt field: canonical path), pulled from the
# arm's banked pretrain tgz when this worker did not train it and placed at its canonical path.
#   MY_ARMS="C6" W=2 nohup bash tools/filler_champ.sh > runs/filler_w2.log 2>&1 &
set -u
cd "${FILLER_ROOT:-$HOME/HRRN}" || exit 1
GCS=${GCS:-gs://qhrrn2-rescue/champ}; R_TAG=${R_TAG:-champ}; FG=$GCS/filler
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz}
PY=${CHAIN_PY:-.venv/bin/python}; ARM_PREC=${ARM_PREC:-default}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}
W=${W:-?}; MY_ARMS=${MY_ARMS:?the arms this worker runs in the chain}
FILLER_ARMS=${FILLER_ARMS:-C0 C6 C3 C5 C4 C1 C2}; JOBS=${JOBS:-k128 d64full arcsuite}   # PI 2026-09-08: every Sudoku row before any ARC job
ARC_BUNDLE=${ARC_BUNDLE:-$FG/arcsuite_bundle.tgz}; ARC_K=${ARC_K:-32}
IDLE_POLL=${IDLE_POLL:-300}; N_FULL=${N_FULL:-422786}; N_SCAN=${N_SCAN:-5000}; K_SCAN=${K_SCAN:-128}
pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=$ARM_PREC "$@"; }
log () { echo "[$(date -u +%FT%TZ)] $*"; }
busy () {  # 1 = a trainer or an evaluator is alive on this worker (the chain's or a previous filler job's)
  pgrep -f 'tools/pretrain[.]py|tools/eval_sudoku_extreme[.]py' >/dev/null 2>&1
}
mine_done () { local a; for a in $MY_ARMS; do gsutil -q stat "$GCS/${a}_ARM_OK" 2>/dev/null || gsutil -q stat "$GCS/${a}_SKIPPED" 2>/dev/null || return 1; done; return 0; }
wait_idle () {
  while :; do
    if mine_done && ! busy; then return 0; fi
    log "FILLER-WAIT w=$W (mine_done=$(mine_done && echo 1 || echo 0) busy=$(busy && echo 1 || echo 0))"; sleep "$IDLE_POLL"
  done
}
grid_of () {  # ARM -> the canonical selected-grid path (from the battery's full_vsel_t16 summary), present on disk
  local arm=$1 d="runs/sxeval_p${R_TAG}${arm}/full_vsel_t16" ck
  if [ ! -f "$d/summary_all.json" ]; then
    mkdir -p /tmp/filler_pull && gsutil -q cp "$GCS/evals/full_${arm}_vsel_t16.tgz" "/tmp/filler_pull/full_${arm}_vsel_t16.tgz" 2>/dev/null \
      && mkdir -p /tmp/filler_pull/x_$arm && tar xzf "/tmp/filler_pull/full_${arm}_vsel_t16.tgz" -C "/tmp/filler_pull/x_$arm" 2>/dev/null
    d=/tmp/filler_pull/x_$arm/$d; [ -f "$d/summary_all.json" ] || { echo ""; return 1; }
  fi
  ck=$(${REAL_PY:-python3} -c "import json; print(json.load(open('$d/summary_all.json'))['ckpt'])" 2>/dev/null); [ -n "$ck" ] || { echo ""; return 1; }
  if [ ! -f "$ck" ]; then   # this worker did not train the arm: pull the banked pretrain tgz, place ONLY the selected grid at its canonical path
    mkdir -p /tmp/filler_pull && gsutil -q cp "$GCS/${arm}_pretrain.tgz" "/tmp/filler_pull/${arm}_pretrain.tgz" 2>/dev/null || { echo ""; return 1; }
    mkdir -p "/tmp/filler_pull/p_$arm" && tar xzf "/tmp/filler_pull/${arm}_pretrain.tgz" -C "/tmp/filler_pull/p_$arm" "$ck" 2>/dev/null
    mkdir -p "$(dirname "$ck")" && cp "/tmp/filler_pull/p_$arm/$ck" "$ck" 2>/dev/null || { echo ""; return 1; }
  fi
  echo "$ck"
}
fill_eval () {  # NAME CK OUTDIR NGATE EXTRA... — NCHIP-way sharded, merged, n-gated, banked under $FG (idempotent by marker)
  local name=$1 CK=$2 O=$3 NGATE=$4; shift 4
  gsutil -q stat "$FG/${name}_OK" 2>/dev/null && { log "FILLER-SKIP $name (done)"; return 0; }
  mkdir -p "$O"; local pids=() rc=0 i p
  log "FILLER-JOB-START $name ck=$CK chips=$NCHIP"
  for i in $(seq 0 $((NCHIP - 1))); do
    pin "$i" $PY tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --out "$O" --shard "$i/$NCHIP" --bank-every 300 "$@" > "$O/shard_$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { log "FILLER-JOB-FAILED $name (a shard failed; partials stay for the resume)"; return 1; }
  JAX_PLATFORMS=cpu $PY tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  ${REAL_PY:-python3} -c "import json,sys; s=json.load(open('$O/summary_all.json')); sys.exit(0 if s['n']==$NGATE else 1)" \
    || { log "FILLER-N-BAD $name"; return 1; }
  tar czf "/tmp/filler_${name}.tgz" "$O" && gsutil -q cp "/tmp/filler_${name}.tgz" "$FG/${name}.tgz" && echo ok | gsutil -q cp - "$FG/${name}_OK"
  log "FILLER-JOB-OK $name"
}
mkdir -p runs; PIDF=runs/filler_w$W.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "FILLER-ALREADY-RUNNING pid=$(cat "$PIDF")"; exit 0; fi
echo $$ > "$PIDF"   # PID-based liveness (never a text pattern: the self-match hazard)
arcsuite_job () {  # the ARC suite (tools/arc_suite.py) for p13Dri (RI) + p13C53 (plain twin) x {valhard, dev30}: 4 pinned processes, banked as one
  local name=arcsuite O=runs/filler_arcsuite pids=() rc=0 i=0 sub set ck
  gsutil -q stat "$FG/${name}_OK" 2>/dev/null && { log "FILLER-SKIP $name (done)"; return 0; }
  if [ ! -f runs/filler_arcsuite_bundle_OK ]; then
    gsutil -q cp "$ARC_BUNDLE" /tmp/arcsuite_bundle.tgz && tar xzf /tmp/arcsuite_bundle.tgz && touch runs/filler_arcsuite_bundle_OK || { log "FILLER-ARC-BUNDLE-FAIL"; return 1; }
  fi
  mkdir -p "$O"; log "FILLER-JOB-START $name chips=$NCHIP k=$ARC_K"
  for sub in p13Dri p13C53; do for set in valhard dev30; do
    ck=runs/pretrain13_${sub#p13}/ckpt_053333.pkl
    [ -f "$O/${sub}_${set}/summary.json" ] && continue
    pin $((i % NCHIP)) $PY tools/arc_suite.py --ckpt "$ck" --set "$set" --k "$ARC_K" --t-total 16 --out "$O/${sub}_${set}" > "$O/${sub}_${set}.log" 2>&1 & pids+=($!); i=$((i + 1))
  done; done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { log "FILLER-JOB-FAILED $name (a run failed; results.jsonl resumes)"; return 1; }
  tar czf "/tmp/filler_${name}.tgz" "$O" && gsutil -q cp "/tmp/filler_${name}.tgz" "$FG/${name}.tgz" && echo ok | gsutil -q cp - "$FG/${name}_OK"
  log "FILLER-JOB-OK $name"
}
log "FILLER START w=$W pid=$$ my_arms=[$MY_ARMS] arms=[$FILLER_ARMS] jobs=[$JOBS] chips=$NCHIP"
claimed_elsewhere () {  # NAME -> 0 if another worker holds the claim (this worker may retry its own)
  local c; for c in $(gsutil ls "$FG/${1}_CLAIM_w*" 2>/dev/null); do case $c in *_CLAIM_w$W) ;; *) return 0;; esac; done; return 1
}
for pass in $(seq 1 "${PASSES:-400}"); do   # 400 x 600 s ≈ 2.8 days: the queue outlives the night (40 would have exited before the first ARM_OK)
  todo=0; did=0
  for job in $JOBS; do
    if [ "$job" = arcsuite ]; then
      gsutil -q stat "$FG/arcsuite_OK" 2>/dev/null && continue
      todo=$((todo + 1))
      pending=""; for a in $FILLER_ARMS; do for j in k128 d64full; do gsutil -q stat "$FG/${j}_${a}_OK" 2>/dev/null || pending="$pending ${j}_$a"; done; done
      [ -z "$pending" ] || [ "${FORCE_ARC:-0}" = 1 ] || { log "FILLER-ARC-DEFERRED (Sudoku rows pending:$pending)"; continue; }
      [ -z "$pending" ] || log "FILLER-ARC-FORCED (PI 2026-09-10: run ARC now; Sudoku pending:$pending)"
      claimed_elsewhere arcsuite && { log "FILLER-CLAIMED arcsuite (another worker)"; continue; }
      wait_idle; echo "$W $(date -u +%FT%TZ)" | gsutil -q cp - "$FG/arcsuite_CLAIM_w$W"; arcsuite_job && did=$((did + 1)); continue
    fi
    for arm in $FILLER_ARMS; do
      case $job in k128) name=k128_$arm;; d64full) name=d64full_$arm;; *) log "FILLER-BAD-JOB $job"; continue;; esac
      gsutil -q stat "$FG/${name}_OK" 2>/dev/null && continue
      todo=$((todo + 1))
      gsutil -q stat "$GCS/${arm}_ARM_OK" 2>/dev/null || { log "FILLER-NOT-READY $arm (no ARM_OK yet)"; continue; }
      claimed_elsewhere "$name" && { log "FILLER-CLAIMED $name (another worker)"; continue; }
      wait_idle
      echo "$W $(date -u +%FT%TZ)" | gsutil -q cp - "$FG/${name}_CLAIM_w$W"
      ck=$(grid_of "$arm") || { log "FILLER-NO-GRID $arm"; continue; }
      case $job in
        k128)    fill_eval "$name" "$ck" "runs/filler_sxscan${K_SCAN}_p${R_TAG}${arm}" "$N_SCAN" \
                   --split test --subsample "$N_SCAN" --t-total 64 --k-init "$K_SCAN" --ema --batch "${SCAN_BATCH:-32}" && did=$((did + 1)) ;;   # 32 rows x 128 draws = the registered scan's 128 x 32 state footprint
        d64full) fill_eval "$name" "$ck" "runs/filler_sxeval_p${R_TAG}${arm}_full_t64" "$N_FULL" --split test --t-total 64 --ema && did=$((did + 1)) ;;
      esac
    done
  done
  [ "$todo" -eq 0 ] && { log "FILLER-ALL-DONE w=$W"; exit 0; }
  log "FILLER-PASS $pass done=$did todo=$todo (sleeping ${PASS_SLEEP:-600} s)"; sleep "${PASS_SLEEP:-600}"
done
log "FILLER-PASSES-EXHAUSTED w=$W"
