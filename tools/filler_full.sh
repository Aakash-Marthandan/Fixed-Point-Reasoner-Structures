#!/bin/bash
# FULL-SET EVALUATION FILLER (2026-09-10; the PI: "get the best models we have for the numbers and evaluate them against
# the full eval set where the numbers would best reflect our work in the paper"). A retargeted copy of tools/filler_champ.sh
# (same mechanics: eval-only, idempotent GCS markers under $GCS/filler/, own dirs runs/filler_*, claims across workers,
# 4-way sharded on this worker's chips, 300 s partials -> a killed job resumes). Differences: the arm order and the job list
# are ENV-driven per worker; two new rows; the ARC suite's gate is a separate list (default: none).
#   d64full : the D64 row on the FULL 422,786 test (the paper's headline depth row on the field's set)        [17.2 h w384 / 7.9 h w192 on 4 chips]
#   d128sub : D128 on a 50k uniform subsample (seed 20260822; supersedes the 20k row)                          [4.0 h w384 / 1.9 h w192]
#   d256sub : D256 on a 50k uniform subsample (supersedes the 5k row)                                           [8.0 h w384 / 3.7 h w192]
#   arcsuite: tools/arc_suite.py on p13Dri / p13C53 x {valhard, dev30} (4 pinned processes)                     [~1 h]
#   W=3 MY_ARMS="C3 C5" FILLER_ARMS="C5" JOBS="d64full d128sub d256sub arcsuite" nohup bash tools/filler_full.sh > runs/filler_full_w3.log 2>&1 &
set -u
cd "${FILLER_ROOT:-$HOME/HRRN}" || exit 1
GCS=${GCS:-gs://qhrrn2-rescue/champ}; R_TAG=${R_TAG:-champ}; FG=$GCS/filler
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz}
PY=${CHAIN_PY:-.venv/bin/python}; ARM_PREC=${ARM_PREC:-default}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}
W=${W:-?}; MY_ARMS=${MY_ARMS:?the arms this worker ran in the chain}
FILLER_ARMS=${FILLER_ARMS:-C0 C2 C1 C5}; JOBS=${JOBS:-d64full}
ARC_GATE_ARMS=${ARC_GATE_ARMS:-}                     # the ARC suite waits for these arms' d64full rows (default: no gate)
ARC_BUNDLE=${ARC_BUNDLE:-$FG/arcsuite_bundle.tgz}; ARC_K=${ARC_K:-32}
IDLE_POLL=${IDLE_POLL:-300}; N_FULL=${N_FULL:-422786}; SUB_N=${SUB_N:-50000}
pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=$ARM_PREC "$@"; }
log () { echo "[$(date -u +%FT%TZ)] $*"; }
busy () { pgrep -f 'tools/pretrain[.]py|tools/eval_sudoku_extreme[.]py|tools/arc_suite[.]py' >/dev/null 2>&1; }
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
  if [ ! -f "$ck" ]; then
    mkdir -p /tmp/filler_pull && gsutil -q cp "$GCS/${arm}_pretrain.tgz" "/tmp/filler_pull/${arm}_pretrain.tgz" 2>/dev/null || { echo ""; return 1; }
    mkdir -p "/tmp/filler_pull/p_$arm" && tar xzf "/tmp/filler_pull/${arm}_pretrain.tgz" -C "/tmp/filler_pull/p_$arm" "$ck" 2>/dev/null
    mkdir -p "$(dirname "$ck")" && cp "/tmp/filler_pull/p_$arm/$ck" "$ck" 2>/dev/null || { echo ""; return 1; }
  fi
  echo "$ck"
}
fill_eval () {  # NAME CK OUTDIR NGATE EXTRA... — NCHIP-way sharded, merged, n-gated, banked under $FG (idempotent by marker; resumes from partials)
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
arcsuite_job () {
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
mkdir -p runs; PIDF=runs/filler_full_w$W.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "FILLER-ALREADY-RUNNING pid=$(cat "$PIDF")"; exit 0; fi
echo $$ > "$PIDF"
log "FILLER-FULL START w=$W pid=$$ my_arms=[$MY_ARMS] arms=[$FILLER_ARMS] jobs=[$JOBS] arc_gate=[$ARC_GATE_ARMS] chips=$NCHIP"
claimed_elsewhere () { local c; for c in $(gsutil ls "$FG/${1}_CLAIM_w*" 2>/dev/null); do case $c in *_CLAIM_w$W) ;; *) return 0;; esac; done; return 1; }
for pass in $(seq 1 "${PASSES:-400}"); do
  todo=0; did=0
  for job in $JOBS; do
    if [ "$job" = arcsuite ]; then
      gsutil -q stat "$FG/arcsuite_OK" 2>/dev/null && continue
      todo=$((todo + 1))
      pending=""; for a in $ARC_GATE_ARMS; do gsutil -q stat "$FG/d64full_${a}_OK" 2>/dev/null || pending="$pending d64full_$a"; done
      [ -z "$pending" ] || { log "FILLER-ARC-DEFERRED (pending:$pending)"; continue; }
      claimed_elsewhere arcsuite && { log "FILLER-CLAIMED arcsuite (another worker)"; continue; }
      wait_idle; echo "$W $(date -u +%FT%TZ)" | gsutil -q cp - "$FG/arcsuite_CLAIM_w$W"; arcsuite_job && did=$((did + 1)); continue
    fi
    for arm in $FILLER_ARMS; do
      case $job in d64full) name=d64full_$arm;; d128sub) name=d128sub_$arm;; d256sub) name=d256sub_$arm;; *) log "FILLER-BAD-JOB $job"; continue;; esac
      gsutil -q stat "$FG/${name}_OK" 2>/dev/null && continue
      todo=$((todo + 1))
      gsutil -q stat "$GCS/${arm}_ARM_OK" 2>/dev/null || { log "FILLER-NOT-READY $arm"; continue; }
      claimed_elsewhere "$name" && { log "FILLER-CLAIMED $name (another worker)"; continue; }
      wait_idle
      echo "$W $(date -u +%FT%TZ)" | gsutil -q cp - "$FG/${name}_CLAIM_w$W"
      ck=$(grid_of "$arm") || { log "FILLER-NO-GRID $arm"; continue; }
      case $job in
        d64full) fill_eval "$name" "$ck" "runs/filler_sxeval_p${R_TAG}${arm}_full_t64" "$N_FULL" --split test --t-total 64 --ema --record-by-step && did=$((did + 1)) ;;
        d128sub) fill_eval "$name" "$ck" "runs/filler_sxeval_p${R_TAG}${arm}_sub${SUB_N}_t128" "$SUB_N" --split test --subsample "$SUB_N" --t-total 128 --ema --record-by-step && did=$((did + 1)) ;;
        d256sub) fill_eval "$name" "$ck" "runs/filler_sxeval_p${R_TAG}${arm}_sub${SUB_N}_t256" "$SUB_N" --split test --subsample "$SUB_N" --t-total 256 --ema --record-by-step && did=$((did + 1)) ;;
      esac
    done
  done
  [ "$todo" -eq 0 ] && { log "FILLER-FULL ALL-DONE w=$W"; exit 0; }
  log "FILLER-PASS $pass done=$did todo=$todo (sleeping ${PASS_SLEEP:-600} s)"; sleep "${PASS_SLEEP:-600}"
done
log "FILLER-PASSES-EXHAUSTED w=$W"
