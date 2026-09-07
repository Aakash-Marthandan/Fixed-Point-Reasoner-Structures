#!/bin/bash
# Ledger: FRONTIER POD RUN (2026-09-07; Plan_2026-09-07_Instrument_Suite §4 / §6 build B4) — "the comparator rows at the
# full protocol": the public TRM-class Sudoku checkpoints (alphaXiv TRM-MLP `trmpub` [+ its 60k grid `trmpub60k`, one row],
# CGAR `cgar`, EqR `eqr` [EMA headline; raw = alt]) PORTED into our checkpoint format (tools/field_ckpts/port_field_ckpt.py,
# fp32 step-1 agreement checked) and read through OUR evaluator on ONE v6e-8 at Night A's exact protocol, plus PASS TWO of
# Night A: the DEC wide arms' 5k x k32 scans that deadlocked on the pods (A3 A7 A5, then A4 A8) and the DEC's D128 rows.
# EVAL-ONLY (no training): every job is one eval, idempotent by its GCS marker ($GCS/evals/<name>_OK), sharded over the 8
# chips with 300 s partial banking, the live 5-min bank of runs/ (tools/live_bank.sh), a static job list ordered for science
# per hour under spot churn (the headline rows of every model before any rider). Battery per model: full D16 (+ exact-by-step
# + halting logits) · full D64 · SCAN20k x k128 t64 (the selector column) · strat-512 k256 screen + majority · D128 on 20k /
# D256 on 5k · census · calibration · init radius (5 eps) · prefix zeroed · fp32 control; EqR extras: the raw-weights alt row,
# its released Langevin protocol (trunc reset + per-pass noise .5), the training-noise row, its own train-1k readout.
# THE DEC SCAN CANARY (Report_2026-09-07_NightA_Verdict §6.8): scan_A3 runs FIRST at --batch $DEC_SCAN_BATCH under a stall
# watchdog (no log/partial progress for $STALL_MIN min -> py-spy dump of every shard -> kill -> STALLED); on a stall the
# recipe steps to --z0-device (draws generated on the device, no host transfer; labeled), then --batch 64 --z0-device; the
# recipe that completes is the one every later DEC scan uses (RECIPE-DEC marker). A DEC scan that fails every variant is
# SCAN-DEADLOCK, labeled, and the chain proceeds. Field-class evals run at the field's bf16 (ARM_PREC default); the fp32
# control at highest. Harness: tools/harness_frontier.sh. Numerics/records/sets identical to Night A's chain_final.sh.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src

GCS=${GCS:-gs://qhrrn2-rescue/frontier}
R_TAG=${R_TAG:-frontier}
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0.npz}
GCS_W1=${GCS_W1:-gs://qhrrn2-rescue/sport2}
SENT=${SENT:-FRONTIER}
SUB=${C1_SUB:-20000}; STRAT=${C1_STRAT:-512}; K_SCAN=${C1_K_SCAN:-128}; N_FULL=${C1_N_FULL:-422786}
N_SCAN_DEC=${C1_N_SCAN_DEC:-5000}; K_SCAN_DEC=${C1_K_SCAN_DEC:-32}
DEC_SCAN_BATCH=${DEC_SCAN_BATCH:-128}; STALL_MIN=${STALL_MIN:-20}; STALL_SEC=${STALL_SEC:-$((STALL_MIN * 60))}; DEC_EVAL_TIMEOUT=${DEC_EVAL_TIMEOUT:-7200}
FRONTIER_MODELS=${FRONTIER_MODELS:-"trmpub eqr cgar"}
DEC_ARMS_FIRST=${DEC_ARMS_FIRST:-"A3"}; DEC_ARMS_NEXT=${DEC_ARMS_NEXT:-"A7 A5"}; DEC_ARMS_LAST=${DEC_ARMS_LAST:-"A4 A8"}
CKPT_DIR=runs/frontier_ckpts
PY=${CHAIN_PY:-python3}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}
ARM_PREC=default   # the field's bf16 matmul for every field-class eval; the fp32 control sets highest

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=$ARM_PREC "$@"; }

echo "=== $SENT START chips=$NCHIP $(date -u +%FT%TZ) ==="
mkdir -p "$(dirname "$NPZ")" "$CKPT_DIR" runs/frontier_sets
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/$(basename "$NPZ")" "$NPZ" || { echo "NPZ-MISSING"; exit 2; }
echo "NPZ-OK"
export JAX_COMPILATION_CACHE_DIR="$PWD/jax_cache"; mkdir -p "$PWD/jax_cache"
gsutil -q cp "$GCS/jax_cache.tgz" /tmp/jc.tgz 2>/dev/null && tar xzf /tmp/jc.tgz 2>/dev/null && echo "COMPILE-CACHE restored"
( ${PYSPY_INSTALL:-$PY -m pip install -q py-spy} >/dev/null 2>&1 && echo "PYSPY-OK" ) || echo "PYSPY-ABSENT (dumps skipped)"

# ---------- LIVE 5-MIN GCS BANKING + fresh-node RESTORE (the evals' 300 s partials survive a preemption; no pretrain dirs here) ----------
GCS="$GCS" R_TAG="$R_TAG" ARMS="" bash tools/live_bank.sh restore
GCS="$GCS" R_TAG="$R_TAG" ARMS="" bash tools/live_bank.sh loop & LB_PID=$!
trap 'kill "$LB_PID" 2>/dev/null' EXIT

# ---------- the checkpoints: pulled per object from $GCS/ckpts, LOAD-VERIFIED (a torn pull is re-pulled once) ----------
verify_ckpt () { JAX_PLATFORMS=cpu ${REAL_PY:-$PY} - "$1" <<'PYEOF'
import pickle, sys
d = pickle.load(open(sys.argv[1], "rb")); assert isinstance(d, dict) and "state" in d and "config" in d; print(d.get("step", 0))
PYEOF
}
fetch_ckpt () {  # NAME -> runs/frontier_ckpts/NAME.pkl (idempotent; load-verified)
  local n=$1 f="$CKPT_DIR/$1.pkl" try
  for try in 1 2; do
    [ -f "$f" ] && verify_ckpt "$f" >/dev/null 2>&1 && { echo "CKPT-OK $n"; return 0; }
    rm -f "$f"; gsutil -q cp "$GCS/ckpts/$n.pkl" "$f" || true
  done
  echo "CKPT-MISSING $n"; return 1
}
fetch_set () { local n=$1 f="runs/frontier_sets/$1"; [ -f "$f" ] || gsutil -q cp "$GCS/sets/$n" "$f" || { echo "SET-MISSING $n"; return 1; }; echo "SET-OK $n"; }

# ---------- eval helpers (chain_final.sh's, byte-for-byte mechanics: GCS marker, tar bank, n-gate) ----------
eval_sharded () {  # NAME CK OUTDIR NSH NGATE EXTRA... — NSH-way sharded over chips, merged, n-gated
  local name=$1 CK=$2 O=$3 NSH=$4 NGATE=$5; shift 5
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "EVAL-SKIP $name"; return 0; }
  mkdir -p "$O"; local pids=() rc=0 i p
  for i in $(seq 0 $((NSH - 1))); do
    pin $((i % NCHIP)) ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "${EVAL_NPZ:-$NPZ}" --out "$O" \
        --shard "$i/$NSH" --bank-every 300 "$@" > "$O/shard_$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { echo "EVAL-SHARD-FAILED $name"; return 1; }
  finish_eval "$name" "$O" "$NGATE"
}
finish_eval () {  # NAME OUTDIR NGATE — merge, n-gate, bank, mark
  local name=$1 O=$2 NGATE=$3
  JAX_PLATFORMS=cpu $PY tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  ${REAL_PY:-python3} -c "import json,sys; s=json.load(open('$O/summary_all.json')); sys.exit(0 if s['n']==$NGATE else 1)" \
    || { echo "EVAL-N-BAD $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "EVAL-OK $name $(date -u +%H:%M)"
}
newest_mtime () {  # DIR -> the newest mtime (s) among the shards' logs and banked partials (portable: no GNU find on the Mac harness)
  ${REAL_PY:-python3} - "$1" <<'PYEOF'
import glob, os, sys
fs = glob.glob(os.path.join(sys.argv[1], "shard_*.log")) + glob.glob(os.path.join(sys.argv[1], "partial_*.npz"))
print(int(max(os.path.getmtime(f) for f in fs)) if fs else "")
PYEOF
}
eval_watched () {  # NAME CK OUTDIR NSH NGATE EXTRA... — eval_sharded + the STALL WATCHDOG (the DEC multi-draw scan deadlock probe)
  local name=$1 CK=$2 O=$3 NSH=$4 NGATE=$5; shift 5
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "EVAL-SKIP $name"; return 0; }
  mkdir -p "$O"; local pids=() rc=0 i p t0 last now stalled=0
  for i in $(seq 0 $((NSH - 1))); do
    pin $((i % NCHIP)) ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --out "$O" \
        --shard "$i/$NSH" --bank-every 300 "$@" > "$O/shard_$i.log" 2>&1 & pids+=($!)
  done
  t0=$(date +%s)
  while :; do
    local alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive -eq 1 ] || break
    last=$(newest_mtime "$O"); now=$(date +%s); [ -n "$last" ] || last=$t0
    if [ $((now - last)) -ge "$STALL_SEC" ]; then
      echo "EVAL-STALLED $name (no progress for $STALL_SEC s at $(date -u +%H:%M))"
      for p in "${pids[@]}"; do command -v py-spy >/dev/null 2>&1 && py-spy dump --pid "$p" > "$O/pyspy_$p.txt" 2>&1; kill "$p" 2>/dev/null; done
      sleep 5; for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
      stalled=1; break
    fi
    sleep "${WATCH_POLL:-60}"
  done
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null || rc=1; done
  [ $stalled -eq 0 ] || { tar czf "/tmp/${name}_stall.tgz" "$O" 2>/dev/null && gsutil -q cp "/tmp/${name}_stall.tgz" "$GCS/evals/${name}_STALL_$(date -u +%H%M).tgz"; return 2; }
  [ $rc -eq 0 ] || { echo "EVAL-SHARD-FAILED $name"; return 1; }
  finish_eval "$name" "$O" "$NGATE"
}
census_one () {  # NAME CK OUTDIR CHIP EXTRA — the explosion census, idempotent
  local name=$1 CK=$2 O=$3 chip=$4; shift 4
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "CENSUS-SKIP $name"; return 0; }
  mkdir -p "$O"
  pin "$chip" ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/explosion_census.py --ckpt "$CK" --npz "$NPZ" --out "$O" --name "$name" "$@" > "$O/run.log" 2>&1 \
    || { echo "CENSUS-FAILED $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "CENSUS-OK $name $(date -u +%H:%M)"
}
calib_one () {  # NAME CK OUTDIR CHIP EXTRA — stall calibration, idempotent
  local name=$1 CK=$2 O=$3 chip=$4; shift 4
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "CALIB-SKIP $name"; return 0; }
  mkdir -p "$O"
  pin "$chip" ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/stall_calibration.py --ckpt "$CK" --npz "$NPZ" --out "$O" "$@" > "$O/run.log" 2>&1 \
    || { echo "CALIB-FAILED $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "CALIB-OK $name $(date -u +%H:%M)"
}

# ---------- the DEC scan (pass two) with the canary recipe ----------
DEC_RECIPE=""   # the flags that completed on the canary: "" (batch only) | "--z0-device" | "--batch 64 --z0-device"
dec_recipe () {  # read the banked recipe marker (a relaunch keeps the canary's finding)
  [ -n "$DEC_RECIPE" ] && return 0
  local r; r=$(gsutil -q cp "$GCS/RECIPE-DEC" - 2>/dev/null) && DEC_RECIPE=$r && echo "RECIPE-DEC restored: [$DEC_RECIPE]"
}
scan_dec () {  # ARM — the wide arm's 5k x k32 t64 scan on its vsel grid; canary variants on a stall; SCAN-DEADLOCK labeled
  local arm=$1 CK="$CKPT_DIR/$arm.pkl" name="scan_$arm" O="runs/sxscan_pfinalA$arm" rc v
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "EVAL-SKIP $name"; return 0; }
  gsutil -q stat "$GCS/${arm}_SCAN_DEADLOCK" 2>/dev/null && { echo "SCAN-DEADLOCK-SKIP $arm"; return 0; }
  fetch_ckpt "$arm" || return 1
  dec_recipe
  export EVAL_TIMEOUT=$DEC_EVAL_TIMEOUT
  local variants; if [ -n "$DEC_RECIPE" ] || gsutil -q stat "$GCS/RECIPE-DEC" 2>/dev/null; then variants="$DEC_RECIPE"; else variants="BATCHONLY --z0-device --batch_64_--z0-device"; fi
  for v in $variants; do
    v=${v//_/ }; [ "$v" = "BATCHONLY" ] && v=""
    rm -rf "$O"; echo "DEC-SCAN-TRY $arm batch=$DEC_SCAN_BATCH flags=[$v] $(date -u +%H:%M)"
    # shellcheck disable=SC2086
    eval_watched "$name" "$CK" "$O" "$NCHIP" "$N_SCAN_DEC" --split test --subsample "$N_SCAN_DEC" --t-total 64 --k-init "$K_SCAN_DEC" --ema --batch "$DEC_SCAN_BATCH" $v; rc=$?
    if [ $rc -eq 0 ]; then
      [ -n "$DEC_RECIPE" ] || { DEC_RECIPE="${v:-BATCHONLY}"; printf '%s' "$DEC_RECIPE" | gsutil -q cp - "$GCS/RECIPE-DEC"; echo "RECIPE-DEC $arm: [$DEC_RECIPE]"; }
      unset EVAL_TIMEOUT; return 0
    fi
    echo "DEC-SCAN-VARIANT-FAILED $arm rc=$rc flags=[$v]"
    [ $rc -eq 2 ] || [ -z "$DEC_RECIPE" ] || break     # a non-stall failure under a fixed recipe: no other variant to try
  done
  unset EVAL_TIMEOUT
  echo "SCAN-DEADLOCK $arm (every variant failed; labeled)"; echo "$(date -u +%FT%TZ) variants=[$variants]" | gsutil -q cp - "$GCS/${arm}_SCAN_DEADLOCK"; return 1
}
dec_d128 () {  # ARM — the DEC's D128 row on the 20k (exact-by-step)
  local arm=$1; fetch_ckpt "$arm" || return 1
  eval_sharded "d128_$arm" "$CKPT_DIR/$arm.pkl" "runs/sxeval_pfinalA$arm/sub20k_t128" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 128 --ema --record-by-step
}

# ---------- the frontier battery ----------
headline () {  # M — the rows the paper needs first: full D16 (+ bits + q), full D64 (+ bits), the 20k x k128 scan
  local m=$1 CK="$CKPT_DIR/$1.pkl"; fetch_ckpt "$m" || return 1
  eval_sharded "full_${m}_vsel_t16" "$CK" "runs/sxeval_p${R_TAG}${m}/full_vsel_t16" "$NCHIP" "$N_FULL" --split test --t-total 16 --ema --record-by-step --record-q
  eval_sharded "full_${m}_vsel_t64" "$CK" "runs/sxeval_p${R_TAG}${m}/full_vsel_t64" "$NCHIP" "$N_FULL" --split test --t-total 64 --ema --record-by-step
  eval_sharded "scan_${m}" "$CK" "runs/sxscan_p${R_TAG}${m}" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 64 --k-init "$K_SCAN" --ema
}
riders () {  # M — the rest of the battery
  local m=$1 CK="$CKPT_DIR/$1.pkl" e; fetch_ckpt "$m" || return 1
  eval_sharded "screen_${m}_vb" "$CK" "runs/sxscreen_p${R_TAG}${m}_vb" "$NCHIP" "$STRAT" --split test --stratified "$STRAT" --t-total 16 --k-init 256 --vote-unverified --ema
  eval_sharded "d128_${m}" "$CK" "runs/sxeval_p${R_TAG}${m}/sub20k_t128" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 128 --ema --record-by-step
  eval_sharded "d256_${m}" "$CK" "runs/sxeval_p${R_TAG}${m}/sub5k_t256" "$NCHIP" 5000 --split test --subsample 5000 --t-total 256 --ema --record-by-step
  census_one "census_${m}_vsel" "$CK" "runs/sxcensus_p${R_TAG}${m}_vsel" 0 --ema
  calib_one "calib_${m}_vsel" "$CK" "runs/sxcalib_p${R_TAG}${m}_vsel" 1 --ema
  for e in 0.03 0.1 0.3 1 3; do
    eval_sharded "initrad_${m}_e${e}" "$CK" "runs/sxeval_p${R_TAG}${m}/initrad_e${e}" "$NCHIP" "$STRAT" --split test --stratified "$STRAT" --t-total 16 --k-init 1 --z0-mode perturb --z0-eps "$e" --ema
  done
  eval_sharded "prefix0_${m}" "$CK" "runs/sxeval_p${R_TAG}${m}/sub20k_t16_prefix0" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 16 --zero-prefix --ema
  ARM_PREC=highest eval_sharded "fp32_${m}" "$CK" "runs/sxeval_p${R_TAG}${m}/sub20k_t16_fp32" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 16 --ema --record-by-step
}
eqr_extras () {  # EqR: the raw-weights alt row, its released Langevin protocol, the training-noise row, the truncated reset, its train-1k
  local CK="$CKPT_DIR/eqr.pkl"; fetch_ckpt eqr || return 1
  eval_sharded "full_eqr_raw_t16" "$CK" "runs/sxeval_p${R_TAG}eqr/full_raw_t16" "$NCHIP" "$N_FULL" --split test --t-total 16 --record-by-step
  eval_sharded "scan_eqr_trunc" "$CK" "runs/sxscan_p${R_TAG}eqr_trunc" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 64 --k-init 32 --z0-mode trunc --ema
  eval_sharded "noise05_eqr_t16" "$CK" "runs/sxeval_p${R_TAG}eqr/sub20k_t16_noise05" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 16 --seg-noise-beta 0.5 --ema --record-by-step
  eval_sharded "noise05_eqr_t64" "$CK" "runs/sxeval_p${R_TAG}eqr/sub20k_t64_noise05" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 64 --seg-noise-beta 0.5 --ema --record-by-step
  eval_sharded "noise05_eqr_scan" "$CK" "runs/sxscan_p${R_TAG}eqr_noise05" "$NCHIP" 5000 --split test --subsample 5000 --t-total 64 --k-init 32 --z0-mode trunc --seg-noise-beta 0.5 --ema
  eval_sharded "noise001_eqr_t16" "$CK" "runs/sxeval_p${R_TAG}eqr/sub20k_t16_noise001" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 16 --seg-noise-beta 0.01 --ema --record-by-step
  fetch_set train_eqr_sx.npz && EVAL_NPZ=runs/frontier_sets/train_eqr_sx.npz eval_sharded "train1k_eqr" "$CK" "runs/sxeval_p${R_TAG}eqr/train1k_t16" "$NCHIP" 1000 --split test --t-total 16 --ema --record-by-step
}
trmpub60k_row () {  # the second released alphaXiv grid: the headline D16 + the 20k D64 (one row of the reproduction column)
  local CK="$CKPT_DIR/trmpub60k.pkl"; fetch_ckpt trmpub60k || return 0
  eval_sharded "full_trmpub60k_vsel_t16" "$CK" "runs/sxeval_p${R_TAG}trmpub60k/full_vsel_t16" "$NCHIP" "$N_FULL" --split test --t-total 16 --ema --record-by-step --record-q
  eval_sharded "sub20k_trmpub60k_t64" "$CK" "runs/sxeval_p${R_TAG}trmpub60k/sub20k_t64" "$NCHIP" "$SUB" --split test --subsample "$SUB" --t-total 64 --ema --record-by-step
}

# ---------- the static job list (science per hour) ----------
rc=0; m=""; arm=""
for arm in $DEC_ARMS_FIRST; do scan_dec "$arm" || rc=1; done            # the canary (the deadlock diagnosis at the source)
for m in $FRONTIER_MODELS; do headline "$m" || rc=1; done                # every model's paper rows
for arm in $DEC_ARMS_NEXT; do scan_dec "$arm" || rc=1; done
for m in $FRONTIER_MODELS; do riders "$m" || rc=1; done
case " $FRONTIER_MODELS " in *" eqr "*) eqr_extras || rc=1;; esac
trmpub60k_row || rc=1
for arm in $DEC_ARMS_LAST; do scan_dec "$arm" || rc=1; done
for arm in $DEC_ARMS_FIRST $DEC_ARMS_NEXT; do dec_d128 "$arm" || rc=1; done

# ---------- completion (idempotent): every marker present or labeled-absent -> the final tarball ----------
missing=""
for f in $(gsutil ls "$GCS/evals/*_OK" 2>/dev/null | sed 's|.*/||; s|_OK$||'); do :; done
tar czf /tmp/${R_TAG}_final.tgz runs/sxeval_p${R_TAG}* runs/sxscan_p${R_TAG}* runs/sxscreen_p${R_TAG}* runs/sxcensus_p${R_TAG}* runs/sxcalib_p${R_TAG}* runs/sxscan_pfinalA* runs/sxeval_pfinalA*/sub20k_t128 2>/dev/null
gsutil -q cp /tmp/${R_TAG}_final.tgz "$GCS/${R_TAG}_final.tgz"
tar czf /tmp/jc.tgz jax_cache 2>/dev/null && gsutil -q cp /tmp/jc.tgz "$GCS/jax_cache.tgz" || true
if [ $rc -eq 0 ]; then
  echo "CHAIN-$SENT-COMPLETE $(date -u +%FT%TZ)"
  if [ "${SELF_TEARDOWN:-0}" = 1 ] && [ -n "${SELF_POD:-}" ] && [ -n "${SELF_ZONE:-}" ]; then
    echo "SELF-TEARDOWN $(date -u +%FT%TZ)"; sleep 20
    gcloud compute tpus tpu-vm delete "$SELF_POD" --zone "$SELF_ZONE" --quiet >/dev/null 2>&1 || echo "SELF-TEARDOWN-FAILED"
  fi
  exit 0
fi
echo "$SENT-INCOMPLETE (a job failed; markers absent for the failed evals; relaunch resumes) $(date -u +%FT%TZ)"; exit 1
