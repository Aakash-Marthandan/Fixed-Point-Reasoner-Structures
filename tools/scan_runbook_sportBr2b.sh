#!/bin/bash
# Ledger: RUNG 2B RIDER SCANS registration (2026-08-30, PI-decided at the verdict).
# Runs the two one-shot 20k breadth scans (C3X = the B-M2 adjudication; D4 =
# informational/carrier-facing) with the chain's PHASE4 eval flags VERBATIM
# (comparability: same subsample seed default, k=128, t=64, banked partials).
#
# HARDENING (the 5-6h-wall design, per PI directive):
#   - STATIC shard assignment — worker x chip -> shard is a FIXED map; there is
#     NO claim system in this runbook, so the rung-2 claim-race class (16-way
#     silently 4-way) cannot exist. 4x4: C3X shards 0-7 on workers 0-1, D4
#     shards 0-7 on workers 2-3 (both scans concurrent). 1x8 fallback: worker 0
#     runs C3X 0-7 then D4 0-7 (sequential scans).
#   - NSH=8 per scan BY CONSTRUCTION (marker banked for the audit trail).
#   - --bank-every 300 + per-shard immutable banking + 5-min partial sync ->
#     any churn costs <=5-10 min.
#   - busy-retry on chip contention (the ec_one/iommu pattern).
#   - SHARD-START/OK/FAILED echoes; no silent waits (the 08-26 lesson).
#   - merges on CPU (JAX_PLATFORMS=cpu), n-gated == SX_SUB before banking.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src

GCS=${GCS:-gs://qhrrn2-rescue/sportBr2b}
R_TAG=${R_TAG:-sportBr2b}
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
SX_SUB=${SX_SUB:-20000}; SX_SUB_K=${SX_SUB_K:-128}; EVAL_BATCH=${EVAL_BATCH:-128}
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0.npz}
GCS_W1=${GCS_W1:-gs://qhrrn2-rescue/sport2}
NSH=8
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}
SENT=${SENT:-SCAN-SPORTBR2B}

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest "$@"; }

echo "=== $SENT START worker=$W/$NW chips=$NCHIP scans=[C3X D4] NSH=$NSH $(date -u +%FT%TZ) ==="

# npz fetch (the chain's line verbatim — the gen-1 FileNotFound fix) + compile cache
mkdir -p "$(dirname "$NPZ")"
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/$(basename "$NPZ")" "$NPZ" || { echo "NPZ-MISSING $(basename "$NPZ")"; exit 2; }
echo "NPZ-OK $(basename "$NPZ")"
export JAX_COMPILATION_CACHE_DIR="$PWD/jax_cache"
mkdir -p "$PWD/jax_cache"
gsutil -q cp "$GCS/jax_cache.tgz" /tmp/jc.tgz 2>/dev/null && tar xzf /tmp/jc.tgz 2>/dev/null && echo "COMPILE-CACHE restored ($(ls jax_cache | wc -l | tr -d ' ') entries)"

fetch_ckpt () {  # ARM -> ensure runs/pretrain${R_TAG}_${ARM}/ckpt_latest.pkl locally
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  mkdir -p "$D"
  [ -f "$D/ckpt_latest.pkl" ] || { gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" && echo "FETCH-$arm-CKPT-OK"; } || { echo "FETCH-$arm-CKPT-FAILED"; return 1; }
}

partial_sync () {  # OUTDIR NS — cooperative background pusher: stops on flag
  # file (no signal games -> no orphan class); iteration-capped as a belt.
  local O=$1 NS=$2 i=0
  rm -f "$O/.sync_stop"
  while [ ! -f "$O/.sync_stop" ] && [ "$i" -lt 200 ]; do
    for f in "$O"/partial_*.npz; do [ -f "$f" ] && gsutil -q cp "$f" "$GCS/$NS/partials/$(basename "$f")" 2>/dev/null; done
    sleep "${SCAN_PARTIAL_SLEEP:-300}"; i=$((i+1))
  done
}

run_shard () {  # ARM NS K CHIP — one shard, idempotent, busy-retry, immutable bank
  local arm=$1 NS=$2 K=$3 c=$4
  gsutil -q stat "$GCS/$NS/summary_s$K.json" 2>/dev/null && { echo "SHARD-SKIP $arm s$K (banked)"; return 0; }
  local D=runs/pretrain${R_TAG}_$arm CK O=runs/sxscan_p${R_TAG}${arm}
  CK="$D/ckpt_latest.pkl"
  mkdir -p "$O"
  for f in $(gsutil ls "$GCS/$NS/partials/partial_*.npz" 2>/dev/null); do
    b=$(basename "$f"); [ -f "$O/$b" ] || gsutil -q cp "$f" "$O/$b" 2>/dev/null || true
  done
  echo "SHARD-START $arm s$K chip=$c $(date -u +%H:%M)"
  for try in $(seq 1 60); do
    if pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --shard "$K/$NSH" --out "$O" --bank-every 300 --batch "$EVAL_BATCH" \
        --split test --subsample "$SX_SUB" --t-total 64 --k-init "$SX_SUB_K" > "$O/shard_s$K.log" 2>&1; then
      gsutil -q cp "$O/summary_s$K.json" "$GCS/$NS/summary_s$K.json" && gsutil -q cp "$O/records_s$K.npz" "$GCS/$NS/records_s$K.npz" \
        && { echo "SHARD-OK $arm s$K $(date -u +%H:%M)"; return 0; } || { echo "SHARD-BANK-FAILED $arm s$K"; return 1; }
    fi
    if grep -qE "resource busy|Couldn't open iommu group" "$O/shard_s$K.log"; then
      echo "SHARD-WAIT $arm s$K (chip busy; retry $try)"; sleep 120; continue
    fi
    echo "SHARD-FAILED $arm s$K (see shard log)"; return 1
  done
  echo "SHARD-FAILED $arm s$K (retries exhausted)"; return 1
}

run_scan_block () {  # ARM NS FIRST_SHARD COUNT — COUNT shards in parallel on chips 0..COUNT-1
  local arm=$1 NS=$2 first=$3 count=$4
  fetch_ckpt "$arm" || return 1
  echo "$NSH" | gsutil -q cp - "$GCS/$NS/NSH.txt" 2>/dev/null || true
  local O=runs/sxscan_p${R_TAG}${arm}; mkdir -p "$O"
  partial_sync "$O" "$NS" >/dev/null 2>&1 & local PS=$!
  local pids=() rc=0
  for i in $(seq 0 $((count-1))); do
    run_shard "$arm" "$NS" $((first+i)) $((i % NCHIP)) & pids+=($!)
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || rc=1; done
  touch "$O/.sync_stop"; wait $PS 2>/dev/null || true
  return $rc
}

merge_scan () {  # ARM NS — first-to-complete merges (idempotent; n-gated)
  local arm=$1 NS=$2
  gsutil -q stat "$GCS/p4x_${arm}.tgz" 2>/dev/null && { echo "MERGE-SKIP $arm (banked)"; return 0; }
  local n; n=$(gsutil ls "$GCS/$NS/summary_s*.json" 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge "$NSH" ] || { echo "MERGE-WAIT $arm ($n/$NSH shards banked)"; return 1; }
  local O=runs/sxscan_p${R_TAG}${arm}; mkdir -p "$O"
  for f in $(gsutil ls "$GCS/$NS/summary_s*.json" "$GCS/$NS/records_s*.npz" 2>/dev/null); do
    b=$(basename "$f"); [ -f "$O/$b" ] || gsutil -q cp "$f" "$O/$b"
  done
  JAX_PLATFORMS=cpu python3 tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  if [ -f "$O/summary_all.json" ] && ! python3 -c "import json,sys;sys.exit(0 if json.load(open('$O/summary_all.json'))['n']==$SX_SUB else 1)"; then
    echo "MERGE-N-BAD $arm (n != $SX_SUB) — refusing to bank"; rm -f "$O/summary_all.json"; return 1
  fi
  [ -f "$O/summary_all.json" ] || { echo "MERGE-FAILED $arm"; return 1; }
  printf 'RIDER SCAN (PI-decided 2026-08-30 at the rung-2b verdict): PHASE4-identical protocol on %s vb ckpt; NSH=8 static; see ledger RUNG 2B RIDER SCANS registration\n' "$arm" > "$O/LABEL.txt"
  tar czf "/tmp/p4x_${arm}.tgz" "$O" 2>/dev/null && gsutil -q cp "/tmp/p4x_${arm}.tgz" "$GCS/p4x_${arm}.tgz" && echo "SCAN-$arm-OK $(date -u +%H:%M)"
}

# ---------- static assignment ----------
rc=0
if [ "$NW" -ge 4 ]; then
  case $W in
    0) run_scan_block C3X p4xc3x 0 4 || rc=1;;
    1) run_scan_block C3X p4xc3x 4 4 || rc=1;;
    2) run_scan_block D4  p4xd4  0 4 || rc=1;;
    3) run_scan_block D4  p4xd4  4 4 || rc=1;;
  esac
else
  run_scan_block C3X p4xc3x 0 8 || rc=1
  run_scan_block D4  p4xd4  0 8 || rc=1
fi

# ---------- completion (any worker; merges are idempotent + n-gated) ----------
for pass in $(seq 1 "${SCAN_WAIT_PASSES:-150}"); do
  ok1=0; ok2=0
  merge_scan C3X p4xc3x && ok1=1
  merge_scan D4  p4xd4  && ok2=1
  if [ "$ok1" -eq 1 ] && [ "$ok2" -eq 1 ]; then
    echo "scans complete" | gsutil -q cp - "$GCS/p4x_COMPLETE.txt" 2>/dev/null || true
    echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
    if [ "${SELF_TEARDOWN:-0}" = 1 ] && [ -n "${SELF_POD:-}" ] && [ -n "${SELF_ZONE:-}" ]; then
      echo "SELF-TEARDOWN: deleting $SELF_POD in $SELF_ZONE (scans banked) $(date -u +%FT%TZ)"; sleep 20
      gcloud compute tpus tpu-vm delete "$SELF_POD" --zone "$SELF_ZONE" --quiet >/dev/null 2>&1 && echo "SELF-TEARDOWN-ISSUED" || echo "SELF-TEARDOWN-FAILED (supervisor/watchdog tears down)"
    fi
    exit 0
  fi
  echo "$SENT-WORKER-DONE worker=$W waiting (C3X:$ok1 D4:$ok2) pass=$pass $(date -u +%H:%M)"
  sleep "${SCAN_POLL_SLEEP:-120}"
done
echo "$SENT-INCOMPLETE worker=$W (wait passes exhausted)"; exit 1
