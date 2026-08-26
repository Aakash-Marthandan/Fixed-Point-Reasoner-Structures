#!/bin/bash
# TAIL RUNBOOK (2026-08-26, PI-authorized): finish rung 1 after the 14:21Z preemption.
# Work: (a) /16 PHASE4 shards s6+s14 (the only missing ones) with the EXACT chain
# invocation (chain_sportB.sh:394-395), fingerprint-gated against the 14 banked
# shards; (b) PHASE4-MID 20k scan of the winner's mid ckpt (predicate = chain's own
# line 416-422, computed YES from banked screens); (c) evaluator --merge (tested
# tool path) for both; (d) package breadth20k.tgz (chain line 429 verbatim); then
# (e) run the UNMODIFIED chain -> pure SKIP-cascade -> completion guard -> final
# tgz -> CHAIN-SPORTB-COMPLETE -> self-teardown. NEVER run the chain before (d):
# on a non-16 node its NSH partition would hollow-merge (b589334 class).
# Idempotent: every unit is GCS-state-guarded; partials bank every 300s (batch-
# boundary) and sync to GCS every 60s (tightened from 300s per PI 15:0xZ).
set -uo pipefail
cd ~/qhrrn2
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
GCS=gs://qhrrn2-rescue/sportB
GCS_W1=gs://qhrrn2-rescue/sport2
NPZ=data/sudoku_extreme/sudoku_extreme_seed0.npz
D=runs/pretrainsportB_B2
O=runs/sxbreadth20k_psportBB2
OM=runs/sxbreadth20k_psportBB2_mid
SELF_POD=${SELF_POD:-qhrrn2-pod2}; SELF_ZONE=${SELF_ZONE:?need SELF_ZONE}
log () { echo "$(date -u +%FT%TZ) $*"; }
trap 'gsutil -q cp ~/qhrrn2/runs/tail_runbook.log "$GCS/tail_runbook_last.log" 2>/dev/null' EXIT

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest "$@"; }   # chain_sportB.sh:64 verbatim
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -cE '^[0-9]+$'); log "TAIL-START chips=$NCHIP zone=$SELF_ZONE"

# ---------- data + reference provenance ----------
mkdir -p "$D" "$O" "$OM" data/sudoku_extreme
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/sudoku_extreme_seed0.npz" "$NPZ" || { log "NPZ-MISSING"; exit 2; }
gsutil -q cp "$GCS/p4/summary_s0.json" /tmp/ref_s0.json || { log "REF-S0-MISSING"; exit 2; }
# The banked artifacts fully specify the instrument: take ckpt/npz paths from the
# reference fingerprint fields, never from assumptions.
REF_CK=$(python3 -c "import json;print(json.load(open('/tmp/ref_s0.json'))['ckpt'])")
REF_NPZ=$(python3 -c "import json;print(json.load(open('/tmp/ref_s0.json'))['npz'])")
[ "$REF_NPZ" = "$NPZ" ] || { log "REF-NPZ-MISMATCH ref=$REF_NPZ"; exit 2; }
case "$REF_CK" in "$D"/ckpt_*.pkl) :;; *) log "REF-CKPT-UNEXPECTED $REF_CK"; exit 2;; esac
STEP=$(basename "$REF_CK" .pkl); STEP=${STEP#ckpt_}
[ -f "$REF_CK" ] || gsutil -q cp "$GCS/B2_ckpt_$STEP.pkl" "$REF_CK" || { log "CKPT-PULL-FAILED B2_ckpt_$STEP"; exit 2; }
gsutil -q cp "$GCS/B2_val_best.txt" "$D/val_best.txt" 2>/dev/null || true
VB=$(cut -d' ' -f1 "$D/val_best.txt" 2>/dev/null)
[ "$VB" = "$STEP" ] || log "WARN vb_step=$VB != ref ckpt step=$STEP (proceeding on the banked reference)"
# mid step: chain_sportB.sh mid_step formula verbatim
MS=$(python3 -c "vb=int('$STEP'); mid=max(5000, round(vb/2/5000)*5000); print(f'{mid:06d}' if mid!=vb else '')")
[ -n "$MS" ] || { log "MID-STEP-EMPTY (mid==vb?) — protocol violation, stopping"; exit 2; }
MCK=$D/ckpt_$MS.pkl
[ -f "$MCK" ] || gsutil -q cp "$GCS/B2_ckpt_$MS.pkl" "$MCK" || { log "MID-CKPT-PULL-FAILED B2_ckpt_$MS"; exit 2; }
log "REFS ck=$REF_CK mid=$MCK"

# ---------- partial restore + 60s GCS sync (chain partial_sync, cadence 60s) ----------
for OD in "$O" "$OM"; do base=$(basename "$OD")
  for f in $(gsutil ls "$GCS/partials/${base}_partial_*.npz" 2>/dev/null); do
    b=$(basename "$f"); b=${b#${base}_}
    [ -f "$OD/$b" ] || gsutil -q cp "$f" "$OD/$b" 2>/dev/null
  done
done
( while true; do sleep 60
    for OD in "$O" "$OM"; do base=$(basename "$OD")
      for f in "$OD"/partial_*.npz; do [ -f "$f" ] && gsutil -q cp "$f" "$GCS/partials/${base}_$(basename "$f")" 2>/dev/null; done
    done
  done ) & SYNC=$!
trap 'pkill -P $SYNC 2>/dev/null; kill $SYNC 2>/dev/null; gsutil -q cp ~/qhrrn2/runs/tail_runbook.log "$GCS/tail_runbook_last.log" 2>/dev/null' EXIT

# ---------- fingerprint gate (provenance fields ONLY — never metrics) ----------
gate () {  # SUMMARY REF -> 0 ok; prints differing provenance fields
  python3 - "$1" "$2" <<'PY'
import json, sys
KEYS = ["ckpt","npz","split","t_total","k_init","init","layout","fpopt_gamma","tau",
        "stratified","subsample","subsample_seed","mi_seed","eta","eta_z","T","d",
        "eta_learned","eta_override","final_map_only","eq_coupled_ab","n"]  # n: /16 shards are exactly 1250 each
mine, ref = (json.load(open(p)) for p in sys.argv[1:3])
bad = [k for k in KEYS if mine.get(k) != ref.get(k)]
for k in bad: print(f"GATE-DIFF {k}: mine={mine.get(k)!r} ref={ref.get(k)!r}")
sys.exit(1 if bad else 0)
PY
}

# ---------- (a) /16 shards s6 + s14 on chips 0,1 (chain :394-395 verbatim flags) ----------
run16 () {  # SHARD CHIP
  local K=$1 c=$2
  gsutil -q stat "$GCS/p4/summary_s$K.json" 2>/dev/null && { log "SKIP-s$K (banked)"; return 0; }
  for try in $(seq 1 300); do
    pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$REF_CK" --npz "$NPZ" --shard "$K/16" --out "$O" --bank-every 300 \
        --split test --subsample 20000 --t-total 64 --k-init 128 > "$O/shard_s$K.log" 2>&1 \
      && { if gate "$O/summary_s$K.json" /tmp/ref_s0.json; then
             gsutil -q cp "$O/summary_s$K.json" "$GCS/p4/summary_s$K.json" && gsutil -q cp "$O/records_s$K.npz" "$GCS/p4/records_s$K.npz" \
               && log "TAIL-s$K-OK (gate passed, banked)"
           else log "TAIL-s$K-GATE-FAIL — NOT uploading (instrument mismatch; investigate)"; return 1; fi; return 0; }
    grep -qE "resource busy|Couldn't open iommu group" "$O/shard_s$K.log" && { sleep 120; continue; }
    log "TAIL-s$K-FAILED (see shard_s$K.log)"; return 1
  done; return 1
}
# ---------- (b) MID 20k scan, 6-way on chips 2..7 (chain :426 flags; per-shard GCS bank) ----------
runmid () {  # SHARD CHIP  (NSH_MID-way self-consistent partition)
  local j=$1 c=$2
  gsutil -q stat "$GCS/p4mid/summary_s$j.json" 2>/dev/null && \
    { gsutil -q cp "$GCS/p4mid/summary_s$j.json" "$OM/summary_s$j.json"; gsutil -q cp "$GCS/p4mid/records_s$j.npz" "$OM/records_s$j.npz"; log "SKIP-mid-s$j (banked)"; return 0; }
  for try in $(seq 1 300); do
    pin "$c" python3 tools/eval_sudoku_extreme.py --ckpt "$MCK" --npz "$NPZ" --shard "$j/$NSH_MID" --out "$OM" --bank-every 300 \
        --split test --subsample 20000 --t-total 64 --k-init 128 > "$OM/shard_s$j.log" 2>&1 \
      && { gsutil -q cp "$OM/summary_s$j.json" "$GCS/p4mid/summary_s$j.json"; gsutil -q cp "$OM/records_s$j.npz" "$GCS/p4mid/records_s$j.npz"; log "MID-s$j-OK (banked)"; return 0; }
    grep -qE "resource busy|Couldn't open iommu group" "$OM/shard_s$j.log" && { sleep 120; continue; }
    log "MID-s$j-FAILED (see mid shard_s$j.log)"; return 1
  done; return 1
}
# mid partition PINNED in GCS: a resume on a different node shape must reuse the
# original NSH_MID or banked /K shards would mix partitions.
if NSH_MID=$(gsutil -q cat "$GCS/p4mid/NSH.txt" 2>/dev/null) && [ -n "$NSH_MID" ]; then
  log "MID partition pinned: $NSH_MID-way (from GCS)"
else
  # >=8 chips: 6-way on the chips not running s6/s14 (full overlap). <8 chips:
  # NCHIP-way over ALL chips — mid shards on chips 0,1 queue behind s6/s14 via
  # the busy-retry loop (best wall on a v6e-4: ~10h vs 16h for 2-way).
  if [ "$NCHIP" -ge 8 ]; then NSH_MID=$((NCHIP - 2)); else NSH_MID=$NCHIP; fi
  echo "$NSH_MID" | gsutil -q cp - "$GCS/p4mid/NSH.txt"
fi
midchip () { if [ "$NCHIP" -ge 8 ]; then echo $((2 + $1 % (NCHIP-2))); else echo $(($1 % NCHIP)); fi; }
if ! gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null; then
  pids=(); rc16=0
  run16 6 0 & pids+=($!)
  run16 14 1 & pids+=($!)
  for j in $(seq 0 $((NSH_MID-1))); do runmid "$j" "$(midchip "$j")" & pids+=($!); done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || { log "TAIL-SHARDS-INCOMPLETE — stopping before merge"; exit 1; }

  # ---------- (c) merges: evaluator --merge (tested path), count-gated ----------
  n=$(gsutil ls "$GCS/p4/summary_s*.json" 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge 16 ] || { log "P4-COUNT-$n-of-16 — refusing merge"; exit 1; }
  for f in $(gsutil ls "$GCS/p4/summary_s*.json" "$GCS/p4/records_s*.npz" 2>/dev/null); do
    b=$(basename "$f"); [ -f "$O/$b" ] || gsutil -q cp "$f" "$O/$b"
  done
  JAX_PLATFORMS=cpu python3 tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  [ -f "$O/summary_all.json" ] || { log "MERGE-FAILED (see merge.log)"; exit 1; }
  NN=$(python3 -c "import json;print(json.load(open('$O/summary_all.json'))['n'])")
  [ "$NN" = "20000" ] || { log "MERGE-N-BAD n=$NN != 20000"; exit 1; }
  log "P4-MERGE-OK n=20000 (16/16 shards)"
  JAX_PLATFORMS=cpu python3 tools/eval_sudoku_extreme.py --merge "$OM" > "$OM/merge.log" 2>&1
  [ -f "$OM/summary_all.json" ] || { log "MID-MERGE-FAILED"; exit 1; }
  NM=$(python3 -c "import json;print(json.load(open('$OM/summary_all.json'))['n'])")
  [ "$NM" = "20000" ] || { log "MID-MERGE-N-BAD n=$NM != 20000"; exit 1; }
  log "MID-MERGE-OK n=20000 ($NSH_MID/$NSH_MID shards)"

  # ---------- (d) package (chain :429 verbatim tar) ----------
  tar czf /tmp/breadth20k.tgz runs/sxbreadth20k_psportB* 2>/dev/null && gsutil -q cp /tmp/breadth20k.tgz "$GCS/breadth20k.tgz" \
    && log "BREADTH20K-BANKED (winner + mid)" || { log "BREADTH20K-PACKAGE-FAILED"; exit 1; }
else log "SKIP shards+merge (breadth20k.tgz already banked)"; fi

# ---------- (e) completion via the UNMODIFIED chain (pure SKIP-cascade now) ----------
gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null || { log "GUARD: breadth20k missing — refusing chain"; exit 1; }
pkill -P $SYNC 2>/dev/null; kill $SYNC 2>/dev/null || true
log "COMPLETION: launching unmodified chain (SKIP-cascade -> final tgz -> sentinel -> teardown)"
env ARMS_W0='B1' ARMS_W1='B2' ARMS_W2='B3' ARMS_W3='B4 B4s1 B5' PRIMARY='B1 B2 B3 B4' RD=64 WS=4 \
    SCREEN_K=256 SX_NPZ=sudoku_extreme_seed0.npz SX_AUG=100 SX_T_STRAT='6 64 256' SX_K_INIT=16 SX_STRAT=512 \
    SX_SUB=20000 SX_SUB_K=128 SX_RET_T=8 MON_EVERY=5000 GCS_W1=$GCS_W1 GCS=$GCS \
    CHAIN_WORKER=0 CHAIN_WORKERS=1 R_TAG=sportB \
    SELF_TEARDOWN=1 SELF_POD="$SELF_POD" SELF_ZONE="$SELF_ZONE" \
    bash tools/chain_sportB.sh
log "TAIL-RUNBOOK-DONE"
