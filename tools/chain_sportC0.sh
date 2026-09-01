#!/bin/bash
# Ledger: CHAMPION TRACK PILOT chain (sportC0, 2026-09-01; Plan_2026-09-01 §3).
# Seven native9 d96 arms, ONE-SHOT (NaN -> amputation to last banked grid +
# STOPPED label, no retries), STATIC worker map (no claim system — the scan
# runbook's hardening), per-arm eval battery incl. the NEW standing stats
# (b1_exact / t1r_at_k / majority) + the canvas C3X/D4 rider evals (program
# review §1). Runs 4x4 (v6e-16: 2 arms/worker DP-4) or 1x8 (all arms
# sequential DP-8). Harness: tools/harness_sportC0.sh (offline stub).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src

GCS=${GCS:-gs://qhrrn2-rescue/sportC0}
GCS_R2B=${GCS_R2B:-gs://qhrrn2-rescue/sportBr2b}
R_TAG=${R_TAG:-sportC0}
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0.npz}
GCS_W1=${GCS_W1:-gs://qhrrn2-rescue/sport2}
SENT=${SENT:-SPORTC0}
STEPS=${C0_STEPS:-50000}
MON=${C0_MON:-2000}
PY=${CHAIN_PY:-python3}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest "$@"; }

echo "=== $SENT START worker=$W/$NW chips=$NCHIP $(date -u +%FT%TZ) ==="
mkdir -p "$(dirname "$NPZ")"
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/$(basename "$NPZ")" "$NPZ" || { echo "NPZ-MISSING"; exit 2; }
echo "NPZ-OK"
export JAX_COMPILATION_CACHE_DIR="$PWD/jax_cache"; mkdir -p "$PWD/jax_cache"
gsutil -q cp "$GCS/jax_cache.tgz" /tmp/jc.tgz 2>/dev/null && tar xzf /tmp/jc.tgz 2>/dev/null && echo "COMPILE-CACHE restored"

# ---------- arm flag registry (locked at registration; native9 d96 ws6) ----
common_flags () {
  echo "--sudoku-extreme $NPZ --sudoku-layout native9 --equilibrium \
        --d 96 --width-scale 6 --K 64 --batch 64 --wd 1e-4 --warmup 500 \
        --fpa-k 4 --fpa-eps 0.2 --beta-flux-nl 1e-6 --lr 1e-3 --lr-end 3e-5 \
        --monitor-every $MON --ckpt-every 1000 --val-every 100000 --dp"
}
arm_flags () {  # ARM -> extra flags beyond common (two-stage P3/P3s1/P6 handled in run_arm)
  case $1 in
    P1)   echo "--T 12 --sudoku-aug 100 --seed 0";;
    P2)   echo "--T 12 --sudoku-aug 100 --seed 0 --ri-p 0.5";;
    P2s1) echo "--T 12 --sudoku-aug 100 --seed 1 --ri-p 0.5";;
    P3)   echo "--T 16 --sudoku-aug 100 --seed 0 --ri-p 0.5";;
    P3s1) echo "--T 16 --sudoku-aug 100 --seed 1 --ri-p 0.5";;
    P5)   echo "--T 12 --sudoku-aug 100 --seed 0 --ri-p 0.5 --ni-sigma 0.01";;
    P6)   echo "--T 16 --sudoku-aug 1000 --seed 0 --ri-p 0.5";;
    *)    return 1;;
  esac
}
two_stage () { case $1 in P3|P3s1|P6) return 0;; *) return 1;; esac; }
# Two-phase split (champion recipe): 35k cosine (1e-3 -> 3e-5) + 15k floor
# continuation at constant 3e-5 from the stage-A ckpt, FRESH optimizer — the
# measured C3X pattern verbatim (R2b-5 CONTINUATION-GROWS).
S_A=$((STEPS * 7 / 10)); S_B=$((STEPS - S_A))

nan_check () {  # DIR -> 0 clean / 1 non-finite tail or missing DONE
  ${REAL_PY:-python3} - "$1" <<'EOF'
import json, math, sys
from pathlib import Path
d = Path(sys.argv[1]); ok = True
mp = d / "metrics.jsonl"
if not mp.exists(): sys.exit(1)
rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip() and '"loss"' in l]
tail = rows[-5:]
if not tail: sys.exit(1)
for r in tail:
    if not math.isfinite(r.get("loss", float("nan"))): ok = False
sys.exit(0 if ok else 1)
EOF
}

amputate () {  # DIR STEPS_WANTED — one-shot rule: final = last banked FINITE grid
  local D=$1
  ${REAL_PY:-python3} - "$D" <<'EOF'
import json, math, pickle, sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
grids = sorted(d.glob("ckpt_0*.pkl"))
best = None
for g in reversed(grids):
    try:
        c = pickle.load(open(g, "rb"))
        leaves = [x for x in __import__("jax").tree.leaves(c["state"]) if hasattr(x, "dtype")]
        if all(np.isfinite(np.asarray(x)).all() for x in leaves):
            best = g; break
    except Exception:
        continue
if best is None:
    print("AMPUTATE-FAILED no finite grid"); sys.exit(1)
c = pickle.load(open(best, "rb")); step = int(c["step"])
import shutil; shutil.copy(best, d / "ckpt_latest.pkl")
mp = d / "metrics.jsonl"
if mp.exists():
    keep = []
    for l in mp.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        s = r.get("step", r.get("val", {}).get("step", r.get("monitor", {}).get("step", 0)))
        if s <= step: keep.append(l)
    mp.write_text("\n".join(keep) + "\n")
(d / "STOPPED.txt").write_text(f"STOPPED final step {step} (NaN halt; one-shot amputation, sportC0)\n")
print(f"AMPUTATED to {best.name} step {step}")
EOF
}

pt () {  # DP pretrain with per-host confinement on multi-host (93a79d4)
  if [ "$NW" -ge 2 ]; then
    TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
      JAX_DEFAULT_MATMUL_PRECISION=highest $PY tools/pretrain.py "$@"
  else
    JAX_DEFAULT_MATMUL_PRECISION=highest $PY tools/pretrain.py "$@"
  fi
}

run_pretrain () {  # ARM — ONE-SHOT (any NaN anywhere -> amputate + STOPPED, no stage-B rescue)
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  gsutil -q stat "$GCS/${arm}_PRETRAIN_OK" 2>/dev/null && { echo "PRETRAIN-SKIP $arm"; return 0; }
  local FL; FL="$(common_flags) $(arm_flags "$arm")" || { echo "BAD-ARM $arm"; return 1; }
  echo "PRETRAIN-START $arm $(date -u +%H:%M)"
  if two_stage "$arm"; then
    local DA=runs/pretrain${R_TAG}_${arm}a
    if ! gsutil -q stat "$GCS/${arm}_STAGEA_OK" 2>/dev/null; then
      pt --out "$DA" $FL --steps "$S_A" > "$DA.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ] || ! nan_check "$DA"; then
        # ONE-SHOT: a hot-phase death IS the arm's outcome — the amputated
        # stage-A grid becomes the final; stage B never runs (no rescue).
        echo "PRETRAIN-NAN $arm stage A (rc=$rc) -> amputate + STOP"
        amputate "$DA" || return 1
        mkdir -p "$D"; cp "$DA/ckpt_latest.pkl" "$D/ckpt_latest.pkl"
        cp "$DA/metrics.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
        cp "$DA/STOPPED.txt" "$D/STOPPED.txt"
        for g in "$DA"/ckpt_0*.pkl; do [ -f "$g" ] && cp "$g" "$D/"; done
      else
        gsutil -q cp "$DA/ckpt_latest.pkl" "$GCS/${arm}_stageA_ckpt.pkl" && echo ok | gsutil -q cp - "$GCS/${arm}_STAGEA_OK"
      fi
    else
      mkdir -p "$DA"; [ -f "$DA/ckpt_latest.pkl" ] || gsutil -q cp "$GCS/${arm}_stageA_ckpt.pkl" "$DA/ckpt_latest.pkl"
    fi
    if [ ! -f "$D/STOPPED.txt" ]; then
      # stage B: floor-lr continuation, fresh optimizer (the measured C3X pattern)
      pt --out "$D" $FL --steps "$S_B" --warmup 100 \
         --lr 3e-5 --lr-end 3e-5 --init-from "$DA/ckpt_latest.pkl" > "$D.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ] || ! nan_check "$D"; then
        echo "PRETRAIN-NAN $arm stage B (rc=$rc) -> amputate"; amputate "$D" || return 1
      fi
    fi
  else
    pt --out "$D" $FL --steps "$STEPS" > "$D.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || ! nan_check "$D"; then
      echo "PRETRAIN-NAN $arm (rc=$rc) -> amputate"; amputate "$D" || return 1
    fi
  fi
  tar czf "/tmp/${arm}_pre.tgz" "$D" 2>/dev/null && gsutil -q cp "/tmp/${arm}_pre.tgz" "$GCS/${arm}_pretrain.tgz"
  gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt.pkl"
  [ -f "$D/STOPPED.txt" ] && gsutil -q cp "$D/STOPPED.txt" "$GCS/${arm}_STOPPED.txt"
  echo ok | gsutil -q cp - "$GCS/${arm}_PRETRAIN_OK"
  echo "PRETRAIN-OK $arm $(date -u +%H:%M)"
}

eval_one () {  # NAME ARM_DIR OUTDIR EXTRA... — chip-pinned single eval, idempotent by GCS marker
  local name=$1 CK=$2 O=$3 chip=$4; shift 4
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "EVAL-SKIP $name"; return 0; }
  mkdir -p "$O"
  pin "$chip" $PY tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --out "$O" "$@" > "$O/run.log" 2>&1 \
    || { echo "EVAL-FAILED $name"; return 1; }
  [ -f "$O/summary_all.json" ] || $PY tools/eval_sudoku_extreme.py --merge "$O" >> "$O/run.log" 2>&1
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "EVAL-OK $name $(date -u +%H:%M)"
}

eval_sharded () {  # NAME CK OUTDIR NSH EXTRA... — NSH-way sharded over chips, merged, n-gated
  local name=$1 CK=$2 O=$3 NSH=$4 NGATE=$5; shift 5
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "EVAL-SKIP $name"; return 0; }
  mkdir -p "$O"; local pids=() rc=0
  for i in $(seq 0 $((NSH - 1))); do
    pin $((i % NCHIP)) $PY tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --out "$O" \
        --shard "$i/$NSH" --bank-every 300 "$@" > "$O/shard_$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { echo "EVAL-SHARD-FAILED $name"; return 1; }
  JAX_PLATFORMS=cpu $PY tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  ${REAL_PY:-python3} -c "import json,sys; s=json.load(open('$O/summary_all.json')); sys.exit(0 if s['n']==$NGATE else 1)" \
    || { echo "EVAL-N-BAD $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "EVAL-OK $name $(date -u +%H:%M)"
}

run_arm () {  # ARM — pretrain then its full battery (vb select, screens, fulls, scan, retfm)
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  run_pretrain "$arm" || return 1
  [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl"
  ${REAL_PY:-python3} tools/select_ckpt.py "$D" > "$D/val_best.txt" 2>/dev/null || true
  local VB; VB=$(awk '{print $1}' "$D/val_best.txt" 2>/dev/null | head -1)
  local VBCK="$D/ckpt_latest.pkl"
  [ -n "$VB" ] && [ -f "$D/ckpt_$VB.pkl" ] && VBCK="$D/ckpt_$VB.pkl"
  # screens: strat-512 k256 + uv + (b1/t1r free) at two banked ckpts + vb.
  # Two-stage arms: stage-A end (the hot-phase peak) + stage-B mid — the
  # funnel-vs-training curve across the phase boundary (AE/two-phase watch).
  if two_stage "$arm" && [ ! -f "$D/STOPPED.txt" ]; then
    local DA=runs/pretrain${R_TAG}_${arm}a
    [ -f "$DA/ckpt_latest.pkl" ] && eval_one "screen_${arm}_sAend" "$DA/ckpt_latest.pkl" \
      "runs/sxscreen_p${R_TAG}${arm}_sAend" 0 \
      --split test --stratified 512 --t-total 64 --k-init 256 --vote-unverified
    [ -f "$D/ckpt_005000.pkl" ] && eval_one "screen_${arm}_sB05" "$D/ckpt_005000.pkl" \
      "runs/sxscreen_p${R_TAG}${arm}_sB05" 0 \
      --split test --stratified 512 --t-total 64 --k-init 256 --vote-unverified
  else
    for st in 015000 035000; do
      [ -f "$D/ckpt_$st.pkl" ] || continue
      eval_one "screen_${arm}_s$st" "$D/ckpt_$st.pkl" "runs/sxscreen_p${R_TAG}${arm}_s$st" 0 \
        --split test --stratified 512 --t-total 64 --k-init 256 --vote-unverified
    done
  fi
  eval_one "screen_${arm}_vb" "$VBCK" "runs/sxscreen_p${R_TAG}${arm}_vb" 1 \
    --split test --stratified 512 --t-total 64 --k-init 256 --vote-unverified
  # retfm (batched final-map retention)
  eval_one "retfm_${arm}" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/retfm_t8" 2 \
    --split test --stratified 512 --t-total 8 --init solution --final-map-only
  # full-test cold (423k, sharded)
  eval_sharded "full_${arm}_t64" "$D/ckpt_latest.pkl" "runs/sxeval_p${R_TAG}${arm}/full_t64" "$NCHIP" 422786 \
    --split test --t-total 64
  # 20k breadth scan k128 (the protocol-table stats ride the records)
  eval_sharded "scan_${arm}" "$VBCK" "runs/sxscan_p${R_TAG}${arm}" "$NCHIP" 20000 \
    --split test --subsample 20000 --t-total 64 --k-init 128
  echo ok | gsutil -q cp - "$GCS/${arm}_ARM_OK"
  echo "ARM-OK $arm $(date -u +%H:%M)"
}

riders () {  # canvas C3X/D4 EqR-statistic evals (program review §1) — idle-chip work
  for src in C3X D4; do
    local CK=runs/pretrainsportBr2b_${src}/ckpt_latest.pkl
    [ -f "$CK" ] || gsutil -q cp "$GCS_R2B/${src}_ckpt.pkl" "$CK" || { echo "RIDER-CKPT-MISS $src"; continue; }
    eval_one "rider_${src}_sel5k" "$CK" "runs/sxrider_${src}_sel5k" 3 \
      --split test --subsample 5000 --t-total 64 --k-init 128 --vote-unverified
  done
}

# ---------- static assignment ----------
rc=0
if [ "$NW" -ge 4 ]; then
  case $W in
    0) run_arm P1 || rc=1; run_arm P6 || rc=1;;
    1) run_arm P2 || rc=1; run_arm P2s1 || rc=1;;
    2) run_arm P3 || rc=1; run_arm P5 || rc=1;;
    3) run_arm P3s1 || rc=1; riders || rc=1;;
  esac
else
  for arm in P1 P2 P2s1 P3 P3s1 P5 P6; do run_arm "$arm" || rc=1; done
  riders || rc=1
fi

# ---------- completion (any worker; idempotent) ----------
need="P1 P2 P2s1 P3 P3s1 P5 P6"
for pass in $(seq 1 "${C0_WAIT_PASSES:-200}"); do
  all=1
  for armx in $need; do gsutil -q stat "$GCS/${armx}_ARM_OK" 2>/dev/null || all=0; done
  if [ "$all" -eq 1 ]; then
    # pull everything and package once
    if ! gsutil -q stat "$GCS/sportC0_final.tgz" 2>/dev/null; then
      for f in $(gsutil ls "$GCS/evals/*.tgz" "$GCS"/*_pretrain.tgz 2>/dev/null); do
        b=$(basename "$f"); [ -f "/tmp/pull_$b" ] || { gsutil -q cp "$f" "/tmp/pull_$b" && tar xzf "/tmp/pull_$b" 2>/dev/null; }
      done
      tar czf /tmp/sportC0_final.tgz runs/pretrain${R_TAG}_* runs/sxscreen_p${R_TAG}* runs/sxeval_p${R_TAG}* runs/sxscan_p${R_TAG}* runs/sxrider_* 2>/dev/null
      gsutil -q cp /tmp/sportC0_final.tgz "$GCS/sportC0_final.tgz"
    fi
    tar czf /tmp/jc.tgz jax_cache 2>/dev/null && gsutil -q cp /tmp/jc.tgz "$GCS/jax_cache.tgz" || true
    echo "CHAIN-$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
    if [ "${SELF_TEARDOWN:-0}" = 1 ] && [ -n "${SELF_POD:-}" ] && [ -n "${SELF_ZONE:-}" ]; then
      echo "SELF-TEARDOWN $(date -u +%FT%TZ)"; sleep 20
      gcloud compute tpus tpu-vm delete "$SELF_POD" --zone "$SELF_ZONE" --quiet >/dev/null 2>&1 || echo "SELF-TEARDOWN-FAILED"
    fi
    exit 0
  fi
  echo "$SENT-WORKER-DONE worker=$W waiting pass=$pass $(date -u +%H:%M)"
  sleep "${C0_POLL_SLEEP:-120}"
done
echo "$SENT-INCOMPLETE worker=$W"; exit 1
