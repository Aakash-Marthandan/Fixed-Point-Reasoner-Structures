#!/bin/bash
# Ledger: sportC2 chain (2026-09-04; Plan_2026-09-04_sportC2 §2-§6) — the pre-parity graft night
# at d128. Derivative of chain_sportC1.sh (identical mechanics: static worker map, ONE-SHOT NaN
# amputation, per-arm banking, idempotent markers, n-gated merges, full at vsel AND final, census
# on both grids, stage-A restore on a node change) + the sportC2 additions: the stall-calibration
# eval on every vsel grid, R4's continuation init-from sportC1's R0, the B0/B1 correct-grid riders.
# ARMS: W0 (B0 + wd 1.0, the base) · R1 (+ persistent carry) · R2 (+ inner cycles K3) · R3 (+ hard
# rows) · R4 (R0 +50k, field regime) · X1 (X0 - digit aug) · X2 (X0 + our group mixer).
# 4x4 static map: w0 W0 R2 · w1 R1 R3 · w2 R4 X1 · w3 X2 + riders. Harness: tools/harness_sportC2.sh.
# Pre-mortem additions (2026-09-04): pt_run = ONE --remat retry on a launch-time HBM OOM (no step logged; labeled,
# never a NaN death; harness S9); R3's labeled hard-feedback inference rows (screen_R3_vb_hard, calib_R3_vsel_hard).
# (sportC1 header follows for the mechanics it inherits:)
# Derivative of the proven chain_sportC0.sh (static worker
# map, ONE-SHOT NaN amputation, per-arm GCS banking, idempotent markers,
# n-gated merges) plus the pilot's hardening: early-NaN-abort rc handled,
# post-death screens skipped, stage-A dir + config.json banked, LOUD
# VB-FALLBACK-FINAL with a pretrain re-pull before select_ckpt, full test at
# vsel AND final (+ the alternate-weights row), explosion census on vsel AND
# final grids, per-arm headline weights (raw on A/B, EMA on R0/X0/X0n) and
# per-arm matmul precision (bf16 passes on the field-recipe arms).
# ARMS: A0 A1 (d128 champion recipe, controls) · B0 B1 (+ z-norm) · R0 (our
# cell, FIELD optimizer regime) · X0 (FIELD baseline: TRM/EqR cell + SOT +
# ACT) · X0n (X0 without ACT; descriptive rider). 4x4 static map: w0 A0 B0 ·
# w1 A1 B1 · w2 R0 · w3 X0 X0n + canvas riders. Harness: tools/harness_sportC1.sh.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src

GCS=${GCS:-gs://qhrrn2-rescue/sportC2}
GCS_C1=${GCS_C1:-gs://qhrrn2-rescue/sportC1}
GCS_R2B=${GCS_R2B:-gs://qhrrn2-rescue/sportBr2b}
R_TAG=${R_TAG:-sportC2}
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0.npz}
GCS_W1=${GCS_W1:-gs://qhrrn2-rescue/sport2}
SENT=${SENT:-SPORTC2}
STEPS_AB=${C1_STEPS_AB:-80000}          # two-phase total: 50k cosine (1e-3 -> 3e-5) + 30k floor @3e-5
S_A=$((STEPS_AB * 5 / 8)); S_B=$((STEPS_AB - S_A))
STEPS_R=${C1_STEPS_R:-50000}            # R0: the field's 50k steps (batch 384)
STEPS_X=${C1_STEPS_X:-50000}            # X0/X0n: the field's 50k SOT steps (batch 768)
MON=${C1_MON:-2000}
SOT_SEG=${C2_SOT_SEGMENTS:-4}
SUB=${C1_SUB:-20000}; STRAT=${C1_STRAT:-512}
PY=${CHAIN_PY:-python3}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}
ARM_PREC=highest   # per-arm matmul precision for pinned evals (set by run_arm)

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=$ARM_PREC "$@"; }

echo "=== $SENT START worker=$W/$NW chips=$NCHIP $(date -u +%FT%TZ) ==="
mkdir -p "$(dirname "$NPZ")"
[ -f "$NPZ" ] || gsutil -q cp "$GCS_W1/$(basename "$NPZ")" "$NPZ" || { echo "NPZ-MISSING"; exit 2; }
echo "NPZ-OK"
export JAX_COMPILATION_CACHE_DIR="$PWD/jax_cache"; mkdir -p "$PWD/jax_cache"
gsutil -q cp "$GCS/jax_cache.tgz" /tmp/jc.tgz 2>/dev/null && tar xzf /tmp/jc.tgz 2>/dev/null && echo "COMPILE-CACHE restored"

# ---------- arm flag registry (locked at registration; plan §12) ----------
native_common () {  # d128 ws8 native9 = 3,004,530 params; every measured mechanism of the champion recipe
  echo "--sudoku-extreme $NPZ --sudoku-layout native9 --equilibrium --d 128 --width-scale 8 --K 64 \
        --T 16 --ri-p 0.5 --fpa-k 4 --fpa-eps 0.2 --beta-flux-nl 1e-6 --sudoku-aug 1000 \
        --monitor-every $MON --ckpt-every 1000 --val-every 100000 --dp"
}
our_regime ()   { echo "--batch 64 --wd 1e-4 --warmup 500 --lr 1e-3 --lr-end 3e-5 --ema 0.999"; }
w0_regime ()    { echo "--batch 64 --wd 1.0 --warmup 500 --lr 1e-3 --lr-end 3e-5 --ema 0.999"; }   # sportC2 W0: the one variable
field_regime () { echo "--batch 384 --wd 1.0 --warmup 2000 --lr 1e-4 --lr-end 1e-4 --beta2 0.95 --ema 0.999"; }
x_common () {       # the field-recipe baseline: TRM/EqR cell + SOT loop in the field's regime (sportC1 plan §11.2)
  echo "--sudoku-extreme $NPZ --sudoku-layout native9 --equilibrium --cell trm --sot \
        --trm-hidden 512 --trm-layers 2 --trm-h-cycles 3 --trm-l-cycles 6 --T 16 \
        --trm-lambda 0.05 --trm-beta 0.01 --sudoku-aug 1000 --loss stablemax \
        --batch 768 --wd 1.0 --warmup 2000 --lr 1e-4 --lr-end 1e-4 --beta2 0.95 --ema 0.999 \
        --fpa-k 0 --beta-flux-nl 0 --monitor-every $MON --ckpt-every 1000 --val-every 100000 --dp"
}
arm_flags () {   # sportC2 registry (plan §2): one variable per arm from its base
  case $1 in
    W0)  echo "$(native_common) $(w0_regime) --seed 0 --z-norm rms";;
    R1)  echo "$(native_common) $(w0_regime) --seed 0 --z-norm rms --sot --sot-segments $SOT_SEG";;
    R2)  echo "$(native_common) $(w0_regime) --seed 0 --z-norm rms --inner-k 3";;
    R3)  echo "$(native_common) $(w0_regime) --seed 0 --z-norm rms --hard-p 0.5";;
    R4)  echo "$(native_common) $(field_regime) --seed 0 --z-norm rms --remat --init-from runs/pretrainsportC1_R0/ckpt_latest.pkl";;
    X1)  echo "$(x_common) --act --seed 0";;
    X2)  echo "$(x_common) --sudoku-digit-aug --act --seed 0 --trm-token-mixer group9";;
    *)   return 1;;
  esac
}
two_stage () { case $1 in W0|R1|R2|R3) return 0;; *) return 1;; esac; }
is_field ()  { case $1 in X1|X2) return 0;; *) return 1;; esac; }
arm_steps () { case $1 in R4) echo "$STEPS_R";; X1|X2) echo "$STEPS_X";; *) echo "$STEPS_AB";; esac; }
arm_prec ()  { case $1 in X1|X2) echo default;; *) echo highest;; esac; }   # the field trains/evals in bf16
head_ema ()  { case $1 in R4|X1|X2) echo "--ema";; *) echo "";; esac; }     # headline weights
alt_ema ()   { case $1 in R4|X1|X2) echo "";; *) echo "--ema";; esac; }
select_key (){ case $1 in R4) echo val_t64_ema;; X1|X2) echo val_t16_ema;; *) echo val_t64;; esac; }
head_t ()    { if is_field "$1"; then echo 16; else echo 64; fi; }             # X headline = EqR's D=16
screen_steps () {  # fixed-step screens (vb rides separately); two-stage: stage-A end + B+5k + B+15k
  case $1 in R4|X1|X2) echo "015000 035000";; *) echo "";; esac; }
fetch_init () {  # R4: the sportC1 R0 checkpoint it continues from (idempotent)
  [ "$1" = R4 ] || return 0
  mkdir -p runs/pretrainsportC1_R0
  [ -f runs/pretrainsportC1_R0/ckpt_latest.pkl ] || gsutil -q cp "$GCS_C1/R0_ckpt.pkl" runs/pretrainsportC1_R0/ckpt_latest.pkl || { echo "INIT-CKPT-MISSING R4"; return 1; }
}

nan_check () {  # DIR -> 0 clean / 1 non-finite tail or missing metrics (the trainer's NAN-ABORT rc=3 lands here too)
  ${REAL_PY:-python3} - "$1" <<'PYEOF'
import json, math, sys
from pathlib import Path
d = Path(sys.argv[1]); mp = d / "metrics.jsonl"
if (d / "NAN_ABORT.txt").exists(): sys.exit(1)
if not mp.exists(): sys.exit(1)
rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip() and '"loss"' in l]
tail = rows[-5:]
if not tail: sys.exit(1)
sys.exit(0 if all(math.isfinite(r.get("loss", float("nan"))) for r in tail) else 1)
PYEOF
}

amputate () {  # DIR — one-shot rule: final = last banked FINITE grid; post-death grids REMOVED (never screened)
  ${REAL_PY:-python3} - "$1" <<'PYEOF'
import json, math, pickle, shutil, sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
def finite(tree):
    st = [tree]
    while st:
        x = st.pop()
        if isinstance(x, dict): st.extend(x.values())
        elif isinstance(x, (list, tuple)): st.extend(x)
        elif hasattr(x, "dtype"):
            a = np.asarray(x)
            if a.dtype.kind in "fc" and not np.isfinite(a).all(): return False
    return True
grids = sorted(d.glob("ckpt_0*.pkl")); best = None
for g in reversed(grids):
    try:
        c = pickle.load(open(g, "rb"))
        if finite(c["state"]): best = g; break
    except Exception:
        continue
if best is None:
    print("AMPUTATE-FAILED no finite grid"); sys.exit(1)
c = pickle.load(open(best, "rb")); step = int(c["step"])
shutil.copy(best, d / "ckpt_latest.pkl")
for g in grids:
    try:
        if int(pickle.load(open(g, "rb"))["step"]) > step: g.unlink()
    except Exception:
        g.unlink()
mp = d / "metrics.jsonl"
if mp.exists():
    keep = []
    for l in mp.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        s = r.get("step", r.get("val", {}).get("step", r.get("monitor", {}).get("step", 0)))
        if s <= step: keep.append(l)
    mp.write_text("\n".join(keep) + "\n")
(d / "STOPPED.txt").write_text(f"STOPPED final step {step} (NaN halt; one-shot amputation, sportC1)\n")
print(f"AMPUTATED to {best.name} step {step}")
PYEOF
}

pt () {  # DP pretrain with per-host confinement on multi-host (93a79d4); the ARM's matmul precision
  if [ "$NW" -ge 2 ]; then
    TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
      JAX_DEFAULT_MATMUL_PRECISION=$ARM_PREC $PY tools/pretrain.py "$@"
  else
    JAX_DEFAULT_MATMUL_PRECISION=$ARM_PREC $PY tools/pretrain.py "$@"
  fi
}

has_loss_rows () { [ -f "$1/metrics.jsonl" ] && grep -q '"loss"' "$1/metrics.jsonl"; }
pt_run () {  # LOG ARM DIR pt-args... — sportC2 pre-mortem (2026-09-04): a LAUNCH-TIME HBM exhaustion (rc != 0 before
  # any step was logged, an OOM message in the log, and no --remat in the args) is NOT a NaN death and must not be
  # amputated as one: ONE retry with --remat (numerics-equivalent: tests/test_eq_remat), labeled on disk
  # (RETRY_REMAT.txt ships in the pretrain tarball). Any failure after the first logged step stays one-shot.
  local log=$1 arm=$2 dir=$3; shift 3
  pt "$@" > "$log" 2>&1; local rc=$?
  if [ $rc -ne 0 ] && ! has_loss_rows "$dir" && ! printf '%s\n' "$@" | grep -qx -- '--remat' \
     && grep -qiE 'RESOURCE_EXHAUSTED|out of memory' "$log"; then
    echo "PRETRAIN-OOM-RETRY-REMAT $arm (rc=$rc before any logged step = HBM exhaustion, not a NaN death; one retry with --remat, numerics-equivalent, labeled)"
    mkdir -p "$dir"; echo "OOM at launch (rc=$rc) -> retried once with --remat $(date -u +%FT%TZ)" >> "$dir/RETRY_REMAT.txt"
    pt "$@" --remat >> "$log" 2>&1; rc=$?
  fi
  return $rc
}

bank_dir () {  # ARM DIR NAME — tar a run dir to GCS (grids + metrics + config + resumes)
  tar czf "/tmp/$3.tgz" "$2" 2>/dev/null && gsutil -q cp "/tmp/$3.tgz" "$GCS/$3.tgz"
}

run_pretrain () {  # ARM — ONE-SHOT (any NaN anywhere -> amputate + STOPPED, no stage-B rescue)
  local arm=$1 D=runs/pretrain${R_TAG}_$1 rc
  gsutil -q stat "$GCS/${arm}_PRETRAIN_OK" 2>/dev/null && { echo "PRETRAIN-SKIP $arm"; return 0; }
  local FL; FL="$(arm_flags "$arm")" || { echo "BAD-ARM $arm"; return 1; }
  echo "PRETRAIN-START $arm $(date -u +%H:%M) prec=$ARM_PREC"
  if two_stage "$arm"; then
    local DA=runs/pretrain${R_TAG}_${arm}a
    if ! gsutil -q stat "$GCS/${arm}_STAGEA_OK" 2>/dev/null; then
      pt_run "$DA.log" "$arm" "$DA" --out "$DA" $FL --steps "$S_A"; rc=$?
      if [ $rc -ne 0 ] || ! nan_check "$DA"; then
        # ONE-SHOT: a hot-phase death IS the arm's outcome — the amputated stage-A grid
        # becomes the final; stage B never runs; config.json + the stage-A dir ship (§4.3).
        echo "PRETRAIN-NAN $arm stage A (rc=$rc) -> amputate + STOP"
        amputate "$DA" || return 1
        mkdir -p "$D"; cp "$DA/ckpt_latest.pkl" "$D/ckpt_latest.pkl"
        cp "$DA/metrics.jsonl" "$D/metrics.jsonl" 2>/dev/null || true
        cp "$DA/config.json" "$D/config.json" 2>/dev/null || true
        cp "$DA/resumes.txt" "$D/resumes.txt" 2>/dev/null || true
        cp "$DA/STOPPED.txt" "$D/STOPPED.txt"
        for g in "$DA"/ckpt_0*.pkl; do [ -f "$g" ] && cp "$g" "$D/"; done
        bank_dir "$arm" "$DA" "${arm}a_pretrain"
      else
        bank_dir "$arm" "$DA" "${arm}a_pretrain"       # §4.3: stage-A dir + metrics + config banked
        gsutil -q cp "$DA/ckpt_latest.pkl" "$GCS/${arm}_stageA_ckpt.pkl" && echo ok | gsutil -q cp - "$GCS/${arm}_STAGEA_OK"
        echo "STAGEA-OK $arm $(date -u +%H:%M)"
      fi
    else
      mkdir -p "$DA"; [ -f "$DA/ckpt_latest.pkl" ] || gsutil -q cp "$GCS/${arm}_stageA_ckpt.pkl" "$DA/ckpt_latest.pkl"
    fi
    if [ ! -f "$D/STOPPED.txt" ]; then
      # stage B: floor-lr continuation, fresh optimizer (the measured C3X pattern; R2b-5)
      pt_run "$D.log" "$arm" "$D" --out "$D" $FL --steps "$S_B" --warmup 100 \
         --lr 3e-5 --lr-end 3e-5 --init-from "$DA/ckpt_latest.pkl"; rc=$?
      if [ $rc -ne 0 ] || ! nan_check "$D"; then
        echo "PRETRAIN-NAN $arm stage B (rc=$rc) -> amputate"; amputate "$D" || return 1
      fi
    fi
  else
    fetch_init "$arm" || return 1
    pt_run "$D.log" "$arm" "$D" --out "$D" $FL --steps "$(arm_steps "$arm")"; rc=$?
    if [ $rc -ne 0 ] || ! nan_check "$D"; then
      echo "PRETRAIN-NAN $arm (rc=$rc) -> amputate"; amputate "$D" || return 1
    fi
  fi
  bank_dir "$arm" "$D" "${arm}_pretrain"
  gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt.pkl"
  [ -f "$D/STOPPED.txt" ] && gsutil -q cp "$D/STOPPED.txt" "$GCS/${arm}_STOPPED.txt"
  echo ok | gsutil -q cp - "$GCS/${arm}_PRETRAIN_OK"
  echo "PRETRAIN-OK $arm $(date -u +%H:%M)"
}

ensure_local_pretrain () {  # ARM — §4.4: after a node change the banked grids + metrics must be local before select_ckpt
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  if [ ! -f "$D/metrics.jsonl" ] || [ -z "$(ls "$D"/ckpt_0*.pkl 2>/dev/null)" ]; then
    if gsutil -q cp "$GCS/${arm}_pretrain.tgz" "/tmp/${arm}_pre_pull.tgz" 2>/dev/null; then
      tar xzf "/tmp/${arm}_pre_pull.tgz" 2>/dev/null && echo "PRETRAIN-RESTORE $arm (grids + metrics re-pulled from GCS)"
    fi
  fi
  mkdir -p "$D"
  [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl"
  [ -f "$D/STOPPED.txt" ] || gsutil -q cp "$GCS/${arm}_STOPPED.txt" "$D/STOPPED.txt" 2>/dev/null || true
  # sportC1 analysis pass 2026-09-03 (PROVENANCE DEFECT, PM-1 class re-entering via the resume path): a two-stage
  # arm's STAGE-A dir must be local too, else the both-stage val-selection silently degrades to stage-B-only on a
  # new node (B0/B1's scans + vsel censuses ran on memorized stage-B grids after the 16:23Z preemption).
  if two_stage "$arm" && [ ! -f "$D/STOPPED.txt" ]; then
    local DA=runs/pretrain${R_TAG}_${arm}a
    if [ ! -f "$DA/metrics.jsonl" ] || [ -z "$(ls "$DA"/ckpt_0*.pkl 2>/dev/null)" ]; then
      if gsutil -q cp "$GCS/${arm}a_pretrain.tgz" "/tmp/${arm}a_pre_pull.tgz" 2>/dev/null; then
        tar xzf "/tmp/${arm}a_pre_pull.tgz" 2>/dev/null && echo "PRETRAIN-RESTORE ${arm}a (stage-A grids + metrics re-pulled from GCS; selection stays two-stage)"
      else
        echo "STAGEA-RESTORE-MISSING $arm (no ${arm}a_pretrain.tgz in GCS: selection is STAGE-B-ONLY — labeled)"
      fi
    fi
  fi
}

stopped_step () { grep -oE 'step [0-9]+' "runs/pretrain${R_TAG}_$1/STOPPED.txt" 2>/dev/null | awk '{print $2}' | head -1; }

eval_one () {  # NAME CK OUTDIR CHIP EXTRA... — chip-pinned single eval, idempotent by GCS marker
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

eval_sharded () {  # NAME CK OUTDIR NSH NGATE EXTRA... — NSH-way sharded over chips, merged, n-gated
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

bank_eval () {  # NAME DIR — bank an eval dir that was produced by copy (vsel == final), same contract as eval_one
  local name=$1 O=$2
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && return 0
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "EVAL-OK $name (copied: vsel == final) $(date -u +%H:%M)"
}

census_one () {  # NAME CK OUTDIR CHIP EXTRA — the explosion census (§4.5), idempotent
  local name=$1 CK=$2 O=$3 chip=$4; shift 4
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "CENSUS-SKIP $name"; return 0; }
  mkdir -p "$O"
  pin "$chip" $PY tools/explosion_census.py --ckpt "$CK" --npz "$NPZ" --out "$O" --name "$name" "$@" > "$O/run.log" 2>&1 \
    || { echo "CENSUS-FAILED $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "CENSUS-OK $name $(date -u +%H:%M)"
}

screen () {  # ARM TAG CK CHIP — strat-512 k256 (+ majority) at the arm's headline depth/weights
  local arm=$1 tag=$2 CK=$3 chip=$4
  eval_one "screen_${arm}_${tag}" "$CK" "runs/sxscreen_p${R_TAG}${arm}_${tag}" "$chip" \
    --split test --stratified "$STRAT" --t-total "$(head_t "$arm")" --k-init 256 --vote-unverified $(head_ema "$arm")
}

run_arm () {  # ARM — pretrain, then the battery (vsel select, screens, retfm, fulls vsel+final+alt, scan, census)
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  ARM_PREC=$(arm_prec "$arm")
  run_pretrain "$arm" || return 1
  ensure_local_pretrain "$arm"
  local HE; HE=$(head_ema "$arm"); local AE; AE=$(alt_ema "$arm"); local TH; TH=$(head_t "$arm")
  # val-selected checkpoint on the headline weights' monitor key, selected over BOTH stages of a
  # two-stage arm (the hot-phase peak is a candidate: the pilot's P1 peaked at 20k of its cosine);
  # ties -> the later stage; LOUD fallback when no banked grid carries a monitor row (§4.4)
  local VB="" VBCK="$D/ckpt_latest.pkl" KEY; KEY=$(select_key "$arm")
  local selB="" selA="" vB="" vA=""
  selB=$($PY tools/select_ckpt.py "$D" --key "$KEY" 2>/dev/null) && vB=$(echo "$selB" | awk '{print $2}')
  if two_stage "$arm" && [ ! -f "$D/STOPPED.txt" ] && [ -d "runs/pretrain${R_TAG}_${arm}a" ]; then
    selA=$($PY tools/select_ckpt.py "runs/pretrain${R_TAG}_${arm}a" --key "$KEY" 2>/dev/null) && vA=$(echo "$selA" | awk '{print $2}')
  fi
  if [ -n "$vA" ] && { [ -z "$vB" ] || ${REAL_PY:-python3} -c "import sys; sys.exit(0 if float('$vA') > float('$vB') else 1)"; }; then
    VB=$(echo "$selA" | awk '{print $1}'); [ -f "runs/pretrain${R_TAG}_${arm}a/ckpt_$VB.pkl" ] && VBCK="runs/pretrain${R_TAG}_${arm}a/ckpt_$VB.pkl"
    echo "A:$selA" > "$D/val_best.txt"
  elif [ -n "$vB" ]; then
    VB=$(echo "$selB" | awk '{print $1}'); [ -f "$D/ckpt_$VB.pkl" ] && VBCK="$D/ckpt_$VB.pkl"
    echo "$( two_stage "$arm" && echo "B:" )$selB" > "$D/val_best.txt"
  fi
  if [ "$VBCK" = "$D/ckpt_latest.pkl" ]; then
    echo "VB-FALLBACK-FINAL $arm (select_ckpt returned no banked step; vsel := final, labeled)"
    echo "FALLBACK-FINAL" > "$D/val_best.txt"
  else
    echo "VALBEST $arm $(cat "$D/val_best.txt") -> $VBCK"
  fi
  local ST; ST=$(stopped_step "$arm")
  # screens: fixed steps (skipped beyond a STOPPED arm's final — §4.2) + vb (always)
  if two_stage "$arm" && [ ! -f "$D/STOPPED.txt" ]; then
    local DA=runs/pretrain${R_TAG}_${arm}a
    [ -f "$DA/ckpt_latest.pkl" ] && screen "$arm" sAend "$DA/ckpt_latest.pkl" 0
    for st in 005000 015000; do [ -f "$D/ckpt_$st.pkl" ] && screen "$arm" "sB$st" "$D/ckpt_$st.pkl" 0; done
  else
    for st in $(screen_steps "$arm"); do
      [ -f "$D/ckpt_$st.pkl" ] || continue
      if [ -n "$ST" ] && [ "$((10#$st))" -gt "$ST" ]; then echo "SCREEN-SKIP $arm s$st (beyond STOPPED final $ST)"; continue; fi
      screen "$arm" "s$st" "$D/ckpt_$st.pkl" 0
    done
  fi
  screen "$arm" vb "$VBCK" 1
  # retfm (native arms only: the field cell has no answer register)
  is_field "$arm" || eval_one "retfm_${arm}" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/retfm_t8" 2 \
      --split test --stratified "$STRAT" --t-total 8 --init solution --final-map-only $HE
  # full test at vsel AND final (§4.2), headline weights; the alternate-weights row on vsel (§12.4)
  eval_sharded "full_${arm}_vsel_t${TH}" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/full_vsel_t${TH}" "$NCHIP" 422786 \
      --split test --t-total "$TH" $HE
  if [ "$VBCK" = "$D/ckpt_latest.pkl" ]; then
    mkdir -p "runs/sxeval_p${R_TAG}${arm}/full_final_t${TH}"
    cp -f "runs/sxeval_p${R_TAG}${arm}/full_vsel_t${TH}/"summary_all.json "runs/sxeval_p${R_TAG}${arm}/full_final_t${TH}/" 2>/dev/null
    echo "FULL-FINAL $arm := vsel (identical grid)"
    bank_eval "full_${arm}_final_t${TH}" "runs/sxeval_p${R_TAG}${arm}/full_final_t${TH}"
  else
    eval_sharded "full_${arm}_final_t${TH}" "$D/ckpt_latest.pkl" "runs/sxeval_p${R_TAG}${arm}/full_final_t${TH}" "$NCHIP" 422786 \
        --split test --t-total "$TH" $HE
  fi
  eval_sharded "full_${arm}_vsel_t${TH}_alt" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/full_vsel_t${TH}_alt" "$NCHIP" 422786 \
      --split test --t-total "$TH" $AE
  # the field arms' depth row (EqR's D=64 column) on the vsel grid
  is_field "$arm" && eval_sharded "full_${arm}_vsel_t64" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/full_vsel_t64" "$NCHIP" 422786 \
      --split test --t-total 64 $HE
  # 20k breadth scan k128 at t64 (the protocol-table stats ride the records); X0n has none (descriptive rider)
  [ "$arm" = X0n ] || eval_sharded "scan_${arm}" "$VBCK" "runs/sxscan_p${R_TAG}${arm}" "$NCHIP" "$SUB" \
      --split test --subsample "$SUB" --t-total 64 --k-init 128 $HE
  # explosion census on the vsel AND final grids (§4.5)
  census_one "census_${arm}_vsel" "$VBCK" "runs/sxcensus_p${R_TAG}${arm}_vsel" 3 $HE
  if [ "$VBCK" = "$D/ckpt_latest.pkl" ]; then
    mkdir -p "runs/sxcensus_p${R_TAG}${arm}_final"; cp -f "runs/sxcensus_p${R_TAG}${arm}_vsel/"* "runs/sxcensus_p${R_TAG}${arm}_final/" 2>/dev/null
    bank_eval "census_${arm}_final" "runs/sxcensus_p${R_TAG}${arm}_final"
  else
    census_one "census_${arm}_final" "$D/ckpt_latest.pkl" "runs/sxcensus_p${R_TAG}${arm}_final" 3 $HE
  fi
  # sportC2: the stall-calibration instrument on the vsel grid (R3's rule; standing on every arm)
  calib_one "calib_${arm}_vsel" "$VBCK" "runs/sxcalib_p${R_TAG}${arm}_vsel" 3 $HE
  if [ "$arm" = R3 ]; then
    # sportC2 pre-mortem (2026-09-04): R3 trained with hard feedback on half its steps but is EVALUATED soft like
    # every arm (the registered rows); these two rows read the same vsel grid in its HARD-feedback inference mode
    # (strat-512 cold + the calibration instrument) — descriptive, labeled, no rule reads them.
    eval_one "screen_${arm}_vb_hard" "$VBCK" "runs/sxscreen_p${R_TAG}${arm}_vb_hard" 1 \
      --split test --stratified "$STRAT" --t-total "$TH" --k-init 0 --hard-feedback $HE
    calib_one "calib_${arm}_vsel_hard" "$VBCK" "runs/sxcalib_p${R_TAG}${arm}_vsel_hard" 3 --hard-feedback $HE
  fi
  echo ok | gsutil -q cp - "$GCS/${arm}_ARM_OK"
  echo "ARM-OK $arm $(date -u +%H:%M)"
}

calib_one () {  # NAME CK OUTDIR CHIP EXTRA — stall calibration (tools/stall_calibration.py), idempotent
  local name=$1 CK=$2 O=$3 chip=$4; shift 4
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "CALIB-SKIP $name"; return 0; }
  mkdir -p "$O"
  pin "$chip" $PY tools/stall_calibration.py --ckpt "$CK" --npz "$NPZ" --out "$O" "$@" > "$O/run.log" 2>&1 \
    || { echo "CALIB-FAILED $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "CALIB-OK $name $(date -u +%H:%M)"
}

riders () {  # sportC1's owed riders: B0/B1 20k scans + B0 EMA full on the CORRECT A:20k grids; canvas C3X/D4 EqR-statistic evals
  ARM_PREC=highest
  for arm in B0 B1; do
    local DA=runs/pretrainsportC1_${arm}a CK=runs/pretrainsportC1_${arm}a/ckpt_020000.pkl
    if [ ! -f "$CK" ]; then
      mkdir -p "$DA"
      if gsutil -q cp "$GCS_C1/${arm}a_pretrain.tgz" "/tmp/${arm}a_rider.tgz" 2>/dev/null; then
        tar xzf "/tmp/${arm}a_rider.tgz" "runs/pretrainsportC1_${arm}a/ckpt_020000.pkl" 2>/dev/null || tar xzf "/tmp/${arm}a_rider.tgz" 2>/dev/null
      fi
    fi
    [ -f "$CK" ] || { echo "RIDER-CKPT-MISS ${arm}a"; continue; }
    eval_sharded "rider_${arm}_vselA20k_scan" "$CK" "runs/sxscan_psportC1${arm}_vselA20k" "$NCHIP" "$SUB" \
      --split test --subsample "$SUB" --t-total 64 --k-init 128
    [ "$arm" = B0 ] && eval_sharded "rider_B0_vselA20k_full_ema" "$CK" "runs/sxeval_psportC1B0/full_vselA20k_t64_ema" "$NCHIP" 422786 \
      --split test --t-total 64 --ema
  done
  for src in C3X D4; do
    local CK=runs/pretrainsportBr2b_${src}/ckpt_latest.pkl
    [ -f "$CK" ] || gsutil -q cp "$GCS_R2B/${src}_ckpt.pkl" "$CK" || { echo "RIDER-CKPT-MISS $src"; continue; }
    eval_one "rider_${src}_sel5k" "$CK" "runs/sxrider_${src}_sel5k" 3 \
      --split test --subsample 5000 --t-total 64 --k-init 128 --vote-unverified
  done
}

# ---------- static assignment (plan §12.7) ----------
rc=0
if [ "$NW" -ge 4 ]; then
  case $W in
    0) run_arm W0 || rc=1; run_arm R2 || rc=1;;
    1) run_arm R1 || rc=1; run_arm R3 || rc=1;;
    2) run_arm R4 || rc=1; run_arm X1 || rc=1;;
    3) run_arm X2 || rc=1; riders || rc=1;;
  esac
else
  for arm in W0 R1 R3 X1 R2 R4 X2; do run_arm "$arm" || rc=1; done   # 8-shape priority order (plan §6)
  riders || rc=1
fi

# ---------- completion (any worker; idempotent) ----------
need="W0 R1 R2 R3 R4 X1 X2"
for pass in $(seq 1 "${C1_WAIT_PASSES:-200}"); do
  all=1
  for armx in $need; do gsutil -q stat "$GCS/${armx}_ARM_OK" 2>/dev/null || all=0; done
  if [ "$all" -eq 1 ]; then
    if ! gsutil -q stat "$GCS/sportC2_final.tgz" 2>/dev/null; then
      for f in $(gsutil ls "$GCS/evals/*.tgz" "$GCS"/*_pretrain.tgz 2>/dev/null); do
        b=$(basename "$f"); [ -f "/tmp/pull_$b" ] || { gsutil -q cp "$f" "/tmp/pull_$b" && tar xzf "/tmp/pull_$b" 2>/dev/null; }
      done
      tar czf /tmp/sportC2_final.tgz runs/pretrain${R_TAG}_* runs/sxscreen_p${R_TAG}* runs/sxeval_p${R_TAG}* runs/sxscan_p${R_TAG}* runs/sxcensus_p${R_TAG}* runs/sxcalib_p${R_TAG}* runs/sxrider_* runs/sxscan_psportC1*_vselA20k runs/sxeval_psportC1B0/full_vselA20k_t64_ema 2>/dev/null
      gsutil -q cp /tmp/sportC2_final.tgz "$GCS/sportC2_final.tgz"
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
  sleep "${C1_POLL_SLEEP:-120}"
done
echo "$SENT-INCOMPLETE worker=$W"; exit 1
