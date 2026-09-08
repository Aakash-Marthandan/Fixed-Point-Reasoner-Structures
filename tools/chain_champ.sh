#!/bin/bash
# Ledger: CHAMPION NIGHT chain (2026-09-08; Plan_2026-09-08_Champion_Night.md = the registration; the PI's decisions
# 2026-09-08: a v6e-16 between the US and Mumbai through churns, all seven arms, 30k steps with a contingency for more):
# "the final Sudoku champion runs, designed from the measurements" — seven one-variable arms of the DEC-w384 champion
# recipe (A5's objective set: FPA k1 eps.2 frac.25 + RI sigma 1; exact S9, NO digit aug; the field regime), each with
# the full DEC-class battery, val-selected on the 512-PUZZLE MONITOR with the EARLIEST-TIE rule and the raw monitor as
# the second key (the Night A selection-instrument lesson):
#   C0 C1 C2  the champion recipe at seeds 0 / 1 / 2 (the claim-bearing triple; the seed floor)        30k
#   C3        + the ONLINE position orbit (a fresh group element per row per step; --sudoku-aug 0)     50k
#   C4        the DEC coupling "attn" (set attention over the other eight fields per cell, 4 x dk 32)  30k
#   C5        DEC-w192 (1.34x the TRM-MLP's MACs: the accuracy-per-MAC point)                           30k
#   C6        + the calibrated commit head (tau .9, BCE weight .1; selective hardening)                50k
# THE EXTENSION RULE (R-CH-EXT, registered): after an arm's budget completes, if its selected grid (earliest tie) sits
# inside the last EXT_WINDOW = 4k steps of the budget, the arm is extended ONCE by EXT_STEPS = 20k (a resume from
# ckpt_latest under the constant field lr: an exact continuation), labeled EXTENDED.txt + the GCS marker, then re-selected.
# Battery per arm (every arm is DEC-class = the wide battery of Night A + the frontier's depth rows): screens at the 2k
# grids {10k, 20k[, 30k, 40k]} + vb (strat-512 k256 + the unverified majority); the headline D16 val-selected FULL on all
# 422,786 (+ exact-by-step + halting logits); final and alt (raw weights) D16 on 50k; the D64 row on 100k (+ bits); D128
# on the 20k scan set and D256 on 5k (+ bits); the 5k x k32 t64 scan at --batch 128 under the stall watchdog (RECIPE-DEC
# = BATCHONLY from the frontier run; the ladder --z0-device -> --batch 64 --z0-device on a stall; SCAN-DEADLOCK labeled);
# census vsel + final; stall calibration; C6 also the 20k D16 commit-head row (--record-commit). The sync-policy A/B
# rider (256 strat rows x k8 t64 in ONE batch on A3's frontier grid under --sync-per-step and the default) runs first on
# the worker that owns C3 -> runs/analysis/champ_sync_ab.json + the SYNC-AB marker.
# Mechanics = the Night A chain's (chain_final.sh): static worker map, per-worker PREFLIGHT (60 full-batch steps of every
# arm it owns; a seed-arm failure stops the night, a treatment-arm failure skips it, labeled), ONE-SHOT NaN amputation,
# pt_run's ONE --remat retry on a launch-time HBM OOM (persisted), per-arm banking, idempotent GCS markers, n-gated merges,
# the live 5-min GCS bank + fresh-node restore, canonical grid paths in every eval's provenance (the pass-two lesson).
# 4x4 map (v6e-16; balanced by the measured hours — training 12.5 h per 30k w384 arm incl. its ~6 h battery, 17 h per 50k
# arm, C5 ~ 7 h): w0 C0 C4 · w1 C1 C2 · w2 C6 · w3 [sync rider] C3 C5 (pole w0 ~ 26 h + extensions). 8x4: one arm per
# worker, the rider on w7.
# 1x8: C0 C1 C2 C5 C4 C3 C6 sequential (science per hour). Harness: tools/harness_champ.sh.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src

GCS=${GCS:-gs://qhrrn2-rescue/champ}
R_TAG=${R_TAG:-champ}
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz}   # the 512-puzzle monitor file (train + test byte-identical to seed0)
GCS_SETS=${GCS_SETS:-gs://qhrrn2-rescue/champ/sets}
GCS_FRONTIER=${GCS_FRONTIER:-gs://qhrrn2-rescue/frontier}
SENT=${SENT:-CHAMP}
STEPS_X=${C1_STEPS_X:-30000}            # the seed / operator / width arms
STEPS_LONG=${C1_STEPS_LONG:-50000}      # the orbit and commit-head arms
EXT_STEPS=${C1_EXT_STEPS:-20000}        # the registered extension (once)
EXT_WINDOW=${C1_EXT_WINDOW:-4000}       # a selected grid inside the last EXT_WINDOW steps of the budget -> extend
DEC_W=${CHAMP_DEC_W:-384}
N_FIN_WIDE=${C1_N_FIN_WIDE:-50000}; N_D64_WIDE=${C1_N_D64_WIDE:-100000}; N_SCAN_WIDE=${C1_N_SCAN_WIDE:-5000}; K_SCAN_WIDE=${C1_K_SCAN_WIDE:-32}
DEC_SCAN_BATCH=${DEC_SCAN_BATCH:-128}; STALL_MIN=${STALL_MIN:-25}; STALL_SEC=${STALL_SEC:-$((STALL_MIN * 60))}; DEC_EVAL_TIMEOUT=${DEC_EVAL_TIMEOUT:-9000}
PF_STEPS=${C1_PF_STEPS:-60}
CKPT_EVERY=${C1_CKPT_EVERY:-500}
SEED_ARMS="C0 C1 C2"; OPTIONAL_ARMS="C3 C4 C5 C6"; ALL_ARMS="C0 C1 C2 C3 C4 C5 C6"
MON=${C1_MON:-2000}
SUB=${C1_SUB:-20000}; STRAT=${C1_STRAT:-512}
SYNC_ROWS=${C1_SYNC_ROWS:-256}; SYNC_K=${C1_SYNC_K:-8}
PY=${CHAIN_PY:-python3}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}
ARM_PREC=default   # the field trains/evals in bf16

pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=$ARM_PREC "$@"; }

echo "=== $SENT START worker=$W/$NW chips=$NCHIP $(date -u +%FT%TZ) ==="
mkdir -p "$(dirname "$NPZ")"
[ -f "$NPZ" ] || gsutil -q cp "$GCS_SETS/$(basename "$NPZ")" "$NPZ" || { echo "NPZ-MISSING"; exit 2; }
echo "NPZ-OK"
export JAX_COMPILATION_CACHE_DIR="$PWD/jax_cache"; mkdir -p "$PWD/jax_cache"
gsutil -q cp "$GCS/jax_cache.tgz" /tmp/jc.tgz 2>/dev/null && tar xzf /tmp/jc.tgz 2>/dev/null && echo "COMPILE-CACHE restored"

# ---------- LIVE 5-MIN GCS BANKING + fresh-node RESTORE (tools/live_bank.sh; the standing policy) ----------
LB_ARMS="$ALL_ARMS"
GCS="$GCS" R_TAG="$R_TAG" ARMS="$LB_ARMS" bash tools/live_bank.sh restore
GCS="$GCS" R_TAG="$R_TAG" ARMS="$LB_ARMS" bash tools/live_bank.sh loop & LB_PID=$!
trap 'kill "$LB_PID" 2>/dev/null' EXIT

# ---------- arm flag registry (locked at registration; plan §2) ----------
loop_common () {  # the field's loop + regime (= Night A's loop_common on the 512-monitor file)
  echo "--sudoku-extreme $NPZ --sudoku-layout native9 --equilibrium --sot --act \
        --trm-layers 2 --trm-h-cycles 3 --trm-l-cycles 6 --T 16 --trm-lambda 0.05 --trm-beta 0.01 \
        --sudoku-aug 1000 --loss stablemax --batch 768 --wd 1.0 --warmup 2000 --lr 1e-4 --lr-end 1e-4 --beta2 0.95 --ema 0.999 \
        --fpa-k 0 --beta-flux-nl 0 --monitor-every $MON --grid-every $MON --ckpt-every $CKPT_EVERY --val-every 100000 --dp"
}
champ_common () { echo "$(loop_common) --cell dec --dec-width $DEC_W --fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --trm-ri-sigma 1.0"; }   # = A5's recipe (B0 by rule)
arm_flags () {   # one variable per arm from C0 (a later flag overrides an earlier one)
  case $1 in
    C0)  echo "$(champ_common) --seed 0";;
    C1)  echo "$(champ_common) --seed 1";;
    C2)  echo "$(champ_common) --seed 2";;
    C3)  echo "$(champ_common) --seed 0 --sudoku-aug 0 --sudoku-orbit-online";;
    C4)  echo "$(champ_common) --seed 0 --dec-coupling attn --dec-attn-heads 4 --dec-attn-dk 32";;
    C5)  echo "$(champ_common) --seed 0 --dec-width 192";;
    C6)  echo "$(champ_common) --seed 0 --dec-commit --dec-commit-tau 0.9 --dec-commit-w 0.1";;
    *)   return 1;;
  esac
}
arm_steps ()  { case $1 in C3|C6) echo "$STEPS_LONG";; *) echo "$STEPS_X";; esac; }
head_ema ()   { echo "--ema"; }                  # headline weights = EMA
alt_ema ()    { echo ""; }                       # the alt row = the raw weights
select_key () { echo val_t16_ema; }
second_key () { echo val_t16; }
head_t ()     { echo 16; }
screen_steps () { case $1 in C3|C6) echo "010000 020000 030000 040000";; *) echo "010000 020000";; esac; }
is_optional () { case " $OPTIONAL_ARMS " in *" $1 "*) return 0;; *) return 1;; esac; }
is_commit ()   { [ "$1" = C6 ]; }
worker_arms () {  # the arms THIS worker runs, in order (the sync rider is bound to the C3 worker)
  if [ "$NW" -ge 8 ]; then case $W in 0) echo "C0";; 1) echo "C1";; 2) echo "C2";; 3) echo "C3";; 4) echo "C4";; 5) echo "C5";; 6) echo "C6";; 7) echo "";; esac
  elif [ "$NW" -ge 4 ]; then case $W in 0) echo "C0 C4";; 1) echo "C1 C2";; 2) echo "C6";; 3) echo "C3 C5";; esac   # balanced by the measured hours (pole w0 ~ 26 h)
  else echo "${CHAMP_ARMS_1X8:-C0 C1 C2 C5 C4 C3 C6}"; fi
}
rider_worker () { if [ "$NW" -ge 8 ]; then [ "$W" = 7 ]; elif [ "$NW" -ge 4 ]; then [ "$W" = 3 ]; else [ "$W" = 0 ]; fi; }

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

log_has_step () { grep -qE '^step +[0-9]+ ' "$1"; }   # THIS launch logged a step (the log is rewritten per launch)
pt_run () {  # LOG ARM DIR pt-args... — ONE --remat retry on a LAUNCH-TIME HBM exhaustion (persisted in RETRY_REMAT.txt); see chain_final.sh
  local log=$1 arm=$2 dir=$3; shift 3
  if [ -f "$dir/RETRY_REMAT.txt" ] && ! printf '%s\n' "$@" | grep -qx -- '--remat'; then
    echo "REMAT-PERSISTED $arm (this arm needed --remat before; launching with it)"; set -- "$@" --remat
  fi
  pt "$@" > "$log" 2>&1; local rc=$?
  if [ $rc -ne 0 ] && ! log_has_step "$log" && ! printf '%s\n' "$@" | grep -qx -- '--remat' \
     && grep -qiE 'RESOURCE_EXHAUSTED|out of memory' "$log"; then
    echo "PRETRAIN-OOM-RETRY-REMAT $arm (rc=$rc before any logged step = HBM exhaustion, not a NaN death; one retry with --remat, numerics-equivalent, labeled)"
    mkdir -p "$dir"; echo "OOM at launch (rc=$rc) -> retried once with --remat $(date -u +%FT%TZ)" >> "$dir/RETRY_REMAT.txt"
    pt "$@" --remat >> "$log" 2>&1; rc=$?
  fi
  return $rc
}

bank_dir () { tar czf "/tmp/$3.tgz" "$2" 2>/dev/null && gsutil -q cp "/tmp/$3.tgz" "$GCS/$3.tgz"; }   # ARM DIR NAME

preflight () {  # 60 full-batch steps of every arm this worker will run: compile + pace at the source, before any arm
  gsutil -q stat "$GCS/PREFLIGHT_OK_w${W}_nw${NW}" 2>/dev/null && { echo "PREFLIGHT-SKIP worker=$W nw=$NW"; return 0; }
  local arm D FL rc ips
  for arm in $(worker_arms); do
    gsutil -q stat "$GCS/${arm}_ARM_OK" 2>/dev/null && continue
    D=runs/preflight${R_TAG}_$arm; FL="$(arm_flags "$arm")" || { echo "BAD-ARM $arm"; return 1; }
    pt_run "$D.log" "pf-$arm" "$D" --out "$D" $FL --steps "$PF_STEPS" --warmup 10 --ckpt-every "$PF_STEPS" --grid-every "$PF_STEPS" --monitor-every 0 --log-every 10; rc=$?
    if [ $rc -ne 0 ] || ! nan_check "$D"; then
      if is_optional "$arm"; then
        echo "PREFLIGHT-FAILED $arm (rc=$rc) -> SKIPPED (treatment arm, labeled)"; echo "preflight failed rc=$rc $(date -u +%FT%TZ)" | gsutil -q cp - "$GCS/${arm}_SKIPPED"
      else
        echo "PREFLIGHT-FAILED $arm (rc=$rc) -> the night stops here (seed arm; stop-and-report)"; return 1
      fi
      continue
    fi
    ips=$(${REAL_PY:-python3} -c "import json; r=[json.loads(l) for l in open('$D/metrics.jsonl') if '\"loss\"' in l]; print(r[-1].get('steps_per_sec','nan') if r else 'nan')" 2>/dev/null)
    echo "PREFLIGHT-OK $arm ${ips} it/s $(date -u +%H:%M)"
  done
  echo ok | gsutil -q cp - "$GCS/PREFLIGHT_OK_w${W}_nw${NW}"
}

select_best () {  # DIR -> "NNNNNN val step" on the registered selection instrument (earliest tie, the raw monitor second)
  $PY tools/select_ckpt.py "$1" --key "$(select_key)" --tie earliest --second-key "$(second_key)" 2>/dev/null
}

run_pretrain () {  # ARM — ONE-SHOT NaN amputation; the registered EXTENSION rule after the budget completes
  local arm=$1 D=runs/pretrain${R_TAG}_$1 rc budget sel best
  gsutil -q stat "$GCS/${arm}_PRETRAIN_OK" 2>/dev/null && { echo "PRETRAIN-SKIP $arm"; return 0; }
  local FL; FL="$(arm_flags "$arm")" || { echo "BAD-ARM $arm"; return 1; }
  budget=$(arm_steps "$arm")
  if [ -f "$D/EXTENDED.txt" ] || gsutil -q stat "$GCS/${arm}_EXTENDED" 2>/dev/null; then
    mkdir -p "$D"; [ -f "$D/EXTENDED.txt" ] || gsutil -q cp "$GCS/${arm}_EXTENDED" "$D/EXTENDED.txt"
    budget=$((budget + EXT_STEPS)); echo "PRETRAIN-EXTENDED-BUDGET $arm $budget (a relaunch keeps the registered extension)"
  fi
  echo "PRETRAIN-START $arm $(date -u +%H:%M) prec=$ARM_PREC steps=$budget"
  pt_run "$D.log" "$arm" "$D" --out "$D" $FL --steps "$budget"; rc=$?
  if [ $rc -ne 0 ] || ! nan_check "$D"; then
    echo "PRETRAIN-NAN $arm (rc=$rc) -> amputate"; amputate "$D" || return 1
  fi
  # R-CH-EXT: the selected grid inside the last EXT_WINDOW steps of the budget -> ONE extension by EXT_STEPS
  if [ ! -f "$D/STOPPED.txt" ] && [ ! -f "$D/EXTENDED.txt" ]; then
    sel=$(select_best "$D") && best=$(echo "$sel" | awk '{print $3}')
    if [ -n "${best:-}" ] && [ "$best" -ge $((budget - EXT_WINDOW)) ]; then
      echo "EXTENDED from $budget to $((budget + EXT_STEPS)) (peak at $best) $(date -u +%FT%TZ)" > "$D/EXTENDED.txt"
      gsutil -q cp "$D/EXTENDED.txt" "$GCS/${arm}_EXTENDED"
      echo "PRETRAIN-EXTEND $arm: selected grid $best inside the last $EXT_WINDOW of $budget -> +$EXT_STEPS (R-CH-EXT, once)"
      pt_run "$D.log" "$arm" "$D" --out "$D" $FL --steps $((budget + EXT_STEPS)); rc=$?
      if [ $rc -ne 0 ] || ! nan_check "$D"; then
        echo "PRETRAIN-NAN $arm in the extension (rc=$rc) -> amputate"; amputate "$D" || return 1
      fi
    else
      echo "PRETRAIN-NO-EXTEND $arm (selected grid ${best:-none} vs budget $budget window $EXT_WINDOW)"
    fi
  fi
  bank_dir "$arm" "$D" "${arm}_pretrain"
  gsutil -q cp "$D/ckpt_latest.pkl" "$GCS/${arm}_ckpt.pkl"
  [ -f "$D/STOPPED.txt" ] && gsutil -q cp "$D/STOPPED.txt" "$GCS/${arm}_STOPPED.txt"
  echo ok | gsutil -q cp - "$GCS/${arm}_PRETRAIN_OK"
  echo "PRETRAIN-OK $arm $(date -u +%H:%M)"
}

ensure_local_pretrain () {  # ARM — after a node change the banked grids + metrics must be local before selection
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  if [ ! -f "$D/metrics.jsonl" ] || [ -z "$(ls "$D"/ckpt_0*.pkl 2>/dev/null)" ]; then
    if gsutil -q cp "$GCS/${arm}_pretrain.tgz" "/tmp/${arm}_pre_pull.tgz" 2>/dev/null; then
      tar xzf "/tmp/${arm}_pre_pull.tgz" 2>/dev/null && echo "PRETRAIN-RESTORE $arm (grids + metrics re-pulled from GCS)"
    fi
  fi
  mkdir -p "$D"
  [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl"
  [ -f "$D/STOPPED.txt" ] || gsutil -q cp "$GCS/${arm}_STOPPED.txt" "$D/STOPPED.txt" 2>/dev/null || true
  [ -f "$D/EXTENDED.txt" ] || gsutil -q cp "$GCS/${arm}_EXTENDED" "$D/EXTENDED.txt" 2>/dev/null || true
}

stopped_step () { grep -oE 'step [0-9]+' "runs/pretrain${R_TAG}_$1/STOPPED.txt" 2>/dev/null | awk '{print $2}' | head -1; }

# ---------- evals (idempotent by GCS marker; sharded over the worker's chips; n-gated) ----------
finish_eval () {  # NAME OUTDIR NGATE — merge, n-gate, bank, mark
  local name=$1 O=$2 NGATE=$3
  JAX_PLATFORMS=cpu $PY tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  ${REAL_PY:-python3} -c "import json,sys; s=json.load(open('$O/summary_all.json')); sys.exit(0 if s['n']==$NGATE else 1)" \
    || { echo "EVAL-N-BAD $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "EVAL-OK $name $(date -u +%H:%M)"
}
eval_one () {  # NAME CK OUTDIR CHIP EXTRA... — chip-pinned single eval, idempotent by GCS marker
  local name=$1 CK=$2 O=$3 chip=$4; shift 4
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "EVAL-SKIP $name"; return 0; }
  mkdir -p "$O"
  pin "$chip" ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --out "$O" "$@" > "$O/run.log" 2>&1 \
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
    pin $((i % NCHIP)) ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --out "$O" \
        --shard "$i/$NSH" --bank-every 300 "$@" > "$O/shard_$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { echo "EVAL-SHARD-FAILED $name"; return 1; }
  finish_eval "$name" "$O" "$NGATE"
}
newest_mtime () {  # DIR -> the newest mtime (s) among the shards' logs and banked partials (portable: no GNU find on the Mac harness)
  ${REAL_PY:-python3} - "$1" <<'PYEOF'
import glob, os, sys
fs = glob.glob(os.path.join(sys.argv[1], "shard_*.log")) + glob.glob(os.path.join(sys.argv[1], "partial_*.npz"))
print(int(max(os.path.getmtime(f) for f in fs)) if fs else "")
PYEOF
}
eval_watched () {  # NAME CK OUTDIR NSH NGATE EXTRA... — eval_sharded + the STALL WATCHDOG (the DEC multi-draw scan); rc 2 = stalled
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
bank_eval () {  # NAME DIR — bank an eval dir that was produced by copy (vsel == final), same contract as eval_one
  local name=$1 O=$2
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && return 0
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "EVAL-OK $name (copied: vsel == final) $(date -u +%H:%M)"
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
calib_one () {  # NAME CK OUTDIR CHIP EXTRA — stall calibration (tools/stall_calibration.py), idempotent
  local name=$1 CK=$2 O=$3 chip=$4; shift 4
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "CALIB-SKIP $name"; return 0; }
  mkdir -p "$O"
  pin "$chip" ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/stall_calibration.py --ckpt "$CK" --npz "$NPZ" --out "$O" "$@" > "$O/run.log" 2>&1 \
    || { echo "CALIB-FAILED $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "CALIB-OK $name $(date -u +%H:%M)"
}
screen () {  # ARM TAG CK — strat-512 k256 (+ majority) at the headline depth/weights, sharded over the worker's chips
  local arm=$1 tag=$2 CK=$3
  eval_sharded "screen_${arm}_${tag}" "$CK" "runs/sxscreen_p${R_TAG}${arm}_${tag}" "$NCHIP" "$STRAT" \
    --split test --stratified "$STRAT" --t-total "$(head_t "$arm")" --k-init 256 --vote-unverified $(head_ema "$arm")
}

DEC_RECIPE=""   # the flags that completed: "" (batch only) | "--z0-device" | "--batch 64 --z0-device" (the frontier run: BATCHONLY)
dec_recipe () {
  [ -n "$DEC_RECIPE" ] && return 0
  local r; r=$(gsutil -q cp "$GCS/RECIPE-DEC" - 2>/dev/null) && DEC_RECIPE=$r && echo "RECIPE-DEC restored: [$DEC_RECIPE]"
}
scan_dec () {  # ARM CK — the 5k x k32 t64 scan at --batch 128 under the watchdog; the variant ladder on a stall; SCAN-DEADLOCK labeled
  local arm=$1 CK=$2 name="scan_$arm" O="runs/sxscan_p${R_TAG}$arm" rc v
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "EVAL-SKIP $name"; return 0; }
  gsutil -q stat "$GCS/${arm}_SCAN_DEADLOCK" 2>/dev/null && { echo "SCAN-DEADLOCK-SKIP $arm"; return 0; }
  dec_recipe
  export EVAL_TIMEOUT=$DEC_EVAL_TIMEOUT
  local variants; if [ -n "$DEC_RECIPE" ] || gsutil -q stat "$GCS/RECIPE-DEC" 2>/dev/null; then variants="$DEC_RECIPE"; else variants="BATCHONLY --z0-device --batch_64_--z0-device"; fi
  for v in $variants; do
    v=${v//_/ }; [ "$v" = "BATCHONLY" ] && v=""
    rm -rf "$O"; echo "DEC-SCAN-TRY $arm batch=$DEC_SCAN_BATCH flags=[$v] $(date -u +%H:%M)"
    # shellcheck disable=SC2086
    eval_watched "$name" "$CK" "$O" "$NCHIP" "$N_SCAN_WIDE" --split test --subsample "$N_SCAN_WIDE" --t-total 64 --k-init "$K_SCAN_WIDE" $(head_ema "$arm") --batch "$DEC_SCAN_BATCH" $v; rc=$?
    if [ $rc -eq 0 ]; then
      [ -n "$DEC_RECIPE" ] || { DEC_RECIPE="${v:-BATCHONLY}"; printf '%s' "$DEC_RECIPE" | gsutil -q cp - "$GCS/RECIPE-DEC"; echo "RECIPE-DEC $arm: [$DEC_RECIPE]"; }
      unset EVAL_TIMEOUT; return 0
    fi
    echo "DEC-SCAN-VARIANT-FAILED $arm rc=$rc flags=[$v]"
    [ $rc -eq 2 ] || [ -z "$DEC_RECIPE" ] || break
  done
  unset EVAL_TIMEOUT
  echo "SCAN-DEADLOCK $arm (every variant failed; labeled)"; echo "$(date -u +%FT%TZ) variants=[$variants]" | gsutil -q cp - "$GCS/${arm}_SCAN_DEADLOCK"; return 1
}

sync_ab () {  # the evaluator's sync-policy A/B pace rider: ONE batch of SYNC_ROWS strat rows x k SYNC_K t64 on A3's frontier grid, both policies, one chip
  gsutil -q stat "$GCS/SYNC-AB" 2>/dev/null && { echo "SYNC-AB-SKIP"; return 0; }
  mkdir -p runs/champ_ckpts runs/analysis
  [ -f runs/champ_ckpts/A3.pkl ] || gsutil -q cp "$GCS_FRONTIER/ckpts/A3.pkl" runs/champ_ckpts/A3.pkl || { echo "SYNC-AB-NOCKPT (A3's frontier grid absent; rider skipped, labeled)"; echo "no ckpt" | gsutil -q cp - "$GCS/SYNC-AB"; return 0; }
  local pol flags O
  for pol in default perstep; do
    flags=""; [ "$pol" = perstep ] && flags="--sync-per-step"
    O="runs/sxsync_p${R_TAG}_$pol"
    # shellcheck disable=SC2086
    eval_one "sync_$pol" runs/champ_ckpts/A3.pkl "$O" 0 --split test --stratified "$SYNC_ROWS" --t-total 64 --k-init "$SYNC_K" --ema --batch "$SYNC_ROWS" $flags || echo "SYNC-AB-FAILED $pol"
  done
  ${REAL_PY:-python3} - "$SYNC_ROWS" "$SYNC_K" <<'PYEOF'
import json, os, sys
from pathlib import Path
out = {"rows": int(sys.argv[1]), "k": int(sys.argv[2]), "zone": os.environ.get("SELF_ZONE", "")}
for pol, key in (("default", "wall_default_s"), ("perstep", "wall_per_step_s")):
    p = Path(f"runs/sxsync_pchamp_{pol}/summary_all.json")
    out[key] = json.loads(p.read_text()).get("wall_s") if p.exists() else None
Path("runs/analysis").mkdir(parents=True, exist_ok=True)
Path("runs/analysis/champ_sync_ab.json").write_text(json.dumps(out, indent=1))
print("SYNC-AB", json.dumps(out))
PYEOF
  gsutil -q cp runs/analysis/champ_sync_ab.json "$GCS/champ_sync_ab.json"; echo ok | gsutil -q cp - "$GCS/SYNC-AB"
  echo "SYNC-AB-OK $(date -u +%H:%M)"
}

run_arm () {  # ARM — pretrain (+ the extension rule), then the DEC-class battery
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  gsutil -q stat "$GCS/${arm}_SKIPPED" 2>/dev/null && { echo "ARM-SKIPPED $arm (labeled)"; return 0; }
  gsutil -q stat "$GCS/${arm}_ARM_OK" 2>/dev/null && { echo "ARM-SKIP $arm (done)"; return 0; }
  run_pretrain "$arm" || return 1
  ensure_local_pretrain "$arm"
  local HE; HE=$(head_ema "$arm"); local AE; AE=$(alt_ema "$arm"); local TH; TH=$(head_t "$arm")
  # the val-selected grid on the 512-puzzle monitor: earliest tie, the raw monitor as the second key (the CANONICAL grid
  # path is what every eval records — the pass-two provenance lesson); LOUD fallback to the final when no grid has a row
  local VB="" VBCK="$D/ckpt_latest.pkl" sel v
  sel=$(select_best "$D") && v=$(echo "$sel" | awk '{print $2}')
  if [ -n "${v:-}" ]; then VB=$(echo "$sel" | awk '{print $1}'); [ -f "$D/ckpt_$VB.pkl" ] && VBCK="$D/ckpt_$VB.pkl"; echo "$sel" > "$D/val_best.txt"; fi
  if [ "$VBCK" = "$D/ckpt_latest.pkl" ]; then
    echo "VB-FALLBACK-FINAL $arm (select_ckpt returned no banked step; vsel := final, labeled)"; echo "FALLBACK-FINAL" > "$D/val_best.txt"
  else
    echo "VALBEST $arm $(cat "$D/val_best.txt") -> $VBCK"
  fi
  local ST; ST=$(stopped_step "$arm")
  for st in $(screen_steps "$arm"); do
    [ -f "$D/ckpt_$st.pkl" ] || continue
    if [ -n "$ST" ] && [ "$((10#$st))" -gt "$ST" ]; then echo "SCREEN-SKIP $arm s$st (beyond STOPPED final $ST)"; continue; fi
    screen "$arm" "s$st" "$D/ckpt_$st.pkl"
  done
  screen "$arm" vb "$VBCK"
  # the headline: the D16 val-selected FULL on all 422,786 (+ the exact bit per step + the halting logits)
  eval_sharded "full_${arm}_vsel_t${TH}" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/full_vsel_t${TH}" "$NCHIP" 422786 \
      --split test --t-total "$TH" $HE --record-by-step --record-q
  if [ "$VBCK" = "$D/ckpt_latest.pkl" ]; then
    mkdir -p "runs/sxeval_p${R_TAG}${arm}/full_final_t${TH}"
    cp -f "runs/sxeval_p${R_TAG}${arm}/full_vsel_t${TH}/"summary_all.json "runs/sxeval_p${R_TAG}${arm}/full_final_t${TH}/" 2>/dev/null
    echo "FULL-FINAL $arm := vsel (identical grid)"
    bank_eval "full_${arm}_final_t${TH}" "runs/sxeval_p${R_TAG}${arm}/full_final_t${TH}"
  else
    eval_sharded "full_${arm}_final_t${TH}" "$D/ckpt_latest.pkl" "runs/sxeval_p${R_TAG}${arm}/full_final_t${TH}" "$NCHIP" "$N_FIN_WIDE" \
        --split test --t-total "$TH" --subsample "$N_FIN_WIDE" $HE
  fi
  eval_sharded "full_${arm}_vsel_t${TH}_alt" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/full_vsel_t${TH}_alt" "$NCHIP" "$N_FIN_WIDE" \
      --split test --t-total "$TH" --subsample "$N_FIN_WIDE" $AE
  # the depth ladder: D64 on 100k, D128 on the 20k scan set, D256 on 5k (the exact bit per step on each)
  eval_sharded "full_${arm}_vsel_t64" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/full_vsel_t64" "$NCHIP" "$N_D64_WIDE" \
      --split test --t-total 64 --subsample "$N_D64_WIDE" $HE --record-by-step
  eval_sharded "d128_${arm}" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/sub20k_t128" "$NCHIP" "$SUB" \
      --split test --subsample "$SUB" --t-total 128 $HE --record-by-step
  eval_sharded "d256_${arm}" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/sub5k_t256" "$NCHIP" 5000 \
      --split test --subsample 5000 --t-total 256 $HE --record-by-step
  # C6: the commit head's rows on the 20k at D16 (the head's probability per cell per step + per-cell correctness bits)
  if is_commit "$arm"; then
    eval_sharded "commit_${arm}" "$VBCK" "runs/sxeval_p${R_TAG}${arm}/sub20k_t${TH}_commit" "$NCHIP" "$SUB" \
        --split test --subsample "$SUB" --t-total "$TH" $HE --record-commit --record-by-step
  fi
  # the breadth scan (the selector column): 5k x k32 t64 at batch 128 under the watchdog
  scan_dec "$arm" "$VBCK" || echo "SCAN-ABSENT $arm (labeled)"
  census_one "census_${arm}_vsel" "$VBCK" "runs/sxcensus_p${R_TAG}${arm}_vsel" 3 $HE
  if [ "$VBCK" = "$D/ckpt_latest.pkl" ]; then
    mkdir -p "runs/sxcensus_p${R_TAG}${arm}_final"; cp -f "runs/sxcensus_p${R_TAG}${arm}_vsel/"* "runs/sxcensus_p${R_TAG}${arm}_final/" 2>/dev/null
    bank_eval "census_${arm}_final" "runs/sxcensus_p${R_TAG}${arm}_final"
  else
    census_one "census_${arm}_final" "$D/ckpt_latest.pkl" "runs/sxcensus_p${R_TAG}${arm}_final" 3 $HE
  fi
  calib_one "calib_${arm}_vsel" "$VBCK" "runs/sxcalib_p${R_TAG}${arm}_vsel" 3 $HE
  echo ok | gsutil -q cp - "$GCS/${arm}_ARM_OK"
  echo "ARM-OK $arm $(date -u +%H:%M)"
}

# ---------- static assignment ----------
rc=0
rider_worker && sync_ab
preflight || { echo "$SENT-PREFLIGHT-ABORT worker=$W $(date -u +%FT%TZ)"; exit 1; }
for arm in $(worker_arms); do run_arm "$arm" || rc=1; done

# ---------- completion (any worker; idempotent) ----------
need="$ALL_ARMS"
echo "COMPLETION-SET nw=$NW need=[$need]"
for pass in $(seq 1 "${C1_WAIT_PASSES:-200}"); do
  all=1
  for armx in $need; do gsutil -q stat "$GCS/${armx}_ARM_OK" 2>/dev/null || gsutil -q stat "$GCS/${armx}_SKIPPED" 2>/dev/null || all=0; done
  if [ "$all" -eq 1 ]; then
    if ! gsutil -q stat "$GCS/champ_final.tgz" 2>/dev/null; then
      for f in $(gsutil ls "$GCS/evals/*.tgz" "$GCS"/*_pretrain.tgz 2>/dev/null); do
        b=$(basename "$f"); [ -f "/tmp/pull_$b" ] || { gsutil -q cp "$f" "/tmp/pull_$b" && tar xzf "/tmp/pull_$b" 2>/dev/null; }
      done
      gsutil -q cp "$GCS/champ_sync_ab.json" runs/analysis/champ_sync_ab.json 2>/dev/null || true
      tar czf /tmp/champ_final.tgz runs/pretrain${R_TAG}_* runs/preflight${R_TAG}_*.log runs/sxscreen_p${R_TAG}* runs/sxeval_p${R_TAG}* runs/sxscan_p${R_TAG}* runs/sxcensus_p${R_TAG}* runs/sxcalib_p${R_TAG}* runs/sxsync_p${R_TAG}* runs/analysis/champ_sync_ab.json 2>/dev/null
      gsutil -q cp /tmp/champ_final.tgz "$GCS/champ_final.tgz"
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
