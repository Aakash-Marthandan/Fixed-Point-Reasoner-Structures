#!/bin/bash
# Ledger: DEC-ARC NIGHT chain (Plan_2026-09-10_DEC-ARC_Build §2; built 2026-09-10; the registration entry locks the
# arm registry below). One spot v6e-8 = 2 workers x 4 chips: w0 runs D0 then D2, w1 runs D1 then N0 (a 1-worker shape
# runs all four in order). Per arm: preflight (60 full-batch steps: compile + pace + HBM at the source), the pretrain
# (the DEC-ARC arms on the field loop with the ARC corpus; N0 = the native rg cell at d96, the d96 rung's A5-class
# arm), the registered EXTENSION rule on the DEC arms (a selected grid inside the last EXT_WINDOW steps of the budget
# -> +EXT_STEPS ONCE, idempotent via EXTENDED.txt + a GCS marker), the val-selected grid (the 2k ARC monitor, EMA key,
# earliest tie, the raw monitor second; N0 on its val rows), then the battery through tools/eval_decarc.py sharded over
# the worker's chips (val-hard + dev-30 with k = 32 latent restarts, the FPA-start ladder, the 8-view dihedral vote and
# the colour-flip row; rg-96 / rt-48 with k = 8; the ARC-AGI-1 evaluation split with k = 8 and ARC1_VIEWS views; the
# FINAL grid on val-hard = the memorization row) — N0 through tools/arc_suite.py on the same sets. Every eval is
# idempotent by a GCS marker, n-gated and banked. The RIDER (RIDER=1; the PI's reserve gate): the champion C2's
# full-set D64 Sudoku row, claimed once by the first worker to finish its arms. Live 5-min bank; the ONE --remat retry
# on a launch-time HBM OOM; one-shot NaN amputation; completion = every arm ARM_OK or SKIPPED -> decarc_final.tgz ->
# CHAIN-DECARC-COMPLETE. Harness: tools/harness_decarc.sh.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src

GCS=${GCS:-gs://qhrrn2-rescue/decarc}
R_TAG=${R_TAG:-decarc}
W=${CHAIN_WORKER:-0}; NW=${CHAIN_WORKERS:-1}
GCS_SETS=${GCS_SETS:-gs://qhrrn2-rescue/decarc/sets}
GCS_CHAMP_SETS=${GCS_CHAMP_SETS:-gs://qhrrn2-rescue/champ/sets}
SENT=${SENT:-DECARC}
STEPS_DEC=${DA_STEPS_DEC:-30000}; STEPS_NAT=${DA_STEPS_NAT:-40000}
EXT_STEPS=${DA_EXT_STEPS:-10000}; EXT_WINDOW=${DA_EXT_WINDOW:-4000}
DEC_W=${DA_DEC_W:-160}; HEADS=${DA_HEADS:-4}; BATCH=${DA_BATCH:-64}; NVAL=${DA_NVAL:-96}; W_VOID=${DA_W_VOID:-0.5}
TABLE_LR=${DA_TABLE_LR:-1e-2}; TABLE_WD=${DA_TABLE_WD:-0.1}   # the task-code table's own optimizer (the audit, plan §10 item 2)
MON=${DA_MON:-2000}; CKPT_EVERY=${DA_CKPT_EVERY:-500}; PF_STEPS=${DA_PF_STEPS:-60}
K_VH=${DA_K_VH:-32}; K_GATE=${DA_K_GATE:-8}; VIEWS_VH=${DA_VIEWS_VH:-8}; ARC1_VIEWS=${DA_ARC1_VIEWS:-8}; ARC1_K=${DA_ARC1_K:-8}
FIT_STEPS=${DA_FIT_STEPS:-600}; T_TOTAL=${DA_T_TOTAL:-16}
RIDER=${RIDER:-0}
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz}
SEED_ARMS="D0 D1"; OPTIONAL_ARMS="D2 N0"; ALL_ARMS="D0 D1 D2 N0"
PY=${CHAIN_PY:-python3}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}
ARM_PREC=default
EVAL_TIMEOUT=${DA_EVAL_TIMEOUT:-14400}
pt () { JAX_DEFAULT_MATMUL_PRECISION=$ARM_PREC $PY tools/pretrain.py "$@"; }
pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=$ARM_PREC "$@"; }

echo "=== $SENT START worker=$W/$NW chips=$NCHIP $(date -u +%FT%TZ) ==="

# ---------- the ARC data on the node (git-ignored; banked as one tarball at launch) ----------
ensure_data () {
  if [ ! -d data/re_arc ] || [ ! -d data/ARC-AGI/data/training ] || [ ! -d data/ConceptARC/corpus ]; then
    gsutil -q cp "$GCS_SETS/arc_data.tgz" /tmp/arc_data.tgz && tar xzf /tmp/arc_data.tgz || { echo "DATA-MISSING (arc_data.tgz not banked)"; return 1; }
  fi
  for d in data/ARC-AGI/data/training data/ARC-AGI/data/evaluation data/ConceptARC/corpus data/re_arc data/re_gate48 data/re_gateb48 data/re_train48; do
    [ -d "$d" ] || { echo "DATA-MISSING $d"; return 1; }
  done
  [ "$(ls data/re_gate48/*.json 2>/dev/null | wc -l | tr -d ' ')" = 48 ] || { echo "DATA-MISSING re_gate48 != 48"; return 1; }
  echo "DATA-OK"
}
ensure_data || { echo "$SENT-DATA-ABORT worker=$W $(date -u +%FT%TZ)"; exit 2; }
export JAX_COMPILATION_CACHE_DIR="$PWD/jax_cache"; mkdir -p "$PWD/jax_cache"
gsutil -q cp "$GCS/jax_cache.tgz" /tmp/jc.tgz 2>/dev/null && tar xzf /tmp/jc.tgz 2>/dev/null && echo "COMPILE-CACHE restored"

# ---------- LIVE 5-MIN GCS BANKING + fresh-node RESTORE (tools/live_bank.sh; the standing policy) ----------
GCS="$GCS" R_TAG="$R_TAG" ARMS="$ALL_ARMS" bash tools/live_bank.sh restore
GCS="$GCS" R_TAG="$R_TAG" ARMS="$ALL_ARMS" bash tools/live_bank.sh loop & LB_PID=$!
trap 'kill "$LB_PID" 2>/dev/null' EXIT

# ---------- arm flag registry (locked at registration; plan §2) ----------
corpus_common () { echo "--equilibrium --rearc --conceptarc --orbit 4 --n-val $NVAL --dp --seed 0"; }
decarc_common () {   # the DEC on the ten-field colour state under the champion loop + regime (plan §1)
  echo "$(corpus_common) --sot --act --cell decarc --dec-width $DEC_W --decarc-heads $HEADS \
        --trm-layers 2 --trm-h-cycles 3 --trm-l-cycles 6 --T 16 --trm-lambda 0.05 --trm-beta 0.01 \
        --loss stablemax --batch $BATCH --wd 1.0 --warmup 2000 --lr 1e-4 --lr-end 1e-4 --beta2 0.95 --ema 0.999 \
        --w-void $W_VOID --table-lr $TABLE_LR --table-wd $TABLE_WD --beta-flux-nl 0 --remat --monitor-every $MON --grid-every $MON --ckpt-every $CKPT_EVERY --val-every 100000"
}
native_common () {   # the d96 rung's A5-class arm (chain_r0.sh: PRICED + NI, B64/T6, the knee; the back-port: 2k val rows + grids)
  echo "$(corpus_common) --d 96 --T 6 --anchor-p 0.3 --beta-flux 3e-5 --beta-flux-nl 1e-5 --ni-sigma 0.01 \
        --monitor-every 0 --val-every $MON --grid-every $MON --ckpt-every $CKPT_EVERY"
}
arm_flags () {   # one variable per DEC arm from D0 (a later flag overrides an earlier one)
  case $1 in
    D0)  echo "$(decarc_common) --trm-ri-sigma 1.0 --fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --seed 0";;
    D1)  echo "$(decarc_common) --trm-ri-sigma 1.0 --fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --seed 1";;
    D2)  echo "$(decarc_common) --trm-ri-sigma 0 --fpa-k 0 --seed 0";;                       # the plain twin
    N0)  echo "$(native_common)";;
    *)   return 1;;
  esac
}
arm_steps ()  { case $1 in N0) echo "$STEPS_NAT";; *) echo "$STEPS_DEC";; esac; }
is_dec ()     { [ "$1" != N0 ]; }
is_optional () { case " $OPTIONAL_ARMS " in *" $1 "*) return 0;; *) return 1;; esac; }
worker_arms () {   # the arms THIS worker runs, in order (two rounds on two hosts; the seed pair first)
  if [ "$NW" -ge 4 ]; then case $W in 0) echo "D0";; 1) echo "D1";; 2) echo "D2";; 3) echo "N0";; *) echo "";; esac
  elif [ "$NW" -ge 2 ]; then case $W in 0) echo "D0 D2";; 1) echo "D1 N0";; *) echo "";; esac
  else echo "${DA_ARMS_1X:-D0 D1 D2 N0}"; fi
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

amputate () {  # DIR — one-shot rule: final = last banked FINITE grid; post-death grids REMOVED (never evaluated)
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
        elif hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.floating):
            if not np.all(np.isfinite(np.asarray(x))): return False
    return True
grids = sorted(d.glob("ckpt_0*.pkl"))
good = None
for g in grids:
    try:
        ck = pickle.load(open(g, "rb"))
        if finite(ck["state"]): good = (g, int(ck["step"]))
    except Exception: pass
if good is None: print("AMPUTATE-NO-FINITE-GRID"); sys.exit(1)
for g in grids:
    if int(g.stem.split("_")[1]) > good[1]: g.unlink()
shutil.copy(good[0], d / "ckpt_latest.pkl")
(d / "STOPPED.txt").write_text(f"non-finite loss; final := {good[0].name} step {good[1]} (one-shot amputation)\n")
print(f"AMPUTATED final := {good[0].name} step {good[1]}")
PYEOF
}

log_has_step () { grep -qE '^step +[0-9]+ ' "$1"; }
pt_run () {  # LOG ARM DIR pt-args... — ONE --remat retry on a LAUNCH-TIME HBM exhaustion (persisted in RETRY_REMAT.txt)
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

preflight () {  # PF_STEPS full-batch steps of every arm this worker will run: compile + pace + HBM at the source
  gsutil -q stat "$GCS/PREFLIGHT_OK_w${W}_nw${NW}" 2>/dev/null && { echo "PREFLIGHT-SKIP worker=$W nw=$NW"; return 0; }
  local arm D FL rc ips
  for arm in $(worker_arms); do
    gsutil -q stat "$GCS/${arm}_ARM_OK" 2>/dev/null && continue
    D=runs/preflight${R_TAG}_$arm; FL="$(arm_flags "$arm")" || { echo "BAD-ARM $arm"; return 1; }
    pt_run "$D.log" "pf-$arm" "$D" --out "$D" $FL --steps "$PF_STEPS" --warmup 10 --ckpt-every "$PF_STEPS" --grid-every "$PF_STEPS" --monitor-every 0 --val-every 100000 --log-every 10; rc=$?
    if [ $rc -ne 0 ] || ! nan_check "$D"; then
      if is_optional "$arm"; then
        echo "PREFLIGHT-FAILED $arm (rc=$rc) -> SKIPPED (optional arm, labeled)"; echo "preflight failed rc=$rc $(date -u +%FT%TZ)" | gsutil -q cp - "$GCS/${arm}_SKIPPED"
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

select_best () {  # ARM DIR -> "NNNNNN val step": the DEC arms on the EMA monitor (earliest tie, the raw monitor second); N0 on its val rows
  if is_dec "$1"; then $PY tools/select_ckpt.py "$2" --key val_t16_ema --tie earliest --second-key val_t16 2>/dev/null
  else $PY tools/select_ckpt.py "$2" --row val --key val_frac --tie earliest --second-key val_pix_mean 2>/dev/null; fi
}

run_pretrain () {  # ARM — ONE-SHOT NaN amputation; the registered EXTENSION rule (DEC arms) after the budget completes
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
  if is_dec "$arm" && [ ! -f "$D/STOPPED.txt" ] && [ ! -f "$D/EXTENDED.txt" ]; then
    sel=$(select_best "$arm" "$D") && best=$(echo "$sel" | awk '{print $3}')
    if [ -n "${best:-}" ] && [ "$best" -ge $((budget - EXT_WINDOW)) ]; then
      echo "EXTENDED from $budget to $((budget + EXT_STEPS)) (peak at $best) $(date -u +%FT%TZ)" > "$D/EXTENDED.txt"
      gsutil -q cp "$D/EXTENDED.txt" "$GCS/${arm}_EXTENDED"
      echo "PRETRAIN-EXTEND $arm: selected grid $best inside the last $EXT_WINDOW of $budget -> +$EXT_STEPS (R-DA-EXT, once)"
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

# ---------- the battery (idempotent by GCS marker; sharded over the worker's chips; n-gated; banked) ----------
set_n () { case $1 in valhard*) echo 48;; dev30) echo 30;; rg96) echo 96;; rt48) echo 48;; arc1eval) echo 400;; *) echo 0;; esac; }
finish_dec_eval () {  # NAME OUTDIR NGATE — summarize, n-gate on the task count, bank, mark
  local name=$1 O=$2 NGATE=$3
  $PY tools/eval_decarc.py --out "$O" --summarize > "$O/summary.log" 2>&1 || { echo "EVAL-SUMMARY-FAILED $name"; return 1; }
  ${REAL_PY:-python3} -c "import json,sys; s=json.load(open('$O/summary.json')); sys.exit(0 if s['n_tasks']==$NGATE and s.get('xcheck_all', True) else 1)" \
    || { echo "EVAL-N-BAD $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "EVAL-OK $name $(date -u +%H:%M)"
}
eval_dec () {  # ARM SETNAME SET CK NSH EXTRA... — the DEC-ARC battery on one set, NSH-way sharded over the chips
  local arm=$1 setname=$2 set=$3 CK=$4 NSH=$5; shift 5
  local name="${arm}_${setname}" O="runs/decarceval_${arm}/${setname}"
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "EVAL-SKIP $name"; return 0; }
  mkdir -p "$O"; local pids=() rc=0
  for i in $(seq 0 $((NSH - 1))); do
    pin $((i % NCHIP)) ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/eval_decarc.py --ckpt "$CK" --set "$set" --out "$O" --shard "$i/$NSH" \
        --steps "$FIT_STEPS" --t-total "$T_TOTAL" "$@" > "$O/shard_$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { echo "EVAL-SHARD-FAILED $name"; return 1; }
  finish_dec_eval "$name" "$O" "$(set_n "$setname")"
}
eval_nat () {  # SETNAME SET CHIP TASKS_CSV EXTRA... — the native control through tools/arc_suite.py (one chip per set)
  local setname=$1 set=$2 chip=$3 tasks=$4; shift 4
  local name="N0_${setname}" O="runs/decarceval_N0/${setname}"
  gsutil -q stat "$GCS/evals/${name}_OK" 2>/dev/null && { echo "EVAL-SKIP $name"; return 0; }
  case $setname in rg96|rt48) [ -n "$tasks" ] || { echo "GATE-TASKS-EMPTY $name (the gate set's ids could not be listed)"; return 1; };; esac
  mkdir -p "$O"
  pin "$chip" ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/arc_suite.py --ckpt "$N0_CK" --set "$set" ${tasks:+--tasks "$tasks"} --out "$O" --steps "$FIT_STEPS" --t-total "$T_TOTAL" "$@" > "$O/run.log" 2>&1 \
    || { echo "EVAL-FAILED $name"; return 1; }
  ${REAL_PY:-python3} -c "import sys; n=sum(1 for l in open('$O/results.jsonl') if l.strip()); sys.exit(0 if n==$(set_n "$setname") else 1)" \
    || { echo "EVAL-N-BAD $name"; return 1; }
  tar czf "/tmp/${name}.tgz" "$O" && gsutil -q cp "/tmp/${name}.tgz" "$GCS/evals/${name}.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/${name}_OK"
  echo "EVAL-OK $name $(date -u +%H:%M)"
}
gate_tasks () {  # SET -> the comma list of the gate set's task ids from the data dirs (the same rule as eval_decarc.task_ids_of; no heavy import)
  ${REAL_PY:-python3} - "$1" <<'PYEOF'
import sys
from pathlib import Path
name = sys.argv[1]
dirs = {"rg96": ("re_gate48", "re_gateb48"), "rt48": ("re_train48",)}[name]
ids = sorted(p.stem for d in dirs for p in (Path("data") / d).glob("*.json") if not p.name.startswith("."))
print(",".join(ids))
PYEOF
}

battery () {  # ARM VBCK D — the registered battery per arm (the DEC rows; N0's native rows)
  local arm=$1 VBCK=$2 D=$3 rc=0
  if is_dec "$arm"; then
    eval_dec "$arm" valhard valhard "$VBCK" "$NCHIP" --ema --k "$K_VH" --ladder 0,0.2,0.4,0.6,0.8 --views "$VIEWS_VH" --flip-test || rc=1
    eval_dec "$arm" dev30 dev30 "$VBCK" "$NCHIP" --ema --k "$K_VH" --ladder 0,0.2,0.4,0.6,0.8 --views "$VIEWS_VH" --flip-test || rc=1
    eval_dec "$arm" rg96 rg96 "$VBCK" "$NCHIP" --ema --k "$K_GATE" --ladder 0 --views 1 || rc=1
    eval_dec "$arm" rt48 rt48 "$VBCK" "$NCHIP" --ema --k "$K_GATE" --ladder 0 --views 1 || rc=1
    eval_dec "$arm" arc1eval arc1eval "$VBCK" "$NCHIP" --ema --k "$ARC1_K" --ladder 0 --views "$ARC1_VIEWS" || rc=1
    if [ "$VBCK" != "$D/ckpt_latest.pkl" ]; then   # the FINAL grid on val-hard = the memorization row
      eval_dec "$arm" valhard_final valhard "$D/ckpt_latest.pkl" "$NCHIP" --ema --k 0 --ladder 0 --views 1 || rc=1
    else
      echo "FINAL-IS-VSEL $arm (identical grid; the final row := the vsel row, labeled)"
    fi
  else
    N0_CK="$VBCK"; local pids=() r=0
    eval_nat valhard valhard 0 "" --k "$K_VH" & pids+=($!)
    eval_nat dev30 dev30 1 "" --k "$K_VH" & pids+=($!)
    eval_nat rg96 valhard 2 "$(gate_tasks rg96)" --k "$K_GATE" & pids+=($!)
    eval_nat rt48 valhard 3 "$(gate_tasks rt48)" --k "$K_GATE" & pids+=($!)
    for p in "${pids[@]}"; do wait "$p" || r=1; done
    [ $r -eq 0 ] || rc=1
  fi
  return $rc
}

run_arm () {  # ARM — pretrain (+ the extension rule), the selection, the battery
  local arm=$1 D=runs/pretrain${R_TAG}_$1
  gsutil -q stat "$GCS/${arm}_SKIPPED" 2>/dev/null && { echo "ARM-SKIPPED $arm (labeled)"; return 0; }
  gsutil -q stat "$GCS/${arm}_ARM_OK" 2>/dev/null && { echo "ARM-SKIP $arm (done)"; return 0; }
  run_pretrain "$arm" || return 1
  ensure_local_pretrain "$arm"
  local VB="" VBCK="$D/ckpt_latest.pkl" sel v
  sel=$(select_best "$arm" "$D") && v=$(echo "$sel" | awk '{print $2}')
  if [ -n "${v:-}" ]; then VB=$(echo "$sel" | awk '{print $1}'); [ -f "$D/ckpt_$VB.pkl" ] && VBCK="$D/ckpt_$VB.pkl"; echo "$sel" > "$D/val_best.txt"; fi
  if [ "$VBCK" = "$D/ckpt_latest.pkl" ]; then
    echo "VB-FALLBACK-FINAL $arm (select_ckpt returned no banked step; vsel := final, labeled)"; echo "FALLBACK-FINAL" > "$D/val_best.txt"
    ${REAL_PY:-python3} -c "import json; json.dump({'step': None, 'val': None, 'ckpt': '$VBCK', 'fallback': 'final'}, open('$D/vsel.json', 'w'))"
  else
    echo "VALBEST $arm $(cat "$D/val_best.txt") -> $VBCK"
    ${REAL_PY:-python3} -c "import json; s='$sel'.split(); json.dump({'step': int(s[2]), 'val': float(s[1]), 'ckpt': '$VBCK'}, open('$D/vsel.json', 'w'))"
  fi
  gsutil -q cp "$D/vsel.json" "$GCS/${arm}_vsel.json"
  battery "$arm" "$VBCK" "$D" || { echo "ARM-PARTIAL $arm (an eval failed; a rerun redoes only the missing rows)"; return 1; }
  echo ok | gsutil -q cp - "$GCS/${arm}_ARM_OK"
  echo "ARM-OK $arm $(date -u +%H:%M)"
}

# ---------- the RIDER: the champion C2's full-set D64 Sudoku row (RIDER=1 only; the PI's reserve gate) ----------
run_rider () {
  [ "$RIDER" = 1 ] || { echo "RIDER-OFF (RIDER=$RIDER)"; return 0; }
  gsutil -q stat "$GCS/evals/rider_C2_full_t64_OK" 2>/dev/null && { echo "RIDER-SKIP (done)"; return 0; }
  if gsutil ls "$GCS/RIDER_CLAIM_w*" >/dev/null 2>&1 && ! gsutil -q stat "$GCS/RIDER_CLAIM_w$W" 2>/dev/null; then echo "RIDER-CLAIMED-ELSEWHERE"; return 0; fi
  echo "w$W $(date -u +%FT%TZ)" | gsutil -q cp - "$GCS/RIDER_CLAIM_w$W"
  mkdir -p data/sudoku_extreme runs/champ_ckpts
  [ -f "$NPZ" ] || gsutil -q cp "$GCS_CHAMP_SETS/$(basename "$NPZ")" "$NPZ" || { echo "RIDER-NPZ-MISSING"; return 1; }
  [ -f runs/champ_ckpts/C2_vsel.pkl ] || gsutil -q cp "$GCS_SETS/C2_vsel.pkl" runs/champ_ckpts/C2_vsel.pkl || { echo "RIDER-CKPT-MISSING"; return 1; }
  local O=runs/sxeval_${R_TAG}_rider/full_vsel_t64 pids=() rc=0
  mkdir -p "$O"
  for i in $(seq 0 $((NCHIP - 1))); do
    pin "$i" ${EVAL_TIMEOUT:+timeout $EVAL_TIMEOUT} $PY tools/eval_sudoku_extreme.py --ckpt runs/champ_ckpts/C2_vsel.pkl --npz "$NPZ" --out "$O" \
        --shard "$i/$NCHIP" --bank-every 300 --split test --t-total 64 --ema --record-by-step > "$O/shard_$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { echo "RIDER-SHARD-FAILED"; return 1; }
  JAX_PLATFORMS=cpu $PY tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  ${REAL_PY:-python3} -c "import json,sys; s=json.load(open('$O/summary_all.json')); sys.exit(0 if s['n']==422786 else 1)" || { echo "RIDER-N-BAD"; return 1; }
  tar czf /tmp/rider_C2_full_t64.tgz "$O" && gsutil -q cp /tmp/rider_C2_full_t64.tgz "$GCS/evals/rider_C2_full_t64.tgz"
  echo ok | gsutil -q cp - "$GCS/evals/rider_C2_full_t64_OK"
  echo "RIDER-OK C2 full-set D64 $(date -u +%H:%M)"
}

# ---------- static assignment ----------
rc=0
preflight || { echo "$SENT-PREFLIGHT-ABORT worker=$W $(date -u +%FT%TZ)"; exit 1; }
for arm in $(worker_arms); do run_arm "$arm" || rc=1; done
run_rider || echo "RIDER-FAILED (labeled; the night's arms are unaffected)"

# ---------- completion (any worker; idempotent) ----------
need="$ALL_ARMS"
echo "COMPLETION-SET nw=$NW need=[$need]"
for pass in $(seq 1 "${C1_WAIT_PASSES:-300}"); do
  all=1
  for armx in $need; do gsutil -q stat "$GCS/${armx}_ARM_OK" 2>/dev/null || gsutil -q stat "$GCS/${armx}_SKIPPED" 2>/dev/null || all=0; done
  if [ "$all" -eq 1 ]; then
    if ! gsutil -q stat "$GCS/${R_TAG}_final.tgz" 2>/dev/null; then
      for f in $(gsutil ls "$GCS/evals/*.tgz" "$GCS"/*_pretrain.tgz 2>/dev/null); do
        b=$(basename "$f"); [ -f "/tmp/pull_$b" ] || { gsutil -q cp "$f" "/tmp/pull_$b" && tar xzf "/tmp/pull_$b" 2>/dev/null; }
      done
      for armx in $need; do gsutil -q cp "$GCS/${armx}_vsel.json" "runs/pretrain${R_TAG}_${armx}/vsel.json" 2>/dev/null || true; done
      tar czf "/tmp/${R_TAG}_final.tgz" runs/pretrain${R_TAG}_* runs/preflight${R_TAG}_*.log runs/decarceval_* runs/sxeval_${R_TAG}_rider 2>/dev/null
      gsutil -q cp "/tmp/${R_TAG}_final.tgz" "$GCS/${R_TAG}_final.tgz"
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
  sleep "${C1_WAIT_SLEEP:-120}"
done
echo "$SENT-WAIT-TIMEOUT worker=$W $(date -u +%FT%TZ)"
exit $rc
