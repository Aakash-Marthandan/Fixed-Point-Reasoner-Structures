#!/bin/bash
# THE X5 LONG RUN chain (2026-09-17; Documentation/Plan_2026-09-17_X5_Long.md = the registration). X5 = TRM's cell at hidden 160 under the
# full recipe, parameter-matched to the width-192 DEC; its 50k row was read at the edge of its budget (the pending Sudoku runs' verdict).
# This chain RESUMES X5's banked 50k state to the COMPUTE-MATCHED budget (EXT_TO, default 960,000 steps = 50k x 272.7 / 14.2, the measured
# same-pod training paces of X5 and of the width-192 DEC; six digits, so every grid keeps the ckpt_NNNNNN name the tools glob) and runs the registered champion battery on the monitor-selected grid. ONE v6e-8 (1 host x 8 chips).
# Nothing under $SRC (the champion prefix) is written: X5_pretrain.tgz (the 50k state + its 25 grids + metrics), sets/ and jax_cache.tgz
# are READ; every marker, tarball and the live bank live under the fresh prefix $GCS.
#   Lane 0  the pre-staged markers under $GCS (X5_EXTENDED = the registered budget; SYNC-AB = no rider; RECIPE-DEC = the scan recipe that
#           completed X5's 50k scan) and the resume point: the live prefix, else X5's banked state from $SRC; ASSERT ckpt_latest >= EXT_FROM
#           (X5L-NO-RESUME-STATE otherwise: the long run never trains from scratch); once X5_PRETRAIN_OK exists the grids come from $GCS only.
#   Lane 1  tools/chain_champ.sh for X5 alone (CHAMP_ALL_ARMS="X5"): preflight, the resume EXT_FROM -> EXT_TO through the registered
#           extension path (the same code that extended C5, C7 and C8; the constant field lr makes it an exact continuation), the long
#           cadence (monitor + grid every X5L_MON steps, ckpt_latest every X5L_CKPT), the registered selection over EVERY banked grid
#           (the 50k run's 25 + the long run's), the champion battery on the selected grid.
#   Lane 2  tools/filler_full.sh job k128 on X5: the k128 t64 restart scan on the identical 5k, on the long run's selected grid.
# Completion: every REQ marker -> the manifest $FINAL + the sentinel; the supervisor tears the node down on either.
# Harness: tools/harness_x5long.sh. The frozen analyzer: tools/analyze_x5long.py.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src
export GCS=${GCS:-gs://qhrrn2-arc/rescue/x5l}
export LIVE_PREFIX=${LIVE_PREFIX:-$GCS/live}
SRC=${X5L_SRC:-gs://qhrrn2-arc/rescue/champ}
SENT=${XL_SENT:-CHAIN-X5LONG}; FINAL=${FINAL_OBJ:-x5long_final.tgz}
W=${CHAIN_WORKER:-0}
RPY=${REAL_PY:-python3}
ARM=X5
EXT_FROM=${X5L_EXT_FROM:-50000}; EXT_TO=${X5L_EXT_TO:-960000}
X5L_MON=${X5L_MON:-10000}; X5L_CKPT=${X5L_CKPT:-5000}
D=runs/pretrainchamp_$ARM
REQ="${ARM}_ARM_OK filler/k128_${ARM}_OK"
ck_step () { JAX_PLATFORMS=cpu $RPY -c "import pickle,sys; print(int(pickle.load(open(sys.argv[1],'rb'))['step']))" "$1" 2>/dev/null; }

echo "=== X5LONG START worker=$W chips=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]') gcs=$GCS src=$SRC (read-only) live=$LIVE_PREFIX budget=$EXT_FROM->$EXT_TO $(date -u +%FT%TZ) ==="
gsutil -q stat "$GCS/$FINAL" 2>/dev/null && { echo "$SENT-COMPLETE worker=$W (final object present)"; exit 0; }
[ "$EXT_TO" -gt "$EXT_FROM" ] && [ "$EXT_TO" -le 999999 ] || { echo "X5L-BAD-BUDGET $EXT_FROM -> $EXT_TO (the grid names are six digits: EXT_TO <= 999999)"; exit 2; }
mkdir -p runs

# ---------- lane 0: the pre-staged markers (under $GCS only) and the resume point ----------
gsutil -q stat "$GCS/${ARM}_EXTENDED" 2>/dev/null || echo "EXTENDED from $EXT_FROM to $EXT_TO (the X5 long run's registration 2026-09-17: the compute-matched budget, not the champion's window rule) $(date -u +%FT%TZ)" | gsutil -q cp - "$GCS/${ARM}_EXTENDED"
gsutil -q stat "$GCS/SYNC-AB" 2>/dev/null || echo "skipped: the X5 long run carries no sync rider" | gsutil -q cp - "$GCS/SYNC-AB"
gsutil -q stat "$GCS/RECIPE-DEC" 2>/dev/null || printf 'BATCHONLY' | gsutil -q cp - "$GCS/RECIPE-DEC"
if ! gsutil -q stat "$GCS/jax_cache.tgz" 2>/dev/null && [ ! -d jax_cache ]; then
  gsutil -q cp "$SRC/jax_cache.tgz" /tmp/x5l_jc_src.tgz 2>/dev/null && tar xzf /tmp/x5l_jc_src.tgz 2>/dev/null && echo "COMPILE-CACHE seeded from $SRC (read-only)"
fi
GCS="$GCS" R_TAG=champ ARMS="$ARM" LIVE_PREFIX="$LIVE_PREFIX" bash tools/live_bank.sh restore
if gsutil -q stat "$GCS/${ARM}_PRETRAIN_OK" 2>/dev/null; then
  echo "X5L-PRETRAIN-DONE (the long run's grids are banked under $GCS; the $SRC 50k state is not restored)"
else
  if [ ! -f "$D/ckpt_latest.pkl" ]; then
    gsutil -q cp "$SRC/${ARM}_pretrain.tgz" /tmp/x5l_src.tgz 2>/dev/null && tar xzf /tmp/x5l_src.tgz 2>/dev/null \
      && echo "X5L-RESTORE-SRC X5's banked state from $SRC/${ARM}_pretrain.tgz (read-only)"
  fi
  st=$(ck_step "$D/ckpt_latest.pkl")
  if [ -z "$st" ] || [ "$st" -lt "$EXT_FROM" ]; then
    echo "X5L-NO-RESUME-STATE (ckpt_latest step ${st:-none} < $EXT_FROM: the long run never trains from scratch) $(date -u +%FT%TZ)"; exit 2
  fi
  echo "X5L-RESUME-POINT step $st (>= $EXT_FROM)"
fi

# ---------- lane 1: the resume to the budget + the champion battery, through the champion chain ----------
echo "XL-LANE1 $ARM via tools/chain_champ.sh budget=$EXT_FROM+$((EXT_TO - EXT_FROM)) mon=$X5L_MON ckpt=$X5L_CKPT $(date -u +%FT%TZ)"
SELF_TEARDOWN=0 CHAMP_ALL_ARMS="$ARM" CHAMP_ARMS_1X8="$ARM" SENT=CHAMPXL C1_WAIT_PASSES=1 C1_POLL_SLEEP=5 \
  C1_STEPS_LONG="$EXT_FROM" C1_EXT_STEPS=$((EXT_TO - EXT_FROM)) C1_MON="$X5L_MON" C1_CKPT_EVERY="$X5L_CKPT" \
  bash tools/chain_champ.sh; rc1=$?
echo "XL-LANE1-END rc=$rc1 $(date -u +%FT%TZ)"
orph=$(pgrep -fa 'tools/pretrain[.]py|tools/eval_sudoku_extreme[.]py' 2>/dev/null | cut -c1-160)
[ -z "$orph" ] || echo "XL-ORPHANS lane 1 left trainer/evaluator processes alive; the filler waits on them: $(echo "$orph" | tr '\n' ';' | cut -c1-400)"

# ---------- lane 2: the k128 row (its own live-bank loop; lane 1's exited with it) ----------
if gsutil -q stat "$GCS/${ARM}_ARM_OK" 2>/dev/null; then
  GCS="$GCS" R_TAG=champ ARMS="$ARM" LIVE_PREFIX="$LIVE_PREFIX" LIVE_EVERY=${LIVE_EVERY:-300} nohup bash tools/live_bank.sh loop > runs/live_bank_xl.log 2>&1 &
  LBP=$!; trap 'kill "$LBP" 2>/dev/null' EXIT
  echo "XL-FILLER arms=[$ARM] jobs=[k128] $(date -u +%FT%TZ)"
  W=2 MY_ARMS="$ARM" FILLER_ARMS="$ARM" JOBS="k128" FILLER_ROOT=$PWD GCS=$GCS R_TAG=champ \
    PASS_SLEEP=${PASS_SLEEP:-60} PASSES=${XL_PASSES:-4} IDLE_POLL=${IDLE_POLL:-60} \
    bash tools/filler_full.sh 2>&1 | tee -a runs/filler_full_xl.log
else
  echo "XL-FILLER-SKIP (${ARM}_ARM_OK absent: lane 1 did not complete)"
fi

# ---------- completion ----------
need=""; for m in $REQ; do gsutil -q stat "$GCS/$m" 2>/dev/null || need="$need $m"; done
[ -z "$need" ] || { echo "X5LONG-INCOMPLETE worker=$W (missing:$need) $(date -u +%FT%TZ)"; exit 1; }
files=""
for p in $D/metrics.jsonl $D/val_best.txt $D/config.json $D/resumes.txt $D/EXTENDED.txt $D/STOPPED.txt runs/preflightchamp_$ARM.log \
         runs/sxeval_pchamp$ARM/*/summary_all.json runs/sxscan_pchamp$ARM/summary_all.json runs/sxscan_pchamp$ARM/records_all.npz \
         runs/filler_sxscan128_pchamp$ARM/summary_all.json runs/filler_sxscan128_pchamp$ARM/records_all.npz runs/filler_full_xl.log; do
  [ -e "$p" ] && files="$files $p"
done
# shellcheck disable=SC2086
tar czf "/tmp/$FINAL" $files 2>/dev/null && gsutil -q cp "/tmp/$FINAL" "$GCS/$FINAL" && echo "FINAL-BANKED $GCS/$FINAL ($(echo $files | wc -w | tr -d ' ') files)"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
