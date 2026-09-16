#!/bin/bash
# THE PENDING SUDOKU RUNS chain (2026-09-17; Documentation/Plan_2026-09-17_Sudoku_Pending_Runs.md = the registration draft; the rows
# paper 1 still owes: the restart column as a triple at k = 128, and the parameter-matched TRM's-cell row under the full recipe).
# ONE v6e-8 (1 host x 8 chips), the champion prefix (GCS=gs://qhrrn2-rescue/champ), idempotent by the champ markers.
#   Lane 1  tools/chain_champ.sh with the X arm(s) (SP_ARMS, default X5 = TRM's cell at hidden 160 under the full recipe: randomized-init +
#           anchor rows + the field's digit augmentation, on the champion loop and regime; a FIXED 50k budget, never extended), through the
#           registered champion battery (preflight, the 512-monitor selection with the earliest tie, D16 full, final and raw rows, D64 on
#           100k, D128 on 20k, D256 on 5k, the 5k x k32 scan, census, calibration, screens).
#   Lane 2  tools/filler_full.sh job k128 on "C7 C8 $SP_ARMS": the k128 t64 restart scan on the identical 5k (the frontier headline scan's
#           flags; the row the champion filler banked as k128_C0..C6, C5's on ckpt_046000). C7's grid = its banked selected grid (46k, the
#           champion prefix); C8's paper grid lives under the extension's prefix -> FILLER_CK_C8 (the registered override; default the
#           c8x 46k grid). SP_ARMS' k128 row runs after lane 1 banked its selected grid.
# Completion: every REQ marker -> a manifest tarball ($FINAL) + the sentinel; the supervisor tears the node down on either.
# Harness: tools/harness_sudokupend.sh. The frozen analyzer: tools/analyze_sudokupend.py.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src
export GCS=${GCS:-gs://qhrrn2-rescue/champ}
export LIVE_PREFIX=${LIVE_PREFIX:-gs://qhrrn2-rescue/champ_pend/live}
export FILLER_CK_C8=${FILLER_CK_C8:-gs://qhrrn2-rescue/c8x/C8_pretrain.tgz|runs/pretrainchamp_C8/ckpt_046000.pkl}
SP_ARMS=${SP_ARMS:-X5}
SENT=${SP_SENT:-CHAIN-SUDOKUPEND}; FINAL=${FINAL_OBJ:-sudokupend_final.tgz}
W=${CHAIN_WORKER:-0}
REQ="filler/k128_C7_OK filler/k128_C8_OK"; for a in $SP_ARMS; do REQ="$REQ ${a}_ARM_OK filler/k128_${a}_OK"; done

echo "=== SUDOKUPEND START worker=$W arms=[$SP_ARMS] chips=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]') live=$LIVE_PREFIX $(date -u +%FT%TZ) ==="
gsutil -q stat "$GCS/$FINAL" 2>/dev/null && { echo "$SENT-COMPLETE worker=$W (final object present)"; exit 0; }

# ---------- lane 1: the X arm(s) through the champion chain ----------
echo "SP-LANE1 $SP_ARMS via tools/chain_champ.sh $(date -u +%FT%TZ)"
SELF_TEARDOWN=0 CHAMP_ARMS_1X8="$SP_ARMS" CHAMP_EXTRA_ARMS="$SP_ARMS" SENT=CHAMPX C1_WAIT_PASSES=1 C1_POLL_SLEEP=5 \
  bash tools/chain_champ.sh; rc1=$?
echo "SP-LANE1-END rc=$rc1 $(date -u +%FT%TZ)"
orph=$(pgrep -fa 'tools/pretrain[.]py|tools/eval_sudoku_extreme[.]py|tools/arc_suite[.]py' 2>/dev/null | cut -c1-160)
[ -z "$orph" ] || echo "SP-ORPHANS lane 1 left trainer/evaluator processes alive; the filler waits on them: $(echo "$orph" | tr '\n' ';' | cut -c1-400)"

# ---------- lane 2: the k128 rows (their own live-bank loop) ----------
R_TAG=champ ARMS="C7 C8 $SP_ARMS" LIVE_EVERY=${LIVE_EVERY:-300} nohup bash tools/live_bank.sh loop > runs/live_bank_sp.log 2>&1 &
LBP=$!; trap 'kill "$LBP" 2>/dev/null' EXIT
filler () {  # FILLER_ARMS JOBS — one filler invocation (W=2; MY_ARMS=C4 is ARM_OK, so wait_idle never blocks on an arm)
  echo "SP-FILLER arms=[$1] jobs=[$2] $(date -u +%FT%TZ)"
  W=2 MY_ARMS="C4" FILLER_ARMS="$1" JOBS="$2" FILLER_ROOT=$PWD GCS=$GCS R_TAG=champ \
    PASS_SLEEP=${PASS_SLEEP:-60} PASSES=${SP_PASSES:-4} IDLE_POLL=${IDLE_POLL:-60} \
    bash tools/filler_full.sh 2>&1 | tee -a runs/filler_full_sp.log
}
filler "C7 C8 $SP_ARMS" "k128"

# ---------- completion ----------
need=""; for m in $REQ; do gsutil -q stat "$GCS/$m" 2>/dev/null || need="$need $m"; done
[ -z "$need" ] || { echo "SUDOKUPEND-INCOMPLETE worker=$W (missing:$need) $(date -u +%FT%TZ)"; exit 1; }
files=""
for a in $SP_ARMS; do
  for p in runs/pretrainchamp_$a/metrics.jsonl runs/pretrainchamp_$a/val_best.txt runs/pretrainchamp_$a/config.json runs/preflightchamp_$a.log \
           runs/sxeval_pchamp$a/*/summary_all.json runs/sxscan_pchamp$a/summary_all.json runs/sxscan_pchamp$a/records_all.npz; do
    [ -e "$p" ] && files="$files $p"
  done
done
for p in runs/filler_sxscan128_pchamp*/summary_all.json runs/filler_sxscan128_pchamp*/records_all.npz runs/filler_full_sp.log; do [ -e "$p" ] && files="$files $p"; done
# shellcheck disable=SC2086
tar czf "/tmp/$FINAL" $files 2>/dev/null && gsutil -q cp "/tmp/$FINAL" "$GCS/$FINAL" && echo "FINAL-BANKED $GCS/$FINAL ($(echo $files | wc -w | tr -d ' ') files)"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
