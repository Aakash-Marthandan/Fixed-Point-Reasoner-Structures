#!/bin/bash
# PAPER FINAL RUNS chain (2026-09-13; Documentation/Plan_2026-09-13_Paper_Final_Runs.md = the registration; the PI: "let's finish
# the recommended runs for the paper first, then we'll close off Sudoku and focus on ARC"). ONE v6e-8 (1 host x 8 chips).
#   Lane 1  tools/chain_champ.sh with C7 C8 = C5's recipe (the symmetric DEC at width 192) at seeds 1 / 2, the registered champion
#           battery unchanged (preflight, 30k + the extension rule, the 512-monitor selection with the earliest tie, D16 full, final
#           and raw rows on 50k, D64 on 100k, D128 on 20k, D256 on 5k, the 5k x k32 scan, census, calibration, screens).
#   Lane 2  tools/filler_full.sh (the filler that banked C2's and C5's full-set rows on 2026-09-12):
#           (a) C7 C8: d64full (all 422,786), d128sub + d256sub (the 50k) = the rows C5 now carries;
#           (b) k128port eqr: EqR's released weights, k128 t64 on the champion arms' identical 5k (the frontier headline scan's flags);
#           (c) C4: d64full (the set-attention arm's D64 on the full set).
# Idempotent by the champ markers (GCS=gs://qhrrn2-rescue/champ; the filler's under champ/filler); the in-flight state is live-banked
# under a FRESH prefix (LIVE_PREFIX=champ_paper/live; the champion night's 6.3 GB live prefix is never restored). The filler runs as
# W=2 so the champion night's stale d64full_C4_CLAIM_w2 reads as this worker's own claim. On every required marker: a manifest tarball
# ($FINAL_OBJ) + the sentinel; the supervisor tears the node down on either. Harness: tools/harness_paperfinal.sh.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src
export GCS=${GCS:-gs://qhrrn2-rescue/champ}
export LIVE_PREFIX=${LIVE_PREFIX:-gs://qhrrn2-rescue/champ_paper/live}
SENT=${PF_SENT:-CHAIN-PAPERFINAL}; FINAL=${FINAL_OBJ:-paperfinal_final.tgz}
W=${CHAIN_WORKER:-0}
REQ="C7_ARM_OK C8_ARM_OK filler/d64full_C7_OK filler/d64full_C8_OK filler/d128sub_C7_OK filler/d128sub_C8_OK filler/d256sub_C7_OK filler/d256sub_C8_OK filler/k128port_eqr_OK filler/d64full_C4_OK"

echo "=== PAPERFINAL START worker=$W chips=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]') live=$LIVE_PREFIX $(date -u +%FT%TZ) ==="
gsutil -q stat "$GCS/$FINAL" 2>/dev/null && { echo "$SENT-COMPLETE worker=$W (final object present)"; exit 0; }

# ---------- lane 1: the width-192 seed pair through the champion chain ----------
echo "PF-LANE1 C7 C8 via tools/chain_champ.sh $(date -u +%FT%TZ)"
SELF_TEARDOWN=0 CHAMP_ARMS_1X8="C7 C8" CHAMP_EXTRA_ARMS="C7 C8" SENT=CHAMPW192 C1_WAIT_PASSES=1 C1_POLL_SLEEP=5 \
  bash tools/chain_champ.sh; rc1=$?
echo "PF-LANE1-END rc=$rc1 $(date -u +%FT%TZ)"
# a killed stall-ladder scan can orphan its evaluator (the watchdog kills the shell wrapper): the filler would wait on it silently.
# Report it LOUDLY for ops (inspect with py-spy; a kill is the PI's call); never kill here.
orph=$(pgrep -fa 'tools/pretrain[.]py|tools/eval_sudoku_extreme[.]py|tools/arc_suite[.]py' 2>/dev/null | cut -c1-160)
[ -z "$orph" ] || echo "PF-ORPHANS lane 1 left trainer/evaluator processes alive; the filler waits on them: $(echo "$orph" | tr '\n' ';' | cut -c1-400)"

# ---------- lane 2: the filler rows (their own live-bank loop; chain_champ's loop ended with it) ----------
R_TAG=champ ARMS="C4 C7 C8" LIVE_EVERY=${LIVE_EVERY:-300} nohup bash tools/live_bank.sh loop > runs/live_bank_pf.log 2>&1 &
LBP=$!; trap 'kill "$LBP" 2>/dev/null' EXIT
filler () {  # FILLER_ARMS JOBS — one filler invocation (W=2; MY_ARMS=C4 is ARM_OK, so wait_idle never blocks on an arm)
  echo "PF-FILLER arms=[$1] jobs=[$2] $(date -u +%FT%TZ)"
  W=2 MY_ARMS="C4" FILLER_ARMS="$1" JOBS="$2" FILLER_ROOT=$PWD GCS=$GCS R_TAG=champ \
    PASS_SLEEP=${PASS_SLEEP:-60} PASSES=${PF_PASSES:-4} IDLE_POLL=${IDLE_POLL:-60} \
    bash tools/filler_full.sh 2>&1 | tee -a runs/filler_full_pf.log
}
filler "C7 C8" "d64full d128sub d256sub"
filler "C4" "k128port"
filler "C4" "d64full"

# ---------- completion ----------
need=""; for m in $REQ; do gsutil -q stat "$GCS/$m" 2>/dev/null || need="$need $m"; done
[ -z "$need" ] || { echo "PAPERFINAL-INCOMPLETE worker=$W (missing:$need) $(date -u +%FT%TZ)"; exit 1; }
files=""
for p in runs/pretrainchamp_C7/metrics.jsonl runs/pretrainchamp_C8/metrics.jsonl runs/pretrainchamp_C7/val_best.txt runs/pretrainchamp_C8/val_best.txt \
         runs/pretrainchamp_C7/EXTENDED.txt runs/pretrainchamp_C8/EXTENDED.txt runs/preflightchamp_C7.log runs/preflightchamp_C8.log \
         runs/sxeval_pchampC7/*/summary_all.json runs/sxeval_pchampC8/*/summary_all.json runs/sxscan_pchampC7/summary_all.json runs/sxscan_pchampC8/summary_all.json \
         runs/filler_sxeval_pchampC4_full_t64/summary_all.json runs/filler_sxeval_pchampC7_*/summary_all.json runs/filler_sxeval_pchampC8_*/summary_all.json \
         runs/filler_sxscan128_pport_eqr/summary_all.json runs/filler_full_pf.log; do
  [ -e "$p" ] && files="$files $p"
done
# shellcheck disable=SC2086
tar czf "/tmp/$FINAL" $files 2>/dev/null && gsutil -q cp "/tmp/$FINAL" "$GCS/$FINAL" && echo "FINAL-BANKED $GCS/$FINAL ($(echo $files | wc -w | tr -d ' ') files)"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
