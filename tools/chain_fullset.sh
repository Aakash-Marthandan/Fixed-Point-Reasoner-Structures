#!/bin/bash
# FULL-SET ROWS CHAIN (2026-09-12; the PI: "run the full eval for C2 D64 as suggested and follow it up with the C5
# recommendations"). The supervisor's chain wrapper around tools/filler_full.sh on ONE v6e-8: lane 1 = C2 d64full (the
# champion by rule's D64 on all 422,786; the seed triple's D64 line complete on the full set), lane 2 = C5 d128sub +
# d256sub (the labeled best's D128 / D256 on the 50k uniform subsample, seed 20260822). Every job is idempotent by the
# champ filler's GCS markers ($GCS/filler/<job>_<arm>_OK; 8-way sharded; 300 s partials -> a killed job resumes) and the
# in-flight partials are live-banked every 5 min under a prefix of their own (a preemption costs <= 5 min). On ALL-DONE a
# manifest tarball + the sentinel let pod.sh tear the node down. Measured paces (the champion pod): 109 puzzle-steps/s
# per chip at w384 -> C2 d64full ~= 8.6 h on 8 chips; the w192 arm ~2.2x faster -> C5's two rows ~= 2.8 h.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src
GCS=${GCS:-gs://qhrrn2-rescue/champ}; LIVE_GCS=${LIVE_GCS:-gs://qhrrn2-rescue/champ_fullset}
SENT=${SENT:-CHAIN-FULLSET}; FINAL=${FINAL_OBJ:-fullset_final.tgz}
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz}
W=${CHAIN_WORKER:-0}
JOBS_ALL="d64full_C2 d128sub_C5 d256sub_C5"
echo "=== FULLSET START worker=$W chips=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]') $(date -u +%FT%TZ) ==="
gsutil -q stat "$GCS/$FINAL" 2>/dev/null && { echo "$SENT-COMPLETE worker=$W (final object present)"; exit 0; }
if [ ! -f "$NPZ" ]; then
  mkdir -p "$(dirname "$NPZ")"; gsutil -q cp "$GCS/sets/$(basename "$NPZ")" "$NPZ" || { echo "$SENT-DATA-ABORT worker=$W (npz)"; exit 2; }
fi
echo "DATA-OK npz=$NPZ"
# the live bank of the in-flight partials (its own prefix: the champion night's live prefix is NOT touched)
GCS=$LIVE_GCS R_TAG=champ ARMS="C2 C5" bash tools/live_bank.sh restore 2>&1 | tail -n 3
GCS=$LIVE_GCS R_TAG=champ ARMS="C2 C5" LIVE_EVERY=${LIVE_EVERY:-300} nohup bash tools/live_bank.sh loop > runs/live_bank.log 2>&1 &
echo "LIVE-BANK loop pid=$! -> $LIVE_GCS/live/runs"
run_lane () {  # ARMS JOBS — one filler pass set; exits ALL-DONE when its jobs' markers exist (retries a failed job after PASS_SLEEP)
  echo "FILLER-LANE arms=[$1] jobs=[$2] $(date -u +%FT%TZ)"
  W=$W MY_ARMS="C2 C5" FILLER_ARMS="$1" JOBS="$2" FILLER_ROOT=$PWD GCS=$GCS R_TAG=champ PASS_SLEEP=${PASS_SLEEP:-120} PASSES=${PASSES:-40} \
    bash tools/filler_full.sh 2>&1 | tee -a "runs/filler_full_w$W.log"
}
run_lane "${FULL_ARMS1:-C2}" "${FULL_JOBS1:-d64full}"
run_lane "${FULL_ARMS2:-C5}" "${FULL_JOBS2:-d128sub d256sub}"
need=""; for j in $JOBS_ALL; do gsutil -q stat "$GCS/filler/${j}_OK" 2>/dev/null || need="$need $j"; done
[ -z "$need" ] || { echo "FULLSET-INCOMPLETE worker=$W (missing:$need)"; exit 1; }
tar czf "/tmp/$FINAL" runs/filler_sxeval_pchamp*/summary_all.json runs/filler_sxeval_pchamp*/merge.log "runs/filler_full_w$W.log" 2>/dev/null \
  && gsutil -q cp "/tmp/$FINAL" "$GCS/$FINAL" && echo "FINAL-BANKED $GCS/$FINAL"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
