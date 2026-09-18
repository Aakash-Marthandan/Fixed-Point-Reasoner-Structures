#!/bin/bash
# THE WIDTH LADDER chain (2026-09-18; Documentation/Plan_2026-09-18_Width_Ladder.md = the registration). A MEASUREMENT for paper 1: the symmetric
# DEC under the registered champion recipe at seed 0 with ONE variable per arm — the hidden size (W128, W256), or OUR reimplementation of SE-RRM's
# mixers on this loop and recipe (SA128 / SA192 / SA256: self-attention over the cells with 2D rotary positions, attention across the fields).
# A FIXED 30,000-step budget (never extended), the 512-puzzle selection with the earliest tie, and the REDUCED battery on the identical 5,000
# test puzzles: 16 and 64 iterations from the fixed start (+ the exact bit per step) and the 5k x k32 t64 restart scan.
# ONE spot v6e-8 per pod under its OWN fresh prefix ($GCS; nothing of an earlier campaign is read but sets/ and the compile cache).
# Completion: every arm's ARM_OK -> a manifest tarball ($FINAL) + the sentinel; the supervisor tears the node down on either.
# Harness: tools/harness_wladder.sh. The frozen analyzer: tools/analyze_wladder.py.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src
export GCS=${GCS:?set GCS to the fresh prefix of this pod}
export LIVE_PREFIX=${LIVE_PREFIX:-$GCS/live}
SRC=${WL_SRC:-gs://qhrrn2-arc/rescue/champ}
WL_ARMS=${WL_ARMS:?set WL_ARMS to the arms of this pod}
SENT=${WL_SENT:-CHAIN-WLADDER}; FINAL=${FINAL_OBJ:-wladder_final.tgz}
W=${CHAIN_WORKER:-0}
REQ=""; for a in $WL_ARMS; do REQ="$REQ ${a}_ARM_OK"; done

echo "=== WLADDER START worker=$W arms=[$WL_ARMS] chips=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]') gcs=$GCS live=$LIVE_PREFIX $(date -u +%FT%TZ) ==="
gsutil -q stat "$GCS/$FINAL" 2>/dev/null && { echo "$SENT-COMPLETE worker=$W (final object present)"; exit 0; }
for a in $WL_ARMS; do case $a in W128|W256|SA128|SA192|SA256) ;; *) echo "WL-BAD-ARM $a (the ladder's arms are W128 W256 SA128 SA192 SA256)"; exit 2;; esac; done
mkdir -p runs
gsutil -q stat "$GCS/SYNC-AB" 2>/dev/null || echo "skipped: the width ladder carries no sync rider" | gsutil -q cp - "$GCS/SYNC-AB"
gsutil -q stat "$GCS/RECIPE-DEC" 2>/dev/null || printf 'BATCHONLY' | gsutil -q cp - "$GCS/RECIPE-DEC"
if ! gsutil -q stat "$GCS/jax_cache.tgz" 2>/dev/null && [ ! -d jax_cache ]; then
  gsutil -q cp "$SRC/jax_cache.tgz" /tmp/wl_jc_src.tgz 2>/dev/null && tar xzf /tmp/wl_jc_src.tgz 2>/dev/null && echo "COMPILE-CACHE seeded from $SRC (read-only)"
fi

echo "WL-LANE1 $WL_ARMS via tools/chain_champ.sh (LADDER_BATTERY=1, fixed 30k) $(date -u +%FT%TZ)"
SELF_TEARDOWN=0 LADDER_BATTERY=1 CHAMP_ALL_ARMS="$WL_ARMS" CHAMP_ARMS_1X8="$WL_ARMS" SENT=CHAMPWL C1_WAIT_PASSES=1 C1_POLL_SLEEP=5 \
  C1_STEPS_X=30000 bash tools/chain_champ.sh; rc1=$?
echo "WL-LANE1-END rc=$rc1 $(date -u +%FT%TZ)"

need=""; for m in $REQ; do gsutil -q stat "$GCS/$m" 2>/dev/null || need="$need $m"; done
[ -z "$need" ] || { echo "WLADDER-INCOMPLETE worker=$W (missing:$need) $(date -u +%FT%TZ)"; exit 1; }
files=""
for a in $WL_ARMS; do
  for p in runs/pretrainchamp_$a/metrics.jsonl runs/pretrainchamp_$a/val_best.txt runs/pretrainchamp_$a/config.json runs/pretrainchamp_$a/resumes.txt runs/preflightchamp_$a.log \
           runs/sxeval_pchamp$a/*/summary_all.json runs/sxeval_pchamp$a/*/records_all.npz runs/sxscan_pchamp$a/summary_all.json runs/sxscan_pchamp$a/records_all.npz; do
    [ -e "$p" ] && files="$files $p"
  done
done
# shellcheck disable=SC2086
tar czf "/tmp/$FINAL" $files 2>/dev/null && gsutil -q cp "/tmp/$FINAL" "$GCS/$FINAL" && echo "FINAL-BANKED $GCS/$FINAL ($(echo $files | wc -w | tr -d ' ') files)"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
