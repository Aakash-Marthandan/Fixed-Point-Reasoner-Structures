#!/bin/bash
# THE RECIPE ABLATION chain (2026-09-19; Documentation/Plan_2026-09-19_Recipe_Ablation.md = the registration). The PI: "Change the three things to
# locate the advantage we have." SE-RRM's two mixers inside OUR block, loop and recipe (the width ladder's SA256) read 98.24 / 99.60 on the 5,000
# against SE-RRM's published 93.73 / 98.22 — so some part of our loop and recipe is worth about 4.5 points at 16 iterations. Each arm is SA256
# with ONE recipe item moved toward SE-RRM's published setup:
#   SA256L  the LOOP        --trm-lambda 0 --trm-beta 0          (TRM's exact loop: no EqR damping, no path noise)
#   SA256S  the START-UP    --trm-ri-sigma 0 --fpa-k 0           (no randomized init, no anchor rows)
#   SA256O  the OPTIMIZER   --batch 272 --lr 5e-4 --lr-end 5e-4  (SE-RRM's batch and learning rate)
# SA256's own protocol otherwise: seed 0, a FIXED 30,000 steps (never extended), the 512-puzzle selection with the earliest tie, the REDUCED
# battery on the identical 5,000 test puzzles (16 and 64 iterations from the fixed start + the 5k x k32 t64 restart scan). The reference is the
# ladder's banked SA256 row; nothing is re-run. ONE spot v6e-8 per arm under its OWN fresh prefix ($GCS); the compile cache is seeded read-only
# from the ladder pod that trained SA256. Completion: every arm's ARM_OK -> a manifest tarball ($FINAL) + the sentinel.
# Harness: tools/harness_sablate.sh. The frozen analyzer: tools/analyze_sablate.py.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src
export GCS=${GCS:?set GCS to the fresh prefix of this pod}
export LIVE_PREFIX=${LIVE_PREFIX:-$GCS/live}
SRC=${AB_SRC:-gs://qhrrn2-arc/rescue/wladder_p1}
WL_ARMS=${AB_ARMS:?set AB_ARMS to the arm(s) of this pod}
SENT=${AB_SENT:-CHAIN-SABLATE}; FINAL=${FINAL_OBJ:-sablate_final.tgz}
W=${CHAIN_WORKER:-0}
REQ=""; for a in $WL_ARMS; do REQ="$REQ ${a}_ARM_OK"; done

echo "=== SABLATE START worker=$W arms=[$WL_ARMS] chips=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]') gcs=$GCS live=$LIVE_PREFIX $(date -u +%FT%TZ) ==="
gsutil -q stat "$GCS/$FINAL" 2>/dev/null && { echo "$SENT-COMPLETE worker=$W (final object present)"; exit 0; }
for a in $WL_ARMS; do case $a in SA256L|SA256S|SA256O|SA256B|SA256BR|SA256U) ;; *) echo "AB-BAD-ARM $a (the ablation's arms are SA256L SA256S SA256O; attribution round 1 adds SA256B SA256BR; round 2 adds SA256U)"; exit 2;; esac; done
mkdir -p runs
gsutil -q stat "$GCS/SYNC-AB" 2>/dev/null || echo "skipped: the recipe ablation carries no sync rider" | gsutil -q cp - "$GCS/SYNC-AB"
gsutil -q stat "$GCS/RECIPE-DEC" 2>/dev/null || printf 'BATCHONLY' | gsutil -q cp - "$GCS/RECIPE-DEC"
if ! gsutil -q stat "$GCS/jax_cache.tgz" 2>/dev/null && [ ! -d jax_cache ]; then
  gsutil -q cp "$SRC/jax_cache.tgz" /tmp/ab_jc_src.tgz 2>/dev/null && tar xzf /tmp/ab_jc_src.tgz 2>/dev/null && echo "COMPILE-CACHE seeded from $SRC (read-only)"
fi

echo "AB-LANE1 $WL_ARMS via tools/chain_champ.sh (LADDER_BATTERY=1, fixed 30k) $(date -u +%FT%TZ)"
SELF_TEARDOWN=0 LADDER_BATTERY=1 CHAMP_ALL_ARMS="$WL_ARMS" CHAMP_ARMS_1X8="$WL_ARMS" SENT=CHAMPAB C1_WAIT_PASSES=1 C1_POLL_SLEEP=5 \
  C1_STEPS_X=30000 bash tools/chain_champ.sh; rc1=$?
echo "AB-LANE1-END rc=$rc1 $(date -u +%FT%TZ)"

need=""; for m in $REQ; do gsutil -q stat "$GCS/$m" 2>/dev/null || need="$need $m"; done
[ -z "$need" ] || { echo "SABLATE-INCOMPLETE worker=$W (missing:$need) $(date -u +%FT%TZ)"; exit 1; }
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
