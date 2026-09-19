#!/bin/bash
# THE ATTENTION-ARM EXTENSION chain (2026-09-19; Documentation/Plan_2026-09-19_SA_Extension.md = the registration). The PI: "extend SA128 / SA192 /
# SA256 from 30k to 50k, then the full champion battery." The width ladder trained the three attention arms (SE-RRM's two mixers inside our block,
# loop and recipe) to a FIXED 30,000 steps and read them on 5,000 puzzles; two of the three were selected at the budget's last grid. This chain
# RESUMES one arm's banked 30k state (optimizer state, EMA, RNG: an exact continuation, the learning rate is constant) to 50,000 steps — the
# paper's width-192 budget and the champion recipe's registered one-time extension (+20,000) — selects over ALL 25 grids with the registered
# rule, and runs the champion battery; lane 2 adds the rows the paper's width-192 seeds carry beyond it (SE_JOBS). ONE v6e-8 per arm, its own
# fresh prefix $GCS. Nothing under $SRC (the ladder's prefix) is written: <ARM>_pretrain.tgz and jax_cache.tgz are READ.
#   Lane 0  the pre-staged markers under $GCS (<ARM>_EXTENDED = the registered budget; SYNC-AB = no rider; RECIPE-DEC = the ladder's scan recipe)
#           and the resume point: the live prefix, else the arm's banked 30k state from $SRC; ASSERT 30,000 <= ckpt_latest <= 50,000
#           (SE-NO-RESUME-STATE otherwise: the extension never trains from scratch). The banked run's config is staged as
#           $D/config_banked.json (and $GCS/<ARM>_config_banked.json): chain_champ.sh's run_pretrain then refuses the arm unless the exact
#           argv equals the banked run's in every key but the budget (tools/resume_flags_guard.py, the trainer's own parser).
#   Lane 1  tools/chain_champ.sh for the arm alone: preflight, the resume 30k -> 50k through the registered extension path, the 512-puzzle
#           selection with the earliest tie over every banked grid, the champion battery (screens, D16 on all 422,786, final and raw rows,
#           D64 on 100k, D128 on 20k, D256 on 5k, the 5k x k32 scan, census, calibration) on the selected grid.
#   Lane 2  tools/filler_full.sh, jobs SE_JOBS on the arm's selected grid (default: d64full = D64 on all 422,786; d128sub / d256sub = D128 /
#           D256 on the 50k; k128 = the 5k x k128 t64 restart scan; comma-separated). SE_JOBS=none = the champion battery alone (option A).
# Completion: every REQ marker -> the manifest $FINAL + the sentinel; the supervisor tears the node down on either.
# Harness: tools/harness_saext.sh. The frozen analyzer: tools/analyze_saext.py.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src
export GCS=${GCS:?set GCS to the fresh prefix of this pod}
export LIVE_PREFIX=${LIVE_PREFIX:-$GCS/live}
ARM=${SE_ARM:?set SE_ARM to this pod arm}
SRC=${SE_SRC:?set SE_SRC to the ladder prefix that banked the arm}
JOBS=${SE_JOBS-d64full,d128sub,d256sub,k128}; JOBS=${JOBS//,/ }; [ "$JOBS" = none ] && JOBS=""   # commas: the value rides pod.sh's command line (no spaces); none = option A
SENT=${SE_SENT:-CHAIN-SAEXT}; FINAL=${FINAL_OBJ:-saext_final.tgz}
W=${CHAIN_WORKER:-0}
RPY=${REAL_PY:-python3}
EXT_FROM=30000; EXT_TO=50000; EXT_BY=$((EXT_TO - EXT_FROM))
D=runs/pretrainchamp_$ARM
ck_step () { JAX_PLATFORMS=cpu $RPY -c "import pickle,sys; print(int(pickle.load(open(sys.argv[1],'rb'))['step']))" "$1" 2>/dev/null; }

echo "=== SAEXT START worker=$W arm=$ARM jobs=[$JOBS] chips=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]') gcs=$GCS src=$SRC (read-only) live=$LIVE_PREFIX budget=$EXT_FROM->$EXT_TO $(date -u +%FT%TZ) ==="
gsutil -q stat "$GCS/$FINAL" 2>/dev/null && { echo "$SENT-COMPLETE worker=$W (final object present)"; exit 0; }
case $ARM in SA128|SA192|SA256) ;; *) echo "SE-BAD-ARM $ARM (the extension's arms are SA128 SA192 SA256)"; exit 2;; esac
for j in $JOBS; do case $j in d64full|d128sub|d256sub|k128) ;; *) echo "SE-BAD-JOB $j (the registered filler jobs are d64full d128sub d256sub k128)"; exit 2;; esac; done
REQ="${ARM}_ARM_OK"; for j in $JOBS; do REQ="$REQ filler/${j}_${ARM}_OK"; done
mkdir -p runs

# ---------- lane 0: the pre-staged markers (under $GCS only) and the resume point ----------
gsutil -q stat "$GCS/${ARM}_EXTENDED" 2>/dev/null || echo "EXTENDED from $EXT_FROM to $EXT_TO (the PI's extension 2026-09-19, registered in Plan_2026-09-19_SA_Extension.md: the paper's 50k budget, not the champion's window rule) $(date -u +%FT%TZ)" | gsutil -q cp - "$GCS/${ARM}_EXTENDED"
gsutil -q stat "$GCS/SYNC-AB" 2>/dev/null || echo "skipped: the extension carries no sync rider" | gsutil -q cp - "$GCS/SYNC-AB"
gsutil -q stat "$GCS/RECIPE-DEC" 2>/dev/null || printf 'BATCHONLY' | gsutil -q cp - "$GCS/RECIPE-DEC"
if ! gsutil -q stat "$GCS/jax_cache.tgz" 2>/dev/null && [ ! -d jax_cache ]; then
  gsutil -q cp "$SRC/jax_cache.tgz" /tmp/se_jc_src.tgz 2>/dev/null && tar xzf /tmp/se_jc_src.tgz 2>/dev/null && echo "COMPILE-CACHE seeded from $SRC (read-only)"
fi
GCS="$GCS" R_TAG=champ ARMS="$ARM" LIVE_PREFIX="$LIVE_PREFIX" bash tools/live_bank.sh restore
if gsutil -q stat "$GCS/${ARM}_PRETRAIN_OK" 2>/dev/null; then
  echo "SE-PRETRAIN-DONE (the extension's 25 grids are banked under $GCS; the $SRC 30k state is not restored)"
else
  SRCTGZ=/tmp/se_src_${ARM}_$$.tgz; rm -f /tmp/se_src_"${ARM}"_*.tgz   # the banked tarball, pulled fresh in THIS life (never a stale copy)
  pull_src () { [ -s "$SRCTGZ" ] || gsutil -q cp "$SRC/${ARM}_pretrain.tgz" "$SRCTGZ" 2>/dev/null; }
  if [ ! -f "$D/ckpt_latest.pkl" ]; then
    pull_src && tar xzf "$SRCTGZ" 2>/dev/null \
      && echo "SE-RESTORE-SRC the banked 30k state from $SRC/${ARM}_pretrain.tgz (read-only)"
  fi
  if [ ! -f "$D/config_banked.json" ]; then   # the BANKED run's config, before the trainer rewrites config.json at the resume
    if gsutil -q stat "$GCS/${ARM}_config_banked.json" 2>/dev/null; then gsutil -q cp "$GCS/${ARM}_config_banked.json" "$D/config_banked.json"
    else
      mkdir -p "$D" && pull_src && tar xzOf "$SRCTGZ" "$D/config.json" > "$D/config_banked.json" 2>/dev/null \
        && $RPY -c "import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if c['argv']['steps']==$EXT_FROM else 1)" "$D/config_banked.json" \
        && gsutil -q cp "$D/config_banked.json" "$GCS/${ARM}_config_banked.json" \
        || { rm -f "$D/config_banked.json"; echo "SE-NO-BANKED-CONFIG (the banked run's config.json could not be staged with steps $EXT_FROM) $(date -u +%FT%TZ)"; exit 2; }
    fi
    echo "SE-BANKED-CONFIG staged (the resume guard compares every key but the budget)"
  fi
  rm -f "$SRCTGZ"
  st=$(ck_step "$D/ckpt_latest.pkl")
  if [ -z "$st" ] || [ "$st" -lt "$EXT_FROM" ] || [ "$st" -gt "$EXT_TO" ]; then
    echo "SE-NO-RESUME-STATE (ckpt_latest step ${st:-none} outside [$EXT_FROM, $EXT_TO]: the extension never trains from scratch) $(date -u +%FT%TZ)"; exit 2
  fi
  echo "SE-RESUME-POINT step $st"
fi

# ---------- lane 1: the resume to 50k + the champion battery, through the champion chain ----------
echo "SE-LANE1 $ARM via tools/chain_champ.sh budget=$EXT_FROM+$EXT_BY (the full champion battery) $(date -u +%FT%TZ)"
SELF_TEARDOWN=0 CHAMP_ALL_ARMS="$ARM" CHAMP_ARMS_1X8="$ARM" SENT=CHAMPSE C1_WAIT_PASSES=1 C1_POLL_SLEEP=5 \
  C1_STEPS_X="$EXT_FROM" C1_EXT_STEPS="$EXT_BY" bash tools/chain_champ.sh; rc1=$?
echo "SE-LANE1-END rc=$rc1 $(date -u +%FT%TZ)"
orph=$(pgrep -fa 'tools/pretrain[.]py|tools/eval_sudoku_extreme[.]py' 2>/dev/null | cut -c1-160)
[ -z "$orph" ] || echo "SE-ORPHANS lane 1 left trainer/evaluator processes alive; the filler waits on them: $(echo "$orph" | tr '\n' ';' | cut -c1-400)"

# ---------- lane 2: the filler rows (their own live-bank loop; lane 1's exited with it) ----------
if [ -z "$JOBS" ]; then
  echo "SE-FILLER-NONE (SE_JOBS=none: the champion battery alone)"
elif gsutil -q stat "$GCS/${ARM}_ARM_OK" 2>/dev/null; then
  GCS="$GCS" R_TAG=champ ARMS="$ARM" LIVE_PREFIX="$LIVE_PREFIX" LIVE_EVERY=${LIVE_EVERY:-300} nohup bash tools/live_bank.sh loop > runs/live_bank_se.log 2>&1 &
  LBP=$!; trap 'kill "$LBP" 2>/dev/null' EXIT
  echo "SE-FILLER arms=[$ARM] jobs=[$JOBS] $(date -u +%FT%TZ)"
  W=2 MY_ARMS="$ARM" FILLER_ARMS="$ARM" JOBS="$JOBS" FILLER_ROOT=$PWD GCS=$GCS R_TAG=champ \
    PASS_SLEEP=${PASS_SLEEP:-60} PASSES=${SE_PASSES:-4} IDLE_POLL=${IDLE_POLL:-60} \
    bash tools/filler_full.sh 2>&1 | tee -a runs/filler_full_se.log
else
  echo "SE-FILLER-SKIP (${ARM}_ARM_OK absent: lane 1 did not complete)"
fi

# ---------- completion ----------
need=""; for m in $REQ; do gsutil -q stat "$GCS/$m" 2>/dev/null || need="$need $m"; done
[ -z "$need" ] || { echo "SAEXT-INCOMPLETE worker=$W (missing:$need) $(date -u +%FT%TZ)"; exit 1; }
files=""
for p in $D/metrics.jsonl $D/val_best.txt $D/config.json $D/config_banked.json $D/resumes.txt $D/EXTENDED.txt $D/STOPPED.txt $D.guard.log runs/preflightchamp_$ARM.log \
         runs/sxeval_pchamp$ARM/*/summary_all.json runs/sxscan_pchamp$ARM/summary_all.json runs/sxscan_pchamp$ARM/records_all.npz \
         runs/sxeval_pchamp$ARM/sub5k_t256/records_all.npz runs/sxscreen_pchamp${ARM}_*/summary_all.json runs/sxcensus_pchamp${ARM}_*/census.json runs/sxcalib_pchamp${ARM}_vsel/calib.json \
         runs/filler_sxeval_pchamp${ARM}_*/summary_all.json runs/filler_sxscan128_pchamp$ARM/summary_all.json runs/filler_sxscan128_pchamp$ARM/records_all.npz runs/filler_full_se.log; do
  [ -e "$p" ] && files="$files $p"
done
# shellcheck disable=SC2086
tar czf "/tmp/$FINAL" $files 2>/dev/null && gsutil -q cp "/tmp/$FINAL" "$GCS/$FINAL" && echo "FINAL-BANKED $GCS/$FINAL ($(echo $files | wc -w | tr -d ' ') files)"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
