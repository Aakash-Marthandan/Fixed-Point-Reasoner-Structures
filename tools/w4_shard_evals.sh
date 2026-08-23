#!/bin/bash
# tools/w4_shard_evals.sh — ops ACCELERATION helper (2026-08-23, PI: "prioritize W4 relaunch and evals since
# it's the long pole; do the rest in parallel"). The chain runs each arm's full-test evals on the arm's OWN chip
# (d32 full t64 ≈ 2.75 h single-chip) while other chips sit idle near the end of a campaign. This helper, run
# detached on the pod beside the chain, waits for an arm's FINAL ckpt, then runs the SAME evaluator with the
# SAME flags sharded over idle chips and --merge'd into the chain's eval dir (summary_all.json + records_all.npz,
# the chain's own file contract). The chain's eval_one is skip-if-present per file, so it skips what is already
# merged; if the chain's single-chip eval of that file had already started, the helper retires that ONE
# redundant process AFTER the merged result is in place (the chain logs EVAL-<arm>-fullNN-FAILED and proceeds —
# the files it needs are present). Nothing here changes any registered number: identical evaluator, identical
# flags, identical outputs.
#   ARMS="W4 W9" CHIPS="1 4 5 6 7" bash tools/w4_shard_evals.sh      (detached: see tools/HANDOFF.md)
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src
ARMS=${ARMS:-"W4 W9"}; CHIPS=${CHIPS:-"1 4 5 6 7"}; TAG=${TAG:-sport2w2}; TS=${TS:-"6 64"}
NPZ=data/sudoku_extreme/sudoku_extreme_seed0.npz
pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest "$@"; }
K=$(echo $CHIPS | wc -w | tr -d ' ')
echo "SHARD-HELPER start $(date -u +%FT%TZ) arms=[$ARMS] chips=[$CHIPS] ts=[$TS]"
for ARM in $ARMS; do
  D=runs/pretrain${TAG}_$ARM; O=runs/sxeval_p${TAG}$ARM
  n=0; until [ -f "$D/.done" ]; do sleep 30; n=$((n+1)); [ $n -gt 240 ] && { echo "SHARD-$ARM-TIMEOUT waiting for $D/.done"; continue 2; }; done
  echo "SHARD-$ARM ckpt final $(date -u +%H:%M)"
  for t in $TS; do
    if [ -f "$O/full_t$t/summary_all.json" ]; then echo "SHARD-$ARM-t$t skip (present)"; continue; fi
    mkdir -p "$O/full_t$t"; i=0; pids=()
    for c in $CHIPS; do
      if [ ! -f "$O/full_t$t/summary_s$i.json" ]; then
        ( pin $c python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" --split test --t-total $t --k-init 0 \
            --shard $i/$K --out "$O/full_t$t" > "runs/shard_${ARM}_t${t}_s$i.log" 2>&1 || echo "SHARD-$ARM-t$t-s$i-FAILED" ) & pids+=($!)
      fi
      i=$((i+1))
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || true; done
    if [ -f "$O/full_t$t/summary_all.json" ]; then echo "SHARD-$ARM-t$t: chain finished first (summary_all present) — leaving it"; continue; fi
    # JAX_PLATFORMS=cpu: the merge is pure numpy over the shard .npz files and must NOT touch the TPU —
    # without it jax tries to init the backend and dies on /dev/vfio busy while the chips run other jobs
    # (live bug 2026-08-23: every in-flight merge failed this way; the shard COMPUTES were valid).
    if JAX_PLATFORMS=cpu python3 tools/eval_sudoku_extreme.py --merge "$O/full_t$t" > "runs/shard_${ARM}_t${t}_merge.log" 2>&1 && [ -f "$O/full_t$t/summary_all.json" ]; then
      echo "SHARD-$ARM-t$t-OK $(date -u +%H:%M) (merged $K shards)"
      # retire the chain's redundant single-chip eval of THIS file, if it is running (merged result is in place)
      for p in $(pgrep -f "sxeval_p${TAG}${ARM}/full_t$t" 2>/dev/null); do
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -- '--shard' && continue
        kill "$p" 2>/dev/null && echo "SHARD-$ARM-t$t: retired the chain's redundant single-chip eval (pid $p) — merged result present"
      done
    else echo "SHARD-$ARM-t$t-MERGE-FAILED (chain will run it itself)"; fi
  done
done
echo "SHARD-HELPER done $(date -u +%FT%TZ)"
