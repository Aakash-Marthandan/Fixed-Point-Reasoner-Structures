#!/bin/bash
# tools/shard_merge_wait.sh — MERGE-ONLY companion to tools/w4_shard_evals.sh (2026-08-23). It NEVER launches
# shards; it waits for an arm's per-shard full-test eval outputs (produced by the sharding helper on the idle
# chips) and CPU-merges them into the chain's eval dir, then retires the chain's redundant single-chip eval of
# that file. The merge is pure numpy over the shard .npz files; JAX_PLATFORMS=cpu keeps it off the (busy) TPU —
# the bug that made w4_shard_evals.sh's in-line merge fail with `/dev/vfio busy`. Outputs are byte-identical to
# a single full-test run (verified: sharded+merged == single, per-puzzle records identical); idempotent.
#   ARM=W9 TS="6 64" bash tools/shard_merge_wait.sh          (run detached; see tools/HANDOFF.md)
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src
ARM=${ARM:?}; TS=${TS:-"6 64"}; TAG=${TAG:-sport2w2}
for t in $TS; do
  O=runs/sxeval_p${TAG}${ARM}/full_t$t
  n=0
  until [ -f "$O/summary_all.json" ] || [ "$(ls $O/summary_s*.json 2>/dev/null | wc -l)" -eq 5 ]; do
    sleep 30; n=$((n+1)); [ $n -gt 240 ] && { echo "MERGEWAIT-$ARM-t$t TIMEOUT"; break; }
  done
  if [ -f "$O/summary_all.json" ]; then echo "MERGEWAIT-$ARM-t$t: summary_all already present (chain or prior merge)"; continue; fi
  if [ "$(ls $O/summary_s*.json 2>/dev/null | wc -l)" -eq 5 ]; then
    JAX_PLATFORMS=cpu python3 tools/eval_sudoku_extreme.py --merge "$O" > runs/mergewait_${ARM}_t$t.log 2>&1
    if [ -f "$O/summary_all.json" ]; then
      python3 -c "import json,glob; tot=sum(json.load(open(f))['n'] for f in glob.glob('$O/summary_s*.json')); s=json.load(open('$O/summary_all.json')); assert s['n']==tot,(s['n'],tot); print('MERGEWAIT-$ARM-t$t-OK n=%d exact=%.5f'%(s['n'],s['exact_acc']))"
      for p in $(pgrep -f "sxeval_p${TAG}${ARM}/full_t$t"); do tr '\0' ' ' </proc/$p/cmdline | grep -q -- '--shard' && continue; kill $p 2>/dev/null && echo "MERGEWAIT-$ARM-t$t retired chain single-chip pid $p"; done
    else echo "MERGEWAIT-$ARM-t$t MERGE-FAILED"; tail -3 runs/mergewait_${ARM}_t$t.log; fi
  fi
done
echo "MERGEWAIT-$ARM-DONE $(date -u +%FT%TZ)"
