#!/bin/bash
# Ledger: chip-parallel sweep launcher (PI throughput directive 2026-07-28).
# Runs K measure.py shards, one per TPU chip, on a multi-chip TPU VM.
# Usage (remotely, via dispatcher run --detach):
#   bash tools/shard_run.sh K [measure.py args...]
# Chip pinning env recipe validated by tools/probe_chips.py BEFORE first use.
set -u
K=$1; shift
mkdir -p runs
pids=()
for i in $(seq 0 $((K - 1))); do
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_CHIPS=$i \
  python3 tools/measure.py --shard "$i/$K" "$@" > "runs/shard_$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "shard_run: all $K shards finished rc=$rc"
tail -n 2 runs/shard_*.log
exit $rc
