#!/bin/bash
# Ledger: chip-parallel sweep launcher (PI throughput directive 2026-07-28).
# Runs K measure.py shards, one per TPU chip, on a multi-chip TPU VM.
# Usage (remotely, via dispatcher run --detach):
#   bash tools/shard_run.sh K [measure.py args...]
# Chip pinning env recipe validated by tools/probe_chips.py BEFORE first use.
set -u
K=$1; shift
# Self-contained env (2026-07-29: `ENV=x cmd1 && cmd2` binds the prefix to
# cmd1 only — a chained second invocation ran on system python and died).
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
mkdir -p runs
pids=()
for i in $(seq 0 $((K - 1))); do
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_CHIPS=$i \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  python3 tools/measure.py --shard "$i/$K" "$@" > "runs/shard_$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "shard_run: all $K shards finished rc=$rc"
tail -n 2 runs/shard_*.log
exit $rc
