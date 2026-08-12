#!/bin/bash
# Ledger: chip-parallel PROBE launcher (cluster-Q battery, 2026-08-12) —
# task-sharded probe_sample across K chips of a single-host pod (v6e-8).
# Chip-pinning env recipe = shard_run.sh's, validated by probe_chips.py.
# Usage (remotely, via dispatcher run --detach):
#   bash tools/shard_probe.sh K TASKS_CSV OUTBASE [probe_sample args...]
# Each shard writes ${OUTBASE}_s$i/results.jsonl (no write races; merge at
# analysis). Round-robin task split.
set -u
K=$1; TASKS=$2; OUTBASE=$3; shift 3
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
mkdir -p runs
IFS=',' read -ra ALL <<< "$TASKS"
pids=()
for i in $(seq 0 $((K - 1))); do
  GROUP=$(for j in "${!ALL[@]}"; do
    if [ $((j % K)) -eq "$i" ]; then printf "%s," "${ALL[$j]}"; fi
  done)
  GROUP=${GROUP%,}
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_CHIPS=$i \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  python3 tools/probe_sample.py --tasks "$GROUP" --out "${OUTBASE}_s$i" "$@" \
    > "runs/shardp_$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "shard_probe: all $K shards finished rc=$rc"
tail -n 1 runs/shardp_*.log
exit $rc
