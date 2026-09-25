#!/bin/bash
# The review-period P1/P2 queue (registered 2026-09-26): three attention widths in parallel, then the C5 and EqR receivers in sequence.
# PID-based waits only (never pattern-based). Launch: nohup bash runs/analysis/rebuttal_20260926/queue.sh > runs/analysis/rebuttal_20260926/logs/queue.log 2>&1 &
set -u
cd /Users/aakash/Projects/HRRN
OUT=runs/analysis/rebuttal_20260926; mkdir -p $OUT/logs
export JAX_PLATFORMS=cpu
echo "queue start $(date -u +%FT%TZ) pid $$"
PIDS=()
for W in 128 192 256; do
  nohup .venv/bin/python tools/rebuttal_p1p2.py run --width $W --group all > $OUT/logs/attention_$W.log 2>&1 &
  PIDS+=($!); echo "launched width $W pid $!"
done
for P in "${PIDS[@]}"; do while kill -0 "$P" 2>/dev/null; do sleep 60; done; echo "pid $P finished $(date -u +%FT%TZ)"; done
for R in C5 EQR; do
  echo "port $R start $(date -u +%FT%TZ)"
  .venv/bin/python tools/rebuttal_p1p2.py port --receiver $R > $OUT/logs/port_$R.log 2>&1; echo "port $R rc=$? $(date -u +%FT%TZ)"
done
.venv/bin/python tools/rebuttal_p1p2.py report > $OUT/logs/report.log 2>&1; echo "report rc=$?"
echo "queue done $(date -u +%FT%TZ)"
