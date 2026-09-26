#!/bin/bash
# P2b queue: waits (PID-based) for the P1c width processes, then runs the three widths in parallel, then the report.
set -u; cd /Users/aakash/Projects/HRRN; OUT=runs/analysis/rebuttal_20260926c; mkdir -p $OUT/logs; export JAX_PLATFORMS=cpu
echo "queue start $(date -u +%FT%TZ) pid $$"
for P in $(grep -E "^launched width" runs/analysis/rebuttal_20260926b/logs/queue.log | awk '{print $NF}'); do while kill -0 "$P" 2>/dev/null; do sleep 60; done; echo "P1c width pid $P finished $(date -u +%FT%TZ)"; done
PIDS=(); for W in 128 192 256; do nohup .venv/bin/python tools/rebuttal_p2b.py run --width $W > $OUT/logs/attention_$W.log 2>&1 & PIDS+=($!); echo "launched width $W pid $!"; done
for P in "${PIDS[@]}"; do while kill -0 "$P" 2>/dev/null; do sleep 60; done; echo "pid $P finished $(date -u +%FT%TZ)"; done
.venv/bin/python tools/rebuttal_p2b.py report > $OUT/logs/report.log 2>&1; echo "report rc=$?"; echo "queue done $(date -u +%FT%TZ)"
