#!/bin/bash
# P1c queue: three widths in parallel + the C5 then EqR receivers alongside; then the report. PID waits only.
set -u; cd /Users/aakash/Projects/HRRN; OUT=runs/analysis/rebuttal_20260926b; mkdir -p $OUT/logs; export JAX_PLATFORMS=cpu
echo "queue start $(date -u +%FT%TZ) pid $$"; PIDS=()
for W in 128 192 256; do nohup .venv/bin/python tools/rebuttal_p1c.py run --width $W > $OUT/logs/attention_$W.log 2>&1 & PIDS+=($!); echo "launched width $W pid $!"; done
( .venv/bin/python tools/rebuttal_p1c.py port --receiver C5 > $OUT/logs/port_C5.log 2>&1; echo "port C5 rc=$? $(date -u +%FT%TZ)"; .venv/bin/python tools/rebuttal_p1c.py port --receiver EQR > $OUT/logs/port_EQR.log 2>&1; echo "port EQR rc=$? $(date -u +%FT%TZ)" ) & PORTS=$!; echo "launched port chain pid $PORTS"
for P in "${PIDS[@]}" $PORTS; do while kill -0 "$P" 2>/dev/null; do sleep 60; done; echo "pid $P finished $(date -u +%FT%TZ)"; done
.venv/bin/python tools/rebuttal_p1c.py report > $OUT/logs/report.log 2>&1; echo "report rc=$?"; echo "queue done $(date -u +%FT%TZ)"
