#!/bin/bash
# P28 queue: phase 1 sharded screening (NS shards per width), finalize; phase 2 arms (NA shards per width; one 32-puzzle block per shard
# when 128 are eligible); merge; the frozen report. Four processes per phase (two per receiver): the measured CPU saturation within memory.
set -u; cd /Users/aakash/Projects/HRRN; OUT=runs/analysis/rebuttal_20261014; mkdir -p $OUT/logs; export JAX_PLATFORMS=cpu; NS=${NS:-2}; NA=${NA:-2}
echo "queue start $(date -u +%FT%TZ) pid $$ NS=$NS NA=$NA"
PIDS=(); for W in 256 192; do for ((S=0; S<NS; S++)); do .venv/bin/python tools/rebuttal_p28.py screen --width $W --shard $S/$NS > $OUT/logs/screen_${W}_$S.log 2>&1 & PIDS+=($!); done; done
RC=0; for P in "${PIDS[@]}"; do wait $P || RC=1; done; echo "screening done rc=$RC $(date -u +%FT%TZ)"; [ $RC -eq 0 ] || exit 1
for W in 192 256; do .venv/bin/python tools/rebuttal_p28.py finalize --width $W > $OUT/logs/finalize_$W.log 2>&1 || exit 1; echo "finalize $W: $(tail -1 $OUT/logs/finalize_$W.log)"; done
PIDS=(); for W in 256 192; do for ((S=0; S<NA; S++)); do .venv/bin/python tools/rebuttal_p28.py arms --width $W --shard $S/$NA > $OUT/logs/arms_${W}_$S.log 2>&1 & PIDS+=($!); done; done
RC=0; for P in "${PIDS[@]}"; do wait $P || RC=1; done; echo "arms done rc=$RC $(date -u +%FT%TZ)"; [ $RC -eq 0 ] || exit 1
for W in 192 256; do .venv/bin/python tools/rebuttal_p28.py merge --width $W > $OUT/logs/merge_$W.log 2>&1 || exit 1; echo "merge $W: $(tail -1 $OUT/logs/merge_$W.log)"; done
.venv/bin/python tools/rebuttal_p28.py report > $OUT/logs/report.log 2>&1; echo "report rc=$?"; echo "queue done $(date -u +%FT%TZ)"
