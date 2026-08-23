#!/bin/bash
# tools/idle_filler.sh — IDLE-CHIP FILLER (2026-08-23; PI chip-utilization policy: "make sure we're using
# the 16 chips efficiently"). Runs DETACHED on a pod worker beside the campaign chain. Every 5 min it looks
# for a chip with no TPU process and, while the worker's chain is still in its per-chip arm phase (no
# QUEUES-DONE yet — after that the sharded phases need every chip), starts the next eval-only FILLER job
# on it: breadth k=256 scans (strat-512 @t64, nested k-curve) of banked WAVE-2 checkpoints that the wave-3
# recipe decision wants context from (W2 T12, W3 RI+NI, W8 priced d32, W9 priced d16). Jobs are claimed
# through a GCS marker so several workers never run the same one; results upload as
# breadth_<arm>_t64_k256.tgz into the campaign bucket. Zero cost (the chips were idle), zero risk to the
# campaign (the chain's sharded phases retry on a busy chip). Stops when no jobs remain.
#   FILLERS="W2 W3 W8 W9" bash tools/idle_filler.sh       (see tools/HANDOFF.md)
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src
GCS=${GCS:-gs://qhrrn2-rescue/sport3a}; GCS_W2=${GCS_W2:-gs://qhrrn2-rescue/sport2w2}; W2_TAG=${W2_TAG:-sport2w2}
FILLERS=${FILLERS:-"W2 W3 W8 W9"}; K=${K:-256}; T=${T:-64}; STRAT=${STRAT:-512}
NPZ=data/sudoku_extreme/sudoku_extreme_seed0.npz
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -cE '^[0-9]+$'); [ "${NCHIP:-0}" -ge 1 ] || NCHIP=8
pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest "$@"; }
busy_chips () { for p in $(pgrep -f 'tools/(pretrain|eval_sudoku_extreme|probe_sudoku)'); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^TPU_VISIBLE_CHIPS=' | cut -d= -f2; done | sort -u; }
echo "FILLER start $(date -u +%FT%TZ) host=$(hostname) chips=$NCHIP jobs=[$FILLERS]"
declare -A RUNNING
while true; do
  # stop handing out work once the chain's sharded phases are imminent/running on this worker
  if grep -q "QUEUES-DONE" runs/detached.log 2>/dev/null; then echo "FILLER: chain QUEUES-DONE on this worker — no new filler jobs"; break; fi
  remaining=""
  for arm in $FILLERS; do
    obj=breadth_${arm}_t${T}_k${K}.tgz
    gsutil -q stat "$GCS/$obj" 2>/dev/null && continue
    gsutil -q stat "$GCS/${obj}.claim" 2>/dev/null && { [ -n "${RUNNING[$arm]:-}" ] || continue; }
    remaining="$remaining $arm"
  done
  [ -n "$remaining" ] || { echo "FILLER: no jobs remain $(date -u +%H:%M)"; break; }
  busy=$(busy_chips); free=""
  for c in $(seq 0 $((NCHIP-1))); do echo "$busy" | grep -qx "$c" || { free=$c; break; }; done
  if [ -n "$free" ]; then
    for arm in $remaining; do
      [ -n "${RUNNING[$arm]:-}" ] && continue
      obj=breadth_${arm}_t${T}_k${K}.tgz
      gsutil -q stat "$GCS/${obj}.claim" 2>/dev/null && continue     # claimed by another worker meanwhile
      echo "$(hostname) $(date -u +%FT%TZ)" | gsutil -q cp - "$GCS/${obj}.claim"
      D=runs/pretrain${W2_TAG}_$arm; mkdir -p "$D"
      [ -f "$D/ckpt_latest.pkl" ] || gsutil -q cp "$GCS_W2/${arm}_ckpt.pkl" "$D/ckpt_latest.pkl" || { echo "FILLER-$arm-NOCKPT"; continue; }
      O=runs/sxbreadth_${arm}_t${T}_k${K}
      echo "FILLER-$arm start on chip $free $(date -u +%H:%M)"
      ( pin "$free" python3 tools/eval_sudoku_extreme.py --ckpt "$D/ckpt_latest.pkl" --npz "$NPZ" --split test --stratified "$STRAT" --t-total "$T" --k-init "$K" --out "$O" > "runs/filler_${arm}.log" 2>&1 \
          && tar czf "/tmp/$obj" "$O" && gsutil -q cp "/tmp/$obj" "$GCS/$obj" && echo "FILLER-$arm-OK $(date -u +%H:%M)" || echo "FILLER-$arm-FAILED (see runs/filler_${arm}.log)" ) &
      RUNNING[$arm]=$!
      break    # one launch per poll; re-check free chips next round
    done
  fi
  sleep 300
done
for a in "${!RUNNING[@]}"; do wait "${RUNNING[$a]}" 2>/dev/null; done
echo "FILLER done $(date -u +%FT%TZ)"
