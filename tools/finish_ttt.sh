#!/bin/bash
# Ledger: phase-3 FINISHING WAVE (2026-08-14) — after the ceiling-kill at
# plain/ewc ~45/48, kl ~27/48: compute the missing (mode, task) pairs from
# every results file present (chain dirs + any kl shards), distribute them
# over all 8 chips as small disjoint slices, run to completion, stage to
# GCS, sentinel. Usage: bash tools/finish_ttt.sh VH_TASKS_CSV
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
VH=$1
GCS=gs://qhrrn2-rescue/p13w2
python3 - <<'EOF' > /tmp/missing.txt
import json, glob, os, sys
vh = sorted(os.environ["VH"].split(","))
jobs = []
for mode in ("plain", "kl", "ewc"):
    done = set()
    pats = [f"runs/ttt_{mode}_p1248c/results.jsonl"] + \
           (glob.glob("runs/ttt_kl_s*/results.jsonl") if mode == "kl" else [])
    for p in pats:
        try:
            for line in open(p):
                try: done.add(json.loads(line)["task"])
                except Exception: pass
        except FileNotFoundError: pass
    missing = [t for t in vh if t not in done]
    for t in missing:
        jobs.append(f"{mode} {t}")
print("\n".join(jobs))
EOF
N=$(wc -l < /tmp/missing.txt)
echo "FINISH-WAVE: $N missing (mode,task) fits"
if [ "$N" -eq 0 ]; then echo "TTT-ALREADY-COMPLETE"; else
  pkill -f probe_lora 2>/dev/null; sleep 5
  for c in 0 1 2 3 4 5 6 7; do
    TASKS_C=$(awk -v c=$c 'NR % 8 == c' /tmp/missing.txt)
    [ -z "$TASKS_C" ] && continue
    ( echo "$TASKS_C" | while read -r MODE T; do
        FLAG=""; [ "$MODE" = "kl" ] && FLAG="--kl 1.0"
        [ "$MODE" = "ewc" ] && FLAG="--ewc 1.0"
        env TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 \
            TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest \
            python3 tools/probe_lora.py \
            --ckpt runs/pretrain12_48c_40k/ckpt_latest.pkl --tasks "$T" \
            --out "runs/ttt_${MODE}_fin" $FLAG \
            >> "runs/finwave_$c.log" 2>&1
      done ) &
  done
  wait
  echo "FINISH-WAVE-DONE"
fi
tar czf /tmp/p13_ttt_final.tgz runs/ttt_* runs/finwave_*.log 2>/dev/null || true
gsutil -q cp /tmp/p13_ttt_final.tgz "$GCS/ttt_final.tgz" \
  && echo "RESCUE-TTT-FINAL-OK" || echo "RESCUE-TTT-FINAL-FAILED"
echo "FINISH-TTT-COMPLETE"
