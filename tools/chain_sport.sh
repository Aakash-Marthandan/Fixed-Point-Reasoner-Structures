#!/bin/bash
# Ledger: S-PORT cell 1 (H-33) — two arms on ONE cheap v5e-1 lane, chosen to
# make the landscape-class comparison decidable in a single night:
#   sudA = base (knee-priced, anchors on, NO RI)   -> H-33 (i) + (iii)
#   sudB = sudA + RI 0.15                          -> H-33 (ii), the EqR lever
# Protocol is deliberately MATCHED to our ARC d16 line (d16/T6, 20k steps,
# anchor-p .3, knee beta 3e-5/1e-5, B64, seed 0) so every cross-domain
# comparison is like-for-like; the only differences are the ones the domain
# forces (one task row, generated corpus, no TTT at probe time).
# Each arm: pretrain -> probe (solve/retention/ladder/multi-init) -> stage.
# Resumable: pretrain resumes from ckpt_latest, probes resume per-puzzle.
# Usage: bash tools/chain_sport.sh
set -uo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
GCS=gs://qhrrn2-rescue/sport
# MIXED-DIFFICULTY training + LADDER eval (design review before launch): a
# single hard setting risks solve=0 everywhere, which VOIDS all three H-33
# readouts instead of answering them. Training over 30-50 givens and probing
# at 50/40/30 finds where propagation depth runs out — informative either way.
COMMON="--equilibrium --d 16 --T 6 --steps 20000 --anchor-p 0.3
        --beta-flux 3e-5 --beta-flux-nl 1e-5
        --sudoku 8000 --sudoku-givens 30 --sudoku-givens-hi 50
        --n-val 64 --seed 0"

run_arm () {
  NAME=$1; shift
  echo "=== S-PORT $NAME pretrain $(date -u +%H:%M) ==="
  mkdir -p "runs/sud_$NAME"
  gsutil -q cp "$GCS/${NAME}_ckpt_live.pkl" "runs/sud_$NAME/ckpt_latest.pkl" \
    2>/dev/null && echo "RESUME-$NAME"
  ( while true; do sleep 300; gsutil -q cp "runs/sud_$NAME/ckpt_latest.pkl" \
      "$GCS/${NAME}_ckpt_live.pkl" 2>/dev/null || true; done ) &
  SY=$!
  # shellcheck disable=SC2086
  python3 tools/pretrain.py --out "runs/sud_$NAME" $COMMON "$@" \
    && echo "PRETRAIN-$NAME-OK"
  kill "$SY" 2>/dev/null || true
  gsutil -q cp "runs/sud_$NAME/ckpt_latest.pkl" "$GCS/${NAME}_ckpt.pkl" || true

  echo "=== S-PORT $NAME probe $(date -u +%H:%M) ==="
  # eval-seed differs from the pretrain seed: held-out puzzles by construction
  python3 tools/probe_sudoku.py --ckpt "runs/sud_$NAME/ckpt_latest.pkl" \
    --n 40 --givens-list 50,40,30 --k-init 16 --eval-seed 99999 \
    --out "runs/sudprobe_$NAME" && echo "PROBE-$NAME-OK"
  tar czf /tmp/sport_${NAME}.tgz "runs/sudprobe_$NAME" "runs/sud_$NAME/metrics.jsonl" \
    2>/dev/null || true
  gsutil -q cp /tmp/sport_${NAME}.tgz "$GCS/${NAME}_results.tgz" || true
}

run_arm sudA
run_arm sudB --ri-p 0.15
echo "CHAIN-SPORT-COMPLETE $(date -u +%H:%M)"
