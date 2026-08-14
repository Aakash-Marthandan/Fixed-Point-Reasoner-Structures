#!/bin/bash
# Ledger: RUNG 0 = pretrain-13-full v2 campaign chain (launch registration
# 2026-08-14). The ladder's seeded anchor: d48/T6 @ 40k, 4 arms x 3 seeds.
#   A1 floors-priced | A2 global-priced (H-32 pair) | A3 plain (G3/Law-4)
#   A4 floors+NI s=.01 (G1 clean attribution)
# Batteries (registered reduced matrix): lad + ladrg + ladrgb (G4 rg-96)
#   x all 12; ladrt (G2) seed-0 arms; samp-mi A1(all)+A2s0; e1e3 (G1)
#   A1+A4 all seeds. Trained scalars ride the ckpts (G7; analyzer-side).
# Durability: live 5-min ckpt sync + post-arm ckpt stage + per-wave battery
# stage + per-task probe resume (the siege ladder, final form).
# Two-pod split (32-chip us-east1-d quota): R0_ARMS env picks this pod's
# arm set; batteries auto-restrict to R0_ARMS substrates.
# Usage: bash tools/chain_r0.sh VH_CSV RG_CSV RB_CSV RT_CSV
set -euo pipefail
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH=src
VH=$1; RG=$2; RB=$3; RT=$4
GCS=gs://qhrrn2-rescue/r0
COMMON="--equilibrium --d 48 --T 6 --anchor-p 0.3 --steps 40000
        --rearc --conceptarc --orbit 4 --dp"
PRICED="--beta-flux 3e-5 --beta-flux-nl 1e-5"
FLOORS="--flux-floors 350,75,50,15,30"

ARMS=${R0_ARMS:-"A1s0 A1s1 A1s2 A2s0 A2s1 A2s2 A3s0 A3s1 A3s2 A4s0 A4s1 A4s2"}

arm_flags () {
  case ${1%s*} in
    A1) echo "$PRICED $FLOORS" ;;
    A2) echo "$PRICED" ;;
    A3) echo "" ;;
    A4) echo "$PRICED $FLOORS --ni-sigma 0.01" ;;
    *)  echo "UNKNOWN-ARM $1" >&2; return 1 ;;
  esac
}

pretrain_arm () {
  TAG=$1
  SEED=${TAG#*s}
  echo "=== PRETRAIN $TAG $(date -u +%H:%M) ==="
  mkdir -p "runs/pretrain13f_$TAG"
  if gsutil -q cp "$GCS/${TAG}_ckpt_live.pkl" \
      "runs/pretrain13f_$TAG/ckpt_latest.pkl" 2>/dev/null; then
    echo "RESUME-$TAG-FROM-GCS"
  fi
  ( while true; do sleep 300; gsutil -q cp \
      "runs/pretrain13f_$TAG/ckpt_latest.pkl" \
      "$GCS/${TAG}_ckpt_live.pkl" 2>/dev/null || true; done ) &
  SYNC_PID=$!
  # shellcheck disable=SC2086
  python3 tools/pretrain.py --out "runs/pretrain13f_$TAG" $COMMON \
    --seed "$SEED" $(arm_flags "$TAG") && echo "PRETRAIN-$TAG-OK"
  kill "$SYNC_PID" 2>/dev/null || true
  gsutil -q cp "runs/pretrain13f_$TAG/ckpt_latest.pkl" \
    "$GCS/${TAG}_ckpt.pkl" && echo "CKPT-STAGE-$TAG-OK" \
    || echo "CKPT-STAGE-$TAG-FAILED"
}

run_waves () {
  local -a QUEUE=("$@")
  local i=0
  while [ $i -lt ${#QUEUE[@]} ]; do
    pids=()
    for c in $(seq 0 $(( ${NCHIPS:-8} - 1 ))); do
      [ $((i)) -ge ${#QUEUE[@]} ] && break
      IFS='|' read -r TAG CMD <<< "${QUEUE[$i]}"
      echo ">>> wave job chip$c: $TAG"
      env TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 \
          TPU_VISIBLE_CHIPS=$c JAX_DEFAULT_MATMUL_PRECISION=highest \
          bash -c "$CMD" > "runs/wave_${TAG}.log" 2>&1 &
      pids+=("$!")
      i=$((i+1))
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    echo "wave done rc=$rc $(date -u +%H:%M)"
    tar czf /tmp/r0_partial.tgz runs/lad_p13f* runs/ladrg_p13f* \
      runs/ladrgb_p13f* runs/ladrt_p13f* runs/samp_p13f* runs/e1e3_p13f* \
      2>/dev/null || true
    gsutil -q cp /tmp/r0_partial.tgz "$GCS/partial_results.tgz" \
      2>/dev/null || true
  done
}

# resume: restore staged partial batteries so probes skip completed tasks
if gsutil -q cp "$GCS/partial_results.tgz" /tmp/r0p.tgz 2>/dev/null; then
  tar xzf /tmp/r0p.tgz -C . 2>/dev/null && echo "RESUME-PARTIAL-RESULTS"
fi

# ---- PHASE 1: pretrains back-to-back, DP-8, zero idle ----
for TAG in $ARMS; do
  if [ -f "runs/pretrain13f_$TAG/.done" ]; then
    echo "SKIP-$TAG (done)"; continue
  fi
  pretrain_arm "$TAG" && touch "runs/pretrain13f_$TAG/.done"
done
echo "PHASE1-OK $(date -u +%H:%M)"

# ---- PHASE 2: batteries, 8-way chip-pinned waves ----
Q=()
for TAG in $ARMS; do
  CK="runs/pretrain13f_$TAG/ckpt_latest.pkl"
  Q+=("lad_$TAG|python3 tools/probe_ladder.py --ckpt $CK --tasks $VH --out runs/lad_p13f$TAG")
  Q+=("rg_$TAG|python3 tools/probe_ladder.py --ckpt $CK --tasks $RG --out runs/ladrg_p13f$TAG")
  Q+=("rb_$TAG|python3 tools/probe_ladder.py --ckpt $CK --tasks $RB --out runs/ladrgb_p13f$TAG")
  case $TAG in *s0)
    Q+=("rt_$TAG|python3 tools/probe_ladder.py --ckpt $CK --tasks $RT --out runs/ladrt_p13f$TAG") ;;
  esac
  case $TAG in A1s*|A2s0)
    Q+=("mi_$TAG|python3 tools/probe_sample.py --ckpt $CK --tasks $VH --out runs/samp_p13f${TAG}_mi --k 16 --temps 0.0 --init random") ;;
  esac
  case $TAG in A1s*|A4s*)
    Q+=("e13_$TAG|python3 tools/probe_e1e3.py --ckpt $CK --tasks $VH --out runs/e1e3_p13f$TAG") ;;
  esac
done
echo "PHASE2: ${#Q[@]} battery jobs"
run_waves "${Q[@]}"
echo "PHASE2-OK $(date -u +%H:%M)"

# ---- PHASE 3: final rescue ----
tar czf /tmp/r0_final.tgz runs/pretrain13f_*/ckpt_latest.pkl \
  runs/lad_p13f* runs/ladrg_p13f* runs/ladrgb_p13f* runs/ladrt_p13f* \
  runs/samp_p13f* runs/e1e3_p13f* runs/wave_*.log 2>/dev/null || true
gsutil cp /tmp/r0_final.tgz "$GCS/r0_final.tgz" && echo "RESCUE-OK"
echo "CHAIN-R0-COMPLETE $(date -u +%H:%M)"
