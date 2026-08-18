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
# RUNG PARAMETRIZATION (2026-08-15): the harness-verified rung-0 logic is
# reused for every rung rather than forked. Defaults = rung 0 exactly.
#   R_D / R_STEPS  width and budget (rung 1: 64 / 53333 = d48-proportional)
#   R_TAG          run-dir + GCS prefix (rung 0: 13f, rung 1: r1)
R_D=${R_D:-48}; R_STEPS=${R_STEPS:-40000}; R_TAG=${R_TAG:-13f}
GCS=gs://qhrrn2-rescue/$([ "$R_TAG" = 13f ] && echo r0 || echo "r${R_TAG}")
COMMON="--equilibrium --d $R_D --T 6 --anchor-p 0.3 --steps $R_STEPS
        --rearc --conceptarc --orbit 4 --dp"
PRICED="--beta-flux 3e-5 --beta-flux-nl 1e-5"
FLOORS="--flux-floors 350,75,50,15,30"

ARMS=${R0_ARMS:-"A1s0 A1s1 A1s2 A2s0 A2s1 A2s2 A3s0 A3s1 A3s2 A4s0 A4s1 A4s2"}
# PER-POD staging object (2026-08-15): both pods were writing the SAME
# partial_results.tgz, so each restore could overwrite this pod's newer
# battery rows with the other pod's older snapshot of them (arm sets are
# disjoint, so no corruption — but real recompute and confusing row counts:
# both pods reported an identical 768 rows). Key the object to the arm set.
PARTIAL="partial_${ARMS%% *}.tgz"

arm_flags () {
  case ${1%s*} in
    A1) echo "$PRICED $FLOORS" ;;
    A2) echo "$PRICED" ;;
    A3) echo "" ;;
    A4) echo "$PRICED $FLOORS --ni-sigma 0.01" ;;
    A5) echo "$PRICED --ni-sigma 0.01" ;;          # global+NI: floors decomposition (rung 1)
    *)  echo "UNKNOWN-ARM $1" >&2; return 1 ;;
  esac
}

pretrain_arm () {
  TAG=$1
  SEED=${TAG#*s}
  echo "=== PRETRAIN $TAG $(date -u +%H:%M) ==="
  mkdir -p "runs/pretrain${R_TAG}_$TAG"
  # COMPLETED-ARM SHORT-CIRCUIT (2026-08-15): ${TAG}_ckpt.pkl is staged ONLY
  # after an arm finishes, so its presence proves completion. The old resume
  # pulled ${TAG}_ckpt_live.pkl (last 5-min sync) and re-ran the arm's tail —
  # measured cost: every completed arm restarted at 36k/40k after a
  # preemption, ~20 min per recovery for a 6-arm pod, recurring because
  # .done markers die with the node. Prefer the complete ckpt and skip.
  if gsutil -q cp "$GCS/${TAG}_ckpt.pkl" \
      "runs/pretrain${R_TAG}_$TAG/ckpt_latest.pkl" 2>/dev/null; then
    touch "runs/pretrain${R_TAG}_$TAG/.done"
    echo "SKIP-$TAG (completed in an earlier life; final ckpt restored)"
    return 0
  fi
  if gsutil -q cp "$GCS/${TAG}_ckpt_live.pkl" \
      "runs/pretrain${R_TAG}_$TAG/ckpt_latest.pkl" 2>/dev/null; then
    echo "RESUME-$TAG-FROM-GCS"
  fi
  ( while true; do sleep 300; gsutil -q cp \
      "runs/pretrain${R_TAG}_$TAG/ckpt_latest.pkl" \
      "$GCS/${TAG}_ckpt_live.pkl" 2>/dev/null || true; done ) &
  SYNC_PID=$!
  # `if` form, not a bare call: under `set -e` a bare failing statement would
  # abort the whole chain, and a command in an if-condition is exempt.
  # shellcheck disable=SC2086
  if python3 tools/pretrain.py --out "runs/pretrain${R_TAG}_$TAG" $COMMON \
      --seed "$SEED" $(arm_flags "$TAG"); then
    RC=0
  else
    RC=$?
  fi
  kill "$SYNC_PID" 2>/dev/null || true
  # Stage the COMPLETE ckpt ONLY on success — its presence is exactly what
  # the short-circuit above reads as proof of completion, so staging it
  # after a crash would make a later run SKIP an unfinished arm.
  if [ "$RC" -eq 0 ]; then
    echo "PRETRAIN-$TAG-OK"
    gsutil -q cp "runs/pretrain${R_TAG}_$TAG/ckpt_latest.pkl" \
      "$GCS/${TAG}_ckpt.pkl" && echo "CKPT-STAGE-$TAG-OK" \
      || echo "CKPT-STAGE-$TAG-FAILED"
  else
    echo "PRETRAIN-$TAG-FAILED rc=$RC (complete-ckpt NOT staged; live ckpt stands)"
  fi
  return "$RC"
}

# Stage every battery results.jsonl produced so far. Probes resume PER TASK
# from these files, so whatever is banked is never recomputed.
stage_partials () {
  tar czf /tmp/r0_partial.tgz runs/lad_p${R_TAG}* runs/ladrg_p${R_TAG}* \
    runs/ladrgb_p${R_TAG}* runs/ladrt_p${R_TAG}* runs/samp_p${R_TAG}* runs/e1e3_p${R_TAG}* \
    2>/dev/null || true
  gsutil -q cp /tmp/r0_partial.tgz "$GCS/$PARTIAL" 2>/dev/null || true
}

run_waves () {
  local -a QUEUE=("$@")
  local i=0
  # LIVE 5-MIN STAGER (2026-08-15, PI): staging only BETWEEN waves meant a
  # mid-wave preemption threw away the whole wave — up to ~40 min x 8 jobs,
  # which is exactly what pod2's final wave lost. The pretrain phase already
  # syncs every 300s; the battery phase now matches it, so a preemption
  # costs <=5 min of rows regardless of when it lands.
  ( while true; do sleep 300; stage_partials; done ) &
  local STAGER=$!
  # kill the stager AND its child sleep on ANY exit path (set -e, ceiling,
  # signal). Plain `kill $STAGER` left an orphan `sleep 300` (harness-
  # verified) that would hold the detached job open past CHAIN-R0-COMPLETE.
  # pkill -P is portable (procps, present on the pod image); setsid is not
  # guaranteed there.
  trap 'pkill -P "$STAGER" 2>/dev/null; kill "$STAGER" 2>/dev/null || true' RETURN
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
    stage_partials
  done
  pkill -P "$STAGER" 2>/dev/null; kill "$STAGER" 2>/dev/null || true
}

# resume: restore staged partial batteries so probes skip completed tasks.
# Prefer this pod's own object; on the FIRST relaunch after the per-pod
# rename none exists yet, so fall back to the legacy shared object (which
# holds whatever this pod banked before the rename) rather than restoring
# nothing. Restoring the other pod's rows too is harmless: arm sets are
# disjoint, and probes only ever read their own arm's directories.
if gsutil -q cp "$GCS/$PARTIAL" /tmp/r0p.tgz 2>/dev/null \
   || gsutil -q cp "$GCS/partial_results.tgz" /tmp/r0p.tgz 2>/dev/null; then
  tar xzf /tmp/r0p.tgz -C . 2>/dev/null && echo "RESUME-PARTIAL-RESULTS"
  # SANITIZE: live (5-min) staging can tar a results.jsonl mid-write, so a
  # restored file may end in a truncated line. Probes tolerate it on resume
  # but would APPEND after it, leaving permanent corruption that the
  # analyzers (which parse rows without try/except) would hit at verdict
  # time. Drop non-parsing lines once, here, before anything appends.
  R_TAG="$R_TAG" python3 - <<'PYSAN'
import json, pathlib, os
TAG = os.environ.get("R_TAG", "13f")

def parses(line):
    try:
        json.loads(line)
        return True
    except Exception:
        return False

fixed = dropped = 0
for p in pathlib.Path("runs").glob(f"*_p{TAG}*/results.jsonl"):
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    good = [l for l in lines if parses(l)]
    if len(good) != len(lines):
        p.write_text("".join(l + "\n" for l in good))
        fixed += 1
        dropped += len(lines) - len(good)
print(f"SANITIZED {fixed} file(s), dropped {dropped} truncated row(s)")
PYSAN
fi

# ---- PHASE 1: pretrains back-to-back, DP-8, zero idle ----
for TAG in $ARMS; do
  if [ -f "runs/pretrain${R_TAG}_$TAG/.done" ]; then
    echo "SKIP-$TAG (done)"; continue
  fi
  pretrain_arm "$TAG" && touch "runs/pretrain${R_TAG}_$TAG/.done"
done
echo "PHASE1-OK $(date -u +%H:%M)"

# ---- PHASE 2: batteries, 8-way chip-pinned waves ----
Q=()
for TAG in $ARMS; do
  CK="runs/pretrain${R_TAG}_$TAG/ckpt_latest.pkl"
  Q+=("lad_$TAG|python3 tools/probe_ladder.py --ckpt $CK --tasks $VH --out runs/lad_p${R_TAG}$TAG")
  Q+=("rg_$TAG|python3 tools/probe_ladder.py --ckpt $CK --tasks $RG --out runs/ladrg_p${R_TAG}$TAG")
  Q+=("rb_$TAG|python3 tools/probe_ladder.py --ckpt $CK --tasks $RB --out runs/ladrgb_p${R_TAG}$TAG")
  case $TAG in *s0)
    Q+=("rt_$TAG|python3 tools/probe_ladder.py --ckpt $CK --tasks $RT --out runs/ladrt_p${R_TAG}$TAG") ;;
  esac
  case $TAG in A1s*|A2s0)
    Q+=("mi_$TAG|python3 tools/probe_sample.py --ckpt $CK --tasks $VH --out runs/samp_p${R_TAG}${TAG}_mi --k 16 --temps 0.0 --init random") ;;
  esac
  case $TAG in A1s*|A4s*)
    Q+=("e13_$TAG|python3 tools/probe_e1e3.py --ckpt $CK --tasks $VH --out runs/e1e3_p${R_TAG}$TAG") ;;
  esac
done
echo "PHASE2: ${#Q[@]} battery jobs"
run_waves "${Q[@]}"
echo "PHASE2-OK $(date -u +%H:%M)"

# ---- PHASE 3: final rescue ----
tar czf /tmp/r0_final.tgz runs/pretrain${R_TAG}_*/ckpt_latest.pkl \
  runs/lad_p${R_TAG}* runs/ladrg_p${R_TAG}* runs/ladrgb_p${R_TAG}* runs/ladrt_p${R_TAG}* \
  runs/samp_p${R_TAG}* runs/e1e3_p${R_TAG}* runs/wave_*.log 2>/dev/null || true
gsutil cp /tmp/r0_final.tgz "$GCS/r0_final.tgz" && echo "RESCUE-OK"
echo "CHAIN-R0-COMPLETE $(date -u +%H:%M)"
