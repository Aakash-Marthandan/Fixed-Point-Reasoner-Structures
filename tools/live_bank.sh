#!/bin/bash
# tools/live_bank.sh — LIVE 5-MIN BANKING of in-flight campaign state to GCS + the fresh-node RESTORE.
# Ledger/HANDOFF 2026-09-04 (PI directive mid-sportC2: "5-min cadence banking to GCS so we progress seamlessly
# through preemptions and node switches without losing existing progress"). The sportC chains banked to GCS only
# at STAGEA-OK / PRETRAIN-OK / EVAL-OK; everything IN FLIGHT (the trainer's 1k-cadence ckpt_latest, the 5k grids,
# metrics, the evaluator's 300 s partials) lived on the node alone, so a node change lost the in-flight pretrain
# and every running eval (2026-09-04: R2/R4 were hand-banked once; R2's 5k-25k grids were lost with the v6e-16).
# chain_r0 (2026-08-14) had this loop; the sportC chains dropped it. This restores it as ONE tool:
#   bash tools/live_bank.sh loop      the 5-min loop (pidfile runs/live_bank.pid, refuses to start twice; the chain
#                                     starts it at launch, and it may be started standalone via setsid nohup)
#   bash tools/live_bank.sh once      one pass: serial `gsutil rsync` (never -m, never deletes) of runs/ to
#                                     $GCS/live/runs/, banked arms' pretrain dirs and static junk excluded; waits
#                                     30 s if a checkpoint was written < 30 s ago (pickle.dump is not atomic)
#   bash tools/live_bank.sh restore   at every chain start: on a FRESH node pull $GCS/live/runs -> runs/ NO-CLOBBER
#                                     (banked arms excluded: their tarballs are the record), then SANITIZE every
#                                     local pretrain dir — a ckpt_latest that does not unpickle -> the newest loadable
#                                     5k grid (LIVE-RESTORE-FALLBACK, labeled, <= 5k steps re-run), none -> removed
#                                     (LIVE-RESTORE-FRESH, labeled). Runs on the same node too (pull skipped when this
#                                     node is the live source), so a checkpoint torn by the wall-recycle SIGTERM can no
#                                     longer enter the NaN/amputation path.
# Verification: tools/harness_sportC2.sh S10 (restore + RESUMED), S10b (torn ckpt -> fallback), S10c (a banked arm's
# stale live copy is NOT restored over its banked final). Env: GCS, R_TAG, ARMS, LIVE_EVERY, REAL_PY (load-verify
# python; the venv's python under JAX_PLATFORMS=cpu on the node — never touches the chips).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
GCS=${GCS:-gs://qhrrn2-rescue/sportC2}; LIVE="$GCS/live"; R_TAG=${R_TAG:-sportC2}
ARMS=${ARMS:-W0 R1 R2 R3 R4 X1 X2}
PIDF=runs/live_bank.pid; LOG=runs/live_bank.log
EXCL='^(pretrainsportC1_|pretrainsportBr2b_|_live_restore|_canary|pretrain6_|cloud/|.*_pull|.*\.tmp\.npz$|.*\.tmp$|live_bank\.(pid|log|out)$)'
PYV=${REAL_PY:-}; [ -n "$PYV" ] || { [ -x .venv/bin/python ] && PYV=.venv/bin/python || PYV=python3; }
say () { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

banked_excl () {  # regex of pretrain dirs whose arm (or stage A) is already BANKED — the tarballs are the record
  local a x=""
  for a in $ARMS; do
    if gsutil -q stat "$GCS/${a}_PRETRAIN_OK" 2>/dev/null; then x="$x|^pretrain${R_TAG}_${a}/|^pretrain${R_TAG}_${a}a/"
    elif gsutil -q stat "$GCS/${a}_STAGEA_OK" 2>/dev/null; then x="$x|^pretrain${R_TAG}_${a}a/"; fi
  done
  echo "${x#|}"
}
sync_once () {   # one serial rsync pass; the 30 s guard waits out a checkpoint mid-write
  mkdir -p runs
  local fresh; fresh=$(find runs -maxdepth 2 -name 'ckpt_*.pkl' -newermt '-30 seconds' 2>/dev/null | head -1)
  [ -n "$fresh" ] && [ -z "${LIVE_NO_GUARD:-}" ] && sleep 30
  local bx x out rc n; bx=$(banked_excl); x="$EXCL"; [ -n "$bx" ] && x="$EXCL|$bx"
  out=$(perl -e 'alarm 900; exec @ARGV' -- gsutil rsync -r -C -x "$x" runs "$LIVE/runs" 2>&1); rc=$?   # bounded: a hung upload never stalls the loop silently
  n=$(printf '%s\n' "$out" | grep -c 'Copying' || true)
  say "LIVE-BANK rc=$rc uploaded=$n"
  [ $rc -eq 0 ] || printf '%s\n' "$out" | tail -3 >> "$LOG"
  return $rc
}
verify_ckpt () {  # FILE -> prints its step, or fails (CPU only; never the chips)
  JAX_PLATFORMS=cpu "$PYV" - "$1" <<'PYEOF'
import pickle, sys
d = pickle.load(open(sys.argv[1], "rb"))
assert isinstance(d, dict) and "state" in d and "step" in d
print(int(d["step"]))
PYEOF
}
sanitize () {   # every local pretrain dir: unloadable ckpt_latest -> newest loadable grid (labeled) / none -> removed (labeled)
  local d st g
  for d in runs/pretrain${R_TAG}_*/; do
    d=${d%/}; [ -f "$d/ckpt_latest.pkl" ] || continue
    if st=$(verify_ckpt "$d/ckpt_latest.pkl" 2>/dev/null); then echo "LIVE-CKPT-OK $d step $st"; continue; fi
    for g in $(ls -r "$d"/ckpt_0*.pkl 2>/dev/null); do
      if st=$(verify_ckpt "$g" 2>/dev/null); then
        cp -f "$g" "$d/ckpt_latest.pkl"; say "LIVE-RESTORE-FALLBACK $d ckpt_latest unloadable -> $(basename "$g") step $st (labeled; <= 5k steps re-run)"
        continue 2
      fi
    done
    rm -f "$d/ckpt_latest.pkl"; say "LIVE-RESTORE-FRESH $d no loadable grid -> the arm restarts from step 0 (labeled)"
  done
}
restore () {
  mkdir -p runs
  if [ -f "$LOG" ]; then
    say "LIVE-RESTORE skip-pull (this node is the live source; local state is at least as new)"
  elif gsutil ls "$LIVE/runs/" >/dev/null 2>&1; then
    local bx n rc; bx=$(banked_excl)
    # 2026-09-04 FIX (found live on the first real node-switch): gsutil rsync REQUIRES the dest dir to EXIST — a bare
    # `rm -rf` left it absent, so every download errored (CommandException) and, under -q 2>&1, silently pulled 0
    # (the harness stub auto-created dest dirs and hid it; the stub now errors like real gsutil). mkdir it, bound the
    # rsync, and log a LOUD warning on any nonzero rc so a future pull failure is never silent again.
    rm -rf runs/_live_restore; mkdir -p runs/_live_restore
    if [ -n "$bx" ]; then perl -e 'alarm 600; exec @ARGV' -- gsutil -q rsync -r -C -x "$bx" "$LIVE/runs" runs/_live_restore >/dev/null 2>&1; rc=$?
    else perl -e 'alarm 600; exec @ARGV' -- gsutil -q rsync -r -C "$LIVE/runs" runs/_live_restore >/dev/null 2>&1; rc=$?; fi
    n=$(find runs/_live_restore -type f 2>/dev/null | wc -l | tr -d ' ')
    [ "${rc:-0}" -ne 0 ] && say "LIVE-RESTORE-PULL-WARN rc=$rc ($n files pulled; rsync error — the banked _pretrain.tgz tarballs are the fallback)"
    [ -d runs/_live_restore ] && cp -Rn runs/_live_restore/. runs/ 2>/dev/null
    rm -rf runs/_live_restore
    say "LIVE-RESTORE pulled=$n files from $LIVE/runs (no-clobber; banked arms excluded: ${bx:-none})"
  else
    say "LIVE-RESTORE none (no live prefix at $LIVE/runs)"
  fi
  sanitize
}
loop () {
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "LIVE-BANK loop already running (pid $(cat "$PIDF"))"; exit 0; fi
  mkdir -p runs; echo $$ > "$PIDF"
  trap 'kill "${SLEEP_PID:-}" 2>/dev/null; rm -f "$PIDF"; exit 0' TERM INT
  say "LIVE-BANK loop start pid=$$ every ${LIVE_EVERY:-300}s -> $LIVE/runs"
  while true; do sync_once || true; sleep "${LIVE_EVERY:-300}" & SLEEP_PID=$!; wait "$SLEEP_PID"; done
}
case ${1:-} in
  loop)    loop;;
  once)    sync_once;;
  restore) restore;;
  *) sed -n '2,22p' "$0"; exit 64;;
esac
