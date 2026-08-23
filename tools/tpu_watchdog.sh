#!/bin/bash
# Layered TPU watchdog (PI directive 2026-08-12: "make the heartbeat robust").
# Layer 1 of 3 — a launchd-driven poll every 15 min that SURVIVES Claude
# sessions and app restarts:
#   - writes runs/tpu_status.txt as a pure INVENTORY SNAPSHOT (one line;
#     changes only when inventory changes -> a session Monitor on this file
#     wakes Claude exactly on real events, not on polls)
#   - appends timestamped history to runs/tpu_status_log.txt
#   - macOS notification to the PI on inventory CHANGE, and an ALARM when
#     the same non-empty inventory persists >= 8h (DMS window watch) —
#     these reach the PI even when no Claude session is alive.
# Layers 2/3 (session-side): Monitor on the snapshot; hourly cron report.
# Install: tools/install_watchdog.sh (launchd, user-level, no sudo).
PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin
cd "$(dirname "$0")/.." || exit 1
ZONES="us-east1-d us-east1-c us-east5-b us-central1-a us-central2-b us-west1-c us-west4-a asia-east1-c"
SNAP=runs/tpu_status.txt
LOG=runs/tpu_status_log.txt
mkdir -p runs
NEW=""
for z in $ZONES; do
  # perl alarm = portable timeout (no coreutils on this Mac); a wedged gcloud
  # connection must never freeze the watchdog (2026-08-14: one hung poll
  # blinded layer 1 for 45 min — the 07-29 bounded-call law applies here too).
  RAW=$(perl -e 'alarm shift; exec @ARGV' 120 \
      gcloud compute tpus tpu-vm list --zone="$z" --project=quantum-llm \
      --format="value(name,state)" 2>/dev/null)
  RC=$?
  if [ "$RC" -ne 0 ]; then
    # A failed probe is NOT an empty zone — surface blindness, never mask it.
    NEW="$NEW$z=PROBE-FAIL "
  else
    R=$(printf '%s' "$RAW" | tr '\n\t' ' :')
    [ -n "$R" ] && NEW="$NEW$z=$R "
  fi
done
NEW=$(echo "$NEW" | sed 's/ *$//')
OLD=$(cat "$SNAP" 2>/dev/null || echo "")
echo "$(date -u +%FT%TZ) | ${NEW:-none}" >> "$LOG"
if [ "$NEW" != "$OLD" ]; then
  echo "$NEW" > "$SNAP"
  /usr/bin/osascript -e "display notification \"${NEW:-all zones clear}\" \
    with title \"QHRRN TPU watchdog: inventory changed\"" 2>/dev/null
fi
if [ -n "$NEW" ]; then
  SAME=$(tail -n 40 "$LOG" | grep -cF "| $NEW")
  if [ "$SAME" -ge 32 ]; then   # 32 polls x 15 min = 8h same inventory
    # 2026-08-23 (PI: the >=8h alarm kept firing through a supervised 16h campaign): the
    # reminder is for an UNSUPERVISED long-lived node. Stay quiet when the one-pod
    # supervisor is alive AND the deadline file is more than 1h away (the supervisor
    # tears down on completion; the deadline delete below is untouched) — log instead.
    SUP_ALIVE=0; SP=$(cat runs/pod_supervisor.pid 2>/dev/null); [ -n "$SP" ] && kill -0 "$SP" 2>/dev/null && SUP_ALIVE=1
    DL=$(head -1 runs/tpu_deadline.txt 2>/dev/null | tr -dc '0-9'); FAR=0
    [ -n "$DL" ] && [ $(( DL - $(date -u +%s) )) -gt 3600 ] && FAR=1
    if [ "$SUP_ALIVE" -eq 1 ] && [ "$FAR" -eq 1 ]; then
      echo "$(date -u +%FT%TZ) | >=8h READY but supervised (pid $SP) with deadline >1h away — alarm suppressed" >> "$LOG"
    else
      /usr/bin/osascript -e "display notification \"$NEW up >=8h — check DMS \
and teardown\" with title \"QHRRN TPU watchdog ALARM\"" 2>/dev/null
    fi
  fi
fi

# ---- HARD BILLING BACKSTOP (2026-08-14, PI stepping away for the night) ----
# The DMS is a GUEST shutdown and does NOT stop TPU billing (07-29 lesson:
# only node DELETION does), and layers 2/3 die with the Claude session. This
# gives layer 1 — the only layer that survives everything — actual teeth:
# past the deadline in runs/tpu_deadline.txt (UTC epoch seconds), every node
# is DELETED. Safe by construction here: chain_r0 is resume-complete from
# GCS with a 5-min live ckpt sync, so a deletion costs <=5 min of compute,
# never the campaign. Remove/extend the file to change the deadline.
DEADLINE_FILE=runs/tpu_deadline.txt
if [ -n "$NEW" ] && [ -f "$DEADLINE_FILE" ]; then
  DEADLINE=$(head -1 "$DEADLINE_FILE" | tr -dc '0-9')
  NOW=$(date -u +%s)
  if [ -n "$DEADLINE" ] && [ "$NOW" -gt "$DEADLINE" ]; then
    for z in $ZONES; do
      for n in $(gcloud compute tpus tpu-vm list --zone="$z" \
                 --project=quantum-llm --format="value(name)" 2>/dev/null); do
        echo "$(date -u +%FT%TZ) | DEADLINE-DELETE $z/$n" >> "$LOG"
        gcloud compute tpus tpu-vm delete "$n" --zone="$z" \
          --project=quantum-llm --quiet >/dev/null 2>&1 &
      done
    done
    wait
    /usr/bin/osascript -e "display notification \"deadline passed — all TPUs \
deleted (work is banked in GCS)\" with title \"QHRRN watchdog: AUTO-TEARDOWN\"" \
      2>/dev/null
  fi
fi
