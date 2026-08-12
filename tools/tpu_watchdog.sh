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
ZONES="us-east1-d us-east1-c us-east5-b us-west4-a asia-east1-c"
SNAP=runs/tpu_status.txt
LOG=runs/tpu_status_log.txt
mkdir -p runs
NEW=""
for z in $ZONES; do
  R=$(gcloud compute tpus tpu-vm list --zone="$z" --project=quantum-llm \
      --format="value(name,state)" 2>/dev/null | tr '\n\t' ' :')
  [ -n "$R" ] && NEW="$NEW$z=$R "
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
    /usr/bin/osascript -e "display notification \"$NEW up >=8h — check DMS \
and teardown\" with title \"QHRRN TPU watchdog ALARM\"" 2>/dev/null
  fi
fi
