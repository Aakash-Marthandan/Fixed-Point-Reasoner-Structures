#!/bin/bash
# tools/ops_watch_pods.sh — the session's wake-up driver for a multi-pod campaign (2026-09-16, the two-pod DEC-ARC night).
# Runs tools/ops_heartbeat.sh in single-tick mode (HB_ONCE=1) for every pod env given, every HB_EVERY seconds (default 900), for at
# most HB_TICKS ticks (default 4 = the hourly heartbeat), appending every line to runs/ops_watch_pods.log and printing it. Ends
# EARLY with the largest heartbeat code of a tick: 10 an ALERT · 20 the campaign COMPLETE · 21 a pod's SHARE-DONE — so a
# background task running it wakes the session exactly on an event or on the hour. Re-arm after handling; drop a finished pod's
# env from the list (its code 21 would end every tick). READ-ONLY: it never acts on a node, a supervisor or a marker.
#   usage: bash tools/ops_watch_pods.sh tools/campaign_decarc_arc_p0.env tools/campaign_decarc_arc_p1.env
cd "$(dirname "$0")/.." || exit 1
LOG=runs/ops_watch_pods.log
[ $# -ge 1 ] || { echo "usage: ops_watch_pods.sh ENV [ENV ...]"; exit 64; }
TICKS=${HB_TICKS:-4}
for t in $(seq 1 "$TICKS"); do
  worst=0
  for env in "$@"; do
    out=$(HB_ONCE=1 POD_ENV="$env" bash tools/ops_heartbeat.sh 2>&1); rc=$?
    printf '%s\n' "$out" | tee -a "$LOG"
    [ "$rc" -gt "$worst" ] && worst=$rc
  done
  # the ARC project's spend record (tools/arc_spend.py, refreshed by the launchd watchdog every 15 min): quoted once per tick
  [ -f runs/arc_spend_log.txt ] && echo "  SPEND $(tail -1 runs/arc_spend_log.txt)" | tee -a "$LOG"
  [ "$worst" -ne 0 ] && { echo "WATCH-END code=$worst tick=$t/$TICKS $(date -u +%FT%TZ)" | tee -a "$LOG"; exit "$worst"; }
  [ "$t" -lt "$TICKS" ] && sleep "${HB_EVERY:-900}"
done
echo "WATCH-END code=0 (the hourly heartbeat) $(date -u +%FT%TZ)" | tee -a "$LOG"
exit 0
