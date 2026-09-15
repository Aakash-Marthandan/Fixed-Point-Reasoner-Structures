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
PATH=${WATCHDOG_PATH:-/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin}   # WATCHDOG_PATH / WATCHDOG_OSASCRIPT: the offline harness only (tools/harness_watchdog_projects.sh)
OSA=${WATCHDOG_OSASCRIPT:-/usr/bin/osascript}
cd "$(dirname "$0")/.." || exit 1
# asia-south1-* added 2026-08-28 (autonomous-mode fix): the Mumbai zones joined
# the campaign rotation on 08-27 but the watchdog never swept them — it reported
# "none" while a Mumbai v6e-16 ran, blinding the spend meter AND leaving the
# deadline-enforcement backstop with a coverage hole. Keep this list a SUPERSET
# of campaign.env ZONES whenever zones are added.
ZONES="us-east1-d us-east1-c us-east5-b us-central1-a us-central2-b us-west1-c us-west4-a asia-east1-c asia-south1-a asia-south1-b asia-south1-c"
# THE ARC ERA (2026-09-15; the PI: "all further ARC project content and compute goes there"; "the monitoring tools ... safe as we
# shouldn't intrude others' work in the shared project funding and compute"). Project ids and the sharing policy come from the
# git-ignored tools/.gcp_local.env (tools/gcp_local.sh); missing -> an ALARM and exit (the backstop is BLIND, never guessing).
# Every project in WATCH_PROJECTS is swept. The Sudoku era's own project keeps its behaviour byte-for-byte: every node listed,
# every node deleted past the deadline, the snapshot token "zone=name:state". In a SHARED project (SHARED_PROJECTS) ONLY nodes
# whose name starts with OWN_PREFIX are listed, reported or deleted — another member's TPU is never touched — and its tokens read
# "project/zone=name:state" (the readers' "[a-z0-9-]+=POD:READY" still extracts the zone). ARC_ZONES = every zone offering v6e
# in the ARC project (probed 2026-09-15); keep it a SUPERSET of the ARC env's ZONES.
SNAP=runs/tpu_status.txt
LOG=runs/tpu_status_log.txt
mkdir -p runs
if ! source tools/gcp_local.sh; then
  echo "$(date -u +%FT%TZ) | BLIND: tools/.gcp_local.env missing or incomplete — no project swept, no deadline enforced" >> "$LOG"
  "$OSA" -e "display notification \"tools/.gcp_local.env missing — the TPU backstop is BLIND\" \
    with title \"QHRRN TPU watchdog ALARM\"" 2>/dev/null
  exit 1
fi
WATCH_PROJECTS=${WATCH_PROJECTS:-"$SUDOKU_PROJECT ${ARC_PROJECT:-}"}
ARC_ZONES="us-east1-d us-east5-a us-east5-b us-central1-a us-central1-b us-central1-c us-west1-c us-south1-a europe-west4-a asia-south1-c"
zones_of () { if is_shared_project "$1"; then echo "$ARC_ZONES"; else echo "$ZONES"; fi; }
mine () { if is_shared_project "$1"; then awk -v p="$OWN_PREFIX" 'index($1, p) == 1'; else cat; fi; }   # stdin rows "name[\tstate]"
tagz () { if is_shared_project "$1"; then echo "$1/$2"; else echo "$2"; fi; }
NEW=""
for proj in $WATCH_PROJECTS; do
for z in $(zones_of "$proj"); do
  # perl alarm = portable timeout (no coreutils on this Mac); a wedged gcloud
  # connection must never freeze the watchdog (2026-08-14: one hung poll
  # blinded layer 1 for 45 min — the 07-29 bounded-call law applies here too).
  RAW=$(perl -e 'alarm shift; exec @ARGV' 120 \
      gcloud compute tpus tpu-vm list --zone="$z" --project="$proj" \
      --format="value(name,state)" 2>/dev/null)
  RC=$?
  if [ "$RC" -ne 0 ]; then
    # A failed probe is NOT an empty zone — surface blindness, never mask it.
    NEW="$NEW$(tagz "$proj" "$z")=PROBE-FAIL "
  else
    R=$(printf '%s' "$RAW" | mine "$proj" | tr '\n\t' ' :')
    R=${R% }
    [ -n "$R" ] && NEW="$NEW$(tagz "$proj" "$z")=$R "
  fi
done
done
NEW=$(echo "$NEW" | sed 's/ *$//')
OLD=$(cat "$SNAP" 2>/dev/null || echo "")
echo "$(date -u +%FT%TZ) | ${NEW:-none}" >> "$LOG"
if [ "$NEW" != "$OLD" ]; then
  echo "$NEW" > "$SNAP"
  "$OSA" -e "display notification \"${NEW:-all zones clear}\" \
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
      "$OSA" -e "display notification \"$NEW up >=8h — check DMS \
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
    for proj in $WATCH_PROJECTS; do
    for z in $(zones_of "$proj"); do
      for n in $(gcloud compute tpus tpu-vm list --zone="$z" \
                 --project="$proj" --format="value(name)" 2>/dev/null | mine "$proj"); do
        echo "$(date -u +%FT%TZ) | DEADLINE-DELETE $(tagz "$proj" "$z")/$n" >> "$LOG"
        gcloud compute tpus tpu-vm delete "$n" --zone="$z" \
          --project="$proj" --quiet >/dev/null 2>&1 &
      done
    done
    done
    wait
    "$OSA" -e "display notification \"deadline passed — all TPUs \
deleted (work is banked in GCS)\" with title \"QHRRN watchdog: AUTO-TEARDOWN\"" \
      2>/dev/null
  fi
fi
