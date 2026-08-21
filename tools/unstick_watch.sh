#!/bin/bash
# Ledger: SELF-HEALING STALL WATCH (2026-08-20). Two real incidents (08-19
# 05:32Z, 08-20 05:32Z): a preemption DURING bring-up (UP/CANARY/LAUNCH)
# leaves pod.sh's supervisor blocked in a bounded-but-long SSH to a dead node
# (up to ~30 min) with NO new log lines — the edge monitor goes silent and the
# hunt stalls. This watch reads the supervisor log's LAST line; when it is a
# bring-up step unchanged past its normal duration AND the node is POSITIVELY
# not READY/CREATING (describe; unknown never acts), it kills the supervisor's
# hung CHILD (dispatcher up/canary or the gcloud ssh/scp) — the step then
# fails fast, the loop tears down and resumes hunting. It NEVER kills the
# supervisor itself and never acts blind. Also alerts once if the supervisor
# process dies while the campaign is incomplete.
# Usage (under a session Monitor): bash tools/unstick_watch.sh
set -u
cd "$(dirname "$0")/.." || exit 1
source tools/campaign.env   # POD
LOG=runs/pod_${POD}.log
PIDF=runs/pod_supervisor.pid
prev=""; same=0; dead_alerted=0
while true; do
  sleep 60
  last=$(tail -1 "$LOG" 2>/dev/null)
  # supervisor liveness (alert once; completion = normal exit)
  SUP=$(cat "$PIDF" 2>/dev/null)
  if [ -n "$SUP" ] && ! kill -0 "$SUP" 2>/dev/null; then
    if ! tail -5 "$LOG" | grep -qE "campaign complete|COMPLETE \(GCS|deadline"; then
      [ "$dead_alerted" -eq 0 ] && { echo "SUPERVISOR DEAD with campaign incomplete — restart per HANDOFF §1"; dead_alerted=1; }
    fi
    prev="$last"; continue
  fi
  dead_alerted=0
  step=""; zone=""
  case "$last" in
    *"UP (bootstrap+data) in "*)  step=UP;     zone=${last##*in }; lim=15;;
    *"CANARY in "*)               step=CANARY; zone=${last##*in }; lim=10;;
    *"LAUNCH chain in "*)         step=LAUNCH; zone=${last##* in }; zone=${zone%% *}; lim=10;;
    *) prev="$last"; same=0; continue;;
  esac
  if [ "$last" = "$prev" ]; then same=$((same+1)); else same=0; fi
  prev="$last"
  if [ "$same" -ge "$lim" ]; then
    st=$(perl -e 'alarm shift; exec @ARGV' 60 gcloud compute tpus tpu-vm describe "$POD" \
         --zone="$zone" --project=quantum-llm --format="value(state)" 2>&1)
    pos=$(printf '%s' "$st" | grep -oE '^(READY|CREATING|PREEMPTED|STOPPING|STOPPED|TERMINATED|REPAIRING)$|NOT_FOUND' | head -1)
    case "$pos" in
      READY|CREATING|"") continue;;   # healthy-or-unknown: never act
      *)  # positively dead/gone AND the loop is blocked on it -> surgical unstick
        KIDS=$(ps -eo pid,ppid,args | awk -v s="$SUP" '$2==s && (/dispatcher.py (up|canary)/ || /launch_detached/){print $1}')
        SSHK=$(pgrep -f "tpu-vm (ssh|scp).*${POD}")
        [ -n "$KIDS$SSHK" ] || continue
        kill $KIDS $SSHK 2>/dev/null
        echo "UNSTUCK: $step in $zone blocked ${same}m on a $pos node — killed hung child (loop will fail the step, tear down, re-hunt)"
        same=0;;
    esac
  fi
done
