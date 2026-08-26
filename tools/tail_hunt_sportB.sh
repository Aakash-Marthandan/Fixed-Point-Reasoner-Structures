#!/bin/bash
# TAIL HUNT+SUPERVISE (local, 2026-08-26): land a small spot node (v6e-8 first,
# v6e-4 fallback — PI 15:0xZ), bring up via the proven dispatcher path, run
# tail_runbook.sh detached, resupervise through preemptions. The normal pod.sh
# supervisor stays DEAD: its chain launch on a non-16 node would hollow-merge.
set -uo pipefail
PROJECT=quantum-llm
POD=qhrrn2-pod2
ZONES="us-east1-d us-east1-c us-east5-b us-central1-a us-central2-b us-west1-c us-west4-a asia-east1-c"
ACCELS="v6e-8 v6e-4"
GCS=gs://qhrrn2-rescue/sportB
RB=/Users/aakash/Projects/HRRN/tools/tail_runbook_sportB.sh
PY=.venv/bin/python
cd /Users/aakash/Projects/HRRN
say () { echo "$(date -u +%FT%TZ) $*"; }
bounded () { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }   # pod.sh:65 verbatim — macOS has NO timeout(1); bare `timeout N` fails command-not-found and MASKS as capacity-dry (the 14:47-15:25Z incident)
STRIKES=0; MAX_STRIKES=8
node_state () { gcloud compute tpus tpu-vm describe "$POD" --zone="$1" --project=$PROJECT --format='value(state)' 2>/dev/null; }
gssh () { bounded 150 gcloud compute tpus tpu-vm ssh "$POD" --zone="$1" --project=$PROJECT --command "$2" 2>/dev/null; }

find_node () {  # any zone where POD exists -> echo zone
  local z; for z in $ZONES; do [ -n "$(node_state "$z")" ] && { echo "$z"; return 0; }; done; return 1
}
delete_node () { say "DELETE $POD in $1 ($2)"; gcloud compute tpus tpu-vm delete "$POD" --zone="$1" --project=$PROJECT --quiet >/dev/null 2>&1; }

node_acc () { gcloud compute tpus tpu-vm describe "$POD" --zone="$1" --project=$PROJECT --format='value(acceleratorType)' 2>/dev/null; }
launch_runbook () {  # ZONE -> 0 launched+verified
  local z=$1
  gssh "$z" "mkdir -p ~/qhrrn2/runs" >/dev/null
  bounded 300 gcloud compute tpus tpu-vm scp "$RB" "$POD:~/qhrrn2/tail_runbook.sh" --zone="$z" --project=$PROJECT >/dev/null 2>&1 || { say "  scp runbook FAILED"; return 1; }
  gssh "$z" "cd ~/qhrrn2 && pgrep -f 'bash tail_runbook[.]sh' >/dev/null && echo ALREADY || { SELF_POD=$POD SELF_ZONE=$z setsid nohup bash tail_runbook.sh > runs/tail_runbook.log 2>&1 < /dev/null & sleep 3; pgrep -f 'bash tail_runbook[.]sh' >/dev/null && echo LAUNCHED || echo LAUNCH-DEAD; }" | grep -E "ALREADY|LAUNCHED|LAUNCH-DEAD" | head -1
}

while true; do
  # SUCCESS check first: sentinel artifacts in GCS end the loop regardless of node state
  if gsutil -q stat "$GCS/sportB_final.tgz" 2>/dev/null && gsutil -q stat "$GCS/breadth20k.tgz" 2>/dev/null; then
    say "TAIL-SUCCESS: breadth20k.tgz + sportB_final.tgz banked"
    z=$(find_node) && { say "node still up post-completion — teardown"; delete_node "$z" "complete"; }
    say "HUNT-EXIT-COMPLETE"; exit 0
  fi
  [ "$STRIKES" -lt "$MAX_STRIKES" ] || { say "HUNT-GIVEUP after $MAX_STRIKES strikes — PI decision needed"; z=$(find_node) && delete_node "$z" "giveup - never leave an idle biller"; exit 1; }

  z=$(find_node) || z=""
  if [ -z "$z" ]; then
    landed=""
    for acc in $ACCELS; do
      for cz in $ZONES; do
        say "CREATE $POD ($acc spot) in $cz"
        if bounded 600 gcloud compute tpus tpu-vm create "$POD" --zone="$cz" --project=$PROJECT \
             --accelerator-type="$acc" --version=v6e-ubuntu-2404 --spot >/dev/null 2>&1; then
          say "  CREATED ($acc) in $cz"; landed="$cz"; break 2
        fi
        say "  no capacity in $cz ($acc)"
      done
    done
    [ -n "$landed" ] || { say "no capacity anywhere — retry in 150s"; sleep 150; continue; }
    z=$landed
    say "UP (dispatcher bootstrap+data) in $z"
    lacc=$(node_acc "$z")
    if ! bounded 1200 $PY tools/dispatcher.py up --name "$POD" --zone "$z" --accelerator "$lacc" --workers 1 --with-data >> runs/tail_dispatcher.log 2>&1; then
      say "  up attempt 1 failed — retrying once"
      [ "$(node_state "$z")" = "READY" ] || { say "  node died during up"; STRIKES=$((STRIKES+1)); delete_node "$z" "died in bring-up"; continue; }
      bounded 1200 $PY tools/dispatcher.py up --name "$POD" --zone "$z" --accelerator "$lacc" --workers 1 --with-data >> runs/tail_dispatcher.log 2>&1 \
        || { say "  up failed twice — strike"; STRIKES=$((STRIKES+1)); delete_node "$z" "bring-up failed"; sleep 150; continue; }
    fi
    r=$(launch_runbook "$z")
    say "  runbook: ${r:-no-response}"
    case "$r" in LAUNCHED|ALREADY) say "RUNBOOK-RUNNING in $z";; *) STRIKES=$((STRIKES+1)); delete_node "$z" "runbook launch failed"; sleep 150; continue;; esac
  fi

  # WATCH: node health + runbook liveness every 120s
  st=$(node_state "$z")
  if [ "$st" != "READY" ]; then
    say "NODE-$st in $z — preempted/dead; delete + re-hunt (banked partials resume)"
    delete_node "$z" "state=$st"
    sleep 30; continue
  fi
  alive=$(gssh "$z" "pgrep -f 'bash tail_runbook[.]sh' >/dev/null && echo YES || echo NO" | grep -E "YES|NO" | head -1)
  if [ "$alive" = "NO" ]; then
    done_line=$(gssh "$z" "tail -3 ~/qhrrn2/runs/tail_runbook.log 2>/dev/null" | tr '\n' ' | ')
    say "RUNBOOK-EXITED: $done_line"
    if echo "$done_line" | grep -q "TAIL-RUNBOOK-DONE\|SELF-TEARDOWN"; then say "runbook finished — verifying GCS next pass"; sleep 30; continue; fi
    if echo "$done_line" | grep -qE "GATE-FAIL|MERGE-N-BAD|REF-|protocol violation"; then
      say "HARD-STOP condition from runbook — leaving node up 30min for inspection, PI decision needed"
      gssh "$z" "tail -40 ~/qhrrn2/runs/tail_runbook.log" | sed 's/^/    /'
      sleep 1800; delete_node "$z" "hard-stop inspected window over"; exit 1
    fi
    STRIKES=$((STRIKES+1)); say "relaunching runbook (idempotent) — strike $STRIKES"
    r=$(launch_runbook "$z"); say "  relaunch: ${r:-no-response}"
    case "$r" in LAUNCHED|ALREADY) ;; *) say "  relaunch failed — recycling node"; delete_node "$z" "runbook relaunch failed"; sleep 30; continue;; esac
  fi
  sleep 120
done
