#!/bin/bash
# Ledger: the NODE-SIDE DELETE GUARD, promoted to a tool (2026-09-01; the
# standing per-node procedure from the 2026-08-30/31 rides — the
# Mac-independent billing backstop; the DMS is a guest shutdown, NEVER a
# billing backstop). Re-run ONCE PER NODE (every CREATED/READY) and per
# deadline extension. Two-call pattern (the v_stop self-match lesson: never
# put a pkill in the same ssh as a payload containing its pattern):
#   call 1: clear any prior guard + cancel/push the DMS to +DMS_MIN
#   call 2: plant `sleep until-deadline-epoch; gcloud ... delete` via setsid,
#           then pgrep-verify a pid (the verify-by-artifact rule).
# Zone is DISCOVERED from the campaign ZONES (nodes move zones on re-hunts —
# 2026-09-01 lesson: a hardcoded zone silently guards nothing).
#   usage: bash tools/plant_guard.sh [attempts]   (default 10, 45s apart)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source tools/campaign.env
DL=$(cat runs/tpu_deadline.txt)
DMS_MIN=${DMS_MIN:-780}
ATTEMPTS=${1:-10}
now=$(date +%s)
[ "$DL" -gt "$now" ] || { echo "GUARD-ABORT deadline $DL is in the past"; exit 1; }

find_zone () {
  for z in $ZONES; do
    st=$(perl -e 'alarm 45; exec @ARGV' -- gcloud compute tpus tpu-vm describe "$POD" \
         --zone="$z" --project=quantum-llm --format='value(state)' 2>/dev/null)
    if [ "$st" = "READY" ]; then echo "$z"; return 0; fi
  done
  return 1
}

for i in $(seq 1 "$ATTEMPTS"); do
  Z=$(find_zone) || { echo "GUARD-RETRY attempt=$i (no READY node yet)"; sleep 45; continue; }
  SSH=(compute tpus tpu-vm ssh "$POD" --zone="$Z" --project=quantum-llm --worker=0
       --ssh-flag "-o StrictHostKeyChecking=no" --ssh-flag "-o UserKnownHostsFile=/dev/null"
       --ssh-flag "-o ConnectTimeout=20")
  if perl -e 'alarm 70; exec @ARGV' -- gcloud "${SSH[@]}" \
      --command "pkill -f 'tpu-vm [d]elete $POD' 2>/dev/null; sudo shutdown -c 2>/dev/null; sudo shutdown -h +$DMS_MIN >/dev/null 2>&1; echo DMS-PUSHED-${DMS_MIN}min"; then
    echo "CALL1-OK zone=$Z attempt=$i"
    if perl -e 'alarm 70; exec @ARGV' -- gcloud "${SSH[@]}" \
        --command "SECS=\$(( $DL - \$(date +%s) )); setsid nohup bash -c \"sleep \$SECS; gcloud compute tpus tpu-vm delete $POD --zone=$Z --project=quantum-llm --quiet\" </dev/null >/tmp/deadline_guard.log 2>&1 & sleep 1; pgrep -f 'tpu-vm [d]elete' && echo GUARD-PLANTED-fires-in-\${SECS}s"; then
      echo "GUARD-OK zone=$Z (fires at epoch $DL = $(date -r "$DL" '+%F %T %Z'))"
      exit 0
    fi
    echo "CALL2-FAILED attempt=$i"
  else
    echo "CALL1-RETRY attempt=$i zone=$Z (ssh not ready)"
  fi
  sleep 45
done
echo "GUARD-PLANT-FAILED after $ATTEMPTS attempts"
exit 1
