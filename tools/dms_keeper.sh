#!/bin/bash
# DMS keeper (2026-09-08): the supervisor's 8.5 h wall recycle re-arms the guest dead-man's switch to +600 min at every
# relaunch (dispatcher run), which puts it BEFORE the campaign deadline and would kill the node-side guard's sleep if a
# recycle ever failed (the 09-04 rule: the DMS must sit past the deadline). Every 30 min: read the scheduled shutdown on
# worker 0; if it is earlier than deadline + 30 min, push it to deadline + 60 min. Never touches the guard or the chain.
cd /Users/aakash/Projects/HRRN || exit 1
source "${POD_ENV:-tools/campaign.env}"   # POD_ENV (2026-09-16, the two-pod night): one keeper per pod
source tools/gcp_local.sh || exit 2   # project ids: the git-ignored tools/.gcp_local.env (2026-09-15)
while :; do
  DL=$(cat runs/tpu_deadline.txt 2>/dev/null); now=$(date +%s)
  Z=$(grep -oE "[a-z0-9-]+=${POD}:READY" runs/tpu_status.txt 2>/dev/null | head -1 | cut -d= -f1)
  if [ -z "$Z" ]; then for z in $ZONES; do   # 2026-09-16: a positive describe over the campaign zones (the watchdog's token omits the zone for a second node in one zone)
    [ "$(perl -e 'alarm 45; exec @ARGV' -- gcloud compute tpus tpu-vm describe "$POD" --zone="$z" --project=${PROJECT:-$SUDOKU_PROJECT} --format='value(state)' 2>/dev/null)" = READY ] && { Z=$z; break; }
  done; fi
  [ -n "$Z" ] || { echo "$(date -u +%FT%TZ) [$POD] no READY node in the campaign zones; idle"; sleep 1800; continue; }
  if [ -n "$DL" ] && [ "$DL" -gt "$now" ]; then
    # read scheduled shutdown on ALL workers; the earliest one governs (the wall recycle re-arms every worker, not just w0)
    usecs=$(perl -e 'alarm 90; exec @ARGV' -- gcloud compute tpus tpu-vm ssh "$POD" --zone "$Z" --project ${PROJECT:-$SUDOKU_PROJECT} --worker all --ssh-flag "-o StrictHostKeyChecking=no" --ssh-flag "-o UserKnownHostsFile=/dev/null" --ssh-flag "-o ConnectTimeout=20" --command "grep -oE 'USEC=[0-9]+' /run/systemd/shutdown/scheduled 2>/dev/null | cut -d= -f2" 2>/dev/null | grep -oE '[0-9]{10,}')
    sched=$(echo "$usecs" | awk '{s=int($1/1000000); if(m==""||s<m)m=s} END{print (m==""?0:m)}')
    if [ "$sched" -lt $((DL + 1800)) ]; then
      dms=$(( (DL - now) / 60 + 60 ))
      out=$(perl -e 'alarm 120; exec @ARGV' -- gcloud compute tpus tpu-vm ssh "$POD" --zone "$Z" --project ${PROJECT:-$SUDOKU_PROJECT} --worker all --ssh-flag "-o StrictHostKeyChecking=no" --ssh-flag "-o UserKnownHostsFile=/dev/null" --ssh-flag "-o ConnectTimeout=20" --command "sudo shutdown -c 2>/dev/null; sudo shutdown -h +$dms >/dev/null 2>&1 && echo PUSHED-\$(hostname|sed 's/.*-w-//')" 2>/dev/null | grep -oE 'PUSHED-[0-9]' | tr '\n' ' ')
      echo "$(date -u +%FT%TZ) DMS was $(date -u -r "$sched" +%FT%TZ 2>/dev/null || echo unknown) < deadline+30min -> pushed +$dms min: ${out:-FAILED}"
    else
      echo "$(date -u +%FT%TZ) DMS ok at $(date -u -r "$sched" +%FT%TZ)"
    fi
  else
    echo "$(date -u +%FT%TZ) no live deadline; idle"
  fi
  sleep 1800
done
