#!/bin/bash
# CHAMPION NIGHT 15-min direct check (2026-09-08; the PI: "launch it and keep the 15 min checks going").
# One compact line per tick from the SOURCE (node state per zone via the supervisor's log + a describe, the supervisor
# pid, the chain's last marker + trainer pace + eval progress via tools/ops_snapshot.sh on a READY node, DMS/guard/LB,
# hours to the deadline, spend estimate), plus event lines on state changes. Never reads accuracy values (VALBEST and
# MONITOR numbers are redacted). Never acts.
cd /Users/aakash/Projects/HRRN || exit 1
source tools/campaign.env
LOG=runs/pod_qhrrn2-pod2.log
prev_state=""; t_launch=$(date +%s); zone_hours=""
# scope every log read to THIS launch (the supervisor log is appended across campaigns: a stale "CREATED in" line misled the first tick)
L0=$(grep -n "SUPERVISE start: pod=$POD" "$LOG" | tail -1 | cut -d: -f1); L0=${L0:-1}
lg () { tail -n +"$L0" "$LOG" 2>/dev/null; }
redact () { sed -E 's/(VALBEST [A-Z0-9]+ [0-9]+) [0-9.]+ ([0-9]+)/\1 <val> \2/g; s/(val_t16(_ema)? )[0-9.]+/\1<val>/g; s/(exact_acc[a-z_]*"?: ?)[0-9.]+/\1<val>/g'; }
while :; do
  now=$(date -u +%H:%MZ); ist=$(TZ=Asia/Kolkata date +%H:%M)
  sup=$( pgrep -f "bash tools/pod.sh supervise" >/dev/null && echo "sup:alive" || echo "SUP:DEAD" )
  dl=$(python3 -c "import time;d=int(open('runs/tpu_deadline.txt').read());print(f'{(d-time.time())/3600:.1f}h')" 2>/dev/null)
  # node state: the supervisor's own last read (cheap) + a positive describe in the zone it names
  last=$(lg | grep -E "CREATED in|node .* in .*|ABSENT everywhere|no capacity|preempt|DOWN |LAUNCH chain|launch w|canary|CANARY|COMPLETE|GUARD|leftover|LEFTOVER|SUPERVISE|code archive" | tail -1 | cut -c1-140)
  z=$(lg | grep -oE "CREATED in [a-z0-9-]+" | tail -1 | awk '{print $3}')
  # an ADOPTED node (a supervisor restart) never prints CREATED: fall back to the supervisor's own "READY <zone> |" state line
  [ -n "$z" ] || z=$(lg | grep -oE "\| READY [a-z0-9-]+ \|" | tail -1 | awk '{print $3}')
  lg | grep -qE "DOWN $POD in $z" 2>/dev/null && [ -n "$z" ] && { dz=$(lg | grep -nE "CREATED in $z|DOWN $POD in $z" | tail -1); case $dz in *DOWN*) z="";; esac; }
  st=""; if [ -n "$z" ]; then st=$(perl -e 'alarm 60; exec @ARGV' -- gcloud compute tpus tpu-vm describe "$POD" --zone="$z" --project=quantum-llm --format='value(state)' 2>/dev/null | grep -oE '^[A-Z]+$' | head -1); fi
  state="${z:-hunting}:${st:-absent}"
  if [ "$state" != "$prev_state" ]; then echo "[$now $ist] EVENT node state $prev_state -> $state | $last"; prev_state=$state; fi
  snap=""
  if [ "$st" = READY ]; then
    snap=$(perl -e 'alarm 170; exec @ARGV' -- bash tools/ops_snapshot.sh 2>/dev/null | redact | grep -E "^ *(T|MARK|PT|EV|DMS|GUARD|DISK|LB) " | tr '\n' ' ' | cut -c1-900)
  fi
  errs=$(lg | tail -200 | grep -cE "FAILED|ABORT|WARNING|SSHFAIL|INCOMPLETE")
  echo "[$now $ist] TICK $sup deadline:$dl node:$state | ${snap:-$last} | recent-warn:$errs"
  lg | grep -q "$SENTINEL" && { echo "[$now $ist] COMPLETE sentinel seen in the supervisor log"; }
  sleep 900
done
