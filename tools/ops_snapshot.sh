#!/bin/bash
# tools/ops_snapshot.sh — READ-ONLY measured snapshot of the live one-pod campaign (2026-09-04, sportC2 Opus ops
# handoff; the "verify at the SOURCE, never a proxy" law). ONE bounded ssh to the READY node; prints one short
# block; NEVER acts. pod.sh status's PROGRESS snippet sticks at the last marker during a pretrain — this reads the
# trainer/eval logs themselves.
#   SNAPSHOT  local clock | watchdog inventory line | supervisor pid | hours to runs/tpu_deadline.txt
#   T         node UTC clock                     PID   detached chain pid alive|DEAD (+ detached.exit)
#   MARK      last chain marker in runs/detached.log (ARM-OK / PRETRAIN-OK / EVAL-OK / RESUMED / ...)
#   PT        newest pretrain log + its mtime: last "step N ... it/s" line (two snapshots -> wall pace = dstep/dt)
#   EV        newest eval dir + its mtime: per-shard "banked partial" progress, or run.log's last line
#   DMS       the guest dead-man's poweroff epoch (re-armed +10 h at every relaunch; must stay > the next wall recycle)
#   GUARD     the node-side delete guard process (sleep-then-delete at the deadline epoch)   DISK  free space on /
#   LB        the last line of runs/live_bank.log (the 5-min live GCS bank; stale > 15 min or absent = the loop is down)
# usage: bash tools/ops_snapshot.sh   (zone from runs/tpu_status.txt, else a positive describe over campaign ZONES)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source tools/campaign.env
NOW=$(date -u +%FT%TZ)
Z=$(grep -oE "[a-z0-9-]+=${POD}:READY" runs/tpu_status.txt 2>/dev/null | head -1 | cut -d= -f1)
if [ -z "$Z" ]; then
  for z in $ZONES; do
    st=$(perl -e 'alarm 45; exec @ARGV' -- gcloud compute tpus tpu-vm describe "$POD" --zone="$z" --project=quantum-llm --format='value(state)' 2>/dev/null)
    [ "$st" = READY ] && { Z=$z; break; }
  done
fi
SP=$(cat runs/pod_supervisor.pid 2>/dev/null); SUP="NOT RUNNING"; [ -n "$SP" ] && kill -0 "$SP" 2>/dev/null && SUP="alive pid $SP"
DL=$(tr -dc '0-9' < runs/tpu_deadline.txt 2>/dev/null); LEFT="?"
[ -n "$DL" ] && LEFT="$(awk -v d="$DL" -v n="$(date +%s)" 'BEGIN{printf "%.1fh", (d-n)/3600}') to $(date -u -r "$DL" +%FT%TZ)"
echo "SNAPSHOT $NOW | watchdog: $(cat runs/tpu_status.txt 2>/dev/null || echo none) | supervisor: $SUP | deadline: $LEFT"
[ -z "$Z" ] && { echo "NODE none READY (watchdog snapshot + positive describe over campaign zones)"; exit 0; }
REMOTE='cd ~/qhrrn2 2>/dev/null || { echo "NOREPO"; exit 0; }
echo "T $(date -u +%FT%TZ)"
P=$(cat runs/detached.pid 2>/dev/null); if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then echo "PID $P alive"; else echo "PID ${P:-none} DEAD exit=$(cat runs/detached.exit 2>/dev/null || echo ?)"; fi
echo "MARK $(grep -E "ARM-OK|PRETRAIN-(START|OK|SKIP|NAN|RESTORE|OOM)|STAGEA|EVAL-(OK|SKIP|FAILED|N-BAD|SHARD)|CENSUS-(OK|SKIP|FAILED)|CALIB-(OK|SKIP|FAILED)|VALBEST|VB-FALLBACK|AMPUTAT|RIDER|WORKER-DONE|COMPLETE|INCOMPLETE|TEARDOWN|BAD-ARM|MISSING" runs/detached.log 2>/dev/null | tail -1 | cut -c1-120)"
L=$(ls -t runs/pretrain'"$R_TAG"'_*.log 2>/dev/null | head -1)
[ -n "$L" ] && echo "PT $(basename "$L" .log) @$(stat -c %y "$L" | cut -c12-19)Z: $(grep -E "^step |RESUMED|INIT-FROM|DP:|NAN|OOM" "$L" | tail -1 | cut -c1-110)"
E=$(ls -td runs/sx*/ 2>/dev/null | head -1)
if [ -n "$E" ]; then
  if ls "$E"shard_*.log >/dev/null 2>&1; then echo "EV $(basename "$E") @$(stat -c %y "$E" | cut -c12-19)Z shards: $(for f in "$E"shard_*.log; do grep "banked partial" "$f" | tail -1 | sed -E "s/.*@ ([0-9]+\/[0-9]+).*/\1/"; done | tr "\n" " ")"
  else echo "EV $(basename "$E") @$(stat -c %y "$E" | cut -c12-19)Z: $(tail -1 "$E"run.log 2>/dev/null | cut -c1-100)"; fi
fi
U=$(grep -oE "USEC=[0-9]+" /run/systemd/shutdown/scheduled 2>/dev/null | cut -d= -f2)
if [ -n "$U" ]; then echo "DMS $(date -u -d @$((U/1000000)) +%FT%TZ) (in $(( (U/1000000 - $(date +%s)) / 60 )) min)"; else echo "DMS none scheduled"; fi
G=$(pgrep -af "tpu-vm [d]elete" | head -1 | cut -c1-90); echo "GUARD ${G:-NONE PLANTED}"
echo "DISK $(df -h / | awk "NR==2{print \$4\" free\"}")"
echo "LB $(tail -1 runs/live_bank.log 2>/dev/null || echo "no live_bank.log — the 5-min live bank is NOT running on this node")"'
perl -e 'alarm 120; exec @ARGV' -- gcloud compute tpus tpu-vm ssh "$POD" --zone="$Z" --project=quantum-llm --worker=0 \
  --ssh-flag "-o StrictHostKeyChecking=no" --ssh-flag "-o UserKnownHostsFile=/dev/null" --ssh-flag "-o ConnectTimeout=20" \
  --command="$REMOTE" 2>/dev/null | grep -vE "^(SSH:|Using ssh|Warning:|Updating|Existing|External IP)" | sed "s/^/  /"
rc=${PIPESTATUS[0]}; [ "$rc" -ne 0 ] && echo "  SSH-FAILED rc=$rc (zone $Z) — an UNKNOWN read, not a state"
exit 0
