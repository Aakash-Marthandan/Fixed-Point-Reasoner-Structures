#!/bin/bash
# FINAL PHASE Night A — SWITCH THE REMAINING ARM (A5) FROM THE v6e-16 TO A v6e-8 (PI 2026-09-05: "after the rest of the
# arms are done, switch to a v6e-8 for A5 so we speed things up efficiently; bank properly so we can resume").
# Sequence (each step verified at the source; the harness's S12b = the 16 -> 8 demotion with an arm in flight):
#   1. bank A5's in-flight state once more from its worker (the chain's own loop banks every 5 min; the trainer's
#      ckpt_latest is written every 500 steps; eval partials every 300 s);
#   2. stop the supervisor and every remote chain; delete the 16 (`pod.sh down`);
#   3. drop A5's remat record from the LIVE prefix: at 96 rows/chip the w384 arm needs ~27 GB of temporaries (< 31 GB),
#      so the 8 runs it WITHOUT remat (~30 % faster); if that is wrong, pt_run's OOM rule retries with remat (labeled);
#   4. set the accelerator ladder to v6e-8 ONLY (us-east1-d first) and restart the supervisor: it hunts the 8, brings it
#      up, and the chain (1 worker) skips the five banked arms, restores the live prefix, preflights A5 on the new shape
#      and RESUMES it (RESUMED at its last 500-step boundary; eval partials in place);
#   5. OWED BY HAND at the new node's first READY: `bash tools/plant_guard.sh 20` -> GUARD-OK.
# Preconditions checked here: the five other arms carry ARM_OK in the bucket; A5 does not; the node is READY.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source tools/campaign.env
ZONE=${SWITCH_ZONE:-us-east1-d}
A5_WORKER=${SWITCH_A5_WORKER:-2}
need="A0 A1 A2 A3 A4"; miss=""
for a in $need; do gsutil -q stat "$GCS/${a}_ARM_OK" 2>/dev/null || miss="$miss $a"; done
[ -z "$miss" ] || { echo "NOT-YET: missing ARM_OK for$miss"; exit 2; }
gsutil -q stat "$GCS/A5_ARM_OK" 2>/dev/null && { echo "A5 already ARM_OK — nothing to switch"; exit 0; }
echo "=== 1. final live bank of A5 from worker $A5_WORKER ==="
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker "$A5_WORKER" --ssh-flag="-o ConnectTimeout=25" \
  --command "cd ~/qhrrn2 && GCS=$GCS R_TAG=$R_TAG ARMS='$ARMS' bash tools/live_bank.sh once 2>&1 | tail -1; ls runs/pretrainfinalA_A5/ | tr '\n' ' '" 2>/dev/null | tail -2
echo "=== 2. stop the supervisor + remote chains; delete the 16 ==="
SPID=$(cat runs/pod_supervisor.pid 2>/dev/null); [ -n "$SPID" ] && kill "$SPID" 2>/dev/null; sleep 2; pkill -f "pod.sh supervise" 2>/dev/null; pkill -f "caffeinate -i -s -w" 2>/dev/null; sleep 1
bash tools/pod.sh stop 2>&1 | grep -E "STOP chain|workers left" | tail -2
bash tools/pod.sh down 2>&1 | tail -2
echo "=== 3. drop A5's remat record from the live prefix (the 8 fits at 96 rows/chip; the OOM rule stands as the net) ==="
gsutil -q rm "$GCS/live/runs/pretrainfinalA_A5/RETRY_REMAT.txt" 2>/dev/null && echo "RETRY_REMAT.txt removed from the live prefix" || echo "(no remat record in the live prefix)"
gsutil ls "$GCS/live/runs/pretrainfinalA_A5/" 2>/dev/null | sed 's|.*/||' | tr '\n' ' '; echo
echo "=== 4. ladder -> v6e-8 only; restart the supervisor ==="
sed -i '' 's/^ACCEL=.*/ACCEL=v6e-8/; s/^ACCEL_LIST=.*/ACCEL_LIST="v6e-8"/' tools/campaign.env
grep -E "^ACCEL|^ZONES" tools/campaign.env
rm -f runs/pod_strikes.txt runs/pod_workers.txt runs/pod_accel.txt runs/pod_supervisor.pid
DRY_SLEEP=150 nohup bash tools/pod.sh supervise 18 > /dev/null 2>&1 &
sleep 4; SPID=$(cat runs/pod_supervisor.pid 2>/dev/null || pgrep -f "pod.sh supervise" | head -1); echo "supervisor pid $SPID"
nohup caffeinate -i -s -w "$SPID" > /dev/null 2>&1 &
echo "=== 5. NEXT (by hand): at the new node's first READY -> bash tools/plant_guard.sh 20 ; then verify at the source: RESUMED A5 + eval partials restored ==="
tail -3 runs/pod_qhrrn2-pod2.log | cut -c1-160
