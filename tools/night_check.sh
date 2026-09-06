#!/bin/bash
# FINAL PHASE Night A — the 15-min DIRECT check (PI 2026-09-05: "check the pod directly at 15 min cadence; measured data,
# never proxies"). Every interval: the node state (describe), the supervisor pid, the bucket's ARM_OK markers, and on EVERY
# worker (ssh) each arm's last logged step, its pace from the metrics TIMESTAMPS (the trainer's own steps_per_sec reads 2x
# high on the field loop), the age of the last row, the live pretrain/eval processes and the chain's last marker line;
# STALE = a live trainer whose last row is older than 12 min; NO-PROC = no trainer/eval alive and no ARM-OK on that worker.
cd "$(dirname "$0")/.." || exit 1
source "${POD_ENV:-tools/campaign.env}"
INTERVAL=${NIGHT_CHECK_INTERVAL:-900}
ZONE=us-east1-d   # overwritten by the per-check discovery below
NW=${NIGHT_CHECK_WORKERS:-4}
REMOTE_ONE='cd ~/qhrrn2 && python3 -c "
import json,glob,datetime as dt,subprocess,re,os
now=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
procs=subprocess.run(\"pgrep -f \\\"tools/pretrain.py|tools/eval_sudoku_extreme.py|explosion_census.py|stall_calibration.py\\\" | wc -l\",shell=True,capture_output=True,text=True).stdout.strip()
last=subprocess.run(\"grep -E \\\"PRETRAIN-START|PRETRAIN-OK|PRETRAIN-NAN|OOM|REMAT|AMPUTAT|EVAL-OK|EVAL-FAILED|EVAL-N-BAD|CENSUS|CALIB|ARM-OK|ARM-SKIPPED|PREFLIGHT|WORKER-DONE|INCOMPLETE|COMPLETE\\\" runs/detached.log 2>/dev/null | tail -1 | cut -c1-80\",shell=True,capture_output=True,text=True).stdout.strip()
mine=set(re.findall(r\"(?:PRETRAIN-START|PRETRAIN-SKIP|ARM-OK|PRETRAIN-OK) (A[0-9])\", open(\"runs/detached.log\").read())) if os.path.exists(\"runs/detached.log\") else set()
out=[]
for p in sorted(glob.glob(\"runs/pretrainfinalA_A*/metrics.jsonl\")):
    arm=p.split(\"/\")[1].replace(\"pretrainfinalA_\",\"\")
    if mine and arm not in mine: continue
    rows=[json.loads(l) for l in open(p) if \"\\\"loss\\\"\" in l]
    if not rows: out.append(arm+\":no-rows\"); continue
    b=rows[-1]; tb=dt.datetime.fromisoformat(b[\"t\"]); age=(now-tb).total_seconds()/60; pace=\"-\"
    if len(rows)>=2:
        a=rows[-2]; d=(tb-dt.datetime.fromisoformat(a[\"t\"])).total_seconds()
        if d>0: pace=\"%.2f\"%((b[\"step\"]-a[\"step\"])/d)
    done=b[\"step\"]>=50000; flag=\"\" if done else (\" STALE\" if (age>12 and int(procs)>0) else \"\")
    out.append(\"%s:step %d%s %sit/s ce %.3f age %.0fm%s\"%(arm,b[\"step\"],\" DONE\" if done else \"\",pace,b[\"ce_in\"],age,flag))
noproc=int(procs)==0 and \"ARM-OK\" not in last and \"WORKER-DONE\" not in last and \"COMPLETE\" not in last
print(\"procs \"+procs+\" | \"+\" | \".join(out)+\" | last: \"+last+(\" | NO-PROC\" if noproc else \"\"))
"'

while true; do
  [ "${NIGHT_CHECK_ONCE:-0}" = 1 ] || sleep "$INTERVAL"
  DL=$(tr -dc '0-9' < runs/tpu_deadline.txt 2>/dev/null); left=$(( (DL - $(date -u +%s)) / 60 ))
  state=ABSENT   # discover the live zone (the ladder can land anywhere in ZONES, Mumbai included)
  for z in ${NIGHT_CHECK_ZONE:-$ZONES}; do st=$(gcloud compute tpus tpu-vm describe "$POD" --zone "$z" --format='value(state)' 2>/dev/null); [ -n "$st" ] && { state="$st"; ZONE=$z; break; }; done
  sup=$(pgrep -f "pod.sh supervise" >/dev/null && echo alive || echo DEAD)
  nok=$(gsutil ls "$GCS/" 2>/dev/null | grep -c "_ARM_OK"); nsk=$(gsutil ls "$GCS/" 2>/dev/null | grep -c "_SKIPPED")
  echo "== CHECK $(date -u +%H:%MZ) | node $state in $ZONE | supervisor $sup | ARM_OK $nok SKIPPED $nsk | cap in ${left} min"
  # ONE gcloud invocation for all workers (each gcloud ssh costs ~40 s of key/metadata setup); lines carry the host's worker index
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all --ssh-flag="-o ConnectTimeout=25" --ssh-flag="-o ServerAliveInterval=15" \
    --command "echo \"WORKER \$(hostname | sed 's/.*-w-//') \$($REMOTE_ONE)\"" 2>/dev/null | grep "^WORKER" | sort | sed 's/^WORKER /  w/' || echo "  SSH-FAIL"
  [ "${NIGHT_CHECK_ONCE:-0}" = 1 ] && exit 0
done
