#!/bin/bash
# Ledger: rung-0 UNATTENDED capacity retry (2026-08-14, PI away for the
# night; three preemptions in the first hour of the east1-d evening churn).
# The siege's "batch-across-spots" pattern, made safe for nobody-watching:
#
#   loop zones -> create (spot) -> bootstrap+data -> canary -> launch chain
#   -> VERIFY the chain is actually running -> exit (work proceeds, banked
#      to GCS every 5 min by the chain itself)
#
# SELF-LIMITING BY DESIGN — the failure mode that matters unattended is a
# pod that lands and then bills while idle, so EVERY failure path after a
# successful create tears the node down immediately rather than leaving it
# up for someone to notice. Layered under this: the watchdog's hard deadline
# (runs/tpu_deadline.txt) deletes everything regardless, and the chain is
# resume-complete from GCS so any teardown costs <=5 min of compute.
#
# Usage: bash tools/r0_retry_loop.sh <POD_NAME> "<R0_ARMS>" [max_hours]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"

POD=$1; ARMS=$2; MAXH=${3:-9}
ZONES="us-east1-d us-east5-b us-central1-a us-west1-c us-central2-b asia-east1-c"
PY=.venv/bin/python
LOGF=runs/r0_retry_${POD}.log
DEADLINE=$(( $(date -u +%s) + MAXH * 3600 ))
source /private/tmp/claude-501/-Users-aakash-Projects-HRRN/48879add-e490-4bd6-b0b6-070311c0b7a0/scratchpad/r0_tasks.sh

say () { echo "$(date -u +%FT%TZ) | $*" >> "$LOGF"; }

teardown () {   # $1 = zone, $2 = reason  — never leave a billing orphan
  say "TEARDOWN $POD ($2)"
  $PY tools/dispatcher.py down --name "$POD" --zone "$1" >> "$LOGF" 2>&1
}

say "retry loop start: pod=$POD arms='$ARMS' deadline=$(date -u -r $DEADLINE +%H:%MZ)"
while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do
  for Z in $ZONES; do
    [ "$(date -u +%s)" -ge "$DEADLINE" ] && break
    # skip zones where the node name already exists (stale/preempted)
    EXIST=$(gcloud compute tpus tpu-vm list --zone="$Z" --project=quantum-llm \
            --format="value(name)" 2>/dev/null | grep -c "^${POD}$")
    [ "$EXIST" -gt 0 ] && teardown "$Z" "stale node before retry"

    say "try create $POD in $Z"
    if ! gcloud compute tpus tpu-vm create "$POD" --zone="$Z" \
         --project=quantum-llm --accelerator-type=v6e-8 \
         --version=v6e-ubuntu-2404 --spot >> "$LOGF" 2>&1; then
      say "  no capacity in $Z"
      continue
    fi
    say "CREATED in $Z — bootstrapping"

    if ! $PY tools/dispatcher.py up --name "$POD" --zone "$Z" \
         --accelerator v6e-8 --with-data >> "$LOGF" 2>&1; then
      teardown "$Z" "bootstrap failed"; continue
    fi
    # canary reference ckpt (fresh pods lack it; mkdir-before-scp is LAW)
    gcloud compute tpus tpu-vm ssh "$POD" --zone="$Z" --project=quantum-llm \
      --command="mkdir -p ~/qhrrn2/runs/pretrain6_d24" >> "$LOGF" 2>&1
    gcloud compute tpus tpu-vm scp runs/pretrain6_d24/ckpt_latest.pkl \
      "$POD:~/qhrrn2/runs/pretrain6_d24/" --zone="$Z" --project=quantum-llm \
      >> "$LOGF" 2>&1
    if ! $PY tools/dispatcher.py canary --name "$POD" --zone "$Z" >> "$LOGF" 2>&1; then
      teardown "$Z" "canary FAILED"; continue
    fi
    say "canary PASS — launching chain"

    $PY tools/dispatcher.py run --name "$POD" --zone "$Z" --detach \
      --wall-time 30600 \
      --cmd "R0_ARMS='$ARMS' bash tools/chain_r0.sh $VH $RG $RB $RT" \
      >> "$LOGF" 2>&1
    sleep 60
    # VERIFY it is really running; an unverified launch is an idle biller
    if gcloud compute tpus tpu-vm ssh "$POD" --zone="$Z" --project=quantum-llm \
       --command="cd ~/qhrrn2 && kill -0 \$(cat runs/detached.pid 2>/dev/null) 2>/dev/null" \
       >> "$LOGF" 2>&1; then
      say "CHAIN RUNNING in $Z — retry loop exits (chain banks to GCS every 5 min)"
      exit 0
    fi
    teardown "$Z" "chain did not start"
  done
  say "all zones dry — sleeping 8 min"
  sleep 480
done
say "deadline reached without a durable window — giving up (nothing left up)"
exit 1
