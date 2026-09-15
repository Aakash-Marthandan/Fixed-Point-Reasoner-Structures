#!/bin/bash
# THE ARC PROJECT CANARY (2026-09-15): the first node in the shared lab project, driven step by step with markers.
# Creates ONE labeled spot v6e-8 named qhrrn2-arc-canary (US zones first, Mumbai last); tries the proven image, falls back to the
# project's listed v6e runtime; bootstraps through the dispatcher with the GCS code archive (node->bucket read) and banks the venv
# (node->bucket write); the canary fit; node-side checks (image, python, gsutil/gcloud present, 8 chips, bucket rw+delete, self
# describe, scopes, egress); then the node DELETES ITSELF (the node-side guard's exact command = the backstop's permission test);
# the Mac verifies absence by a positive describe and deletes it itself if the node could not.
set -uo pipefail
cd /Users/aakash/Projects/HRRN || exit 1
export PATH=$PWD/.venv/bin:$PATH
source tools/gcp_local.sh || { echo "CANARY-ABORT identity file"; exit 2; }
export CLOUDSDK_ACTIVE_CONFIG_NAME=$ARC_GCLOUD_CONFIG QHRRN_GCP_PROJECT=$ARC_PROJECT
export POD_LABELS="program=qhrrn2,owner=aakash,purpose=arc-canary"
NAME=qhrrn2-arc-canary; P=$ARC_PROJECT
ZONES="us-east1-d us-east5-b us-east5-a us-central1-a us-central1-b us-central1-c us-west1-c us-south1-a asia-south1-c"
RT=v6e-ubuntu-2404
say () { echo "$(date -u +%FT%TZ) $*"; }
b () { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }
# the deadline knob: never shorter than now + 3 h while the canary lives (the launchd watchdog deletes our nodes past it)
now=$(date +%s); dl=$(tr -dc '0-9' < runs/tpu_deadline.txt 2>/dev/null); want=$((now + 3*3600))
if [ -z "$dl" ] || [ "$dl" -lt "$want" ]; then echo "$want" > runs/tpu_deadline.txt; say "DEADLINE-KNOB set to $(date -u -r $want +%FT%TZ) (was $(date -u -r ${dl:-0} +%FT%TZ))"; fi
# the code archive (identity files and backups excluded) in the ARC bucket
SHA=$(git rev-parse --short HEAD); ARCH="gs://qhrrn2-arc/decarc/ops/code_${SHA}dirty$(date -u +%H%M)canary.tgz"
COPYFILE_DISABLE=1 tar czf /tmp/qhrrn2_canary_code.tgz --exclude='tools/.gcp_local.env' --exclude='tools/.gcp_identity' --exclude='*.bak' --exclude='__pycache__' src tests tools requirements.txt pyproject.toml \
  && gsutil -q cp /tmp/qhrrn2_canary_code.tgz "$ARCH" && export QHRRN_CODE_TGZ="$ARCH" && say "CODE-ARCHIVE $ARCH" || { say "CANARY-ABORT code archive upload"; exit 2; }
# create: zones in order; a version error switches the image once; capacity errors move on
Z=""
for z in $ZONES; do
  for attempt in 1 2; do
    say "CREATE $NAME v6e-8 spot runtime=$RT zone=$z"
    out=$(b 600 gcloud compute tpus tpu-vm create "$NAME" --zone="$z" --project="$P" --accelerator-type=v6e-8 --version="$RT" --spot --labels="$POD_LABELS" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then Z=$z; say "CREATED zone=$z runtime=$RT"; break 2; fi
    say "CREATE-FAILED rc=$rc: $(printf '%s' "$out" | grep -vE '^(Create request issued|Waiting for operation)' | tr '\n' ' ' | cut -c1-400)"
    st=$(b 90 gcloud compute tpus tpu-vm describe "$NAME" --zone="$z" --project="$P" --format='value(state)' 2>/dev/null)
    if [ "$st" = READY ]; then Z=$z; say "LEFTOVER READY record adopted in $z"; break 2; fi
    [ -n "$st" ] && { say "LEFTOVER $st record in $z — deleting"; b 300 gcloud compute tpus tpu-vm delete "$NAME" --zone="$z" --project="$P" --quiet >/dev/null 2>&1; sleep 20; }
    if [ "$RT" = v6e-ubuntu-2404 ] && printf '%s' "$out" | grep -qiE 'version|runtime'; then RT=v2-alpha-tpuv6e; say "IMAGE-SWITCH -> $RT (the project does not offer v6e-ubuntu-2404)"; continue; fi
    break
  done
done
[ -n "$Z" ] || { say "CANARY-NO-CAPACITY in every zone (nothing created)"; exit 3; }
export QHRRN_TPU_RUNTIME=$RT
cleanup_from_mac () { b 300 python tools/dispatcher.py down --name "$NAME" --zone "$Z" >/dev/null 2>&1; st=$(b 90 gcloud compute tpus tpu-vm describe "$NAME" --zone="$Z" --project="$P" --format='value(state)' 2>/dev/null); say "MAC-DELETE rc done; describe now: ${st:-ABSENT}"; }
T0=$(date +%s)
say "UP (bootstrap + data) zone=$Z"
b 2400 python tools/dispatcher.py up --name "$NAME" --zone "$Z" --accelerator v6e-8 --with-data > /tmp/qhrrn2_canary_up.log 2>&1; rc=$?
grep -E "identity guard|code from GCS|falling back|VENV-|bootstrap|READY|Error|error:|FAILED|Traceback" /tmp/qhrrn2_canary_up.log | grep -v "account=" | tail -12 | sed 's/^/  /'
say "UP rc=$rc after $(( $(date +%s) - T0 ))s"
[ $rc -eq 0 ] || { say "CANARY-UP-FAILED"; tail -30 /tmp/qhrrn2_canary_up.log | grep -v "account=" | sed 's/^/  | /'; cleanup_from_mac; exit 4; }
SSH=(compute tpus tpu-vm ssh "$NAME" --zone="$Z" --project="$P" --ssh-flag "-o StrictHostKeyChecking=no" --ssh-flag "-o UserKnownHostsFile=/dev/null" --ssh-flag "-o ConnectTimeout=20")
b 120 gcloud "${SSH[@]}" --command "mkdir -p ~/qhrrn2/runs/pretrain6_d24" >/dev/null 2>&1
b 300 gcloud compute tpus tpu-vm scp runs/pretrain6_d24/ckpt_latest.pkl "$NAME:~/qhrrn2/runs/pretrain6_d24/" --zone="$Z" --project="$P" >/dev/null 2>&1 && say "CKPT-SCP-OK"
say "CANARY FIT"
b 1900 python tools/dispatcher.py canary --name "$NAME" --zone "$Z" > /tmp/qhrrn2_canary_fit.log 2>&1; rc=$?
grep -E ">>> canary|CANARY-PASS|Error|Traceback" /tmp/qhrrn2_canary_fit.log | tail -4 | sed 's/^/  /'
say "CANARY rc=$rc"
say "NODE CHECKS"
b 600 gcloud "${SSH[@]}" --command '
echo "OS $(lsb_release -ds 2>/dev/null || grep PRETTY /etc/os-release)"; echo "SYSPY $(python3 --version 2>&1)"
echo "TOOLS gsutil=$(command -v gsutil || echo MISSING) gcloud=$(command -v gcloud || echo MISSING)"
cd ~/qhrrn2 && echo "VENVPY $(.venv/bin/python --version 2>&1)" && echo "JAX $(.venv/bin/python -c "import jax; d=jax.devices(); print(len(d), d[0])" 2>&1 | tail -1)"
T=gs://qhrrn2-arc/decarc/ops/canary_rw_$(hostname).txt
echo canary | gsutil -q cp - "$T" && echo "GCS-WRITE-OK" || echo "GCS-WRITE-FAILED"
gsutil -q stat "$T" && echo "GCS-STAT-OK" || echo "GCS-STAT-FAILED"
gsutil -q rm "$T" && echo "GCS-DELETE-OK" || echo "GCS-DELETE-FAILED"
gsutil ls gs://qhrrn2-arc/decarc/ops/ 2>&1 | grep -c "venv_" | sed "s/^/VENV-OBJECTS /"
echo "SCOPES $(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes | tr "\n" " ")"
echo "EGRESS pypi $(curl -s -o /dev/null -w "%{http_code}" --max-time 15 https://pypi.org/simple/)"
echo "NPROC $(nproc) MEM $(free -g | awk "/Mem:/{print \$2}") GB DISK $(df -h / | awk "NR==2{print \$4}") free"
' 2>/dev/null | grep -E "^(OS|SYSPY|TOOLS|VENVPY|JAX|GCS-|VENV-OBJECTS|SCOPES|EGRESS|NPROC)" | sed 's/^/  /'
say "SELF-DESCRIBE + SELF-DELETE from the node (the guard's command)"
b 300 gcloud "${SSH[@]}" --command "gcloud compute tpus tpu-vm describe $NAME --zone=$Z --project=$P --format=\"value(state)\" 2>&1 | tail -1 | sed \"s/^/NODE-DESCRIBE /\"; gcloud compute tpus tpu-vm delete $NAME --zone=$Z --project=$P --quiet --async 2>&1 | tail -2 | sed \"s/^/NODE-DELETE /\"" 2>/dev/null | grep -E "^NODE-" | sed 's/^/  /'
for i in $(seq 1 20); do
  st=$(b 90 gcloud compute tpus tpu-vm describe "$NAME" --zone="$Z" --project="$P" --format='value(state)' 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$st" | grep -qiE 'not.?found|NOT_FOUND'; then say "SELF-DELETE-VERIFIED: the node is ABSENT (positive NOT_FOUND) after $((i*20))s"; break; fi
  say "  node state: $(printf '%s' "$st" | tail -1 | cut -c1-80)"; sleep 20
done
st=$(b 90 gcloud compute tpus tpu-vm describe "$NAME" --zone="$Z" --project="$P" --format='value(state)' 2>/dev/null)
[ -n "$st" ] && { say "SELF-DELETE-NOT-EFFECTIVE (state $st) — deleting from the Mac"; cleanup_from_mac; }
say "FLEET (ours) in $Z: [$(b 90 gcloud compute tpus tpu-vm list --zone="$Z" --project="$P" --format='value(name,state)' 2>/dev/null | awk -v p="$OWN_PREFIX" 'index($1,p)==1' | tr '\n' ' ')]"
say "CANARY-DONE runtime=$RT zone=$Z wall=$(( $(date +%s) - T0 ))s"
