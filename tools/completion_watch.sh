#!/bin/bash
# Ledger: unattended completion watcher (2026-08-14, PI away). Neither
# chain_r0 nor chain_sport deletes its own node — they were written for a
# session that would notice the sentinel and tear down. Unattended, a
# finished chain would bill idle until the watchdog deadline hours later.
# This closes that gap: poll the remote log, and on the completion sentinel
# (or a dead job with the sentinel already present) tear the node down.
# Results are safe by then — every chain stages to GCS before its sentinel.
#
# Deliberately does NOT tear down on a job that merely died: a crashed or
# preempted chain is resumable, and deleting on failure would race the
# retry loops. Only the SENTINEL triggers teardown here.
#
# Usage: bash tools/completion_watch.sh <POD> <ZONE> <SENTINEL> [max_hours]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"
POD=$1; ZONE=$2; SENT=$3; MAXH=${4:-12}
LOGF=runs/complwatch_${POD}.log
END=$(( $(date -u +%s) + MAXH * 3600 ))

say () { echo "$(date -u +%FT%TZ) | $*" >> "$LOGF"; }
say "watching $POD/$ZONE for '$SENT'"

while [ "$(date -u +%s)" -lt "$END" ]; do
  sleep 600   # API-load trim (2026-08-18)
  # node gone (preempted/deleted) -> nothing to watch; let the retry loop own it
  ALIVE=$(gcloud compute tpus tpu-vm list --zone="$ZONE" --project=quantum-llm \
          --format="value(name)" 2>/dev/null | grep -c "^${POD}$")
  if [ "$ALIVE" -eq 0 ]; then
    # TOLERATE absence (fixed 2026-08-15): the first version EXITED here,
    # so both pod watchers quit during a hunter's re-create window and a
    # finished chain would then have billed idle until the deadline. The
    # hunter owns recovery; this watcher just keeps waiting for the node to
    # come back and for its sentinel to appear.
    say "node absent — waiting (hunter owns recovery)"
    continue
  fi
  HIT=$(gcloud compute tpus tpu-vm ssh "$POD" --zone="$ZONE" \
        --project=quantum-llm --command="grep -c '$SENT' ~/qhrrn2/runs/detached.log 2>/dev/null" \
        2>/dev/null | tr -dc '0-9' | head -c 3)
  if [ -n "$HIT" ] && [ "$HIT" -gt 0 ] 2>/dev/null; then
    say "SENTINEL SEEN — rescuing then tearing down"
    .venv/bin/python tools/dispatcher.py run --name "$POD" --zone "$ZONE" \
      --cmd "echo rescue-pass" >> "$LOGF" 2>&1      # pulls the rescue tarball
    .venv/bin/python tools/dispatcher.py down --name "$POD" --zone "$ZONE" \
      >> "$LOGF" 2>&1
    say "TEARDOWN DONE"
    exit 0
  fi
done
say "watch window elapsed without sentinel — leaving node to the watchdog deadline"
