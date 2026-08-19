#!/bin/bash
# tools/pod.sh — THE ops tool for the one-pod campaign (2026-08-18 night).
#
# PI directive: ONE stable spot v6e-8 pod finishes the rung — no multi-pod,
# no queued resources, no on-demand. This file replaces the hunter +
# completion-watcher + merge trio (three loops, three logs, three
# half-overlapping state machines) with ONE loop, ONE log, ONE state table.
# Every value comes from tools/campaign.env; task lists from tools/r0_tasks.sh.
#
#   pod.sh supervise [hours]   the loop (single instance; adopts a live chain)
#   pod.sh status              one screen of truth (node, job, arms, bank, loop)
#   pod.sh relaunch            (re)launch the chain on the READY node (resumes)
#   pod.sh stop                controlled stop of the remote chain (verified)
#   pod.sh down                rescue + delete the node (PI standing order)
#   pod.sh log [n]             last n supervisor lines
#
# STATE TABLE (one poll = one row; the loop never acts on an UNKNOWN read):
#   node ABSENT (every zone read positively)  -> create(spot) -> up -> canary -> launch
#   node CREATING / STOPPING / REPAIRING       -> wait
#   node PREEMPTED / STOPPED / TERMINATED       -> down (delete)   [next poll hunts]
#   node READY + chain COMPLETE (sentinel/GCS)  -> down -> exit 0
#   node READY + chain RUNNING                  -> log progress
#   node READY + chain IDLE (crash/ceiling/kill)-> relaunch (<=3 per node life,
#                                                  then down + exit 3 LOUDLY)
#   node READY + SSH unreachable x6 (30 min)    -> down (sick node)  [next poll hunts]
#   describe/list FAILED (network)              -> nothing; count; keep polling
# Completion is ALSO read from GCS ($GCS/$FINAL_OBJ) before touching the node,
# so a finished chain on an unreachable node still gets torn down.
#
# What survives what: this loop lives in the Mac's process table (not app
# restarts — restart it; it adopts). tools/tpu_watchdog.sh (launchd) is the
# billing backstop and DELETES everything past runs/tpu_deadline.txt.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"
# shellcheck source=campaign.env
source tools/campaign.env
# shellcheck source=r0_tasks.sh
source tools/r0_tasks.sh          # VH RG RB RT
PY=.venv/bin/python
PROJECT=quantum-llm
LOG=${POD_LOG:-runs/pod_${POD}.log}          # overridable ONLY for the offline harness
PIDF=${POD_PIDF:-runs/pod_supervisor.pid}
POLL=${POLL:-300}
CHAIN_CMD="${CHAIN_EXTRA_ENV:+$CHAIN_EXTRA_ENV }R_TAG='$R_TAG' R_D='$R_D' R_STEPS='$R_STEPS' R0_ARMS='$ARMS' bash ${CHAIN_SCRIPT:-tools/chain_r0.sh} $VH $RG $RB $RT"

say () { echo "$(date -u +%FT%TZ) | $*" | tee -a "$LOG"; }
notify () { [ -n "${POD_QUIET:-}" ] && return 0; /usr/bin/osascript -e "display notification \"$2\" with title \"QHRRN pod: $1\"" 2>/dev/null || true; }
# bounded run: perl alarm = portable timeout (no coreutils on this Mac)
bounded () { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }
gssh () {   # gssh ZONE CMD  (bounded 150 s, gcloud chatter stripped)
  bounded 150 gcloud compute tpus tpu-vm ssh "$POD" --zone="$1" --project=$PROJECT \
    --command="$2" 2>/dev/null | grep -vE "^(SSH:|Using ssh|Warning:|Updating|Existing)"
}

# ---------- reads ----------
# node_where: prints "ZONE STATE" | "ABSENT" | "UNKNOWN". ABSENT only when EVERY
# zone was read positively and none holds the pod (never create blind).
node_where () {
  local z out rc unknown=0
  for z in $ZONES; do
    out=$(bounded 90 gcloud compute tpus tpu-vm list --zone="$z" --project=$PROJECT \
          --format="value(name,state)" 2>/dev/null); rc=$?
    if [ "$rc" -ne 0 ]; then unknown=1; continue; fi
    st=$(printf '%s\n' "$out" | awk -v p="$POD" '$1==p{print $2}' | head -1)
    if [ -n "$st" ]; then echo "$z $st"; return 0; fi
  done
  [ "$unknown" -eq 1 ] && echo UNKNOWN || echo ABSENT
}
# job_state ZONE: prints one line: COMPLETE | RUNNING <pid> | IDLE <exit> | SSHFAIL,
# then a PROGRESS line (last chain sentinel + last it/s line).
job_state () {
  local out
  out=$(gssh "$1" "cd ~/qhrrn2 2>/dev/null || { echo NOREPO; exit 0; }; \
if grep -q '$SENTINEL' runs/detached.log 2>/dev/null; then echo COMPLETE; \
elif P=\$(cat runs/detached.pid 2>/dev/null) && [ -n \"\$P\" ] && kill -0 \$P 2>/dev/null; then echo \"RUNNING \$P\"; \
else echo \"IDLE \$(cat runs/detached.exit 2>/dev/null || echo killed)\"; fi; \
echo \"PROGRESS \$(grep -E '^=== PRETRAIN|PHASE1-OK|PHASE2:|wave done|PHASE2-OK|^SKIP-|RESUMED' runs/detached.log 2>/dev/null | tail -1 | cut -c1-60) | \$(grep 'it/s' runs/detached.log 2>/dev/null | tail -1 | cut -c1-44)\"")
  if ! printf '%s' "$out" | grep -qE '^(COMPLETE|RUNNING|IDLE|NOREPO)'; then echo SSHFAIL; return; fi
  printf '%s\n' "$out"
}
gcs_complete () { gsutil -q stat "$GCS/$FINAL_OBJ" 2>/dev/null; }
still_ready () {   # ZONE -> 0 if READY/CREATING/unknown, 1 if positively dead
  local st
  st=$(bounded 90 gcloud compute tpus tpu-vm describe "$POD" --zone="$1" --project=$PROJECT \
       --format="value(state)" 2>/dev/null | grep -oE '^[A-Z]+$' | head -1)
  if [ -n "$st" ] && [ "$st" != READY ] && [ "$st" != CREATING ]; then
    say "  precheck: node is $st — abandoning this attempt"; return 1; fi
  return 0
}

# ---------- verbs (each returns 0/nonzero, logs what it did) ----------
v_down () {   # ZONE REASON
  say "DOWN $POD in $1 ($2)"
  $PY tools/dispatcher.py down --name "$POD" --zone "$1" >> "$LOG" 2>&1
  local rc=$?; say "  down rc=$rc"; return $rc
}
v_launch () { # ZONE -> 0 launched/already running
  say "LAUNCH chain in $1 (wall ${WALL}s): arms=$ARMS tag=$R_TAG d=$R_D steps=$R_STEPS"
  $PY tools/launch_detached.py --name "$POD" --zone "$1" --wall-time "$WALL" \
      --cmd "$CHAIN_CMD" >> "$LOG" 2>&1
  local rc=$?
  case $rc in 0) say "  launch: detached + verified"; return 0;;
              7) say "  launch: already running (guard)"; return 0;;
              *) say "  launch FAILED rc=$rc"; return 1;; esac
}
v_bring_up () { # ZONE (node exists) -> 0 = chain launched
  still_ready "$1" || return 1
  say "UP (bootstrap+data) in $1"
  if ! $PY tools/dispatcher.py up --name "$POD" --zone "$1" --accelerator "$ACCEL" --with-data >> "$LOG" 2>&1; then
    say "  up attempt 1 failed — retrying once"; still_ready "$1" || return 1
    $PY tools/dispatcher.py up --name "$POD" --zone "$1" --accelerator "$ACCEL" --with-data >> "$LOG" 2>&1 \
      || { say "  up failed twice"; return 1; }
  fi
  gssh "$1" "mkdir -p ~/qhrrn2/$(dirname "$CANARY_CKPT")" >/dev/null
  bounded 300 gcloud compute tpus tpu-vm scp "$CANARY_CKPT" "$POD:~/qhrrn2/$(dirname "$CANARY_CKPT")/" \
      --zone="$1" --project=$PROJECT >> "$LOG" 2>&1
  still_ready "$1" || return 1
  say "CANARY in $1"
  $PY tools/dispatcher.py canary --name "$POD" --zone "$1" >> "$LOG" 2>&1 \
      || { say "  canary FAILED"; return 1; }
  still_ready "$1" || return 1
  v_launch "$1"
}
v_hunt () {   # try every zone once; 0 = chain launched somewhere
  local z
  for z in $ZONES; do
    say "CREATE $POD ($ACCEL spot) in $z"
    if ! bounded 600 gcloud compute tpus tpu-vm create "$POD" --zone="$z" --project=$PROJECT \
         --accelerator-type="$ACCEL" --version=v6e-ubuntu-2404 --spot >> "$LOG" 2>&1; then
      say "  no capacity in $z"; continue
    fi
    say "  CREATED in $z"; notify "created" "$POD landed in $z"
    if v_bring_up "$z"; then return 0; fi
    v_down "$z" "bring-up failed — never leave an idle biller"
  done
  return 1
}
v_ensure_final () {   # ZONE — sentinel seen but the chain's own final upload may have failed: redo it (bounded) before any teardown
  if gcs_complete; then say "  final object present in GCS"; return 0; fi
  say "  final object ABSENT in GCS — re-running the chain's final rescue remotely"
  gssh "$1" "cd ~/qhrrn2 && tar czf /tmp/r0_final.tgz runs/pretrain${R_TAG}_*/ckpt_latest.pkl runs/*_p${R_TAG}* runs/wave_*.log 2>/dev/null; gsutil -q cp /tmp/r0_final.tgz $GCS/$FINAL_OBJ && echo RESCUE-REDO-OK" | tee -a "$LOG"
  if gcs_complete; then say "  final object now present"; return 0; fi
  say "  WARNING: final object still absent — down's own rescue (runs/cloud/<pod>-<stamp>.tgz) is the last copy; per-arm ckpts + partial_${ARMS%% *}.tgz remain in GCS"
  notify "final upload failed" "check runs/cloud rescue + GCS partials before analysis"; return 1
}
v_stop () {   # ZONE — kill the detached tree (setsid => pgid = pid), verify.
  # NOTE the [.] in pkill patterns: the pattern must not match THIS ssh command
  # line (the merge script's STOP killed its own shell that way, 08-18).
  say "STOP chain in $1"
  gssh "$1" "cd ~/qhrrn2 && P=\$(cat runs/detached.pid 2>/dev/null); \
if [ -n \"\$P\" ] && kill -0 \$P 2>/dev/null; then kill -TERM -- -\$P 2>/dev/null || kill -TERM \$P; sleep 6; fi; \
pkill -TERM -f 'tools/chain_r0[.]sh|tools/pretrain[.]py|tools/probe_[a-z]' 2>/dev/null; sleep 4; \
pkill -KILL -f 'tools/chain_r0[.]sh|tools/pretrain[.]py|tools/probe_[a-z]' 2>/dev/null; sleep 2; \
if [ -n \"\$P\" ] && kill -0 \$P 2>/dev/null; then echo 'STOP: sh STILL ALIVE'; else echo 'STOP: detached sh gone'; fi; \
echo \"STOP: workers left=\$(pgrep -fc 'tools/pretrain[.]py|tools/probe_[a-z]|tools/chain_r0[.]sh')\"" | tee -a "$LOG"
}

# ---------- subcommands ----------
cmd_status () {
  echo "=== $POD status $(date -u +%FT%TZ) ==="
  local nw; nw=$(node_where); echo "node: $nw"
  echo "supervisor: $( [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null && echo "RUNNING pid $(cat $PIDF)" || echo "NOT RUNNING")"
  echo "deadline: $(python3 -c "import time;d=int(open('runs/tpu_deadline.txt').read());print(f'{(d-time.time())/3600:.1f}h')" 2>/dev/null || echo none)"
  echo "GCS bank ($GCS):"; gsutil ls -l "$GCS/" 2>/dev/null | grep -vE "TOTAL" | awk '{printf "  %s  %s\n",$2,$3}' | sed "s#$GCS/##"
  case $nw in
    *" READY")
      local z=${nw%% *}
      echo "--- remote:"; job_state "$z"
      gssh "$z" "cd ~/qhrrn2 && for a in $ARMS; do d=runs/pretrain${R_TAG}_\$a; printf '  %s: %s\n' \$a \"\$([ -f \$d/.done ] && echo DONE || ([ -f \$d/ckpt_latest.pkl ] && echo partial || echo -))\"; done; for f in runs/*_p${R_TAG}*/results.jsonl; do [ -f \"\$f\" ] && echo \"  \$f: \$(wc -l < \"\$f\") rows\"; done 2>/dev/null | tail -30";;
  esac
  echo "--- last supervisor lines:"; tail -4 "$LOG" 2>/dev/null
}

cmd_supervise () {
  local HOURS=${1:-14}
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "supervisor already running (pid $(cat "$PIDF")) — one instance only"; exit 1; fi
  echo $$ > "$PIDF"
  local END RELAUNCHES=0 UNKNOWN=0 SSHFAIL=0 nw z st js
  # ONE KNOB: the loop's end time IS the watchdog deadline (runs/tpu_deadline.txt,
  # re-read every poll so extending the file extends both layers). Without it
  # the watchdog would delete the node at its deadline and this loop would
  # re-create it -> a create/delete cycle. HOURS is only the fallback.
  local END_FALLBACK=$(( $(date -u +%s) + HOURS*3600 ))   # fixed at start: no deadline file => HOURS cap, never "forever"
  end_time () { local d; d=$(tr -dc '0-9' < "${POD_DEADLINE_FILE:-runs/tpu_deadline.txt}" 2>/dev/null); [ -n "$d" ] && echo "$d" || echo "$END_FALLBACK"; }
  END=$(end_time)
  say "SUPERVISE start: pod=$POD zones='$ZONES' arms='$ARMS' until=$(date -u -r "$END" +%FT%TZ) (watchdog deadline) poll=${POLL}s pid=$$"
  while END=$(end_time) && [ "$(date -u +%s)" -lt "$END" ]; do
    if gcs_complete; then
      say "COMPLETE (GCS $FINAL_OBJ present)"; nw=$(node_where)
      case $nw in ABSENT|UNKNOWN) say "  node $nw — nothing to tear down";; *) v_down "${nw%% *}" "campaign complete";; esac
      notify "COMPLETE" "rung chain finished; node torn down"; rm -f "$PIDF"; exit 0
    fi
    nw=$(node_where)
    case $nw in
      UNKNOWN)
        UNKNOWN=$((UNKNOWN+1)); say "read UNKNOWN (x$UNKNOWN) — network? not acting"; sleep "$POLL"; continue;;
      ABSENT)
        UNKNOWN=0; say "node ABSENT everywhere — hunting"
        if v_hunt; then RELAUNCHES=0; SSHFAIL=0; sleep "$POLL"; else say "  all zones dry — sleeping 8 min"; sleep 480; fi
        continue;;
    esac
    z=${nw%% *}; st=${nw#* }; UNKNOWN=0
    case $st in
      READY)
        js=$(job_state "$z")
        case ${js%%$'\n'*} in
          COMPLETE)   say "COMPLETE (sentinel) in $z"; v_ensure_final "$z"; v_down "$z" "campaign complete"; notify "COMPLETE" "rung chain finished; node torn down"; rm -f "$PIDF"; exit 0;;
          RUNNING*)   SSHFAIL=0; say "READY $z | ${js%%$'\n'*} | $(printf '%s' "$js" | grep PROGRESS | cut -c10-)";;
          IDLE*|NOREPO)
            SSHFAIL=0; say "READY $z | chain ${js%%$'\n'*} — relaunching (attempt $((RELAUNCHES+1))/3)"
            if [ "$RELAUNCHES" -ge 3 ]; then
              say "chain died 3x on this node — tearing down and EXITING (needs eyes)"; v_down "$z" "repeated chain death"
              notify "NEEDS EYES" "chain died 3x; node deleted; supervisor exited"; rm -f "$PIDF"; exit 3
            fi
            case ${js%%$'\n'*} in NOREPO) v_bring_up "$z" || v_down "$z" "bring-up failed";; *) v_launch "$z" || v_down "$z" "launch failed";; esac
            RELAUNCHES=$((RELAUNCHES+1));;
          SSHFAIL)
            SSHFAIL=$((SSHFAIL+1)); say "READY $z but SSH unreachable (x$SSHFAIL)"
            if [ "$SSHFAIL" -ge 6 ]; then say "  node sick (30 min unreachable) — down"; v_down "$z" "ssh dead 30 min"; SSHFAIL=0; RELAUNCHES=0; fi;;
        esac;;
      CREATING|STOPPING|REPAIRING|DELETING) say "node $st in $z — waiting"; sleep 120; continue;;
      *)  say "node $st in $z (positively not READY) — down"; v_down "$z" "node $st"; RELAUNCHES=0; SSHFAIL=0;;
    esac
    sleep "$POLL"
  done
  say "SUPERVISE reached the watchdog deadline — exiting WITHOUT creating; the watchdog deletes what is up (work is banked). Extend runs/tpu_deadline.txt and restart to continue."; notify "deadline" "supervisor stopped at watchdog deadline"; rm -f "$PIDF"; exit 2
}

case ${1:-} in
  supervise) cmd_supervise "${2:-14}";;
  status)    cmd_status;;
  log)       tail -"${2:-30}" "$LOG";;
  relaunch|stop|down)
    nw=$(node_where); case $nw in ABSENT|UNKNOWN) echo "node $nw — nothing to $1"; exit 2;; esac
    z=${nw%% *}
    case $1 in relaunch) v_launch "$z";; stop) v_stop "$z";; down) v_down "$z" "manual";; esac;;
  *) sed -n '2,20p' "$0"; exit 64;;
esac
