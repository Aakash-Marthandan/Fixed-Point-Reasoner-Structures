#!/bin/bash
# DEC-ARC ops heartbeat: ONE direct ssh measurement every 15 min, one compact line + ALERT lines on trouble.
# Exits on the campaign's completion (drives the close). Coverage: supervisor-dead, node-not-READY, past-deadline,
# DMS<deadline, stale live-bank, NAN/AMPUTATE/FAILED/ABORT markers, host thrash, eval stall, the COST-ABORT marker.
# 2026-09-15 (the ARC era): the pod name, the bucket prefix, the final object and the sentinel come from the campaign env
# (tools/campaign.env, or POD_ENV); the Sudoku-era pods read byte-identically. HB_SENTINEL / HB_FINAL still override.
# 2026-09-16 (the two-pod night): POD_ENV selects the pod (its env, its log, its pid file); SHARE_MARK (the env's) = this pod's
# worker share is banked and its supervisor tears the node down — the heartbeat reports POD-SHARE-DONE and stops alerting on
# the (legitimately) exited supervisor. HB_ONCE=1 = one tick, then exit with a code for a waiter:
#   0 quiet · 10 at least one ALERT · 20 the campaign COMPLETE (the final object / sentinel) · 21 this pod's SHARE-DONE
# (tools/ops_watch_pods.sh runs the ticks for several pods and ends on the first non-zero code = the session's wake-up).
cd /Users/aakash/Projects/HRRN
source "${POD_ENV:-tools/campaign.env}" || { echo "ops_heartbeat: the campaign env failed to load (the identity file?)"; exit 2; }
POD=${POD:-qhrrn2-pod2}
PODLOG=runs/pod_${POD}.log
SENT_=${HB_SENTINEL:-${SENTINEL:-CHAIN-DECARC-COMPLETE}}
FINAL_=${HB_FINAL:-${GCS:-gs://qhrrn2-rescue/decarc_pilot}/${FINAL_OBJ:-decarc_final.tgz}}
while true; do
  ts=$(date -u +%H:%M:%SZ)
  SP=$(cat "runs/pod_${POD}_supervisor.pid" 2>/dev/null)
  sup=$(kill -0 "$SP" 2>/dev/null && echo up || echo DOWN)
  dl=$(tr -dc '0-9' < runs/tpu_deadline.txt 2>/dev/null); now=$(date +%s); dlm=$(( (${dl:-now}-now)/60 ))
  snap=$(POD_ENV=${POD_ENV:-tools/campaign.env} bash tools/ops_snapshot.sh 2>/dev/null)
  wd=$(echo "$snap" | grep -oE "${POD}:[A-Z_]+" | head -1)
  mark=$(echo "$snap" | grep -E "^  MARK" | head -1 | sed 's/^  MARK //' | cut -c1-56)
  pt=$(echo "$snap" | grep -E "^  PT" | head -1 | grep -oE "step +[0-9]+.*it/s" | tr -s ' ' | cut -c1-46)
  ev=$(echo "$snap" | grep -E "^  EV" | head -1 | sed 's/^  EV //' | cut -c1-40)
  dms=$(echo "$snap" | grep -E "^  DMS" | grep -oE "in [0-9-]+ min")
  lb=$(echo "$snap" | grep -E "^  LB" | grep -oE "rc=[0-9]+ uploaded=[0-9]+")
  load=$(echo "$snap" | grep -E "^  LOAD" | sed 's/^  LOAD //' | cut -c1-40)
  l1=$(echo "$load" | awk '{print int($1)}'); np=$(echo "$load" | grep -oE "nproc=[0-9]+" | cut -d= -f2)
  evage=$(echo "$snap" | grep -E "^  EV" | grep -oE "age=[0-9]+s" | tr -dc '0-9'); evtasks=$(echo "$snap" | grep -E "^  EV" | grep -oE "tasks=[0-9]+" | cut -d= -f2)
  suptail=$(tail -3 "$PODLOG" 2>/dev/null | grep -oE "PREEMPT[A-Z]*|node [A-Z]+ in|down|relaunch|DEMOTE[D]?|sick|unreachable|STRIKE|SHARE-DONE|COST-ABORT" | tr '\n' ',' )
  done_=""
  if echo "$snap $mark" | grep -q "$SENT_" || gsutil -q stat "$FINAL_" 2>/dev/null; then done_=complete
  elif [ -n "${SHARE_MARK:-}" ] && gsutil -q stat "${GCS}/${SHARE_MARK}" 2>/dev/null; then done_=share; fi
  echo "$ts [$POD] | sup:$sup dl:${dlm}m ${wd:-<no-node>} | ${mark:-?} | ${pt:-idle} | ev:${ev:-none} | load:${load:-?} | dms${dms:-?} lb:${lb:-?} ${suptail:+sup:$suptail}"
  n_alert=0
  alert () { echo "  ALERT $*"; n_alert=$((n_alert + 1)); }
  if [ -z "$done_" ]; then   # a finished pod's supervisor has exited by design: no supervisor / node alerts for it
    [ "$sup" = DOWN ] && alert "supervisor process DOWN (restart: POD_ENV=${POD_ENV:-tools/campaign.env} DRY_SLEEP=480 nohup bash tools/pod.sh supervise <h> &)"
    [ -n "$wd" ] && ! echo "$wd" | grep -q READY && alert "node not READY: $wd"
  fi
  [ "${dlm:-1}" -lt 0 ] && alert "past deadline knob"
  echo "$mark" | grep -qiE "NAN|AMPUTATE|FAILED|ABORT|N-BAD|SKIPPED|INCOMPLETE|NO-PORT|NO-GRID|DEADLOCK|EXHAUSTED|ORPHANS|NO-RESUME-STATE|VALSET-MISSING|VAL-N-BAD|XROW-FAILED|XROW-N-BAD|NO-GRIDS|ROW-N-BAD|TEST-FAILED|TEST-N-BAD|NO-GRID|VALBEST-MISMATCH" && alert "chain marker: $mark"
  echo "$snap" | grep -qiE "DMS.*in -" && alert "DMS before deadline"
  [ -n "$l1" ] && [ -n "$np" ] && [ "$l1" -gt $(( np * 3 / 2 )) ] && alert "host load $l1 > 1.5 x nproc $np (the thrash class: something besides the chain is on the host)"
  echo "$mark" | grep -qE "VALBEST|VB-FALLBACK|PRETRAIN-OK|EVAL-OK|COST-OK" && [ -n "$evage" ] && [ "$evage" -gt 1800 ] && alert "eval stall: newest eval file ${evage}s old (tasks=${evtasks:-?}) while in the battery"
  gsutil -q stat "${GCS:-gs://qhrrn2-rescue/decarc_pilot}/CHAIN-COST-ABORT" 2>/dev/null && alert "CHAIN-COST-ABORT marker present: the chain refused a battery (projected wall > DA_COST_BUDGET_H); the supervisors tear down and exit 4; the protocol decision is the PI's"
  if [ "$done_" = complete ]; then echo "  CAMPAIGN-COMPLETE ($SENT_ / the final object) — run the close"; [ -n "${HB_ONCE:-}" ] && exit 20; break; fi
  if [ "$done_" = share ]; then echo "  POD-SHARE-DONE ($POD's worker share is banked; its supervisor tears the node down; the other pod finalizes)"; [ -n "${HB_ONCE:-}" ] && exit 21; break; fi
  if [ -n "${HB_ONCE:-}" ]; then [ "$n_alert" -gt 0 ] && exit 10; exit 0; fi
  sleep 900
done
