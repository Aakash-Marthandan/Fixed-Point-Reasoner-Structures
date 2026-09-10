#!/bin/bash
# DEC-ARC ops heartbeat: ONE direct ssh measurement every 15 min, one compact line + ALERT lines on trouble.
# Exits on CHAIN-DECARC-COMPLETE (drives the close). Coverage: supervisor-dead, node-not-READY, past-deadline,
# DMS<deadline, stale live-bank, NAN/AMPUTATE/FAILED/ABORT markers, node-gone.
cd /Users/aakash/Projects/HRRN
PODLOG=runs/pod_qhrrn2-pod2.log
while true; do
  ts=$(date -u +%H:%M:%SZ)
  SP=$(cat runs/pod_qhrrn2-pod2_supervisor.pid 2>/dev/null)
  sup=$(kill -0 "$SP" 2>/dev/null && echo up || echo DOWN)
  dl=$(tr -dc '0-9' < runs/tpu_deadline.txt 2>/dev/null); now=$(date +%s); dlm=$(( (${dl:-now}-now)/60 ))
  snap=$(bash tools/ops_snapshot.sh 2>/dev/null)
  wd=$(echo "$snap" | grep -oE "qhrrn2-pod2:[A-Z_]+" | head -1)
  mark=$(echo "$snap" | grep -E "^  MARK" | head -1 | sed 's/^  MARK //' | cut -c1-56)
  pt=$(echo "$snap" | grep -E "^  PT" | head -1 | grep -oE "step +[0-9]+.*it/s" | tr -s ' ' | cut -c1-46)
  ev=$(echo "$snap" | grep -E "^  EV" | head -1 | sed 's/^  EV //' | cut -c1-40)
  dms=$(echo "$snap" | grep -E "^  DMS" | grep -oE "in [0-9-]+ min")
  lb=$(echo "$snap" | grep -E "^  LB" | grep -oE "rc=[0-9]+ uploaded=[0-9]+")
  load=$(echo "$snap" | grep -E "^  LOAD" | sed 's/^  LOAD //' | cut -c1-40)
  l1=$(echo "$load" | awk '{print int($1)}'); np=$(echo "$load" | grep -oE "nproc=[0-9]+" | cut -d= -f2)
  evage=$(echo "$snap" | grep -E "^  EV" | grep -oE "age=[0-9]+s" | tr -dc '0-9'); evtasks=$(echo "$snap" | grep -E "^  EV" | grep -oE "tasks=[0-9]+" | cut -d= -f2)
  suptail=$(tail -3 "$PODLOG" 2>/dev/null | grep -oE "PREEMPT[A-Z]*|node [A-Z]+ in|down|relaunch|DEMOTE[D]?|sick|unreachable|STRIKE" | tr '\n' ',' )
  echo "$ts | sup:$sup dl:${dlm}m ${wd:-<no-snap>} | ${mark:-?} | ${pt:-${ev:-idle}} | load:${load:-?} | dms${dms:-?} lb:${lb:-?} ${suptail:+sup:$suptail}"
  [ "$sup" = DOWN ] && echo "  ALERT supervisor process DOWN (restart: DRY_SLEEP=480 nohup bash tools/pod.sh supervise <h> &)"
  [ -n "$wd" ] && ! echo "$wd" | grep -q READY && echo "  ALERT node not READY: $wd"
  [ "${dlm:-1}" -lt 0 ] && echo "  ALERT past deadline knob"
  echo "$mark" | grep -qiE "NAN|AMPUTATE|FAILED|ABORT|N-BAD|SKIPPED" && echo "  ALERT chain marker: $mark"
  echo "$snap" | grep -qiE "DMS.*in -" && echo "  ALERT DMS before deadline"
  [ -n "$l1" ] && [ -n "$np" ] && [ "$l1" -gt $(( np * 3 / 2 )) ] && echo "  ALERT host load $l1 > 1.5 x nproc $np (the thrash class: something besides the chain is on the host)"
  echo "$mark" | grep -qE "VALBEST|VB-FALLBACK|PRETRAIN-OK|EVAL-OK" && [ -n "$evage" ] && [ "$evage" -gt 1800 ] && echo "  ALERT eval stall: newest eval file ${evage}s old (tasks=${evtasks:-?}) while in the battery"
  if echo "$snap $mark" | grep -q "CHAIN-DECARC-COMPLETE" || gsutil -q stat gs://qhrrn2-rescue/decarc_pilot/decarc_final.tgz 2>/dev/null; then
    echo "  PILOT-COMPLETE — verify G1 done + pull decarc_pilot/decarc_final.tgz before node loss"; break
  fi
  sleep 900
done
