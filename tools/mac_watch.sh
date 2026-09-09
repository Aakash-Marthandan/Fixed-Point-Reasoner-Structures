#!/bin/bash
# CHAMPION NIGHT — Mac-side 15-min direct check (2026-09-08; PI: "maintain the 15 minute checks on both the pod and
# the mac with regular measurements over proxies"). One compact line per tick from the SOURCE: the field-checkpoint
# harness chain (retain -> draws) by PID + its row/partial files, the PI-approved full-aug vote queue by PID + its
# progress line (accuracy values redacted — ops reads none), Mac disk + battery. Never acts.
cd /Users/aakash/Projects/HRRN || exit 1
H=runs/field_ckpts/harness
redact () { sed -E 's/(exact|token-acc|token_acc|exact_acc)[ =:]+[0-9.]+/\1 <val>/g'; }
rows () { [ -f "$H/arc_out/$1/rows.jsonl" ] && wc -l < "$H/arc_out/$1/rows.jsonl" | tr -d ' ' || echo 0; }
prtl () { [ -f "$H/arc_out/$1/rows_partial.jsonl" ] && wc -l < "$H/arc_out/$1/rows_partial.jsonl" | tr -d ' ' || echo 0; }
while :; do
  now=$(date -u +%H:%MZ); ist=$(TZ=Asia/Kolkata date +%H:%M)
  # field harness chain
  fh=$(pgrep -f "run_field_arc[.]py" | head -1)
  if [ -n "$fh" ]; then mode=$(ps -o args= -p "$fh" | grep -oE '\-\-mode [a-z]+' | awk '{print $2}'); fhs="fh:$mode(pid $fh)"; else fhs="fh:idle"; fi
  # approved vote queue
  vq=$(pgrep -f "mac_vote_queue[.]sh" | head -1)
  vp=$(pgrep -f "eval_arc_theirs[.]py" | head -1)
  if [ -n "$vp" ]; then vs="vote:RUNNING(pid $vp) $(tail -1 "$H/arc_logs/vote40_bf16.log" 2>/dev/null | redact | cut -c1-70)"
  elif [ -n "$vq" ]; then vs="vote:queued(waits on draws)"
  else vs="vote:$(grep -q VOTE-DONE /private/tmp/claude-501/-Users-aakash-Projects-HRRN/1a2de5df-ce75-4c6d-a519-55c0fb5da0b1/scratchpad/mac_vote_queue.log 2>/dev/null && echo DONE || echo absent)"; fi
  # DIRECT progress from the active field-harness log (rows.jsonl is written only at the end; the "N/419" line is the truth)
  flog=$(ls -t "$H"/arc_logs/*.log 2>/dev/null | head -1)
  fprog=$(grep -E "[0-9]+/[0-9]+" "$flog" 2>/dev/null | tail -1 | sed -E 's/[0-9]+\.[0-9]+/<v>/g' | cut -c1-46)
  disk=$(df -h /Users/aakash | tail -1 | awk '{print $4}')
  batt=$(pmset -g batt | tail -1 | grep -oE '[0-9]+%[^;]*;[^;]*' | head -1 | tr -d ' ')
  echo "[$now $ist] MAC $fhs progress:[${fprog:-none} in $(basename ${flog:-none} .log)] | $vs | disk:$disk batt:$batt"
  sleep 900
done
