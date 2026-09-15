#!/bin/bash
# Ledger: THE ARC ERA ops — the offline harness for tools/tpu_watchdog.sh's multi-project sweep (2026-09-15). A fake gcloud on
# WATCHDOG_PATH answers `tpus tpu-vm list/delete` from a fixture; a stub osascript logs notifications; each scenario runs the watchdog
# in its own sandbox (tools/ + runs/), so nothing real is listed, deleted or notified.
#   W1 inventory: quantum-llm lists every node; anita-hunter lists ONLY qhrrn2-* nodes (tagged "anita-hunter/zone="); no delete
#   W2 past the deadline: deletes exactly our nodes (quantum-llm: all; anita-hunter: qhrrn2-* only) — never another member's
#   W3 WATCH_PROJECTS=quantum-llm: the new script's snapshot, log lines (timestamps stripped) and delete calls equal HEAD's
#   W4 a failed probe in the shared project surfaces as "anita-hunter/zone=PROBE-FAIL"
#   W5 the readers' zone regex ("[a-z0-9-]+=POD:READY", ops_snapshot.sh / dms_keeper.sh) extracts the zone from a tagged token
#   W6 prefix discipline: "qhrrn2x" and "lab-qhrrn2-node" in the shared project are not ours
# usage: bash tools/harness_watchdog_projects.sh
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d "${TMPDIR:-/tmp}/wdh.XXXXXX"); PASS=0; FAIL=0
ok () { if eval "$2"; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }
mkdir -p "$T/bin"
cat > "$T/bin/gcloud" <<'EOF'
#!/bin/bash
verb=""; name=""; zone=""; proj=""; fmt=""; prev=""
for a in "$@"; do
  case "$a" in
    --zone=*) zone=${a#--zone=} ;; --project=*) proj=${a#--project=} ;; --format=*) fmt=${a#--format=} ;;
    list|delete) verb=$a ;;
    -*) ;;
    *) [ "$prev" = delete ] && name=$a ;;
  esac
  [ "$prev" = --zone ] && zone=$a; [ "$prev" = --project ] && proj=$a; prev=$a
done
if [ "$verb" = list ]; then
  grep -q "^$proj $zone PROBE-FAIL" "$WDH_FIX" && exit 1
  awk -v p="$proj" -v z="$zone" -v f="$fmt" '$1==p && $2==z && $3!="PROBE-FAIL" { if (f ~ /state/) printf "%s\t%s\n", $3, $4; else print $3 }' "$WDH_FIX"
  exit 0
fi
[ "$verb" = delete ] && echo "DELETE $proj $zone $name" >> "$WDH_CALLS"
exit 0
EOF
printf '#!/bin/bash\necho "NOTIFY $*" >> "$WDH_NOTE"\n' > "$T/bin/osascript"; chmod +x "$T/bin/gcloud" "$T/bin/osascript"
FIXTURE="quantum-llm us-east1-d qhrrn2-pod2 READY
quantum-llm asia-south1-c sudoku-node READY
anita-hunter us-east1-d qhrrn2-arc-pod READY
anita-hunter us-east1-d labmate-tpu READY
anita-hunter us-central1-a other-team READY
anita-hunter us-west1-c qhrrn2x READY
anita-hunter us-south1-a lab-qhrrn2-node READY"
sandbox () {   # NAME SCRIPT_SRC FIXTURE DEADLINE_OFFSET_S -> runs the watchdog once in $T/NAME
  local d="$T/$1"; mkdir -p "$d/tools" "$d/runs"; cp "$2" "$d/tools/tpu_watchdog.sh"; printf '%s\n' "$3" > "$d/fix.txt"
  echo $(( $(date -u +%s) + $4 )) > "$d/runs/tpu_deadline.txt"
  WDH_FIX="$d/fix.txt" WDH_CALLS="$d/calls.txt" WDH_NOTE="$d/note.txt" WATCHDOG_PATH="$T/bin:/usr/bin:/bin" WATCHDOG_OSASCRIPT="$T/bin/osascript" \
    env ${WP:+"WATCH_PROJECTS=$WP"} bash "$d/tools/tpu_watchdog.sh"; touch "$d/calls.txt" "$d/note.txt"
}
NEWW="$ROOT/tools/tpu_watchdog.sh"
echo "W1 inventory (future deadline)"
sandbox w1 "$NEWW" "$FIXTURE" 3600
S1=$(cat "$T/w1/runs/tpu_status.txt")
ok "W1 quantum-llm lists every node"           '[ "${S1#*us-east1-d=qhrrn2-pod2:READY}" != "$S1" ] && [ "${S1#*asia-south1-c=sudoku-node:READY}" != "$S1" ]'
ok "W1 anita-hunter ours, tagged"               '[ "${S1#*anita-hunter/us-east1-d=qhrrn2-arc-pod:READY}" != "$S1" ]'
ok "W1 other members invisible (snap + log)"    '! grep -qE "labmate|other-team" "$T/w1/runs/tpu_status.txt" "$T/w1/runs/tpu_status_log.txt"'
ok "W1 no delete before the deadline"           '[ ! -s "$T/w1/calls.txt" ]'
echo "W2 past the deadline"
sandbox w2 "$NEWW" "$FIXTURE" -60
C2=$(sort "$T/w2/calls.txt" | tr '\n' ';')
ok "W2 deletes exactly our nodes"               '[ "$C2" = "DELETE anita-hunter us-east1-d qhrrn2-arc-pod;DELETE quantum-llm asia-south1-c sudoku-node;DELETE quantum-llm us-east1-d qhrrn2-pod2;" ]'
ok "W2 never another member's TPU"              '! grep -qE "labmate|other-team|qhrrn2x|lab-qhrrn2-node" "$T/w2/calls.txt"'
ok "W2 tagged DEADLINE-DELETE log line"         'grep -q "DEADLINE-DELETE anita-hunter/us-east1-d/qhrrn2-arc-pod" "$T/w2/runs/tpu_status_log.txt"'
echo "W3 quantum-llm only: new == HEAD (PATH/osascript lines of the HEAD copy pointed at the stubs, nothing else)"
git -C "$ROOT" show HEAD:tools/tpu_watchdog.sh | sed -e 's#^PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin#PATH='"$T"'/bin:/usr/bin:/bin#' -e 's#/usr/bin/osascript#'"$T"'/bin/osascript#g' > "$T/head_wd.sh"
WP=quantum-llm sandbox w3new "$NEWW" "$FIXTURE" -60
sandbox w3old "$T/head_wd.sh" "$FIXTURE" -60
strip () { sed -E 's/^[0-9T:-]+Z \| //' "$1"; }
ok "W3 snapshot identical"                      'cmp -s "$T/w3new/runs/tpu_status.txt" "$T/w3old/runs/tpu_status.txt"'
ok "W3 log lines identical (timestamps off)"    '[ "$(strip "$T/w3new/runs/tpu_status_log.txt")" = "$(strip "$T/w3old/runs/tpu_status_log.txt")" ]'
ok "W3 delete calls identical"                  '[ "$(sort "$T/w3new/calls.txt")" = "$(sort "$T/w3old/calls.txt")" ] && [ -s "$T/w3old/calls.txt" ]'
ok "W3 notifications identical"                 '[ "$(cat "$T/w3new/note.txt")" = "$(cat "$T/w3old/note.txt")" ]'
echo "W4 a failed probe in the shared project"
sandbox w4 "$NEWW" "$FIXTURE
anita-hunter us-east5-b PROBE-FAIL" 3600
ok "W4 PROBE-FAIL tagged"                       'grep -q "anita-hunter/us-east5-b=PROBE-FAIL" "$T/w4/runs/tpu_status.txt"'
echo "W5 readers"
Z5=$(grep -oE "[a-z0-9-]+=qhrrn2-arc-pod:READY" "$T/w1/runs/tpu_status.txt" | head -1 | cut -d= -f1)
ok "W5 zone extracted from the tagged token"    '[ "$Z5" = us-east1-d ]'
echo "W6 prefix discipline"
ok "W6 qhrrn2x / lab-qhrrn2-node not ours"      '! grep -qE "qhrrn2x|lab-qhrrn2-node" "$T/w1/runs/tpu_status.txt" "$T/w2/calls.txt"'
echo "harness_watchdog_projects: $PASS passed, $FAIL failed ($T)"
[ "$FAIL" -eq 0 ]
