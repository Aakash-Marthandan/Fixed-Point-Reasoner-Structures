#!/bin/bash
# Ledger: THE ARC ERA ops — the offline harness for the SHARED-PROJECT safety layer (2026-09-15; the PI: spot only in the shared
# project; "the monitoring tools ... safe as we shouldn't intrude others' work in the shared project funding and compute"; "git
# ignore the credential related details"). Fictional projects only (own-proj = the Sudoku era's own, lab-shared = a shared lab
# project); a fake gcloud on PATH answers from a fixture and records every call; a stub osascript logs notifications; every tool
# runs in a sandbox. Nothing real is listed, created, deleted or notified.
#   W1 watchdog inventory: own-proj lists every node; lab-shared lists ONLY qhrrn2-* nodes (tagged "lab-shared/zone="); no delete
#   W2 past the deadline: deletes exactly our nodes — never another member's
#   W3 own project only: the new watchdog equals the pre-ARC watchdog (1a94adf; its project literal read from the local identity
#      file at run time, so this harness names no real project) on snapshot, log lines, delete calls and notifications
#   W4 a failed probe in the shared project surfaces as "lab-shared/zone=PROBE-FAIL"
#   W5 the readers' zone regex ("[a-z0-9-]+=POD:READY") extracts the zone from a tagged token
#   W6 prefix discipline: "qhrrn2x" and "lab-qhrrn2-node" are not ours
#   W7 the local identity file missing: the watchdog ALARMS "BLIND", exits 1 and calls gcloud zero times
#   G1-G5 pod.sh: refuses a non-prefixed POD, a missing POD_LABELS, PROJECT != QHRRN_GCP_PROJECT and a missing identity file in the
#      shared project (exit 2, zero gcloud calls); passes with a prefixed, labeled POD (the `log` verb, no cloud)
#   D1-D5 dispatcher.py: refuses a non-prefixed name, --on-demand in a spot-only project and a missing policy file (exit 2, zero
#      gcloud calls); a dry-run create in the shared project refuses without POD_LABELS and prints --spot + --labels with them;
#      status lists only our nodes in the shared project
# usage: bash tools/harness_shared_project.sh
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d "${TMPDIR:-/tmp}/sph.XXXXXX"); PASS=0; FAIL=0
ok () { if eval "$2"; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }
mkdir -p "$T/bin"
cat > "$T/bin/gcloud" <<'EOF'
#!/bin/bash
echo "$*" >> "${WDH_ALL:-/dev/null}"
verb=""; name=""; zone=""; proj=""; fmt=""; prev=""
for a in "$@"; do
  case "$a" in
    --zone=*) zone=${a#--zone=} ;; --project=*) proj=${a#--project=} ;; --format=*) fmt=${a#--format=} ;;
    list|delete|describe|get-value) verb=$a ;;
    -*) ;;
    *) { [ "$prev" = delete ] || [ "$prev" = describe ]; } && name=$a ;;
  esac
  [ "$prev" = --zone ] && zone=$a; [ "$prev" = --project ] && proj=$a; prev=$a
done
case "$verb" in
  get-value) [ "$prev" = account ] && echo "${FAKE_ACCOUNT:-}"; [ "$prev" = project ] && echo "${FAKE_PROJECT:-}"; exit 0 ;;
  list)
    grep -q "^$proj $zone PROBE-FAIL" "$WDH_FIX" && exit 1
    awk -v p="$proj" -v z="$zone" -v f="$fmt" '$1==p && $2==z && $3!="PROBE-FAIL" { if (f ~ /state/) printf "%s\t%s\n", $3, $4; else print $3 }' "$WDH_FIX"
    exit 0 ;;
  describe) awk -v p="$proj" -v z="$zone" -v n="$name" '$1==p && $2==z && $3==n { print $4 }' "$WDH_FIX"; exit 0 ;;
  delete) echo "DELETE $proj $zone $name" >> "$WDH_CALLS"; exit 0 ;;
esac
exit 0
EOF
printf '#!/bin/bash\necho "NOTIFY $*" >> "$WDH_NOTE"\n' > "$T/bin/osascript"; chmod +x "$T/bin/gcloud" "$T/bin/osascript"
cat > "$T/local.env" <<'EOF'
QHRRN_GCP_ACCOUNT=pi@example.com
SUDOKU_PROJECT=own-proj
ARC_PROJECT=lab-shared
ARC_GCLOUD_CONFIG=arc-cfg
SHARED_PROJECTS="lab-shared"
SPOT_ONLY_PROJECTS="lab-shared"
OWN_PREFIX=qhrrn2-
EOF
FIXTURE="own-proj us-east1-d qhrrn2-pod2 READY
own-proj asia-south1-c sudoku-node READY
lab-shared us-east1-d qhrrn2-arc-pod READY
lab-shared us-east1-d labmate-tpu READY
lab-shared us-central1-a other-team READY
lab-shared us-west1-c qhrrn2x READY
lab-shared us-south1-a lab-qhrrn2-node READY"
sandbox () {   # NAME SCRIPT_SRC FIXTURE DEADLINE_OFFSET_S [nolocal] -> runs the watchdog once in $T/NAME
  local d="$T/$1"; mkdir -p "$d/tools" "$d/runs"; cp "$2" "$d/tools/tpu_watchdog.sh"; cp "$ROOT/tools/gcp_local.sh" "$d/tools/"
  [ "${5:-}" = nolocal ] || cp "$T/local.env" "$d/tools/.gcp_local.env"
  printf '%s\n' "$3" > "$d/fix.txt"; echo $(( $(date -u +%s) + $4 )) > "$d/runs/tpu_deadline.txt"
  WDH_FIX="$d/fix.txt" WDH_CALLS="$d/calls.txt" WDH_NOTE="$d/note.txt" WDH_ALL="$d/all.txt" WATCHDOG_PATH="$T/bin:/usr/bin:/bin" \
    WATCHDOG_OSASCRIPT="$T/bin/osascript" env -u GCP_LOCAL_ENV ${WP:+"WATCH_PROJECTS=$WP"} bash "$d/tools/tpu_watchdog.sh"; echo $? > "$d/rc.txt"
  touch "$d/calls.txt" "$d/note.txt" "$d/all.txt"
}
NEWW="$ROOT/tools/tpu_watchdog.sh"
echo "W1 inventory (future deadline)"
sandbox w1 "$NEWW" "$FIXTURE" 3600
S1=$(cat "$T/w1/runs/tpu_status.txt")
ok "W1 own project lists every node"            '[ "${S1#*us-east1-d=qhrrn2-pod2:READY}" != "$S1" ] && [ "${S1#*asia-south1-c=sudoku-node:READY}" != "$S1" ]'
ok "W1 shared project: ours, tagged"            '[ "${S1#*lab-shared/us-east1-d=qhrrn2-arc-pod:READY}" != "$S1" ]'
ok "W1 other members invisible (snap + log)"    '! grep -qE "labmate|other-team" "$T/w1/runs/tpu_status.txt" "$T/w1/runs/tpu_status_log.txt"'
ok "W1 no delete before the deadline"           '[ ! -s "$T/w1/calls.txt" ]'
echo "W2 past the deadline"
sandbox w2 "$NEWW" "$FIXTURE" -60
C2=$(sort "$T/w2/calls.txt" | tr '\n' ';')
ok "W2 deletes exactly our nodes"               '[ "$C2" = "DELETE lab-shared us-east1-d qhrrn2-arc-pod;DELETE own-proj asia-south1-c sudoku-node;DELETE own-proj us-east1-d qhrrn2-pod2;" ]'
ok "W2 never another member's TPU"              '! grep -qE "labmate|other-team|qhrrn2x|lab-qhrrn2-node" "$T/w2/calls.txt"'
ok "W2 tagged DEADLINE-DELETE log line"         'grep -q "DEADLINE-DELETE lab-shared/us-east1-d/qhrrn2-arc-pod" "$T/w2/runs/tpu_status_log.txt"'
echo "W3 own project only: new == the pre-ARC watchdog (1a94adf)"
REAL_SUDOKU=$(bash -c 'source "$1/tools/gcp_local.sh" >/dev/null 2>&1 && echo "$SUDOKU_PROJECT"' _ "$ROOT")
if [ -n "$REAL_SUDOKU" ]; then
  git -C "$ROOT" show 1a94adf:tools/tpu_watchdog.sh | sed -e "s#--project=$REAL_SUDOKU#--project=own-proj#g" \
    -e 's#^PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin#PATH='"$T"'/bin:/usr/bin:/bin#' -e 's#/usr/bin/osascript#'"$T"'/bin/osascript#g' > "$T/old_wd.sh"
  WP=own-proj sandbox w3new "$NEWW" "$FIXTURE" -60
  sandbox w3old "$T/old_wd.sh" "$FIXTURE" -60
  strip () { sed -E 's/^[0-9T:-]+Z \| //' "$1"; }
  ok "W3 snapshot identical"                    'cmp -s "$T/w3new/runs/tpu_status.txt" "$T/w3old/runs/tpu_status.txt"'
  ok "W3 log lines identical (timestamps off)"  '[ "$(strip "$T/w3new/runs/tpu_status_log.txt")" = "$(strip "$T/w3old/runs/tpu_status_log.txt")" ]'
  ok "W3 delete calls identical"                '[ "$(sort "$T/w3new/calls.txt")" = "$(sort "$T/w3old/calls.txt")" ] && [ -s "$T/w3old/calls.txt" ]'
  ok "W3 notifications identical"               '[ "$(cat "$T/w3new/note.txt")" = "$(cat "$T/w3old/note.txt")" ]'
else
  ok "W3 needs the real tools/.gcp_local.env"   'false'
fi
echo "W4 a failed probe in the shared project"
sandbox w4 "$NEWW" "$FIXTURE
lab-shared us-east5-b PROBE-FAIL" 3600
ok "W4 PROBE-FAIL tagged"                       'grep -q "lab-shared/us-east5-b=PROBE-FAIL" "$T/w4/runs/tpu_status.txt"'
echo "W5 readers"
Z5=$(grep -oE "[a-z0-9-]+=qhrrn2-arc-pod:READY" "$T/w1/runs/tpu_status.txt" | head -1 | cut -d= -f1)
ok "W5 zone extracted from the tagged token"    '[ "$Z5" = us-east1-d ]'
echo "W6 prefix discipline"
ok "W6 qhrrn2x / lab-qhrrn2-node not ours"      '! grep -qE "qhrrn2x|lab-qhrrn2-node" "$T/w1/runs/tpu_status.txt" "$T/w2/calls.txt"'
echo "W7 the identity file missing"
sandbox w7 "$NEWW" "$FIXTURE" -60 nolocal
ok "W7 exit 1, BLIND logged and notified"       '[ "$(cat "$T/w7/rc.txt")" = 1 ] && grep -q BLIND "$T/w7/runs/tpu_status_log.txt" && grep -q BLIND "$T/w7/note.txt"'
ok "W7 zero gcloud calls, zero deletes"         '[ ! -s "$T/w7/all.txt" ] && [ ! -s "$T/w7/calls.txt" ]'
echo "G pod.sh guards (the log verb; sandboxed state files)"
podrun () {   # NAME ENV_BODY [LOCAL_ENV] -> rc + output of `pod.sh log 1`
  local d="$T/$1"; mkdir -p "$d"; printf '%s\n' "$2" > "$d/pod.env"; echo "LOG-OK" > "$d/pod.log"
  ( cd "$ROOT" && PATH="$T/bin:$PATH" WDH_ALL="$d/all.txt" POD_ENV="$d/pod.env" POD_LOG="$d/pod.log" POD_PIDF="$d/pod.pid" \
      GCP_LOCAL_ENV="${3:-$T/local.env}" env -u PROJECT -u QHRRN_GCP_PROJECT -u POD_LABELS bash tools/pod.sh log 1 > "$d/out.txt" 2>&1; echo $? > "$d/rc.txt" )
  touch "$d/all.txt"
}
BASE=$(cat "$ROOT/tools/campaign_decarc.env")   # the registered DEC-ARC env (every knob pod.sh reads); each case overrides project, POD, labels
podrun g1 "$BASE; PROJECT=lab-shared; POD=labmate-pod; POD_LABELS=program=qhrrn2"
ok "G1 non-prefixed POD refused (exit 2)"       '[ "$(cat "$T/g1/rc.txt")" = 2 ] && grep -q "lacks the qhrrn2- prefix" "$T/g1/out.txt" && [ ! -s "$T/g1/all.txt" ]'
podrun g2 "$BASE; PROJECT=lab-shared; POD=qhrrn2-arc-pod"
ok "G2 missing POD_LABELS refused (exit 2)"     '[ "$(cat "$T/g2/rc.txt")" = 2 ] && grep -q "POD_LABELS unset" "$T/g2/out.txt" && [ ! -s "$T/g2/all.txt" ]'
podrun g3 "$BASE; PROJECT=lab-shared; export QHRRN_GCP_PROJECT=own-proj; POD=qhrrn2-arc-pod; POD_LABELS=program=qhrrn2"
ok "G3 PROJECT != QHRRN_GCP_PROJECT refused"    '[ "$(cat "$T/g3/rc.txt")" = 2 ] && grep -q "differ" "$T/g3/out.txt"'
podrun g4 "$BASE; PROJECT=lab-shared; POD=qhrrn2-arc-pod; POD_LABELS=program=qhrrn2" /nonexistent
ok "G4 identity file missing refused (exit 2)"  '[ "$(cat "$T/g4/rc.txt")" = 2 ] && grep -q "gcp_local.env missing" "$T/g4/out.txt"'
podrun g5 "$BASE; PROJECT=lab-shared; POD=qhrrn2-arc-pod; POD_LABELS=program=qhrrn2"
ok "G5 prefixed + labeled POD passes the guards" '[ "$(cat "$T/g5/rc.txt")" = 0 ] && grep -q "LOG-OK" "$T/g5/out.txt" && [ ! -s "$T/g5/all.txt" ]'
podrun g6 "$BASE; POD=qhrrn2-pod2"
ok "G6 own project (PROJECT unset) passes"      '[ "$(cat "$T/g6/rc.txt")" = 0 ] && grep -q "LOG-OK" "$T/g6/out.txt"'
echo "D dispatcher.py guards"
disp () {   # NAME LOCAL_ENV ARGS... -> rc + output
  local d="$T/$1" le=$2; shift 2; mkdir -p "$d"
  ( cd "$ROOT" && PATH="$T/bin:$PATH" WDH_ALL="$d/all.txt" WDH_FIX="$T/w1/fix.txt" GCP_LOCAL_ENV="$le" QHRRN_GCP_ACCOUNT=pi@example.com \
      QHRRN_GCP_PROJECT=lab-shared FAKE_ACCOUNT=pi@example.com FAKE_PROJECT=lab-shared .venv/bin/python tools/dispatcher.py "$@" > "$d/out.txt" 2>&1
    echo $? > "$d/rc.txt" ); touch "$d/all.txt"
}
disp d1 "$T/local.env" status --name labmate-tpu --zone us-east1-d
ok "D1 non-prefixed name refused (exit 2)"      '[ "$(cat "$T/d1/rc.txt")" = 2 ] && grep -q "lacks the qhrrn2- prefix" "$T/d1/out.txt" && [ ! -s "$T/d1/all.txt" ]'
disp d2 "$T/local.env" up --name qhrrn2-arc-pod --zone us-east1-d --on-demand --dry-run
ok "D2 --on-demand refused (spot only)"         '[ "$(cat "$T/d2/rc.txt")" = 2 ] && grep -q "spot-only" "$T/d2/out.txt" && [ ! -s "$T/d2/all.txt" ]'
disp d3 /nonexistent status --name qhrrn2-arc-pod --zone us-east1-d
ok "D3 policy file missing refused (exit 2)"    '[ "$(cat "$T/d3/rc.txt")" = 2 ] && grep -q "gcp_local.env missing" "$T/d3/out.txt" && [ ! -s "$T/d3/all.txt" ]'
disp d4 "$T/local.env" up --name qhrrn2-arc-pod --zone us-east1-d --accelerator v6e-8 --dry-run
ok "D4 dry create without POD_LABELS refused"   '[ "$(cat "$T/d4/rc.txt")" = 2 ] && grep -q "without POD_LABELS" "$T/d4/out.txt" && ! grep -q "tpu-vm create" "$T/d4/out.txt"'
( export POD_LABELS=program=qhrrn2,owner=pi; disp d4b "$T/local.env" up --name qhrrn2-arc-pod --zone us-east1-d --accelerator v6e-8 --dry-run )
ok "D4b dry create prints --spot and --labels"  'grep -E "tpu-vm create qhrrn2-arc-pod .*--spot --labels=program=qhrrn2,owner=pi" -q "$T/d4b/out.txt"'
disp d5 "$T/local.env" status --name qhrrn2-arc-pod --zone us-east1-d
ok "D5 status lists only our nodes"             'grep -q "fleet: 1 VM(s) in us-east1-d (ours only: shared project): qhrrn2-arc-pod" "$T/d5/out.txt" && ! grep -q labmate "$T/d5/out.txt"'
disp d6 "$T/local.env" up --name "" --zone us-east1-d --dry-run
ok "D6 an empty node name refused (exit 2)"     '[ "$(cat "$T/d6/rc.txt")" = 2 ] && grep -q "empty node name" "$T/d6/out.txt" && [ ! -s "$T/d6/all.txt" ]'
echo "Z the loader and the ARC env sourced from zsh (the operator's shell) as well as bash"
for sh_ in bash zsh; do
  command -v "$sh_" >/dev/null || { ok "Z $sh_ present" 'false'; continue; }
  out=$(cd "$ROOT" && GCP_LOCAL_ENV="$T/local.env" "$sh_" -c 'source tools/campaign_decarc_arc.env && echo "P=$PROJECT C=$CLOUDSDK_ACTIVE_CONFIG_NAME Q=$QHRRN_GCP_PROJECT POD=$POD S=$(is_shared_project "$PROJECT" && echo y) O=$(is_spot_only_project "$PROJECT" && echo y) N=$(is_own_name "$POD" && echo y)"' 2>&1)
  ok "Z $sh_: the ARC env loads fully"          '[ "$out" = "P=lab-shared C=arc-cfg Q=lab-shared POD=qhrrn2-arc-pod S=y O=y N=y" ]'
  out=$(cd "$ROOT" && GCP_LOCAL_ENV=/nonexistent "$sh_" -c 'source tools/campaign_decarc_arc.env; echo "rc=$? P=${PROJECT:-unset}"' 2>/dev/null | tail -1)
  ok "Z $sh_: a missing identity file stops the env" '[ "${out#rc=}" != "0 P=unset" ] && [ "${out##* }" = "P=unset" ]'
done
echo "harness_shared_project: $PASS passed, $FAIL failed ($T)"
[ "$FAIL" -eq 0 ]
