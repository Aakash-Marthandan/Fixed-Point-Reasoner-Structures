#!/bin/bash
# tools/harness_pod_state.sh — the offline harness of pod.sh's supervise STATE TABLE (2026-09-15; the DEC-ARC night's
# CHAIN-COST-ABORT marker, the build item the pilot's close left open: "pod.sh's state table learns CHAIN-COST-ABORT").
# No cloud, no node, no side effects outside a sandbox: a fake gcloud on PATH answers every node read from a fixture (one
# READY node of our name in the first zone, or none), answers ssh with the job_state line "IDLE 3" (= the chain exited 3
# on the marker) and records every delete; a fake gsutil answers stat from a sandbox dir; POD_ENV / POD_LOG / POD_PIDF /
# POD_DEADLINE_FILE keep every state file in the sandbox; POLL=1, MAX_RELAUNCH=1, POD_QUIET=1; the env under test is the
# ARC env's shape (a shared, spot-only project with a prefixed, labeled node).
# Scenarios:
#   P1 the marker present, the node READY + IDLE  -> COST-ABORT logged, ONE delete, NO launch, exit 4
#   P2 no marker, the node READY + IDLE            -> the pre-existing path: a relaunch attempted (LAUNCH chain), then
#                                                     'chain died 1x' (MAX_RELAUNCH=1) -> delete, exit 3 — the edit is inert
#   P3 the marker present, the node ABSENT         -> COST-ABORT logged, no delete, exit 4
# 2026-09-16 (the two-pod night):
#   P4 this pod's SHARE_DONE marker, node READY    -> SHARE-DONE logged, ONE delete, no launch, exit 0
#   P5 the OTHER pod's SHARE marker only            -> ignored: the pre-existing relaunch path (P2's), and the relaunch
#                                                     re-plants the node guard + DMS (v_guard after every relaunch)
#   P6 node ABSENT, every zone dry                  -> the create carries --version=$TPU_RUNTIME --spot --labels; the supervisor
#                                                     stops at the deadline (exit 2) without creating anything else
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d /tmp/hps.XXXXXX)
trap 'rm -rf "$T"' EXIT
PASS=0; FAIL=0
ok () { if eval "$2"; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }
mkdir -p "$T/bin"
cat > "$T/bin/gcloud" <<'EOF'
#!/bin/bash
echo "$*" >> "${HPS_ALL:-/dev/null}"
verb=""; name=""; zone=""; fmt=""; prev=""
for a in "$@"; do
  case "$a" in
    --zone=*) zone=${a#--zone=} ;; --format=*) fmt=${a#--format=} ;;
    list|delete|describe|get-value|ssh|scp|create) [ -z "$verb" ] && verb=$a ;;
    -*) ;;
    *) { [ "$prev" = delete ] || [ "$prev" = describe ] || [ "$prev" = ssh ] || [ "$prev" = scp ]; } && name=$a ;;
  esac
  [ "$prev" = --zone ] && zone=$a; prev=$a
done
have () { [ -f "$HPS_FIX" ] && grep -q "^$zone $1\$" "$HPS_FIX"; }
case "$verb" in
  get-value) [ "$prev" = account ] && echo "${FAKE_ACCOUNT:-}"; [ "$prev" = project ] && echo "${FAKE_PROJECT:-}"; exit 0 ;;
  list)  if have "${HPS_NODE}"; then printf '%s\tREADY\n' "$HPS_NODE"; fi; exit 0 ;;
  describe) if have "$name"; then case "$fmt" in *ipAddress*) echo "10.0.0.1";; *) echo READY;; esac; fi; exit 0 ;;
  ssh)   printf 'IDLE 3\nPROGRESS ARM-OK D0 | \n'; exit 0 ;;   # the chain exited 3 (the marker); a neutral progress line
  scp)   exit 0 ;;
  create) echo "CREATE $*" >> "$HPS_CALLS"; echo "ERROR: There is no more capacity in the zone" >&2; exit 1 ;;
  delete) echo "DELETE $zone $name" >> "$HPS_CALLS"; exit 0 ;;
esac
exit 0
EOF
cat > "$T/bin/gsutil" <<'EOF'
#!/bin/bash
args=(); for a in "$@"; do [ "$a" = "-q" ] || [ "$a" = "-m" ] || args+=("$a"); done
case "${args[0]:-}" in
  stat) [ -f "$HPS_GCS/$(basename "${args[1]}")" ] ;;
  cp|ls) exit 0 ;;
  *) exit 0 ;;
esac
EOF
printf '#!/bin/bash\nexit 0\n' > "$T/bin/osascript"; chmod +x "$T/bin/gcloud" "$T/bin/gsutil" "$T/bin/osascript"
cat > "$T/local.env" <<'EOF'
QHRRN_GCP_ACCOUNT=pi@example.com
SUDOKU_PROJECT=own-proj
ARC_PROJECT=lab-shared
ARC_GCLOUD_CONFIG=arc-cfg
SHARED_PROJECTS="lab-shared"
SPOT_ONLY_PROJECTS="lab-shared"
OWN_PREFIX=qhrrn2-
EOF
# the env under test: the ARC env's shape (project, node name, labels, one zone) over the registered DEC-ARC knobs
mkenv () {
  cat > "$1" <<EOF
source tools/gcp_local.sh || return 2 2>/dev/null || exit 2
export CLOUDSDK_ACTIVE_CONFIG_NAME=\$ARC_GCLOUD_CONFIG
export QHRRN_GCP_PROJECT=\$ARC_PROJECT
PROJECT=\$ARC_PROJECT
POD=qhrrn2-arc-pod
POD_LABELS=program=qhrrn2,owner=pi,purpose=arc
ZONES="us-east1-d"
ACCEL=v6e-8
ACCEL_LIST="v6e-8"
BIG_MAX_STRIKES=99
R_TAG=decarc
R_D=160
R_STEPS=30000
ARMS="D0 D1 D2 N0"
CHAIN_SCRIPT=tools/chain_decarc.sh
CHAIN_EXTRA_ENV="SELF_TEARDOWN=0 RIDER=0"
TPU_RUNTIME=v2-alpha-tpuv6e
SHARE_MARK=SHARE_DONE_w0
WALL=30600
SENTINEL=CHAIN-DECARC-COMPLETE
GCS=gs://qhrrn2-arc/decarc
FINAL_OBJ=decarc_final.tgz
CANARY_CKPT=runs/pretrain6_d24/ckpt_latest.pkl
EOF
}
scenario () {   # NAME NODE_PRESENT(1/0) MARKER_PRESENT(1/0) [SHARE_MARKER_NAME] [DEADLINE_OFFSET_S]
  local d="$T/$1"; mkdir -p "$d/gcs"; mkenv "$d/pod.env"; : > "$d/fix.txt"; : > "$d/calls.txt"; : > "$d/all.txt"
  [ "$2" = 1 ] && echo "us-east1-d qhrrn2-arc-pod" > "$d/fix.txt"
  [ "$3" = 1 ] && echo "D0 2026-09-15T00:00:00Z" > "$d/gcs/CHAIN-COST-ABORT"
  [ -n "${4:-}" ] && echo "w0 2026-09-16T00:00:00Z" > "$d/gcs/$4"
  echo $(( $(date -u +%s) + ${5:-3600} )) > "$d/deadline.txt"
  ( cd "$ROOT" && PATH="$T/bin:$PATH" HPS_ALL="$d/all.txt" HPS_FIX="$d/fix.txt" HPS_CALLS="$d/calls.txt" HPS_GCS="$d/gcs" HPS_NODE=qhrrn2-arc-pod \
      GCP_LOCAL_ENV="$T/local.env" FAKE_ACCOUNT=pi@example.com FAKE_PROJECT=lab-shared \
      POD_ENV="$d/pod.env" POD_LOG="$d/pod.log" POD_PIDF="$d/pod.pid" POD_WFILE="$d/workers.txt" POD_AFILE="$d/accel.txt" POD_SFILE="$d/strikes.txt" \
      POD_DEADLINE_FILE="$d/deadline.txt" POLL=1 MAX_RELAUNCH=1 POD_QUIET=1 DRY_SLEEP=1 \
      env -u PROJECT -u QHRRN_GCP_PROJECT -u POD_LABELS perl -e 'alarm 240; exec @ARGV' -- bash tools/pod.sh supervise 1 > "$d/out.txt" 2>&1; echo $? > "$d/rc.txt" )
}
echo "P1 the COST-ABORT marker present, the node READY + IDLE"
scenario p1 1 1
ok "P1 exit 4"                                   '[ "$(cat "$T/p1/rc.txt")" = 4 ]'
ok "P1 COST-ABORT logged, not relaunching"       'grep -q "COST-ABORT (marker gs://qhrrn2-arc/decarc/CHAIN-COST-ABORT present" "$T/p1/pod.log" && grep -q "not relaunching" "$T/p1/pod.log"'
ok "P1 the node torn down exactly once"          '[ "$(grep -c "^DELETE us-east1-d qhrrn2-arc-pod" "$T/p1/calls.txt")" = 1 ] && grep -q "down rc=0" "$T/p1/pod.log"'
ok "P1 no launch, no relaunch attempt"           '! grep -q "LAUNCH chain\|IDLE — relaunching" "$T/p1/pod.log"'
ok "P1 the pid file removed"                     '[ ! -f "$T/p1/pod.pid" ]'
echo "P2 no marker, the node READY + IDLE (the pre-existing relaunch path; MAX_RELAUNCH=1 bounds it)"
scenario p2 1 0
ok "P2 exit 3 (chain died 1x)"                   '[ "$(cat "$T/p2/rc.txt")" = 3 ] && grep -q "chain died 1x" "$T/p2/pod.log"'
ok "P2 a relaunch was attempted"                 'grep -q "IDLE — relaunching (attempt 1/1)" "$T/p2/pod.log" && grep -q "LAUNCH chain in us-east1-d" "$T/p2/pod.log"'
ok "P2 no COST-ABORT line"                       '! grep -q "COST-ABORT" "$T/p2/pod.log"'
ok "P2 the node torn down at the end"            'grep -q "^DELETE us-east1-d qhrrn2-arc-pod" "$T/p2/calls.txt"'
echo "P3 the marker present, the node ABSENT"
scenario p3 0 1
ok "P3 exit 4"                                   '[ "$(cat "$T/p3/rc.txt")" = 4 ]'
ok "P3 COST-ABORT logged, nothing to tear down"  'grep -q "COST-ABORT (marker" "$T/p3/pod.log" && grep -q "node ABSENT — nothing to tear down" "$T/p3/pod.log"'
ok "P3 zero deletes, no create"                  '[ ! -s "$T/p3/calls.txt" ] && ! grep -q "CREATE" "$T/p3/pod.log"'
echo "P4 this pod's SHARE_DONE_w0 marker present, the node READY (the env's SHARE_MARK=SHARE_DONE_w0)"
scenario p4 1 0 SHARE_DONE_w0
ok "P4 exit 0"                                   '[ "$(cat "$T/p4/rc.txt")" = 0 ]'
ok "P4 SHARE-DONE logged"                        'grep -q "SHARE-DONE (marker gs://qhrrn2-arc/decarc/SHARE_DONE_w0 present" "$T/p4/pod.log"'
ok "P4 the node torn down exactly once"          '[ "$(grep -c "^DELETE us-east1-d qhrrn2-arc-pod" "$T/p4/calls.txt")" = 1 ]'
ok "P4 no launch, no relaunch, pid file removed" '! grep -q "LAUNCH chain\|IDLE — relaunching" "$T/p4/pod.log" && [ ! -f "$T/p4/pod.pid" ]'
echo "P5 only the OTHER pod's marker (SHARE_DONE_w1) present, the node READY + IDLE"
scenario p5 1 0 SHARE_DONE_w1
ok "P5 the other pod's marker is ignored (exit 3 by the relaunch cap, as P2)" '[ "$(cat "$T/p5/rc.txt")" = 3 ] && ! grep -q "SHARE-DONE" "$T/p5/pod.log"'
ok "P5 the relaunch re-plants the node guard (v_guard after every relaunch)" 'awk "/IDLE — relaunching/{r=1} r && /guard re-planted|plant_guard failed/{g=1} END{exit !g}" "$T/p5/pod.log"'
echo "P6 the node ABSENT, every zone dry: the create's image, spot and labels (TPU_RUNTIME=v2-alpha-tpuv6e); stop at the deadline"
scenario p6 0 0 "" 25
ok "P6 exit 2 at the deadline, nothing created"  '[ "$(cat "$T/p6/rc.txt")" = 2 ] && grep -q "reached the watchdog deadline" "$T/p6/pod.log"'
ok "P6 the create carries the runtime, spot and labels" 'grep -q "^CREATE compute tpus tpu-vm create qhrrn2-arc-pod --zone=us-east1-d --project=lab-shared --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e --spot --labels=program=qhrrn2,owner=pi,purpose=arc" "$T/p6/calls.txt"'
ok "P6 no delete of anything"                    '! grep -q "^DELETE" "$T/p6/calls.txt"'
echo "harness_pod_state: $PASS passed, $FAIL failed ($T)"
[ "$FAIL" -eq 0 ]
