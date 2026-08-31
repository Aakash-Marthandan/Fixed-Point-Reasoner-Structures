#!/bin/bash
# Offline stub harness for scan_runbook_sportBr2b.sh (2026-08-30). No cloud, no
# model code. Per-worker filesystem isolation (O4). Scenarios:
#  S1 fresh 4x4 (static map: 16 shard-jobs, both merges n-gated, sentinel)
#  S2 idempotent re-run (all SKIP, immediate COMPLETE)
#  S3 merge n-gate refusal (short n -> MERGE-N-BAD, no tgz, INCOMPLETE)
#  S4 fresh 1x8 fallback (sequential scans, COMPLETE)
set -uo pipefail
SB=$(mktemp -d /tmp/scan2bharness.XXXXXX); REPO=$(cd "$(dirname "$0")/.." && pwd)
FAKE="$SB/gcs"; mkdir -p "$FAKE"
PASS=0; FAIL=0
ck () { if eval "$2"; then PASS=$((PASS+1)); echo "  PASS  $1"; else FAIL=$((FAIL+1)); echo "  FAIL  $1"; fi; }

seed_fake () { echo ck > "$FAKE/sportBr2b%C3X_ckpt.pkl"; echo ck > "$FAKE/sportBr2b%D4_ckpt.pkl"; echo npz > "$FAKE/sport2%sudoku_extreme_seed0.npz"; }

mkworker () {  # N -> isolated home with stubs
  local WD="$SB/w$1"
  mkdir -p "$WD/tools" "$WD/runs" "$WD/bin" "$WD/data/sudoku_extreme" "$WD/.venv/bin"
  cp "$REPO/tools/scan_runbook_sportBr2b.sh" "$WD/tools/"
  : # npz now fetched by the runbook (gen-1 incident fix)
  cat > "$WD/bin/gsutil" <<EOF
#!/bin/bash
FAKE="$FAKE"
q=0; [ "\$1" = "-q" ] && { q=1; shift; }
cmd=\$1; shift
p2f () { echo "\$FAKE/\$(echo "\$1" | sed 's|gs://[^/]*/||; s|/|%|g')"; }
case \$cmd in
  stat) [ -f "\$(p2f "\$1")" ];;
  cp) if [ "\$1" = "-" ]; then cat > "\$(p2f "\$2")"; elif [[ "\$1" == gs://* ]]; then src=\$(p2f "\$1"); [ -f "\$src" ] && cat "\$src" > "\$2" || exit 1; else cat "\$1" > "\$(p2f "\$2")"; fi;;
  ls) pat=\$(echo "\$1" | sed 's|gs://[^/]*/||; s|/|%|g; s|\*|.*|g'); found=0; for f in "\$FAKE"/*; do b=\$(basename "\$f"); if [[ "\$b" =~ ^\$pat\$ ]]; then echo "gs://bucket/\$(echo "\$b" | sed 's|%|/|g')"; found=1; fi; done; [ \$found -eq 1 ];;
  rm) rm -f "\$(p2f "\$1")";;
esac
EOF
  cat > "$WD/bin/gcloud" <<'EOF'
#!/bin/bash
exit 0
EOF
  cat > "$WD/tools/eval_sudoku_extreme.py" <<'EOF'
import json, sys, os
args = sys.argv[1:]
def get(flag, default=None):
    return args[args.index(flag)+1] if flag in args else default
out = get("--out", get("--merge"))
os.makedirs(out, exist_ok=True)
if "--merge" in args:
    n = 0
    for f in sorted(os.listdir(out)):
        if f.startswith("summary_s") and f.endswith(".json"):
            n += json.load(open(os.path.join(out, f)))["n"]
    json.dump({"n": n, "vote_at_k": {"128": 0.87}}, open(os.path.join(out, "summary_all.json"), "w"))
else:
    k = get("--shard").split("/")[0]
    per = 2500 if not os.environ.get("HARNESS_SHORT_SHARD") else 2400
    json.dump({"n": per, "shard": k}, open(os.path.join(out, f"summary_s{k}.json"), "w"))
    open(os.path.join(out, f"records_s{k}.npz"), "wb").write(b"x")
    open(os.path.join(out, f"partial_stub_s{k}.npz"), "wb").write(b"p")
EOF
  cat > "$WD/.venv/bin/python3" <<EOF
#!/bin/bash
exec /usr/bin/python3 "\$@"
EOF
  chmod +x "$WD/bin/gsutil" "$WD/bin/gcloud" "$WD/.venv/bin/python3"
  echo "$WD"
}

runw () {  # WD W NW extra_env
  local WD=$1 W=$2 NW=$3; shift 3
  (cd "$WD" && PATH="$WD/bin:$PATH" CHAIN_WORKER=$W CHAIN_WORKERS=$NW NCHIP_OVERRIDE=4 \
   SCAN_WAIT_PASSES=3 SCAN_POLL_SLEEP=1 SCAN_PARTIAL_SLEEP=1 SX_SUB=20000 "$@" bash tools/scan_runbook_sportBr2b.sh) 
}

seed_fake
echo "== S1 fresh 4x4 =="
W0=$(mkworker 0); W1=$(mkworker 1); W2=$(mkworker 2); W3=$(mkworker 3)
L0=$(runw "$W0" 0 4 2>&1); L2=$(runw "$W2" 2 4 2>&1); L1=$(runw "$W1" 1 4 2>&1); L3=$(runw "$W3" 3 4 2>&1)
ck "16 shard summaries banked" "[ \$(ls $FAKE/*summary_s*.json 2>/dev/null | wc -l) -eq 16 ]"
ck "both scan tgzs banked" "ls $FAKE/*p4x_C3X.tgz >/dev/null 2>&1 && ls $FAKE/*p4x_D4.tgz >/dev/null 2>&1"
ck "COMPLETE marker banked" "ls $FAKE/*p4x_COMPLETE.txt >/dev/null 2>&1"
ck "sentinel emitted" "echo \"\$L3\$L1\" | grep -q 'SCAN-SPORTBR2B-COMPLETE'"
ck "NSH pins = 8 both" "grep -q 8 $FAKE/*p4xc3x*NSH.txt && grep -q 8 $FAKE/*p4xd4*NSH.txt"
ck "static map: w0 ran C3X s0-3" "echo \"\$L0\" | grep -q 'SHARD-OK C3X s3' && ! echo \"\$L0\" | grep -q 'SHARD-OK D4'"
echo "== S2 idempotent =="
L=$(runw "$W0" 0 4 2>&1)
ck "all SKIP" "echo \"\$L\" | grep -q 'SHARD-SKIP C3X s0' && ! echo \"\$L\" | grep -q 'SHARD-START'"
ck "immediate COMPLETE" "echo \"\$L\" | grep -q 'SCAN-SPORTBR2B-COMPLETE'"
echo "== S3 merge n-gate refusal =="
rm -rf "$FAKE"; mkdir -p "$FAKE"; seed_fake
W4=$(mkworker 4)
L=$(HARNESS_SHORT_SHARD=1 runw "$W4" 0 1 2>&1)
ck "MERGE-N-BAD fired" "echo \"\$L\" | grep -q 'MERGE-N-BAD'"
ck "no tgz banked" "! ls $FAKE/*p4x_C3X.tgz >/dev/null 2>&1"
ck "INCOMPLETE exit" "echo \"\$L\" | grep -q 'SCAN-SPORTBR2B-INCOMPLETE'"
echo "== S4 fresh 1x8 fallback =="
rm -rf "$FAKE"; mkdir -p "$FAKE"; seed_fake
W5=$(mkworker 5)
L=$(runw "$W5" 0 1 2>&1)
ck "both scans sequential complete" "ls $FAKE/*p4x_C3X.tgz >/dev/null 2>&1 && ls $FAKE/*p4x_D4.tgz >/dev/null 2>&1"
ck "sentinel" "echo \"\$L\" | grep -q 'SCAN-SPORTBR2B-COMPLETE'"
echo; echo "HARNESS: $PASS PASS / $FAIL FAIL"
[ $FAIL -eq 0 ]
