#!/bin/bash
# Ledger: RUNG 2 offline stub harness (2026-08-27) — chain_sportBr2.sh end-to-end
# with NO cloud and NO model code: stub gsutil/gcloud + stub tools/{pretrain.py,
# eval_sudoku_extreme.py,select_ckpt.py,probe_sudoku.py}. COMMITTED (harnesses
# used to die with sessions — this one is repo-carried per the PI's ops mandate).
# O4: per-worker FILESYSTEM ISOLATION — each stub worker runs in its OWN sandbox
# home sharing only the fake GCS (the shared-dir blind spot missed both
# multi-host bugs; never again).
# Scenarios: S1 fresh 1x8 -> COMPLETE (+ coincide-skip, FULLVB-skip, depth rider,
# optional demos) · S2 idempotent re-run · S3 2-worker isolated coop -> WORKER-
# DONE then COMPLETE · S4 optional-task failure NEVER blocks · S5 pretrain
# failure -> WORKER-DONE only (guard holds).
set -uo pipefail
SB=${HARNESS_DIR:-$(mktemp -d /tmp/r2harness.XXXXXX)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
FAKE="$SB/gcs"
PASS=0; FAIL=0
ck () { if eval "$2"; then PASS=$((PASS+1)); echo "  PASS  $1"; else FAIL=$((FAIL+1)); echo "  FAIL  $1"; fi; }

make_stubs () {  # WORKDIR — stub tools + bin into an isolated worker home
  local WD=$1
  mkdir -p "$WD/tools" "$WD/runs" "$WD/data/sudoku_extreme" "$WD/bin" "$WD/src"
  cp "$REPO/tools/chain_sportBr2.sh" "$WD/tools/"
  cat > "$WD/bin/gsutil" <<EOF
#!/bin/bash
FAKE="$FAKE"
EOF
  cat >> "$WD/bin/gsutil" <<'EOF'
args=(); for a in "$@"; do [ "$a" = "-q" ] || [ "$a" = "-n" ] || args+=("$a"); done
cmd=${args[0]}; args=("${args[@]:1}")
map () { printf '%s' "${1/gs:\/\/qhrrn2-rescue/$FAKE}"; }
case $cmd in
  stat) p=$(map "${args[0]}"); [ -e "$p" ] ;;
  cat)  p=$(map "${args[0]}"); cat "$p" 2>/dev/null ;;
  rm)   p=$(map "${args[0]}"); rm -f "$p" ;;
  ls)   rc=1; for g in "${args[@]}"; do p=$(map "$g"); for f in $p; do [ -e "$f" ] && { echo "${f/$FAKE/gs://qhrrn2-rescue}"; rc=0; }; done; done; exit $rc ;;
  cp)   src=${args[0]}; dst=${args[1]}
        if [ "$src" = "-" ]; then p=$(map "$dst"); mkdir -p "$(dirname "$p")"; cat > "$p"
        elif [ "$dst" = "-" ]; then p=$(map "$src"); cat "$p" 2>/dev/null
        else s=$(map "$src"); d=$(map "$dst"); [ -e "$s" ] || exit 1; mkdir -p "$(dirname "$d")"; cp "$s" "$d"; fi ;;
  *) exit 64 ;;
esac
EOF
  cat > "$WD/bin/gcloud" <<'EOF'
#!/bin/bash
echo "stub-gcloud $*"; exit 0
EOF
  chmod +x "$WD/bin/gsutil" "$WD/bin/gcloud"
  # ---- stub pretrain: instant; ckpt grid + metrics + config
  cat > "$WD/tools/pretrain.py" <<'EOF'
import argparse, json, os, pickle, sys
ap = argparse.ArgumentParser()
ap.add_argument('--out'); ap.add_argument('--steps', type=int, default=50000)
ap.add_argument('--seed', type=int, default=0)
a, _ = ap.parse_known_args()
arm = os.path.basename(a.out).split('_')[-1]
if os.environ.get('STUB_PRETRAIN_FAIL') == arm:
    print('boom'); sys.exit(1)
os.makedirs(a.out, exist_ok=True)
for s in range(5000, a.steps + 1, 5000):
    pickle.dump(dict(step=s), open(f"{a.out}/ckpt_{s:06d}.pkl", 'wb'))
pickle.dump(dict(step=a.steps), open(f"{a.out}/ckpt_latest.pkl", 'wb'))
with open(f"{a.out}/metrics.jsonl", 'w') as fh:
    for s in range(0, a.steps + 1, 2000):
        fh.write(json.dumps({"step": s, "loss": 0.5}) + "\n")
        fh.write(json.dumps({"monitor": {"step": s, "val_t64": 0.1, "ret_final_t8": 1.0, "eta": 0.8, "lam_joint_max": 0.9, "lam_max_max": 0.2}}) + "\n")
open(f"{a.out}/config.json", 'w').write('{"stub": true}')
EOF
  # ---- stub select_ckpt: vb from env STUB_VB_<arm> else final
  cat > "$WD/tools/select_ckpt.py" <<'EOF'
import os, pickle, sys
d = sys.argv[1]; arm = os.path.basename(d).split('_')[-1]
vb = os.environ.get(f'STUB_VB_{arm}')
if not vb:
    vb = f"{pickle.load(open(f'{d}/ckpt_latest.pkl','rb'))['step']:06d}"
print(f"{vb} 0.1000 {vb}")
EOF
  # ---- stub evaluator: shard/merge/plain modes; env-tunable per (arm, context)
  cat > "$WD/tools/eval_sudoku_extreme.py" <<'EOF'
import argparse, json, os, pathlib
ap = argparse.ArgumentParser()
for f in ('--ckpt','--npz','--out','--shard','--split','--init','--merge'):
    ap.add_argument(f)
for f in ('--stratified','--t-total','--k-init','--batch','--subsample'):
    ap.add_argument(f, type=int)
ap.add_argument('--bank-every', type=float, default=0)
ap.add_argument('--final-map-only', action='store_true')
ap.add_argument('--vote-unverified', action='store_true')
a = ap.parse_args()
def env(k, d):
    v = os.environ.get(k); return float(v) if v else d
if a.merge:
    d = pathlib.Path(a.merge)
    ss = sorted(d.glob('summary_s*.json'))
    js = [json.loads(p.read_text()) for p in ss]
    tot = sum(j['n'] for j in js)
    out = dict(js[0]); out.update(n=tot, shard='merged')
    (d / 'summary_all.json').write_text(json.dumps(out)); raise SystemExit(0)
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
# identify arm from ckpt path
ck = a.ckpt or ''
arm = 'X'
for c in ('C1s1','C1','C2','C3','C4'):
    if f'_{c}/' in ck or ck.endswith(f'_{c}.pkl') or f'sportBr2_{c}' in ck: arm = c; break
if '_d3demo_' in ck: arm = 'D3'
kind = out.name
n = a.stratified or a.subsample or 64
s = dict(ckpt=a.ckpt, npz=a.npz, split=a.split, n=n, t_total=a.t_total, k_init=a.k_init or 0,
         init=a.init or 'void', wall_s=1.0, rating_bins=[0]*9, givens_kept_frac=1.0)
if (a.init == 'solution') and a.final_map_only:
    s['exact_acc'] = env(f'STUB_RETFM_{arm}', 1.0)
elif a.init == 'solution':
    s['exact_acc'] = 1.0
else:
    s['exact_acc'] = env(f'STUB_COLD_{arm}', 0.25)
if a.k_init:
    v256 = env(f'STUB_SCR_{arm}_{kind}', env(f'STUB_SCR_{arm}', 0.7))
    s['vote_at_k'] = {'16': round(v256*0.55,4), '64': round(v256*0.8,4), '128': round(v256*0.9,4), '256': v256}
    if (a.k_init or 0) <= 128: s['vote_at_k'] = {k: v for k, v in s['vote_at_k'].items() if int(k) <= a.k_init}
if a.vote_unverified:
    s['majority_vote_at_k'] = {'16': 0.3, '64': 0.4, '128': 0.45}
if a.shard:
    K, N = a.shard.split('/')
    s['n'] = n // int(N)
    (out / f'summary_s{K}.json').write_text(json.dumps(s))
    (out / f'records_s{K}.npz').write_bytes(b'stub')
else:
    (out / 'summary_all.json').write_text(json.dumps(s))
    (out / 'records_all.npz').write_bytes(b'stub')
EOF
  # ---- stub probe
  cat > "$WD/tools/probe_sudoku.py" <<'EOF'
import argparse, json, os, pathlib
ap = argparse.ArgumentParser(); ap.add_argument('--out')
a, _ = ap.parse_known_args()
p = pathlib.Path(a.out); p.mkdir(parents=True, exist_ok=True)
with open(p / 'results.jsonl', 'w') as fh:
    for i in range(4):
        fh.write(json.dumps({"task": i, "gt_retention": 1, "q_ladder": [1,1,1,1,1,1], "I_s": [1000,30,20,10,1]}) + "\n")
EOF
  printf 'npzstub' > "$WD/data/sudoku_extreme/sudoku_extreme_seed0.npz"
}

seed_fake_gcs () {
  rm -rf "$FAKE"; mkdir -p "$FAKE/sportBr2" "$FAKE/sport2" "$FAKE/sportB"
  printf 'npzstub' > "$FAKE/sport2/sudoku_extreme_seed0.npz"
  python3 -c "import pickle;pickle.dump(dict(step=50000),open('$FAKE/sportB/B2_ckpt.pkl','wb'))"
  python3 -c "import pickle;pickle.dump(dict(step=20000),open('$FAKE/sport2/S5_ckpt.pkl','wb'))"
}

run_chain () {  # WORKDIR W NW [env k=v...]
  local WD=$1 w=$2 nw=$3; shift 3
  ( cd "$WD" && PATH="$WD/bin:$PATH" NCHIP=2 CHAIN_WORKER=$w CHAIN_WORKERS=$nw \
    ARMS_W0='C1 C2' ARMS_W1='C1s1 C3 C4' PRIMARY='C1 C1s1 C2 C3' CARRIER_FULLS='C1 C1s1' \
    SX_STRAT=64 SX_SUB=128 PHASE2_SLEEP=1 P4_POLL_SLEEP=1 P4_WAIT_PASSES=3 P4_WAIT_PASSES2=6 \
    SHARD_RETRY_SLEEP=1 PARTIAL_SYNC_SLEEP=2 SELF_TEARDOWN=1 SELF_POD=stubpod SELF_ZONE=stubzone \
    env "$@" bash tools/chain_sportBr2.sh )
}

echo "=== harness sandbox: $SB"
# ---------------- S1: fresh single-worker (1x8-shape; NCHIP=2 for speed) ----------------
seed_fake_gcs
W1="$SB/w_single"; rm -rf "$W1"; make_stubs "$W1"
# make C4's m2 (40000) coincide with a stubbed vb to exercise the coincide-skip
run_chain "$W1" 0 1 STUB_VB_C4=040000 STUB_SCR_C2_vb=0.30 STUB_SCR_C4_vb=0.60 > "$SB/s1.log" 2>&1
ck "S1 sentinel COMPLETE"            "grep -q 'CHAIN-SPORTBR2-COMPLETE' $SB/s1.log"
ck "S1 five pretrains"               "[ \$(grep -c 'PRETRAIN-C.*-OK' $SB/s1.log) -eq 5 ]"
ck "S1 screens ran (>=13 OK) + coincide skip" "[ \$(grep -c 'SCREEN-C.*-OK' $SB/s1.log) -ge 13 ] && grep -q 'SCREEN-C4-m2-SKIP' $SB/s1.log"
ck "S1 FULLVB skip (vb=final)"       "grep -q 'FULLVB-C1-SKIP' $SB/s1.log"
ck "S1 probes"                       "grep -q 'PROBES4-OK' $SB/s1.log"
ck "S1 PHASE4 coop + winner marker"  "grep -q 'PHASE4-OK' $SB/s1.log && [ -f $FAKE/sportBr2/p4winner.txt ]"
ck "S1 depth rider"                  "grep -q 'P4DEPTH-OK' $SB/s1.log && [ -f $FAKE/sportBr2/depth_t256.tgz ]"
ck "S1 d3 demos (optional) ran"      "[ -f $FAKE/sportBr2/d3demo_b2d64.tgz ] && [ -f $FAKE/sportBr2/d3demo_s5d16.tgz ]"
ck "S1 final banked"                 "[ -f $FAKE/sportBr2/sportBr2_final.tgz ]"
ck "S1 teardown issued"              "grep -q 'SELF-TEARDOWN-ISSUED' $SB/s1.log"
# ---------------- S2: idempotent re-run ----------------
run_chain "$W1" 0 1 STUB_VB_C4=040000 > "$SB/s2.log" 2>&1
ck "S2 all SKIP"                     "[ \$(grep -c 'SKIP-C.* (GCS complete)\\|SKIP-C.* (done)' $SB/s2.log) -eq 5 ]"
ck "S2 COMPLETE again"               "grep -q 'CHAIN-SPORTBR2-COMPLETE' $SB/s2.log"
ck "S2 no re-screens"                "! grep -q 'SCREEN-C.*-OK' $SB/s2.log"
# ---------------- S3: 2 workers, ISOLATED filesystems, coop PHASE4 ----------------
seed_fake_gcs
WA="$SB/w_a"; WB="$SB/w_b"; rm -rf "$WA" "$WB"; make_stubs "$WA"; make_stubs "$WB"
run_chain "$WA" 0 2 > "$SB/s3a.log" 2>&1
ck "S3 worker0 WORKER-DONE (not COMPLETE)" "grep -q 'CHAIN-SPORTBR2-WORKER-DONE' $SB/s3a.log && ! grep -q 'CHAIN-SPORTBR2-COMPLETE' $SB/s3a.log"
run_chain "$WB" 1 2 > "$SB/s3b.log" 2>&1
ck "S3 worker1 finishes -> COMPLETE" "grep -q 'CHAIN-SPORTBR2-COMPLETE' $SB/s3b.log"
ck "S3 coop shards from both homes"  "grep -q 'P4-SHARD-s0-OK\\|P4-SHARD-s1-OK' $SB/s3a.log || grep -q 'P4-SHARD-s2-OK\\|P4-SHARD-s3-OK' $SB/s3b.log"
ck "S3 final banked"                 "[ -f $FAKE/sportBr2/sportBr2_final.tgz ]"
# ---------------- S4: optional-task failure never blocks ----------------
seed_fake_gcs
rm -f "$FAKE/sportB/B2_ckpt.pkl"        # d3demo:b2d64 source missing -> must not block
W4="$SB/w_opt"; rm -rf "$W4"; make_stubs "$W4"
run_chain "$W4" 0 1 > "$SB/s4.log" 2>&1
ck "S4 d3demo failed"                "grep -q 'D3DEMO-b2d64-NOCKPT\\|D3DEMO-b2d64-FAILED' $SB/s4.log"
ck "S4 STILL COMPLETE"               "grep -q 'CHAIN-SPORTBR2-COMPLETE' $SB/s4.log"
# ---------------- S5: pretrain failure -> guard refuses completion ----------------
seed_fake_gcs
W5="$SB/w_fail"; rm -rf "$W5"; make_stubs "$W5"
run_chain "$W5" 0 1 STUB_PRETRAIN_FAIL=C2 P4_WAIT_PASSES=2 > "$SB/s5.log" 2>&1
ck "S5 pretrain C2 FAILED"           "grep -q 'PRETRAIN-C2-FAILED' $SB/s5.log"
ck "S5 WORKER-DONE only, no sentinel" "grep -q 'CHAIN-SPORTBR2-WORKER-DONE' $SB/s5.log && ! grep -q 'CHAIN-SPORTBR2-COMPLETE' $SB/s5.log"
ck "S5 no final object"              "[ ! -f $FAKE/sportBr2/sportBr2_final.tgz ]"

# ---------------- S6: node-shape change mid-PHASE4 — the partition pin (pre-mortem catch) ----------------
seed_fake_gcs
W6="$SB/w_pin"; rm -rf "$W6"; make_stubs "$W6"
# simulate a dead 4-way run: NSH pinned at 4, shards s0/s1/s3 banked (n=32 each of SX_SUB=128)
mkdir -p "$FAKE/sportBr2/p4"
printf '4' > "$FAKE/sportBr2/p4/NSH.txt"
python3 - "$FAKE" <<'PY'
import json, sys, pathlib
fake = pathlib.Path(sys.argv[1])
for k in (0, 1, 3):
    (fake/'sportBr2/p4'/f'summary_s{k}.json').write_text(json.dumps(dict(n=32, ckpt='x', vote_at_k={'128': .6}, rating_bins=[0]*9, wall_s=1)))
    (fake/'sportBr2/p4'/f'records_s{k}.npz').write_bytes(b'stub')
PY
run_chain "$W6" 0 1 > "$SB/s6.log" 2>&1
ck "S6 pin honored (4-way from GCS, not 2)" "grep -q 'P4 partition pinned: 4-way (from GCS)' $SB/s6.log"
ck "S6 only the missing shard ran"          "grep -q 'P4-SHARD-s2-OK' $SB/s6.log && ! grep -q 'P4-SHARD-s0-OK\\|P4-SHARD-s1-OK\\|P4-SHARD-s3-OK' $SB/s6.log"
ck "S6 merge n-gate passed + PHASE4-OK"     "grep -q 'PHASE4-OK' $SB/s6.log && ! grep -q 'P4-MERGE-N-BAD' $SB/s6.log"
ck "S6 COMPLETE"                            "grep -q 'CHAIN-SPORTBR2-COMPLETE' $SB/s6.log"

echo; echo "harness: $PASS/$((PASS+FAIL))"
[ "$FAIL" -eq 0 ]
