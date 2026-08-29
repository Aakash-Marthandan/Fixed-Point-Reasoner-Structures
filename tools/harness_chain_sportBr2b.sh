#!/bin/bash
# Ledger: RUNG 2B offline stub harness (2026-08-29) — chain_sportBr2b.sh end-to-end
# with NO cloud and NO model code (stub gsutil/gcloud + stub tools). COMMITTED.
# Per-worker FILESYSTEM ISOLATION (the O4 discipline). Scenarios:
#  S1 fresh 1-worker -> gate PASS, PHASE4 capped-claim scan, depth rider, COMPLETE
#  S2 idempotent re-run (all SKIP, no re-screens)
#  S3 2-worker isolated coop -> WORKER-DONE then COMPLETE
#  S4 gate FAIL -> no scan, COMPLETE with banked FAIL marker (breadth not required)
#  S5 pretrain hard-fail (non-NaN) -> WORKER-DONE only (guard holds)
#  S6 partition pin honored across a shape change (rung-2 pre-mortem catch)
#  S7 NaN mid-arm -> AUTOMATED AMPUTATION (finite-grid final, STOPPED label,
#     impossible screens = legit zero-byte skips, evals proceed) -> COMPLETE
#  S8 idempotent AFTER a NaN stop (no re-amputation; SKIP from GCS)
#  S9 zero-byte screen SELF-HEALS (the C4_vb class: invalid empty -> re-run)
#  S10 stale claim with owner stamp parsed correctly (no arith error; takeover)
set -uo pipefail
SB=${HARNESS_DIR:-$(mktemp -d /tmp/r2bharness.XXXXXX)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
FAKE="$SB/gcs"
PASS=0; FAIL=0
ck () { if eval "$2"; then PASS=$((PASS+1)); echo "  PASS  $1"; else FAIL=$((FAIL+1)); echo "  FAIL  $1"; fi; }

make_stubs () {  # WORKDIR — stub tools + bin into an isolated worker home
  local WD=$1
  mkdir -p "$WD/tools" "$WD/runs" "$WD/data/sudoku_extreme" "$WD/bin" "$WD/src"
  cp "$REPO/tools/chain_sportBr2b.sh" "$WD/tools/"
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
  # ---- stub pretrain: instant; grids + metrics + config; NaN mode per arm
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
if os.environ.get('STUB_PRETRAIN_NAN') == arm:
    # clean grids to 20000; a poisoned 25000 grid + latest; metrics tail NaN; die
    for s in range(5000, 20001, 5000):
        pickle.dump(dict(step=s), open(f"{a.out}/ckpt_{s:06d}.pkl", 'wb'))
    pickle.dump(dict(step=25000, bad=float('nan')), open(f"{a.out}/ckpt_025000.pkl", 'wb'))
    pickle.dump(dict(step=25100, bad=float('nan')), open(f"{a.out}/ckpt_latest.pkl", 'wb'))
    with open(f"{a.out}/metrics.jsonl", 'w') as fh:
        for s in range(0, 25001, 2000):
            fh.write(json.dumps({"step": s, "loss": 0.5, "A_total": 1e4}) + "\n")
            fh.write(json.dumps({"monitor": {"step": s, "val_t64": 0.1, "ret_final_t8": 1.0, "eta": 0.8, "lam_joint_max": 0.9, "lam_max_max": 0.2}}) + "\n")
        fh.write(json.dumps({"step": 25100, "loss": float('nan'), "A_total": 1e14}) + "\n")
    open(f"{a.out}/config.json", 'w').write('{"stub": true}')
    sys.exit(1)
for s in range(5000, a.steps + 1, 5000):
    pickle.dump(dict(step=s), open(f"{a.out}/ckpt_{s:06d}.pkl", 'wb'))
pickle.dump(dict(step=a.steps), open(f"{a.out}/ckpt_latest.pkl", 'wb'))
with open(f"{a.out}/metrics.jsonl", 'w') as fh:
    for s in range(0, a.steps + 1, 2000):
        fh.write(json.dumps({"step": s, "loss": 0.5, "A_total": 5e4}) + "\n")
        fh.write(json.dumps({"monitor": {"step": s, "val_t64": 0.1, "ret_final_t8": 1.0, "eta": 0.8, "lam_joint_max": 0.9, "lam_max_max": 0.2}}) + "\n")
open(f"{a.out}/config.json", 'w').write('{"stub": true}')
EOF
  # ---- stub select_ckpt: vb from env STUB_VB_<arm> else the final step
  cat > "$WD/tools/select_ckpt.py" <<'EOF'
import os, pickle, sys
d = sys.argv[1]; arm = os.path.basename(d).split('_')[-1]
vb = os.environ.get(f'STUB_VB_{arm}')
if not vb:
    vb = f"{pickle.load(open(f'{d}/ckpt_latest.pkl','rb'))['step']:06d}"
print(f"{vb} 0.1000 {vb}")
EOF
  # ---- stub evaluator: shard/merge/plain; screen values via STUB_SCR_<arm>[_<kind>]
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
ck = a.ckpt or ''
arm = 'X'
for c in ('C3X','D1','D2','D3','D4'):
    if f'_{c}/' in ck or ck.endswith(f'_{c}.pkl'): arm = c; break
name = out.name
kind = name.split('_')[-1] if name.startswith('sxscreen') else name
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
import argparse, json, pathlib
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
  rm -rf "$FAKE"; mkdir -p "$FAKE/sportBr2b" "$FAKE/sport2" "$FAKE/sportBr2"
  printf 'npzstub' > "$FAKE/sport2/sudoku_extreme_seed0.npz"
  python3 -c "import pickle;pickle.dump(dict(step=20000),open('$FAKE/sportBr2/C3_ckpt.pkl','wb'))"
}

run_chain () {  # WORKDIR W NW [env k=v...]
  local WD=$1 w=$2 nw=$3; shift 3
  ( cd "$WD" && PATH="$WD/bin:$PATH" NCHIP=2 CHAIN_WORKER=$w CHAIN_WORKERS=$nw \
    ARMS_W0='D1 D3' ARMS_W1='D2 D4' PRIMARY='D1 D3' \
    SX_STRAT=64 SX_SUB=128 PHASE2_SLEEP=1 P4_POLL_SLEEP=1 P4_WAIT_PASSES=3 P4_WAIT_PASSES2=6 P4_CLAIM_PASSES=8 \
    SHARD_RETRY_SLEEP=1 PARTIAL_SYNC_SLEEP=2 CACHE_SETTLE_SLEEP=1 SELF_TEARDOWN=1 SELF_POD=stubpod SELF_ZONE=stubzone \
    env "$@" bash tools/chain_sportBr2b.sh )
}

echo "=== harness sandbox: $SB"
# ---------------- S1: fresh single-worker, gate PASS ----------------
seed_fake_gcs
W1="$SB/w_single"; rm -rf "$W1"; make_stubs "$W1"
run_chain "$W1" 0 1 ARMS_W0='D1 D2 D3 D4 C3X' ARMS_W1='' STUB_SCR_D1_vb=0.91 STUB_SCR_D2_vb=0.87 > "$SB/s1.log" 2>&1
ck "S1 sentinel COMPLETE"            "grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s1.log"
ck "S1 four pretrains OK"            "[ \$(grep -c 'PRETRAIN-D.*-OK' $SB/s1.log) -eq 4 ]"
ck "S1 C3X init fetched + pretrain OK" "grep -q 'C3X-INIT fetched' $SB/s1.log && grep -q 'PRETRAIN-C3X-OK' $SB/s1.log"
ck "S1 all 18 screens ran (15 D + 3 C3X)" "[ \$(grep -cE 'SCREEN-(D|C3X).*-OK' $SB/s1.log) -eq 18 ]"
ck "S1 vb screens always run (5)"    "[ \$(grep -cE 'SCREEN-(D.|C3X)-vb-OK' $SB/s1.log) -eq 5 ]"
ck "S1 C3X full + in guard"          "[ -f $FAKE/sportBr2b/full_C3X_t64.tgz ]"
ck "S1 FULLVB skip (vb=final)"       "grep -q 'FULLVB-D1-SKIP' $SB/s1.log"
ck "S1 gate PASS + capped scan"      "grep -q 'P4-GATE: PASS' $SB/s1.log && grep -q 'PHASE4-OK D1' $SB/s1.log"
ck "S1 spawn echoes + inflight"      "grep -q 'P4-SPAWN s0' $SB/s1.log && grep -q 'P4-INFLIGHT' $SB/s1.log"
ck "S1 depth rider"                  "grep -q 'P4DEPTH-OK' $SB/s1.log && [ -f $FAKE/sportBr2b/depth_t256.tgz ]"
ck "S1 probes4 + optional D4 probe"  "grep -q 'PROBES4-OK' $SB/s1.log && grep -q 'PROBE-D4-OK' $SB/s1.log"
ck "S1 final banked + teardown"      "[ -f $FAKE/sportBr2b/sportBr2b_final.tgz ] && grep -q 'SELF-TEARDOWN-ISSUED' $SB/s1.log"
# ---------------- S2: idempotent re-run ----------------
run_chain "$W1" 0 1 ARMS_W0='D1 D2 D3 D4' ARMS_W1='' STUB_SCR_D1_vb=0.91 STUB_SCR_D2_vb=0.87 > "$SB/s2.log" 2>&1
ck "S2 all four SKIP"                "[ \$(grep -c 'SKIP-D.* (GCS complete)\\|SKIP-D.* (done)' $SB/s2.log) -eq 4 ]"
ck "S2 COMPLETE again, no re-screens" "grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s2.log && ! grep -q 'SCREEN-D.*-OK' $SB/s2.log"
# ---------------- S3: 2 workers, isolated homes, coop ----------------
seed_fake_gcs
WA="$SB/w_a"; WB="$SB/w_b"; rm -rf "$WA" "$WB"; make_stubs "$WA"; make_stubs "$WB"
run_chain "$WA" 0 2 STUB_SCR_D1_vb=0.91 STUB_SCR_D2_vb=0.87 > "$SB/s3a.log" 2>&1
ck "S3 worker0 WORKER-DONE"          "grep -q 'CHAIN-SPORTBR2B-WORKER-DONE' $SB/s3a.log && ! grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s3a.log"
run_chain "$WB" 1 2 STUB_SCR_D1_vb=0.91 STUB_SCR_D2_vb=0.87 > "$SB/s3b.log" 2>&1
ck "S3 worker1 -> COMPLETE"          "grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s3b.log"
ck "S3 final banked"                 "[ -f $FAKE/sportBr2b/sportBr2b_final.tgz ]"
# ---------------- S4: gate FAIL -> no scan, still COMPLETE ----------------
seed_fake_gcs
W4="$SB/w_gate"; rm -rf "$W4"; make_stubs "$W4"
run_chain "$W4" 0 1 ARMS_W0='D1 D2 D3 D4' ARMS_W1='' STUB_SCR_D1_vb=0.80 STUB_SCR_D2_vb=0.87 > "$SB/s4.log" 2>&1
ck "S4 gate FAIL"                    "grep -q 'P4-GATE: FAIL' $SB/s4.log && grep -q 'P4-GATE-FAIL' $SB/s4.log"
ck "S4 no scan, no depth"            "! grep -q 'PHASE4-OK' $SB/s4.log && ! grep -q 'P4DEPTH-OK' $SB/s4.log"
ck "S4 COMPLETE w/ FAIL marker"      "grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s4.log && grep -q '^FAIL' $FAKE/sportBr2b/p4gate.txt"
# ---------------- S5: pretrain hard-fail (non-NaN) -> guard holds ----------------
seed_fake_gcs
W5="$SB/w_fail"; rm -rf "$W5"; make_stubs "$W5"
run_chain "$W5" 0 1 ARMS_W0='D1 D2 D3 D4' ARMS_W1='' STUB_PRETRAIN_FAIL=D1 P4_WAIT_PASSES=2 > "$SB/s5.log" 2>&1
ck "S5 D1 FAILED (no amputation)"    "grep -q 'PRETRAIN-D1-FAILED' $SB/s5.log && ! grep -q 'AMPUTATE-D1' $SB/s5.log"
ck "S5 WORKER-DONE only"             "grep -q 'CHAIN-SPORTBR2B-WORKER-DONE' $SB/s5.log && ! grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s5.log"
ck "S5 no final object"              "[ ! -f $FAKE/sportBr2b/sportBr2b_final.tgz ]"
# ---------------- S6: partition pin across a shape change ----------------
seed_fake_gcs
W6="$SB/w_pin"; rm -rf "$W6"; make_stubs "$W6"
mkdir -p "$FAKE/sportBr2b/p4"
printf '4' > "$FAKE/sportBr2b/p4/NSH.txt"
python3 - "$FAKE" <<'PY'
import json, sys, pathlib
fake = pathlib.Path(sys.argv[1])
for k in (0, 1, 3):
    (fake/'sportBr2b/p4'/f'summary_s{k}.json').write_text(json.dumps(dict(n=32, ckpt='x', vote_at_k={'128': .8}, rating_bins=[0]*9, wall_s=1)))
    (fake/'sportBr2b/p4'/f'records_s{k}.npz').write_bytes(b'stub')
PY
run_chain "$W6" 0 1 ARMS_W0='D1 D2 D3 D4' ARMS_W1='' STUB_SCR_D1_vb=0.91 STUB_SCR_D2_vb=0.87 > "$SB/s6.log" 2>&1
ck "S6 pin honored (4-way from GCS)" "grep -q 'P4 partition pinned: 4-way (from GCS)' $SB/s6.log"
ck "S6 only missing shard ran"       "grep -q 'P4-SHARD-s2-OK' $SB/s6.log && ! grep -q 'P4-SHARD-s0-OK\\|P4-SHARD-s1-OK\\|P4-SHARD-s3-OK' $SB/s6.log"
ck "S6 n-gate + COMPLETE"            "grep -q 'PHASE4-OK' $SB/s6.log && ! grep -q 'P4-MERGE-N-BAD' $SB/s6.log && grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s6.log"
# ---------------- S7: NaN mid-arm -> automated amputation -> COMPLETE ----------------
seed_fake_gcs
W7="$SB/w_nan"; rm -rf "$W7"; make_stubs "$W7"
run_chain "$W7" 0 1 ARMS_W0='D1 D2 D3 D4' ARMS_W1='' STUB_PRETRAIN_NAN=D3 STUB_SCR_D1_vb=0.91 STUB_SCR_D2_vb=0.87 > "$SB/s7.log" 2>&1
ck "S7 amputation fired, finite grid chosen (20000, not the poisoned 25000)" "grep -q 'AMPUTATE-D3-OK final=20000' $SB/s7.log"
ck "S7 STOPPED label banked"         "grep -q 'STOPPED final step 20000' $FAKE/sportBr2b/D3_STOPPED.txt"
ck "S7 metrics truncated to <=20000" "python3 -c \"
import json,sys
rows=[json.loads(l) for l in open('$W7/runs/pretrainsportBr2b_D3/metrics.jsonl') if l.strip()]
mx=max((r['monitor']['step'] if 'monitor' in r else r['step']) for r in rows)
sys.exit(0 if mx<=20000 else 1)\""
ck "S7 impossible screens = legit zero-byte skips" "grep -q 'SCREEN-D3-s025000-SKIP' $SB/s7.log && grep -q 'SCREEN-D3-s040000-SKIP' $SB/s7.log && [ -f $FAKE/sportBr2b/screen_D3_s025000_k256.tgz ] && [ ! -s $FAKE/sportBr2b/screen_D3_s025000_k256.tgz ]"
ck "S7 vb screen ran on the stopped final" "grep -q 'SCREEN-D3-vb-OK step=020000' $SB/s7.log"
ck "S7 evals proceed on the final"   "grep -q 'EVALCHEAP-D3-OK' $SB/s7.log"
ck "S7 COMPLETE with a stopped arm"  "grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s7.log"
# ---------------- S8: idempotent after the NaN stop ----------------
run_chain "$W7" 0 1 ARMS_W0='D1 D2 D3 D4' ARMS_W1='' STUB_PRETRAIN_NAN=D3 STUB_SCR_D1_vb=0.91 STUB_SCR_D2_vb=0.87 > "$SB/s8.log" 2>&1
ck "S8 D3 SKIPs from GCS, no re-amputation" "grep -q 'SKIP-D3 (GCS complete)\\|SKIP-D3 (done)' $SB/s8.log && ! grep -q 'AMPUTATE-D3' $SB/s8.log"
ck "S8 COMPLETE again"               "grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s8.log"
# ---------------- S9: zero-byte screen self-heals (the C4_vb class) ----------------
: > "$FAKE/sportBr2b/screen_D1_s010000_k256.tgz"     # corrupt: empty but D1 is NOT stopped
rm -f "$FAKE/sportBr2b/sportBr2b_final.tgz"
run_chain "$W7" 0 1 ARMS_W0='D1 D2 D3 D4' ARMS_W1='' STUB_PRETRAIN_NAN=D3 STUB_SCR_D1_vb=0.91 STUB_SCR_D2_vb=0.87 > "$SB/s9.log" 2>&1
ck "S9 invalid zero-byte detected + re-run" "grep -q 'SCREEN-OBJ-INVALID scr:D1:s010000' $SB/s9.log && grep -q 'SCREEN-D1-s010000-OK' $SB/s9.log"
ck "S9 healed object nonzero"        "[ -s $FAKE/sportBr2b/screen_D1_s010000_k256.tgz ]"
ck "S9 COMPLETE"                     "grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s9.log"
# ---------------- S10: stale owner-stamped claim parsed + taken over ----------------
printf '0 w9\n' > "$FAKE/sportBr2b/claim_screen_D2_s010000_k256"
rm -f "$FAKE/sportBr2b/screen_D2_s010000_k256.tgz" "$FAKE/sportBr2b/sportBr2b_final.tgz"
run_chain "$W7" 0 1 ARMS_W0='D1 D2 D3 D4' ARMS_W1='' STUB_PRETRAIN_NAN=D3 STUB_SCR_D1_vb=0.91 STUB_SCR_D2_vb=0.87 > "$SB/s10.log" 2>&1
ck "S10 stale claim taken over (first-field parse, no arith error)" "grep -q 'CLAIM-STALE scr:D2:s010000' $SB/s10.log && ! grep -qi 'arith\\|syntax error' $SB/s10.log"
ck "S10 screen re-ran + COMPLETE"    "grep -q 'SCREEN-D2-s010000-OK' $SB/s10.log && grep -q 'CHAIN-SPORTBR2B-COMPLETE' $SB/s10.log"

echo; echo "harness: $PASS/$((PASS+FAIL))"
[ "$FAIL" -eq 0 ]