#!/bin/bash
# Ledger: CHAMPION TRACK PILOT offline stub harness (2026-09-01; the house law:
# no chain launches without an end-to-end offline pass). Stubs: gsutil/gcloud
# -> a local FAKE_GCS dir; $CHAIN_PY -> a stub that emulates pretrain.py /
# eval_sudoku_extreme.py / (merge) by writing plausible artifacts; REAL python
# still runs the chain's own helpers (nan_check / amputate / n-gates), so the
# amputation path is executed for real on stub-written checkpoints.
# Scenarios: S1 fresh 1x8 -> COMPLETE; S2 idempotent rerun (all SKIP);
# S3 NaN one-shot amputation (single-stage arm AND stage-A of a two-stage arm
# -> STOPPED, no stage-B rescue); S4 fresh 4x4 static map -> COMPLETE from the
# waiting worker; S5 two-stage ordering (stage-A marker precedes stage B).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export REAL_PY="$REPO/.venv/bin/python3"
PASS=0; FAIL=0
ok () { PASS=$((PASS+1)); echo "  PASS  $1"; }
bad () { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }

mk_sandbox () {
  SB=$(mktemp -d /tmp/hc0.XXXXXX)
  mkdir -p "$SB/repo/tools" "$SB/repo/runs" "$SB/repo/data/sudoku_extreme" "$SB/gcs/sportC0" "$SB/gcs/sportBr2b" "$SB/bin"
  cp "$REPO/tools/chain_sportC0.sh" "$SB/repo/tools/"
  : > "$SB/repo/data/sudoku_extreme/sudoku_extreme_seed0.npz"
  echo "fake" > "$SB/gcs/sportBr2b/C3X_ckpt.pkl"; echo "fake" > "$SB/gcs/sportBr2b/D4_ckpt.pkl"

  # gsutil shim: gs://qhrrn2-rescue/X -> $SB/gcs/X (cp/stat/ls; '-' stdin cp)
  cat > "$SB/bin/gsutil" <<SH
#!/bin/bash
GB="$SB/gcs"
SH
  cat >> "$SB/bin/gsutil" <<'SH'
map () { echo "$1" | sed "s|gs://qhrrn2-rescue/|$GB/|"; }
args=(); for a in "$@"; do [ "$a" = "-q" ] || [ "$a" = "-m" ] || args+=("$a"); done
cmd=${args[0]:-}
case $cmd in
  stat) p=$(map "${args[1]}"); [ -f "$p" ];;
  ls)   p=$(map "${args[1]}"); ls -d ${p} 2>/dev/null | sed "s|$GB/|gs://qhrrn2-rescue/|";;
  cp)   src=${args[1]}; dst=${args[2]}
        if [ "$src" = "-" ]; then p=$(map "$dst"); mkdir -p "$(dirname "$p")"; cat > "$p"
        elif [[ "$src" == gs://* ]]; then p=$(map "$src"); [ -f "$p" ] && { mkdir -p "$(dirname "$dst")" 2>/dev/null; cp "$p" "$dst"; } || exit 1
        else p=$(map "$dst"); mkdir -p "$(dirname "$p")"; cp "$src" "$p"; fi;;
  *) exit 0;;
esac
SH
  chmod +x "$SB/bin/gsutil"
  cat > "$SB/bin/gcloud" <<'SH'
#!/bin/bash
echo "gcloud-stub $*"; exit 0
SH
  chmod +x "$SB/bin/gcloud"

  # the tool stub: dispatches on the SCRIPT PATH argument
  cat > "$SB/bin/stubpy" <<'PYEOF'
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
import numpy as np

argv = sys.argv[1:]
tool = argv[0] if argv else ""
def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

if tool.endswith("pretrain.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    steps = int(flag("--steps", "100"))
    nan_arm = os.environ.get("STUB_NAN_ARM", "")
    base = out.name.replace("pretrainsportC0_", "")
    is_nan = bool(nan_arm) and base == nan_arm
    rows = []
    for s in (steps // 2, steps):
        loss = float("nan") if (is_nan and s == steps) else 0.5
        rows.append(json.dumps({"step": s, "loss": loss, "ce_in": .4, "I_total": 1e5, "A_total": 5.0,
                                "rule_H": 0.0, "lr": 1e-3, "steps_per_sec": 99.0, "t": "T"}))
        rows.append(json.dumps({"monitor": {"step": s, "val_t64": 0.3, "ret_final_t8": 1.0,
                                            "ret_sched_t8": 1.0, "eta": 0.8, "lam_joint_max": 0.9}}))
    (out / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    # banked grids: a mid ckpt (finite) + latest (poisoned when NaN)
    import pickle
    fin = {"state": {"model": {"w": np.ones(2, np.float32)}}, "step": steps // 2, "config": {}}
    pickle.dump(fin, open(out / f"ckpt_{steps//2:06d}.pkl", "wb"))
    last = {"state": {"model": {"w": (np.full(2, np.nan, np.float32) if is_nan else np.ones(2, np.float32))}},
            "step": steps, "config": {}}
    pickle.dump(last, open(out / f"ckpt_{steps:06d}.pkl", "wb"))
    pickle.dump(last, open(out / "ckpt_latest.pkl", "wb"))
    (out / "config.json").write_text("{}")
    sys.exit(1 if is_nan else 0)

if tool.endswith("eval_sudoku_extreme.py"):
    if "--merge" in argv:
        d = Path(flag("--merge"))
        recs = sorted(d.glob("records_s*.npz"))
        n = sum(int(np.load(p)["n"]) for p in recs) if recs else 0
        (d / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": .3, "exact_acc_vote": .8}))
        np.savez(d / "records_all.npz", n=np.asarray(n))
        sys.exit(0)
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    shard = flag("--shard")
    sub = flag("--subsample"); strat = flag("--stratified")
    n_total = int(sub) if sub else (int(strat) if strat else 422786)
    if shard:
        i, K = map(int, shard.split("/"))
        n = n_total // K + (1 if i < n_total % K else 0)
        np.savez(out / f"records_s{i}.npz", n=np.asarray(n))
        (out / f"summary_s{i}.json").write_text(json.dumps({"n": n}))
    else:
        np.savez(out / "records_all.npz", n=np.asarray(n_total))
        (out / "summary_all.json").write_text(json.dumps(
            {"n": n_total, "exact_acc": .3, "exact_acc_vote": .85, "b1_exact": .4,
             "vote_at_k": {"128": .85}, "t1r_at_k": {"128": .5}}))
    sys.exit(0)

if tool.endswith("select_ckpt.py"):
    print("000050 0.3000 50"); sys.exit(0)
sys.exit(0)
PYEOF
  chmod +x "$SB/bin/stubpy"
}

run_chain () {  # W NW [extra VAR=val...] — env(1), because "$@"-expanded
  # assignments are NOT prefix-assignments in bash (found by this harness).
  local w=$1 nw=$2; shift 2
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" \
     CHAIN_WORKER=$w CHAIN_WORKERS=$nw \
     NCHIP_OVERRIDE=4 C0_STEPS=100 C0_WAIT_PASSES=3 C0_POLL_SLEEP=1 "$@" \
     bash tools/chain_sportC0.sh > "$SB/w${w}.log" 2>&1)
}

echo "== S1 fresh 1x8 =="
mk_sandbox
run_chain 0 1
grep -q "CHAIN-SPORTC0-COMPLETE" "$SB/w0.log" && ok "S1 complete sentinel" || { bad "S1 sentinel"; tail -5 "$SB/w0.log"; }
[ -f "$SB/gcs/sportC0/P3_STAGEA_OK" ] && ok "S1 two-stage stage-A marker" || bad "S1 stage-A marker"
[ -f "$SB/gcs/sportC0/sportC0_final.tgz" ] && ok "S1 final tgz" || bad "S1 final tgz"
n_ok=$(ls "$SB/gcs/sportC0/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' ')
[ "$n_ok" = 7 ] && ok "S1 7/7 arm markers" || bad "S1 arm markers ($n_ok)"
ls "$SB/gcs/sportC0/evals/" | grep -q "rider_C3X_sel5k_OK" && ok "S1 canvas rider ran" || bad "S1 rider"
SB1=$SB

echo "== S2 idempotent rerun =="
(cd "$SB1/repo" && PATH="$SB1/bin:$PATH" CHAIN_PY="$SB1/bin/stubpy" CHAIN_WORKER=0 CHAIN_WORKERS=1 \
   NCHIP_OVERRIDE=4 C0_STEPS=100 C0_WAIT_PASSES=3 C0_POLL_SLEEP=1 bash tools/chain_sportC0.sh > "$SB1/re.log" 2>&1)
grep -q "CHAIN-SPORTC0-COMPLETE" "$SB1/re.log" && ok "S2 complete again" || bad "S2 complete"
n_skip=$(grep -c "PRETRAIN-SKIP" "$SB1/re.log")
[ "$n_skip" = 7 ] && ok "S2 all pretrains skipped" || bad "S2 skips ($n_skip)"

echo "== S3 NaN one-shot amputation (single-stage P2; stage-A P3a) =="
mk_sandbox
run_chain 0 1 STUB_NAN_ARM=P2
grep -q "PRETRAIN-NAN P2" "$SB/w0.log" && ok "S3 NaN detected" || bad "S3 NaN detect"
grep -q "STOPPED final step 50" "$SB/repo/runs/pretrainsportC0_P2/STOPPED.txt" 2>/dev/null \
  && ok "S3 amputated to last finite grid" || bad "S3 amputation"
grep -q "CHAIN-SPORTC0-COMPLETE" "$SB/w0.log" && ok "S3 completes despite stop" || bad "S3 complete"
mk_sandbox
run_chain 0 1 STUB_NAN_ARM=P3a
grep -q "amputate + STOP" "$SB/w0.log" && ok "S3b stage-A death -> STOP (no stage-B rescue)" || bad "S3b stage-A stop"
[ -f "$SB/repo/runs/pretrainsportC0_P3/STOPPED.txt" ] && ok "S3b STOPPED label on final dir" || bad "S3b label"
if grep -q "INIT-FROM\|stage B" "$SB/repo/runs/pretrainsportC0_P3.log" 2>/dev/null; then bad "S3b stage B ran"; else ok "S3b stage B never ran"; fi

echo "== S4 fresh 4x4 static map =="
mk_sandbox
for w in 0 1 2 3; do run_chain $w 4 & done; wait
if grep -q "CHAIN-SPORTC0-COMPLETE" "$SB"/w*.log; then ok "S4 complete from a worker"; else bad "S4 complete"; tail -3 "$SB"/w*.log; fi
n_ok=$(ls "$SB/gcs/sportC0/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' ')
[ "$n_ok" = 7 ] && ok "S4 7/7 arms across the static map" || bad "S4 arms ($n_ok)"

echo "== S5 two-stage ordering =="
grep -q "P3_STAGEA_OK" <(ls "$SB/gcs/sportC0/") && ok "S5 stage-A marker exists" || bad "S5 stage-A marker"

echo
echo "harness: $PASS PASS / $FAIL FAIL"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
