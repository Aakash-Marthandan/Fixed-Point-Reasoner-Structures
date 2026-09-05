#!/bin/bash
# Ledger: FINAL PHASE Night A offline stub harness (2026-09-05; the house law: no chain launches without an
# end-to-end offline pass; the 2026-09-04 lesson: a harness adapted from the previous campaign is unverified
# until RUN, and every negative scenario asserts the staged failure fired). Stubs: gsutil/gcloud -> a local
# FAKE_GCS dir; $CHAIN_PY -> a stub emulating pretrain.py / eval_sudoku_extreme.py / explosion_census.py /
# stall_calibration.py / select_ckpt.py; REAL python still runs the chain's own helpers (nan_check / amputate /
# n-gates). Scenarios: S1 fresh 1x8 -> COMPLETE (6 single-stage arms; fulls vsel+final+alt at D16 + the D64
# depth row on EVERY arm; scans on every arm; no retfm; census vsel+final; calib on every arm; EMA headline);
# S1r the ARM REGISTRY reaches the trainer (A3 --cell dec --dec-width 256 without digit aug; A4 with it; A1/A5
# --fpa-k 1; A2/A5 --trm-ri-sigma 1.0; A0 --seed 1; --grid-every on every arm); S2 idempotent rerun; S3 NaN
# one-shot (A2) + the trainer's NAN-ABORT rc=3 (A1) + S3d post-death screens never run; S4 fresh 4x4 static map;
# S5 select_ckpt failure -> VB-FALLBACK-FINAL; S6 banked pretrain, no local dir -> re-pull; S9 launch-time HBM
# OOM -> ONE --remat retry (A3) and the always-OOM negative (no ARM_OK, INCOMPLETE); S10/S10b/S10c live bank.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export REAL_PY="$REPO/.venv/bin/python3"
PASS=0; FAIL=0
ok () { PASS=$((PASS+1)); echo "  PASS  $1"; }
bad () { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }

mk_sandbox () {
  SB=$(mktemp -d /tmp/hfa.XXXXXX)
  mkdir -p "$SB/repo/tools" "$SB/repo/runs" "$SB/repo/data/sudoku_extreme" "$SB/gcs/finalA" "$SB/bin"
  cp "$REPO/tools/chain_final.sh" "$REPO/tools/live_bank.sh" "$SB/repo/tools/"
  : > "$SB/repo/data/sudoku_extreme/sudoku_extreme_seed0.npz"
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
  ls)   rc=1; for g in "${args[@]:1}"; do p=$(map "$g"); for f in $p; do [ -e "$f" ] && { echo "$f" | sed "s|$GB/|gs://qhrrn2-rescue/|"; rc=0; }; done; done; exit $rc;;
  cp)   src=${args[1]}; dst=${args[2]}
        if [ "$src" = "-" ]; then p=$(map "$dst"); mkdir -p "$(dirname "$p")"; cat > "$p"
        elif [[ "$src" == gs://* ]]; then p=$(map "$src"); [ -f "$p" ] && { mkdir -p "$(dirname "$dst")" 2>/dev/null; cp "$p" "$dst"; } || exit 1
        else p=$(map "$dst"); mkdir -p "$(dirname "$p")"; cp "$src" "$p"; fi;;
  rsync) x=""; pos=(); i=1
         while [ $i -lt ${#args[@]} ]; do a=${args[$i]}
           case $a in -x) i=$((i+1)); x=${args[$i]};; -r|-C|-n|-d|-c) ;; *) pos+=("$a");; esac; i=$((i+1)); done
         src=$(map "${pos[0]}"); dst=$(map "${pos[1]}")
         [ -d "$src" ] || exit 1
         case "${pos[1]}" in gs://*) : ;; *) [ -d "$dst" ] || { echo "CommandException: arg ($dst) does not name a directory, bucket, or bucket subdir." >&2; exit 1; };; esac
         ( cd "$src" && find . -type f | sed 's|^\./||' ) | while read -r rel; do
           if [ -n "$x" ] && printf '%s\n' "$rel" | grep -qE "$x"; then continue; fi
           s="$src/$rel"; d="$dst/$rel"
           if [ ! -f "$d" ] || ! cmp -s "$s" "$d"; then mkdir -p "$(dirname "$d")"; cp "$s" "$d"; echo "Copying file://$rel"; fi
         done; exit 0;;
  *) exit 0;;
esac
SH
  chmod +x "$SB/bin/gsutil"
  printf '#!/bin/bash\necho "gcloud-stub $*"; exit 0\n' > "$SB/bin/gcloud"; chmod +x "$SB/bin/gcloud"
  cat > "$SB/bin/stubpy" <<'PYEOF'
#!/usr/bin/env python3
import json, os, pickle, sys
from pathlib import Path
import numpy as np
argv = sys.argv[1:]
tool = argv[0] if argv else ""
def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

if tool.endswith("pretrain.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    steps = int(flag("--steps", "100"))
    base = out.name.replace("pretrainfinalA_", "").replace("preflightfinalA_", "")
    if "preflight" in out.name and base == os.environ.get("STUB_PREFLIGHT_FAIL", ""):
        print("XLA compile error: staged preflight failure", file=sys.stderr); sys.exit(1)
    latest = out / "ckpt_latest.pkl"   # S10: the trainer resumes from a local ckpt_latest (and crashes on a torn one)
    if latest.exists():
        try:
            prev = pickle.load(open(latest, "rb")); print(f"RESUMED from {latest} at step {int(prev['step'])}", flush=True)
        except Exception as e:
            print(f"ckpt_latest unloadable: {e}", file=sys.stderr); sys.exit(1)
    if base == os.environ.get("STUB_OOM_ARM", "") and ("--remat" not in argv or os.environ.get("STUB_OOM_ALWAYS") == "1"):
        # S9: a LAUNCH-TIME HBM exhaustion — nothing logged, rc 1 (NOT a NaN death); the --remat retry succeeds
        # unless STUB_OOM_ALWAYS=1 (S9b: the arm OOMs even with --remat, e.g. one already launched on it)
        print("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 17179869184 bytes.", file=sys.stderr); sys.exit(1)
    arm_run = out.name.startswith("pretrainfinalA_")      # a mid-training NaN is staged in the ARM run only (a 60-step preflight never sees a step-25k death)
    is_nan = arm_run and base == os.environ.get("STUB_NAN_ARM", "")
    is_abort = arm_run and base == os.environ.get("STUB_NANABORT_ARM", "")
    cell = flag("--cell", "rg"); key = "val_t16" if cell in ("trm", "dec") else "val_t64"
    if "--ema" in argv: key_ema = key + "_ema"
    rows = []
    for s in (steps // 2, steps):
        loss = float("nan") if ((is_nan or is_abort) and s == steps) else 0.5
        rows.append(json.dumps({"step": s, "loss": loss, "ce_in": .04, "I_total": 1e5, "A_total": 5.0, "rule_H": 0.0, "lr": 1e-3, "steps_per_sec": 99.0, "t": "T"}))
        mon = {"step": s, key: 0.3, "ret_final_t8": 1.0, "ret_sched_t8": 1.0, "eta": 0.85, "lam_joint_max": 0.9}
        if "--ema" in argv: mon[key_ema] = 0.31
        rows.append(json.dumps({"monitor": mon}))
    (out / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    poisoned = is_nan or is_abort
    def grid(step):
        bad = poisoned and step > steps // 2          # the NaN started mid-run: every later grid is poisoned
        w = np.full(2, np.nan, np.float32) if bad else np.ones(2, np.float32)
        return {"state": {"model": {"w": w}}, "step": step, "config": {}}
    for st in sorted({steps // 2, steps} | {k for k in range(5000, steps + 1, 5000)}):
        pickle.dump(grid(st), open(out / f"ckpt_{st:06d}.pkl", "wb"))
    pickle.dump(grid(steps), open(out / "ckpt_latest.pkl", "wb"))
    (out / "config.json").write_text(json.dumps({"stub": True, "argv": argv}))
    if is_abort:
        (out / "NAN_ABORT.txt").write_text(f"NAN-ABORT at step {steps}\n"); sys.exit(3)
    sys.exit(1 if is_nan else 0)

if tool.endswith("eval_sudoku_extreme.py"):
    if "--merge" in argv:
        d = Path(flag("--merge"))
        recs = sorted(d.glob("records_s*.npz"))
        n = sum(int(np.load(p)["n"]) for p in recs) if recs else 0
        (d / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4}))
        np.savez(d / "records_all.npz", n=np.asarray(n)); sys.exit(0)
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    shard = flag("--shard"); sub = flag("--subsample"); strat = flag("--stratified")
    n_total = int(sub) if sub else (int(strat) if strat else 422786)
    prov = {"t_total": int(flag("--t-total", "64")), "ema": "--ema" in argv, "argv": argv}
    if shard:
        i, K = map(int, shard.split("/")); n = n_total // K + (1 if i < n_total % K else 0)
        np.savez(out / f"records_s{i}.npz", n=np.asarray(n)); (out / f"summary_s{i}.json").write_text(json.dumps({"n": n, **prov}))
    else:
        np.savez(out / "records_all.npz", n=np.asarray(n_total))
        (out / "summary_all.json").write_text(json.dumps({"n": n_total, "exact_acc": .3, "exact_acc_vote": .85, "b1_exact": .4,
                                                          "vote_at_k": {"128": .85}, "t1r_at_k": {"128": .5}, **prov}))
    sys.exit(0)

if tool.endswith("stall_calibration.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "calib.json").write_text(json.dumps({"ckpt": flag("--ckpt"), "topk_correct_stalled": 0.8, "n": 512, "ema": "--ema" in argv}))
    sys.exit(0)

if tool.endswith("explosion_census.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "census.json").write_text(json.dumps({"rows": [{"t": 64, "exploded_frac": 0.0, "n": 512}, {"t": 256, "exploded_frac": 0.0, "n": 64}], "ema": "--ema" in argv}))
    sys.exit(0)

if tool.endswith("select_ckpt.py"):
    d = Path(argv[1]); arm = d.name.replace("pretrainfinalA_", "")
    if arm in os.environ.get("STUB_SELECT_FAIL", "").split(","): print("NONE", file=sys.stderr); sys.exit(1)
    if not (d / "metrics.jsonl").exists(): print("NONE", file=sys.stderr); sys.exit(1)
    banked = sorted(d.glob("ckpt_0*.pkl"))
    if not banked: print("NONE", file=sys.stderr); sys.exit(1)
    st = banked[0].name[5:11]
    val = os.environ.get(f"STUB_SELECT_VAL_{arm}", "0.3000")
    print(f"{st} {val} {int(st)}"); sys.exit(0)
sys.exit(0)
PYEOF
  chmod +x "$SB/bin/stubpy"
  echo "  (sandbox $SB)"
}

run_chain () {  # W NW [extra VAR=val...]
  local w=$1 nw=$2; shift 2
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" \
     CHAIN_WORKER=$w CHAIN_WORKERS=$nw NCHIP_OVERRIDE=4 C1_STEPS_X=100 \
     C1_WAIT_PASSES=3 C1_POLL_SLEEP=1 LIVE_NO_GUARD=1 "$@" bash tools/chain_final.sh > "$SB/w${w}.log" 2>&1)
}

echo "== S1 fresh 1x8 =="
mk_sandbox; run_chain 0 1
grep -q "CHAIN-FINALA-COMPLETE" "$SB/w0.log" && ok "S1 complete sentinel" || { bad "S1 sentinel"; tail -8 "$SB/w0.log"; }
n_ok=$(ls "$SB/gcs/finalA/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '); [ "$n_ok" = 6 ] && ok "S1 6/6 core arm markers (A6 not run on the 8-shape)" || bad "S1 arm markers ($n_ok)"
! grep -q "PRETRAIN-START A6" "$SB/w0.log" && [ ! -f "$SB/gcs/finalA/A6_ARM_OK" ] && ok "S1 A6 never runs on the 8-shape" || bad "S1 A6 ran on the 8-shape"
[ "$(grep -c 'PREFLIGHT-OK' "$SB/w0.log")" = 6 ] && grep -q "PREFLIGHT-OK A3 99.0 it/s" "$SB/w0.log" && ok "S1 preflight of all six arms before any arm (pace read)" || { bad "S1 preflight"; grep PREFLIGHT "$SB/w0.log" | head -3; }
first_pf=$(grep -n "PREFLIGHT-OK" "$SB/w0.log" | tail -1 | cut -d: -f1); first_arm=$(grep -n "PRETRAIN-START" "$SB/w0.log" | head -1 | cut -d: -f1); [ "$first_pf" -lt "$first_arm" ] && ok "S1 every preflight precedes the first arm" || bad "S1 preflight order"
[ -f "$SB/gcs/finalA/finalA_final.tgz" ] && ok "S1 final tgz" || bad "S1 final tgz"
allc=1; for a in A0 A1 A2 A3 A4 A5; do [ -f "$SB/gcs/finalA/evals/census_${a}_vsel_OK" ] && [ -f "$SB/gcs/finalA/evals/census_${a}_final_OK" ] || allc=0; done; [ $allc = 1 ] && ok "S1 census vsel+final on every arm" || bad "S1 census"
allf=1; for a in A0 A1 A2 A3 A4 A5; do for r in full_${a}_vsel_t16 full_${a}_final_t16 full_${a}_vsel_t16_alt full_${a}_vsel_t64; do [ -f "$SB/gcs/finalA/evals/${r}_OK" ] || allf=0; done; done; [ $allf = 1 ] && ok "S1 fulls vsel+final+alt at D16 + the D64 depth row on every arm" || bad "S1 fulls"
alls=1; for a in A0 A1 A2 A3 A4 A5; do [ -f "$SB/gcs/finalA/evals/scan_${a}_OK" ] && [ -f "$SB/gcs/finalA/evals/calib_${a}_vsel_OK" ] || alls=0; done; [ $alls = 1 ] && ok "S1 scan + calib on every arm" || bad "S1 scans/calib"
[ -z "$(ls "$SB/gcs/finalA/evals/" | grep retfm)" ] && ok "S1 no retfm (field-loop cells)" || bad "S1 retfm ran"
eargv () { "$REAL_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('argv', [])))" "$1"; }
sa0=$(eargv "$SB/repo/runs/sxscan_pfinalAA0/summary_s0.json"); sa3=$(eargv "$SB/repo/runs/sxscan_pfinalAA3/summary_s0.json")
echo "$sa0" | grep -q -- "--subsample 20000 --t-total 64 --k-init 128" && echo "$sa3" | grep -q -- "--subsample 5000 --t-total 64 --k-init 32" && ok "S1 scans: X0-class 20k x k128, wide 5k x k32 (labeled)" || bad "S1 scan classes: A0 [$sa0] A3 [$sa3]"
d0=$(eargv "$SB/repo/runs/sxeval_pfinalAA0/full_vsel_t64/summary_s0.json"); d3=$(eargv "$SB/repo/runs/sxeval_pfinalAA3/full_vsel_t64/summary_s0.json")
! echo "$d0" | grep -q -- "--subsample" && echo "$d3" | grep -q -- "--subsample 100000" && ok "S1 D64 row: full for X0-class, 100k for the wide arms" || bad "S1 D64 classes"
v3=$(eargv "$SB/repo/runs/sxeval_pfinalAA3/full_vsel_t16/summary_s0.json"); a3=$(eargv "$SB/repo/runs/sxeval_pfinalAA3/full_vsel_t16_alt/summary_s0.json")
! echo "$v3" | grep -q -- "--subsample" && echo "$a3" | grep -q -- "--subsample 50000" && ok "S1 wide arm: headline D16 vsel full on 422,786; alt on 50k" || bad "S1 wide fulls"
n3=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/sxscan_pfinalAA3/summary_all.json'))['n'])"); [ "$n3" = 5000 ] && ok "S1 wide scan n-gate 5000" || bad "S1 wide scan n ($n3)"
grep -q '"ema": true' "$SB/repo/runs/sxeval_pfinalAA3/full_vsel_t16/summary_s0.json" && ok "S1 EMA headline" || bad "S1 headline weights"
grep -q "screen_A3_vb_OK" <(ls "$SB/gcs/finalA/evals/") && ok "S1 vb screen on every arm (fixed-step screens need 15k grids: asserted in S3d)" || bad "S1 screens"
grep -q "PRETRAIN-START A3 .*prec=default" "$SB/w0.log" && ok "S1 bf16 precision on the field arms" || bad "S1 precision"
[ -z "$(grep -E 'STAGEA|stage A|stage B' "$SB/w0.log")" ] && ok "S1 no two-stage machinery ran" || bad "S1 two-stage leak"
echo "== S1r the ARM REGISTRY reaches the trainer (the 2026-09-04 PM-5 lesson: arm names/flags verified by the run) =="
argv () { "$REAL_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['argv']))" "$SB/repo/runs/pretrainfinalA_$1/config.json"; }
a3=$(argv A3); a4=$(argv A4); a1=$(argv A1); a2=$(argv A2); a5=$(argv A5); a0=$(argv A0)
echo "$a3" | grep -q -- "--cell dec --dec-width 384" && ! echo "$a3" | grep -q -- "--sudoku-digit-aug" && ok "S1r A3 = DEC w384 WITHOUT digit aug" || bad "S1r A3 flags: $a3"
echo "$a4" | grep -q -- "--cell dec" && echo "$a4" | grep -q -- "--sudoku-digit-aug" && ok "S1r A4 = DEC + digit aug" || bad "S1r A4 flags"
echo "$a1" | grep -q -- "--cell trm --trm-hidden 512 --sudoku-digit-aug" && echo "$a1" | grep -q -- "--fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25" && ok "S1r A1 = X0 + FPA anchor rows" || bad "S1r A1 flags"
echo "$a2" | grep -q -- "--trm-ri-sigma 1.0" && ! echo "$a2" | grep -q -- "--fpa-k 1" && ok "S1r A2 = X0 + RI only" || bad "S1r A2 flags"
echo "$a5" | grep -q -- "--cell dec" && echo "$a5" | grep -q -- "--fpa-k 1" && echo "$a5" | grep -q -- "--trm-ri-sigma 1.0" && ! echo "$a5" | grep -q -- "--sudoku-digit-aug" && ok "S1r A5 = DEC + FPA + RI, no digit aug" || bad "S1r A5 flags"
echo "$a0" | grep -q -- "--seed 1" && echo "$a1" | grep -q -- "--seed 0" && ok "S1r A0 seed 1, A1 seed 0 (the noise pair vs sportC1's X0 seed 0)" || bad "S1r seeds"
for a in A0 A3; do echo "$(argv $a)" | grep -q -- "--grid-every 2000" && echo "$(argv $a)" | grep -q -- "--monitor-every 2000" && echo "$(argv $a)" | grep -q -- "--ckpt-every 500" || { bad "S1r cadences $a"; break; }; done; ok "S1r grids at the monitor cadence (2000), ckpt_latest every 500 steps (the 5-min bank)"
SB1=$SB

echo "== S2 idempotent rerun =="
(cd "$SB1/repo" && env PATH="$SB1/bin:$PATH" CHAIN_PY="$SB1/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 C1_STEPS_X=100 C1_WAIT_PASSES=3 C1_POLL_SLEEP=1 LIVE_NO_GUARD=1 bash tools/chain_final.sh > "$SB1/re.log" 2>&1)
grep -q "CHAIN-FINALA-COMPLETE" "$SB1/re.log" && ok "S2 complete again" || bad "S2 complete"
n_skip=$(grep -c "PRETRAIN-SKIP" "$SB1/re.log"); [ "$n_skip" = 6 ] && ok "S2 all pretrains skipped" || bad "S2 skips ($n_skip)"
grep -q "PREFLIGHT-SKIP" "$SB1/re.log" && ok "S2 preflight idempotent" || bad "S2 preflight reran"
[ "$(grep -c 'EVAL-OK\|CENSUS-OK\|CALIB-OK' "$SB1/re.log")" = 0 ] && ok "S2 no re-evals" || bad "S2 re-evals ran"

echo "== S3 NaN one-shot (A2); the trainer's NAN-ABORT rc=3 (A1); S3d post-death screens never run =="
mk_sandbox; run_chain 0 1 STUB_NAN_ARM=A2
grep -q "PRETRAIN-NAN A2" "$SB/w0.log" && grep -q "STOPPED final step 50" "$SB/repo/runs/pretrainfinalA_A2/STOPPED.txt" && ok "S3 A2 amputated to the last finite grid" || bad "S3 A2 amputation"
[ ! -f "$SB/repo/runs/pretrainfinalA_A2/ckpt_000100.pkl" ] && ok "S3 post-death grid removed" || bad "S3 post-death grid kept"
grep -q "CHAIN-FINALA-COMPLETE" "$SB/w0.log" && ok "S3 completes despite the stop" || bad "S3 complete"
mk_sandbox; run_chain 0 1 STUB_NANABORT_ARM=A1
grep -q "PRETRAIN-NAN A1 (rc=3)" "$SB/w0.log" && [ -f "$SB/repo/runs/pretrainfinalA_A1/STOPPED.txt" ] && ok "S3c trainer NAN-ABORT rc=3 -> amputation" || bad "S3c nan-abort path"
grep -q "CHAIN-FINALA-COMPLETE" "$SB/w0.log" && ok "S3c complete" || bad "S3c complete"
mk_sandbox; run_chain 0 1 STUB_NAN_ARM=A2 C1_STEPS_X=40000
grep -q "AMPUTATED to ckpt_020000.pkl step 20000" "$SB/w0.log" && ok "S3d amputated to the last finite 5k grid (20000)" || { bad "S3d amputation"; grep -E 'AMPUTAT|STOPPED' "$SB/w0.log" | head -3; }
[ -f "$SB/gcs/finalA/evals/screen_A2_s015000_OK" ] && [ ! -f "$SB/gcs/finalA/evals/screen_A2_s035000_OK" ] && ok "S3d s015000 screened, s035000 (post-death) never ran" || bad "S3d post-death screen"

echo "== S4 fresh 4x4 static map =="
mk_sandbox; for w in 0 1 2 3; do run_chain $w 4 & done; wait
if grep -q "CHAIN-FINALA-COMPLETE" "$SB"/w*.log; then ok "S4 complete from a worker"; else bad "S4 complete"; tail -3 "$SB"/w*.log; fi
n_ok=$(ls "$SB/gcs/finalA/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '); [ "$n_ok" = 7 ] && ok "S4 7/7 arms across the static map (A6 on the 16)" || bad "S4 arms ($n_ok)"
grep -q "PRETRAIN-START A3" "$SB/w0.log" && ! grep -q "PRETRAIN-START A0" "$SB/w0.log" && grep -q "PRETRAIN-START A4" "$SB/w1.log" && grep -q "PRETRAIN-START A5" "$SB/w2.log" && grep -q "PRETRAIN-START A6" "$SB/w3.log" && grep -q "PRETRAIN-START A0" "$SB/w3.log" && grep -q "PRETRAIN-START A2" "$SB/w3.log" && ok "S4 map (13:30Z re-map): A3 w0 · A4 w1 · A5 w2 · A6 A0 A1 A2 w3" || bad "S4 map"
a6=$("$REAL_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['argv']))" "$SB/repo/runs/pretrainfinalA_A6/config.json")
echo "$a6" | grep -q -- "--cell dec" && echo "$a6" | grep -q -- "--dec-width 512" && echo "$a6" | grep -q -- "--remat" && ! echo "$a6" | grep -q -- "--sudoku-digit-aug" && ok "S4r A6 = DEC w512, remat, no digit aug" || bad "S4r A6 flags: $a6"
grep -q "PREFLIGHT-OK A6" "$SB/w3.log" && [ -f "$SB/gcs/finalA/PREFLIGHT_OK_w3_nw4" ] && ok "S4 A6 preflighted on its own worker (shape-specific marker)" || bad "S4 A6 preflight"
[ -f "$SB/gcs/finalA/evals/full_A6_vsel_t16_OK" ] && [ -f "$SB/gcs/finalA/evals/scan_A6_OK" ] && ok "S4 A6 full battery" || bad "S4 A6 battery"

echo "== S4b fresh 8x4 static map (the v6e-32): one wide arm per worker, the seed pairs on w4/w5 =="
mk_sandbox; for w in 0 1 2 3 4 5 6 7; do run_chain $w 8 & done; wait
if grep -q "CHAIN-FINALA-COMPLETE" "$SB"/w*.log; then ok "S4b complete from a worker"; else bad "S4b complete"; tail -3 "$SB"/w*.log; fi
n_ok=$(ls "$SB/gcs/finalA/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '); [ "$n_ok" = 9 ] && ok "S4b 9/9 arms across the 8x4 map" || bad "S4b arms ($n_ok)"
grep -q "PRETRAIN-START A7" "$SB/w4.log" && grep -q "PRETRAIN-START A8" "$SB/w5.log" && grep -q "PRETRAIN-START A0" "$SB/w6.log" && grep -q "PRETRAIN-START A1" "$SB/w6.log" && grep -q "PRETRAIN-START A2" "$SB/w7.log" && ok "S4b map: A7 w4 · A8 w5 · A0+A1 w6 · A2 w7" || bad "S4b map"
a7=$(argv A7); a8=$(argv A8)
echo "$a7" | grep -q -- "--cell dec --dec-width 384" && echo "$a7" | grep -q -- "--seed 1" && ! echo "$a7" | grep -q -- "--sudoku-digit-aug" && ok "S4b A7 = A3's flags at seed 1" || bad "S4b A7 flags: $a7"
echo "$a8" | grep -q -- "--dec-width 512" && echo "$a8" | grep -q -- "--remat" && echo "$a8" | grep -q -- "--seed 1" && ok "S4b A8 = A6's flags at seed 1" || bad "S4b A8 flags: $a8"
sa7=$(eargv "$SB/repo/runs/sxscan_pfinalAA7/summary_s0.json"); echo "$sa7" | grep -q -- "--subsample 5000 --t-total 64 --k-init 32" && ok "S4b the seed arms get the wide battery" || bad "S4b A7 battery"
echo "== S4c the 16 does NOT run the seed arms (A7/A8 absent, the night completes with 7) =="
mk_sandbox; for w in 0 1 2 3; do run_chain $w 4 & done; wait
n_ok=$(ls "$SB/gcs/finalA/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '); [ "$n_ok" = 7 ] && ! grep -q "PRETRAIN-START A7\|PRETRAIN-START A8" "$SB"/w*.log && grep -q "CHAIN-FINALA-COMPLETE" "$SB"/w*.log && ok "S4c 7 arms on the 16, A7/A8 never run, complete" || bad "S4c ($n_ok)"

echo "== S12 DEMOTION 32 -> 16 with arms IN FLIGHT: A3 (w0) and A7 (w4) live-banked at step 40; the 16 resumes A3, never starts A7, re-preflights on its shape, completes with 7 =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && for a in A3 A7; do env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_$a --steps 40 --ema 0.999 --cell dec >/dev/null 2>&1
    mkdir -p "$SB/gcs/finalA/live/runs/pretrainfinalA_$a"; cp runs/pretrainfinalA_$a/* "$SB/gcs/finalA/live/runs/pretrainfinalA_$a/"; done; rm -rf runs
  echo ok > "$SB/gcs/finalA/PREFLIGHT_OK_w0_nw8" )   # the 32's own preflight marker must NOT satisfy the 16
for w in 0 1 2 3; do run_chain $w 4 & done; wait
grep -q "RESUMED from runs/pretrainfinalA_A3/ckpt_latest.pkl at step 40" "$SB/repo/runs/pretrainfinalA_A3.log" && ok "S12 A3 resumed from the live bank on the 16" || { bad "S12 A3 resume"; head -3 "$SB/repo/runs/pretrainfinalA_A3.log" 2>/dev/null; }
! grep -q "PRETRAIN-START A7" "$SB"/w*.log && [ ! -f "$SB/gcs/finalA/A7_ARM_OK" ] && [ -f "$SB/gcs/finalA/live/runs/pretrainfinalA_A7/ckpt_latest.pkl" ] && ok "S12 A7 not started on the 16; its live state left banked" || bad "S12 A7 handling"
grep -q "PREFLIGHT-OK A3" "$SB/w0.log" && ! grep -q "PREFLIGHT-SKIP" "$SB/w0.log" && [ -f "$SB/gcs/finalA/PREFLIGHT_OK_w0_nw4" ] && ok "S12 the 16 re-preflighted its shape (the 32's marker did not satisfy it)" || { bad "S12 preflight on the new shape"; grep PREFLIGHT "$SB/w0.log" | head -3; }
grep -q "COMPLETION-SET nw=4 need=\[A0 A1 A2 A3 A4 A5 A6\] optional-not-awaited=\[A7 A8 \]" "$SB"/w*.log && grep -q "CHAIN-FINALA-COMPLETE" "$SB"/w*.log && [ "$(ls "$SB/gcs/finalA/"*_ARM_OK | wc -l | tr -d ' ')" = 7 ] && ok "S12 completes with the 16's seven arms, the seed arms named as not awaited" || { bad "S12 completion"; grep COMPLETION "$SB"/w*.log | head -2; }

echo "== S13 RESUME after a remat-needing OOM (the 14:25Z defect): the arm dir holds loss rows + RETRY_REMAT.txt; the relaunch must start WITH remat and RESUME, never amputate =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_A3 --steps 40 --ema 0.999 --cell dec >/dev/null 2>&1
  echo "OOM at launch (rc=1) -> retried once with --remat" > runs/pretrainfinalA_A3/RETRY_REMAT.txt
  mkdir -p "$SB/gcs/finalA/live/runs/pretrainfinalA_A3"; cp runs/pretrainfinalA_A3/* "$SB/gcs/finalA/live/runs/pretrainfinalA_A3/"; rm -rf runs )
run_chain 0 1 STUB_OOM_ARM=A3
grep -q "REMAT-PERSISTED A3" "$SB/w0.log" && grep -q "RESUMED from runs/pretrainfinalA_A3/ckpt_latest.pkl at step 40" "$SB/repo/runs/pretrainfinalA_A3.log" && ok "S13 remat persisted: the relaunch resumed at step 40 with --remat, no OOM" || { bad "S13 persisted remat"; grep -E 'REMAT|OOM|NAN|AMPUT' "$SB/w0.log" | head -4; }
! grep -q "PRETRAIN-NAN A3" "$SB/w0.log" && ! grep -q "AMPUTATE" "$SB/w0.log" && [ -f "$SB/gcs/finalA/A3_ARM_OK" ] && ok "S13 no NaN path, no amputation, A3 completes" || bad "S13 misclassification"
echo "== S13b RESUME with loss rows but NO remat record (e.g. a shape change): a compile-time OOM in THIS launch is an OOM (retry with remat), not a NaN death =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_A3 --steps 40 --ema 0.999 --cell dec >/dev/null 2>&1
  mkdir -p "$SB/gcs/finalA/live/runs/pretrainfinalA_A3"; cp runs/pretrainfinalA_A3/* "$SB/gcs/finalA/live/runs/pretrainfinalA_A3/"; rm -rf runs )
run_chain 0 1 STUB_OOM_ARM=A3
grep -q "PRETRAIN-OOM-RETRY-REMAT A3" "$SB/w0.log" && ! grep -q "PRETRAIN-NAN A3" "$SB/w0.log" && grep -q "RESUMED from runs/pretrainfinalA_A3/ckpt_latest.pkl at step 40" "$SB/repo/runs/pretrainfinalA_A3.log" && [ -f "$SB/gcs/finalA/A3_ARM_OK" ] && ok "S13b OOM on a resume -> remat retry -> resumed, completes" || { bad "S13b resume OOM path"; grep -E 'REMAT|OOM|NAN|AMPUT|RESUMED' "$SB/w0.log" "$SB/repo/runs/pretrainfinalA_A3.log" | head -5; }

echo "== S12b DEMOTION 16 -> 8: A3 banked (ARM_OK), A4 in flight at step 40 on the 16's w1; the 8's single worker skips A3, resumes A4, preflights the five unbanked arms, completes with 6 =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_A3 --steps 100 --ema 0.999 --cell dec >/dev/null 2>&1
  tar czf "$SB/gcs/finalA/A3_pretrain.tgz" runs/pretrainfinalA_A3 && cp runs/pretrainfinalA_A3/ckpt_latest.pkl "$SB/gcs/finalA/A3_ckpt.pkl"; echo ok > "$SB/gcs/finalA/A3_PRETRAIN_OK"; echo ok > "$SB/gcs/finalA/A3_ARM_OK"; rm -rf runs
  mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_A4 --steps 40 --ema 0.999 --cell dec >/dev/null 2>&1
  mkdir -p "$SB/gcs/finalA/live/runs/pretrainfinalA_A4"; cp runs/pretrainfinalA_A4/* "$SB/gcs/finalA/live/runs/pretrainfinalA_A4/"; rm -rf runs
  echo ok > "$SB/gcs/finalA/PREFLIGHT_OK_w1_nw4" )
run_chain 0 1
grep -q "RESUMED from runs/pretrainfinalA_A4/ckpt_latest.pkl at step 40" "$SB/repo/runs/pretrainfinalA_A4.log" && ok "S12b A4 resumed from the live bank on the 8" || { bad "S12b A4 resume"; head -3 "$SB/repo/runs/pretrainfinalA_A4.log" 2>/dev/null; }
grep -q "PRETRAIN-SKIP A3" "$SB/w0.log" && [ "$(grep -c 'PREFLIGHT-OK' "$SB/w0.log")" = 5 ] && ! grep -q "PREFLIGHT-OK A3" "$SB/w0.log" && ok "S12b banked A3 skipped everywhere; the five unbanked arms preflighted on the 8" || { bad "S12b skip/preflight"; grep -E 'PREFLIGHT|SKIP' "$SB/w0.log" | head -4; }
grep -q "CHAIN-FINALA-COMPLETE" "$SB/w0.log" && [ "$(ls "$SB/gcs/finalA/"*_ARM_OK | wc -l | tr -d ' ')" = 6 ] && ok "S12b completes with the 8's six arms" || bad "S12b completion"

echo "== S5 select_ckpt failure -> LOUD fallback =="
mk_sandbox; run_chain 0 1 STUB_SELECT_FAIL=A1
grep -q "VB-FALLBACK-FINAL A1" "$SB/w0.log" && grep -q "FALLBACK-FINAL" "$SB/repo/runs/pretrainfinalA_A1/val_best.txt" && ok "S5 fallback echoed + labeled" || bad "S5 fallback"
grep -q "FULL-FINAL A1 := vsel" "$SB/w0.log" && [ -f "$SB/repo/runs/sxeval_pfinalAA1/full_final_t16/summary_all.json" ] && ok "S5 final := vsel copy" || bad "S5 final copy"
grep -q "CHAIN-FINALA-COMPLETE" "$SB/w0.log" && ok "S5 complete" || bad "S5 complete"

echo "== S6 banked pretrain, no local dir -> re-pull before select =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_A0 --steps 100 --ema 0.999 --cell trm >/dev/null 2>&1
  tar czf "$SB/gcs/finalA/A0_pretrain.tgz" runs/pretrainfinalA_A0 && cp runs/pretrainfinalA_A0/ckpt_latest.pkl "$SB/gcs/finalA/A0_ckpt.pkl"
  echo ok > "$SB/gcs/finalA/A0_PRETRAIN_OK"; rm -rf runs/pretrainfinalA_A0 )
run_chain 0 1
grep -q "PRETRAIN-SKIP A0" "$SB/w0.log" && grep -q "PRETRAIN-RESTORE A0" "$SB/w0.log" && ok "S6 banked arm re-pulled before select" || bad "S6 re-pull"
grep -q "VALBEST A0" "$SB/w0.log" && ! grep -q "VB-FALLBACK-FINAL A0" "$SB/w0.log" && ok "S6 select_ckpt worked on the restored metrics" || bad "S6 select after restore"

echo "== S9 launch-time HBM OOM (no step logged) -> ONE retry with --remat (A3); the always-OOM negative =="
mk_sandbox; run_chain 0 1 STUB_OOM_ARM=A3
grep -q "PRETRAIN-OOM-RETRY-REMAT A3" "$SB/w0.log" && [ -f "$SB/repo/runs/pretrainfinalA_A3/RETRY_REMAT.txt" ] && ok "S9 OOM at launch -> retried once with --remat, labeled on disk" || { bad "S9 retry"; grep -E 'OOM|REMAT' "$SB/w0.log" | head -3; }
[ -f "$SB/gcs/finalA/A3_ARM_OK" ] && [ ! -f "$SB/repo/runs/pretrainfinalA_A3/STOPPED.txt" ] && ok "S9 A3 completed clean after the retry (no STOPPED label)" || bad "S9 A3 outcome"
grep -q "CHAIN-FINALA-COMPLETE" "$SB/w0.log" && ok "S9 complete" || bad "S9 complete"
mk_sandbox; run_chain 0 1 STUB_OOM_ARM=A3 STUB_OOM_ALWAYS=1
if grep -q "PRETRAIN-OOM-RETRY-REMAT pf-A3" "$SB/w0.log" && grep -q "PREFLIGHT-FAILED A3" "$SB/w0.log" && grep -q "FINALA-PREFLIGHT-ABORT" "$SB/w0.log" && [ ! -f "$SB/gcs/finalA/A3_ARM_OK" ] && ! grep -q "CHAIN-FINALA-COMPLETE" "$SB/w0.log" && ! grep -q "PRETRAIN-START" "$SB/w0.log"; then ok "S9b always-OOM: the preflight's own --remat retry fails too -> abort BEFORE any arm (stop-and-report, no silent ARM_OK)"; else bad "S9b always-OOM path"; grep -E 'OOM|PREFLIGHT|ABORT|COMPLETE' "$SB/w0.log" | head -4; fi

echo "== S10 LIVE BANK: in-flight state in the live prefix, FRESH node -> pulled no-clobber, pretrain RESUMES, eval partial in place, loop banks + cleans up =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_A3 --steps 40 --ema 0.999 --cell dec >/dev/null 2>&1
  mkdir -p "$SB/gcs/finalA/live/runs/pretrainfinalA_A3" "$SB/gcs/finalA/live/runs/sxscan_pfinalAA0"
  cp runs/pretrainfinalA_A3/* "$SB/gcs/finalA/live/runs/pretrainfinalA_A3/"; echo partial > "$SB/gcs/finalA/live/runs/sxscan_pfinalAA0/partial_s0.npz"; rm -rf runs )
run_chain 0 1
grep -q "LIVE-RESTORE pulled=" "$SB/w0.log" && ok "S10 live prefix pulled on a fresh node" || { bad "S10 pull"; grep LIVE "$SB/w0.log" | head -3; }
grep -q "RESUMED from runs/pretrainfinalA_A3/ckpt_latest.pkl at step 40" "$SB/repo/runs/pretrainfinalA_A3.log" && ok "S10 A3 RESUMED from the live ckpt (step 40)" || { bad "S10 resume"; head -3 "$SB/repo/runs/pretrainfinalA_A3.log"; }
[ -f "$SB/repo/runs/sxscan_pfinalAA0/partial_s0.npz" ] && ok "S10 eval partial restored into place" || bad "S10 partial"
grep -q "LIVE-BANK loop start" "$SB/w0.log" && grep -q "LIVE-BANK rc=0" "$SB/w0.log" && ok "S10 the 5-min loop started and banked" || { bad "S10 loop"; grep LIVE-BANK "$SB/w0.log" | head -3; }
sleep 1; [ ! -f "$SB/repo/runs/live_bank.pid" ] && ok "S10 loop stopped with the chain (pidfile gone)" || bad "S10 loop cleanup"
grep -q "CHAIN-FINALA-COMPLETE" "$SB/w0.log" && ok "S10 complete" || bad "S10 complete"

echo "== S10b LIVE BANK negative: a TORN live ckpt_latest (staged) -> LIVE-RESTORE-FALLBACK to the newest loadable grid =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_A3 --steps 40 --ema 0.999 --cell dec >/dev/null 2>&1
  mkdir -p "$SB/gcs/finalA/live/runs/pretrainfinalA_A3"; cp runs/pretrainfinalA_A3/* "$SB/gcs/finalA/live/runs/pretrainfinalA_A3/"
  head -c 1000 /dev/urandom > "$SB/gcs/finalA/live/runs/pretrainfinalA_A3/ckpt_latest.pkl"; rm -rf runs )
"$REAL_PY" -c "import pickle,sys; pickle.load(open(sys.argv[1],'rb'))" "$SB/gcs/finalA/live/runs/pretrainfinalA_A3/ckpt_latest.pkl" 2>/dev/null && bad "S10b staged torn ckpt is loadable (stage failed)" || ok "S10b the torn ckpt really is unloadable (staged failure verified)"
run_chain 0 1
grep -q "LIVE-RESTORE-FALLBACK runs/pretrainfinalA_A3 ckpt_latest unloadable -> ckpt_000040.pkl step 40" "$SB/w0.log" && ok "S10b torn ckpt_latest -> fallback to the newest loadable grid" || { bad "S10b fallback"; grep LIVE "$SB/w0.log" | head -3; }
grep -q "RESUMED from runs/pretrainfinalA_A3/ckpt_latest.pkl at step 40" "$SB/repo/runs/pretrainfinalA_A3.log" && ! grep -q "PRETRAIN-NAN A3" "$SB/w0.log" && ok "S10b A3 resumed at step 40, no NaN path" || bad "S10b resume"

echo "== S10c LIVE BANK negative: a BANKED arm's stale live copy (step 7) is NOT restored over its banked final (step 100) =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_A0 --steps 7 --ema 0.999 --cell trm >/dev/null 2>&1
  mkdir -p "$SB/gcs/finalA/live/runs/pretrainfinalA_A0"; cp runs/pretrainfinalA_A0/* "$SB/gcs/finalA/live/runs/pretrainfinalA_A0/"; rm -rf runs
  mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainfinalA_A0 --steps 100 --ema 0.999 --cell trm >/dev/null 2>&1
  tar czf "$SB/gcs/finalA/A0_pretrain.tgz" runs/pretrainfinalA_A0 && cp runs/pretrainfinalA_A0/ckpt_latest.pkl "$SB/gcs/finalA/A0_ckpt.pkl"
  echo ok > "$SB/gcs/finalA/A0_PRETRAIN_OK"; rm -rf runs )
[ -f "$SB/gcs/finalA/live/runs/pretrainfinalA_A0/ckpt_000007.pkl" ] && ok "S10c the stale live copy really is staged in the prefix" || bad "S10c stage"
run_chain 0 1
grep -q "banked arms excluded: .*pretrainfinalA_A0/" "$SB/w0.log" && grep -q "PRETRAIN-RESTORE A0 " "$SB/w0.log" && ok "S10c banked A0 excluded from the live pull; its tarball restored instead" || { bad "S10c exclusion"; grep -E 'LIVE|RESTORE' "$SB/w0.log" | head -4; }
st=$("$REAL_PY" -c "import pickle,sys; print(pickle.load(open(sys.argv[1],'rb'))['step'])" "$SB/repo/runs/pretrainfinalA_A0/ckpt_latest.pkl" 2>/dev/null)
[ "$st" = 100 ] && [ ! -f "$SB/repo/runs/pretrainfinalA_A0/ckpt_000007.pkl" ] && ok "S10c A0's local final is the banked step-100 grid; no stale step-7 files" || bad "S10c final grid (step=$st)"

echo "== S11 PREFLIGHT negatives: the optional arm's failure is SKIPPED (labeled) and the night completes; a core arm's failure aborts before any arm =="
mk_sandbox; for w in 0 1 2 3; do run_chain $w 4 STUB_PREFLIGHT_FAIL=A6 & done; wait
grep -q "PREFLIGHT-FAILED A6 .*SKIPPED" "$SB/w3.log" && [ -f "$SB/gcs/finalA/A6_SKIPPED" ] && ok "S11 A6 preflight failure -> SKIPPED marker, labeled" || { bad "S11 A6 skip"; grep PREFLIGHT "$SB/w3.log" | head -3; }
! grep -q "PRETRAIN-START A6" "$SB"/w*.log && [ ! -f "$SB/gcs/finalA/A6_ARM_OK" ] && ok "S11 A6 never trained" || bad "S11 A6 trained after a failed preflight"
grep -q "CHAIN-FINALA-COMPLETE" "$SB"/w*.log && n_ok=$(ls "$SB/gcs/finalA/"*_ARM_OK | wc -l | tr -d ' ') && [ "$n_ok" = 6 ] && ok "S11 night completes with the six core arms" || bad "S11 completion ($n_ok)"
mk_sandbox; run_chain 0 1 STUB_PREFLIGHT_FAIL=A3
grep -q "PREFLIGHT-FAILED A3 .*stop-and-report" "$SB/w0.log" && grep -q "FINALA-PREFLIGHT-ABORT" "$SB/w0.log" && ok "S11b core-arm preflight failure -> abort, stop-and-report" || { bad "S11b abort"; grep -E 'PREFLIGHT|ABORT' "$SB/w0.log" | head -3; }
[ -z "$(ls "$SB/gcs/finalA/"*_ARM_OK 2>/dev/null)" ] && ! grep -q "CHAIN-FINALA-COMPLETE" "$SB/w0.log" && ok "S11b no arm ran, no sentinel" || bad "S11b arms ran after the abort"

echo; echo "harness: $PASS PASS / $FAIL FAIL"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
