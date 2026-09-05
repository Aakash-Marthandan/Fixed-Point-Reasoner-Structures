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
    base = out.name.replace("pretrainfinalA_", "")
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
    is_nan = base == os.environ.get("STUB_NAN_ARM", "")
    is_abort = base == os.environ.get("STUB_NANABORT_ARM", "")
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
    prov = {"t_total": int(flag("--t-total", "64")), "ema": "--ema" in argv}
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
n_ok=$(ls "$SB/gcs/finalA/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '); [ "$n_ok" = 6 ] && ok "S1 6/6 arm markers" || bad "S1 arm markers ($n_ok)"
[ -f "$SB/gcs/finalA/finalA_final.tgz" ] && ok "S1 final tgz" || bad "S1 final tgz"
allc=1; for a in A0 A1 A2 A3 A4 A5; do [ -f "$SB/gcs/finalA/evals/census_${a}_vsel_OK" ] && [ -f "$SB/gcs/finalA/evals/census_${a}_final_OK" ] || allc=0; done; [ $allc = 1 ] && ok "S1 census vsel+final on every arm" || bad "S1 census"
allf=1; for a in A0 A1 A2 A3 A4 A5; do for r in full_${a}_vsel_t16 full_${a}_final_t16 full_${a}_vsel_t16_alt full_${a}_vsel_t64; do [ -f "$SB/gcs/finalA/evals/${r}_OK" ] || allf=0; done; done; [ $allf = 1 ] && ok "S1 fulls vsel+final+alt at D16 + the D64 depth row on every arm" || bad "S1 fulls"
alls=1; for a in A0 A1 A2 A3 A4 A5; do [ -f "$SB/gcs/finalA/evals/scan_${a}_OK" ] && [ -f "$SB/gcs/finalA/evals/calib_${a}_vsel_OK" ] || alls=0; done; [ $alls = 1 ] && ok "S1 scan + calib on every arm" || bad "S1 scans/calib"
[ -z "$(ls "$SB/gcs/finalA/evals/" | grep retfm)" ] && ok "S1 no retfm (field-loop cells)" || bad "S1 retfm ran"
grep -q '"ema": true' "$SB/repo/runs/sxeval_pfinalAA3/full_vsel_t16/summary_s0.json" && ok "S1 EMA headline" || bad "S1 headline weights"
grep -q "screen_A3_vb_OK" <(ls "$SB/gcs/finalA/evals/") && ok "S1 vb screen on every arm (fixed-step screens need 15k grids: asserted in S3d)" || bad "S1 screens"
grep -q "PRETRAIN-START A3 .*prec=default" "$SB/w0.log" && ok "S1 bf16 precision on the field arms" || bad "S1 precision"
[ -z "$(grep -E 'STAGEA|stage A|stage B' "$SB/w0.log")" ] && ok "S1 no two-stage machinery ran" || bad "S1 two-stage leak"
echo "== S1r the ARM REGISTRY reaches the trainer (the 2026-09-04 PM-5 lesson: arm names/flags verified by the run) =="
argv () { "$REAL_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['argv']))" "$SB/repo/runs/pretrainfinalA_$1/config.json"; }
a3=$(argv A3); a4=$(argv A4); a1=$(argv A1); a2=$(argv A2); a5=$(argv A5); a0=$(argv A0)
echo "$a3" | grep -q -- "--cell dec --dec-width 256" && ! echo "$a3" | grep -q -- "--sudoku-digit-aug" && ok "S1r A3 = DEC w256 WITHOUT digit aug" || bad "S1r A3 flags: $a3"
echo "$a4" | grep -q -- "--cell dec" && echo "$a4" | grep -q -- "--sudoku-digit-aug" && ok "S1r A4 = DEC + digit aug" || bad "S1r A4 flags"
echo "$a1" | grep -q -- "--cell trm --trm-hidden 512 --sudoku-digit-aug" && echo "$a1" | grep -q -- "--fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25" && ok "S1r A1 = X0 + FPA anchor rows" || bad "S1r A1 flags"
echo "$a2" | grep -q -- "--trm-ri-sigma 1.0" && ! echo "$a2" | grep -q -- "--fpa-k 1" && ok "S1r A2 = X0 + RI only" || bad "S1r A2 flags"
echo "$a5" | grep -q -- "--cell dec" && echo "$a5" | grep -q -- "--fpa-k 1" && echo "$a5" | grep -q -- "--trm-ri-sigma 1.0" && ! echo "$a5" | grep -q -- "--sudoku-digit-aug" && ok "S1r A5 = DEC + FPA + RI, no digit aug" || bad "S1r A5 flags"
echo "$a0" | grep -q -- "--seed 1" && echo "$a1" | grep -q -- "--seed 0" && ok "S1r A0 seed 1, A1 seed 0 (the noise pair vs sportC1's X0 seed 0)" || bad "S1r seeds"
for a in A0 A3; do echo "$(argv $a)" | grep -q -- "--grid-every 2000" && echo "$(argv $a)" | grep -q -- "--monitor-every 2000" || { bad "S1r grid cadence $a"; break; }; done; ok "S1r grids banked at the monitor cadence (2000)"
SB1=$SB

echo "== S2 idempotent rerun =="
(cd "$SB1/repo" && env PATH="$SB1/bin:$PATH" CHAIN_PY="$SB1/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 C1_STEPS_X=100 C1_WAIT_PASSES=3 C1_POLL_SLEEP=1 LIVE_NO_GUARD=1 bash tools/chain_final.sh > "$SB1/re.log" 2>&1)
grep -q "CHAIN-FINALA-COMPLETE" "$SB1/re.log" && ok "S2 complete again" || bad "S2 complete"
n_skip=$(grep -c "PRETRAIN-SKIP" "$SB1/re.log"); [ "$n_skip" = 6 ] && ok "S2 all pretrains skipped" || bad "S2 skips ($n_skip)"
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
n_ok=$(ls "$SB/gcs/finalA/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '); [ "$n_ok" = 6 ] && ok "S4 6/6 arms across the static map" || bad "S4 arms ($n_ok)"
grep -q "PRETRAIN-START A3" "$SB/w0.log" && grep -q "PRETRAIN-START A4" "$SB/w1.log" && grep -q "PRETRAIN-START A5" "$SB/w2.log" && grep -q "WORKER-3-IDLE" "$SB/w3.log" && ok "S4 map: A3 w0 · A4 w1 · A5 w2 · w3 idle" || bad "S4 map"

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
if grep -q "PRETRAIN-OOM-RETRY-REMAT A3" "$SB/w0.log" && grep -q "AMPUTATE-FAILED" "$SB/w0.log" && [ ! -f "$SB/gcs/finalA/A3_ARM_OK" ] && grep -q "FINALA-INCOMPLETE" "$SB/w0.log"; then ok "S9b always-OOM: one retry, then AMPUTATE-FAILED + INCOMPLETE (stop-and-report, no silent ARM_OK)"; else bad "S9b always-OOM path"; grep -E 'OOM|AMPUTATE|INCOMPLETE|COMPLETE' "$SB/w0.log" | head -4; fi

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

echo; echo "harness: $PASS PASS / $FAIL FAIL"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
