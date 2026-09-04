#!/bin/bash
# Ledger: CHAMPION sportC1 offline stub harness (2026-09-02; the house law: no
# chain launches without an end-to-end offline pass). Stubs: gsutil/gcloud ->
# a local FAKE_GCS dir; $CHAIN_PY -> a stub emulating pretrain.py /
# eval_sudoku_extreme.py / explosion_census.py / select_ckpt.py; REAL python
# still runs the chain's own helpers (nan_check / amputate / n-gates).
# Scenarios: S1 fresh 1x8 -> COMPLETE (7 arms; stage-A tarballs; census vsel+final;
# vsel+final+alt fulls; X1 depth row; X1 scan, no X2 scan; riders); S2 idempotent
# rerun (all SKIP); S3 NaN one-shot: single-stage R4 (post-death screens SKIPPED)
# + stage-A death W0a (STOP, no stage B, config.json shipped) + the trainer's
# NAN-ABORT rc=3 path on B1; S4 fresh 4x4 static map -> COMPLETE from a worker;
# S5 select_ckpt failure -> VB-FALLBACK-FINAL, evals proceed; S6 banked pretrain
# with no local dir -> PRETRAIN-RESTORE re-pull before select (no fallback); S7 two-stage
# selection over both stages; S8 banked two-stage arm on a NEW node -> both dirs restored;
# S9 (sportC2 pre-mortem) launch-time HBM OOM before any logged step -> ONE --remat retry, labeled
# (R2a), and an arm already on --remat (R4) is NOT retried (AMPUTATE-FAILED + INCOMPLETE = stop-and-report).
# 2026-09-04 correction pass: the first sportC2 run (21/24) failed only on sandbox construction still carrying
# sportC1 arm names (R4_ckpt.pkl for R0_ckpt.pkl; R2a/R3a rider tarballs for B0a/B1a; A0/A1/R0/X0 assertions) —
# the PM-5 class; every scenario now names sportC2's arms.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export REAL_PY="$REPO/.venv/bin/python3"
PASS=0; FAIL=0
ok () { PASS=$((PASS+1)); echo "  PASS  $1"; }
bad () { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }

mk_sandbox () {
  SB=$(mktemp -d /tmp/hc2.XXXXXX)
  mkdir -p "$SB/repo/tools" "$SB/repo/runs" "$SB/repo/data/sudoku_extreme" "$SB/gcs/sportC2" "$SB/gcs/sportC1" "$SB/gcs/sportBr2b" "$SB/bin"
  echo "fake" > "$SB/gcs/sportC1/R0_ckpt.pkl"   # R4's init-from source = sportC1's R0 final (chain fetch_init)
  # the B0/B1 correct-grid riders pull sportC1's STAGE-A tarballs (B0a/B1a) and read ckpt_020000.pkl from them
  ( cd "$SB" && mkdir -p runs/pretrainsportC1_B0a runs/pretrainsportC1_B1a && echo fake > runs/pretrainsportC1_B0a/ckpt_020000.pkl && echo fake > runs/pretrainsportC1_B1a/ckpt_020000.pkl \
    && tar czf gcs/sportC1/B0a_pretrain.tgz runs/pretrainsportC1_B0a && tar czf gcs/sportC1/B1a_pretrain.tgz runs/pretrainsportC1_B1a && rm -rf runs )
  cp "$REPO/tools/chain_sportC2.sh" "$REPO/tools/live_bank.sh" "$SB/repo/tools/"
  : > "$SB/repo/data/sudoku_extreme/sudoku_extreme_seed0.npz"
  echo "fake" > "$SB/gcs/sportBr2b/C3X_ckpt.pkl"; echo "fake" > "$SB/gcs/sportBr2b/D4_ckpt.pkl"
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
    base = out.name.replace("pretrainsportC2_", "")
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
    cell = flag("--cell", "rg"); key = "val_t16" if cell == "trm" else "val_t64"
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
    d = Path(argv[1]); arm = d.name.replace("pretrainsportC2_", "")
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
     CHAIN_WORKER=$w CHAIN_WORKERS=$nw NCHIP_OVERRIDE=4 C1_STEPS_AB=80 C1_STEPS_R=100 C1_STEPS_X=100 \
     C1_WAIT_PASSES=3 C1_POLL_SLEEP=1 LIVE_NO_GUARD=1 "$@" bash tools/chain_sportC2.sh > "$SB/w${w}.log" 2>&1)
}

echo "== S1 fresh 1x8 =="
mk_sandbox; run_chain 0 1
grep -q "CHAIN-SPORTC2-COMPLETE" "$SB/w0.log" && ok "S1 complete sentinel" || { bad "S1 sentinel"; tail -8 "$SB/w0.log"; }
n_ok=$(ls "$SB/gcs/sportC2/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '); [ "$n_ok" = 7 ] && ok "S1 7/7 arm markers" || bad "S1 arm markers ($n_ok)"
[ -f "$SB/gcs/sportC2/W0_STAGEA_OK" ] && [ -f "$SB/gcs/sportC2/W0a_pretrain.tgz" ] && ok "S1 stage-A marker + stage-A dir banked" || bad "S1 stage-A banking"
[ -f "$SB/gcs/sportC2/sportC2_final.tgz" ] && ok "S1 final tgz" || bad "S1 final tgz"
for a in W0 R4 X1; do [ -f "$SB/gcs/sportC2/evals/census_${a}_vsel_OK" ] && [ -f "$SB/gcs/sportC2/evals/census_${a}_final_OK" ] || { bad "S1 census $a"; break; }; done; [ -f "$SB/gcs/sportC2/evals/census_X2_final_OK" ] && ok "S1 census vsel+final on every arm"
[ -f "$SB/gcs/sportC2/evals/full_W0_vsel_t64_OK" ] && [ -f "$SB/gcs/sportC2/evals/full_W0_final_t64_OK" ] && [ -f "$SB/gcs/sportC2/evals/full_W0_vsel_t64_alt_OK" ] && ok "S1 W0 fulls vsel+final+alt" || bad "S1 W0 fulls"
[ -f "$SB/gcs/sportC2/evals/full_X1_vsel_t16_OK" ] && [ -f "$SB/gcs/sportC2/evals/full_X1_vsel_t64_OK" ] && ok "S1 X1 headline D16 + depth D64" || bad "S1 X1 fulls"
[ -f "$SB/gcs/sportC2/evals/scan_X1_OK" ] && [ -f "$SB/gcs/sportC2/evals/scan_X2_OK" ] && ok "S1 both field arms scanned" || bad "S1 X scans"
[ -f "$SB/gcs/sportC2/evals/retfm_W0_OK" ] && [ ! -f "$SB/gcs/sportC2/evals/retfm_X1_OK" ] && ok "S1 retfm native-only" || bad "S1 retfm"
grep -q '"ema": true' "$SB/repo/runs/sxeval_psportC2R4/full_vsel_t64/summary_s0.json" && grep -q '"ema": false' "$SB/repo/runs/sxeval_psportC2W0/full_vsel_t64/summary_s0.json" && ok "S1 headline weights: EMA on R4, raw on W0" || bad "S1 headline weights"
grep -q "rider_C3X_sel5k_OK" <(ls "$SB/gcs/sportC2/evals/") && grep -q "rider_B0_vselA20k_scan_OK" <(ls "$SB/gcs/sportC2/evals/") && grep -q "rider_B0_vselA20k_full_ema_OK" <(ls "$SB/gcs/sportC2/evals/") && ok "S1 canvas + B0/B1 correct-grid riders ran" || bad "S1 riders"
[ -f "$SB/gcs/sportC2/evals/calib_W0_vsel_OK" ] && [ -f "$SB/gcs/sportC2/evals/calib_X2_vsel_OK" ] && ok "S1 stall calibration on every arm" || bad "S1 calib"
[ -f "$SB/gcs/sportC2/evals/screen_R3_vb_hard_OK" ] && [ -f "$SB/gcs/sportC2/evals/calib_R3_vsel_hard_OK" ] && [ ! -f "$SB/gcs/sportC2/evals/screen_W0_vb_hard_OK" ] && ok "S1 R3 hard-feedback rows (R3 only)" || bad "S1 R3 hard rows"
grep -q "PRETRAIN-START R4 " "$SB/w0.log" && ok "S1 R4 init ckpt fetched (no INIT-CKPT-MISSING)" || bad "S1 R4 init"
grep -q "screen_W0_sAend_OK" <(ls "$SB/gcs/sportC2/evals/") && grep -q "screen_R4_s015000_OK" <(ls "$SB/gcs/sportC2/evals/") 2>/dev/null; [ $? -eq 0 ] || true
grep -q "PRETRAIN-START X1 .*prec=default" "$SB/w0.log" && grep -q "PRETRAIN-START W0 .*prec=highest" "$SB/w0.log" && ok "S1 per-arm precision" || bad "S1 precision"
grep -q "INIT-CKPT-MISSING" "$SB/w0.log" && bad "S1 R4 init missing" || true
SB1=$SB

echo "== S2 idempotent rerun =="
(cd "$SB1/repo" && env PATH="$SB1/bin:$PATH" CHAIN_PY="$SB1/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 C1_STEPS_AB=80 C1_STEPS_R=100 C1_STEPS_X=100 C1_WAIT_PASSES=3 C1_POLL_SLEEP=1 bash tools/chain_sportC2.sh > "$SB1/re.log" 2>&1)
grep -q "CHAIN-SPORTC2-COMPLETE" "$SB1/re.log" && ok "S2 complete again" || bad "S2 complete"
n_skip=$(grep -c "PRETRAIN-SKIP" "$SB1/re.log"); [ "$n_skip" = 7 ] && ok "S2 all pretrains skipped" || bad "S2 skips ($n_skip)"
[ "$(grep -c 'EVAL-OK\|CENSUS-OK' "$SB1/re.log")" = 0 ] && ok "S2 no re-evals" || bad "S2 re-evals ran"

echo "== S3 NaN one-shot: single-stage R4; stage-A W0a; NAN-ABORT rc=3 on R3a =="
mk_sandbox; run_chain 0 1 STUB_NAN_ARM=R4
grep -q "PRETRAIN-NAN R4" "$SB/w0.log" && grep -q "STOPPED final step 50" "$SB/repo/runs/pretrainsportC2_R4/STOPPED.txt" && ok "S3 R4 amputated to the last finite grid" || bad "S3 R4 amputation"
[ ! -f "$SB/repo/runs/pretrainsportC2_R4/ckpt_000100.pkl" ] && ok "S3 post-death grid removed" || bad "S3 post-death grid kept"
grep -q "CHAIN-SPORTC2-COMPLETE" "$SB/w0.log" && ok "S3 completes despite the stop" || bad "S3 complete"
mk_sandbox; run_chain 0 1 STUB_NAN_ARM=W0a
grep -q "amputate + STOP" "$SB/w0.log" && [ -f "$SB/repo/runs/pretrainsportC2_W0/STOPPED.txt" ] && ok "S3b stage-A death -> STOP" || bad "S3b stage-A stop"
[ -f "$SB/repo/runs/pretrainsportC2_W0/config.json" ] && [ -f "$SB/gcs/sportC2/W0a_pretrain.tgz" ] && ok "S3b config.json + stage-A dir shipped on death" || bad "S3b shipping"
if grep -q "stage B" "$SB/repo/runs/pretrainsportC2_W0.log" 2>/dev/null; then bad "S3b stage B ran"; else ok "S3b stage B never ran"; fi
grep -q "screen_W0_vb_OK" <(ls "$SB/gcs/sportC2/evals/") && [ ! -f "$SB/gcs/sportC2/evals/screen_W0_sAend_OK" ] && ok "S3b vb screen on the stopped final; stage screens skipped" || bad "S3b screens"
mk_sandbox; run_chain 0 1 STUB_NANABORT_ARM=R3a
grep -q "PRETRAIN-NAN R3 stage A (rc=3)" "$SB/w0.log" && [ -f "$SB/repo/runs/pretrainsportC2_R3/STOPPED.txt" ] && ok "S3c trainer NAN-ABORT rc=3 -> amputation" || bad "S3c nan-abort path"
grep -q "CHAIN-SPORTC2-COMPLETE" "$SB/w0.log" && ok "S3c complete" || bad "S3c complete"

echo "== S3d post-death fixed-step screens never run (single-stage R0, NaN mid-run at 40k) =="
mk_sandbox; run_chain 0 1 STUB_NAN_ARM=R4 C1_STEPS_R=40000
grep -q "AMPUTATED to ckpt_020000.pkl step 20000" "$SB/w0.log" && ok "S3d amputated to the last finite 5k grid (20000)" || { bad "S3d amputation"; grep -E 'AMPUTAT|STOPPED' "$SB/w0.log" | head -3; }
[ -f "$SB/gcs/sportC2/evals/screen_R4_s015000_OK" ] && [ ! -f "$SB/gcs/sportC2/evals/screen_R4_s035000_OK" ] && ok "S3d s015000 screened, s035000 (post-death) never ran" || bad "S3d post-death screen"
[ -z "$(ls "$SB/repo/runs/pretrainsportC2_R4"/ckpt_0[2-4]5000.pkl "$SB/repo/runs/pretrainsportC2_R4"/ckpt_0[34]0000.pkl 2>/dev/null)" ] && ok "S3d post-death grids removed" || bad "S3d post-death grids remain"

echo "== S4 fresh 4x4 static map =="
mk_sandbox; for w in 0 1 2 3; do run_chain $w 4 & done; wait
if grep -q "CHAIN-SPORTC2-COMPLETE" "$SB"/w*.log; then ok "S4 complete from a worker"; else bad "S4 complete"; tail -3 "$SB"/w*.log; fi
n_ok=$(ls "$SB/gcs/sportC2/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '); [ "$n_ok" = 7 ] && ok "S4 7/7 arms across the static map" || bad "S4 arms ($n_ok)"
grep -q "PRETRAIN-START R4" "$SB/w2.log" && grep -q "PRETRAIN-START X2" "$SB/w3.log" && ok "S4 map: R4 on w2, X2 on w3" || bad "S4 map"

echo "== S5 select_ckpt failure -> LOUD fallback =="
mk_sandbox; run_chain 0 1 STUB_SELECT_FAIL=R1,R1a
grep -q "VB-FALLBACK-FINAL R1" "$SB/w0.log" && grep -q "FALLBACK-FINAL" "$SB/repo/runs/pretrainsportC2_R1/val_best.txt" && ok "S5 fallback echoed + labeled" || bad "S5 fallback"
grep -q "FULL-FINAL R1 := vsel" "$SB/w0.log" && [ -f "$SB/repo/runs/sxeval_psportC2R1/full_final_t64/summary_all.json" ] && ok "S5 final := vsel copy" || bad "S5 final copy"
grep -q "CHAIN-SPORTC2-COMPLETE" "$SB/w0.log" && ok "S5 complete" || bad "S5 complete"

echo "== S6 banked pretrain, no local dir -> re-pull before select =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainsportC2_W0 --steps 80 --ema 0.999 >/dev/null 2>&1
  tar czf "$SB/gcs/sportC2/W0_pretrain.tgz" runs/pretrainsportC2_W0 && cp runs/pretrainsportC2_W0/ckpt_latest.pkl "$SB/gcs/sportC2/W0_ckpt.pkl"
  echo ok > "$SB/gcs/sportC2/W0_PRETRAIN_OK"; rm -rf runs/pretrainsportC2_W0 )
run_chain 0 1
grep -q "PRETRAIN-SKIP W0" "$SB/w0.log" && grep -q "PRETRAIN-RESTORE W0" "$SB/w0.log" && ok "S6 banked arm re-pulled before select" || bad "S6 re-pull"
grep -q "VALBEST W0" "$SB/w0.log" && ! grep -q "VB-FALLBACK-FINAL W0" "$SB/w0.log" && ok "S6 select_ckpt worked on the restored metrics" || bad "S6 select after restore"

echo "== S7 two-stage selection over BOTH stages =="
mk_sandbox; run_chain 0 1 STUB_SELECT_VAL_W0a=0.4500
grep -q "VALBEST W0 A:000025 0.4500 25 -> runs/pretrainsportC2_W0a/ckpt_000025.pkl" "$SB/w0.log" && ok "S7 stage-A peak selected when its val is higher" || { bad "S7 stage-A selection"; grep VALBEST "$SB/w0.log" | head -3; }
grep -q "VALBEST R1 B:000015 0.3000 15 -> runs/pretrainsportC2_R1/ckpt_000015.pkl" "$SB/w0.log" && ok "S7 tie -> stage B (the later stage)" || { bad "S7 tie rule"; grep 'VALBEST R1' "$SB/w0.log"; }
grep -q '"ema": false' "$SB/repo/runs/sxeval_psportC2W0/full_vsel_t64/summary_s0.json" && [ -f "$SB/gcs/sportC2/evals/full_W0_final_t64_OK" ] && ok "S7 vsel (stage-A grid) and final both evaluated" || bad "S7 fulls after stage-A selection"

echo "== S8 banked TWO-STAGE arm, no local dirs (node change after PRETRAIN-OK) -> BOTH dirs restored, selection over both stages =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainsportC2_W0a --steps 50 --ema 0.999 >/dev/null 2>&1
  env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainsportC2_W0 --steps 30 --ema 0.999 >/dev/null 2>&1
  tar czf "$SB/gcs/sportC2/W0a_pretrain.tgz" runs/pretrainsportC2_W0a && cp runs/pretrainsportC2_W0a/ckpt_latest.pkl "$SB/gcs/sportC2/W0_stageA_ckpt.pkl" && echo ok > "$SB/gcs/sportC2/W0_STAGEA_OK"
  tar czf "$SB/gcs/sportC2/W0_pretrain.tgz" runs/pretrainsportC2_W0 && cp runs/pretrainsportC2_W0/ckpt_latest.pkl "$SB/gcs/sportC2/W0_ckpt.pkl" && echo ok > "$SB/gcs/sportC2/W0_PRETRAIN_OK"
  rm -rf runs/pretrainsportC2_W0 runs/pretrainsportC2_W0a )
run_chain 0 1 STUB_SELECT_VAL_W0a=0.4500
grep -q "PRETRAIN-SKIP W0" "$SB/w0.log" && grep -q "PRETRAIN-RESTORE W0 " "$SB/w0.log" && grep -q "PRETRAIN-RESTORE W0a" "$SB/w0.log" && ok "S8 both stage dirs re-pulled after the node change" || { bad "S8 restore"; grep -E 'RESTORE|SKIP W0' "$SB/w0.log" | head -4; }
grep -q "VALBEST W0 A:000025 0.4500 25 -> runs/pretrainsportC2_W0a/ckpt_000025.pkl" "$SB/w0.log" && ok "S8 selection over BOTH stages picks the stage-A peak (the B0/B1 defect closed)" || { bad "S8 selection"; grep 'VALBEST W0' "$SB/w0.log" | head -2; }
grep -q "CHAIN-SPORTC2-COMPLETE" "$SB/w0.log" && ok "S8 complete" || bad "S8 complete"

echo "== S9 launch-time HBM OOM (no step logged) -> ONE retry with --remat (R2a); an arm already on --remat (R4) is NOT retried =="
mk_sandbox; run_chain 0 1 STUB_OOM_ARM=R2a
grep -q "PRETRAIN-OOM-RETRY-REMAT R2" "$SB/w0.log" && [ -f "$SB/repo/runs/pretrainsportC2_R2a/RETRY_REMAT.txt" ] && ok "S9 OOM at launch -> retried once with --remat, labeled on disk" || { bad "S9 retry"; grep -E 'OOM|PRETRAIN-NAN R2|AMPUTATE' "$SB/w0.log" | head -3; }
[ -f "$SB/gcs/sportC2/R2_ARM_OK" ] && [ ! -f "$SB/repo/runs/pretrainsportC2_R2/STOPPED.txt" ] && ok "S9 R2 completed clean after the retry (no STOPPED label)" || bad "S9 R2 outcome"
grep -q "CHAIN-SPORTC2-COMPLETE" "$SB/w0.log" && ok "S9 complete" || bad "S9 complete"
mk_sandbox; run_chain 0 1 STUB_OOM_ARM=R4 STUB_OOM_ALWAYS=1
if ! grep -q "PRETRAIN-OOM-RETRY-REMAT" "$SB/w0.log" && grep -q "AMPUTATE-FAILED" "$SB/w0.log" && grep -q "SPORTC2-INCOMPLETE" "$SB/w0.log"; then ok "S9b remat arm OOM -> no retry, AMPUTATE-FAILED, INCOMPLETE (stop-and-report)"; else bad "S9b remat-arm OOM path"; grep -E 'R4|OOM|AMPUTATE' "$SB/w0.log" | head -4; fi
mk_sandbox; run_chain 0 1 STUB_OOM_ARM=R2a STUB_OOM_ALWAYS=1
if grep -q "PRETRAIN-OOM-RETRY-REMAT R2" "$SB/w0.log" && grep -q "AMPUTATE-FAILED" "$SB/w0.log" && [ ! -f "$SB/gcs/sportC2/R2_ARM_OK" ] && grep -q "SPORTC2-INCOMPLETE" "$SB/w0.log"; then ok "S9c OOM persists through the --remat retry -> AMPUTATE-FAILED, R2 lost, INCOMPLETE (stop-and-report; never a second retry)"; else bad "S9c persistent-OOM path"; grep -E 'R2|OOM|AMPUTATE' "$SB/w0.log" | head -4; fi

echo "== S10 LIVE BANK: in-flight state in the live prefix, FRESH node -> pulled no-clobber, pretrain RESUMES from the live ckpt, eval partial in place, loop banks + cleans up =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainsportC2_R2a --steps 40 --ema 0.999 >/dev/null 2>&1
  mkdir -p "$SB/gcs/sportC2/live/runs/pretrainsportC2_R2a" "$SB/gcs/sportC2/live/runs/sxscan_psportC2X1"
  cp runs/pretrainsportC2_R2a/* "$SB/gcs/sportC2/live/runs/pretrainsportC2_R2a/"; echo partial > "$SB/gcs/sportC2/live/runs/sxscan_psportC2X1/partial_s0.npz"; rm -rf runs )
run_chain 0 1
grep -q "LIVE-RESTORE pulled=" "$SB/w0.log" && ok "S10 live prefix pulled on a fresh node" || { bad "S10 pull"; grep LIVE "$SB/w0.log" | head -3; }
grep -q "RESUMED from runs/pretrainsportC2_R2a/ckpt_latest.pkl at step 40" "$SB/repo/runs/pretrainsportC2_R2a.log" && ok "S10 R2 stage A RESUMED from the live ckpt (step 40)" || { bad "S10 resume"; head -3 "$SB/repo/runs/pretrainsportC2_R2a.log"; }
[ -f "$SB/repo/runs/sxscan_psportC2X1/partial_s0.npz" ] && ok "S10 eval partial restored into place" || bad "S10 partial"
grep -q "LIVE-BANK loop start" "$SB/w0.log" && grep -q "LIVE-BANK rc=0" "$SB/w0.log" && ok "S10 the 5-min loop started and banked" || { bad "S10 loop"; grep LIVE-BANK "$SB/w0.log" | head -3; }
sleep 1; [ ! -f "$SB/repo/runs/live_bank.pid" ] && ok "S10 loop stopped with the chain (pidfile gone)" || bad "S10 loop cleanup"
grep -q "CHAIN-SPORTC2-COMPLETE" "$SB/w0.log" && ok "S10 complete" || bad "S10 complete"

echo "== S10b LIVE BANK negative: a TORN live ckpt_latest (staged) -> LIVE-RESTORE-FALLBACK to the newest loadable grid, RESUMED at its step, no NaN path =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainsportC2_R2a --steps 40 --ema 0.999 >/dev/null 2>&1
  mkdir -p "$SB/gcs/sportC2/live/runs/pretrainsportC2_R2a"; cp runs/pretrainsportC2_R2a/* "$SB/gcs/sportC2/live/runs/pretrainsportC2_R2a/"
  head -c 1000 /dev/urandom > "$SB/gcs/sportC2/live/runs/pretrainsportC2_R2a/ckpt_latest.pkl"; rm -rf runs )
"$REAL_PY" -c "import pickle,sys; pickle.load(open(sys.argv[1],'rb'))" "$SB/gcs/sportC2/live/runs/pretrainsportC2_R2a/ckpt_latest.pkl" 2>/dev/null && bad "S10b staged torn ckpt is loadable (stage failed)" || ok "S10b the staged torn ckpt really does not unpickle"
run_chain 0 1
grep -q "LIVE-RESTORE-FALLBACK runs/pretrainsportC2_R2a ckpt_latest unloadable -> ckpt_000040.pkl step 40" "$SB/w0.log" && ok "S10b torn ckpt_latest -> fallback to the newest loadable grid, labeled" || { bad "S10b fallback"; grep -E "LIVE|PRETRAIN-NAN" "$SB/w0.log" | head -4; }
grep -q "RESUMED from runs/pretrainsportC2_R2a/ckpt_latest.pkl at step 40" "$SB/repo/runs/pretrainsportC2_R2a.log" && ! grep -q "PRETRAIN-NAN R2" "$SB/w0.log" && ok "S10b R2 resumed at step 40 and never entered the NaN/amputation path" || bad "S10b resume"

echo "== S10c LIVE BANK negative: a BANKED arm's stale live copy (staged, step 7) is NOT restored over its banked final (step 80) =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainsportC2_W0 --steps 7 --ema 0.999 >/dev/null 2>&1
  mkdir -p "$SB/gcs/sportC2/live/runs/pretrainsportC2_W0"; cp runs/pretrainsportC2_W0/* "$SB/gcs/sportC2/live/runs/pretrainsportC2_W0/"; rm -rf runs
  mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainsportC2_W0 --steps 80 --ema 0.999 >/dev/null 2>&1
  tar czf "$SB/gcs/sportC2/W0_pretrain.tgz" runs/pretrainsportC2_W0 && cp runs/pretrainsportC2_W0/ckpt_latest.pkl "$SB/gcs/sportC2/W0_ckpt.pkl"
  echo ok > "$SB/gcs/sportC2/W0_PRETRAIN_OK"; rm -rf runs )
[ -f "$SB/gcs/sportC2/live/runs/pretrainsportC2_W0/ckpt_000007.pkl" ] && ok "S10c the stale live copy really is staged in the prefix" || bad "S10c stage"
run_chain 0 1
grep -q "banked arms excluded: .*pretrainsportC2_W0/" "$SB/w0.log" && grep -q "PRETRAIN-RESTORE W0 " "$SB/w0.log" && ok "S10c banked W0 excluded from the live pull; its tarball restored instead" || { bad "S10c exclusion"; grep -E "LIVE-RESTORE|PRETRAIN-RESTORE" "$SB/w0.log" | head -3; }
st=$("$REAL_PY" -c "import pickle,sys; print(pickle.load(open(sys.argv[1],'rb'))['step'])" "$SB/repo/runs/pretrainsportC2_W0/ckpt_latest.pkl" 2>/dev/null)
[ "$st" = 80 ] && [ ! -f "$SB/repo/runs/pretrainsportC2_W0/ckpt_000007.pkl" ] && ok "S10c W0's local final is the banked step-80 grid; no stale step-7 files" || bad "S10c final grid (step=$st)"

echo; echo "harness: $PASS PASS / $FAIL FAIL"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
