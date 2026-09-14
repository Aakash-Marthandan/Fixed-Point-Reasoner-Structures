#!/bin/bash
# Ledger: THE C8 EXTENSION offline stub harness (2026-09-14; the house law: no chain launches without an end-to-end offline pass; every
# negative scenario asserts the staged failure fired). Reuses tools/harness_champ.sh's sandbox and stubs (its lines 1-157) and patches
# the stub evaluator for the held-out split (--split val -> n STUB_VAL_N, the split in the provenance, a peak grid, a staged failure).
# Fixture = the paper-final GCS state under champ (C8's 30k-analog state at step 100 with grids 50/100, C5/C7 grids, C8's paper-final
# markers and rows, the compile cache) + the validation set under c8x/sets. Scenarios:
#   X1 fresh -> COMPLETE (the resume from 100, the extension to 120 through chain_champ, preflight + battery on C8 only, the val curves
#      of C8/C5/C7, the filler rows, the extra rows for g_val and the hypothesis grid; NOTHING under champ written or deleted)
#   X2 idempotent rerun (final present -> nothing re-runs; final absent -> only the manifest)
#   X3 g_val == g_mon == the hypothesis grid -> no extra rows
#   X4 C8's source state missing -> C8X-NO-RESUME-STATE, no training (never from scratch)
#   X5 a node change after C8_PRETRAIN_OK -> the extended grids from c8x, the 30k source never extracted
#   X6 a mid-extension resume from the live prefix (step 110) -> no source extraction, RESUMED at 110
#   X7 a val grid that fails twice -> INCOMPLETE naming it, no sentinel
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
_HC=$(mktemp /tmp/hc8x_src.XXXXXX); sed -n '1,157p' "$HERE/harness_champ.sh" > "$_HC"   # bash 3.2 (macOS) cannot source a process substitution
grep -q '^n_ok () ' "$_HC" && grep -q '^mk_sandbox () ' "$_HC" || { echo "harness_champ.sh layout changed: lines 1-157 no longer hold the helpers"; exit 2; }
# shellcheck disable=SC1090
source "$_HC"; rm -f "$_HC"

tree_sig () { (cd "$1" 2>/dev/null && find . -type f | sort | while read -r f; do printf '%s %s\n' "$f" "$(cksum < "$f")"; done) | cksum; }
mk_x () {
  mk_sandbox
  cp "$REPO/tools/filler_full.sh" "$REPO/tools/chain_c8x.sh" "$REPO/tools/c8x_valcurve.py" "$SB/repo/tools/"
  rm -rf /tmp/filler_pull /tmp/c8x_src /tmp/c8x_c8_src.tgz /tmp/c8x_c8_ext.tgz /tmp/c8x_vsel.tgz /tmp/c8x_val_*.tgz /tmp/c8x_xrow_*.tgz /tmp/c8x_jc_src.tgz
  local G="$SB/gcs/champ" X="$SB/gcs/c8x"
  mkdir -p "$G/evals" "$G/filler" "$X/sets"
  : > "$X/sets/sudoku_extreme_seed0_val10k.npz"
  for m in C8_ARM_OK C8_PRETRAIN_OK SYNC-AB evals/full_C8_vsel_t16_OK filler/d64full_C8_OK; do echo "paper-final" > "$G/$m"; done
  printf 'BATCHONLY' > "$G/RECIPE-DEC"
  "$REAL_PY" - "$SB/fx" <<'PYEOF'
import json, pickle, sys
from pathlib import Path
fx = Path(sys.argv[1])
def arm(name, grids, latest, mon):
    d = fx / "src" / name / "runs" / f"pretrainchamp_{name}"; d.mkdir(parents=True, exist_ok=True)
    for g in grids: pickle.dump({"state": {"model": {}}, "step": g, "config": {}}, open(d / f"ckpt_{g:06d}.pkl", "wb"))
    pickle.dump({"state": {"model": {}}, "step": latest, "config": {}}, open(d / "ckpt_latest.pkl", "wb"))
    rows = []
    for s, v in mon:
        rows.append(json.dumps({"step": s, "loss": .5, "ce_in": .4, "steps_per_sec": 99.0}))
        rows.append(json.dumps({"monitor": {"step": s, "val_t16": v - .01, "val_t16_ema": v, "n_val": 512}}))
    (d / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    (d / "config.json").write_text(json.dumps({"argv": ["tools/pretrain.py", "--seed", "2", "--dec-width", "192"]}))
    (d / "val_best.txt").write_text("000050 0.3 50\n")
arm("C8", [50, 100], 100, [(50, .30), (100, .29)])
arm("C5", [50, 100, 120], 120, [(50, .30), (100, .31), (120, .31)])
arm("C7", [50, 100, 120], 120, [(50, .30), (100, .31), (120, .30)])
(fx / "jc" / "jax_cache").mkdir(parents=True, exist_ok=True); (fx / "jc" / "jax_cache" / "entry").write_text("cache")
PYEOF
  for a in C8 C5 C7; do tar czf "$G/${a}_pretrain.tgz" -C "$SB/fx/src/$a" "runs/pretrainchamp_$a"; done
  tar czf "$G/jax_cache.tgz" -C "$SB/fx/jc" jax_cache
  "$REAL_PY" - "$SB/bin/stubpy" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
old = '        (d / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4, "wall_s": 5.0}))\n'
new = ('        s0 = sorted(d.glob("summary_s*.json")); prov0 = json.loads(s0[0].read_text()) if s0 else {}; prov0.pop("n", None)\n'
       '        (d / "summary_all.json").write_text(json.dumps({**prov0, "n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4, "wall_s": 5.0}))\n')
assert s.count(old) == 1, "stub merge line not found"; s = s.replace(old, new)
old = '    prov = {"t_total": int(flag("--t-total", "64")), "ema": "--ema" in argv, "argv": argv, "ckpt": flag("--ckpt")}\n'
new = ('    prov = {"t_total": int(flag("--t-total", "64")), "ema": "--ema" in argv, "argv": argv, "ckpt": flag("--ckpt"), "split": flag("--split", "test")}\n'
       '    val_acc = None\n'
       '    if flag("--split") == "val":\n'
       '        import re as _re\n'
       '        n_total = int(os.environ.get("STUB_VAL_N", "10000"))\n'
       '        _st = int(_re.search(r"ckpt_(\\d+)", flag("--ckpt")).group(1)); _arm = _re.search(r"pretrainchamp_(C\\d)", flag("--ckpt")).group(1)\n'
       '        if f"{_arm}:{_st:06d}" in os.environ.get("STUB_VAL_FAIL", "").split(","): print("staged val failure", file=sys.stderr); sys.exit(1)\n'
       '        val_acc = 0.95 if str(_st) == os.environ.get("STUB_VAL_PEAK", "") else 0.90\n')
assert s.count(old) == 1, "stub prov line not found"; s = s.replace(old, new)
old = '        (out / "summary_all.json").write_text(json.dumps({"n": n_total, "exact_acc": .3,'
new = '        (out / "summary_all.json").write_text(json.dumps({"n": n_total, "exact_acc": (val_acc if val_acc is not None else .3),'
assert s.count(old) == 1, "stub summary line not found"; s = s.replace(old, new)
p.write_text(s)
PYEOF
  CHAMP_SIG0=$(tree_sig "$G")
}
run_x () {  # [VAR=val...]
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 \
     C1_STEPS_X=100 C1_STEPS_LONG=120 C1_EXT_STEPS=20 C1_EXT_WINDOW=10 LIVE_NO_GUARD=1 LIVE_EVERY=1 STALL_SEC=3 WATCH_POLL=1 \
     PASS_SLEEP=1 IDLE_POLL=1 X_PASSES=2 C8X_EXT_FROM=100 C8X_EXT_TO=120 VAL_MIN_STEP=0 C8X_HYP_GRID=000120 C8X_PF_GRID=000050 \
     "$@" bash tools/chain_c8x.sh > "$SB/x.log" 2>&1; echo $? > "$SB/x.rc")
}
has () { grep -q -- "$1" "$SB/x.log"; }
line_of () { grep -n -- "$1" "$SB/x.log" | head -1 | cut -d: -f1; }

echo "== X1 fresh -> COMPLETE (g_mon = the paper-final grid 50, g_val = 100, the hypothesis grid 120) =="
mk_x; run_x STUB_VAL_PEAK=100; X="$SB/gcs/c8x"
[ "$(cat "$SB/x.rc")" = 0 ] && has "CHAIN-C8X-COMPLETE" && [ -f "$X/c8x_final.tgz" ] && ok "X1 rc 0, sentinel, c8x_final.tgz" || { bad "X1 completion (rc $(cat "$SB/x.rc"))"; tail -15 "$SB/x.log"; }
has "C8X-RESTORE-SRC" && has "C8X-RESUME-POINT step 100" && grep -q "RESUMED from runs/pretrainchamp_C8/ckpt_latest.pkl at step 100" "$SB/repo/runs/pretrainchamp_C8.log" && has "PRETRAIN-EXTENDED-BUDGET C8 120" && ok "X1 the resume from the banked state at 100 through the registered extension path to 120" || bad "X1 resume: $(grep -E 'RESUME|RESTORE|EXTENDED-BUDGET' "$SB/x.log" | tr '\n' ' ')"
[ "$(grep -c 'PRETRAIN-START' "$SB/x.log")" = 1 ] && has "PRETRAIN-START C8" && [ "$(grep -c 'PREFLIGHT-OK' "$SB/x.log")" = 1 ] && has "PREFLIGHT-OK C8" && ok "X1 one preflight and one pretrain, both C8" || bad "X1 arms: $(grep -E 'PREFLIGHT-OK|PRETRAIN-START' "$SB/x.log" | tr '\n' ' ')"
has "SYNC-AB-SKIP" && has "CHAIN-CHAMPC8X-COMPLETE" && ! has "CHAMPC8X-INCOMPLETE" && ok "X1 the rider skipped; the inner chain completes on C8 alone" || bad "X1 inner chain: $(grep -E 'SYNC-AB|CHAMPC8X' "$SB/x.log" | tr '\n' ' ')"
allb=1; for r in full_C8_vsel_t16 full_C8_final_t16 full_C8_vsel_t16_alt full_C8_vsel_t64 d128_C8 d256_C8 scan_C8 census_C8_vsel calib_C8_vsel screen_C8_vb; do [ -f "$X/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done
[ $allb = 1 ] && [ -f "$X/C8_ARM_OK" ] && [ -f "$X/C8_pretrain.tgz" ] && ok "X1 the battery, ARM_OK and the extended pretrain tarball under c8x" || bad "X1 battery"
nval=$(ls "$X"/val/*_OK 2>/dev/null | wc -l | tr -d ' ')
[ "$nval" = 9 ] && [ -f "$X/val/C8_s000120_OK" ] && [ -f "$X/val/C5_s000050_OK" ] && [ -f "$X/val/C7_s000120_OK" ] && ok "X1 the val curves: 9 rows (C8 50/100/120, C5 and C7 50/100/120)" || bad "X1 val rows ($nval)"
ve=$(eargv "$SB/repo/runs/c8x_val_pC8_s000120/summary_all.json")
echo "$ve" | grep -q -- "--split val --t-total 16 --ema" && echo "$ve" | grep -q "val10k" && ok "X1 val rows on the held-out file, D16, EMA" || bad "X1 val flags: $ve"
has "C8X-PICKS g_mon=000050 g_val=000100 hyp=000120" && has "C8X-XROWS \[d16full_s000100 d64full_s000100 d16full_s000120 d64sub100k_s000120\]" && ok "X1 the picks and the four extra rows" || bad "X1 picks: $(grep -E 'PICKS|XROWS' "$SB/x.log" | tr '\n' ' ')"
allx=1; for r in d16full_s000100 d64full_s000100 d16full_s000120 d64sub100k_s000120; do [ -f "$X/xrows/${r}_OK" ] && [ -f "$X/xrows/$r.tgz" ] || { allx=0; echo "    missing xrow $r"; }; done
nx=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/c8x_xrow_pchampC8_d64sub100k_s000120/summary_all.json'))['n'], json.load(open('$SB/repo/runs/c8x_xrow_pchampC8_d16full_s000100/summary_all.json'))['n'])")
[ $allx = 1 ] && [ "$nx" = "100000 422786" ] && ok "X1 the extra rows banked with their n-gates" || bad "X1 xrows ($nx)"
xe=$(eargv "$SB/repo/runs/c8x_xrow_pchampC8_d64full_s000100/summary_s0.json")
echo "$xe" | grep -q -- "--ckpt runs/pretrainchamp_C8/ckpt_000100.pkl" && echo "$xe" | grep -q -- "--split test --t-total 64 --ema --record-by-step" && echo "$xe" | grep -q "mon512" && ok "X1 the g_val D64 row: its grid, the test split of the monitor file, full set" || bad "X1 xrow flags: $xe"
allf=1; for m in d64full_C8 d128sub_C8 d256sub_C8; do [ -f "$X/filler/${m}_OK" ] || { allf=0; echo "    missing filler $m"; }; done
[ $allf = 1 ] && ok "X1 the filler rows for g_mon under c8x" || bad "X1 filler"
[ "$(tree_sig "$SB/gcs/champ")" = "$CHAMP_SIG0" ] && ok "X1 NOTHING under champ written or deleted" || bad "X1 champ state changed"
[ -d "$X/live/runs" ] && [ ! -d "$SB/gcs/champ/live" ] && ok "X1 the live bank under c8x/live only" || bad "X1 live prefix"
l1=$(line_of "C8X-LANE1-END"); l2=$(line_of "C8X-LANE2"); l3=$(line_of "C8X-FILLER"); lc=$(line_of "CHAIN-C8X-COMPLETE")
[ -n "$l1" ] && [ -n "$l2" ] && [ -n "$l3" ] && [ -n "$lc" ] && [ "$l1" -lt "$l2" ] && [ "$l2" -lt "$l3" ] && [ "$l3" -lt "$lc" ] && ok "X1 lane order 1 -> 2 -> 3 -> sentinel" || bad "X1 order ($l1 $l2 $l3 $lc)"
tar tzf "$X/c8x_final.tgz" | grep -q "runs/analysis/c8x_valcurve.json" && tar tzf "$X/c8x_final.tgz" | grep -q "pretrainchamp_C8/metrics.jsonl" && tar tzf "$X/c8x_final.tgz" | grep -q "c8x_xrow_pchampC8_d16full_s000100/summary_all.json" && ok "X1 the manifest carries the curve, the metrics and the extra rows" || bad "X1 manifest"
SBX1=$SB

echo "== X2 idempotent rerun =="
SB=$SBX1; run_x STUB_VAL_PEAK=100
[ "$(cat "$SB/x.rc")" = 0 ] && has "(final object present)" && ! has "PRETRAIN-START" && ok "X2 final present -> complete at once" || { bad "X2"; tail -5 "$SB/x.log"; }
rm -f "$SB/gcs/c8x/c8x_final.tgz"; run_x STUB_VAL_PEAK=100
[ "$(cat "$SB/x.rc")" = 0 ] && ! has "PRETRAIN-START\|C8X-XROW-START\|FILLER-JOB-START" && has "C8X-VAL-PASS 0 grids" && has "FINAL-BANKED" && ok "X2b markers present -> nothing re-runs; only the manifest is rebuilt" || { bad "X2b"; grep -E "PRETRAIN-START|XROW-START|FILLER-JOB-START|VAL-PASS" "$SB/x.log" | head -5; }

echo "== X3 g_val == g_mon == the hypothesis grid -> no extra rows =="
mk_x; run_x STUB_PEAK_LAST_ARM=C8 STUB_VAL_PEAK=120
[ "$(cat "$SB/x.rc")" = 0 ] && has "C8X-PICKS g_mon=000120 g_val=000120" && has "C8X-XROWS \[\]" && [ ! -d "$SB/gcs/c8x/xrows" ] && has "CHAIN-C8X-COMPLETE" && ok "X3 the picks coincide: zero extra rows, COMPLETE" || { bad "X3"; grep -E "PICKS|XROWS|COMPLETE" "$SB/x.log"; }

echo "== X4 C8's source state missing -> no training =="
mk_x; rm -f "$SB/gcs/champ/C8_pretrain.tgz"; CHAMP_SIG0=$(tree_sig "$SB/gcs/champ"); run_x STUB_VAL_PEAK=100
[ "$(cat "$SB/x.rc")" = 2 ] && has "C8X-NO-RESUME-STATE" && ! has "PRETRAIN-START\|PREFLIGHT" && [ ! -f "$SB/gcs/c8x/C8_PRETRAIN_OK" ] && [ ! -d "$SB/repo/runs/pretrainchamp_C8" ] && ok "X4 NO-RESUME-STATE rc 2: no preflight, no pretrain, nothing trained from scratch" || { bad "X4"; tail -6 "$SB/x.log"; }

echo "== X5 a node change after C8_PRETRAIN_OK -> the extended grids from c8x =="
mk_x; run_x STUB_VAL_PEAK=100; SB5=$SB; sleep 3   # let the finished run's background live-bank pass settle before the node change
rm -rf "$SB/repo/runs" /tmp/c8x_c8_src.tgz /tmp/c8x_c8_ext.tgz /tmp/c8x_vsel.tgz; mkdir -p "$SB/repo/runs"
rm -f "$SB/gcs/c8x/c8x_final.tgz" "$SB/gcs/c8x/val/C8_s000120_OK" "$SB/gcs/c8x/xrows/d16full_s000120_OK"; rm -rf "$SB/gcs/c8x/live"
run_x STUB_VAL_PEAK=100
[ "$(cat "$SB/x.rc")" = 0 ] && has "C8X-PRETRAIN-DONE" && ! has "C8X-RESTORE-SRC" && ! has "PRETRAIN-START" && has "C8X-VAL-OK C8_s000120" && has "C8X-XROW-OK d16full_s000120" && has "CHAIN-C8X-COMPLETE" && ok "X5 the extended grids re-pulled from c8x; the 30k source never extracted; the missing rows re-run" || { bad "X5"; grep -E "PRETRAIN|RESTORE|VAL-OK C8_s000120|XROW|COMPLETE|INCOMPLETE" "$SB/x.log" | head -8; }

echo "== X6 a mid-extension resume from the live prefix (step 110) =="
mk_x
"$REAL_PY" - "$SB/gcs/c8x/live/runs/pretrainchamp_C8" <<'PYEOF'
import json, pickle, sys
from pathlib import Path
d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
for g in (50, 100): pickle.dump({"state": {"model": {}}, "step": g, "config": {}}, open(d / f"ckpt_{g:06d}.pkl", "wb"))
pickle.dump({"state": {"model": {}}, "step": 110, "config": {}}, open(d / "ckpt_latest.pkl", "wb"))
rows = [json.dumps({"step": s, "loss": .5, "ce_in": .4}) for s in (50, 100)] + [json.dumps({"monitor": {"step": s, "val_t16": v - .01, "val_t16_ema": v, "n_val": 512}}) for s, v in ((50, .30), (100, .29))]
(d / "metrics.jsonl").write_text("\n".join(rows) + "\n")
(d / "config.json").write_text(json.dumps({"argv": ["tools/pretrain.py", "--seed", "2"]}))
PYEOF
run_x STUB_VAL_PEAK=100
[ "$(cat "$SB/x.rc")" = 0 ] && ! has "C8X-RESTORE-SRC" && has "C8X-RESUME-POINT step 110" && grep -q "RESUMED from runs/pretrainchamp_C8/ckpt_latest.pkl at step 110" "$SB/repo/runs/pretrainchamp_C8.log" && has "CHAIN-C8X-COMPLETE" && ok "X6 the in-flight state from the live prefix; no source extraction; RESUMED at 110" || { bad "X6"; grep -E "RESTORE|RESUME|COMPLETE" "$SB/x.log" | head -6; }

echo "== X7 a val grid that fails twice -> INCOMPLETE, no sentinel =="
mk_x; run_x STUB_VAL_PEAK=100 STUB_VAL_FAIL=C5:000100
[ "$(cat "$SB/x.rc")" = 1 ] && [ "$(grep -c 'C8X-VAL-N-BAD C5_s000100' "$SB/x.log")" = 2 ] && has "C8X-INCOMPLETE worker=0 (missing: val/C5_s000100)" && ! has "CHAIN-C8X-COMPLETE" && [ ! -f "$SB/gcs/c8x/c8x_final.tgz" ] && [ -f "$SB/gcs/c8x/val/C5_s000120_OK" ] && ok "X7 retried once, INCOMPLETE names the grid, no sentinel, the other rows banked" || { bad "X7"; grep -E "N-BAD|INCOMPLETE|COMPLETE" "$SB/x.log" | head -5; }

echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
