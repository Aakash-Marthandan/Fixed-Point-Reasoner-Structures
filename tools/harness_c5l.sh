#!/bin/bash
# Ledger: THE WIDTH-192 LONG RUN offline stub harness (2026-09-14; the house law: no chain launches without an end-to-end offline pass; every
# negative scenario asserts the staged failure fired). Reuses tools/harness_champ.sh's sandbox and stubs (its lines 1-157), patched: the
# stub trainer writes a grid every STUB_GRID_EVERY steps; the stub evaluator knows the held-out (split val -> n 10,000) and train-1k (split
# train -> n 1,000) instruments with the split in the provenance, a peak grid and a staged failure; every evaluator call is logged.
# Fixture = the read-only sources: champ/C5_pretrain.tgz (the 50k analog at step 100, with EXTENDED.txt and the champion night's
# RETRY_REMAT.txt), champ/C1_pretrain.tgz, finalA/A5|A8_pretrain.tgz, c8x/val/C5_s* (the reused held-out rows), c8x/sets, the compile cache.
#   Y1 fresh -> COMPLETE (resume 100 -> 200 pretrain-only, no battery, no --remat carried; C5 + wide instruments; picks; test rows on the
#      50k subsample ONLY; champ/ finalA/ c8x/ byte-identical)   Y2 idempotent rerun   Y3 the source state missing -> no training
#   Y4 a node change after PRETRAIN_OK -> nothing re-trained, the missing rows re-run   Y5 a failing row -> INCOMPLETE naming it
#   Y6 a launch-time OOM still retries once with --remat   Y7 the picks coincide with the paper grid -> only the final grid's test rows
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
_HC=$(mktemp /tmp/hc5l_src.XXXXXX); sed -n '1,157p' "$HERE/harness_champ.sh" > "$_HC"   # bash 3.2 (macOS) cannot source a process substitution
grep -q '^n_ok () ' "$_HC" && grep -q '^mk_sandbox () ' "$_HC" || { echo "harness_champ.sh layout changed: lines 1-157 no longer hold the helpers"; exit 2; }
# shellcheck disable=SC1090
source "$_HC"; rm -f "$_HC"

tree_sig () { (cd "$1" 2>/dev/null && find . -type f | sort | while read -r f; do printf '%s %s\n' "$f" "$(cksum < "$f")"; done) | cksum; }
mk_l () {
  mk_sandbox
  cp "$REPO/tools/chain_c5l.sh" "$REPO/tools/c5l_curves.py" "$SB/repo/tools/"
  rm -rf /tmp/c5l_src /tmp/c5l_c5_src.tgz /tmp/c5l_c5_ext.tgz /tmp/c5l_val_*.tgz /tmp/c5l_tr1k_*.tgz /tmp/c5l_test_*.tgz /tmp/c5l_reuse_*.tgz /tmp/c5l_jc_src.tgz
  local G="$SB/gcs/champ" F="$SB/gcs/finalA" X="$SB/gcs/c8x"
  mkdir -p "$G/evals" "$F" "$X/sets" "$X/val"
  : > "$X/sets/sudoku_extreme_seed0_val10k.npz"
  for m in C5_ARM_OK C5_PRETRAIN_OK C1_ARM_OK SYNC-AB; do echo "champion night" > "$G/$m"; done
  "$REAL_PY" - "$SB/fx" <<'PYEOF'
import json, pickle, sys
from pathlib import Path
fx = Path(sys.argv[1])
def run(sub, grids, latest, mon, extra=()):
    d = fx / sub; d.mkdir(parents=True, exist_ok=True)
    for g in grids: pickle.dump({"state": {"model": {}}, "step": g, "config": {}}, open(d / f"ckpt_{g:06d}.pkl", "wb"))
    pickle.dump({"state": {"model": {}}, "step": latest, "config": {}}, open(d / "ckpt_latest.pkl", "wb"))
    rows = []
    for s, v in mon:
        rows.append(json.dumps({"step": s, "loss": .5, "ce_in": .4, "train_exact": .2, "steps_per_sec": 99.0}))
        rows.append(json.dumps({"monitor": {"step": s, "val_t16": v - .01, "val_t16_ema": v, "n_val": 512}}))
    (d / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    (d / "config.json").write_text(json.dumps({"argv": ["tools/pretrain.py", "--seed", "0", "--dec-width", "192"]}))
    for name, text in extra: (d / name).write_text(text)
grids = [20, 40, 60, 80, 100]
run("C5/runs/pretrainchamp_C5", grids, 100, [(g, .30 if g == 80 else .29) for g in grids],
    extra=(("EXTENDED.txt", "EXTENDED from 60 to 100 (peak at 60)\n"), ("RETRY_REMAT.txt", "OOM at launch -> retried once with --remat\n"), ("resumes.txt", "60\n")))
run("C1/runs/pretrainchamp_C1", grids, 100, [(g, .29) for g in grids])
for a in ("A5", "A8"): run(f"{a}/runs/pretrainfinalA_{a}", grids, 100, [(g, .29) for g in grids])
for g in (40, 60, 80, 100):
    d = fx / "reuse" / "runs" / f"c8x_val_pC5_s{g:06d}"; d.mkdir(parents=True, exist_ok=True)
    (d / "summary_all.json").write_text(json.dumps({"n": 10000, "exact_acc": .93 if g == 80 else .91, "split": "val", "ema": True, "t_total": 16, "ckpt": f"runs/pretrainchamp_C5/ckpt_{g:06d}.pkl"}))
(fx / "jc" / "jax_cache").mkdir(parents=True, exist_ok=True); (fx / "jc" / "jax_cache" / "entry").write_text("cache")
PYEOF
  tar czf "$G/C5_pretrain.tgz" -C "$SB/fx/C5" runs/pretrainchamp_C5
  tar czf "$G/C1_pretrain.tgz" -C "$SB/fx/C1" runs/pretrainchamp_C1
  for a in A5 A8; do tar czf "$F/${a}_pretrain.tgz" -C "$SB/fx/$a" "runs/pretrainfinalA_$a"; done
  for g in 000040 000060 000080 000100; do tar czf "$X/val/C5_s$g.tgz" -C "$SB/fx/reuse" "runs/c8x_val_pC5_s$g"; echo ok > "$X/val/C5_s${g}_OK"; done
  tar czf "$G/jax_cache.tgz" -C "$SB/fx/jc" jax_cache
  "$REAL_PY" - "$SB/bin/stubpy" "$SB/evals.log" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text(); log = sys.argv[2]
old = '    mon_steps = sorted({s for s in range(2000, steps + 1, 2000)} | {steps // 2, steps})\n'
new = '    _G = int(os.environ.get("STUB_GRID_EVERY", "2000")); mon_steps = sorted({s for s in range(_G, steps + 1, _G)} | {steps // 2, steps})\n'
assert s.count(old) == 1, "stub grid line"; s = s.replace(old, new)
old = '        (d / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4, "wall_s": 5.0}))\n'
new = ('        s0 = sorted(d.glob("summary_s*.json")); prov0 = json.loads(s0[0].read_text()) if s0 else {}; prov0.pop("n", None)\n'
       '        (d / "summary_all.json").write_text(json.dumps({**prov0, "n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4, "wall_s": 5.0}))\n')
assert s.count(old) == 1, "stub merge line"; s = s.replace(old, new)
old = '    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)\n    arm = out.name.replace("sxscan_pchamp", "")\n'
new = ('    open(' + repr(log) + ', "a").write(" ".join(argv) + "\\n")\n'
       '    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)\n    arm = out.name.replace("sxscan_pchamp", "")\n')
assert s.count(old) == 1, "stub eval head"; s = s.replace(old, new)
old = '    prov = {"t_total": int(flag("--t-total", "64")), "ema": "--ema" in argv, "argv": argv, "ckpt": flag("--ckpt")}\n'
new = ('    prov = {"t_total": int(flag("--t-total", "64")), "ema": "--ema" in argv, "argv": argv, "ckpt": flag("--ckpt"), "split": flag("--split", "test")}\n'
       '    val_acc = None\n'
       '    if flag("--split") in ("val", "train"):\n'
       '        import re as _re\n'
       '        n_total = int(os.environ.get("STUB_VAL_N", "10000")) if flag("--split") == "val" else int(os.environ.get("STUB_TR1K_N", "1000"))\n'
       '        _st = int(_re.search(r"ckpt_(\\d+)", flag("--ckpt")).group(1)); _run = _re.search(r"pretrain(?:champ|finalA)_([CA]\\d)", flag("--ckpt")).group(1)\n'
       '        _kind = "val" if flag("--split") == "val" else "tr1k"\n'
       '        if f"{_run}:{_st:06d}:{_kind}" in os.environ.get("STUB_VAL_FAIL", "").split(","): print("staged row failure", file=sys.stderr); sys.exit(1)\n'
       '        val_acc = 0.95 if (_kind == "val" and _run == "C5" and str(_st) == os.environ.get("STUB_VAL_PEAK", "")) else 0.90\n')
assert s.count(old) == 1, "stub prov line"; s = s.replace(old, new)
old = '        (out / "summary_all.json").write_text(json.dumps({"n": n_total, "exact_acc": .3,'
new = '        (out / "summary_all.json").write_text(json.dumps({"n": n_total, "exact_acc": (val_acc if val_acc is not None else .3),'
assert s.count(old) == 1, "stub summary line"; s = s.replace(old, new)
p.write_text(s)
PYEOF
  SRC_SIG0=$(tree_sig "$SB/gcs/champ")-$(tree_sig "$SB/gcs/finalA")-$(tree_sig "$SB/gcs/c8x")
}
run_l () {  # [VAR=val...]
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 \
     C1_STEPS_X=60 C1_STEPS_LONG=120 C1_EXT_WINDOW=10 LIVE_NO_GUARD=1 LIVE_EVERY=1 STALL_SEC=3 WATCH_POLL=1 STUB_GRID_EVERY=20 \
     C5L_STEP=20 C5L_EXT_FROM=100 C5L_EXT_TO=200 C5L_EARLY="000020" C5L_TR1K_FROM=20 C5L_PICK_MIN=40 C5L_REUSE_LO=40 C5L_REUSE_HI=100 \
     C5L_PAPER_GRID=000080 C5L_WIDE_RUNS="C1 A5 A8" C5L_WIDE_GRIDS="000020 000060 000100" \
     "$@" bash tools/chain_c5l.sh > "$SB/l.log" 2>&1; echo $? > "$SB/l.rc")
}
has () { grep -q -- "$1" "$SB/l.log"; }
line_of () { grep -n -- "$1" "$SB/l.log" | head -1 | cut -d: -f1; }
nok () { ls "$SB/gcs/c5l/$1/"*_OK 2>/dev/null | wc -l | tr -d ' '; }

echo "== Y1 fresh -> COMPLETE =="
mk_l; run_l STUB_VAL_PEAK=160; L="$SB/gcs/c5l"
[ "$(cat "$SB/l.rc")" = 0 ] && has "CHAIN-C5L-COMPLETE" && [ -f "$L/c5l_final.tgz" ] && ok "Y1 rc 0, sentinel, c5l_final.tgz" || { bad "Y1 completion (rc $(cat "$SB/l.rc"))"; tail -15 "$SB/l.log"; }
has "C5L-RESTORE-SRC" && has "C5L-RESUME-POINT step 100" && grep -q "RESUMED from runs/pretrainchamp_C5/ckpt_latest.pkl at step 100" "$SB/repo/runs/pretrainchamp_C5.log" && has "PRETRAIN-EXTENDED-BUDGET C5 200" && ok "Y1 the resume from the banked state at 100 through the extension path to 200" || bad "Y1 resume: $(grep -E 'RESUME|RESTORE|EXTENDED-BUDGET' "$SB/l.log" | tr '\n' ' ')"
! has "REMAT-PERSISTED" && ! pargv C5 | grep -q -- "--remat" && [ ! -f "$SB/repo/runs/pretrainchamp_C5/RETRY_REMAT.txt" ] && ok "Y1 RETRY_REMAT.txt not carried: no --remat in the training argv" || bad "Y1 remat carried: $(pargv C5)"
has "ARM-OK C5 pretrain-only" && ! has "EVAL-OK\|EVAL-FAILED\|DEC-SCAN\|CENSUS-\|CALIB-" && [ ! -d "$SB/repo/runs/sxeval_pchampC5" ] && [ -z "$(ls -d "$SB/repo/runs/sxscreen_pchampC5"* 2>/dev/null)" ] && ok "Y1 pretrain-only: no battery (no screens, no evaluation rows)" || bad "Y1 battery ran: $(grep -E 'EVAL-|SCREEN' "$SB/l.log" | head -3 | tr '\n' ' ')"
[ "$(nok val)" = 15 ] && [ "$(nok tr1k)" = 19 ] && [ -f "$L/val/C5_s000020_OK" ] && [ -f "$L/val/C5_s000200_OK" ] && [ ! -f "$L/val/C5_s000060_OK" ] && [ -f "$L/tr1k/C5_s000060_OK" ] && [ -f "$L/val/A8_s000100_OK" ] && [ -f "$L/tr1k/C1_s000020_OK" ] && ok "Y1 rows: held-out 15 (C5 early + 120-200; 40-100 reused; wide 9), train-1k 19 (C5 20-200; wide 9)" || bad "Y1 rows val $(nok val) tr1k $(nok tr1k)"
ve=$(eargv "$SB/repo/runs/c5l_tr1k_pA5_s000060/summary_all.json")
echo "$ve" | grep -q -- "--split train --t-total 16 --ema" && echo "$ve" | grep -q "val10k" && echo "$ve" | grep -q "pretrainfinalA_A5/ckpt_000060.pkl" && ok "Y1 the train-1k row: split train, D16, EMA, the run's own grid" || bad "Y1 tr1k flags: $ve"
has "C5L-PICKS g_mon=000080 g_val=000160 final=000200 paper=000080" && has "C5L-TEST-ROWS \[d16_s000160 d64_s000160 d16_s000200 d64_s000200\]" && [ "$(nok test)" = 4 ] && ok "Y1 picks (g_val from the merged c5l + c8x curve) and 4 test rows (g_mon = the paper grid: none)" || bad "Y1 picks/test: $(grep -E 'PICKS|TEST-ROWS' "$SB/l.log" | tr '\n' ' ') test $(nok test)"
nfull=$(grep -- "--split test" "$SB/evals.log" | grep -vc -- "--subsample 50000"); ntest=$(grep -c -- "--split test" "$SB/evals.log")
[ "$ntest" -gt 0 ] && [ "$nfull" = 0 ] && ok "Y1 every test-set evaluation on the 50k subsample ($ntest calls; none on the full set)" || bad "Y1 full-set evaluations: $nfull of $ntest"
nt=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/c5l_test_pC5_d64_s000160/summary_all.json'))['n'])")
[ "$nt" = 50000 ] && ok "Y1 the test row's n-gate 50,000" || bad "Y1 test n $nt"
[ "$(tree_sig "$SB/gcs/champ")-$(tree_sig "$SB/gcs/finalA")-$(tree_sig "$SB/gcs/c8x")" = "$SRC_SIG0" ] && ok "Y1 champ/, finalA/, c8x/ byte-identical (nothing written or deleted)" || bad "Y1 a source tree changed"
[ -d "$L/live/runs" ] && [ ! -d "$SB/gcs/champ/live" ] && ok "Y1 the live bank under c5l/live only" || bad "Y1 live prefix"
l1=$(line_of "C5L-LANE1-END"); l2=$(line_of "C5L-LANE2"); l3=$(line_of "C5L-TEST-ROWS"); l4=$(line_of "C5L-LANE4"); lc=$(line_of "CHAIN-C5L-COMPLETE")
[ -n "$l1" ] && [ -n "$l4" ] && [ "$l1" -lt "$l2" ] && [ "$l2" -lt "$l3" ] && [ "$l3" -lt "$l4" ] && [ "$l4" -lt "$lc" ] && ok "Y1 lane order 1 -> 2 -> 3 -> 4 -> sentinel" || bad "Y1 order ($l1 $l2 $l3 $l4 $lc)"
tar tzf "$L/c5l_final.tgz" | grep -q "runs/analysis/c5l_curves.json" && tar tzf "$L/c5l_final.tgz" | grep -q "c5l_tr1k_pA8_s000100/summary_all.json" && tar tzf "$L/c5l_final.tgz" | grep -q "pretrainchamp_C5/metrics.jsonl" && ok "Y1 the manifest carries the curves, the rows and the metrics" || bad "Y1 manifest"
"$REAL_PY" -c "import json; d=json.load(open('$SB/repo/runs/analysis/c5l_curves.json')); assert [s for s,_ in d['C5']['val']] == [20,40,60,80,100,120,140,160,180,200], d['C5']['val']; assert len(d['A8']['tr1k']) == 3" 2>/dev/null && ok "Y1 the curves json merges the reused and new held-out rows" || bad "Y1 curves json"
SBY1=$SB

echo "== Y2 idempotent rerun =="
SB=$SBY1; run_l STUB_VAL_PEAK=160
[ "$(cat "$SB/l.rc")" = 0 ] && has "(final object present)" && ! has "PRETRAIN-START" && ok "Y2 final present -> complete at once" || { bad "Y2"; tail -4 "$SB/l.log"; }
rm -f "$SB/gcs/c5l/c5l_final.tgz"; run_l STUB_VAL_PEAK=160
[ "$(cat "$SB/l.rc")" = 0 ] && ! has "PRETRAIN-START\|C5L-ROW-OK\|C5L-TEST-START" && has "C5L-ROWS-PASS c5-1 0 rows" && has "C5L-ROWS-PASS wide-1 0 rows" && has "FINAL-BANKED" && ok "Y2b markers present -> nothing re-runs; only the manifest" || { bad "Y2b"; grep -E "PRETRAIN-START|ROW-OK|TEST-START|ROWS-PASS" "$SB/l.log" | head -5; }

echo "== Y3 the source state missing -> no training =="
mk_l; rm -f "$SB/gcs/champ/C5_pretrain.tgz"; SRC_SIG0=$(tree_sig "$SB/gcs/champ")-$(tree_sig "$SB/gcs/finalA")-$(tree_sig "$SB/gcs/c8x"); run_l STUB_VAL_PEAK=160
[ "$(cat "$SB/l.rc")" = 2 ] && has "C5L-NO-RESUME-STATE" && ! has "PREFLIGHT\|PRETRAIN-START" && [ ! -d "$SB/repo/runs/pretrainchamp_C5" ] && ok "Y3 NO-RESUME-STATE rc 2: no preflight, no pretrain, nothing from scratch" || { bad "Y3"; tail -5 "$SB/l.log"; }

echo "== Y4 a node change after PRETRAIN_OK =="
mk_l; run_l STUB_VAL_PEAK=160; sleep 3
rm -rf "$SB/repo/runs" /tmp/c5l_src /tmp/c5l_c5_src.tgz /tmp/c5l_c5_ext.tgz /tmp/c5l_reuse_*.tgz; mkdir -p "$SB/repo/runs"
rm -f "$SB/gcs/c5l/c5l_final.tgz" "$SB/gcs/c5l/val/C5_s000140_OK" "$SB/gcs/c5l/tr1k/A5_s000060_OK" "$SB/gcs/c5l/test/d64_s000200_OK"; rm -rf "$SB/gcs/c5l/live"
run_l STUB_VAL_PEAK=160
[ "$(cat "$SB/l.rc")" = 0 ] && has "C5L-PRETRAIN-DONE" && ! has "C5L-RESTORE-SRC" && ! has "PRETRAIN-START" && has "C5L-ROW-OK val/C5_s000140" && has "C5L-ROW-OK tr1k/A5_s000060" && has "C5L-TEST-OK d64_s000200" && has "C5L-VALBEST-REPLAY 000080" && has "(the pick recorded at ARM_OK: 000080)" && has "C5L-PICKS g_mon=000080 g_val=000160" && has "CHAIN-C5L-COMPLETE" && ok "Y4 nothing re-trained; the grids re-pulled from c5l; the monitor pick replayed = the recorded pick; the three missing rows re-run" || { bad "Y4"; grep -E "PRETRAIN|RESTORE|ROW-OK|TEST-OK|PICKS|COMPLETE|INCOMPLETE" "$SB/l.log" | head -8; }

echo "== Y5 a failing row -> INCOMPLETE =="
mk_l; run_l STUB_VAL_PEAK=160 STUB_VAL_FAIL=A5:000060:val
[ "$(cat "$SB/l.rc")" = 1 ] && [ "$(grep -c 'C5L-ROW-N-BAD val/A5_s000060' "$SB/l.log")" = 2 ] && has "C5L-INCOMPLETE worker=0 (missing: val/A5_s000060)" && ! has "CHAIN-C5L-COMPLETE" && [ ! -f "$SB/gcs/c5l/c5l_final.tgz" ] && [ -f "$SB/gcs/c5l/tr1k/A5_s000060_OK" ] && ok "Y5 retried once, INCOMPLETE names the row, no sentinel, the others banked" || { bad "Y5"; grep -E "N-BAD|INCOMPLETE|COMPLETE" "$SB/l.log" | head -4; }

echo "== Y6 a launch-time OOM still retries once with --remat =="
mk_l; run_l STUB_VAL_PEAK=160 STUB_OOM_ARM=C5
[ "$(cat "$SB/l.rc")" = 0 ] && has "PRETRAIN-OOM-RETRY-REMAT C5" && pargv C5 | grep -q -- "--remat" && has "CHAIN-C5L-COMPLETE" && ok "Y6 the OOM retry with --remat (labeled), then COMPLETE" || { bad "Y6"; grep -E "OOM|REMAT|COMPLETE" "$SB/l.log" | head -4; }

echo "== Y7 the picks coincide with the paper grid =="
mk_l; run_l STUB_VAL_PEAK=
[ "$(cat "$SB/l.rc")" = 0 ] && has "C5L-PICKS g_mon=000080 g_val=000080" && has "C5L-TEST-ROWS \[d16_s000200 d64_s000200\]" && [ "$(nok test)" = 2 ] && ok "Y7 g_val = g_mon = the paper grid (the reused peak): only the final grid's two test rows" || { bad "Y7"; grep -E "PICKS|TEST-ROWS" "$SB/l.log"; }

echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
