#!/bin/bash
# Ledger: THE X5 LONG RUN offline stub harness (2026-09-17; the house law: no chain launches without an end-to-end offline pass).
# Reuses tools/harness_champ.sh's sandbox, stubs and helpers (its lines 1-157) and adds tools/filler_full.sh + tools/chain_x5long.sh with the
# pending Sudoku runs' end state as the fixture: the champion prefix ($SRC) holds X5's banked 50k state (X5_pretrain.tgz: ckpt_latest at
# 50,000, its grids and monitor rows) and X5's completion markers, which the long run must neither read as its own nor write. Scenarios:
#   L1 fresh -> COMPLETE: the 50k state restored from $SRC, the resume 50,000 -> 960,000 through the registered extension path (RESUMED,
#      never from scratch), the long cadence and X5's levers at the trainer, the champion battery + the k128 row on the selected grid under
#      the fresh prefix, $SRC byte-identical, the manifest, the sentinel order.
#   L2 idempotent rerun / the manifest alone rebuilt.   L3 no resume state, L3b a state short of 50,000 -> X5L-NO-RESUME-STATE rc 2, nothing trained.
#   L4 a preemption: the live prefix holds a later state -> the resume point is the live one, $SRC's tarball never extracted.
#   L5 X5's preflight failure -> abort, INCOMPLETE.     L6 a relaunch after the pretrain is banked -> PRETRAIN-SKIP, the battery completes.
#   L7 a budget that would break the six-digit grid names -> X5L-BAD-BUDGET rc 2.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
_HC=$(mktemp /tmp/hxl_src.XXXXXX); sed -n '1,157p' "$HERE/harness_champ.sh" > "$_HC"
grep -q '^n_ok () ' "$_HC" && grep -q '^mk_sandbox () ' "$_HC" || { echo "harness_champ.sh layout changed: lines 1-157 no longer hold the helpers"; exit 2; }
# shellcheck disable=SC1090
source "$_HC"; rm -f "$_HC"

X5FLAGS="--cell trm --trm-hidden 160 --sudoku-digit-aug --fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --trm-ri-sigma 1.0 --seed 0"
mk_xl () {  # the champion prefix as the pending runs left it; the fresh x5l prefix empty
  mk_sandbox
  cp "$REPO/tools/filler_full.sh" "$REPO/tools/chain_x5long.sh" "$SB/repo/tools/"
  rm -rf /tmp/filler_pull /tmp/x5l_src.tgz /tmp/x5l_jc_src.tgz
  local G="$SB/gcs/champ"
  mkdir -p "$G/filler" "$G/evals" "$SB/fx"
  for a in C0 C1 C2 C3 C4 C5 C6 C7 C8 X5; do echo ok > "$G/${a}_ARM_OK"; echo ok > "$G/${a}_PRETRAIN_OK"; done
  echo ok > "$G/filler/k128_X5_OK"; echo ok > "$G/SYNC-AB"; printf 'BATCHONLY' > "$G/RECIPE-DEC"; echo "the pending runs' manifest" > "$G/sudokupend_final.tgz"
  echo ok > "$G/PREFLIGHT_OK_w0_nw1"
  if [ "${NO_STATE:-0}" != 1 ]; then   # X5's banked 50k state = the stub trainer's own 50k run (rising to its end, as the real one)
    (cd "$SB/fx" && STUB_PEAK_LAST_ARM=X5 "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainchamp_X5 $X5FLAGS --steps "${STATE_STEPS:-50000}" > /dev/null 2>&1)
    tar czf "$G/X5_pretrain.tgz" -C "$SB/fx" runs/pretrainchamp_X5
    cp "$SB/fx/runs/pretrainchamp_X5/ckpt_latest.pkl" "$G/X5_ckpt.pkl"
  fi
  # the stub merge carries the shards' provenance (ckpt, argv), as the real evaluator's summary_all.json does (grid_of reads its ckpt)
  "$REAL_PY" - "$SB/bin/stubpy" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
old = '        (d / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4, "wall_s": 5.0}))\n'
new = ('        s0 = sorted(d.glob("summary_s*.json")); prov0 = json.loads(s0[0].read_text()) if s0 else {}; prov0.pop("n", None)\n'
       '        (d / "summary_all.json").write_text(json.dumps({**prov0, "n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4, "wall_s": 5.0}))\n')
assert s.count(old) == 1, "stub merge line not found"
p.write_text(s.replace(old, new))
PYEOF
  (cd "$SB/gcs/champ" && find . -type f | sort | while read -r f; do printf '%s %s\n' "$(cksum < "$f")" "$f"; done) > "$SB/src_before.txt"
}
run_xl () {  # [VAR=val...]
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 \
     GCS=gs://qhrrn2-rescue/x5l X5L_SRC=gs://qhrrn2-rescue/champ LIVE_PREFIX=gs://qhrrn2-rescue/x5l/live \
     C1_STEPS_X=30000 C1_EXT_WINDOW=4000 LIVE_NO_GUARD=1 LIVE_EVERY=1 STALL_SEC=3 WATCH_POLL=1 PASS_SLEEP=1 IDLE_POLL=1 XL_PASSES=2 "$@" \
     bash tools/chain_x5long.sh > "$SB/xl.log" 2>&1; echo $? > "$SB/xl.rc")
}
src_same () { (cd "$SB/gcs/champ" && find . -type f | sort | while read -r f; do printf '%s %s\n' "$(cksum < "$f")" "$f"; done) | cmp -s - "$SB/src_before.txt"; }
line_of () { grep -n "$1" "$SB/xl.log" | head -1 | cut -d: -f1; }
ckst () { "$REAL_PY" -c "import pickle,sys; print(pickle.load(open(sys.argv[1],'rb'))['step'])" "$1"; }

echo "== L1 fresh -> COMPLETE (X5's monitor rising to the end: the selected grid is the budget's last) =="
mk_xl; run_xl STUB_PEAK_LAST_ARM=X5; G="$SB/gcs/x5l"; PL="$SB/repo/runs/pretrainchamp_X5.log"
[ "$(cat "$SB/xl.rc")" = 0 ] && grep -q "CHAIN-X5LONG-COMPLETE" "$SB/xl.log" && [ -f "$G/x5long_final.tgz" ] && ok "L1 rc 0, sentinel, x5long_final.tgz" || { bad "L1 completion (rc $(cat "$SB/xl.rc"))"; tail -14 "$SB/xl.log"; }
grep -q "X5L-RESTORE-SRC" "$SB/xl.log" && grep -q "X5L-RESUME-POINT step 50000" "$SB/xl.log" && ok "L1 the 50k state restored from the champion prefix; resume point 50000" || bad "L1 resume point"
grep -q "PRETRAIN-EXTENDED-BUDGET X5 960000" "$SB/xl.log" && grep -q "PRETRAIN-START X5 .*steps=960000" "$SB/xl.log" && ! grep -q "PRETRAIN-EXTEND X5" "$SB/xl.log" && ok "L1 the registered budget 960,000 through the extension path (the window rule never fires)" || bad "L1 budget: $(grep 'PRETRAIN-' "$SB/xl.log" | tr '\n' ' ')"
grep -q "RESUMED from .* at step 50000" "$PL" && [ "$(ckst "$SB/repo/runs/pretrainchamp_X5/ckpt_latest.pkl")" = 960000 ] && ok "L1 the trainer RESUMED at 50,000 (never from scratch) and ended at 960,000" || bad "L1 resume in the trainer log: $(head -2 "$PL")"
"$REAL_PY" - "$SB/repo/runs/pretrainchamp_X5/metrics.jsonl" <<'PYEOF' && ok "L1 one continuous curve: the 50k run's monitor rows kept, the long run's appended, no step twice" || bad "L1 metrics continuity"
import json, sys
st = [json.loads(l)["monitor"]["step"] for l in open(sys.argv[1]) if '"monitor"' in l]
assert st == sorted(st) and len(st) == len(set(st)), "duplicate or unordered monitor steps"
assert min(st) <= 2000 and 50000 in st and max(st) == 960000, (min(st), max(st))
PYEOF
x5=$(pargv X5)
echo "$x5" | grep -q -- "$X5FLAGS" && echo "$x5" | grep -q -- "--monitor-every 10000 --grid-every 10000 --ckpt-every 5000 " && echo "$x5" | grep -q -- "--steps 960000" && echo "$x5" | grep -q -- "--batch 768 --wd 1.0 " && echo "$x5" | grep -q -- "--lr 1e-4 --lr-end 1e-4" && ! echo "$x5" | grep -q -- "--cell dec\|--dec-width" && ok "L1 the trainer got X5's levers, the constant lr, the long cadence (10k / 10k / 5k) and --steps 960000" || bad "L1 trainer argv: $x5"
[ "$(grep -c 'PREFLIGHT-OK' "$SB/xl.log")" = 1 ] && grep -q "PREFLIGHT-OK X5" "$SB/xl.log" && ok "L1 X5 preflights under the fresh prefix (the champion prefix's PREFLIGHT marker is not read)" || bad "L1 preflight: $(grep PREFLIGHT "$SB/xl.log" | tr '\n' ' ')"
grep -q "VALBEST X5 960000" "$SB/xl.log" && ok "L1 the registered selection over every grid picks the long run's peak (960000)" || bad "L1 selection: $(grep VALBEST "$SB/xl.log")"
allb=1; for r in full_X5_vsel_t16 full_X5_final_t16 full_X5_vsel_t16_alt full_X5_vsel_t64 d128_X5 d256_X5 scan_X5 census_X5_vsel census_X5_final calib_X5_vsel screen_X5_vb screen_X5_s010000 screen_X5_s020000 screen_X5_s030000 screen_X5_s040000; do [ -f "$G/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done
[ $allb = 1 ] && [ -f "$G/X5_ARM_OK" ] && [ -f "$G/X5_pretrain.tgz" ] && [ -f "$G/X5_PRETRAIN_OK" ] && ok "L1 the champion battery (11 rows + four screens), the pretrain bank and ARM_OK under the fresh prefix" || bad "L1 battery"
s16=$("$REAL_PY" -c "import json; s=json.load(open('$SB/repo/runs/sxeval_pchampX5/full_vsel_t16/summary_all.json')); print(s['n'], s.get('ckpt'))")
[ "$s16" = "422786 runs/pretrainchamp_X5/ckpt_960000.pkl" ] && ok "L1 D16 on all 422,786, on the selected grid" || bad "L1 D16 row: $s16"
kx=$(eargv "$SB/repo/runs/filler_sxscan128_pchampX5/summary_s0.json")
[ -f "$G/filler/k128_X5_OK" ] && [ -f "$G/filler/k128_X5.tgz" ] && echo "$kx" | grep -q -- "--ckpt runs/pretrainchamp_X5/ckpt_960000.pkl" && echo "$kx" | grep -q -- "--split test --subsample 5000 --t-total 64 --k-init 128 --ema" && ok "L1 the k128 row on the long run's selected grid with the headline scan's flags" || bad "L1 k128: $kx"
src_same && ok "L1 the champion prefix byte-identical (nothing written under \$SRC; its X5 markers not taken as ours)" || { bad "L1 \$SRC changed"; (cd "$SB/gcs/champ" && find . -type f | sort) | diff - <(awk '{print $3}' "$SB/src_before.txt") | head -5; }
[ -d "$SB/gcs/x5l/live/runs" ] && [ ! -d "$SB/gcs/champ/live" ] && ok "L1 the live bank under the fresh prefix only" || bad "L1 live prefix"
grep -q "EXTENDED from 50000 to 960000" "$G/X5_EXTENDED" && [ "$(cat "$G/RECIPE-DEC")" = BATCHONLY ] && [ -f "$G/SYNC-AB" ] && ! grep -q "SYNC-AB-OK" "$SB/xl.log" && ok "L1 the pre-staged markers (budget, scan recipe, no sync rider)" || bad "L1 pre-staged markers"
ch=$(line_of "CHAIN-CHAMPXL-COMPLETE"); f1=$(line_of "XL-FILLER arms"); sp=$(line_of "CHAIN-X5LONG-COMPLETE")
[ -n "$ch" ] && [ -n "$f1" ] && [ -n "$sp" ] && [ "$ch" -lt "$f1" ] && [ "$f1" -lt "$sp" ] && ok "L1 order: the inner chain's sentinel, the filler, the outer sentinel" || bad "L1 sentinel order ($ch $f1 $sp)"
! grep -q "SELF-TEARDOWN" "$SB/xl.log" && ok "L1 no self-teardown (the supervisor owns it)" || bad "L1 self-teardown"
m=$(tar tzf "$G/x5long_final.tgz"); echo "$m" | grep -q "pretrainchamp_X5/metrics.jsonl" && echo "$m" | grep -q "pretrainchamp_X5/resumes.txt\|pretrainchamp_X5/EXTENDED.txt" && echo "$m" | grep -q "sxeval_pchampX5/full_vsel_t16/summary_all.json" && echo "$m" | grep -q "filler_sxscan128_pchampX5/records_all.npz" && ok "L1 the manifest carries the curve, the budget label, the summaries and the k128 records" || bad "L1 manifest: $(echo "$m" | tr '\n' ' ' | cut -c1-300)"
SBL1=$SB

echo "== L2 idempotent rerun =="
SB=$SBL1; run_xl STUB_PEAK_LAST_ARM=X5
[ "$(cat "$SB/xl.rc")" = 0 ] && grep -q "(final object present)" "$SB/xl.log" && ! grep -q "PRETRAIN-START\|FILLER-JOB-START" "$SB/xl.log" && ok "L2 rerun: complete at once, nothing re-run" || { bad "L2 rerun"; tail -5 "$SB/xl.log"; }
rm -f "$SB/gcs/x5l/x5long_final.tgz"; run_xl STUB_PEAK_LAST_ARM=X5
[ "$(cat "$SB/xl.rc")" = 0 ] && ! grep -q "PRETRAIN-START\|FILLER-JOB-START" "$SB/xl.log" && grep -q "X5L-PRETRAIN-DONE" "$SB/xl.log" && grep -q "FINAL-BANKED" "$SB/xl.log" && [ -f "$SB/gcs/x5l/x5long_final.tgz" ] && ok "L2b markers present, manifest absent -> only the manifest is rebuilt" || bad "L2b manifest rebuild"

echo "== L3 no resume state -> the long run never trains from scratch =="
NO_STATE=1 mk_xl; run_xl
[ "$(cat "$SB/xl.rc")" = 2 ] && grep -q "X5L-NO-RESUME-STATE" "$SB/xl.log" && ! grep -q "PRETRAIN-START\|XL-LANE1\|CHAIN-X5LONG-COMPLETE" "$SB/xl.log" && [ ! -f "$SB/gcs/x5l/X5_ARM_OK" ] && ok "L3 rc 2, no lane 1, no pretrain, no sentinel" || { bad "L3 no-state (rc $(cat "$SB/xl.rc"))"; tail -5 "$SB/xl.log"; }

echo "== L3b a state SHORT of the resume point (30,000 < 50,000) is refused too =="
STATE_STEPS=30000 mk_xl; run_xl
[ "$(cat "$SB/xl.rc")" = 2 ] && grep -q "X5L-NO-RESUME-STATE (ckpt_latest step 30000 < 50000" "$SB/xl.log" && ! grep -q "PRETRAIN-START\|XL-LANE1" "$SB/xl.log" && ok "L3b rc 2: a 30k state is not the registered resume point" || { bad "L3b short state (rc $(cat "$SB/xl.rc"))"; tail -4 "$SB/xl.log"; }

echo "== L4 a preemption: the live prefix holds a later state =="
mk_xl
mkdir -p "$SB/lv" && (cd "$SB/lv" && mkdir -p runs && tar xzf "$SB/gcs/champ/X5_pretrain.tgz" && STUB_PEAK_LAST_ARM=X5 "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainchamp_X5 $X5FLAGS --steps 400000 > /dev/null 2>&1)
mkdir -p "$SB/gcs/x5l/live/runs" && cp -R "$SB/lv/runs/pretrainchamp_X5" "$SB/gcs/x5l/live/runs/"
echo "EXTENDED from 50000 to 960000 (pre-staged by the first launch)" > "$SB/gcs/x5l/X5_EXTENDED"
run_xl STUB_PEAK_LAST_ARM=X5; PL="$SB/repo/runs/pretrainchamp_X5.log"
[ "$(cat "$SB/xl.rc")" = 0 ] && grep -q "X5L-RESUME-POINT step 400000" "$SB/xl.log" && ! grep -q "X5L-RESTORE-SRC" "$SB/xl.log" && grep -q "RESUMED from .* at step 400000" "$PL" && grep -q "PRETRAIN-START X5 .*steps=960000" "$SB/xl.log" && grep -q "CHAIN-X5LONG-COMPLETE" "$SB/xl.log" && ok "L4 the live state (400,000) is the resume point; the 50k tarball is never extracted over it; the budget stands; COMPLETE" || { bad "L4 preemption resume"; grep "X5L-\|PRETRAIN-" "$SB/xl.log" | head -6; head -2 "$PL"; }
src_same && ok "L4 the champion prefix byte-identical" || bad "L4 \$SRC changed"

echo "== L5 X5's preflight failure -> abort, INCOMPLETE =="
mk_xl; run_xl STUB_PREFLIGHT_FAIL=X5
[ "$(cat "$SB/xl.rc")" = 1 ] && grep -q "CHAMPXL-PREFLIGHT-ABORT" "$SB/xl.log" && ! grep -q "PRETRAIN-START" "$SB/xl.log" && grep -q "XL-FILLER-SKIP" "$SB/xl.log" && grep -q "X5LONG-INCOMPLETE" "$SB/xl.log" && ! grep -q "CHAIN-X5LONG-COMPLETE" "$SB/xl.log" && ok "L5 preflight failure: no pretrain, no filler, INCOMPLETE, no sentinel" || { bad "L5 preflight abort (rc $(cat "$SB/xl.rc"))"; tail -5 "$SB/xl.log"; }

echo "== L6 a relaunch on a fresh node after the pretrain is banked =="
SB=$SBL1; G="$SB/gcs/x5l"
rm -f "$G/x5long_final.tgz" "$G/X5_ARM_OK" "$G/filler/k128_X5_OK" "$G/filler/k128_X5.tgz" "$G/evals/full_X5_vsel_t64_OK" "$G/evals/full_X5_vsel_t64.tgz"; rm -rf "$SB/repo/runs" "$G/live" /tmp/filler_pull; mkdir -p "$SB/repo/runs"
run_xl STUB_PEAK_LAST_ARM=X5
[ "$(cat "$SB/xl.rc")" = 0 ] && grep -q "X5L-PRETRAIN-DONE" "$SB/xl.log" && grep -q "PRETRAIN-SKIP X5" "$SB/xl.log" && ! grep -q "PRETRAIN-START\|X5L-RESTORE-SRC" "$SB/xl.log" && grep -q "PRETRAIN-RESTORE X5" "$SB/xl.log" && grep -q "VALBEST X5 960000" "$SB/xl.log" && [ -f "$G/evals/full_X5_vsel_t64_OK" ] && [ -f "$G/filler/k128_X5_OK" ] && grep -q "CHAIN-X5LONG-COMPLETE" "$SB/xl.log" && ok "L6 PRETRAIN-SKIP, the long run's grids re-pulled from the fresh prefix (never the 50k tarball), the missing rows re-run, COMPLETE" || { bad "L6 relaunch after the bank (rc $(cat "$SB/xl.rc"))"; grep "X5L-\|PRETRAIN-\|VALBEST\|INCOMPLETE" "$SB/xl.log" | head -8; }

echo "== L7 a budget past the six-digit grid names is refused =="
mk_xl; run_xl X5L_EXT_TO=1000000
[ "$(cat "$SB/xl.rc")" = 2 ] && grep -q "X5L-BAD-BUDGET" "$SB/xl.log" && ! grep -q "XL-LANE1" "$SB/xl.log" && ok "L7 rc 2, nothing launched" || bad "L7 budget guard (rc $(cat "$SB/xl.rc"))"

echo; echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
