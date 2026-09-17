#!/bin/bash
# Ledger: THE PENDING SUDOKU RUNS offline stub harness (2026-09-17; the house law: no chain launches without an end-to-end offline pass).
# Reuses tools/harness_champ.sh's sandbox, stubs and helpers (its lines 1-157) and adds tools/filler_full.sh + tools/chain_sudokupend.sh with
# the paper-final state as the fixture (C7 and C8 done under the champion prefix with C7's banked selected grid at 46k; C8's paper grid under
# the c8x prefix = the registered override; the champion night's k128 rows for C0-C6; a stale champ/live prefix). Scenarios:
#   Q1 fresh -> COMPLETE: X5 through the champion battery on a FIXED budget (PRETRAIN-FIXED-BUDGET, never PRETRAIN-EXTEND), the three k128 rows
#      with the frontier scan's flags, C8's row on the override grid (046000), C7's on its banked grid, X5's on its selected grid; the manifest.
#   Q2 idempotent rerun; Q3 the C8 override object missing -> INCOMPLETE, no sentinel; Q4 X5's preflight failure -> abort, INCOMPLETE.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
_HC=$(mktemp /tmp/hsp_src.XXXXXX); sed -n '1,157p' "$HERE/harness_champ.sh" > "$_HC"
grep -q '^n_ok () ' "$_HC" && grep -q '^mk_sandbox () ' "$_HC" || { echo "harness_champ.sh layout changed: lines 1-157 no longer hold the helpers"; exit 2; }
# shellcheck disable=SC1090
source "$_HC"; rm -f "$_HC"

mk_sp () {
  mk_sandbox
  cp "$REPO/tools/filler_full.sh" "$REPO/tools/chain_sudokupend.sh" "$SB/repo/tools/"
  rm -rf /tmp/filler_pull
  local G="$SB/gcs/champ"
  mkdir -p "$G/filler" "$G/evals" "$SB/gcs/champ/live/runs" "$SB/gcs/c8x"
  for a in C0 C1 C2 C3 C4 C5 C6 C7 C8; do echo ok > "$G/${a}_ARM_OK"; echo ok > "$G/${a}_PRETRAIN_OK"; done
  for a in C0 C1 C2 C3 C4 C5 C6; do echo ok > "$G/filler/k128_${a}_OK"; done
  echo ok > "$G/SYNC-AB"; printf 'BATCHONLY' > "$G/RECIPE-DEC"; echo "the paper-final manifest" > "$G/paperfinal_final.tgz"
  echo stale > "$SB/gcs/champ/live/runs/STALE_CHAMP_LIVE_MARKER"
  mkdir -p "$SB/fx/runs/sxeval_pchampC7/full_vsel_t16" "$SB/fx/runs/pretrainchamp_C7" "$SB/fx8/runs/pretrainchamp_C8"
  "$REAL_PY" - "$SB/fx" "$SB/fx8" <<'PYEOF'
import json, pickle, sys
from pathlib import Path
fx, fx8 = Path(sys.argv[1]), Path(sys.argv[2])
(fx / "runs/sxeval_pchampC7/full_vsel_t16/summary_all.json").write_text(json.dumps({"n": 422786, "ckpt": "runs/pretrainchamp_C7/ckpt_046000.pkl"}))
pickle.dump({"state": {"model": {}}, "step": 46000, "config": {}}, open(fx / "runs/pretrainchamp_C7/ckpt_046000.pkl", "wb"))
pickle.dump({"state": {"model": {}}, "step": 46000, "config": {}}, open(fx8 / "runs/pretrainchamp_C8/ckpt_046000.pkl", "wb"))
PYEOF
  tar czf "$G/evals/full_C7_vsel_t16.tgz" -C "$SB/fx" runs/sxeval_pchampC7/full_vsel_t16
  tar czf "$G/C7_pretrain.tgz" -C "$SB/fx" runs/pretrainchamp_C7
  [ "${NO_C8X:-0}" = 1 ] || tar czf "$SB/gcs/c8x/C8_pretrain.tgz" -C "$SB/fx8" runs/pretrainchamp_C8
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
}
run_sp () {  # [VAR=val...]
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 \
     C1_STEPS_X=30000 C1_STEPS_LONG=50000 C1_EXT_STEPS=20000 C1_EXT_WINDOW=4000 LIVE_NO_GUARD=1 LIVE_EVERY=1 STALL_SEC=3 WATCH_POLL=1 \
     PASS_SLEEP=1 IDLE_POLL=1 SP_PASSES=2 FILLER_CK_C8="$SB/gcs/c8x/C8_pretrain.tgz,runs/pretrainchamp_C8/ckpt_046000.pkl" "$@" \
     bash tools/chain_sudokupend.sh > "$SB/sp.log" 2>&1; echo $? > "$SB/sp.rc")
}
line_of () { grep -n "$1" "$SB/sp.log" | head -1 | cut -d: -f1; }

echo "== Q1 fresh -> COMPLETE (full-scale stub budgets, as the champion harness's S3d; X5's monitor rising to the end so a C arm WOULD extend) =="
mk_sp; run_sp STUB_PEAK_LAST_ARM=X5; G="$SB/gcs/champ"
[ "$(cat "$SB/sp.rc")" = 0 ] && grep -q "CHAIN-SUDOKUPEND-COMPLETE" "$SB/sp.log" && [ -f "$G/sudokupend_final.tgz" ] && ok "Q1 rc 0, sentinel, sudokupend_final.tgz" || { bad "Q1 completion (rc $(cat "$SB/sp.rc"))"; tail -14 "$SB/sp.log"; }
[ "$(grep -c 'PREFLIGHT-OK' "$SB/sp.log")" = 1 ] && grep -q "PREFLIGHT-OK X5" "$SB/sp.log" && ok "Q1 preflight of X5 only" || bad "Q1 preflight: $(grep PREFLIGHT "$SB/sp.log" | tr '\n' ' ')"
grep -q "PRETRAIN-START X5 .*steps=50000" "$SB/sp.log" && grep -q "PRETRAIN-FIXED-BUDGET X5 50000" "$SB/sp.log" && ! grep -q "PRETRAIN-EXTEND\|EXTENDED from" "$SB/sp.log" && [ ! -f "$G/X5_EXTENDED" ] && grep -qE "VALBEST X5 0(48|50)000" "$SB/sp.log" && ok "Q1 X5 on the fixed 50k budget, its selected grid inside the last 4k, and NOT extended (a C arm would have been)" || bad "Q1 budget/extension: $(grep -E 'PRETRAIN-(START|FIXED|EXTEND|NO-EXTEND)|VALBEST X5' "$SB/sp.log" | tr '\n' ' ')"
x5=$(pargv X5)
echo "$x5" | grep -q -- "--cell trm --trm-hidden 160 --sudoku-digit-aug --fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --trm-ri-sigma 1.0 --seed 0" && echo "$x5" | grep -q -- "--sudoku-aug 1000 " && echo "$x5" | grep -q -- "--batch 768 --wd 1.0 " && ! echo "$x5" | grep -q -- "--cell dec\|--dec-width" && ok "Q1 X5 = the champion loop + TRM's cell at hidden 160 + the two levers + digit aug (no DEC flags)" || bad "Q1 X5 flags: $x5"
allb=1; for r in full_X5_vsel_t16 full_X5_final_t16 full_X5_vsel_t16_alt full_X5_vsel_t64 d128_X5 d256_X5 scan_X5 census_X5_vsel census_X5_final calib_X5_vsel screen_X5_vb; do [ -f "$G/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done
for st in 010000 020000 030000 040000; do [ -f "$G/evals/screen_X5_s${st}_OK" ] || { allb=0; echo "    missing screen_X5_s$st"; }; done
[ $allb = 1 ] && [ -f "$G/X5_ARM_OK" ] && ok "Q1 the champion battery on X5 (11 rows + the four fixed-step screens at 10k-40k) + ARM_OK" || bad "Q1 battery"
allf=1; for m in k128_C7 k128_C8 k128_X5; do [ -f "$G/filler/${m}_OK" ] && [ -f "$G/filler/${m}.tgz" ] || { allf=0; echo "    missing filler $m"; }; done
[ $allf = 1 ] && ok "Q1 the three k128 rows banked (tgz + OK)" || bad "Q1 k128 rows"
k7=$(eargv "$SB/repo/runs/filler_sxscan128_pchampC7/summary_s0.json"); k8=$(eargv "$SB/repo/runs/filler_sxscan128_pchampC8/summary_s0.json"); kx=$(eargv "$SB/repo/runs/filler_sxscan128_pchampX5/summary_s0.json")
echo "$k7" | grep -q -- "--ckpt runs/pretrainchamp_C7/ckpt_046000.pkl" && echo "$k8" | grep -q -- "--ckpt runs/pretrainchamp_C8/ckpt_046000.pkl" && echo "$kx" | grep -q -- "--ckpt runs/pretrainchamp_X5/ckpt_" && ok "Q1 C7 on its banked 46k grid, C8 on the OVERRIDE grid (c8x 46k), X5 on its selected grid" || bad "Q1 k128 grids: C7[$k7] C8[$k8] X5[$kx]"
for k in "$k7" "$k8" "$kx"; do echo "$k" | grep -q -- "--split test --subsample 5000 --t-total 64 --k-init 128 --ema" && echo "$k" | grep -q "mon512" || { bad "Q1 k128 flags: $k"; break; }; done; ok "Q1 the k128 rows carry the frontier headline scan's flags (5k, t64, k128, EMA, the mon512 corpus)"
nk=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/filler_sxscan128_pchampC8/summary_all.json'))['n'])"); n16=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/sxeval_pchampX5/full_vsel_t16/summary_all.json'))['n'])")
[ "$nk" = 5000 ] && [ "$n16" = 422786 ] && ok "Q1 n-gates 5000 / 422,786" || bad "Q1 n ($nk $n16)"
[ "$(cat "$G/paperfinal_final.tgz")" = "the paper-final manifest" ] && [ -f "$G/filler/k128_C5_OK" ] && [ ! -f "$G/evals/full_C7_vsel_t16_OK.new" ] && ok "Q1 the paper-final and champion markers untouched" || bad "Q1 earlier markers touched"
[ ! -e "$SB/repo/runs/STALE_CHAMP_LIVE_MARKER" ] && [ -d "$SB/gcs/champ_pend/live/runs" ] && grep -q "champ_pend/live" "$SB/sp.log" && ok "Q1 the fresh live prefix (champ_pend/live); the stale champion live prefix never restored" || bad "Q1 live prefix"
ch=$(line_of "CHAIN-CHAMPX-COMPLETE"); f1=$(line_of "SP-FILLER"); sp=$(line_of "CHAIN-SUDOKUPEND-COMPLETE")
[ -n "$ch" ] && [ -n "$f1" ] && [ -n "$sp" ] && [ "$ch" -lt "$f1" ] && [ "$f1" -lt "$sp" ] && ok "Q1 order: the inner chain's sentinel, the filler, the supervisor's sentinel" || bad "Q1 sentinel order ($ch $f1 $sp)"
! grep -q "SELF-TEARDOWN" "$SB/sp.log" && ok "Q1 no self-teardown (the supervisor owns it)" || bad "Q1 self-teardown"
tar tzf "$G/sudokupend_final.tgz" | grep -q "filler_sxscan128_pchampC8/summary_all.json" && tar tzf "$G/sudokupend_final.tgz" | grep -q "pretrainchamp_X5/metrics.jsonl" && tar tzf "$G/sudokupend_final.tgz" | grep -q "sxscan_pchampX5/summary_all.json" && ok "Q1 the manifest carries the k128 summaries, X5's metrics and its k32 scan" || bad "Q1 manifest: $(tar tzf "$G/sudokupend_final.tgz" | head -8 | tr '\n' ' ')"
SBQ1=$SB

echo "== Q2 idempotent rerun =="
SB=$SBQ1; run_sp
[ "$(cat "$SB/sp.rc")" = 0 ] && grep -q "(final object present)" "$SB/sp.log" && ! grep -q "PRETRAIN-START\|FILLER-JOB-START" "$SB/sp.log" && ok "Q2 rerun: complete at once, nothing re-run" || { bad "Q2 rerun"; tail -5 "$SB/sp.log"; }
rm -f "$SB/gcs/champ/sudokupend_final.tgz"; run_sp
[ "$(cat "$SB/sp.rc")" = 0 ] && ! grep -q "PRETRAIN-START\|FILLER-JOB-START" "$SB/sp.log" && grep -q "FINAL-BANKED" "$SB/sp.log" && [ -f "$SB/gcs/champ/sudokupend_final.tgz" ] && ok "Q2b markers present, manifest absent -> only the manifest is rebuilt" || bad "Q2b manifest rebuild"

echo "== Q3 the C8 override object missing -> INCOMPLETE, no sentinel =="
NO_C8X=1 mk_sp; run_sp
[ "$(cat "$SB/sp.rc")" = 1 ] && grep -q "FILLER-NO-GRID C8" "$SB/sp.log" && grep -q "SUDOKUPEND-INCOMPLETE.*k128_C8_OK" "$SB/sp.log" && ! grep -q "CHAIN-SUDOKUPEND-COMPLETE" "$SB/sp.log" && [ ! -f "$SB/gcs/champ/sudokupend_final.tgz" ] && [ -f "$SB/gcs/champ/filler/k128_C7_OK" ] && [ -f "$SB/gcs/champ/filler/k128_X5_OK" ] && ok "Q3 INCOMPLETE without the override object; C7's and X5's rows still banked; no sentinel" || { bad "Q3 override missing (rc $(cat "$SB/sp.rc"))"; grep -E "FILLER-NO-GRID|INCOMPLETE|COMPLETE" "$SB/sp.log" | head -5; }

echo "== Q4 X5's preflight failure -> abort, INCOMPLETE =="
mk_sp; run_sp STUB_PREFLIGHT_FAIL=X5
[ "$(cat "$SB/sp.rc")" = 1 ] && grep -q "CHAMPX-PREFLIGHT-ABORT" "$SB/sp.log" && ! grep -q "PRETRAIN-START" "$SB/sp.log" && grep -q "SUDOKUPEND-INCOMPLETE" "$SB/sp.log" && ! grep -q "CHAIN-SUDOKUPEND-COMPLETE" "$SB/sp.log" && ok "Q4 preflight failure: no pretrain, INCOMPLETE, no sentinel" || { bad "Q4 preflight (rc $(cat "$SB/sp.rc"))"; grep -E "PREFLIGHT|INCOMPLETE|COMPLETE" "$SB/sp.log" | head -5; }

echo; echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
