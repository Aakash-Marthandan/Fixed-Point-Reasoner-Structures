#!/bin/bash
# Ledger: PAPER FINAL RUNS offline stub harness (2026-09-13; the house law: no chain launches without an end-to-end offline pass).
# Reuses tools/harness_champ.sh's sandbox, stubs and helpers (its lines 1-157: gsutil -> a local FAKE_GCS, $CHAIN_PY -> the stub
# trainer/evaluator/census/calibration, select_ckpt.py REAL) and adds tools/filler_full.sh + tools/chain_paperfinal.sh with the
# champion night's GCS state as the fixture (C0-C6 done, the rider, BATCHONLY, champ_final.tgz, the stale d64full_C4_CLAIM_w2,
# C4's banked grid, EqR's ported checkpoint, a stale champ/live prefix). Scenarios: P1 fresh -> COMPLETE (C7/C8 through the
# champion battery, the filler rows, EqR's k128 on the identical 5k, C4's full row despite the stale claim; nothing of the champion
# night touched; the fresh live prefix); P2 idempotent rerun; P3 the ported checkpoint missing -> INCOMPLETE, no sentinel;
# P4 a seed-arm preflight failure -> the pair stops, the riders still bank, INCOMPLETE; P5 the extension rule on C7 only.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1090
_HC=$(mktemp /tmp/hpf_src.XXXXXX); sed -n '1,157p' "$HERE/harness_champ.sh" > "$_HC"   # bash 3.2 (macOS) cannot source a process substitution
grep -q '^n_ok () ' "$_HC" && grep -q '^mk_sandbox () ' "$_HC" || { echo "harness_champ.sh layout changed: lines 1-157 no longer hold the helpers"; exit 2; }
source "$_HC"; rm -f "$_HC"

mk_pf () {
  mk_sandbox
  cp "$REPO/tools/filler_full.sh" "$REPO/tools/chain_paperfinal.sh" "$SB/repo/tools/"
  rm -rf /tmp/filler_pull
  local G="$SB/gcs/champ"
  mkdir -p "$G/filler" "$G/evals" "$SB/gcs/champ/live/runs"
  for a in C0 C1 C2 C3 C4 C5 C6; do echo ok > "$G/${a}_ARM_OK"; echo ok > "$G/${a}_PRETRAIN_OK"; done
  echo ok > "$G/SYNC-AB"; printf 'BATCHONLY' > "$G/RECIPE-DEC"; echo "the champion night's final" > "$G/champ_final.tgz"
  echo "2 2026-09-10T00:00:00Z" > "$G/filler/d64full_C4_CLAIM_w2"; echo ok > "$G/filler/k128_C4_OK"
  echo stale > "$SB/gcs/champ/live/runs/STALE_CHAMP_LIVE_MARKER"
  mkdir -p "$SB/fx/runs/sxeval_pchampC4/full_vsel_t16" "$SB/fx/runs/pretrainchamp_C4"
  "$REAL_PY" - "$SB/fx" <<'PYEOF'
import json, pickle, sys
from pathlib import Path
fx = Path(sys.argv[1])
(fx / "runs/sxeval_pchampC4/full_vsel_t16/summary_all.json").write_text(json.dumps({"n": 422786, "ckpt": "runs/pretrainchamp_C4/ckpt_020000.pkl"}))
pickle.dump({"state": {"model": {}}, "step": 20000, "config": {}}, open(fx / "runs/pretrainchamp_C4/ckpt_020000.pkl", "wb"))
PYEOF
  tar czf "$G/evals/full_C4_vsel_t16.tgz" -C "$SB/fx" runs/sxeval_pchampC4/full_vsel_t16
  tar czf "$G/C4_pretrain.tgz" -C "$SB/fx" runs/pretrainchamp_C4
  [ "${NO_PORT:-0}" = 1 ] || "$REAL_PY" -c "import pickle; pickle.dump({'state': {'model': {}}, 'config': {'cell_kind': 'trm'}}, open('$SB/gcs/frontier/ckpts/eqr.pkl', 'wb'))"
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
run_pf () {  # [VAR=val...]
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 \
     C1_STEPS_X=100 C1_STEPS_LONG=120 C1_EXT_STEPS=20 C1_EXT_WINDOW=10 LIVE_NO_GUARD=1 LIVE_EVERY=1 STALL_SEC=3 WATCH_POLL=1 \
     PASS_SLEEP=1 IDLE_POLL=1 PF_PASSES=2 "$@" bash tools/chain_paperfinal.sh > "$SB/pf.log" 2>&1; echo $? > "$SB/pf.rc")
}
line_of () { grep -n "$1" "$SB/pf.log" | head -1 | cut -d: -f1; }

echo "== P1 fresh -> COMPLETE =="
mk_pf; run_pf; G="$SB/gcs/champ"
[ "$(cat "$SB/pf.rc")" = 0 ] && grep -q "CHAIN-PAPERFINAL-COMPLETE" "$SB/pf.log" && [ -f "$G/paperfinal_final.tgz" ] && ok "P1 rc 0, sentinel, paperfinal_final.tgz" || { bad "P1 completion (rc $(cat "$SB/pf.rc"))"; tail -12 "$SB/pf.log"; }
[ "$(grep -c 'PREFLIGHT-OK' "$SB/pf.log")" = 2 ] && grep -q "PREFLIGHT-OK C7" "$SB/pf.log" && grep -q "PREFLIGHT-OK C8" "$SB/pf.log" && ok "P1 preflight of C7 and C8 only" || bad "P1 preflight: $(grep PREFLIGHT "$SB/pf.log" | tr '\n' ' ')"
[ "$(grep -c 'PRETRAIN-START' "$SB/pf.log")" = 2 ] && [ "$(line_of 'PRETRAIN-START C7')" -lt "$(line_of 'PRETRAIN-START C8')" ] && ok "P1 pretrain C7 then C8, nothing else" || bad "P1 pretrain order: $(grep PRETRAIN-START "$SB/pf.log" | tr '\n' ' ')"
allb=1; for a in C7 C8; do for r in full_${a}_vsel_t16 full_${a}_final_t16 full_${a}_vsel_t16_alt full_${a}_vsel_t64 d128_${a} d256_${a} scan_${a} census_${a}_vsel census_${a}_final calib_${a}_vsel screen_${a}_vb; do [ -f "$G/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done; [ -f "$G/${a}_ARM_OK" ] || allb=0; done
[ $allb = 1 ] && ok "P1 the champion battery on C7 and C8 (11 rows each) + ARM_OK" || bad "P1 battery"
allf=1; for m in d64full_C7 d64full_C8 d128sub_C7 d128sub_C8 d256sub_C7 d256sub_C8 k128port_eqr d64full_C4; do [ -f "$G/filler/${m}_OK" ] && [ -f "$G/filler/${m}.tgz" ] || { allf=0; echo "    missing filler $m"; }; done
[ $allf = 1 ] && ok "P1 the eight filler rows banked (tgz + OK)" || bad "P1 filler rows"
c7=$(pargv C7); c8=$(pargv C8)
echo "$c7" | grep -q -- "--dec-width 384 .*--seed 1 --dec-width 192" && echo "$c8" | grep -q -- "--seed 2 --dec-width 192" && ! echo "$c7" | grep -q -- "--dec-coupling attn\|--dec-commit\|--sudoku-orbit-online" && ok "P1 C7/C8 = C5's recipe at seeds 1/2 (w192; no other variable)" || bad "P1 registry: $c7"
kp=$(eargv "$SB/repo/runs/filler_sxscan128_pport_eqr/summary_s0.json")
echo "$kp" | grep -q -- "--ckpt runs/frontier_ckpts/eqr.pkl" && echo "$kp" | grep -q -- "--split test --subsample 5000 --t-total 64 --k-init 128 --ema" && ! echo "$kp" | grep -q -- "--batch" && echo "$kp" | grep -q "mon512" && ok "P1 EqR k128: the frontier scan's flags on the 5k of the 512-monitor file (the champion arms' identical set)" || bad "P1 k128port flags: $kp"
nk=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/filler_sxscan128_pport_eqr/summary_all.json'))['n'])"); n64=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/filler_sxeval_pchampC7_full_t64/summary_all.json'))['n'])"); n128=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/filler_sxeval_pchampC8_sub50000_t128/summary_all.json'))['n'])")
[ "$nk" = 5000 ] && [ "$n64" = 422786 ] && [ "$n128" = 50000 ] && ok "P1 n-gates 5000 / 422,786 / 50,000" || bad "P1 n ($nk $n64 $n128)"
grep -q "FILLER-JOB-START d64full_C4" "$SB/pf.log" && ! grep -q "FILLER-CLAIMED d64full_C4" "$SB/pf.log" && ok "P1 C4's full row ran despite the stale w2 claim (no marker deleted)" || bad "P1 stale claim"
[ "$(cat "$G/champ_final.tgz")" = "the champion night's final" ] && [ -f "$G/filler/d64full_C4_CLAIM_w2" ] && [ -f "$G/filler/k128_C4_OK" ] && [ -f "$SB/gcs/champ/live/runs/STALE_CHAMP_LIVE_MARKER" ] && ok "P1 nothing of the champion night overwritten or deleted (final tgz, claim, k128 marker, live prefix)" || bad "P1 champion state touched"
[ ! -e "$SB/repo/runs/STALE_CHAMP_LIVE_MARKER" ] && [ -d "$SB/gcs/champ_paper/live/runs" ] && grep -q "champ_paper/live" "$SB/pf.log" && ok "P1 the fresh live prefix (champ_paper/live) banked; the champion live prefix never restored" || bad "P1 live prefix"
ch=$(line_of "CHAIN-CHAMPW192-COMPLETE"); f1=$(line_of "PF-FILLER"); pf=$(line_of "CHAIN-PAPERFINAL-COMPLETE")
[ -n "$ch" ] && [ -n "$f1" ] && [ -n "$pf" ] && [ "$ch" -lt "$f1" ] && [ "$f1" -lt "$pf" ] && ok "P1 order: the inner chain's own sentinel, then the fillers, then the supervisor's sentinel" || bad "P1 sentinel order ($ch $f1 $pf)"
! grep -q "SELF-TEARDOWN" "$SB/pf.log" && ok "P1 no self-teardown (the supervisor owns it)" || bad "P1 self-teardown"
[ ! -f "$G/evals/full_C0_vsel_t16_OK" ] && [ ! -f "$G/C0_EXTENDED" ] && ok "P1 C0-C6 untouched (no new markers)" || bad "P1 C0-C6 touched"
tar tzf "$G/paperfinal_final.tgz" | grep -q "filler_sxscan128_pport_eqr/summary_all.json" && tar tzf "$G/paperfinal_final.tgz" | grep -q "pretrainchamp_C8/metrics.jsonl" && ok "P1 the manifest carries the summaries and the metrics" || bad "P1 manifest"
SBP1=$SB

echo "== P2 idempotent rerun =="
SB=$SBP1; run_pf
[ "$(cat "$SB/pf.rc")" = 0 ] && grep -q "(final object present)" "$SB/pf.log" && ! grep -q "PRETRAIN-START\|FILLER-JOB-START" "$SB/pf.log" && ok "P2 rerun: complete at once, nothing re-run" || { bad "P2 rerun"; tail -5 "$SB/pf.log"; }
rm -f "$SB/gcs/champ/paperfinal_final.tgz"; run_pf
[ "$(cat "$SB/pf.rc")" = 0 ] && ! grep -q "PRETRAIN-START\|FILLER-JOB-START" "$SB/pf.log" && grep -q "FINAL-BANKED" "$SB/pf.log" && [ -f "$SB/gcs/champ/paperfinal_final.tgz" ] && ok "P2b markers present, manifest absent -> only the manifest is rebuilt" || { bad "P2b"; tail -6 "$SB/pf.log"; }

echo "== P3 the ported checkpoint missing -> INCOMPLETE, no sentinel =="
NO_PORT=1 mk_pf; run_pf
[ "$(cat "$SB/pf.rc")" = 1 ] && grep -q "FILLER-NO-PORT eqr" "$SB/pf.log" && grep -q "PAPERFINAL-INCOMPLETE.*k128port_eqr_OK" "$SB/pf.log" && ! grep -q "CHAIN-PAPERFINAL-COMPLETE" "$SB/pf.log" && [ ! -f "$SB/gcs/champ/paperfinal_final.tgz" ] && [ -f "$SB/gcs/champ/filler/d64full_C4_OK" ] && ok "P3 NO-PORT -> INCOMPLETE rc 1, no sentinel, the other rows banked" || { bad "P3"; tail -6 "$SB/pf.log"; }

echo "== P4 a seed-arm preflight failure (C8) -> the pair stops, the riders bank, INCOMPLETE =="
mk_pf; run_pf STUB_PREFLIGHT_FAIL=C8
[ "$(cat "$SB/pf.rc")" = 1 ] && grep -q "CHAMPW192-PREFLIGHT-ABORT" "$SB/pf.log" && ! grep -q "PRETRAIN-START" "$SB/pf.log" && grep -q "PAPERFINAL-INCOMPLETE" "$SB/pf.log" && ! grep -q "CHAIN-PAPERFINAL-COMPLETE" "$SB/pf.log" && [ -f "$SB/gcs/champ/filler/k128port_eqr_OK" ] && [ -f "$SB/gcs/champ/filler/d64full_C4_OK" ] && [ ! -f "$SB/gcs/champ/C8_SKIPPED" ] && ok "P4 preflight abort: no pretrain, no SKIPPED label on a seed arm, riders banked, INCOMPLETE" || { bad "P4"; tail -8 "$SB/pf.log"; }

echo "== P5 the extension rule on C7 only =="
mk_pf; run_pf STUB_PEAK_LAST_ARM=C7
grep -q "PRETRAIN-EXTEND C7" "$SB/pf.log" && [ -f "$SB/gcs/champ/C7_EXTENDED" ] && [ ! -f "$SB/gcs/champ/C8_EXTENDED" ] && grep -q "CHAIN-PAPERFINAL-COMPLETE" "$SB/pf.log" && ok "P5 C7 extended once (registered rule), C8 not; completion" || { bad "P5"; grep -n "EXTEND" "$SB/pf.log" | head -4; }

echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
