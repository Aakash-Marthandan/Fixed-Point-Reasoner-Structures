#!/bin/bash
# Ledger: THE WIDTH LADDER offline stub harness (2026-09-18; the house law: no chain launches without an end-to-end offline pass).
# Reuses tools/harness_champ.sh's sandbox, stubs and helpers (its lines 1-157) and adds tools/chain_wladder.sh. Scenarios:
#   L1 fresh prefix -> COMPLETE: every arm on a FIXED 30k budget (PRETRAIN-FIXED-BUDGET, never PRETRAIN-EXTEND even with a monitor rising to the
#      end), the trainer's argv carries the arm's ONE variable (the width; the attention mixers), the REDUCED battery only (D16 + D64 on 5,000 +
#      the k32 scan; no full-set row, no D128 / D256 / census / calibration / screens), the n-gates, the manifest, the seeded markers.
#   L2 idempotent rerun; L3 a preflight failure -> abort, INCOMPLETE, no sentinel; L4 an arm outside the ladder is refused before any spend.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
_HC=$(mktemp /tmp/hwl_src.XXXXXX); sed -n '1,157p' "$HERE/harness_champ.sh" > "$_HC"
grep -q '^n_ok () ' "$_HC" && grep -q '^mk_sandbox () ' "$_HC" || { echo "harness_champ.sh layout changed: lines 1-157 no longer hold the helpers"; exit 2; }
# shellcheck disable=SC1090
source "$_HC"; rm -f "$_HC"

mk_wl () { mk_sandbox; cp "$REPO/tools/chain_wladder.sh" "$SB/repo/tools/"; mkdir -p "$SB/gcs/champ/sets" "$SB/gcs/wl"; echo stale > "$SB/gcs/champ/STALE_CHAMP_MARKER"; }
run_wl () {  # ARMS [VAR=val...]
  local arms=$1; shift
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 \
     C1_EXT_STEPS=20000 C1_EXT_WINDOW=4000 LIVE_NO_GUARD=1 LIVE_EVERY=1 STALL_SEC=3 WATCH_POLL=1 \
     GCS=gs://qhrrn2-rescue/wl GCS_SETS=gs://qhrrn2-rescue/champ/sets WL_SRC=gs://qhrrn2-rescue/champ LIVE_PREFIX=gs://qhrrn2-rescue/wl/live WL_ARMS="$arms" "$@" \
     bash tools/chain_wladder.sh > "$SB/wl.log" 2>&1; echo $? > "$SB/wl.rc")
}

echo "== L1 fresh prefix -> COMPLETE (the monitor of the LAST arm rises to the end, so a champion arm WOULD extend) =="
mk_wl; run_wl "W256 W128 SA128" STUB_PEAK_LAST_ARM=SA128; G="$SB/gcs/wl"
[ "$(cat "$SB/wl.rc")" = 0 ] && grep -q "CHAIN-WLADDER-COMPLETE" "$SB/wl.log" && [ -f "$G/wladder_final.tgz" ] && ok "L1 rc 0, sentinel, wladder_final.tgz" || { bad "L1 completion (rc $(cat "$SB/wl.rc"))"; tail -14 "$SB/wl.log"; }
[ "$(grep -c 'PREFLIGHT-OK' "$SB/wl.log")" = 3 ] && ok "L1 preflight of the three arms only" || bad "L1 preflight: $(grep PREFLIGHT "$SB/wl.log" | tr '\n' ' ')"
f=1; for a in W256 W128 SA128; do grep -q "PRETRAIN-START $a .*steps=30000" "$SB/wl.log" && grep -q "PRETRAIN-FIXED-BUDGET $a 30000" "$SB/wl.log" && [ ! -f "$G/${a}_EXTENDED" ] || { f=0; echo "    budget $a"; }; done
[ $f = 1 ] && ! grep -q "PRETRAIN-EXTEND\|EXTENDED from" "$SB/wl.log" && ok "L1 every arm on the FIXED 30k budget, never extended" || bad "L1 budget"
w=$(pargv W128); s=$(pargv SA128); w2=$(pargv W256)
echo "$w" | grep -q -- "--cell dec .*--seed 0 --dec-width 128" && ! echo "$w" | grep -q -- "--dec-token-mixer" && echo "$w2" | grep -q -- "--seed 0 --dec-width 256" && ok "L1 the width arms: the champion recipe, seed 0, ONE variable (the width)" || bad "L1 width argv: $w"
echo "$s" | grep -q -- "--seed 0 --dec-width 128 --dec-token-mixer attn --dec-tok-dk 32 --dec-coupling attn --dec-attn-heads 4 --dec-attn-dk 32" && ok "L1 the mixer arm: attention over the cells (2D rotary) + attention across the fields" || bad "L1 mixer argv: $s"
for a in "$w" "$s"; do echo "$a" | grep -q -- "--fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --trm-ri-sigma 1.0" && echo "$a" | grep -q -- "--sudoku-aug 1000 " && echo "$a" | grep -q -- "--batch 768 --wd 1.0 " && echo "$a" | grep -q "mon512" || { bad "L1 recipe flags: $a"; break; }; done; ok "L1 the registered recipe on every arm (anchor rows, randomized init, aug 1000, batch 768, wd 1, the 512-puzzle file)"
allb=1; for a in W256 W128 SA128; do for r in l5k_${a}_vsel_t16 l5k_${a}_vsel_t64 scan_${a}; do [ -f "$G/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done; [ -f "$G/${a}_ARM_OK" ] || allb=0; done
[ $allb = 1 ] && ok "L1 the reduced battery on every arm (D16 + D64 on 5,000, the k32 scan) + ARM_OK" || { bad "L1 battery"; grep -n "SCAN\|scan_\|RECIPE" "$SB/wl.log" | head -8 | cut -c1-200; }
none=1; for a in W256 W128 SA128; do for r in full_${a}_vsel_t16 full_${a}_vsel_t64 d128_${a} d256_${a} census_${a}_vsel calib_${a}_vsel screen_${a}_vb; do [ -f "$G/evals/${r}_OK" ] && { none=0; echo "    unexpected $r"; }; done; done
[ $none = 1 ] && ok "L1 no full-set, D128 / D256, census, calibration or screen row" || bad "L1 extra rows ran"
e16=$(eargv "$SB/repo/runs/sxeval_pchampW128/sub5k_vsel_t16/summary_s0.json"); e64=$(eargv "$SB/repo/runs/sxeval_pchampSA128/sub5k_vsel_t64/summary_s0.json")
echo "$e16" | grep -q -- "--split test --subsample 5000 --t-total 16 --ema --record-by-step" && echo "$e64" | grep -q -- "--split test --subsample 5000 --t-total 64 --ema --record-by-step" && echo "$e64" | grep -q -- "--ckpt runs/pretrainchamp_SA128/ckpt_" && ok "L1 the rows' flags: the identical 5,000, EMA, the exact bit per step, the arm's own selected grid" || bad "L1 eval flags: $e16 | $e64"
n16=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/sxeval_pchampW256/sub5k_vsel_t16/summary_all.json'))['n'])"); [ "$n16" = 5000 ] && ok "L1 n-gate 5,000" || bad "L1 n ($n16)"
[ -f "$G/SYNC-AB" ] && [ "$(cat "$G/RECIPE-DEC")" = BATCHONLY ] && ! grep -q "SYNC-AB-OK\|sync_" "$SB/wl.log" && ok "L1 the fresh prefix seeded (no sync rider, the scan recipe)" || bad "L1 seeded markers"
[ "$(cat "$SB/gcs/champ/STALE_CHAMP_MARKER")" = stale ] && [ "$(ls "$SB/gcs/champ" | wc -l | tr -d ' ')" = 2 ] && ok "L1 the champion prefix untouched (read-only)" || bad "L1 the champion prefix was written: $(ls "$SB/gcs/champ" | tr '\n' ' ')"
! grep -q "SELF-TEARDOWN" "$SB/wl.log" && ok "L1 no self-teardown (the supervisor owns it)" || bad "L1 self-teardown"
t=$(tar tzf "$G/wladder_final.tgz"); echo "$t" | grep -q "pretrainchamp_W128/metrics.jsonl" && echo "$t" | grep -q "sxeval_pchampSA128/sub5k_vsel_t64/summary_all.json" && echo "$t" | grep -q "sxscan_pchampW256/summary_all.json" && ok "L1 the manifest carries the metrics, the 5k rows and the scans" || bad "L1 manifest: $(echo "$t" | head -5 | tr '\n' ' ')"
SBL1=$SB

echo "== L2 idempotent rerun =="
SB=$SBL1; run_wl "W256 W128 SA128"
[ "$(cat "$SB/wl.rc")" = 0 ] && grep -q "(final object present)" "$SB/wl.log" && ! grep -q "PRETRAIN-START" "$SB/wl.log" && ok "L2 rerun: complete at once, nothing re-run" || { bad "L2 rerun"; tail -5 "$SB/wl.log"; }
rm -f "$SB/gcs/wl/wladder_final.tgz"; run_wl "W256 W128 SA128"
[ "$(cat "$SB/wl.rc")" = 0 ] && ! grep -q "PRETRAIN-START" "$SB/wl.log" && grep -q "FINAL-BANKED" "$SB/wl.log" && ok "L2b markers present, manifest absent -> only the manifest is rebuilt" || { bad "L2b"; tail -5 "$SB/wl.log"; }

echo "== L3 a preflight failure -> abort, INCOMPLETE, no sentinel =="
mk_wl; run_wl "SA256 SA192" STUB_PREFLIGHT_FAIL=SA192
[ "$(cat "$SB/wl.rc")" = 1 ] && grep -q "CHAMPWL-PREFLIGHT-ABORT" "$SB/wl.log" && ! grep -q "PRETRAIN-START" "$SB/wl.log" && grep -q "WLADDER-INCOMPLETE" "$SB/wl.log" && ! grep -q "CHAIN-WLADDER-COMPLETE" "$SB/wl.log" && ok "L3 preflight failure: abort before any training, INCOMPLETE, no sentinel" || { bad "L3"; tail -8 "$SB/wl.log"; }

echo "== L4 an arm outside the ladder is refused =="
mk_wl; run_wl "W256 C5"
[ "$(cat "$SB/wl.rc")" = 2 ] && grep -q "WL-BAD-ARM C5" "$SB/wl.log" && ! grep -q "PRETRAIN-START\|PREFLIGHT" "$SB/wl.log" && ok "L4 a non-ladder arm is refused before any work" || { bad "L4"; tail -5 "$SB/wl.log"; }

echo; echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
