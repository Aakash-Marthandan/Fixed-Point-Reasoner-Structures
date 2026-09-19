#!/bin/bash
# Ledger: THE RECIPE ABLATION offline stub harness (2026-09-19; the house law: no chain launches without an end-to-end offline pass).
# Reuses tools/harness_champ.sh's sandbox, stubs and helpers (its lines 1-157) and adds tools/chain_sablate.sh. Each arm = SA256 with ONE recipe
# item moved toward SE-RRM's published setup; what matters most here is that EXACTLY that one override reaches the trainer, last, and nothing else.
#   A1 one arm per pod, fresh prefix -> COMPLETE, for each of the three arms: the fixed 30k budget (never extended though the monitor rises to the
#      end), SA256's flags verbatim + the arm's override as the LAST word on its keys, the OTHER two arms' overrides absent, the reduced battery on
#      the identical 5,000 (and no full-battery row), the fresh prefix seeded, the read-only source untouched, the manifest, no self-teardown.
#   A2 idempotent rerun / the manifest alone rebuilt.   A3 a preflight failure -> abort, INCOMPLETE.   A4 an arm outside the ablation is refused.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
_HC=$(mktemp /tmp/hab_src.XXXXXX); sed -n '1,157p' "$HERE/harness_champ.sh" > "$_HC"
grep -q '^n_ok () ' "$_HC" && grep -q '^mk_sandbox () ' "$_HC" || { echo "harness_champ.sh layout changed: lines 1-157 no longer hold the helpers"; exit 2; }
# shellcheck disable=SC1090
source "$_HC"; rm -f "$_HC"

SA256="--seed 0 --dec-width 256 --dec-token-mixer attn --dec-tok-dk 32 --dec-coupling attn --dec-attn-heads 4 --dec-attn-dk 32"
mk_ab () { mk_sandbox; cp "$REPO/tools/chain_sablate.sh" "$SB/repo/tools/"; mkdir -p "$SB/gcs/champ/sets" "$SB/gcs/wladder_p1"; echo stale > "$SB/gcs/wladder_p1/STALE_LADDER_MARKER"; echo ok > "$SB/gcs/wladder_p1/SA256_ARM_OK"; }
run_ab () {  # ARM [VAR=val...]
  local arm=$1; shift
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 \
     C1_EXT_STEPS=20000 C1_EXT_WINDOW=4000 LIVE_NO_GUARD=1 LIVE_EVERY=1 STALL_SEC=3 WATCH_POLL=1 \
     GCS="gs://qhrrn2-rescue/ab_$arm" GCS_SETS=gs://qhrrn2-rescue/champ/sets AB_SRC=gs://qhrrn2-rescue/wladder_p1 LIVE_PREFIX="gs://qhrrn2-rescue/ab_$arm/live" AB_ARMS="$arm" "$@" \
     bash tools/chain_sablate.sh > "$SB/ab.log" 2>&1; echo $? > "$SB/ab.rc")
}
last_of () { echo "$1" | tr ' ' '\n' | awk -v k="$2" '$0==k{getline v; r=v} END{print r}'; }   # ARGV KEY -> the LAST value given for KEY (argparse keeps the last)

# ARM | the keys it overrides and the value that must be the LAST word | the keys it must leave at SA256's value
spec () { case $1 in
  SA256L) echo "--trm-lambda=0 --trm-beta=0|--trm-ri-sigma=1.0 --fpa-k=1 --batch=768 --lr=1e-4 --lr-end=1e-4";;
  SA256S) echo "--trm-ri-sigma=0 --fpa-k=0|--trm-lambda=0.05 --trm-beta=0.01 --batch=768 --lr=1e-4 --lr-end=1e-4";;
  SA256O) echo "--batch=272 --lr=5e-4 --lr-end=5e-4|--trm-lambda=0.05 --trm-beta=0.01 --trm-ri-sigma=1.0 --fpa-k=1";; esac; }

for arm in SA256L SA256S SA256O; do
  echo "== A1 $arm: one pod, fresh prefix -> COMPLETE (its monitor rises to the end, so a champion arm WOULD extend) =="
  mk_ab; run_ab "$arm" STUB_PEAK_LAST_ARM=$arm; G="$SB/gcs/ab_$arm"
  [ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && [ -f "$G/sablate_final.tgz" ] && ok "A1 $arm rc 0, sentinel, sablate_final.tgz" || { bad "A1 $arm completion (rc $(cat "$SB/ab.rc"))"; tail -12 "$SB/ab.log"; }
  grep -q "PRETRAIN-START $arm .*steps=30000" "$SB/ab.log" && grep -q "PRETRAIN-FIXED-BUDGET $arm 30000" "$SB/ab.log" && ! grep -q "PRETRAIN-EXTEND\|EXTENDED from" "$SB/ab.log" && [ ! -f "$G/${arm}_EXTENDED" ] && [ "$(grep -c 'PREFLIGHT-OK' "$SB/ab.log")" = 1 ] && ok "A1 $arm the FIXED 30k budget, never extended; preflight of this arm only" || bad "A1 $arm budget: $(grep 'PRETRAIN-\|PREFLIGHT' "$SB/ab.log" | tr '\n' ' ' | cut -c1-200)"
  x=$(pargv "$arm"); sp=$(spec "$arm"); good=1
  echo "$x" | grep -q -- "--cell dec .*$SA256" || { good=0; echo "    SA256's flags not verbatim: $x"; }
  for kv in ${sp%%|*} ${sp##*|}; do k=${kv%%=*}; v=${kv#*=}; [ "$(last_of "$x" "$k")" = "$v" ] || { good=0; echo "    $k: last value '$(last_of "$x" "$k")' != '$v'"; }; done
  [ $good = 1 ] && echo "$x" | grep -q -- "--sudoku-aug 1000 " && echo "$x" | grep -q -- "--wd 1.0 " && echo "$x" | grep -q -- "--ema 0.999" && echo "$x" | grep -q "mon512" && ok "A1 $arm SA256's flags verbatim; its override is the LAST word on its keys; the other two arms' keys stay at SA256's values" || bad "A1 $arm trainer argv"
  allb=1; for r in l5k_${arm}_vsel_t16 l5k_${arm}_vsel_t64 scan_${arm}; do [ -f "$G/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done; [ -f "$G/${arm}_ARM_OK" ] || allb=0
  none=1; for r in full_${arm}_vsel_t16 full_${arm}_vsel_t64 d128_${arm} d256_${arm} census_${arm}_vsel calib_${arm}_vsel screen_${arm}_vb; do [ -f "$G/evals/${r}_OK" ] && { none=0; echo "    unexpected $r"; }; done
  [ $allb = 1 ] && [ $none = 1 ] && ok "A1 $arm the reduced battery (D16 + D64 on 5,000, the k32 scan) and no full-battery row" || bad "A1 $arm battery"
  e64=$(eargv "$SB/repo/runs/sxeval_pchamp$arm/sub5k_vsel_t64/summary_s0.json"); n16=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/sxeval_pchamp$arm/sub5k_vsel_t16/summary_all.json'))['n'])")
  echo "$e64" | grep -q -- "--split test --subsample 5000 --t-total 64 --ema --record-by-step" && echo "$e64" | grep -q -- "--ckpt runs/pretrainchamp_$arm/ckpt_" && [ "$n16" = 5000 ] && ok "A1 $arm the rows' flags: the identical 5,000, EMA, the exact bit per step, on the arm's selected grid" || bad "A1 $arm row flags: $e64 (n $n16)"
  [ -f "$G/SYNC-AB" ] && [ "$(cat "$G/RECIPE-DEC")" = BATCHONLY ] && [ "$(ls "$SB/gcs/wladder_p1" | tr '\n' ' ')" = "SA256_ARM_OK STALE_LADDER_MARKER " ] && [ "$(cat "$SB/gcs/wladder_p1/STALE_LADDER_MARKER")" = stale ] && ! grep -q "SELF-TEARDOWN" "$SB/ab.log" && ok "A1 $arm the fresh prefix seeded; the ladder's prefix untouched (its SA256 marker not taken as ours); no self-teardown" || bad "A1 $arm prefix hygiene: $(ls "$SB/gcs/wladder_p1" | tr '\n' ' ')"
  t=$(tar tzf "$G/sablate_final.tgz"); echo "$t" | grep -q "pretrainchamp_$arm/metrics.jsonl" && echo "$t" | grep -q "pretrainchamp_$arm/config.json" && echo "$t" | grep -q "sxeval_pchamp$arm/sub5k_vsel_t64/records_all.npz" && echo "$t" | grep -q "sxscan_pchamp$arm/summary_all.json" && ok "A1 $arm the manifest carries the metrics, the config, the 5k rows' records and the scan" || bad "A1 $arm manifest: $(echo "$t" | tr '\n' ' ' | cut -c1-260)"
  [ "$arm" = SA256L ] && SBA1=$SB
done

echo "== A2 idempotent rerun =="
SB=$SBA1; run_ab SA256L
[ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "(final object present)" "$SB/ab.log" && ! grep -q "PRETRAIN-START" "$SB/ab.log" && ok "A2 rerun: complete at once, nothing re-run" || { bad "A2 rerun"; tail -5 "$SB/ab.log"; }
rm -f "$SB/gcs/ab_SA256L/sablate_final.tgz"; run_ab SA256L
[ "$(cat "$SB/ab.rc")" = 0 ] && ! grep -q "PRETRAIN-START" "$SB/ab.log" && grep -q "FINAL-BANKED" "$SB/ab.log" && ok "A2b markers present, manifest absent -> only the manifest is rebuilt" || { bad "A2b"; tail -5 "$SB/ab.log"; }

echo "== A3 a preflight failure -> abort, INCOMPLETE, no sentinel =="
mk_ab; run_ab SA256O STUB_PREFLIGHT_FAIL=SA256O
[ "$(cat "$SB/ab.rc")" = 1 ] && grep -q "CHAMPAB-PREFLIGHT-ABORT" "$SB/ab.log" && ! grep -q "PRETRAIN-START" "$SB/ab.log" && grep -q "SABLATE-INCOMPLETE" "$SB/ab.log" && ! grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && ok "A3 preflight failure: abort before any training, INCOMPLETE, no sentinel" || { bad "A3"; tail -8 "$SB/ab.log"; }

echo "== A4 an arm outside the ablation is refused (the reference itself is never re-run) =="
mk_ab; run_ab SA256
[ "$(cat "$SB/ab.rc")" = 2 ] && grep -q "AB-BAD-ARM SA256 " "$SB/ab.log" && ! grep -q "PRETRAIN-START\|PREFLIGHT" "$SB/ab.log" && ok "A4 SA256 itself is refused before any work" || { bad "A4"; tail -5 "$SB/ab.log"; }

echo; echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
