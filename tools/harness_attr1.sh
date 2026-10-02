#!/bin/bash
# Ledger: SE-RRM ATTRIBUTION ROUND 1 offline stub harness (2026-10-02; Note_2026-10-02_Attribution_Round1_Registration.md). Reuses
# tools/harness_champ.sh's sandbox, stubs and helpers (its lines 1-157) and tools/chain_sablate.sh, as tools/harness_sablate.sh does. Each arm is
# SA256 with SE-RRM's released optimizer (batch 272; lr 1e-4 = ours) at its own fixed budget; what matters is that EXACTLY those keys reach the
# trainer, last, and nothing else, and that both arms run in sequence on ONE pod.
#   B1 one arm per pod -> COMPLETE, for each arm: its fixed budget (36,000 / 84,000; never extended though the monitor rises to the end), SA256's
#      flags verbatim + its override as the LAST word on its keys (SA256BR also on the monitor/grid cadence 5,600), every other key at SA256's
#      value, the reduced battery on the identical 5,000 and no full-battery row, the manifest.
#   B2 BOTH arms on ONE pod (the launch configuration) -> COMPLETE, both ARM_OK, each at its own budget, one preflight per arm, one manifest.
#   B3 idempotent rerun.   B4 a preflight failure -> abort, INCOMPLETE.   B5 the reference and an unknown arm are refused.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
_HC=$(mktemp /tmp/hat_src.XXXXXX); sed -n '1,157p' "$HERE/harness_champ.sh" > "$_HC"
grep -q '^n_ok () ' "$_HC" && grep -q '^mk_sandbox () ' "$_HC" || { echo "harness_champ.sh layout changed: lines 1-157 no longer hold the helpers"; exit 2; }
# shellcheck disable=SC1090
source "$_HC"; rm -f "$_HC"

SA256="--seed 0 --dec-width 256 --dec-token-mixer attn --dec-tok-dk 32 --dec-coupling attn --dec-attn-heads 4 --dec-attn-dk 32"
mk_ab () { mk_sandbox; cp "$REPO/tools/chain_sablate.sh" "$SB/repo/tools/"; mkdir -p "$SB/gcs/champ/sets" "$SB/gcs/wladder_p1"; echo stale > "$SB/gcs/wladder_p1/STALE_LADDER_MARKER"; echo ok > "$SB/gcs/wladder_p1/SA256_ARM_OK"; }
run_ab () {  # "ARMS" PREFIX [VAR=val...]
  local arms=$1 pre=$2; shift 2
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 \
     C1_EXT_STEPS=20000 C1_EXT_WINDOW=4000 LIVE_NO_GUARD=1 LIVE_EVERY=1 STALL_SEC=3 WATCH_POLL=1 \
     GCS="gs://qhrrn2-rescue/$pre" GCS_SETS=gs://qhrrn2-rescue/champ/sets AB_SRC=gs://qhrrn2-rescue/wladder_p1 LIVE_PREFIX="gs://qhrrn2-rescue/$pre/live" AB_ARMS="$arms" "$@" \
     bash tools/chain_sablate.sh > "$SB/ab.log" 2>&1; echo $? > "$SB/ab.rc")
}
last_of () { echo "$1" | tr ' ' '\n' | awk -v k="$2" '$0==k{getline v; r=v} END{print r}'; }   # ARGV KEY -> the LAST value given for KEY
budget_of () { case $1 in SA256B) echo 36000;; SA256BR) echo 84000;; esac; }
# ARM | the keys it overrides with the value that must be the LAST word | the keys it must leave at SA256's value
spec () { case $1 in
  SA256B)  echo "--batch=272|--lr=1e-4 --lr-end=1e-4 --trm-lambda=0.05 --trm-beta=0.01 --trm-ri-sigma=1.0 --fpa-k=1 --monitor-every=2000 --grid-every=2000";;
  SA256BR) echo "--batch=272 --monitor-every=5600 --grid-every=5600|--lr=1e-4 --lr-end=1e-4 --trm-lambda=0.05 --trm-beta=0.01 --trm-ri-sigma=1.0 --fpa-k=1";; esac; }
check_arm () {  # ARM GCSDIR LABEL
  local arm=$1 G=$2 L=$3 b x sp good kv k v allb none r t
  b=$(budget_of "$arm")
  grep -q "PRETRAIN-START $arm .*steps=$b" "$SB/ab.log" && grep -q "PRETRAIN-FIXED-BUDGET $arm $b" "$SB/ab.log" && ! grep -q "PRETRAIN-EXTEND $arm\|EXTENDED from" "$SB/ab.log" && [ ! -f "$G/${arm}_EXTENDED" ] \
    && ok "$L $arm its FIXED $b budget, never extended" || bad "$L $arm budget: $(grep "PRETRAIN-.*$arm" "$SB/ab.log" | tr '\n' ' ' | cut -c1-220)"
  x=$(pargv "$arm"); sp=$(spec "$arm"); good=1
  echo "$x" | grep -q -- "--cell dec .*$SA256" || { good=0; echo "    SA256's flags not verbatim: $x"; }
  for kv in ${sp%%|*} ${sp##*|}; do k=${kv%%=*}; v=${kv#*=}; [ "$(last_of "$x" "$k")" = "$v" ] || { good=0; echo "    $k: last value '$(last_of "$x" "$k")' != '$v'"; }; done
  [ "$(last_of "$x" --steps)" = "$b" ] || { good=0; echo "    --steps last value '$(last_of "$x" --steps)' != $b"; }
  [ $good = 1 ] && echo "$x" | grep -q -- "--sudoku-aug 1000 " && echo "$x" | grep -q -- "--wd 1.0 " && echo "$x" | grep -q -- "--ema 0.999" && echo "$x" | grep -q "mon512" \
    && ok "$L $arm SA256's flags verbatim; its override is the LAST word on its keys; every other key at SA256's value; --steps $b" || bad "$L $arm trainer argv"
  allb=1; for r in l5k_${arm}_vsel_t16 l5k_${arm}_vsel_t64 scan_${arm}; do [ -f "$G/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done; [ -f "$G/${arm}_ARM_OK" ] || allb=0
  none=1; for r in full_${arm}_vsel_t16 full_${arm}_vsel_t64 d128_${arm} d256_${arm} census_${arm}_vsel calib_${arm}_vsel screen_${arm}_vb; do [ -f "$G/evals/${r}_OK" ] && { none=0; echo "    unexpected $r"; }; done
  [ $allb = 1 ] && [ $none = 1 ] && ok "$L $arm the reduced battery (D16 + D64 on 5,000, the k32 scan) and no full-battery row" || bad "$L $arm battery"
  t=$(tar tzf "$G/sablate_final.tgz"); echo "$t" | grep -q "pretrainchamp_$arm/metrics.jsonl" && echo "$t" | grep -q "pretrainchamp_$arm/config.json" && echo "$t" | grep -q "sxeval_pchamp$arm/sub5k_vsel_t64/records_all.npz" && echo "$t" | grep -q "sxscan_pchamp$arm/summary_all.json" \
    && ok "$L $arm the manifest carries its metrics, config, 5k rows and scan" || bad "$L $arm manifest"
}

for arm in SA256B SA256BR; do
  echo "== B1 $arm: one pod, fresh prefix -> COMPLETE (its monitor rises to the end, so an extendable arm WOULD extend) =="
  mk_ab; run_ab "$arm" "attr_$arm" STUB_PEAK_LAST_ARM=$arm; G="$SB/gcs/attr_$arm"
  [ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && [ -f "$G/sablate_final.tgz" ] && ok "B1 $arm rc 0, sentinel, final manifest" || { bad "B1 $arm completion (rc $(cat "$SB/ab.rc"))"; tail -12 "$SB/ab.log"; }
  check_arm "$arm" "$G" B1
  [ "$(grep -c 'PREFLIGHT-OK' "$SB/ab.log")" = 1 ] && ok "B1 $arm one preflight" || bad "B1 $arm preflights $(grep -c 'PREFLIGHT-OK' "$SB/ab.log")"
  [ "$(ls "$SB/gcs/wladder_p1" | tr '\n' ' ')" = "SA256_ARM_OK STALE_LADDER_MARKER " ] && ! grep -q "SELF-TEARDOWN" "$SB/ab.log" && ok "B1 $arm the ladder's prefix untouched; no self-teardown" || bad "B1 $arm prefix hygiene"
done

echo "== B2 both arms in sequence on ONE pod (the launch configuration) =="
mk_ab; run_ab "SA256B SA256BR" attr_p0; G="$SB/gcs/attr_p0"; SBB2=$SB
[ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && [ -f "$G/SA256B_ARM_OK" ] && [ -f "$G/SA256BR_ARM_OK" ] && ok "B2 rc 0, sentinel, both ARM_OK" || { bad "B2 completion (rc $(cat "$SB/ab.rc"))"; tail -12 "$SB/ab.log"; }
check_arm SA256B "$G" B2; check_arm SA256BR "$G" B2
[ "$(grep -c 'PREFLIGHT-OK' "$SB/ab.log")" = 2 ] && ok "B2 one preflight per arm" || bad "B2 preflights $(grep -c 'PREFLIGHT-OK' "$SB/ab.log")"

echo "== B3 idempotent rerun =="
SB=$SBB2; run_ab "SA256B SA256BR" attr_p0
[ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "(final object present)" "$SB/ab.log" && ! grep -q "PRETRAIN-START" "$SB/ab.log" && ok "B3 rerun: complete at once, nothing re-run" || { bad "B3 rerun"; tail -5 "$SB/ab.log"; }

echo "== B4 a preflight failure -> abort, INCOMPLETE, no sentinel =="
mk_ab; run_ab SA256BR attr_pf STUB_PREFLIGHT_FAIL=SA256BR
[ "$(cat "$SB/ab.rc")" = 1 ] && grep -q "CHAMPAB-PREFLIGHT-ABORT" "$SB/ab.log" && ! grep -q "PRETRAIN-START" "$SB/ab.log" && grep -q "SABLATE-INCOMPLETE" "$SB/ab.log" && ! grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && ok "B4 preflight failure: abort before any training, INCOMPLETE, no sentinel" || { bad "B4"; tail -8 "$SB/ab.log"; }

echo "== B5 the reference and an unknown arm are refused before any work =="
mk_ab; run_ab SA256 attr_ref
[ "$(cat "$SB/ab.rc")" = 2 ] && grep -q "AB-BAD-ARM SA256 " "$SB/ab.log" && ! grep -q "PRETRAIN-START\|PREFLIGHT" "$SB/ab.log" && ok "B5 SA256 itself is refused" || { bad "B5 SA256"; tail -5 "$SB/ab.log"; }
mk_ab; run_ab "SA256B SA256X" attr_unk
[ "$(cat "$SB/ab.rc")" = 2 ] && grep -q "AB-BAD-ARM SA256X " "$SB/ab.log" && ! grep -q "PRETRAIN-START\|PREFLIGHT" "$SB/ab.log" && ok "B5 an unknown arm in the list refuses the whole pod" || { bad "B5 unknown"; tail -5 "$SB/ab.log"; }

echo; echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
