#!/bin/bash
# Ledger: SE-RRM ATTRIBUTION ROUND 3 offline stub harness (2026-10-03; Note_2026-10-03_Attribution_Round3_Registration.md). Reuses
# tools/harness_champ.sh's sandbox, stubs and helpers (its lines 1-157) and tools/chain_sablate.sh, as tools/harness_attr2.sh does. SA256UD = SA256U
# + SE-RRM's dropout 0.2; SA256D = SA256 + the same; both at SA256's fixed 30k. What matters: EXACTLY the registered flags reach the trainer, once
# each, nothing else moves, the budgets are fixed, and both arms run in sequence on ONE pod.
#   D1 each arm on its own pod -> COMPLETE: its FIXED 30,000 budget (never extended though the monitor rises to the end); the trainer argv token
#      for token = --out, SA256's registry flags, the arm's flags, --steps 30000; every other key at SA256's value; the reduced battery with the
#      per-iteration bits; the manifest.
#   D2 BOTH arms in sequence on ONE pod (the launch configuration) -> COMPLETE, both ARM_OK, one preflight per arm, one manifest.
#   D3 idempotent rerun.   D4 a preflight failure -> abort, INCOMPLETE.   D5 the reference and an unknown arm are refused.
#   D6 MUTANTS (the harness's own sensitivity; each must FAIL D1's checks at the named check): SA256UD without its dropout flag; SA256D at the
#      wrong rate (0.1); SA256UD out of fixed_budget (the rising monitor then extends it); SA256D's explicit budget wrong (36,000).
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
suffix_of () { case $1 in SA256UD) echo "--dec-single-state --dec-dropout 0.2";; SA256D) echo "--dec-dropout 0.2";; esac; }
nss_of () { case $1 in SA256UD) echo 1;; SA256D) echo 0;; esac; }   # how many --dec-single-state tokens the arm must carry
expect_argv () {  # ARM NPZ -> the argv the arm must receive, built from the (sandbox) chain's own registry at the chain's defaults (MON 2000, CKPT 500, DEC_W 384)
  sed -n '/^loop_common () {/,/^arm_steps ()/p' "$SB/repo/tools/chain_champ.sh" | sed '$d' > "$SB/armfns.sh"    # a file: bash 3.2 cannot source a process substitution
  bash -c 'source "$1"; NPZ=$2; MON=2000; CKPT_EVERY=500; DEC_W=384
           echo "--out runs/pretrainchamp_$3 $(arm_flags SA256) $4 --steps 30000"' _ "$SB/armfns.sh" "$2" "$1" "$(suffix_of "$1")" | tr -s ' \n' '  ' | sed 's/ *$//'
}
check_arm () {  # ARM GCSDIR LABEL
  local a=$1 G=$2 L=$3 x x0 exp good kv k v allb none r t nd ns
  grep -q "PRETRAIN-START $a .*steps=30000" "$SB/ab.log" && grep -q "PRETRAIN-FIXED-BUDGET $a 30000" "$SB/ab.log" && ! grep -q "PRETRAIN-EXTEND $a\|EXTENDED from" "$SB/ab.log" && [ ! -f "$G/${a}_EXTENDED" ] \
    && ok "$L $a its FIXED 30000 budget, never extended" || bad "$L $a budget: $(grep "PRETRAIN-.*$a" "$SB/ab.log" | tr '\n' ' ' | cut -c1-220)"
  x=$(pargv "$a"); nd=$(echo "$x" | tr ' ' '\n' | grep -cx -- '--dec-dropout'); ns=$(echo "$x" | tr ' ' '\n' | grep -cx -- '--dec-single-state')
  [ "$nd" = 1 ] && [ "$(last_of "$x" --dec-dropout)" = 0.2 ] && [ "$ns" = "$(nss_of "$a")" ] \
    && ok "$L $a --dec-dropout 0.2 exactly once; --dec-single-state x$(nss_of "$a")" || bad "$L $a --dec-dropout count $nd value '$(last_of "$x" --dec-dropout)'; --dec-single-state count $ns"
  x0=$(echo "$x" | sed 's|^tools/pretrain.py ||' | tr -s ' ' | sed 's/ *$//'); exp=$(expect_argv "$a" "$(last_of "$x" --sudoku-extreme)")
  [ "$x0" = "$exp" ] && ok "$L $a trainer argv token for token = --out, SA256's registry flags, $(suffix_of "$a"), --steps 30000" \
    || { bad "$L $a trainer argv != the registry's"; echo "    got: $x0" | cut -c1-260; echo "    exp: $exp" | cut -c1-260; }
  good=1
  echo "$x" | grep -q -- "--cell dec .*$SA256" || { good=0; echo "    SA256's flags not verbatim: $x"; }
  for kv in --batch=768 --lr=1e-4 --lr-end=1e-4 --trm-lambda=0.05 --trm-beta=0.01 --trm-ri-sigma=1.0 --fpa-k=1 --monitor-every=2000 --grid-every=2000 --steps=30000; do
    k=${kv%%=*}; v=${kv#*=}; [ "$(last_of "$x" "$k")" = "$v" ] || { good=0; echo "    $k: last value '$(last_of "$x" "$k")' != '$v'"; }; done
  [ $good = 1 ] && echo "$x" | grep -q -- "--sudoku-aug 1000 " && echo "$x" | grep -q -- "--wd 1.0 " && echo "$x" | grep -q -- "--ema 0.999" && echo "$x" | grep -q "mon512" \
    && ok "$L $a SA256's flags verbatim; every other key at SA256's value; --steps 30000" || bad "$L $a keys"
  allb=1; for r in l5k_${a}_vsel_t16 l5k_${a}_vsel_t64 scan_${a}; do [ -f "$G/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done; [ -f "$G/${a}_ARM_OK" ] || allb=0
  none=1; for r in full_${a}_vsel_t16 full_${a}_vsel_t64 d128_${a} d256_${a} census_${a}_vsel calib_${a}_vsel screen_${a}_vb; do [ -f "$G/evals/${r}_OK" ] && { none=0; echo "    unexpected $r"; }; done
  [ $allb = 1 ] && [ $none = 1 ] && ok "$L $a the reduced battery (D16 + D64 on 5,000, the k32 scan) and no full-battery row" || bad "$L $a battery"
  grep -q -- "--record-by-step" <(cat "$SB/repo/runs/sxeval_pchamp${a}/sub5k_vsel_t64/"*.json 2>/dev/null) && ok "$L $a the D64 row records the per-iteration bits (--record-by-step)" || bad "$L $a D64 without --record-by-step"
  t=$(tar tzf "$G/sablate_final.tgz" 2>/dev/null); echo "$t" | grep -q "pretrainchamp_$a/metrics.jsonl" && echo "$t" | grep -q "pretrainchamp_$a/config.json" && echo "$t" | grep -q "sxeval_pchamp$a/sub5k_vsel_t64/records_all.npz" && echo "$t" | grep -q "sxscan_pchamp$a/summary_all.json" \
    && ok "$L $a the manifest carries its metrics, config, 5k rows and scan" || bad "$L $a manifest"
}

for arm in SA256UD SA256D; do
  echo "== D1 $arm: one pod, fresh prefix -> COMPLETE (its monitor rises to the end, so an extendable arm WOULD extend) =="
  mk_ab; run_ab "$arm" "attr3_$arm" STUB_PEAK_LAST_ARM=$arm; G="$SB/gcs/attr3_$arm"
  [ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && [ -f "$G/sablate_final.tgz" ] && ok "D1 $arm rc 0, sentinel, final manifest" || { bad "D1 $arm completion (rc $(cat "$SB/ab.rc"))"; tail -12 "$SB/ab.log"; }
  check_arm "$arm" "$G" D1
  [ "$(grep -c 'PREFLIGHT-OK' "$SB/ab.log")" = 1 ] && ok "D1 $arm one preflight" || bad "D1 $arm preflights $(grep -c 'PREFLIGHT-OK' "$SB/ab.log")"
  [ "$(ls "$SB/gcs/wladder_p1" | tr '\n' ' ')" = "SA256_ARM_OK STALE_LADDER_MARKER " ] && ! grep -q "SELF-TEARDOWN" "$SB/ab.log" && ok "D1 $arm the ladder's prefix untouched; no self-teardown" || bad "D1 $arm prefix hygiene"
done

echo "== D2 both arms in sequence on ONE pod (the launch configuration) =="
mk_ab; run_ab "SA256UD SA256D" attr3_p0; G="$SB/gcs/attr3_p0"; SBD2=$SB
[ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && [ -f "$G/SA256UD_ARM_OK" ] && [ -f "$G/SA256D_ARM_OK" ] && ok "D2 rc 0, sentinel, both ARM_OK" || { bad "D2 completion (rc $(cat "$SB/ab.rc"))"; tail -12 "$SB/ab.log"; }
check_arm SA256UD "$G" D2; check_arm SA256D "$G" D2
[ "$(grep -c 'PREFLIGHT-OK' "$SB/ab.log")" = 2 ] && ok "D2 one preflight per arm" || bad "D2 preflights $(grep -c 'PREFLIGHT-OK' "$SB/ab.log")"

echo "== D3 idempotent rerun =="
SB=$SBD2; run_ab "SA256UD SA256D" attr3_p0
[ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "(final object present)" "$SB/ab.log" && ! grep -q "PRETRAIN-START" "$SB/ab.log" && ok "D3 rerun: complete at once, nothing re-run" || { bad "D3 rerun"; tail -5 "$SB/ab.log"; }

echo "== D4 a preflight failure -> abort, INCOMPLETE, no sentinel =="
mk_ab; run_ab SA256UD attr3_pf STUB_PREFLIGHT_FAIL=SA256UD
[ "$(cat "$SB/ab.rc")" = 1 ] && grep -q "CHAMPAB-PREFLIGHT-ABORT" "$SB/ab.log" && ! grep -q "PRETRAIN-START" "$SB/ab.log" && grep -q "SABLATE-INCOMPLETE" "$SB/ab.log" && ! grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && ok "D4 preflight failure: abort before any training, INCOMPLETE, no sentinel" || { bad "D4"; tail -8 "$SB/ab.log"; }

echo "== D5 the reference and an unknown arm are refused before any work =="
mk_ab; run_ab SA256 attr3_ref
[ "$(cat "$SB/ab.rc")" = 2 ] && grep -q "AB-BAD-ARM SA256 " "$SB/ab.log" && ! grep -q "PRETRAIN-START\|PREFLIGHT" "$SB/ab.log" && ok "D5 SA256 itself is refused" || { bad "D5 SA256"; tail -5 "$SB/ab.log"; }
mk_ab; run_ab "SA256UD SA256Z" attr3_unk
[ "$(cat "$SB/ab.rc")" = 2 ] && grep -q "AB-BAD-ARM SA256Z " "$SB/ab.log" && ! grep -q "PRETRAIN-START\|PREFLIGHT" "$SB/ab.log" && ok "D5 an unknown arm in the list refuses the whole pod" || { bad "D5 unknown"; tail -5 "$SB/ab.log"; }

echo "== D6 mutants: each must FAIL D1's checks at the named check =="
mutant () {  # LABEL ARM SED-EXPR EXPECTED-FAIL-FRAGMENT
  local lab=$1 arm=$2 expr=$3 frag=$4 out nf
  mk_ab; sed -i.bak "$expr" "$SB/repo/tools/chain_champ.sh"
  cmp -s "$SB/repo/tools/chain_champ.sh" "$SB/repo/tools/chain_champ.sh.bak" && { bad "D6 $lab: the mutation did not apply"; return; }
  run_ab "$arm" "attr3_mut" STUB_PEAK_LAST_ARM=$arm; out=$(check_arm "$arm" "$SB/gcs/attr3_mut" "MUT"); nf=$(echo "$out" | grep -c '^  FAIL')
  [ "$nf" -ge 1 ] && echo "$out" | grep -q "^  FAIL  MUT $arm $frag" && ok "D6 $lab caught: $nf check(s) fail, incl. '$frag'" || { bad "D6 $lab NOT caught (failures $nf)"; echo "$out" | head -12; }
}
mutant "SA256UD without its dropout flag" SA256UD 's#SA256UD) echo "$(arm_flags SA256U) --dec-dropout 0.2";;#SA256UD) echo "$(arm_flags SA256U)";;#' "--dec-dropout count 0"
mutant "SA256D at the wrong rate (0.1)" SA256D 's#SA256D)  echo "$(arm_flags SA256) --dec-dropout 0.2";;#SA256D)  echo "$(arm_flags SA256) --dec-dropout 0.1";;#' "--dec-dropout count 1 value '0.1'"
mutant "SA256UD out of fixed_budget" SA256UD 's#SA256UD|SA256D) return 0;;#SA256D) return 0;;#' "budget"
mutant "SA256D's explicit budget wrong (36,000)" SA256D 's#SA256UD|SA256D) echo 30000;;#SA256UD) echo 30000;; SA256D) echo 36000;;#' "budget"

echo; echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
