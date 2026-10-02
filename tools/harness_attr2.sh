#!/bin/bash
# Ledger: SE-RRM ATTRIBUTION ROUND 2 offline stub harness (2026-10-02; Note_2026-10-02_Attribution_Round2_Registration.md). Reuses
# tools/harness_champ.sh's sandbox, stubs and helpers (its lines 1-157) and tools/chain_sablate.sh, as tools/harness_attr1.sh does. SA256U is SA256
# with SE-RRM's single-state recurrence (--dec-single-state) at SA256's fixed 30k budget; what matters is that EXACTLY that one flag reaches the
# trainer, once, that nothing else moves, and that the budget is fixed.
#   U1 SA256U on one pod -> COMPLETE: its FIXED 30,000 budget (never extended though the monitor rises to the end); the trainer argv token for token
#      = --out, SA256's flags from the chain's own registry, --dec-single-state, --steps 30000; every other key at SA256's value; the reduced battery
#      on the identical 5,000 and no full-battery row; the manifest.
#   U2 idempotent rerun.   U3 a preflight failure -> abort, INCOMPLETE.   U4 the reference and an unknown arm are refused.
#   U5 MUTANTS (the harness's own sensitivity; each must FAIL U1's checks at the named check): the arm entry without its flag; a wrong explicit
#      budget (36,000); the arm out of fixed_budget (the rising monitor then extends it).
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
expect_argv () {  # NPZ -> the argv SA256U must receive, built from the (sandbox) chain's own registry at the chain's defaults (MON 2000, CKPT 500, DEC_W 384)
  sed -n '/^loop_common () {/,/^arm_steps ()/p' "$SB/repo/tools/chain_champ.sh" | sed '$d' > "$SB/armfns.sh"    # a file: bash 3.2 cannot source a process substitution
  bash -c 'source "$1"; NPZ=$2; MON=2000; CKPT_EVERY=500; DEC_W=384
           echo "--out runs/pretrainchamp_SA256U $(arm_flags SA256) --dec-single-state --steps 30000"' _ "$SB/armfns.sh" "$1" | tr -s ' \n' '  ' | sed 's/ *$//'
}
check_u () {  # GCSDIR LABEL
  local G=$1 L=$2 x x0 exp good kv k v allb none r t nflag
  grep -q "PRETRAIN-START SA256U .*steps=30000" "$SB/ab.log" && grep -q "PRETRAIN-FIXED-BUDGET SA256U 30000" "$SB/ab.log" && ! grep -q "PRETRAIN-EXTEND SA256U\|EXTENDED from" "$SB/ab.log" && [ ! -f "$G/SA256U_EXTENDED" ] \
    && ok "$L SA256U its FIXED 30000 budget, never extended" || bad "$L SA256U budget: $(grep "PRETRAIN-.*SA256U" "$SB/ab.log" | tr '\n' ' ' | cut -c1-220)"
  x=$(pargv SA256U); nflag=$(echo "$x" | tr ' ' '\n' | grep -cx -- '--dec-single-state')
  [ "$nflag" = 1 ] && ok "$L SA256U --dec-single-state reaches the trainer exactly once" || bad "$L SA256U --dec-single-state count $nflag"
  x0=$(echo "$x" | sed 's|^tools/pretrain.py ||' | tr -s ' ' | sed 's/ *$//'); exp=$(expect_argv "$(last_of "$x" --sudoku-extreme)")
  [ "$x0" = "$exp" ] && ok "$L SA256U trainer argv token for token = --out, SA256's registry flags, --dec-single-state, --steps 30000" \
    || { bad "$L SA256U trainer argv != the registry's"; echo "    got: $x0" | cut -c1-260; echo "    exp: $exp" | cut -c1-260; }
  good=1
  echo "$x" | grep -q -- "--cell dec .*$SA256" || { good=0; echo "    SA256's flags not verbatim: $x"; }
  for kv in --batch=768 --lr=1e-4 --lr-end=1e-4 --trm-lambda=0.05 --trm-beta=0.01 --trm-ri-sigma=1.0 --fpa-k=1 --monitor-every=2000 --grid-every=2000 --steps=30000; do
    k=${kv%%=*}; v=${kv#*=}; [ "$(last_of "$x" "$k")" = "$v" ] || { good=0; echo "    $k: last value '$(last_of "$x" "$k")' != '$v'"; }; done
  [ $good = 1 ] && echo "$x" | grep -q -- "--sudoku-aug 1000 " && echo "$x" | grep -q -- "--wd 1.0 " && echo "$x" | grep -q -- "--ema 0.999" && echo "$x" | grep -q "mon512" \
    && ok "$L SA256U SA256's flags verbatim; every other key at SA256's value; --steps 30000" || bad "$L SA256U keys"
  allb=1; for r in l5k_SA256U_vsel_t16 l5k_SA256U_vsel_t64 scan_SA256U; do [ -f "$G/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done; [ -f "$G/SA256U_ARM_OK" ] || allb=0
  none=1; for r in full_SA256U_vsel_t16 full_SA256U_vsel_t64 d128_SA256U d256_SA256U census_SA256U_vsel calib_SA256U_vsel screen_SA256U_vb; do [ -f "$G/evals/${r}_OK" ] && { none=0; echo "    unexpected $r"; }; done
  [ $allb = 1 ] && [ $none = 1 ] && ok "$L SA256U the reduced battery (D16 + D64 on 5,000, the k32 scan) and no full-battery row" || bad "$L SA256U battery"
  grep -q -- "--record-by-step" <(cat "$SB/repo/runs/sxeval_pchampSA256U/sub5k_vsel_t64/"*.json 2>/dev/null) && ok "$L SA256U the D64 row records the per-iteration bits (--record-by-step)" || bad "$L SA256U D64 without --record-by-step"
  t=$(tar tzf "$G/sablate_final.tgz" 2>/dev/null); echo "$t" | grep -q "pretrainchamp_SA256U/metrics.jsonl" && echo "$t" | grep -q "pretrainchamp_SA256U/config.json" && echo "$t" | grep -q "sxeval_pchampSA256U/sub5k_vsel_t64/records_all.npz" && echo "$t" | grep -q "sxscan_pchampSA256U/summary_all.json" \
    && ok "$L SA256U the manifest carries its metrics, config, 5k rows and scan" || bad "$L SA256U manifest"
}

echo "== U1 SA256U: one pod, fresh prefix -> COMPLETE (its monitor rises to the end, so an extendable arm WOULD extend) =="
mk_ab; run_ab SA256U attr2_p0 STUB_PEAK_LAST_ARM=SA256U; G="$SB/gcs/attr2_p0"; SBU1=$SB
[ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && [ -f "$G/sablate_final.tgz" ] && ok "U1 rc 0, sentinel, final manifest" || { bad "U1 completion (rc $(cat "$SB/ab.rc"))"; tail -12 "$SB/ab.log"; }
check_u "$G" U1
[ "$(grep -c 'PREFLIGHT-OK' "$SB/ab.log")" = 1 ] && ok "U1 one preflight" || bad "U1 preflights $(grep -c 'PREFLIGHT-OK' "$SB/ab.log")"
[ "$(ls "$SB/gcs/wladder_p1" | tr '\n' ' ')" = "SA256_ARM_OK STALE_LADDER_MARKER " ] && ! grep -q "SELF-TEARDOWN" "$SB/ab.log" && ok "U1 the ladder's prefix untouched; no self-teardown" || bad "U1 prefix hygiene"

echo "== U2 idempotent rerun =="
SB=$SBU1; run_ab SA256U attr2_p0
[ "$(cat "$SB/ab.rc")" = 0 ] && grep -q "(final object present)" "$SB/ab.log" && ! grep -q "PRETRAIN-START" "$SB/ab.log" && ok "U2 rerun: complete at once, nothing re-run" || { bad "U2 rerun"; tail -5 "$SB/ab.log"; }

echo "== U3 a preflight failure -> abort, INCOMPLETE, no sentinel =="
mk_ab; run_ab SA256U attr2_pf STUB_PREFLIGHT_FAIL=SA256U
[ "$(cat "$SB/ab.rc")" = 1 ] && grep -q "CHAMPAB-PREFLIGHT-ABORT" "$SB/ab.log" && ! grep -q "PRETRAIN-START" "$SB/ab.log" && grep -q "SABLATE-INCOMPLETE" "$SB/ab.log" && ! grep -q "CHAIN-SABLATE-COMPLETE" "$SB/ab.log" && ok "U3 preflight failure: abort before any training, INCOMPLETE, no sentinel" || { bad "U3"; tail -8 "$SB/ab.log"; }

echo "== U4 the reference and an unknown arm are refused before any work =="
mk_ab; run_ab SA256 attr2_ref
[ "$(cat "$SB/ab.rc")" = 2 ] && grep -q "AB-BAD-ARM SA256 " "$SB/ab.log" && ! grep -q "PRETRAIN-START\|PREFLIGHT" "$SB/ab.log" && ok "U4 SA256 itself is refused" || { bad "U4 SA256"; tail -5 "$SB/ab.log"; }
mk_ab; run_ab "SA256U SA256Z" attr2_unk
[ "$(cat "$SB/ab.rc")" = 2 ] && grep -q "AB-BAD-ARM SA256Z " "$SB/ab.log" && ! grep -q "PRETRAIN-START\|PREFLIGHT" "$SB/ab.log" && ok "U4 an unknown arm in the list refuses the whole pod" || { bad "U4 unknown"; tail -5 "$SB/ab.log"; }

echo "== U5 mutants: each must FAIL U1's checks at the named check =="
mutant () {  # LABEL SED-EXPR EXPECTED-FAIL-FRAGMENT
  local lab=$1 expr=$2 frag=$3 out nf
  mk_ab; sed -i.bak "$expr" "$SB/repo/tools/chain_champ.sh"
  cmp -s "$SB/repo/tools/chain_champ.sh" "$SB/repo/tools/chain_champ.sh.bak" && { bad "U5 $lab: the mutation did not apply"; return; }
  run_ab SA256U "attr2_mut" STUB_PEAK_LAST_ARM=SA256U; out=$(check_u "$SB/gcs/attr2_mut" "MUT"); nf=$(echo "$out" | grep -c '^  FAIL')
  [ "$nf" -ge 1 ] && echo "$out" | grep -q "^  FAIL  MUT SA256U $frag" && ok "U5 $lab caught: $nf check(s) fail, incl. '$frag'" || { bad "U5 $lab NOT caught (failures $nf)"; echo "$out" | head -12; }
}
mutant "the arm without its flag" 's#SA256U)  echo "$(arm_flags SA256) --dec-single-state";;#SA256U)  echo "$(arm_flags SA256)";;#' "--dec-single-state count 0"
mutant "a wrong explicit budget (36,000)" 's#SA256U) echo 30000;;#SA256U) echo 36000;;#' "budget"
mutant "the arm out of fixed_budget" 's#SA256BR|SA256U) return 0;;#SA256BR) return 0;;#' "budget"

echo; echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
