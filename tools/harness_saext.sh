#!/bin/bash
# Ledger: THE ATTENTION-ARM EXTENSION offline stub harness (2026-09-19; the house law: no chain launches without an end-to-end offline pass).
# Reuses tools/harness_champ.sh's sandbox, stubs and helpers (its lines 1-157) and adds tools/filler_full.sh + tools/chain_saext.sh +
# tools/resume_flags_guard.py (which imports the REAL tools/pretrain.py to parse argv). The fixture is the width ladder's end state: the ladder
# prefix ($SRC) holds the arm's banked 30k state (<ARM>_pretrain.tgz: ckpt_latest at 30,000, its 15 grids and monitor rows, and a config.json in
# the REAL trainer's format — argv = vars(parse_args(the arm's flags)) — as the ladder banked it) plus the ladder's own markers, which the
# extension must neither read as its own nor write. The arm flags come from tools/chain_champ.sh itself. Scenarios:
#   E1 each arm fresh -> COMPLETE: the 30k state restored from $SRC; the banked config staged; the RESUME GUARD passes on every key but the
#      budget; the resume 30,000 -> 50,000 through the extension path (RESUMED, never from scratch; no second extension); one continuous
#      monitor curve (25 grids); the selection over ALL 25 grids (a peak inside the extension, a peak inside the ladder's 30k, a peak at the
#      last grid); the full champion battery + the four filler rows on the selected grid with their registered flags and sizes; $SRC
#      byte-identical; the manifest; the sentinel order.
#   E2 idempotent rerun / the manifest alone rebuilt.   E3 the banked config differs from the chain's argv (an integer, a float, a boolean)
#      -> the guard refuses the arm: no training, INCOMPLETE.   E4 no resume state / a state short of 30,000 / a banked config that is not
#      the 30k run's -> rc 2, nothing trained.   E5 a preemption: the live prefix holds a later state -> the live resume point.
#   E6 an arm or a job outside the registration -> rc 2.   E7 SE_JOBS=none -> the champion battery alone (option A).   E8 a relaunch on a fresh node after
#      the pretrain is banked -> PRETRAIN-SKIP, the battery and the fillers complete on the banked grids.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
_HC=$(mktemp /tmp/hse_src.XXXXXX); sed -n '1,157p' "$HERE/harness_champ.sh" > "$_HC"
grep -q '^n_ok () ' "$_HC" && grep -q '^mk_sandbox () ' "$_HC" || { echo "harness_champ.sh layout changed: lines 1-157 no longer hold the helpers"; exit 2; }
# shellcheck disable=SC1090
source "$_HC"; rm -f "$_HC"

# the arm flags, from the chain itself (its own defaults; bash 3.2 cannot source a process substitution)
_FF=$(mktemp /tmp/hse_ff.XXXXXX)
{ grep -E '^(NPZ|MON|CKPT_EVERY|DEC_W)=' "$REPO/tools/chain_champ.sh"; awk '/^loop_common \(\)/{p=1} p{print} /^arm_flags \(\)/{f=1} f&&/^}/{exit}' "$REPO/tools/chain_champ.sh"; } > "$_FF"
# shellcheck disable=SC1090
source "$_FF"; rm -f "$_FF"
type arm_flags > /dev/null 2>&1 || { echo "could not load arm_flags from tools/chain_champ.sh"; exit 2; }
src_of () { case $1 in SA128) echo wladder_p0;; *) echo wladder_p1;; esac; }

mk_se () {  # ARM [BANK_STEPS] [EXTRA_BANKED_FLAGS] — the ladder prefix as the width ladder left it; the fresh saext prefix empty
  local arm=$1 bst=${2:-30000} xtra=${3:-}; local sp; sp=$(src_of "$arm")
  mk_sandbox
  cp "$REPO/tools/filler_full.sh" "$REPO/tools/chain_saext.sh" "$REPO/tools/resume_flags_guard.py" "$REPO/tools/pretrain.py" "$REPO/tools/dev30.py" "$SB/repo/tools/"
  ln -s "$REPO/src" "$SB/repo/src"
  rm -rf /tmp/filler_pull /tmp/se_jc_src.tgz; rm -f /tmp/se_src_*.tgz
  local G="$SB/gcs/$sp"; mkdir -p "$G/evals" "$SB/fx"
  # the stub merge carries the shards' provenance (ckpt, argv), as the real evaluator's summary_all.json does (grid_of reads its ckpt)
  "$REAL_PY" - "$SB/bin/stubpy" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
old = '        (d / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4, "wall_s": 5.0}))\n'
new = ('        s0 = sorted(d.glob("summary_s*.json")); prov0 = json.loads(s0[0].read_text()) if s0 else {}; prov0.pop("n", None)\n'
       '        (d / "summary_all.json").write_text(json.dumps({**prov0, "n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4, "wall_s": 5.0}))\n')
assert s.count(old) == 1, "stub merge line not found"
s = s.replace(old, new)
# STUB_PEAK_AT="<ARM>:<step>": the monitor peaks at exactly that grid (the selection must find it across the 30k join)
old2 = '            v = (0.30 + 0.0001 * s) if peak_last else (0.30 if s <= steps // 2 else 0.29)   # peak_last: rising to the end; else a tie over the first half\n'
new2 = ('            pk = os.environ.get("STUB_PEAK_AT", ""); pk_arm, _, pk_step = pk.partition(":")\n'
        '            v = (0.60 if s == int(pk_step) else 0.25) if (pk_arm == base and pk_step) else ((0.30 + 0.0001 * s) if peak_last else (0.30 if s <= steps // 2 else 0.29))\n')
assert s.count(old2) == 1, "stub monitor line not found"
p.write_text(s.replace(old2, new2))
PYEOF
  echo ok > "$G/${arm}_ARM_OK"; echo ok > "$G/${arm}_PRETRAIN_OK"; echo ok > "$G/evals/l5k_${arm}_vsel_t16_OK"; echo ok > "$G/SYNC-AB"; printf 'BATCHONLY' > "$G/RECIPE-DEC"
  echo ok > "$G/PREFLIGHT_OK_w0_nw1"; echo "the ladder's compile cache" > "$G/jax_cache.tgz"; echo "the ladder's manifest" > "$G/wladder_final.tgz"
  if [ "${NO_STATE:-0}" != 1 ]; then
    local FL; FL="$(arm_flags "$arm") $xtra"
    # shellcheck disable=SC2086
    (cd "$SB/fx" && STUB_PEAK_AT="${BANK_PEAK:-}" "$SB/bin/stubpy" tools/pretrain.py --out "runs/pretrainchamp_$arm" $FL --steps "$bst" > /dev/null 2>&1)
    # the banked config in the REAL trainer's format: argv = vars(the trainer's parse of this exact argv)
    "$REAL_PY" - "$SB/fx/runs/pretrainchamp_$arm/config.json" "$REPO/tools" <<'PYEOF' || { echo "fixture: the real-format config could not be built"; exit 2; }
import json, sys
p, tools = sys.argv[1], sys.argv[2]; sys.path.insert(0, tools)
import pretrain
argv = json.load(open(p))["argv"][1:]
sys.argv = ["pretrain.py"] + argv; a = vars(pretrain.parse_args())
json.dump({"argv": a, "config": {"stub": True}}, open(p, "w"))
PYEOF
    "$REAL_PY" - "$SB/fx/runs/pretrainchamp_$arm" "$bst" <<'PYEOF'
import json, os, sys
d, bst = sys.argv[1], int(sys.argv[2]); half = bst // 2
if half % 2000:   # the stub logs a monitor row + a grid at half its budget; the real trainer (and the ladder's banked state) has 2,000-step grids only
    rows = [l for l in open(os.path.join(d, "metrics.jsonl")) if l.strip()]
    keep = [l for l in rows if json.loads(l).get("step", json.loads(l).get("monitor", {}).get("step")) != half]
    open(os.path.join(d, "metrics.jsonl"), "w").write("".join(keep))
    g = os.path.join(d, f"ckpt_{half:06d}.pkl"); os.path.exists(g) and os.remove(g)
PYEOF
    [ "$arm" = SA192 ] && echo 25500 > "$SB/fx/runs/pretrainchamp_$arm/resumes.txt"
    tar czf "$G/${arm}_pretrain.tgz" -C "$SB/fx" "runs/pretrainchamp_$arm"
    cp "$SB/fx/runs/pretrainchamp_$arm/ckpt_latest.pkl" "$G/${arm}_ckpt.pkl"
  fi
  (cd "$SB/gcs/$sp" && find . -type f | sort | while read -r f; do printf '%s %s\n' "$(cksum < "$f")" "$f"; done) > "$SB/src_before.txt"
  SE_ARM_=$arm; SE_SRCP=$sp
}
run_se () {  # [VAR=val...]
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 \
     GCS="gs://qhrrn2-rescue/saext_$SE_ARM_" SE_ARM="$SE_ARM_" SE_SRC="gs://qhrrn2-rescue/$SE_SRCP" LIVE_PREFIX="gs://qhrrn2-rescue/saext_$SE_ARM_/live" \
     C1_EXT_WINDOW=4000 LIVE_NO_GUARD=1 LIVE_EVERY=1 STALL_SEC=3 WATCH_POLL=1 PASS_SLEEP=1 IDLE_POLL=1 SE_PASSES=2 "$@" \
     bash tools/chain_saext.sh > "$SB/se.log" 2>&1; echo $? > "$SB/se.rc")
}
src_same () { (cd "$SB/gcs/$SE_SRCP" && find . -type f | sort | while read -r f; do printf '%s %s\n' "$(cksum < "$f")" "$f"; done) | cmp -s - "$SB/src_before.txt"; }
line_of () { grep -n "$1" "$SB/se.log" | head -1 | cut -d: -f1; }
ckst () { "$REAL_PY" -c "import pickle,sys; print(pickle.load(open(sys.argv[1],'rb'))['step'])" "$1"; }
rowck () { "$REAL_PY" -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['n'], s.get('ckpt'))" "$1" 2>/dev/null; }

# ---------------- E1: each arm fresh -> COMPLETE ----------------
for spec in "SA256 40000 -" "SA192 - 28000" "SA128 last -"; do
  set -- $spec; arm=$1; ext_peak=$2; bank_peak=$3
  echo "== E1 $arm fresh -> COMPLETE (the validation peak: $([ "$ext_peak" = last ] && echo 'the last grid, 50,000' || { [ "$ext_peak" = - ] && echo "inside the LADDER's 30k, at $bank_peak" || echo "inside the extension, at $ext_peak"; })) =="
  BANK_PEAK=$([ "$bank_peak" = - ] && echo "" || echo "$arm:$bank_peak") mk_se "$arm"
  JB=SE_JOBS=d64full,d128sub,d256sub,k128; [ "$arm" = SA192 ] && JB=SE_UNSET=1   # the env's explicit comma form; SA192 = the default (SE_JOBS unset)
  case $ext_peak in last) run_se "$JB" STUB_PEAK_LAST_ARM="$arm"; want=050000;; -) run_se "$JB"; want=0$bank_peak;; *) run_se "$JB" STUB_PEAK_AT="$arm:$ext_peak"; want=0$ext_peak;; esac
  G="$SB/gcs/saext_$arm"; D="$SB/repo/runs/pretrainchamp_$arm"; PL="$D.log"
  [ "$(cat "$SB/se.rc")" = 0 ] && grep -q "CHAIN-SAEXT-COMPLETE" "$SB/se.log" && [ -f "$G/saext_final.tgz" ] && ok "E1 $arm rc 0, sentinel, saext_final.tgz" || { bad "E1 $arm completion (rc $(cat "$SB/se.rc"))"; tail -14 "$SB/se.log"; }
  grep -q "SE-RESTORE-SRC" "$SB/se.log" && grep -q "SE-BANKED-CONFIG staged" "$SB/se.log" && grep -q "SE-RESUME-POINT step 30000" "$SB/se.log" && [ -f "$G/${arm}_config_banked.json" ] && cmp -s "$G/${arm}_config_banked.json" "$D/config_banked.json" \
    && ok "E1 $arm the 30k state restored from the ladder prefix; the banked config staged (node + fresh prefix); resume point 30000" || bad "E1 $arm restore: $(grep '^SE-' "$SB/se.log" | tr '\n' ' ' | cut -c1-300)"
  grep -q "PRETRAIN-RESUME-FLAGS-OK $arm RESUME-FLAGS-IDENTICAL .* allowed changes: steps 30000->50000" "$SB/se.log" && [ "$(line_of "PRETRAIN-RESUME-FLAGS-OK")" -lt "$(line_of "PRETRAIN-START $arm")" ] \
    && ok "E1 $arm the resume guard ran BEFORE the trainer: every key identical to the banked run's but steps 30000 -> 50000" || bad "E1 $arm guard: $(grep 'RESUME-FLAGS' "$SB/se.log" | cut -c1-300)"
  grep -q "PRETRAIN-EXTENDED-BUDGET $arm 50000" "$SB/se.log" && grep -q "PRETRAIN-START $arm .*steps=50000" "$SB/se.log" && grep -q "PRETRAIN-FIXED-BUDGET $arm 50000" "$SB/se.log" && ! grep -q "PRETRAIN-EXTEND $arm\|EXTENDED from 50000" "$SB/se.log" \
    && ok "E1 $arm the registered budget 50,000 through the extension path; no second extension (fixed budget)" || bad "E1 $arm budget: $(grep 'PRETRAIN-' "$SB/se.log" | tr '\n' ' ' | cut -c1-300)"
  grep -q "RESUMED from .* at step 30000" "$PL" && [ "$(ckst "$D/ckpt_latest.pkl")" = 50000 ] && ok "E1 $arm the trainer RESUMED at 30,000 (never from scratch) and ended at 50,000" || bad "E1 $arm trainer resume: $(head -2 "$PL")"
  "$REAL_PY" - "$D/metrics.jsonl" <<'PYEOF' && ok "E1 $arm one continuous monitor curve: 25 grids 2,000..50,000, the ladder's 15 kept, no step twice" || bad "E1 $arm metrics continuity"
import json, sys
st = [json.loads(l)["monitor"]["step"] for l in open(sys.argv[1]) if '"monitor"' in l]
assert st == sorted(st) and len(st) == len(set(st)) and st == list(range(2000, 50001, 2000)), st
PYEOF
  x=$(pargv "$arm"); fl=$(arm_flags "$arm"); fl=$(echo $fl)   # word-split: the chain's echo spans continued lines
  echo "$x" | grep -qF -- "$fl" && [ "$(echo "$x" | tr ' ' '\n' | awk '$0=="--steps"{getline v; r=v} END{print r}')" = 50000 ] && echo "$x" | grep -q -- "--monitor-every 2000 --grid-every 2000 --ckpt-every 500 " \
    && ok "E1 $arm the trainer argv = the chain's arm flags verbatim + --steps 50000 (the ladder's cadence)" || bad "E1 $arm trainer argv: $x"
  grep -q "VALBEST $arm $want" "$SB/se.log" && ok "E1 $arm the selection over all 25 grids picks $want" || bad "E1 $arm selection: $(grep VALBEST "$SB/se.log") (want $want)"
  CKW="runs/pretrainchamp_$arm/ckpt_$want.pkl"
  allb=1; for r in full_${arm}_vsel_t16 full_${arm}_final_t16 full_${arm}_vsel_t16_alt full_${arm}_vsel_t64 d128_${arm} d256_${arm} scan_${arm} census_${arm}_vsel census_${arm}_final calib_${arm}_vsel screen_${arm}_vb screen_${arm}_s010000 screen_${arm}_s020000; do [ -f "$G/evals/${r}_OK" ] || { allb=0; echo "    missing $r"; }; done
  [ $allb = 1 ] && [ -f "$G/${arm}_ARM_OK" ] && [ -f "$G/${arm}_PRETRAIN_OK" ] && [ -f "$G/${arm}_pretrain.tgz" ] && ! grep -q "LADDER_BATTERY\|ladder battery" "$SB/se.log" \
    && ok "E1 $arm the FULL champion battery (screens at 10k / 20k / the selection, D16 full, final, raw, D64, D128, D256, k32 scan, census x2, calibration)" || bad "E1 $arm battery"
  R="$SB/repo/runs"
  [ "$(rowck "$R/sxeval_pchamp$arm/full_vsel_t16/summary_all.json")" = "422786 $CKW" ] && [ "$(rowck "$R/sxeval_pchamp$arm/full_vsel_t64/summary_all.json")" = "100000 $CKW" ] \
    && [ "$(rowck "$R/sxeval_pchamp$arm/sub20k_t128/summary_all.json")" = "20000 $CKW" ] && [ "$(rowck "$R/sxeval_pchamp$arm/sub5k_t256/summary_all.json")" = "5000 $CKW" ] \
    && [ "$(rowck "$R/sxscan_pchamp$arm/summary_all.json")" = "5000 $CKW" ] && [ "$(rowck "$R/sxeval_pchamp$arm/full_vsel_t16_alt/summary_all.json")" = "50000 $CKW" ] \
    && [ "$(rowck "$R/sxeval_pchamp$arm/full_final_t16/summary_all.json")" = "50000 runs/pretrainchamp_$arm/ckpt_latest.pkl" ] \
    && ok "E1 $arm the battery's sizes (422,786 / 100k / 20k / 5k / the 5k scan / the raw row on 50k) on the selected grid; the final row on ckpt_latest (50k) even when the selection is the last grid" || bad "E1 $arm row sizes / grid: D16 $(rowck "$R/sxeval_pchamp$arm/full_vsel_t16/summary_all.json") | final $(rowck "$R/sxeval_pchamp$arm/full_final_t16/summary_all.json")"
  e16=$(eargv "$R/sxeval_pchamp$arm/full_vsel_t16/summary_s0.json")
  echo "$e16" | grep -q -- "--split test --t-total 16 --ema --record-by-step --record-q" && ok "E1 $arm the headline row's flags (EMA, the exact bit per step, the halting logits)" || bad "E1 $arm D16 flags: $e16"
  fb=1
  for spec2 in "d64full filler_sxeval_pchamp${arm}_full_t64 422786 --split test --t-total 64 --ema --record-by-step" \
               "d128sub filler_sxeval_pchamp${arm}_sub50000_t128 50000 --split test --subsample 50000 --t-total 128 --ema --record-by-step" \
               "d256sub filler_sxeval_pchamp${arm}_sub50000_t256 50000 --split test --subsample 50000 --t-total 256 --ema --record-by-step" \
               "k128 filler_sxscan128_pchamp${arm} 5000 --split test --subsample 5000 --t-total 64 --k-init 128 --ema"; do
    job=${spec2%% *}; rest=${spec2#* }; dir=${rest%% *}; rest=${rest#* }; n=${rest%% *}; flg=${rest#* }
    [ -f "$G/filler/${job}_${arm}_OK" ] && [ "$(rowck "$R/$dir/summary_all.json")" = "$n $CKW" ] && eargv "$R/$dir/summary_s0.json" | grep -q -- "$flg" || { fb=0; echo "    filler $job: $(rowck "$R/$dir/summary_all.json") | $(eargv "$R/$dir/summary_s0.json" 2>/dev/null | cut -c1-160)"; }
  done
  [ $fb = 1 ] && ok "E1 $arm the four filler rows (D64 on all 422,786, D128 / D256 on the 50k, the k128 scan) on the selected grid with the paper's width-192 flags" || bad "E1 $arm filler rows"
  src_same && ok "E1 $arm the ladder prefix byte-identical (nothing written under \$SRC; its $arm markers not taken as ours)" || { bad "E1 $arm \$SRC changed"; (cd "$SB/gcs/$SE_SRCP" && find . -type f | sort) | diff - <(awk '{print $3}' "$SB/src_before.txt") | head -5; }
  grep -q "EXTENDED from 30000 to 50000" "$G/${arm}_EXTENDED" && [ "$(cat "$G/RECIPE-DEC")" = BATCHONLY ] && [ -f "$G/SYNC-AB" ] && ! grep -q "SYNC-AB-OK" "$SB/se.log" && [ -d "$G/live/runs" ] && [ ! -d "$SB/gcs/$SE_SRCP/live" ] \
    && ok "E1 $arm the pre-staged markers (budget, scan recipe, no sync rider); the live bank under the fresh prefix only" || bad "E1 $arm pre-staged markers"
  ch=$(line_of "CHAIN-CHAMPSE-COMPLETE"); f1=$(line_of "SE-FILLER arms"); sp=$(line_of "CHAIN-SAEXT-COMPLETE")
  [ -n "$ch" ] && [ -n "$f1" ] && [ -n "$sp" ] && [ "$ch" -lt "$f1" ] && [ "$f1" -lt "$sp" ] && ! grep -q "SELF-TEARDOWN" "$SB/se.log" && ok "E1 $arm order: the inner chain's sentinel, the filler, the outer sentinel; no self-teardown" || bad "E1 $arm sentinel order ($ch $f1 $sp)"
  m=$(tar tzf "$G/saext_final.tgz"); mb=1
  for want_f in "pretrainchamp_$arm/metrics.jsonl" "pretrainchamp_$arm/config.json" "pretrainchamp_$arm/config_banked.json" "pretrainchamp_$arm/EXTENDED.txt" "pretrainchamp_$arm/val_best.txt" "pretrainchamp_$arm.guard.log" \
                "sxeval_pchamp$arm/full_vsel_t16/summary_all.json" "sxscan_pchamp$arm/records_all.npz" "filler_sxscan128_pchamp$arm/records_all.npz" "filler_sxeval_pchamp${arm}_full_t64/summary_all.json" "sxcalib_pchamp${arm}_vsel/calib.json"; do
    echo "$m" | grep -q "$want_f" || { mb=0; echo "    manifest lacks $want_f"; }
  done
  [ "$arm" != SA192 ] || echo "$m" | grep -q "pretrainchamp_SA192/resumes.txt" || { mb=0; echo "    manifest lacks SA192's resumes.txt"; }
  [ $mb = 1 ] && ok "E1 $arm the manifest (metrics, both configs, the guard log, the rows, the scans$([ "$arm" = SA192 ] && echo ', the ladder resume record'))" || bad "E1 $arm manifest"
  [ "$arm" = SA256 ] && SBE1=$SB
done

# ---------------- E2 ----------------
echo "== E2 idempotent rerun =="
SB=$SBE1; SE_ARM_=SA256; SE_SRCP=wladder_p1; run_se STUB_PEAK_AT=SA256:40000
[ "$(cat "$SB/se.rc")" = 0 ] && grep -q "(final object present)" "$SB/se.log" && ! grep -q "PRETRAIN-START\|FILLER-JOB-START" "$SB/se.log" && ok "E2 rerun: complete at once, nothing re-run" || { bad "E2 rerun"; tail -5 "$SB/se.log"; }
rm -f "$SB/gcs/saext_SA256/saext_final.tgz"; run_se STUB_PEAK_AT=SA256:40000
[ "$(cat "$SB/se.rc")" = 0 ] && ! grep -q "PRETRAIN-START\|FILLER-JOB-START" "$SB/se.log" && grep -q "SE-PRETRAIN-DONE" "$SB/se.log" && grep -q "FINAL-BANKED" "$SB/se.log" && [ -f "$SB/gcs/saext_SA256/saext_final.tgz" ] && ok "E2b markers present, manifest absent -> only the manifest is rebuilt" || bad "E2b manifest rebuild"

# ---------------- E3 ----------------
for mm in "--dec-attn-heads 8" "--lr 2e-4 --lr-end 2e-4" "--sot"; do
  echo "== E3 the banked run differs from the chain's argv [$mm] -> the guard refuses the arm =="
  if [ "$mm" = "--sot" ]; then   # a boolean the chain passes but the banked run did not: build the banked config WITHOUT --sot
    mk_se SA256; "$REAL_PY" - "$SB/gcs/wladder_p1/SA256_pretrain.tgz" "$SB/fx" <<'PYEOF'
import json, sys, tarfile, os
t, fx = sys.argv[1], sys.argv[2]; p = os.path.join(fx, "runs/pretrainchamp_SA256/config.json"); c = json.load(open(p)); c["argv"]["sot"] = False; json.dump(c, open(p, "w"))
with tarfile.open(t, "w:gz") as z: z.add(os.path.join(fx, "runs/pretrainchamp_SA256"), arcname="runs/pretrainchamp_SA256")
PYEOF
    (cd "$SB/gcs/wladder_p1" && find . -type f | sort | while read -r f; do printf '%s %s\n' "$(cksum < "$f")" "$f"; done) > "$SB/src_before.txt"
  else mk_se SA256 30000 "$mm"; fi
  run_se STUB_PEAK_AT=SA256:40000; G="$SB/gcs/saext_SA256"
  [ "$(cat "$SB/se.rc")" = 1 ] && grep -q "PRETRAIN-RESUME-FLAGS-ABORT SA256" "$SB/se.log" && grep -q "RESUME-FLAGS-DIFFER" "$SB/se.log" && ! grep -q "PRETRAIN-START" "$SB/se.log" && [ ! -f "$SB/repo/runs/pretrainchamp_SA256.log" ] \
     && [ ! -f "$G/SA256_ARM_OK" ] && [ ! -f "$G/SA256_PRETRAIN_OK" ] && grep -q "SAEXT-INCOMPLETE" "$SB/se.log" && ! grep -q "CHAIN-SAEXT-COMPLETE" "$SB/se.log" \
    && ok "E3 [$mm] refused before the trainer: $(grep -o 'RESUME-FLAGS-DIFFER [a-z_0-9]*' "$SB/se.log" | head -2 | tr '\n' ' ')no pretrain, INCOMPLETE" || { bad "E3 [$mm] (rc $(cat "$SB/se.rc"))"; grep "RESUME-FLAGS\|PRETRAIN-" "$SB/se.log" | head -4 | cut -c1-240; }
done

# ---------------- E4 ----------------
echo "== E4 no resume state / a state short of 30,000 / a banked config that is not the 30k run's =="
NO_STATE=1 mk_se SA256; run_se
[ "$(cat "$SB/se.rc")" = 2 ] && ! grep -q "SE-LANE1\|PRETRAIN-START" "$SB/se.log" && [ ! -f "$SB/gcs/saext_SA256/SA256_ARM_OK" ] && ok "E4a no banked state: rc 2 ($(grep -o 'SE-NO-[A-Z-]*' "$SB/se.log" | head -1)), nothing launched" || { bad "E4a (rc $(cat "$SB/se.rc"))"; tail -4 "$SB/se.log"; }
mk_se SA256 20000; run_se
[ "$(cat "$SB/se.rc")" = 2 ] && grep -q "SE-NO-BANKED-CONFIG" "$SB/se.log" && ! grep -q "SE-LANE1" "$SB/se.log" && ok "E4b a 20k banked run: its config is not the 30k run's -> rc 2, nothing launched" || { bad "E4b (rc $(cat "$SB/se.rc"))"; tail -4 "$SB/se.log"; }
mk_se SA256; "$REAL_PY" - "$SB/gcs/wladder_p1/SA256_pretrain.tgz" "$SB/fx" <<'PYEOF'
import pickle, sys, tarfile, os
t, fx = sys.argv[1], sys.argv[2]; p = os.path.join(fx, "runs/pretrainchamp_SA256/ckpt_latest.pkl"); c = pickle.load(open(p, "rb")); c["step"] = 20000; pickle.dump(c, open(p, "wb"))
with tarfile.open(t, "w:gz") as z: z.add(os.path.join(fx, "runs/pretrainchamp_SA256"), arcname="runs/pretrainchamp_SA256")
PYEOF
run_se
[ "$(cat "$SB/se.rc")" = 2 ] && grep -q "SE-NO-RESUME-STATE (ckpt_latest step 20000 outside \[30000, 50000\]" "$SB/se.log" && ! grep -q "SE-LANE1" "$SB/se.log" && ok "E4c a ckpt_latest at 20,000 -> SE-NO-RESUME-STATE rc 2" || { bad "E4c (rc $(cat "$SB/se.rc"))"; tail -4 "$SB/se.log"; }

# ---------------- E5 ----------------
echo "== E5 a preemption: the live prefix holds a later state (40,000) and the banked config =="
mk_se SA192; fl=$(arm_flags SA192)
mkdir -p "$SB/lv" && (cd "$SB/lv" && mkdir -p runs && tar xzf "$SB/gcs/wladder_p1/SA192_pretrain.tgz" && cp runs/pretrainchamp_SA192/config.json runs/pretrainchamp_SA192/config_banked.json \
  && "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainchamp_SA192 $fl --steps 40000 > /dev/null 2>&1)
mkdir -p "$SB/gcs/saext_SA192/live/runs" && cp -R "$SB/lv/runs/pretrainchamp_SA192" "$SB/gcs/saext_SA192/live/runs/"
echo "EXTENDED from 30000 to 50000 (pre-staged by the first launch)" > "$SB/gcs/saext_SA192/SA192_EXTENDED"; cp "$SB/lv/runs/pretrainchamp_SA192/config_banked.json" "$SB/gcs/saext_SA192/SA192_config_banked.json"
run_se STUB_PEAK_AT=SA192:44000; PL="$SB/repo/runs/pretrainchamp_SA192.log"
[ "$(cat "$SB/se.rc")" = 0 ] && grep -q "SE-RESUME-POINT step 40000" "$SB/se.log" && ! grep -q "SE-RESTORE-SRC" "$SB/se.log" && grep -q "RESUMED from .* at step 40000" "$PL" && grep -q "PRETRAIN-RESUME-FLAGS-OK SA192" "$SB/se.log" \
   && grep -q "VALBEST SA192 044000" "$SB/se.log" && grep -q "CHAIN-SAEXT-COMPLETE" "$SB/se.log" && ok "E5 the live state (40,000) is the resume point; \$SRC's tarball never extracted; the guard passes on the banked config; complete" || { bad "E5 (rc $(cat "$SB/se.rc"))"; grep "^SE-\|RESUMED\|VALBEST\|RESUME-FLAGS" "$SB/se.log" "$PL" | head -6; }
src_same && ok "E5 the ladder prefix byte-identical" || bad "E5 \$SRC changed"

# ---------------- E6 ----------------
echo "== E6 an arm or a job outside the registration =="
mk_se SA256; SE_ARM_=SA256L; run_se
[ "$(cat "$SB/se.rc")" = 2 ] && grep -q "SE-BAD-ARM SA256L" "$SB/se.log" && ! grep -q "SE-LANE1\|SE-RESTORE" "$SB/se.log" && ok "E6a an arm outside the extension is refused before any work" || { bad "E6a"; tail -3 "$SB/se.log"; }
mk_se SA256; run_se SE_JOBS=d64full,k32
[ "$(cat "$SB/se.rc")" = 2 ] && grep -q "SE-BAD-JOB k32" "$SB/se.log" && ! grep -q "SE-LANE1\|SE-RESTORE" "$SB/se.log" && ok "E6b a filler job outside the registration is refused before any work" || { bad "E6b"; tail -3 "$SB/se.log"; }

# ---------------- E7 ----------------
echo "== E7 SE_JOBS=none -> the champion battery alone (option A) =="
mk_se SA128; run_se SE_JOBS=none STUB_PEAK_AT=SA128:36000; G="$SB/gcs/saext_SA128"
[ "$(cat "$SB/se.rc")" = 0 ] && grep -q "SE-FILLER-NONE" "$SB/se.log" && grep -q "CHAIN-SAEXT-COMPLETE" "$SB/se.log" && [ -f "$G/SA128_ARM_OK" ] && [ ! -d "$G/filler" ] && ! grep -q "FILLER-JOB-START" "$SB/se.log" \
  && ok "E7 complete on the champion battery alone; no filler row" || { bad "E7 (rc $(cat "$SB/se.rc"))"; tail -4 "$SB/se.log"; }

# ---------------- E8 ----------------
echo "== E8 a relaunch on a fresh node after the pretrain is banked =="
SB=$SBE1; SE_ARM_=SA256; SE_SRCP=wladder_p1; G="$SB/gcs/saext_SA256"
rm -f "$G/saext_final.tgz" "$G/SA256_ARM_OK" "$G/filler/k128_SA256_OK" "$G/filler/k128_SA256.tgz" "$G/evals/full_SA256_vsel_t64_OK" "$G/evals/full_SA256_vsel_t64.tgz"; rm -rf "$SB/repo/runs" "$G/live" /tmp/filler_pull; mkdir -p "$SB/repo/runs"
run_se STUB_PEAK_AT=SA256:40000
[ "$(cat "$SB/se.rc")" = 0 ] && grep -q "SE-PRETRAIN-DONE" "$SB/se.log" && grep -q "PRETRAIN-SKIP SA256" "$SB/se.log" && ! grep -q "PRETRAIN-START\|SE-RESTORE-SRC\|RESUME-FLAGS" "$SB/se.log" && grep -q "PRETRAIN-RESTORE SA256" "$SB/se.log" \
   && grep -q "VALBEST SA256 040000" "$SB/se.log" && [ -f "$G/evals/full_SA256_vsel_t64_OK" ] && [ -f "$G/filler/k128_SA256_OK" ] && grep -q "CHAIN-SAEXT-COMPLETE" "$SB/se.log" \
  && ok "E8 PRETRAIN-SKIP; the 25 banked grids restored from the fresh prefix; the same selection (040000); the missing rows re-run; complete" || { bad "E8 (rc $(cat "$SB/se.rc"))"; grep "SE-\|PRETRAIN-\|VALBEST" "$SB/se.log" | head -8; }

echo; echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
