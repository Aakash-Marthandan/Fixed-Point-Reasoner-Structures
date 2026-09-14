#!/bin/bash
# THE WIDTH-192 LONG RUN chain (2026-09-14; Documentation/Plan_2026-09-14_W192_Long.md = the registration; the PI: "let's do it, this is
# worth looking into for the ARC DEC port — but use a 50k subsample for evals to test things rather than the full"). ONE v6e-8.
# Nothing under champ/, c8x/ or finalA/ is written: champ/C5_pretrain.tgz (C5's 50k state), champ/C1_pretrain.tgz, finalA/A{5,7,8}_pretrain.tgz,
# c8x/val/C5_s*.tgz (C5's held-out rows 10k-50k), c8x/sets/ (the 10k held-out set) and champ/jax_cache.tgz are READ; everything written
# lives under $GCS (gs://qhrrn2-rescue/c5l). No evaluation touches the full test set: test rows use the 50k subsample (seed 20260822).
#   Lane 0  the resume point: the live prefix, else C5's banked state from champ (RETRY_REMAT.txt of the 4-chip champion night not carried:
#           8 chips x 96 rows as C7/C8; pt_run's own OOM retry stands) -> ASSERT ckpt_latest >= EXT_FROM (C5L-NO-RESUME-STATE otherwise)
#   Lane 1  tools/chain_champ.sh for C5 alone, pretrain-only (CHAMP_PRETRAIN_ONLY=1): preflight, the resume EXT_FROM -> EXT_TO through the
#           registered extension path (budget = C1_STEPS_X + C1_EXT_STEPS), the registered monitor selection over every banked grid (g_mon)
#   Lane 2  C5's instruments per grid, one grid per chip: held-out (10,000, split val, D16, EMA) at C5_EARLY and EXT_FROM+STEP..EXT_TO
#           (REUSE_LO..REUSE_HI read from c8x); train-1k (1,000, split train, D16, EMA) at every grid C5_TR1K_FROM..EXT_TO; g_val = the
#           held-out argmax over >= 10k (tools/c5l_curves.py)
#   Lane 3  test rows on the 50k subsample: D16 and D64 for {g_val, g_mon, EXT_TO} minus the paper's grid, sharded over the chips
#   Lane 4  the wide runs' instruments (eval only): held-out + train-1k at WIDE_GRIDS for WIDE_RUNS (C1 w384 champion s1, A5 w384 champion
#           s0, A7 w384 plain s1, A8 w512 plain s1)
# Idempotent by the markers under $GCS; completion = the manifest $FINAL + the sentinel. Harness: tools/harness_c5l.sh.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src
export GCS=${GCS:-gs://qhrrn2-rescue/c5l}
export LIVE_PREFIX=${LIVE_PREFIX:-$GCS/live}
SRC_CHAMP=${C5L_SRC_CHAMP:-gs://qhrrn2-rescue/champ}; SRC_FINALA=${C5L_SRC_FINALA:-gs://qhrrn2-rescue/finalA}; SRC_C8X=${C5L_SRC_C8X:-gs://qhrrn2-rescue/c8x}
SENT=${L_SENT:-CHAIN-C5L}; FINAL=${FINAL_OBJ:-c5l_final.tgz}
W=${CHAIN_WORKER:-0}
PY=${CHAIN_PY:-python3}; RPY=${REAL_PY:-python3}
TEST_NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz}
VAL_NPZ=${VAL_NPZ:-data/sudoku_extreme/sudoku_extreme_seed0_val10k.npz}
N_VAL=${N_VAL:-10000}; N_TR1K=${N_TR1K:-1000}; N_TEST=${N_TEST:-50000}
STEP=${C5L_STEP:-2000}; EXT_FROM=${C5L_EXT_FROM:-50000}; EXT_TO=${C5L_EXT_TO:-150000}
C5_EARLY=${C5L_EARLY:-004000 006000 008000}; C5_TR1K_FROM=${C5L_TR1K_FROM:-4000}; PICK_MIN=${C5L_PICK_MIN:-10000}
REUSE_LO=${C5L_REUSE_LO:-10000}; REUSE_HI=${C5L_REUSE_HI:-50000}
PAPER_GRID=${C5L_PAPER_GRID:-046000}
WIDE_RUNS=${C5L_WIDE_RUNS:-C1 A5 A7 A8}
WIDE_GRIDS=${C5L_WIDE_GRIDS:-004000 006000 008000 010000 012000 014000 016000 018000 020000 022000 024000 026000 028000 030000 034000 038000 042000 046000 050000}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}
D=runs/pretrainchamp_C5
pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c "$@"; }
ck_step () { JAX_PLATFORMS=cpu $RPY -c "import pickle,sys; print(int(pickle.load(open(sys.argv[1],'rb'))['step']))" "$1" 2>/dev/null; }
seqgrid () { local s; for s in $(seq "$1" "$STEP" "$2"); do printf '%06d\n' "$s"; done; }

echo "=== C5L START worker=$W chips=$NCHIP gcs=$GCS (sources read-only: champ, finalA, c8x) live=$LIVE_PREFIX $(date -u +%FT%TZ) ==="
gsutil -q stat "$GCS/$FINAL" 2>/dev/null && { echo "$SENT-COMPLETE worker=$W (final object present)"; exit 0; }
mkdir -p data/sudoku_extreme runs/analysis
[ -f "$VAL_NPZ" ] || gsutil -q cp "$SRC_C8X/sets/$(basename "$VAL_NPZ")" "$VAL_NPZ" || { echo "C5L-VALSET-MISSING $SRC_C8X/sets/$(basename "$VAL_NPZ")"; exit 2; }
echo "C5L-VALSET-OK $(basename "$VAL_NPZ")"

# ---------- lane 0: the pre-staged markers (under $GCS only) and the resume point ----------
gsutil -q stat "$GCS/C5_EXTENDED" 2>/dev/null || echo "EXTENDED from $EXT_FROM to $EXT_TO (the PI's long run 2026-09-14: the width-192 clocks; not the registered champion rule) $(date -u +%FT%TZ)" | gsutil -q cp - "$GCS/C5_EXTENDED"
gsutil -q stat "$GCS/SYNC-AB" 2>/dev/null || echo "skipped: the long run carries no sync rider" | gsutil -q cp - "$GCS/SYNC-AB"
if ! gsutil -q stat "$GCS/jax_cache.tgz" 2>/dev/null && [ ! -d jax_cache ]; then
  gsutil -q cp "$SRC_CHAMP/jax_cache.tgz" /tmp/c5l_jc_src.tgz 2>/dev/null && tar xzf /tmp/c5l_jc_src.tgz 2>/dev/null && echo "COMPILE-CACHE seeded from $SRC_CHAMP (read-only)"
fi
GCS="$GCS" R_TAG=champ ARMS="C5" LIVE_PREFIX="$LIVE_PREFIX" bash tools/live_bank.sh restore
if gsutil -q stat "$GCS/C5_PRETRAIN_OK" 2>/dev/null; then
  echo "C5L-PRETRAIN-DONE (the extended grids are banked under $GCS; the $SRC_CHAMP 50k state is not restored)"
else
  if [ ! -f "$D/ckpt_latest.pkl" ]; then
    gsutil -q cp "$SRC_CHAMP/C5_pretrain.tgz" /tmp/c5l_c5_src.tgz 2>/dev/null && tar xzf /tmp/c5l_c5_src.tgz --exclude="*/RETRY_REMAT.txt" 2>/dev/null \
      && echo "C5L-RESTORE-SRC C5's banked state from $SRC_CHAMP/C5_pretrain.tgz (read-only; RETRY_REMAT.txt of the 4-chip champion night not carried)"
  fi
  st=$(ck_step "$D/ckpt_latest.pkl")
  if [ -z "$st" ] || [ "$st" -lt "$EXT_FROM" ]; then
    echo "C5L-NO-RESUME-STATE (ckpt_latest step ${st:-none} < $EXT_FROM: the long run never trains from scratch) $(date -u +%FT%TZ)"; exit 2
  fi
  echo "C5L-RESUME-POINT step $st (the long run $EXT_FROM -> $EXT_TO)"
fi

# ---------- lane 1: C5 through the champion chain, pretrain-only ----------
echo "C5L-LANE1 C5 via tools/chain_champ.sh (pretrain-only) $(date -u +%FT%TZ)"
SELF_TEARDOWN=0 CHAMP_PRETRAIN_ONLY=1 CHAMP_ALL_ARMS="C5" CHAMP_ARMS_1X8="C5" SENT=CHAMPC5L C1_WAIT_PASSES=1 C1_POLL_SLEEP=5 \
  C1_EXT_STEPS=$((EXT_TO - ${C1_STEPS_X:-30000})) bash tools/chain_champ.sh; rc1=$?
echo "C5L-LANE1-END rc=$rc1 $(date -u +%FT%TZ)"
orph=$(pgrep -fa 'tools/pretrain[.]py|tools/eval_sudoku_extreme[.]py' 2>/dev/null | cut -c1-160)
[ -z "$orph" ] || echo "C5L-ORPHANS lane 1 left trainer/evaluator processes alive (a kill is the PI's call): $(echo "$orph" | tr '\n' ';' | cut -c1-400)"
gsutil -q stat "$GCS/C5_ARM_OK" 2>/dev/null || { echo "C5L-INCOMPLETE (lane 1: no C5_ARM_OK; rc=$rc1) $(date -u +%FT%TZ)"; exit 1; }

R_TAG=champ ARMS="C5" LIVE_EVERY=${LIVE_EVERY:-300} nohup bash tools/live_bank.sh loop > runs/live_bank_c5l.log 2>&1 &
LBP=$!; trap 'kill "$LBP" 2>/dev/null' EXIT

# ---------- the grids of every run ----------
grids_dir () {  # RUN -> the directory holding the run's banked grids (C5 local; the wide runs pulled read-only outside runs/)
  local run=$1 P tgz sub
  if [ "$run" = C5 ]; then
    if [ ! -f "$D/ckpt_$(printf '%06d' "$EXT_TO").pkl" ]; then
      gsutil -q cp "$GCS/C5_pretrain.tgz" /tmp/c5l_c5_ext.tgz 2>/dev/null && tar xzf /tmp/c5l_c5_ext.tgz 2>/dev/null && echo "C5L-GRIDS-REPULL C5 from $GCS" >&2
    fi
    [ -f "$D/ckpt_$(printf '%06d' "$EXT_TO").pkl" ] || return 1
    echo "$D"; return 0
  fi
  case $run in C1) tgz="$SRC_CHAMP/C1_pretrain.tgz"; sub="runs/pretrainchamp_C1";; A5|A7|A8) tgz="$SRC_FINALA/${run}_pretrain.tgz"; sub="runs/pretrainfinalA_$run";; *) return 1;; esac
  P=/tmp/c5l_src/$run
  if [ -z "$(ls "$P/$sub"/ckpt_0*.pkl 2>/dev/null)" ]; then
    mkdir -p "$P" && gsutil -q cp "$tgz" "/tmp/c5l_src/${run}_pretrain.tgz" 2>/dev/null && tar xzf "/tmp/c5l_src/${run}_pretrain.tgz" -C "$P" 2>/dev/null \
      && echo "C5L-SRC-PULL $run from $tgz (read-only)" >&2
  fi
  [ -n "$(ls "$P/$sub"/ckpt_0*.pkl 2>/dev/null)" ] || return 1
  echo "$P/$sub"
}
OKS=""
refresh_oks () { OKS=$(gsutil ls "$GCS/val/*_OK" "$GCS/tr1k/*_OK" "$GCS/test/*_OK" 2>/dev/null); }
banked () { echo "$OKS" | grep -q "/$1/$2_OK\$"; }   # KIND NAME
row_finish () {  # KIND RUN STEP -> n-gate (n, split, EMA, t16), bank, mark
  local kind=$1 run=$2 st=$3 O n split
  O=runs/c5l_${kind}_p${run}_s${st}
  case $kind in val) n=$N_VAL; split=val;; tr1k) n=$N_TR1K; split=train;; esac
  [ -f "$O/summary_all.json" ] || JAX_PLATFORMS=cpu $PY tools/eval_sudoku_extreme.py --merge "$O" >> "$O/run.log" 2>&1
  $RPY -c "import json,sys; s=json.load(open('$O/summary_all.json')); sys.exit(0 if s['n']==$n and s.get('split')=='$split' and s.get('ema') and s.get('t_total')==16 else 1)" 2>/dev/null \
    || { echo "C5L-ROW-N-BAD ${kind}/${run}_s${st}"; return 1; }
  tar czf "/tmp/c5l_${kind}_${run}_s${st}.tgz" "$O" && gsutil -q cp "/tmp/c5l_${kind}_${run}_s${st}.tgz" "$GCS/$kind/${run}_s${st}.tgz" \
    && echo ok | gsutil -q cp - "$GCS/$kind/${run}_s${st}_OK" && echo "C5L-ROW-OK ${kind}/${run}_s${st} $(date -u +%H:%M)"
}
rows_pass () {  # LABEL then lines "KIND RUN STEP" -> evaluate the unbanked ones, one grid per chip per round
  local label=$1; shift
  local want=("$@") jobs=() x G i c pids grp p
  refresh_oks
  for x in "${want[@]}"; do
    set -- $x; banked "$1" "$2_s$3" && continue
    G=$(grids_dir "$2") || { echo "C5L-NO-GRIDS $2"; continue; }
    [ -f "$G/ckpt_$3.pkl" ] || { echo "C5L-NO-GRID $2 $3"; continue; }
    jobs+=("$1 $2 $3 $G/ckpt_$3.pkl")
  done
  echo "C5L-ROWS-PASS $label ${#jobs[@]} rows to evaluate $(date -u +%H:%M)"
  i=0
  while [ "$i" -lt "${#jobs[@]}" ]; do
    pids=(); grp=()
    for c in $(seq 0 $((NCHIP - 1))); do
      [ "$i" -lt "${#jobs[@]}" ] || break
      set -- ${jobs[$i]}
      local O=runs/c5l_$1_p$2_s$3 SPLIT
      [ "$1" = val ] && SPLIT="--split val" || SPLIT="--split train"
      rm -rf "$O"; mkdir -p "$O"
      pin "$c" $PY tools/eval_sudoku_extreme.py --ckpt "$4" --npz "$VAL_NPZ" --out "$O" $SPLIT --t-total 16 --ema > "$O/run.log" 2>&1 & pids+=($!); grp+=("$1 $2 $3"); i=$((i + 1))
    done
    for p in "${pids[@]}"; do wait "$p"; done
    for p in "${grp[@]}"; do set -- $p; row_finish "$1" "$2" "$3" || true; done
  done
}

# ---------- lane 2: C5's instruments ----------
C5_VAL=(); C5_TR=()
for s in $C5_EARLY $(seqgrid $((EXT_FROM + STEP)) "$EXT_TO"); do C5_VAL+=("val C5 $s"); done
for s in $(seqgrid "$C5_TR1K_FROM" "$EXT_TO"); do C5_TR+=("tr1k C5 $s"); done
echo "C5L-LANE2 C5 instruments: held-out ${#C5_VAL[@]} new grids (+ c8x reuse $REUSE_LO..$REUSE_HI), train-1k ${#C5_TR[@]} grids $(date -u +%FT%TZ)"
for s in $(seqgrid "$REUSE_LO" "$REUSE_HI"); do   # C5's held-out rows 10k-50k from the C8 extension (read-only), on disk for the curve
  [ -f "runs/c8x_val_pC5_s$s/summary_all.json" ] && continue
  gsutil -q cp "$SRC_C8X/val/C5_s$s.tgz" "/tmp/c5l_reuse_$s.tgz" 2>/dev/null && tar xzf "/tmp/c5l_reuse_$s.tgz" 2>/dev/null
done
for pass in 1 2; do rows_pass "c5-$pass" "${C5_VAL[@]}" "${C5_TR[@]}"; done
for m in $(gsutil ls "$GCS/val/C5_s*_OK" 2>/dev/null); do   # after a node change: the banked held-out rows back on disk for the pick
  n=$(basename "$m" _OK); [ -f "runs/c5l_val_pC5_s${n##*_s}/summary_all.json" ] && continue
  gsutil -q cp "$GCS/val/$n.tgz" "/tmp/c5l_val_$n.tgz" 2>/dev/null && tar xzf "/tmp/c5l_val_$n.tgz" 2>/dev/null
done
gval=$($RPY tools/c5l_curves.py --root runs --kind val --run C5 --prefixes c5l c8x --n "$N_VAL" --min-step "$PICK_MIN") \
  || { echo "C5L-INCOMPLETE (no held-out curve for C5) $(date -u +%FT%TZ)"; exit 1; }
if [ ! -f "$D/val_best.txt" ]; then   # after a node change: C5_pretrain.tgz is banked before the pick is written -> replay the registered selector
  grids_dir C5 > /dev/null || { echo "C5L-INCOMPLETE (no C5 grids to replay the monitor pick) $(date -u +%FT%TZ)"; exit 1; }
  sel=$($PY tools/select_ckpt.py "$D" --key val_t16_ema --tie earliest --second-key val_t16 2>/dev/null) && echo "$sel" > "$D/val_best.txt"
  rec=$(gsutil -q cp "$GCS/C5_ARM_OK" - 2>/dev/null | sed -n 's/^pretrain-only: \([0-9]*\).*/\1/p')
  echo "C5L-VALBEST-REPLAY ${sel:-none} (the pick recorded at ARM_OK: ${rec:-none})"
  [ -z "$rec" ] || [ "$rec" = "$(echo "$sel" | awk '{print $1}')" ] || { echo "C5L-VALBEST-MISMATCH replay ${sel:-none} vs recorded $rec"; exit 1; }
fi
gmon=$(awk '{print $1; exit}' "$D/val_best.txt" 2>/dev/null)
[ -n "$gmon" ] && [ "$gmon" != FALLBACK-FINAL ] || { echo "C5L-INCOMPLETE (no g_mon: val_best.txt ${gmon:-absent}) $(date -u +%FT%TZ)"; exit 1; }
echo "C5L-PICKS g_mon=$gmon g_val=$gval final=$(printf '%06d' "$EXT_TO") paper=$PAPER_GRID"

# ---------- lane 3: test rows on the 50k subsample ----------
TROWS=""
for g in $gval $gmon $(printf '%06d' "$EXT_TO"); do
  [ "$g" = "$PAPER_GRID" ] && continue
  case " $TROWS " in *" d16_s$g "*) continue;; esac
  TROWS="$TROWS d16_s$g d64_s$g"
done
echo "C5L-TEST-ROWS [${TROWS# }] (the 50k subsample, seed 20260822; never the full set)"
test_row () {  # NAME (d16_sSTEP | d64_sSTEP) -> sharded over the chips (partials every 300 s), merged, n-gated, banked $GCS/test/NAME.tgz + NAME_OK
  local name=$1 dep=${1%_s*} st=${1##*_s} O CK T rc p pids=() i
  O=runs/c5l_test_pC5_$name; CK=$D/ckpt_$st.pkl; T=${dep#d}
  refresh_oks; banked test "$name" && { echo "C5L-TEST-SKIP $name"; return 0; }
  [ -f "$CK" ] || { echo "C5L-TEST-NO-GRID $name ($CK)"; return 1; }
  mkdir -p "$O"; echo "C5L-TEST-START $name ck=$CK $(date -u +%H:%M)"
  for i in $(seq 0 $((NCHIP - 1))); do
    pin "$i" $PY tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$TEST_NPZ" --out "$O" --shard "$i/$NCHIP" --bank-every 300 \
      --split test --subsample "$N_TEST" --t-total "$T" --ema --record-by-step > "$O/shard_$i.log" 2>&1 & pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { echo "C5L-TEST-FAILED $name (a shard failed; partials stay for the resume)"; return 1; }
  JAX_PLATFORMS=cpu $PY tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  $RPY -c "import json,sys; s=json.load(open('$O/summary_all.json')); sys.exit(0 if s['n']==$N_TEST and s.get('t_total')==$T and s.get('ema') else 1)" 2>/dev/null || { echo "C5L-TEST-N-BAD $name"; return 1; }
  tar czf "/tmp/c5l_test_$name.tgz" "$O" && gsutil -q cp "/tmp/c5l_test_$name.tgz" "$GCS/test/$name.tgz" && echo ok | gsutil -q cp - "$GCS/test/${name}_OK" \
    && echo "C5L-TEST-OK $name $(date -u +%H:%M)"
}
for pass in 1 2; do for r in $TROWS; do test_row "$r" || true; done; done

# ---------- lane 4: the wide runs' instruments ----------
WIDE=()
for run in $WIDE_RUNS; do for s in $WIDE_GRIDS; do WIDE+=("val $run $s" "tr1k $run $s"); done; done
echo "C5L-LANE4 wide runs [$WIDE_RUNS] x ${#WIDE[@]} rows $(date -u +%FT%TZ)"
for pass in 1 2; do rows_pass "wide-$pass" "${WIDE[@]}"; done

# ---------- completion ----------
refresh_oks; need=""
gsutil -q stat "$GCS/C5_ARM_OK" 2>/dev/null || need="$need C5_ARM_OK"
for x in "${C5_VAL[@]}" "${C5_TR[@]}" "${WIDE[@]}"; do set -- $x; banked "$1" "$2_s$3" || need="$need $1/$2_s$3"; done
for r in $TROWS; do banked test "$r" || need="$need test/$r"; done
[ -z "$need" ] || { echo "C5L-INCOMPLETE worker=$W (missing:$(echo "$need" | cut -c1-600)) $(date -u +%FT%TZ)"; exit 1; }
$RPY - <<PYEOF > runs/analysis/c5l_curves.json 2>/dev/null
import json, sys
sys.path.insert(0, "tools")
from c5l_curves import curve
out = {"C5": {"val": curve("runs", "val", "C5", ("c5l", "c8x"), $N_VAL), "tr1k": curve("runs", "tr1k", "C5", ("c5l",), $N_TR1K)}}
for run in "$WIDE_RUNS".split():
    out[run] = {"val": curve("runs", "val", run, ("c5l",), $N_VAL), "tr1k": curve("runs", "tr1k", run, ("c5l",), $N_TR1K)}
print(json.dumps(out))
PYEOF
files=""
for p in $D/metrics.jsonl $D/val_best.txt $D/EXTENDED.txt $D/config.json $D/resumes.txt runs/preflightchamp_C5.log \
         runs/c5l_*/summary_all.json runs/analysis/c5l_curves.json; do
  [ -e "$p" ] && files="$files $p"
done
# shellcheck disable=SC2086
tar czf "/tmp/$FINAL" $files 2>/dev/null && gsutil -q cp "/tmp/$FINAL" "$GCS/$FINAL" && echo "FINAL-BANKED $GCS/$FINAL ($(echo $files | wc -w | tr -d ' ') files)"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
