#!/bin/bash
# THE C8 EXTENSION chain (2026-09-14; Documentation/Plan_2026-09-14_C8_Extension.md = the registration; the PI: "extend C8, the w192
# seed with the lowest score, to 50k steps ... check if we get a better optimal point than the current step"). ONE v6e-8 (1 x 8 chips).
# Nothing under $SRC (gs://qhrrn2-rescue/champ) is written: C8_pretrain.tgz (the 30k state), C5/C7_pretrain.tgz, jax_cache.tgz and
# sets/ are READ; every marker, tarball and the live bank live under the fresh prefix $GCS (gs://qhrrn2-rescue/c8x).
#   Lane 0  the resume point: restore the live prefix; else C8's banked 30k state from $SRC; ASSERT ckpt_latest >= step $EXT_FROM
#           (C8X-NO-RESUME-STATE otherwise: the extension never trains from scratch); once C8_PRETRAIN_OK exists the extended grids
#           come from $GCS only (the 30k tarball is never extracted over them).
#   Lane 1  tools/chain_champ.sh for C8 alone (CHAMP_ALL_ARMS="C8") with the pre-staged $GCS/C8_EXTENDED marker: preflight, the
#           resume $EXT_FROM -> $EXT_TO (the registered extension path, the same code that extended C5 and C7), the registered
#           selection over every banked grid (g_mon), the champion battery on g_mon.
#   Lane 2  the low-noise selection curves: every banked grid >= VAL_MIN_STEP of C8, C5 and C7 at D16 (EMA) on the 10,000-puzzle
#           held-out set (tools/sx_make_valset.py), one grid per chip; g_val = tools/c8x_valcurve.py's pick for C8.
#   Lane 3  the filler rows for g_mon (d64full, d128sub, d256sub); the extra test rows: g_val (if not g_mon and not the paper-final
#           grid) -> D16 full + D64 full; the PI's hypothesis grid $HYP (if not among them) -> D16 full + D64 on the 100k.
# Idempotent by the markers under $GCS; completion = the manifest $FINAL + the sentinel. Harness: tools/harness_c8x.sh.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src
export GCS=${GCS:-gs://qhrrn2-rescue/c8x}
export LIVE_PREFIX=${LIVE_PREFIX:-$GCS/live}
SRC=${C8X_SRC:-gs://qhrrn2-rescue/champ}
SENT=${X_SENT:-CHAIN-C8X}; FINAL=${FINAL_OBJ:-c8x_final.tgz}
W=${CHAIN_WORKER:-0}
PY=${CHAIN_PY:-python3}; RPY=${REAL_PY:-python3}
NPZ=${SX_NPZ_PATH:-data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz}
VAL_NPZ=${VAL_NPZ:-data/sudoku_extreme/sudoku_extreme_seed0_val10k.npz}; VAL_N=${VAL_N:-10000}; VAL_MIN_STEP=${VAL_MIN_STEP:-10000}
VAL_ARMS=${VAL_ARMS:-C8 C5 C7}
EXT_FROM=${C8X_EXT_FROM:-30000}; EXT_TO=${C8X_EXT_TO:-50000}
HYP=${C8X_HYP_GRID:-046000}; PF_GRID=${C8X_PF_GRID:-022000}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=${NCHIP_OVERRIDE:-4}
D=runs/pretrainchamp_C8
pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c "$@"; }
ck_step () { JAX_PLATFORMS=cpu $RPY -c "import pickle,sys; print(int(pickle.load(open(sys.argv[1],'rb'))['step']))" "$1" 2>/dev/null; }
step_of () { basename "$1" .pkl | sed 's/^ckpt_//'; }

echo "=== C8X START worker=$W chips=$NCHIP gcs=$GCS src=$SRC (read-only) live=$LIVE_PREFIX $(date -u +%FT%TZ) ==="
gsutil -q stat "$GCS/$FINAL" 2>/dev/null && { echo "$SENT-COMPLETE worker=$W (final object present)"; exit 0; }
mkdir -p data/sudoku_extreme runs/analysis
[ -f "$VAL_NPZ" ] || gsutil -q cp "$GCS/sets/$(basename "$VAL_NPZ")" "$VAL_NPZ" || { echo "C8X-VALSET-MISSING $GCS/sets/$(basename "$VAL_NPZ")"; exit 2; }
echo "C8X-VALSET-OK $(basename "$VAL_NPZ")"

# ---------- lane 0: the pre-staged markers (under $GCS only) and the resume point ----------
gsutil -q stat "$GCS/C8_EXTENDED" 2>/dev/null || echo "EXTENDED from $EXT_FROM to $EXT_TO (the PI's counterfactual 2026-09-14, not the registered rule, which missed by one monitor puzzle) $(date -u +%FT%TZ)" | gsutil -q cp - "$GCS/C8_EXTENDED"
gsutil -q stat "$GCS/SYNC-AB" 2>/dev/null || echo "skipped: the C8 extension carries no sync rider" | gsutil -q cp - "$GCS/SYNC-AB"
if ! gsutil -q stat "$GCS/jax_cache.tgz" 2>/dev/null && [ ! -d jax_cache ]; then
  gsutil -q cp "$SRC/jax_cache.tgz" /tmp/c8x_jc_src.tgz 2>/dev/null && tar xzf /tmp/c8x_jc_src.tgz 2>/dev/null && echo "COMPILE-CACHE seeded from $SRC (read-only)"
fi
GCS="$GCS" R_TAG=champ ARMS="C8" LIVE_PREFIX="$LIVE_PREFIX" bash tools/live_bank.sh restore
if gsutil -q stat "$GCS/C8_PRETRAIN_OK" 2>/dev/null; then
  echo "C8X-PRETRAIN-DONE (the extended grids are banked under $GCS; the $SRC 30k state is not restored)"
else
  if [ ! -f "$D/ckpt_latest.pkl" ]; then
    gsutil -q cp "$SRC/C8_pretrain.tgz" /tmp/c8x_c8_src.tgz 2>/dev/null && tar xzf /tmp/c8x_c8_src.tgz 2>/dev/null \
      && echo "C8X-RESTORE-SRC C8's banked state from $SRC/C8_pretrain.tgz (read-only)"
  fi
  st=$(ck_step "$D/ckpt_latest.pkl")
  if [ -z "$st" ] || [ "$st" -lt "$EXT_FROM" ]; then
    echo "C8X-NO-RESUME-STATE (ckpt_latest step ${st:-none} < $EXT_FROM: the extension never trains from scratch) $(date -u +%FT%TZ)"; exit 2
  fi
  echo "C8X-RESUME-POINT step $st (the extension $EXT_FROM -> $EXT_TO)"
fi

# ---------- lane 1: C8 through the champion chain (the extension path, the registered selection, the battery) ----------
echo "C8X-LANE1 C8 via tools/chain_champ.sh $(date -u +%FT%TZ)"
SELF_TEARDOWN=0 CHAMP_ALL_ARMS="C8" CHAMP_ARMS_1X8="C8" CHAMP_EXTRA_ARMS="C8" SENT=CHAMPC8X C1_WAIT_PASSES=1 C1_POLL_SLEEP=5 \
  bash tools/chain_champ.sh; rc1=$?
echo "C8X-LANE1-END rc=$rc1 $(date -u +%FT%TZ)"
orph=$(pgrep -fa 'tools/pretrain[.]py|tools/eval_sudoku_extreme[.]py' 2>/dev/null | cut -c1-160)
[ -z "$orph" ] || echo "C8X-ORPHANS lane 1 left trainer/evaluator processes alive (a kill is the PI's call): $(echo "$orph" | tr '\n' ';' | cut -c1-400)"
gsutil -q stat "$GCS/C8_ARM_OK" 2>/dev/null || { echo "C8X-INCOMPLETE (lane 1: no C8_ARM_OK; rc=$rc1) $(date -u +%FT%TZ)"; exit 1; }

R_TAG=champ ARMS="C8" LIVE_EVERY=${LIVE_EVERY:-300} nohup bash tools/live_bank.sh loop > runs/live_bank_c8x.log 2>&1 &
LBP=$!; trap 'kill "$LBP" 2>/dev/null' EXIT

# ---------- lane 2: the low-noise selection curves on the held-out 10k ----------
grids_dir () {  # ARM -> the directory holding the arm's banked grids (C8: local, re-pulled from $GCS; C5/C7: read-only pulls outside runs/)
  local arm=$1 P
  if [ "$arm" = C8 ]; then
    if [ ! -f "$D/ckpt_$(printf '%06d' "$EXT_TO").pkl" ]; then
      gsutil -q cp "$GCS/C8_pretrain.tgz" /tmp/c8x_c8_ext.tgz 2>/dev/null && tar xzf /tmp/c8x_c8_ext.tgz 2>/dev/null && echo "C8X-GRIDS-REPULL C8 from $GCS" >&2
    fi
    [ -f "$D/ckpt_$(printf '%06d' "$EXT_TO").pkl" ] || return 1
    echo "$D"
  else
    P=/tmp/c8x_src/$arm
    if [ -z "$(ls "$P/runs/pretrainchamp_$arm"/ckpt_0*.pkl 2>/dev/null)" ]; then
      mkdir -p "$P" && gsutil -q cp "$SRC/${arm}_pretrain.tgz" "/tmp/c8x_src/${arm}_pretrain.tgz" 2>/dev/null && tar xzf "/tmp/c8x_src/${arm}_pretrain.tgz" -C "$P" 2>/dev/null
    fi
    [ -n "$(ls "$P/runs/pretrainchamp_$arm"/ckpt_0*.pkl 2>/dev/null)" ] || return 1
    echo "$P/runs/pretrainchamp_$arm"
  fi
}
val_steps () {  # DIR -> the banked grid steps >= VAL_MIN_STEP (six digits)
  local ck s; for ck in $(ls "$1"/ckpt_0*.pkl 2>/dev/null | sort); do s=$(step_of "$ck"); [ "$((10#$s))" -ge "$VAL_MIN_STEP" ] && echo "$s"; done
}
val_finish () {  # ARM STEP -> n-gate (n, split), bank, mark
  local name=${1}_s$2 O=runs/c8x_val_p$1_s$2
  [ -f "$O/summary_all.json" ] || JAX_PLATFORMS=cpu $PY tools/eval_sudoku_extreme.py --merge "$O" >> "$O/run.log" 2>&1
  $RPY -c "import json,sys; s=json.load(open('$O/summary_all.json')); sys.exit(0 if s['n']==$VAL_N and s.get('split')=='val' and s.get('ema') else 1)" 2>/dev/null \
    || { echo "C8X-VAL-N-BAD $name"; return 1; }
  tar czf "/tmp/c8x_val_$name.tgz" "$O" && gsutil -q cp "/tmp/c8x_val_$name.tgz" "$GCS/val/$name.tgz" && echo ok | gsutil -q cp - "$GCS/val/${name}_OK" \
    && echo "C8X-VAL-OK $name $(date -u +%H:%M)"
}
val_pass () {
  local arm G s jobs=() i c pids grp p
  for arm in $VAL_ARMS; do
    G=$(grids_dir "$arm") || { echo "C8X-VAL-NO-GRIDS $arm"; continue; }
    for s in $(val_steps "$G"); do gsutil -q stat "$GCS/val/${arm}_s${s}_OK" 2>/dev/null || jobs+=("$arm $s $G/ckpt_$s.pkl"); done
  done
  echo "C8X-VAL-PASS ${#jobs[@]} grids to evaluate $(date -u +%H:%M)"
  i=0
  while [ "$i" -lt "${#jobs[@]}" ]; do
    pids=(); grp=()
    for c in $(seq 0 $((NCHIP - 1))); do
      [ "$i" -lt "${#jobs[@]}" ] || break
      set -- ${jobs[$i]}; rm -rf "runs/c8x_val_p$1_s$2"; mkdir -p "runs/c8x_val_p$1_s$2"
      pin "$c" $PY tools/eval_sudoku_extreme.py --ckpt "$3" --npz "$VAL_NPZ" --out "runs/c8x_val_p$1_s$2" --split val --t-total 16 --ema \
        > "runs/c8x_val_p$1_s$2/run.log" 2>&1 & pids+=($!); grp+=("$1 $2"); i=$((i + 1))
    done
    for p in "${pids[@]}"; do wait "$p"; done
    for p in "${grp[@]}"; do set -- $p; val_finish "$1" "$2" || true; done
  done
}
echo "C8X-LANE2 val curves arms=[$VAL_ARMS] n=$VAL_N min_step=$VAL_MIN_STEP $(date -u +%FT%TZ)"
val_pass; val_pass   # the second pass re-runs only a grid whose first attempt did not bank
for m in $(gsutil ls "$GCS/val/*_OK" 2>/dev/null); do   # after a node change: the banked rows back on disk for the curve
  n=$(basename "$m" _OK); [ -f "runs/c8x_val_p${n%_s*}_s${n##*_s}/summary_all.json" ] && continue
  gsutil -q cp "$GCS/val/$n.tgz" "/tmp/c8x_val_$n.tgz" 2>/dev/null && tar xzf "/tmp/c8x_val_$n.tgz" 2>/dev/null
done
gval=$($RPY tools/c8x_valcurve.py --root runs --arms $VAL_ARMS --select C8 --n "$VAL_N" --min-step "$VAL_MIN_STEP" --out runs/analysis/c8x_valcurve.json) \
  || { echo "C8X-INCOMPLETE (no val curve for C8) $(date -u +%FT%TZ)"; exit 1; }
vs=runs/sxeval_pchampC8/full_vsel_t16/summary_all.json
if [ ! -f "$vs" ]; then gsutil -q cp "$GCS/evals/full_C8_vsel_t16.tgz" /tmp/c8x_vsel.tgz 2>/dev/null && tar xzf /tmp/c8x_vsel.tgz 2>/dev/null; fi
gmon=$($RPY -c "import json,re; print(re.search(r'ckpt_(\d+)', json.load(open('$vs'))['ckpt']).group(1))" 2>/dev/null) \
  || { echo "C8X-INCOMPLETE (no g_mon provenance) $(date -u +%FT%TZ)"; exit 1; }
echo "C8X-PICKS g_mon=$gmon g_val=$gval hyp=$HYP paper-final=$PF_GRID"

# ---------- lane 3: the filler rows for g_mon, then the extra test rows ----------
echo "C8X-FILLER arms=[C8] jobs=[d64full d128sub d256sub] $(date -u +%FT%TZ)"
W=0 MY_ARMS="C8" FILLER_ARMS="C8" JOBS="d64full d128sub d256sub" FILLER_ROOT=$PWD GCS=$GCS R_TAG=champ \
  PASS_SLEEP=${PASS_SLEEP:-60} PASSES=${X_PASSES:-4} IDLE_POLL=${IDLE_POLL:-60} bash tools/filler_full.sh 2>&1 | tee -a runs/filler_full_c8x.log
XROWS=""
case " $gmon $PF_GRID " in *" $gval "*) ;; *) XROWS="d16full_s$gval d64full_s$gval";; esac
case " $gmon $gval $PF_GRID " in *" $HYP "*) ;; *) XROWS="$XROWS d16full_s$HYP d64sub100k_s$HYP";; esac
echo "C8X-XROWS [${XROWS# }]"
xrow () {  # NAME -> sharded over the chips (partials every 300 s), merged, n-gated, banked $GCS/xrows/NAME.tgz + NAME_OK
  local name=$1 kind=${1%_s*} st=${1##*_s} O CK NG rc p pids=() i
  O=runs/c8x_xrow_pchampC8_$name; CK=$D/ckpt_$st.pkl
  gsutil -q stat "$GCS/xrows/${name}_OK" 2>/dev/null && { echo "C8X-XROW-SKIP $name"; return 0; }
  [ -f "$CK" ] || { echo "C8X-XROW-NO-GRID $name ($CK)"; return 1; }
  case $kind in
    d16full) NG=422786; set -- --split test --t-total 16 --ema --record-by-step;;
    d64full) NG=422786; set -- --split test --t-total 64 --ema --record-by-step;;
    d64sub100k) NG=100000; set -- --split test --subsample 100000 --t-total 64 --ema --record-by-step;;
    *) echo "C8X-XROW-BAD $name"; return 1;;
  esac
  mkdir -p "$O"; echo "C8X-XROW-START $name ck=$CK $(date -u +%H:%M)"
  for i in $(seq 0 $((NCHIP - 1))); do
    pin "$i" $PY tools/eval_sudoku_extreme.py --ckpt "$CK" --npz "$NPZ" --out "$O" --shard "$i/$NCHIP" --bank-every 300 "$@" > "$O/shard_$i.log" 2>&1 & pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || { echo "C8X-XROW-FAILED $name (a shard failed; partials stay for the resume)"; return 1; }
  JAX_PLATFORMS=cpu $PY tools/eval_sudoku_extreme.py --merge "$O" > "$O/merge.log" 2>&1
  $RPY -c "import json,sys; s=json.load(open('$O/summary_all.json')); sys.exit(0 if s['n']==$NG else 1)" 2>/dev/null || { echo "C8X-XROW-N-BAD $name"; return 1; }
  tar czf "/tmp/c8x_xrow_$name.tgz" "$O" && gsutil -q cp "/tmp/c8x_xrow_$name.tgz" "$GCS/xrows/$name.tgz" && echo ok | gsutil -q cp - "$GCS/xrows/${name}_OK" \
    && echo "C8X-XROW-OK $name $(date -u +%H:%M)"
}
for pass in 1 2; do for r in $XROWS; do xrow "$r" || true; done; done

# ---------- completion ----------
need=""
for m in C8_ARM_OK filler/d64full_C8_OK filler/d128sub_C8_OK filler/d256sub_C8_OK; do gsutil -q stat "$GCS/$m" 2>/dev/null || need="$need $m"; done
for arm in $VAL_ARMS; do
  G=$(grids_dir "$arm" 2>/dev/null) || { need="$need val/${arm}_grids"; continue; }
  for s in $(val_steps "$G"); do gsutil -q stat "$GCS/val/${arm}_s${s}_OK" 2>/dev/null || need="$need val/${arm}_s$s"; done
done
for r in $XROWS; do gsutil -q stat "$GCS/xrows/${r}_OK" 2>/dev/null || need="$need xrows/$r"; done
[ -z "$need" ] || { echo "C8X-INCOMPLETE worker=$W (missing:$need) $(date -u +%FT%TZ)"; exit 1; }
files=""
for p in $D/metrics.jsonl $D/val_best.txt $D/EXTENDED.txt $D/config.json $D/resumes.txt runs/preflightchamp_C8.log \
         runs/sxeval_pchampC8/*/summary_all.json runs/sxscan_pchampC8/summary_all.json runs/filler_sxeval_pchampC8_*/summary_all.json \
         runs/c8x_val_p*/summary_all.json runs/c8x_xrow_pchampC8_*/summary_all.json runs/analysis/c8x_valcurve.json runs/filler_full_c8x.log; do
  [ -e "$p" ] && files="$files $p"
done
# shellcheck disable=SC2086
tar czf "/tmp/$FINAL" $files 2>/dev/null && gsutil -q cp "/tmp/$FINAL" "$GCS/$FINAL" && echo "FINAL-BANKED $GCS/$FINAL ($(echo $files | wc -w | tr -d ' ') files)"
echo "$SENT-COMPLETE worker=$W $(date -u +%FT%TZ)"
