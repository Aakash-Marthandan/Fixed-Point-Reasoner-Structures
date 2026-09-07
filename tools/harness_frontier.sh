#!/bin/bash
# Ledger: FRONTIER POD RUN offline stub harness (2026-09-07; the house law: no chain launches without an end-to-end offline
# pass; every negative scenario asserts the staged failure fired). Stubs: gsutil/gcloud/timeout/py-spy -> a local FAKE_GCS dir;
# $CHAIN_PY -> a stub emulating eval_sudoku_extreme.py (records_s*/summary_s* per shard with the flags recorded; --merge sums n),
# explosion_census.py, stall_calibration.py; REAL python runs the chain's own helpers (the n-gate). Scenarios:
#   S1 fresh -> COMPLETE: every job's marker present (3 models x battery + EqR extras + trmpub60k + the DEC scans A3 A7 A5 A4 A8 +
#      D128 rows); the DEC canary at --batch 128 with no extra flag -> RECIPE-DEC=BATCHONLY; the flag registry reaches the
#      evaluator (record-by-step + record-q on the D16 fulls; k128 on the 20k scans; z0-mode perturb with the 5 eps; zero-prefix;
#      the fp32 control at highest precision; EqR's trunc reset + seg-noise .5 / .01 + the train-1k npz; --ema everywhere but the
#      raw row); the final tarball banked.
#   S2 idempotent rerun -> every job EVAL-SKIP / CENSUS-SKIP / CALIB-SKIP, no second tarball write needed, COMPLETE.
#   S3 the DEC canary STALLS at batch-only (stub hangs) -> EVAL-STALLED + the stall tarball + the --z0-device variant completes ->
#      RECIPE-DEC=--z0-device and every later DEC scan runs with --z0-device (never batch-only again).
#   S3b every variant stalls -> SCAN-DEADLOCK marker for A3, the chain proceeds to the frontier rows, INCOMPLETE at the end, and
#      a RERUN skips the deadlocked arm by its marker.
#   S4 a shard fails (rc 1) on one frontier eval -> EVAL-SHARD-FAILED, no marker, the rest of the battery runs, INCOMPLETE; the
#      rerun redoes ONLY that eval.
#   S5 a missing checkpoint -> CKPT-MISSING, that model's jobs skipped, INCOMPLETE.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export REAL_PY="$REPO/.venv/bin/python3"
PASS=0; FAIL=0
ok () { PASS=$((PASS+1)); echo "  PASS  $1"; }
bad () { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }

mk_sandbox () {
  SB=$(mktemp -d /tmp/hfr.XXXXXX)
  mkdir -p "$SB/repo/tools" "$SB/repo/runs" "$SB/repo/data/sudoku_extreme" "$SB/gcs/frontier/ckpts" "$SB/gcs/frontier/sets" "$SB/gcs/sport2" "$SB/bin"
  cp "$REPO/tools/chain_frontier.sh" "$REPO/tools/live_bank.sh" "$SB/repo/tools/"
  : > "$SB/repo/data/sudoku_extreme/sudoku_extreme_seed0.npz"
  # the checkpoints in the fake bucket (loadable pickles)
  for n in trmpub cgar eqr trmpub60k A3 A7 A5 A4 A8; do
    "$REAL_PY" -c "import pickle,sys; pickle.dump({'state':{},'config':{},'step':0}, open(sys.argv[1],'wb'))" "$SB/gcs/frontier/ckpts/$n.pkl"
  done
  : > "$SB/gcs/frontier/sets/train_eqr_sx.npz"
  cat > "$SB/bin/gsutil" <<SH
#!/bin/bash
GB="$SB/gcs"
SH
  cat >> "$SB/bin/gsutil" <<'SH'
map () { echo "$1" | sed "s|gs://qhrrn2-rescue/|$GB/|"; }
args=(); for a in "$@"; do [ "$a" = "-q" ] || [ "$a" = "-m" ] || args+=("$a"); done
cmd=${args[0]:-}
case $cmd in
  stat) p=$(map "${args[1]}"); [ -f "$p" ];;
  ls)   rc=1; for g in "${args[@]:1}"; do p=$(map "$g"); for f in $p; do [ -e "$f" ] && { echo "$f" | sed "s|$GB/|gs://qhrrn2-rescue/|"; rc=0; }; done; done; exit $rc;;
  cp)   src=${args[1]}; dst=${args[2]}
        if [ "$src" = "-" ]; then p=$(map "$dst"); mkdir -p "$(dirname "$p")"; cat > "$p"
        elif [[ "$src" == gs://* ]]; then p=$(map "$src"); [ -f "$p" ] || exit 1; if [ "$dst" = "-" ]; then cat "$p"; else mkdir -p "$(dirname "$dst")" 2>/dev/null; cp "$p" "$dst"; fi
        else p=$(map "$dst"); mkdir -p "$(dirname "$p")"; cp "$src" "$p"; fi;;
  rsync) x=""; pos=(); i=1
         while [ $i -lt ${#args[@]} ]; do a=${args[$i]}
           case $a in -x) i=$((i+1)); x=${args[$i]};; -r|-C|-n|-d|-c) ;; *) pos+=("$a");; esac; i=$((i+1)); done
         src=$(map "${pos[0]}"); dst=$(map "${pos[1]}")
         [ -d "$src" ] || exit 1
         case "${pos[1]}" in gs://*) : ;; *) [ -d "$dst" ] || { echo "CommandException: arg ($dst) does not name a directory, bucket, or bucket subdir." >&2; exit 1; };; esac
         ( cd "$src" && find . -type f | sed 's|^\./||' ) | while read -r rel; do
           if [ -n "$x" ] && printf '%s\n' "$rel" | grep -qE "$x"; then continue; fi
           s="$src/$rel"; d="$dst/$rel"
           if [ ! -f "$d" ] || ! cmp -s "$s" "$d"; then mkdir -p "$(dirname "$d")"; cp "$s" "$d"; echo "Copying file://$rel"; fi
         done; exit 0;;
  *) exit 0;;
esac
SH
  chmod +x "$SB/bin/gsutil"
  printf '#!/bin/bash\necho "gcloud-stub $*"; exit 0\n' > "$SB/bin/gcloud"; chmod +x "$SB/bin/gcloud"
  printf '#!/bin/bash\nshift\nexec "$@"\n' > "$SB/bin/timeout"; chmod +x "$SB/bin/timeout"
  printf '#!/bin/bash\necho "py-spy-stub dump $*"; exit 0\n' > "$SB/bin/py-spy"; chmod +x "$SB/bin/py-spy"
  cat > "$SB/bin/stubpy" <<'PYEOF'
#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
import numpy as np
argv = sys.argv[1:]
tool = argv[0] if argv else ""
def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

if tool.endswith("eval_sudoku_extreme.py"):
    if "--merge" in argv:
        d = Path(flag("--merge")); recs = sorted(d.glob("records_s*.npz"))
        n = sum(int(np.load(p)["n"]) for p in recs) if recs else 0
        s0 = json.loads(sorted(d.glob("summary_s*.json"))[0].read_text()) if recs else {}
        (d / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": .3, **{k: v for k, v in s0.items() if k != "n"}}))
        np.savez(d / "records_all.npz", n=np.asarray(n)); sys.exit(0)
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    name = out.name; ck = Path(flag("--ckpt")).stem
    # staged failures
    if ck in os.environ.get("STUB_STALL_ARMS", "").split(",") and "--k-init" in argv and int(flag("--k-init", "0")) > 0:
        # the DEC multi-draw scan deadlock: hang unless the recipe carries the flag that "fixes" it (STUB_STALL_FIX; "NONE" = every variant hangs)
        fix = os.environ.get("STUB_STALL_FIX", "--z0-device")
        if fix == "NONE" or fix not in argv:
            (out / f"shard_{flag('--shard', '0/1').split('/')[0]}.log").write_text("step 1 in 3s (incl. compile)\n")
            time.sleep(int(os.environ.get("STUB_STALL_SLEEP", "120"))); sys.exit(0)
    if os.environ.get("STUB_FAIL_EVAL", "") and os.environ["STUB_FAIL_EVAL"] in str(out) and not (out / "recovered").exists():
        print("staged shard failure", file=sys.stderr); sys.exit(1)
    shard = flag("--shard"); sub = flag("--subsample"); strat = flag("--stratified"); lim = flag("--limit")
    n_total = int(sub) if sub else (int(strat) if strat else (1000 if "train_eqr" in str(flag("--npz", "")) else 422786))
    prov = {"t_total": int(flag("--t-total", "64")), "ema": "--ema" in argv, "k_init": int(flag("--k-init", "0")), "argv": argv,
            "prec": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", ""), "z0_mode": flag("--z0-mode", "gauss"), "z0_eps": flag("--z0-eps"),
            "seg_noise_beta": flag("--seg-noise-beta"), "zero_prefix": "--zero-prefix" in argv, "rbs": "--record-by-step" in argv,
            "rq": "--record-q" in argv, "batch": flag("--batch"), "z0_device": "--z0-device" in argv, "npz": flag("--npz")}
    if shard:
        i, K = map(int, shard.split("/")); n = n_total // K + (1 if i < n_total % K else 0)
        np.savez(out / f"records_s{i}.npz", n=np.asarray(n)); (out / f"summary_s{i}.json").write_text(json.dumps({"n": n, **prov}))
    else:
        np.savez(out / "records_all.npz", n=np.asarray(n_total)); (out / "summary_all.json").write_text(json.dumps({"n": n_total, **prov}))
    sys.exit(0)

if tool.endswith("stall_calibration.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "calib.json").write_text(json.dumps({"ckpt": flag("--ckpt"), "ema": "--ema" in argv})); sys.exit(0)

if tool.endswith("explosion_census.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "census.json").write_text(json.dumps({"ckpt": flag("--ckpt"), "ema": "--ema" in argv})); sys.exit(0)
sys.exit(0)
PYEOF
  chmod +x "$SB/bin/stubpy"
  echo "  (sandbox $SB)"
}

run_chain () {  # [extra VAR=val...]
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" NCHIP_OVERRIDE=4 LIVE_NO_GUARD=1 \
     PYSPY_INSTALL=true WATCH_POLL=1 STALL_SEC=3 "$@" bash tools/chain_frontier.sh > "$SB/w.log" 2>&1); echo $? > "$SB/rc"
}
summ () { "$REAL_PY" -c "import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps({k: s.get(k) for k in sys.argv[2:]}))" "$@"; }
mark () { [ -f "$SB/gcs/frontier/evals/${1}_OK" ]; }

# ---------- S1 fresh -> COMPLETE ----------
echo "S1 fresh"; mk_sandbox; run_chain
if grep -q "CHAIN-FRONTIER-COMPLETE" "$SB/w.log" && [ "$(cat "$SB/rc")" = 0 ]; then ok "S1 COMPLETE rc 0"; else bad "S1 completion"; tail -5 "$SB/w.log"; fi
n_ok=$(ls "$SB/gcs/frontier/evals/" | grep -c "_OK$"); [ "$n_ok" -ge 60 ] && ok "S1 markers: $n_ok evals banked" || bad "S1 markers only $n_ok"
for m in trmpub eqr cgar; do
  for j in full_${m}_vsel_t16 full_${m}_vsel_t64 scan_${m} screen_${m}_vb d128_${m} d256_${m} census_${m}_vsel calib_${m}_vsel initrad_${m}_e0.03 initrad_${m}_e3 prefix0_${m} fp32_${m}; do mark "$j" || bad "S1 missing $j"; done
done
for j in full_eqr_raw_t16 scan_eqr_trunc noise05_eqr_t16 noise05_eqr_t64 noise05_eqr_scan noise001_eqr_t16 train1k_eqr full_trmpub60k_vsel_t16 sub20k_trmpub60k_t64 scan_A3 scan_A7 scan_A5 scan_A4 scan_A8 d128_A3 d128_A7 d128_A5; do mark "$j" || bad "S1 missing $j"; done
ok "S1 every named job marked (or reported above)"
[ "$(cat "$SB/gcs/frontier/RECIPE-DEC")" = "BATCHONLY" ] && ok "S1 RECIPE-DEC=BATCHONLY (the canary passed at --batch 128, no extra flag)" || bad "S1 recipe: $(cat "$SB/gcs/frontier/RECIPE-DEC" 2>/dev/null)"
s=$(summ "$SB/repo/runs/sxeval_pfrontiertrmpub/full_vsel_t16/summary_all.json" n t_total ema rbs rq prec); echo "$s" | grep -q '"n": 422786, "t_total": 16, "ema": true, "rbs": true, "rq": true, "prec": "default"' && ok "S1 full D16: n 422786, EMA, exact-by-step + q, bf16" || bad "S1 full D16 flags: $s"
s=$(summ "$SB/repo/runs/sxscan_pfrontiereqr/summary_all.json" n t_total k_init z0_mode); echo "$s" | grep -q '"n": 20000, "t_total": 64, "k_init": 128, "z0_mode": "gauss"' && ok "S1 scan: 20k x k128 t64 gauss (the column)" || bad "S1 scan flags: $s"
s=$(summ "$SB/repo/runs/sxscan_pfrontiereqr_trunc/summary_all.json" k_init z0_mode); echo "$s" | grep -q '"k_init": 32, "z0_mode": "trunc"' && ok "S1 EqR trunc reset scan k32" || bad "S1 trunc: $s"
s=$(summ "$SB/repo/runs/sxeval_pfrontiereqr/sub20k_t64_noise05/summary_all.json" seg_noise_beta n); echo "$s" | grep -q '"seg_noise_beta": "0.5", "n": 20000' && ok "S1 EqR Langevin .5 row" || bad "S1 noise: $s"
s=$(summ "$SB/repo/runs/sxeval_pfrontiereqr/full_raw_t16/summary_all.json" ema n); echo "$s" | grep -q '"ema": false, "n": 422786' && ok "S1 EqR raw alt row without --ema" || bad "S1 raw: $s"
s=$(summ "$SB/repo/runs/sxeval_pfrontiercgar/initrad_e0.3/summary_all.json" z0_mode z0_eps k_init n); echo "$s" | grep -q '"z0_mode": "perturb", "z0_eps": "0.3", "k_init": 1, "n": 512' && ok "S1 init radius eps .3 on strat-512" || bad "S1 initrad: $s"
s=$(summ "$SB/repo/runs/sxeval_pfrontiercgar/sub20k_t16_fp32/summary_all.json" prec); echo "$s" | grep -q '"prec": "highest"' && ok "S1 fp32 control at highest" || bad "S1 fp32: $s"
s=$(summ "$SB/repo/runs/sxeval_pfrontiertrmpub/sub20k_t16_prefix0/summary_all.json" zero_prefix); echo "$s" | grep -q '"zero_prefix": true' && ok "S1 prefix zeroed" || bad "S1 prefix: $s"
s=$(summ "$SB/repo/runs/sxeval_pfrontiereqr/train1k_t16/summary_all.json" n npz); echo "$s" | grep -q '"n": 1000' && echo "$s" | grep -q train_eqr_sx && ok "S1 EqR train-1k on its own npz" || bad "S1 train1k: $s"
s=$(summ "$SB/repo/runs/sxscan_pfinalAA7/summary_all.json" n k_init batch z0_device); echo "$s" | grep -q '"n": 5000, "k_init": 32, "batch": "128", "z0_device": false' && ok "S1 DEC scan A7: 5k x k32 at batch 128, host draws (the canary recipe)" || bad "S1 DEC scan: $s"
[ -f "$SB/gcs/frontier/frontier_final.tgz" ] && ok "S1 final tarball banked" || bad "S1 tarball"
grep -q "SELF-TEARDOWN" "$SB/w.log" && bad "S1 teardown fired without SELF_TEARDOWN=1" || ok "S1 no teardown without the env"

# ---------- S2 idempotent rerun ----------
echo "S2 rerun"; run_chain
n_skip=$(grep -c -E "EVAL-SKIP|CENSUS-SKIP|CALIB-SKIP" "$SB/w.log"); n_run=$(grep -c "EVAL-OK" "$SB/w.log")
[ "$n_run" -eq 0 ] && [ "$n_skip" -ge 60 ] && grep -q "CHAIN-FRONTIER-COMPLETE" "$SB/w.log" && ok "S2 rerun: $n_skip skips, 0 re-runs, COMPLETE" || bad "S2 rerun skips=$n_skip runs=$n_run"

# ---------- S3 the DEC canary stalls at batch-only; --z0-device completes ----------
echo "S3 canary stall -> z0-device"; mk_sandbox; run_chain STUB_STALL_ARMS=A3,A7,A5,A4,A8 STUB_STALL_FIX=--z0-device STUB_STALL_SLEEP=8 STALL_SEC=3 WATCH_POLL=2
grep -q "EVAL-STALLED scan_A3" "$SB/w.log" && ok "S3 the canary STALLED at batch-only (watchdog fired)" || { bad "S3 no stall"; grep -E "scan_A3|STALL|RECIPE" "$SB/w.log" | head -5; }
ls "$SB/gcs/frontier/evals/" | grep -q "scan_A3_STALL_" && ok "S3 the stall tarball banked" || bad "S3 stall tarball"
[ "$(cat "$SB/gcs/frontier/RECIPE-DEC" 2>/dev/null)" = "--z0-device" ] && ok "S3 RECIPE-DEC=--z0-device" || bad "S3 recipe: $(cat "$SB/gcs/frontier/RECIPE-DEC" 2>/dev/null)"
s=$(summ "$SB/repo/runs/sxscan_pfinalAA8/summary_all.json" z0_device batch); echo "$s" | grep -q '"z0_device": true, "batch": "128"' && ok "S3 every later DEC scan used the canary's recipe (A8 with --z0-device)" || bad "S3 A8 recipe: $s"
[ "$(grep -c "DEC-SCAN-TRY A7" "$SB/w.log")" -eq 1 ] && ok "S3 A7 tried ONCE (no batch-only retry after the recipe)" || bad "S3 A7 tries: $(grep -c "DEC-SCAN-TRY A7" "$SB/w.log")"
grep -q "CHAIN-FRONTIER-COMPLETE" "$SB/w.log" && ok "S3 COMPLETE" || bad "S3 completion"

# ---------- S3b every variant stalls -> SCAN-DEADLOCK, proceed, INCOMPLETE; the rerun skips the arm ----------
echo "S3b deadlock"; mk_sandbox; run_chain STUB_STALL_ARMS=A3 STUB_STALL_FIX=NONE STUB_STALL_SLEEP=8 STALL_SEC=3 WATCH_POLL=2
[ "$(grep -c "EVAL-STALLED scan_A3" "$SB/w.log")" -eq 3 ] && ok "S3b three variants stalled" || bad "S3b stalls: $(grep -c "EVAL-STALLED scan_A3" "$SB/w.log")"
[ -f "$SB/gcs/frontier/A3_SCAN_DEADLOCK" ] && grep -q "SCAN-DEADLOCK A3" "$SB/w.log" && ok "S3b SCAN-DEADLOCK marker, labeled" || bad "S3b marker"
mark full_trmpub_vsel_t16 && mark scan_A7 && ok "S3b the chain proceeded (frontier rows + A7's scan ran)" || bad "S3b proceed"
grep -q "FRONTIER-INCOMPLETE" "$SB/w.log" && [ "$(cat "$SB/rc")" = 1 ] && ok "S3b INCOMPLETE rc 1 (a job failed)" || bad "S3b end state"
[ -z "$(cat "$SB/gcs/frontier/RECIPE-DEC" 2>/dev/null)" ] && bad "S3b recipe must be set by A7's success" || ok "S3b RECIPE-DEC set by the next arm: $(cat "$SB/gcs/frontier/RECIPE-DEC")"
run_chain STUB_STALL_ARMS=A3 STUB_STALL_FIX=NONE STUB_STALL_SLEEP=8 STALL_SEC=3 WATCH_POLL=2
grep -q "SCAN-DEADLOCK-SKIP A3" "$SB/w.log" && [ "$(grep -c "DEC-SCAN-TRY A3" "$SB/w.log")" -eq 0 ] && ok "S3b rerun skips the deadlocked arm by its marker" || bad "S3b rerun"

# ---------- S4 a shard failure on one frontier eval ----------
echo "S4 shard failure"; mk_sandbox; run_chain STUB_FAIL_EVAL=sxscan_pfrontiercgar
grep -q "EVAL-SHARD-FAILED scan_cgar" "$SB/w.log" && ! mark scan_cgar && ok "S4 EVAL-SHARD-FAILED, no marker" || bad "S4 failure path"
mark screen_cgar_vb && mark scan_A7 && ok "S4 the battery continued past the failure" || bad "S4 continue"
grep -q "FRONTIER-INCOMPLETE" "$SB/w.log" && ok "S4 INCOMPLETE" || bad "S4 end"
touch "$SB/repo/runs/sxscan_pfrontiercgar/recovered"; run_chain STUB_FAIL_EVAL=sxscan_pfrontiercgar
[ "$(grep -c "EVAL-OK" "$SB/w.log")" -eq 1 ] && mark scan_cgar && grep -q "CHAIN-FRONTIER-COMPLETE" "$SB/w.log" && ok "S4 rerun redid ONLY the failed eval -> COMPLETE" || bad "S4 rerun: $(grep -c EVAL-OK "$SB/w.log") evals"

# ---------- S5 a missing checkpoint ----------
echo "S5 missing ckpt"; mk_sandbox; rm "$SB/gcs/frontier/ckpts/cgar.pkl"; run_chain
grep -q "CKPT-MISSING cgar" "$SB/w.log" && ! mark full_cgar_vsel_t16 && mark full_eqr_vsel_t16 && grep -q "FRONTIER-INCOMPLETE" "$SB/w.log" && ok "S5 CKPT-MISSING -> that model skipped, the others ran, INCOMPLETE" || bad "S5"

echo; echo "harness: $PASS PASS / $FAIL FAIL"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
