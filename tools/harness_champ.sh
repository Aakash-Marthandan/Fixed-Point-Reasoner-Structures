#!/bin/bash
# Ledger: CHAMPION NIGHT offline stub harness (2026-09-08; the house law: no chain launches without an end-to-end
# offline pass; every negative scenario asserts the staged failure fired; the 2026-09-04 lesson: a harness adapted from
# the previous campaign is unverified until RUN). Stubs: gsutil/gcloud -> a local FAKE_GCS dir; $CHAIN_PY -> a stub
# emulating pretrain.py / eval_sudoku_extreme.py / explosion_census.py / stall_calibration.py; select_ckpt.py runs REAL
# (the extension rule and the earliest-tie selection go through the registered tool); REAL python runs the chain's own
# helpers (nan_check / amputate / n-gates / newest_mtime). Scenarios: S1 fresh 1x8 -> COMPLETE (7 arms, the rider first,
# the DEC-class battery on every arm, the commit row on C6 only); S1r the ARM REGISTRY reaches the trainer; S2 idempotent
# rerun; SE the EXTENSION rule (peak at the end -> +EXT once, resumed; peak early -> none; a relaunch keeps the extended
# budget; the earliest-tie selection); S3 NaN one-shot + NAN-ABORT rc=3; S3d fixed-step screens at the 2k grids of each
# budget; S4 4x4 map (+ the rider on w3) and S4b 8x4; S5 no monitor rows -> LOUD fallback, no extension; S6 banked pretrain
# re-pull; S7 the DEC scan stall ladder (BATCHONLY stalls -> --z0-device completes -> RECIPE-DEC reused) and S7b every
# variant stalls -> SCAN-DEADLOCK labeled, the arm proceeds; S8 preflight failure: treatment arm SKIPPED / seed arm ABORT;
# S9 launch-time OOM -> ONE --remat retry; S10 live-bank restore + RESUMED; S11 the sync rider without A3's grid.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export REAL_PY="$REPO/.venv/bin/python3"
PASS=0; FAIL=0
ok () { PASS=$((PASS+1)); echo "  PASS  $1"; }
bad () { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }

mk_sandbox () {
  SB=$(mktemp -d /tmp/hch.XXXXXX)
  mkdir -p "$SB/repo/tools" "$SB/repo/runs" "$SB/repo/data/sudoku_extreme" "$SB/gcs/champ/sets" "$SB/gcs/frontier/ckpts" "$SB/bin"
  cp "$REPO/tools/chain_champ.sh" "$REPO/tools/live_bank.sh" "$REPO/tools/select_ckpt.py" "$SB/repo/tools/"
  : > "$SB/gcs/champ/sets/sudoku_extreme_seed0_mon512.npz"
  [ "${NO_A3:-0}" = 1 ] || echo a3 > "$SB/gcs/frontier/ckpts/A3.pkl"
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
        elif [[ "$src" == gs://* ]]; then p=$(map "$src"); if [ "$dst" = "-" ]; then cat "$p" 2>/dev/null || exit 1; else [ -f "$p" ] && { mkdir -p "$(dirname "$dst")" 2>/dev/null; cp "$p" "$dst"; } || exit 1; fi
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
  cat > "$SB/bin/stubpy" <<'PYEOF'
#!/usr/bin/env python3
import json, os, pickle, sys, time
from pathlib import Path
import numpy as np
argv = sys.argv[1:]
tool = argv[0] if argv else ""
def flag(name, default=None):
    return argv[len(argv) - 1 - argv[::-1].index(name) + 1] if name in argv else default   # the LAST occurrence wins (argparse)
if tool.endswith("select_ckpt.py"):
    os.execv(os.environ["REAL_PY"], [os.environ["REAL_PY"], os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "..", "repo", "tools", "select_ckpt.py")] + argv[1:])
if tool.endswith("pretrain.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    steps = int(flag("--steps", "100"))
    base = out.name.replace("pretrainchamp_", "").replace("preflightchamp_", "")
    if "preflight" in out.name and base == os.environ.get("STUB_PREFLIGHT_FAIL", ""):
        print("XLA compile error: staged preflight failure", file=sys.stderr); sys.exit(1)
    latest = out / "ckpt_latest.pkl"; start = 0
    if latest.exists():
        try:
            prev = pickle.load(open(latest, "rb")); start = int(prev["step"]); print(f"RESUMED from {latest} at step {start}", flush=True)
        except Exception as e:
            print(f"ckpt_latest unloadable: {e}", file=sys.stderr); sys.exit(1)
    if base == os.environ.get("STUB_OOM_ARM", "") and ("--remat" not in argv or os.environ.get("STUB_OOM_ALWAYS") == "1"):
        print("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 17179869184 bytes.", file=sys.stderr); sys.exit(1)
    arm_run = out.name.startswith("pretrainchamp_")
    is_nan = arm_run and base == os.environ.get("STUB_NAN_ARM", "")
    is_abort = arm_run and base == os.environ.get("STUB_NANABORT_ARM", "")
    peak_last = base in os.environ.get("STUB_PEAK_LAST_ARM", "").split(",")
    no_mon = base in os.environ.get("STUB_NO_MONITOR_ARM", "").split(",")
    rows = []
    mon_steps = sorted({s for s in range(2000, steps + 1, 2000)} | {steps // 2, steps})
    for s in mon_steps:
        if s <= start: continue
        loss = float("nan") if ((is_nan or is_abort) and s == steps) else 0.5
        rows.append(json.dumps({"step": s, "loss": loss, "ce_in": .04, "I_total": 1e5, "A_total": 5.0, "rule_H": 0.0, "lr": 1e-4, "steps_per_sec": 99.0, "t": "T"}))
        if not no_mon:
            v = (0.30 + 0.0001 * s) if peak_last else (0.30 if s <= steps // 2 else 0.29)   # peak_last: rising to the end; else a tie over the first half
            mon = {"step": s, "val_t16": v - 0.01, "val_t16_ema": v, "n_val": 512, "eta": 1.0, "eta_z": 1.0}
            rows.append(json.dumps({"monitor": mon}))
    with open(out / "metrics.jsonl", "a") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    poisoned = is_nan or is_abort
    def grid(step):
        bad = poisoned and step > steps // 2
        w = np.full(2, np.nan, np.float32) if bad else np.ones(2, np.float32)
        return {"state": {"model": {"w": w}}, "step": step, "config": {}}
    for st in mon_steps:
        if st <= start: continue
        pickle.dump(grid(st), open(out / f"ckpt_{st:06d}.pkl", "wb"))
    pickle.dump(grid(steps), open(out / "ckpt_latest.pkl", "wb"))
    (out / "config.json").write_text(json.dumps({"stub": True, "argv": argv}))
    if is_abort:
        (out / "NAN_ABORT.txt").write_text(f"NAN-ABORT at step {steps}\n"); sys.exit(3)
    sys.exit(1 if is_nan else 0)
if tool.endswith("eval_sudoku_extreme.py"):
    if "--merge" in argv:
        d = Path(flag("--merge"))
        recs = sorted(d.glob("records_s*.npz"))
        n = sum(int(np.load(p)["n"]) for p in recs) if recs else 0
        (d / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": .3, "exact_acc_vote": .8, "b1_exact": .4, "wall_s": 5.0}))
        np.savez(d / "records_all.npz", n=np.asarray(n)); sys.exit(0)
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    arm = out.name.replace("sxscan_pchamp", "")
    if flag("--k-init") == "32" and arm == os.environ.get("STUB_STALL_ARM", "") and ("--z0-device" not in argv or os.environ.get("STUB_STALL_ALWAYS") == "1"):
        time.sleep(1000)   # the staged deadlock: no log line, no partial
    shard = flag("--shard"); sub = flag("--subsample"); strat = flag("--stratified")
    n_total = int(sub) if sub else (int(strat) if strat else 422786)
    prov = {"t_total": int(flag("--t-total", "64")), "ema": "--ema" in argv, "argv": argv, "ckpt": flag("--ckpt")}
    if shard:
        i, K = map(int, shard.split("/")); n = n_total // K + (1 if i < n_total % K else 0)
        np.savez(out / f"records_s{i}.npz", n=np.asarray(n)); (out / f"summary_s{i}.json").write_text(json.dumps({"n": n, **prov}))
    else:
        np.savez(out / "records_all.npz", n=np.asarray(n_total))
        (out / "summary_all.json").write_text(json.dumps({"n": n_total, "exact_acc": .3, "exact_acc_vote": .85, "b1_exact": .4, "wall_s": (7.0 if "--sync-per-step" in argv else 3.0),
                                                          "vote_at_k": {"32": .85}, "t1r_at_k": {"32": .5}, **prov}))
    sys.exit(0)
if tool.endswith("stall_calibration.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "calib.json").write_text(json.dumps({"ckpt": flag("--ckpt"), "topk_correct_stalled": 0.8, "n": 512, "ema": "--ema" in argv})); sys.exit(0)
if tool.endswith("explosion_census.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    (out / "census.json").write_text(json.dumps({"rows": [{"t": 64, "exploded_frac": 0.0, "n": 512}, {"t": 256, "exploded_frac": 0.0, "n": 64}], "ema": "--ema" in argv})); sys.exit(0)
sys.exit(0)
PYEOF
  chmod +x "$SB/bin/stubpy"
  echo "  (sandbox $SB)"
}
run_chain () {  # W NW [extra VAR=val...]
  local w=$1 nw=$2; shift 2
  (cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" \
     CHAIN_WORKER=$w CHAIN_WORKERS=$nw NCHIP_OVERRIDE=4 C1_STEPS_X=100 C1_STEPS_LONG=120 C1_EXT_STEPS=20 C1_EXT_WINDOW=10 \
     C1_WAIT_PASSES=3 C1_POLL_SLEEP=1 LIVE_NO_GUARD=1 STALL_SEC=3 WATCH_POLL=1 "$@" bash tools/chain_champ.sh > "$SB/w${w}.log" 2>&1)
}
eargv () { "$REAL_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('argv', [])))" "$1"; }
pargv () { "$REAL_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['argv']))" "$SB/repo/runs/pretrainchamp_$1/config.json"; }
n_ok () { ls "$SB/gcs/champ/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '; }

echo "== S1 fresh 1x8 =="
mk_sandbox; run_chain 0 1
grep -q "CHAIN-CHAMP-COMPLETE" "$SB/w0.log" && ok "S1 complete sentinel" || { bad "S1 sentinel"; tail -8 "$SB/w0.log"; }
[ "$(n_ok)" = 7 ] && ok "S1 7/7 arm markers" || bad "S1 arm markers ($(n_ok))"
[ "$(grep -c 'PREFLIGHT-OK' "$SB/w0.log")" = 7 ] && ok "S1 preflight of all seven arms" || { bad "S1 preflight"; grep PREFLIGHT "$SB/w0.log" | head -3; }
first_pf=$(grep -n "PREFLIGHT-OK" "$SB/w0.log" | tail -1 | cut -d: -f1); first_arm=$(grep -n "PRETRAIN-START" "$SB/w0.log" | head -1 | cut -d: -f1); [ "$first_pf" -lt "$first_arm" ] && ok "S1 every preflight precedes the first arm" || bad "S1 preflight order"
sync_l=$(grep -n "SYNC-AB-OK" "$SB/w0.log" | head -1 | cut -d: -f1); [ -n "$sync_l" ] && [ "$sync_l" -lt "$first_pf" ] && [ -f "$SB/gcs/champ/SYNC-AB" ] && ok "S1 the sync rider ran FIRST and marked" || { bad "S1 rider order"; grep -n "SYNC" "$SB/w0.log" | head -3; }
"$REAL_PY" -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['wall_default_s']==3.0 and d['wall_per_step_s']==7.0 and d['rows']==256 and d['k']==8" "$SB/repo/runs/analysis/champ_sync_ab.json" 2>/dev/null && ok "S1 champ_sync_ab.json carries both walls (256 rows x k8)" || bad "S1 sync json"
grep -m1 "PRETRAIN-START" "$SB/w0.log" | grep -q "PRETRAIN-START C0" && grep "PRETRAIN-START" "$SB/w0.log" | tail -1 | grep -q "C6" && ok "S1 1x8 order: C0 first, C6 last" || bad "S1 1x8 order"
[ -f "$SB/gcs/champ/champ_final.tgz" ] && ok "S1 final tgz" || bad "S1 final tgz"
allf=1; for a in C0 C1 C2 C3 C4 C5 C6; do for r in full_${a}_vsel_t16 full_${a}_final_t16 full_${a}_vsel_t16_alt full_${a}_vsel_t64 d128_${a} d256_${a} scan_${a} census_${a}_vsel census_${a}_final calib_${a}_vsel screen_${a}_vb; do [ -f "$SB/gcs/champ/evals/${r}_OK" ] || { allf=0; echo "    missing $r"; }; done; done; [ $allf = 1 ] && ok "S1 the DEC-class battery on every arm (fulls vsel/final/alt, D64, D128, D256, scan, census x2, calib, vb screen)" || bad "S1 battery"
[ -f "$SB/gcs/champ/evals/commit_C6_OK" ] && [ ! -f "$SB/gcs/champ/evals/commit_C0_OK" ] && ok "S1 the commit row on C6 only" || bad "S1 commit row"
h0=$(eargv "$SB/repo/runs/sxeval_pchampC0/full_vsel_t16/summary_s0.json"); echo "$h0" | grep -q -- "--t-total 16 --ema --record-by-step --record-q" && ! echo "$h0" | grep -q -- "--subsample" && ok "S1 headline: D16 vsel full on 422,786 with bits + halting logits" || bad "S1 headline flags: $h0"
n0=$("$REAL_PY" -c "import json; print(json.load(open('$SB/repo/runs/sxeval_pchampC0/full_vsel_t16/summary_all.json'))['n'])"); [ "$n0" = 422786 ] && ok "S1 headline n-gate 422,786" || bad "S1 headline n ($n0)"
s0=$(eargv "$SB/repo/runs/sxscan_pchampC0/summary_s0.json"); echo "$s0" | grep -q -- "--subsample 5000 --t-total 64 --k-init 32 --ema --batch 128" && ok "S1 scan: 5k x k32 t64 at batch 128" || bad "S1 scan flags: $s0"
d64=$(eargv "$SB/repo/runs/sxeval_pchampC0/full_vsel_t64/summary_s0.json"); echo "$d64" | grep -q -- "--t-total 64 --subsample 100000 --ema --record-by-step" && ok "S1 D64 row on 100k with bits" || bad "S1 D64: $d64"
d128=$(eargv "$SB/repo/runs/sxeval_pchampC0/sub20k_t128/summary_s0.json"); d256=$(eargv "$SB/repo/runs/sxeval_pchampC0/sub5k_t256/summary_s0.json")
echo "$d128" | grep -q -- "--subsample 20000 --t-total 128 --ema --record-by-step" && echo "$d256" | grep -q -- "--subsample 5000 --t-total 256 --ema --record-by-step" && ok "S1 D128 on 20k, D256 on 5k with bits" || bad "S1 depth rows"
cm=$(eargv "$SB/repo/runs/sxeval_pchampC6/sub20k_t16_commit/summary_s0.json"); echo "$cm" | grep -q -- "--subsample 20000 --t-total 16 --ema --record-commit --record-by-step" && ok "S1 C6 commit row: 20k D16 --record-commit" || bad "S1 commit flags: $cm"
alt=$(eargv "$SB/repo/runs/sxeval_pchampC0/full_vsel_t16_alt/summary_s0.json"); echo "$alt" | grep -q -- "--subsample 50000" && ! echo "$alt" | grep -q -- "--ema" && ok "S1 alt row = raw weights on 50k" || bad "S1 alt: $alt"
grep -q "npz.*mon512" "$SB/repo/runs/sxeval_pchampC0/full_vsel_t16/summary_s0.json" && ok "S1 evals read the 512-monitor npz (test set byte-identical)" || bad "S1 npz"
grep -q "VALBEST C0 000050" "$SB/w0.log" && ok "S1 EARLIEST tie selected (steps 50 and 100 tie at .30 -> 50)" || { bad "S1 earliest tie"; grep VALBEST "$SB/w0.log" | head -2; }
grep -q "PRETRAIN-NO-EXTEND C0 (selected grid 50 vs budget 100 window 10)" "$SB/w0.log" && ok "S1 no extension when the peak is early" || { bad "S1 no-extend"; grep EXTEND "$SB/w0.log" | head -3; }
[ "$(grep -c 'PRETRAIN-NO-EXTEND' "$SB/w0.log")" = 7 ] && [ ! -f "$SB/gcs/champ/C0_EXTENDED" ] && ok "S1 no arm extended, no markers" || bad "S1 extension markers"
echo "== S1r the ARM REGISTRY reaches the trainer =="
c0=$(pargv C0); c1=$(pargv C1); c2=$(pargv C2); c3=$(pargv C3); c4=$(pargv C4); c5=$(pargv C5); c6=$(pargv C6)
echo "$c0" | grep -q -- "--cell dec --dec-width 384 --fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --trm-ri-sigma 1.0 --seed 0" && ! echo "$c0" | grep -q -- "--sudoku-digit-aug\|--sudoku-orbit-online\|--dec-commit\|--dec-coupling attn" && ok "S1r C0 = DEC-w384 + FPA + RI, seed 0, no digit aug / orbit / attn / commit" || bad "S1r C0: $c0"
echo "$c1" | grep -q -- "--seed 1" && echo "$c2" | grep -q -- "--seed 2" && ok "S1r C1/C2 seeds 1/2" || bad "S1r seeds"
echo "$c3" | grep -q -- "--sudoku-aug 1000 .*--sudoku-aug 0 --sudoku-orbit-online" && echo "$c3" | grep -q -- "--steps 120" && ok "S1r C3 = online orbit (aug 0 overrides 1000), the long budget" || bad "S1r C3: $c3"
echo "$c4" | grep -q -- "--dec-coupling attn --dec-attn-heads 4 --dec-attn-dk 32" && ok "S1r C4 = attention coupling 4 x 32" || bad "S1r C4: $c4"
echo "$c5" | grep -q -- "--dec-width 384 .*--dec-width 192" && ok "S1r C5 = w192 (the later flag wins)" || bad "S1r C5: $c5"
echo "$c6" | grep -q -- "--dec-commit --dec-commit-tau 0.9 --dec-commit-w 0.1" && echo "$c6" | grep -q -- "--steps 120" && ok "S1r C6 = commit head tau .9 w .1, the long budget" || bad "S1r C6: $c6"
echo "$c0" | grep -q -- "--grid-every 2000" && echo "$c0" | grep -q -- "--monitor-every 2000" && echo "$c0" | grep -q -- "--ckpt-every 500" && echo "$c0" | grep -q "mon512" && ok "S1r cadences 2000/2000/500 on the 512-monitor file" || bad "S1r cadences"
echo "$c0" | grep -q -- "--batch 768 --wd 1.0 --warmup 2000 --lr 1e-4 --lr-end 1e-4 --beta2 0.95 --ema 0.999" && ok "S1r the field regime" || bad "S1r regime"
SB1=$SB

echo "== S2 idempotent rerun =="
(cd "$SB1/repo" && env PATH="$SB1/bin:$PATH" CHAIN_PY="$SB1/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 C1_STEPS_X=100 C1_STEPS_LONG=120 C1_EXT_STEPS=20 C1_EXT_WINDOW=10 C1_WAIT_PASSES=3 C1_POLL_SLEEP=1 LIVE_NO_GUARD=1 bash tools/chain_champ.sh > "$SB1/re.log" 2>&1)
grep -q "CHAIN-CHAMP-COMPLETE" "$SB1/re.log" && ok "S2 complete again" || bad "S2 complete"
[ "$(grep -c 'ARM-SKIP ' "$SB1/re.log")" = 7 ] && grep -q "PREFLIGHT-SKIP" "$SB1/re.log" && grep -q "SYNC-AB-SKIP" "$SB1/re.log" && ok "S2 arms, preflight and rider all skipped" || bad "S2 skips"
[ "$(grep -c 'EVAL-OK\|CENSUS-OK\|CALIB-OK\|PRETRAIN-START' "$SB1/re.log")" = 0 ] && ok "S2 no re-evals, no re-pretrain" || bad "S2 re-ran something"

echo "== SE the EXTENSION rule: C1 peaks at its last grid -> +20 once, resumed at 100; a relaunch keeps the budget; the rerun does not extend twice =="
mk_sandbox; run_chain 0 1 STUB_PEAK_LAST_ARM=C1
grep -q "PRETRAIN-EXTEND C1: selected grid 100 inside the last 10 of 100 -> +20" "$SB/w0.log" && [ -f "$SB/gcs/champ/C1_EXTENDED" ] && grep -q "EXTENDED from 100 to 120 (peak at 100)" "$SB/repo/runs/pretrainchamp_C1/EXTENDED.txt" && ok "SE C1 extended once, labeled + marked" || { bad "SE extend"; grep -E "EXTEND" "$SB/w0.log" | head -4; }
grep -q "RESUMED from runs/pretrainchamp_C1/ckpt_latest.pkl at step 100" "$SB/repo/runs/pretrainchamp_C1.log" && pargv C1 | grep -q -- "--steps 120" && ok "SE the extension RESUMED at 100 with --steps 120" || { bad "SE resume"; head -3 "$SB/repo/runs/pretrainchamp_C1.log"; }
grep -q "VALBEST C1 000120" "$SB/w0.log" && ok "SE re-selected after the extension (the rising peak now at 120)" || { bad "SE reselect"; grep "VALBEST C1" "$SB/w0.log"; }
[ "$(grep -c 'PRETRAIN-NO-EXTEND' "$SB/w0.log")" = 6 ] && [ "$(ls "$SB/gcs/champ/"*_EXTENDED | wc -l | tr -d ' ')" = 1 ] && ok "SE the other six not extended" || bad "SE others"
(cd "$SB/repo" && env PATH="$SB/bin:$PATH" CHAIN_PY="$SB/bin/stubpy" REAL_PY="$REAL_PY" CHAIN_WORKER=0 CHAIN_WORKERS=1 NCHIP_OVERRIDE=4 C1_STEPS_X=100 C1_STEPS_LONG=120 C1_EXT_STEPS=20 C1_EXT_WINDOW=10 C1_WAIT_PASSES=3 C1_POLL_SLEEP=1 LIVE_NO_GUARD=1 STUB_PEAK_LAST_ARM=C1 bash tools/chain_champ.sh > "$SB/re.log" 2>&1)
! grep -q "PRETRAIN-EXTEND" "$SB/re.log" && grep -q "CHAIN-CHAMP-COMPLETE" "$SB/re.log" && ok "SE the rerun never extends twice" || bad "SE rerun"
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" STUB_PEAK_LAST_ARM=C1 "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainchamp_C1 --steps 100 --ema 0.999 --cell dec >/dev/null 2>&1
  echo "EXTENDED from 100 to 120 (peak at 100)" > runs/pretrainchamp_C1/EXTENDED.txt; cp runs/pretrainchamp_C1/EXTENDED.txt "$SB/gcs/champ/C1_EXTENDED"
  mkdir -p "$SB/gcs/champ/live/runs/pretrainchamp_C1"; cp runs/pretrainchamp_C1/* "$SB/gcs/champ/live/runs/pretrainchamp_C1/"; rm -rf runs )
run_chain 0 1 STUB_PEAK_LAST_ARM=C1
grep -q "PRETRAIN-EXTENDED-BUDGET C1 120" "$SB/w0.log" && grep -q "RESUMED from runs/pretrainchamp_C1/ckpt_latest.pkl at step 100" "$SB/repo/runs/pretrainchamp_C1.log" && ! grep -q "PRETRAIN-EXTEND C1:" "$SB/w0.log" && ok "SE a relaunch mid-extension keeps the extended budget (resumed at 100 toward 120, no second extension)" || { bad "SE relaunch budget"; grep -E "EXTEND|RESUMED" "$SB/w0.log" "$SB/repo/runs/pretrainchamp_C1.log" | head -4; }

echo "== S3 NaN one-shot (C2); the trainer's NAN-ABORT rc=3 (C4) =="
mk_sandbox; run_chain 0 1 STUB_NAN_ARM=C2
grep -q "PRETRAIN-NAN C2" "$SB/w0.log" && grep -q "STOPPED final step 50" "$SB/repo/runs/pretrainchamp_C2/STOPPED.txt" && ok "S3 C2 amputated to the last finite grid" || { bad "S3 C2 amputation"; grep -E "NAN|AMPUT" "$SB/w0.log" | head -3; }
[ ! -f "$SB/repo/runs/pretrainchamp_C2/ckpt_000100.pkl" ] && ! grep -q "PRETRAIN-EXTEND C2" "$SB/w0.log" && ok "S3 post-death grid removed; a STOPPED arm is never extended" || bad "S3 post-death"
grep -q "CHAIN-CHAMP-COMPLETE" "$SB/w0.log" && ok "S3 completes despite the stop" || bad "S3 complete"
mk_sandbox; run_chain 0 1 STUB_NANABORT_ARM=C4
grep -q "PRETRAIN-NAN C4 (rc=3)" "$SB/w0.log" && [ -f "$SB/repo/runs/pretrainchamp_C4/STOPPED.txt" ] && grep -q "CHAIN-CHAMP-COMPLETE" "$SB/w0.log" && ok "S3c trainer NAN-ABORT rc=3 -> amputation, complete" || bad "S3c nan-abort path"

echo "== S3d fixed-step screens at the 2k grids of each budget (30k arms: 10k 20k; 50k arms: 10k..40k) =="
mk_sandbox; run_chain 0 1 C1_STEPS_X=30000 C1_STEPS_LONG=50000 C1_EXT_WINDOW=10
[ -f "$SB/gcs/champ/evals/screen_C0_s010000_OK" ] && [ -f "$SB/gcs/champ/evals/screen_C0_s020000_OK" ] && [ ! -f "$SB/gcs/champ/evals/screen_C0_s030000_OK" ] && ok "S3d C0 screens at 10k, 20k only" || bad "S3d C0 screens"
[ -f "$SB/gcs/champ/evals/screen_C3_s030000_OK" ] && [ -f "$SB/gcs/champ/evals/screen_C3_s040000_OK" ] && [ -f "$SB/gcs/champ/evals/screen_C6_s040000_OK" ] && ok "S3d C3/C6 screens up to 40k" || bad "S3d long screens"
grep -q "VALBEST C0 002000" "$SB/w0.log" && ok "S3d earliest tie at scale (the 2k grid of the tie 2k..15k)" || { bad "S3d tie"; grep "VALBEST C0" "$SB/w0.log"; }

echo "== S4 fresh 4x4 static map (+ the rider on w3) =="
mk_sandbox; for w in 0 1 2 3; do run_chain $w 4 & done; wait
grep -q "CHAIN-CHAMP-COMPLETE" "$SB"/w*.log && ok "S4 complete from a worker" || { bad "S4 complete"; tail -3 "$SB"/w*.log; }
[ "$(n_ok)" = 7 ] && ok "S4 7/7 arms" || bad "S4 arms ($(n_ok))"
grep -q "PRETRAIN-START C0" "$SB/w0.log" && grep -q "PRETRAIN-START C4" "$SB/w0.log" && grep -q "PRETRAIN-START C1" "$SB/w1.log" && grep -q "PRETRAIN-START C2" "$SB/w1.log" && grep -q "PRETRAIN-START C6" "$SB/w2.log" && grep -q "PRETRAIN-START C3" "$SB/w3.log" && grep -q "PRETRAIN-START C5" "$SB/w3.log" && ok "S4 map: w0 C0 C4 · w1 C1 C2 · w2 C6 · w3 C3 C5" || bad "S4 map"
grep -q "SYNC-AB-OK" "$SB/w3.log" && ! grep -q "SYNC-AB-OK" "$SB/w0.log" && ok "S4 the rider on w3 only" || bad "S4 rider worker"
for w in 0 1 2 3; do [ -f "$SB/gcs/champ/PREFLIGHT_OK_w${w}_nw4" ] || bad "S4 preflight marker w$w"; done; ok "S4 shape-specific preflight markers"
echo "== S4b fresh 8x4 =="
mk_sandbox; for w in 0 1 2 3 4 5 6 7; do run_chain $w 8 & done; wait
grep -q "CHAIN-CHAMP-COMPLETE" "$SB"/w*.log && [ "$(n_ok)" = 7 ] && grep -q "PRETRAIN-START C6" "$SB/w6.log" && grep -q "SYNC-AB-OK" "$SB/w7.log" && ! grep -q "PRETRAIN-START" "$SB/w7.log" && ok "S4b 8x4: one arm per worker, the rider on w7, complete" || bad "S4b"

echo "== S5 no monitor rows (C1) -> LOUD fallback, no extension =="
mk_sandbox; run_chain 0 1 STUB_NO_MONITOR_ARM=C1
grep -q "VB-FALLBACK-FINAL C1" "$SB/w0.log" && grep -q "FALLBACK-FINAL" "$SB/repo/runs/pretrainchamp_C1/val_best.txt" && grep -q "PRETRAIN-NO-EXTEND C1 (selected grid none" "$SB/w0.log" && ok "S5 fallback echoed + labeled; no extension without a selection" || { bad "S5 fallback"; grep -E "FALLBACK|EXTEND C1" "$SB/w0.log" | head -3; }
grep -q "FULL-FINAL C1 := vsel" "$SB/w0.log" && grep -q "CHAIN-CHAMP-COMPLETE" "$SB/w0.log" && ok "S5 final := vsel copy, complete" || bad "S5 final copy"

echo "== S6 banked pretrain, no local dir -> re-pull before select =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainchamp_C0 --steps 100 --ema 0.999 --cell dec >/dev/null 2>&1
  tar czf "$SB/gcs/champ/C0_pretrain.tgz" runs/pretrainchamp_C0 && cp runs/pretrainchamp_C0/ckpt_latest.pkl "$SB/gcs/champ/C0_ckpt.pkl"
  echo ok > "$SB/gcs/champ/C0_PRETRAIN_OK"; rm -rf runs/pretrainchamp_C0 )
run_chain 0 1
grep -q "PRETRAIN-SKIP C0" "$SB/w0.log" && grep -q "PRETRAIN-RESTORE C0" "$SB/w0.log" && grep -q "VALBEST C0" "$SB/w0.log" && ok "S6 banked arm re-pulled, selected" || bad "S6 re-pull"

echo "== S7 the DEC scan stall ladder: C0's scan stalls at BATCHONLY -> --z0-device completes -> RECIPE-DEC reused by C1 =="
mk_sandbox; run_chain 0 1 STUB_STALL_ARM=C0
grep -q "EVAL-STALLED scan_C0" "$SB/w0.log" && grep -q "DEC-SCAN-VARIANT-FAILED C0 rc=2 flags=\[\]" "$SB/w0.log" && ok "S7 the staged stall fired and was killed by the watchdog" || { bad "S7 stall"; grep -E "STALL|VARIANT" "$SB/w0.log" | head -4; }
grep -q "RECIPE-DEC C0: \[--z0-device\]" "$SB/w0.log" && [ "$(cat "$SB/gcs/champ/RECIPE-DEC")" = "--z0-device" ] && [ -f "$SB/gcs/champ/evals/scan_C0_OK" ] && ok "S7 the next variant completed; recipe banked" || bad "S7 recipe"
[ "$(grep -c 'DEC-SCAN-TRY C1' "$SB/w0.log")" = 1 ] && grep -q "DEC-SCAN-TRY C1 batch=128 flags=\[--z0-device\]" "$SB/w0.log" && ok "S7 C1 used the recipe directly (one try)" || bad "S7 reuse"
grep -q "CHAIN-CHAMP-COMPLETE" "$SB/w0.log" && ok "S7 complete" || bad "S7 complete"
mk_sandbox; run_chain 0 1 STUB_STALL_ARM=C0 STUB_STALL_ALWAYS=1
grep -q "SCAN-DEADLOCK C0" "$SB/w0.log" && [ -f "$SB/gcs/champ/C0_SCAN_DEADLOCK" ] && [ ! -f "$SB/gcs/champ/evals/scan_C0_OK" ] && [ -f "$SB/gcs/champ/C0_ARM_OK" ] && [ -f "$SB/gcs/champ/evals/calib_C0_vsel_OK" ] && ok "S7b every variant stalled -> SCAN-DEADLOCK labeled; the arm's census/calib ran; ARM_OK" || { bad "S7b deadlock path"; grep -E "DEADLOCK|ARM-OK C0" "$SB/w0.log" | head -3; }
[ "$(grep -c 'EVAL-STALLED scan_C0' "$SB/w0.log")" = 3 ] && grep -q "CHAIN-CHAMP-COMPLETE" "$SB/w0.log" && ok "S7b three variants tried; complete" || bad "S7b variants ($(grep -c 'EVAL-STALLED scan_C0' "$SB/w0.log"))"

echo "== S8 preflight failure: a treatment arm (C4) is SKIPPED, a seed arm (C1) ABORTS the worker =="
mk_sandbox; run_chain 0 1 STUB_PREFLIGHT_FAIL=C4
grep -q "PREFLIGHT-FAILED C4 (rc=1) -> SKIPPED" "$SB/w0.log" && [ -f "$SB/gcs/champ/C4_SKIPPED" ] && grep -q "ARM-SKIPPED C4" "$SB/w0.log" && [ "$(n_ok)" = 6 ] && grep -q "CHAIN-CHAMP-COMPLETE" "$SB/w0.log" && ok "S8 C4 skipped, labeled; complete with 6" || { bad "S8 treatment skip"; grep -E "PREFLIGHT|SKIP" "$SB/w0.log" | head -4; }
mk_sandbox; run_chain 0 1 STUB_PREFLIGHT_FAIL=C1
grep -q "PREFLIGHT-FAILED C1 (rc=1) -> the night stops here" "$SB/w0.log" && grep -q "CHAMP-PREFLIGHT-ABORT" "$SB/w0.log" && ! grep -q "PRETRAIN-START" "$SB/w0.log" && [ "$(n_ok)" = 0 ] && ok "S8b a seed-arm preflight failure aborts before any arm" || bad "S8b seed abort"

echo "== S9 launch-time HBM OOM (no step logged) -> ONE retry with --remat (C0) =="
mk_sandbox; run_chain 0 1 STUB_OOM_ARM=C0
grep -q "PRETRAIN-OOM-RETRY-REMAT C0" "$SB/w0.log" && [ -f "$SB/repo/runs/pretrainchamp_C0/RETRY_REMAT.txt" ] && [ -f "$SB/gcs/champ/C0_ARM_OK" ] && [ ! -f "$SB/repo/runs/pretrainchamp_C0/STOPPED.txt" ] && ok "S9 OOM -> remat retry, labeled, clean" || { bad "S9"; grep -E "OOM|REMAT" "$SB/w0.log" | head -3; }

echo "== S10 LIVE BANK: C0 in flight at step 50 on the live prefix, FRESH node -> RESUMED =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretrainchamp_C0 --steps 50 --ema 0.999 --cell dec >/dev/null 2>&1
  mkdir -p "$SB/gcs/champ/live/runs/pretrainchamp_C0"; cp runs/pretrainchamp_C0/* "$SB/gcs/champ/live/runs/pretrainchamp_C0/"; rm -rf runs )
run_chain 0 1
grep -q "LIVE-RESTORE pulled=" "$SB/w0.log" && grep -q "RESUMED from runs/pretrainchamp_C0/ckpt_latest.pkl at step 50" "$SB/repo/runs/pretrainchamp_C0.log" && grep -q "CHAIN-CHAMP-COMPLETE" "$SB/w0.log" && ok "S10 restored + RESUMED at 50, complete" || { bad "S10"; grep -E "LIVE|RESUMED" "$SB/w0.log" "$SB/repo/runs/pretrainchamp_C0.log" | head -4; }

echo "== S11 the sync rider without A3's frontier grid -> labeled, the night proceeds =="
NO_A3=1 mk_sandbox; run_chain 0 1
grep -q "SYNC-AB-NOCKPT" "$SB/w0.log" && [ -f "$SB/gcs/champ/SYNC-AB" ] && grep -q "CHAIN-CHAMP-COMPLETE" "$SB/w0.log" && ok "S11 rider skipped, labeled; complete" || bad "S11"

echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
