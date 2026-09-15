#!/bin/bash
# Ledger: DEC-ARC NIGHT offline stub harness (2026-09-10; the house law: no chain launches without an end-to-end offline
# pass; every negative scenario asserts the staged failure fired; the 2026-09-04 lesson: a harness adapted from the
# previous campaign is unverified until RUN). Stubs: gsutil/gcloud/timeout -> a local FAKE_GCS dir; $CHAIN_PY -> a stub
# emulating pretrain.py / eval_decarc.py / arc_suite.py / eval_sudoku_extreme.py (select_ckpt.py runs REAL: the extension
# rule, the earliest-tie selection and the native --row val selection go through the registered tool); REAL python runs
# the chain's own helpers (nan_check / amputate / n-gates / gate_tasks through the real eval_decarc.task_ids_of on the
# real data dirs). Scenarios: S1 fresh 1-worker -> COMPLETE (4 arms; the registry reaches the trainer; every set's
# shards + summarize + n-gate; N0 through arc_suite on four chips; vsel.json written); S2 idempotent rerun; S3 the
# EXTENSION rule (a DEC peak at the end -> +EXT once, resumed; early -> none; N0 never extends; a relaunch keeps the
# extended budget); S4 the 2-worker map (w0 D0 D2, w1 D1 N0; completion by either); S5 NaN one-shot amputation; S6
# preflight failure: an optional arm SKIPPED, a seed arm ABORTS; S7 launch-time OOM -> ONE --remat retry; S8 the rider
# (RIDER=1: claimed once, n-gated 422786; RIDER=0: off); S9 a shard failure -> ARM-PARTIAL, the rerun redoes only the
# missing row; S10 missing data -> DATA-ABORT before any launch; S11 the live-bank restore -> RESUMED.
# 2026-09-15 (the Sudoku lessons folded in; Plan §12): S1 also asserts the registry's --decarc-eval-start, the mon96 row (n-gated
# NVAL, k 0, no re-fit flags), N0's bridge row (48 rows, k 0) and --trace-fused 1 on every DEC eval; S15 asserts the bridge
# carries NO --fit-t while the night rows do; S17 the ARC env's exact CHAIN_EXTRA_ENV (the ARC bucket mapped; every knob reaches
# the tools; NOTHING written under the Sudoku era's prefix); S18 the plateau-keyed extension (a flat, declining monitor extends
# under DA_EXT_PLATEAU_PP and not without); S19 the fused-trace cross-check failing on the 'chip' -> the eager fallback (--trace-fused
# 0) and completion; S20 DA_MON96=0 DA_BRIDGE=0 remove the two rows (default-inert switches).
# 2026-09-16 (the two-pod night): S21 two pods = two separate node repos sharing one bucket, CHAIN_WORKERS=2 + DA_SHARE_EXIT=1 +
# a live prefix per pod: the pod that finishes its share first banks SHARE_DONE_w<W> and exits 0 without the sentinel; the other
# pod finds every arm done and finalizes; each pod's live bank writes only its own prefix; both finishing orders. S22 an
# ARM-PARTIAL no longer waits in the completion loop: the worker exits 1 at once (OWN-ARMS-INCOMPLETE) and a rerun completes.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export REAL_PY="$REPO/.venv/bin/python3"
PASS=0; FAIL=0
ok () { PASS=$((PASS+1)); echo "  PASS  $1"; }
bad () { FAIL=$((FAIL+1)); echo "  FAIL  $1"; }

mk_sandbox () {
  SB=$(mktemp -d /tmp/hda.XXXXXX)
  mkdir -p "$SB/repo/tools" "$SB/repo/runs" "$SB/repo/data" "$SB/gcs/decarc/sets" "$SB/gcs/champ/sets" "$SB/gcs/arc/sets" "$SB/bin"
  cp "$REPO/tools/chain_decarc.sh" "$REPO/tools/live_bank.sh" "$REPO/tools/select_ckpt.py" "$REPO/tools/eval_decarc.py" "$REPO/tools/valhard.json" "$REPO/tools/dev30.py" "$SB/repo/tools/"
  ln -s "$REPO/src" "$SB/repo/src"
  if [ "${NO_DATA:-0}" != 1 ]; then for d in ARC-AGI ConceptARC re_arc re_gate48 re_gateb48 re_train48; do ln -s "$REPO/data/$d" "$SB/repo/data/$d"; done; fi
  : > "$SB/gcs/champ/sets/sudoku_extreme_seed0_mon512.npz"; echo c2 > "$SB/gcs/decarc/sets/C2_vsel.pkl"
  : > "$SB/gcs/arc/sets/sudoku_extreme_seed0_mon512.npz"; echo c2 > "$SB/gcs/arc/sets/C2_vsel.pkl"   # the ARC project's sets (gs://qhrrn2-arc/sets)
  cat > "$SB/bin/gsutil" <<SH
#!/bin/bash
GB="$SB/gcs"
SH
  cat >> "$SB/bin/gsutil" <<'SH'
map () { echo "$1" | sed "s|gs://qhrrn2-arc/|$GB/arc/|; s|gs://qhrrn2-rescue/|$GB/|"; }
unmap () { sed "s|$GB/arc/|gs://qhrrn2-arc/|; s|$GB/|gs://qhrrn2-rescue/|"; }
args=(); for a in "$@"; do [ "$a" = "-q" ] || [ "$a" = "-m" ] || args+=("$a"); done
cmd=${args[0]:-}
case $cmd in
  stat) p=$(map "${args[1]}"); [ -f "$p" ];;
  ls)   rc=1; for g in "${args[@]:1}"; do p=$(map "$g"); for f in $p; do [ -e "$f" ] && { echo "$f" | unmap; rc=0; }; done; done; exit $rc;;
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
HERE = os.path.dirname(os.path.abspath(sys.argv[0]))
if tool.endswith("select_ckpt.py"):
    os.execv(os.environ["REAL_PY"], [os.environ["REAL_PY"], os.path.join(HERE, "..", "repo", "tools", "select_ckpt.py")] + argv[1:])
if tool.endswith("pretrain.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    steps = int(flag("--steps", "100"))
    base = out.name.replace("pretraindecarc_", "").replace("preflightdecarc_", "")
    dec = flag("--cell", "rg") == "decarc"
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
    arm_run = out.name.startswith("pretraindecarc_")
    is_nan = arm_run and base == os.environ.get("STUB_NAN_ARM", "")
    peak_last = base in os.environ.get("STUB_PEAK_LAST_ARM", "").split(",")
    rows = []
    mon_steps = sorted({s for s in range(2000, steps + 1, 2000)} | {steps})
    for s in mon_steps:
        if s <= start: continue
        loss = float("nan") if (is_nan and s == steps) else 0.5
        rows.append(json.dumps({"step": s, "loss": loss, "ce_in": .4, "q_loss": .01, "train_exact": .3, "halt_frac": .5, "mean_steps": 4.0, "I_total": 0.0, "A_total": 0.0, "rule_H": 0.0, "lr": 1e-4, "steps_per_sec": 1.1, "t": "T"}))
        if int(flag("--monitor-every", "0")) and dec:
            v = 0.30 + (0.001 * s / 8000.0 if peak_last else -0.001 * s / 8000.0)   # monotone in the ABSOLUTE step (a resumed extension keeps rising)
            rows.append(json.dumps({"monitor": {"val_t16": v - 0.02, "val_pix": .5, "n_val": 100, "val_t16_ema": v, "val_pix_ema": .5 + s / 1e6, "step": s, "wall_s": 9.0}}))
        if int(flag("--val-every", "0")) and not dec and s % int(flag("--val-every")) == 0:
            ne = 10 + (s // 2000 if peak_last else -(s // 2000) % 5)
            rows.append(json.dumps({"val": {"val_exact": ne, "val_total": 100, "val_pix_mean": .6, "obj_consistency": 0, "obj_consistency_n": 0, "step": s}}))
    with open(out / "metrics.jsonl", "a") as f:
        for r in rows: f.write(r + "\n")
    cfg = {"cell_kind": "decarc" if dec else "rg", "T": int(flag("--T", "6")), "d_task": 32, "canvas": 32}
    ck = {"state": {"model": {"w": np.zeros(2, np.float32)}, "table": np.zeros((3, 10, 32), np.float32)}, "opt_state": {}, "rng": np.zeros(2, np.uint32), "step": steps, "config": cfg, "state_ema": {"model": {"w": np.zeros(2, np.float32)}, "table": np.zeros((3, 10, 32), np.float32)}}
    if is_nan: ck["state"]["model"]["w"] = np.array([np.nan, 0.0], np.float32)
    grid = int(flag("--grid-every", "2000"))
    for s in mon_steps:
        if s <= start: continue
        if s % grid == 0 or s == steps:
            g = dict(ck); g["step"] = s
            if is_nan and s < steps: g = dict(ck, state={"model": {"w": np.zeros(2, np.float32)}, "table": ck["state"]["table"]})
            pickle.dump(g, open(out / f"ckpt_{s:06d}.pkl", "wb"))
    pickle.dump(ck, open(latest, "wb"))
    (out / "config.json").write_text(json.dumps({"argv": argv, "n_tasks": 3}))
    if dec and arm_run:   # the real trainer writes the monitor tasks (table row, id, query count) for the evaluator's mon96 set
        nv = int(flag("--n-val", "96")); (out / "val_tasks.json").write_text(json.dumps([{"t": i, "tid": f"mon{i:02d}", "n_q": 1} for i in range(nv)]))
    print(f"step {steps:6d}  loss 0.5", flush=True); print("DONE"); sys.exit(0)
if tool.endswith("eval_decarc.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    if "--summarize" in argv:
        n = sum(1 for f in out.glob("results*.jsonl") for l in open(f) if l.strip())
        pv = json.load(open(sorted(out.glob("provenance*.json"))[0]))
        s = {"ckpt": pv["ckpt"], "ema": pv["ema"], "set": pv["set"], "n_tasks": n, "n_queries": 3 * n, "xcheck_all": True, "clean_exact_limit": .08, "exact_by_step": [.05, .07, .08], "lost": 0,
             "auc_res_z": .9, "oracle": .12, "retention_gt": .4, "retention_gt_solved": .95, "converged_wrong_of_converged": .95, "flips_per_cell": 1.0,
             "flip": {"flip_rate_colour": .02, "flip_rate_floor": .02}, "vote": {"pass1": .08, "pass2": .11, "any_view": .2}}
        (out / "summary.json").write_text(json.dumps(s)); print(json.dumps(s)); sys.exit(0)
    sys.path.insert(0, os.path.join(HERE, "..", "repo", "tools"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("ed_ids", os.path.join(HERE, "..", "repo", "tools", "eval_decarc.py"))
    # the task lists come from the REAL tool's set loaders (data dirs linked into the sandbox)
    ids = None
    try:
        import json as _j
        setn = flag("--set", "valhard")
        if setn == "valhard": ids = _j.load(open(os.path.join(HERE, "..", "repo", "tools", "valhard.json")))["valhard"]
        elif setn == "dev30":
            sys.path.insert(0, os.path.join(HERE, "..", "repo", "tools")); import dev30; ids = sorted(dev30.MANIFEST)
        elif setn == "rg96": ids = sorted(p[:-5] for d in ("re_gate48", "re_gateb48") for p in os.listdir(os.path.join(HERE, "..", "repo", "data", d)) if p.endswith(".json"))
        elif setn == "rt48": ids = sorted(p[:-5] for p in os.listdir(os.path.join(HERE, "..", "repo", "data", "re_train48")) if p.endswith(".json"))
        elif setn == "arc1eval": ids = sorted(p[:-5] for p in os.listdir(os.path.join(HERE, "..", "repo", "data", "ARC-AGI", "data", "evaluation")) if p.endswith(".json"))
        elif setn == "mon96":   # the real tool: the ids from the checkpoint dir's val_tasks.json (no fit)
            assert int(flag("--views", "8")) <= 1 and "--flip-test" not in argv, "mon96 takes no re-fit flags"
            ids = [r["tid"] for r in _j.load(open(os.path.join(os.path.dirname(flag("--ckpt")), "val_tasks.json")))]
    except Exception as e:
        print(f"set load failed: {e}", file=sys.stderr); sys.exit(1)
    lim = int(flag("--limit", "0"))
    if lim: ids = ids[:lim]   # the real tool: ids[:limit] BEFORE the shard slice
    sh = flag("--shard"); tag = ""
    if sh:
        i, n = (int(v) for v in sh.split("/")); ids = ids[i::n]; tag = f"_{i}"
    if os.environ.get("STUB_EVAL_FAIL", "") == f"{Path(flag('--ckpt')).parent.name.replace('pretraindecarc_', '')}:{flag('--set')}:{tag.strip('_')}" and not (out / "failed_once").exists():
        (out / "failed_once").write_text("x"); print("staged shard failure", file=sys.stderr); sys.exit(1)
    key = f"{Path(flag('--ckpt')).parent.name.replace('pretraindecarc_', '')}:{flag('--set')}:{tag.strip('_')}"
    if os.environ.get("STUB_EVAL_STALL", "") == key and (os.environ.get("STUB_EVAL_STALL_ALWAYS") == "1" or not (out / "stalled_once").exists()):
        (out / "stalled_once").write_text("x"); time.sleep(int(os.environ.get("STUB_STALL_SLEEP", "12"))); sys.exit(0)   # a starved shard: alive, silent
    (out / f"provenance{tag}.json").write_text(json.dumps({"ckpt": flag("--ckpt"), "ema": "--ema" in argv, "set": flag("--set"), "k": int(flag("--k", "32")), "views": int(flag("--views", "8")), "argv": argv}))
    rf = out / f"results{tag}.jsonl"; done = set()
    if rf.exists():
        done = {json.loads(l)["task"] for l in rf.read_text().splitlines() if l.strip()}
    with open(rf, "a") as f:
        for t in ids:
            if t not in done: f.write(json.dumps({"task": t, "queries": [{"q": 0}]}) + "\n")
    sys.exit(0)
if tool.endswith("arc_suite.py"):
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    key = f"N0:{Path(flag('--out')).name}:"   # keyed by the OUT DIR (rg96 / rt48 / s<i> share --set valhard)
    if os.environ.get("STUB_EVAL_STALL", "") == key and (os.environ.get("STUB_EVAL_STALL_ALWAYS") == "1" or not (out / "stalled_once").exists()):
        (out / "stalled_once").write_text("x"); time.sleep(int(os.environ.get("STUB_STALL_SLEEP", "12"))); sys.exit(0)
    ids = flag("--tasks").split(",") if flag("--tasks") else (json.load(open(os.path.join(HERE, "..", "repo", "tools", "valhard.json")))["valhard"] if flag("--set", "valhard") == "valhard" else [f"t{i}" for i in range(30)])
    with open(out / "results.jsonl", "w") as f:
        for t in ids: f.write(json.dumps({"task": t}) + "\n")
    (out / "summary.json").write_text(json.dumps({"ckpt": flag("--ckpt"), "n_queries": len(ids), "clean_exact_limit": .07, "oracle": .15, "argv": argv}))
    sys.exit(0)
if tool.endswith("cost_probe.py"):
    out = Path(flag("--out")); out.parent.mkdir(parents=True, exist_ok=True)
    s = float(os.environ.get("STUB_COST_S", "0.01")); dec = "decarc_D" in flag("--ckpt")   # 0.01 s/step -> the full registered battery projects ~1.8 h on the sandbox's 4 chips (inside the 8 h budget); S16 sets 100
    rec = {"ckpt": flag("--ckpt"), "set": flag("--set"), "task": "t", "cell_kind": "decarc" if dec else "rg", "B": 6, "n_support": 3, "fit_T": int(flag("--fit-t", "16")), "cfg_T": 16, "t_total": int(flag("--t-total", "16")), "steps": int(flag("--steps", "8")), "compile_s": 1.0, "s_per_step": s, "val_first_s": 1.0, "val_s": 0.1, "trace_first_s": 1.0, "trace_s": 0.2, "device": "stub", "argv": argv}
    if dec: rec.update({"trace_dec_first_s": 1.0, "trace_dec_s": 0.2, "trace_dec_eager_s": 1.0, "xcheck_fused_eager": os.environ.get("STUB_XCHECK_FAIL") != "1", "xcheck_fused_probe": True})
    out.write_text(json.dumps(rec)); sys.exit(0)
if tool.endswith("eval_sudoku_extreme.py"):
    if "--merge" in argv:
        d = Path(flag("--merge")); n = sum(json.load(open(p))["n"] for p in d.glob("shard_*.json"))
        (d / "summary_all.json").write_text(json.dumps({"n": n, "exact": .99})); sys.exit(0)
    out = Path(flag("--out")); out.mkdir(parents=True, exist_ok=True)
    i, k = (int(v) for v in flag("--shard", "0/1").split("/"))
    n = 422786 // k + (1 if i < 422786 % k else 0)
    (out / f"shard_{i}.json").write_text(json.dumps({"n": n, "argv": argv})); sys.exit(0)
print(f"stubpy: unknown tool {tool}", file=sys.stderr); sys.exit(2)
PYEOF
  chmod +x "$SB/bin/stubpy"
}

run_chain () {  # W NW [extra VAR=val...]
  local w=$1 nw=$2; shift 2
  mkdir -p "$SB/tmp"
  ( cd "$SB/repo" && env PATH="$SB/bin:$PATH" TMPDIR="$SB/tmp" CHAIN_PY="$SB/bin/stubpy" CHAIN_WORKER="$w" CHAIN_WORKERS="$nw" NCHIP_OVERRIDE=4 \
      LIVE_EVERY=1 C1_WAIT_PASSES=3 C1_WAIT_SLEEP=0 DA_PF_STEPS=20 DA_STEPS_DEC=8000 DA_STEPS_NAT=6000 DA_EXT_STEPS=4000 DA_EXT_WINDOW=2000 \
      DA_MON=2000 DA_CKPT_EVERY=1000 DA_EVAL_STALL_SEC=3 WATCH_EVERY=1 "$@" bash tools/chain_decarc.sh > "$SB/w$w.log" 2>&1 )
}
run_chain_in () {  # REPO W NW [extra VAR=val...] — a second node's repo in the same sandbox bucket (S21)
  local repo=$1 w=$2 nw=$3; shift 3
  mkdir -p "$repo/tmp"
  ( cd "$repo" && env PATH="$SB/bin:$PATH" TMPDIR="$repo/tmp" CHAIN_PY="$SB/bin/stubpy" CHAIN_WORKER="$w" CHAIN_WORKERS="$nw" NCHIP_OVERRIDE=4 \
      LIVE_EVERY=1 C1_WAIT_PASSES=3 C1_WAIT_SLEEP=0 DA_PF_STEPS=20 DA_STEPS_DEC=8000 DA_STEPS_NAT=6000 DA_EXT_STEPS=4000 DA_EXT_WINDOW=2000 \
      DA_MON=2000 DA_CKPT_EVERY=1000 DA_EVAL_STALL_SEC=3 WATCH_EVERY=1 "$@" bash tools/chain_decarc.sh > "$SB/$(basename "$repo")_w$w.log" 2>&1 )
}
mk_repo2 () {  # a second node's repo: the same tools (copied), src and data linked, its own runs/
  local r="$SB/node$1"; mkdir -p "$r/tools" "$r/runs" "$r/data"
  cp "$SB/repo/tools/"* "$r/tools/"; ln -s "$REPO/src" "$r/src"
  for d in ARC-AGI ConceptARC re_arc re_gate48 re_gateb48 re_train48; do ln -s "$REPO/data/$d" "$r/data/$d"; done
  # the stub finds its tool paths relative to $SB/repo (HERE/../repo): the chain's own helpers run inside this node's repo
  echo "$r"
}
pargv () { "$REAL_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['argv']))" "$SB/repo/runs/pretraindecarc_$1/config.json"; }
eargv () { "$REAL_PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('argv', [])))" "$1"; }
n_ok () { ls "$SB/gcs/decarc/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' '; }

echo "== S1 fresh 1-worker run -> COMPLETE (4 arms; the registry; every set; N0 through arc_suite; vsel.json) =="
mk_sandbox; run_chain 0 1
grep -q "DATA-OK" "$SB/w0.log" && ok "S1 data present" || bad "S1 data"
[ "$(n_ok)" = 4 ] && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && [ -f "$SB/gcs/decarc/decarc_final.tgz" ] && ok "S1 complete with 4 ARM_OK + decarc_final.tgz" || bad "S1 complete ($(n_ok) ok)"
A0=$(pargv D0); A2=$(pargv D2); AN=$(pargv N0)
echo "$A0" | grep -q -- "--cell decarc --dec-width 160 --decarc-heads 4" && echo "$A0" | grep -q -- "--trm-ri-sigma 1.0 --fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --seed 0" && echo "$A0" | grep -q -- "--w-void 0.5 --table-lr 1e-2 --table-wd 0.1" && echo "$A0" | grep -q -- "--rearc --conceptarc --orbit 4 --n-val 96 --dp" && echo "$A0" | grep -q -- "--steps 8000" && echo "$A0" | grep -q -- "--wd 0.1 --warmup 2000 --lr 2e-4 --lr-end 2e-4" && ok "S1r D0 registry reaches the trainer (the field ARC regime)" || bad "S1r D0 registry: $A0"
echo "$A2" | grep -q -- "--trm-ri-sigma 0 --fpa-k 0 --seed 0" && ! echo "$A2" | grep -q -- "--trm-ri-sigma 1.0" && ok "S1r D2 = the plain twin" || bad "S1r D2: $A2"
echo "$AN" | grep -q -- "--d 96 --T 6 --anchor-p 0.3 --beta-flux 3e-5 --beta-flux-nl 1e-5 --ni-sigma 0.01" && echo "$AN" | grep -q -- "--steps 6000" && ! echo "$AN" | grep -q -- "--cell" && echo "$AN" | grep -q -- "--val-every 2000" && ok "S1r N0 = the native A5-class d96 arm with the val back-port" || bad "S1r N0: $AN"
for s in valhard dev30 mon96 rg96 rt48 arc1eval valhard_final; do [ -f "$SB/gcs/decarc/evals/D0_${s}_OK" ] || bad "S1 D0 eval $s missing"; done
echo "$A0" | grep -q -- "--decarc-eval-start fieldfix" && ok "S1r D0 registry carries the deterministic evaluation start (fieldfix)" || bad "S1r eval start: $A0"
"$REAL_PY" -c "import json; s=json.load(open('$SB/repo/runs/decarceval_D0/mon96/summary.json')); assert s['n_tasks']==96, s" && ok "S1 the mon96 row n-gated at NVAL 96" || bad "S1 mon96 n-gate"
EV=$(eargv "$SB/repo/runs/decarceval_D0/mon96/provenance_0.json"); echo "$EV" | grep -q -- "--k 0" && echo "$EV" | grep -q -- "--views 1" && ! echo "$EV" | grep -q -- "--flip-test" && echo "$EV" | grep -q -- "--ladder 0,0.2,0.4,0.6,0.8" && ok "S1 mon96 flags (k 0, views 1, no flip, the ladder)" || bad "S1 mon96 flags: $EV"
EV=$(eargv "$SB/repo/runs/decarceval_D0/valhard/provenance_0.json"); echo "$EV" | grep -q -- "--trace-fused 1" && ! echo "$EV" | grep -q -- "--start-rows" && ok "S1 every DEC eval carries --trace-fused 1; no start rows by default" || bad "S1 trace-fused: $EV"
[ -f "$SB/gcs/decarc/evals/N0_valhard_bridge_OK" ] && [ "$(cat "$SB/repo/runs/decarceval_N0/valhard_bridge/results.jsonl" | wc -l | tr -d " ")" = 48 ] && EB=$(eargv "$SB/repo/runs/decarceval_N0/valhard_bridge/summary.json") && echo "$EB" | grep -q -- "--k 0" && ! echo "$EB" | grep -q -- "--fit-t" && ok "S1 N0's bridge row: 48 tasks, k 0, the deployed fit" || bad "S1 bridge"
grep -q "SELECT-KEYS D0 key=val_t16_ema key2=val_t16 start=fieldfix" "$SB/w0.log" && ok "S1 the registered two selection keys by default (no third key, no plateau)" || bad "S1 select keys: $(grep SELECT-KEYS "$SB/w0.log" | head -1)"
[ -f "$SB/gcs/decarc/evals/D0_arc1eval_OK" ] && [ "$(ls "$SB/repo/runs/decarceval_D0/valhard/results_"*.jsonl | wc -l | tr -d ' ')" = 4 ] && ok "S1 D0's six sets banked; val-hard 4-way sharded" || bad "S1 D0 sets"
"$REAL_PY" -c "import json; s=json.load(open('$SB/repo/runs/decarceval_D0/valhard/summary.json')); assert s['n_tasks']==48 and s['ema'] is True, s; s=json.load(open('$SB/repo/runs/decarceval_D0/arc1eval/summary.json')); assert s['n_tasks']==400, s; s=json.load(open('$SB/repo/runs/decarceval_D0/rg96/summary.json')); assert s['n_tasks']==96, s" && ok "S1 n-gates (48 / 400 / 96) and --ema recorded" || bad "S1 n-gates"
EV=$(eargv "$SB/repo/runs/decarceval_D0/valhard/provenance_0.json"); echo "$EV" | grep -q -- "--k 32" && echo "$EV" | grep -q -- "--views 8" && echo "$EV" | grep -q -- "--flip-test" && echo "$EV" | grep -q -- "--ladder 0,0.2,0.4,0.6,0.8" && ok "S1 val-hard battery flags (k32, 8 views, flip, ladder)" || bad "S1 val-hard flags: $EV"
EV=$(eargv "$SB/repo/runs/decarceval_D0/rg96/provenance_0.json"); echo "$EV" | grep -q -- "--k 8" && echo "$EV" | grep -q -- "--views 1" && ok "S1 rg-96 flags (k8, no vote)" || bad "S1 rg96 flags: $EV"
for s in valhard dev30 rg96 rt48 arc1eval; do [ -f "$SB/gcs/decarc/evals/N0_${s}_OK" ] || bad "S1 N0 eval $s missing"; done
"$REAL_PY" -c "import json; s=json.load(open('$SB/repo/runs/decarceval_N0/arc1eval/summary.json')); assert s['n_tasks']==400 and s['shards']==4, s" && ok "S1 N0 public row: 400 tasks over 4 shards, merged" || bad "S1 N0 arc1eval merge"
EV=$(eargv "$SB/repo/runs/decarceval_N0/rg96/summary.json"); echo "$EV" | grep -q -- "--tasks" && echo "$EV" | grep -q "rg_00d62c1b" && echo "$EV" | grep -q -- "--k 8" && ok "S1 N0 rg-96 through arc_suite --tasks (96 ids), k8" || bad "S1 N0 rg96: ${EV:0:200}"
"$REAL_PY" -c "import json; v=json.load(open('$SB/repo/runs/pretraindecarc_D0/vsel.json')); assert v['step'] and v['ckpt'].endswith('.pkl'), v; v=json.load(open('$SB/repo/runs/pretraindecarc_N0/vsel.json')); assert v['step'], v" && [ -f "$SB/gcs/decarc/D0_vsel.json" ] && ok "S1 vsel.json written and banked (DEC on the EMA monitor, N0 on its val rows)" || bad "S1 vsel"
grep -q "RIDER-OFF" "$SB/w0.log" && ok "S1 rider off by default" || bad "S1 rider default"
[ "$(grep -c "COST-OK" "$SB/w0.log")" = 4 ] && [ -f "$SB/gcs/decarc/D0_cost_probe.json" ] && [ -f "$SB/gcs/decarc/D0_COST_OK" ] && grep -q "COST-PROBE D0 projected_h=" "$SB/w0.log" && ok "S1 the cost probe ran before every battery (4 COST-OK; json banked)" || bad "S1 cost probe: $(grep -c COST-OK "$SB/w0.log")"
EV=$(eargv "$SB/repo/runs/decarceval_D0/valhard/provenance_0.json"); EN=$(eargv "$SB/repo/runs/decarceval_N0/valhard/summary.json"); ! echo "$EV" | grep -q -- "--limit" && ! echo "$EV" | grep -q -- "--fit-t" && ! echo "$EN" | grep -q -- "--tasks" && ! echo "$EN" | grep -q -- "--fit-t" && ok "S1 no LIMIT / no FIT_T: the eval commands carry no --limit, no --fit-t, no --tasks on the native sets (byte-identical night path)" || bad "S1 limit/fit-t leak: $EV | $EN"

echo "== S2 idempotent rerun: nothing re-runs, completion again =="
run_chain 0 1
[ "$(grep -c 'PRETRAIN-START' "$SB/w0.log")" = 0 ] && [ "$(grep -c 'ARM-SKIP .* (done)' "$SB/w0.log")" = 4 ] && [ "$(grep -c 'EVAL-OK' "$SB/w0.log")" = 0 ] && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && ok "S2 idempotent (4 arms skipped before their evals)" || bad "S2 idempotent"

echo "== S3 the EXTENSION rule: D0's peak at the end -> +EXT once, resumed; D1 early -> none; N0 never =="
mk_sandbox; run_chain 0 1 STUB_PEAK_LAST_ARM=D0,N0
grep -q "PRETRAIN-EXTEND D0: selected grid 8000 inside the last 2000 of 8000 -> +4000" "$SB/w0.log" && [ -f "$SB/repo/runs/pretraindecarc_D0/EXTENDED.txt" ] && grep -q "RESUMED from runs/pretraindecarc_D0/ckpt_latest.pkl at step 8000" "$SB/repo/runs/pretraindecarc_D0.log" && ok "S3 D0 extended once, resumed" || bad "S3 D0 extension"
grep -q "PRETRAIN-NO-EXTEND D1" "$SB/w0.log" && ok "S3 D1 no extension (early peak)" || bad "S3 D1"
! grep -q "PRETRAIN-EXTEND N0\|PRETRAIN-NO-EXTEND N0" "$SB/w0.log" && grep -q "PRETRAIN-OK N0" "$SB/w0.log" && ok "S3 N0 never extends" || bad "S3 N0"
"$REAL_PY" -c "import json; v=json.load(open('$SB/repo/runs/pretraindecarc_D0/vsel.json')); assert v['step']==12000, v" && ok "S3 D0 re-selected after the extension (12000)" || bad "S3 D0 reselect"
rm -f "$SB/gcs/decarc/D0_PRETRAIN_OK" "$SB/gcs/decarc/D0_ARM_OK"; run_chain 0 1 STUB_PEAK_LAST_ARM=D0,N0
grep -q "PRETRAIN-EXTENDED-BUDGET D0 12000" "$SB/w0.log" && [ "$(grep -c 'PRETRAIN-EXTEND D0' "$SB/w0.log")" = 0 ] && ok "S3 a relaunch keeps the extended budget, no second extension" || bad "S3 relaunch"

echo "== S4 the 2-worker map: w0 D0 D2, w1 D1 N0; completion by either =="
mk_sandbox; run_chain 0 2; run_chain 1 2
grep -q "PRETRAIN-START D0" "$SB/w0.log" && grep -q "PRETRAIN-START D2" "$SB/w0.log" && ! grep -q "PRETRAIN-START D1" "$SB/w0.log" && grep -q "PRETRAIN-START D1" "$SB/w1.log" && grep -q "PRETRAIN-START N0" "$SB/w1.log" && ok "S4 map" || bad "S4 map"
[ "$(n_ok)" = 4 ] && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w1.log" && grep -q "DECARC-WORKER-DONE worker=0" "$SB/w0.log" && ok "S4 complete by w1; w0 waited" || bad "S4 complete"

echo "== S5 NaN one-shot amputation on D1 (STOPPED, evaluated on the last finite grid) =="
mk_sandbox; run_chain 0 1 STUB_NAN_ARM=D1
grep -q "PRETRAIN-NAN D1" "$SB/w0.log" && [ -f "$SB/repo/runs/pretraindecarc_D1/STOPPED.txt" ] && [ -f "$SB/gcs/decarc/D1_ARM_OK" ] && [ "$(n_ok)" = 4 ] && ok "S5 amputated, still evaluated, complete" || bad "S5"

echo "== S6 preflight failure: D2 (optional) SKIPPED; D0 (seed) aborts the worker =="
mk_sandbox; run_chain 0 1 STUB_PREFLIGHT_FAIL=D2
grep -q "PREFLIGHT-FAILED D2 (rc=1) -> SKIPPED" "$SB/w0.log" && [ -f "$SB/gcs/decarc/D2_SKIPPED" ] && [ "$(n_ok)" = 3 ] && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && ok "S6 optional arm skipped, complete" || bad "S6a"
mk_sandbox; run_chain 0 1 STUB_PREFLIGHT_FAIL=D0
grep -q "PREFLIGHT-FAILED D0 (rc=1) -> the night stops here" "$SB/w0.log" && grep -q "DECARC-PREFLIGHT-ABORT" "$SB/w0.log" && ! grep -q "PRETRAIN-START" "$SB/w0.log" && ok "S6b seed arm aborts before any launch" || bad "S6b"

echo "== S7 launch-time HBM OOM -> ONE --remat retry (N0; the DEC arms carry --remat by registry) =="
mk_sandbox; run_chain 0 1 STUB_OOM_ARM=N0
grep -q "PRETRAIN-OOM-RETRY-REMAT N0" "$SB/w0.log" && [ -f "$SB/repo/runs/pretraindecarc_N0/RETRY_REMAT.txt" ] && [ -f "$SB/gcs/decarc/N0_ARM_OK" ] && ok "S7 remat retry on N0" || bad "S7"
pargv D0 | grep -q -- "--remat" && ok "S7 the DEC arms launch with --remat by registry" || bad "S7 D0 remat flag"

echo "== S8 the rider: RIDER=1 claimed once, n-gated 422786; the marker makes the second worker skip =="
mk_sandbox; run_chain 0 1 RIDER=1
grep -q "RIDER-OK C2 full-set D64" "$SB/w0.log" && [ -f "$SB/gcs/decarc/evals/rider_C2_full_t64_OK" ] && [ -f "$SB/gcs/decarc/RIDER_CLAIM_w0" ] && ok "S8 rider ran, n-gated" || bad "S8 rider"
run_chain 1 1 RIDER=1
grep -q "RIDER-SKIP (done)" "$SB/w1.log" && ok "S8 rider idempotent" || bad "S8 rider rerun"

echo "== S9 a shard failure on D0's dev-30 -> ARM-PARTIAL; the rerun redoes only that row =="
mk_sandbox; run_chain 0 1 STUB_EVAL_FAIL="D0:dev30:1"
grep -q "EVAL-SHARD-FAILED D0_dev30" "$SB/w0.log" && grep -q "ARM-PARTIAL D0" "$SB/w0.log" && [ ! -f "$SB/gcs/decarc/D0_ARM_OK" ] && [ -f "$SB/gcs/decarc/evals/D0_valhard_OK" ] && [ -f "$SB/gcs/decarc/evals/D0_rg96_OK" ] && ok "S9 partial arm: the other rows banked" || bad "S9a"
run_chain 0 1 STUB_EVAL_FAIL="D0:dev30:1"
grep -q "EVAL-SKIP D0_valhard" "$SB/w0.log" && grep -q "EVAL-OK D0_dev30" "$SB/w0.log" && [ -f "$SB/gcs/decarc/D0_ARM_OK" ] && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && ok "S9b rerun redid only dev-30; complete" || bad "S9b"

echo "== S10 missing data -> DATA-ABORT before any launch =="
NO_DATA=1 mk_sandbox; run_chain 0 1
grep -q "DATA-MISSING" "$SB/w0.log" && grep -q "DECARC-DATA-ABORT" "$SB/w0.log" && ! grep -q "PRETRAIN-START\|PREFLIGHT" "$SB/w0.log" && ok "S10 data abort" || bad "S10"

echo "== S11 LIVE BANK: D0 in flight at step 2000 on the live prefix, FRESH node -> RESUMED =="
mk_sandbox
( cd "$SB/repo" && mkdir -p runs && env PATH="$SB/bin:$PATH" "$SB/bin/stubpy" tools/pretrain.py --out runs/pretraindecarc_D0 --steps 2000 --cell decarc --monitor-every 2000 --grid-every 2000 >/dev/null 2>&1
  mkdir -p "$SB/gcs/decarc/live/runs/pretraindecarc_D0"; cp runs/pretraindecarc_D0/* "$SB/gcs/decarc/live/runs/pretraindecarc_D0/"; rm -rf runs )
run_chain 0 1
grep -q "LIVE-RESTORE pulled=" "$SB/w0.log" && grep -q "RESUMED from runs/pretraindecarc_D0/ckpt_latest.pkl at step 2000" "$SB/repo/runs/pretraindecarc_D0.log" && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && ok "S11 live-bank restore + RESUMED" || bad "S11"

echo "== S12 the registry guard: warmup >= the DEC steps -> REGISTRY-ABORT before any launch =="
mk_sandbox; run_chain 0 1 DA_WARMUP=8000
grep -q "REGISTRY-BAD warmup 8000 >= DEC steps 8000" "$SB/w0.log" && grep -q "DECARC-REGISTRY-ABORT" "$SB/w0.log" && ! grep -q "PREFLIGHT\|PRETRAIN-START" "$SB/w0.log" && ok "S12 registry guard aborts before launch" || bad "S12"
mk_sandbox; run_chain 0 1 DA_WARMUP=200
pargv D0 | grep -q -- "--warmup 200" && [ "$(n_ok)" = 4 ] && ok "S12b the warmup knob reaches the trainer" || bad "S12b"

echo "== S13 the eval STALL watchdog: a silent shard is killed after EVAL_STALL_SEC and retried once (resume-safe) =="
mk_sandbox; run_chain 0 1 STUB_EVAL_STALL="D0:dev30:1"
grep -q "EVAL-STALLED D0_dev30" "$SB/w0.log" && grep -q "EVAL-STALL-RETRY D0_dev30" "$SB/w0.log" && grep -q "EVAL-OK D0_dev30" "$SB/w0.log" && [ -f "$SB/gcs/decarc/D0_ARM_OK" ] && [ "$(n_ok)" = 4 ] && ok "S13 stall killed, retried, banked; complete" || bad "S13 ($(grep -c 'EVAL-STALLED D0_dev30' "$SB/w0.log") stalled, $(grep -c 'EVAL-OK D0_dev30' "$SB/w0.log") ok)"
"$REAL_PY" -c "import json; s=json.load(open('$SB/repo/runs/decarceval_D0/dev30/summary.json')); assert s['n_tasks']==30, s" && ok "S13 the retried set has every task exactly once" || bad "S13 n-gate after retry"
mk_sandbox; run_chain 0 1 STUB_EVAL_STALL="D0:dev30:1" STUB_EVAL_STALL_ALWAYS=1
[ "$(grep -c 'EVAL-STALLED D0_dev30' "$SB/w0.log")" = 2 ] && grep -q "EVAL-SHARD-FAILED D0_dev30" "$SB/w0.log" && grep -q "ARM-PARTIAL D0" "$SB/w0.log" && [ ! -f "$SB/gcs/decarc/D0_ARM_OK" ] && ok "S13b a persistent stall fails the row after one retry (ARM-PARTIAL, rerunnable)" || bad "S13b"
mk_sandbox; run_chain 0 1 STUB_EVAL_STALL="N0:valhard:"
grep -q "EVAL-STALLED N0_valhard" "$SB/w0.log" && grep -q "EVAL-OK N0_valhard" "$SB/w0.log" && [ -f "$SB/gcs/decarc/N0_ARM_OK" ] && ok "S13c the native single-process set is watched too" || bad "S13c"

echo "== S14 the PILOT LIMIT: DA_LIMIT=8 -> every eval set is its first 8 tasks, every n-gate = 8, completion =="
mk_sandbox; run_chain 0 1 DA_LIMIT=8
grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && [ "$(n_ok)" = 4 ] && ok "S14 complete with 4 ARM_OK under LIMIT 8" || bad "S14 complete ($(n_ok) ok)"
"$REAL_PY" -c "import json; [ (lambda s: (_ for _ in ()).throw(AssertionError(s)) if s['n_tasks']!=8 else None)(json.load(open(f'$SB/repo/runs/decarceval_D0/{x}/summary.json'))) for x in ('valhard','dev30','rg96','rt48','arc1eval','valhard_final')]; s=json.load(open('$SB/repo/runs/decarceval_N0/arc1eval/summary.json')); assert s['n_tasks']==8 and s['shards']==4, s" && ok "S14 every D0 set + the N0 public row n-gated at 8" || bad "S14 n-gates"
EV=$(eargv "$SB/repo/runs/decarceval_D0/valhard/provenance_0.json"); echo "$EV" | grep -q -- "--limit 8" && ok "S14 eval_decarc carries --limit 8" || bad "S14 --limit: $EV"
EN=$(eargv "$SB/repo/runs/decarceval_N0/valhard/summary.json"); ED=$(eargv "$SB/repo/runs/decarceval_N0/dev30/summary.json"); [ "$(echo "$EN" | sed -n "s/.*--tasks \([^ ]*\).*/\1/p" | tr "," "\n" | wc -l | tr -d " ")" = 8 ] && [ "$(echo "$ED" | sed -n "s/.*--tasks \([^ ]*\).*/\1/p" | tr "," "\n" | wc -l | tr -d " ")" = 8 ] && echo "$ED" | grep -q "1e0a9b12" && ok "S14 the native val-hard / dev-30 rows carry --tasks with 8 ids (dev-30 from the manifest literal)" || bad "S14 native --tasks: $EN | $ED"
[ "$(cat "$SB/repo/runs/decarceval_N0/rg96/results.jsonl" | wc -l | tr -d " ")" = 8 ] && ok "S14 rg-96 native row = 8 tasks" || bad "S14 rg96 rows"

echo "== S15 the FIT_T knob: DA_FIT_T=1 -> every fit (DEC shards, the native sets, the native public row) carries --fit-t 1 =="
mk_sandbox; run_chain 0 1 DA_FIT_T=1
grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && [ "$(n_ok)" = 4 ] && ok "S15 complete under FIT_T 1" || bad "S15 complete ($(n_ok) ok)"
EV=$(eargv "$SB/repo/runs/decarceval_D0/valhard/provenance_0.json"); EG=$(eargv "$SB/repo/runs/decarceval_D0/arc1eval/provenance_0.json"); EN=$(eargv "$SB/repo/runs/decarceval_N0/valhard/summary.json"); EP=$(eargv "$SB/repo/runs/decarceval_N0/arc1eval/s0/summary.json"); echo "$EV" | grep -q -- "--fit-t 1" && echo "$EG" | grep -q -- "--fit-t 1" && echo "$EN" | grep -q -- "--fit-t 1" && echo "$EP" | grep -q -- "--fit-t 1" && echo "$EV" | grep -q -- "--t-total 16" && ok "S15 --fit-t 1 on the DEC sets, the native sets and the native public row; --t-total untouched" || bad "S15 fit-t: $EV | $EG | $EN | $EP"
EB=$(eargv "$SB/repo/runs/decarceval_N0/valhard_bridge/summary.json"); ! echo "$EB" | grep -q -- "--fit-t" && echo "$EB" | grep -q -- "--k 0" && ok "S15 the bridge row keeps the DEPLOYED fit (no --fit-t) while the night rows carry --fit-t 1" || bad "S15 bridge: $EB"

echo "== S16 the COST PROBE budget: a 100 s/step fit -> COST-ABORT on D0 before its battery, the CHAIN-COST-ABORT marker, nothing else launched; the rerun stands down =="
mk_sandbox; run_chain 0 1 STUB_COST_S=100
grep -q "COST-ABORT D0 projected" "$SB/w0.log" && grep -q "DECARC-COST-ABORT worker=0 arm=D0" "$SB/w0.log" && [ -f "$SB/gcs/decarc/CHAIN-COST-ABORT" ] && [ ! -f "$SB/gcs/decarc/D0_COST_OK" ] && [ "$(grep -c "PRETRAIN-START" "$SB/w0.log")" = 1 ] && [ ! -f "$SB/gcs/decarc/evals/D0_valhard_OK" ] && ! grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && ok "S16 COST-ABORT before D0's battery; only D0's pretrain ran; marker set" || bad "S16 abort: $(grep -c PRETRAIN-START "$SB/w0.log") pretrains; $(ls "$SB/gcs/decarc/" | tr "\n" " ")"
run_chain 0 1 STUB_COST_S=100
grep -q "COST-ABORT-STANDING" "$SB/w0.log" && [ "$(grep -c "PRETRAIN-START" "$SB/w0.log")" = 0 ] && ok "S16 the rerun stands down on the marker" || bad "S16 rerun"
rm -f "$SB/gcs/decarc/CHAIN-COST-ABORT"; run_chain 0 1
grep -q "COST-OK D0" "$SB/w0.log" && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && [ "$(n_ok)" = 4 ] && ok "S16 marker removed + a sane cost -> the chain completes (D0's pretrain resumed from its grid)" || bad "S16 recovery ($(n_ok) ok)"

echo "== S17 the ARC env: the exact CHAIN_EXTRA_ENV of tools/campaign_decarc_arc.env (the ARC bucket; every knob reaches the tools; nothing written under the Sudoku prefix) =="
mk_sandbox
ARC_ENV=$(grep '^CHAIN_EXTRA_ENV=' "$REPO/tools/campaign_decarc_arc.env" | sed 's/^CHAIN_EXTRA_ENV="//; s/"$//')
echo "$ARC_ENV" | grep -q "GCS=gs://qhrrn2-arc/decarc" && echo "$ARC_ENV" | grep -q "GCS_SETS=gs://qhrrn2-arc/sets" && ok "S17 the ARC env names the ARC bucket for the markers and the sets" || bad "S17 env: $ARC_ENV"
BEFORE=$(cd "$SB/gcs" && find decarc champ -type f | sort | md5)
run_chain 0 1 $ARC_ENV
AFTER=$(cd "$SB/gcs" && find decarc champ -type f | sort | md5)
grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && [ "$(ls "$SB/gcs/arc/decarc/"*_ARM_OK 2>/dev/null | wc -l | tr -d ' ')" = 4 ] && [ -f "$SB/gcs/arc/decarc/decarc_final.tgz" ] && ok "S17 complete under gs://qhrrn2-arc/decarc (4 ARM_OK + the final tarball)" || bad "S17 complete ($(ls "$SB/gcs/arc/decarc/" 2>/dev/null | tr '\n' ' '))"
[ "$BEFORE" = "$AFTER" ] && ok "S17 NOTHING written under the Sudoku era's prefixes (decarc/, champ/ byte-identical)" || bad "S17 the Sudoku prefix changed"
A0=$(pargv D0); echo "$A0" | grep -q -- "--decarc-eval-start fieldfix" && ok "S17 the evaluation start reaches the trainer" || bad "S17 D0 argv: $A0"
EV=$(eargv "$SB/repo/runs/decarceval_D0/valhard/provenance_0.json"); echo "$EV" | grep -q -- "--fit-t 1" && echo "$EV" | grep -q -- "--steps 150" && echo "$EV" | grep -q -- "--k 16" && echo "$EV" | grep -q -- "--views 2" && echo "$EV" | grep -q -- "--start-rows buffers,symfix,rifix" && echo "$EV" | grep -q -- "--trace-fused 1" && ok "S17 val-hard: fit-t 1, 150 steps, k 16, views 2, the start rows, the fused trace" || bad "S17 val-hard flags: $EV"
EG=$(eargv "$SB/repo/runs/decarceval_D0/arc1eval/provenance_0.json"); echo "$EG" | grep -q -- "--views 1" && echo "$EG" | grep -q -- "--k 4" && ok "S17 the public row: views 1, k 4" || bad "S17 arc1eval flags: $EG"
grep -q "SELECT-KEYS D0 key=val_t16_ema key2=val_t16 key3=val_pix_ema plateau=0.02 start=fieldfix" "$SB/w0.log" && grep -qE "PRETRAIN-(NO-)?EXTEND D1.*plateau end" "$SB/w0.log" && ok "S17 the third key and the plateau reach the selection (the stub's flat monitor extends under the plateau, as S18 asserts)" || bad "S17 select: $(grep -E 'SELECT-KEYS D0|EXTEND D1' "$SB/w0.log" | head -2)"
grep -q "COST-OK D0 projected" "$SB/w0.log" && ! grep -q "COST-ABORT" "$SB/w0.log" && ok "S17 the cost probe accepts the battery under the 6 h budget (stub paces)" || bad "S17 cost: $(grep -E 'COST-' "$SB/w0.log" | head -3)"
[ -f "$SB/gcs/arc/decarc/evals/D0_mon96_OK" ] && [ -f "$SB/gcs/arc/decarc/evals/N0_valhard_bridge_OK" ] && ok "S17 the mon96 and bridge rows banked under the ARC prefix" || bad "S17 rows"

echo "== S18 the plateau-keyed extension: a flat declining monitor (every grid within 2 pp of the max) extends under DA_EXT_PLATEAU_PP and not without =="
mk_sandbox; run_chain 0 1 DA_EXT_PLATEAU_PP=0.02
grep -q "PRETRAIN-EXTEND D0: plateau end 8000 (within 0.02 of the max; selected grid 2000) inside the last 2000 of 8000 -> +4000" "$SB/w0.log" && [ -f "$SB/repo/runs/pretraindecarc_D0/EXTENDED.txt" ] && grep -q "plateau end 8000" "$SB/repo/runs/pretraindecarc_D0/EXTENDED.txt" && ok "S18 D0 extended on the plateau's end (the argmax at 2000 would not have)" || bad "S18 extend: $(grep -E 'PRETRAIN-(NO-)?EXTEND D0' "$SB/w0.log")"
"$REAL_PY" -c "import json; v=json.load(open('$SB/repo/runs/pretraindecarc_D0/vsel.json')); assert v['step']==2000, v" && ok "S18 the SELECTION is still the earliest maximum (2000): the plateau keys only the extension" || bad "S18 vsel"
! grep -q "PRETRAIN-EXTEND N0" "$SB/w0.log" && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && ok "S18 N0 never extends; complete" || bad "S18 N0/complete"
mk_sandbox; run_chain 0 1
grep -q "PRETRAIN-NO-EXTEND D0 (selected grid 2000 vs budget 8000 window 2000)" "$SB/w0.log" && ok "S18b without the knob the argmax rule stands (byte-identical message)" || bad "S18b: $(grep -E 'PRETRAIN-(NO-)?EXTEND D0' "$SB/w0.log")"

echo "== S19 the fused-trace cross-check fails on the 'chip' -> the eager fallback (--trace-fused 0), labeled; completion =="
mk_sandbox; run_chain 0 1 STUB_XCHECK_FAIL=1
grep -q "TRACE-FUSED-XCHECK-FAILED D0" "$SB/w0.log" && EV=$(eargv "$SB/repo/runs/decarceval_D0/valhard/provenance_0.json") && echo "$EV" | grep -q -- "--trace-fused 0" && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && [ "$(n_ok)" = 4 ] && ok "S19 fallback to the eager trace on every DEC row; complete" || bad "S19: $(grep -c TRACE-FUSED "$SB/w0.log") $(eargv "$SB/repo/runs/decarceval_D0/valhard/provenance_0.json" | grep -o -- '--trace-fused [01]')"
EG=$(eargv "$SB/repo/runs/decarceval_D0/arc1eval/provenance_0.json"); echo "$EG" | grep -q -- "--trace-fused 0" && [ -f "$SB/gcs/decarc/D0_TRACE_EAGER" ] && ok "S19 the fallback holds for the arm's later rows too; the marker persists it" || bad "S19 arc1eval: $EG"
rm -f "$SB/gcs/decarc/evals/D0_rt48_OK" "$SB/gcs/decarc/D0_ARM_OK"; run_chain 0 1
grep -q "TRACE-EAGER-STANDING D0" "$SB/w0.log" && grep -q "COST-SKIP D0" "$SB/w0.log" && ER=$(eargv "$SB/repo/runs/decarceval_D0/rt48/provenance_0.json") && echo "$ER" | grep -q -- "--trace-fused 0" && ok "S19b a rerun (the probe skipped) keeps the eager path for the re-done row" || bad "S19b rerun: $(grep -c TRACE-EAGER-STANDING "$SB/w0.log")"

echo "== S20 DA_MON96=0 DA_BRIDGE=0: the two rows are switches (off -> absent; every other row and completion unchanged) =="
mk_sandbox; run_chain 0 1 DA_MON96=0 DA_BRIDGE=0
[ ! -f "$SB/gcs/decarc/evals/D0_mon96_OK" ] && [ ! -f "$SB/gcs/decarc/evals/N0_valhard_bridge_OK" ] && [ -f "$SB/gcs/decarc/evals/D0_valhard_OK" ] && [ -f "$SB/gcs/decarc/evals/N0_valhard_OK" ] && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && [ "$(n_ok)" = 4 ] && ok "S20 rows off, the night otherwise unchanged, complete" || bad "S20"

echo "== S21 TWO PODS (separate nodes, one bucket): w0 = D0 D2 on node A, w1 = D1 N0 on node B; DA_SHARE_EXIT; a live prefix per pod =="
mk_sandbox; NA=$(mk_repo2 A); NB=$(mk_repo2 B)
run_chain_in "$NB" 1 2 DA_SHARE_EXIT=1 LIVE_PREFIX=gs://qhrrn2-rescue/decarc/live_w1
LB=$SB/nodeB_w1.log
grep -q "DECARC-SHARE-DONE worker=1 arms=\[D1 N0\]" "$LB" && [ -f "$SB/gcs/decarc/SHARE_DONE_w1" ] && ! grep -q "CHAIN-DECARC-COMPLETE" "$LB" && [ ! -f "$SB/gcs/decarc/decarc_final.tgz" ] && ok "S21 node B (the short share) banks SHARE_DONE_w1 and exits without the sentinel" || bad "S21 node B share: $(grep -E 'SHARE-DONE|COMPLETE|WORKER-DONE|INCOMPLETE' "$LB" | head -3)"
( cd "$NB" && cat runs/detached.exit 2>/dev/null ); grep -q "PRETRAIN-START D1" "$LB" && grep -q "PRETRAIN-START N0" "$LB" && ! grep -q "PRETRAIN-START D0\|PRETRAIN-START D2" "$LB" && ok "S21 node B ran only its share (D1, N0)" || bad "S21 node B arms"
[ -d "$SB/gcs/decarc/live_w1/runs/pretraindecarc_D1" ] || [ -d "$SB/gcs/decarc/live_w1/runs/pretraindecarc_N0" ] || [ -n "$(ls "$SB/gcs/decarc/live_w1/runs" 2>/dev/null)" ] && [ ! -d "$SB/gcs/decarc/live" ] && ok "S21 node B's live bank wrote only live_w1 (nothing under the default live/)" || bad "S21 live_w1: $(ls "$SB/gcs/decarc" | tr '\n' ' ')"
run_chain_in "$NA" 0 2 DA_SHARE_EXIT=1 LIVE_PREFIX=gs://qhrrn2-rescue/decarc/live_w0
LA=$SB/nodeA_w0.log
grep -q "CHAIN-DECARC-COMPLETE worker=0" "$LA" && [ -f "$SB/gcs/decarc/decarc_final.tgz" ] && [ "$(n_ok)" = 4 ] && [ ! -f "$SB/gcs/decarc/SHARE_DONE_w0" ] && ok "S21 node A (the long share) finds every arm done and finalizes (final + sentinel; no SHARE_DONE_w0)" || bad "S21 node A: $(grep -E 'SHARE-DONE|COMPLETE|INCOMPLETE' "$LA" | head -3)"
tar tzf "$SB/gcs/decarc/decarc_final.tgz" | grep -q "decarceval_D1/valhard/summary.json" && tar tzf "$SB/gcs/decarc/decarc_final.tgz" | grep -q "decarceval_N0/arc1eval/summary.json" && tar tzf "$SB/gcs/decarc/decarc_final.tgz" | grep -q "decarceval_D0/valhard/summary.json" && ok "S21 the final tarball carries both pods' rows (pulled from the bucket)" || bad "S21 final contents"
[ -n "$(ls "$SB/gcs/decarc/live_w0/runs" 2>/dev/null)" ] && ! ls "$SB/gcs/decarc/live_w0/runs" | grep -q "pretraindecarc_D1\|pretraindecarc_N0" && ok "S21 node A's live bank wrote only its own arms under live_w0" || bad "S21 live_w0: $(ls "$SB/gcs/decarc/live_w0/runs" 2>/dev/null | tr '\n' ' ')"
echo "== S21c the final gate: a banked row whose tarball is missing from the bucket -> FINAL-INCOMPLETE, exit 1, no final =="
mk_sandbox; NA=$(mk_repo2 A); NB=$(mk_repo2 B)
run_chain_in "$NB" 1 2 DA_SHARE_EXIT=1 LIVE_PREFIX=gs://qhrrn2-rescue/decarc/live_w1
rm -f "$SB/gcs/decarc/evals/N0_rt48.tgz"
run_chain_in "$NA" 0 2 DA_SHARE_EXIT=1 LIVE_PREFIX=gs://qhrrn2-rescue/decarc/live_w0
grep -q "DECARC-FINAL-INCOMPLETE worker=0 missing summaries: N0_rt48" "$SB/nodeA_w0.log" && [ ! -f "$SB/gcs/decarc/decarc_final.tgz" ] && ! grep -q "CHAIN-DECARC-COMPLETE" "$SB/nodeA_w0.log" && ok "S21c the incomplete final is refused (N0_rt48 named), nothing banked" || bad "S21c: $(grep -E 'FINAL-|COMPLETE' "$SB/nodeA_w0.log" | head -2)"
echo "== S21b the reverse order: node A (w0) finishes first -> SHARE_DONE_w0; node B (w1) finalizes =="
mk_sandbox; NA=$(mk_repo2 A); NB=$(mk_repo2 B)
run_chain_in "$NA" 0 2 DA_SHARE_EXIT=1 LIVE_PREFIX=gs://qhrrn2-rescue/decarc/live_w0
run_chain_in "$NB" 1 2 DA_SHARE_EXIT=1 LIVE_PREFIX=gs://qhrrn2-rescue/decarc/live_w1
grep -q "DECARC-SHARE-DONE worker=0 arms=\[D0 D2\]" "$SB/nodeA_w0.log" && [ -f "$SB/gcs/decarc/SHARE_DONE_w0" ] && grep -q "CHAIN-DECARC-COMPLETE worker=1" "$SB/nodeB_w1.log" && [ -f "$SB/gcs/decarc/decarc_final.tgz" ] && [ "$(n_ok)" = 4 ] && ok "S21b either order completes exactly once" || bad "S21b"
run_chain_in "$NA" 0 2 DA_SHARE_EXIT=1 LIVE_PREFIX=gs://qhrrn2-rescue/decarc/live_w0
grep -q "ARM-SKIP D0 (done)" "$SB/nodeA_w0.log" && grep -q "CHAIN-DECARC-COMPLETE worker=0" "$SB/nodeA_w0.log" && ! grep -q "PRETRAIN-START\|EVAL-OK" "$SB/nodeA_w0.log" && ok "S21b a relaunch of a finished pod re-runs nothing" || bad "S21b relaunch: $(grep -E 'PRETRAIN-START|EVAL-OK|COMPLETE|SHARE' "$SB/nodeA_w0.log" | head -3)"
echo "== S22 an ARM-PARTIAL exits 1 at once (OWN-ARMS-INCOMPLETE; no 20-h wait); the rerun completes =="
mk_sandbox; run_chain 0 1 STUB_EVAL_FAIL="D0:rt48:1" C1_WAIT_PASSES=600 C1_WAIT_SLEEP=120
grep -q "ARM-PARTIAL D0" "$SB/w0.log" && grep -q "DECARC-OWN-ARMS-INCOMPLETE worker=0" "$SB/w0.log" && ! grep -q "WORKER-DONE" "$SB/w0.log" && ok "S22 the partial arm exits at once instead of waiting (C1_WAIT_PASSES 600 x 120 s would be 20 h)" || bad "S22: $(grep -E 'PARTIAL|INCOMPLETE|WORKER-DONE' "$SB/w0.log" | head -3)"
run_chain 0 1 STUB_EVAL_FAIL="D0:rt48:1"
grep -q "EVAL-OK D0_rt48" "$SB/w0.log" && grep -q "CHAIN-DECARC-COMPLETE" "$SB/w0.log" && [ "$(n_ok)" = 4 ] && ok "S22b the rerun redoes only the missing row and completes" || bad "S22b"

echo "== RESULT: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
