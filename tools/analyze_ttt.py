# Ledger: wave-2 READOUT 5 (2026-08-14) — the convert-phase TTT adjudication
# as registered 2026-08-13 (cluster V entry): probe_lora {plain, --kl 1.0,
# --ewc 1.0} x 48 val-hard on pretrain12_48c_40k, 2000 steps, selection =
# earliest-exact-else-final (NOTE: the [H-12] selection caveat carries — the
# best-checkpoint retest fix was NOT in this chain; alpha/rank=2 likewise).
#   PREDICTION: ewc >= kl >= plain on retention at comparable exactness.
#   DECISION:   winner becomes the convert-phase TTT default.
#   KILL:       ewc <= plain => curvature-then != damage-now, KL stands alone.
# Result sets are UNIONS (dedup-last) across the chain + shard + finishing
# dirs per the Day-3 close manifest. Comparators: same-substrate keyhole
# battery (lad_p1248c40k, e_t-only 600-step fits) computed live; the [H-12]
# LoRA cell (32c substrate: ret 8, exact 6) quoted with its substrate caveat.
"""
  .venv/bin/python tools/analyze_ttt.py
"""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "ttt_adjudication_20260814.txt"

UNION = {
    "plain": ["ttt_plain_p1248c", "ttt_plain_fin"],
    "kl":    ["ttt_kl_p1248c", "ttt_kl_s3", "ttt_kl_s4", "ttt_kl_fin"],
    "ewc":   ["ttt_ewc_p1248c", "ttt_ewc_fin"],
}

LINES: list[str] = []


def say(s=""):
    LINES.append(s)
    print(s)


def load_union(dirs):
    rows, dups = {}, 0
    for d in dirs:
        p = RUNS / d / "results.jsonl"
        for line in p.read_text().splitlines():
            r = json.loads(line)
            dups += r["task"] in rows
            rows[r["task"]] = r          # dedup-last
    return rows, dups


def keyhole_comparator():
    p = RUNS / "lad_p1248c40k" / "results.jsonl"
    ret = ex = n = 0
    for line in p.read_text().splitlines():
        r = json.loads(line)
        for q in r["queries"]:
            ret += q["gt_retention"]; ex += q["exact_T"]; n += 1
    return ret, ex, n


def main():
    say("=" * 92)
    say("READOUT 5 — CONVERT-PHASE TTT ADJUDICATION (plain / kl / ewc on p1248c40k)")
    say("=" * 92)
    arms = {}
    for mode, dirs in UNION.items():
        rows, dups = load_union(dirs)
        per = {}
        ret = ex = 0
        for t, r in rows.items():
            for qi, q in enumerate(r["queries"]):
                per[(t, qi)] = (bool(q["gt_retention"]), bool(q["exact_T"]))
                ret += q["gt_retention"]; ex += q["exact_T"]
        sel = Counter("final" if r["sel_step"] >= 2000 else "early"
                      for r in rows.values())
        arms[mode] = dict(rows=rows, per=per, ret=ret, ex=ex, dups=dups, sel=sel)
        say(f"{mode:6s} tasks={len(rows)} dups_overwritten={dups} "
            f"pairs={len(per)} sel(early/final)={sel['early']}/{sel['final']} "
            f"n_adapted={next(iter(rows.values()))['n_adapted']}")

    tasksets = [frozenset(a["rows"]) for a in arms.values()]
    say(f"task-set identical across modes: {len(set(tasksets)) == 1}")

    kret, kex, kn = keyhole_comparator()
    say()
    say(f'{"arm":8s} {"exact@sel":>9s} {"GT-ret":>7s}')
    say(f'{"keyhole":8s} {kex:>9d} {kret:>7d}   (same-substrate e_t-only 600-step, exact@T conv.)')
    for mode in ("plain", "kl", "ewc"):
        a = arms[mode]
        say(f'{mode:8s} {a["ex"]:>9d} {a["ret"]:>7d}')
    say(f'{"[H-12]":8s} {6:>9d} {8:>7d}   (LoRA on 32c substrate — cross-substrate caveat)')

    # pairwise McNemar on retention
    say()
    say("pairwise retention flips (b = first-only, c = second-only, exact binomial p):")
    for m1, m2 in (("plain", "kl"), ("plain", "ewc"), ("kl", "ewc")):
        pa, pb = arms[m1]["per"], arms[m2]["per"]
        keys = set(pa) & set(pb)
        b = sum(1 for k in keys if pa[k][0] and not pb[k][0])
        c = sum(1 for k in keys if pb[k][0] and not pa[k][0])
        n = b + c
        p = (sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2) if n else 1.0
        say(f"  {m1:6s} vs {m2:6s}: b={b:>2d} c={c:>2d} p={min(p,1):.4f}")

    # registered adjudication
    r = {m: arms[m]["ret"] for m in arms}
    e = {m: arms[m]["ex"] for m in arms}
    say()
    say(f"PREDICTION (ewc >= kl >= plain on retention): "
        f"{'HOLDS' if r['ewc'] >= r['kl'] >= r['plain'] else 'FAILS'} "
        f"(ret ewc {r['ewc']} / kl {r['kl']} / plain {r['plain']}; "
        f"exact ewc {e['ewc']} / kl {e['kl']} / plain {e['plain']})")
    say(f"KILL (ewc <= plain): {'FIRES — curvature-then != damage-now, KL stands alone' if r['ewc'] <= r['plain'] else 'does not fire'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n")
    say()
    say(f"artifact -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
