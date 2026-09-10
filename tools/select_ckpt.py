# Ledger: SPRINT S2 wave 3a (2026-08-23) — VAL-SELECTED checkpoint: among the
# BANKED checkpoints (ckpt_NNNNNN.pkl, every 5k steps) of a pretrain dir, pick
# the step with the best monitor val@t64 (metrics.jsonl {"monitor": ...} rows,
# written by tools/pretrain.py --monitor-every). Legitimate early stopping: the
# monitor puzzles are the 64 train-file rows disjoint from train AND test
# (never the test set). Ties -> the LATER step. Prints "NNNNNN val step" so
# the chain can name the ckpt; exit 1 when no banked ckpt has a monitor row.
# CHAMPION NIGHT (2026-09-08; the Night A selection-instrument lesson: a 64-puzzle
# monitor ties at its maximum and the later-tie rule picked the memorization side):
#   --tie earliest       ties on the key -> the EARLIEST step (the trajectory law: the
#                        peak precedes memorization; default "later" = the pre-existing rule)
#   --second-key KEY     a second monitor field breaks ties on the first (the chain passes the
#                        RAW-weights monitor val_t16 under the EMA key val_t16_ema)
# The default call is byte-identical to the pre-existing behaviour.
"""  .venv/bin/python tools/select_ckpt.py runs/pretrainsport3a_A3  """
from __future__ import annotations
import json, re, sys
from pathlib import Path

def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

def main():
    d = Path(sys.argv[1])
    # sportC1: --key names the monitor field to select on (val_t64 = the native
    # arms' raw weights; val_t64_ema / val_t16_ema = the EMA rows of R0 / X0)
    key = _opt("--key", "val_t64")
    key2 = _opt("--second-key", None)
    tie = _opt("--tie", "later")
    # DEC-ARC BUILD (2026-09-10): --row val reads the native ARC loop's {"val": ...} rows (val20_eval: val_exact / val_total /
    # val_pix_mean) instead of the {"monitor": ...} rows; --key val_frac = val_exact / val_total. The default is byte-identical.
    row = _opt("--row", "monitor")
    assert row in ("monitor", "val"), row
    assert tie in ("later", "earliest"), tie
    banked = {int(re.search(r"ckpt_(\d+)\.pkl$", p.name).group(1)) for p in d.glob("ckpt_[0-9]*.pkl")}
    rows = []
    for l in (d / "metrics.jsonl").read_text().splitlines():
        try:
            r = json.loads(l)
        except Exception:
            continue
        if row in r:
            m = dict(r[row])
            if row == "val" and "val_total" in m:
                m["val_frac"] = float(m["val_exact"]) / max(int(m["val_total"]), 1)
            if key in m:
                v2 = float(m[key2]) if (key2 and key2 in m) else 0.0
                rows.append((int(m["step"]), float(m[key]), v2))
    cand = [(v, v2, s) for s, v, v2 in rows if s in banked]
    if not cand:
        print("NONE", file=sys.stderr); sys.exit(1)
    sgn = -1 if tie == "earliest" else 1
    v, v2, s = max(cand, key=lambda x: (x[0], x[1], sgn * x[2]))   # best val; second key; ties -> earliest/later step
    print(f"{s:06d} {v:.4f} {s}")

if __name__ == "__main__":
    main()
