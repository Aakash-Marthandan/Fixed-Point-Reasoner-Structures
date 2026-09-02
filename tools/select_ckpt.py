# Ledger: SPRINT S2 wave 3a (2026-08-23) — VAL-SELECTED checkpoint: among the
# BANKED checkpoints (ckpt_NNNNNN.pkl, every 5k steps) of a pretrain dir, pick
# the step with the best monitor val@t64 (metrics.jsonl {"monitor": ...} rows,
# written by tools/pretrain.py --monitor-every). Legitimate early stopping: the
# monitor puzzles are the 64 train-file rows disjoint from train AND test
# (never the test set). Ties -> the LATER step. Prints "NNNNNN val step" so
# the chain can name the ckpt; exit 1 when no banked ckpt has a monitor row.
"""  .venv/bin/python tools/select_ckpt.py runs/pretrainsport3a_A3  """
from __future__ import annotations
import json, re, sys
from pathlib import Path

def main():
    d = Path(sys.argv[1])
    # sportC1: --key names the monitor field to select on (val_t64 = the native
    # arms' raw weights; val_t64_ema / val_t16_ema = the EMA rows of R0 / X0)
    key = sys.argv[sys.argv.index("--key") + 1] if "--key" in sys.argv else "val_t64"
    banked = {int(re.search(r"ckpt_(\d+)\.pkl$", p.name).group(1)) for p in d.glob("ckpt_[0-9]*.pkl")}
    rows = []
    for l in (d / "metrics.jsonl").read_text().splitlines():
        try:
            r = json.loads(l)
        except Exception:
            continue
        if "monitor" in r and key in r["monitor"]:
            rows.append((int(r["monitor"]["step"]), float(r["monitor"][key])))
    cand = [(v, s) for s, v in rows if s in banked]
    if not cand:
        print("NONE", file=sys.stderr); sys.exit(1)
    v, s = max(cand, key=lambda x: (x[0], x[1]))   # best val; ties -> later step
    print(f"{s:06d} {v:.4f} {s}")

if __name__ == "__main__":
    main()
