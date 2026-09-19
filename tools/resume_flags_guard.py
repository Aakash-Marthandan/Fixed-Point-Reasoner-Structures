#!/usr/bin/env python3
"""THE RESUME FLAGS GUARD (2026-09-19; the attention-arm extension, Plan_2026-09-19_SA_Extension.md).  A resumed run continues the banked
optimizer state, EMA and RNG with WHATEVER flags it is launched with; a flag that differs from the banked run's would change the run silently.
This guard parses the exact argv the chain is about to pass to tools/pretrain.py with the trainer's own parser and compares every key to the
banked run's saved argv (config.json "argv" = vars(args) at its launch).  Keys in --allow may differ (the extension: steps; out is compared
as a path).  Exit 0 = identical apart from the allowed keys; exit 3 = a difference (each printed); exit 2 = unreadable input.

  python tools/resume_flags_guard.py --config runs/pretrainchamp_SA256/config.json --allow steps -- --out runs/pretrainchamp_SA256 <flags...> --steps 50000
  python tools/resume_flags_guard.py --selftest
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def same(x, y):
    if isinstance(x, bool) or isinstance(y, bool): return type(x) is type(y) and x == y      # True == 1 in Python: a bool matches only a bool
    if x is None or y is None or isinstance(x, str) or isinstance(y, str): return x == y
    if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)): return list(x or []) == list(y or [])
    try: return float(x) == float(y)
    except (TypeError, ValueError): return x == y


def compare(saved, now, allow):
    """keys whose values differ (a key present on one side only counts), minus the allowed keys; each as (key, saved, now)."""
    return [(k, saved.get(k, "<absent>"), now.get(k, "<absent>")) for k in sorted(set(saved) | set(now))
            if k not in allow and not same(saved.get(k, "<absent>"), now.get(k, "<absent>"))]


def parse_with_trainer(argv):
    import pretrain                                  # the trainer's own parser (module import runs nothing: main() is guarded)
    old = sys.argv
    try:
        sys.argv = ["pretrain.py"] + list(argv); return vars(pretrain.parse_args())
    finally:
        sys.argv = old


def main():
    if "--selftest" in sys.argv: return selftest()
    if "--" not in sys.argv: print("RESUME-GUARD-BAD-USAGE (the trainer argv follows --)"); return 2
    i = sys.argv.index("--"); ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--allow", default="steps")
    a = ap.parse_args(sys.argv[1:i]); argv = sys.argv[i + 1:]
    try:
        saved = json.loads(Path(a.config).read_text())["argv"]; assert isinstance(saved, dict)
    except Exception as e:
        print(f"RESUME-GUARD-UNREADABLE {a.config}: {e}"); return 2
    now = parse_with_trainer(argv); allow = {k.strip() for k in a.allow.split(",") if k.strip()}
    d = compare(saved, now, allow)
    if d:
        for k, s, n in d: print(f"RESUME-FLAGS-DIFFER {k}: banked {s!r} now {n!r}")
        return 3
    changed = [(k, saved.get(k), now.get(k)) for k in sorted(allow) if not same(saved.get(k), now.get(k))]
    print(f"RESUME-FLAGS-IDENTICAL {len(now)} keys compared; allowed changes: {', '.join(f'{k} {s}->{n}' for k, s, n in changed) or 'none'}")
    return 0


def selftest():
    n = 0
    assert same(1e-4, 0.0001) and same(True, True) and not same(True, 1) and same(None, None) and not same(None, 0) and same([1, 2], (1, 2)) and not same("a", "b"); n += 1
    s = dict(steps=30000, lr=1e-4, dp=True, out="runs/x", z0_device=None)
    assert compare(s, dict(s, steps=50000), {"steps"}) == []; n += 1
    assert compare(s, dict(s, steps=50000, lr=5e-4), {"steps"}) == [("lr", 1e-4, 5e-4)]; n += 1
    assert compare(s, {k: v for k, v in s.items() if k != "dp"}, {"steps"}) == [("dp", True, "<absent>")]; n += 1
    assert compare(s, dict(s, z0_device="cpu"), {"steps"}) == [("z0_device", None, "cpu")]; n += 1
    print(f"selftest OK: {n}/{n}"); return 0


if __name__ == "__main__":
    sys.exit(main())
