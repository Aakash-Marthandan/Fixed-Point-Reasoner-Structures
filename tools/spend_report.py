# Ledger: spend instrument (PI directive 2026-08-14: "keep track of the
# spending"). Billed TPU time is DERIVED from the launchd watchdog's
# inventory log (runs/tpu_status_log.txt, 15-min polls since 2026-08-12)
# rather than estimated from memory: each poll line lists the nodes that
# existed at that instant, so a node's billed span is the number of polls it
# appears in x the poll interval. Accuracy is +-1 poll (15 min) per span.
#
# MEASURED vs INFERRED (the labeling law): the SPANS are measured from the
# log; the DOLLARS are inferred by multiplying spans by list spot rates —
# actual invoiced cost can differ (spot rates float; sustained-use and
# credits apply). Treat output as a planning tally, and reconcile against
# the console periodically.
#
# Rates: v6e-8 spot ~$6.5/h (8 chips), v5e-1 spot ~$0.55/h — the figures the
# budget architecture (2026-08-14) is written against.
"""
  .venv/bin/python tools/spend_report.py [--since YYYY-MM-DD]
"""
from __future__ import annotations
import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs" / "tpu_status_log.txt"
POLL = timedelta(minutes=15)

RATES = {"pod": 6.5, "default": 0.55}     # $/h by node-name class


def rate_for(name: str) -> float:
    return RATES["pod"] if "pod" in name else RATES["default"]


def parse(since: datetime | None):
    """-> {node: [(start, end)]} spans, from consecutive presence in polls."""
    seen: dict[str, datetime] = {}      # node -> first poll of current span
    last: dict[str, datetime] = {}      # node -> last poll seen present
    spans: dict[str, list[tuple[datetime, datetime]]] = {}
    prev_ts = None
    for line in LOG.read_text().splitlines():
        m = re.match(r"(\S+) \| (.*)", line.strip())
        if not m:
            continue
        ts = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
        body = m.group(2)
        present = set()
        if body != "none":
            for tok in body.split():
                # zone=node:STATE  (a zone may carry several node tokens).
                # BILLING STATE FILTER (fixed 2026-08-14 before first use):
                # only READY bills. A PREEMPTED node keeps appearing in
                # listings until it is deleted but its VM is stopped, and
                # CREATING has no allocated resource yet — counting either
                # inflates the tally (the first draft did, by ~2 spans).
                for node, state in re.findall(r"([A-Za-z0-9-]+):([A-Z]+)", tok):
                    if state == "READY":
                        present.add(node)
        for node in present:
            if node not in seen:
                seen[node] = ts
            last[node] = ts
        for node in list(seen):
            if node not in present:      # span closed at the previous poll
                spans.setdefault(node, []).append((seen[node], last[node] + POLL))
                del seen[node]
        prev_ts = ts
    for node, start in seen.items():     # still-open spans
        spans.setdefault(node, []).append((start, last[node] + POLL))
    if since:
        spans = {n: [(a, b) for a, b in v if b >= since] for n, v in spans.items()}
        spans = {n: v for n, v in spans.items() if v}
    return spans, prev_ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="YYYY-MM-DD (UTC)")
    a = ap.parse_args()
    since = (datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc)
             if a.since else None)
    spans, last_poll = parse(since)

    print("=" * 78)
    print("TPU SPEND — spans MEASURED from the watchdog inventory log (+-15 min);")
    print("            dollars INFERRED at list spot rates (v6e-8 $6.5/h, v5e-1 $0.55/h)")
    print("=" * 78)
    total = 0.0
    for node in sorted(spans):
        hrs = sum((b - a).total_seconds() for a, b in spans[node]) / 3600
        cost = hrs * rate_for(node)
        total += cost
        print(f"  {node:16s} {len(spans[node]):>2d} span(s)  {hrs:>5.2f} h  ${cost:>6.2f}")
        for s, e in spans[node]:
            print(f"      {s:%m-%d %H:%M} -> {e:%m-%d %H:%M}Z")
    print(f"  {'TOTAL':16s} {'':>2s}          {'':>5s}    ${total:>6.2f}")
    if last_poll:
        print(f"\n  last watchdog poll: {last_poll:%Y-%m-%d %H:%M}Z")
    print("  NOTE: the watchdog was installed 2026-08-12 — spend before that")
    print("  date is not derivable here (ledger entries carry it: ~$215 through 08-12).")


if __name__ == "__main__":
    main()
