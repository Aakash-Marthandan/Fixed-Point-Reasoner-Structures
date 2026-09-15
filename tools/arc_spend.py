"""THE ARC PROJECT SPEND RECORD (2026-09-15; the PI: "Keep fresh record of the spend estimate on this google cloud project").

The lab's billing account is not visible to our account, so the spend is ESTIMATED from what we can measure at the source:
  * every node of ours (the OWN_PREFIX names) that ever existed in the ARC project: its start (the API's createTime, or a logged
    create request), its end (a logged teardown, a verified NOT_FOUND, or the first sweep that no longer lists it — an upper bound
    within one sweep), its zone and accelerator — kept in a persistent ledger (runs/arc_spend_ledger.json) so deleted nodes stay counted;
  * the program's v6e-8 spot rates (ops memory item 7; hand-derived from the Sudoku project's billing, 2026-08-30): US $6.82/h,
    Mumbai (asia-south1) $8.00/h — labeled INFERRED (the lab's list prices and discounts may differ);
  * the bucket's size (Standard, US multi-region, $0.026/GB-month) and a ROUGH inter-continental transfer estimate for nodes outside
    the US (the venv + data pulls per bring-up at $0.12/GB, the bucket's growth written from Asia at $0.08/GB).
BILLING ASSUMPTION (conservative): a node is charged from its create request to its deletion, CREATING minutes included; a create that
failed for capacity allocated nothing and is counted at $0.

Every run: lists our nodes in every ARC zone (read-only; other members' nodes are never listed, counted or written), merges them into the
ledger, closes the spans of nodes no longer listed (the pod logs' exact teardown time when one matches), and rewrites
runs/arc_spend.md (the human-readable record) + runs/arc_spend.json, appending one line to runs/arc_spend_log.txt.
The launchd watchdog runs it after every sweep (15 min), so the record stays fresh without any session; the heartbeat quotes the line.

  .venv/bin/python tools/arc_spend.py            # refresh + print the summary line
  .venv/bin/python tools/arc_spend.py --quiet    # refresh silently (the watchdog hook)
  .venv/bin/python tools/arc_spend.py --seed-json FILE   # add measured spans of nodes the tool never saw (idempotent by key)
  .venv/bin/python tools/arc_spend.py --selftest
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LEDGER = RUNS / "arc_spend_ledger.json"
REPORT_MD = RUNS / "arc_spend.md"
REPORT_JSON = RUNS / "arc_spend.json"
LOG = RUNS / "arc_spend_log.txt"
ARC_ZONES = ["us-east1-d", "us-east5-a", "us-east5-b", "us-central1-a", "us-central1-b", "us-central1-c", "us-west1-c",
             "us-south1-a", "europe-west4-a", "asia-south1-c"]   # every zone offering v6e in the ARC project (probed 2026-09-15)
RATES_V6E8 = {"us-": 6.82, "asia-south1": 8.00}                  # $/h per v6e-8 spot node; INFERRED (ops memory item 7)
RATE_SOURCE = "the program's v6e-8 spot rates (ops memory item 7, hand-derived 2026-08-30): US $6.82/h, Mumbai $8.00/h — inferred"
STORAGE_PER_GB_MONTH = 0.026   # Standard storage, US multi-region
XFER_TO_ASIA_PER_GB = 0.12     # bucket (US multi-region) -> a node in Asia (rough)
XFER_FROM_ASIA_PER_GB = 0.08   # a node in Asia -> the bucket (rough)
BRINGUP_PULL_GB = 0.12         # the ARC data tarball + the code archive per bring-up (the venv tarball measured and added when readable)
# the walls per pod, node-hours for the whole night (low, high). CORRECTED 2026-09-15 17:00Z from the MEASURED wall pace on the night's own
# pods (D0 1.93 it/s, D1 1.96 it/s from the metrics' timestamps; the pilot's 5.42 it/s was the trainer's printed rate, 2.8x high on the field
# loop): a DEC arm = pretrain 4.3 h + monitors 0.9 + fits 4.0-4.6 + traces 0.3 + probe/compile 0.5 ~ 10.1-10.7 h (+1.7 h if extended);
# pod0 = D0 + D2 ~ 20.2-25.9 h (27.9 h on the eager-trace fallback); pod1 = D1 + N0 ~ 11.4-13.7 h (N0's pace unmeasured on this image)
POD_WALL_BANDS_H = {"qhrrn2-arc-pod0": (20.2, 25.9), "qhrrn2-arc-pod1": (11.4, 13.7)}


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_time(s: str) -> dt.datetime:
    s = s.strip().replace("Z", "+00:00")
    m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(\+00:00)?$", s)
    if not m:
        raise ValueError(f"unparseable time {s!r}")
    return dt.datetime.fromisoformat(m.group(1)).replace(tzinfo=dt.timezone.utc)


def fmt(t: dt.datetime | None) -> str:
    return "—" if t is None else t.strftime("%Y-%m-%dT%H:%M:%SZ")


def rate_of(zone: str, accel: str) -> float | None:
    if accel != "v6e-8":
        return None
    for prefix, r in RATES_V6E8.items():
        if zone.startswith(prefix):
            return r
    return None


def local_policy() -> dict:
    p = Path(os.environ.get("GCP_LOCAL_ENV") or (ROOT / "tools" / ".gcp_local.env"))
    pol = {}
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            pol[k.strip()] = v.strip().strip('"').strip("'")
    return pol


# ---------- the sources ----------
def gcloud_json(args: list[str], pol: dict, timeout: int = 90):
    env = dict(os.environ)
    if pol.get("ARC_GCLOUD_CONFIG"):
        env["CLOUDSDK_ACTIVE_CONFIG_NAME"] = pol["ARC_GCLOUD_CONFIG"]
    r = subprocess.run(["gcloud", *args, "--format=json"], capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-300:])
    return json.loads(r.stdout or "[]")


def list_our_nodes(pol: dict, lister=None) -> tuple[list[dict], list[str]]:
    """-> (nodes of ours, zones that failed to list). A failed zone is NEVER read as empty."""
    lister = lister or (lambda z: gcloud_json(["compute", "tpus", "tpu-vm", "list", f"--zone={z}", f"--project={pol['ARC_PROJECT']}"], pol))
    prefix = pol.get("OWN_PREFIX", "qhrrn2-")
    out, failed = [], []
    for z in ARC_ZONES:
        try:
            rows = lister(z)
        except Exception:
            failed.append(z)
            continue
        for n in rows:
            name = str(n.get("name", "")).rsplit("/", 1)[-1]
            if not name.startswith(prefix):
                continue      # another member's node: never listed, counted or recorded
            accel = str(n.get("acceleratorType", "")).rsplit("/", 1)[-1]
            out.append({"name": name, "zone": z, "accel": accel, "state": n.get("state", "?"), "createTime": n.get("createTime")})
    return out, failed


TEARDOWN_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) \|\s+down rc=0")
DOWN_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) \| DOWN (\S+) in (\S+)")


def logged_teardown(name: str, zone: str, after: dt.datetime) -> dt.datetime | None:
    """The first `down rc=0` that follows a `DOWN <name> in <zone>` after the node's start, in runs/pod_<name>.log."""
    p = RUNS / f"pod_{name}.log"
    if not p.exists():
        return None
    pending = False
    for line in p.read_text(errors="replace").splitlines():
        m = DOWN_RE.match(line)
        if m and m.group(2) == name and m.group(3) == zone and parse_time(m.group(1)) >= after:
            pending = True
            continue
        m = TEARDOWN_RE.match(line)
        if m and pending:
            return parse_time(m.group(1))
    return None


# ---------- the ledger ----------
def key_of(n: dict) -> str:
    return f"{n['name']}@{n['zone']}@{n['start']}"


def merge(ledger: dict, seen: list[dict], failed_zones: list[str], now: dt.datetime, teardown=logged_teardown) -> dict:
    nodes = ledger.setdefault("nodes", {})
    live = set()
    for s in seen:
        start = fmt(parse_time(s["createTime"])) if s.get("createTime") else fmt(now)
        rec = {"name": s["name"], "zone": s["zone"], "accel": s["accel"], "start": start}
        k = key_of(rec)
        live.add(k)
        cur = nodes.get(k) or {**rec, "end": None, "end_source": None, "first_seen": fmt(now), "source": "the TPU API listing"}
        cur.update({"state": s["state"], "last_seen": fmt(now)})
        if cur.get("end") is not None:   # a record closed earlier but listed again (e.g. a slow deletion): reopen
            cur.update({"end": None, "end_source": None})
        nodes[k] = cur
    for k, rec in nodes.items():
        if k in live or rec.get("end") is not None:
            continue
        if rec["zone"] in failed_zones:
            continue   # an UNKNOWN read of its zone: never close a span on it
        t = teardown(rec["name"], rec["zone"], parse_time(rec["start"]))
        if t is not None:
            rec.update({"end": fmt(t), "end_source": "the supervisor's logged teardown"})
        else:
            rec.update({"end": fmt(now), "end_source": "the first sweep that no longer lists it (an upper bound within one sweep)"})
    return ledger


def seed(ledger: dict, rows: list[dict]) -> int:
    nodes = ledger.setdefault("nodes", {}); added = 0
    for r in rows:
        rec = {"name": r["name"], "zone": r["zone"], "accel": r.get("accel", "v6e-8"), "start": fmt(parse_time(r["start"]))}
        k = key_of(rec)
        if k in nodes:
            continue
        nodes[k] = {**rec, "end": fmt(parse_time(r["end"])) if r.get("end") else None, "end_source": r.get("end_source"),
                    "state": r.get("state", "DELETED" if r.get("end") else "?"), "source": r.get("source", "seeded"),
                    "note": r.get("note", ""), "billed": r.get("billed", True)}
        added += 1
    for r in rows:   # the failed-capacity creates are kept as a count (never billed)
        pass
    return added


# ---------- the estimate ----------
def estimate(ledger: dict, now: dt.datetime, bucket_bytes: int | None, venv_bytes: int | None, bucket_bytes_at_first_asia: int | None) -> dict:
    rows, compute, running_rate = [], 0.0, 0.0
    hours_by_pod: dict[str, float] = {}
    asia_bringups = 0
    for k, r in sorted(ledger.get("nodes", {}).items(), key=lambda kv: kv[1]["start"]):
        start = parse_time(r["start"]); end = parse_time(r["end"]) if r.get("end") else None
        h = max(((end or now) - start).total_seconds() / 3600.0, 0.0)
        rate = rate_of(r["zone"], r["accel"]) if r.get("billed", True) else 0.0
        cost = None if rate is None else h * rate
        if cost is not None:
            compute += cost
        if end is None and rate:
            running_rate += rate
        hours_by_pod[r["name"]] = hours_by_pod.get(r["name"], 0.0) + h
        if r["zone"].startswith("asia-") and r.get("billed", True):
            asia_bringups += 1
        rows.append({**r, "hours": round(h, 3), "rate": rate, "cost": None if cost is None else round(cost, 2)})
    storage = 0.0
    if bucket_bytes is not None:
        created = min((parse_time(r["start"]) for r in ledger.get("nodes", {}).values()), default=now)
        months = max((now - created).total_seconds() / (30 * 86400), 0.0)
        storage = bucket_bytes / 1e9 * STORAGE_PER_GB_MONTH * max(months, 1 / 30)
    xfer = 0.0
    if asia_bringups:
        pull_gb = BRINGUP_PULL_GB + (venv_bytes or 0) / 1e9
        xfer += asia_bringups * pull_gb * XFER_TO_ASIA_PER_GB
        if bucket_bytes is not None and bucket_bytes_at_first_asia is not None:
            xfer += max(bucket_bytes - bucket_bytes_at_first_asia, 0) / 1e9 * XFER_FROM_ASIA_PER_GB
    proj_lo = proj_hi = compute
    for pod, (lo, hi) in POD_WALL_BANDS_H.items():
        live_rate = sum((rate_of(r["zone"], r["accel"]) or 0.0) for r in ledger.get("nodes", {}).values() if r["name"] == pod and not r.get("end"))
        if not live_rate:
            continue   # a pod with no live node adds nothing to the projection (finished, stopped or between nodes)
        done = hours_by_pod.get(pod, 0.0)
        proj_lo += max(lo - done, 0.0) * live_rate
        proj_hi += max(hi - done, 0.0) * live_rate
    other = storage + xfer
    return {"as_of": fmt(now), "compute_to_date": round(compute, 2), "storage_to_date": round(storage, 2), "transfer_to_date": round(xfer, 2),
            "total_to_date": round(compute + other, 2), "running_rate_per_h": round(running_rate, 2),
            "projected_total_low": round(proj_lo + other, 2), "projected_total_high": round(proj_hi + other, 2),
            "bucket_gb": None if bucket_bytes is None else round(bucket_bytes / 1e9, 3), "rows": rows}


def render_md(est: dict, failed_zones: list[str], failed_creates: int) -> str:
    L = [f"# ARC project spend — ESTIMATE as of {est['as_of']}",
         "",
         f"**To date ≈ ${est['total_to_date']:.2f}** (compute ${est['compute_to_date']:.2f} · storage ${est['storage_to_date']:.2f} · transfer ${est['transfer_to_date']:.2f}). "
         f"Running now: ${est['running_rate_per_h']:.2f}/h. "
         f"Projected for the DEC-ARC night: ≈ ${est['projected_total_low']:.2f}–{est['projected_total_high']:.2f} (the registered walls per live pod; churn not included).",
         "",
         "The lab's billing account is not visible to us; this is an estimate from measured node lifetimes. Rates: " + RATE_SOURCE + ". "
         "A node counts from its create request to its deletion (CREATING minutes included, conservatively); a create refused for capacity allocated nothing and counts $0. "
         f"Storage: the bucket's size ({est['bucket_gb']} GB) at ${STORAGE_PER_GB_MONTH}/GB-month. Transfer: a rough inter-continental estimate for nodes outside the US.",
         ""]
    if failed_zones:
        L += [f"**UNKNOWN reads this refresh:** {', '.join(failed_zones)} (their nodes keep their last known state; a span is never closed on a failed read).", ""]
    L += ["| node | zone | start (UTC) | end (UTC) | hours | $/h | $ | end from |", "|---|---|---|---|---:|---:|---:|---|"]
    for r in est["rows"]:
        end = r["end"] or f"running ({r.get('state', '?')})"
        rate = "—" if r["rate"] is None else f"{r['rate']:.2f}"
        cost = "—" if r["cost"] is None else f"{r['cost']:.2f}"
        note = r.get("end_source") or ("—" if r["end"] is None else "")
        if r.get("note"):
            note = f"{note}; {r['note']}" if note else r["note"]
        L.append(f"| {r['name']} | {r['zone']} | {r['start']} | {end} | {r['hours']:.2f} | {rate} | {cost} | {note} |")
    L += ["", f"Creates refused for capacity (no node allocated, $0): {failed_creates}.", "",
          "Refreshed by the launchd watchdog after every sweep (15 min) and by the ops heartbeat; history in runs/arc_spend_log.txt."]
    return "\n".join(L) + "\n"


def count_failed_creates() -> int:
    n = 0
    for p in RUNS.glob("pod_qhrrn2-arc-*.log"):
        n += len(re.findall(r"\|\s+no capacity in ", p.read_text(errors="replace")))
    return n


def bucket_size(pol: dict, url: str) -> int | None:
    env = dict(os.environ)
    if pol.get("ARC_GCLOUD_CONFIG"):
        env["CLOUDSDK_ACTIVE_CONFIG_NAME"] = pol["ARC_GCLOUD_CONFIG"]
    try:
        r = subprocess.run(["gcloud", "storage", "du", "-s", url], capture_output=True, text=True, timeout=120, env=env)
        if r.returncode != 0:
            return None
        return int(r.stdout.split()[0])
    except Exception:
        return None


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def refresh(quiet: bool = False) -> dict:
    pol = local_policy()
    now = utcnow()
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"nodes": {}}
    seen, failed = list_our_nodes(pol)
    merge(ledger, seen, failed, now)
    bb = bucket_size(pol, "gs://qhrrn2-arc")
    vb = None
    try:
        env = dict(os.environ, CLOUDSDK_ACTIVE_CONFIG_NAME=pol.get("ARC_GCLOUD_CONFIG", ""))
        r = subprocess.run(["gcloud", "storage", "du", "-s", "gs://qhrrn2-arc/decarc/ops/venv_*"], capture_output=True, text=True, timeout=120, env=env)
        vb = int(r.stdout.split()[0]) if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        vb = None
    if bb is not None and ledger.get("bucket_bytes_at_first_asia") is None and any(
            n["zone"].startswith("asia-") for n in ledger["nodes"].values()):
        ledger["bucket_bytes_at_first_asia"] = bb
    est = estimate(ledger, now, bb, vb, ledger.get("bucket_bytes_at_first_asia"))
    atomic_write(LEDGER, json.dumps(ledger, indent=1))
    atomic_write(REPORT_JSON, json.dumps({**est, "unknown_zones": failed}, indent=1))
    atomic_write(REPORT_MD, render_md(est, failed, count_failed_creates()))
    line = (f"{est['as_of']} | to date ${est['total_to_date']:.2f} (compute {est['compute_to_date']:.2f}, storage {est['storage_to_date']:.2f}, "
            f"transfer {est['transfer_to_date']:.2f}) | running ${est['running_rate_per_h']:.2f}/h | projected ${est['projected_total_low']:.0f}-{est['projected_total_high']:.0f}"
            + (f" | UNKNOWN zones: {' '.join(failed)}" if failed else ""))
    with open(LOG, "a") as f:
        f.write(line + "\n")
    if not quiet:
        print(line)
    return est


# ---------- selftest ----------
def selftest():
    ok = 0
    t0 = parse_time("2026-09-15T16:00:00Z")
    led = {"nodes": {}}
    # 1. a live node listed by the API: counted from createTime; running rate; the projection adds its remaining band
    seen = [{"name": "qhrrn2-arc-pod0", "zone": "asia-south1-c", "accel": "v6e-8", "state": "READY", "createTime": "2026-09-15T16:00:00.5Z"}]
    merge(led, seen, [], t0 + dt.timedelta(hours=2), teardown=lambda *a: None)
    est = estimate(led, t0 + dt.timedelta(hours=2), None, None, None)
    assert abs(est["compute_to_date"] - 16.0) < 1e-6 and est["running_rate_per_h"] == 8.0, est; ok += 1
    other = est["storage_to_date"] + est["transfer_to_date"]   # an Asia bring-up adds its (rounded) transfer term to the projection too
    lo, hi = POD_WALL_BANDS_H["qhrrn2-arc-pod0"]   # the projection = the node's rate x the pod's band (the elapsed 2 h is inside the band)
    assert abs(est["projected_total_low"] - other - 8.0 * lo) < 0.02 and abs(est["projected_total_high"] - other - 8.0 * hi) < 0.02, est; ok += 1
    # 2. the node disappears; a logged teardown gives the exact end
    merge(led, [], [], t0 + dt.timedelta(hours=3), teardown=lambda n, z, a: parse_time("2026-09-15T18:30:00Z"))
    r = next(iter(led["nodes"].values()))
    assert r["end"] == "2026-09-15T18:30:00Z" and "logged teardown" in r["end_source"], r; ok += 1
    est = estimate(led, t0 + dt.timedelta(hours=5), None, None, None)
    assert abs(est["compute_to_date"] - 20.0) < 1e-6 and est["running_rate_per_h"] == 0.0 and est["projected_total_low"] == est["total_to_date"], est; ok += 1
    # 3. a failed read of the zone never closes a span; an absent node without a logged teardown closes at the sweep (upper bound)
    led2 = {"nodes": {}}
    merge(led2, [{"name": "qhrrn2-arc-pod1", "zone": "us-east1-d", "accel": "v6e-8", "state": "READY", "createTime": "2026-09-15T16:00:00Z"}], [], t0, teardown=lambda *a: None)
    merge(led2, [], ["us-east1-d"], t0 + dt.timedelta(hours=1), teardown=lambda *a: None)
    assert next(iter(led2["nodes"].values()))["end"] is None; ok += 1
    merge(led2, [], [], t0 + dt.timedelta(hours=1, minutes=15), teardown=lambda *a: None)
    r2 = next(iter(led2["nodes"].values()))
    assert r2["end"] == "2026-09-15T17:15:00Z" and "upper bound" in r2["end_source"]; ok += 1
    # 4. another member's node is never counted; the prefix filter
    nodes, failed = list_our_nodes({"ARC_PROJECT": "p", "OWN_PREFIX": "qhrrn2-"}, lister=lambda z: [
        {"name": f"projects/p/locations/{z}/nodes/labmate-tpu", "acceleratorType": "v6e-8", "state": "READY", "createTime": "2026-09-15T00:00:00Z"},
        {"name": f"projects/p/locations/{z}/nodes/qhrrn2-arc-pod0", "acceleratorType": "v6e-8", "state": "READY", "createTime": "2026-09-15T00:00:00Z"}] if z == "asia-south1-c" else [])
    assert [n["name"] for n in nodes] == ["qhrrn2-arc-pod0"] and not failed; ok += 1
    nodes, failed = list_our_nodes({"ARC_PROJECT": "p"}, lister=lambda z: (_ for _ in ()).throw(RuntimeError("denied")) if z == "us-east1-d" else [])
    assert failed == ["us-east1-d"] and nodes == []; ok += 1
    # 5. seeding is idempotent; an unbilled seed counts $0; rates by region
    led3 = {"nodes": {}}
    rows = [{"name": "qhrrn2-arc-canary", "zone": "us-east1-d", "start": "2026-09-15T15:24:33Z", "end": "2026-09-15T15:40:27Z", "end_source": "verified NOT_FOUND"},
            {"name": "x", "zone": "us-east1-d", "start": "2026-09-15T15:00:00Z", "end": "2026-09-15T16:00:00Z", "billed": False}]
    assert seed(led3, rows) == 2 and seed(led3, rows) == 0; ok += 1
    est = estimate(led3, t0 + dt.timedelta(hours=1), None, None, None)
    assert abs(est["compute_to_date"] - (15 * 60 + 54) / 3600 * 6.82) < 0.01, est; ok += 1
    assert rate_of("asia-south1-c", "v6e-8") == 8.0 and rate_of("us-central1-b", "v6e-8") == 6.82 and rate_of("europe-west4-a", "v6e-8") is None and rate_of("us-east1-d", "v6e-16") is None; ok += 1
    # 6. storage and transfer lines: an Asia node adds its bring-up pulls and the bucket's growth written from Asia
    led4 = {"nodes": {}}
    seed(led4, [{"name": "qhrrn2-arc-pod0", "zone": "asia-south1-c", "start": "2026-09-15T16:00:00Z", "end": "2026-09-15T17:00:00Z"}])
    est = estimate(led4, t0 + dt.timedelta(hours=1), bucket_bytes=int(12e9), venv_bytes=int(2e9), bucket_bytes_at_first_asia=int(2e9))
    assert abs(est["transfer_to_date"] - ((0.12 + 2.0) * 0.12 + 10.0 * 0.08)) < 0.01 and est["storage_to_date"] > 0, est; ok += 1
    md = render_md(est, ["us-east5-a"], 3)
    assert "UNKNOWN reads" in md and "Creates refused for capacity (no node allocated, $0): 3." in md and "| qhrrn2-arc-pod0 | asia-south1-c |" in md; ok += 1
    print(f"arc_spend selftest OK: {ok}/{ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--seed-json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.seed_json:
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"nodes": {}}
        n = seed(ledger, json.load(open(a.seed_json)))
        atomic_write(LEDGER, json.dumps(ledger, indent=1))
        print(f"seeded {n} span(s)")
    refresh(quiet=a.quiet)


if __name__ == "__main__":
    main()
