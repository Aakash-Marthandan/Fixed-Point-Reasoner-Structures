# Ledger: cluster S plumbing (2026-08-12) — convert probe_sample --save-preds
# results (one or more dirs/shards, e.g. multi-init + Langevin runs) into the
# eval_c3 members-file schema {task: {"member_query_preds": [[grid/query]
# per member]}}. Candidates are deduped per query, ordered by visit count
# (heaviest first), capped at --max-members; short queries pad with the det
# pred so every member has one grid per query (eval_c3 indexes m[qi]).
"""
  python tools/samp_to_members.py --out runs/s_members.json \
      runs/scand_p1248c40k_s0 runs/scand_p1248c40k_s1 ...
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-members", type=int, default=32)
    a = ap.parse_args()

    tasks = {}  # tid -> [per-query dict key->(visits, grid)]
    det = {}    # tid -> [per-query det grid]
    for d in a.dirs:
        p = Path(d) / "results.jsonl"
        if not p.exists():
            print(f"missing {p} — skipped")
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            tid = r["task"]
            qs = tasks.setdefault(tid, [])
            dts = det.setdefault(tid, [])
            for qi, q in enumerate(r["queries"]):
                while len(qs) <= qi:
                    qs.append({})
                while len(dts) <= qi:
                    dts.append(None)
                if q.get("det_pred") is not None:
                    dts[qi] = q["det_pred"]
                for rec in q["sigmas"].values():
                    for c in rec.get("cands", []):
                        key = json.dumps(c["grid"])
                        n0, _ = qs[qi].get(key, (0, None))
                        qs[qi][key] = (n0 + c["n"], c["grid"])

    out = {}
    for tid, qs in tasks.items():
        per_q = []
        for qi, cand_map in enumerate(qs):
            ranked = sorted(cand_map.values(), key=lambda t: -t[0])
            grids = [g for _, g in ranked[:a.max_members]]
            if not grids:
                grids = [det[tid][qi]]
            per_q.append(grids)
        n_m = min(a.max_members, max(len(g) for g in per_q))
        members = []
        for m in range(n_m):
            members.append([
                per_q[qi][m] if m < len(per_q[qi]) else
                (det[tid][qi] if det[tid][qi] is not None else per_q[qi][0])
                for qi in range(len(per_q))])
        out[tid] = {"member_query_preds": members}
    Path(a.out).write_text(json.dumps(out))
    n_mem = {t: len(v["member_query_preds"]) for t, v in out.items()}
    print(f"wrote {a.out}: {len(out)} tasks, members "
          f"min {min(n_mem.values())} max {max(n_mem.values())}")


if __name__ == "__main__":
    main()
