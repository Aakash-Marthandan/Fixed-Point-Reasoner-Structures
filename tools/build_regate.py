# Ledger: C20c (family-transfer gate set, registered 2026-08-10).
# Builds the FROZEN rg_ gate: 48 seeded families from the C20b gate-100
# (minus any dev-30 ids — dev-30 rules are a reserved claim set), each as
# one ARC-format JSON task: 3 support pairs + 3 query pairs, difficulty
# uniform (0,1), all instances verified. Deterministic given SEED; the
# committed checksum file (tools/re_gate48.sha256) makes any regeneration
# verifiable — the JSONs themselves live under git-ignored data/.
"""
  python tools/build_regate.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from qhrrn2 import rearc

SEED = 20260810
N_TASKS = 48
N_SUPPORT = 3
N_QUERY = 3
OUT = Path(__file__).resolve().parents[1] / "data" / "re_gate48"
SHA_FILE = Path(__file__).resolve().parent / "re_gate48.sha256"


def gate_families():
    import dev30
    _, gate = rearc.family_split()
    pool = [f for f in gate if f not in set(dev30.MANIFEST)]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pool))[:N_TASKS]
    return sorted(pool[i] for i in idx)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1)
    digest = hashlib.sha256()
    built = []
    for fam in gate_families():
        pairs = []
        tries = 0
        while len(pairs) < N_SUPPORT + N_QUERY and tries < 40:
            tries += 1
            p = rearc.sample_instance(fam, rng)
            if p is not None:
                pairs.append(p)
        if len(pairs) < N_SUPPORT + N_QUERY:
            print(f"  SKIP {fam}: generator refused ({len(pairs)} pairs)")
            continue
        task = {
            "train": [{"input": x.tolist(), "output": y.tolist()}
                      for x, y in pairs[:N_SUPPORT]],
            "test": [{"input": x.tolist(), "output": y.tolist()}
                     for x, y in pairs[N_SUPPORT:]],
        }
        blob = json.dumps(task, sort_keys=True, separators=(",", ":"))
        (OUT / f"rg_{fam}.json").write_text(blob)
        digest.update(f"rg_{fam}:".encode() + blob.encode())
        built.append(f"rg_{fam}")
    sha = digest.hexdigest()
    print(f"built {len(built)} gate tasks -> {OUT}")
    print(f"sha256 {sha}")
    return built, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify existing build against the committed sha")
    a = ap.parse_args()
    built, sha = build()
    if a.check:
        want = SHA_FILE.read_text().split()[0]
        assert sha == want, f"gate set drifted: {sha} != committed {want}"
        print("CHECK OK: matches committed sha")
    else:
        SHA_FILE.write_text(sha + f"  re_gate48 seed={SEED} n={len(built)}\n")
        print(f"sha written to {SHA_FILE}")
    print(",".join(built))


if __name__ == "__main__":
    main()
