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
# --set gate (default): unseen families (C20c). --set train: TRAINED families,
# fresh instances (the Q2 attribution cell, registered 2026-08-11) — an
# independent rng chain from the corpus sampler's (SEED+7 vs corpus seed 0),
# so instances are disjoint from pretraining draws.
# --set gateb (G4, rung-0 launch registration 2026-08-14): the DISJOINT
# second 48 — the SAME seeded permutation's indices [48:96] over the same
# 99-family gate pool, so rg-96 := frozen rg-48 ∪ rb-48 with the frozen set
# untouched bit-exact (historical comparability preserved; resolution
# doubled). perm_seed stays SEED for all gate-pool sets (disjointness by
# construction); the instance chain gets its own documented seed.
SETS = {
    "gate": dict(prefix="rg", out="re_gate48", sha="re_gate48.sha256",
                 seed=SEED, perm_seed=SEED, lo=0, hi=48),
    "train": dict(prefix="rt", out="re_train48", sha="re_train48.sha256",
                  seed=SEED + 7, perm_seed=SEED + 7, lo=0, hi=48),
    "gateb": dict(prefix="rb", out="re_gateb48", sha="re_gateb48.sha256",
                  seed=SEED + 13, perm_seed=SEED, lo=48, hi=96),
}


def families_for(which: str):
    import dev30
    train, gate = rearc.family_split()
    if which in ("gate", "gateb"):
        pool = [f for f in gate if f not in set(dev30.MANIFEST)]
    else:
        pool = sorted(set(train) - set(dev30.MANIFEST))  # = the 271 trained
    cfg = SETS[which]
    rng = np.random.default_rng(cfg["perm_seed"])
    idx = rng.permutation(len(pool))[cfg["lo"]:cfg["hi"]]
    return sorted(pool[i] for i in idx)


def build(which: str = "gate"):
    cfg = SETS[which]
    out_dir = Path(__file__).resolve().parents[1] / "data" / cfg["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"] + 1)
    digest = hashlib.sha256()
    built = []
    for fam in families_for(which):
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
        pref = cfg["prefix"]
        (out_dir / f"{pref}_{fam}.json").write_text(blob)
        digest.update(f"{pref}_{fam}:".encode() + blob.encode())
        built.append(f"{pref}_{fam}")
    sha = digest.hexdigest()
    print(f"built {len(built)} {which} tasks -> {out_dir}")
    print(f"sha256 {sha}")
    return built, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="gate", choices=list(SETS))
    ap.add_argument("--check", action="store_true",
                    help="verify existing build against the committed sha")
    a = ap.parse_args()
    cfg = SETS[a.set]
    sha_file = Path(__file__).resolve().parent / cfg["sha"]
    built, sha = build(a.set)
    if a.check:
        want = sha_file.read_text().split()[0]
        assert sha == want, f"{a.set} set drifted: {sha} != committed {want}"
        print("CHECK OK: matches committed sha")
    else:
        sha_file.write_text(sha + f"  {cfg['out']} seed={cfg['seed']} n={len(built)}\n")
        print(f"sha written to {sha_file}")
    print(",".join(built))


if __name__ == "__main__":
    main()
