"""Bounded implementation validation for the recorded message-replay experiment.

No intervention outcome is regenerated or overwritten. Check all 256 first
states against the unmodified block at batch 64, and one intact 16-iteration
batch per checkpoint against that block and the archived batch-128 execution.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rebuttal_p2b import R4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/analysis/p2b_validation_20260926"


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return dict(bitwise=bool(np.array_equal(a, b)),
                max_abs=float(np.max(np.abs(a - b))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, required=True, choices=(128, 192, 256))
    w = ap.parse_args().width
    r = R4(w, out=OUT)
    hooked = r.DC._block
    with np.load(r.STUDY / "inputs/interventions.npz") as d:
        ids, puzzles = d["ids"], d["puz"]
    rows = []
    for b in range(0, 256, 64):
        pp, ii = puzzles[b:b+64], ids[b:b+64]
        r.DC._block = r.original_block
        native, state, _ = r.trajectory(pp, ii, steps=16 if b == 0 else 1)
        r.DC._block = hooked
        x, y, z, lg, pr, bank = r.first_step(pp)
        first = dict(carries=compare(z, state[1]), feedback=compare(y, state[0]),
                     logits=compare(lg, state[2]))
        assert all(v["bitwise"] for v in first.values()), first
        # Replay at the first iteration must preserve complete state, not just its readout.
        jnp = r.jnp
        void = r.jax.nn.one_hot(jnp.full((9, 9), r.G.VOID, jnp.int32), 11).transpose(2, 0, 1)
        yy, zz, ll, _, _ = r.one_step(x, jnp.broadcast_to(void, y.shape), None,
                                     True, "replay", bank, True)
        replay = dict(carries=compare(zz, z), feedback=compare(yy, y), logits=compare(ll, lg))
        assert all(v["bitwise"] for v in replay.values()), replay
        row = dict(batch_start=b, n=len(ii), first_state=first, replay_first_state=replay)
        if b == 0:
            recorded = [lg]
            for t in range(1, 16):
                y, z, ll, _, _ = r.one_step(x, y, z, False, "record", jnp.zeros(1), False)
                recorded.append(ll)
            recorded = np.stack(recorded)
            with np.load(r.STUDY / f"attention_{w}/interventions/intact/batch_0000.npz") as d:
                reference = d["logits"][:, :64]
            row["intact_record_vs_original_batch64"] = compare(recorded, native["logits"])
            row["intact_record_vs_archived_batch128"] = compare(recorded, reference)
            assert row["intact_record_vs_original_batch64"]["bitwise"], row
            assert row["intact_record_vs_archived_batch128"]["bitwise"], row
        rows.append(row)
        print(w, b, "passed", flush=True)
    source_paths = [Path(__file__), ROOT/"tools/rebuttal_p2b.py", ROOT/"tools/rebuttal_p1p2.py", r.path]
    result = dict(width=w, n_first_states=256, intact_n=64, intact_iterations=16,
                  passed=True, checks=rows,
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (r.dir / "validation.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
