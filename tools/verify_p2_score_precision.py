"""Verify P2 readouts after float32 casting, without modifying the original runner.

This is a numerical validation replay, not an additional experimental condition.
The original trajectory is instrumented only after constructing each edited state.
The default sample is the first 16 intervention puzzles, with all 16 iterations.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np
import rebuttal_p1p2 as original


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--width', type=int, choices=original.WIDTHS, required=True)
    ap.add_argument('--n', type=int, default=16)
    args = ap.parse_args()
    out = original.ROOT / 'runs/analysis/p2_precision_20260926'
    runner = original.R2(args.width, out)
    records = []

    def check(before, after, mode, iteration):
        jnp = runner.jnp
        v = jnp.asarray(runner.lm, jnp.float32)
        old = np.asarray(jnp.einsum('bfsw,w->bsf', jnp.asarray(before, jnp.float32), v))
        new = np.asarray(jnp.einsum('bfsw,w->bsf', after, v))
        # A projection/reconstruction sham checks casting round-off separately.
        par, perp = original.project(before, runner.lm)
        sham = np.asarray(jnp.einsum('bfsw,w->bsf', jnp.asarray(par + perp, jnp.float32), v))
        records.append(dict(mode=mode, iteration=iteration + 1,
                            max_score_shift=float(np.abs(old-new).max()),
                            max_sham_score_shift=float(np.abs(old-sham).max()),
                            changed_argmax_cells=int((old.argmax(-1)!=new.argmax(-1)).sum()),
                            scores=int(old.size)))

    source = textwrap.dedent(inspect.getsource(original.R2.trajectory))
    anchor = 'z = z.at[:, 0].set(jnp.asarray(h2, jnp.float32))'
    assert source.count(anchor) == 1
    instrumented = source.replace(anchor, anchor + '\n                _precision_check(h, z[:, 0], mode, t)')
    scope = dict(vars(original), _precision_check=check)
    exec(compile(instrumented, '<P2 precision-only instrumentation>', 'exec'), scope)
    trajectory = scope['trajectory'].__get__(runner, original.R2)
    with np.load(runner.STUDY / 'inputs/interventions.npz') as data:
        ids, puz = data['ids'][:args.n], data['puz'][:args.n]
    _, first, _ = runner.trajectory(puz, ids, steps=1)
    for mode in ('null_h0', 'null_random'):
        trajectory(puz, ids, branch=first, mode=mode)
        print('checked', args.width, mode, flush=True)
    result = dict(width=args.width, n=len(ids), steps=original.STEPS,
                  source_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
                  instrumentation_sha256=hashlib.sha256(instrumented.encode()).hexdigest(),
                  checkpoint_sha256=runner.meta['checkpoint_sha256'], ids=ids.tolist(),
                  max_score_shift=max(r['max_score_shift'] for r in records),
                  changed_argmax_cells=sum(r['changed_argmax_cells'] for r in records),
                  records=records)
    assert result['max_score_shift'] < 1e-2
    path = out / f'attention_{args.width}/precision.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','ids')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
