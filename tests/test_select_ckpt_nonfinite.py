"""Exercise the checkpoint-selection CLI without JAX or real checkpoints."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'select_ckpt.py'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('field', ['score', 'secondary', 'tertiary'])
def test_nonfinite_candidate_cannot_win_or_extend_plateau(tmp_path, bad, field):
    rows = []
    for step in [1, 2]:
        (tmp_path / f'ckpt_{step:06d}.pkl').touch()
        metrics = dict(step=step, score=.8, secondary=.7, tertiary=.6)
        if step == 2:
            metrics[field] = bad
        rows.append(json.dumps({'monitor': metrics}))
    (tmp_path / 'metrics.jsonl').write_text('\n'.join(rows))
    result = subprocess.run([
        sys.executable, str(SCRIPT), str(tmp_path), '--key', 'score',
        '--second-key', 'secondary', '--third-key', 'tertiary', '--plateau-pp', '.1',
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '000001 0.8000 1 1'
    assert 'non-finite' in result.stderr


def test_all_nonfinite_candidates_fail(tmp_path):
    (tmp_path / 'ckpt_000001.pkl').touch()
    (tmp_path / 'metrics.jsonl').write_text(json.dumps({
        'monitor': {'step': 1, 'val_t64': float('nan')},
    }))
    result = subprocess.run([
        sys.executable, str(SCRIPT), str(tmp_path),
    ], capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout == ''
    assert 'NONE' in result.stderr
