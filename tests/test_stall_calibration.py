"""Regression checks for training-normalized, digit-conditional confidence."""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from stall_calibration import digit_probabilities, topk_empty_metrics
from qhrrn2.objective import log_stablemax


def test_training_normalizer_changes_confidence_without_changing_digit_rank():
    logits = np.zeros((2, 11), np.float32)
    logits[0, 1] = 5.0
    logits[1, 1:10] = [-4, 2, -1, 3, 0, 1, -2, -3, 4]
    stable = np.asarray(digit_probabilities(logits, "stablemax"))
    soft = np.asarray(digit_probabilities(logits, "softmax"))
    np.testing.assert_allclose(stable[0].max(), 6 / 14, atol=1e-6)
    np.testing.assert_allclose(soft[0].max(), np.exp(5) / (np.exp(5) + 8), atol=1e-6)
    np.testing.assert_array_equal(stable.argmax(-1), logits[:, 1:10].argmax(-1))
    np.testing.assert_array_equal(soft.argmax(-1), stable.argmax(-1))
    np.testing.assert_array_equal(np.argsort(stable[1]), np.argsort(soft[1]))
    assert soft[0].max() > .9 and stable[0].max() < .9
    trained_full = np.exp(np.asarray(log_stablemax(jnp.asarray(logits))))[:, 1:10]
    trained_conditional = trained_full / trained_full.sum(-1, keepdims=True)
    np.testing.assert_allclose(stable, trained_conditional, atol=1e-6)


@pytest.mark.parametrize("normalizer", ["stablemax", "softmax"])
def test_confidence_conditions_on_digits_and_ignores_special_token_logits(normalizer):
    logits = np.zeros((1, 11), np.float32)
    logits[0, 4] = 3.0
    expected = np.asarray(digit_probabilities(logits, normalizer))
    logits[0, [0, 10]] = [1e4, 2e4]
    actual = np.asarray(digit_probabilities(logits, normalizer))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-6)
    assert np.isfinite(actual).all()


def test_cross_cell_confidence_rank_is_not_invariant_to_normalization():
    # StableMax preserves digit ranking within a cell, not rankings between
    # cells. Subtracting each cell's max would incorrectly erase this example.
    logits = np.zeros((2, 11), np.float32)
    logits[0, 1:10] = [100] + [94] * 8
    logits[1, 1:10] = [5] + [0] * 8
    stable = np.asarray(digit_probabilities(logits, "stablemax")).max(-1)
    soft = np.asarray(digit_probabilities(logits, "softmax")).max(-1)
    assert stable[0] < stable[1]
    assert soft[0] > soft[1]


def test_topk_mask_excludes_givens_even_when_fewer_than_k_cells_are_empty():
    confidence = np.array([[1., .8, .7, .99], [1., 1., 1., 1.]])
    empty = np.array([[False, True, True, False], [False] * 4])
    pred = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    solution = np.array([[1, 2, 9, 4], [1, 2, 3, 4]])
    accuracy, mean_conf = topk_empty_metrics(confidence, pred, solution, empty, 5)
    np.testing.assert_allclose([accuracy[0], mean_conf[0]], [.5, .75])
    assert np.isnan(accuracy[1]) and np.isnan(mean_conf[1])


def test_unknown_normalization_fails_instead_of_silently_using_softmax():
    with pytest.raises(ValueError, match="Unsupported checkpoint loss_kind"):
        digit_probabilities(np.zeros((1, 11)), "unknown")
