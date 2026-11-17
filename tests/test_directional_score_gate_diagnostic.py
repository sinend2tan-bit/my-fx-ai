"""Synthetic-only tests for the private 55% directional PnL diagnostic.

These tests protect the helper's documented BUY/SELL selection semantics.
They do not change or validate model accuracy, probability calibration, or
the production validation gate.
"""

import numpy as np
import pytest

import scanner


def test_directional_predictions_match_the_55_percent_score_rule():
    scores = np.array(
        [
            [0.20, 0.25, 0.55],  # BUY exactly at threshold
            [0.55, 0.20, 0.25],  # SELL exactly at threshold
            [0.45, 0.10, 0.45],  # tie: abstain
            [0.549, 0.001, 0.450],  # below threshold: abstain
            [0.40, 0.00, 0.60],  # BUY above threshold
            [0.60, 0.00, 0.40],  # SELL above threshold
        ],
        dtype=float,
    )

    actual = scanner.directional_predictions_from_scores(scores)

    np.testing.assert_array_equal(actual, np.array([1, -1, 0, 0, 1, -1]))


def test_directional_predictions_require_strict_directional_dominance():
    scores = np.array(
        [
            [0.55, 0.00, 0.45],
            [0.45, 0.00, 0.55],
            [0.55, 0.00, 0.55],
        ],
        dtype=float,
    )

    actual = scanner.directional_predictions_from_scores(scores)

    np.testing.assert_array_equal(actual, np.array([-1, 1, 0]))


@pytest.mark.parametrize(
    "scores",
    [
        np.array([0.2, 0.3, 0.5]),
        np.zeros((2, 2)),
        np.array([[0.2, np.nan, 0.8]]),
        np.array([[0.2, np.inf, 0.8]]),
        np.array([[-0.1, 0.2, 0.9]]),
        np.array([[0.2, 0.2, 1.1]]),
        np.array([[0.2, 0.2, 0.5]]),  # row sum below 1
        np.array([[0.2, 0.2, 0.7]]),  # row sum above 1
    ],
)
def test_directional_predictions_reject_malformed_scores(scores):
    with pytest.raises(ValueError):
        scanner.directional_predictions_from_scores(scores)


@pytest.mark.parametrize("threshold", [-0.1, 100.1, np.nan, np.inf])
def test_directional_predictions_reject_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        scanner.directional_predictions_from_scores(
            np.array([[0.2, 0.2, 0.6]], dtype=float),
            threshold_pct=threshold,
        )


def test_directional_predictions_allow_explicit_threshold_override():
    scores = np.array([[0.50, 0.10, 0.40]], dtype=float)

    actual = scanner.directional_predictions_from_scores(
        scores,
        threshold_pct=45.0,
    )

    np.testing.assert_array_equal(actual, np.array([-1]))
