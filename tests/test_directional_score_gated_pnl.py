"""Verify that forward PnL diagnostics apply the live directional score gate."""
import numpy as np
import pytest

import scanner


def test_score_gate_matches_live_buy_sell_threshold_and_tie_behavior():
    scores = np.array([
        [0.10, 0.20, 0.70],  # BUY above threshold
        [0.65, 0.20, 0.15],  # SELL above threshold
        [0.40, 0.05, 0.55],  # BUY qualifies and beats SELL
        [0.45, 0.10, 0.45],  # tie: no direction
        [0.54, 0.40, 0.06],  # SELL below threshold
        [0.20, 0.60, 0.20],  # neutral dominant
    ])
    assert scanner.directional_predictions_from_scores(scores).tolist() == [1, -1, 1, 0, 0, 0]


def test_score_gate_uses_same_configurable_threshold_for_both_sides():
    scores = np.array([[0.60, 0.20, 0.20], [0.20, 0.20, 0.60]])
    assert scanner.directional_predictions_from_scores(scores, 60).tolist() == [-1, 1]
    assert scanner.directional_predictions_from_scores(scores, 61).tolist() == [0, 0]


@pytest.mark.parametrize(
    "scores",
    [
        np.array([0.2, 0.8, 0.0]),
        np.array([[0.2, 0.8]]),
        np.array([[np.nan, 0.0, 1.0]]),
        np.array([[1.1, 0.0, 0.0]]),
    ],
)
def test_score_gate_rejects_malformed_or_nonfinite_scores(scores):
    with pytest.raises(ValueError):
        scanner.directional_predictions_from_scores(scores)


def test_score_gate_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        scanner.directional_predictions_from_scores(np.array([[0.2, 0.2, 0.6]]), 101)
