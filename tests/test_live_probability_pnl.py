"""Tests for the separate App-aligned 55% OOF reference filter."""

import numpy as np
import pytest

import scanner


def test_live_probability_filter_applies_55_percent_to_predicted_direction_only():
    predictions = np.array([-1, -1, 0, 1, 1, 1], dtype=int)
    probabilities = np.array([
        [0.55, 0.20, 0.25],  # SELL exactly at threshold: retain
        [0.549, 0.20, 0.251],  # SELL below threshold: abstain
        [0.10, 0.80, 0.10],  # NEUTRAL: no trade
        [0.20, 0.20, 0.60],  # BUY above threshold: retain
        [0.20, 0.25, 0.55],  # BUY exactly at threshold: retain
        [0.20, 0.251, 0.549],  # BUY below threshold: abstain
    ])

    original = predictions.copy()
    filtered = scanner.apply_live_probability_threshold(predictions, probabilities)

    assert filtered.tolist() == [-1, 0, 0, 1, 1, 0]
    assert np.array_equal(predictions, original)


def test_live_probability_filter_does_not_infer_opposite_direction():
    predictions = np.array([1, -1], dtype=int)
    probabilities = np.array([
        [0.70, 0.10, 0.20],  # Model predicted BUY, but BUY score is below 55%
        [0.20, 0.10, 0.70],  # Model predicted SELL, but SELL score is below 55%
    ])

    assert scanner.apply_live_probability_threshold(predictions, probabilities).tolist() == [0, 0]


@pytest.mark.parametrize(
    "predictions, probabilities",
    [
        ([1], [[0.2, 0.3]]),
        ([1, -1], [[0.2, 0.3, 0.5]]),
        ([1], [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]),
    ],
)
def test_live_probability_filter_rejects_invalid_probability_shape(predictions, probabilities):
    with pytest.raises(ValueError):
        scanner.apply_live_probability_threshold(predictions, probabilities)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_live_probability_filter_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        scanner.apply_live_probability_threshold([1], [[0.1, 0.1, 0.8]], threshold)
