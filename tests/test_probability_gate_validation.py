"""Parity tests for the Scanner's production 55% direction gate."""

import numpy as np
import pytest

import scanner


@pytest.mark.parametrize(
    ("sell_probability", "buy_probability", "expected"),
    [
        (0.20, 0.56, 1),
        (0.56, 0.20, -1),
        (0.10, 0.55, 1),
        (0.55, 0.10, -1),
        (0.54, 0.30, 0),
        (0.30, 0.54, 0),
        (0.55, 0.55, 0),
        (0.70, 0.20, -1),
        (0.20, 0.70, 1),
    ],
)
def test_direction_from_probabilities_matches_production_threshold(
    sell_probability, buy_probability, expected
):
    assert scanner.direction_from_probabilities(
        sell_probability, buy_probability
    ) == expected


@pytest.mark.parametrize(
    ("sell_probability", "buy_probability"),
    [
        (None, 0.90),
        (0.90, None),
        (float("nan"), 0.90),
        (0.90, float("inf")),
        (-0.10, 0.90),
    ],
)
def test_direction_from_probabilities_fails_closed_on_invalid_values(
    sell_probability, buy_probability
):
    assert scanner.direction_from_probabilities(
        sell_probability, buy_probability
    ) == 0
