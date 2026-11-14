"""Synthetic tests for the live 55% direction threshold reference PnL.

The reference metric must remain separate from the existing OOF PnL and gates.
No real data, network, or model training is used.
"""

import numpy as np
import pandas as pd
import pytest

import scanner


class _FixedProbabilityModel:
    def __init__(self, buy_probability):
        self.buy_probability = float(buy_probability)

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([-1, 0, 1])
        self.feature_importances_ = np.full(X.shape[1], 1.0 / X.shape[1])
        return self

    def predict(self, X):
        # BUY remains the argmax even at 54%, but is below the live 55% gate.
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        buy = self.buy_probability
        neutral = 0.80 - buy
        sell = 0.20
        return np.tile([sell, neutral, buy], (len(X), 1))


def _validation_frame(n=1000):
    rng = np.random.default_rng(20261114)
    index = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        rng.normal(size=(n, len(scanner.FEATURE_COLUMNS))),
        columns=scanner.FEATURE_COLUMNS,
        index=index,
    )
    frame["Target"] = np.tile([-1, 0, 1], n // 3 + 1)[:n]
    frame["ATR"] = 1.0
    frame["ADX"] = 30.0
    frame["ATR_Ratio"] = 1.0
    frame["Future_Open_1"] = 150.0
    frame["Future_Close"] = 151.2
    for step in range(1, scanner.LOOKAHEAD_BARS + 1):
        frame[f"Future_High_{step}"] = 151.2
        frame[f"Future_Low_{step}"] = 150.2
    return frame


@pytest.mark.parametrize(
    ("buy_probability", "expected_reference_trades"),
    [(0.54, 0), (0.55, None)],
)
def test_live55_reference_pnl_applies_exact_55_percent_threshold_without_changing_gate(
    monkeypatch, buy_probability, expected_reference_trades
):
    monkeypatch.setattr(
        scanner,
        "RandomForestClassifier",
        lambda **kwargs: _FixedProbabilityModel(buy_probability),
    )

    result = scanner.walk_forward_validation(
        _validation_frame(),
        pair_name="米ドル/円 (USD/JPY)",
    )

    reference = result["simulated_pnl_live55_reference"]
    existing = result["simulated_pnl"]

    assert existing["trades"] > 0
    assert reference["trades"] == (
        existing["trades"] if expected_reference_trades is None
        else expected_reference_trades
    )
    assert result["total_pnl_trades"] == existing["trades"]
    assert all(
        "simulated_pnl_live55_reference" in fold
        for fold in result["fold_details"]
    )
    # The original ungated PnL still drives the existing gate and pass result.
    assert result["pnl_stability_ok"] is True
    assert result["passed"] is False
