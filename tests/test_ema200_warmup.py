from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aifx_indicator_utils import ema_with_warmup
import scanner


def test_ema200_is_unavailable_until_200_observations():
    prices = pd.Series(np.linspace(100.0, 120.0, 200))
    result = ema_with_warmup(prices, 200)
    assert result.iloc[:199].isna().all()
    assert np.isfinite(result.iloc[199])


def test_200_observation_gate_is_not_full_ema_convergence():
    def seed_residual(observations):
        baseline = pd.Series(np.full(observations, 100.0))
        perturbed = baseline.copy()
        perturbed.iloc[0] = 200.0
        baseline_ema = ema_with_warmup(baseline, 200).iloc[-1]
        perturbed_ema = ema_with_warmup(perturbed, 200).iloc[-1]
        return float(perturbed_ema - baseline_ema)

    # At the first visible output, the initial seed still contributes ~13.67%.
    assert seed_residual(200) == pytest.approx(
        100.0 * (1.0 - 2.0 / 201.0) ** 199
    )
    assert seed_residual(200) > 13.0
    # The contribution is still just above 1% at 461 observations, and
    # falls below 1% at 462 observations.
    assert seed_residual(461) > 1.0
    assert seed_residual(462) < 1.0


def test_short_segment_never_emits_a_partial_ema200():
    prices = pd.Series(np.linspace(100.0, 110.0, 199))
    result = ema_with_warmup(prices, 200)
    assert result.isna().all()


def test_ema_warmup_preserves_series_index_and_rejects_invalid_span():
    index = pd.date_range("2026-01-01", periods=205, freq="h", tz="UTC")
    prices = pd.Series(np.linspace(100.0, 105.0, len(index)), index=index)
    result = ema_with_warmup(prices, 200)
    assert result.index.equals(index)
    with pytest.raises(ValueError):
        ema_with_warmup(prices, 0)
    with pytest.raises(ValueError):
        ema_with_warmup(prices, True)


def test_app_and_scanner_both_use_the_shared_ema200_warmup():
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "app.py").read_text(encoding="utf-8")
    scanner_source = (root / "scanner.py").read_text(encoding="utf-8")
    assert "EMA200_MIN_PERIODS = 200" in app_source
    assert 'ema_with_warmup(close, EMA200_MIN_PERIODS)' in app_source
    assert "EMA200_MIN_PERIODS = 200" in scanner_source
    assert 'ema_with_warmup(close, EMA200_MIN_PERIODS)' in scanner_source
    assert "MODEL_VERSION = 11" in app_source
    assert "MODEL_VERSION = 24" in scanner_source


def test_scanner_raw_minimum_covers_all_walk_forward_folds_and_purge():
    # 959 = 199 EMA warm-up + 4 * (180 train + 8 purge) + 8 label horizon.
    assert scanner.MIN_WALK_FORWARD_FEATURE_ROWS == 752
    assert scanner.MIN_RAW_BARS_1H_FOR_WALK_FORWARD == 959
    fold_size = scanner.MIN_WALK_FORWARD_FEATURE_ROWS // (
        scanner.N_WALK_FORWARD_FOLDS + 1
    )
    assert fold_size - scanner.LOOKAHEAD_BARS == scanner.MIN_TRAIN_ROWS
