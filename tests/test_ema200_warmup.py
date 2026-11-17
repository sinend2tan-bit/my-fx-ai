from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aifx_indicator_utils import ema_with_warmup


def test_ema200_is_unavailable_until_200_observations():
    prices = pd.Series(np.linspace(100.0, 120.0, 200))
    result = ema_with_warmup(prices, 200)
    assert result.iloc[:199].isna().all()
    assert np.isfinite(result.iloc[199])


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
    assert 'ema_with_warmup(close, 200)' in app_source
    assert 'ema_with_warmup(close, 200)' in scanner_source
    assert "MODEL_VERSION = 11" in app_source
    assert "MODEL_VERSION = 24" in scanner_source
