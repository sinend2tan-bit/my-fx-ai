"""Regression tests for hourly forecast-window continuity."""

import pandas as pd

from aifx_time_validation import (
    forward_hourly_bar_gap_allowed,
    has_contiguous_hourly_window,
)


def test_consecutive_hourly_bars_are_allowed():
    idx = pd.date_range("2026-11-09 00:00", periods=9, freq="h", tz="UTC")
    assert has_contiguous_hourly_window(idx)


def test_friday_to_monday_market_closure_is_allowed():
    idx = pd.DatetimeIndex([
        "2026-11-06 20:00+00:00",  # Friday
        "2026-11-06 21:00+00:00",
        "2026-11-09 00:00+00:00",  # Monday
    ])
    assert has_contiguous_hourly_window(idx)


def test_weekday_missing_hour_is_rejected():
    idx = pd.DatetimeIndex([
        "2026-11-09 09:00+00:00",
        "2026-11-09 10:00+00:00",
        "2026-11-09 12:00+00:00",
    ])
    assert not has_contiguous_hourly_window(idx)


def test_duplicate_or_reverse_timestamps_are_rejected():
    friday = pd.Timestamp("2026-11-06 20:00+00:00")
    assert not forward_hourly_bar_gap_allowed(friday, friday)
    assert not forward_hourly_bar_gap_allowed(
        friday + pd.Timedelta(hours=1), friday
    )


def test_sunday_to_monday_gap_is_not_assumed_to_be_weekend_closure():
    idx = pd.DatetimeIndex([
        "2026-11-08 23:00+00:00",
        "2026-11-09 01:00+00:00",
    ])
    assert not has_contiguous_hourly_window(idx)



def _extract_app_forecast_updater():
    """Extract the production update_forecast_history function without running Streamlit UI."""
    import ast
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    import numpy as np

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "update_forecast_history"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "pd": pd,
        "datetime": datetime,
        "JST": ZoneInfo("Asia/Tokyo"),
        "LOOKAHEAD_BARS": 8,
        "has_contiguous_hourly_window": has_contiguous_hourly_window,
        "_normalise_timestamp": lambda value: pd.Timestamp(value).isoformat(),
        "save_forecast_history": lambda _items: None,
    }
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["update_forecast_history"]


def test_app_forecast_is_not_scored_across_weekday_gap():
    updater = _extract_app_forecast_updater()
    idx = pd.DatetimeIndex([
        "2026-11-09 09:00+00:00",
        "2026-11-09 10:00+00:00",
        "2026-11-09 12:00+00:00",
    ])
    frame = pd.DataFrame({"Close": [150.0, 150.1, 150.2]}, index=idx)
    item = {
        "ticker": "USDJPY=X",
        "interval": "1h",
        "status": "PENDING",
        "bar_timestamp": idx[0].isoformat(),
        "reference_close": 150.0,
        "signal": "BUY",
        "horizon_bars": 2,
    }
    result = updater([item], frame, "USDJPY=X", "1h")
    assert result[0]["status"] == "UNEVALUABLE_DISCONTINUOUS"
    assert "不連続" in result[0]["evaluation_note"]
    assert "evaluated_close" not in result[0]


def test_app_forecast_scoring_remains_available_for_contiguous_window():
    updater = _extract_app_forecast_updater()
    idx = pd.date_range("2026-11-09 09:00", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"Close": [150.0, 150.1, 150.2]}, index=idx)
    item = {
        "ticker": "USDJPY=X",
        "interval": "1h",
        "status": "PENDING",
        "bar_timestamp": idx[0].isoformat(),
        "reference_close": 150.0,
        "signal": "BUY",
        "horizon_bars": 2,
    }
    result = updater([item], frame, "USDJPY=X", "1h")
    assert result[0]["status"] == "CORRECT"
    assert result[0]["evaluated_close"] == 150.2
