"""Synthetic tests for App 1h timestamp continuity.

All OHLCV is deterministic and local. These tests do not access external data,
train a model, or change any model/gate configuration.
"""

import ast
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scanner

from test_stage1_input_validation import _app_namespace, _extract_function, make_frame


def _exec_app_function(name, namespace):
    node = _extract_function(name)
    code = compile(ast.Module(body=[node], type_ignores=[]), "app_extract", "exec")
    exec(code, namespace)


def test_app_gap_policy_matches_scanner_weekday_and_weekend_rules():
    ns = {"pd": pd}
    _exec_app_function("_app_forward_bar_gap_allowed", ns)
    app_allowed = ns["_app_forward_bar_gap_allowed"]

    pairs = [
        ("2026-01-07 00:00", "2026-01-07 01:00", True),
        ("2026-01-07 00:00", "2026-01-07 02:00", False),
        ("2026-01-09 00:00", "2026-01-12 08:00", True),   # 80h
        ("2026-01-09 00:00", "2026-01-12 09:00", False),  # 81h
        ("2026-01-09 23:00", "2026-01-13 00:00", False),  # Friday -> Tuesday
        ("2026-01-07 01:00", "2026-01-07 01:00", False),
    ]
    for left, right, expected in pairs:
        prev = pd.Timestamp(left, tz="UTC")
        nxt = pd.Timestamp(right, tz="UTC")
        assert app_allowed(prev, nxt) is expected
        assert app_allowed(prev, nxt) == scanner._forward_bar_gap_allowed(prev, nxt)


def test_app_1h_label_window_is_invalidated_by_weekday_gap():
    raw = make_frame(n=500, start="2026-01-05 00:00")
    gap_ts = raw.index[260]
    frame = raw.drop(index=[gap_ts])

    ns, _st = _app_namespace(lambda: frame.copy())
    _exec_app_function("_app_forward_bar_gap_allowed", ns)
    result = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")

    assert result is not None
    previous_ts = raw.index[259]
    next_ts = raw.index[261]
    assert next_ts - previous_ts == pd.Timedelta(hours=2)

    # Eight signal rows whose 8-bar forward windows cross the missing hour
    # remain visible for audit, but are excluded from training by Target=NaN.
    assert previous_ts in result.index
    assert pd.isna(result.loc[previous_ts, "Target"])
    assert next_ts in result.index
    assert pd.notna(result.loc[next_ts, "Target"])
    quality = result.attrs["data_quality"]
    assert quality["dropped_label_discontinuous"] == 8
    assert quality["dropped_label_insufficient_future"] == 8
    assert quality["trainable_rows"] == (
        quality["feature_rows"]
        - quality["dropped_label_discontinuous"]
        - quality["dropped_label_insufficient_future"]
    )


def _forecast_namespace():
    ns = {
        "pd": pd,
        "np": np,
        "datetime": datetime,
        "JST": ZoneInfo("Asia/Tokyo"),
        "LOOKAHEAD_BARS": 8,
        "_normalise_timestamp": lambda value: pd.Timestamp(value).isoformat(),
        "save_forecast_history": lambda items: None,
    }
    _exec_app_function("_app_forward_bar_gap_allowed", ns)
    _exec_app_function("update_forecast_history", ns)
    return ns


def _pending_forecast(timestamp):
    return {
        "ticker": "USDJPY=X",
        "interval": "1h",
        "bar_timestamp": pd.Timestamp(timestamp).isoformat(),
        "reference_close": 100.0,
        "signal": "BUY",
        "model_class_prediction": 1,
        "horizon_bars": 8,
        "status": "PENDING",
    }


def test_saved_forecast_is_marked_unevaluable_when_window_crosses_weekday_gap():
    index = pd.date_range("2026-01-05 00:00", periods=10, freq="1h", tz="UTC")
    index = index.delete(4)
    frame = pd.DataFrame(
        {"Close": np.linspace(100.0, 101.0, len(index)), "Target": np.ones(len(index))},
        index=index,
    )
    item = _pending_forecast(index[0])

    ns = _forecast_namespace()
    result = ns["update_forecast_history"]([item], frame, "USDJPY=X", "1h", 8)

    assert result[0]["status"] == "UNEVALUABLE_GAP"
    assert "evaluated_close" not in result[0]
    assert "target_label_match" not in result[0]
    assert "欠損" in result[0]["evaluation_note"]


def test_saved_forecast_allows_weekend_market_closure():
    friday = pd.Timestamp("2026-01-09 20:00", tz="UTC")
    monday = pd.date_range("2026-01-12 00:00", periods=8, freq="1h", tz="UTC")
    index = pd.DatetimeIndex([friday, *monday])
    frame = pd.DataFrame(
        {"Close": np.linspace(100.0, 101.0, len(index)), "Target": np.ones(len(index))},
        index=index,
    )
    item = _pending_forecast(friday)

    ns = _forecast_namespace()
    result = ns["update_forecast_history"]([item], frame, "USDJPY=X", "1h", 8)

    assert result[0]["status"] == "CORRECT"
    assert result[0]["target_label_match"] is True
