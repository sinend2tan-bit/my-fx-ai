"""Regression tests for discontinuous hourly candidate-price outcome windows."""
import ast
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from aifx_time_validation import has_contiguous_hourly_window

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _outcome_updater(saved_calls):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "update_entry_tracking_outcomes"
    )
    namespace = {
        "pd": pd,
        "np": np,
        "datetime": datetime,
        "JST": ZoneInfo("Asia/Tokyo"),
        "has_contiguous_hourly_window": has_contiguous_hourly_window,
        "_normalise_timestamp": lambda value: pd.Timestamp(value).isoformat(),
        "save_entry_tracking": lambda items: saved_calls.append([dict(item) for item in items]),
    }
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace["update_entry_tracking_outcomes"]


def _item(start):
    return {
        "ticker": "USDJPY=X",
        "interval": "1h",
        "status": "PENDING",
        "bar_timestamp": start.isoformat(),
        "candidate_price": 100.0,
    }


def _frame(index, hit_positions=()):
    rows = []
    for position in range(len(index)):
        if position in hit_positions:
            rows.append({"Open": 100.0, "High": 100.1, "Low": 99.9, "Close": 100.0})
        else:
            rows.append({"Open": 101.0, "High": 101.1, "Low": 100.9, "Close": 101.0})
    return pd.DataFrame(rows, index=index)


def test_hourly_entry_tracking_rejects_gap_before_later_hit():
    start = pd.Timestamp("2026-11-16T00:00:00Z")
    index = pd.DatetimeIndex([start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=3)])
    items = [_item(start)]
    saved = []
    update = _outcome_updater(saved)

    result = update(items, _frame(index, hit_positions={1}), "USDJPY=X", "1h", horizon=2)

    assert result[0]["status"] == "UNEVALUABLE_DISCONTINUOUS"
    assert "採点対象外" in result[0]["evaluation_note"]
    assert saved


def test_hourly_entry_tracking_rejects_gap_when_no_price_touch():
    start = pd.Timestamp("2026-11-16T00:00:00Z")
    index = pd.DatetimeIndex([start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=3)])
    items = [_item(start)]
    saved = []
    update = _outcome_updater(saved)

    result = update(items, _frame(index), "USDJPY=X", "1h", horizon=2)

    assert result[0]["status"] == "UNEVALUABLE_DISCONTINUOUS"
    assert result[0]["status"] != "NOT_REACHED"


def test_hourly_entry_tracking_keeps_hit_before_later_gap():
    start = pd.Timestamp("2026-11-16T00:00:00Z")
    index = pd.DatetimeIndex([start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=3)])
    items = [_item(start)]
    saved = []
    update = _outcome_updater(saved)

    result = update(items, _frame(index, hit_positions={0}), "USDJPY=X", "1h", horizon=2)

    assert result[0]["status"] == "HIT"
    assert result[0]["bars_to_touch"] == 1


def test_non_hourly_entry_tracking_keeps_existing_row_count_behavior():
    start = pd.Timestamp("2026-11-16T00:00:00Z")
    index = pd.DatetimeIndex([start + pd.Timedelta(minutes=15), start + pd.Timedelta(minutes=45)])
    items = [_item(start)]
    saved = []
    update = _outcome_updater(saved)

    result = update(items, _frame(index), "USDJPY=X", "15m", horizon=2)

    assert result[0]["status"] == "NOT_REACHED"
