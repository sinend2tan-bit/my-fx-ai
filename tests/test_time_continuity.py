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
