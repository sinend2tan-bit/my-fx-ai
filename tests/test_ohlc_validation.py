import numpy as np
import pandas as pd
import pytest

from aifx_ohlc_validation import (
    add_4h_bucket_diagnostics,
    clean_ohlc_source,
    ohlc_valid_mask,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])


def test_accepts_valid_candles_and_equal_boundaries():
    data = frame([
        [100.0, 105.0, 95.0, 102.0],
        [100.0, 102.0, 98.0, 102.0],
        [100.0, 100.0, 100.0, 100.0],
    ])
    assert ohlc_valid_mask(data).tolist() == [True, True, True]


@pytest.mark.parametrize(
    "row",
    [
        [100.0, 99.0, 95.0, 98.0],   # High below Open
        [100.0, 105.0, 103.0, 102.0], # Low above Close
        [100.0, 95.0, 96.0, 98.0],   # High below Low
        [0.0, 105.0, 95.0, 102.0],   # non-positive Open
        [100.0, 105.0, -1.0, 102.0], # non-positive Low
        [100.0, np.inf, 95.0, 102.0],
        [100.0, 105.0, np.nan, 102.0],
    ],
)
def test_rejects_invalid_candle_geometry_and_values(row):
    assert ohlc_valid_mask(frame([row])).tolist() == [False]


def test_preserves_duplicate_index_alignment():
    data = frame([
        [100.0, 105.0, 95.0, 102.0],
        [100.0, 99.0, 95.0, 98.0],
    ])
    data.index = pd.DatetimeIndex(["2026-01-01", "2026-01-01"])
    result = ohlc_valid_mask(data)
    assert result.index.equals(data.index)
    assert result.tolist() == [True, False]


def test_requires_all_ohlc_columns():
    with pytest.raises(ValueError, match="Missing required OHLC columns"):
        ohlc_valid_mask(pd.DataFrame({"Open": [1.0]}))



def test_invalid_source_bar_makes_four_hour_bucket_incomplete():
    index = pd.date_range("2026-01-01 00:00", periods=4, freq="h", tz="UTC")
    data = frame([
        [100.0, 105.0, 95.0, 102.0],
        [102.0, 104.0, 101.0, 103.0],
        [103.0, 99.0, 98.0, 100.0],  # impossible: High below Open
        [100.0, 106.0, 99.0, 105.0],
    ])
    data.index = index
    valid_source = data.loc[ohlc_valid_mask(data)]
    bucket_count = valid_source["Close"].resample("4h", origin="start_day").count()
    assert bucket_count.iloc[0] == 3
    assert bucket_count.iloc[0] != 4

def test_clean_ohlc_source_counts_each_first_exclusion_reason():
    index = pd.date_range("2026-01-01 00:00", periods=8, freq="h", tz="UTC")
    data = frame([
        [100.0, 105.0, 95.0, 102.0],
        [102.0, 106.0, 101.0, 103.0],
        [103.0, 107.0, 102.0, 104.0],
        [104.0, 108.0, 103.0, 105.0],
        [105.0, 109.0, 104.0, 106.0],
        [106.0, 110.0, 105.0, 107.0],
        [107.0, 111.0, 106.0, 108.0],
        [108.0, 112.0, 107.0, 109.0],
    ])
    data.index = index
    data.loc[index[1], "High"] = np.nan
    data.loc[index[5], "High"] = 100.0  # impossible: High below Open
    duplicate = data.loc[[index[6]]].copy()
    invalid_timestamp = frame([[100.0, 105.0, 95.0, 102.0]])
    invalid_timestamp.index = pd.DatetimeIndex([pd.NaT], tz="UTC")
    data = pd.concat([data, duplicate, invalid_timestamp])

    cleaned, diagnostics = clean_ohlc_source(data)
    hourly_counts = cleaned["Close"].resample("4h", origin="start_day").count()
    add_4h_bucket_diagnostics(
        diagnostics,
        hourly_counts,
        pd.Timestamp("2026-01-01 08:00", tz="UTC"),
    )

    assert diagnostics == {
        "input_rows": 10,
        "dropped_invalid_timestamp": 1,
        "dropped_off_hour_grid_timestamp": 0,
        "dropped_missing_or_nonfinite": 1,
        "dropped_duplicate_timestamp": 1,
        "dropped_invalid_ohlc": 1,
        "validated_source_rows": 6,
        "completed_4h_buckets": 2,
        "complete_4h_buckets": 0,
        "dropped_underfull_4h_buckets": 2,
        "dropped_overfull_4h_buckets": 0,
        "dropped_non_four_4h_buckets": 2,
    }
    assert hourly_counts.tolist() == [3, 3]

def test_duplicate_order_matches_legacy_missing_then_keep_last_contract():
    index = pd.DatetimeIndex([
        "2026-01-01 00:00",
        "2026-01-01 00:00",
        "2026-01-01 01:00",
        "2026-01-01 01:00",
    ], tz="UTC")
    data = frame([
        [100.0, 105.0, 95.0, 102.0],  # older valid duplicate
        [100.0, np.nan, 95.0, 102.0], # missing latest duplicate is dropped first
        [102.0, 107.0, 101.0, 105.0], # older valid duplicate
        [102.0, 101.0, 99.0, 100.0],  # latest remaining duplicate has impossible High
    ])
    data.index = index

    cleaned, diagnostics = clean_ohlc_source(data)

    # Missing duplicate is removed before de-duplication, so the older
    # non-missing row remains; impossible geometry on the last remaining
    # duplicate is rejected and does not fall back to its older duplicate.
    assert cleaned.index.tolist() == [index[0]]
    assert diagnostics["dropped_off_hour_grid_timestamp"] == 0
    assert diagnostics["dropped_missing_or_nonfinite"] == 1
    assert diagnostics["dropped_duplicate_timestamp"] == 1
    assert diagnostics["dropped_invalid_ohlc"] == 1
    assert diagnostics["validated_source_rows"] == 1

def test_off_hour_grid_rows_are_removed_before_4h_bucket_counting():
    index = pd.DatetimeIndex([
        "2026-01-01 00:00",
        "2026-01-01 00:30",
        "2026-01-01 01:00",
        "2026-01-01 02:00",
        "2026-01-01 03:00",
        "2026-01-01 04:00",
        "2026-01-01 05:00",
        "2026-01-01 06:00",
    ], tz="UTC")
    data = frame([
        [100.0, 105.0, 95.0, 102.0],
        [102.0, 106.0, 101.0, 103.0],
        [103.0, 107.0, 102.0, 104.0],
        [104.0, 108.0, 103.0, 105.0],
        [105.0, 109.0, 104.0, 106.0],
        [106.0, 110.0, 105.0, 107.0],
        [107.0, 111.0, 106.0, 108.0],
        [108.0, 112.0, 107.0, 109.0],
    ])
    data.index = index
    cleaned, diagnostics = clean_ohlc_source(data)
    counts = cleaned["Close"].resample("4h", origin="start_day").count()
    add_4h_bucket_diagnostics(
        diagnostics, counts, pd.Timestamp("2026-01-01 08:00", tz="UTC")
    )

    assert counts.tolist() == [4, 3]
    assert diagnostics["dropped_off_hour_grid_timestamp"] == 1
    assert diagnostics["completed_4h_buckets"] == 2
    assert diagnostics["complete_4h_buckets"] == 1
    assert diagnostics["dropped_underfull_4h_buckets"] == 1
    assert diagnostics["dropped_overfull_4h_buckets"] == 0
    assert diagnostics["dropped_non_four_4h_buckets"] == 1


def test_overfull_4h_bucket_diagnostic_is_defensive():
    counts = pd.Series(
        [5], index=pd.DatetimeIndex(["2026-01-01 00:00"], tz="UTC")
    )
    diagnostics = {
        "completed_4h_buckets": 0,
        "complete_4h_buckets": 0,
        "dropped_underfull_4h_buckets": 0,
        "dropped_overfull_4h_buckets": 0,
        "dropped_non_four_4h_buckets": 0,
    }
    add_4h_bucket_diagnostics(
        diagnostics, counts, pd.Timestamp("2026-01-01 04:00", tz="UTC")
    )
    assert diagnostics["dropped_overfull_4h_buckets"] == 1

def test_clean_ohlc_source_rejects_timezone_naive_index():
    data = frame([
        [100.0, 105.0, 95.0, 102.0],
        [102.0, 106.0, 101.0, 103.0],
    ])
    data.index = pd.date_range("2026-01-01 00:00", periods=2, freq="h")

    with pytest.raises(ValueError, match="timezone-aware"):
        clean_ohlc_source(data)


def test_clean_ohlc_source_normalizes_aware_index_to_utc():
    data = frame([
        [100.0, 105.0, 95.0, 102.0],
        [102.0, 106.0, 101.0, 103.0],
        [103.0, 107.0, 102.0, 104.0],
        [104.0, 108.0, 103.0, 105.0],
    ])
    data.index = pd.date_range(
        "2026-01-01 09:00", periods=4, freq="h", tz="Asia/Tokyo"
    )

    cleaned, diagnostics = clean_ohlc_source(data)

    assert str(cleaned.index.tz) == "UTC"
    assert cleaned.index.equals(
        pd.date_range("2026-01-01 00:00", periods=4, freq="h", tz="UTC")
    )
    assert diagnostics["validated_source_rows"] == 4
