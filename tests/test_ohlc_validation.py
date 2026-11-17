import numpy as np
import pandas as pd
import pytest

from aifx_ohlc_validation import ohlc_valid_mask


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
