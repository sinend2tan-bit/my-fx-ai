"""Shared OHLC candle-geometry validation for App and Scanner.

This module validates source candles before indicators or resampling. It does not
repair prices or infer missing values; invalid rows are excluded by the caller.
"""
import numpy as np
import pandas as pd


def ohlc_valid_mask(frame, required=None):
    """Return a boolean Series for finite, positive, geometrically valid OHLC rows.

    A candle is valid only when:
      * Open, High, Low and Close are finite numeric values greater than zero;
      * High >= max(Open, Close);
      * Low <= min(Open, Close);
      * High >= Low.

    The returned Series preserves the original index, including duplicate labels.
    Missing required columns raise ValueError so callers cannot silently skip checks.
    """
    columns = list(required or ("Open", "High", "Low", "Close"))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")
    if not {"Open", "High", "Low", "Close"}.issubset(columns):
        raise ValueError("required must include Open, High, Low and Close")

    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    numeric = values[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
    finite_positive = np.isfinite(numeric).all(axis=1) & (numeric > 0).all(axis=1)
    oc_max = values[["Open", "Close"]].max(axis=1).to_numpy(dtype=float)
    oc_min = values[["Open", "Close"]].min(axis=1).to_numpy(dtype=float)
    high = values["High"].to_numpy(dtype=float)
    low = values["Low"].to_numpy(dtype=float)
    valid = finite_positive & (high >= oc_max) & (low <= oc_min) & (high >= low)
    return pd.Series(valid, index=frame.index, dtype=bool)
