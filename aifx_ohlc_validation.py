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


def clean_ohlc_source(frame, required=None):
    """Clean 1h source rows for 4h aggregation and return reason-specific counts.

    The order intentionally matches the existing 4h input contract:
    invalid timestamps are removed first, then timestamps not aligned to the
    UTC hourly grid are removed, then rows with missing/non-finite required
    OHLC are removed, then remaining duplicate timestamps keep the last row,
    and finally impossible candle geometry is rejected. Thus a
    missing-value duplicate can fall back to an older non-missing row (as in
    the legacy dropna-before-dedup order), but a last remaining duplicate with
    impossible OHLC geometry is rejected without falling back to an older row.
    """
    columns = list(required or ("Open", "High", "Low", "Close"))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")
    if not {"Open", "High", "Low", "Close"}.issubset(columns):
        raise ValueError("required must include Open, High, Low and Close")

    cleaned = frame.copy()
    diagnostics = {
        "input_rows": int(len(cleaned)),
        "dropped_invalid_timestamp": 0,
        "dropped_off_hour_grid_timestamp": 0,
        "dropped_missing_or_nonfinite": 0,
        "dropped_duplicate_timestamp": 0,
        "dropped_invalid_ohlc": 0,
        "validated_source_rows": 0,
        "completed_4h_buckets": 0,
        "complete_4h_buckets": 0,
        "dropped_underfull_4h_buckets": 0,
        "dropped_overfull_4h_buckets": 0,
        "dropped_non_four_4h_buckets": 0,
    }

    valid_timestamp = cleaned.index.notna()
    diagnostics["dropped_invalid_timestamp"] = int((~valid_timestamp).sum())
    cleaned = cleaned.loc[valid_timestamp].copy()

    if not isinstance(cleaned.index, pd.DatetimeIndex):
        raise ValueError("4h source index must be a DatetimeIndex")
    if cleaned.index.tz is None:
        raise ValueError("4h source index must be timezone-aware")
    # Normalize inside the shared helper as well as at the callers. This keeps
    # the UTC-grid contract fail-closed if a future caller omits normalization.
    cleaned.index = cleaned.index.tz_convert("UTC")
    on_hour_grid = (
        (cleaned.index.minute == 0)
        & (cleaned.index.second == 0)
        & (cleaned.index.microsecond == 0)
        & (cleaned.index.nanosecond == 0)
    )
    diagnostics["dropped_off_hour_grid_timestamp"] = int((~on_hour_grid).sum())
    cleaned = cleaned.loc[on_hour_grid].copy()

    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    rows_before = len(cleaned)
    finite_ok = cleaned[columns].notna().all(axis=1).to_numpy()
    finite_ok &= np.isfinite(cleaned[columns].to_numpy(dtype=float)).all(axis=1)
    cleaned = cleaned.loc[finite_ok].copy()
    diagnostics["dropped_missing_or_nonfinite"] = int(rows_before - len(cleaned))

    rows_before = len(cleaned)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    diagnostics["dropped_duplicate_timestamp"] = int(rows_before - len(cleaned))

    valid_ohlc = ohlc_valid_mask(cleaned, columns).to_numpy()
    diagnostics["dropped_invalid_ohlc"] = int((~valid_ohlc).sum())
    cleaned = cleaned.loc[valid_ohlc].copy()
    diagnostics["validated_source_rows"] = int(len(cleaned))
    return cleaned, diagnostics


def add_4h_bucket_diagnostics(diagnostics, hourly_counts, cutoff):
    """Add counts for ended 4h buckets; ignore the still-forming last bucket."""
    completed = hourly_counts.loc[
        (hourly_counts.index + pd.Timedelta(hours=4)) <= cutoff
    ]
    diagnostics["completed_4h_buckets"] = int(len(completed))
    diagnostics["complete_4h_buckets"] = int((completed == 4).sum())
    diagnostics["dropped_underfull_4h_buckets"] = int((completed < 4).sum())
    diagnostics["dropped_overfull_4h_buckets"] = int((completed > 4).sum())
    diagnostics["dropped_non_four_4h_buckets"] = int((completed != 4).sum())
    return diagnostics
