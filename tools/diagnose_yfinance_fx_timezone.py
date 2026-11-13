#!/usr/bin/env python3
"""Read-only Yahoo Finance/yfinance timezone diagnostic for FX daily candles.

This script does not write application data, place orders, or change model state.
It compares yfinance history(), download(ignore_tz=False), and download(default)
for the five FX tickers used by the app.
"""
from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

SYMBOLS = ["USDJPY=X", "GBPJPY=X", "EURJPY=X", "AUDJPY=X", "EURUSD=X"]
OHLC = ["Open", "High", "Low", "Close"]


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def flatten_download(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance may put the ticker in either level depending on version.
        for level in range(frame.columns.nlevels):
            values = frame.columns.get_level_values(level)
            if symbol in values:
                frame = frame.xs(symbol, axis=1, level=level, drop_level=True)
                break
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
    return frame


def frame_summary(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        return {"rows": 0, "empty": True}
    idx = frame.index
    tz = getattr(idx, "tz", None)
    valid = frame[[c for c in OHLC if c in frame.columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    return {
        "rows": int(len(frame)),
        "index_type": type(idx).__name__,
        "index_timezone": str(tz) if tz is not None else None,
        "index_min": str(idx.min()) if len(idx) else None,
        "index_max": str(idx.max()) if len(idx) else None,
        "first_3_index": [str(x) for x in idx[:3]],
        "last_3_index": [str(x) for x in idx[-3:]],
        "ohlc_columns_present": [c for c in OHLC if c in frame.columns],
        "ohlc_non_null_rows": int(valid.dropna().shape[0]) if not valid.empty else 0,
    }


def compare_by_calendar_date(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    if left is None or right is None or left.empty or right.empty:
        return {"comparable": False, "reason": "one_or_both_frames_empty"}
    if not set(OHLC).issubset(left.columns) or not set(OHLC).issubset(right.columns):
        return {"comparable": False, "reason": "OHLC_columns_missing"}

    a = left[OHLC].copy()
    b = right[OHLC].copy()
    # Compare calendar labels as displayed by each source; preserve original
    # timezone information separately in frame_summary().
    a.index = pd.Index([pd.Timestamp(x).strftime("%Y-%m-%d") for x in a.index], name="date")
    b.index = pd.Index([pd.Timestamp(x).strftime("%Y-%m-%d") for x in b.index], name="date")
    a = a[~a.index.duplicated(keep="last")]
    b = b[~b.index.duplicated(keep="last")]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"comparable": True, "overlap_dates": 0, "equal_ohlc_rows": 0}
    aa = a.loc[common].apply(pd.to_numeric, errors="coerce")
    bb = b.loc[common].apply(pd.to_numeric, errors="coerce")
    delta = (aa - bb).abs()
    row_equal = delta.le(1e-10).all(axis=1) & aa.notna().all(axis=1) & bb.notna().all(axis=1)
    max_diff = delta.max().max()
    return {
        "comparable": True,
        "overlap_dates": int(len(common)),
        "equal_ohlc_rows": int(row_equal.sum()),
        "different_ohlc_rows": int((~row_equal).sum()),
        "max_absolute_ohlc_difference": float(max_diff) if pd.notna(max_diff) else None,
        "first_5_different_dates": [str(x) for x in common[~row_equal][:5]],
    }


def diagnose(symbol: str) -> dict:
    result = {"symbol": symbol}
    ticker = yf.Ticker(symbol)

    try:
        getter = getattr(ticker, "get_history_metadata", None)
        metadata = getter() if callable(getter) else getattr(ticker, "history_metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        keys = [
            "exchangeTimezoneName", "timezone", "currency", "exchangeName",
            "fullExchangeName", "instrumentType", "gmtoffset", "firstTradeDate",
        ]
        result["history_metadata"] = {k: json_safe(metadata.get(k)) for k in keys}
        result["history_metadata_available"] = bool(metadata)
    except Exception as exc:
        result["history_metadata_error"] = f"{type(exc).__name__}: {exc}"

    try:
        hist = ticker.history(
            period="2y", interval="1d", auto_adjust=False, actions=False,
            raise_errors=True,
        )
        result["ticker_history"] = frame_summary(hist)
    except Exception as exc:
        hist = pd.DataFrame()
        result["ticker_history_error"] = f"{type(exc).__name__}: {exc}"

    # Some yfinance versions populate history_metadata only after history()
    # has fetched the chart response; retry metadata after that first request.
    if not result.get("history_metadata_available"):
        try:
            getter = getattr(ticker, "get_history_metadata", None)
            metadata = getter() if callable(getter) else getattr(ticker, "history_metadata", None)
            if isinstance(metadata, dict):
                keys = [
                    "exchangeTimezoneName", "timezone", "currency", "exchangeName",
                    "fullExchangeName", "instrumentType", "gmtoffset", "firstTradeDate",
                ]
                result["history_metadata"] = {k: json_safe(metadata.get(k)) for k in keys}
                result["history_metadata_available"] = bool(metadata)
        except Exception as exc:
            result["history_metadata_retry_error"] = f"{type(exc).__name__}: {exc}"

    frames = {}
    for mode, kwargs in (
        ("download_ignore_tz_false", {"ignore_tz": False}),
        ("download_default", {}),
    ):
        try:
            data = yf.download(
                symbol, period="2y", interval="1d", auto_adjust=False,
                progress=False, threads=False, timeout=20, **kwargs,
            )
            data = flatten_download(data, symbol)
            frames[mode] = data
            result[mode] = frame_summary(data)
        except Exception as exc:
            frames[mode] = pd.DataFrame()
            result[mode + "_error"] = f"{type(exc).__name__}: {exc}"

    result["comparison_history_vs_ignore_tz_false"] = compare_by_calendar_date(
        hist, frames.get("download_ignore_tz_false", pd.DataFrame())
    )
    result["comparison_ignore_tz_false_vs_default"] = compare_by_calendar_date(
        frames.get("download_ignore_tz_false", pd.DataFrame()),
        frames.get("download_default", pd.DataFrame()),
    )
    return result


def main() -> None:
    report = {
        "diagnostic": "read_only_yfinance_fx_daily_timezone",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "pandas_version": pd.__version__,
        "symbols": [],
    }
    for symbol in SYMBOLS:
        try:
            report["symbols"].append(diagnose(symbol))
        except Exception as exc:
            report["symbols"].append({
                "symbol": symbol,
                "fatal_error": f"{type(exc).__name__}: {exc}",
            })
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
