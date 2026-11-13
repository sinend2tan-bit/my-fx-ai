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
from collections.abc import Mapping
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

SYMBOLS = ["USDJPY=X", "GBPJPY=X", "EURJPY=X", "AUDJPY=X", "EURUSD=X"]
OHLC = ["Open", "High", "Low", "Close"]
METADATA_KEYS = [
    "exchangeTimezoneName", "timezone", "currency", "exchangeName",
    "fullExchangeName", "instrumentType", "gmtoffset", "firstTradeDate",
]


def json_safe(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
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
    different_dates = common[~row_equal][:5]
    difference_samples = []
    for date in different_dates:
        difference_samples.append({
            "date": str(date),
            "left_ohlc": {col: json_safe(aa.loc[date, col]) for col in OHLC},
            "right_ohlc": {col: json_safe(bb.loc[date, col]) for col in OHLC},
            "absolute_difference": {col: json_safe(delta.loc[date, col]) for col in OHLC},
        })
    max_diff = delta.max().max()
    return {
        "comparable": True,
        "overlap_dates": int(len(common)),
        "equal_ohlc_rows": int(row_equal.sum()),
        "different_ohlc_rows": int((~row_equal).sum()),
        "max_absolute_ohlc_difference": float(max_diff) if pd.notna(max_diff) else None,
        "first_5_different_dates": [str(x) for x in different_dates],
        "difference_samples": difference_samples,
    }


def read_metadata_after_history(ticker) -> dict:
    """Read cached base metadata without requesting lazy intraday tradingPeriods."""
    result = {
        "metadata_type": None,
        "history_metadata_available": False,
        "history_metadata": {key: None for key in METADATA_KEYS},
    }
    try:
        getter = getattr(ticker, "get_history_metadata", None)
        metadata = getter() if callable(getter) else getattr(ticker, "history_metadata", None)
        result["metadata_type"] = type(metadata).__name__ if metadata is not None else None
        if isinstance(metadata, Mapping):
            values = {key: json_safe(metadata.get(key)) for key in METADATA_KEYS}
            result["history_metadata"] = values
            result["history_metadata_available"] = any(value is not None for value in values.values())
        elif isinstance(metadata, dict):
            values = {key: json_safe(metadata.get(key)) for key in METADATA_KEYS}
            result["history_metadata"] = values
            result["history_metadata_available"] = any(value is not None for value in values.values())
        else:
            result["metadata_type_warning"] = "metadata object is not a Mapping"
    except Exception as exc:
        result["history_metadata_error"] = f"{type(exc).__name__}: {exc}"
    return result




def compare_daily_to_hourly_boundaries(ticker, daily: pd.DataFrame) -> dict:
    """Compare daily OHLC against hourly aggregation under three date boundaries."""
    try:
        hourly = ticker.history(
            period="3mo", interval="1h", auto_adjust=False, actions=False,
            raise_errors=True,
        )
    except Exception as exc:
        return {"available": False, "reason": f"hourly_history_error: {type(exc).__name__}: {exc}"}
    if hourly is None or hourly.empty or not set(OHLC).issubset(hourly.columns):
        return {"available": False, "reason": "hourly_history_empty_or_missing_ohlc"}
    if not isinstance(hourly.index, pd.DatetimeIndex) or hourly.index.tz is None:
        return {
            "available": False,
            "reason": "hourly_index_not_timezone_aware",
            "hourly_index_timezone": str(getattr(hourly.index, "tz", None)),
        }
    if daily is None or daily.empty or not set(OHLC).issubset(daily.columns):
        return {"available": False, "reason": "daily_history_empty_or_missing_ohlc"}
    if not isinstance(daily.index, pd.DatetimeIndex) or daily.index.tz is None:
        return {"available": False, "reason": "daily_index_not_timezone_aware"}

    daily_frame = daily[OHLC].copy()
    daily_frame.index = pd.Index(daily_frame.index.strftime("%Y-%m-%d"), name="date")
    daily_frame = daily_frame[~daily_frame.index.duplicated(keep="last")]
    if len(daily_frame):
        daily_frame = daily_frame.loc[daily_frame.index < daily_frame.index.max()]
    daily_frame = daily_frame.apply(pd.to_numeric, errors="coerce")

    boundary_keys = {
        "London_calendar_day": hourly.index.strftime("%Y-%m-%d"),
        "UTC_calendar_day": hourly.index.tz_convert("UTC").strftime("%Y-%m-%d"),
        "New_York_17h_session": (
            hourly.index.tz_convert("America/New_York") - pd.Timedelta(hours=17)
        ).strftime("%Y-%m-%d"),
    }
    boundary_results = {}
    for boundary_name, keys in boundary_keys.items():
        source = hourly[OHLC].copy()
        source["group_date"] = keys
        grouped = source.groupby("group_date", sort=True).agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            hourly_rows=("Close", "count"),
        )
        grouped = grouped[grouped["hourly_rows"] >= 20].copy()
        comparisons = []
        for shift_days in (-1, 0, 1):
            candidate = grouped[OHLC].copy()
            candidate.index = (
                pd.to_datetime(candidate.index) + pd.Timedelta(days=shift_days)
            ).strftime("%Y-%m-%d")
            common = daily_frame.index.intersection(candidate.index)
            if len(common) == 0:
                comparisons.append({
                    "daily_label_shift_days": shift_days,
                    "overlap_dates": 0,
                    "equal_ohlc_rows": 0,
                    "different_ohlc_rows": 0,
                })
                continue
            left = daily_frame.loc[common]
            right = candidate.loc[common].apply(pd.to_numeric, errors="coerce")
            delta = (left - right).abs()
            equal = delta.le(1e-8).all(axis=1) & left.notna().all(axis=1) & right.notna().all(axis=1)
            comparisons.append({
                "daily_label_shift_days": shift_days,
                "overlap_dates": int(len(common)),
                "equal_ohlc_rows": int(equal.sum()),
                "different_ohlc_rows": int((~equal).sum()),
                "max_absolute_ohlc_difference": float(delta.max().max()) if not delta.empty else None,
            })
        boundary_results[boundary_name] = {
            "eligible_hourly_groups": int(len(grouped)),
            "comparisons": comparisons,
        }

    return {
        "available": True,
        "hourly_index_timezone": str(hourly.index.tz),
        "hourly_rows": int(len(hourly)),
        "hourly_start": str(hourly.index.min()),
        "hourly_end": str(hourly.index.max()),
        "daily_index_timezone": str(daily.index.tz),
        "boundaries": boundary_results,
        "note": "Hourly aggregation is a diagnostic proxy; partial/missing hourly bars can prevent exact OHLC matching.",
    }


def fetch_raw_chart(ticker, symbol: str, interval: str, period: str = "3mo") -> dict:
    """Fetch raw Yahoo Chart API timestamps before yfinance normalizes daily dates."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    response = ticker._data.cache_get(
        url=url, params={"range": period, "interval": interval}, timeout=20
    )
    payload = response.json()
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo chart response has no result")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_list = (result.get("indicators") or {}).get("quote") or []
    if not quote_list:
        raise RuntimeError("Yahoo chart response has no quote data")
    quote = quote_list[0]
    rows = []
    for i, timestamp in enumerate(timestamps):
        row = {"timestamp": int(timestamp)}
        for column in OHLC:
            values = quote.get(column.lower()) or []
            row[column] = values[i] if i < len(values) else None
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for column in OHLC:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    metadata = result.get("meta") or {}
    return {
        "frame": frame,
        "exchange_timezone": metadata.get("exchangeTimezoneName"),
        "meta_currency": metadata.get("currency"),
        "raw_timestamp_count": len(timestamps),
    }


def compare_raw_daily_windows_to_hourly(ticker, symbol: str) -> dict:
    """Test whether each raw daily timestamp interval contains matching hourly OHLC."""
    try:
        daily = fetch_raw_chart(ticker, symbol, "1d", "3mo")
        hourly = fetch_raw_chart(ticker, symbol, "1h", "3mo")
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    d = daily["frame"]
    h = hourly["frame"]
    if d.empty or h.empty:
        return {"available": False, "reason": "raw_daily_or_hourly_empty"}
    h = h.sort_values("timestamp")
    d = d.sort_values("timestamp")
    hourly_times = h["timestamp"].to_numpy()
    compared = []
    for i in range(len(d) - 1):
        start = int(d.iloc[i]["timestamp"])
        end = int(d.iloc[i + 1]["timestamp"])
        bars = h[(h["timestamp"] >= start) & (h["timestamp"] < end)]
        if len(bars) < 20:
            continue
        aggregate = {
            "Open": bars["Open"].iloc[0],
            "High": bars["High"].max(),
            "Low": bars["Low"].min(),
            "Close": bars["Close"].iloc[-1],
        }
        daily_values = {column: d.iloc[i][column] for column in OHLC}
        valid = all(pd.notna(daily_values[k]) and pd.notna(aggregate[k]) for k in OHLC)
        deltas = {
            key: abs(float(daily_values[key]) - float(aggregate[key]))
            if valid else None
            for key in OHLC
        }
        compared.append({
            "daily_raw_timestamp": start,
            "daily_raw_utc": pd.Timestamp(start, unit="s", tz="UTC").isoformat(),
            "daily_raw_exchange_time": pd.Timestamp(start, unit="s", tz="UTC").tz_convert(
                daily["exchange_timezone"] or "UTC"
            ).isoformat(),
            "window_end_raw_utc": pd.Timestamp(end, unit="s", tz="UTC").isoformat(),
            "hourly_rows": int(len(bars)),
            "matches_ohlc": bool(valid and all(v <= 1e-8 for v in deltas.values())),
            "daily_ohlc": {k: json_safe(v) for k, v in daily_values.items()},
            "hourly_aggregated_ohlc": {k: json_safe(v) for k, v in aggregate.items()},
            "absolute_differences": deltas,
        })
    matched = sum(1 for row in compared if row["matches_ohlc"])
    mismatch_samples = [row for row in compared if not row["matches_ohlc"]][:5]
    return {
        "available": True,
        "exchange_timezone_daily": daily["exchange_timezone"],
        "exchange_timezone_hourly": hourly["exchange_timezone"],
        "daily_raw_timestamp_count": daily["raw_timestamp_count"],
        "hourly_raw_timestamp_count": hourly["raw_timestamp_count"],
        "daily_raw_first_5": [
            {"epoch": int(row["timestamp"]),
             "utc": pd.Timestamp(int(row["timestamp"]), unit="s", tz="UTC").isoformat(),
             "exchange": pd.Timestamp(int(row["timestamp"]), unit="s", tz="UTC").tz_convert(
                 daily["exchange_timezone"] or "UTC"
             ).isoformat()}
            for _, row in d.head(5).iterrows()
        ],
        "daily_raw_last_5": [
            {"epoch": int(row["timestamp"]),
             "utc": pd.Timestamp(int(row["timestamp"]), unit="s", tz="UTC").isoformat(),
             "exchange": pd.Timestamp(int(row["timestamp"]), unit="s", tz="UTC").tz_convert(
                 daily["exchange_timezone"] or "UTC"
             ).isoformat()}
            for _, row in d.tail(5).iterrows()
        ],
        "hourly_raw_first_3_utc": [
            pd.Timestamp(int(v), unit="s", tz="UTC").isoformat()
            for v in h["timestamp"].head(3)
        ],
        "hourly_raw_last_3_utc": [
            pd.Timestamp(int(v), unit="s", tz="UTC").isoformat()
            for v in h["timestamp"].tail(3)
        ],
        "compared_daily_windows": len(compared),
        "matched_ohlc_windows": matched,
        "mismatched_ohlc_windows": len(compared) - matched,
        "hourly_rows_per_daily_window_min": min((x["hourly_rows"] for x in compared), default=None),
        "hourly_rows_per_daily_window_max": max((x["hourly_rows"] for x in compared), default=None),
        "first_5_mismatch_samples": mismatch_samples,
        "note": "This compares raw Yahoo chart epoch windows [daily timestamp, next daily timestamp) with raw hourly chart bars; it does not infer a broker's candle convention.",
    }

def diagnose(symbol: str) -> dict:
    result = {"symbol": symbol}
    ticker = yf.Ticker(symbol)

    try:
        hist = ticker.history(
            period="2y", interval="1d", auto_adjust=False, actions=False,
            raise_errors=True,
        )
        result["ticker_history"] = frame_summary(hist)
    except Exception as exc:
        hist = pd.DataFrame()
        result["ticker_history_error"] = f"{type(exc).__name__}: {exc}"

    # In current yfinance, metadata may be a lazy Mapping rather than dict.
    # Read only base fields after history(); do not touch lazy tradingPeriods.
    result.update(read_metadata_after_history(ticker))
    result["hourly_boundary_comparison"] = compare_daily_to_hourly_boundaries(ticker, hist)
    result["raw_chart_window_comparison"] = compare_raw_daily_windows_to_hourly(ticker, symbol)

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
