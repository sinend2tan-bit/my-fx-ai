#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FX簡易スキャナー（分析・通知用のみ）
- 松井証券FXへの接続・注文・自動発注は行いません。
- Yahoo Finance (yfinance) の参考価格データを使います。実際の約定価格とは異なる場合があります。
- 売買シグナルは投資助言や利益の保証ではありません。
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

JST = timezone.utc  # timestamp is stored in UTC; display is converted to Japan time below.
STATE_FILE = Path("scan_state_simple.json")
HISTORY_FILE = Path("scan_history_simple.json")

PAIRS: dict[str, str] = {
    "USD/JPY": "USDJPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/USD": "EURUSD=X",
}
PERIOD = "60d"
INTERVAL = "1h"
MIN_BARS = 80
LOOKAHEAD_BARS = 8  # label horizon for historical evaluation only
SPREAD_PIPS = {
    "USD/JPY": 0.2,
    "EUR/JPY": 0.5,
    "GBP/JPY": 1.0,
    "AUD/JPY": 0.6,
    "EUR/USD": 0.4,
}


def now_jst() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def download_bars(ticker: str) -> pd.DataFrame:
    """Download hourly candles and discard the latest candle conservatively."""
    if yf is None:
        raise RuntimeError("yfinance がありません。requirements.txt の依存関係をインストールしてください。")
    frame = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )
    if frame is None or frame.empty:
        raise RuntimeError("価格データを取得できませんでした。")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close"]
    missing = [col for col in needed if col not in frame.columns]
    if missing:
        raise RuntimeError(f"価格データに必要な列がありません: {missing}")
    frame = frame[needed].copy()
    for col in needed:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna().sort_index()
    # 最新の足は形成中の可能性があるため分析から外す
    if len(frame) > 1:
        frame = frame.iloc[:-1]
    if len(frame) < MIN_BARS:
        raise RuntimeError(f"データ不足です（{len(frame)}本）。最低{MIN_BARS}本必要です。")
    return frame


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["SMA20"] = close.rolling(20).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))
    # True Range / ATR
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR14"] = true_range.rolling(14).mean()
    df["Return8"] = close.pct_change(LOOKAHEAD_BARS)
    return df


def classify_signal(row: pd.Series) -> tuple[str, int, list[str]]:
    """Transparent rule-based signal; confidence is a heuristic score, not probability."""
    close = safe_float(row.get("Close"))
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    rsi = safe_float(row.get("RSI"))
    if None in (close, ema20, ema50, rsi):
        return "WAIT", 0, ["指標データ不足"]
    score = 0
    reasons: list[str] = []
    if close > ema20:
        score += 1
        reasons.append("価格がEMA20より上")
    else:
        score -= 1
        reasons.append("価格がEMA20より下")
    if ema20 > ema50:
        score += 1
        reasons.append("EMA20がEMA50より上")
    else:
        score -= 1
        reasons.append("EMA20がEMA50より下")
    if 50 <= rsi <= 68:
        score += 1
        reasons.append("RSIは上昇寄りの範囲")
    elif 32 <= rsi < 50:
        score -= 1
        reasons.append("RSIは弱めの範囲")
    elif rsi > 75:
        reasons.append("RSI高水準のため追い買い注意")
    elif rsi < 25:
        reasons.append("RSI低水準のため追い売り注意")
    signal = "BUY" if score >= 2 else "SELL" if score <= -2 else "WAIT"
    # This is a rule-strength score, deliberately not called a probability.
    strength = min(90, 50 + abs(score) * 12) if signal != "WAIT" else 50
    return signal, strength, reasons


def analyze_pair(name: str, ticker: str) -> dict[str, Any]:
    frame = add_indicators(download_bars(ticker))
    row = frame.iloc[-1]
    signal, strength, reasons = classify_signal(row)
    close = safe_float(row["Close"])
    atr = safe_float(row["ATR14"])
    rsi = safe_float(row["RSI"])
    ema20 = safe_float(row["EMA20"])
    ema50 = safe_float(row["EMA50"])
    pip = 0.01 if "JPY" in name else 0.0001
    spread = SPREAD_PIPS.get(name, 1.0)
    spread_price = spread * pip
    spread_to_atr = (spread_price / atr) if atr and atr > 0 else None
    caution = []
    if spread_to_atr is None:
        caution.append("ATRまたはスプレッドを評価できません")
    elif spread_to_atr > 0.15:
        caution.append("参考スプレッドがATRに対して大きめです")
    if rsi is not None and (rsi >= 75 or rsi <= 25):
        caution.append("RSIが極端な水準です。反転・急変動に注意")
    if signal == "WAIT":
        caution.append("方向性が一致していないため様子見")
    return {
        "pair": name,
        "ticker": ticker,
        "signal": signal,
        "signal_strength_score": strength,
        "score_note": "確率ではなく、単純ルールの強さを示す目安",
        "price": close,
        "ema20": ema20,
        "ema50": ema50,
        "rsi14": rsi,
        "atr14": atr,
        "spread_reference_pips": spread,
        "spread_to_atr_ratio": round(spread_to_atr, 4) if spread_to_atr is not None else None,
        "bar_time": str(frame.index[-1]),
        "reasons": reasons,
        "cautions": caution,
        "data_source": "Yahoo Finance / yfinance (参考データ)",
        "auto_order": False,
    }


def load_history() -> list[dict[str, Any]]:
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    if yf is None:
        print("エラー: yfinance がインストールされていません。")
        print("pip install yfinance pandas")
        return 2

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for name, ticker in PAIRS.items():
        try:
            results.append(analyze_pair(name, ticker))
        except Exception as exc:
            errors.append({"pair": name, "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "schema_version": 1,
        "created_at": now_jst(),
        "mode": "simple_rule_based_analysis_only",
        "interval": INTERVAL,
        "period": PERIOD,
        "auto_order": False,
        "warning": (
            "これは参考分析であり、売買推奨・利益保証ではありません。"
            "Yahoo Financeの価格は松井証券FXの実勢価格や約定価格と異なる場合があります。"
        ),
        "results": results,
        "errors": errors,
    }
    save_json(STATE_FILE, report)
    history = load_history()
    history.append(report)
    # Keep the most recent 200 scans only.
    save_json(HISTORY_FILE, history[-200:])

    print(f"FX簡易スキャナー — {report['created_at']}")
    print("分析専用 / 自動発注なし")
    print("-" * 68)
    for item in results:
        print(
            f"{item['pair']:7} {item['signal']:4} "
            f"score={item['signal_strength_score']:2} "
            f"price={item['price']} RSI={item['rsi14']}"
        )
        for note in item["cautions"]:
            print(f"  注意: {note}")
    for item in errors:
        print(f"{item['pair']}: 取得エラー — {item['error']}")
    print(f"\n結果保存: {STATE_FILE}")
    print(f"履歴保存: {HISTORY_FILE}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
