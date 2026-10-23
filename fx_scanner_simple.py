
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FX簡易スキャナー（分析・通知用のみ）

・松井証券FXへの接続、注文、自動発注は行いません。
・Yahoo Finance の参考価格を使用します。
・シグナルやスコアは勝率ではなく、利益を保証しません。
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


JST = ZoneInfo("Asia/Tokyo")
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
SPREAD_PIPS = {
    "USD/JPY": 0.2,
    "EUR/JPY": 0.5,
    "GBP/JPY": 1.0,
    "AUD/JPY": 0.6,
    "EUR/USD": 0.4,
}


def now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def download_bars(ticker: str) -> pd.DataFrame:
    """1時間足を取得し、形成中の可能性がある最新足を除外する。"""
    if yf is None:
        raise RuntimeError(
            "yfinance がありません。requirements.txt を確認してください。"
        )

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
        raise RuntimeError(f"必要な価格列がありません: {missing}")

    frame = frame[needed].copy()

    for col in needed:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna().sort_index()

    # 最新の足は形成中の可能性があるため除外
    if len(frame) > 1:
        frame = frame.iloc[:-1]

    if len(frame) < MIN_BARS:
        raise RuntimeError(
            f"データ不足です（{len(frame)}本）。"
            f"最低{MIN_BARS}本必要です。"
        )

    return frame


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # EMA20の5本前からの変化率（%）
    df["EMA20Slope5"] = df["EMA20"].pct_change(5) * 100

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))

    # RSIの計算上、値下がりがない場合は100として扱う
    no_losses = loss.eq(0) & gain.gt(0)
    df.loc[no_losses, "RSI"] = 100

    no_moves = loss.eq(0) & gain.eq(0)
    df.loc[no_moves, "RSI"] = 50

    # ATR 14
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()

    return df


def classify_signal(
    row: pd.Series,
) -> tuple[str, int, list[str], int]:
    """
    ルールベースの判定。
    scoreは方向判定用の点数で、勝率・成功確率ではない。
    """
    close = safe_float(row.get("Close"))
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    slope = safe_float(row.get("EMA20Slope5"))
    rsi = safe_float(row.get("RSI"))

    if None in (close, ema20, ema50, slope, rsi):
        return "WAIT", 50, ["指標データ不足"], 0

    score = 0
    reasons: list[str] = []

    # 価格と短期移動平均線
    if close > ema20:
        score += 1
        reasons.append("価格はEMA20より上")
    elif close < ema20:
        score -= 1
        reasons.append("価格はEMA20より下")
    else:
        reasons.append("価格とEMA20はほぼ同水準")

    # 短期と長期の移動平均線
    if ema20 > ema50:
        score += 1
        reasons.append("EMA20はEMA50より上")
    elif ema20 < ema50:
        score -= 1
        reasons.append("EMA20はEMA50より下")
    else:
        reasons.append("EMA20とEMA50はほぼ同水準")

    # EMA20の傾き。ごく小さな変化は中立扱い
    if slope > 0.02:
        score += 1
        reasons.append("EMA20は直近5本で上向き")
    elif slope < -0.02:
        score -= 1
        reasons.append("EMA20は直近5本で下向き")
    else:
        reasons.append("EMA20の傾きは小さく、方向感が弱い")

    # RSIによる勢いの確認
    if 50 <= rsi <= 68:
        score += 1
        reasons.append("RSIは上昇寄りの範囲")
    elif 32 <= rsi < 50:
        score -= 1
        reasons.append("RSIは弱めの範囲")
    elif rsi > 75:
        reasons.append("RSIが高水準。追い買いに注意")
    elif rsi < 25:
        reasons.append("RSIが低水準。追い売りに注意")
    else:
        reasons.append("RSIは中立寄りの範囲")

    # 複数条件の一致度による判定
    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "WAIT"

    # 表示用のルール強度スコア。勝率ではない
    strength = min(90, 50 + abs(score) * 8) if signal != "WAIT" else 50

    return signal, strength, reasons, score


def analyze_pair(name: str, ticker: str) -> dict[str, Any]:
    frame = add_indicators(download_bars(ticker))
    row = frame.iloc[-1]

    signal, strength, reasons, direction_score = classify_signal(row)

    close = safe_float(row["Close"])
    atr = safe_float(row["ATR14"])
    rsi = safe_float(row["RSI"])
    ema20 = safe_float(row["EMA20"])
    ema50 = safe_float(row["EMA50"])
    slope = safe_float(row["EMA20Slope5"])

    if close is None or atr is None or atr <= 0:
        raise RuntimeError("価格またはATRの値が不正です。")

    pip = 0.01 if "JPY" in name else 0.0001
    spread = SPREAD_PIPS.get(name, 1.0)
    spread_price = spread * pip
    spread_to_atr = spread_price / atr

    cautions: list[str] = []

    if spread_to_atr > 0.15:
        cautions.append("参考スプレッドがATRに対して大きめです")

    if rsi is not None and (rsi >= 75 or rsi <= 25):
        cautions.append("RSIが極端な水準です。急変動に注意")

    if signal == "WAIT":
        cautions.append("方向性が弱い、または条件が一致していません")

    if len(frame) < MIN_BARS:
        cautions.append("分析データが少ないため注意が必要です")

    return {
        "pair": name,
        "ticker": ticker,
        "signal": signal,
        "direction_score": direction_score,
        "signal_strength_score": strength,
        "score_note": "ルールの一致度を示す目安。勝率・成功確率ではありません",
        "price": close,
        "ema20": ema20,
        "ema50": ema50,
        "ema20_slope_5bar_percent": slope,
        "rsi14": rsi,
        "atr14": atr,
        "spread_reference_pips": spread,
        "spread_to_atr_ratio": round(spread_to_atr, 4),
        "bar_time": str(frame.index[-1]),
        "reasons": reasons,
        "cautions": cautions,
        "data_source": "Yahoo Finance / yfinance（参考データ）",
        "auto_order": False,
    }


def load_history() -> list[dict[str, Any]]:
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> int:
    if yf is None:
        print("エラー: yfinance がインストールされていません。")
        return 2

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for name, ticker in PAIRS.items():
        try:
            results.append(analyze_pair(name, ticker))
        except Exception as exc:
            errors.append({
                "pair": name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    report = {
        "schema_version": 2,
        "created_at": now_jst(),
        "mode": "simple_rule_based_analysis_only",
        "interval": INTERVAL,
        "period": PERIOD,
        "auto_order": False,
        "warning": (
            "これは参考分析であり、売買推奨・利益保証ではありません。"
            "スコアは勝率ではありません。"
            "Yahoo Financeの価格は松井証券FXの価格や約定価格と"
            "異なる場合があります。"
        ),
        "results": results,
        "errors": errors,
    }

    save_json(STATE_FILE, report)

    history = load_history()
    history.append(report)
    save_json(HISTORY_FILE, history[-200:])

    print(f"FX簡易スキャナー — {report['created_at']}")
    print("分析専用 / 自動発注なし / スコアは勝率ではありません")
    print("-" * 68)

    for item in results:
        print(
            f"{item['pair']:7} {item['signal']:4} "
            f"方向スコア={item['direction_score']:+d} "
            f"強度目安={item['signal_strength_score']} "
            f"価格={item['price']}"
        )
        print(
            f"  RSI={item['rsi14']} "
            f"EMA20傾き(5本)={item['ema20_slope_5bar_percent']}%"
        )

        for reason in item["reasons"]:
            print(f"  判定理由: {reason}")

        for caution in item["cautions"]:
            print(f"  注意: {caution}")

    for item in errors:
        print(f"{item['pair']}: 取得エラー — {item['error']}")

    print(f"\n結果保存: {STATE_FILE}")
    print(f"履歴保存: {HISTORY_FILE}")

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
