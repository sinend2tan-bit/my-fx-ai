
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FX簡易スキャナー＋過去データ検証

・分析専用。松井証券FXへの接続・注文・自動発注は行いません。
・Yahoo Financeの参考価格を使用します。
・スコアは勝率ではありません。
・過去検証は将来の利益を保証しません。
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

PAIRS = {
    "USD/JPY": "USDJPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/USD": "EURUSD=X",
}

PERIOD = "60d"
INTERVAL = "1h"
MIN_BARS = 80

# 過去シグナル発生から何本後の価格を評価するか
LOOKAHEAD_BARS = 8

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
    if yf is None:
        raise RuntimeError("yfinance がインストールされていません。")

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

    # 形成中の可能性がある最新足を除外
    if len(frame) > 1:
        frame = frame.iloc[:-1]

    if len(frame) < MIN_BARS:
        raise RuntimeError(
            f"データ不足です（{len(frame)}本）。最低{MIN_BARS}本必要です。"
        )

    return frame


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA20Slope5"] = df["EMA20"].pct_change(5) * 100

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[loss.eq(0) & gain.gt(0), "RSI"] = 100
    df.loc[loss.eq(0) & gain.eq(0), "RSI"] = 50

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
    """ルールベースの方向判定。勝率の予測ではありません。"""
    close = safe_float(row.get("Close"))
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    slope = safe_float(row.get("EMA20Slope5"))
    rsi = safe_float(row.get("RSI"))

    if None in (close, ema20, ema50, slope, rsi):
        return "WAIT", 50, ["指標データ不足"], 0

    score = 0
    reasons = []

    if close > ema20:
        score += 1
        reasons.append("価格はEMA20より上")
    elif close < ema20:
        score -= 1
        reasons.append("価格はEMA20より下")
    else:
        reasons.append("価格とEMA20はほぼ同水準")

    if ema20 > ema50:
        score += 1
        reasons.append("EMA20はEMA50より上")
    elif ema20 < ema50:
        score -= 1
        reasons.append("EMA20はEMA50より下")
    else:
        reasons.append("EMA20とEMA50はほぼ同水準")

    if slope > 0.02:
        score += 1
        reasons.append("EMA20は直近5本で上向き")
    elif slope < -0.02:
        score -= 1
        reasons.append("EMA20は直近5本で下向き")
    else:
        reasons.append("EMA20の傾きは小さい")

    if 50 <= rsi <= 68:
        score += 1
        reasons.append("RSIは上昇寄り")
    elif 32 <= rsi < 50:
        score -= 1
        reasons.append("RSIは弱め")
    elif rsi > 75:
        reasons.append("RSIが高水準。追い買いに注意")
    elif rsi < 25:
        reasons.append("RSIが低水準。追い売りに注意")
    else:
        reasons.append("RSIは中立寄り")

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "WAIT"

    strength = min(90, 50 + abs(score) * 8) if signal != "WAIT" else 50
    return signal, strength, reasons, score


def backtest(
    frame: pd.DataFrame,
    pair: str,
) -> dict[str, Any]:
    """
    過去のシグナルを8本後の終値で評価する簡易検証。

    手数料・スプレッドは指定した参考スプレッドだけを控除。
    スリッページ、スワップ、実際の約定条件は再現しない。
    """
    pip = 0.01 if "JPY" in pair else 0.0001
    spread = SPREAD_PIPS.get(pair, 1.0)

    evaluated = 0
    wins = 0
    losses = 0
    neutral = 0
    net_pips_list = []
    gross_pips_list = []

    # 未来の価格を使ってシグナルを作らない。
    # 各時点の指標で判定し、その後8本の価格変化で評価する。
    end = len(frame) - LOOKAHEAD_BARS

    for i in range(end):
        row = frame.iloc[i]
        signal, _, _, _ = classify_signal(row)

        if signal not in ("BUY", "SELL"):
            continue

        entry = safe_float(row.get("Close"))
        future = safe_float(
            frame.iloc[i + LOOKAHEAD_BARS].get("Close")
        )

        if entry is None or future is None or pip <= 0:
            continue

        move_pips = (future - entry) / pip

        if signal == "SELL":
            move_pips *= -1

        # 方向一致率では、価格変化がゼロのケースを別枠にする
        if abs(move_pips) < 1e-9:
            neutral += 1
            continue

        evaluated += 1
        gross_pips_list.append(move_pips)

        net_pips = move_pips - spread
        net_pips_list.append(net_pips)

        if move_pips > 0:
            wins += 1
        else:
            losses += 1

    if evaluated:
        hit_rate = round(wins / evaluated * 100, 2)
        avg_gross = round(sum(gross_pips_list) / evaluated, 2)
        avg_net = round(sum(net_pips_list) / evaluated, 2)
        total_net = round(sum(net_pips_list), 2)
        positive_net = sum(1 for p in net_pips_list if p > 0)
        net_positive_rate = round(positive_net / evaluated * 100, 2)
    else:
        hit_rate = None
        avg_gross = None
        avg_net = None
        total_net = None
        net_positive_rate = None

    return {
        "pair": pair,
        "evaluation_method": (
            f"シグナル時点の終値から{LOOKAHEAD_BARS}本後の終値まで"
        ),
        "lookahead_bars": LOOKAHEAD_BARS,
        "spread_assumption_pips": spread,
        "sample_count": evaluated,
        "neutral_count": neutral,
        "direction_hit_rate_percent": hit_rate,
        "net_positive_rate_percent": net_positive_rate,
        "average_gross_pips": avg_gross,
        "average_net_pips_after_reference_spread": avg_net,
        "total_net_pips_after_reference_spread": total_net,
        "warning": (
            "過去データの簡易検証です。将来の勝率・利益を保証しません。"
            "実際の約定、スリッページ、スワップ、取引可能時間は"
            "完全には再現していません。"
        ),
    }


def analyze_pair(name: str, ticker: str) -> dict[str, Any]:
    frame = add_indicators(download_bars(ticker))
    row = frame.iloc[-1]

    signal, strength, reasons, direction_score = classify_signal(row)

    close = safe_float(row.get("Close"))
    atr = safe_float(row.get("ATR14"))
    rsi = safe_float(row.get("RSI"))
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    slope = safe_float(row.get("EMA20Slope5"))

    if close is None or atr is None or atr <= 0:
        raise RuntimeError("価格またはATRの値が不正です。")

    pip = 0.01 if "JPY" in name else 0.0001
    spread = SPREAD_PIPS.get(name, 1.0)
    spread_to_atr = (spread * pip) / atr

    cautions = []

    if spread_to_atr > 0.15:
        cautions.append("参考スプレッドがATRに対して大きめです")

    if rsi is not None and (rsi >= 75 or rsi <= 25):
        cautions.append("RSIが極端な水準です。急変動に注意")

    if signal == "WAIT":
        cautions.append("方向性が弱い、または条件が一致していません")

    historical = backtest(frame, name)

    if historical["sample_count"] < 30:
        cautions.append(
            "過去検証の件数が少ないため、成績を判断するには不十分です"
        )

    return {
        "pair": name,
        "ticker": ticker,
        "signal": signal,
        "direction_score": direction_score,
        "signal_strength_score": strength,
        "score_note": "ルールの一致度。勝率・成功確率ではありません",
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
        "historical_backtest": historical,
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

    results = []
    errors = []

    for name, ticker in PAIRS.items():
        try:
            results.append(analyze_pair(name, ticker))
        except Exception as exc:
            errors.append({
                "pair": name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    report = {
        "schema_version": 3,
        "created_at": now_jst(),
        "mode": "simple_analysis_with_historical_backtest",
        "interval": INTERVAL,
        "period": PERIOD,
        "lookahead_bars": LOOKAHEAD_BARS,
        "auto_order": False,
        "warning": (
            "分析・過去検証専用です。スコアは勝率ではありません。"
            "過去の成績は将来の利益を保証しません。"
            "Yahoo Financeの価格は松井証券FXの価格・約定価格と"
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
    print("分析・過去検証専用 / 自動発注なし")
    print("スコアや過去検証結果は将来の勝率を保証しません")
    print("-" * 68)

    for item in results:
        print(
            f"\n{item['pair']} {item['signal']} "
            f"方向スコア={item['direction_score']:+d} "
            f"強度目安={item['signal_strength_score']}"
        )
        print(
            f"価格={item['price']} "
            f"RSI={item['rsi14']} "
            f"EMA20傾き(5本)={item['ema20_slope_5bar_percent']}%"
        )

        for reason in item["reasons"]:
            print(f"  判定理由: {reason}")

        for caution in item["cautions"]:
            print(f"  注意: {caution}")

        bt = item["historical_backtest"]
        print("  --- 過去データ検証 ---")
        print(f"  検証件数: {bt['sample_count']}")
        print(f"  方向一致率: {bt['direction_hit_rate_percent']}%")
        print(f"  スプレッド控除後の平均参考損益: "
              f"{bt['average_net_pips_after_reference_spread']} pips")
        print(f"  スプレッド控除後の合計参考損益: "
              f"{bt['total_net_pips_after_reference_spread']} pips")
        print("  ※過去データの結果であり、将来の成績を保証しません")

    for item in errors:
        print(f"\n{item['pair']}: 取得エラー — {item['error']}")

    print(f"\n結果保存: {STATE_FILE}")
    print(f"履歴保存: {HISTORY_FILE}")

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
