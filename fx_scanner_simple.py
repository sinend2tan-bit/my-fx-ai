import json
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# FX Simple Scanner - 複数期間・売買方向別検証版
# 分析専用 / 口座接続なし / 自動発注なし
# =========================================================

JST = ZoneInfo("Asia/Tokyo")

PERIOD = "2y"
INTERVAL = "1h"
MIN_BARS = 500

LOOKAHEAD_BARS = 8
TRAIN_RATIO = 0.70

MIN_TEST_SAMPLES = 30
MIN_TRAIN_SAMPLES = 30
MIN_DIRECTION_HIT_RATE = 0.50
MIN_AVG_NET_PIPS = 0.0

# 後半30%を3つの連続した期間に分けて検証
TEST_WINDOWS = 3
MIN_WINDOW_SAMPLES = 10

STATE_FILE = "scan_state_simple.json"
HISTORY_FILE = "scan_history_simple.json"

PAIRS = {
    "USD/JPY": {"ticker": "USDJPY=X", "spread": 0.2},
    "EUR/JPY": {"ticker": "EURJPY=X", "spread": 0.5},
    "GBP/JPY": {"ticker": "GBPJPY=X", "spread": 1.0},
    "AUD/JPY": {"ticker": "AUDJPY=X", "spread": 0.6},
    "EUR/USD": {"ticker": "EURUSD=X", "spread": 0.4},
}


def now_jst():
    return datetime.now(JST).isoformat(timespec="seconds")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def download_prices(ticker):
    try:
        df = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=20,
        )
    except Exception as exc:
        return None, f"価格データ取得エラー: {exc}"

    if df is None or df.empty:
        return None, "価格データが取得できませんでした"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    if any(col not in df.columns for col in required):
        return None, "必要な価格列がありません"

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 最後の足は未確定の可能性があるため除外
    if len(df) > 1:
        df = df.iloc[:-1].copy()

    if len(df) < MIN_BARS:
        return None, f"データ不足: {len(df)}本。最低{MIN_BARS}本必要です"

    return df, None


def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA20_SLOPE"] = df["EMA20"] - df["EMA20"].shift(5)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI"] = 50

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    return df.replace([np.inf, -np.inf], np.nan)


def make_signal(row):
    values = [
        row.get("Close"),
        row.get("EMA20"),
        row.get("EMA50"),
        row.get("EMA20_SLOPE"),
        row.get("RSI"),
        row.get("ATR"),
    ]
    if any(pd.isna(v) for v in values):
        return "WAIT", 0, ["指標データ不足"]

    score = 0
    reasons = []

    if row["Close"] > row["EMA20"]:
        score += 1
        reasons.append("価格がEMA20より上")
    elif row["Close"] < row["EMA20"]:
        score -= 1
        reasons.append("価格がEMA20より下")

    if row["EMA20"] > row["EMA50"]:
        score += 1
        reasons.append("EMA20がEMA50より上")
    elif row["EMA20"] < row["EMA50"]:
        score -= 1
        reasons.append("EMA20がEMA50より下")

    if row["EMA20_SLOPE"] > 0:
        score += 1
        reasons.append("EMA20は上向き")
    elif row["EMA20_SLOPE"] < 0:
        score -= 1
        reasons.append("EMA20は下向き")

    if row["RSI"] >= 75:
        reasons.append("RSIが高く、追い買いに注意")
    elif row["RSI"] <= 25:
        reasons.append("RSIが低く、追い売りに注意")

    if score >= 2 and row["RSI"] < 75:
        return "BUY", score, reasons
    if score <= -2 and row["RSI"] > 25:
        return "SELL", score, reasons

    return "WAIT", score, reasons


def backtest(df, pip_size, spread, start_i, end_i):
    trades = []
    i = max(start_i, 55)

    last_signal_i = min(
        end_i - LOOKAHEAD_BARS,
        len(df) - LOOKAHEAD_BARS - 1,
    )

    while i <= last_signal_i:
        signal, score, _ = make_signal(df.iloc[i])

        if signal in ("BUY", "SELL"):
            entry_i = i + 1
            exit_i = i + LOOKAHEAD_BARS
            entry_price = float(df["Open"].iloc[entry_i])
            exit_price = float(df["Close"].iloc[exit_i])

            if (
                not math.isfinite(entry_price)
                or not math.isfinite(exit_price)
                or entry_price <= 0
            ):
                i += LOOKAHEAD_BARS
                continue

            direction = 1 if signal == "BUY" else -1
            gross_pips = (
                (exit_price - entry_price) * direction / pip_size
            )
            net_pips = gross_pips - spread

            trades.append({
                "signal_time": str(df.index[i]),
                "entry_time": str(df.index[entry_i]),
                "exit_time": str(df.index[exit_i]),
                "signal": signal,
                "score": int(score),
                "gross_pips": float(gross_pips),
                "net_pips": float(net_pips),
                "win": bool(net_pips > 0),
            })

        i += LOOKAHEAD_BARS

    return trades


def summarize_trades(trades):
    count = len(trades)
    if count == 0:
        return {
            "samples": 0,
            "hit_rate_pct": None,
            "wins": 0,
            "losses": 0,
            "avg_win_pips": None,
            "avg_loss_pips": None,
            "profit_loss_ratio": None,
            "avg_net_pips": None,
            "total_net_pips": 0.0,
        }

    net = np.array([t["net_pips"] for t in trades], dtype=float)
    winners = net[net > 0]
    losers = net[net < 0]

    avg_win = float(winners.mean()) if len(winners) else None
    avg_loss = float(abs(losers.mean())) if len(losers) else None

    pl_ratio = (
        avg_win / avg_loss
        if avg_win is not None and avg_loss is not None and avg_loss > 0
        else None
    )

    return {
        "samples": count,
        "hit_rate_pct": round(float((net > 0).mean() * 100), 2),
        "wins": int((net > 0).sum()),
        "losses": int((net < 0).sum()),
        "avg_win_pips": round(avg_win, 2) if avg_win is not None else None,
        "avg_loss_pips": round(avg_loss, 2) if avg_loss is not None else None,
        "profit_loss_ratio": round(pl_ratio, 2) if pl_ratio is not None else None,
        "avg_net_pips": round(float(net.mean()), 2),
        "total_net_pips": round(float(net.sum()), 2),
    }


def summarize_by_direction(trades):
    return {
        direction: summarize_trades(
            [t for t in trades if t["signal"] == direction]
        )
        for direction in ("BUY", "SELL")
    }


def evaluate_pair(name, config):
    df, error = download_prices(config["ticker"])
    if error:
        return {
            "pair": name,
            "error": error,
            "decision": "判定保留",
        }

    df = add_indicators(df)
    pip_size = 0.01 if "JPY" in config["ticker"] else 0.0001

    split_i = int(len(df) * TRAIN_RATIO)
    train_end = max(56, split_i - LOOKAHEAD_BARS)

    train_trades = backtest(
        df, pip_size, config["spread"], 55, train_end
    )
    test_trades = backtest(
        df, pip_size, config["spread"], split_i, len(df)
    )

    train_stats = summarize_trades(train_trades)
    test_stats = summarize_trades(test_trades)

    # BUYとSELLを別々に集計
    direction_stats = summarize_by_direction(test_trades)

    # 後半のテスト期間を3分割して安定性を確認
    window_stats = []
    test_length = len(df) - split_i
    boundaries = np.linspace(
        split_i, len(df), TEST_WINDOWS + 1, dtype=int
    )

    for n in range(TEST_WINDOWS):
        start_i = int(boundaries[n])
        end_i = int(boundaries[n + 1])

        window_trades = backtest(
            df, pip_size, config["spread"], start_i, end_i
        )
        window_stats.append({
            "window": n + 1,
            "from": str(df.index[start_i]),
            "to": str(df.index[max(start_i, end_i - 1)]),
            "stats": summarize_trades(window_trades),
        })

    # 期間別の安定性は診断情報として記録する。
    # 既存の最低判定基準を勝手に緩めない。
    usable_windows = [
        w["stats"] for w in window_stats
        if w["stats"]["samples"] >= MIN_WINDOW_SAMPLES
    ]
    positive_windows = sum(
        1 for s in usable_windows
        if s["avg_net_pips"] is not None and s["avg_net_pips"] > 0
    )

    current_signal, current_score, reasons = make_signal(df.iloc[-1])

    quality_ok = (
        train_stats["samples"] >= MIN_TRAIN_SAMPLES
        and test_stats["samples"] >= MIN_TEST_SAMPLES
        and test_stats["hit_rate_pct"] is not None
        and test_stats["hit_rate_pct"] >= MIN_DIRECTION_HIT_RATE * 100
        and test_stats["avg_net_pips"] is not None
        and test_stats["avg_net_pips"] > MIN_AVG_NET_PIPS
    )

    if quality_ok:
        decision = current_signal
        decision_reason = (
            "全体テストの最低基準を満たしました。"
            "期間別の安定性と売買方向別の成績も確認してください。"
            "将来の利益を保証するものではありません。"
        )
    else:
        decision = "判定保留"
        decision_reason = (
            "サンプル数またはテスト成績が基準未達です。"
            "この結果だけで売買判断をしないでください。"
        )

    return {
        "pair": name,
        "bars": len(df),
        "latest_bar_time": str(df.index[-1]),
        "signal": current_signal,
        "direction_score": int(current_score),
        "decision": decision,
        "decision_reason": decision_reason,
        "reasons": reasons,
        "train_period": {
            "from": str(df.index[0]),
            "to": str(df.index[max(0, split_i - 1)]),
        },
        "test_period": {
            "from": str(df.index[split_i]),
            "to": str(df.index[-1]),
        },
        "train": train_stats,
        "test": test_stats,
        "test_by_direction": direction_stats,
        "test_windows": window_stats,
        "window_stability": {
            "usable_windows": len(usable_windows),
            "positive_windows": positive_windows,
            "note": (
                "期間別の件数が少ない場合、比較結果は不安定です。"
                "これは独立した将来予測ではありません。"
            ),
        },
        "notes": [
            "強度スコアは勝率や確率ではありません。",
            "次の足の始値でエントリーする想定です。",
            "スプレッドは固定値を差し引いた参考値です。",
            "スリッページ、スワップ、実際の約定差は未反映です。",
            "期間別成績は過去の診断であり、将来の成績を保証しません。",
        ],
    }


def print_stats(label, stats):
    print(f"\n{label}")
    print(f"サンプル数: {stats['samples']}")
    print(f"勝率: {stats['hit_rate_pct']}%")
    print(f"勝ち / 負け: {stats['wins']} / {stats['losses']}")
    print(f"平均利益: {stats['avg_win_pips']} pips")
    print(f"平均損失: {stats['avg_loss_pips']} pips")
    print(f"損益比: {stats['profit_loss_ratio']}")
    print(f"平均損益: {stats['avg_net_pips']} pips")
    print(f"合計損益: {stats['total_net_pips']} pips")


def main():
    print("=" * 64)
    print("FX Simple Scanner - 複数期間・方向別検証版")
    print(f"実行時刻: {now_jst()}")
    print("分析専用 / 自動発注なし / 口座接続なし")
    print("=" * 64)

    results = []

    for name, config in PAIRS.items():
        print(f"\n--- {name} ---")
        result = evaluate_pair(name, config)
        results.append(result)

        if "error" in result:
            print(f"状態: {result['decision']}")
            print(f"理由: {result['error']}")
            continue

        print(f"使用データ本数: {result['bars']}")
        print(f"現在のサイン: {result['signal']}")
        print(f"方向スコア: {result['direction_score']}")
        print(f"検証判定: {result['decision']}")
        print(result["decision_reason"])

        print_stats("前半・確認期間", result["train"])
        print_stats("後半・全体テスト", result["test"])

        print("\nテスト期間のBUY・SELL別成績")
        for direction, stats in result["test_by_direction"].items():
            print_stats(direction, stats)

        print("\nテスト期間の3分割検証")
        for window in result["test_windows"]:
            print(
                f"期間{window['window']}: "
                f"{window['from']} ～ {window['to']}"
            )
            print_stats("期間別成績", window["stats"])

        stability = result["window_stability"]
        print(
            "プラス損益の期間数: "
            f"{stability['positive_windows']} / "
            f"{stability['usable_windows']} "
            "(最低件数を満たした期間のみ)"
        )

        print("\n注意点:")
        for note in result["notes"]:
            print(f"- {note}")

    payload = {
        "schema_version": 5,
        "last_run_jst": now_jst(),
        "mode": "multi_window_direction_split_no_auto_orders",
        "interval": INTERVAL,
        "lookahead_bars": LOOKAHEAD_BARS,
        "train_ratio": TRAIN_RATIO,
        "test_windows": TEST_WINDOWS,
        "results": results,
    }

    save_json(STATE_FILE, payload)

    history = load_history()
    history.append(payload)
    save_json(HISTORY_FILE, history[-200:])

    print("\n" + "=" * 64)
    print(f"結果保存: {STATE_FILE}")
    print(f"履歴保存: {HISTORY_FILE}")
    print("実行終了")
    print("=" * 64)


if __name__ == "__main__":
    main()