import json
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
# =========================================================
# FX Simple Scanner
# 慎重判定・複数期間・方向別検証版
# 前半の平均損益がプラスであることも必須条件にする
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
MIN_DIRECTION_SAMPLES = 30
MIN_DIRECTION_HIT_RATE = 0.50
# 前半・後半とも、平均損益がこの値を上回ること
# 0.0の場合、平均損益が0以下なら不合格
MIN_TRAIN_AVG_NET_PIPS = 0.0
MIN_AVG_NET_PIPS = 0.0
MIN_DIRECTION_AVG_NET_PIPS = 0.0
TEST_WINDOWS = 3
MIN_WINDOW_SAMPLES = 10
MIN_POSITIVE_WINDOWS = 3
# 1取引あたりのスリッページ想定値（pips）
# 実際の値を保証するものではありません
SLIPPAGE_PIPS = 0.2
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
def timestamp_jst(value):
    """日時を日本時間のISO形式に統一する。"""
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("Asia/Tokyo").isoformat()
    except Exception:
        return str(value)
def json_default(value):
    """NumPy・Pandasの値をJSONに保存できる形式にする。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            return None
        return result
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return timestamp_jst(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
def save_json(path, data):
    """一時ファイル経由でJSONを保存する。"""
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
            allow_nan=False,
        )
    os.replace(temp_path, path)
def load_history():
    """過去の実行履歴を読み込む。"""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []
def download_prices(ticker):
    """価格を取得し、OHLCの整合性を確認する。"""
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
    valid = (
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["High"] >= df["Low"])
        & (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
    )
    df = df.loc[valid].copy()
    # 最後の足は未確定の可能性があるため除外
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    if len(df) < MIN_BARS:
        return None, (
            f"データ不足: {len(df)}本。"
            f"最低{MIN_BARS}本必要です"
        )
    return df, None
def add_indicators(df):
    """テクニカル指標を計算する。"""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    df["EMA20"] = close.ewm(
        span=20,
        adjust=False,
    ).mean()
    df["EMA50"] = close.ewm(
        span=50,
        adjust=False,
    ).mean()
    df["EMA20_SLOPE"] = (
        df["EMA20"] - df["EMA20"].shift(5)
    )
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI",
    ] = 100
    df.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI",
    ] = 50
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
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()
    return df.replace([np.inf, -np.inf], np.nan)
def make_signal(row):
    """EMAとRSIを使い、BUY・SELL・WAITを判定する。"""
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
    """
    指定範囲でバックテストする。
    ・シグナル足の次の足の始値でエントリー
    ・シグナル足からLOOKAHEAD_BARS後の終値で決済
    ・検証範囲内で決済できる取引だけ採用
    ・保有期間が重複しないようにする
    """
    trades = []
    start_i = max(int(start_i), 55)
    end_i = min(int(end_i), len(df))
    if end_i <= start_i:
        return trades
    last_signal_i = min(
        end_i - LOOKAHEAD_BARS - 1,
        len(df) - LOOKAHEAD_BARS - 1,
    )
    i = start_i
    while i <= last_signal_i:
        signal, score, _ = make_signal(df.iloc[i])
        if signal not in ("BUY", "SELL"):
            i += 1
            continue
        entry_i = i + 1
        exit_i = i + LOOKAHEAD_BARS
        if entry_i >= end_i or exit_i >= end_i:
            i += 1
            continue
        try:
            entry_price = float(df["Open"].iloc[entry_i])
            exit_price = float(df["Close"].iloc[exit_i])
        except (ValueError, TypeError, IndexError):
            i += 1
            continue
        if (
            not math.isfinite(entry_price)
            or not math.isfinite(exit_price)
            or entry_price <= 0
            or exit_price <= 0
        ):
            i += 1
            continue
        direction = 1 if signal == "BUY" else -1
        gross_pips = (
            (exit_price - entry_price)
            * direction
            / pip_size
        )
        total_cost_pips = float(spread) + SLIPPAGE_PIPS
        net_pips = gross_pips - total_cost_pips
        if not math.isfinite(net_pips):
            i += 1
            continue
        trades.append({
            "signal_time": timestamp_jst(df.index[i]),
            "entry_time": timestamp_jst(df.index[entry_i]),
            "exit_time": timestamp_jst(df.index[exit_i]),
            "signal": signal,
            "score": int(score),
            "gross_pips": round(float(gross_pips), 4),
            "spread_pips": float(spread),
            "slippage_pips": float(SLIPPAGE_PIPS),
            "net_pips": round(float(net_pips), 4),
            "win": bool(net_pips > 0),
        })
        # 決済足の終値で決済する想定。
        # 次の確認は決済足から再開する。
        i = exit_i
    return trades
def summarize_trades(trades):
    """取引成績を集計する。"""
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
    net = np.array(
        [t["net_pips"] for t in trades],
        dtype=float,
    )
    net = net[np.isfinite(net)]
    if len(net) == 0:
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
    winners = net[net > 0]
    losers = net[net < 0]
    avg_win = float(winners.mean()) if len(winners) else None
    avg_loss = float(abs(losers.mean())) if len(losers) else None
    pl_ratio = (
        avg_win / avg_loss
        if (
            avg_win is not None
            and avg_loss is not None
            and avg_loss > 0
        )
        else None
    )
    return {
        "samples": int(len(net)),
        "hit_rate_pct": round(
            float((net > 0).mean() * 100),
            2,
        ),
        "wins": int((net > 0).sum()),
        "losses": int((net < 0).sum()),
        "avg_win_pips": (
            round(avg_win, 2) if avg_win is not None else None
        ),
        "avg_loss_pips": (
            round(avg_loss, 2) if avg_loss is not None else None
        ),
        "profit_loss_ratio": (
            round(pl_ratio, 2) if pl_ratio is not None else None
        ),
        "avg_net_pips": round(float(net.mean()), 2),
        "total_net_pips": round(float(net.sum()), 2),
    }
def summarize_by_direction(trades):
    """BUYとSELLを分けて集計する。"""
    return {
        direction: summarize_trades(
            [t for t in trades if t["signal"] == direction]
        )
        for direction in ("BUY", "SELL")
    }
def evaluate_pair(name, config):
    """1通貨ペアを分析・検証する。"""
    df, error = download_prices(config["ticker"])
    if error:
        return {
            "pair": name,
            "error": error,
            "signal": "WAIT",
            "decision": "判定保留",
            "decision_reason": error,
        }
    df = add_indicators(df)
    pip_size = (
        0.01 if "JPY" in config["ticker"] else 0.0001
    )
    split_i = int(len(df) * TRAIN_RATIO)
    if split_i <= 55 or split_i >= len(df):
        return {
            "pair": name,
            "error": "訓練期間とテスト期間を分けるためのデータが不足",
            "signal": "WAIT",
            "decision": "判定保留",
            "decision_reason": "期間分割ができないため",
        }
    # 前半の終端を手前にずらし、境界をまたぐ決済を避ける
    train_end = max(56, split_i - LOOKAHEAD_BARS)
    train_trades = backtest(
        df,
        pip_size,
        config["spread"],
        55,
        train_end,
    )
    test_trades = backtest(
        df,
        pip_size,
        config["spread"],
        split_i,
        len(df),
    )
    train_stats = summarize_trades(train_trades)
    test_stats = summarize_trades(test_trades)
    direction_stats = summarize_by_direction(test_trades)
    # 後半のテスト期間を3つに分割
    window_stats = []
    boundaries = np.linspace(
        split_i,
        len(df),
        TEST_WINDOWS + 1,
        dtype=int,
    )
    for n in range(TEST_WINDOWS):
        start_i = int(boundaries[n])
        end_i = int(boundaries[n + 1])
        window_trades = backtest(
            df,
            pip_size,
            config["spread"],
            start_i,
            end_i,
        )
        window_stats.append({
            "window": n + 1,
            "from": timestamp_jst(df.index[start_i]),
            "to": timestamp_jst(
                df.index[max(start_i, end_i - 1)]
            ),
            "stats": summarize_trades(window_trades),
        })
    usable_windows = [
        w["stats"]
        for w in window_stats
        if w["stats"]["samples"] >= MIN_WINDOW_SAMPLES
    ]
    positive_windows = sum(
        1
        for stats in usable_windows
        if (
            stats["avg_net_pips"] is not None
            and stats["avg_net_pips"] > 0
        )
    )
    current_signal, current_score, reasons = make_signal(
        df.iloc[-1]
    )
    # 前半の検証条件
    train_ok = (
        train_stats["samples"] >= MIN_TRAIN_SAMPLES
        and train_stats["avg_net_pips"] is not None
        and train_stats["avg_net_pips"] > MIN_TRAIN_AVG_NET_PIPS
    )
    # 後半全体の検証条件
    overall_ok = (
        test_stats["samples"] >= MIN_TEST_SAMPLES
        and test_stats["avg_net_pips"] is not None
        and test_stats["avg_net_pips"] > MIN_AVG_NET_PIPS
    )
    # 現在のサイン方向と同じ方向の過去成績を確認
    current_direction_stats = None
    direction_ok = False
    if current_signal in ("BUY", "SELL"):
        current_direction_stats = direction_stats[current_signal]
        direction_ok = (
            current_direction_stats["samples"]
            >= MIN_DIRECTION_SAMPLES
            and current_direction_stats["hit_rate_pct"] is not None
            and current_direction_stats["hit_rate_pct"]
            >= MIN_DIRECTION_HIT_RATE * 100
            and current_direction_stats["avg_net_pips"] is not None
            and current_direction_stats["avg_net_pips"]
            > MIN_DIRECTION_AVG_NET_PIPS
        )
    # 3期間すべてで最低サンプル数とプラス損益を要求
    stability_ok = (
        len(usable_windows) == TEST_WINDOWS
        and positive_windows >= MIN_POSITIVE_WINDOWS
    )
    failed_checks = []
    if train_stats["samples"] < MIN_TRAIN_SAMPLES:
        failed_checks.append("前半の検証サンプル数が不足")
    if train_stats["avg_net_pips"] is None:
        failed_checks.append("前半の平均損益を計算できない")
    elif train_stats["avg_net_pips"] <= MIN_TRAIN_AVG_NET_PIPS:
        failed_checks.append(
            "前半の平均損益が基準未達（0 pips以下）"
        )
    if test_stats["samples"] < MIN_TEST_SAMPLES:
        failed_checks.append("後半のテストサンプル数が不足")
    if test_stats["avg_net_pips"] is None:
        failed_checks.append("後半全体の平均損益を計算できない")
    elif test_stats["avg_net_pips"] <= MIN_AVG_NET_PIPS:
        failed_checks.append("後半全体の平均損益が基準未達")
    if current_signal == "WAIT":
        failed_checks.append("現在のテクニカルサインが様子見")
    elif not direction_ok:
        failed_checks.append(
            f"現在のサイン方向（{current_signal}）の成績が基準未達"
        )
    if len(usable_windows) < TEST_WINDOWS:
        failed_checks.append(
            "3期間すべてで最低サンプル数を満たしていない"
        )
    elif positive_windows < MIN_POSITIVE_WINDOWS:
        failed_checks.append(
            "3期間すべての平均損益がプラスではない"
        )
    # 前半・後半・方向別・期間別のすべてを満たすこと
    quality_ok = (
        train_ok
        and overall_ok
        and direction_ok
        and stability_ok
        and current_signal in ("BUY", "SELL")
    )
    if quality_ok:
        decision = current_signal
        decision_reason = (
            "前半・後半の平均損益、現在の売買方向別成績、"
            "3期間の安定性が設定基準を満たしました。"
            "将来の利益を保証するものではありません。"
        )
    else:
        decision = "判定保留"
        if not failed_checks:
            failed_checks.append("総合条件を満たしていません")
        decision_reason = " / ".join(failed_checks)
    return {
        "pair": name,
        "bars": len(df),
        "latest_bar_time": timestamp_jst(df.index[-1]),
        "signal": current_signal,
        "direction_score": int(current_score),
        "decision": decision,
        "decision_reason": decision_reason,
        "reasons": reasons,
        "train_period": {
            "from": timestamp_jst(df.index[0]),
            "to": timestamp_jst(
                df.index[max(0, split_i - 1)]
            ),
        },
        "test_period": {
            "from": timestamp_jst(df.index[split_i]),
            "to": timestamp_jst(df.index[-1]),
        },
        "train": train_stats,
        "test": test_stats,
        "test_by_direction": direction_stats,
        "current_direction_test": current_direction_stats,
        "test_windows": window_stats,
        "cost_assumptions": {
            "spread_pips": float(config["spread"]),
            "slippage_pips_per_trade": float(SLIPPAGE_PIPS),
            "note": (
                "固定値による仮定です。実際のスプレッド、"
                "スリッページ、スワップ、約定差とは異なります。"
            ),
        },
        "window_stability": {
            "usable_windows": len(usable_windows),
            "positive_windows": positive_windows,
            "required_windows": TEST_WINDOWS,
            "minimum_samples_per_window": MIN_WINDOW_SAMPLES,
            "passed": stability_ok,
            "note": (
                "期間別成績は独立した将来予測ではありません。"
            ),
        },
        "quality_checks": {
            "train_passed": train_ok,
            "overall_test_passed": overall_ok,
            "current_direction_passed": direction_ok,
            "window_stability_passed": stability_ok,
            "all_passed": quality_ok,
            "failed_checks": failed_checks,
        },
        "notes": [
            "強度スコアは勝率や確率ではありません。",
            "次の足の始値でエントリーする想定です。",
            "決済はシグナル足から所定本数後の終値を使います。",
            "固定スプレッドとスリッページ想定値を差し引いています。",
            "スワップ、実際の約定差、指値・逆指値の約定条件は未反映です。",
            "前半の平均損益が0 pips以下の場合は判定保留です。",
            "後半の平均損益もプラスである必要があります。",
            "過去の検証成績は将来の利益を保証しません。",
            "判定が通っても、実際の取引を推奨するものではありません。",
            "シグナルがない場合は1本ずつ確認し、取引期間の重複を避けます。",
        ],
    }
def print_stats(label, stats):
    """成績を表示する。"""
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
    print("FX Simple Scanner - 前半成績も必須とする慎重判定版")
    print(f"実行時刻: {now_jst()}")
    print("分析専用 / 自動発注なし / 口座接続なし")
    print("=" * 64)
    results = []
    for name, config in PAIRS.items():
        print(f"\n--- {name} ---")
        try:
            result = evaluate_pair(name, config)
        except Exception as exc:
            result = {
                "pair": name,
                "signal": "WAIT",
                "decision": "判定保留",
                "error": f"分析中のエラー: {exc}",
                "decision_reason": (
                    "分析中にエラーが発生したため判定を保留しました。"
                ),
            }
        results.append(result)
        if "error" in result:
            print(f"状態: {result['decision']}")
            print(f"理由: {result['error']}")
            continue
        print(f"使用データ本数: {result['bars']}")
        print(f"最新足の時刻: {result['latest_bar_time']}")
        print(f"現在のサイン: {result['signal']}")
        print(f"方向スコア: {result['direction_score']}")
        print(f"検証判定: {result['decision']}")
        print(f"判定理由: {result['decision_reason']}")
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
            f"{stability['usable_windows']}"
        )
        checks = result["quality_checks"]
        print("\n判定条件の確認")
        print(f"前半の成績: {checks['train_passed']}")
        print(f"後半全体の成績: {checks['overall_test_passed']}")
        print(f"方向別成績: {checks['current_direction_passed']}")
        print(f"期間の安定性: {checks['window_stability_passed']}")
        print(f"総合判定: {checks['all_passed']}")
        if checks["failed_checks"]:
            print("未達条件:")
            for item in checks["failed_checks"]:
                print(f"- {item}")
        print("\n注意点:")
        for note in result["notes"]:
            print(f"- {note}")
    payload = {
        "schema_version": 9,
        "last_run_jst": now_jst(),
        "mode": (
            "strict_train_test_direction_window_validation"
            "_with_cost_assumptions_no_auto_orders_v3"
        ),
        "interval": INTERVAL,
        "lookahead_bars": LOOKAHEAD_BARS,
        "train_ratio": TRAIN_RATIO,
        "test_windows": TEST_WINDOWS,
        "cost_assumptions": {
            "spread_source": "PAIRS設定の固定値",
            "slippage_pips_per_trade": SLIPPAGE_PIPS,
        },
        "criteria": {
            "min_train_samples": MIN_TRAIN_SAMPLES,
            "min_train_avg_net_pips_exclusive": (
                MIN_TRAIN_AVG_NET_PIPS
            ),
            "min_test_samples": MIN_TEST_SAMPLES,
            "min_avg_net_pips_exclusive": MIN_AVG_NET_PIPS,
            "min_direction_samples": MIN_DIRECTION_SAMPLES,
            "min_direction_hit_rate_pct": (
                MIN_DIRECTION_HIT_RATE * 100
            ),
            "min_direction_avg_net_pips_exclusive": (
                MIN_DIRECTION_AVG_NET_PIPS
            ),
            "min_window_samples": MIN_WINDOW_SAMPLES,
            "min_positive_windows": MIN_POSITIVE_WINDOWS,
        },
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