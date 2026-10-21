# FXバックグラウンド分析・複数期間の学習検証（参考情報のみ・自動発注なし）
# Discord通知は行わず、実行レポートを scan_state.json に保存します。
# 注意：これは検証記録用です。利益を保証するものではありません。
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

JST = ZoneInfo("Asia/Tokyo")
STATE_FILE = "scan_state.json"
LOOKAHEAD_BARS = 8
MIN_TRAIN_ROWS = 180
N_WALK_FORWARD_FOLDS = 3

PAIRS = {
    "米ドル/円 (USD/JPY)": "USDJPY=X",
    "英ポンド/円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ/円 (EUR/JPY)": "EURJPY=X",
    "豪ドル/円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ/米ドル (EUR/USD)": "EURUSD=X",
}
SPREAD = {
    "USDJPY=X": 0.2,
    "EURUSD=X": 0.4,
    "GBPJPY=X": 1.0,
    "EURJPY=X": 0.5,
    "AUDJPY=X": 0.6,
}
FEATURES = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200",
    "Dev_EMA20_200", "Vol_Ratio", "RSI", "RSI_Diff",
    "MACD_Hist_Ratio", "BB_PctB", "ADX", "ATR_Ratio",
    "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Stoch_K", "CCI", "ROC",
]


def bars(ticker, interval, period):
    """取得価格から指標と先読みラベルを作成。未完成の足は除外する。"""
    source = "1h" if interval == "4h" else interval
    d = yf.download(
        ticker,
        period=period,
        interval=source,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=25,
    )
    if d is None or d.empty:
        return None

    if isinstance(d.columns, pd.MultiIndex):
        try:
            d = d.xs(ticker, axis=1, level=-1)
        except Exception:
            try:
                d = d.xs(ticker, axis=1, level=0)
            except Exception:
                d.columns = [
                    item[0] if isinstance(item, tuple) else item
                    for item in d.columns
                ]

    required = ["Open", "High", "Low", "Close"]
    if not set(required).issubset(d.columns):
        return None

    d = d[required].apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    if interval == "4h":
        d = d.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }).dropna()

    # 最後の足は形成中の可能性があるため使用しない
    if len(d) > 2:
        d = d.iloc[:-1]
    if len(d) < 220:
        return None

    c, h, l, o = d["Close"], d["High"], d["Low"], d["Open"]
    d["SMA_20"] = c.rolling(20, min_periods=10).mean()
    d["EMA_20"] = c.ewm(span=20, adjust=False).mean()
    d["EMA_200"] = c.ewm(span=200, adjust=False).mean()

    prev = c.shift()
    tr = pd.concat(
        [h - l, (h - prev).abs(), (l - prev).abs()], axis=1
    ).max(axis=1)
    d["ATR"] = tr.rolling(14, min_periods=7).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = (100 - 100 / (1 + rs)).fillna(50)
    d["RSI_Diff"] = d["RSI"].diff()

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    d["MACD_Hist_Ratio"] = hist / c.replace(0, np.nan)

    std = c.rolling(20, min_periods=10).std()
    upper, lower = d["SMA_20"] + 2 * std, d["SMA_20"] - 2 * std
    d["BB_PctB"] = (c - lower) / (upper - lower).replace(0, np.nan)

    d["Return_1"] = c.pct_change()
    d["Return_5"] = c.pct_change(5)
    d["Dev_SMA20"] = (c - d["SMA_20"]) / c.replace(0, np.nan)
    d["Dev_EMA200"] = (c - d["EMA_200"]) / c.replace(0, np.nan)
    d["Dev_EMA20_200"] = (d["EMA_20"] - d["EMA_200"]) / c.replace(0, np.nan)
    d["ATR_Ratio"] = d["ATR"] / c.replace(0, np.nan)
    d["Vol_Ratio"] = d["ATR"] / d["ATR"].rolling(50, min_periods=10).mean().replace(0, np.nan)

    up, down = h.diff(), -l.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    safe_atr = d["ATR"].replace(0, np.nan)
    pdi = 100 * plus.rolling(14, min_periods=7).mean() / safe_atr
    mdi = 100 * minus.rolling(14, min_periods=7).mean() / safe_atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["ADX"] = dx.rolling(14, min_periods=7).mean().fillna(20)

    rng = (h - l).replace(0, np.nan)
    d["Upper_Wick_Ratio"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    d["Lower_Wick_Ratio"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    lo14 = l.rolling(14, min_periods=7).min()
    hi14 = h.rolling(14, min_periods=7).max()
    d["Stoch_K"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)

    tp = (h + l + c) / 3
    tps = tp.rolling(20, min_periods=10).mean()
    mad = tp.rolling(20, min_periods=10).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    d["CCI"] = (tp - tps) / (0.015 * mad.replace(0, np.nan))
    d["ROC"] = c.pct_change(10) * 100

    pip = 0.01 if "JPY" in ticker else 0.0001
    spread = SPREAD.get(ticker, 0.5) * pip
    barrier = d["ATR"].clip(lower=spread * 2).to_numpy()
    cv, hv, lv = c.to_numpy(), h.to_numpy(), l.to_numpy()
    y = np.full(len(d), np.nan)

    # 8本先までのバリア判定。最終8本は未来データ不足のため未ラベル。
    for i in range(max(0, len(d) - LOOKAHEAD_BARS)):
        upper_barrier = cv[i] + barrier[i] + spread
        lower_barrier = cv[i] - barrier[i] - spread
        label = 0
        for j in range(i + 1, min(i + LOOKAHEAD_BARS + 1, len(d))):
            hit_up = hv[j] >= upper_barrier
            hit_down = lv[j] <= lower_barrier
            # 同一足で上下両方に触れた場合、順序不明のため中立ラベルにする
            if hit_up and hit_down:
                label = 0
                break
            if hit_up:
                label = 1
                break
            if hit_down:
                label = -1
                break
        y[i] = label

    d["Target"] = y
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["ATR"])


def trend(d):
    if d is None or d.empty:
        return "不明"
    close = float(d["Close"].iloc[-1])
    ema20 = float(d["EMA_20"].iloc[-1])
    ema200 = float(d["EMA_200"].iloc[-1])
    if close > ema20 > ema200:
        return "上昇"
    if close < ema20 < ema200:
        return "下降"
    return "中立"


def make_model(seed=42):
    return RandomForestClassifier(
        n_estimators=220,
        max_depth=7,
        min_samples_leaf=8,
        class_weight=None,  # sample_weight側でクラス重みを一度だけ適用
        random_state=seed,
        n_jobs=-1,
    )


def walk_forward_validation(X, y):
    """
    Expanding-window検証。各検証区間の直前8サンプルを学習から除外し、
    先読みラベルが検証期間にまたがる可能性を抑える。
    """
    n = len(X)
    if n < MIN_TRAIN_ROWS + 3 * 30:
        return {
            "passed": False,
            "mean_balanced_accuracy_pct": 0.0,
            "mean_baseline_balanced_accuracy_pct": 0.0,
            "folds_passed": 0,
            "fold_count": 0,
            "fold_results": [],
            "reason": "複数期間の検証に必要なデータが不足",
        }

    # 最初の約55%を初期学習用、残りを3つの連続した検証区間にする
    initial_train_end = int(n * 0.55)
    remaining = n - initial_train_end
    block = remaining // N_WALK_FORWARD_FOLDS
    fold_results = []

    for fold_idx in range(N_WALK_FORWARD_FOLDS):
        val_start = initial_train_end + fold_idx * block
        val_end = n if fold_idx == N_WALK_FORWARD_FOLDS - 1 else initial_train_end + (fold_idx + 1) * block
        train_end = max(0, val_start - LOOKAHEAD_BARS)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[val_start:val_end], y.iloc[val_start:val_end]

        if (
            len(X_train) < MIN_TRAIN_ROWS
            or len(X_val) < 30
            or y_train.nunique() < 2
            or y_val.nunique() < 2
        ):
            fold_results.append({
                "fold": fold_idx + 1,
                "status": "skipped",
                "reason": "学習または検証データのクラス数・件数が不足",
                "train_samples": int(len(X_train)),
                "validation_samples": int(len(X_val)),
            })
            continue

        model = make_model(seed=42 + fold_idx)
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=weights)
        pred = model.predict(X_val)
        bal = balanced_accuracy_score(y_val, pred) * 100

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        base_pred = baseline.predict(X_val)
        base_bal = balanced_accuracy_score(y_val, base_pred) * 100

        fold_passed = bool(bal >= max(38.0, base_bal + 3.0))
        fold_results.append({
            "fold": fold_idx + 1,
            "status": "ok",
            "train_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "balanced_accuracy_pct": round(float(bal), 1),
            "baseline_balanced_accuracy_pct": round(float(base_bal), 1),
            "passed": fold_passed,
        })

    valid_folds = [f for f in fold_results if f.get("status") == "ok"]
    if len(valid_folds) < 2:
        return {
            "passed": False,
            "mean_balanced_accuracy_pct": 0.0,
            "mean_baseline_balanced_accuracy_pct": 0.0,
            "folds_passed": 0,
            "fold_count": len(valid_folds),
            "fold_results": fold_results,
            "reason": "有効な検証区間が2つ未満",
        }

    mean_bal = float(np.mean([f["balanced_accuracy_pct"] for f in valid_folds]))
    mean_base = float(np.mean([f["baseline_balanced_accuracy_pct"] for f in valid_folds]))
    folds_passed = sum(bool(f["passed"]) for f in valid_folds)

    # 複数区間の平均だけでなく、少なくとも半数超の区間で条件を満たすことを要求
    passed = bool(
        mean_bal >= max(38.0, mean_base + 3.0)
        and folds_passed >= (len(valid_folds) // 2 + 1)
    )
    return {
        "passed": passed,
        "mean_balanced_accuracy_pct": round(mean_bal, 1),
        "mean_baseline_balanced_accuracy_pct": round(mean_base, 1),
        "folds_passed": int(folds_passed),
        "fold_count": int(len(valid_folds)),
        "fold_results": fold_results,
        "reason": "複数期間の検証条件を満たした" if passed else "複数期間の検証品質条件を満たさない",
    }


def analyze(d):
    fallback = {
        "dir": "様子見",
        "buy": 0,
        "sell": 0,
        "quality": False,
        "bal": 0.0,
        "baseline_bal": 0.0,
        "samples": 0,
        "validation": {
            "passed": False,
            "mean_balanced_accuracy_pct": 0.0,
            "mean_baseline_balanced_accuracy_pct": 0.0,
            "folds_passed": 0,
            "fold_count": 0,
            "fold_results": [],
            "reason": "データ不足",
        },
    }
    if d is None or len(d) < MIN_TRAIN_ROWS:
        return fallback

    xa = d.reindex(columns=FEATURES).replace([np.inf, -np.inf], np.nan)
    labels = pd.to_numeric(d["Target"], errors="coerce")
    mask = labels.notna() & xa.notna().all(axis=1)
    X = xa.loc[mask]
    y = labels.loc[mask].astype(int)

    if len(X) < MIN_TRAIN_ROWS or y.nunique() < 2:
        return fallback

    validation = walk_forward_validation(X, y)
    quality = bool(validation["passed"])

    # 現在のシグナル生成用モデル。検証の合否とは別に学習するが、
    # quality=False の場合は買い・売りシグナルを出さない。
    model = make_model(seed=43)
    weights = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(X, y, sample_weight=weights)

    latest = xa.dropna()
    if latest.empty:
        return fallback

    probabilities = {
        int(cls): float(prob)
        for cls, prob in zip(
            model.classes_, model.predict_proba(latest.iloc[[-1]])[0]
        )
    }
    buy = probabilities.get(1, 0.0)
    sell = probabilities.get(-1, 0.0)
    rsi = float(d["RSI"].iloc[-1])
    direction = "様子見"

    if quality and buy >= 0.53 and buy > sell and rsi < 78:
        direction = "買い"
    elif quality and sell >= 0.53 and sell > buy and rsi > 22:
        direction = "売り"

    return {
        "dir": direction,
        "buy": round(buy * 100),
        "sell": round(sell * 100),
        "quality": quality,
        "bal": validation["mean_balanced_accuracy_pct"],
        "baseline_bal": validation["mean_baseline_balanced_accuracy_pct"],
        "samples": int(len(X)),
        "validation": validation,
    }


def load_previous_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def main():
    old = load_previous_state()
    interval = os.getenv("FX_SCAN_INTERVAL", "1h")
    interval = interval if interval in ("1h", "4h") else "1h"
    period = "60d" if interval == "1h" else "90d"

    report, metrics = [], {}
    for label, ticker in PAIRS.items():
        try:
            cur = bars(ticker, interval, period)
            four = bars(ticker, "4h", "90d")
            daily = bars(ticker, "1d", "2y")
            one = cur if interval == "1h" else bars(ticker, "1h", "60d")

            if any(frame is None for frame in (cur, four, daily, one)):
                report.append(f"{label}: データ不足・判定保留")
                metrics[ticker] = {"pair": label, "status": "insufficient_data"}
                continue

            result = analyze(cur)
            direction = result["dir"]
            trends = [trend(one), trend(four), trend(daily)]
            matching_direction = {"買い": "上昇", "売り": "下降"}.get(direction)
            matching = sum(item == matching_direction for item in trends) if matching_direction else 0
            opposing = (
                (direction == "買い" and "下降" in trends)
                or (direction == "売り" and "上昇" in trends)
            )

            stamp = pd.Timestamp(cur.index[-1])
            stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            age = (pd.Timestamp.now(tz="UTC") - stamp).total_seconds() / 60
            pip = 0.01 if "JPY" in ticker else 0.0001
            enough = float(cur["ATR"].iloc[-1]) / pip >= SPREAD.get(ticker, 0.5) * 3

            if age > (210 if interval == "1h" else 600):
                decision, reason = "判定保留", "データが古い可能性"
            elif not result["quality"]:
                decision, reason = "見送り", "複数期間の時系列検証条件を満たさない"
            elif not enough:
                decision, reason = "見送り", "推定スプレッド負担が大きい可能性"
            elif direction == "様子見":
                decision, reason = "様子見", "明確なAI方向シグナルなし"
            elif opposing:
                decision, reason = "方向不一致・見送り", "時間足の一部が逆方向"
            elif matching < 2:
                decision, reason = "様子見", "時間足一致が不十分"
            else:
                decision, reason = "候補あり", f"AI方向と複数時間足が一致（{matching}/3）"

            close = float(cur["Close"].iloc[-1])
            barstamp = stamp.tz_convert(JST).strftime("%Y-%m-%d %H:%M")
            report.append(
                f'{label}: {decision} / {direction} / '
                f'買い{result["buy"]}%・売り{result["sell"]}% / '
                f"価格参考 {close:.5f} / {reason}"
            )
            metrics[ticker] = {
                "pair": label,
                "status": "ok",
                "decision": decision,
                "direction": direction,
                "close": close,
                "bar_time_jst": barstamp,
                "data_age_minutes": round(age, 1),
                "validation_balanced_accuracy_pct": result["bal"],
                "baseline_balanced_accuracy_pct": result["baseline_bal"],
                "quality_gate_passed": result["quality"],
                "training_samples": result["samples"],
                "validation": result["validation"],
                "timeframe_trends": trends,
            }
        except Exception as exc:
            report.append(f"{label}: 取得・分析失敗 ({type(exc).__name__})")
            metrics[ticker] = {
                "pair": label,
                "status": "error",
                "error_type": type(exc).__name__,
            }

    now = datetime.now(JST).isoformat()
    previous_history = old.get("history", [])
    if not isinstance(previous_history, list):
        previous_history = []

    state = {
        "schema_version": 3,
        "last_run_jst": now,
        "mode": "multi_period_validation_no_discord_no_auto_orders",
        "interval": interval,
        "summary": report,
        "metrics": metrics,
        "note": (
            "複数期間の時系列検証レポートです。学習済みモデルを永続保存・app.pyへ配布する機能はありません。"
            "検証指標や予測確率は将来の利益を保証しません。実取引前に取引コスト込みのバックテストが必要です。"
        ),
        "history": previous_history[-99:],
    }
    state["history"].append({"time_jst": now, "metrics": metrics})

    temp_file = STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp_file, STATE_FILE)

    print("\n".join(report))
    print("Completed; Discord notifications disabled; multi-period validation report saved.")


if __name__ == "__main__":
    main()
