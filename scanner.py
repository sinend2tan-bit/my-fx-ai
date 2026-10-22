
# FXバックグラウンド分析・複数期間の学習検証
# 参考情報専用・自動発注なし・Discord通知なし
# 急変動時の判定保留、クラス別再現率、判定理由の記録を追加
# 注意：利益や予測精度を保証するものではありません。

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

JST = ZoneInfo("Asia/Tokyo")
STATE_FILE = "scan_state.json"
LOOKAHEAD_BARS = 8
MIN_TRAIN_ROWS = 180
N_WALK_FORWARD_FOLDS = 3
FAST_MOVE_ATR_MULTIPLE = 2.5

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
    """価格データを取得し、テクニカル指標と学習用ラベルを作成する。"""
    source = "1h" if interval == "4h" else interval

    try:
        d = yf.download(
            ticker,
            period=period,
            interval=source,
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=25,
        )
    except Exception:
        return None

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

    d = (
        d[required]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_index()
    )

    if interval == "4h":
        d = d.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }).dropna()

    # 最後の足は形成中の可能性があるため除外
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
        [h - l, (h - prev).abs(), (l - prev).abs()],
        axis=1,
    ).max(axis=1)
    d["ATR"] = tr.rolling(14, min_periods=7).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = (100 - 100 / (1 + rs)).fillna(50)
    d["RSI_Diff"] = d["RSI"].diff()

    macd = (
        c.ewm(span=12, adjust=False).mean()
        - c.ewm(span=26, adjust=False).mean()
    )
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    d["MACD_Hist_Ratio"] = hist / c.replace(0, np.nan)

    std = c.rolling(20, min_periods=10).std()
    upper = d["SMA_20"] + 2 * std
    lower = d["SMA_20"] - 2 * std
    d["BB_PctB"] = (
        (c - lower) / (upper - lower).replace(0, np.nan)
    )

    d["Return_1"] = c.pct_change()
    d["Return_5"] = c.pct_change(5)
    d["Dev_SMA20"] = (c - d["SMA_20"]) / c.replace(0, np.nan)
    d["Dev_EMA200"] = (c - d["EMA_200"]) / c.replace(0, np.nan)
    d["Dev_EMA20_200"] = (
        (d["EMA_20"] - d["EMA_200"]) / c.replace(0, np.nan)
    )
    d["ATR_Ratio"] = d["ATR"] / c.replace(0, np.nan)
    d["Vol_Ratio"] = (
        d["ATR"]
        / d["ATR"].rolling(50, min_periods=10).mean().replace(0, np.nan)
    )

    up, down = h.diff(), -l.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    safe_atr = d["ATR"].replace(0, np.nan)
    pdi = 100 * plus.rolling(14, min_periods=7).mean() / safe_atr
    mdi = 100 * minus.rolling(14, min_periods=7).mean() / safe_atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["ADX"] = dx.rolling(14, min_periods=7).mean().fillna(20)

    rng = (h - l).replace(0, np.nan)
    body_high = pd.concat([o, c], axis=1).max(axis=1)
    body_low = pd.concat([o, c], axis=1).min(axis=1)
    d["Upper_Wick_Ratio"] = (h - body_high) / rng
    d["Lower_Wick_Ratio"] = (body_low - l) / rng

    lo14 = l.rolling(14, min_periods=7).min()
    hi14 = h.rolling(14, min_periods=7).max()
    d["Stoch_K"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)

    tp = (h + l + c) / 3
    tps = tp.rolling(20, min_periods=10).mean()
    mad = tp.rolling(20, min_periods=10).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )
    d["CCI"] = (tp - tps) / (0.015 * mad.replace(0, np.nan))
    d["ROC"] = c.pct_change(10) * 100

    pip = 0.01 if "JPY" in ticker else 0.0001
    spread = SPREAD.get(ticker, 0.5) * pip
    barrier = d["ATR"].clip(lower=spread * 2).to_numpy()
    cv, hv, lv = c.to_numpy(), h.to_numpy(), l.to_numpy()
    y = np.full(len(d), np.nan)

    # 8本先までの値動きでラベル作成
    # 上下両方に同じ足で触れた場合は順序不明なので中立扱い
    for i in range(max(0, len(d) - LOOKAHEAD_BARS)):
        upper_barrier = cv[i] + barrier[i] + spread
        lower_barrier = cv[i] - barrier[i] - spread
        label = 0

        for j in range(i + 1, min(i + LOOKAHEAD_BARS + 1, len(d))):
            hit_up = hv[j] >= upper_barrier
            hit_down = lv[j] <= lower_barrier

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
    return d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURES + ["ATR"]
    )


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
        class_weight=None,
        random_state=seed,
        n_jobs=-1,
    )


def empty_validation(reason):
    return {
        "passed": False,
        "mean_balanced_accuracy_pct": 0.0,
        "mean_baseline_balanced_accuracy_pct": 0.0,
        "folds_passed": 0,
        "fold_count": 0,
        "class_recall_pct": {
            "sell": 0.0,
            "neutral": 0.0,
            "buy": 0.0,
        },
        "fold_results": [],
        "reason": reason,
    }


def walk_forward_validation(X, y):
    """時系列順の検証。検証区間の直前8サンプルを学習から除外する。"""
    n = len(X)

    if n < MIN_TRAIN_ROWS + 3 * 30:
        return empty_validation("複数期間の検証に必要なデータが不足")

    initial_train_end = int(n * 0.55)
    remaining = n - initial_train_end
    block = remaining // N_WALK_FORWARD_FOLDS
    fold_results = []

    for fold_idx in range(N_WALK_FORWARD_FOLDS):
        val_start = initial_train_end + fold_idx * block
        val_end = (
            n if fold_idx == N_WALK_FORWARD_FOLDS - 1
            else initial_train_end + (fold_idx + 1) * block
        )
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
        result = empty_validation("有効な検証区間が2つ未満")
        result["fold_results"] = fold_results
        result["fold_count"] = len(valid_folds)
        return result

    mean_bal = float(np.mean([
        f["balanced_accuracy_pct"] for f in valid_folds
    ]))
    mean_base = float(np.mean([
        f["baseline_balanced_accuracy_pct"] for f in valid_folds
    ]))
    folds_passed = sum(bool(f["passed"]) for f in valid_folds)

    # 最後の有効検証区間で、売り・中立・買いの再現率を個別に計算
    last_valid = next(
        f for f in reversed(fold_results) if f.get("status") == "ok"
    )
    last_idx = int(last_valid["fold"]) - 1
    val_start = initial_train_end + last_idx * block
    val_end = (
        n if last_idx == N_WALK_FORWARD_FOLDS - 1
        else initial_train_end + (last_idx + 1) * block
    )
    train_end = max(0, val_start - LOOKAHEAD_BARS)

    recall_model = make_model(seed=42 + last_idx)
    recall_y_train = y.iloc[:train_end]
    recall_weights = compute_sample_weight(
        class_weight="balanced",
        y=recall_y_train,
    )
    recall_model.fit(
        X.iloc[:train_end],
        recall_y_train,
        sample_weight=recall_weights,
    )
    recall_pred = recall_model.predict(X.iloc[val_start:val_end])
    recalls = recall_score(
        y.iloc[val_start:val_end],
        recall_pred,
        labels=[-1, 0, 1],
        average=None,
        zero_division=0,
    ) * 100

    class_recall = {
        "sell": round(float(recalls[0]), 1),
        "neutral": round(float(recalls[1]), 1),
        "buy": round(float(recalls[2]), 1),
    }

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
        "class_recall_pct": class_recall,
        "fold_results": fold_results,
        "reason": (
            "複数期間の検証条件を満たした"
            if passed else "複数期間の検証品質条件を満たさない"
        ),
    }


def fast_move_detected(d):
    """直近3本の終値変化がATRの2.5倍以上なら急変動として扱う。"""
    if d is None or len(d) < 4:
        return False, 0.0

    atr = float(d["ATR"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return False, 0.0

    recent = d.iloc[-3:]
    move = abs(float(recent["Close"].iloc[-1]) - float(recent["Close"].iloc[0]))
    ratio = move / atr

    return bool(ratio >= FAST_MOVE_ATR_MULTIPLE), round(float(ratio), 2)


def analyze(d):
    fallback = {
        "dir": "様子見",
        "buy": 0,
        "sell": 0,
        "quality": False,
        "bal": 0.0,
        "baseline_bal": 0.0,
        "samples": 0,
        "validation": empty_validation("データ不足"),
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

    model = make_model(seed=43)
    weights = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(X, y, sample_weight=weights)

    latest = xa.dropna()
    if latest.empty:
        return fallback

    probabilities = {
        int(cls): float(prob)
        for cls, prob in zip(
            model.classes_,
            model.predict_proba(latest.iloc[[-1]])[0],
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
                metrics[ticker] = {
                    "pair": label,
                    "status": "insufficient_data",
                    "wait_reason": "必要な時間足データを取得できない",
                }
                continue

            result = analyze(cur)
            fast_move, fast_move_atr_ratio = fast_move_detected(cur)
            direction = result["dir"]

            trends = [trend(one), trend(four), trend(daily)]
            matching_direction = {"買い": "上昇", "売り": "下降"}.get(direction)
            matching = (
                sum(item == matching_direction for item in trends)
                if matching_direction else 0
            )
            opposing = (
                (direction == "買い" and "下降" in trends)
                or (direction == "売り" and "上昇" in trends)
            )

            stamp = pd.Timestamp(cur.index[-1])
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            else:
                stamp = stamp.tz_convert("UTC")

            age = (
                pd.Timestamp.now(tz="UTC") - stamp
            ).total_seconds() / 60

            pip = 0.01 if "JPY" in ticker else 0.0001
            enough = (
                float(cur["ATR"].iloc[-1]) / pip
                >= SPREAD.get(ticker, 0.5) * 3
            )

            # 判定の優先順位：急変動、古いデータ、検証品質、コスト等
            if fast_move:
                decision = "急変動・判定保留"
                reason = (
                    f"直近3本の値動きがATRの"
                    f"{fast_move_atr_ratio}倍"
                )
            elif age > (210 if interval == "1h" else 600):
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
                decision = "候補あり"
                reason = f"AI方向と複数時間足が一致（{matching}/3）"

            close = float(cur["Close"].iloc[-1])
            barstamp = stamp.tz_convert(JST).strftime("%Y-%m-%d %H:%M")

            report.append(
                f"{label}: {decision} / {direction} / "
                f"買い{result['buy']}%・売り{result['sell']}% / "
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
                "class_recall_pct": result["validation"].get(
                    "class_recall_pct", {}
                ),
                "fast_move_detected": fast_move,
                "fast_move_atr_ratio": fast_move_atr_ratio,
                "wait_reason": reason,
                "timeframe_trends": trends,
            }

        except Exception as exc:
            report.append(f"{label}: 取得・分析失敗 ({type(exc).__name__})")
            metrics[ticker] = {
                "pair": label,
                "status": "error",
                "error_type": type(exc).__name__,
                "wait_reason": "データ取得または分析中にエラー",
            }

    now = datetime.now(JST).isoformat()
    previous_history = old.get("history", [])
    if not isinstance(previous_history, list):
        previous_history = []

    state = {
        "schema_version": 4,
        "last_run_jst": now,
        "mode": "multi_period_validation_no_discord_no_auto_orders",
        "interval": interval,
        "summary": report,
        "metrics": metrics,
        "note": (
            "複数期間の時系列検証レポートです。"
            "予測確率・検証指標は将来の利益を保証しません。"
            "急変動検出は直近3本の終値変化とATRを使う簡易判定です。"
            "実取引前にスプレッド等の取引コストを含む検証が必要です。"
        ),
        "history": previous_history[-99:],
    }

    state["history"].append({
        "time_jst": now,
        "metrics": metrics,
    })

    temp_file = STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    os.replace(temp_file, STATE_FILE)

    print("\n".join(report))
    print(
        "Completed; Discord notifications disabled; "
        "multi-period validation report saved."
    )


if __name__ == "__main__":
    main()
