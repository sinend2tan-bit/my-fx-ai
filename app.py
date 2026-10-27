# app.py
# redeploy trigger 2026-10-26 (reflect PR #7: plain HTML importance bars)
# Pro FX Analyzer & Signal Pro — 安全性・検証整合性・モバイル表示改善版
# 価格分析・注文補助専用です。証券口座接続・自動発注は行いません。
# 予測確率・バックテスト結果は将来の利益や実際の勝率を保証しません。

import html
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    import requests
except Exception:
    requests = None

st.set_page_config(page_title="AI FX分析ツール", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# Safety notice: this application is an analysis aid only; never sends orders to a broker.

SETTINGS_FILE = "user_settings.json"
FEEDBACK_FILE = "trade_feedback.json"
ENTRY_TRACK_FILE = "entry_candidate_history.json"
FORECAST_TRACK_FILE = "forecast_outcome_history.json"
PAIR_SCAN_OUTCOME_FILE = "pair_scan_outcomes.json"
TRADE_JOURNAL_FILE = "actual_trade_journal.json"
JST = ZoneInfo("Asia/Tokyo")

DEFAULT_SETTINGS = {
    "account_balance": 500000,
    "quantity_wan": 0.10,
    "auto_refresh": False,
    "auto_pair_scan": False,
    "pair_scan_interval_minutes": 15,
    "pair_scan_timeframe": "1時間足",
    "risk_percent": 1.0,
    "rr_ratio": 1.5,
    "trap_width": 30,
    "ranges": {},
    "trade_memo": "",
    "webhook_url": "",
}
FEATURE_COLUMNS = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200", "Dev_EMA20_200",
    "Vol_Ratio", "RSI", "RSI_Diff", "MACD_Hist_Ratio", "BB_PctB", "ADX",
    "ATR_Ratio", "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Stoch_K", "CCI", "ROC",
]
FEATURE_LABELS = {
    "Return_1": "1本リターン", "Return_5": "5本リターン",
    "Dev_SMA20": "SMA20乖離", "Dev_EMA200": "EMA200乖離",
    "Dev_EMA20_200": "EMA20/200乖離", "Vol_Ratio": "ボラティリティ比",
    "RSI": "RSI", "RSI_Diff": "RSI変化", "MACD_Hist_Ratio": "MACDヒストグラム比",
    "BB_PctB": "ボリンジャー%B", "ADX": "ADX", "ATR_Ratio": "ATR比",
    "Upper_Wick_Ratio": "上ヒゲ比率", "Lower_Wick_Ratio": "下ヒゲ比率",
    "Stoch_K": "ストキャス%K", "CCI": "CCI", "ROC": "ROC",
}
PAIRS = {
    "米ドル/円 (USD/JPY)": "USDJPY=X",
    "英ポンド/円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ/円 (EUR/JPY)": "EURJPY=X",
    "豪ドル/円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ/米ドル (EUR/USD)": "EURUSD=X",
}
TIMEFRAMES = {
    "5分足": {"interval": "5m", "period": "7d"},
    "15分足": {"interval": "15m", "period": "30d"},
    "1時間足": {"interval": "1h", "period": "60d"},
    "4時間足": {"interval": "4h", "period": "90d"},
    "日足": {"interval": "1d", "period": "2y"},
}
DEFAULT_SPREAD_PIPS = {
    "USDJPY=X": 0.2, "EURUSD=X": 0.4, "GBPJPY=X": 1.0,
    "EURJPY=X": 0.5, "AUDJPY=X": 0.6,
}
LOOKAHEAD_BARS = 8
MODEL_VERSION = 6

st.markdown("""
<style>
.stApp { background: #0b1220; color: #e5e7eb; }
[data-testid="stSidebar"] { background: #111827; }
.update-time { padding: 8px 12px; background:#172033; border-radius:8px; margin:8px 0 14px; }
.param-box,.news-card { padding:14px; border:1px solid #334155; border-radius:10px; background:#111827; margin:8px 0; }
.alert-box,.warn-box { padding:12px; border-radius:8px; background:#422006; border:1px solid #f59e0b; margin:10px 0; }
.badge-buy,.badge-sell,.badge-wait { padding:8px 12px; border-radius:8px; font-weight:800; display:inline-block; }
.badge-buy { background:#14532d; color:#86efac; } .badge-sell { background:#7f1d1d; color:#fecaca; }
.badge-wait { background:#374151; color:#e5e7eb; }
.custom-table { width:100%; border-collapse:collapse; } .custom-table th,.custom-table td { padding:8px; border-bottom:1px solid #334155; text-align:left; }

/* iPad / Pixel 向け：文字のコントラストとタップ領域を改善 */
html, body, [class*="css"] { -webkit-text-size-adjust: 100%; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color:#0b1220 !important;
    color:#e5e7eb !important;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background-color:#111827 !important;
}
[data-testid="stSidebar"] * ,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
.stRadio label, .stCheckbox label {
    color:#e5e7eb !important;
}
[data-testid="stCaptionContainer"] { color:#cbd5e1 !important; }
[data-testid="stMetricLabel"] { color:#cbd5e1 !important; }
[data-testid="stMetricValue"] { color:#f8fafc !important; }
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stTextArea"] textarea {
    font-size:16px !important;
    min-height:44px;
}
.stButton > button, [data-testid="stDownloadButton"] button {
    min-height:44px;
    white-space:normal;
    line-height:1.35;
}
[data-testid="stTabs"] [role="tab"] {
    color:#cbd5e1 !important;
    white-space:normal !important;
    height:auto !important;
    min-height:48px;
    padding:8px 10px !important;
    font-size:clamp(13px, 1.5vw, 16px) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:#fca5a5 !important;
}
[data-testid="stPlotlyChart"] { width:100% !important; }
[data-testid="stDataFrame"], .custom-table { max-width:100%; }
.news-card, .param-box, .alert-box, .warn-box {
    overflow-wrap:anywhere;
    word-break:normal;
}
code { overflow-wrap:anywhere; }

/* 画面を簡潔に：余白・見出し・折りたたみ領域を整理 */
.block-container { max-width: 1450px; padding-top: 1.15rem; padding-bottom: 2rem; }
[data-testid="stExpander"] { border: 1px solid #334155; border-radius: 10px; }
[data-testid="stExpander"] summary { min-height: 44px; }
[data-testid="stMetric"] { background: #111827; border: 1px solid #263449; padding: 10px 12px; border-radius: 10px; }
[data-testid="stMetricLabel"] { font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { font-size: clamp(1.15rem, 2.5vw, 1.8rem) !important; }
[data-testid="stRadio"] > div { gap: 0.35rem; }
[data-testid="stRadio"] label { padding: 0.2rem 0.1rem; }
[data-testid="stSelectbox"] { margin-bottom: 0.25rem; }
[data-testid="stMarkdownContainer"] h3 { margin-top: 0.65rem; }
/* 縦向きタブレット・スマートフォンでは横幅を節約 */
@media (max-width: 900px) {
    .block-container {
        padding-left:1rem !important;
        padding-right:1rem !important;
        padding-top:1rem !important;
        max-width:100% !important;
    }
    [data-testid="stTabs"] [role="tab"] {
        min-width:115px;
        flex:1 1 auto;
    }
    [data-testid="stSidebar"] {
        min-width: min(82vw, 320px) !important;
    }
    h1 { font-size:1.65rem !important; }
    h2 { font-size:1.35rem !important; }
    h3 { font-size:1.15rem !important; }
    [data-testid="stMetricValue"] { font-size:clamp(1.25rem, 4vw, 2rem) !important; }
}
@media (max-width: 600px) {
    .block-container {
        padding-left:.65rem !important;
        padding-right:.65rem !important;
    }
    [data-testid="stTabs"] [role="tab"] {
        min-width:105px;
        padding:7px 8px !important;
        font-size:12px !important;
    }
    [data-testid="stPlotlyChart"] {
        overflow-x:auto;
    }
    .custom-table th, .custom-table td {
        padding:6px 5px;
        font-size:13px;
        overflow-wrap:anywhere;
    }
}
</style>
""", unsafe_allow_html=True)


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
    return str(obj)


def load_settings():
    result = DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for key in DEFAULT_SETTINGS:
                if key in loaded:
                    result[key] = loaded[key]
    except (OSError, ValueError, TypeError):
        pass
    # Validate persisted values before they are passed to Streamlit widgets.
    try:
        result["account_balance"] = int(np.clip(float(result.get("account_balance", 500000)), 10000, 100000000000))
    except (TypeError, ValueError, OverflowError):
        result["account_balance"] = DEFAULT_SETTINGS["account_balance"]
    try:
        result["quantity_wan"] = float(np.clip(float(result.get("quantity_wan", 0.10)), 0.01, 1000.0))
    except (TypeError, ValueError, OverflowError):
        result["quantity_wan"] = DEFAULT_SETTINGS["quantity_wan"]
    try:
        result["risk_percent"] = float(np.clip(float(result.get("risk_percent", 1.0)), 0.1, 5.0))
    except (TypeError, ValueError, OverflowError):
        result["risk_percent"] = DEFAULT_SETTINGS["risk_percent"]
    try:
        result["rr_ratio"] = float(np.clip(float(result.get("rr_ratio", 1.5)), 0.5, 5.0))
    except (TypeError, ValueError, OverflowError):
        result["rr_ratio"] = DEFAULT_SETTINGS["rr_ratio"]
    try:
        result["trap_width"] = int(np.clip(float(result.get("trap_width", 30)), 10, 500))
    except (TypeError, ValueError, OverflowError):
        result["trap_width"] = DEFAULT_SETTINGS["trap_width"]
    result["auto_refresh"] = bool(result.get("auto_refresh", False))
    result["auto_pair_scan"] = bool(result.get("auto_pair_scan", False))
    try:
        result["pair_scan_interval_minutes"] = int(result.get("pair_scan_interval_minutes", 15))
    except (TypeError, ValueError):
        result["pair_scan_interval_minutes"] = 15
    if result["pair_scan_interval_minutes"] not in (15, 30, 60):
        result["pair_scan_interval_minutes"] = 15
    if result.get("pair_scan_timeframe") not in ("1時間足", "4時間足"):
        result["pair_scan_timeframe"] = "1時間足"
    result["trade_memo"] = str(result.get("trade_memo", ""))[:20000]
    # Never trust a webhook URL restored from a settings backup.
    result["webhook_url"] = str(result.get("webhook_url", "")) if str(result.get("webhook_url", "")).lower().startswith("https://") else ""
    if not isinstance(result.get("ranges"), dict):
        result["ranges"] = {}
    else:
        cleaned_ranges = {}
        for pair, bounds in result["ranges"].items():
            if pair not in PAIRS.values() or not isinstance(bounds, dict):
                continue
            try:
                lo, hi = float(bounds.get("lower")), float(bounds.get("upper"))
                if np.isfinite([lo, hi]).all() and 0 < lo < hi:
                    cleaned_ranges[pair] = {"lower": lo, "upper": hi}
            except (TypeError, ValueError, OverflowError):
                continue
        result["ranges"] = cleaned_ranges
    return result


def save_settings():
    payload = {k: st.session_state.get(k, v) for k, v in DEFAULT_SETTINGS.items()}
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    except OSError as exc:
        st.warning(f"設定を保存できませんでした: {exc}")


def reset_settings_callback():
    """Reset persisted settings before widgets are instantiated on the next run."""
    for key, value in DEFAULT_SETTINGS.items():
        st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value
    try:
        if os.path.exists(SETTINGS_FILE):
            os.remove(SETTINGS_FILE)
    except OSError:
        pass
    st.session_state["settings_loaded"] = True


def load_feedback_data():
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, list) else []
    except (OSError, ValueError, TypeError):
        return []


def save_feedback_data(items):
    try:
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2, default=_json_default)
    except OSError as exc:
        st.error(f"フィードバックを保存できませんでした: {exc}")


def load_trade_journal():
    """Load manually confirmed realized trade results; never treated as model labels."""
    try:
        with open(TRADE_JOURNAL_FILE, "r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, list) else []
    except (OSError, ValueError, TypeError):
        return []


def save_trade_journal(items):
    try:
        with open(TRADE_JOURNAL_FILE, "w", encoding="utf-8") as f:
            json.dump(items[-2000:], f, ensure_ascii=False, indent=2, default=_json_default)
    except OSError as exc:
        st.error(f"実取引記録を保存できませんでした: {exc}")


def load_entry_tracking():
    """Load persisted candidate-price observations; these are not broker executions."""
    try:
        with open(ENTRY_TRACK_FILE, "r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, list) else []
    except (OSError, ValueError, TypeError):
        return []


def save_entry_tracking(items):
    try:
        # Keep the file bounded so it remains practical on mobile/local hosting.
        with open(ENTRY_TRACK_FILE, "w", encoding="utf-8") as f:
            json.dump(items[-1000:], f, ensure_ascii=False, indent=2, default=_json_default)
    except OSError as exc:
        st.error(f"候補価格の検証履歴を保存できませんでした: {exc}")


def load_forecast_history():
    """Load automatic signal-evaluation records; these are not broker trade results."""
    try:
        with open(FORECAST_TRACK_FILE, "r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, list) else []
    except (OSError, ValueError, TypeError):
        return []


def save_forecast_history(items):
    try:
        with open(FORECAST_TRACK_FILE, "w", encoding="utf-8") as f:
            json.dump(items[-3000:], f, ensure_ascii=False, indent=2, default=_json_default)
    except OSError as exc:
        st.warning(f"自動予測検証履歴を保存できませんでした: {exc}")


def build_data_backup_bundle():
    """Export user settings and local logs; secrets such as webhook URLs are excluded."""
    settings = load_settings().copy()
    settings["webhook_url"] = ""
    return {
        "format": "ai_fx_analyzer_backup",
        "version": 1,
        "created_at": datetime.now(JST).isoformat(),
        "notice": "Webhook URL and credentials are intentionally excluded.",
        "settings": settings,
        "forecast_history": load_forecast_history(),
        "entry_tracking": load_entry_tracking(),
        "trade_journal": load_trade_journal(),
        "feedback_history": load_feedback_data(),
    }


def restore_data_backup_bundle(bundle):
    """Validate a backup bundle before restoring known local JSON data files."""
    if not isinstance(bundle, dict) or bundle.get("format") != "ai_fx_analyzer_backup":
        raise ValueError("このファイルはAI FX分析ツールのバックアップではありません。")
    settings = bundle.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("設定データの形式が正しくありません。")
    safe_settings = {k: settings[k] for k in DEFAULT_SETTINGS if k in settings}
    safe_settings["webhook_url"] = ""  # Never restore secrets from a file.
    lists = {
        FORECAST_TRACK_FILE: bundle.get("forecast_history", []),
        ENTRY_TRACK_FILE: bundle.get("entry_tracking", []),
        TRADE_JOURNAL_FILE: bundle.get("trade_journal", []),
        FEEDBACK_FILE: bundle.get("feedback_history", []),
    }
    if any(not isinstance(v, list) for v in lists.values()):
        raise ValueError("履歴データの形式が正しくありません。復元を中止しました。")
    # Validate all content before writing any files.
    json.dumps({"settings": safe_settings, "lists": lists}, ensure_ascii=False, default=_json_default)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(safe_settings, f, ensure_ascii=False, indent=2, default=_json_default)
    for filename, items in lists.items():
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(items[-3000:], f, ensure_ascii=False, indent=2, default=_json_default)


def update_forecast_history(items, df, ticker, interval, horizon=LOOKAHEAD_BARS):
    """Resolve saved BUY/SELL/WAIT signals after N completed bars using close-to-close direction.

    This measures directional follow-through only, not profitability or executable fills.
    """
    if df is None or df.empty:
        return items
    changed = False
    idx = pd.DatetimeIndex(df.index)
    idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    close_values = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    for item in items:
        if item.get("ticker") != ticker or item.get("interval") != interval or item.get("status") != "PENDING":
            continue
        try:
            saved_ts = pd.Timestamp(item["bar_timestamp"])
            saved_ts = saved_ts.tz_localize("UTC") if saved_ts.tzinfo is None else saved_ts.tz_convert("UTC")
            matches = np.flatnonzero(idx_utc == saved_ts)
            if len(matches) == 0:
                # If the saved bar has rolled out of the provider's lookback window,
                # don't leave the forecast pending forever or score it against a wrong bar.
                latest_ts = idx_utc[-1]
                expected_horizon = max(1, int(item.get("horizon_bars", horizon)))
                interval_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(str(interval), 60)
                elapsed_minutes = max(0.0, (latest_ts - saved_ts).total_seconds() / 60.0)
                if elapsed_minutes > interval_minutes * (expected_horizon + 3):
                    item.update({
                        "status": "EXPIRED_NO_DATA",
                        "evaluated_at": datetime.now(JST).isoformat(),
                        "evaluation_note": "保存元の確定足が取得データ範囲外になったため採点対象外",
                    })
                    changed = True
                continue
            base_pos = int(matches[-1])
            target_pos = base_pos + int(item.get("horizon_bars", horizon))
            if target_pos >= len(df):
                continue
            base_close = float(item["reference_close"])
            future_close = float(close_values[target_pos])
            signal = item.get("signal")
            if signal == "BUY":
                result = "CORRECT" if future_close > base_close else "INCORRECT" if future_close < base_close else "FLAT"
            elif signal == "SELL":
                result = "CORRECT" if future_close < base_close else "INCORRECT" if future_close > base_close else "FLAT"
            else:
                # WAIT is tracked separately; no directional hit rate is assigned.
                result = "WAIT_CHECKED"
            item.update({"status": result, "evaluated_close": future_close,
                         "evaluated_bar_timestamp": _normalise_timestamp(idx[target_pos]),
                         "evaluated_at": datetime.now(JST).isoformat(),
                         "move_pct": (future_close / base_close - 1.0) * 100.0 if base_close else None})
            # Separately score the model's predicted class against the exact first-touch
            # Target used for training. This is distinct from close-to-close direction.
            try:
                target_value = pd.to_numeric(pd.Series([df["Target"].iloc[base_pos]]), errors="coerce").iloc[0]
                predicted_class = item.get("model_class_prediction")
                if pd.notna(target_value) and predicted_class in (-1, 0, 1):
                    item["training_target"] = int(target_value)
                    item["target_label_match"] = bool(int(predicted_class) == int(target_value))
                    item["target_label_scored_at"] = datetime.now(JST).isoformat()
            except (KeyError, TypeError, ValueError, IndexError, OverflowError):
                pass
            changed = True
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            continue
    if changed:
        save_forecast_history(items)
    return items


def record_forecast_and_update(df, ticker, interval, signal, probs, horizon=LOOKAHEAD_BARS):
    """Automatically log each completed bar once, then score matured forecasts."""
    if df is None or df.empty:
        return []
    items = load_forecast_history()
    bar_ts = _normalise_timestamp(df.index[-1])
    exists = any(x.get("ticker") == ticker and x.get("interval") == interval and x.get("bar_timestamp") == bar_ts for x in items)
    if not exists:
        raw_signal = "BUY" if str(signal).startswith("BUY") else "SELL" if str(signal).startswith("SELL") else "WAIT"
        _class_probabilities = {-1: int(probs.get("sell", 0)), 0: int(probs.get("wait", 100)), 1: int(probs.get("buy", 0))}
        _model_class_prediction = max((-1, 0, 1), key=lambda _c: _class_probabilities[_c])
        items.append({"ticker": ticker, "interval": interval, "bar_timestamp": bar_ts,
                      "reference_close": float(df["Close"].iloc[-1]), "signal": raw_signal,
                      "model_class_prediction": int(_model_class_prediction),
                      "buy_pct": int(probs.get("buy", 0)), "sell_pct": int(probs.get("sell", 0)),
                      "wait_pct": int(probs.get("wait", 100)), "horizon_bars": int(horizon),
                      "status": "PENDING", "created_at": datetime.now(JST).isoformat(),
                      "skip_reason": ("AI判定が様子見、または買い・売りの条件が十分にそろっていない" if raw_signal == "WAIT" else "方向シグナルあり。時間足・データ鮮度・リスク条件は別途確認"),
                      "data_source": "Yahoo Finance OHLC", "is_broker_trade": False})
        items = items[-3000:]
    items = update_forecast_history(items, df, ticker, interval, horizon)
    save_forecast_history(items)
    return items


def update_entry_tracking_outcomes(items, df, ticker, interval, horizon=8):
    """Evaluate whether later OHLC bars touched saved candidate prices.

    OHLC touch is only a market-data estimate, not proof of a broker fill.
    If both entry and other levels fall within one bar, intrabar ordering is unknown.
    """
    if df is None or df.empty:
        return items
    changed = False
    idx = pd.DatetimeIndex(df.index)
    for item in items:
        if item.get("ticker") != ticker or item.get("interval") != interval:
            continue
        if item.get("status") in ("HIT", "NOT_REACHED", "EXPIRED"):
            continue
        try:
            saved_ts = pd.Timestamp(item.get("bar_timestamp"))
            if saved_ts.tzinfo is None:
                saved_ts = saved_ts.tz_localize("UTC")
            else:
                saved_ts = saved_ts.tz_convert("UTC")
            idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            future_positions = np.flatnonzero(idx_utc > saved_ts)
            if len(future_positions) == 0:
                continue
            positions = future_positions[:horizon]
            price = float(item["candidate_price"])
            hit_position = None
            for pos in positions:
                row = df.iloc[int(pos)]
                if float(row["Low"]) <= price <= float(row["High"]):
                    hit_position = int(pos)
                    break
            if hit_position is not None:
                item["status"] = "HIT"
                item["bars_to_touch"] = int(np.flatnonzero(positions == hit_position)[0]) + 1
                item["touch_bar_timestamp"] = _normalise_timestamp(idx[hit_position])
                item["evaluated_at"] = datetime.now(JST).isoformat()
                changed = True
            elif len(future_positions) >= horizon:
                item["status"] = "NOT_REACHED"
                item["evaluated_at"] = datetime.now(JST).isoformat()
                changed = True
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    if changed:
        save_entry_tracking(items)
    return items


def clean_series(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_to_tokyo_tz(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce")
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert(JST)
    return out


def flatten_yf_df(df):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance has changed the order of MultiIndex levels across versions.
        out.columns = [str(c[0] if str(c[0]).lower() in {"open","high","low","close","adj close","volume"} else c[-1]) for c in out.columns]
    out.columns = [str(c).replace(" ", "_").title() for c in out.columns]
    return out


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


@st.cache_data(ttl=300, show_spinner=False)
def get_usdjpy_rate():
    try:
        d = yf.download("USDJPY=X", period="5d", interval="1d", progress=False, threads=False)
        d = flatten_yf_df(d)
        if not d.empty and "Close" in d:
            vals = clean_series(d["Close"]).dropna()
            if not vals.empty and float(vals.iloc[-1]) > 0:
                return float(vals.iloc[-1])
    except Exception:
        pass
    return None


def _impact_from_text(title, summary=""):
    """見出しの語句だけで暫定分類する。実際の影響度や方向を断定しない。"""
    text = f"{title} {summary}".lower()
    high_terms = [
        "fomc", "interest rate", "policy rate", "inflation", "cpi", "core pce", "pce price",
        "payroll", "nonfarm", "jobs report", "unemployment rate", "central bank", "boj",
        "bank of japan", "federal reserve", "fed decision", "powell", "植田", "日銀", "金融政策",
        "政策金利", "米雇用統計", "雇用統計", "消費者物価", "物価指数", "介入", "intervention",
        "gdp", "利上げ", "利下げ", "緊急会合", "emergency meeting"
    ]
    if any(x in text for x in high_terms):
        return "影響度：高", "重要イベント関連の語句を検出。発表内容・時刻・市場反応を確認してください", "方向は見出しだけでは判定しない"
    medium_terms = ["yen", "dollar", "treasury", "bond", "economy", "employment", "trade", "pmi", "retail sales", "小売", "景気", "為替"]
    if any(x in text for x in medium_terms):
        return "影響度：中", "為替・景気関連の語句を検出。本文と市場反応を確認してください", "方向は本文と市場反応を確認"
    return "影響度：低", "見出し内の関連語句は限定的です（影響がないという意味ではありません）", "方向判定なし"


# Official central-bank feeds, keyed by currency code. These carry official
# announcements only, not a full economic calendar or real-time newswire.
CENTRAL_BANK_FEEDS = [
    ("JPY", "日本銀行（公式）", "https://www.boj.or.jp/rss/whatsnew.xml"),
    ("USD", "米連邦準備制度（公式）", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    ("EUR", "欧州中央銀行（公式）", "https://www.ecb.europa.eu/rss/press.html"),
    ("GBP", "英イングランド銀行（公式）", "https://www.bankofengland.co.uk/rss/news"),
    ("AUD", "豪準備銀行（公式）", "https://www.rba.gov.au/rss/rss-cb-media-releases.xml"),
]

# Macro / monetary-policy terms used to triage official-feed headlines.
_OFFICIAL_FEED_TERMS = (
    "金融政策", "政策金利", "物価", "経済", "為替", "日銀", "総裁", "金融",
    "monetary", "fomc", "interest rate", "inflation", "economic",
    "federal funds", "policy rate", "bank rate", "cash rate",
)

_RSS10_NS = "http://purl.org/rss/1.0/"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_DC_NS = "http://purl.org/dc/elements/1.1/"


def _central_bank_feeds_for_ticker(ticker):
    """Return the official feeds relevant to the pair's two currencies."""
    code = str(ticker or "").upper()
    return [(publisher, url) for currency, publisher, url in CENTRAL_BANK_FEEDS if currency in code]


def _feed_node_text(node, names):
    """First non-empty text among candidate element names (namespace aware)."""
    for name in names:
        el = node.find(name)
        if el is not None and el.text and el.text.strip():
            return el.text.strip()
    return ""


def _fetch_official_feed_items(publisher, url, errors):
    """Fetch one official central-bank feed and return triaged headline items.

    Handles RSS 2.0, Atom and RSS 1.0 (RDF) shapes. Headlines are kept only
    when they contain a macro / monetary-policy related term, so unrelated
    notices from large official feeds do not crowd out the list.
    """
    items = []
    try:
        response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0 FX-analysis-tool"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        errors.append(f"{publisher}: {type(exc).__name__}")
        return items
    nodes = root.findall(".//item")
    if not nodes:
        nodes = root.findall(f".//{{{_ATOM_NS}}}entry")
    if not nodes:
        nodes = root.findall(f".//{{{_RSS10_NS}}}item")
    for node in nodes:
        title = _feed_node_text(node, ["title", f"{{{_ATOM_NS}}}title", f"{{{_RSS10_NS}}}title"])
        if not title:
            continue
        # Keep macro / monetary-policy related headlines rather than unrelated notices.
        lower_title = title.lower()
        if not any(term in lower_title for term in _OFFICIAL_FEED_TERMS):
            continue
        link = _feed_node_text(node, ["link", f"{{{_RSS10_NS}}}link"])
        if not link:
            for name in ("link", f"{{{_ATOM_NS}}}link"):
                el = node.find(name)
                if el is not None and el.attrib.get("href"):
                    link = el.attrib["href"]
                    break
        if not link:
            link = "#"
        raw_date = _feed_node_text(node, ["pubDate", f"{{{_ATOM_NS}}}updated", f"{{{_ATOM_NS}}}published", f"{{{_DC_NS}}}date"])
        published_ts = None
        try:
            parsed_date = pd.to_datetime(raw_date, utc=True, errors="coerce") if raw_date else pd.NaT
            if not pd.isna(parsed_date):
                published_ts = float(parsed_date.timestamp())
                time_str = parsed_date.tz_convert(JST).strftime("%Y/%m/%d %H:%M JST")
            else:
                time_str = raw_date[:25] if raw_date else "時刻不明"
        except Exception:
            time_str = raw_date[:25] if raw_date else "時刻不明"
        impact, reason, direction = _impact_from_text(title)
        items.append({"title": title, "publisher": publisher, "time": time_str,
                      "published_ts": published_ts, "link": link, "impact": impact, "reason": reason,
                      "direction": direction, "source": publisher})
    return items


# Pair-specific Japanese search terms for Google News RSS. Yahoo Finance
# headlines are effectively unusable for these FX pairs (yfinance returns
# zero items and Yahoo's own headline RSS carries months-old leftovers),
# so the headline layer uses Google News search RSS instead: free, no API
# key, and fresh (the "when:3d" operator limits results to recent days).
GOOGLE_NEWS_QUERIES = {
    "USDJPY=X": "ドル円",
    "GBPJPY=X": "ポンド円",
    "EURJPY=X": "ユーロ円",
    "AUDJPY=X": "豪ドル円",
    "EURUSD=X": "ユーロドル",
}


def _google_news_query_for_ticker(ticker):
    """Japanese search term for the pair, with a generic FX fallback."""
    code = str(ticker or "").upper()
    if code in GOOGLE_NEWS_QUERIES:
        return GOOGLE_NEWS_QUERIES[code]
    base = code.replace("=X", "")
    return base or "為替"


def _fetch_google_news_items(ticker, errors):
    """Fetch recent headlines for the pair from Google News search RSS.

    Returns items in the same shape as the official-feed items, with
    source "Googleニュース" and the outlet name kept as publisher. Up to
    30 raw items are read; the merge step caps the combined list at 12.
    """
    from urllib.parse import quote_plus

    items = []
    query = _google_news_query_for_ticker(ticker)
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(f"{query} when:3d")
        + "&hl=ja&gl=JP&ceid=JP:ja"
    )
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0 FX-analysis-tool"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        errors.append(f"Googleニュース: {type(exc).__name__}")
        return items
    for node in root.findall(".//item")[:30]:
        title = _feed_node_text(node, ["title"])
        if not title:
            continue
        publisher = _feed_node_text(node, ["source"]) or "Googleニュース"
        link = _feed_node_text(node, ["link"]) or "#"
        raw_date = _feed_node_text(node, ["pubDate"])
        published_ts = None
        try:
            parsed_date = pd.to_datetime(raw_date, utc=True, errors="coerce") if raw_date else pd.NaT
            if not pd.isna(parsed_date):
                published_ts = float(parsed_date.timestamp())
                time_str = parsed_date.tz_convert(JST).strftime("%Y/%m/%d %H:%M JST")
            else:
                time_str = raw_date[:25] if raw_date else "時刻不明"
        except Exception:
            time_str = raw_date[:25] if raw_date else "時刻不明"
        impact, reason, direction = _impact_from_text(title)
        items.append({"title": title, "publisher": publisher, "time": time_str,
                      "published_ts": published_ts, "link": link, "impact": impact, "reason": reason,
                      "direction": direction, "source": "Googleニュース"})
    return items


def _news_title_key(item):
    """Normalized title used for duplicate detection in the merge.

    Google News titles end with a " - <outlet>" suffix; the same story from
    different outlets (or with the suffix present/absent) should still be
    treated as one headline, so the suffix matching the item's publisher
    is stripped before comparing.
    """
    title = str(item.get("title", "")).strip()
    publisher = str(item.get("publisher", "")).strip()
    if publisher:
        suffix = f" - {publisher}"
        if title.lower().endswith(suffix.lower()):
            title = title[: -len(suffix)].strip()
    return title.lower()


def _merge_news_items(*groups):
    """Merge headline groups, dropping duplicates and sorting newest first.

    Duplicates are detected by title or by link. Items without a parsed
    timestamp keep their relative order at the end of the list.
    """
    merged = []
    seen_titles = set()
    seen_links = set()
    for group in groups:
        for item in group:
            title_key = _news_title_key(item)
            link_key = str(item.get("link", "")).strip()
            if title_key and title_key in seen_titles:
                continue
            if link_key and link_key != "#" and link_key in seen_links:
                continue
            if title_key:
                seen_titles.add(title_key)
            if link_key and link_key != "#":
                seen_links.add(link_key)
            merged.append(item)
    merged.sort(key=lambda x: (x.get("published_ts") is None, -(x.get("published_ts") or 0.0)))
    return merged[:12]


# News is cached for 15 minutes to reduce repeated requests during ordinary UI reruns.
@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_and_impact(ticker):
    """Fetch Google News headlines plus the pair's official central-bank feeds.

    Yahoo Finance headlines are not used: for these FX pairs yfinance
    returns zero items and Yahoo's headline RSS serves stale leftovers.
    Google News search RSS (pair-specific Japanese query, last 3 days)
    provides the headline layer instead. The official feeds for the
    pair's two currencies are always merged in, duplicates are removed
    and the combined list is sorted newest first, capped at 12.
    This is headline keyword triage only, not semantic AI or a directional
    forecast. Returns source/status metadata so an empty feed is visible
    to the user.
    """
    errors = []
    headline_items = []
    official_items = []
    if requests is not None:
        headline_items = _fetch_google_news_items(ticker, errors)
        # Always merge the official central-bank feeds for this pair's
        # currencies. These are official announcements, not a full
        # economic calendar or real-time newswire, so they complement
        # rather than replace the headline layer.
        for publisher, url in _central_bank_feeds_for_ticker(ticker):
            official_items.extend(_fetch_official_feed_items(publisher, url, errors))

    items = _merge_news_items(headline_items, official_items)
    if items:
        used_sources = []
        for item in items:
            src = item.get("source", "")
            if src and src not in used_sources:
                used_sources.append(src)
        status = f"取得成功（{'、'.join(used_sources)}／見出しのキーワード分類のみ）"
    elif requests is None:
        status = "取得失敗：requests が利用できず、ニュースRSSへの接続もできません"
    else:
        status = "取得できるニュースがありません。ネットワーク、配信元、取得制限を確認してください"
    return {"items": items, "status": status, "errors": errors}


def summarize_news_risk(items):
    """取得できた見出しを簡易集計。経済指標カレンダーや将来の発表時刻ではない。"""
    if not items:
        return {
            "level": "確認できず", "kind": "unknown",
            "message": "ニュースを取得できていません。リスクが低いという意味ではないため、取引前に公式の経済指標予定を確認してください。",
            "high_count": 0, "recent_count": 0,
        }
    now_ts = datetime.now(JST).timestamp()
    high_items = [x for x in items if str(x.get("impact", "")).startswith("影響度：高")]
    recent_high = []
    timestamp_unknown = 0
    for item in high_items:
        ts = item.get("published_ts")
        try:
            ts = float(ts) if ts is not None else None
        except (TypeError, ValueError):
            ts = None
        if ts is None:
            timestamp_unknown += 1
        elif 0 <= now_ts - ts <= 24 * 60 * 60:
            recent_high.append(item)
    if recent_high:
        return {
            "level": "重要ニュースあり・要確認", "kind": "high",
            "message": f"過去24時間以内の見出しに重要イベント関連語句が {len(recent_high)} 件あります。発表済みか、結果・市場反応・今後の予定を公式情報で確認してください。",
            "high_count": len(high_items), "recent_count": len(recent_high),
        }
    if high_items:
        suffix = "掲載時刻を確認できない見出しもあります。" if timestamp_unknown else ""
        return {
            "level": "重要語句あり・時刻確認", "kind": "medium",
            "message": f"重要イベント関連語句を含む見出しが {len(high_items)} 件あります。古いニュースの可能性もあるため、掲載時刻と公式発表を確認してください。{suffix}",
            "high_count": len(high_items), "recent_count": 0,
        }
    return {
        "level": "高影響語句は未検出", "kind": "low",
        "message": "取得した見出しから重要イベント関連語句を検出しませんでした。ただしニュースの網羅性は保証されず、経済指標の発表がないことや相場が安全であることを意味しません。",
        "high_count": 0, "recent_count": 0,
    }


# Price data is cached for 60 seconds. This is data freshness, not a UI auto-refresh timer.
@st.cache_data(ttl=60, show_spinner=False)
def load_and_process_data(ticker, period, interval, label=""):
    try:
        actual_interval = "1h" if interval == "4h" else interval
        df = yf.download(ticker, period=period, interval=actual_interval, auto_adjust=False,
                         progress=False, threads=False, timeout=20)
        df = flatten_yf_df(df)
        required = ["Open", "High", "Low", "Close"]
        if df.empty or not set(required).issubset(df.columns):
            return None
        df = df[required + (["Volume"] if "Volume" in df.columns else [])].copy()
        df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=required).sort_index()
        if interval == "4h":
            agg = {"Open":"first","High":"max","Low":"min","Close":"last"}
            if "Volume" in df: agg["Volume"] = "sum"
            df = df.resample("4h").agg(agg).dropna(subset=required)
        # Discard the newest bar because it may still be forming.
        if len(df) > 2:
            df = df.iloc[:-1]
        if len(df) < 80:
            return None
        close, high, low, opn = df["Close"], df["High"], df["Low"], df["Open"]
        df["SMA_20"] = close.rolling(20, min_periods=10).mean()
        df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA_200"] = close.ewm(span=200, adjust=False).mean()
        prev = close.shift(1)
        tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14, min_periods=7).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = (100 - 100/(1+rs)).fillna(50)
        df["RSI_Diff"] = df["RSI"].diff()
        ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        df["MACD_Hist_Ratio"] = df["MACD_Hist"] / close.replace(0, np.nan)
        std = close.rolling(20, min_periods=10).std()
        df["Upper_Band"], df["Lower_Band"] = df["SMA_20"] + 2*std, df["SMA_20"] - 2*std
        df["BB_PctB"] = (close - df["Lower_Band"]) / (df["Upper_Band"] - df["Lower_Band"]).replace(0, np.nan)
        df["Return_1"] = close.pct_change()
        df["Return_5"] = close.pct_change(5)
        df["Dev_SMA20"] = (close - df["SMA_20"]) / close.replace(0, np.nan)
        df["Dev_EMA200"] = (close - df["EMA_200"]) / close.replace(0, np.nan)
        df["Dev_EMA20_200"] = (df["EMA_20"] - df["EMA_200"]) / close.replace(0, np.nan)
        df["ATR_Ratio"] = df["ATR"] / close.replace(0, np.nan)
        df["Vol_Ratio"] = df["ATR"] / df["ATR"].rolling(50, min_periods=10).mean().replace(0, np.nan)
        up_move, down_move = high.diff(), -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        atr_safe = df["ATR"].replace(0, np.nan)
        plus_di = 100 * plus_dm.rolling(14, min_periods=7).mean() / atr_safe
        minus_di = 100 * minus_dm.rolling(14, min_periods=7).mean() / atr_safe
        dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
        df["ADX"] = dx.rolling(14, min_periods=7).mean().fillna(20)
        rng = (high-low).replace(0, np.nan)
        df["Upper_Wick_Ratio"] = (high - pd.concat([opn, close], axis=1).max(axis=1)) / rng
        df["Lower_Wick_Ratio"] = (pd.concat([opn, close], axis=1).min(axis=1) - low) / rng
        low14, high14 = low.rolling(14, min_periods=7).min(), high.rolling(14, min_periods=7).max()
        df["Stoch_K"] = 100 * (close-low14)/(high14-low14).replace(0, np.nan)
        tp = (high+low+close)/3
        tp_sma = tp.rolling(20, min_periods=10).mean()
        mad = tp.rolling(20, min_periods=10).apply(lambda x: np.mean(np.abs(x-np.mean(x))), raw=True)
        df["CCI"] = (tp-tp_sma)/(0.015*mad.replace(0, np.nan))
        df["ROC"] = close.pct_change(10)*100

        # First-touch labels: determine which barrier is touched first in the next N bars.
        # If both barriers are touched within the same OHLC bar, intrabar order is unknowable,
        # so that sample is labeled WAIT (0) rather than guessing.
        pip_unit = 0.01 if "JPY" in ticker else 0.0001
        spread = DEFAULT_SPREAD_PIPS.get(ticker, 0.5) * pip_unit
        barrier = (df["ATR"] * 1.0).clip(lower=spread * 2).to_numpy(dtype=float)
        close_values = close.to_numpy(dtype=float)
        high_values = high.to_numpy(dtype=float)
        low_values = low.to_numpy(dtype=float)
        target = np.full(len(df), np.nan, dtype=float)
        usable = max(0, len(df) - LOOKAHEAD_BARS)
        for i in range(usable):
            distance = barrier[i] + spread
            upper_barrier = close_values[i] + distance
            lower_barrier = close_values[i] - distance
            label = 0
            for j in range(i + 1, min(i + LOOKAHEAD_BARS + 1, len(df))):
                hit_up = high_values[j] >= upper_barrier
                hit_down = low_values[j] <= lower_barrier
                if hit_up and hit_down:
                    label = 0  # same-bar order cannot be inferred from OHLC data
                    break
                if hit_up:
                    label = 1
                    break
                if hit_down:
                    label = -1
                    break
            target[i] = label
        df["Target"] = target
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.dropna(subset=FEATURE_COLUMNS + ["Close", "ATR"])
    except Exception:
        return None


def send_webhook_alert(url, message):
    if not url or requests is None:
        return False
    # Only permit HTTPS endpoints; never include secrets or account information in the message.
    if not str(url).lower().startswith("https://"):
        return False
    try:
        response = requests.post(url, json={"content": message}, timeout=5)
        return 200 <= response.status_code < 300
    except Exception:
        return False


def _model_filename(ticker, interval):
    clean_ticker = str(ticker).replace("=X", "").replace("/", "")
    return f"model_{clean_ticker}_{interval}.joblib"


def _normalise_timestamp(value):
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    except Exception:
        return str(value)


def analyze_signal_with_backtest(df_current, df_htf, ticker, interval, force_retrain=False):
    """Returns the same tuple shape expected by the UI."""
    st.session_state.pop("model_eval_metrics", None)
    fallback = ("WAIT (様子見)", 0.0, 0, 0.0, "データ不足",
                {"buy": 0, "sell": 0, "wait": 100}, pd.Series(dtype=float), pd.Series(dtype=int))
    if df_current is None or len(df_current) < 180:
        return fallback
    try:
        data = df_current.copy().sort_index()
        X_all = data.reindex(columns=FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan)
        labels = pd.to_numeric(data["Target"], errors="coerce")
        labelled_mask = labels.notna() & X_all.notna().all(axis=1)
        labelled = data.loc[labelled_mask].copy()
        X = X_all.loc[labelled_mask]
        y = labels.loc[labelled_mask].astype(int)
        if len(X) < 160 or y.nunique() < 2:
            return fallback

        # Chronological holdout with an embargo at least as long as the label horizon.
        test_n = min(120, max(40, int(len(X) * 0.20)))
        split = len(X) - test_n
        train_end = max(1, split - LOOKAHEAD_BARS)
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        if len(X_train) < 100 or len(X_test) < 20 or y_train.nunique() < 2:
            return fallback

        eval_model = RandomForestClassifier(
            n_estimators=220, max_depth=7, min_samples_leaf=8,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        eval_model.fit(X_train, y_train, sample_weight=weights)
        pred_test = eval_model.predict(X_test)
        accuracy = float(accuracy_score(y_test, pred_test) * 100)
        balanced_accuracy = float(balanced_accuracy_score(y_test, pred_test) * 100)
        # Multiclass Brier score on chronological holdout; lower is better.
        # This is a probability-quality diagnostic, not a trading-profit metric.
        eval_prob = eval_model.predict_proba(X_test)
        eval_classes = [int(c) for c in eval_model.classes_]
        y_true_matrix = np.zeros((len(y_test), len(eval_classes)), dtype=float)
        y_test_values = y_test.to_numpy(dtype=int)
        for _ci, _class_value in enumerate(eval_classes):
            y_true_matrix[:, _ci] = (y_test_values == _class_value).astype(float)
        brier_score = float(np.mean(np.sum((eval_prob - y_true_matrix) ** 2, axis=1)))
        # A simple majority-class baseline helps detect a model that looks accurate
        # only because one label dominates the test set.
        baseline_model = DummyClassifier(strategy="most_frequent")
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_accuracy = float(accuracy_score(y_test, baseline_pred) * 100)
        baseline_balanced_accuracy = float(balanced_accuracy_score(y_test, baseline_pred) * 100)
        model_quality_ok = bool(
            balanced_accuracy >= max(38.0, baseline_balanced_accuracy + 3.0)
        )
        st.session_state["model_eval_metrics"] = {
            "balanced_accuracy": balanced_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "baseline_balanced_accuracy": baseline_balanced_accuracy,
            "brier_score": brier_score,
            "quality_ok": model_quality_ok,
            "test_count": int(len(y_test)),
            "train_count": int(len(X_train)),
            "test_start": str(X_test.index[0]),
            "test_end": str(X_test.index[-1]),
        }
        test_count = int(len(y_test))

        # History contains only truly out-of-sample test predictions; prior bars are NaN.
        history = pd.Series(np.nan, index=data.index, dtype=float)
        history.loc[X_test.index] = pred_test.astype(float)

        model_path = _model_filename(ticker, interval)
        model = None
        if not force_retrain and os.path.exists(model_path):
            try:
                saved = joblib.load(model_path)
                if (isinstance(saved, dict) and saved.get("version") == MODEL_VERSION
                    and saved.get("features") == FEATURE_COLUMNS
                    and isinstance(saved.get("model"), RandomForestClassifier)):
                    age_hours = (time.time() - os.path.getmtime(model_path)) / 3600
                    if age_hours <= 4:
                        model = saved["model"]
            except Exception:
                model = None

        # Retrain with all eligible labelled rows. Feedback reweights only matching training rows.
        if model is None or force_retrain:
            candidate_model = RandomForestClassifier(
                n_estimators=260, max_depth=7, min_samples_leaf=8,
                class_weight="balanced_subsample", random_state=43, n_jobs=-1
            )
            # Trade outcome buttons are review logs, not reliable market labels.
            # Do not promote a new model when out-of-sample balanced accuracy fails
            # to clear a simple baseline by a small margin.
            sw = compute_sample_weight(class_weight="balanced", y=y)
            candidate_model.fit(X, y, sample_weight=sw)
            if model_quality_ok:
                model = candidate_model
                try:
                    joblib.dump({"version": MODEL_VERSION, "features": FEATURE_COLUMNS,
                                 "model": model, "trained_at": datetime.now(JST).isoformat(),
                                 "validation_balanced_accuracy": balanced_accuracy,
                                 "baseline_balanced_accuracy": baseline_balanced_accuracy,
                                 "feedback_applied": 0}, model_path)
                except Exception:
                    pass
            else:
                # Use the validation-only model for this run; do not save/promote
                # the full-data candidate when its holdout result is weak.
                model = eval_model

        current_features = X_all.dropna()
        if current_features.empty:
            return fallback
        latest_X = current_features.iloc[[-1]]
        raw = model.predict_proba(latest_X)[0]
        prob_map = {int(c): float(p) for c, p in zip(model.classes_, raw)}
        buy_p, sell_p, wait_p = prob_map.get(1, 0.0), prob_map.get(-1, 0.0), prob_map.get(0, 0.0)
        probs = {"buy": int(round(buy_p*100)), "sell": int(round(sell_p*100)), "wait": int(round(wait_p*100))}
        close = float(data["Close"].iloc[-1])
        rsi = float(data["RSI"].iloc[-1]) if "RSI" in data else 50.0
        adx = float(data["ADX"].iloc[-1]) if "ADX" in data else 20.0
                # 売買判定とは別に、方向候補を表示するための参考情報
        candidate = None
        candidate_reason = "買い・売りの確率差が小さい"

        if buy_p >= 0.40 and buy_p > sell_p + 0.08 and rsi < 80:
            candidate = "買い候補"
            candidate_reason = "買い確率が売り確率を8ポイント超上回る"
        elif sell_p >= 0.40 and sell_p > buy_p + 0.08 and rsi > 20:
            candidate = "売り候補"
            candidate_reason = "売り確率が買い確率を8ポイント超上回る"

        st.session_state["directional_candidate"] = {
            "ticker": ticker,
            "interval": interval,
            "candidate": candidate,
            "reason": candidate_reason,
            "buy": probs["buy"],
            "sell": probs["sell"],
            "quality_ok": bool(model_quality_ok),
            "test_count": int(test_count),
        }
        if not model_quality_ok:
            status = "WAIT (様子見)"
        elif buy_p >= 0.53 and buy_p > sell_p and rsi < 78:
            status = "BUY (買い)"
        elif sell_p >= 0.53 and sell_p > buy_p and rsi > 22:
            status = "SELL (売り)"
        else:
            status = "WAIT (様子見)"
        # Forecast logging is performed after freshness and multi-timeframe safety gates.
        # This avoids recording a raw BUY/SELL recommendation when the final UI decision is WAIT.
        atr_ratio = float(data["ATR_Ratio"].iloc[-1]) if "ATR_Ratio" in data else 0.0
        market_type = "トレンド相場" if adx >= 25 else "レンジ相場"
        if atr_ratio > 0.01:
            market_type += "・高ボラ"
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        return status, max(buy_p, sell_p)*100, test_count, accuracy, market_type, probs, importances, history
    except Exception as exc:
        st.warning(f"AI分析で問題が発生したため、様子見にしています（{type(exc).__name__}）。")
        return fallback


# Load persisted settings once into Streamlit state.
if "settings_loaded" not in st.session_state:
    _settings = load_settings()
    for _k, _v in _settings.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v
    st.session_state["settings_loaded"] = True

st.title("AI FX分析ツール")
st.caption("価格分析・注文条件の計算補助専用｜口座接続・自動発注なし。まず通貨ペアと時間足を選び、判定・データ鮮度・想定損失を確認してください。表示確率や分類一致率は実際の勝率・利益を保証しません。")

# iPad / スマートフォンで selectbox の検索キーボードが出る不便さを避けるため、
# 通貨ペア・時間足はタップで切り替えるラジオボタンにする。
# 既存の session_state キーを維持し、保存済みの選択状態との互換性を保つ。
col_sel1, col_sel2 = st.columns([3, 2])
with col_sel1:
    st.markdown("**通貨ペア（タップして選択）**")
    selected_label = st.radio(
        "通貨ペア", list(PAIRS.keys()),
        index=list(PAIRS.keys()).index(st.session_state.get("selected_pair_label", list(PAIRS.keys())[0]))
        if st.session_state.get("selected_pair_label", list(PAIRS.keys())[0]) in PAIRS else 0,
        key="selected_pair_label", horizontal=True, label_visibility="collapsed",
    )
with col_sel2:
    st.markdown("**時間足（タップして選択）**")
    tf_label = st.radio(
        "時間足", list(TIMEFRAMES.keys()),
        index=list(TIMEFRAMES.keys()).index(st.session_state.get("selected_tf_label", list(TIMEFRAMES.keys())[2]))
        if st.session_state.get("selected_tf_label", list(TIMEFRAMES.keys())[2]) in TIMEFRAMES else 2,
        key="selected_tf_label", horizontal=True, label_visibility="collapsed",
    )
ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]
is_jpy = "JPY" in ticker
pip_unit = 0.01 if is_jpy else 0.0001
price_fmt = "%.3f" if is_jpy else "%.5f"
spread_pips = DEFAULT_SPREAD_PIPS.get(ticker, 0.5)

st.sidebar.header("⚙️ 資金 & リスク設定")
st.sidebar.number_input("口座資金 (円)", min_value=10000, step=10000, key="account_balance", on_change=save_settings, help="松井証券FXの現在の実際の評価額に合わせて手入力してください。口座残高・評価額を自動取得する機能はありません。")
st.sidebar.caption("⚠️ ここは手入力の計算用金額です。実際の評価額と違う場合、許容損失額・推奨数量もずれます。")
st.sidebar.number_input("1注文の基本数量 (万通貨)", min_value=0.01, step=0.01, key="quantity_wan", on_change=save_settings)
st.sidebar.markdown("---")
st.sidebar.header("🕒 取引スタイル")
trade_style = st.sidebar.radio(
    "保有時間に合わせて分析の見方を切り替え",
    ["短期（数分〜数時間）", "デイトレード（当日決済）", "中期（1〜数日）", "長期（数日以上）"],
    index=["短期（数分〜数時間）", "デイトレード（当日決済）", "中期（1〜数日）", "長期（数日以上）"].index(
        st.session_state.get("trade_style", "デイトレード（当日決済）")
    ) if st.session_state.get("trade_style", "デイトレード（当日決済）") in ["短期（数分〜数時間）", "デイトレード（当日決済）", "中期（1〜数日）", "長期（数日以上）"] else 1,
    key="trade_style",
    help="表示する推奨時間足と確認ポイントを切り替えます。AIモデル自体の学習条件は自動変更しません。",
)
_STYLE_GUIDANCE = {
    "短期（数分〜数時間）": ("15分足で方向・形を確認し、5分足はエントリータイミングの補助に使います。1時間足・4時間足も逆方向の大きな流れがないか確認してください。", "15分足 → 5分足（補助） → 1時間足・4時間足で逆行リスク確認"),
    "デイトレード（当日決済）": ("4時間足で大きな方向、1時間足で当日の流れ、15分足で成行エントリーのタイミングを確認します。日足は重要な節目と逆方向リスクの参考にします。", "日足（背景） → 4時間足 → 1時間足 → 15分足（タイミング）"),
    "中期（1〜数日）": ("4時間足で方向を決め、1時間足でエントリーを検討します。15分足はタイミング調整に限定し、5分足の細かな値動きに振り回されないようにします。日足も背景として確認します。", "日足（背景） → 4時間足 → 1時間足 → 15分足（補助）"),
    "長期（数日以上）": ("日足で相場全体の方向、4時間足でエントリー候補を確認します。週末・経済指標・スワップ・保有中のギャップリスクを確認してください。", "日足 → 4時間足 → 1時間足（補助）"),
}
_style_text, _style_sequence = _STYLE_GUIDANCE[trade_style]
st.sidebar.caption("推奨：" + _style_sequence)
st.sidebar.caption(_style_text)
st.sidebar.markdown("---")
st.sidebar.header("🎯 単発トレード リスク設定")
st.sidebar.number_input("1取引あたりの許容リスク (%)", min_value=0.1, max_value=5.0, step=0.1, key="risk_percent", on_change=save_settings)
st.sidebar.number_input("デフォルト リスクリワード比 (RR)", min_value=0.5, max_value=5.0, step=0.1, key="rr_ratio", on_change=save_settings)
def apply_conservative_risk_defaults():
    # A transparent conservative preset, not a claim of AI-optimized profitability.
    st.session_state["risk_percent"] = 0.5
    st.session_state["rr_ratio"] = 1.5
    save_settings()
st.sidebar.button("🛡️ 安全寄りの基本設定を適用", on_click=apply_conservative_risk_defaults,
                  help="許容リスク0.5%、RR1.5に設定します。利益を保証する設定ではありません。")
st.sidebar.caption("※リスク率は資金管理の設定です。AIが勝率を保証して自動最適化するものではありません。")
_sidebar_balance = float(st.session_state.get("account_balance", 500000))
_sidebar_risk = float(st.session_state.get("risk_percent", 1.0))
st.sidebar.info(f"現在の1取引あたり損失上限の目安：**{_sidebar_balance * _sidebar_risk / 100:,.0f}円**\n\n損切り滑り・スプレッド拡大で実損失が上限を超えることがあります。")
st.sidebar.markdown("---")
st.sidebar.header("🔄 AIモデル管理（自己学習つき）")
if st.sidebar.button("🧠 AIモデルを今すぐ強制再学習 (知見反映)"):
    model_file = _model_filename(ticker, tf_config["interval"])
    try:
        if os.path.exists(model_file):
            os.remove(model_file)
    except OSError:
        pass
    st.cache_data.clear()
    st.session_state["force_retrain_once"] = True
    st.toast("再学習を要求しました。次の分析でモデルを再作成します。", icon="🧠")
    st.rerun()
m_file_info = _model_filename(ticker, tf_config["interval"])
if os.path.exists(m_file_info):
    m_time_dt = datetime.fromtimestamp(os.path.getmtime(m_file_info), tz=JST)
    m_time = m_time_dt.strftime("%Y-%m-%d %H:%M")
    model_age_hours = max(0.0, (time.time() - os.path.getmtime(m_file_info)) / 3600)
    st.sidebar.caption(f"🤖 最終学習日時: `{m_time}`")
    if model_age_hours > 4:
        st.sidebar.warning("モデルが4時間以上経過しています。次回分析時に再学習されます。")
    else:
        st.sidebar.caption(f"モデル経過時間: 約 {model_age_hours:.1f} 時間")
    st.sidebar.caption("※再学習はアプリが実行・再分析された時に行われます。取引結果の記録だけでは学習されません。")
else:
    st.sidebar.caption("🤖 現在のモデル状態: `初回自動作成待ち`")
    st.sidebar.caption("※十分な価格データを取得できた次回分析時にモデルを作成します。")
st.sidebar.markdown("---")
st.sidebar.header("🔄 更新・メンテナンス")
st.sidebar.checkbox(
    "5分ごとに自動更新（画面維持）",
    key="auto_refresh",
    on_change=save_settings,
    help="対応するStreamlitではメイン表示領域だけを5分ごとに再実行します。AI分析も再計算される場合があります。初期設定はオフです。",
)
st.sidebar.info(
    "自動更新は初期設定でオフです。オンにすると対応環境ではメイン表示領域だけを5分ごとに更新します。 "
    "ニュースは15分キャッシュ、価格データは60秒キャッシュです。自動発注は行いません。"
)
with st.sidebar.expander("🔔 Discord通知設定・接続テスト", expanded=False):
    st.text_input("Discord Webhook URL", key="webhook_url", on_change=save_settings, placeholder="https://discord.com/api/webhooks/...")
    st.caption("Webhook URLは通知先への送信権限を持つ秘密情報です。公開・共有しないでください。")
    if st.button("📨 Discordへテスト通知", key="discord_test_notification", use_container_width=True):
        webhook_url_test = str(st.session_state.get("webhook_url", "")).strip()
        if not webhook_url_test:
            st.error("Webhook URLが未入力です。DiscordでWebhookを作成し、URLを入力してください。")
        elif requests is None:
            st.error("requestsライブラリが利用できません。実行環境で `pip install requests` を実行してください。")
        elif not webhook_url_test.startswith("https://discord.com/api/webhooks/") and not webhook_url_test.startswith("https://discordapp.com/api/webhooks/"):
            st.error("Discord Webhook URLの形式を確認してください。URL全体を入力する必要があります。")
        else:
            save_settings()
            test_ok = send_webhook_alert(webhook_url_test, "✅ AI FX分析ツールからのテスト通知です。これが届けばDiscordへの送信経路は動作しています。")
            if test_ok:
                st.success("Discordへの送信要求が成功しました。Discordの通知先チャンネルを確認してください。")
            else:
                st.error("送信に失敗しました。Webhook URL、Discord側のWebhook状態、実行環境の通信制限を確認してください。")
    st.caption("自動通知は条件成立時のみ送信されます。テスト通知はボタンを押した時点で送信します。")
if st.sidebar.button("💾 設定を即時手動保存"):
    save_settings()
    st.toast("設定を保存しました！", icon="💾")
with st.sidebar.expander("📂 設定ファイル管理 (バックアップ)", expanded=False):
    safe_backup_settings = load_settings().copy()
    safe_backup_settings["webhook_url"] = ""
    st.download_button("📥 設定をバックアップ (.json)", data=json.dumps(safe_backup_settings, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="fx_analyzer_settings.json", mime="application/json", use_container_width=True)
    _full_backup = build_data_backup_bundle()
    st.download_button(
        "🗃️ 設定・予測履歴・取引記録をまとめて保存",
        data=json.dumps(_full_backup, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8"),
        file_name=f"ai_fx_full_backup_{datetime.now(JST).strftime('%Y%m%d')}.json",
        mime="application/json", use_container_width=True,
        help="設定、予測検証履歴、注文候補の検証履歴、実取引記録、フィードバックを保存します。Webhook URLは含めません。",
    )
    st.caption("端末変更・アプリ更新前に、まとめてバックアップを保存してください。Webhook URLはセキュリティのため保存対象外です。")
    full_backup_upload = st.file_uploader("📤 まとめて保存したバックアップを復元", type=["json"], key="full_data_backup_upload")
    if full_backup_upload is not None:
        st.warning("復元すると、現在の設定・履歴ファイルがバックアップ内の内容に置き換わります。先に現在のデータをバックアップしてください。")
        if st.button("⚠️ バックアップを復元する", key="restore_full_data_backup"):
            try:
                _bundle = json.load(full_backup_upload)
                restore_data_backup_bundle(_bundle)
                st.success("設定と履歴を復元しました。アプリを再読み込みしてください。Webhook URLは復元していません。")
                st.rerun()
            except Exception as exc:
                st.error(f"バックアップの復元に失敗しました: {exc}")
    uploaded_file = st.file_uploader("📤 設定のみを復元 (アップロード)", type=["json"], key="settings_only_upload")
    if uploaded_file is not None:
        try:
            imported_data = json.load(uploaded_file)
            if isinstance(imported_data, dict):
                safe_data = {k: imported_data[k] for k in DEFAULT_SETTINGS if k in imported_data}
                safe_data["webhook_url"] = ""  # Never import a webhook URL from an untrusted backup.
                with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                    json.dump(safe_data, f, ensure_ascii=False, indent=2)
                st.success("設定を復元しました。再読み込みしてください。")
        except Exception as exc:
            st.error(f"設定の復元に失敗しました: {exc}")
st.sidebar.button("🔄 設定を初期値にリセット", on_click=reset_settings_callback)
if st.sidebar.button("🔄 最新データに更新"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("更新ボタンは価格・ニュースのキャッシュを消して再取得します。選択中の画面は維持されます。連続して押す必要はありません。")

def _render_main_area():
    with st.spinner("最新相場データ & ニュースを取得中..."):
        data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
        data_4h = load_and_process_data(ticker, "90d", "4h", "4時間足 (中期・リピート用)")
        data_1h = data if tf_config["interval"] == "1h" else load_and_process_data(ticker, "60d", "1h", "1時間足 (方向確認用)")
        data_15m = data if tf_config["interval"] == "15m" else load_and_process_data(ticker, "30d", "15m", "15分足 (タイミング確認用)")
        data_htf = load_and_process_data(ticker, "2y", "1d", "日足")
        news_result = fetch_news_and_impact(ticker)
        news_items = news_result.get("items", []) if isinstance(news_result, dict) else []
        news_status = news_result.get("status", "ニュース取得状況は不明です") if isinstance(news_result, dict) else "ニュース取得状況は不明です"
        news_errors = news_result.get("errors", []) if isinstance(news_result, dict) else []
    if data_4h is None and data is not None:
        data_4h = data
    if data is None or data.empty:
        st.error("データの取得に失敗したか、分析に必要な本数が不足しています。時間をおいて更新してください。")
        st.stop()

    last_bar_index = pd.Timestamp(data.index[-1])
    last_bar_jst = safe_to_tokyo_tz(data).index[-1].strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<div class="update-time">最終確定足の時刻: <b>{last_bar_jst} JST</b> (最新の未確定足は除外。取得時刻とは異なります)</div>', unsafe_allow_html=True)
    # Timeframe-aware freshness check. Yahoo timestamps can be delayed and are not Matsui live quotes.
    data_is_stale = False
    freshness_check_ok = False
    age_minutes = None
    try:
        # Do not guess the timezone of a naive timestamp. If the provider omits timezone
        # information, freshness is unknown and new-entry decisions are held for safety.
        if last_bar_index.tzinfo is None:
            freshness_check_ok = False
            data_is_stale = True
            st.warning(
                "価格データの時刻にタイムゾーン情報がありません。UTCと決めつけず、"
                "安全のため新規注文判断を保留します。配信元の時刻仕様と松井証券FXの現在Bid/Askを確認してください。"
            )
        else:
            last_bar_utc = last_bar_index.tz_convert("UTC")
            now_utc = pd.Timestamp.now(tz="UTC")
            age_minutes = max(0.0, (now_utc - last_bar_utc).total_seconds() / 60.0)
            freshness_check_ok = True
            # Limits are deliberately conservative; FX closes and provider delays can create false alarms.
            stale_limits = {"5m": 25, "15m": 60, "1h": 210, "4h": 600, "1d": 2160}
            stale_limit = stale_limits.get(tf_config["interval"], 210)
            data_is_stale = age_minutes > stale_limit
            if data_is_stale:
                st.warning(
                    f"価格データが古い可能性があります（最終確定足から約{age_minutes / 60:.1f}時間、判定基準 {stale_limit / 60:.1f}時間）。"
                    "休場や配信遅延の可能性があります。Yahoo Financeの時刻は松井証券FXのリアルタイムレートではありません。"
                    "松井証券の現在Bid/Askを確認し、データ更新を確認できるまで新規注文判断を保留してください。"
                )
            else:
                st.caption(f"データ鮮度の目安：最終確定足から約{age_minutes / 60:.1f}時間（配信元の遅延・時刻仕様を完全には検出できません）。")
    except Exception:
        data_is_stale = True
        freshness_check_ok = False
        st.warning("価格データの時刻を検証できません。安全のため新規注文判断を保留し、証券会社の現在レートを確認してください。")
    latest_price = float(clean_series(data["Close"]).iloc[-1])
    latest_atr = float(clean_series(data["ATR"]).iloc[-1]) if "ATR" in data else latest_price * 0.005
    force_retrain = bool(st.session_state.pop("force_retrain_once", False))
    status, conf, total_signals_count, win_rate, m_type, prob_dict, feature_importances, ai_signal_history = analyze_signal_with_backtest(
        data, data_htf, ticker, tf_config["interval"], force_retrain=force_retrain
    )
    raw_model_status = status
    if data_is_stale:
        status = "WAIT（データ鮮度未確認）"

    # Conservative day-trading filter: do not show BUY/SELL as the final recommendation
    # when required higher-timeframe trend data is missing, neutral, or conflicting.
    def _quick_trend_sign(frame):
        try:
            if frame is None or frame.empty or "Close" not in frame:
                return None
            close = clean_series(frame["Close"]).dropna()
            if len(close) < 50:
                return None
            e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            last = close.iloc[-1]
            if last > e20 > e50:
                return 1
            if last < e20 < e50:
                return -1
            return 0
        except Exception:
            return None

    mtf_gate_reason = None
    # Apply a conservative multi-timeframe consistency check for every trading style.
    # Required timeframes are descriptive EMA trend filters, not standalone signals.
    _trend_requirements = {
        "短期（数分〜数時間）": [("15分足", data_15m), ("1時間足", data_1h)],
        "デイトレード（当日決済）": [("4時間足", data_4h), ("1時間足", data_1h)],
        "中期（1〜数日）": [("日足", data_htf), ("4時間足", data_4h)],
        "長期（数日以上）": [("日足", data_htf), ("4時間足", data_4h)],
    }
    _required = _trend_requirements.get(trade_style, _trend_requirements["デイトレード（当日決済）"])
    _required_signs = [(name, _quick_trend_sign(frame)) for name, frame in _required]
    _direction = 1 if str(status).startswith("BUY") else -1 if str(status).startswith("SELL") else 0
    if not data_is_stale and _direction != 0:
        if any(sign is None for _, sign in _required_signs):
            status = "WAIT（複数時間足データ不足）"
            mtf_gate_reason = "必要な時間足のデータが不足しているため、AIの買い・売り判定を保留しました。"
        elif any(sign == 0 for _, sign in _required_signs):
            status = "WAIT（時間足の方向不明瞭）"
            mtf_gate_reason = "必要な時間足の方向が中立・混在のため、AIの買い・売り判定を保留しました。"
        elif any(sign != _direction for _, sign in _required_signs):
            _against = [name for name, sign in _required_signs if sign != _direction]
            status = "WAIT（AIと時間足が不一致）"
            mtf_gate_reason = "AI判定と時間足の方向が一致しません（" + "・".join(_against) + "）。無理に方向を決めず、判定を保留しました。"
    # Save the final gated decision only when the selected price series passed freshness checks.
    # Stale/unknown timestamps must not create apparently actionable forecast records.
    if not data_is_stale and freshness_check_ok:
        record_forecast_and_update(data, ticker, tf_config["interval"], status, prob_dict, LOOKAHEAD_BARS)

    std_series = clean_series(data["Close"]).rolling(20, min_periods=1).std()
    band_width_pct = float((std_series.iloc[-1] * 4.0) / (latest_price + 1e-10) * 100)
    adx_val_curr = float(clean_series(data["ADX"]).iloc[-1]) if "ADX" in data else 20.0
    vol_ratio_series = clean_series(data["Vol_Ratio"])
    vol_ratio_curr = float(vol_ratio_series.iloc[-1]) if len(vol_ratio_series) else 1.0
    is_volatility_expanding = bool(vol_ratio_series.notna().any() and vol_ratio_curr > vol_ratio_series.mean() * 1.2)
    is_squeeze = band_width_pct < 0.35
    vol_alert_msg = None
    if is_volatility_expanding or adx_val_curr > 28.0:
        if prob_dict["buy"] > prob_dict["sell"]:
            vol_alert_msg = f"⚠️ 【急変動・上方ブレイク警戒】買い方向確率 {prob_dict['buy']}%（予測保証ではありません）"
        else:
            vol_alert_msg = f"⚠️ 【急変動・下方ブレイク警戒】売り方向確率 {prob_dict['sell']}%（予測保証ではありません）"
    elif is_squeeze:
        vol_alert_msg = f"⚡ 【ボラティリティ収縮】ボリンジャー幅が狭くなっています（{band_width_pct:.2f}%）。急変動を予測・保証するものではありません。"
    if vol_alert_msg:
        st.markdown(f'<div class="alert-box">{html.escape(vol_alert_msg)}</div>', unsafe_allow_html=True)
        if is_volatility_expanding or adx_val_curr > 28.0:
            st.warning("値動きが強まっている可能性があります。成行で追いかけず、Bid/Ask・スプレッド・損切り幅を確認してください。急変動の方向や経済指標の発表時刻を、この検知だけで確定することはできません。")
        webhook_url = st.session_state.get("webhook_url", "")
        if webhook_url:
            key = f"last_webhook_{ticker}_{tf_config['interval']}"
            time_key = key + "_time"
            current_time = time.time()
            if st.session_state.get(key) != vol_alert_msg or current_time - st.session_state.get(time_key, 0) > 1800:
                if send_webhook_alert(webhook_url, f"FX参考アラート {selected_label}\n{vol_alert_msg}"):
                    st.session_state[key] = vol_alert_msg
                    st.session_state[time_key] = current_time

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("確定足レート", price_fmt % latest_price)
    m2.markdown("**AI 推奨アクション**")
    badge_html = '<div class="badge-buy">BUY</div>' if status.startswith("BUY") else '<div class="badge-sell">SELL</div>' if status.startswith("SELL") else '<div class="badge-wait">WAIT</div>'
    m2.markdown(badge_html, unsafe_allow_html=True)
    m3.metric("時系列テスト分類一致率", f"{win_rate:.1f}%", f"検証数: {total_signals_count}本")
    m4.metric("相場環境", m_type, f"ATR: {latest_atr/pip_unit:.1f} pips")
    st.caption("価格データはYahoo Finance由来で、松井証券FXのリアルタイムBid/Askや約定価格とは異なる場合があります。分類一致率は予測ラベルの一致率で、実際の勝率・利益率ではありません。スプレッドは参考値です。")
    if mtf_gate_reason:
        st.warning(mtf_gate_reason)
    # BUY/SELLの正式判定とは分けて、方向候補を表示
    _candidate_info = st.session_state.get("directional_candidate", {})

    if (
        _candidate_info.get("ticker") == ticker
        and _candidate_info.get("interval") == tf_config["interval"]
        and _candidate_info.get("candidate")
    ):
        _candidate_label = _candidate_info["candidate"]
        _quality_label = (
            "検証基準クリア"
            if _candidate_info.get("quality_ok")
            else "検証基準未達"
        )

        st.info(
            f"方向候補：{_candidate_label} "
            f"（買い確率 {_candidate_info.get('buy', 0)}%・"
            f"売り確率 {_candidate_info.get('sell', 0)}%）"
            f"／{_quality_label}。"
            "参考候補であり、売買推奨や利益予測ではありません。"
            "データ鮮度・時間足・スプレッド・損切り条件を確認してください。"
        )   
    elif trade_style == "デイトレード（当日決済）" and not data_is_stale:
        st.caption("デイトレードの最終推奨には4時間足・1時間足の方向一致を適用しています。詳細は下の折りたたみ欄で確認できます。")

    # Multi-timeframe context panel: trend alignment is a descriptive filter, not a standalone signal.
    def _trend_summary(frame):
        try:
            if frame is None or frame.empty or "Close" not in frame or len(frame) < 50:
                return "データ不足", None
            close = clean_series(frame["Close"]).dropna()
            if len(close) < 50:
                return "データ不足", None
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            last = float(close.iloc[-1])
            if last > ema20 > ema50:
                return "上昇", 1
            if last < ema20 < ema50:
                return "下降", -1
            return "中立・混在", 0
        except Exception:
            return "判定不可", None

    _trend_frames = [("日足（参考）", data_htf), ("4時間足", data_4h), ("1時間足", data_1h), ("15分足", data_15m)]
    _trend_rows = []
    _trend_signs = {}
    for _tf_name, _tf_frame in _trend_frames:
        _trend_text, _trend_sign = _trend_summary(_tf_frame)
        _trend_rows.append({"時間足": _tf_name, "EMA20/50による方向の目安": _trend_text})
        _trend_signs[_tf_name] = _trend_sign
    with st.expander("🧭 マルチ時間足の方向確認（詳細）", expanded=False):
        st.caption("終値とEMA20・EMA50の位置関係による簡易判定です。デイトレードでは4時間足・1時間足を方向確認、15分足をタイミング補助、日足を背景確認に使います。将来の方向や利益を保証せず、AI推奨アクションを置き換えるものではありません。")
        st.dataframe(pd.DataFrame(_trend_rows), use_container_width=True, hide_index=True)
        if trade_style == "短期（数分〜数時間）":
            _required_trends = [_trend_signs.get("15分足"), _trend_signs.get("1時間足")]
            _mode_label = "15分足・1時間足"
        elif trade_style == "デイトレード（当日決済）":
            _required_trends = [_trend_signs.get("4時間足"), _trend_signs.get("1時間足")]
            _mode_label = "4時間足・1時間足（15分足はタイミング補助）"
        elif trade_style == "中期（1〜数日）":
            _required_trends = [_trend_signs.get("日足（参考）"), _trend_signs.get("4時間足"), _trend_signs.get("1時間足")]
            _mode_label = "日足・4時間足・1時間足"
        else:
            _required_trends = [_trend_signs.get("日足（参考）"), _trend_signs.get("4時間足")]
            _mode_label = "日足・4時間足"
        if None in _required_trends:
            st.info(f"{trade_style}：{_mode_label}の方向一致は、データ不足のため判定できません。")
        elif all(x == _required_trends[0] for x in _required_trends) and _required_trends[0] in (1, -1):
            st.success(f"{trade_style}：{_mode_label}の方向は一致しています（{ '上昇' if _required_trends[0] == 1 else '下降' }）。ほかの条件とリスクも確認してください。")
        else:
            st.warning(f"{trade_style}：{_mode_label}の方向が一致していないか、中立です。無理なエントリーを避けるための注意表示です。")
        if trade_style == "デイトレード（当日決済）" and _trend_signs.get("日足（参考）") in (1, -1):
            _daily_sign = _trend_signs["日足（参考）"]
            _core_signs = [_trend_signs.get("4時間足"), _trend_signs.get("1時間足")]
            if all(x in (1, -1) for x in _core_signs) and any(x != _daily_sign for x in _core_signs):
                st.info("日足の方向と短い時間足の方向が一部異なります。日足は背景情報として扱い、逆行リスクと損切り幅を確認してください。")
        if data_is_stale:
            st.error("データ鮮度を確認できないため、AI推奨は売買判断に使わず『判定保留』として扱ってください。")
    _eval_metrics_display = st.session_state.get("model_eval_metrics", {})
    if _eval_metrics_display:
        _em1, _em2, _em3 = st.columns(3)
        _em1.metric("バランス一致率", f"{_eval_metrics_display.get('balanced_accuracy', 0):.1f}%")
        _em2.metric("多数派だけの基準", f"{_eval_metrics_display.get('baseline_balanced_accuracy', 0):.1f}%")
        _em3.metric("基準モデル比", "条件を満たす" if _eval_metrics_display.get("quality_ok", False) else "優位性未確認")
        with st.expander("AI検証の詳細", expanded=False):
            _ed1, _ed2, _ed3 = st.columns(3)
            _ed1.metric("検証サンプル", f"{_eval_metrics_display.get('test_count', 0)}件")
            _ed2.metric("学習サンプル", f"{_eval_metrics_display.get('train_count', 0)}件")
            _ed3.metric("Brierスコア", f"{_eval_metrics_display.get('brier_score', float('nan')):.3f}", help="多クラス確率誤差の目安。低いほど良い傾向ですが、利益率ではありません。")
            st.caption(f"検証期間：{_eval_metrics_display.get('test_start', '—')} ～ {_eval_metrics_display.get('test_end', '—')}")
        st.caption("バランス一致率は各クラスの当たりやすさを平均した指標です。Brierスコアは予測確率の誤差の目安です。どちらも実際の勝率・利益や将来の精度を保証しません。")
    st.markdown("---")

    swing_high_4h = float(clean_series(data_4h["High"]).iloc[-300:].max()) if data_4h is not None else latest_price * 1.03
    swing_low_4h = float(clean_series(data_4h["Low"]).iloc[-300:].min()) if data_4h is not None else latest_price * 0.97
    atr_4h = float(clean_series(data_4h["ATR"]).iloc[-1]) if data_4h is not None and "ATR" in data_4h else latest_price * 0.005

    def is_valid_range(r_low, r_up, current_p):
        return bool(np.isfinite([r_low, r_up, current_p]).all() and r_low > 0 and r_up > r_low
                    and r_low >= current_p * 0.4 and r_up <= current_p * 1.6)

    if ticker not in st.session_state["ranges"] or not is_valid_range(
        float(st.session_state["ranges"].get(ticker, {}).get("lower", 0)),
        float(st.session_state["ranges"].get(ticker, {}).get("upper", 0)), latest_price
    ):
        st.session_state["ranges"][ticker] = {"lower": float(swing_low_4h), "upper": float(swing_high_4h)}
        save_settings()
    key_lower, key_upper = f"in_lower_{ticker}", f"in_upper_{ticker}"
    current_range = st.session_state["ranges"][ticker]
    if key_lower not in st.session_state: st.session_state[key_lower] = float(current_range["lower"])
    if key_upper not in st.session_state: st.session_state[key_upper] = float(current_range["upper"])
    with st.expander("⚙️ リピート自動売買のレンジ調整", expanded=False):
        rc1, rc2 = st.columns(2)
        def update_range_callback():
            lo, hi = float(st.session_state[key_lower]), float(st.session_state[key_upper])
            if lo < hi:
                st.session_state["ranges"][ticker] = {"lower": lo, "upper": hi}
                save_settings()
        def trigger_auto_calc_callback():
            st.session_state["ranges"][ticker] = {"lower": float(swing_low_4h), "upper": float(swing_high_4h)}
            st.session_state[key_lower] = float(swing_low_4h)
            st.session_state[key_upper] = float(swing_high_4h)
            save_settings()
        in_lower = rc1.number_input("レンジ下限", step=0.1 if is_jpy else 0.001, format=price_fmt, key=key_lower, on_change=update_range_callback)
        in_upper = rc2.number_input("レンジ上限", step=0.1 if is_jpy else 0.001, format=price_fmt, key=key_upper, on_change=update_range_callback)
        st.button("✨ 4時間足高値・安値（広め）からレンジを自動計算", on_click=trigger_auto_calc_callback)
    user_lower, user_upper = min(float(st.session_state[key_lower]), float(st.session_state[key_upper])), max(float(st.session_state[key_lower]), float(st.session_state[key_upper]))
    if user_upper <= user_lower:
        user_upper = user_lower + pip_unit * 100
    user_half = (user_upper + user_lower) / 2
    stop_buffer_pips = max(10.0, round((atr_4h / pip_unit) * 1.5, 1))
    stop_buffer_val = stop_buffer_pips * pip_unit
    buy_stop_loss, sell_stop_loss = user_lower - stop_buffer_val, user_upper + stop_buffer_val

    # 定期スキャン：中期トレード向けに1h/4hを選択し、日足・4h・1hの方向を比較。
    st.markdown("### 🔎 中期トレード候補スキャナー")
    st.caption("5通貨ペアをまとめて確認し、数時間〜数日保有向けの候補をランキングします。候補は分析上の参考で、注文可能・利益確実を意味しません。")
    # iPadで selectbox の検索欄にフォーカスしてキーボードが出るのを避けるため、
    # 短い候補は radio で直接タップ選択できるようにする。
    scan_control_cols = st.columns([1, 1, 1, 1])
    with scan_control_cols[0]:
        st.markdown("**スキャン時間足**")
        scan_tf_label = st.radio(
            "スキャン時間足", ["1時間足", "4時間足"],
            key="pair_scan_timeframe", horizontal=True, label_visibility="collapsed",
            help="中期トレードでは1時間足を基本、4時間足はゆったりした候補探しに使います。",
        )
    with scan_control_cols[1]:
        st.markdown("**自動チェック間隔**")
        scan_interval_minutes = st.radio(
            "自動チェック間隔", [15, 30, 60],
            key="pair_scan_interval_minutes", horizontal=True,
            format_func=lambda x: f"{x}分", label_visibility="collapsed",
        )
    with scan_control_cols[2]:
        st.checkbox("定期スキャンを有効化", key="auto_pair_scan", on_change=save_settings, help="アプリ画面が開いている間だけ動作します。バックグラウンド常時監視ではありません。")
    with scan_control_cols[3]:
        run_pair_scan_button = st.button("🔍 今すぐ全通貨ペア確認", type="primary", use_container_width=True)
    scan_configs = {
        "1時間足": {"interval": "1h", "period": "60d"},
        "4時間足": {"interval": "4h", "period": "90d"},
    }
    scan_config = scan_configs[scan_tf_label]
    st.caption(f"スキャン対象: 5通貨ペア / {scan_tf_label}・4時間足・日足を比較 / 価格データ: Yahoo Finance（松井証券の実レートではありません）")

    if "pair_scan_results" not in st.session_state:
        st.session_state["pair_scan_results"] = {}
    now_epoch = time.time()
    previous_scan_epoch = float(st.session_state.get("pair_scan_last_epoch", 0) or 0)
    auto_scan_due = bool(st.session_state.get("auto_pair_scan", False)) and (
        now_epoch - previous_scan_epoch >= int(scan_interval_minutes) * 60
    )
    run_pair_scan = bool(run_pair_scan_button or auto_scan_due)
    if run_pair_scan:
        scan_results = []
        original_eval_metrics = st.session_state.get("model_eval_metrics")
        progress = st.progress(0, text="5通貨ペアを確認しています…")
        pair_items = list(PAIRS.items())
        for idx, (pair_label, pair_ticker) in enumerate(pair_items, start=1):
            try:
                scan_data = load_and_process_data(pair_ticker, scan_config["period"], scan_config["interval"], scan_tf_label)
                scan_4h = load_and_process_data(pair_ticker, "90d", "4h", "4時間足")
                scan_1h = scan_data if scan_config["interval"] == "1h" else load_and_process_data(pair_ticker, "60d", "1h", "1時間足")
                scan_daily = load_and_process_data(pair_ticker, "2y", "1d", "日足")
                if scan_data is None or scan_data.empty or scan_4h is None or scan_1h is None or scan_daily is None:
                    scan_results.append({"通貨ペア": pair_label, "順位": "—", "判定": "判定保留", "AI方向": "—", "時間足一致": "確認不可", "買い確率": "—", "売り確率": "—", "価格": "—", "損切り目安": "—", "利確目安": "—", "スプレッド目安(pips)": DEFAULT_SPREAD_PIPS.get(pair_ticker, 0.5), "理由": "必要な時間足データが不足しています"})
                else:
                    result = analyze_signal_with_backtest(scan_data, scan_daily, pair_ticker, scan_config["interval"])
                    sig, confidence, test_n, accuracy, market_type, probs, importances, history = result
                    metrics = st.session_state.get("model_eval_metrics", {})
                    last_price_scan = float(clean_series(scan_data["Close"]).iloc[-1])
                    scan_last_ts = pd.Timestamp(scan_data.index[-1])
                    scan_last_utc = scan_last_ts.tz_localize("UTC") if scan_last_ts.tzinfo is None else scan_last_ts.tz_convert("UTC")
                    scan_age_min = max(0.0, (pd.Timestamp.now(tz="UTC") - scan_last_utc).total_seconds() / 60.0)
                    stale_limits = {"1h": 210, "4h": 600}
                    scan_is_stale = scan_age_min > stale_limits.get(scan_config["interval"], 210)
                    atr_scan = float(clean_series(scan_data["ATR"]).iloc[-1]) if "ATR" in scan_data else 0.0
                    pip_scan = 0.01 if "JPY" in pair_ticker else 0.0001
                    spread_scan = DEFAULT_SPREAD_PIPS.get(pair_ticker, 0.5)
                    quality_ok = bool(metrics.get("quality_ok", False))
                    enough_movement = atr_scan / pip_scan >= spread_scan * 3 if pip_scan else False

                    def _trend_dir(frame):
                        try:
                            close_v = float(clean_series(frame["Close"]).iloc[-1])
                            ema20_v = float(clean_series(frame["EMA_20"]).iloc[-1])
                            ema200_v = float(clean_series(frame["EMA_200"]).iloc[-1])
                            if not np.isfinite([close_v, ema20_v, ema200_v]).all(): return "不明"
                            if close_v > ema20_v and ema20_v > ema200_v: return "上昇"
                            if close_v < ema20_v and ema20_v < ema200_v: return "下降"
                            return "中立"
                        except Exception:
                            return "不明"
                    trend_1h, trend_4h, trend_d = _trend_dir(scan_1h), _trend_dir(scan_4h), _trend_dir(scan_daily)
                    ai_direction = "買い" if str(sig).startswith("BUY") else "売り" if str(sig).startswith("SELL") else "様子見"
                    trend_values = [trend_1h, trend_4h, trend_d]
                    matching = sum(1 for t in trend_values if t == ai_direction.replace("買い", "上昇").replace("売り", "下降")) if ai_direction != "様子見" else 0
                    opposing = ai_direction == "買い" and "下降" in trend_values or ai_direction == "売り" and "上昇" in trend_values
                    alignment_text = f"1h:{trend_1h} / 4h:{trend_4h} / 日足:{trend_d}"

                    # 安全性を維持した「押し目監視」分類。買いシグナルではなく、監視リスト用。
                    # 日足・4時間足が上昇、1時間足が上昇/中立で、価格が1h EMA20付近にある場合のみ。
                    pullback_watch = False
                    pullback_rebound_confirmed = False
                    pullback_detail = "条件外"
                    rebound_detail = "反発条件を未確認"
                    try:
                        close_series_1h = clean_series(scan_1h["Close"])
                        open_series_1h = clean_series(scan_1h["Open"])
                        low_series_1h = clean_series(scan_1h["Low"])
                        close_1h = float(close_series_1h.iloc[-1])
                        prev_close_1h = float(close_series_1h.iloc[-2])
                        open_1h = float(open_series_1h.iloc[-1])
                        low_1h = float(low_series_1h.iloc[-1])
                        ema20_1h = float(clean_series(scan_1h["EMA_20"]).iloc[-1])
                        atr_1h = float(clean_series(scan_1h["ATR"]).iloc[-1])
                        rsi_series_1h = clean_series(scan_1h["RSI"]) if "RSI" in scan_1h else pd.Series(dtype=float)
                        rsi_now = float(rsi_series_1h.iloc[-1]) if len(rsi_series_1h) >= 2 else np.nan
                        rsi_prev = float(rsi_series_1h.iloc[-2]) if len(rsi_series_1h) >= 2 else np.nan
                        if np.isfinite([close_1h, prev_close_1h, open_1h, low_1h, ema20_1h, atr_1h]).all() and atr_1h > 0:
                            lower_zone = ema20_1h - 0.75 * atr_1h
                            upper_zone = ema20_1h + 0.25 * atr_1h
                            in_pullback_zone = lower_zone <= close_1h <= upper_zone
                            # The candle's low must have reached the EMA20 neighborhood; a green
                            # candle alone is not enough to label a pullback rebound.
                            touched_ema_zone = low_1h <= ema20_1h + 0.10 * atr_1h and low_1h >= ema20_1h - 0.90 * atr_1h
                            bullish_background = trend_d == "上昇" and trend_4h == "上昇" and trend_1h in ("上昇", "中立")
                            green_candle = close_1h > open_1h
                            close_recovered = close_1h > prev_close_1h
                            rsi_rising = np.isfinite([rsi_now, rsi_prev]).all() and rsi_now > rsi_prev
                            pullback_rebound_confirmed = bool(
                                bullish_background and in_pullback_zone and touched_ema_zone
                                and green_candle and close_recovered and rsi_rising
                            )
                            pullback_watch = bullish_background and in_pullback_zone and ai_direction == "様子見"
                            if bullish_background and in_pullback_zone:
                                pullback_detail = "1h EMA20付近"
                                missing = []
                                if not touched_ema_zone: missing.append("安値のEMA20接近なし")
                                if not green_candle: missing.append("陽線未確認")
                                if not close_recovered: missing.append("前足終値を上回らず")
                                if not rsi_rising: missing.append("RSI上向き未確認")
                                if pullback_rebound_confirmed:
                                    rebound_detail = "反発兆候あり（EMA20接近・陽線・終値上昇・RSI上向き）"
                                else:
                                    rebound_detail = "反発確認待ち：" + "、".join(missing)
                            elif bullish_background:
                                pullback_detail = "上昇基調だがEMA20付近ではない"
                                rebound_detail = "価格が押し目ゾーン外"
                            else:
                                pullback_detail = "日足・4h・1hの上昇条件未達"
                                rebound_detail = "上位足の上昇条件未達"
                    except (KeyError, IndexError, TypeError, ValueError):
                        pullback_detail = "押し目条件を計算できません"
                        rebound_detail = "反発条件を計算できません"

                    if scan_is_stale:
                        decision, reason = "判定保留", f"データが古い可能性（約{scan_age_min/60:.1f}時間経過）"
                    elif not quality_ok:
                        decision, reason = "見送り", "時系列検証の品質条件を満たさない"
                    elif not enough_movement:
                        decision, reason = "見送り", "ATRに対して推定スプレッド負担が大きい可能性"
                    elif ai_direction == "様子見" and pullback_watch and pullback_rebound_confirmed:
                        decision, reason = "反発兆候あり", "押し目ゾーンで陽線・前足終値超え・RSI上向きを確認。ただしAI方向は様子見のため買いシグナルではなく、次回再判定待ち"
                    elif ai_direction == "様子見" and pullback_watch:
                        decision, reason = "押し目監視", "日足・4hが上昇、1h価格がEMA20付近。反発条件は未達または未確認。買い確定ではありません"
                    elif ai_direction == "様子見":
                        decision, reason = "様子見", f"明確なAI方向シグナルなし（押し目条件：{pullback_detail}）"
                    elif opposing:
                        decision, reason = "方向不一致・見送り", "1h・4h・日足のいずれかがAI方向と逆"
                    elif matching < 2:
                        decision, reason = "様子見", "時間足の方向一致が十分でない"
                    else:
                        decision, reason = "候補あり", f"AI方向と複数時間足が一致（{matching}/3）"
                    # ATR-based indicative levels only; never presented as executable broker prices.
                    stop_distance = max(atr_scan * 1.5, pip_scan * spread_scan * 3)
                    rr = float(st.session_state.get("rr_ratio", 1.5))
                    if ai_direction == "買い":
                        stop_price, target_price = last_price_scan - stop_distance, last_price_scan + stop_distance * rr
                    elif ai_direction == "売り":
                        stop_price, target_price = last_price_scan + stop_distance, last_price_scan - stop_distance * rr
                    else:
                        stop_price = target_price = np.nan
                    fmt_price = lambda v: f"{v:.3f}" if "JPY" in pair_ticker else f"{v:.5f}"
                    scan_results.append({
                        "通貨ペア": pair_label, "順位": "—", "判定": decision, "AI方向": ai_direction,
                        "時間足一致": alignment_text, "一致数": matching,
                        "押し目状態": pullback_detail, "反発確認": rebound_detail,
                        "買い確率": f"{probs.get('buy', 0)}%", "売り確率": f"{probs.get('sell', 0)}%",
                        "価格": fmt_price(last_price_scan), "損切り目安": fmt_price(stop_price) if np.isfinite(stop_price) else "—",
                        "利確目安": fmt_price(target_price) if np.isfinite(target_price) else "—",
                        "確定足時刻": safe_to_tokyo_tz(scan_data).index[-1].strftime("%Y-%m-%d %H:%M"),
                        "データ経過時間": f"{scan_age_min/60:.1f}時間", "ATR(pips)": round(atr_scan / pip_scan, 1),
                        "スプレッド目安(pips)": spread_scan,
                        "検証一致率": f"{metrics.get('balanced_accuracy', 0):.1f}%" if metrics else "不足",
                        "検証件数": int(metrics.get("n_test", 0) or 0), "理由": reason,
                        "判定詳細": f"時間足={alignment_text}。AI={ai_direction}。一致={matching}/3。データ鮮度={'要確認' if scan_is_stale else 'OK'}。検証品質={'OK' if quality_ok else '未達'}。ATR/スプレッド条件={'OK' if enough_movement else '未達'}。押し目={pullback_detail}。反発={rebound_detail}。",
                    })
            except Exception as exc:
                scan_results.append({"通貨ペア": pair_label, "順位": "—", "判定": "確認失敗", "AI方向": "—", "時間足一致": "確認不可", "買い確率": "—", "売り確率": "—", "価格": "—", "損切り目安": "—", "利確目安": "—", "スプレッド目安(pips)": DEFAULT_SPREAD_PIPS.get(pair_ticker, 0.5), "理由": f"分析エラー: {type(exc).__name__}"})
            progress.progress(idx / len(pair_items), text=f"確認中: {pair_label} ({idx}/{len(pair_items)})")
        progress.empty()
        # Rank candidates first; all other states remain visible below them.
        priority = {"候補あり": 0, "反発兆候あり": 1, "押し目監視": 2, "様子見": 3, "方向不一致・見送り": 4, "見送り": 5, "判定保留": 6, "確認失敗": 7}
        scan_results.sort(key=lambda r: (priority.get(r.get("判定"), 9), -int(r.get("一致数", 0) or 0), -float(str(r.get("買い確率", "0")).replace("%", "0") or 0) if str(r.get("買い確率", "—")).replace("%", "").isdigit() else 0))
        for rank, row in enumerate(scan_results, start=1): row["順位"] = rank
        scan_now = datetime.now(JST)
        scan_time_text = scan_now.strftime("%Y-%m-%d %H:%M:%S JST")
        st.session_state["pair_scan_results"] = {"time": scan_time_text, "timeframe": scan_tf_label, "interval_minutes": int(scan_interval_minutes), "rows": scan_results}
        st.session_state["pair_scan_last_epoch"] = time.time()

        # Candidate outcome tracking: evaluate the price after 6h and 24h on later scans.
        # This is directional follow-up only, not actual trade P/L or a guarantee of fill prices.
        try:
            outcome_rows = []
            if os.path.exists(PAIR_SCAN_OUTCOME_FILE):
                with open(PAIR_SCAN_OUTCOME_FILE, "r", encoding="utf-8") as of:
                    loaded_outcomes = json.load(of)
                    if isinstance(loaded_outcomes, list): outcome_rows = loaded_outcomes
            current_prices = {}
            for current_row in scan_results:
                label = current_row.get("通貨ペア")
                ticker_for_row = PAIRS.get(label)
                raw_price = str(current_row.get("価格", "")).replace(",", "")
                try: current_price = float(raw_price)
                except (TypeError, ValueError): current_price = np.nan
                if ticker_for_row and np.isfinite(current_price) and current_price > 0:
                    current_prices[label] = {"price": current_price, "ticker": ticker_for_row}

            for event in outcome_rows:
                if event.get("状態") != "計測中": continue
                try: event_time = pd.Timestamp(event.get("記録時刻"))
                except (TypeError, ValueError): continue
                if event_time.tzinfo is None: event_time = event_time.tz_localize(JST)
                else: event_time = event_time.tz_convert(JST)
                elapsed_hours = (scan_now - event_time.to_pydatetime()).total_seconds() / 3600
                live = current_prices.get(event.get("通貨ペア"))
                if not live: continue
                start_price = float(event.get("候補価格", 0) or 0)
                pip_size = float(event.get("pip単位", 0) or 0)
                if start_price <= 0 or pip_size <= 0: continue
                for horizon in (6, 24):
                    key = f"{horizon}時間後"
                    if elapsed_hours >= horizon and not event.get(key):
                        raw_move_pips = (live["price"] - start_price) / pip_size
                        direction = event.get("AI方向")
                        favorable_pips = raw_move_pips if direction == "買い" else -raw_move_pips
                        if favorable_pips > 0.1: outcome_label = "方向一致"
                        elif favorable_pips < -0.1: outcome_label = "逆方向"
                        else: outcome_label = "ほぼ横ばい"
                        event[key] = {"評価時刻": scan_time_text, "評価価格": live["price"], "方向一致pips": round(favorable_pips, 1), "結果": outcome_label}
                if event.get("6時間後") and event.get("24時間後"):
                    event["状態"] = "評価完了"

            existing_keys = {str(e.get("イベントID")) for e in outcome_rows}
            for row in scan_results:
                if row.get("判定") != "候補あり": continue
                label = row.get("通貨ペア", "")
                direction = row.get("AI方向", "")
                if direction not in ("買い", "売り") or label not in current_prices: continue
                candle_time = str(row.get("確定足時刻", scan_time_text))
                event_id = f"{label}|{scan_tf_label}|{direction}|{candle_time}"
                if event_id in existing_keys: continue
                ticker_for_row = current_prices[label]["ticker"]
                pip_size = 0.01 if "JPY" in ticker_for_row else 0.0001
                outcome_rows.append({
                    "イベントID": event_id, "記録時刻": scan_now.isoformat(), "通貨ペア": label,
                    "時間足": scan_tf_label, "判定": row.get("判定"), "AI方向": direction,
                    "候補価格": current_prices[label]["price"], "pip単位": pip_size,
                    "時間足一致": row.get("時間足一致", "—"), "理由": row.get("理由", "—"),
                    "状態": "計測中", "6時間後": None, "24時間後": None,
                })
                existing_keys.add(event_id)
            with open(PAIR_SCAN_OUTCOME_FILE, "w", encoding="utf-8") as of:
                json.dump(outcome_rows[-2000:], of, ensure_ascii=False, indent=2, default=_json_default)
        except (OSError, ValueError, TypeError, KeyError) as outcome_exc:
            st.warning(f"候補後の自動採点を保存できませんでした（{type(outcome_exc).__name__}）。")

        # Persist a compact scan history for later review; keep last 500 snapshots.
        try:
            history_path = "pair_scan_history.json"
            old_history = []
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as hf:
                    loaded_history = json.load(hf)
                    if isinstance(loaded_history, list): old_history = loaded_history
            old_history.append({"time": scan_time_text, "timeframe": scan_tf_label, "rows": scan_results})
            with open(history_path, "w", encoding="utf-8") as hf:
                json.dump(old_history[-500:], hf, ensure_ascii=False, indent=2, default=_json_default)
        except (OSError, ValueError, TypeError):
            st.warning("スキャン履歴を保存できませんでした。保存先フォルダの書き込み権限を確認してください。")
        webhook_url = st.session_state.get("webhook_url", "")
        if webhook_url:
            alert_state = st.session_state.setdefault("pair_candidate_alert_keys", [])
            new_candidate_lines = []
            for row in scan_results:
                if row.get("判定") != "候補あり": continue
                label = row.get("通貨ペア", "不明")
                pair_ticker = PAIRS.get(label)
                if not pair_ticker: continue
                candidate_key = f"{pair_ticker}|{scan_config['interval']}|{row.get('AI方向')}|{row.get('確定足時刻', scan_time_text)}"
                if candidate_key not in alert_state:
                    new_candidate_lines.append(f"{label}: {row.get('AI方向')} / 価格参考 {row.get('価格')} / {row.get('理由')}")
                    alert_state.append(candidate_key)
            st.session_state["pair_candidate_alert_keys"] = alert_state[-100:]
            if new_candidate_lines:
                alert_message = "中期FX候補スキャン（参考情報・自動発注なし）\n" + "\n".join(new_candidate_lines)
                if send_webhook_alert(webhook_url, alert_message): st.toast("新しい候補の通知を送信しました", icon="🔔")
        if original_eval_metrics is not None: st.session_state["model_eval_metrics"] = original_eval_metrics
        else: st.session_state.pop("model_eval_metrics", None)

    saved_scan = st.session_state.get("pair_scan_results", {})
    if saved_scan.get("rows") and saved_scan.get("timeframe") == scan_tf_label:
        scan_df = pd.DataFrame(saved_scan["rows"])
        available_count = int((scan_df["判定"] == "候補あり").sum()) if "判定" in scan_df else 0
        sca, scb, scc = st.columns(3)
        sca.metric("トレード候補", f"{available_count} ペア")
        scb.metric("チェック対象", f"{len(scan_df)} ペア")
        scc.metric("最終スキャン", saved_scan.get("time", "—"))
        st.dataframe(scan_df, use_container_width=True, hide_index=True)
        st.download_button("📥 候補一覧をCSV保存", data=scan_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"midterm_pair_scan_{scan_config['interval']}.csv", mime="text/csv", use_container_width=False)
        st.caption("損切り・利確はATRから算出した目安で、実際のBid/Ask・スプレッド・約定価格を反映していません。発注前に松井証券FXの価格で再計算してください。候補が0件なら無理に取引せず、見送りを優先してください。")
    elif saved_scan.get("rows"):
        st.info(f"保存済み一覧は {saved_scan.get('timeframe')} です。上のスキャン時間足を合わせると表示できます。")
    else:
        st.info("まだスキャン結果がありません。「今すぐ全通貨ペア確認」を押すか、定期スキャンを有効にしてください。")
    with st.expander("🕘 過去のスキャン履歴", expanded=False):
        try:
            with open("pair_scan_history.json", "r", encoding="utf-8") as hf:
                scan_history_rows = json.load(hf)
            if isinstance(scan_history_rows, list) and scan_history_rows:
                history_view = []
                for snapshot in scan_history_rows[-20:][::-1]:
                    for item in snapshot.get("rows", []):
                        history_view.append({"スキャン時刻": snapshot.get("time", "—"), "時間足": snapshot.get("timeframe", "—"), "通貨ペア": item.get("通貨ペア", "—"), "判定": item.get("判定", "—"), "AI方向": item.get("AI方向", "—"), "参考価格": item.get("価格", "—")})
                if history_view:
                    st.dataframe(pd.DataFrame(history_view), use_container_width=True, hide_index=True)
                    st.caption("履歴は過去のスキャン時点の記録です。候補後の値動きや実際の取引損益を自動採点した結果ではありません。")
            else:
                st.info("まだ履歴がありません。スキャンを実行すると保存されます。")
        except (OSError, ValueError, TypeError):
            st.info("まだ履歴がありません。スキャンを実行すると保存されます。")
    with st.expander("📊 候補後の自動採点（6時間後・24時間後）", expanded=False):
        st.caption("『候補あり』になった時点の価格と、その後のスキャン価格を比較します。実際の約定損益、最大逆行幅、スプレッド・スワップは含みません。アプリが起動していない間は評価が遅れる場合があります。")
        try:
            with open(PAIR_SCAN_OUTCOME_FILE, "r", encoding="utf-8") as of:
                outcome_view_rows = json.load(of)
            if isinstance(outcome_view_rows, list) and outcome_view_rows:
                display_rows = []
                for event in outcome_view_rows[-100:][::-1]:
                    row_view = {"記録時刻": event.get("記録時刻", "—"), "通貨ペア": event.get("通貨ペア", "—"), "時間足": event.get("時間足", "—"), "AI方向": event.get("AI方向", "—"), "候補価格": event.get("候補価格", "—"), "状態": event.get("状態", "—")}
                    for horizon in (6, 24):
                        outcome = event.get(f"{horizon}時間後") or {}
                        row_view[f"{horizon}時間後結果"] = outcome.get("結果", "計測待ち")
                        row_view[f"{horizon}時間後方向一致pips"] = outcome.get("方向一致pips", "—")
                    display_rows.append(row_view)
                outcome_df = pd.DataFrame(display_rows)
                st.dataframe(outcome_df, use_container_width=True, hide_index=True)
                completed_6h = [e for e in outcome_view_rows if isinstance(e.get("6時間後"), dict)]
                completed_24h = [e for e in outcome_view_rows if isinstance(e.get("24時間後"), dict)]
                if completed_6h:
                    hit6 = sum(1 for e in completed_6h if e["6時間後"].get("結果") == "方向一致") / len(completed_6h) * 100
                    st.metric("6時間後の方向一致率（参考）", f"{hit6:.1f}%", help="評価済み候補のうち、候補方向に価格が動いていた割合。将来の勝率を示すものではありません。")
                if completed_24h:
                    hit24 = sum(1 for e in completed_24h if e["24時間後"].get("結果") == "方向一致") / len(completed_24h) * 100
                    st.metric("24時間後の方向一致率（参考）", f"{hit24:.1f}%", help="評価済み候補のうち、候補方向に価格が動いていた割合。将来の勝率を示すものではありません。")
                st.download_button("📥 候補後採点をCSV保存", data=outcome_df.to_csv(index=False).encode("utf-8-sig"), file_name="pair_scan_outcomes.csv", mime="text/csv")
            else:
                st.info("採点データはまだありません。『候補あり』が出ると記録が始まり、6時間後・24時間後のスキャンで採点します。")
        except (OSError, ValueError, TypeError):
            st.info("採点データはまだありません。スキャンを実行すると記録されます。")
    st.caption("自動スキャンはアプリ画面が開いている間だけ動作します。Yahoo Financeのデータ取得頻度・遅延は保証されません。通知は新しい候補が検出された場合のみで、自動発注は行いません。")

    # st.tabs() はボタン操作などの再実行後に先頭タブへ戻ることがあるため、
    # session_state に選択中の画面を保存するラジオ式ナビゲーションに変更。
    APP_SECTIONS = [
        "📈 チャート",
        "⚡ 単発注文の計算",
        "📰 ニュース・経済指標",
        "📋 リピート注文・リスク",
        "🤖 AI検証・履歴",
    ]
    # Compact navigation is easier to use on iPad/phone than a row of long, clipped labels.
    active_section = st.radio(
        "表示する画面",
        APP_SECTIONS,
        key="active_app_section_v2",
        label_visibility="collapsed",
        horizontal=True,
    )

    if active_section == APP_SECTIONS[0]:
        st.markdown("##### 📈 メインチャート")
        st.caption("通常は固定表示です。表示が動いて見失うのを防ぎ、表示本数や指標は下の操作欄から変更できます。")
        chart_interactive = st.toggle(
            "チャートの拡大・移動を有効にする",
            value=False,
            key="chart_interactive_toggle",
            help="通常はOFFがおすすめです。ONにするとチャート上で拡大・移動できます。"
        )
        qc1, qc2, qc3, qc4, qc5 = st.columns([1.2, 1.2, 1.2, 2.0, 2.4])
        with qc1: btn_50 = st.button("直近 50本", use_container_width=True)
        with qc2: btn_120 = st.button("標準 120本", use_container_width=True)
        with qc3: btn_250 = st.button("長期 250本", use_container_width=True)
        if "chart_bars" not in st.session_state: st.session_state["chart_bars"] = 120
        if btn_50: st.session_state["chart_bars"] = 50
        if btn_120: st.session_state["chart_bars"] = 120
        if btn_250: st.session_state["chart_bars"] = 250
        with qc4:
            bars_count = st.slider("表示本数調整", 30, 300, value=int(st.session_state["chart_bars"]), step=10, key="chart_bars_slider", label_visibility="collapsed")
            st.session_state["chart_bars"] = bars_count
        with qc5:
            st.markdown("**サブ指標**")
            sub_indicator = st.radio(
                "サブ指標", ["RSI (14)", "MACD", "なし"],
                index=0, key="sub_indicator_select",
                horizontal=True, label_visibility="collapsed",
            )
        toggle_c1, toggle_c2, toggle_c3, toggle_c4 = st.columns(4)
        with toggle_c1: show_ma = st.checkbox("移動平均線(SMA/EMA)", value=True, key="show_ma_check")
        with toggle_c2: show_bb = st.checkbox("ボリンジャーバンド", value=True, key="show_bb_check")
        with toggle_c3: show_repeat_lines = st.checkbox("リピートレンジ表示", value=("4時間足" in tf_label), key="show_repeat_lines_check")
        with toggle_c4: show_ai_markers = st.checkbox("AI売買シグナル表示", value=True, key="show_ai_markers_check")
        df_chart = safe_to_tokyo_tz(data.tail(bars_count)).copy()
        df_chart = df_chart.loc[~df_chart.index.duplicated(keep="last")]
        x_index = df_chart.index
        has_sub = sub_indicator != "なし"
        fig = make_subplots(rows=2 if has_sub else 1, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25] if has_sub else [1.0], vertical_spacing=0.06)
        fig.add_trace(go.Candlestick(x=x_index, open=clean_series(df_chart["Open"]), high=clean_series(df_chart["High"]),
            low=clean_series(df_chart["Low"]), close=clean_series(df_chart["Close"]),
            increasing_line_color="#22c55e", increasing_fillcolor="#22c55e",
            decreasing_line_color="#ef4444", decreasing_fillcolor="#ef4444", name="価格"), row=1, col=1)
        if show_ma:
            for col, color, name, width in [("SMA_20","#f59e0b","SMA20",1.5),("EMA_200","#38bdf8","EMA200",2.0)]:
                if col in df_chart:
                    fig.add_trace(go.Scatter(x=x_index,y=clean_series(df_chart[col]),line=dict(color=color,width=width),name=name),row=1,col=1)
        if show_bb and "Upper_Band" in df_chart and "Lower_Band" in df_chart:
            fig.add_trace(go.Scatter(x=x_index,y=clean_series(df_chart["Upper_Band"]),line=dict(color="rgba(148,163,184,0.4)",width=1,dash="dot"),name="BB +2σ"),row=1,col=1)
            fig.add_trace(go.Scatter(x=x_index,y=clean_series(df_chart["Lower_Band"]),line=dict(color="rgba(148,163,184,0.4)",width=1,dash="dot"),fill="tonexty",fillcolor="rgba(148,163,184,0.05)",name="BB -2σ"),row=1,col=1)
        if show_ai_markers and ai_signal_history is not None and not ai_signal_history.empty:
            common_idx = df_chart.index.intersection(ai_signal_history.dropna().index)
            if len(common_idx):
                preds = ai_signal_history.loc[common_idx]
                buy_idx = preds.index[preds == 1]
                sell_idx = preds.index[preds == -1]
                if len(buy_idx):
                    fig.add_trace(go.Scatter(x=buy_idx,y=df_chart.loc[buy_idx,"Low"]-latest_atr*.3,mode="markers",marker=dict(symbol="triangle-up",size=11,color="#22c55e"),name="検証区間AI買い"),row=1,col=1)
                if len(sell_idx):
                    fig.add_trace(go.Scatter(x=sell_idx,y=df_chart.loc[sell_idx,"High"]+latest_atr*.3,mode="markers",marker=dict(symbol="triangle-down",size=11,color="#ef4444"),name="検証区間AI売り"),row=1,col=1)
        if show_repeat_lines:
            fig.add_hrect(y0=user_lower,y1=user_upper,fillcolor="rgba(56,189,248,0.06)",line_width=0,row=1,col=1)
            for y, color, dash, label in [(sell_stop_loss,"#dc2626","dashdot","売SL"),(user_upper,"#f87171","dash","上限"),(user_half,"#c084fc","dot","中央"),(user_lower,"#4ade80","dash","下限"),(buy_stop_loss,"#16a34a","dashdot","買SL")]:
                if np.isfinite(y): fig.add_hline(y=y,line_dash=dash,line_color=color,annotation_text=label,row=1,col=1)
        if has_sub and sub_indicator == "RSI (14)" and "RSI" in df_chart:
            fig.add_trace(go.Scatter(x=x_index,y=clean_series(df_chart["RSI"]),line=dict(color="#a855f7",width=1.8),name="RSI"),row=2,col=1)
            fig.add_hline(y=70,line_dash="dot",line_color="#ef4444",row=2,col=1)
            fig.add_hline(y=30,line_dash="dot",line_color="#22c55e",row=2,col=1)
            fig.update_yaxes(range=[0,100],side="right",row=2,col=1)
        elif has_sub and sub_indicator == "MACD" and "MACD" in df_chart:
            fig.add_trace(go.Scatter(x=x_index,y=clean_series(df_chart["MACD"]),line=dict(color="#38bdf8"),name="MACD"),row=2,col=1)
            fig.add_trace(go.Scatter(x=x_index,y=clean_series(df_chart["MACD_Signal"]),line=dict(color="#f59e0b"),name="Signal"),row=2,col=1)
            fig.add_trace(go.Bar(x=x_index,y=clean_series(df_chart["MACD_Hist"]),name="Hist"),row=2,col=1)
            fig.update_yaxes(side="right",row=2,col=1)
        chart_high, chart_low = float(df_chart["High"].max()), float(df_chart["Low"].min())
        span = max(chart_high-chart_low, chart_high*.002)
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=580,
            margin=dict(l=10, r=70, t=25, b=10),
            template="plotly_dark",
            hovermode="x unified" if chart_interactive else False,
            dragmode="pan" if chart_interactive else False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            uirevision=f"{ticker}-{tf_config['interval']}-{bars_count}-{sub_indicator}",
        )
        fig.update_xaxes(
            type="date",
            rangebreaks=[dict(bounds=["sat", "mon"])] if tf_config["interval"] in ("5m", "15m", "1h") else [],
            showspikes=chart_interactive,
            spikemode="across" if chart_interactive else None,
            fixedrange=not chart_interactive,
        )
        fig.update_yaxes(
            range=[chart_low-span*.05, chart_high+span*.05],
            side="right",
            tickformat=".3f" if is_jpy else ".5f",
            fixedrange=not chart_interactive,
            row=1,
            col=1,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "scrollZoom": bool(chart_interactive),
                "displayModeBar": bool(chart_interactive),
                "displaylogo": False,
                "doubleClick": "reset" if chart_interactive else False,
                "responsive": True,
            },
            key="main_chart_plot",
        )
        if chart_interactive:
            st.caption("操作ヒント：ドラッグで移動、ホイール／ピンチで拡大、ダブルタップ（ダブルクリック）でリセットできます。操作後に見失った場合は、このスイッチをOFFにして固定表示へ戻してください。")
        else:
            st.caption("固定表示中：チャートを触っても拡大・移動しません。見たい範囲は『直近50本／標準120本／長期250本』または表示本数スライダーで切り替えてください。")

    elif active_section == APP_SECTIONS[1]:
        st.markdown("##### ⚡ 単発トレード（スキャル・デイトレ）AI発注アシスタント")
        st.caption("AIの分析結果から注文条件の参考値を計算します。発注は行いません。")

        # Signal transition log: records distinct signal states for the selected pair/timeframe/bar.
        # This is an in-session history only; it does not know whether an order actually filled.
        signal_history_key = "signal_transition_history"
        if signal_history_key not in st.session_state:
            st.session_state[signal_history_key] = []
        signal_now = "BUY" if status.startswith("BUY") else "SELL" if status.startswith("SELL") else "WAIT"
        signal_bar_ts = _normalise_timestamp(data.index[-1])
        signal_log = st.session_state[signal_history_key]
        relevant_log = [
            x for x in signal_log
            if x.get("ticker") == ticker and x.get("interval") == tf_config["interval"]
        ]
        previous_signal = relevant_log[-1].get("signal") if relevant_log else None
        already_logged = any(
            x.get("ticker") == ticker
            and x.get("interval") == tf_config["interval"]
            and x.get("bar_timestamp") == signal_bar_ts
            and x.get("signal") == signal_now
            for x in signal_log
        )
        if not already_logged:
            signal_log.append({
                "recorded_at": datetime.now(JST).isoformat(timespec="seconds"),
                "ticker": ticker,
                "interval": tf_config["interval"],
                "bar_timestamp": signal_bar_ts,
                "previous_signal": previous_signal or "（初回）",
                "signal": signal_now,
                "close": float(latest_price),
                "confidence_pct": float(conf),
            })
            st.session_state[signal_history_key] = signal_log[-100:]
            signal_log = st.session_state[signal_history_key]

        st.caption("実際の注文・建玉はこのアプリでは取得できません。松井証券FXの注文照会で確認してください。")

        current_usdjpy_rate = get_usdjpy_rate()
        if not is_jpy and current_usdjpy_rate is None:
            st.error("為替換算用のUSD/JPYレートが取得できません。換算が必要なため数量計算を停止します。")
            usd_rate = None
        else:
            usd_rate = latest_price if ticker == "USDJPY=X" else current_usdjpy_rate
        pip_val_per_wan = 100.0 if is_jpy else (0.0001 * usd_rate * 10000.0 if usd_rate else 0.0)
        account_bal = float(st.session_state["account_balance"])
        risk_pct = float(st.session_state["risk_percent"])
        max_risk_yen = account_bal * risk_pct / 100.0
        ai_direction = "BUY (買い)" if status.startswith("BUY") else "SELL (売り)" if status.startswith("SELL") else "WAIT (様子見)"
        ai_sl_pips = max(1.0, round((latest_atr / pip_unit) * 1.5, 1))
        ai_rr_ratio = float(st.session_state["rr_ratio"])
        ai_tp_pips = round(ai_sl_pips * ai_rr_ratio, 1)

        # Reference-price calculator only; this does not reproduce Matsui order types.
        st.markdown("#### 🧾 単発トレードの参考値")
        st.caption("AIが方向・損切り幅・利確幅・数量の参考値を計算します。実際の注文方式や条件を再現・選択する機能ではありません。松井証券FXの注文画面で注文方式を選んでください。口座接続・自動発注は行いません。")
        entry_style = st.radio(
            "参考価格の計算パターン（松井証券FXの注文方式ではありません）",
            ["成行（現在レートを基準）", "押し目・戻りの参考価格", "ブレイク方向の参考価格"],
            horizontal=True,
            help="ここで選ぶのは参考価格の計算方法だけです。松井証券FXの実際の注文方式・有効期限・決済条件は注文画面で確認してください。",
            key="ifdoco_entry_style",
        )
        check_live_quote = st.checkbox(
            "松井証券FXの現在レートを入力して、参考価格との差を確認する",
            value=entry_style.startswith("成行"),
            key="ifdoco_check_live_quote",
        )
        matsui_quote = None
        if check_live_quote:
            matsui_quote = st.number_input(
                "松井証券FXの現在レート（画面に表示されている価格）",
                min_value=0.0,
                value=0.0,
                step=0.01 if is_jpy else 0.0001,
                format=price_fmt,
                help="初期値は0です。松井証券FXの画面に表示されている現在レートを確認して入力してください。確定足終値を現在レートとして流用しないでください。",
                key="ifdoco_matsui_quote",
            )
            if matsui_quote <= 0:
                st.warning("現在レートが無効です。価格位置の確認は行えません。")
                matsui_quote = None

        ai_is_buy = ai_direction.startswith("BUY")
        ai_sl_pips = max(1.0, round((latest_atr / pip_unit) * 1.5, 1))
        ai_rr_ratio = float(st.session_state["rr_ratio"])
        ai_tp_pips = round(ai_sl_pips * ai_rr_ratio, 1)

        # 1時間足の確定終値とリアルタイム価格の乖離を明示し、
        # 現在レートを入力した場合はその価格を基準に注文候補を再計算する。
        quote_gap_pips = None
        quote_gap_threshold_pips = max(5.0, (latest_atr / pip_unit) * 0.25)
        quote_gap_large = False
        if matsui_quote is not None:
            quote_gap_pips = abs(matsui_quote - latest_price) / pip_unit
            quote_gap_large = quote_gap_pips > quote_gap_threshold_pips
            entry_reference_price = float(matsui_quote)
            entry_reference_label = "入力した松井証券FXの現在レート"
        else:
            # Do not imply that a candle close is a live executable quote.
            entry_reference_price = float(latest_price)
            entry_reference_label = "最終確定足の終値（参考値のみ・現在レートではありません）"

        gap_acknowledged = False
        if matsui_quote is not None:
            st.markdown("##### 現在レートと確定足の差")
            gap_col1, gap_col2, gap_col3 = st.columns(3)
            gap_col1.metric("確定足終値", price_fmt % latest_price)
            gap_col2.metric("入力した現在レート", price_fmt % matsui_quote)
            gap_col3.metric("価格差", f"{quote_gap_pips:.1f} pips", f"しきい値 {quote_gap_threshold_pips:.1f} pips" if quote_gap_large else "通常範囲")
            if quote_gap_large:
                st.warning(
                    f"確定足と現在レートの差が大きめです（{quote_gap_pips:.1f} pips）。"
                    "古い確定足の価格をそのまま注文に使わないでください。以下の候補は入力した現在レートを基準に再計算します。"
                )
                gap_acknowledged = st.checkbox(
                    "松井証券FXの現在レートを再確認し、このレートを基準にした候補値であることを確認しました",
                    value=False,
                    key="ifdoco_gap_acknowledged",
                )
            else:
                st.info("現在レートを基準に注文候補を計算します。Bid/Askの違いと注文画面の条件は別途確認してください。")
        else:
            st.warning(
                f"現在レートが未入力です。確定足終値 {price_fmt % latest_price} はリアルタイム価格ではありません。"
                "注文候補は参考表示に限り、松井証券FXの現在レートを入力するまで転記用の価格として使わないでください。"
            )
            if entry_style.startswith("成行"):
                st.error("現在レートを入力するまで参考値を確定扱いにしないでください。")

        entry_offset_atr = max(float(latest_atr) * (0.5 if entry_style.startswith("押し目") else 0.2), pip_unit)
        if entry_style.startswith("成行"):
            ai_entry = float(matsui_quote) if matsui_quote is not None else float(latest_price)
            entry_order_type = "成行"
        elif ai_is_buy:
            ai_entry = entry_reference_price - entry_offset_atr if entry_style.startswith("押し目") else entry_reference_price + entry_offset_atr
            entry_order_type = "指値" if entry_style.startswith("押し目") else "逆指値"
        else:
            ai_entry = entry_reference_price + entry_offset_atr if entry_style.startswith("押し目") else entry_reference_price - entry_offset_atr
            entry_order_type = "指値" if entry_style.startswith("押し目") else "逆指値"

        st.markdown("##### 🧭 AI候補価格の計算根拠")
        st.write(f"- 計算の基準価格：**{entry_reference_label}**（{price_fmt % entry_reference_price}）")
        st.write(f"- 計算方法（到達確率の予測ではありません）：基準価格 {'−' if entry_style.startswith('押し目') and ai_is_buy else '+' if entry_style.startswith('押し目') else '+' if ai_is_buy else '−'} ATRに基づく幅（{entry_offset_atr / pip_unit:.1f} pips）")
        if matsui_quote is not None:
            distance_to_candidate = abs(float(ai_entry) - float(matsui_quote)) / pip_unit
            st.metric("現在レートから候補価格まで", f"{distance_to_candidate:.1f} pips")
            if entry_style.startswith("成行"):
                st.caption("成行は価格を指定しない注文です。表示価格は参考値で、急変時はスリッページが発生する可能性があります。")
            elif entry_style.startswith("押し目") and ai_is_buy:
                st.caption("買いの押し目指値は現在レートより下に置く候補です。価格がそこまで下がる保証はなく、届かなければ未約定になります。")
            elif ai_is_buy:
                st.caption("買いのブレイク逆指値は現在レートより上に置く候補です。上昇を追う方式なので、約定価格や損失リスクを注文前に確認してください。")
        else:
            st.warning("松井証券FXの現在レートが未入力のため、この候補価格は確定足終値を基準にした仮計算です。実際の注文価格として使わず、現在レートを入力して再計算してください。")

        # Persistent validation of whether a proposed price was touched in the next 8 completed bars.
        # Requires a manually confirmed Matsui quote; candle-close-only estimates are not tracked by default.
        st.markdown("##### 🧪 候補価格の到達検証（履歴を保存）")
        st.caption("候補価格が、その後8本の確定足の高値・安値の範囲に入ったかを記録します。到達は約定を意味しません。Bid/Ask、スプレッド、配信元の差、指値の順番は再現できません。")
        if matsui_quote is None:
            st.info("松井証券FXの現在レートを入力すると、現在レート基準の候補を検証履歴に保存できます。")
        elif status.startswith("WAIT") or status.startswith("様子見"):
            st.warning("AI判定が様子見のため、候補価格の検証保存はできません。")
        else:
            if st.button("📌 この候補価格を検証履歴に保存", key="save_entry_candidate", use_container_width=True):
                track_items = load_entry_tracking()
                bar_ts = _normalise_timestamp(data.index[-1])
                duplicate = any(x.get("ticker") == ticker and x.get("interval") == tf_config["interval"]
                                and x.get("bar_timestamp") == bar_ts and x.get("entry_style") == entry_style
                                for x in track_items)
                if duplicate:
                    st.warning("同じ通貨ペア・時間足・確定足・エントリー方針の候補は既に記録されています。")
                else:
                    track_items.append({
                        "ticker": ticker, "interval": tf_config["interval"],
                        "bar_timestamp": bar_ts, "saved_at": datetime.now(JST).isoformat(),
                        "signal": ai_direction, "entry_style": entry_style,
                        "reference_label": entry_reference_label,
                        "reference_price": float(entry_reference_price),
                        "candidate_price": float(ai_entry),
                        "side": "BUY" if ai_is_buy else "SELL",
                        "source": "yfinance OHLC; candidate based on manually entered Matsui quote",
                        "status": "PENDING", "horizon_bars": LOOKAHEAD_BARS,
                    })
                    save_entry_tracking(track_items)
                    st.success("検証履歴に保存しました。今後の更新で後続確定足を確認します。")

        quote_position_ok = None
        if matsui_quote is not None and entry_order_type == "成行":
            quote_position_ok = True
        elif matsui_quote is not None and ai_direction.startswith("BUY"):
            quote_position_ok = (ai_entry < matsui_quote) if entry_order_type == "指値" else (ai_entry > matsui_quote)
        elif matsui_quote is not None and ai_direction.startswith("SELL"):
            quote_position_ok = (ai_entry > matsui_quote) if entry_order_type == "指値" else (ai_entry < matsui_quote)
        ai_sl_price = ai_entry - ai_sl_pips*pip_unit if ai_is_buy else ai_entry + ai_sl_pips*pip_unit
        ai_tp_price = ai_entry + ai_tp_pips*pip_unit if ai_is_buy else ai_entry - ai_tp_pips*pip_unit
        ai_loss_per_wan = (ai_sl_pips + spread_pips) * pip_val_per_wan
        raw_ai_wan = max_risk_yen / ai_loss_per_wan if ai_loss_per_wan > 0 and np.isfinite(ai_loss_per_wan) else 0.0
        floored_ai_wan = np.floor(max(0.0, raw_ai_wan) * 100) / 100
        ai_rec_wan = float(floored_ai_wan) if floored_ai_wan >= .01 else 0.0
        # Final guard against rounding / malformed values exceeding the configured risk budget.
        if ai_rec_wan > 0 and ai_loss_per_wan * ai_rec_wan > max_risk_yen + 1e-6:
            ai_rec_wan = max(0.0, np.floor((max_risk_yen / ai_loss_per_wan) * 100) / 100 - 0.01)
        # Refuse a recommendation if the take-profit distance does not exceed
        # the estimated spread; a positive-looking gross target may be uneconomic.
        is_risk_exceeded = ai_rec_wan <= 0 or pip_val_per_wan <= 0 or ai_tp_pips <= spread_pips
        ai_profit_yen = (ai_tp_pips-spread_pips)*pip_val_per_wan*ai_rec_wan
        ai_loss_yen = (ai_sl_pips+spread_pips)*pip_val_per_wan*ai_rec_wan
        if ai_rec_wan > 0 and is_jpy:
            # JPY-quoted pairs: notional is already denominated in JPY.
            req_margin_yen = (ai_entry * ai_rec_wan * 10000) / 25
        elif ai_rec_wan > 0 and usd_rate is not None and np.isfinite(usd_rate) and usd_rate > 0:
            # EUR/USD etc.: convert USD notional to JPY using the latest USD/JPY rate.
            req_margin_yen = (ai_entry * float(usd_rate) * ai_rec_wan * 10000) / 25
        else:
            req_margin_yen = 0.0
        st.markdown("#### 🤖 AI提案トレードプラン")
        quality_metrics = st.session_state.get("model_eval_metrics", {})
        if quality_metrics and not quality_metrics.get("quality_ok", True):
            st.warning(
                "今回の時系列検証ではAIモデルが基準モデルを十分に上回りませんでした。"
                "安全側に倒し、売買方向・推奨数量は見送り扱いにしています。"
            )
        if ai_direction.startswith("WAIT"):
            st.warning("現在はAI判定がWAIT（様子見）です。推奨方向・注文数量を提示しません。")
        elif is_risk_exceeded:
            st.error("安全な最小数量を許容リスク内に収められないか、換算データが不足しています。今回は見送り候補です。")
        else:
            st.info(f"参考シグナル: {status}（モデル出力確信度 {conf:.1f}%、分類一致率 {win_rate:.1f}%）")
        if ai_direction.startswith("WAIT") or is_risk_exceeded:
            st.markdown(f'<div class="param-box"><b>【注文補助】</b><br>売買方向: <code>WAIT（見送り）</code><br>推奨数量: <code>0.00 万通貨</code><br>参考価格: <code>{price_fmt % ai_entry}</code></div>',unsafe_allow_html=True)
        elif matsui_quote is None or (quote_gap_large and not gap_acknowledged):
            st.warning("現在レートの入力、または価格差の確認が完了するまで、転記用の注文プランは確定扱いにしません。下記の数値は参考候補です。")
            st.markdown(f'<div class="param-box"><b>【参考候補・未確認】</b><br>方向: <code>{"買い" if ai_is_buy else "売り"}</code><br>注文種類: <code>{entry_order_type}</code><br>候補価格: <code>{price_fmt % ai_entry}</code><br>利確候補: <code>{price_fmt % ai_tp_price}</code><br>損切候補: <code>{price_fmt % ai_sl_price}</code><br>推奨数量: <code>{ai_rec_wan:.2f} 万通貨</code></div>',unsafe_allow_html=True)
        else:
            if ai_tp_pips <= spread_pips:
                st.warning("利確幅が想定スプレッド以下です。コストを考慮すると不利な可能性があるため、利確幅を見直してください。")
            st.markdown(f"""<div class="param-box"><b>【参考パラメータ】</b>（想定スプレッド {spread_pips:.1f} pips）<br>
            ・通貨ペア: <code>{html.escape(selected_label)}</code> | 方向: <code>{"買い" if ai_is_buy else "売り"}</code><br>
            ・新規価格目安: <code>{price_fmt % ai_entry}</code><br>・利確(TP): <code>{price_fmt % ai_tp_price}</code> (+{ai_tp_pips:.1f} pips / 純益概算 +{int(ai_profit_yen):,}円)<br>
            ・損切(SL): <code>{price_fmt % ai_sl_price}</code> (-{ai_sl_pips:.1f} pips / 損失概算 -{int(ai_loss_yen):,}円)<br>
            ・数量: <code>{ai_rec_wan:.2f} 万通貨</code> ({int(ai_rec_wan*10000):,}通貨) | 必要証拠金概算: <code>{int(req_margin_yen):,}円</code></div>""",unsafe_allow_html=True)
            copy_text = (
                f"通貨ペア: {selected_label}\n方向: {'買い' if ai_is_buy else '売り'}\n"
                f"参考価格パターン: {entry_style}\n参考価格候補: {price_fmt % ai_entry}\n利確(TP): {price_fmt % ai_tp_price} ({ai_tp_pips:.1f} pips)\n"
                f"損切(SL): {price_fmt % ai_sl_price} ({ai_sl_pips:.1f} pips)\n"
                f"数量: {ai_rec_wan:.2f} 万通貨\n必要証拠金概算: {int(req_margin_yen):,}円\n"
                f"想定損失: {int(ai_loss_yen):,}円\n想定スプレッド: {spread_pips:.1f} pips\n"
                "※参考値です。発注前に現在レート・スプレッド・証拠金を確認してください。"
            )
            st.markdown("**コピー用テキスト**（コード枠のコピー機能を使用できます）")
            st.code(copy_text, language="text")

        if not ai_direction.startswith("WAIT") and not is_risk_exceeded and matsui_quote is not None and (not quote_gap_large or gap_acknowledged):
            st.markdown("##### 注文前に確認する参考値")
            st.markdown(
                f"""<div class="param-box">
                <b>AI分析の参考値（注文方式は未選択）</b><br>
                ・方向の参考: <code>{'買い' if ai_is_buy else '売り'}</code><br>
                ・価格候補: <code>{price_fmt % ai_entry}</code><br>
                ・利確幅の参考: <code>{price_fmt % ai_tp_price}</code><br>
                ・損切り幅の参考: <code>{price_fmt % ai_sl_price}</code><br>
                ・数量の参考: <code>{ai_rec_wan:.2f} 万通貨</code><br>
                ・実際の注文方式・決済条件は松井証券FXの画面で選択・確認してください。
                </div>""",
                unsafe_allow_html=True,
            )
            if quote_position_ok:
                st.success(f"入力した現在レート {price_fmt % matsui_quote} と参考価格の位置関係を計算しました。実際の注文方式との適合性は判定していません。")
            else:
                st.warning(f"入力した現在レート {price_fmt % matsui_quote} と参考価格の位置関係に注意してください。実際の注文方式との適合性は判定していません。")
            st.caption("入力したレートは参考値で、Bid/Askの別を自動判別していません。実際の注文方式・有効期限・決済条件は松井証券FXの注文画面で確認してください。")

        st.markdown("##### 📝 トレード結果フィードバック")
        st.caption("結果の記録用です。現在の足と実際の取引条件が一致するとは限らないため、この記録をそのままAIの教師ラベルには使いません。")
        fb_col1, fb_col2 = st.columns(2)
        feedback_ts = _normalise_timestamp(data.index[-1])
        def record_feedback(is_win):
            fb_list = load_feedback_data()
            # Avoid duplicate feedback for the same ticker/timeframe/bar.
            existing = [x for x in fb_list if x.get("ticker")==ticker and x.get("interval")==tf_config["interval"] and _normalise_timestamp(x.get("timestamp"))==feedback_ts]
            if existing:
                for item in fb_list:
                    if item.get("ticker")==ticker and item.get("interval")==tf_config["interval"] and _normalise_timestamp(item.get("timestamp"))==feedback_ts:
                        item["is_win"] = bool(is_win)
                        item["updated_at"] = datetime.now(JST).isoformat()
            else:
                fb_list.append({"ticker":ticker,"interval":tf_config["interval"],"timestamp":feedback_ts,
                                "is_win":bool(is_win),"signal":status,"reference_close":latest_price,
                                "created_at":datetime.now(JST).isoformat()})
            save_feedback_data(fb_list)
        with fb_col1:
            if st.button("👍 利益になった結果を記録",use_container_width=True):
                record_feedback(True)
                st.success("結果を記録しました。AIの再学習データには自動混入させず、取引記録として保持します。")
        with fb_col2:
            if st.button("👎 損失になった結果を記録",use_container_width=True):
                record_feedback(False)
                st.warning("結果を記録しました。AIの再学習データには自動混入させず、取引記録として保持します。")

        with st.expander("📒 実取引の損益記録・振り返り", expanded=False):
            st.caption("松井証券FXの約定履歴・実現損益を見ながら手入力してください。ここに保存した結果はAIの再学習には自動投入しません。")
            journal_col1, journal_col2 = st.columns(2)
            with journal_col1:
                journal_pair = st.selectbox("記録する通貨ペア", list(PAIRS.keys()), index=list(PAIRS.keys()).index(selected_label), key="journal_pair")
                journal_side = st.radio("売買方向", ["買い", "売り"], horizontal=True, key="journal_side")
                journal_qty = st.number_input("約定数量（万通貨）", min_value=0.01, max_value=1000.0, value=float(st.session_state.get("quantity_wan", 0.10)), step=0.01, key="journal_qty")
            with journal_col2:
                journal_entry = st.number_input("実際の新規約定価格", min_value=0.0, value=0.0, step=0.01 if "JPY" in PAIRS[journal_pair] else 0.0001, format="%.3f" if "JPY" in PAIRS[journal_pair] else "%.5f", key="journal_entry")
                journal_exit = st.number_input("実際の決済約定価格", min_value=0.0, value=0.0, step=0.01 if "JPY" in PAIRS[journal_pair] else 0.0001, format="%.3f" if "JPY" in PAIRS[journal_pair] else "%.5f", key="journal_exit")
                journal_pnl = st.number_input("実現損益（円・手数料等反映後）", value=0.0, step=100.0, key="journal_pnl", help="松井証券FXの実現損益を入力してください。利益はプラス、損失はマイナスです。")
            journal_note = st.text_input("メモ（任意）", max_chars=300, key="journal_note", placeholder="例：米CPI前、成行、予定より早く決済")
            if st.button("💾 この実取引を保存", key="save_actual_trade_journal", use_container_width=True):
                if journal_entry <= 0 or journal_exit <= 0 or journal_qty <= 0:
                    st.error("新規価格・決済価格・数量を確認してください。すべて正の値が必要です。")
                else:
                    journal_items = load_trade_journal()
                    journal_items.append({
                        "id": datetime.now(JST).strftime("%Y%m%d%H%M%S%f"),
                        "recorded_at": datetime.now(JST).isoformat(timespec="seconds"),
                        "pair": journal_pair,
                        "ticker": PAIRS[journal_pair],
                        "side": journal_side,
                        "quantity_wan": float(journal_qty),
                        "entry_price": float(journal_entry),
                        "exit_price": float(journal_exit),
                        "realized_pnl_yen": float(journal_pnl),
                        "result": "利益" if journal_pnl > 0 else "損失" if journal_pnl < 0 else "±0円",
                        "note": journal_note.strip(),
                    })
                    save_trade_journal(journal_items)
                    st.success("実取引記録を保存しました。次の再表示でも履歴を読み込みます。")
            journal_items = load_trade_journal()
            if journal_items:
                journal_df = pd.DataFrame(journal_items)
                pnl_values = pd.to_numeric(journal_df.get("realized_pnl_yen", pd.Series(dtype=float)), errors="coerce").fillna(0)
                wins = int((pnl_values > 0).sum())
                settled = int((pnl_values != 0).sum())
                jm1, jm2, jm3 = st.columns(3)
                jm1.metric("記録件数", f"{len(journal_df)}件")
                jm2.metric("実現損益合計", f"{pnl_values.sum():+,.0f}円")
                jm3.metric("利益取引率", f"{wins / settled * 100:.1f}%" if settled else "—", help="±0円の取引は分母から除外。過去記録の集計であり将来の勝率を示しません。")
                journal_view = journal_df.rename(columns={
                    "recorded_at":"記録日時", "pair":"通貨ペア", "side":"方向", "quantity_wan":"数量(万通貨)",
                    "entry_price":"新規価格", "exit_price":"決済価格", "realized_pnl_yen":"実現損益(円)",
                    "result":"結果", "note":"メモ"
                })
                display_cols = [c for c in ["記録日時", "通貨ペア", "方向", "数量(万通貨)", "新規価格", "決済価格", "実現損益(円)", "結果", "メモ"] if c in journal_view.columns]
                st.dataframe(journal_view[display_cols].tail(30), use_container_width=True, hide_index=True)
                st.download_button("📥 実取引記録をCSV保存", data=journal_view[display_cols].to_csv(index=False).encode("utf-8-sig"), file_name="actual_trade_journal.csv", mime="text/csv", key="download_actual_trade_journal")
                if st.button("🗑️ 実取引記録を全削除", key="clear_actual_trade_journal"):
                    save_trade_journal([])
                    st.success("実取引記録を削除しました。画面を再表示すると集計も更新されます。")
            else:
                st.info("まだ実取引記録はありません。決済後に実現損益を入力すると、履歴と合計を確認できます。")
        st.markdown("---")
        with st.expander("✅ 発注直前の安全チェック", expanded=False):
            st.caption("注文前に松井証券FXの画面と照合してください。チェックを入れても発注可否を保証するものではありません。")
            st.checkbox("通貨ペア・売買方向・成行／指値／逆指値を確認した", key="preflight_pair_side")
            st.checkbox("松井証券FXの最新Bid/Askとスプレッドを確認した", key="preflight_bid_ask")
            st.checkbox("損切り・利確の価格と数量を注文画面で確認した", key="preflight_sl_tp_qty")
            st.checkbox("想定損失額と必要証拠金を確認した", key="preflight_risk_margin")
            st.checkbox("経済指標・急変・週末リスクを確認した", key="preflight_events")
            st.warning("成行は入力した参考価格で約定する保証がありません。実際の損失はスリッページ・価格飛び・スプレッド拡大で計算上限を超える場合があります。")
        # Keep the manual calculator synchronized with a newly entered Matsui quote
        # unless the user has deliberately overridden that price themselves.
        _manual_existing_entry = st.session_state.get("manual_calc_entry_price")
        _manual_last_auto_entry = st.session_state.get("manual_calc_last_auto_entry")
        _manual_price_is_auto = (
            _manual_existing_entry is None
            or _manual_last_auto_entry is None
            or abs(float(_manual_existing_entry) - float(_manual_last_auto_entry)) < (pip_unit / 10)
        )
        if matsui_quote is not None and _manual_price_is_auto:
            st.session_state["manual_calc_entry_price"] = float(matsui_quote)
            st.session_state["manual_calc_last_auto_entry"] = float(matsui_quote)
        elif _manual_existing_entry is None:
            st.session_state["manual_calc_last_auto_entry"] = float(latest_price)

        with st.expander("🛠️ 注文条件の手動微調整（必要な場合のみ）", expanded=False):
            st.caption("通常はAI提案の注文プランを確認してください。ここは条件を自分で変更したい場合だけ使います。")
            sc1, sc2 = st.columns(2)
            with sc1:
                # AI direction preselects the manual calculator; WAIT is never treated as a sell signal.
                trade_side = st.radio("売買方向（計算用）",["BUY (買い)","SELL (売り)"],index=0 if not status.startswith("SELL") else 1,horizontal=True, key="manual_calc_trade_side")
                # If the user has entered a Matsui quote above, use it as the initial calculation price.
                _manual_entry_default = float(matsui_quote) if matsui_quote is not None else float(latest_price)
                entry_price = st.number_input("エントリー想定レート", min_value=0.0, value=_manual_entry_default, format=price_fmt, step=0.01 if is_jpy else 0.0001, key="manual_calc_entry_price")
            with sc2:
                sl_mode = st.radio("損切(SL)の決め方",["ATRベース (推奨)","固定 pips"],index=0,horizontal=True, key="manual_calc_sl_mode")
                if sl_mode == "ATRベース (推奨)":
                    atr_multiplier = st.slider("ATR倍率",0.5,3.0,1.5,0.1, key="manual_calc_atr_multiplier")
                    sl_pips = max(1.0,round(latest_atr/pip_unit*atr_multiplier,1))
                    st.info(f"現在のATRから算出した損切幅: **{sl_pips:.1f} pips**")
                else:
                    sl_pips = st.number_input("損切幅 (pips)",min_value=1.0,value=float(ai_sl_pips),step=1.0, key="manual_calc_fixed_sl")
            rr_value = st.slider("リスクリワード比 (RR)",0.5,4.0,value=float(min(4,max(.5,st.session_state["rr_ratio"]))),step=.1, key="manual_calc_rr")
            slippage_pips = st.number_input("想定スリッページ (片道 pips)", min_value=0.0, max_value=20.0, value=1.0, step=0.5, key="manual_calc_slippage", help="成行約定のずれを見込むための仮定値です。実際のスリッページを予測・保証するものではありません。")
        tp_pips = round(sl_pips*rr_value,1)
        is_buy_manual = trade_side.startswith("BUY")
        sl_price = entry_price-sl_pips*pip_unit if is_buy_manual else entry_price+sl_pips*pip_unit
        tp_price = entry_price+tp_pips*pip_unit if is_buy_manual else entry_price-tp_pips*pip_unit
        loss_per_wan = (sl_pips + spread_pips + 2.0 * slippage_pips) * pip_val_per_wan
        raw_manual_wan = max_risk_yen / loss_per_wan if loss_per_wan > 0 and np.isfinite(loss_per_wan) else 0.0
        manual_wan = float(np.floor(max(0.0, raw_manual_wan) * 100) / 100) if np.isfinite(raw_manual_wan) else 0.0
        if manual_wan < 0.01:
            manual_wan = 0.0
        # Final conservative guard: never round the suggested quantity above the risk budget.
        while manual_wan >= 0.01 and (not np.isfinite(loss_per_wan) or loss_per_wan * manual_wan > max_risk_yen + 1e-6):
            manual_wan = round(manual_wan - 0.01, 2)
        if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(sl_pips) or sl_pips <= 0:
            manual_wan = 0.0
            st.error("入力レートまたは損切幅が無効です。正の価格と損切幅を入力してください。")
        expected_profit_yen = (tp_pips - spread_pips - 2.0 * slippage_pips) * pip_val_per_wan * manual_wan
        actual_loss_yen = (sl_pips + spread_pips + 2.0 * slippage_pips) * pip_val_per_wan * manual_wan
        res_col1,res_col2,res_col3,res_col4=st.columns(4)
        res_col1.metric("調整後の手動ロット数",f"{manual_wan:.2f} 万通貨",f"許容リスク上限 {risk_pct}%")
        res_col2.metric("損切(SL) レート",price_fmt%sl_price,f"-{sl_pips:.1f} pips")
        res_col3.metric("利確(TP) レート",price_fmt%tp_price,f"+{tp_pips:.1f} pips")
        res_col4.metric("想定純損益",f"{expected_profit_yen:+,.0f}円",f"最大損失 -{actual_loss_yen:,.0f}円")
        if expected_profit_yen <= 0:
            st.warning("利確幅が想定スプレッド以下のため、コスト控除後の利益が見込めない計算です。")
        st.caption(f"手動調整欄は計算補助です。損益見積りには想定スプレッド {spread_pips:.1f} pips と、片道スリッページ {slippage_pips:.1f} pips×往復2回を加算しています。実際の約定価格・コスト・価格飛びで損益は変わり、損失が計算値を超える場合があります。")

    elif active_section == APP_SECTIONS[2]:
        st.markdown("##### 📰 為替ニュース・経済イベントリスク確認")
        st.caption("見出し語句による補助表示です。公式の経済指標カレンダーではなく、発表予定時刻・市場反応・売買方向を保証しません。")
        risk_summary = summarize_news_risk(news_items)
        if risk_summary["kind"] == "high":
            st.error(f"🚨 ニュース注意度：{risk_summary['level']}\n\n{risk_summary['message']}")
        elif risk_summary["kind"] == "medium":
            st.warning(f"⚠️ ニュース注意度：{risk_summary['level']}\n\n{risk_summary['message']}")
        elif risk_summary["kind"] == "low":
            st.info(f"ℹ️ ニュース注意度：{risk_summary['level']}\n\n{risk_summary['message']}")
        else:
            st.warning(f"⚠️ ニュース注意度：{risk_summary['level']}\n\n{risk_summary['message']}")
        st.caption("取引前チェック：米CPI・雇用統計・FOMC、日銀会合、要人発言、週末・祝日の流動性を公式発表元で確認してください。")
        if news_items:
            st.success(f"ニュース取得状態：{news_status}（{len(news_items)}件）")
        else:
            st.warning(f"ニュース取得状態：{news_status}")
            if news_errors:
                st.caption("接続診断：" + " / ".join(news_errors[:3]))
        st.markdown("---")
        st.markdown(f"##### 🌐 `{selected_label}` 関連ニュース（見出し参考判定）")
        if not news_items:
            st.info("関連ニュースを取得できませんでした。")
        else:
            for news in news_items:
                st.markdown(f"""<div class="news-card">
                <div style="display:flex;justify-content:space-between;gap:10px;margin-bottom:6px;">
                <span style="font-weight:bold;color:#38bdf8;">{html.escape(str(news.get('publisher','')))} / {html.escape(str(news.get('source','')))} ({html.escape(str(news.get('time','')) )})</span>
                <span>{html.escape(str(news.get('impact','')))}</span></div>
                <div style="font-size:1.05rem;font-weight:800;margin-bottom:8px;">
                <a href="{html.escape(str(news.get('link','#')),quote=True)}" target="_blank" rel="noopener noreferrer" style="color:#f8fafc;text-decoration:none;">{html.escape(str(news.get('title','')))} 🔗</a></div>
                <div style="background:#0f172a;padding:10px 14px;border-radius:4px;line-height:1.6;">
                🎯 <b>キーワード参考判定</b>: <span style="color:#facc15;font-weight:bold;">{html.escape(str(news.get('direction','')))}</span><br>
                💡 <b>判定理由</b>: {html.escape(str(news.get('reason','')))}</div></div>""",unsafe_allow_html=True)

    elif active_section == APP_SECTIONS[3]:
        st.markdown("##### 📋 松井証券リピート自動売買（ハーフ＆ハーフ）設定アシスタント")
        st.caption("参考計算です。証券会社の仕様・保有ポジション・実際の証拠金計算を完全には再現しません。")
        if latest_price < user_lower or latest_price > user_upper:
            st.markdown(f'<div class="warn-box">【レンジアウト警告】現在レート <code>{price_fmt%latest_price}</code> は設定レンジ <code>{price_fmt%user_lower}</code>～<code>{price_fmt%user_upper}</code> の外側です。</div>',unsafe_allow_html=True)
        atr_4h_series = clean_series(data_4h["ATR"]).dropna() if data_4h is not None and "ATR" in data_4h else pd.Series(dtype=float)
        atr_4h_val = float(atr_4h_series.iloc[-1]) if not atr_4h_series.empty and float(atr_4h_series.iloc[-1]) > 0 else max(float(latest_atr), pip_unit * 30)
        ai_rec_trap_pips = int(np.clip(round((atr_4h_val / pip_unit) * 0.8), 10, 500))
        trap_key = f"trap_w_{ticker}"
        if trap_key not in st.session_state:
            st.session_state[trap_key] = int(np.clip(st.session_state.get("trap_width", ai_rec_trap_pips), 10, 500))
        def apply_ai_trap_callback():
            st.session_state[trap_key] = int(ai_rec_trap_pips)
            st.session_state["trap_width"] = int(ai_rec_trap_pips)
            save_settings()
        def update_trap_callback():
            st.session_state["trap_width"] = int(st.session_state[trap_key])
            save_settings()
        rc_col1,rc_col2=st.columns([2,1])
        with rc_col1:
            trap_width_pips=st.number_input("注文幅 / 利確幅 (pips)",min_value=10,max_value=500,step=5,key=trap_key,on_change=update_trap_callback)
        with rc_col2:
            st.markdown("<br>",unsafe_allow_html=True)
            st.button(f"🤖 AI推奨値 ({ai_rec_trap_pips} pips) を適用",on_click=apply_ai_trap_callback)
        half_range_pips = abs(user_upper-user_half)/pip_unit
        spacing_pips = max(1.0, float(trap_width_pips))
        half_grid_count = max(1, int(np.floor(half_range_pips / spacing_pips)) + 1)
        total_grid_count = half_grid_count * 2
        quantity_wan_val = max(0.0, float(st.session_state["quantity_wan"]))
        account_balance_val = max(0.0, float(st.session_state["account_balance"]))
        order_units = max(0, int(quantity_wan_val * 10000))
        usd_rate_repeat = latest_price if ticker == "USDJPY=X" else get_usdjpy_rate()
        risk_calc_available = bool(is_jpy or (usd_rate_repeat is not None and np.isfinite(usd_rate_repeat) and usd_rate_repeat > 0))
        if is_jpy:
            pip_value_yen = (order_units / 10000) * 100.0
        elif risk_calc_available:
            pip_value_yen = order_units * 0.0001 * float(usd_rate_repeat)
        else:
            pip_value_yen = 0.0
        # Conservative gross stress estimate: assume each grid order is open and moves adversely
        # to the stop buffer beyond its own half-range edge. This is not a broker execution model.
        buy_levels = [user_lower + i * spacing_pips * pip_unit for i in range(half_grid_count)]
        sell_levels = [user_upper - i * spacing_pips * pip_unit for i in range(half_grid_count)]
        buy_loss_pips = sum(max(0.0, (level - buy_stop_loss) / pip_unit) for level in buy_levels)
        sell_loss_pips = sum(max(0.0, (sell_stop_loss - level) / pip_unit) for level in sell_levels)
        max_loss_yen = (buy_loss_pips + sell_loss_pips) * pip_value_yen
        if order_units > 0 and is_jpy:
            margin_per_order = (latest_price * order_units) / 25.0
        elif order_units > 0 and risk_calc_available:
            margin_per_order = (latest_price * float(usd_rate_repeat) * order_units) / 25.0
        else:
            margin_per_order = 0.0
        total_margin_yen = margin_per_order * total_grid_count
        risk_ratio = max_loss_yen / account_balance_val * 100 if risk_calc_available and account_balance_val > 0 else float("nan")
        st.markdown(f"""<div class="param-box"><b>【ハーフ＆ハーフ推奨設定値】</b>（4時間足ATR基準）<br>
        ・買い設定（下半）: <code>{price_fmt%user_lower}</code>～<code>{price_fmt%user_half}</code> | 停止目安: <code>{price_fmt%buy_stop_loss}</code><br>
        ・売り設定（上半）: <code>{price_fmt%user_half}</code>～<code>{price_fmt%user_upper}</code> | 停止目安: <code>{price_fmt%sell_stop_loss}</code><br>
        ・注文幅/利確幅: <code>{trap_width_pips} pips</code>（AI参考値 {ai_rec_trap_pips} pips） | 片側注文本数目安: <code>{half_grid_count}</code>（合計 {total_grid_count}）</div>""",unsafe_allow_html=True)
        st.code(f"""[松井証券リピート注文 設定値]
    通貨ペア: {selected_label}
    注文種別: ハーフ＆ハーフ
    買いレンジ: {price_fmt%user_lower} - {price_fmt%user_half} (停止目安: {price_fmt%buy_stop_loss})
    売りレンジ: {price_fmt%user_half} - {price_fmt%user_upper} (停止目安: {price_fmt%sell_stop_loss})
    注文幅 / 利確幅: {trap_width_pips} pips
    1本あたり数量: {quantity_wan_val:.2f} 万通貨""",language="text")
        st.markdown("##### 🛡️ リスク・資金シミュレーション")
        rc1,rc2,rc3=st.columns(3)
        rc1.metric("想定含み損シナリオ", f"約 {int(max_loss_yen):,} 円" if risk_calc_available else "算出不可")
        rc2.metric("全注文成立時の必要証拠金概算", f"約 {int(total_margin_yen):,} 円" if risk_calc_available else "算出不可")
        rc3.metric("資金リスク比率", f"{risk_ratio:.1f}%" if np.isfinite(risk_ratio) else "算出不可")
        if not risk_calc_available:
            st.error("USD/JPY換算レートを取得できないため、円換算のリスク試算は停止しています。レート取得後に再更新してください。")
        elif risk_ratio > 40:
            st.error("警告: 単純試算の損失シナリオが資金の40%を超えています。")
        elif risk_ratio > 20:
            st.warning("損失シナリオが大きめです。注文数・数量・レンジを見直してください。")
        else:
            st.info("この単純試算だけでは安全性を判断できません。")
        st.markdown("---")
        st.markdown("##### 📌 損益見通しについて")
        st.caption(
            "ATRだけから月間の約定回数・利益を信頼性高く予測することはできないため、"
            "根拠の弱い月間・年間利益予測は表示しません。上の想定損失・必要証拠金の概算を確認し、"
            "松井証券FXの最新条件と注文画面で最終確認してください。"
        )

    elif active_section == APP_SECTIONS[4]:
        st.markdown("##### 🤖 AI予測モデルの評価と内訳")
        st.info("モデルは価格データのラベルで再学習します。予測方向は確定足ごとに自動記録し、後続の確定足で自動採点します。利益/損失ボタンは取引メモで、教師データへ自動混入しません。再学習・採点はアプリが動作して分析処理が走った時に進みます。分類一致率は収益性を示すものではありません。")
        # Automatic signal evaluation summary for the current pair/timeframe.
        forecast_items = load_forecast_history()
        forecast_items = update_forecast_history(forecast_items, data, ticker, tf_config["interval"], LOOKAHEAD_BARS)
        relevant_forecasts = [x for x in forecast_items if x.get("ticker") == ticker and x.get("interval") == tf_config["interval"]]
        resolved_forecasts = [x for x in relevant_forecasts if x.get("status") in ("CORRECT", "INCORRECT", "FLAT")]
        directional_forecasts = [x for x in resolved_forecasts if x.get("signal") in ("BUY", "SELL")]
        expired_forecasts = [x for x in relevant_forecasts if x.get("status") == "EXPIRED_NO_DATA"]
        correct_forecasts = sum(x.get("status") == "CORRECT" for x in directional_forecasts)
        st.markdown("##### 🔁 自動予測記録・方向性検証")
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("記録した予測", f"{len(relevant_forecasts)} 件")
        fc2.metric("検証済み買い/売り", f"{len(directional_forecasts)} 件")
        fc3.metric("方向一致率", f"{(correct_forecasts / len(directional_forecasts) * 100):.1f}%" if directional_forecasts else "データ不足")
        fc4.metric("採点対象外", f"{len(expired_forecasts)} 件")
        st.caption(f"予測後{LOOKAHEAD_BARS}本の確定足終値で方向を自動採点します。WAITは方向一致率から除外。取得範囲外になった未検証記録は誤採点せず『採点対象外』へ変更します。これはYahoo Finance価格の方向性検証で、実際の取引勝率・利益・約定を示すものではありません。検証はアプリが起動し、対象データを取得したときに進みます。")
        _label_scored = [x for x in relevant_forecasts if isinstance(x.get("target_label_match"), bool)]
        if _label_scored:
            _label_hits = sum(bool(x.get("target_label_match")) for x in _label_scored)
            with st.expander("学習ラベルとの一致（補助評価）", expanded=False):
                st.metric("学習ラベル一致率", f"{_label_hits / len(_label_scored) * 100:.1f}%", f"検証済み: {len(_label_scored)}件")
                st.caption("モデルの予測クラスと、学習に使うATR利確・損切りの先着ラベルを比較します。終値方向の一致率とは別指標で、実取引の勝率や利益を示しません。件数が少ない間は参考値です。")
        else:
            with st.expander("学習ラベルとの一致（補助評価）", expanded=False):
                st.info("学習ラベルの採点記録がまだありません。予測後の確定足が十分に蓄積すると表示されます。")
        _wait_records = [x for x in relevant_forecasts if x.get("signal") == "WAIT"]
        _reason_counts = {}
        for _item in _wait_records:
            _reason = _item.get("skip_reason") or "過去記録のため理由の詳細なし"
            _reason_counts[_reason] = _reason_counts.get(_reason, 0) + 1
        if _reason_counts:
            with st.expander("🛑 見送り・様子見の記録", expanded=False):
                st.caption("保存済みのAI判定から見送り傾向を確認します。旧記録は理由が保存されていない場合があります。")
                st.dataframe(pd.DataFrame([{"記録理由": k, "件数": v} for k, v in _reason_counts.items()]), use_container_width=True, hide_index=True)

        # Cross-pair/timeframe view from already stored forecast records; no extra market-data calls.
        _all_resolved = [x for x in forecast_items if x.get("status") in ("CORRECT", "INCORRECT", "FLAT") and x.get("signal") in ("BUY", "SELL")]
        if _all_resolved:
            _all_df = pd.DataFrame(_all_resolved)
            _all_df["通貨ペア"] = _all_df["ticker"].map({v: k for k, v in PAIRS.items()}).fillna(_all_df["ticker"])
            _all_df["時間足"] = _all_df["interval"].map({"5m":"5分足", "15m":"15分足", "1h":"1時間足", "4h":"4時間足", "1d":"日足"}).fillna(_all_df["interval"])
            _all_df["一致"] = _all_df["status"].eq("CORRECT")
            _all_df["不一致"] = _all_df["status"].eq("INCORRECT")
            _all_df["横ばい"] = _all_df["status"].eq("FLAT")
            _summary = _all_df.groupby(["通貨ペア", "時間足"], dropna=False).agg(
                検証件数=("status", "size"), 方向一致=("一致", "sum"), 方向不一致=("不一致", "sum"), 横ばい=("横ばい", "sum")
            ).reset_index()
            _summary["方向一致率（横ばい除外）"] = _summary.apply(
                lambda r: f"{r['方向一致'] / (r['方向一致'] + r['方向不一致']) * 100:.1f}%"
                if (r['方向一致'] + r['方向不一致']) else "データ不足", axis=1
            )
            _summary["件数の目安"] = _summary["検証件数"].apply(lambda n: "少数・参考値" if n < 30 else ("中程度" if n < 100 else "件数あり・継続検証"))
            st.markdown("##### 🌐 通貨ペア・時間足別の検証成績")
            st.dataframe(_summary.sort_values(["通貨ペア", "時間足"]), use_container_width=True, hide_index=True)
            st.caption("保存済み記録のみを集計。30件未満は特に不安定です。件数が増えても利益・勝率を保証せず、未使用期間での検証も必要です。")
            st.download_button("📥 全通貨ペア・時間足の検証集計CSV", data=_summary.to_csv(index=False).encode("utf-8-sig"), file_name="forecast_validation_pair_timeframe_summary.csv", mime="text/csv")
        else:
            st.info("通貨ペア・時間足別の集計は、各組み合わせの予測検証記録が蓄積すると表示されます。")

        if relevant_forecasts:
            # Confidence bands are descriptive historical summaries, not guarantees.
            _resolved_df = pd.DataFrame(resolved_forecasts)
            if not _resolved_df.empty:
                _directional_df = _resolved_df[_resolved_df["signal"].isin(["BUY", "SELL"])].copy()
                if not _directional_df.empty:
                    _directional_df["予測確信度"] = _directional_df.apply(
                        lambda r: max(float(r.get("buy_pct", 0) or 0), float(r.get("sell_pct", 0) or 0)), axis=1
                    )
                    _directional_df["確信度帯"] = pd.cut(
                        _directional_df["予測確信度"],
                        bins=[-1, 49, 59, 69, 100],
                        labels=["49%以下", "50–59%", "60–69%", "70%以上"],
                    )
                    _calibration = _directional_df.groupby("確信度帯", observed=False).agg(
                        件数=("status", "size"),
                        方向一致=("status", lambda s: int((s == "CORRECT").sum())),
                        方向不一致=("status", lambda s: int((s == "INCORRECT").sum())),
                        横ばい=("status", lambda s: int((s == "FLAT").sum())),
                    ).reset_index()
                    _calibration["方向一致率※横ばい除外"] = _calibration.apply(
                        lambda r: f"{r['方向一致'] / (r['方向一致'] + r['方向不一致']) * 100:.1f}%"
                        if (r["方向一致"] + r["方向不一致"]) > 0 else "データ不足", axis=1
                    )
                    st.markdown("##### 🎯 予測確信度別の実績")
                    st.dataframe(_calibration, use_container_width=True, hide_index=True)
                    st.caption("各確信度帯は件数が少ないと大きく変動します。30件未満は参考値として扱い、確信度の数値を実際の的中確率と同一視しないでください。")
                    st.caption("確信度帯はモデル出力の買い/売り確率の最大値で分類。少数件では大きくぶれるため、件数と合わせて確認してください。将来の的中を保証する指標ではありません。")

                    _by_signal = _directional_df.groupby("signal").agg(
                        検証件数=("status", "size"),
                        方向一致=("status", lambda s: int((s == "CORRECT").sum())),
                        方向不一致=("status", lambda s: int((s == "INCORRECT").sum())),
                        横ばい=("status", lambda s: int((s == "FLAT").sum())),
                    ).reset_index()
                    _by_signal["方向一致率※横ばい除外"] = _by_signal.apply(
                        lambda r: f"{r['方向一致'] / (r['方向一致'] + r['方向不一致']) * 100:.1f}%"
                        if (r["方向一致"] + r["方向不一致"]) > 0 else "データ不足", axis=1
                    )
                    st.markdown("##### ↕️ 買い・売り別の検証")
                    st.dataframe(_by_signal, use_container_width=True, hide_index=True)
            _fh = pd.DataFrame(relevant_forecasts).tail(30)
            _cols = [c for c in ["bar_timestamp", "signal", "buy_pct", "sell_pct", "wait_pct", "status", "reference_close", "evaluated_close"] if c in _fh.columns]
            st.dataframe(_fh[_cols].rename(columns={"bar_timestamp":"予測足時刻", "signal":"予測方向", "buy_pct":"買い(%)", "sell_pct":"売り(%)", "wait_pct":"様子見(%)", "status":"検証結果", "reference_close":"予測時価格", "evaluated_close":"検証時価格"}), use_container_width=True, hide_index=True)
            st.download_button("📥 自動予測検証CSVを保存", data=pd.DataFrame(relevant_forecasts).to_csv(index=False).encode("utf-8-sig"), file_name=f"forecast_validation_{ticker.replace('=X','')}_{tf_config['interval']}.csv", mime="text/csv")
        # Update saved entry-candidate outcomes from subsequent completed bars.
        tracking_items = update_entry_tracking_outcomes(load_entry_tracking(), data, ticker, tf_config["interval"], LOOKAHEAD_BARS)
        st.markdown("##### 🎯 候補価格の到達実績")
        relevant_tracking = [x for x in tracking_items if x.get("ticker") == ticker and x.get("interval") == tf_config["interval"]]
        resolved_tracking = [x for x in relevant_tracking if x.get("status") in ("HIT", "NOT_REACHED")]
        hit_count = sum(x.get("status") == "HIT" for x in resolved_tracking)
        pending_count = sum(x.get("status") == "PENDING" for x in relevant_tracking)
        hit_rate = (hit_count / len(resolved_tracking) * 100) if resolved_tracking else None
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("検証済み候補", f"{len(resolved_tracking)} 件")
        tc2.metric("8本以内に到達", f"{hit_count} 件")
        tc3.metric("到達率", f"{hit_rate:.1f}%" if hit_rate is not None else "データ不足")
        st.caption(f"未判定: {pending_count} 件。判定対象は後続の確定足最大{LOOKAHEAD_BARS}本です。到達率は利益率・約定率・将来予測精度ではなく、yfinanceのOHLCデータによる参考値です。")
        if relevant_tracking:
            track_df = pd.DataFrame(relevant_tracking)
            display_cols = [c for c in ["saved_at", "signal", "entry_style", "reference_price", "candidate_price", "status", "bars_to_touch"] if c in track_df.columns]
            st.dataframe(track_df[display_cols].rename(columns={"saved_at":"記録日時", "signal":"AI方向", "entry_style":"方針", "reference_price":"基準価格", "candidate_price":"候補価格", "status":"結果", "bars_to_touch":"到達までの本数"}).tail(30), use_container_width=True, hide_index=True)
            st.download_button("📥 候補価格検証履歴CSV", data=track_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"entry_candidate_history_{ticker.replace('=X','')}_{tf_config['interval']}.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.caption("まだ検証履歴がありません。単発トレードタブで松井証券FXの現在レートを入力し、候補を保存してください。")
        st.markdown("---")
        pcol1,pcol2=st.columns(2)
        with pcol1:
            st.markdown("**最新確定バーのモデル出力確率（総合スコア）**")
            for label,key,color in [("BUY (買い)","buy","#22c55e"),("SELL (売り)","sell","#ef4444"),("WAIT (様子見)","wait","#94a3b8")]:
                st.write(f"{label}: {prob_dict[key]}%")
                st.progress(max(0,min(1,prob_dict[key]/100)))
        with pcol2:
            st.markdown("**時系列アウト・オブ・サンプル検証**")
            st.metric("テスト区間の分類一致率",f"{win_rate:.1f}%",f"検証数: {total_signals_count}本")
            st.caption("分類一致率は売買損益の勝率ではありません。")
            eval_info = st.session_state.get("model_eval_metrics", {})
            if eval_info:
                st.metric("クラス偏り補正の一致率", f"{eval_info.get('balanced_accuracy', 0):.1f}%")
                st.caption(
                    f"単純基準モデルの補正一致率: {eval_info.get('baseline_balanced_accuracy', 0):.1f}%。"
                    "AIが基準を十分に上回らない場合は売買を見送り、弱い候補モデルを保存しません。"
                )
                if eval_info.get("quality_ok"):
                    st.success("今回の検証では、AIモデルが基準モデルを設定した幅だけ上回りました。ただし利益を保証しません。")
                else:
                    st.warning("今回の検証ではAIモデルの優位性を確認できませんでした。見送り判定を優先します。")
            current_fb=load_feedback_data()
            st.info(f"蓄積された取引結果フィードバック記録: **{len(current_fb)} 件**（市場ラベルとは分離して保存）")
            if st.button("🗑️ 自己学習フィードバック履歴をリセット"):
                try:
                    if os.path.exists(FEEDBACK_FILE): os.remove(FEEDBACK_FILE)
                except OSError:
                    pass
                st.toast("フィードバック記録をリセットしました。保存済みモデルの市場ラベル学習には影響しません。")
                st.rerun()
            if ai_signal_history is not None and not ai_signal_history.empty:
                sig_export_df=ai_signal_history.rename("予測ラベル(1=買い,-1=売り,0=様子見)").rename_axis("日時").reset_index()
                st.download_button("📥 AIシグナル履歴CSVダウンロード",data=sig_export_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"ai_signal_history_{ticker.replace('=X','')}.csv",mime="text/csv",use_container_width=True)
        st.markdown("---")
        st.markdown("##### 📊 テクニカル指標ダッシュボード（確定足）")
        def latest_finite(column, default):
            if column not in data:
                return float(default)
            values = clean_series(data[column]).dropna()
            return float(values.iloc[-1]) if not values.empty and np.isfinite(values.iloc[-1]) else float(default)
        curr_rsi = latest_finite("RSI", 50)
        curr_adx = latest_finite("ADX", 25)
        curr_bb = latest_finite("BB_PctB", 0.5)
        curr_macd = latest_finite("MACD_Hist", 0)
        curr_stoch = latest_finite("Stoch_K", 50)
        st.markdown(f"""<table class="custom-table"><thead><tr><th>指標名</th><th>確定足数値</th><th>状態判定</th></tr></thead><tbody>
        <tr><td><b>RSI (14)</b></td><td><code>{curr_rsi:.1f}</code></td><td>{'買われ過ぎ' if curr_rsi>70 else '売られ過ぎ' if curr_rsi<30 else 'ニュートラル'}</td></tr>
        <tr><td><b>ADX</b></td><td><code>{curr_adx:.1f}</code></td><td>{'強いトレンド' if curr_adx>25 else 'レンジ/膠着'}</td></tr>
        <tr><td><b>ボリンジャー%B</b></td><td><code>{curr_bb:.2f}</code></td><td>{'上方バンド付近' if curr_bb>.85 else '下方バンド付近' if curr_bb<.15 else 'バンド内'}</td></tr>
        <tr><td><b>MACDヒストグラム</b></td><td><code>{curr_macd:.5f}</code></td><td>{'上昇モメンタム' if curr_macd>0 else '下落/弱含み'}</td></tr>
        <tr><td><b>ストキャス%K</b></td><td><code>{curr_stoch:.1f}</code></td><td>{'高値圏' if curr_stoch>80 else '安値圏' if curr_stoch<20 else '中立圏'}</td></tr>
        </tbody></table>""",unsafe_allow_html=True)
        if feature_importances is not None and not feature_importances.empty:
            st.markdown("---")
            col_imp_title,col_imp_btn=st.columns([3,1])
            with col_imp_title: st.markdown("##### 📊 AIの判断根拠（特徴量重要度 TOP 10）")
            with col_imp_btn:
                csv_data=feature_importances.rename(index=lambda x:FEATURE_LABELS.get(x,x)).rename("重要度スコア").rename_axis("指標名").reset_index()
                st.download_button("📥 重要度CSV",data=csv_data.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"ai_feature_importances_{ticker.replace('=X','')}.csv",mime="text/csv",key="download_csv_btn")
            top10_imp=feature_importances.nlargest(10)
            _imp_max=float(top10_imp.max()) if len(top10_imp) else 0.0
            _bar_rows=[]
            for _feat,_val in top10_imp.items():
                _pct=(float(_val)/_imp_max*100) if _imp_max>0 else 0.0
                _bar_rows.append(f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;"><div style="width:118px;text-align:right;font-size:12px;white-space:nowrap;overflow:hidden;">{FEATURE_LABELS.get(_feat,_feat)}</div><div style="flex:1;background:rgba(255,255,255,0.08);border-radius:4px;"><div style="width:{_pct:.1f}%;background:#6366f1;height:16px;border-radius:4px;"></div></div><div style="width:52px;font-size:12px;">{float(_val):.3f}</div></div>')
            st.markdown('<div style="padding:6px 2px;">'+"".join(_bar_rows)+'</div><div style="font-size:11px;opacity:0.7;margin-top:2px;">重要度スコア（モデルの特徴量重要度です。勝率ではありません）</div>',unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("##### 📝 簡易トレードノート（考察・反省メモ）")
        def update_memo_callback(): save_settings()
        st.text_area("トレードノートを入力",key="trade_memo",height=150,on_change=update_memo_callback,placeholder="例: 米雇用統計発表前のため見送り。")
        memo_text=st.session_state.get("trade_memo","")
        if memo_text.strip():
            st.download_button("📥 トレードノートをテキスト保存 (.txt)",data=memo_text.encode("utf-8"),
                               file_name=f"trade_memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",mime="text/plain",key="download_memo_btn")


# Streamlit fragments keep the sidebar/navigation stable during periodic updates.
# On older Streamlit versions, fall back to normal manual reruns rather than failing to start.
if hasattr(st, "fragment"):
    _fragment = st.fragment(run_every=300 if (st.session_state.get("auto_refresh", False) or st.session_state.get("auto_pair_scan", False)) else None)(_render_main_area)
    _fragment()
else:
    if st.session_state.get("auto_refresh", False) or st.session_state.get("auto_pair_scan", False):
        st.sidebar.warning("現在のStreamlitは部分自動更新に未対応です。自動更新・定期スキャンは手動操作時のみ実行されます。Streamlitの更新をご検討ください。")
    _render_main_area()