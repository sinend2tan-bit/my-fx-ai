# app.py
# redeploy trigger 2026-10-26 (reflect PR #7: plain HTML importance bars)
# Pro FX Analyzer & Signal Pro — 安全性・検証整合性・モバイル表示改善版
# 価格分析・注文補助専用です。証券口座接続・自動発注は行いません。
# 予測確率・バックテスト結果は将来の利益や実際の勝率を保証しません。

import html
import json
import os
import tempfile
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
MODEL_VERSION = 8

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


def load_pair_scan_outcomes():
    """Load candidate outcome tracking rows (post-candidate auto scoring)."""
    try:
        with open(PAIR_SCAN_OUTCOME_FILE, "r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, list) else []
    except (OSError, ValueError, TypeError):
        return []


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
        "pair_scan_outcomes": load_pair_scan_outcomes(),
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
    # Backups created before pair scan outcomes were added to the bundle do
    # not have this key; restoring one must not wipe the current file.
    if "pair_scan_outcomes" in bundle:
        lists[PAIR_SCAN_OUTCOME_FILE] = bundle.get("pair_scan_outcomes")
    if any(not isinstance(v, list) for v in lists.values()):
        raise ValueError("履歴データの形式が正しくありません。復元を中止しました。")
    # Reject malformed rows before touching current files. All consumers expect
    # dictionary records and call .get() on them during later evaluation/rendering.
    if any(not isinstance(row, dict) for items in lists.values() for row in items):
        raise ValueError("履歴データに不正な行があります。復元を中止しました。")

    # Serialize every target before writing anything, then stage files beside their
    # destinations so os.replace remains atomic on the same filesystem.
    payloads = {
        SETTINGS_FILE: json.dumps(safe_settings, ensure_ascii=False, indent=2, default=_json_default),
    }
    payloads.update({
        filename: json.dumps(items[-3000:], ensure_ascii=False, indent=2, default=_json_default)
        for filename, items in lists.items()
    })
    staged = {}
    originals = {}
    replaced = []
    try:
        for filename, payload in payloads.items():
            directory = os.path.dirname(os.path.abspath(filename)) or "."
            if os.path.exists(filename):
                with open(filename, "rb") as current_file:
                    originals[filename] = current_file.read()
            else:
                originals[filename] = None
            fd, temp_path = tempfile.mkstemp(prefix=".aifx-restore-", suffix=".tmp", dir=directory)
            staged[filename] = temp_path
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

        try:
            for filename, temp_path in staged.items():
                os.replace(temp_path, filename)
                replaced.append(filename)
        except OSError as write_error:
            # Restore any destination already replaced if a later replace fails.
            rollback_errors = []
            for filename in reversed(replaced):
                try:
                    original = originals[filename]
                    if original is None:
                        if os.path.exists(filename):
                            os.remove(filename)
                    else:
                        directory = os.path.dirname(os.path.abspath(filename)) or "."
                        fd, rollback_path = tempfile.mkstemp(
                            prefix=".aifx-rollback-", suffix=".tmp", dir=directory
                        )
                        with os.fdopen(fd, "wb") as f:
                            f.write(original)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(rollback_path, filename)
                except OSError as rollback_error:
                    rollback_errors.append(f"{filename}: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    f"復元に失敗し、一部ファイルの原状回復も失敗しました: {'; '.join(rollback_errors)}"
                ) from write_error
            raise
    finally:
        for temp_path in staged.values():
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


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


# 段階1（016）追加レビュー対応: データ取得の停止理由の記録。
# キャッシュ対象のコア処理（_load_and_process_data_cached）は停止理由を
# 戻り値で返し、警告表示はキャッシュの外側の load_and_process_data() が
# 行う。Streamlit の要素リプレイ（キャッシュヒット時にキャッシュ内要素が
# 再生される仕様）に依存せず、ヒット時にも毎回確実に表示するため。
# _LOAD_STOP_REASONS は (ticker, interval) -> {"code", "message"} の記録で、
# データ取得失敗・データ不足では記録されないため、停止と区別できる。
# スクリプトは再実行のたびに読み直されるため、この記録は1回の画面描画の
# 中でのみ有効（永続化はしない）。
_LOAD_STOP_REASONS = {}
_LOAD_STOP_WARNED = set()


def get_load_stop_reason(ticker, interval):
    """直近の load_and_process_data() で記録された停止理由を返す。

    停止していない場合（正常に取得できた場合・取得失敗・データ不足の
    場合）は None。停止理由がある場合のみ辞書を返すため、呼び出し側で
    「時刻情報を確認できないための停止」と「単なるデータ不足」を
    区別できる。
    """
    return _LOAD_STOP_REASONS.get((ticker, interval))


# Price data is cached for 60 seconds. This is data freshness, not a UI auto-refresh timer.
@st.cache_data(ttl=60, show_spinner=False)
def _load_and_process_data_cached(ticker, period, interval, label=""):
    """load_and_process_data() の実処理（キャッシュ対象）。

    戻り値は (DataFrame | None, 停止理由 | None) のタプル。停止理由は
    {"code", "message"} の辞書で、表示はキャッシュの外側のラッパーが
    行う（この関数内では st.warning を呼ばない）。
    """
    try:
        actual_interval = "1h" if interval == "4h" else interval
        df = yf.download(ticker, period=period, interval=actual_interval, auto_adjust=False,
                         progress=False, threads=False, timeout=20)
        df = flatten_yf_df(df)
        required = ["Open", "High", "Low", "Close"]
        if df.empty or not set(required).issubset(df.columns):
            return None, None
        df = df[required + (["Volume"] if "Volume" in df.columns else [])].copy()
        data_quality = None
        # タイムゾーン安全停止の理由（1h・1d 共通）: 本番環境の
        # yfinance の版が未確認であることを前提に、実行時に index の
        # 型とタイムゾーンを検査する。タイムゾーンを確定できない場合は
        # 処理を停止し、理由を表示する（UTC と推定して処理を続ける
        # ことはしない）。
        tz_unavailable_reason = {
            "code": "tz_unavailable",
            "message": (
                "価格データの時刻にタイムゾーン情報がないため、分析を停止しました。"
                "UTCと推定して処理することはありません。配信元の時刻仕様を確認してください。"
            ),
        }
        if interval == "1h":
            # 段階1（SYN-FX-016）: 入力検証・C-1 カウンタは1時間足
            # だけに適用する（日足は適用範囲外。下記の elif を参照）。
            # 処理順序は、ChatGPT 決定（案A）により鮮度除外を品質検証
            # の前に置く: DatetimeIndex の確認 → タイムゾーンの確認
            # （tz-aware は UTC へ正規化、tz-naive は停止）→ 有効
            # timestamp の最大値（最新 timestamp）群の鮮度除外 →
            # input_rows の確定 → (0) timestamp 不正 → (1) 欠損・非有限値
            # → (2) 重複 timestamp (keep="last") → (3) OHLC 整合性
            # → (4) 特徴量計算による除外。NaT 行は鮮度除外に含めず、
            # (0) で dropped_invalid_timestamp として計上する。
            # input_rows は、最新 timestamp 群の鮮度除外後に品質検証
            # パイプラインへ入る行数とする。鮮度除外はデータ品質の
            # 除外理由ではないため C-1 の品質カウンタには計上しない。
            # 各行は最初に該当した理由で一度だけ計上する。
            data_quality = {
                "input_rows": 0,
                "dropped_invalid_timestamp": 0,
                "dropped_missing_or_nonfinite": 0,
                "dropped_duplicate_timestamp": 0,
                "dropped_invalid_ohlc": 0,
                "dropped_feature_nan": 0,
                "feature_rows": 0,
                "dropped_label_discontinuous": 0,
                "dropped_label_insufficient_future": 0,
                "dropped_label_other": 0,
                "trainable_rows": 0,
            }
            # index が日時でない場合は時刻を確定できないため処理を停止する。
            if not isinstance(df.index, pd.DatetimeIndex):
                return None, {
                    "code": "timestamp_format_unconfirmed",
                    "message": "価格データの時刻形式を確認できないため、分析を停止しました。",
                }
            if df.index.tz is None:
                return None, tz_unavailable_reason
            df.index = df.index.tz_convert("UTC")
            # 鮮度除外: 有効 timestamp の最大値（最新 timestamp）に
            # 属する行をすべて除外する（未確定の最新足。複数行ある
            # 場合はその群全体）。NaT 行はここでは除外せず、次の
            # (0) で計上する。
            valid_timestamp = df.index.notna()
            if valid_timestamp.any():
                latest_timestamp = df.index[valid_timestamp].max()
                freshness_rows = valid_timestamp & (df.index == latest_timestamp)
                df = df.loc[~freshness_rows].copy()
            # input_rows は、鮮度除外後に品質検証パイプラインへ
            # 入る行数（NaT 行を含む）。
            data_quality["input_rows"] = len(df)
            # (0) timestamp の不正行（NaT）を除外する。
            valid_timestamp = df.index.notna()
            data_quality["dropped_invalid_timestamp"] = int((~valid_timestamp).sum())
            df = df.loc[valid_timestamp].copy()
            # (1) 欠損・非有限値の行を除外する。
            df = df.apply(pd.to_numeric, errors="coerce")
            rows_before = len(df)
            df = df.dropna(subset=required)
            finite_ok = np.isfinite(df[required].to_numpy(dtype=float)).all(axis=1)
            df = df.loc[finite_ok].copy()
            data_quality["dropped_missing_or_nonfinite"] = int(rows_before - len(df))
            # (2) 重複 timestamp を keep="last" で整理する。
            rows_before = len(df)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            data_quality["dropped_duplicate_timestamp"] = int(rows_before - len(df))
            # (3) OHLC 整合性を検証する。keep="last" で残った行が不正な
            # 場合は古い重複行へ戻らず、その timestamp 自体を除外する。
            oc_max = df[["Open", "Close"]].max(axis=1)
            oc_min = df[["Open", "Close"]].min(axis=1)
            ohlc_ok = (
                (df["High"] >= oc_max)
                & (df["Low"] <= oc_min)
                & (df["High"] >= df["Low"])
                & (df[required] > 0).all(axis=1)
            ).to_numpy()
            data_quality["dropped_invalid_ohlc"] = int((~ohlc_ok).sum())
            df = df.loc[ohlc_ok].copy()
        elif interval == "1d":
            # 日足は段階1の適用範囲外（ChatGPT 再レビューによる適用
            # 範囲の是正）: 入力検証・C-1 カウンタ・data_quality は
            # 適用せず、従来（base）の処理順序・データ加工に戻す。
            # ここで維持するのはタイムゾーン安全停止仕様のみで、
            # DatetimeIndex でない場合と tz-naive の場合は停止し、
            # 理由を返す（日足は段階0で全ペア tz-naive と確認されて
            # おり、根拠が得られるまで停止対象）。tz-aware の日足は
            # tz 変換も含めて従来経路で処理する。
            if not isinstance(df.index, pd.DatetimeIndex):
                return None, {
                    "code": "timestamp_format_unconfirmed",
                    "message": "価格データの時刻形式を確認できないため、分析を停止しました。",
                }
            if df.index.tz is None:
                return None, tz_unavailable_reason
            df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=required).sort_index()
        elif interval == "4h":
            # 段階2（SYN-FX-017）: 4時間足の源データは1時間足の
            # ダウンロードであり、段階1の1時間足と同じ実行時検査を
            # 適用する。DatetimeIndex でない場合とタイムゾーンなしの
            # 場合は、UTC と推定せず、段階1と同じ停止理由の仕組みで
            # 停止する。タイムゾーン付きの場合は、集約の前に UTC へ
            # 正規化する（次の集約ブロックで origin="start_day" を
            # 適用し、Scanner と同じ絶対時刻の区間境界にするため）。
            if not isinstance(df.index, pd.DatetimeIndex):
                return None, {
                    "code": "timestamp_format_unconfirmed",
                    "message": "価格データの時刻形式を確認できないため、分析を停止しました。",
                }
            if df.index.tz is None:
                return None, tz_unavailable_reason
            df.index = df.index.tz_convert("UTC")
            df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=required)
            # 重複 timestamp の整理は Scanner の共通整形と文字どおり
            # 同じ式・同じ順序（dropna の後、sort_index の前、
            # keep="last"）で行う。これにより後段の完全性判定の本数が
            # 「異なる timestamp の本数」になり、集約値（OHLC・
            # Volume）も Scanner と一致する（段階2再レビュー対応・
            # 案b）。重複のないデータでは挙動は変わらない。
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=required).sort_index()
        if interval == "4h":
            agg = {"Open":"first","High":"max","Low":"min","Close":"last"}
            if "Volume" in df: agg["Volume"] = "sum"
            # 段階2（SYN-FX-017）: UTC の 00:00 起点で集約し（Scanner と
            # 同じ resample("4h", origin="start_day")）、区間ごとに元の
            # 1時間足の本数を数えて、4本そろった区間だけを採用する
            # （Scanner の完全性判定と同じ考え方）。最終区間の一律除外は
            # この完全性判定に置き換えたため、下の iloc[:-1] の対象から
            # 4時間足を外す。
            # cutoff は Scanner と同一式（現在時刻の5分前、配信遅延への
            # 猶予）。終了時刻（index+1h）が cutoff 以前の1時間足だけを
            # 残し、形成中の足を集約前に除外する。採用条件にも区間の
            # 終了（index+4h <= cutoff）を Scanner と同じく両項で明示
            # する（段階2再レビュー対応）。cutoff で完全な区間が80未満
            # になれば、既存の不足扱い（len(df) < 80）にそのまま乗る。
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=5)
            df = df.loc[(df.index + pd.Timedelta(hours=1)) <= cutoff].copy()
            hourly_counts = df["Close"].resample("4h", origin="start_day").count()
            df = df.resample("4h", origin="start_day").agg(agg).dropna(subset=required)
            complete_counts = hourly_counts.reindex(df.index).fillna(0)
            df = df.loc[
                ((df.index + pd.Timedelta(hours=4)) <= cutoff)
                & (complete_counts == 4)
            ].copy()
        # Discard the newest bar because it may still be forming.
        # 1時間足は品質検証の前で最新 timestamp 群を鮮度除外済みのため、
        # ここでは除外しない（案A）。4時間足は完全性判定に置き換えた
        # ため除外しない。その他の時間足は従来どおり。
        if interval not in ("1h", "4h") and len(df) > 2:
            df = df.iloc[:-1]
        if len(df) < 80:
            return None, None
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
        result = df.dropna(subset=FEATURE_COLUMNS + ["Close", "ATR"])
        if data_quality is not None:
            # (4) 特徴量計算による除外と、ラベル判定段階の計上。
            # ラベル判定段階は段階1では現行判定の記録のみ（ラベル定義・
            # 8本窓の定義は変更しない）。App の現行ラベル計算は対象行に
            # 必ずラベルを付けるため、不連続・その他の理由でラベルが
            # 無効になる行は存在せず、末尾の未来窓不足の行だけが
            # ラベルなしで残る。
            data_quality["dropped_feature_nan"] = int(len(df) - len(result))
            data_quality["feature_rows"] = int(len(result))
            data_quality["dropped_label_discontinuous"] = 0
            data_quality["dropped_label_insufficient_future"] = int(
                result["Target"].isna().sum()
            )
            data_quality["dropped_label_other"] = 0
            data_quality["trainable_rows"] = int(
                data_quality["feature_rows"]
                - data_quality["dropped_label_insufficient_future"]
            )
            result.attrs["data_quality"] = data_quality
        return result, None
    except Exception:
        return None, None


def load_and_process_data(ticker, period, interval, label=""):
    """価格データの取得・加工（公開関数）。

    戻り値は従来どおり DataFrame か None（呼び出し側との契約は不変）。
    実処理は _load_and_process_data_cached()（キャッシュ対象）が行い、
    停止理由の記録と警告表示はここ（キャッシュの外側）で行うため、
    キャッシュヒット時にも停止理由が毎回表示される。同一の画面描画の
    中で同じ時間足・同じ理由の警告は1回だけ表示する。
    """
    result, stop_reason = _load_and_process_data_cached(
        ticker, period, interval, label
    )
    key = (ticker, interval)
    if stop_reason is not None:
        _LOAD_STOP_REASONS[key] = stop_reason
        warn_key = (interval, stop_reason.get("code"))
        if warn_key not in _LOAD_STOP_WARNED:
            _LOAD_STOP_WARNED.add(warn_key)
            st.warning(stop_reason["message"])
    else:
        _LOAD_STOP_REASONS.pop(key, None)
    return result


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


def _save_trained_model(model_path, payload):
    """学習済みモデルを保存し、成功したら True、失敗したら False を返す。

    保存失敗の事実だけを戻り値で知らせる。例外の詳細や保存先パスは
    呼び出し側・画面へ伝えない（露出防止）。保存形式・保存先は従来
    どおりで、成功時の動作は変わらない。
    """
    try:
        joblib.dump(payload, model_path)
        return True
    except Exception:
        return False


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

        # Retrain with all eligible labelled rows. Only class-balancing sample weights are used; trade outcome feedback is stored separately and is not used in model fitting.
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
                saved_ok = _save_trained_model(model_path, {
                    "version": MODEL_VERSION, "features": FEATURE_COLUMNS,
                    "model": model, "trained_at": datetime.now(JST).isoformat(),
                    "validation_balanced_accuracy": balanced_accuracy,
                    "baseline_balanced_accuracy": baseline_balanced_accuracy,
                    "feedback_applied": 0})
                # 保存の成否だけを記録する（例外詳細・パスは保持しない）。
                # 表示はメイン表示領域とサイドバーのモデル状態表示で行う。
                st.session_state["model_save_status"] = {
                    "ticker": ticker,
                    "interval": interval,
                    "saved": bool(saved_ok),
                }
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
st.caption("単発トレードの方向判断と、リピート注文（ハーフ＆ハーフ）の設定支援専用です。口座接続・自動発注は行いません。通貨ペアと時間足を選び、方向とデータ鮮度を確認してください。")

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

st.sidebar.header("📦 注文数量")
st.sidebar.number_input("1注文の基本数量 (万通貨)", min_value=0.01, step=0.01, key="quantity_wan", on_change=save_settings, help="リピート注文の1本あたり数量など、数量計算の基本値です。")
st.sidebar.markdown("---")
trade_style = str(st.session_state.get("trade_style", "デイトレード（当日決済）"))
st.sidebar.markdown("---")
with st.sidebar.expander("🔄 AIモデル管理", expanded=False):
    if st.button("🧠 AIモデルを今すぐ強制再学習"):
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
        st.caption(f"🤖 最終学習日時: `{m_time}`")
        if model_age_hours > 4:
            st.warning("モデルが4時間以上経過しています。次回分析時に再学習されます。")
        else:
            st.caption(f"モデル経過時間: 約 {model_age_hours:.1f} 時間")
        st.caption("※再学習はアプリが実行・再分析された時に行われます。取引結果の記録だけでは学習されません。")
    else:
        st.caption("🤖 現在のモデル状態: `初回自動作成待ち`")
        st.caption("※十分な価格データを取得できた次回分析時にモデルを作成します。")
_save_status = st.session_state.get("model_save_status") or {}
if (
    _save_status.get("saved") is False
    and _save_status.get("ticker") == ticker
    and _save_status.get("interval") == tf_config["interval"]
):
    st.sidebar.warning(
        "学習したモデルを保存できませんでした。"
        "今回の分析には利用できますが、アプリを再起動するとモデルは残りません。"
    )
with st.sidebar.expander("🛠️ 更新・メンテナンス", expanded=False):
    st.checkbox(
        "5分ごとに自動更新（画面維持）",
        key="auto_refresh",
        on_change=save_settings,
        help="対応するStreamlitではメイン表示領域だけを5分ごとに再実行します。AI分析も再計算される場合があります。初期設定はオフです。",
    )
    st.info(
        "自動更新は初期設定でオフです。オンにすると対応環境ではメイン表示領域だけを5分ごとに更新します。 "
        "ニュースは15分キャッシュ、価格データは60秒キャッシュです。自動発注は行いません。"
    )
    if st.button("💾 設定を即時手動保存"):
        save_settings()
        st.toast("設定を保存しました！", icon="💾")
    st.markdown("**📂 設定ファイル管理 (バックアップ)**")
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
        help="設定、予測検証履歴、注文候補の検証履歴、候補後の自動採点、実取引記録、フィードバックを保存します。Webhook URLは含めません。",
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
    st.button("🔄 設定を初期値にリセット", on_click=reset_settings_callback)
    if st.button("🔄 最新データに更新"):
        st.cache_data.clear()
        st.rerun()
    st.caption("更新ボタンは価格・ニュースのキャッシュを消して再取得します。選択中の画面は維持されます。連続して押す必要はありません。")

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
    if data is None or data.empty:
        st.error("データの取得に失敗したか、分析に必要な本数が不足しています。時間をおいて更新してください。")
        st.stop()

    last_bar_index = pd.Timestamp(data.index[-1])
    last_bar_jst = safe_to_tokyo_tz(data).index[-1].strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<div class="update-time">最終確定足の時刻: <b>{last_bar_jst} JST</b> (最新の未確定足は除外。取得時刻とは異なります)</div>', unsafe_allow_html=True)
    # 段階1（016）: 1時間足の入力検証カウンタ（C-1）をデータ情報の近くに表示する。
    _data_quality = data.attrs.get("data_quality") if hasattr(data, "attrs") else None
    if _data_quality:
        st.caption(
            "データ品質（入力検証）: "
            f"入力 {_data_quality['input_rows']}件 / "
            f"除外 時刻不正 {_data_quality['dropped_invalid_timestamp']}件・"
            f"欠損/非有限 {_data_quality['dropped_missing_or_nonfinite']}件・"
            f"重複時刻 {_data_quality['dropped_duplicate_timestamp']}件・"
            f"OHLC不整合 {_data_quality['dropped_invalid_ohlc']}件・"
            f"特徴量不足 {_data_quality['dropped_feature_nan']}件 / "
            f"特徴量行 {_data_quality['feature_rows']}件 / "
            f"ラベルなし（未来窓不足） {_data_quality['dropped_label_insufficient_future']}件 / "
            f"学習対象 {_data_quality['trainable_rows']}件"
        )
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
    # モデル保存に失敗した場合は、その場でも知らせる（サイドバーにも
    # 次回描画時に残る）。例外詳細・保存先パスは表示しない。
    _save_status_main = st.session_state.get("model_save_status") or {}
    if (
        _save_status_main.get("saved") is False
        and _save_status_main.get("ticker") == ticker
        and _save_status_main.get("interval") == tf_config["interval"]
    ):
        st.warning(
            "学習したモデルを保存できませんでした。"
            "今回の分析には利用できますが、アプリを再起動するとモデルは残りません。"
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
    if mtf_gate_reason and data_htf is None and get_load_stop_reason(ticker, "1d"):
        mtf_gate_reason += "なお、日足データは時刻情報を確認できないため停止中です（タイムゾーン未確認）。"
    # Save the final gated decision only when the selected price series passed freshness checks.
    # Stale/unknown timestamps must not create apparently actionable forecast records.
    if not data_is_stale and freshness_check_ok:
        record_forecast_and_update(data, ticker, tf_config["interval"], status, prob_dict, LOOKAHEAD_BARS)
        # Continue evaluating previously saved candidate-price records even though
        # their history display was removed from the simplified UI. This preserves
        # forward outcome tracking without creating candidates or changing signals.
        update_entry_tracking_outcomes(
            load_entry_tracking(), data, ticker, tf_config["interval"], LOOKAHEAD_BARS
        )

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
        # 画面簡素化のためバナー表示はしない。通知送信（内部処理）のみ継続する。
        webhook_url = st.session_state.get("webhook_url", "")
        if webhook_url:
            key = f"last_webhook_{ticker}_{tf_config['interval']}"
            time_key = key + "_time"
            current_time = time.time()
            if st.session_state.get(key) != vol_alert_msg or current_time - st.session_state.get(time_key, 0) > 1800:
                if send_webhook_alert(webhook_url, f"FX参考アラート {selected_label}\n{vol_alert_msg}"):
                    st.session_state[key] = vol_alert_msg
                    st.session_state[time_key] = current_time

    # ==== 画面セクション: 単発トレード（方向判断） ====
    # 方向は4時間足・1時間足のEMA20/50トレンドの一致で示す。AI判定・確率・
    # 一致率などの分析数値や、数値の価格目標はこの画面には出さない
    # （内部の分析・予測記録・フォワード評価は従来どおり動作する）。
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

    def _series_fresh(frame, interval):
        """方向判断に使う系列の鮮度を、選択中の時間足と同じ基準で個別に確認する。

        鮮度が「確認できた場合のみ」True。データなし・時刻のタイムゾーン
        不明・検証例外・基準超過はいずれも False（=判断保留側に倒す）。
        """
        try:
            if frame is None or frame.empty:
                return False
            last_idx = frame.index[-1]
            if last_idx.tzinfo is None:
                return False
            age = max(
                0.0,
                (pd.Timestamp.now(tz="UTC") - last_idx.tz_convert("UTC")).total_seconds() / 60.0,
            )
            limits = {"5m": 25, "15m": 60, "1h": 210, "4h": 600, "1d": 2160}
            return age <= limits.get(interval, 210)
        except Exception:
            return False

    st.markdown("---")
    st.markdown("## 🎯 単発トレード")
    _core_trends = [(name, *_trend_summary(frame)) for name, frame in (("4時間足", data_4h), ("1時間足", data_1h))]
    # 方向判断に使う4時間足・1時間足の鮮度を、それぞれ独立に検証する。
    # 選択中の時間足が新しくても、片方でも鮮度を確認できなければ保留する。
    if not (_series_fresh(data_4h, "4h") and _series_fresh(data_1h, "1h")):
        _single_direction = "判断保留"
        _single_reason = "4時間足・1時間足のデータ鮮度が確認できないため、方向判断を保留します。"
    elif any(sign is None for _, _, sign in _core_trends):
        _single_direction = "判断保留"
        _single_reason = "方向判断に使う時間足のデータが不足しているため、保留します。"
    elif all(sign == 1 for _, _, sign in _core_trends):
        _single_direction = "買い方向"
        _single_reason = None
    elif all(sign == -1 for _, _, sign in _core_trends):
        _single_direction = "売り方向"
        _single_reason = None
    else:
        _single_direction = "判断保留"
        _single_reason = "4時間足と1時間足の方向が一致していない（または中立・混在の）ため、保留します。"
    if _single_direction == "買い方向":
        st.success("方向：買い方向（上昇トレンド）")
    elif _single_direction == "売り方向":
        st.markdown('<div class="badge-sell">方向：売り方向（下降トレンド）</div>', unsafe_allow_html=True)
    else:
        st.warning("方向：判断保留")
        st.caption(_single_reason)
    for _tf_name, _tf_text, _tf_sign in _core_trends:
        st.markdown(f"- {_tf_name}： {_tf_text}")
    st.caption("方向は4時間足・1時間足の終値とEMA20/50の位置関係によるトレンド確認です。価格データはYahoo Financeの確定足をもとにしており、リアルタイム価格としては表示していません。")
    st.caption("数値のエントリー価格・決済価格は表示しません。自動発注は行いません。実際の注文は証券会社の画面で確認してください。")


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
    st.markdown("---")
    st.markdown("## 📋 リピート注文（ハーフ＆ハーフ）")
    st.caption("広めの設定レンジを基本に、買いレンジ・売りレンジ、注文幅／利確幅、1注文数量、注文本数の目安を確認します。")
    with st.expander("⚙️ リピート注文のレンジ設定", expanded=False):
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


    st.caption("参考計算です。証券会社の仕様・保有ポジション・実際の証拠金計算を完全には再現しません。")
    st.caption("レンジが広くても、相場状況によってはレンジ外になる可能性があります。完全に防げるものではありません。")
    if latest_price < user_lower or latest_price > user_upper:
        st.markdown(f'<div class="warn-box">【レンジアウト警告】直近の確定足は設定レンジ <code>{price_fmt%user_lower}</code>～<code>{price_fmt%user_upper}</code> の外側です。</div>',unsafe_allow_html=True)
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
    trap_width_pips=st.number_input("注文幅 / 利確幅 (pips)",min_value=10,max_value=500,step=5,key=trap_key,on_change=update_trap_callback)
    st.button(f"ATR参考値 ({ai_rec_trap_pips} pips) を適用",on_click=apply_ai_trap_callback)
    half_range_pips = abs(user_upper-user_half)/pip_unit
    spacing_pips = max(1.0, float(trap_width_pips))
    half_grid_count = max(1, int(np.floor(half_range_pips / spacing_pips)) + 1)
    total_grid_count = half_grid_count * 2
    quantity_wan_val = max(0.0, float(st.session_state["quantity_wan"]))
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
    st.markdown(f"""<div class="param-box"><b>【ハーフ＆ハーフ設定値の目安】</b>（4時間足ATR基準）<br>
    ・買い設定（下半）: <code>{price_fmt%user_lower}</code>～<code>{price_fmt%user_half}</code> | 停止目安: <code>{price_fmt%buy_stop_loss}</code><br>
    ・売り設定（上半）: <code>{price_fmt%user_half}</code>～<code>{price_fmt%user_upper}</code> | 停止目安: <code>{price_fmt%sell_stop_loss}</code><br>
    ・注文幅/利確幅: <code>{trap_width_pips} pips</code>（ATR参考値 {ai_rec_trap_pips} pips） | 片側注文本数目安: <code>{half_grid_count}</code>（合計 {total_grid_count}）</div>""",unsafe_allow_html=True)
    st.code(f"""[松井証券リピート注文 設定値]
通貨ペア: {selected_label}
注文種別: ハーフ＆ハーフ
買いレンジ: {price_fmt%user_lower} - {price_fmt%user_half} (停止目安: {price_fmt%buy_stop_loss})
売りレンジ: {price_fmt%user_half} - {price_fmt%user_upper} (停止目安: {price_fmt%sell_stop_loss})
注文幅 / 利確幅: {trap_width_pips} pips
1本あたり数量: {quantity_wan_val:.2f} 万通貨""",language="text")
    st.markdown("##### 🛡️ リスク・資金シミュレーション")
    rc1,rc2=st.columns(2)
    rc1.metric("想定含み損シナリオ", f"約 {int(max_loss_yen):,} 円" if risk_calc_available else "算出不可")
    rc2.metric("全注文成立時の必要証拠金概算", f"約 {int(total_margin_yen):,} 円" if risk_calc_available else "算出不可")
    if not risk_calc_available:
        st.error("USD/JPY換算レートを取得できないため、円換算のリスク試算は停止しています。レート取得後に再更新してください。")
    else:
        st.info("この単純試算だけでは安全性を判断できません。想定損失と必要証拠金の概算として確認してください。")
    st.markdown("---")
    st.markdown("##### 📌 損益見通しについて")
    st.caption(
        "ATRだけから月間の約定回数・利益を信頼性高く予測することはできないため、"
        "根拠の弱い月間・年間利益予測は表示しません。上の想定損失・必要証拠金の概算を確認し、"
        "松井証券FXの最新条件と注文画面で最終確認してください。"
    )

    # 手動記録は通常画面を2機能に保つため、初期状態で閉じた領域に隔離する。
    with st.expander("🗂️ 取引記録・振り返り（必要なときだけ）", expanded=False):
        st.caption(
            "以下は任意の記録機能です。通常画面には表示しません。"
            "利益・損失のフィードバックや実取引損益は市場ラベル・AI再学習データに自動投入しません。"
        )
        st.markdown("##### 📝 分析結果へのフィードバック")
        st.caption(
            "現在の足と実際の取引条件が一致するとは限らないため、"
            "参考メモとして保存します。"
        )
        feedback_ts = _normalise_timestamp(data.index[-1])

        def record_feedback(is_win):
            fb_list = load_feedback_data()
            existing = [
                x for x in fb_list
                if x.get("ticker") == ticker
                and x.get("interval") == tf_config["interval"]
                and _normalise_timestamp(x.get("timestamp")) == feedback_ts
            ]
            if existing:
                for item in fb_list:
                    if (
                        item.get("ticker") == ticker
                        and item.get("interval") == tf_config["interval"]
                        and _normalise_timestamp(item.get("timestamp")) == feedback_ts
                    ):
                        item["is_win"] = bool(is_win)
                        item["updated_at"] = datetime.now(JST).isoformat()
            else:
                fb_list.append({
                    "ticker": ticker,
                    "interval": tf_config["interval"],
                    "timestamp": feedback_ts,
                    "is_win": bool(is_win),
                    "signal": status,
                    "reference_close": latest_price,
                    "created_at": datetime.now(JST).isoformat(),
                })
            save_feedback_data(fb_list)

        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            if st.button("👍 利益になった結果を記録", use_container_width=True, key="record_feedback_win"):
                record_feedback(True)
                st.success("結果を記録しました。AIの再学習データには自動投入しません。")
        with fb_col2:
            if st.button("👎 損失になった結果を記録", use_container_width=True, key="record_feedback_loss"):
                record_feedback(False)
                st.warning("結果を記録しました。AIの再学習データには自動投入しません。")

        st.markdown("##### 📒 実取引の損益記録")
        st.caption(
            "証券会社の約定履歴・実現損益を見ながら手入力してください。"
            "保存した結果はAIの再学習には自動投入しません。"
        )
        journal_col1, journal_col2 = st.columns(2)
        with journal_col1:
            journal_pair = st.selectbox(
                "記録する通貨ペア", list(PAIRS.keys()),
                index=list(PAIRS.keys()).index(selected_label), key="journal_pair"
            )
            journal_side = st.radio(
                "売買方向", ["買い", "売り"], horizontal=True, key="journal_side"
            )
            journal_qty = st.number_input(
                "約定数量（万通貨）", min_value=0.01, max_value=1000.0,
                value=float(st.session_state.get("quantity_wan", 0.10)),
                step=0.01, key="journal_qty"
            )
        with journal_col2:
            journal_entry = st.number_input(
                "実際の新規約定価格", min_value=0.0, value=0.0,
                step=0.01 if "JPY" in PAIRS[journal_pair] else 0.0001,
                format="%.3f" if "JPY" in PAIRS[journal_pair] else "%.5f",
                key="journal_entry"
            )
            journal_exit = st.number_input(
                "実際の決済約定価格", min_value=0.0, value=0.0,
                step=0.01 if "JPY" in PAIRS[journal_pair] else 0.0001,
                format="%.3f" if "JPY" in PAIRS[journal_pair] else "%.5f",
                key="journal_exit"
            )
            journal_pnl = st.number_input(
                "実現損益（円・手数料等反映後）", value=0.0,
                step=100.0, key="journal_pnl",
                help="実際の実現損益を入力してください。利益はプラス、損失はマイナスです。"
            )
        journal_note = st.text_input(
            "メモ（任意）", max_chars=300, key="journal_note",
            placeholder="例：経済指標前、成行、予定より早く決済"
        )
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
            pnl_values = pd.to_numeric(
                journal_df.get("realized_pnl_yen", pd.Series(dtype=float)),
                errors="coerce"
            ).fillna(0)
            wins = int((pnl_values > 0).sum())
            settled = int((pnl_values != 0).sum())
            jm1, jm2, jm3 = st.columns(3)
            jm1.metric("記録件数", f"{len(journal_df)}件")
            jm2.metric("実現損益合計", f"{pnl_values.sum():+,.0f}円")
            jm3.metric(
                "利益取引率", f"{wins / settled * 100:.1f}%" if settled else "—",
                help="±0円の取引は分母から除外。過去記録の集計であり将来の勝率を示しません。"
            )
            journal_view = journal_df.rename(columns={
                "recorded_at": "記録日時", "pair": "通貨ペア", "side": "方向",
                "quantity_wan": "数量(万通貨)", "entry_price": "新規価格",
                "exit_price": "決済価格", "realized_pnl_yen": "実現損益(円)",
                "result": "結果", "note": "メモ"
            })
            display_cols = [
                c for c in [
                    "記録日時", "通貨ペア", "方向", "数量(万通貨)",
                    "新規価格", "決済価格", "実現損益(円)", "結果", "メモ"
                ] if c in journal_view.columns
            ]
            st.dataframe(
                journal_view[display_cols].tail(30),
                use_container_width=True, hide_index=True
            )
            st.download_button(
                "📥 実取引記録をCSV保存",
                data=journal_view[display_cols].to_csv(index=False).encode("utf-8-sig"),
                file_name="actual_trade_journal.csv", mime="text/csv",
                key="download_actual_trade_journal"
            )
        else:
            st.info("まだ実取引記録はありません。決済後に実現損益を入力すると履歴を確認できます。")


# Streamlit fragments keep the sidebar/navigation stable during periodic updates.
# On older Streamlit versions, fall back to normal manual reruns rather than failing to start.
if hasattr(st, "fragment"):
    _fragment = st.fragment(run_every=300 if (st.session_state.get("auto_refresh", False) or st.session_state.get("auto_pair_scan", False)) else None)(_render_main_area)
    _fragment()
else:
    if st.session_state.get("auto_refresh", False) or st.session_state.get("auto_pair_scan", False):
        st.sidebar.warning("現在のStreamlitは部分自動更新に未対応です。自動更新・定期スキャンは手動操作時のみ実行されます。Streamlitの更新をご検討ください。")
    _render_main_area()