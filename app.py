# app.py
# Pro FX Analyzer & Signal Pro
# 価格分析・注文補助専用です。証券口座接続・自動発注は行いません。
# 予測確率・バックテスト結果は将来の利益や実際の勝率を保証しません。

import html
import json
import os
import time
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
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

try:
    import requests
except Exception:
    requests = None

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="Pro FX Analyzer & Signal Pro", page_icon="📈", layout="wide")

SETTINGS_FILE = "user_settings.json"
FEEDBACK_FILE = "trade_feedback.json"
JST = ZoneInfo("Asia/Tokyo")

DEFAULT_SETTINGS = {
    "account_balance": 500000,
    "quantity_wan": 0.10,
    "auto_refresh": False,
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
}
DEFAULT_SPREAD_PIPS = {
    "USDJPY=X": 0.2, "EURUSD=X": 0.4, "GBPJPY=X": 1.0,
    "EURJPY=X": 0.5, "AUDJPY=X": 0.6,
}
LOOKAHEAD_BARS = 8
MODEL_VERSION = 4

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
    if not isinstance(result.get("ranges"), dict):
        result["ranges"] = {}
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
    text = f"{title} {summary}".lower()
    high_terms = ["fomc", "interest rate", "inflation", "cpi", "payroll", "jobs report",
                  "central bank", "boj", "bank of japan", "fed", "intervention", "gdp"]
    if any(x in text for x in high_terms):
        return "影響度：高", "材料の重要度が高い可能性があります", "方向は本文と実際の結果を確認"
    medium_terms = ["yen", "dollar", "treasury", "bond", "economy", "employment", "trade"]
    if any(x in text for x in medium_terms):
        return "影響度：中", "為替関連キーワードを検出", "方向は本文と市場反応を確認"
    return "影響度：低", "関連度は限定的な可能性", "方向判定なし"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_and_impact(ticker):
    items = []
    try:
        d = yf.Ticker(ticker)
        news = d.news or []
        for n in news[:12]:
            content = n.get("content", n) if isinstance(n, dict) else {}
            title = content.get("title") or n.get("title") or "タイトルなし"
            publisher = content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else n.get("publisher", "")
            pubdate = content.get("pubDate") or n.get("providerPublishTime")
            try:
                if isinstance(pubdate, (int, float)):
                    time_str = datetime.fromtimestamp(pubdate, tz=JST).strftime("%m/%d %H:%M")
                elif pubdate:
                    time_str = str(pubdate)[:16]
                else:
                    time_str = "時刻不明"
            except Exception:
                time_str = "時刻不明"
            canonical = content.get("canonicalUrl", {}) if isinstance(content.get("canonicalUrl"), dict) else {}
            link = canonical.get("url") or n.get("link") or "#"
            impact, reason, direction = _impact_from_text(str(title), str(content.get("summary", "")))
            items.append({"title": str(title), "publisher": str(publisher or "配信元不明"),
                          "time": time_str, "link": str(link), "impact": impact,
                          "reason": reason, "direction": direction})
    except Exception:
        return []
    return items


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
            model = RandomForestClassifier(
                n_estimators=260, max_depth=7, min_samples_leaf=8,
                class_weight="balanced_subsample", random_state=43, n_jobs=-1
            )
            # Trade outcome buttons do not contain enough information to alter a market label
            # safely (trade side, entry/exit, and timing may differ from the current bar).
            # Keep them as a review log; train only on reproducible first-touch market labels.
            sw = compute_sample_weight(class_weight="balanced", y=y)
            model.fit(X, y, sample_weight=sw)
            try:
                joblib.dump({"version": MODEL_VERSION, "features": FEATURE_COLUMNS,
                             "model": model, "trained_at": datetime.now(JST).isoformat(),
                             "feedback_applied": 0}, model_path)
            except Exception:
                pass

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
        if buy_p >= 0.53 and buy_p > sell_p and rsi < 78:
            status = "BUY (買い)"
        elif sell_p >= 0.53 and sell_p > buy_p and rsi > 22:
            status = "SELL (売り)"
        else:
            status = "WAIT (様子見)"
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

st.title("Pro FX Analyzer & Signal Pro")
st.caption("AI分析・注文補助・リピート設定をまとめた参考ツールです。実際の発注は行いません。")

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    selected_label = st.selectbox("通貨ペア", list(PAIRS.keys()), key="selected_pair_label")
with col_sel2:
    tf_label = st.selectbox("時間足", list(TIMEFRAMES.keys()), key="selected_tf_label")
ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]
is_jpy = "JPY" in ticker
pip_unit = 0.01 if is_jpy else 0.0001
price_fmt = "%.3f" if is_jpy else "%.5f"
spread_pips = DEFAULT_SPREAD_PIPS.get(ticker, 0.5)

st.sidebar.header("⚙️ 資金 & リスク設定")
st.sidebar.number_input("口座資金 (円)", min_value=10000, step=50000, key="account_balance", on_change=save_settings)
st.sidebar.number_input("1注文の基本数量 (万通貨)", min_value=0.01, step=0.01, key="quantity_wan", on_change=save_settings)
st.sidebar.markdown("---")
st.sidebar.header("🎯 単発トレード リスク設定")
st.sidebar.number_input("1取引あたりの許容リスク (%)", min_value=0.1, max_value=5.0, step=0.1, key="risk_percent", on_change=save_settings)
st.sidebar.number_input("デフォルト リスクリワード比 (RR)", min_value=0.5, max_value=5.0, step=0.1, key="rr_ratio", on_change=save_settings)
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
    m_time = datetime.fromtimestamp(os.path.getmtime(m_file_info), tz=JST).strftime("%m/%d %H:%M")
    st.sidebar.caption(f"🤖 最終学習日時: `{m_time}`")
else:
    st.sidebar.caption("🤖 現在のモデル状態: `初回自動作成待ち`")
st.sidebar.markdown("---")
st.sidebar.header("🔄 更新・メンテナンス")
if HAS_AUTOREFRESH:
    st.sidebar.checkbox("60秒ごとに自動更新", key="auto_refresh", on_change=save_settings)
    if st.session_state["auto_refresh"]:
        st_autorefresh(interval=60000, key="datarefresh")
else:
    st.sidebar.caption("💡 `pip install streamlit-autorefresh` で自動更新有効")
with st.sidebar.expander("🔔 通知設定 (Webhook)", expanded=False):
    st.text_input("Webhook URL (Discord等)", key="webhook_url", on_change=save_settings, placeholder="https://discord.com/api/webhooks/...")
    st.caption("URLはローカル設定ファイルに保存されます。共有環境では入力しないでください。")
if st.sidebar.button("💾 設定を即時手動保存"):
    save_settings()
    st.toast("設定を保存しました！", icon="💾")
with st.sidebar.expander("📂 設定ファイル管理 (バックアップ)", expanded=False):
    safe_backup_settings = load_settings().copy()
    safe_backup_settings["webhook_url"] = ""
    st.download_button("📥 設定をバックアップ (.json)", data=json.dumps(safe_backup_settings, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="fx_analyzer_settings.json", mime="application/json", use_container_width=True)
    uploaded_file = st.file_uploader("📤 設定を復元 (アップロード)", type=["json"])
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

with st.spinner("最新相場データ & ニュースを取得中..."):
    data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
    data_4h = load_and_process_data(ticker, "90d", "4h", "4時間足 (中期・リピート用)")
    data_htf = load_and_process_data(ticker, "2y", "1d", "日足")
    news_items = fetch_news_and_impact(ticker)
if data_4h is None and data is not None:
    data_4h = data
if data is None or data.empty:
    st.error("データの取得に失敗したか、分析に必要な本数が不足しています。時間をおいて更新してください。")
    st.stop()

last_bar_jst = safe_to_tokyo_tz(data).index[-1].strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f'<div class="update-time">最終確定足の時刻: <b>{last_bar_jst} JST</b> (最新の未確定足は除外。取得時刻とは異なります)</div>', unsafe_allow_html=True)
latest_price = float(clean_series(data["Close"]).iloc[-1])
latest_atr = float(clean_series(data["ATR"]).iloc[-1]) if "ATR" in data else latest_price * 0.005
force_retrain = bool(st.session_state.pop("force_retrain_once", False))
status, conf, total_signals_count, win_rate, m_type, prob_dict, feature_importances, ai_signal_history = analyze_signal_with_backtest(
    data, data_htf, ticker, tf_config["interval"], force_retrain=force_retrain
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
    st.markdown(f'<div class="alert-box">{html.escape(vol_alert_msg)}</div>', unsafe_allow_html=True)
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
st.caption("分類一致率は、過去のテスト区間でモデルの3クラス分類が一致した割合です。実際のトレード勝率ではありません。")
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

tab_chart, tab_single, tab_news, tab_repeat, tab_ai = st.tabs([
    "📈 メインチャート", "⚡ 単発トレード (AI全自動アシスト)",
    "📰 経済指標＆為替ニュースAI予測", "📋 松井証券 リピート設定 & リスク管理",
    "🤖 AI分析詳細 & 自己学習ノート"
])

with tab_chart:
    st.markdown("##### 📈 インタラクティブ・メインチャート")
    st.caption("ボタンで表示本数を切替。十字カーソルとマウスホイールでズームできます。")
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
        sub_indicator = st.selectbox("サブ指標", ["RSI (14)", "MACD", "なし"], index=0, key="sub_indicator_select", label_visibility="collapsed")
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
    fig.update_layout(xaxis_rangeslider_visible=False,height=580,margin=dict(l=10,r=70,t=25,b=10),template="plotly_dark",
                      hovermode="x unified",showlegend=True,legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1))
    fig.update_xaxes(type="date",rangebreaks=[dict(bounds=["sat","mon"])] if tf_config["interval"] in ("5m","15m","1h") else [],showspikes=True,spikemode="across")
    fig.update_yaxes(range=[chart_low-span*.05,chart_high+span*.05],side="right",tickformat=".3f" if is_jpy else ".5f",row=1,col=1)
    st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"displayModeBar":True,"displaylogo":False})

with tab_single:
    st.markdown("##### ⚡ 単発トレード（スキャル・デイトレ）AI発注アシスタント")
    st.caption("AIの分析結果から注文条件の参考値を計算します。発注は行いません。")
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
    ai_entry = latest_price
    ai_is_buy = ai_direction.startswith("BUY")
    ai_sl_price = ai_entry - ai_sl_pips*pip_unit if ai_is_buy else ai_entry + ai_sl_pips*pip_unit
    ai_tp_price = ai_entry + ai_tp_pips*pip_unit if ai_is_buy else ai_entry - ai_tp_pips*pip_unit
    ai_loss_per_wan = (ai_sl_pips + spread_pips) * pip_val_per_wan
    raw_ai_wan = max_risk_yen / ai_loss_per_wan if ai_loss_per_wan > 0 else 0
    floored_ai_wan = np.floor(raw_ai_wan*100)/100
    ai_rec_wan = float(floored_ai_wan) if floored_ai_wan >= .01 else 0.0
    is_risk_exceeded = ai_rec_wan <= 0 or pip_val_per_wan <= 0
    ai_profit_yen = max(0.0, (ai_tp_pips-spread_pips)*pip_val_per_wan*ai_rec_wan)
    ai_loss_yen = (ai_sl_pips+spread_pips)*pip_val_per_wan*ai_rec_wan
    req_margin_yen = ((ai_entry*ai_rec_wan*10000)/25 if is_jpy else (ai_entry*usd_rate*ai_rec_wan*10000)/25) if usd_rate else 0
    st.markdown("#### 🤖 AI提案トレードプラン")
    if ai_direction.startswith("WAIT"):
        st.warning("現在はAI判定がWAIT（様子見）です。推奨方向・注文数量を提示しません。")
    elif is_risk_exceeded:
        st.error("安全な最小数量を許容リスク内に収められないか、換算データが不足しています。今回は見送り候補です。")
    else:
        st.info(f"参考シグナル: {status}（モデル出力確信度 {conf:.1f}%、分類一致率 {win_rate:.1f}%）")
    if ai_direction.startswith("WAIT") or is_risk_exceeded:
        st.markdown(f'<div class="param-box"><b>【注文補助】</b><br>売買方向: <code>WAIT（見送り）</code><br>推奨数量: <code>0.00 万通貨</code><br>確定足価格: <code>{price_fmt % ai_entry}</code></div>',unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="param-box"><b>【注文コピー用パラメータ】</b>（想定スプレッド {spread_pips:.1f} pips）<br>
        ・通貨ペア: <code>{html.escape(selected_label)}</code> | 方向: <code>{"買い" if ai_is_buy else "売り"}</code><br>
        ・新規価格目安: <code>{price_fmt % ai_entry}</code><br>・利確(TP): <code>{price_fmt % ai_tp_price}</code> (+{ai_tp_pips:.1f} pips / 純益概算 +{int(ai_profit_yen):,}円)<br>
        ・損切(SL): <code>{price_fmt % ai_sl_price}</code> (-{ai_sl_pips:.1f} pips / 損失概算 -{int(ai_loss_yen):,}円)<br>
        ・数量: <code>{ai_rec_wan:.2f} 万通貨</code> ({int(ai_rec_wan*10000):,}通貨) | 必要証拠金概算: <code>{int(req_margin_yen):,}円</code></div>""",unsafe_allow_html=True)

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
    st.markdown("---")
    st.markdown("##### 🛠️ 注文条件の手動微調整")
    sc1, sc2 = st.columns(2)
    with sc1:
        trade_side = st.radio("売買方向",["BUY (買い)","SELL (売り)"],index=0 if status.startswith("BUY") else 1,horizontal=True)
        entry_price = st.number_input("エントリー想定レート",value=float(latest_price),format=price_fmt,step=0.01 if is_jpy else 0.0001)
    with sc2:
        sl_mode = st.radio("損切(SL)の決め方",["ATRベース (推奨)","固定 pips"],horizontal=True)
        if sl_mode == "ATRベース (推奨)":
            atr_multiplier = st.slider("ATR倍率",0.5,3.0,1.5,0.1)
            sl_pips = max(1.0,round(latest_atr/pip_unit*atr_multiplier,1))
            st.info(f"現在のATRから算出した損切幅: **{sl_pips:.1f} pips**")
        else:
            sl_pips = st.number_input("損切幅 (pips)",min_value=1.0,value=float(ai_sl_pips),step=1.0)
    rr_value = st.slider("リスクリワード比 (RR)",0.5,4.0,value=float(min(4,max(.5,st.session_state["rr_ratio"]))),step=.1)
    tp_pips = round(sl_pips*rr_value,1)
    is_buy_manual = trade_side.startswith("BUY")
    sl_price = entry_price-sl_pips*pip_unit if is_buy_manual else entry_price+sl_pips*pip_unit
    tp_price = entry_price+tp_pips*pip_unit if is_buy_manual else entry_price-tp_pips*pip_unit
    loss_per_wan = (sl_pips+spread_pips)*pip_val_per_wan
    raw_manual_wan = max_risk_yen/loss_per_wan if loss_per_wan>0 else 0
    manual_wan = float(np.floor(raw_manual_wan*100)/100) if np.floor(raw_manual_wan*100)/100>=.01 else 0.0
    expected_profit_yen = (tp_pips-spread_pips)*pip_val_per_wan*manual_wan
    actual_loss_yen = (sl_pips+spread_pips)*pip_val_per_wan*manual_wan
    res_col1,res_col2,res_col3,res_col4=st.columns(4)
    res_col1.metric("調整後の手動ロット数",f"{manual_wan:.2f} 万通貨",f"許容リスク上限 {risk_pct}%")
    res_col2.metric("損切(SL) レート",price_fmt%sl_price,f"-{sl_pips:.1f} pips")
    res_col3.metric("利確(TP) レート",price_fmt%tp_price,f"+{tp_pips:.1f} pips")
    res_col4.metric("想定純損益",f"{expected_profit_yen:+,.0f}円",f"最大損失 -{actual_loss_yen:,.0f}円")
    if expected_profit_yen <= 0:
        st.warning("利確幅が想定スプレッド以下のため、コスト控除後の利益が見込めない計算です。")
    st.caption("手動調整欄は計算補助です。入力値・スプレッド・約定価格によって実損益は変わります。")

with tab_news:
    st.markdown("##### 📰 リアルタイム為替ニュース & キーワード分析予測")
    st.caption("ニュース見出しのキーワード参考表示です。本文理解や実際の市場反応を保証するAI予測ではありません。")
    with st.expander("🔔 主要経済指標と為替への影響パターン",expanded=True):
        st.markdown("""
        | 経済指標・イベント | 注目ポイント | ドル円への一般的な影響例 |
        | :--- | :--- | :--- |
        | **FOMC** | 政策金利・議長発言 | タカ派はドル高要因、ハト派はドル安要因になり得る |
        | **米雇用統計 (NFP)** | 雇用者数・平均時給 | 予想との差と金利見通しが重要 |
        | **米CPI** | インフレ率 | 予想との差と金融政策見通しが重要 |
        | **日銀会合** | 政策金利・総裁会見 | 政策修正は円相場を動かす場合がある |
        """)
    st.markdown("---")
    st.markdown(f"##### 🌐 `{selected_label}` 関連のニュース & キーワード参考判定")
    if not news_items:
        st.info("関連ニュースを取得できませんでした。")
    else:
        for news in news_items:
            st.markdown(f"""<div class="news-card">
            <div style="display:flex;justify-content:space-between;gap:10px;margin-bottom:6px;">
            <span style="font-weight:bold;color:#38bdf8;">{html.escape(str(news.get('publisher','')))} ({html.escape(str(news.get('time','')) )})</span>
            <span>{html.escape(str(news.get('impact','')))}</span></div>
            <div style="font-size:1.05rem;font-weight:800;margin-bottom:8px;">
            <a href="{html.escape(str(news.get('link','#')),quote=True)}" target="_blank" rel="noopener noreferrer" style="color:#f8fafc;text-decoration:none;">{html.escape(str(news.get('title','')))} 🔗</a></div>
            <div style="background:#0f172a;padding:10px 14px;border-radius:4px;line-height:1.6;">
            🎯 <b>キーワード参考判定</b>: <span style="color:#facc15;font-weight:bold;">{html.escape(str(news.get('direction','')))}</span><br>
            💡 <b>判定理由</b>: {html.escape(str(news.get('reason','')))}</div></div>""",unsafe_allow_html=True)

with tab_repeat:
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
    st.markdown("##### 📈 リターン・回転数シミュレーション（参考値）")
    st.caption("実際の利確回数を予測する根拠は限定的です。参考シナリオであり、利益予測として利用しないでください。")
    net_trap_pips = max(0.0, float(trap_width_pips) - spread_pips)
    estimated_monthly_turns = max(0, int((latest_atr / (max(1.0, float(trap_width_pips)) * pip_unit)) * 6))
    est_monthly_profit = estimated_monthly_turns * net_trap_pips * pip_value_yen
    est_annual_profit = est_monthly_profit * 12
    est_monthly_roi = est_monthly_profit / account_balance_val * 100 if account_balance_val > 0 else 0
    est_annual_roi = est_annual_profit / account_balance_val * 100 if account_balance_val > 0 else 0
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("仮定上の月間回数", f"{estimated_monthly_turns} 回")
    sc2.metric("仮定上の月間損益", f"{est_monthly_profit:,.0f} 円" if risk_calc_available else "算出不可")
    sc3.metric("仮定上の月間比率", f"{est_monthly_roi:.1f}%" if risk_calc_available else "算出不可")
    sc4.metric("単純年換算比率", f"{est_annual_roi:.1f}%" if risk_calc_available else "算出不可")

with tab_ai:
    st.markdown("##### 🤖 AI予測モデルの評価と内訳")
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
        top10_imp=feature_importances.nlargest(10).sort_values(ascending=True)
        fig_imp=go.Figure(go.Bar(x=top10_imp.values,y=[FEATURE_LABELS.get(x,x) for x in top10_imp.index],orientation="h"))
        fig_imp.update_layout(height=340,margin=dict(l=10,r=20,t=10,b=30),template="plotly_dark",xaxis_title="重要度スコア",yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp,use_container_width=True,config={"displayModeBar":False})
    st.markdown("---")
    st.markdown("##### 📝 簡易トレードノート（考察・反省メモ）")
    def update_memo_callback(): save_settings()
    st.text_area("トレードノートを入力",key="trade_memo",height=150,on_change=update_memo_callback,placeholder="例: 米雇用統計発表前のため見送り。")
    memo_text=st.session_state.get("trade_memo","")
    if memo_text.strip():
        st.download_button("📥 トレードノートをテキスト保存 (.txt)",data=memo_text.encode("utf-8"),
                           file_name=f"trade_memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",mime="text/plain",key="download_memo_btn")
