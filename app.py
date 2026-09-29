import json
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 0. 画面基本設定 & CSSデザイン定義（ハイコントラスト化）
# ==========================================
st.set_page_config(
    page_title="AI FX デイトレ & リピートアナライザー Pro v6.3.6",
    layout="wide",
    initial_sidebar_state="expanded",
)

# iPad・白背景/黒背景の双方に対応したハイコントラストCSS
st.markdown("""
<style>
    /* 全体フォント・余白調整 */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    
    /* 入力フォーム・セレクトボックスのラベル文字（濃色でくっきり表示） */
    label[data-testid="stWidgetLabel"] p {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* メトリクス（数値の上の項目名）の視認性向上 */
    [data-testid="stMetricLabel"] {
        font-size: 1.0rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    /* メトリクス（数値自体）の強調 */
    [data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    /* キャプション（注意書き・補足文）の同化防止 */
    [data-testid="stCaptionContainer"], .stCaption p {
        font-size: 0.95rem !important;
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* ステータスバッジ */
    .status-badge-buy {
        background-color: #15803d;
        color: #ffffff;
        border: 1px solid #16a34a;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.15rem;
        display: inline-block;
    }
    .status-badge-sell {
        background-color: #b91c1c;
        color: #ffffff;
        border: 1px solid #dc2626;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.15rem;
        display: inline-block;
    }
    .status-badge-hold {
        background-color: #854d0e;
        color: #ffffff;
        border: 1px solid #ca8a04;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.15rem;
        display: inline-block;
    }
    
    /* パラメータ表示ボックス（背景を暗色に固定し文字を白・水色で極小同化を防ぐ） */
    .param-box {
        background-color: #1e293b !important;
        border-left: 6px solid #3b82f6;
        padding: 18px;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #f8fafc !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .param-box b.label-title {
        color: #94a3b8 !important;
    }
    .param-box code {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 2px 8px !important;
        background-color: #334155 !important;
        color: #38bdf8 !important;
        border: 1px solid #475569;
        border-radius: 4px;
    }
    .param-box-buy { border-left-color: #22c55e !important; }
    .param-box-sell { border-left-color: #ef4444 !important; }

    /* タブのデザイン */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-size: 1.0rem !important;
        font-weight: 700;
        color: #334155 !important;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #0284c7 !important;
        border-bottom-color: #0284c7 !important;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_COLUMNS = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200", "Dev_EMA20_200", "Vol_Ratio",
    "RSI", "RSI_Diff", "MACD_Hist_Ratio", "BB_PctB", "ADX",
    "ATR_Ratio", "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Stoch_K"
]

SETTINGS_FILE = "user_settings.json"

DEFAULT_SETTINGS = {
    "account_balance": 200000,
    "quantity_wan": 0.20,
    "discord_url": "",
    "enable_notify": False,
    "auto_refresh": False,
    "refresh_interval": 180,
    "selected_pair_label": "米ドル / 円 (USD/JPY)",
    "selected_tf_label": "15分足 (デイトレエントリー用)",
    "last_notified_status": {},
}

# ==========================================
# 1. 設定ファイルの永続化 & 補助関数
# ==========================================
def load_user_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                merged = DEFAULT_SETTINGS.copy()
                merged.update(saved)
                return merged
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_user_settings():
    settings = {
        "account_balance": st.session_state.get("account_balance", DEFAULT_SETTINGS["account_balance"]),
        "quantity_wan": st.session_state.get("quantity_wan", DEFAULT_SETTINGS["quantity_wan"]),
        "discord_url": st.session_state.get("discord_url", DEFAULT_SETTINGS["discord_url"]),
        "enable_notify": st.session_state.get("enable_notify", DEFAULT_SETTINGS["enable_notify"]),
        "auto_refresh": st.session_state.get("auto_refresh", DEFAULT_SETTINGS["auto_refresh"]),
        "refresh_interval": st.session_state.get("refresh_interval", DEFAULT_SETTINGS["refresh_interval"]),
        "selected_pair_label": st.session_state.get("selected_pair_label", DEFAULT_SETTINGS["selected_pair_label"]),
        "selected_tf_label": st.session_state.get("selected_tf_label", DEFAULT_SETTINGS["selected_tf_label"]),
        "last_notified_status": st.session_state.get("last_notified_status", DEFAULT_SETTINGS["last_notified_status"]),
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

if "initialized" not in st.session_state:
    saved_settings = load_user_settings()
    for key, val in saved_settings.items():
        st.session_state[key] = val
    st.session_state["initialized"] = True

def send_discord_notification(webhook_url, title, message, color=0x00FF00):
    if not webhook_url:
        return False, "URL未設定"
    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
    }
    try:
        res = requests.post(webhook_url, json=payload, timeout=5)
        if res.status_code in [200, 204]:
            return True, "送信成功"
        else:
            return False, f"ステータスコード: {res.status_code}"
    except Exception as e:
        return False, str(e)

def get_signal_type(status_str):
    if status_str.startswith("BUY"):
        return "BUY"
    elif status_str.startswith("SELL"):
        return "SELL"
    return "HOLD"

def clean_series(s):
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s

# ==========================================
# 2. 通貨ペア & 時間軸設定
# ==========================================
PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}

TIMEFRAMES = {
    "5分足 (超短期スキャル用)": {"period": "7d", "interval": "5m"},
    "15分足 (デイトレエントリー用)": {"period": "1mo", "interval": "15m"},
    "1時間足 (デイトレメイン用)": {"period": "6mo", "interval": "1h"},
    "4時間足 (中期トレンド用)": {"period": "2y", "interval": "1h"},
    "日足 (スイング・環境認識用)": {"period": "2y", "interval": "1d"},
}

PAIR_ATR_CONFIG = {
    "USDJPY=X": {"atr_mult": 0.20, "min_pips": 15},
    "GBPJPY=X": {"atr_mult": 0.25, "min_pips": 20},
    "EURJPY=X": {"atr_mult": 0.20, "min_pips": 15},
    "AUDJPY=X": {"atr_mult": 0.18, "min_pips": 12},
    "EURUSD=X": {"atr_mult": 0.18, "min_pips": 12},
}

# --- ヘッダー・設定エリア ---
st.title("AI FX デイトレ & リピートアナライザー Pro")

with st.container():
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        selected_label = st.selectbox("分析通貨ペア", list(PAIRS.keys()), key="selected_pair_label", on_change=save_user_settings)
    with col_s2:
        tf_label = st.selectbox("時間軸", list(TIMEFRAMES.keys()), key="selected_tf_label", on_change=save_user_settings)

ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]
atr_cfg = PAIR_ATR_CONFIG.get(ticker, {"atr_mult": 0.20, "min_pips": 15})

is_jpy_pair = "JPY" in ticker
pip_unit = 0.01 if is_jpy_pair else 0.0001
price_fmt = ".3f" if is_jpy_pair else ".5f"

# サイドバー設定
st.sidebar.header("⚙️ システム設定")
if st.sidebar.button("🔄 最新データに更新", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("自動更新を有効にする", key="auto_refresh", on_change=save_user_settings)
refresh_interval = st.sidebar.selectbox(
    "更新間隔", options=[60, 180, 300], format_func=lambda x: f"{x // 60}分ごと", key="refresh_interval", on_change=save_user_settings
)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 松井証券トレード資金設定")
account_balance = st.sidebar.number_input("口座資金 (円)", min_value=10000, max_value=100000000, step=50000, key="account_balance", on_change=save_user_settings)
quantity_wan = st.sidebar.number_input("注文数量 (万通貨)", min_value=0.0001, max_value=10.0, step=0.01, format="%.4f", key="quantity_wan", on_change=save_user_settings)
custom_quantity = int(round(quantity_wan * 10000))

st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Discord 通知設定")
discord_url = st.sidebar.text_input("Webhook URL", type="password", key="discord_url", on_change=save_user_settings)
enable_notify = st.sidebar.checkbox("AI売買シグナル時に通知", key="enable_notify", on_change=save_user_settings)

if st.sidebar.button("🧪 テスト送信", use_container_width=True):
    if discord_url:
        ok, msg = send_discord_notification(discord_url, "テスト通知成功", f"選択中の通貨ペア: **{selected_label}**\n連携は正常です。", color=0x3498DB)
        if ok:
            st.sidebar.success("送信完了")
        else:
            st.sidebar.error(f"送信失敗: {msg}")
    else:
        st.sidebar.warning("URLを入力してください")

# ==========================================
# 3. データ取得 & インジケーター計算エンジン
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    df = pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        pass

    if not df.empty and "4時間足" in tf_name and interval == "1h":
        try:
            tz_before = df.index.tz
            df = df.resample("4h", closed="left", label="left", origin="start_day").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            if tz_before is not None and df.index.tz is None:
                df.index = df.index.tz_localize(tz_before)
        except Exception:
            pass

    if df.empty or len(df) < 50:
        return None

    try:
        c_series = clean_series(df["Close"])
        h_series = clean_series(df["High"])
        l_series = clean_series(df["Low"])
        o_series = clean_series(df["Open"])

        new_cols = {}
        new_cols["SMA_20"] = c_series.rolling(window=20).mean()
        new_cols["SMA_50"] = c_series.rolling(window=50).mean()
        new_cols["EMA_20"] = c_series.ewm(span=20, adjust=False).mean()
        new_cols["EMA_200"] = c_series.ewm(span=200, adjust=False).mean()

        high_low = h_series - l_series
        new_cols["ATR"] = high_low.rolling(window=14).mean()
        new_cols["ATR_SMA20"] = new_cols["ATR"].rolling(window=20).mean()
        new_cols["ATR_Ratio"] = new_cols["ATR"] / (new_cols["ATR_SMA20"] + 1e-10)

        total_range = high_low + 1e-10
        open_close_max = pd.concat([o_series, c_series], axis=1).max(axis=1)
        open_close_min = pd.concat([o_series, c_series], axis=1).min(axis=1)
        new_cols["Upper_Wick_Ratio"] = (h_series - open_close_max) / total_range
        new_cols["Lower_Wick_Ratio"] = (open_close_min - l_series) / total_range

        new_cols["Return_1"] = c_series.diff(1) / (c_series.shift(1) + 1e-10)
        new_cols["Return_5"] = c_series.diff(5) / (c_series.shift(5) + 1e-10)
        
        new_cols["Dev_SMA20"] = (c_series - new_cols["SMA_20"]) / (new_cols["SMA_20"] + 1e-10)
        new_cols["Dev_EMA200"] = (c_series - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Dev_EMA20_200"] = (new_cols["EMA_20"] - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Vol_Ratio"] = new_cols["ATR"] / (c_series + 1e-10)

        delta = c_series.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        new_cols["RSI"] = 100.0 - (100.0 / (1.0 + rs))
        new_cols["RSI_Diff"] = new_cols["RSI"].diff(1)

        low_14 = l_series.rolling(window=14).min()
        high_14 = h_series.rolling(window=14).max()
        new_cols["Stoch_K"] = 100.0 * (c_series - low_14) / ((high_14 - low_14) + 1e-10)

        ema12 = c_series.ewm(span=12, adjust=False).mean()
        ema26 = c_series.ewm(span=26, adjust=False).mean()
        new_cols["MACD"] = ema12 - ema26
        new_cols["MACD_Signal"] = new_cols["MACD"].ewm(span=9, adjust=False).mean()
        new_cols["MACD_Hist"] = new_cols["MACD"] - new_cols["MACD_Signal"]
        new_cols["MACD_Hist_Ratio"] = new_cols["MACD_Hist"] / (c_series + 1e-10)

        std20 = c_series.rolling(window=20).std()
        new_cols["Upper_Band"] = new_cols["SMA_20"] + (std20 * 2)
        new_cols["Lower_Band"] = new_cols["SMA_20"] - (std20 * 2)
        new_cols["BB_Width"] = (new_cols["Upper_Band"] - new_cols["Lower_Band"]) / (new_cols["SMA_20"] + 1e-10)
        new_cols["BB_PctB"] = (c_series - new_cols["Lower_Band"]) / ((new_cols["Upper_Band"] - new_cols["Lower_Band"]) + 1e-10)

        tr = pd.concat([high_low, (h_series - c_series.shift(1)).abs(), (l_series - c_series.shift(1)).abs()], axis=1).max(axis=1)
        up_move = h_series - h_series.shift(1)
        down_move = l_series.shift(1) - l_series
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
        sum_di = (plus_di + minus_di).replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / sum_di
        new_cols["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(25.0)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        f_high = pd.concat([h_series.shift(-1), h_series.shift(-2), h_series.shift(-3)], axis=1)
        f_low = pd.concat([l_series.shift(-1), l_series.shift(-2), l_series.shift(-3)], axis=1)

        target_pips = new_cols["ATR"] * 0.8  
        future_max_up = f_high.max(axis=1) - c_series
        future_max_down = c_series - f_low.min(axis=1)

        conditions = [
            (future_max_up >= target_pips) & (future_max_up > future_max_down),
            (future_max_down >= target_pips) & (future_max_down > future_max_up),
        ]
        
        target_series = np.select(conditions, [1, -1], default=0)
        df["Target"] = target_series.astype(float)
        if len(df) > 3:
            df.iloc[-3:, df.columns.get_loc("Target")] = np.nan

        feature_cols = [c for c in df.columns if c != "Target"]
        df = df.dropna(subset=feature_cols)

        if df.empty:
            return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=60, show_spinner=False)
def get_mtf_trends(symbol: str) -> dict:
    trends = {}
    for name, params in TIMEFRAMES.items():
        sub_d = load_and_process_data(symbol, params["period"], params["interval"], name)
        if sub_d is not None and len(sub_d) >= 20:
            c_price = float(clean_series(sub_d['Close']).iloc[-1])
            ema_series = clean_series(sub_d['EMA_200']) if 'EMA_200' in sub_d.columns else clean_series(sub_d['SMA_20'])
            c_ema = float(ema_series.iloc[-1])
            if c_price > c_ema:
                trends[name.split(" ")[0]] = "上昇 📈"
            elif c_price < c_ema:
                trends[name.split(" ")[0]] = "下降 📉"
            else:
                trends[name.split(" ")[0]] = "レンジ ➡️"
        else:
            trends[name.split(" ")[0]] = "判定中..."
    return trends

@st.cache_data(ttl=300, show_spinner=False)
def run_backtest(X_bt, y_bt, test_len):
    cumulative_wins = []
    trade_count, correct_count = 0, 0
    
    max_valid_idx = len(X_bt) - 3
    if max_valid_idx <= 30:
        return [], 0.0, 0, 0

    valid_test_len = min(test_len, max_valid_idx - 10)
    step_size = max(1, valid_test_len // 15)

    for i in range(0, valid_test_len, step_size):
        idx = max_valid_idx - valid_test_len + i
        train_end = max(1, idx - 3)
        valid_train = ~y_bt.iloc[:train_end].isna()
        if valid_train.sum() < 30:
            continue
        y_sub = y_bt.iloc[:train_end][valid_train]
        if len(np.unique(y_sub)) < 2:
            continue
        sub_model = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42)
        sub_model.fit(X_bt.iloc[:train_end][valid_train], y_sub)
        p = sub_model.predict(X_bt.iloc[[idx]])[0]
        actual = y_bt.iloc[idx]

        if p != 0 and not pd.isna(actual):
            trade_count += 1
            if p == actual:
                correct_count += 1
        current_win_rate = (correct_count / trade_count * 100) if trade_count > 0 else 0.0
        cumulative_wins.append((i + 1, current_win_rate))

    win_rate = (correct_count / trade_count * 100) if trade_count > 0 else 0.0
    return cumulative_wins, win_rate, trade_count, correct_count

@st.cache_data(ttl=60, show_spinner=False)
def analyze_signal(df_current, df_higher, usdjpy_df=None, current_symbol=""):
    if df_current is None or len(df_current) < 50:
        return "HOLD", 50.0, "不明", {}
    try:
        avail = [f for f in FEATURE_COLUMNS if f in df_current.columns]
        X = df_current[avail]
        y = df_current["Target"]
        train_mask = ~y.isna()
        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        X_train = X_train_full.iloc[-1000:] if len(X_train_full) > 1000 else X_train_full
        y_train = y_train_full.iloc[-1000:] if len(y_train_full) > 1000 else y_train_full

        if len(np.unique(y_train)) < 2:
            return "HOLD (分析不可: クラス不足)", 50.0, "判定不可", {}

        model = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)

        importances = dict(zip(avail, model.feature_importances_))

        X_latest = X.iloc[[-1]]
        prob_array = model.predict_proba(X_latest)[0]
        prob_dict = dict(zip(model.classes_, prob_array))
        prob_up, prob_down = prob_dict.get(1, 0.0), prob_dict.get(-1, 0.0)
        confidence = max(prob_up, prob_down) * 100

        c_series = clean_series(df_current["Close"])
        latest_price = float(c_series.iloc[-1])
        latest_ema200 = float(clean_series(df_current["EMA_200"]).iloc[-1]) if "EMA_200" in df_current.columns else latest_price
        latest_sma20 = float(clean_series(df_current["SMA_20"]).iloc[-1]) if "SMA_20" in df_current.columns else latest_price
        latest_adx = float(clean_series(df_current["ADX"]).iloc[-1]) if "ADX" in df_current.columns else 25.0
        latest_rsi = float(clean_series(df_current["RSI"]).iloc[-1]) if "RSI" in df_current.columns else 50.0
        latest_atr = float(clean_series(df_current["ATR"]).iloc[-1]) if "ATR" in df_current.columns else 0.1
        upper_band = float(clean_series(df_current["Upper_Band"]).iloc[-1]) if "Upper_Band" in df_current.columns else latest_price
        lower_band = float(clean_series(df_current["Lower_Band"]).iloc[-1]) if "Lower_Band" in df_current.columns else latest_price
        
        if "BB_Width" in df_current.columns:
            bb_w_series = clean_series(df_current["BB_Width"])
            latest_bb_width = float(bb_w_series.iloc[-1])
            avg_bb_series = bb_w_series.rolling(window=20).mean()
            avg_bb_width = float(avg_bb_series.iloc[-1]) if not avg_bb_series.empty and not pd.isna(avg_bb_series.iloc[-1]) else 0.05
        else:
            latest_bb_width = 0.05
            avg_bb_width = 0.05
        
        is_squeezed = latest_bb_width < (avg_bb_width * 0.75)

        htf_trend = "FLAT"
        if df_higher is not None and not df_higher.empty and "EMA_200" in df_higher.columns:
            h_close = float(clean_series(df_higher["Close"]).iloc[-1])
            h_ema200 = float(clean_series(df_higher["EMA_200"]).iloc[-1])
            htf_trend = "UP" if h_close > h_ema200 else "DOWN" if h_close < h_ema200 else "FLAT"

        is_cross_jpy = "JPY" in current_symbol and current_symbol != "USDJPY=X"
        usdjpy_strong_up = usdjpy_strong_down = False
        if is_cross_jpy and usdjpy_df is not None and not usdjpy_df.empty and "EMA_200" in usdjpy_df.columns:
            uj_close = float(clean_series(usdjpy_df["Close"]).iloc[-1])
            uj_ema = float(clean_series(usdjpy_df["EMA_200"]).iloc[-1])
            uj_rsi = float(clean_series(usdjpy_df["RSI"]).iloc[-1]) if "RSI" in usdjpy_df.columns else 50.0
            if uj_close > uj_ema and uj_rsi > 58: usdjpy_strong_up = True
            elif uj_close < uj_ema and uj_rsi < 42: usdjpy_strong_down = True

        h_series_curr = clean_series(df_current["High"])
        l_series_curr = clean_series(df_current["Low"])
        recent_50_high = float(h_series_curr.iloc[-50:-1].max())
        recent_50_low = float(l_series_curr.iloc[-50:-1].min())
        is_far_from_sma = abs(latest_price - latest_sma20) > (latest_atr * 1.5)
        is_near_support = (latest_price - recent_50_low) < (latest_atr * 0.8)
        is_near_resistance = (recent_50_high - latest_price) < (latest_atr * 0.8)

        HIGH_THRESHOLD = 0.60

        if latest_adx > 22.0:
            market_type = "トレンド相場"
            if prob_up >= HIGH_THRESHOLD and htf_trend != "DOWN":
                if usdjpy_strong_down: status = "HOLD (ストッパー: ドル円急落中)"
                elif latest_price <= latest_ema200: status = "HOLD (逆張り警戒: 200EMA下)"
                elif is_far_from_sma: status = "HOLD (高値掴み回避)"
                elif is_near_resistance: status = "HOLD (抵抗線直前)"
                else: status = "BUY"
            elif prob_down >= HIGH_THRESHOLD and htf_trend != "UP":
                if usdjpy_strong_up: status = "HOLD (ストッパー: ドル円急騰中)"
                elif latest_price >= latest_ema200: status = "HOLD (逆張り警戒: 200EMA上)"
                elif is_far_from_sma: status = "HOLD (安値掴み回避)"
                elif is_near_support: status = "HOLD (支持線直前)"
                else: status = "SELL"
            else: status = f"HOLD (確信度不足: {confidence:.1f}%)"
        else:
            market_type = "レンジ相場"
            if is_squeezed:
                status = "HOLD (ブレイクアウト警戒)"
            else:
                if (latest_price <= lower_band or latest_rsi <= 32.0) and prob_up >= 0.56:
                    status = "BUY (レンジ逆張り)" if not usdjpy_strong_down else "HOLD (ストッパー: ドル円逆行)"
                elif (latest_price >= upper_band or latest_rsi >= 68.0) and prob_down >= 0.56:
                    status = "SELL (レンジ逆張り)" if not usdjpy_strong_up else "HOLD (ストッパー: ドル円逆行)"
                else:
                    status = "HOLD (レンジ内静観)"

        return status, confidence, market_type, importances
    except Exception:
        return "HOLD", 50.0, "不明", {}

# ==========================================
# 4. メインデータロード & 画面描画処理
# ==========================================
with st.spinner("最新市場データとAIモデルを読込中..."):
    usdjpy_data = load_and_process_data("USDJPY=X", tf_config["period"], tf_config["interval"], tf_label)
    data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
    higher_tf_data = load_and_process_data(ticker, "1y", "1d", "日足 (スイング・環境認識用)")

now_datetime_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
current_day, current_hour_jst, current_minute_jst = now_datetime_jst.weekday(), now_datetime_jst.hour, now_datetime_jst.minute

now_datetime_ny = now_datetime_jst.astimezone(ZoneInfo("America/New_York"))
is_summer_time = now_datetime_ny.dst().total_seconds() != 0 

is_weekend = (current_day == 5 and current_hour_jst >= 6) or (current_day == 6) or (current_day == 0 and current_hour_jst < 6)
is_low_liquidity = 3 <= current_hour_jst <= 7

if is_summer_time:
    is_ny_open = 21 <= current_hour_jst <= 23
    is_econ_indicator_time = (current_hour_jst == 21 and current_minute_jst >= 15) or (current_hour_jst == 22 and current_minute_jst <= 45)
else:
    is_ny_open = current_hour_jst >= 22 or current_hour_jst == 0
    is_econ_indicator_time = (current_hour_jst == 22 and current_minute_jst >= 15) or (current_hour_jst == 23 and current_minute_jst <= 45)

# 警告メッセージ表示
if is_weekend: st.error("【週末クローズ中】現在市場は休業時間帯です。表示レートは最終終値となります。")
elif is_econ_indicator_time: st.error("【経済指標 警戒時間帯】急変の危険があります。エントリー自重をお勧めします。")
elif is_low_liquidity: st.warning("【低流動性時間帯】早朝のためスプレッド拡大にご注意ください。")

if data is None or len(data) < 10:
    st.error("データ取得に失敗しました。時間足を切り替えるか少し置いて再実行してください。")
else:
    market_status, confidence, market_type, importances = analyze_signal(data, higher_tf_data, usdjpy_df=usdjpy_data, current_symbol=ticker)
    if is_econ_indicator_time and (market_status.startswith("BUY") or market_status.startswith("SELL")):
        market_status = "HOLD (指標警戒帯)"

    c_close = clean_series(data["Close"])
    latest_price = float(c_close.iloc[-1])
    latest_bar_time = str(data.index[-1])

    # 重複通知防止
    if enable_notify and discord_url and (market_status.startswith("BUY") or market_status.startswith("SELL")):
        if "last_notified_status" not in st.session_state:
            st.session_state["last_notified_status"] = {}
        
        notify_key = f"{ticker}_{tf_label}"
        last_notified_bar = st.session_state["last_notified_status"].get(f"{notify_key}_time", "")
        last_status = st.session_state["last_notified_status"].get(f"{notify_key}_status", "")

        current_sig_type = get_signal_type(market_status)
        last_sig_type = get_signal_type(last_status)

        if (last_notified_bar != latest_bar_time) or (current_sig_type != last_sig_type and current_sig_type != "HOLD"):
            color_val = 0x2ECC71 if current_sig_type == "BUY" else 0xE74C3C
            msg_body = (
                f"**【{selected_label}】** AIシグナル発生\n"
                f"時間軸: {tf_label}\n"
                f"現在レート: `{latest_price:{price_fmt}}`\n"
                f"判定: **{market_status}** (確信度: `{confidence:.1f}%`)"
            )
            ok, _ = send_discord_notification(discord_url, f"AI FXシグナル通知 [{selected_label}]", msg_body, color=color_val)
            if ok:
                st.session_state["last_notified_status"][f"{notify_key}_time"] = latest_bar_time
                st.session_state["last_notified_status"][f"{notify_key}_status"] = market_status
                save_user_settings()

    available_features = [f for f in FEATURE_COLUMNS if f in data.columns]
    X_bt, y_bt = data[available_features], data["Target"]
    test_len = min(30, len(X_bt) - 10)

    if test_len > 5:
        cumulative_wins, win_rate, trade_count, correct_count = run_backtest(X_bt, y_bt, test_len)
    else:
        cumulative_wins, win_rate, trade_count, correct_count = [], 0.0, 0, 0

    if "BB_Width" in data.columns:
        bb_w_series = clean_series(data["BB_Width"])
        latest_bb_width = float(bb_w_series.iloc[-1])
        avg_bb_series = bb_w_series.rolling(window=20).mean()
        avg_bb_width = float(avg_bb_series.iloc[-1]) if not avg_bb_series.empty and not pd.isna(avg_bb_series.iloc[-1]) else 0.05
    else:
        latest_bb_width = 0.05
        avg_bb_width = 0.05
    is_squeezed = latest_bb_width < (avg_bb_width * 0.75)

    latest_adx = float(clean_series(data["ADX"]).iloc[-1]) if "ADX" in data.columns else 25.0
    htf_close = latest_price
    htf_sma50 = float(clean_series(data["SMA_50"]).iloc[-1]) if "SMA_50" in data.columns else latest_price
    if higher_tf_data is not None and not higher_tf_data.empty and "SMA_50" in higher_tf_data.columns:
        htf_close = float(clean_series(higher_tf_data["Close"]).iloc[-1])
        htf_sma50 = float(clean_series(higher_tf_data["SMA_50"]).iloc[-1])

    if htf_close > htf_sma50 * 1.002: long_term_trend = "上昇 📈"
    elif htf_close < htf_sma50 * 0.998: long_term_trend = "下降 📉"
    else: long_term_trend = "レンジ ➡️"

    latest_rsi = float(clean_series(data["RSI"]).iloc[-1]) if "RSI" in data.columns else 50.0
    latest_atr = float(clean_series(data["ATR"]).iloc[-1]) if "ATR" in data.columns else 0.1

    c_low = clean_series(data["Low"])
    c_high = clean_series(data["High"])
    buffer_margin = 10 * pip_unit
    structural_buy_sl = round(float(c_low.iloc[-20:].min()) - buffer_margin, 3 if is_jpy_pair else 5)
    structural_sell_sl = round(float(c_high.iloc[-20:].max()) + buffer_margin, 3 if is_jpy_pair else 5)

    conf_factor, adx_bonus = confidence / 50.0, 0.2 if latest_adx > 25 else 0.0
    if "レンジ" in market_type:
        ai_tp_mult, ai_sl_mult = 0.8, 0.6
    else:
        ai_tp_mult = round(max(1.0, min(2.5, 1.2 * conf_factor + adx_bonus)), 2)
        ai_sl_mult = round(max(0.6, min(1.5, 0.8 / (conf_factor * 0.8))), 2)

    if market_status.startswith("BUY"):
        calc_tp = latest_price + (latest_atr * ai_tp_mult)
        calc_sl = min(latest_price - (latest_atr * ai_sl_mult), structural_buy_sl)
    elif market_status.startswith("SELL"):
        calc_tp = latest_price - (latest_atr * ai_tp_mult)
        calc_sl = max(latest_price + (latest_atr * ai_sl_mult), structural_sell_sl)
    else:
        calc_tp = latest_price + (latest_atr * ai_tp_mult)
        calc_sl = latest_price - (latest_atr * ai_sl_mult)

    raw_atr_pips = latest_atr / pip_unit
    ai_recommended_width = int(max(round(raw_atr_pips * atr_cfg["atr_mult"], 1), atr_cfg["min_pips"]))
    recommended_slippage = round(max(0.5, raw_atr_pips * 0.05), 1)

    sl_distance_pips = max(20.0, round((latest_atr * ai_sl_mult) / pip_unit, 1))
    allowed_loss_jpy = account_balance * 0.02
    
    uj_rate = float(clean_series(usdjpy_data["Close"]).iloc[-1]) if (usdjpy_data is not None and not usdjpy_data.empty) else 155.0
    pip_value_per_unit = 0.01 if is_jpy_pair else 0.0001 * uj_rate

    safe_single_units = max(100, min(int(allowed_loss_jpy / (sl_distance_pips * pip_value_per_unit)), 50000))
    safe_single_wan = round(safe_single_units / 10000.0, 4)

    # ---------------------------------------------------------
    # 🌟 メインダッシュボード (最重要指標ヒーローカード)
    # ---------------------------------------------------------
    with st.container(border=True):
        m_head1, m_head2, m_head3, m_head4 = st.columns([1.2, 1.5, 1, 1])
        
        m_head1.metric("現在レート", f"{latest_price:{price_fmt}}")
        
        if market_status.startswith("BUY"):
            badge_html = f'<div class="status-badge-buy">BUY 買い ({confidence:.1f}%)</div>'
        elif market_status.startswith("SELL"):
            badge_html = f'<div class="status-badge-sell">SELL 売り ({confidence:.1f}%)</div>'
        else:
            badge_html = f'<div class="status-badge-hold">{market_status}</div>'
            
        m_head2.markdown("**AI総合判定 / 確信度**")
        m_head2.markdown(badge_html, unsafe_allow_html=True)
        
        m_head3.metric("相場環境", market_type, "トレンド" if latest_adx > 22 else "レンジ")
        m_head4.metric("直近AI勝率", f"{win_rate:.1f}%" if trade_count > 0 else "N/A", f"{correct_count}勝 / {trade_count}戦")

    # セカンダリ指標
    with st.container():
        sec1, sec2, sec3, sec4 = st.columns(4)
        sec1.metric("RSI (14)", f"{latest_rsi:.1f}")
        sec2.metric("ADX (トレンド強度)", f"{latest_adx:.1f}")
        sec3.metric("日足 トレンド", long_term_trend)
        sec4.metric("適正スリッページ", f"{recommended_slippage} pips")

    # ---------------------------------------------------------
    # 📊 マルチタイムフレーム (MTF) サマリー
    # ---------------------------------------------------------
    st.markdown("##### 🌐 マルチタイムフレーム (MTF) トレンド")
    mtf_trends = get_mtf_trends(ticker)
    mtf_cols = st.columns(len(TIMEFRAMES))
    for idx, (tf_name_key, t_val) in enumerate(mtf_trends.items()):
        mtf_cols[idx].metric(label=tf_name_key, value=t_val)

    st.markdown("---")

    # ---------------------------------------------------------
    # 📑 メイン操作タブ
    # ---------------------------------------------------------
    tab_repeat, tab_single, tab_speed, tab_chart, tab_scanner, tab_backtest, tab_metrics = st.tabs([
        "📋 リピート注文 (松井証券)", "🎯 デイトレ単発 (AI)", "⚡ スピード注文", "📈 チャート", "🔍 全ペアスキャン", "📊 バックテスト", "📋 運用サマリー"
    ])

    # --- TAB 1: リピート注文 ---
    with tab_repeat:
        st.subheader("📋 松井証券FX 自動売買（リピート注文）設定")
        jpy_rate = latest_price if is_jpy_pair else latest_price * uj_rate
        margin_per_unit = (jpy_rate * custom_quantity) / 25.0
        max_allowable_grids = max(2, int((account_balance * 0.5) / max(margin_per_unit, 1.0)))
        safe_half_range_val = max(1, max_allowable_grids // 2) * ai_recommended_width * pip_unit

        rep_lower = round(latest_price - safe_half_range_val, 3 if is_jpy_pair else 5)
        rep_upper = round(latest_price + safe_half_range_val, 3 if is_jpy_pair else 5)
        buffer_val = max(latest_atr * 1.5, 0.4 if is_jpy_pair else 0.04)
        rep_buy_stop = round(rep_lower - buffer_val, 3 if is_jpy_pair else 5)
        rep_sell_stop = round(rep_upper + buffer_val, 3 if is_jpy_pair else 5)
        buffer_pips = round(buffer_val / pip_unit, 1)

        total_est_wan = round((custom_quantity * max_allowable_grids) / 10000.0, 2)

        st.caption(f"💡 ATR ({raw_atr_pips:.1f} pips) に基づく推奨注文値幅: **{ai_recommended_width} pips** | 口座適正本数: **最大{max_allowable_grids}本**")

        rep_c1, rep_c2 = st.columns(2)
        with rep_c1:
            st.markdown("#### 🟢 買いリピート設定")
            st.markdown(f"""
            <div class="param-box param-box-buy">
            <b class="label-title">通貨ペア</b>    : {selected_label}<br>
            <b class="label-title">売買区分</b>    : 買<br>
            <b class="label-title">レンジ上限</b>  : <code>{rep_upper}</code><br>
            <b class="label-title">レンジ下限</b>  : <code>{rep_lower}</code><br>
            <b class="label-title">数量（万）</b>  : <code>{quantity_wan}</code><br>
            <b class="label-title">注文値幅</b>    : <code>{ai_recommended_width} pips</code><br>
            <b class="label-title">益出し幅</b>    : <code>{ai_recommended_width} pips</code><br>
            <b class="label-title">運用停止</b>    : <code>{rep_buy_stop}</code> (-{buffer_pips}pips)
            </div>
            """, unsafe_allow_html=True)
            
        with rep_c2:
            st.markdown("#### 🔴 売りリピート設定")
            st.markdown(f"""
            <div class="param-box param-box-sell">
            <b class="label-title">通貨ペア</b>    : {selected_label}<br>
            <b class="label-title">売買区分</b>    : 売<br>
            <b class="label-title">レンジ上限</b>  : <code>{rep_upper}</code><br>
            <b class="label-title">レンジ下限</b>  : <code>{rep_lower}</code><br>
            <b class="label-title">数量（万）</b>  : <code>{quantity_wan}</code><br>
            <b class="label-title">注文値幅</b>    : <code>{ai_recommended_width} pips</code><br>
            <b class="label-title">益出し幅</b>    : <code>{ai_recommended_width} pips</code><br>
            <b class="label-title">運用停止</b>    : <code>{rep_sell_stop}</code> (+{buffer_pips}pips)
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 2: デイトレ単発 ---
    with tab_single:
        st.subheader("🎯 デイトレ単発エントリーガイド")
        tp_pips_val = abs(calc_tp - latest_price) / pip_unit
        sl_pips_val = abs(latest_price - calc_sl) / pip_unit
        rr_ratio = (tp_pips_val / sl_pips_val) if sl_pips_val > 0 else 0.0

        if market_status.startswith("BUY"):
            st.success(f"🟢 **買いシグナル** (確信度: {confidence:.1f}% | Risk-Reward比: {rr_ratio:.2f})")
        elif market_status.startswith("SELL"):
            st.error(f"🔴 **売りシグナル** (確信度: {confidence:.1f}% | Risk-Reward比: {rr_ratio:.2f})")
        else:
            st.warning(f"🟡 **静観フィルター作動中 ({market_status})**")

        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("推奨注文ロット", f"{safe_single_wan} 万通貨")
        t_col2.metric("エントリー目安", f"{latest_price:{price_fmt}}")
        t_col3.metric("利確 (TP)", f"{calc_tp:{price_fmt}}", f"+{tp_pips_val:.1f} pips")
        t_col4.metric("損切 (SL)", f"{calc_sl:{price_fmt}}", f"-{sl_pips_val:.1f} pips")

        if importances:
            with st.expander("🧠 AI判定の主要根拠 (特徴量重要度 Top 5)"):
                imp_df = pd.DataFrame(list(importances.items()), columns=["指標名", "影響度"]).sort_values(by="影響度", ascending=False).head(5)
                st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)

    # --- TAB 3: スピード注文 ---
    with tab_speed:
        st.subheader("⚡ 松井証券FX アプリ【スピード注文】設定")
        sp_tp_pips = round(abs(calc_tp - latest_price) / pip_unit, 1)
        sp_sl_pips = round(abs(latest_price - calc_sl) / pip_unit, 1)
        
        sp_col1, sp_col2, sp_col3 = st.columns(3)
        sp_col1.metric("推奨ロット", f"{safe_single_wan} 万通貨")
        sp_col2.metric("益出し幅", f"{sp_tp_pips} pips")
        sp_col3.metric("損切り幅", f"{sp_sl_pips} pips")

        rec_dir = "買い (ASK)" if market_status.startswith("BUY") else "売り (BID)" if market_status.startswith("SELL") else "静観"
        st.markdown(f"""
        <div class="param-box">
        <b class="label-title">注文方向</b> : {rec_dir}<br>
        <b class="label-title">注文数量</b> : <code>{safe_single_wan}</code> 万通貨 ({safe_single_units:,} 通貨)<br>
        <b class="label-title">益出し幅</b> : <code>{sp_tp_pips} pips</code><br>
        <b class="label-title">損切り幅</b> : <code>{sp_sl_pips} pips</code><br>
        <b class="label-title">スリッページ上限</b> : <code>{recommended_slippage} pips</code>
        </div>
        """, unsafe_allow_html=True)

    # --- TAB 4: チャート ---
    with tab_chart:
        df_chart = data.tail(60).copy()

        try:
            if df_chart.index.tz is None:
                df_chart.index = df_chart.index.tz_localize("UTC")
            df_chart.index = df_chart.index.tz_convert("Asia/Tokyo")
        except Exception:
            pass

        is_daily = "日足" in tf_label
        chart_x = df_chart.index.strftime("%Y-%m-%d" if is_daily else "%m-%d %H:%M")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(
            x=chart_x,
            open=clean_series(df_chart["Open"]),
            high=clean_series(df_chart["High"]),
            low=clean_series(df_chart["Low"]),
            close=clean_series(df_chart["Close"]),
            name="ローソク足"
        ), row=1, col=1)

        for col, color, width, dash in [("SMA_20", "orange", 1, "solid"), ("EMA_200", "#3498db", 1.5, "solid"), ("Upper_Band", "gray", 1, "dash"), ("Lower_Band", "gray", 1, "dash")]:
            if col in df_chart.columns:
                fig.add_trace(go.Scatter(x=chart_x, y=clean_series(df_chart[col]), mode="lines", name=col, line=dict(color=color, width=width, dash=dash)), row=1, col=1)
        
        if market_status.startswith("BUY") or market_status.startswith("SELL"):
            fig.add_hline(y=calc_tp, line_dash="dash", line_color="#2ECC71", annotation_text="TP (利確)", row=1, col=1)
            fig.add_hline(y=calc_sl, line_dash="dash", line_color="#E74C3C", annotation_text="SL (損切)", row=1, col=1)

        if "RSI" in df_chart.columns:
            fig.add_trace(go.Scatter(x=chart_x, y=clean_series(df_chart["RSI"]), mode="lines", name="RSI", line=dict(color="#9b59b6", width=1.5)), row=2, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color="#e74c3c", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#2ecc71", row=2, col=1)
        fig.update_xaxes(type="category", nticks=8)
        fig.update_layout(xaxis_rangeslider_visible=False, height=480, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 5: スキャン ---
    with tab_scanner:
        st.subheader("🔍 全監視通貨ペア AI防衛スキャン")
        if st.button("全通貨ペアをスキャン実行", use_container_width=True):
            scan_results = []
            progress_bar_scan = st.progress(0)
            
            with st.spinner("一括解析中..."):
                sub_usdjpy_df = load_and_process_data("USDJPY=X", tf_config["period"], tf_config["interval"], tf_label)
                for idx_p, (p_label, p_symbol) in enumerate(PAIRS.items()):
                    sub_df = load_and_process_data(p_symbol, tf_config["period"], tf_config["interval"], tf_label)
                    sub_htf = load_and_process_data(p_symbol, "1y", "1d", "日足 (スイング・環境認識用)")
                    if sub_df is not None and len(sub_df) > 10:
                        s_status, s_conf, s_mtype, _ = analyze_signal(sub_df, sub_htf, usdjpy_df=sub_usdjpy_df, current_symbol=p_symbol)
                        scan_results.append({
                            "通貨ペア": p_label,
                            "相場タイプ": s_mtype,
                            "AI判定": s_status,
                            "確信度 (%)": round(s_conf, 1),
                            "ADX": round(float(clean_series(sub_df["ADX"]).iloc[-1]) if "ADX" in sub_df.columns else 25.0, 1)
                        })
                    progress_bar_scan.progress(min(1.0, (idx_p + 1) / len(PAIRS)))

            progress_bar_scan.empty()
            if scan_results:
                st.dataframe(pd.DataFrame(scan_results).sort_values(by="確信度 (%)", ascending=False), use_container_width=True)

    # --- TAB 6: バックテスト ---
    with tab_backtest:
        st.subheader("📊 時系列ウォークフォワード検証")
        if cumulative_wins:
            st.line_chart(pd.DataFrame(cumulative_wins, columns=["ステップ", "適合率 (%)"]).set_index("ステップ"))
            st.metric("トレード実行時 適合率", f"{win_rate:.1f}%" if trade_count > 0 else "N/A", f"{correct_count}勝 / {trade_count}回")

    # --- TAB 7: 運用サマリー ---
    with tab_metrics:
        st.subheader("📋 運用設定サマリー")
        sum_c1, sum_c2 = st.columns(2)
        sum_c1.metric("口座資金", f"{account_balance:,} 円")
        sum_c1.metric("1回あたり数量", f"{quantity_wan} 万通貨 ({custom_quantity:,} 通貨)")
        sum_c2.metric("分析通貨ペア", selected_label)
        sum_c2.metric("分析時間足", tf_label)

if auto_refresh:
    st.caption(f"🔄 自動更新有効 ({refresh_interval}秒間隔)")
    st_autorefresh(interval=refresh_interval * 1000, limit=100, key="data_refresh")
