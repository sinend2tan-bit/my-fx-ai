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
# 0. 画面基本設定 & 共通定数
# ==========================================
st.set_page_config(
    page_title="プロ版 AI FXデイトレ & リピートアナライザー Pro v6.3.6",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLUMNS = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200", "Dev_EMA20_200",
    "EMA20_Cross_SMA50", "Vol_Ratio", "RSI", "RSI_Diff", "MACD_Hist_Ratio",
    "BB_PctB", "ADX", "ATR_Ratio", "ATR_Slope", "Upper_Wick_Ratio",
    "Lower_Wick_Ratio", "Stoch_K"
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
            "color": int(color),
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
    """DataFrame化してしまったカラムを安全に1次元Seriesへ平坦化する補助関数"""
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

st.title("⚡ Pro AI FX デイトレ & リピートアナライザー (v6.3.6)")

col_s1, col_s2 = st.columns(2)
with col_s1:
    selected_label = st.selectbox("通貨ペアを選択", list(PAIRS.keys()), key="selected_pair_label", on_change=save_user_settings)
with col_s2:
    tf_label = st.selectbox("時間軸を選択", list(TIMEFRAMES.keys()), key="selected_tf_label", on_change=save_user_settings)

ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]
atr_cfg = PAIR_ATR_CONFIG.get(ticker, {"atr_mult": 0.20, "min_pips": 15})

is_jpy_pair = "JPY" in ticker
pip_unit = 0.01 if is_jpy_pair else 0.0001
price_fmt = ".3f" if is_jpy_pair else ".5f"

st.sidebar.header("⚙️ システム設定 & 口座管理")
if st.sidebar.button("🔄 今すぐ最新データに更新", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("自動更新を有効にする", key="auto_refresh", on_change=save_user_settings)
refresh_interval = st.sidebar.selectbox(
    "更新間隔を選択", options=[60, 180, 300], format_func=lambda x: f"{x // 60}分ごと", key="refresh_interval", on_change=save_user_settings
)

st.sidebar.subheader("📋 松井証券トレード資金設定（基本入力）")
account_balance = st.sidebar.number_input("口座資金 (円)", min_value=10000, max_value=100000000, step=50000, key="account_balance", on_change=save_user_settings)
quantity_wan = st.sidebar.number_input("注文数量 (万通貨)", min_value=0.0001, max_value=10.0, step=0.01, format="%.4f", key="quantity_wan", on_change=save_user_settings)
custom_quantity = int(round(quantity_wan * 10000))

st.sidebar.subheader("🔔 Discord 通知設定")
discord_url = st.sidebar.text_input("Webhook URL", type="password", key="discord_url", on_change=save_user_settings)
enable_notify = st.sidebar.checkbox("AI売買シグナル時に通知する", key="enable_notify", on_change=save_user_settings)

if st.sidebar.button("🧪 Discord テスト送信"):
    if discord_url:
        ok, msg = send_discord_notification(discord_url, "🧪 テスト通知成功", f"選択中の通貨ペア: **{selected_label}**\n連携は正常です！", color=0x3498DB)
        if ok:
            st.sidebar.success("テスト通知を送信しました！")
        else:
            st.sidebar.error(f"送信失敗: {msg}")
    else:
        st.sidebar.warning("Webhook URLを入力してください。")

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
        new_cols["ATR_Slope"] = new_cols["ATR"].diff(3) / (new_cols["ATR"].shift(3) + 1e-10)

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
        new_cols["EMA20_Cross_SMA50"] = (new_cols["EMA_20"] - new_cols["SMA_50"]) / (new_cols["SMA_50"] + 1e-10)
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
        sub_model = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=42)
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

        model = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)

        importances = dict(zip(avail, model.feature_importances_))

        X_latest = X.iloc[[-1]]
        prob_array = model.predict_proba(X_latest)[0]
        prob_dict = dict(zip(model.classes_, prob_array))
        prob_up, prob_down = prob_dict.get(1, 0.0), prob_dict.get(-1, 0.0)
        confidence = float(max(prob_up, prob_down) * 100)

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
# 4. バックグラウンド全通貨監視 & 通知処理
# ==========================================
if auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

usdjpy_df_global = load_and_process_data("USDJPY=X", tf_config["period"], tf_config["interval"], tf_label)

if enable_notify and discord_url:
    current_status_map = st.session_state.get("last_notified_status", {})
    
    for p_name, p_code in PAIRS.items():
        sub_d = load_and_process_data(p_code, tf_config["period"], tf_config["interval"], tf_label)
        if sub_d is not None and len(sub_d) >= 50:
            h_conf = TIMEFRAMES["4時間足 (中期トレンド用)"]
            df_h = load_and_process_data(p_code, h_conf["period"], h_conf["interval"], "4時間足 (中期トレンド用)")
            
            st_str, conf_val, m_type, _ = analyze_signal(sub_d, df_h, usdjpy_df_global, p_code)
            sig = get_signal_type(st_str)
            last_sig = current_status_map.get(p_code, "HOLD")

            if sig != "HOLD" and sig != last_sig:
                cur_price = float(clean_series(sub_d["Close"]).iloc[-1])
                color_code = 0x2ECC71 if sig == "BUY" else 0xE74C3C
                msg_text = (
                    f"**通貨ペア**: {p_name}\n"
                    f"**シグナル**: **{st_str}**\n"
                    f"** AI 確信度**: {conf_val:.1f}%\n"
                    f"**現在価格**: {cur_price:{price_fmt}}\n"
                    f"**相場環境**: {m_type}\n"
                    f"**発生時刻**: {datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} JST"
                )
                send_discord_notification(discord_url, f"🚨 AI売買シグナル発生 [{p_name}]", msg_text, color=color_code)
                current_status_map[p_code] = sig

    st.session_state["last_notified_status"] = current_status_map
    save_user_settings()

# ==========================================
# 5. メインデータロード & UI描画
# ==========================================
df = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)

htf_key = "日足 (スイング・環境認識用)" if tf_label != "日足 (スイング・環境認識用)" else "4時間足 (中期トレンド用)"
htf_cfg = TIMEFRAMES[htf_key]
df_higher = load_and_process_data(ticker, htf_cfg["period"], htf_cfg["interval"], htf_key)

if df is None:
    st.error("データの取得に失敗しました。時間をおいて再試行してください。")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 デイトレAI単発分析",
    "🔄 リピートFX最適設定",
    "📡 全通貨ペアAI比較",
    "🧪 AI精度検証 (バックテスト)"
])

# ------------------------------------------
# TAB 1: デイトレAI単発分析
# ------------------------------------------
with tab1:
    status, confidence, market_type, feature_importances = analyze_signal(df, df_higher, usdjpy_df_global, ticker)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    latest_close = float(clean_series(df["Close"]).iloc[-1])
    latest_atr = float(clean_series(df["ATR"]).iloc[-1])
    
    with col_m1:
        st.metric("現在価格", f"{latest_close:{price_fmt}}")
    with col_m2:
        if status.startswith("BUY"):
            st.metric("AI売買判定", status, delta="買い推奨", delta_color="normal")
        elif status.startswith("SELL"):
            st.metric("AI売買判定", status, delta="売り推奨", delta_color="inverse")
        else:
            st.metric("AI売買判定", status, delta="様子見", delta_color="off")
    with col_m3:
        st.metric("AI予測確信度", f"{confidence:.1f}%")
    with col_m4:
        st.metric("相場タイプ", market_type, delta=f"ATR: {latest_atr:{price_fmt}}")

    st.markdown("---")

    col_chart, col_side = st.columns([2, 1])

    with col_chart:
        st.subheader("📈 リアルタイム・マルチテクニカルチャート")
        
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("ローソク足 & 200EMA / ボリンジャーバンド", "RSI (14)", "MACD")
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="価格"
        ), row=1, col=1)

        if "EMA_20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], line=dict(color="orange", width=1.5), name="20 EMA"), row=1, col=1)
        if "EMA_200" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_200"], line=dict(color="purple", width=2), name="200 EMA"), row=1, col=1)
        if "Upper_Band" in df.columns and "Lower_Band" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["Upper_Band"], line=dict(color="gray", width=1, dash="dash"), name="BB Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["Lower_Band"], line=dict(color="gray", width=1, dash="dash"), name="BB Lower"), row=1, col=1)

        if "RSI" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="blue", width=1.5), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="blue", width=1), name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color="orange", width=1), name="Signal"), row=3, col=1)
            if "MACD_Hist" in df.columns:
                colors = np.where(df["MACD_Hist"] >= 0, "green", "red")
                fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=colors, name="Hist"), row=3, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, height=650, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.subheader("💡 注文・リスク決済目安")
        
        calculated_tp_pips = max(latest_atr * atr_cfg["atr_mult"] * 100, atr_cfg["min_pips"])
        calculated_sl_pips = calculated_tp_pips * 0.8  

        if is_jpy_pair:
            tp_price = latest_close + (calculated_tp_pips * 0.01) if status.startswith("BUY") else latest_close - (calculated_tp_pips * 0.01)
            sl_price = latest_close - (calculated_sl_pips * 0.01) if status.startswith("BUY") else latest_close + (calculated_sl_pips * 0.01)
        else:
            tp_price = latest_close + (calculated_tp_pips * 0.0001) if status.startswith("BUY") else latest_close - (calculated_tp_pips * 0.0001)
            sl_price = latest_close - (calculated_sl_pips * 0.0001) if status.startswith("BUY") else latest_close + (calculated_sl_pips * 0.0001)

        st.info(
            f"**推奨利確 (TP)**: {tp_price:{price_fmt}} (+{calculated_tp_pips:.1f} pips)\n\n"
            f"**推奨損切 (SL)**: {sl_price:{price_fmt}} (-{calculated_sl_pips:.1f} pips)\n\n"
            f"**リスクリワード比**: 1 : 1.25"
        )

        st.markdown("---")
        st.subheader("💰 想定損益シミュレーター")
        
        # 1pipsあたりの損益計算
        if is_jpy_pair:
            profit_per_pip = custom_quantity * 0.01  # 例: 2,000通貨で1pip=20円
        else:
            # ドルストレート等の場合、現在ドル円レート換算
            uj_close = float(clean_series(usdjpy_df_global["Close"]).iloc[-1]) if usdjpy_df_global is not None else 150.0
            profit_per_pip = custom_quantity * 0.0001 * uj_close

        expected_profit = calculated_tp_pips * profit_per_pip
        expected_loss = calculated_sl_pips * profit_per_pip

        st.write(f"注文数量: **{custom_quantity:,} 通貨** ({quantity_wan:.2f} 万通貨)")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("利確時見込み益", f"+{int(expected_profit):,} 円", delta=f"+{calculated_tp_pips:.1f} pips")
        with col_p2:
            st.metric("損切時見込み損", f"-{int(expected_loss):,} 円", delta=f"-{calculated_sl_pips:.1f} pips", delta_color="inverse")

        st.markdown("---")
        st.subheader("🌐 上位足トレンド環境（MTF）")
        mtf_data = get_mtf_trends(ticker)
        for tf_k, tf_v in mtf_data.items():
            st.write(f"・**{tf_k}**: {tf_v}")

    st.markdown("---")
    st.subheader("🧠 AIトレーディングアドバイス & 判定根拠")
    
    explanation_text = ""
    if "HOLD" in status:
        if "ストッパー" in status:
            explanation_text = "⚠️ **相関ストッパー作動中**: ドル円が急変しているため、クロス円取引の連動リスクを考慮しAIがシグナルを一時停止しています。"
        elif "逆張り警戒" in status:
            explanation_text = "🛡️ **トレンド順張り保護**: 長期200EMAに逆らう方向のシグナルが発生したため、騙しを回避するために静観（HOLD）を推奨しています。"
        elif "高値掴み" in status or "安値掴み" in status:
            explanation_text = "🛑 **過熱感警戒**: 短期移動平均線から価格が乖離しすぎています。押目や戻りを待つのが安全です。"
        else:
            explanation_text = "☕ **静観推奨**: 明確なトレンド方向またはエントリー基準の確信度が得られていません。引きつけてチャンスを待ちましょう。"
    elif "BUY" in status:
        explanation_text = "🚀 **買いシグナル成立**: 上昇勢いが優勢であり、上位足トレンドとの整合性が高まっています。推奨TP/SLをセットしてエントリーを検討できます。"
    else:
        explanation_text = "🔻 **売りシグナル成立**: 下降圧力が強まっており、下落余地が大きいと判断されました。リスク管理を徹底した上でショートを検討できます。"

    st.warning(f"**【AIアクションプラン】** {explanation_text}")

# ------------------------------------------
# TAB 2: リピートFX最適設定
# ------------------------------------------
with tab2:
    st.subheader("🔄 リピートFX (松井証券等) 最適グリッド設定")
    
    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        st.markdown("#### 📊 現在の相場ボラティリティ指標")
        st.write(f"・**15分足 ATR**: {latest_atr:{price_fmt}} ({latest_atr / pip_unit:.1f} pips)")
        
        # 過去30日間のレンジを計算
        df_30d = df.iloc[-30*24*4:] if len(df) > 30*24*4 else df
        min_p_30 = float(clean_series(df_30d["Low"]).min())
        max_p_30 = float(clean_series(df_30d["High"]).max())
        range_pips = (max_p_30 - min_p_30) / pip_unit

        st.write(f"・**直近想定レンジ**: {min_p_30:{price_fmt}} ～ {max_p_30:{price_fmt}} ({range_pips:.0f} pips)")

    with col_r2:
        st.markdown("#### ⚙️ AI推奨注文パラメータ")
        rec_width = max(latest_atr * 1.2 / pip_unit, 10.0)
        rec_profit = rec_width * 1.1

        st.success(
            f"**推奨注文幅 (注文間隔)**: **{rec_width:.1f} pips**\n\n"
            f"**推奨利確幅 (決済トレール)**: **{rec_profit:.1f} pips**\n\n"
            f"**1本あたりの推奨数量**: **{quantity_wan:.2f} 万通貨**"
        )

# ------------------------------------------
# TAB 3: 全通貨ペアAI比較
# ------------------------------------------
with tab3:
    st.subheader("📡 全通貨ペア リアルタイムAI一括スキャン")
    
    scan_results = []
    progress_bar = st.progress(0)
    
    for idx, (p_name, p_code) in enumerate(PAIRS.items()):
        sub_d = load_and_process_data(p_code, tf_config["period"], tf_config["interval"], tf_label)
        if sub_d is not None and len(sub_d) >= 50:
            st_str, conf_v, m_t, _ = analyze_signal(sub_d, df_higher, usdjpy_df_global, p_code)
            c_p = float(clean_series(sub_d["Close"]).iloc[-1])
            scan_results.append({
                "通貨ペア": p_name,
                "現在価格": f"{c_p:{'.3f' if 'JPY' in p_code else '.5f'}}",
                "AI売買判定": st_str,
                "確信度": f"{conf_v:.1f}%",
                "相場タイプ": m_t,
            })
        progress_bar.progress((idx + 1) / len(PAIRS))
    
    progress_bar.empty()

    if scan_results:
        scan_df = pd.DataFrame(scan_results)
        st.dataframe(scan_df, use_container_width=True)
    else:
        st.warning("スキャン結果を取得できませんでした。")

# ------------------------------------------
# TAB 4: AI精度検証 (バックテスト)
# ------------------------------------------
with tab4:
    st.subheader("🧪 AI予測モデルの精度バックテスト検証")
    st.write("過去データを用いたウォークフォワード検証を行い、AIシグナルの勝率と累積パフォーマンスを評価します。")

    if st.button("🚀 バックテストを実行"):
        with st.spinner("過去データでAIモデルをバックテスト中..."):
            avail = [f for f in FEATURE_COLUMNS if f in df.columns]
            X_bt = df[avail]
            y_bt = df["Target"]

            cumulative_wins, win_rate, total_trades, correct_trades = run_backtest(X_bt, y_bt, test_len=100)

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("検証トレード数", f"{total_trades} 回")
            with col_b2:
                st.metric("正解数", f"{correct_trades} 回")
            with col_b3:
                st.metric("AI予測勝率", f"{win_rate:.1f}%")

            if cumulative_wins:
                win_df = pd.DataFrame(cumulative_wins, columns=["Trade_Num", "Win_Rate"])
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=win_df["Trade_Num"], y=win_df["Win_Rate"], mode="lines+markers", name="勝率 (%)"))
                fig_bt.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="勝率 50%")
                fig_bt.update_layout(title="トレード回数に伴うAI勝率の推移", xaxis_title="トレード試行回数", yaxis_title="勝率 (%)", height=400)
                st.plotly_chart(fig_bt, use_container_width=True)
