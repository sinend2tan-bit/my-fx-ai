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
    """DataFrame化してしまったカラムを安全に1次元Seriesへ平坦化する補助関数"""
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s

# ==========================================
# 2. データ取得 & インジケーター計算エンジン
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    df = pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
            
        # yfinanceの多重インデックス対策
        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.levels[1]:
                df = df.xs(symbol, axis=1, level=1)
            else:
                df.columns = df.columns.get_level_values(0)
    except Exception:
        return None

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
        c_series = clean_series(df["Close"]).astype(float)
        h_series = clean_series(df["High"]).astype(float)
        l_series = clean_series(df["Low"]).astype(float)
        o_series = clean_series(df["Open"]).astype(float)

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
        valid_mask = ~y_bt.iloc[:train_end].isna()
        if valid_mask.sum() < 30:
            continue
            
        X_sub = X_bt.iloc[:train_end][valid_mask]
        y_sub = y_bt.iloc[:train_end][valid_mask]
        
        if len(np.unique(y_sub)) < 2:
            continue
            
        sub_model = RandomForestClassifier(n_estimators=60, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42)
        sub_model.fit(X_sub, y_sub)
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

        model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)

        importances = dict(zip(avail, model.feature_importances_))

        X_latest = X.iloc[[-1]]
        prob_array = model.predict_proba(X_latest)[0]
        prob_dict = dict(zip(model.classes_, prob_array))
        prob_up, prob_down = prob_dict.get(1.0, 0.0), prob_dict.get(-1.0, 0.0)
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
# 3. UIヘッダー & サイドバー設定
# ==========================================
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
# 4. データ取得 & 市場時間帯の警告判定
# ==========================================
with st.spinner("データとAIを初期化中..."):
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

if is_weekend: st.error("🛑 **【週末・為替市場クローズ中】**: 現在外国為替市場は休業時間帯です。表示価格は最終クローズ値となります。")
elif is_econ_indicator_time: st.error("🚨 **【重要経済指標 警戒タイムゾーン】**: 突発的乱高下の危険がある時間帯です。新規エントリーは自重をお勧めします。")
elif is_low_liquidity: st.warning("⚠️ **【流動性低下タイムゾーン】**: オセアニア時間の早朝です。スプレッド拡大にご注意ください。")
elif is_ny_open: st.info("🔥 **【NY市場オープンタイムゾーン】**: ボラティリティが高まる時間帯です。")
# ==========================================
# 5. データ検証 & AI判定実行
# ==========================================
if data is None or data.empty:
    st.error("データの取得に失敗したか、データ量が不足しています。しばらく時間を置くか、別の通貨ペア/時間軸を選択してください。")
    st.stop()

signal_status, signal_conf, market_type, feat_importances = analyze_signal(
    data, higher_tf_data, usdjpy_data, ticker
)

c_series = clean_series(data["Close"])
latest_price = float(c_series.iloc[-1])
prev_price = float(c_series.iloc[-2]) if len(c_series) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / (prev_price + 1e-10)) * 100
latest_atr = float(clean_series(data["ATR"]).iloc[-1]) if "ATR" in data.columns else 0.1

# ==========================================
# 6. Discord 通知ロジック
# ==========================================
last_bar_time = str(data.index[-1])
notify_key = f"{ticker}_{tf_label}_{last_bar_time}"
last_notified = st.session_state.get("last_notified_status", {})

if enable_notify and discord_url:
    sig_type = get_signal_type(signal_status)
    if sig_type in ["BUY", "SELL"]:
        if last_notified.get("key") != notify_key or last_notified.get("status") != signal_status:
            color_code = 0x2ECC71 if sig_type == "BUY" else 0xE74C3C
            msg_body = (
                f"**通貨ペア**: {selected_label}\n"
                f"**時間軸**: {tf_label}\n"
                f"**シグナル**: **{signal_status}**\n"
                f"**確信度**: {signal_conf:.1f}%\n"
                f"**現在価格**: {latest_price:{price_fmt}}\n"
                f"**市場環境**: {market_type}\n"
                f"**日時**: {now_datetime_jst.strftime('%Y-%m-%d %H:%M:%S')} JST"
            )
            ok, _ = send_discord_notification(discord_url, f"🚨 AI売買シグナル検知: {sig_type}", msg_body, color=color_code)
            if ok:
                st.session_state["last_notified_status"] = {"key": notify_key, "status": signal_status}
                save_user_settings()

# ==========================================
# 7. 主要指標メトリクス表示
# ==========================================
st.markdown("---")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric(
        "現在価格",
        f"{latest_price:{price_fmt}}",
        f"{price_change:+{price_fmt}} ({pct_change:+.2f}%)"
    )

with m_col2:
    st.metric(
        "AI 判定シグナル",
        signal_status,
        f"確信度: {signal_conf:.1f}%"
    )

with m_col3:
    adx_val = float(clean_series(data['ADX']).iloc[-1]) if "ADX" in data.columns else 25.0
    st.metric(
        "市場環境",
        market_type,
        f"ADX: {adx_val:.1f}"
    )

with m_col4:
    sl_pips = latest_atr * 1.5
    tp_pips = latest_atr * 2.0
    st.metric(
        "推奨 ATR 目安 (1.5x / 2.0x)",
        f"SL: {sl_pips / pip_unit:.1f} pips",
        f"TP: {tp_pips / pip_unit:.1f} pips"
    )

# ==========================================
# 8. メイン タブ UI コンポーネント
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 メインチャート & 売買分析",
    "🔄 松井証券 リピート試算",
    "🔍 全通貨ペア 一括スキャン",
    "🧠 AIモデル分析 & 検証"
])

# ------------------------------------------
# Tab 1: チャート & マルチタイムフレーム
# ------------------------------------------
with tab1:
    st.subheader(f"📈 {selected_label} - {tf_label} インタラクティブチャート")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=clean_series(data['Open']),
        high=clean_series(data['High']),
        low=clean_series(data['Low']),
        close=clean_series(data['Close']),
        name="価格"
    ), row=1, col=1)

    if "SMA_20" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['SMA_20']),
            mode='lines', name='SMA 20', line=dict(color='orange', width=1.2)
        ), row=1, col=1)
    if "EMA_200" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['EMA_200']),
            mode='lines', name='EMA 200', line=dict(color='purple', width=1.8)
        ), row=1, col=1)
    if "Upper_Band" in data.columns and "Lower_Band" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['Upper_Band']),
            mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dot')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['Lower_Band']),
            mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dot')
        ), row=1, col=1)

    if "RSI" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['RSI']),
            mode='lines', name='RSI (14)', line=dict(color='cyan', width=1.2)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🌐 マルチタイムフレーム (MTF) トレンド環境認識")
    mtf_trends = get_mtf_trends(ticker)
    cols = st.columns(len(mtf_trends))
    for i, (tf_n, trend_val) in enumerate(mtf_trends.items()):
        with cols[i]:
            st.metric(tf_n, trend_val)

# ------------------------------------------
# Tab 2: 松井証券リピート注文（手動/自動）シミュレーター
# ------------------------------------------
with tab2:
    st.subheader("🔄 松井証券リピート注文（手動/自動）シミュレーター")
    st.info("想定レンジ幅内に一定間隔（値幅）でトラップを分散配置する想定リスク・リターン試算ツールです。")

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        default_min = float(round(latest_price * 0.97, 2 if is_jpy_pair else 4))
        default_max = float(round(latest_price * 1.03, 2 if is_jpy_pair else 4))
        range_min = st.number_input("想定レンジ下限", value=default_min, format=price_fmt)
        range_max = st.number_input("想定レンジ上限", value=default_max, format=price_fmt)
    with r_col2:
        grid_pips = st.number_input("注文間隔 (pips)", min_value=5.0, max_value=500.0, value=20.0, step=5.0)
        tp_pips_repeat = st.number_input("利確幅 (pips)", min_value=5.0, max_value=500.0, value=20.0, step=5.0)

    grid_step = grid_pips * pip_unit
    if range_max > range_min and grid_step > 0:
        trap_count = int((range_max - range_min) / grid_step) + 1
        total_units = trap_count * custom_quantity
        
        # 必要証拠金算定 (レバレッジ25倍ベース)
        if is_jpy_pair:
            margin_required = (range_max * custom_quantity * trap_count) / 25.0
        else:
            uj_rate = float(clean_series(usdjpy_data['Close']).iloc[-1]) if usdjpy_data is not None else 150.0
            margin_required = (range_max * uj_rate * custom_quantity * trap_count) / 25.0

        profit_per_trap = custom_quantity * (tp_pips_repeat * pip_unit)
        profit_per_trap_jpy = profit_per_trap if is_jpy_pair else profit_per_trap * (float(clean_series(usdjpy_data['Close']).iloc[-1]) if usdjpy_data is not None else 150.0)

        rc_m1, rc_m2, rc_m3, rc_m4 = st.columns(4)
        rc_m1.metric("仕掛け本数", f"{trap_count} 本")
        rc_m2.metric("総注文数量", f"{total_units / 10000:.2f} 万通貨")
        rc_m3.metric("推定必要証拠金", f"約 {int(margin_required):,} 円")
        rc_m4.metric("1回当たり決済益", f"約 {int(profit_per_trap_jpy):,} 円")

        st.markdown("#### 💡 資金管理判定")
        if margin_required > account_balance:
            st.error(f"⚠️ 推定必要証拠金 ({int(margin_required):,}円) が設定口座資金 ({account_balance:,}円) を超過しています。仕掛け幅を広げるか、1本あたりの注文数量を抑えてください。")
        else:
            safety_margin = account_balance - margin_required
            st.success(f"✅ 設定口座資金の範囲内です（推奨耐え余力・自由余力: 約 {int(safety_margin):,}円）。")

# ------------------------------------------
# Tab 3: 全通貨ペア 一括スキャン
# ------------------------------------------
with tab3:
    st.subheader("🔍 全通貨ペア AIスキャン（市場環境 & シグナル一括診断）")
    if st.button("🚀 全通貨ペアをスキャン実行", use_container_width=True):
        scan_results = []
        with st.spinner("全対象通貨ペアのデータ取得 & AI解析を実行中..."):
            for p_label, p_ticker in PAIRS.items():
                try:
                    p_data = load_and_process_data(p_ticker, tf_config["period"], tf_config["interval"], tf_label)
                    p_higher = load_and_process_data(p_ticker, "1y", "1d", "日足 (スイング・環境認識用)")
                    if p_data is not None and not p_data.empty:
                        p_status, p_conf, p_mtype, _ = analyze_signal(p_data, p_higher, usdjpy_data, p_ticker)
                        p_price = float(clean_series(p_data["Close"]).iloc[-1])
                        p_fmt = ".3f" if "JPY" in p_ticker else ".5f"
                        scan_results.append({
                            "通貨ペア": p_label,
                            "現在価格": f"{p_price:{p_fmt}}",
                            "AI判定": p_status,
                            "確信度": f"{p_conf:.1f}%",
                            "相場環境": p_mtype
                        })
                    else:
                        scan_results.append({
                            "通貨ペア": p_label, "現在価格": "-", "AI判定": "データ取得失敗", "確信度": "-", "相場環境": "-"
                        })
                except Exception:
                    scan_results.append({
                        "通貨ペア": p_label, "現在価格": "-", "AI判定": "エラー発生", "確信度": "-", "相場環境": "-"
                    })
        
        if scan_results:
            scan_df = pd.DataFrame(scan_results)
            st.dataframe(scan_df, use_container_width=True, hide_index=True)
    else:
        st.info("「全通貨ペアをスキャン実行」ボタンを押すと、すべての監視通貨ペアのAI診断結果を一覧で表示します。")

# ------------------------------------------
# Tab 4: AIモデル分析 & バックテスト
# ------------------------------------------
with tab4:
    st.subheader("🧠 AIモデル特徴量寄与度 & ウォークフォワード検証")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("#### 📊 特徴量寄与度 (Feature Importance)")
        if feat_importances:
            imp_df = pd.DataFrame(list(feat_importances.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation='h',
                marker_color='teal'
            ))
            fig_imp.update_layout(height=400, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("特徴量データが準備されていません。")

    with b_col2:
        st.markdown("#### 🧪 バックテスト (Walk-Forward Validation)")
        if st.button("🔄 バックテストを実行する", use_container_width=True):
            with st.spinner("直近データによる検証を実行中..."):
                avail = [f for f in FEATURE_COLUMNS if f in data.columns]
                X_bt = data[avail]
                y_bt = data["Target"]
                
                cum_wins, final_win_rate, t_count, c_count = run_backtest(X_bt, y_bt, test_len=100)
                
                st.metric("テスト取引回数", f"{t_count} 回")
                st.metric("正解数 / 最終勝率", f"{c_count} 回 / {final_win_rate:.1f}%")

                if cum_wins:
                    bt_df = pd.DataFrame(cum_wins, columns=["Step", "WinRate"])
                    fig_bt = go.Figure(go.Scatter(
                        x=bt_df["Step"], y=bt_df["WinRate"],
                        mode="lines+markers", name="勝率(%)", line=dict(color='springgreen')
                    ))
                    fig_bt.add_hline(y=50, line_dash="dash", line_color="gray")
                    fig_bt.update_layout(height=260, template="plotly_dark", title="検証勝率推移 (%)", margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.write("ボタンを押すと直近データを用いたウォークフォワード検証を行い、推移勝率を表示します。")

# ==========================================
# 9. 自動更新タイマーの発動
# ==========================================
if auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")
