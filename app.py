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
    page_title="プロ版 AI FXデイトレ & リピートアナライザー Pro v6.4.1",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLUMNS = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200", "Vol_Ratio",
    "RSI", "RSI_Diff", "MACD_Hist_Ratio", "BB_PctB", "ADX",
    "ATR_Ratio", "Upper_Wick_Ratio", "Lower_Wick_Ratio",
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

st.title("⚡ Pro AI FX デイトレ & リピートアナライザー (v6.4.1)")

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
        new_cols = {}
        new_cols["SMA_20"] = df["Close"].rolling(window=20).mean()
        new_cols["SMA_50"] = df["Close"].rolling(window=50).mean()
        new_cols["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        high_low = df["High"] - df["Low"]
        new_cols["ATR"] = high_low.rolling(window=14).mean()
        new_cols["ATR_SMA20"] = new_cols["ATR"].rolling(window=20).mean()
        new_cols["ATR_Ratio"] = new_cols["ATR"] / (new_cols["ATR_SMA20"] + 1e-10)

        total_range = high_low + 1e-10
        new_cols["Upper_Wick_Ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / total_range
        new_cols["Lower_Wick_Ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / total_range

        new_cols["Return_1"] = df["Close"].diff(1) / df["Close"].shift(1)
        new_cols["Return_5"] = df["Close"].diff(5) / df["Close"].shift(5)
        
        new_cols["Dev_SMA20"] = (df["Close"] - new_cols["SMA_20"]) / (new_cols["SMA_20"] + 1e-10)
        new_cols["Dev_EMA200"] = (df["Close"] - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Vol_Ratio"] = new_cols["ATR"] / (df["Close"] + 1e-10)

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        new_cols["RSI"] = 100.0 - (100.0 / (1.0 + rs))
        new_cols["RSI_Diff"] = new_cols["RSI"].diff(1)

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        new_cols["MACD"] = ema12 - ema26
        new_cols["MACD_Signal"] = new_cols["MACD"].ewm(span=9, adjust=False).mean()
        new_cols["MACD_Hist"] = new_cols["MACD"] - new_cols["MACD_Signal"]
        new_cols["MACD_Hist_Ratio"] = new_cols["MACD_Hist"] / (df["Close"] + 1e-10)

        std20 = df["Close"].rolling(window=20).std()
        new_cols["Upper_Band"] = new_cols["SMA_20"] + (std20 * 2)
        new_cols["Lower_Band"] = new_cols["SMA_20"] - (std20 * 2)
        new_cols["BB_Width"] = (new_cols["Upper_Band"] - new_cols["Lower_Band"]) / (new_cols["SMA_20"] + 1e-10)
        new_cols["BB_PctB"] = (df["Close"] - new_cols["Lower_Band"]) / ((new_cols["Upper_Band"] - new_cols["Lower_Band"]) + 1e-10)

        tr = pd.concat([high_low, (df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
        up_move = df["High"] - df["High"].shift(1)
        down_move = df["Low"].shift(1) - df["Low"]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
        sum_di = (plus_di + minus_di).replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / sum_di
        new_cols["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(25.0)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # 未来5本先まで監視期間を延ばして傾向を把握
        f_high = pd.concat([df["High"].shift(-i) for i in range(1, 6)], axis=1)
        f_low = pd.concat([df["Low"].shift(-i) for i in range(1, 6)], axis=1)

        target_pips = df["ATR"] * 0.8  
        future_max_up = f_high.max(axis=1) - df["Close"]
        future_max_down = df["Close"] - f_low.min(axis=1)

        conditions = [
            (future_max_up >= target_pips) & (future_max_up > future_max_down),
            (future_max_down >= target_pips) & (future_max_down > future_max_up),
        ]
        
        target_series = np.select(conditions, [1, -1], default=0)
        df["Target"] = target_series.astype(float)
        
        if len(df) > 5:
            df.iloc[-5:, df.columns.get_loc("Target")] = np.nan

        feature_cols = [c for c in df.columns if c != "Target"]
        df = df.dropna(subset=feature_cols)

        if df.empty:
            return None
        return df
    except Exception:
        return None

# MTFトレンド取得関数
@st.cache_data(ttl=60, show_spinner=False)
def get_mtf_trends(symbol: str) -> dict:
    trends = {}
    for name, params in TIMEFRAMES.items():
        sub_d = load_and_process_data(symbol, params["period"], params["interval"], name)
        if sub_d is not None and len(sub_d) >= 20:
            c_price = sub_d['Close'].iloc[-1]
            c_ema = sub_d['EMA_200'].iloc[-1] if 'EMA_200' in sub_d.columns else sub_d['SMA_20'].iloc[-1]
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
    step_size = max(1, test_len // 15)

    for i in range(0, test_len, step_size):
        idx = len(X_bt) - test_len + i
        train_end = max(1, idx - 3)
        valid_train = ~y_bt.iloc[:train_end].isna()
        if valid_train.sum() < 30:
            continue
        y_sub = y_bt.iloc[:train_end][valid_train]
        if len(np.unique(y_sub)) < 2:
            continue
        sub_model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=6, 
            min_samples_leaf=5, 
            random_state=42, 
            class_weight="balanced_subsample"
        )
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
def analyze_signal(df_current, df_higher, usdjpy_df=None, current_symbol="", n_estimators_override=None):
    if df_current is None or len(df_current) < 50:
        return "HOLD", 50.0, "不明"
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
            return "HOLD (分析不可: クラス不足)", 50.0, "判定不可"

        trees_count = n_estimators_override if n_estimators_override is not None else 120
        model = RandomForestClassifier(
            n_estimators=trees_count, 
            max_depth=6, 
            min_samples_leaf=5, 
            random_state=42, 
            class_weight="balanced_subsample"
        )
        model.fit(X_train, y_train)

        X_latest = X.iloc[[-1]]
        prob_array = model.predict_proba(X_latest)[0]
        prob_dict = dict(zip(model.classes_, prob_array))
        prob_up, prob_down = prob_dict.get(1, 0.0), prob_dict.get(-1, 0.0)
        confidence = max(prob_up, prob_down) * 100

        latest_price = df_current["Close"].iloc[-1]
        latest_ema200 = df_current["EMA_200"].iloc[-1] if "EMA_200" in df_current.columns else latest_price
        latest_adx = df_current["ADX"].iloc[-1] if "ADX" in df_current.columns else 25.0
        latest_rsi = df_current["RSI"].iloc[-1] if "RSI" in df_current.columns else 50.0
        latest_atr = df_current["ATR"].iloc[-1] if "ATR" in df_current.columns else 0.1
        upper_band = df_current["Upper_Band"].iloc[-1] if "Upper_Band" in df_current.columns else latest_price
        lower_band = df_current["Lower_Band"].iloc[-1] if "Lower_Band" in df_current.columns else latest_price
        
        if "BB_Width" in df_current.columns:
            latest_bb_width = df_current["BB_Width"].iloc[-1]
            avg_bb_series = df_current["BB_Width"].rolling(window=20).mean()
            avg_bb_width = avg_bb_series.iloc[-1] if not avg_bb_series.empty and not pd.isna(avg_bb_series.iloc[-1]) else 0.05
        else:
            latest_bb_width = 0.05
            avg_bb_width = 0.05
        
        is_squeezed = latest_bb_width < (avg_bb_width * 0.75)

        htf_trend = "FLAT"
        if df_higher is not None and not df_higher.empty and "EMA_200" in df_higher.columns:
            htf_trend = "UP" if df_higher["Close"].iloc[-1] > df_higher["EMA_200"].iloc[-1] else "DOWN" if df_higher["Close"].iloc[-1] < df_higher["EMA_200"].iloc[-1] else "FLAT"

        is_cross_jpy = "JPY" in current_symbol and current_symbol != "USDJPY=X"
        usdjpy_strong_up = usdjpy_strong_down = False
        if is_cross_jpy and usdjpy_df is not None and not usdjpy_df.empty and "EMA_200" in usdjpy_df.columns:
            uj_close, uj_ema = usdjpy_df["Close"].iloc[-1], usdjpy_df["EMA_200"].iloc[-1]
            uj_rsi = usdjpy_df["RSI"].iloc[-1] if "RSI" in usdjpy_df.columns else 50.0
            if uj_close > uj_ema and uj_rsi > 58: usdjpy_strong_up = True
            elif uj_close < uj_ema and uj_rsi < 42: usdjpy_strong_down = True

        recent_50_high = df_current["High"].iloc[-50:-1].max()
        recent_50_low = df_current["Low"].iloc[-50:-1].min()
        is_near_support = (latest_price - recent_50_low) < (latest_atr * 0.8)
        is_near_resistance = (recent_50_high - latest_price) < (latest_atr * 0.8)

        HIGH_THRESHOLD = 0.55
        # スキャン時など軽量実行(trees_count <= 80)の際は自動で確信度閾値を0.48に補正
        current_threshold = 0.48 if trees_count <= 80 else HIGH_THRESHOLD

        if latest_adx > 22.0:
            market_type = "トレンド相場"
            if prob_up >= current_threshold and htf_trend != "DOWN":
                if usdjpy_strong_down: status = "HOLD (ストッパー: ドル円急落中)"
                elif latest_price <= latest_ema200: status = "HOLD (逆張り警戒: 200EMA下)"
                elif is_near_resistance: status = "HOLD (抵抗線直前)"
                else: status = "BUY"
            elif prob_down >= current_threshold and htf_trend != "UP":
                if usdjpy_strong_up: status = "HOLD (ストッパー: ドル円急騰中)"
                elif latest_price >= latest_ema200: status = "HOLD (逆張り警戒: 200EMA上)"
                elif is_near_support: status = "HOLD (支持線直前)"
                else: status = "SELL"
            else: status = f"HOLD (確信度不足: {confidence:.1f}%)"
        else:
            market_type = "レンジ相場"
            if is_squeezed:
                status = "HOLD (ブレイクアウト警戒)"
            else:
                if (latest_price <= lower_band or latest_rsi <= 32.0) and prob_up >= 0.50:
                    status = "BUY (レンジ逆張り)" if not usdjpy_strong_down else "HOLD (ストッパー: ドル円逆行)"
                elif (latest_price >= upper_band or latest_rsi >= 68.0) and prob_down >= 0.50:
                    status = "SELL (レンジ逆張り)" if not usdjpy_strong_up else "HOLD (ストッパー: ドル円逆行)"
                else:
                    status = "HOLD (レンジ内静観)"

        return status, confidence, market_type
    except Exception:
        return "HOLD", 50.0, "不明"
# ==========================================
# 4. メインデータロード & 画面描画処理
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

if data is None or len(data) < 10:
    st.error("🚨 リアルタイムデータの取得に失敗しました。時間足を変更するか、1〜2分待ってから「最新データに更新」を押してください。")
else:
    market_status, confidence, market_type = analyze_signal(data, higher_tf_data, usdjpy_df=usdjpy_data, current_symbol=ticker)
    if is_econ_indicator_time and (market_status.startswith("BUY") or market_status.startswith("SELL")):
        market_status = "HOLD (指標発表警戒時間帯)"

    if enable_notify and discord_url and (market_status.startswith("BUY") or market_status.startswith("SELL")):
        if "last_notified_status" not in st.session_state:
            st.session_state["last_notified_status"] = {}
        last_sig_key = f"{ticker}_{tf_label}"
        last_status = st.session_state["last_notified_status"].get(last_sig_key, "")
        
        current_sig_type = get_signal_type(market_status)
        last_sig_type = get_signal_type(last_status)

        if current_sig_type != last_sig_type and current_sig_type != "HOLD":
            color_val = 0x2ECC71 if current_sig_type == "BUY" else 0xE74C3C
            msg_body = (
                f"**【{selected_label}】** のAIシグナルが発生しました！\n"
                f"⏱️ **時間軸**: {tf_label}\n"
                f"💵 **現在レート**: `{data['Close'].iloc[-1]:{price_fmt}}`\n"
                f"🤖 **AI判定**: **{market_status}**\n"
                f"🎯 **確信度**: `{confidence:.1f}%`\n"
                f"📊 **相場環境**: {market_type}"
            )
            ok, _ = send_discord_notification(discord_url, f"🚨 AI FXシグナル通知 [{selected_label}]", msg_body, color=color_val)
            if ok:
                st.session_state["last_notified_status"][last_sig_key] = market_status
                save_user_settings()

    available_features = [f for f in FEATURE_COLUMNS if f in data.columns]
    X_bt, y_bt = data[available_features], data["Target"]
    test_len = min(30, len(X_bt) - 10)

    if test_len > 5:
        with st.spinner("バックテストを実行中..."):
            cumulative_wins, win_rate, trade_count, correct_count = run_backtest(X_bt, y_bt, test_len)
    else:
        cumulative_wins, win_rate, trade_count, correct_count = [], 0.0, 0, 0

    if "BB_Width" in data.columns:
        latest_bb_width = data["BB_Width"].iloc[-1]
        avg_bb_series = data["BB_Width"].rolling(window=20).mean()
        avg_bb_width = avg_bb_series.iloc[-1] if not avg_bb_series.empty and not pd.isna(avg_bb_series.iloc[-1]) else 0.05
    else:
        latest_bb_width = 0.05
        avg_bb_width = 0.05
    is_squeezed = latest_bb_width < (avg_bb_width * 0.75)

    latest_adx = data["ADX"].iloc[-1] if "ADX" in data.columns else 25.0
    htf_close, htf_sma50 = data["Close"].iloc[-1], data["SMA_50"].iloc[-1]
    if higher_tf_data is not None and not higher_tf_data.empty and "SMA_50" in higher_tf_data.columns:
        htf_close, htf_sma50 = higher_tf_data["Close"].iloc[-1], higher_tf_data["SMA_50"].iloc[-1]

    if htf_close > htf_sma50 * 1.002: long_term_trend = "📈 強気上昇"
    elif htf_close < htf_sma50 * 0.998: long_term_trend = "📉 弱気下降"
    else: long_term_trend = "➡️ レンジ相場"

    latest_price = data["Close"].iloc[-1]
    latest_rsi = data["RSI"].iloc[-1] if "RSI" in data.columns else 50.0
    latest_atr = data["ATR"].iloc[-1] if "ATR" in data.columns else 0.1

    buffer_margin = 10 * pip_unit
    structural_buy_sl = round(data["Low"].iloc[-20:].min() - buffer_margin, 3 if is_jpy_pair else 5)
    structural_sell_sl = round(data["High"].iloc[-20:].max() + buffer_margin, 3 if is_jpy_pair else 5)

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
    pip_value_per_unit = 0.01 if is_jpy_pair else 0.0001 * (usdjpy_data["Close"].iloc[-1] if (usdjpy_data is not None and not usdjpy_data.empty) else 155.0)

    safe_single_units = max(100, min(int(allowed_loss_jpy / (sl_distance_pips * pip_value_per_unit)), 50000))
    safe_single_wan = round(safe_single_units / 10000.0, 4)

    if is_squeezed: st.error("⚡ **【スクイーズ発生】**: ボリンジャーバンドが収縮中です。ブレイクアウトにご注意ください。")

    # メトリクスカード表示
    st.divider()
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("現在レート", f"{latest_price:{price_fmt}}")
    m_col2.metric("AI識別・現在の相場環境", market_type, "🔥トレンド状態" if latest_adx > 22 else "💤レンジ・揉み合い")
    m_col3.metric("直近AI勝率 (トレード実行時)", f"{win_rate:.1f}% ({correct_count}勝/{trade_count}戦)" if trade_count > 0 else "N/A", "直近シグナル発生なし" if trade_count == 0 else None)

    m_col4, m_col5, m_col6 = st.columns(3)
    m_col4.metric("RSI (14)", f"{latest_rsi:.1f}")
    m_col5.metric("ADX (トレンド強度)", f"{latest_adx:.1f}")
    m_col6.metric("上位足 (日足) トレンド", long_term_trend)

    # マルチタイムフレーム (MTF) トレンド一覧パネル（高速化キャッシュ対応）
    st.markdown("##### 🌐 マルチタイムフレーム (MTF) トレンド一覧")
    mtf_trends = get_mtf_trends(ticker)
    mtf_cols = st.columns(len(TIMEFRAMES))
    for idx, (tf_name_key, t_val) in enumerate(mtf_trends.items()):
        mtf_cols[idx].metric(label=tf_name_key, value=t_val)

    st.divider()

    tab_repeat, tab_single, tab_speed, tab_chart, tab_scanner, tab_backtest, tab_metrics = st.tabs([
        "📋 リピート注文 (松井証券専用)", "🎯 デイトレ単発 (高精度AI)", "⚡ スピード注文", "📈 ローソク足チャート", "🔍 全ペアスキャン", "📊 バックテスト", "📋 注文履歴・成績サマリー"
    ])

    with tab_repeat:
        st.subheader("📋 松井証券FX 自動売買（リピート注文）入力用パラメータ")
        st.write(f"💡 **直近の相場変動幅 (ATR = {raw_atr_pips:.1f} pips) に連動し、値幅 `{ai_recommended_width} pips` を自動設定しました。**")
        jpy_rate = latest_price if is_jpy_pair else latest_price * (usdjpy_data["Close"].iloc[-1] if (usdjpy_data is not None and not usdjpy_data.empty) else 155.0)
        margin_per_unit = (jpy_rate * custom_quantity) / 25.0
        max_allowable_grids = max(2, int((account_balance * 0.5) / max(margin_per_unit, 1.0)))
        safe_half_range_val = max(1, max_allowable_grids // 2) * ai_recommended_width * pip_unit

        rep_lower, rep_upper = round(latest_price - safe_half_range_val, 3 if is_jpy_pair else 5), round(latest_price + safe_half_range_val, 3 if is_jpy_pair else 5)
        buffer_val = max(latest_atr * 1.5, 0.4 if is_jpy_pair else 0.04)
        rep_buy_stop, rep_sell_stop = round(rep_lower - buffer_val, 3 if is_jpy_pair else 5), round(rep_upper + buffer_val, 3 if is_jpy_pair else 5)
        buffer_pips = round(buffer_val / pip_unit, 1)

        total_est_wan = round((custom_quantity * max_allowable_grids) / 10000.0, 2)
        st.info(f"🛡️ **証拠金管理**: 口座資金 **{account_balance:,}円** に対し、1本あたり **{quantity_wan}万通貨 ({custom_quantity:,}通貨)** で同時設定できる安全上限は **{max_allowable_grids}本** です。（想定最大必要証拠金: 約{int(margin_per_unit * max_allowable_grids):,}円）")

        rep_c1, rep_c2 = st.columns(2)
        with rep_c1:
            st.markdown("#### 🟢 買いリピート設定 (BUY)")
            st.code(f"通貨ペア    : {selected_label}\n売買区分    : 買\nレンジ上限  : {rep_upper}\nレンジ下限  : {rep_lower}\n数量（万）  : {quantity_wan}  (※松井アプリ用)\n注文値幅    : {ai_recommended_width} pips  (ATR自動連動)\n益出し幅    : {ai_recommended_width} pips  (ATR自動連動)\n運用停止ライン: {rep_buy_stop} (-{buffer_pips}pips)\n----------------------------------------\n【構成案内】最大仕掛け本数: {max_allowable_grids}本 ({total_est_wan}万通貨分)", language="text")
        with rep_c2:
            st.markdown("#### 🔴 売りリピート設定 (SELL)")
            st.code(f"通貨ペア    : {selected_label}\n売買区分    : 売\nレンジ上限  : {rep_upper}\nレンジ下限  : {rep_lower}\n数量（万）  : {quantity_wan}  (※松井アプリ用)\n注文値幅    : {ai_recommended_width} pips  (ATR自動連動)\n益出し幅    : {ai_recommended_width} pips  (ATR自動連動)\n運用停止ライン: {rep_sell_stop} (+{buffer_pips}pips)\n----------------------------------------\n【構成案内】最大仕掛け本数: {max_allowable_grids}本 ({total_est_wan}万通貨分)", language="text")

    with tab_single:
        st.subheader("🎯 デイトレ単発トレード（高精度ノイズ除去モデル）")
        st.info(f"🛡️ **口座資金 ({account_balance:,}円) に基づく単発適正数量ガイド**: 1回のリスクを資金2%（{int(allowed_loss_jpy):,}円）以下に抑える推奨注文数量は **`{safe_single_wan}万通貨` ({safe_single_units:,}通貨)** です。")
        
        tp_pips_val = abs(calc_tp - latest_price) / pip_unit
        sl_pips_val = abs(latest_price - calc_sl) / pip_unit
        rr_ratio = (tp_pips_val / sl_pips_val) if sl_pips_val > 0 else 0.0

        if market_status.startswith("BUY"):
            st.success(f"🟢 **高確信 買いシグナル確定 ({market_status})** （確信度: {confidence:.1f}% | RR比: {rr_ratio:.2f}）")
            t_col1, t_col2, t_col3 = st.columns(3)
            t_col1.metric("新規買い目安", f"{latest_price:{price_fmt}}"); t_col1.code(f"{latest_price:{price_fmt}}", language="text")
            t_col2.metric("利確目標 (TP)", f"{calc_tp:{price_fmt}}", f"+{tp_pips_val:.1f} pips"); t_col2.code(f"{calc_tp:{price_fmt}}", language="text")
            t_col3.metric("防衛SL", f"{calc_sl:{price_fmt}}", f"-{sl_pips_val:.1f} pips"); t_col3.code(f"{calc_sl:{price_fmt}}", language="text")
        elif market_status.startswith("SELL"):
            st.error(f"🔴 **高確信 売りシグナル確定 ({market_status})** （確信度: {confidence:.1f}% | RR比: {rr_ratio:.2f}）")
            t_col1, t_col2, t_col3 = st.columns(3)
            t_col1.metric("新規売り目安", f"{latest_price:{price_fmt}}"); t_col1.code(f"{latest_price:{price_fmt}}", language="text")
            t_col2.metric("利確目標 (TP)", f"{calc_tp:{price_fmt}}", f"-{tp_pips_val:.1f} pips"); t_col2.code(f"{calc_tp:{price_fmt}}", language="text")
            t_col3.metric("防衛SL", f"{calc_sl:{price_fmt}}", f"+{sl_pips_val:.1f} pips"); t_col3.code(f"{calc_sl:{price_fmt}}", language="text")
        else:
            st.warning(f"🟡 **静観フィルター発動中 ({market_status})**")
            st.info("💡 **解説:** ノイズ除去・ドル円連動ストッパー・指標発表警戒帯等の安全装置により、騙しリスクが高い場面では自動で「HOLD」判定になります。")
    
    with tab_speed:
        st.subheader("⚡ 松井証券FX アプリ【スピード注文】設定用")
        sp_tp_pips = round(abs(calc_tp - latest_price) / pip_unit, 1)
        sp_sl_pips = round(abs(latest_price - calc_sl) / pip_unit, 1)
        
        sp_col1, sp_col2, sp_col3 = st.columns(3)
        sp_col1.metric("資金ベース推奨数量 (万)", f"{safe_single_wan}万 ({safe_single_units:,}通貨)")
        sp_col2.metric("益出し幅 (利確)", f"{sp_tp_pips} pips")
        sp_col3.metric("防衛損切り幅 (損切)", f"{sp_sl_pips} pips")

        rec_dir = "買い (ASK)" if market_status.startswith("BUY") else "売り (BID)" if market_status.startswith("SELL") else "様子見 (静観)"
        st.code(f"通貨ペア: {selected_label}\n推奨エントリー: {rec_dir}\n【推奨安全数量】: {safe_single_wan} 万通貨 ({safe_single_units:,} 通貨)\n益出し幅: {sp_tp_pips} pips\n損切り幅: {sp_sl_pips} pips\n許容スリッページ: {recommended_slippage} pips", language="text")

    with tab_chart:
        st.subheader("📈 Pro仕様 インタラクティブ・ローソク足チャート (Plotly)")
        df_chart = data.tail(60).copy()

        try:
            if df_chart.index.tz is None:
                df_chart.index = df_chart.index.tz_localize("UTC")
            df_chart.index = df_chart.index.tz_convert("Asia/Tokyo")
        except Exception:
            pass

        is_daily = "日足" in tf_label
        chart_x = df_chart.index.strftime("%Y-%m-%d" if is_daily else "%m-%d %H:%M")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=chart_x, open=df_chart["Open"], high=df_chart["High"], low=df_chart["Low"], close=df_chart["Close"], name="ローソク足"), row=1, col=1)

        for col, color, width, dash in [("SMA_20", "orange", 1, "solid"), ("SMA_50", "blue", 1, "solid"), ("EMA_200", "white", 1.5, "solid"), ("Upper_Band", "gray", 1, "dash"), ("Lower_Band", "gray", 1, "dash")]:
            if col in df_chart.columns:
                fig.add_trace(go.Scatter(x=chart_x, y=df_chart[col], mode="lines", name=col, line=dict(color=color, width=width, dash=dash)), row=1, col=1)
        
        if market_status.startswith("BUY") or market_status.startswith("SELL"):
            fig.add_hline(y=calc_tp, line_dash="dash", line_color="#2ECC71", annotation_text="TP (利確目安)", row=1, col=1)
            fig.add_hline(y=calc_sl, line_dash="dash", line_color="#E74C3C", annotation_text="SL (損切目安)", row=1, col=1)

        if "RSI" in df_chart.columns:
            fig.add_trace(go.Scatter(x=chart_x, y=df_chart["RSI"], mode="lines", name="RSI(14)", line=dict(color="purple", width=1.5)), row=2, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_xaxes(type="category", nticks=10)
        fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=5, r=5, t=20, b=5), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab_scanner:
        st.subheader("🔍 全監視通貨ペア AI防衛スキャン")
        if st.button("🚀 全ペアを一括スキャン実行", use_container_width=True):
            scan_results = []
            progress_bar_scan, status_text_scan = st.progress(0), st.empty()
            
            with st.spinner("全通貨ペアを分析中..."):
                for idx_p, (p_label, p_symbol) in enumerate(PAIRS.items()):
                    status_text_scan.text(f"スキャン中... {p_label}")
                    sub_df = load_and_process_data(p_symbol, tf_config["period"], tf_config["interval"], tf_label)
                    sub_htf = load_and_process_data(p_symbol, "1y", "1d", "日足 (スイング・環境認識用)")
                    if sub_df is not None and len(sub_df) > 10:
                        # スキャン精度向上のため、学習の深さを80本に引き上げ
                        s_status, s_conf, s_mtype = analyze_signal(sub_df, sub_htf, usdjpy_df=usdjpy_data, current_symbol=p_symbol, n_estimators_override=80)
                        scan_results.append({
                            "通貨ペア": p_label, "相場環境": s_mtype, "AI総合判定": s_status, "確信度 (%)": round(s_conf, 1),
                            "ADX (強度)": round(sub_df["ADX"].iloc[-1] if "ADX" in sub_df.columns else 25.0, 1)
                        })
                    progress_bar_scan.progress(min(1.0, (idx_p + 1) / len(PAIRS)))

            progress_bar_scan.empty(); status_text_scan.empty()
            if scan_results:
                st.dataframe(pd.DataFrame(scan_results).sort_values(by="確信度 (%)", ascending=False), use_container_width=True)

    with tab_backtest:
        st.subheader("📊 改良型モデルの時系列ウォークフォワード検証")
        st.caption("※AIが相場状況を危険と判断し、「HOLD（静観）」としてエントリーを見送ったステップは分母から除外した『純粋なトレード実行勝率』を表示しています。")
        if cumulative_wins:
            st.line_chart(pd.DataFrame(cumulative_wins, columns=["検証ステップ", "累積適合率 (%)"]).set_index("検証ステップ"))
            st.metric("時系列検証の適合率（トレード実行時）", f"{win_rate:.1f}% ({correct_count}回適合 / {trade_count}回エントリー)" if trade_count > 0 else "N/A", None if trade_count > 0 else "直近シグナル発生なし")

    with tab_metrics:
        st.subheader("📋 注文履歴・成績サマリー")
        st.info("💡 アプリの稼働状況や、設定されている資金・ロット数のサマリーをここで一元管理できます。")
        sum_c1, sum_c2 = st.columns(2)
        sum_c1.metric("現在の口座資金", f"{account_balance:,} 円"); sum_c1.metric("設定中の注文ロット数", f"{quantity_wan} 万通貨 ({custom_quantity:,} 通貨)")
        sum_c2.metric("選択中の通貨ペア", selected_label); sum_c2.metric("選択中の時間足", tf_label)

    st.divider()
    with st.expander("📄 学習データテーブル確認（相対化済みの特徴量）"):
        st.dataframe(data[available_features + ["ATR", "BB_Width"]].tail(10))

if auto_refresh:
    st.caption(f"🔄 自動更新が有効です ({refresh_interval}秒ごと)")
    st_autorefresh(interval=refresh_interval * 1000, limit=100, key="data_refresh")
