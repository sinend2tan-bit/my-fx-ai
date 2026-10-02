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

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ==========================================
# 0. 画面基本設定 & CSSデザイン定義
# ==========================================
st.set_page_config(
    page_title="AI FX 環境認識 & リピートアナライザー Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1280px; }
    label[data-testid="stWidgetLabel"] p { font-size: 1.05rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 1.0rem !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 800 !important; }
    [data-testid="stCaptionContainer"], .stCaption p { font-size: 0.95rem !important; font-weight: 600 !important; }
    
    .status-badge-buy { background-color: #15803d; color: #ffffff; border: 1px solid #16a34a; padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 1.15rem; display: inline-block; }
    .status-badge-sell { background-color: #b91c1c; color: #ffffff; border: 1px solid #dc2626; padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 1.15rem; display: inline-block; }
    .status-badge-hold { background-color: #854d0e; color: #ffffff; border: 1px solid #ca8a04; padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 1.15rem; display: inline-block; }
    
    .param-box { background-color: rgba(30, 41, 59, 0.8) !important; border-left: 6px solid #3b82f6; padding: 18px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 1.05rem; line-height: 1.8; color: #f8fafc !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .param-box b.label-title { color: #94a3b8 !important; }
    .param-box code { font-size: 1.1rem !important; font-weight: 700 !important; padding: 2px 8px !important; background-color: rgba(51, 65, 85, 0.8) !important; color: #38bdf8 !important; border: 1px solid #475569; border-radius: 4px; }
    .param-box-buy { border-left-color: #22c55e !important; }
    .param-box-sell { border-left-color: #ef4444 !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 18px; font-size: 1.0rem !important; font-weight: 700; border-radius: 6px 6px 0 0; }
    .stTabs [aria-selected="true"] { color: #0284c7 !important; border-bottom-color: #0284c7 !important; }
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
            pass
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
    # 安全ガード：選択項目が定義に存在しない場合は初期値に戻す
    if st.session_state.get("selected_pair_label") not in PAIRS:
        st.session_state["selected_pair_label"] = DEFAULT_SETTINGS["selected_pair_label"]
    if st.session_state.get("selected_tf_label") not in TIMEFRAMES:
        st.session_state["selected_tf_label"] = DEFAULT_SETTINGS["selected_tf_label"]
    st.session_state["initialized"] = True

def send_discord_notification(webhook_url, title, message, color=0x00FF00):
    if not webhook_url:
        return False, "URL未設定"
    payload = {
        "embeds": [{
            "title": title, "description": message, "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
    }
    try:
        res = requests.post(webhook_url, json=payload, timeout=5)
        if res.status_code in [200, 204]: return True, "送信成功"
        else: return False, f"ステータスコード: {res.status_code}"
    except Exception as e:
        return False, str(e)

def clean_series(s):
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s

def get_upcoming_market_events(now_jst, is_summer):
    h, m = now_jst.hour, now_jst.minute
    current_total_min = h * 60 + m

    london_open = 16 * 60 if is_summer else 17 * 60
    us_econ_indicator = (21 * 60 + 30) if is_summer else (22 * 60 + 30)
    ny_open = (22 * 60 + 30) if is_summer else (23 * 60 + 30)
    early_morning = 3 * 60

    events = [
        {"name": "欧州市場オープン (ロンドン)", "time_str": "16:00" if is_summer else "17:00", "min": london_open},
        {"name": "米国 主要経済指標発表", "time_str": "21:30" if is_summer else "22:30", "min": us_econ_indicator},
        {"name": "米国市場オープン (NY)", "time_str": "22:30" if is_summer else "23:30", "min": ny_open},
        {"name": "早朝・広スプレッド警戒帯", "time_str": "03:00", "min": early_morning},
    ]

    upcoming = []
    for ev in events:
        diff = ev["min"] - current_total_min
        if diff < -120:
            diff += 24 * 60
        upcoming.append({"event": ev["name"], "time": ev["time_str"], "left_min": diff})

    upcoming.sort(key=lambda x: x["left_min"])
    return upcoming

# ==========================================
# 2. データ取得 & インジケーター計算エンジン
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    df = pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        
        # 2次元・階層型カラム（MultiIndex）の整理
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    except Exception:
        return None

    if "4時間足" in tf_name and interval == "1h":
        try:
            tz_before = df.index.tz
            df = df.resample("4h", closed="left", label="left").agg({
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

        # 疑似トリプルバリア法によるTarget生成
        lookahead = 5
        f_high = pd.concat([h_series.shift(-i) for i in range(1, lookahead + 1)], axis=1)
        f_low = pd.concat([l_series.shift(-i) for i in range(1, lookahead + 1)], axis=1)

        tp_target = new_cols["ATR"] * 1.0
        sl_target = new_cols["ATR"] * 0.5

        future_max_up = f_high.max(axis=1) - c_series
        future_max_down = c_series - f_low.min(axis=1)

        cond_buy = (future_max_up >= tp_target) & (future_max_down < sl_target)
        cond_sell = (future_max_down >= tp_target) & (future_max_up < sl_target)

        target_array = np.select([cond_buy, cond_sell], [1, -1], default=0)
        df["Target"] = pd.Series(target_array, index=df.index, dtype=float)
        
        if len(df) > lookahead: 
            df.iloc[-lookahead:, df.columns.get_loc("Target")] = np.nan

        feature_cols = [c for c in df.columns if c != "Target"]
        df = df.dropna(subset=feature_cols)

        if df.empty: return None
        return df
    except Exception:
        return None
# ==========================================
# パート2: AI学習モデル・バックテスト・MTF分析
# ==========================================
@st.cache_data(ttl=120, show_spinner=False)
def get_mtf_trends(symbol: str) -> dict:
    trends = {}
    for name, params in TIMEFRAMES.items():
        sub_d = load_and_process_data(symbol, params["period"], params["interval"], name)
        if sub_d is not None and len(sub_d) >= 20:
            c_price = float(clean_series(sub_d['Close']).iloc[-1])
            ema_series = clean_series(sub_d['EMA_200']) if 'EMA_200' in sub_d.columns else clean_series(sub_d['SMA_20']) if 'SMA_20' in sub_d.columns else None
            if ema_series is not None:
                c_ema = float(ema_series.iloc[-1])
                if c_price > c_ema: trends[name.split(" ")[0]] = "上昇 📈"
                elif c_price < c_ema: trends[name.split(" ")[0]] = "下降 📉"
                else: trends[name.split(" ")[0]] = "レンジ ➡️"
            else:
                trends[name.split(" ")[0]] = "判定不能"
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
        if valid_train.sum() < 30: continue
        
        y_sub = y_bt.iloc[:train_end][valid_train]
        if len(np.unique(y_sub)) < 2: continue
        
        sub_model = RandomForestClassifier(n_estimators=60, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42)
        sub_model.fit(X_bt.iloc[:train_end][valid_train], y_sub)
        p = sub_model.predict(X_bt.iloc[[idx]])[0]
        actual = y_bt.iloc[idx]

        if p != 0 and not pd.isna(actual):
            trade_count += 1
            if p == actual: correct_count += 1
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
            latest_bb_width, avg_bb_width = 0.05, 0.05
            
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

        h_series_curr, l_series_curr = clean_series(df_current["High"]), clean_series(df_current["Low"])
        recent_50_high, recent_50_low = float(h_series_curr.iloc[-50:-1].max()), float(l_series_curr.iloc[-50:-1].min())
        is_far_from_sma = abs(latest_price - latest_sma20) > (latest_atr * 1.5)
        is_near_support = (latest_price - recent_50_low) < (latest_atr * 0.8)
        is_near_resistance = (recent_50_high - latest_price) < (latest_atr * 0.8)

        HIGH_THRESHOLD = 0.65

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
            if is_squeezed: status = "HOLD (ブレイクアウト警戒)"
            else:
                if (latest_price <= lower_band or latest_rsi <= 32.0) and prob_up >= 0.58:
                    status = "BUY (レンジ逆張り)" if not usdjpy_strong_down else "HOLD (ストッパー: ドル円逆行)"
                elif (latest_price >= upper_band or latest_rsi >= 68.0) and prob_down >= 0.58:
                    status = "SELL (レンジ逆張り)" if not usdjpy_strong_up else "HOLD (ストッパー: ドル円逆行)"
                else: status = "HOLD (レンジ内静観)"
        return status, confidence, market_type, importances
    except Exception:
        return "HOLD", 50.0, "不明", {}
# ==========================================
# パート3: メインUI・ダッシュボード描画
# ==========================================
st.title("AI FX 環境認識 & リピートアナライザー Pro")

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

st.sidebar.header("⚙️ システム設定")
if st.sidebar.button("🔄 最新データに更新", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("自動更新を有効にする", key="auto_refresh", on_change=save_user_settings)
refresh_interval = st.sidebar.selectbox("更新間隔", options=[60, 180, 300], format_func=lambda x: f"{x // 60}分ごと", key="refresh_interval", on_change=save_user_settings)

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
        if ok: st.sidebar.success("送信完了")
        else: st.sidebar.error(f"送信失敗: {msg}")
    else: st.sidebar.warning("URLを入力してください")

with st.spinner("最新市場データとAIモデルを読込中..."):
    usdjpy_data = load_and_process_data("USDJPY=X", tf_config["period"], tf_config["interval"], tf_label)
    data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
    higher_tf_data = load_and_process_data(ticker, "1y", "1d", "日足 (スイング・環境認識用)")

now_datetime_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
current_day, current_hour_jst, current_minute_jst = now_datetime_jst.weekday(), now_datetime_jst.hour, now_datetime_jst.minute

now_datetime_ny = now_datetime_jst.astimezone(ZoneInfo("America/New_York"))
is_summer_time = now_datetime_ny.dst().total_seconds() != 0 

is_weekend = (current_day == 5 and current_hour_jst >= 6) or (current_day == 6) or (current_day == 0 and current_hour_jst < 6)

if is_summer_time:
    is_econ_indicator_time = (current_hour_jst == 21 and current_minute_jst >= 15) or (current_hour_jst == 22 and current_minute_jst <= 45)
else:
    is_econ_indicator_time = (current_hour_jst == 22 and current_minute_jst >= 15) or (current_hour_jst == 23 and current_minute_jst <= 45)

if is_weekend: st.error("【週末クローズ中】現在市場は休業時間帯です。表示レートは最終終値となります。")
if data is None or len(data) < 10:
    st.error("データ取得に失敗しました。時間足を切り替えるか少し置いて再実行してください。")
else:
    market_status, confidence, market_type, importances = analyze_signal(data, higher_tf_data, usdjpy_df=usdjpy_data, current_symbol=ticker)
    if is_econ_indicator_time and (market_status.startswith("BUY") or market_status.startswith("SELL")):
        market_status = "HOLD (指標警戒帯)"

    c_close = clean_series(data["Close"])
    latest_price = float(c_close.iloc[-1])

    upcoming_events = get_upcoming_market_events(now_datetime_jst, is_summer_time)
    next_ev = upcoming_events[0]

    bb_w_series = clean_series(data["BB_Width"]) if "BB_Width" in data.columns else None
    is_pre_breakout = False
    if bb_w_series is not None and len(bb_w_series) >= 20:
        current_w = bb_w_series.iloc[-1]
        min_20_w = bb_w_series.iloc[-20:].min()
        if current_w <= min_20_w * 1.05:
            is_pre_breakout = True

    with st.container():
        ev_col1, ev_col2 = st.columns([2, 1])
        with ev_col1:
            if next_ev["left_min"] <= 60:
                st.error(f"⏰ **【直前警戒】{next_ev['event']} ({next_ev['time']}) まで あと {next_ev['left_min']} 分！**\nトレンド急変の危険があります。新規注文・リピート設定に十分ご注意ください。")
            elif next_ev["left_min"] <= 120:
                st.warning(f"⏳ **【事前察知】{next_ev['event']} ({next_ev['time']}) まで あと {next_ev['left_min']} 分**")
            else:
                st.info(f"🟢 次の注目イベント: **{next_ev['event']} ({next_ev['time']})** (あと {next_ev['left_min'] // 60}時間{next_ev['left_min'] % 60}分)")

        with ev_col2:
            if is_pre_breakout:
                st.error("💥 **ボラティリティ爆発の予兆**\nレンジ極小収縮中。上下どちらかへの強烈なブレイクアウトに警戒。")
            else:
                st.success("🌊 ボラティリティ: 正常レンジ内")

    with st.container():
        m_head1, m_head2, m_head3, m_head4 = st.columns([1.2, 1.5, 1, 1])
        m_head1.metric("現在レート", f"{latest_price:{price_fmt}}")
        
        if market_status.startswith("BUY"): badge_html = f'<div class="status-badge-buy">BUY 買い ({confidence:.1f}%)</div>'
        elif market_status.startswith("SELL"): badge_html = f'<div class="status-badge-sell">SELL 売り ({confidence:.1f}%)</div>'
        else: badge_html = f'<div class="status-badge-hold">{market_status}</div>'
            
        m_head2.markdown("**AI総合環境認識 / 確信度**")
        m_head2.markdown(badge_html, unsafe_allow_html=True)
        
        latest_adx = float(clean_series(data["ADX"]).iloc[-1]) if "ADX" in data.columns else 25.0
        m_head3.metric("相場環境", market_type, "トレンド" if latest_adx > 22 else "レンジ")
        
        available_features = [f for f in FEATURE_COLUMNS if f in data.columns]
        X_bt, y_bt = data[available_features], data["Target"]
        test_len = min(30, len(X_bt) - 10)
        cumulative_wins, win_rate, trade_count, correct_count = run_backtest(X_bt, y_bt, test_len) if test_len > 5 else ([], 0.0, 0, 0)
        m_head4.metric("直近AI勝率", f"{win_rate:.1f}%" if trade_count > 0 else "N/A", f"{correct_count}勝 / {trade_count}戦")

    with st.container():
        sec1, sec2, sec3, sec4 = st.columns(4)
        latest_rsi = float(clean_series(data["RSI"]).iloc[-1]) if "RSI" in data.columns else 50.0
        latest_atr = float(clean_series(data["ATR"]).iloc[-1]) if "ATR" in data.columns else 0.1
        
        htf_close = latest_price
        htf_sma50 = float(clean_series(data["SMA_50"]).iloc[-1]) if "SMA_50" in data.columns else latest_price
        if higher_tf_data is not None and not higher_tf_data.empty and "SMA_50" in higher_tf_data.columns:
            htf_close = float(clean_series(higher_tf_data["Close"]).iloc[-1])
            htf_sma50 = float(clean_series(higher_tf_data["SMA_50"]).iloc[-1])

        if htf_close > htf_sma50 * 1.002: long_term_trend = "上昇 📈"
        elif htf_close < htf_sma50 * 0.998: long_term_trend = "下降 📉"
        else: long_term_trend = "レンジ ➡️"

        raw_atr_pips = latest_atr / pip_unit
        recommended_slippage = round(max(0.5, raw_atr_pips * 0.05), 1)

        sec1.metric("RSI (14)", f"{latest_rsi:.1f}")
        sec2.metric("ADX (トレンド強度)", f"{latest_adx:.1f}")
        sec3.metric("日足 トレンド", long_term_trend)
        sec4.metric("適正スリッページ", f"{recommended_slippage} pips")

    st.markdown("##### 🌐 マルチタイムフレーム (MTF) トレンド")
    mtf_trends = get_mtf_trends(ticker)
    mtf_cols = st.columns(len(TIMEFRAMES))
    for idx, (tf_name_key, t_val) in enumerate(mtf_trends.items()):
        mtf_cols[idx].metric(label=tf_name_key, value=t_val)
    st.markdown("---")
    
    tab_repeat, tab_single, tab_speed, tab_chart, tab_scanner, tab_backtest, tab_metrics = st.tabs([
        "📋 リピート注文 (松井証券)", "🎯 デイトレ参考 (AI)", "⚡ スピード注文", "📈 チャート", "🔍 全ペアスキャン", "📊 バックテスト", "📋 運用サマリー"
    ])

    ai_recommended_width = int(max(round(raw_atr_pips * atr_cfg["atr_mult"], 1), atr_cfg["min_pips"]))

    c_low, c_high = clean_series(data["Low"]), clean_series(data["High"])
    recent_50_high = float(c_high.iloc[-50:].max())
    recent_50_low = float(c_low.iloc[-50:].min())
    range_pips = (recent_50_high - recent_50_low) / pip_unit

    base_grids = int(range_pips / max(1, ai_recommended_width))
    
    uj_rate = float(clean_series(usdjpy_data["Close"]).iloc[-1]) if (usdjpy_data is not None and not usdjpy_data.empty) else 155.0
    jpy_rate = latest_price if is_jpy_pair else latest_price * uj_rate
    margin_per_unit = (jpy_rate * custom_quantity) / 25.0
    
    max_allowable_grids = max(3, int((account_balance * 0.5) / max(margin_per_unit, 1.0)))

    ai_grids = min(max_allowable_grids, max(3, base_grids))

    if ai_grids % 2 == 0:
        ai_grids -= 1
    if ai_grids < 3:
        ai_grids = 3
        
    target_grids = ai_grids

    with tab_repeat:
        st.subheader("📋 松井証券FX 自動売買（リピート注文）最適化ヘルパー")
        
        side_grids = (target_grids - 1) // 2  
        
        p_decimals = 3 if is_jpy_pair else 5
        grid_width_val = round(ai_recommended_width * pip_unit, p_decimals)

        center_price = round(round(latest_price / max(1e-5, grid_width_val)) * grid_width_val, p_decimals)
        
        rep_lower = round(center_price - (side_grids * grid_width_val), p_decimals)
        rep_upper = round(center_price + (side_grids * grid_width_val), p_decimals)

        buffer_val = max(latest_atr * 1.5, 0.4 if is_jpy_pair else 0.04)
        rep_buy_stop = round(rep_lower - buffer_val, p_decimals)
        rep_sell_stop = round(rep_upper + buffer_val, p_decimals)
        buffer_pips = round(buffer_val / pip_unit, 1)

        st.info(
            f"✨ **AI最適化完了**: 現在のボラティリティ（直近の変動幅 {round(range_pips, 1)} pips）と"
            f"口座資金を元に、最適な注文本数を **{target_grids}本** と自動判定しました。"
        )
        
        st.caption(
            f"💡 ATR基準の推測注文値幅: **{ai_recommended_width} pips** | "
            f"AI設定モード: **【{target_grids}本モード (中心価格±{side_grids}本)】** | "
            f"資金許容最大: **約{max_allowable_grids}本**"
        )

        rep_c1, rep_c2 = st.columns(2)
        with rep_c1:
            st.markdown("#### 🟢 買いリピート推奨設定")
            st.markdown(f"""
            <div class="param-box param-box-buy">
            <b class="label-title">通貨ペア</b>    : {selected_label}<br>
            <b class="label-title">売買区分</b>    : 買<br>
            <b class="label-title">レンジ上限</b>  : <code>{rep_upper}</code><br>
            <b class="label-title">レンジ下限</b>  : <code>{rep_lower}</code><br>
            <b class="label-title">数量（万）</b>  : <code>{quantity_wan}</code><br>
            <b class="label-title">注文値幅</b>    : <code>{ai_recommended_width} pips</code><br>
            <b class="label-title">益出し幅</b>    : <code>{ai_recommended_width} pips</code><br>
            <b class="label-title">運用停止(SL)</b>: <code>{rep_buy_stop}</code> (-{buffer_pips}pips)
            </div>
            """, unsafe_allow_html=True)
            
        with rep_c2:
            st.markdown("#### 🔴 売りリピート推奨設定")
            st.markdown(f"""
            <div class="param-box param-box-sell">
            <b class="label-title">通貨ペア</b>    : {selected_label}<br>
            <b class="label-title">売買区分</b>    : 売<br>
            <b class="label-title">レンジ上限</b>  : <code>{rep_upper}</code><br>
            <b class="label-title">レンジ下限</b>  : <code>{rep_lower}</code><br>
            <b class="label-title">数量（万）</b>  : <code>{quantity_wan}</code><br>
            <b class="label-title">注文値幅</b>    : <code>{ai_recommended_width} pips</code><br>
            <b class="label-title">益出し幅</b>    : <code>{ai_recommended_width} pips</code><br>
            <b class="label-title">運用停止(SL)</b>: <code>{rep_sell_stop}</code> (+{buffer_pips}pips)
            </div>
            """, unsafe_allow_html=True)
            
        total_margin_req = margin_per_unit * target_grids
        usage_percent = (total_margin_req / max(1.0, account_balance)) * 100
        if usage_percent > 70:
            st.error(f"⚠️ 警告: 証拠金使用率が約 {usage_percent:.1f}% に達します。数量を減らすか、資金を追加してください。")
        elif usage_percent > 40:
            st.warning(f"⚠️ 注意: 証拠金使用率が約 {usage_percent:.1f}% です。急変動時の含み損に注意してください。")

    ai_tp_mult = 1.0  
    ai_sl_mult = 0.5  
    
    buffer_margin = 10 * pip_unit
    structural_buy_sl = round(float(c_low.iloc[-20:].min()) - buffer_margin, 3 if is_jpy_pair else 5)
    structural_sell_sl = round(float(c_high.iloc[-20:].max()) + buffer_margin, 3 if is_jpy_pair else 5)

    if market_status.startswith("BUY"):
        calc_tp = latest_price + (latest_atr * ai_tp_mult)
        calc_sl = min(latest_price - (latest_atr * ai_sl_mult), structural_buy_sl)
    elif market_status.startswith("SELL"):
        calc_tp = latest_price - (latest_atr * ai_tp_mult)
        calc_sl = max(latest_price + (latest_atr * ai_sl_mult), structural_sell_sl)
    else:
        calc_tp = latest_price + (latest_atr * ai_tp_mult)
        calc_sl = latest_price - (latest_atr * ai_sl_mult)

    sl_distance_pips = max(10.0, round((latest_atr * ai_sl_mult) / pip_unit, 1))
    pip_value_per_unit = 0.01 if is_jpy_pair else 0.0001 * uj_rate
    safe_single_units = max(100, min(int((account_balance * 0.02) / max(0.01, sl_distance_pips * pip_value_per_unit)), 50000))
    safe_single_wan = round(safe_single_units / 10000.0, 4)

    with tab_single:
        st.subheader("🎯 単発デイトレ分析（環境認識用）")
        tp_pips_val = abs(calc_tp - latest_price) / pip_unit
        sl_pips_val = abs(latest_price - calc_sl) / pip_unit
        rr_ratio = (tp_pips_val / sl_pips_val) if sl_pips_val > 0 else 0.0

        if market_status.startswith("BUY"): st.success(f"🟢 **買いシグナル傾向** (確信度: {confidence:.1f}% | R:R比: {rr_ratio:.2f})")
        elif market_status.startswith("SELL"): st.error(f"🔴 **売りシグナル傾向** (確信度: {confidence:.1f}% | R:R比: {rr_ratio:.2f})")
        else: st.warning(f"🟡 **静観判定適用中 ({market_status})**")

        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("推奨単発ロット", f"{safe_single_wan} 万通貨")
        t_col2.metric("現在目安", f"{latest_price:{price_fmt}}")
        t_col3.metric("目標利確 (TP)", f"{calc_tp:{price_fmt}}", f"+{tp_pips_val:.1f} pips")
        t_col4.metric("撤退損切 (SL)", f"{calc_sl:{price_fmt}}", f"-{sl_pips_val:.1f} pips")

        if importances:
            with st.expander("🧠 AI判断の注目指標 TOP 5"):
                imp_df = pd.DataFrame(list(importances.items()), columns=["指標名", "影響度"]).sort_values(by="影響度", ascending=False).head(5)
                st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)

    with tab_speed:
        st.subheader("⚡ スピード注文 設定参照")
        sp_tp_pips = round(abs(calc_tp - latest_price) / pip_unit, 1)
        sp_sl_pips = round(abs(latest_price - calc_sl) / pip_unit, 1)
        
        sp_col1, sp_col2, sp_col3 = st.columns(3)
        sp_col1.metric("推奨ロット", f"{safe_single_wan} 万通貨")
        sp_col2.metric("益出し幅", f"{sp_tp_pips} pips")
        sp_col3.metric("損切り幅", f"{sp_sl_pips} pips")

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
            x=chart_x, open=clean_series(df_chart["Open"]), high=clean_series(df_chart["High"]),
            low=clean_series(df_chart["Low"]), close=clean_series(df_chart["Close"]), name="ローソク足"
        ), row=1, col=1)

        for col, color, width, dash in [("SMA_20", "orange", 1, "solid"), ("EMA_200", "#3498db", 1.5, "solid"), ("Upper_Band", "gray", 1, "dash"), ("Lower_Band", "gray", 1, "dash")]:
            if col in df_chart.columns:
                fig.add_trace(go.Scatter(x=chart_x, y=clean_series(df_chart[col]), mode="lines", name=col, line=dict(color=color, width=width, dash=dash)), row=1, col=1)

        for idx_x, t_str in enumerate(chart_x):
            try:
                time_part = t_str.split(" ")[-1]
                hour_val = int(time_part.split(":")[0])

                indicator_start = 21 if is_summer_time else 22
                if indicator_start <= hour_val <= (indicator_start + 2):
                    fig.add_vrect(x0=idx_x-0.5, x1=idx_x+0.5, fillcolor="rgba(239, 68, 68, 0.15)", layer="below", line_width=0)
                elif 3 <= hour_val <= 6:
                    fig.add_vrect(x0=idx_x-0.5, x1=idx_x+0.5, fillcolor="rgba(234, 179, 8, 0.12)", layer="below", line_width=0)
            except Exception:
                pass

        if "RSI" in df_chart.columns:
            fig.add_trace(go.Scatter(x=chart_x, y=clean_series(df_chart["RSI"]), mode="lines", name="RSI", line=dict(color="#9b59b6", width=1.5)), row=2, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color="#e74c3c", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#2ecc71", row=2, col=1)
        fig.update_xaxes(type="category", nticks=8)
        fig.update_layout(xaxis_rangeslider_visible=False, height=480, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

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
                            "通貨ペア": p_label, "相場タイプ": s_mtype, "AI判定": s_status,
                            "確信度 (%)": round(s_conf, 1),
                            "ADX": round(float(clean_series(sub_df["ADX"]).iloc[-1]) if "ADX" in sub_df.columns else 25.0, 1)
                        })
                    progress_bar_scan.progress(min(1.0, (idx_p + 1) / len(PAIRS)))

            progress_bar_scan.empty()
            if scan_results:
                st.dataframe(pd.DataFrame(scan_results).sort_values(by="確信度 (%)", ascending=False), use_container_width=True)

    with tab_backtest:
        st.subheader("📊 時系列ウォークフォワード検証")
        if cumulative_wins:
            st.line_chart(pd.DataFrame(cumulative_wins, columns=["ステップ", "適合率 (%)"]).set_index("ステップ"))
            st.metric("トレード実行時 適合率", f"{win_rate:.1f}%" if trade_count > 0 else "N/A", f"{correct_count}勝 / {trade_count}回")

    with tab_metrics:
        st.subheader("📋 運用設定サマリー")
        sum_c1, sum_c2 = st.columns(2)
        sum_c1.metric("口座資金", f"{account_balance:,} 円")
        sum_c1.metric("1回あたり数量", f"{quantity_wan} 万通貨 ({custom_quantity:,} 通貨)")
        sum_c1.metric("AI推奨 注文本数", f"{target_grids} 本")
        sum_c2.metric("分析通貨ペア", selected_label)
        sum_c2.metric("分析時間足", tf_label)

if auto_refresh:
    st.caption(f"🔄 自動更新有効 ({refresh_interval}秒間隔)")
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_interval * 1000, limit=100, key="data_refresh")
