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
def load_and_process_data(symbol, period, interval, tf_name=""):
    df = pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    except Exception:
        pass

    if not df.empty and "4時間足" in tf_name and interval == "1h":
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
        if len(df) > 3: df.iloc[-3:, df.columns.get_loc("Target")] = np.nan

        feature_cols = [c for c in df.columns if c != "Target"]
        df = df.dropna(subset=feature_cols)

        if df.empty: return None
        return df
    except Exception:
        return None
# ==========================================
# 3. AIモデル学習 & シグナル判定
# ==========================================
@st.cache_resource(show_spinner=False)
def train_ai_model(df):
    try:
        df_train = df.dropna(subset=["Target"])
        if len(df_train) < 50:
            return None
        
        X = df_train[FEATURE_COLUMNS].fillna(0)
        y = df_train["Target"]
        
        # クラスの偏りを補正して学習
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
        model.fit(X, y)
        return model
    except Exception:
        return None

def generate_signal(model, df):
    if model is None or df.empty:
        return "HOLD", 0.0, {}

    try:
        # 最新のローソク足の特徴量を取得
        latest_features = df[FEATURE_COLUMNS].iloc[-1:].fillna(0)
        prob = model.predict_proba(latest_features)[0]
        classes = model.classes_
        
        prob_dict = {str(c): p for c, p in zip(classes, prob)}
        prob_buy = prob_dict.get('1.0', 0.0)
        prob_sell = prob_dict.get('-1.0', 0.0)
        
        # 閾値設定（0.55以上で強いシグナルとみなす）
        threshold = 0.55
        
        if prob_buy > threshold and prob_buy > prob_sell:
            return "BUY", prob_buy, prob_dict
        elif prob_sell > threshold and prob_sell > prob_buy:
            return "SELL", prob_sell, prob_dict
        else:
            return "HOLD", max(prob_buy, prob_sell), prob_dict
            
    except Exception:
        return "HOLD", 0.0, {}

# ==========================================
# 4. リピート注文 (トラップ) の最適化ロジック
# ==========================================
def calculate_repeat_orders(current_price, atr, symbol, balance, quantity_wan, is_buy):
    # シンボルごとの設定を取得
    config = PAIR_ATR_CONFIG.get(symbol, {"atr_mult": 0.20, "min_pips": 15})
    
    # 1万通貨あたりの必要証拠金（レバレッジ25倍、大まかな概算）
    margin_per_10k = 60000  # デフォルト
    if "JPY" in symbol:
        margin_per_10k = (current_price * 10000) / 25
    elif symbol == "EURUSD=X":
        # 簡易的に1EUR=160円として計算
        margin_per_10k = (160 * 10000) / 25
        
    pip_value = 0.01 if "JPY" in symbol else 0.0001
    
    # ATRに基づくトラップ幅の計算
    trap_width = max(atr * config["atr_mult"], config["min_pips"] * pip_value)
    
    # 必要証拠金の算出
    req_margin_per_order = margin_per_10k * quantity_wan
    
    if req_margin_per_order <= 0:
        return None, "証拠金計算エラー: ロット数が不正です"
        
    # 資金とロット数から設置可能な最大本数を算出
    max_orders_by_margin = int(balance / req_margin_per_order)
    
    # 安全ガード: 最低3本置けない場合は資金不足として警告
    if max_orders_by_margin < 3:
        return None, f"【警告】証拠金が不足しています。現在のロット({quantity_wan}万通貨)では最低3本分の証拠金(約{int(req_margin_per_order * 3):,}円)が必要です。"
        
    # トラップ本数の上限（最大15本に制限）
    num_orders = min(15, max_orders_by_margin)
    
    orders = []
    for i in range(num_orders):
        if is_buy:
            entry_p = current_price - (trap_width * i)
            tp_p = entry_p + trap_width
            sl_p = current_price - (trap_width * (num_orders + 2))
        else:
            entry_p = current_price + (trap_width * i)
            tp_p = entry_p - trap_width
            sl_p = current_price + (trap_width * (num_orders + 2))
            
        orders.append({"entry": entry_p, "tp": tp_p, "sl": sl_p})
        
    total_margin = req_margin_per_order * num_orders
    risk_pips = (trap_width * num_orders + trap_width * 2) / pip_value
    
    return {
        "num_orders": num_orders,
        "trap_width_pips": trap_width / pip_value,
        "total_margin": total_margin,
        "risk_pips": risk_pips,
        "orders": orders
    }, None

# ==========================================
# 5. チャート描画ロジック
# ==========================================
def create_dashboard_chart(df, symbol_name, tf_name, signal, repeat_info):
    # タイムゾーンの安全な処理
    df_plot = df.copy()
    if df_plot.index.tz is not None:
        try:
            df_plot.index = df_plot.index.tz_convert('Asia/Tokyo')
        except Exception:
            pass
            
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
        subplot_titles=(f"{symbol_name} ({tf_name}) - AIシグナル: {signal}", "RSI & MACD ヒストグラム")
    )

    # 1. ローソク足
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'],
        name="Price", increasing_line_color='#22c55e', decreasing_line_color='#ef4444'
    ), row=1, col=1)

    # 2. 移動平均線 (20, 200)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['SMA_20'], mode='lines', 
        name='SMA(20)', line=dict(color='#f59e0b', width=1.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['EMA_200'], mode='lines', 
        name='EMA(200)', line=dict(color='#3b82f6', width=2)
    ), row=1, col=1)
    
    # 3. ボリンジャーバンド (±2σ)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['Upper_Band'], mode='lines',
        name='BB+2σ', line=dict(color='rgba(148, 163, 184, 0.4)', width=1, dash='dot')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['Lower_Band'], mode='lines',
        name='BB-2σ', line=dict(color='rgba(148, 163, 184, 0.4)', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(148, 163, 184, 0.05)'
    ), row=1, col=1)

    # 4. リピート注文の可視化 (シグナル点灯時のみ)
    if repeat_info and signal in ["BUY", "SELL"]:
        is_buy = (signal == "BUY")
        color = "rgba(34, 197, 94, 0.8)" if is_buy else "rgba(239, 68, 68, 0.8)"
        
        # 代表して最初の3本と損切りラインだけ描画してチャートの煩雑化を防ぐ
        display_orders = repeat_info['orders'][:3]
        for idx, ord_data in enumerate(display_orders):
            fig.add_hline(
                y=ord_data['entry'], line_dash="dash", line_color=color, line_width=1.5,
                annotation_text=f" Entry {idx+1}", annotation_position="right",
                row=1, col=1
            )
            
        # 損切りライン
        sl_price = repeat_info['orders'][0]['sl']
        fig.add_hline(
            y=sl_price, line_dash="solid", line_color="rgba(249, 115, 22, 0.8)", line_width=2,
            annotation_text=" Stop Loss", annotation_position="bottom right",
            row=1, col=1
        )

    # 5. RSI と MACD (サブプロット)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['RSI'], mode='lines', 
        name='RSI', line=dict(color='#8b5cf6', width=1.5)
    ), row=2, col=1)
    
    # RSIの基準線 (30, 70)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(148, 163, 184, 0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(148, 163, 184, 0.5)", row=2, col=1)
    
    colors_macd = ['#22c55e' if val >= 0 else '#ef4444' for val in df_plot['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot['MACD_Hist'],
        name='MACD Hist', marker_color=colors_macd, opacity=0.5
    ), row=2, col=1)

    # レイアウトの美化
    fig.update_layout(
        height=650, margin=dict(l=20, r=60, t=40, b=20),
        paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
        font=dict(color='#f8fafc', size=11),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=False
    )
    
    # グリッド線の設定
    fig.update_xaxes(gridcolor='#1e293b', showline=True, linewidth=1, linecolor='#334155', row=1, col=1)
    fig.update_xaxes(gridcolor='#1e293b', showline=True, linewidth=1, linecolor='#334155', row=2, col=1)
    fig.update_yaxes(gridcolor='#1e293b', showline=True, linewidth=1, linecolor='#334155', row=1, col=1)
    fig.update_yaxes(gridcolor='#1e293b', showline=True, linewidth=1, linecolor='#334155', row=2, col=1)

    return fig
# ==========================================
# 6. Streamlit サイドバー設定 (パラメータ入力)
# ==========================================
st.sidebar.markdown("### ⚙️ 実行環境・リスク設定")

# 選択フォーム
selected_pair_label = st.sidebar.selectbox("監視通貨ペア", list(PAIRS.keys()), key="selected_pair_label", on_change=save_user_settings)
selected_tf_label = st.sidebar.selectbox("時間足", list(TIMEFRAMES.keys()), key="selected_tf_label", on_change=save_user_settings)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 資金・ロット設定")
st.sidebar.number_input("口座資金 (円)", min_value=10000, max_value=100000000, step=10000, key="account_balance", on_change=save_user_settings)
st.sidebar.number_input("1注文あたりのロット (万通貨)", min_value=0.01, max_value=10.0, step=0.01, format="%.2f", key="quantity_wan", on_change=save_user_settings)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔔 Discord自動通知設定")
st.sidebar.text_input("Webhook URL", placeholder="https://discord.com/api/webhooks/...", key="discord_url", on_change=save_user_settings)
st.sidebar.checkbox("シグナル変化時にDiscordへ通知する", key="enable_notify", on_change=save_user_settings)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ 自動更新設定")
st.sidebar.checkbox("自動リフレッシュを有効にする", key="auto_refresh", on_change=save_user_settings)
st.sidebar.slider("更新間隔 (秒)", min_value=30, max_value=600, step=30, key="refresh_interval", on_change=save_user_settings)

# 自動リフレッシュの処理
if st.session_state.get("auto_refresh", False):
    interval_ms = int(st.session_state.get("refresh_interval", 180)) * 1000
    st_autorefresh(interval=interval_ms, key="datarefresh")

# ==========================================
# 7. メイン画面の構築 & 実行ロジック
# ==========================================
symbol = PAIRS[selected_pair_label]
tf_info = TIMEFRAMES[selected_tf_label]

# データロード & AI学習
with st.spinner(f"{selected_pair_label} ({selected_tf_label}) のデータを取得・AI解析中..."):
    df = load_and_process_data(symbol, tf_info["period"], tf_info["interval"], selected_tf_label)
    
if df is None or df.empty:
    st.error("データの取得に失敗したか、データ量が不足しています。しばらく経ってから再度お試しください。")
    st.stop()

model = train_ai_model(df)
signal, confidence, prob_dict = generate_signal(model, df)

# 最新情報抽出
latest = df.iloc[-1]
current_price = float(latest["Close"])
atr_val = float(latest["ATR"])
rsi_val = float(latest["RSI"])

# 現在時刻・市場イベント確認
now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
is_summer = 3 <= now_jst.month <= 10  # 簡易的なサマータイム判定
events = get_upcoming_market_events(now_jst, is_summer)

# Discord自動通知判定
last_notified = st.session_state.get("last_notified_status", {})
current_key = f"{selected_pair_label}_{selected_tf_label}"
previous_signal = last_notified.get(current_key, "NONE")

if st.session_state.get("enable_notify", False) and st.session_state.get("discord_url", ""):
    if signal != previous_signal and previous_signal != "NONE" and signal in ["BUY", "SELL"]:
        color_code = 0x22c55e if signal == "BUY" else 0xef4444
        notif_msg = f"**通貨ペア:** {selected_pair_label}\n**時間足:** {selected_tf_label}\n**AIシグナル:** {signal} (確信度: {confidence*100:.1f}%)\n**現在値:** {current_price:.3f}"
        success, res_msg = send_discord_notification(st.session_state["discord_url"], f"🚨 AIシグナル点灯: {signal}", notif_msg, color_code)
        if success:
            last_notified[current_key] = signal
            st.session_state["last_notified_status"] = last_notified
            save_user_settings()
    elif previous_signal == "NONE":
        last_notified[current_key] = signal
        st.session_state["last_notified_status"] = last_notified
        save_user_settings()

# ==========================================
# 8. ダッシュボードのUI描画
# ==========================================
# トップヘッダー情報
col_h1, col_h2, col_h3, col_h4 = st.columns([1.2, 1.0, 1.0, 1.2])

with col_h1:
    st.markdown(f"### 📈 {selected_pair_label}")
    st.caption(f"選択時間足: {selected_tf_label}")

with col_h2:
    st.metric("現在価格", f"{current_price:.3f}", f"RSI: {rsi_val:.1f}")

with col_h3:
    if signal == "BUY":
        st.markdown(f'**AIシグナル**<br><span class="status-badge-buy">▲ BUY (買い)</span>', unsafe_allow_html=True)
    elif signal == "SELL":
        st.markdown(f'**AIシグナル**<br><span class="status-badge-sell">▼ SELL (売り)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'**AIシグナル**<br><span class="status-badge-hold">■ HOLD (様子見)</span>', unsafe_allow_html=True)

with col_h4:
    st.metric("AI予測確信度", f"{confidence*100:.1f}%", f"上昇確率: {prob_dict.get('1.0', 0)*100:.1f}%")

st.markdown("---")

# タブ構成による詳細表示
tab_chart, tab_strategy, tab_info = st.tabs(["📊 チャート & インジケーター", "🛠️ 推奨リピート注文プラン", "⏰ 市場環境・経済指標スケジュール"])

with tab_chart:
    # リピート情報の事前計算（チャート描画用）
    is_buy_signal = (signal == "BUY")
    repeat_info, _ = calculate_repeat_orders(
        current_price, atr_val, symbol, 
        st.session_state["account_balance"], 
        st.session_state["quantity_wan"], 
        is_buy_signal
    )
    
    # チャート描画
    fig = create_dashboard_chart(df, selected_pair_label, selected_tf_label, signal, repeat_info)
    st.plotly_chart(fig, use_container_width=True)

with tab_strategy:
    st.markdown("#### 🤖 AI連動 リピート注文 (トラップ) 戦略プラン")
    st.markdown("現在の相場ボラティリティ（ATR）とAIシグナルに基づき、最も効率的なリピート系注文パラメータを自動算出します。")
    
    if signal == "HOLD":
        st.info("💡 現在のシグナルは **HOLD（様子見）** です。明確なトレンドやブレイクアウトシグナルが出るまでリピート注文の新規構築はお控えいただくことを推奨します。")
    
    is_buy_plan = (signal in ["BUY", "HOLD"])  # HOLD時は便宜上買い基準で表示
    plan_data, err_msg = calculate_repeat_orders(
        current_price, atr_val, symbol, 
        st.session_state["account_balance"], 
        st.session_state["quantity_wan"], 
        is_buy_plan
    )
    
    if err_msg:
        st.error(err_msg)
    elif plan_data:
        box_class = "param-box param-box-buy" if is_buy_plan else "param-box param-box-sell"
        direction_text = "買い（リピート・ロング）" if is_buy_plan else "売り（リピート・ショート）"
        
        st.markdown(f"""
        <div class="{box_class}">
            <b class="label-title">■ 推奨トレード方向:</b> <code>{direction_text}</code><br>
            <b class="label-title">■ 推奨トラップ幅:</b> <code>{plan_data['trap_width_pips:.1f' if isinstance(plan_data['trap_width_pips'], float) else 's']} pips</code> (ATR連動)<br>
            <b class="label-title">■ 設置注文本数:</b> <code>{plan_data['num_orders']} 本</code><br>
            <b class="label-title">■ 必要総証拠金（目安）:</b> <code>約 {int(plan_data['total_margin']):,} 円</code> (口座資金: {int(st.session_state['account_balance']):,} 円)<br>
            <b class="label-title">■ 想定最大リスク幅:</b> <code>約 {plan_data['risk_pips']:.1f} pips</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### 📋 発注予定ライン一覧 (上位3本抜粋)")
        orders_df = pd.DataFrame(plan_data['orders'])
        orders_df.columns = ["エントリー価格", "利食い目標 (TP)", "損切り価格 (SL)"]
        st.dataframe(orders_df.head(5), use_container_width=True)

with tab_info:
    st.markdown("#### 🌍 本日の重要市場イベント・時間帯リスク")
    st.markdown(f"**現在の日本時間 (JST):** {now_jst.strftime('%Y-%m-%d %H:%M:%S')}")
    
    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.markdown("##### 📌 注目すべき時間帯と残り時間")
        for ev in events:
            time_label = f"あと約 {ev['left_min']} 分" if ev['left_min'] >= 0 else f"経過 ({abs(ev['left_min'])}分前)"
            if ev['left_min'] < 0 and ev['left_min'] > -60:
                time_label = f"🔴 通過直後 (警戒中)"
            st.markdown(f"- **{ev['event']}** ({ev['time_str']}予定): ` {time_label} `")
            
    with col_ev2:
        st.markdown("##### 💡 運用上のアドバイス")
        st.markdown("""
        - **ロンドン・NY市場オープン直後**は急激なスプレッド拡大やノイズが発生しやすいため、トラップ注文の幅に余裕を持たせてください。
        - **金曜日の深夜・早朝**にかけてはポジションを持ち越さないか、損切りラインを必ず設定するよう心がけてください。
        """)

# フッター
st.markdown("---")
st.caption("※本アプリケーションはAIによる相場分析およびリピート注文の目安を提供するものであり、実際の投資成果を保証するものではありません。投資の最終判断はご自身の責任で行ってください。")
