import json
import os
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
# 0. 画面基本設定 & タッチデバイス(iPad/Android)用CSS
# ==========================================
st.set_page_config(
    page_title="AI FX デイトレ & リピート アナライザー",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
    label[data-testid="stWidgetLabel"] p { font-size: 1.05rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 0.95rem !important; font-weight: 700 !important; color: #94a3b8; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 800 !important; }
    
    .status-badge-buy { background-color: #15803d; color: #ffffff; border: 1px solid #16a34a; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 1.1rem; display: inline-block; }
    .status-badge-sell { background-color: #b91c1c; color: #ffffff; border: 1px solid #dc2626; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 1.1rem; display: inline-block; }
    .status-badge-hold { background-color: #854d0e; color: #ffffff; border: 1px solid #ca8a04; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 1.1rem; display: inline-block; }
    
    .param-box { background-color: rgba(30, 41, 59, 0.8) !important; border-left: 6px solid #3b82f6; padding: 16px; border-radius: 8px; font-family: monospace, sans-serif; font-size: 1.0rem; line-height: 1.8; color: #f8fafc !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 10px; }
    .param-box b.label-title { color: #94a3b8 !important; font-family: sans-serif; }
    .param-box code { font-size: 1.05rem !important; font-weight: 700 !important; padding: 2px 6px !important; background-color: rgba(51, 65, 85, 0.9) !important; color: #38bdf8 !important; border: 1px solid #475569; border-radius: 4px; }
    .param-box-buy { border-left-color: #22c55e !important; }
    .param-box-sell { border-left-color: #ef4444 !important; }
    .param-box-half { border-left-color: #a855f7 !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 4px; overflow-x: auto; scrollbar-width: none; }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    .stTabs [data-baseweb="tab"] { padding: 10px 12px; font-size: 0.95rem !important; font-weight: 700; border-radius: 6px 6px 0 0; white-space: nowrap; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; background-color: rgba(56, 189, 248, 0.05); }

    /* モバイル(スマホ)向けの最適化 */
    @media (max-width: 768px) {
        [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
        .param-box { font-size: 0.9rem; padding: 12px; }
        .param-box code { font-size: 0.95rem !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.85rem !important; padding: 8px 10px; }
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
    "account_balance": 500000,
    "quantity_wan": 0.10,
    "discord_url": "",
    "enable_notify": False,
    "auto_refresh": False,
    "refresh_interval": 180,
    "selected_pair_label": "米ドル / 円 (USD/JPY)",
    "selected_tf_label": "15分足 (デイトレエントリー用)",
    "grid_density": "標準 (ATR1.0倍)",
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
    "4時間足 (中期・リピート用)": {"period": "2y", "interval": "1h"},
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
# 1. 設定管理 & 補助関数
# ==========================================
def load_user_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                merged = DEFAULT_SETTINGS.copy()
                merged.update(saved)
                return merged
        except Exception: pass
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
        "grid_density": st.session_state.get("grid_density", DEFAULT_SETTINGS["grid_density"]),
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception: pass

if "initialized" not in st.session_state:
    saved_settings = load_user_settings()
    for key, val in saved_settings.items(): st.session_state[key] = val
    if st.session_state.get("selected_pair_label") not in PAIRS: st.session_state["selected_pair_label"] = DEFAULT_SETTINGS["selected_pair_label"]
    if st.session_state.get("selected_tf_label") not in TIMEFRAMES: st.session_state["selected_tf_label"] = DEFAULT_SETTINGS["selected_tf_label"]
    st.session_state["initialized"] = True

def clean_series(s): return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s

def get_upcoming_market_events(now_jst, is_summer):
    current_min = now_jst.hour * 60 + now_jst.minute
    events = [
        {"name": "欧州市場OP", "time": "16:00" if is_summer else "17:00", "min": 16*60 if is_summer else 17*60},
        {"name": "米国指標", "time": "21:30" if is_summer else "22:30", "min": 21*60+30 if is_summer else 22*60+30},
        {"name": "NY市場OP", "time": "22:30" if is_summer else "23:30", "min": 22*60+30 if is_summer else 23*60+30},
    ]
    upcoming = []
    for ev in events:
        diff = ev["min"] - current_min
        if diff < -120: diff += 24 * 60
        upcoming.append({"event": ev["name"], "time": ev["time"], "left_min": diff})
    return sorted(upcoming, key=lambda x: x["left_min"])

# ==========================================
# 2. データ処理 & AIモデル (キャッシュ活用)
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        
        if "4時間足" in tf_name and interval == "1h":
            tz_before = df.index.tz
            df = df.resample("4h", closed="left", label="left").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
            if tz_before is not None and df.index.tz is None: df.index = df.index.tz_localize(tz_before)
            
        if len(df) < 50: return None
        
        c = clean_series(df["Close"])
        h = clean_series(df["High"])
        l = clean_series(df["Low"])
        o = clean_series(df["Open"])

        new_cols = {}
        new_cols["SMA_20"] = c.rolling(20).mean()
        new_cols["SMA_50"] = c.rolling(50).mean()
        new_cols["EMA_20"] = c.ewm(span=20, adjust=False).mean()
        new_cols["EMA_200"] = c.ewm(span=200, adjust=False).mean()

        hl = h - l
        new_cols["ATR"] = hl.rolling(14).mean()
        
        new_cols["Return_1"] = c.diff(1) / (c.shift(1) + 1e-10)
        new_cols["Return_5"] = c.diff(5) / (c.shift(5) + 1e-10)
        new_cols["Dev_SMA20"] = (c - new_cols["SMA_20"]) / (new_cols["SMA_20"] + 1e-10)
        new_cols["Dev_EMA200"] = (c - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Dev_EMA20_200"] = (new_cols["EMA_20"] - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Vol_Ratio"] = new_cols["ATR"] / (c + 1e-10)

        delta = c.diff()
        rs = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean() / ((-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean() + 1e-10)
        new_cols["RSI"] = 100 - (100 / (1 + rs))
        new_cols["RSI_Diff"] = new_cols["RSI"].diff(1)

        new_cols["Stoch_K"] = 100 * (c - l.rolling(14).min()) / ((h.rolling(14).max() - l.rolling(14).min()) + 1e-10)

        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        new_cols["MACD_Hist_Ratio"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c + 1e-10)

        std20 = c.rolling(20).std()
        new_cols["Upper_Band"] = new_cols["SMA_20"] + (std20 * 2)
        new_cols["Lower_Band"] = new_cols["SMA_20"] - (std20 * 2)
        new_cols["BB_PctB"] = (c - new_cols["Lower_Band"]) / ((new_cols["Upper_Band"] - new_cols["Lower_Band"]) + 1e-10)
        
        tr = pd.concat([hl, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        up, down = h - h.shift(1), l.shift(1) - l
        plus_di = 100 * pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / (tr.ewm(alpha=1/14, adjust=False).mean() + 1e-10)
        minus_di = 100 * pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index).ewm(alpha=1/14, adjust=False).mean() / (tr.ewm(alpha=1/14, adjust=False).mean() + 1e-10)
        new_cols["ADX"] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)).ewm(alpha=1/14, adjust=False).mean().fillna(25.0)

        total_range = hl + 1e-10
        open_close_max = pd.concat([o, c], axis=1).max(axis=1)
        open_close_min = pd.concat([o, c], axis=1).min(axis=1)
        new_cols["Upper_Wick_Ratio"] = (h - open_close_max) / total_range
        new_cols["Lower_Wick_Ratio"] = (open_close_min - l) / total_range

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # Target生成
        lookahead = 5
        f_high = pd.concat([h.shift(-i) for i in range(1, lookahead + 1)], axis=1).max(axis=1) - c
        f_low = c - pd.concat([l.shift(-i) for i in range(1, lookahead + 1)], axis=1).min(axis=1)
        tp_t, sl_t = new_cols["ATR"] * 1.0, new_cols["ATR"] * 0.5
        cond_buy = (f_high >= tp_t) & (f_low < sl_t)
        cond_sell = (f_low >= tp_t) & (f_high < sl_t)
        df["Target"] = np.select([cond_buy, cond_sell], [1, -1], default=0)
        if len(df) > lookahead: df.iloc[-lookahead:, df.columns.get_loc("Target")] = np.nan
        df = df.dropna(subset=[c for c in df.columns if c != "Target"])
        return df if not df.empty else None
    except Exception: return None

@st.cache_data(ttl=60, show_spinner=False)
def analyze_signal(df_current, df_higher):
    if df_current is None or len(df_current) < 50: return "HOLD", 50.0, "不明"
    try:
        avail = [f for f in FEATURE_COLUMNS if f in df_current.columns]
        X, y = df_current[avail], df_current["Target"]
        train_mask = ~y.isna()
        X_train, y_train = X[train_mask].iloc[-1000:], y[train_mask].iloc[-1000:]
        if len(np.unique(y_train)) < 2: return "HOLD", 50.0, "判定不可"

        model = RandomForestClassifier(n_estimators=60, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        probs = dict(zip(model.classes_, model.predict_proba(X.iloc[[-1]])[0]))
        prob_up, prob_down = probs.get(1.0, 0.0), probs.get(-1.0, 0.0)
        conf = max(prob_up, prob_down) * 100

        c_val = float(clean_series(df_current["Close"]).iloc[-1])
        ema200 = float(clean_series(df_current["EMA_200"]).iloc[-1])
        adx = float(clean_series(df_current["ADX"]).iloc[-1])
        
        m_type = "トレンド" if adx > 25 else "レンジ"
        if prob_up >= 0.65 and c_val > ema200: status = "BUY"
        elif prob_down >= 0.65 and c_val < ema200: status = "SELL"
        else: status = f"HOLD ({conf:.1f}%)"
        
        return status, conf, m_type
    except Exception: return "HOLD", 50.0, "エラー"

@st.cache_data(ttl=120, show_spinner=False)
def get_mtf_trends(symbol):
    trends = {}
    for name, p in TIMEFRAMES.items():
        d = load_and_process_data(symbol, p["period"], p["interval"], name)
        if d is not None and len(d) > 20:
            c, ema = float(clean_series(d['Close']).iloc[-1]), float(clean_series(d['EMA_200']).iloc[-1])
            trends[name.split(" ")[0]] = "上昇 📈" if c > ema else "下降 📉" if c < ema else "レンジ"
        else: trends[name.split(" ")[0]] = "-"
    return trends

# ==========================================
# 3. UI構築 & メイン処理
# ==========================================
# 📱 トップナビゲーション (スマホ考慮)
col_nav1, col_nav2 = st.columns([1, 1])
with col_nav1:
    selected_label = st.selectbox("通貨ペア (共通)", list(PAIRS.keys()), key="selected_pair_label", on_change=save_user_settings)
with col_nav2:
    tf_label = st.selectbox("時間足 (デイトレ用)", list(TIMEFRAMES.keys()), key="selected_tf_label", on_change=save_user_settings)

ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]
is_jpy = "JPY" in ticker
pip_unit = 0.01 if is_jpy else 0.0001
price_fmt = "%.3f" if is_jpy else "%.5f"

# ⚙️ サイドバー設定
st.sidebar.header("⚙️ 資金・リピート設定")
account_balance = st.sidebar.number_input("口座資金 (円)", min_value=10000, max_value=100000000, step=50000, key="account_balance", on_change=save_user_settings)
quantity_wan = st.sidebar.number_input("1注文の数量 (万通貨)", min_value=0.0001, max_value=10.0, step=0.01, format="%.4f", key="quantity_wan", on_change=save_user_settings)
grid_density = st.sidebar.select_slider("トラップ密度", options=["広め", "標準", "狭め"], key="grid_density", on_change=save_user_settings)
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("自動更新 (Web用)", key="auto_refresh", on_change=save_user_settings)
refresh_interval = st.sidebar.selectbox("更新間隔", options=[60, 180, 300], format_func=lambda x: f"{x//60}分", key="refresh_interval", on_change=save_user_settings)
if st.sidebar.button("🔄 手動更新", use_container_width=True): st.cache_data.clear(); st.rerun()

# 📊 データ取得
with st.spinner("市場データを解析中..."):
    data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
    data_htf = load_and_process_data(ticker, "1y", "1d", "日足")
    data_4h = load_and_process_data(ticker, "2y", "1h", "4時間足 (中期・リピート用)")

if data is None or data_4h is None:
    st.error("データの取得に失敗しました。時間足を変更するか少しお待ちください。")
    st.stop()

# 基本指標計算
latest_price = float(clean_series(data["Close"]).iloc[-1])
latest_atr = float(clean_series(data["ATR"]).iloc[-1])
market_status, confidence, market_type = analyze_signal(data, data_htf)

# アラート・時間判定
now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
is_summer = now_jst.astimezone(ZoneInfo("America/New_York")).dst().total_seconds() != 0
next_ev = get_upcoming_market_events(now_jst, is_summer)[0]

with st.container():
    if next_ev["left_min"] <= 60: st.error(f"⏰ **警戒**: {next_ev['event']} ({next_ev['time']}) まで残り {next_ev['left_min']} 分")
    elif next_ev["left_min"] <= 120: st.warning(f"⏳ **予告**: {next_ev['event']} ({next_ev['time']}) まで残り {next_ev['left_min']} 分")

    col_h1, col_h2, col_h3 = st.columns([1.2, 1.5, 1.3])
    col_h1.metric("現在レート", price_fmt % latest_price)
    
    badge = f'<div class="status-badge-buy">BUY 買い ({confidence:.1f}%)</div>' if market_status.startswith("BUY") else \
            f'<div class="status-badge-sell">SELL 売り ({confidence:.1f}%)</div>' if market_status.startswith("SELL") else \
            f'<div class="status-badge-hold">{market_status}</div>'
    col_h2.markdown("**AI 環境認識シグナル**")
    col_h2.markdown(badge, unsafe_allow_html=True)
    
    col_h3.metric("相場判定", market_type, f"ボラティリティ: {latest_atr/pip_unit:.1f}pips")

# ==========================================
# 4. メインタブ構成 (スマホ横幅考慮)
# ==========================================
tab_daytrade, tab_repeat, tab_chart, tab_risk = st.tabs([
    "🎯 デイトレAI", "🔁 リピート(松井)", "📈 チャート", "🛡️ リスク管理"
])

# 🎯 Tab 1: デイトレAI分析 (パターンB)
with tab_daytrade:
    st.subheader("🎯 デイトレード 裁量補助")
    tp_mult, sl_mult = 1.0, 0.5
    calc_tp = latest_price + (latest_atr * tp_mult) if market_status.startswith("BUY") else latest_price - (latest_atr * tp_mult)
    calc_sl = latest_price - (latest_atr * sl_mult) if market_status.startswith("BUY") else latest_price + (latest_atr * sl_mult)
    
    tp_pips = abs(calc_tp - latest_price) / pip_unit
    sl_pips = abs(latest_price - calc_sl) / pip_unit
    
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("目標利確 (TP)", price_fmt % calc_tp, f"+{tp_pips:.1f} pips")
    tc2.metric("撤退損切 (SL)", price_fmt % calc_sl, f"-{sl_pips:.1f} pips")
    tc3.metric("R:R比率", f"{tp_pips/sl_pips:.2f}" if sl_pips>0 else "N/A")

    st.markdown("##### 🌐 マルチタイムフレーム (MTF) トレンド")
    mtf_trends = get_mtf_trends(ticker)
    mtf_cols = st.columns(len(TIMEFRAMES))
    for idx, (k, v) in enumerate(mtf_trends.items()): mtf_cols[idx].metric(k, v)

# 🔁 Tab 2: リピートFX設定 (パターンA)
with tab_repeat:
    st.subheader("📋 松井証券 自動売買パラメータ算出")
    
    # 4時間足ベースのレンジ算出
    c_4h, h_4h, l_4h = clean_series(data_4h["Close"]), clean_series(data_4h["High"]), clean_series(data_4h["Low"])
    atr_4h = float(clean_series(data_4h["ATR"]).iloc[-1])
    swing_high = float(h_4h.iloc[-100:].max())
    swing_low = float(l_4h.iloc[-100:].min())
    
    st.markdown("###### 🎯 予想レンジの微調整")
    r_col1, r_col2 = st.columns(2)
    user_lower = r_col1.number_input("レンジ下限", value=swing_low, step=0.1 if is_jpy else 0.001, format=price_fmt)
    user_upper = r_col2.number_input("レンジ上限", value=swing_high, step=0.1 if is_jpy else 0.001, format=price_fmt)
    
    # 値幅と本数の計算
    density_mult = 1.5 if grid_density == "広め" else 0.7 if grid_density == "狭め" else 1.0
    atr_pips_4h = atr_4h / pip_unit
    trap_width_pips = max(PAIR_ATR_CONFIG[ticker]["min_pips"], int(round(atr_pips_4h * density_mult)))
    
    range_pips = abs(user_upper - user_lower) / pip_unit
    grid_count = max(2, int(range_pips // trap_width_pips) + 1)
    
    stop_buffer = round(atr_pips_4h * 1.5, 1)
    buy_stop = user_lower - (stop_buffer * pip_unit)
    sell_stop = user_upper + (stop_buffer * pip_unit)
    
    st.info(f"💡 AI推奨注文値幅: **{trap_width_pips} pips** | 算出注文本数: **{grid_count} 本**")
    
    mode = st.radio("注文タイプ", ["ハーフ＆ハーフ", "買いリピート", "売りリピート"], horizontal=True)
    
    if mode == "買いリピート":
        st.markdown(f"""
        <div class="param-box param-box-buy">
        <b class="label-title">売買区分</b> : 買<br>
        <b class="label-title">レンジ上限</b> : <code>{price_fmt % user_upper}</code><br>
        <b class="label-title">レンジ下限</b> : <code>{price_fmt % user_lower}</code><br>
        <b class="label-title">注文本数</b>   : <code>{grid_count} 本</code><br>
        <b class="label-title">注文数量</b>   : <code>{quantity_wan} 万通貨</code><br>
        <b class="label-title">注文/益出幅</b>: <code>{trap_width_pips} pips</code><br>
        <b class="label-title">運用停止(SL)</b>: <code>{price_fmt % buy_stop}</code> (-{stop_buffer}pips)
        </div>
        """, unsafe_allow_html=True)
    elif mode == "売りリピート":
        st.markdown(f"""
        <div class="param-box param-box-sell">
        <b class="label-title">売買区分</b> : 売<br>
        <b class="label-title">レンジ上限</b> : <code>{price_fmt % user_upper}</code><br>
        <b class="label-title">レンジ下限</b> : <code>{price_fmt % user_lower}</code><br>
        <b class="label-title">注文本数</b>   : <code>{grid_count} 本</code><br>
        <b class="label-title">注文数量</b>   : <code>{quantity_wan} 万通貨</code><br>
        <b class="label-title">注文/益出幅</b>: <code>{trap_width_pips} pips</code><br>
        <b class="label-title">運用停止(SL)</b>: <code>{price_fmt % sell_stop}</code> (+{stop_buffer}pips)
        </div>
        """, unsafe_allow_html=True)
    else:
        half_p = (user_upper + user_lower) / 2.0
        h_buy_cnt = max(1, int((half_p - user_lower) / (trap_width_pips * pip_unit)))
        h_sell_cnt = max(1, int((user_upper - half_p) / (trap_width_pips * pip_unit)))
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown(f"""
            <div class="param-box param-box-buy">
            <b class="label-title">【買い (下半)】</b><br>
            <b class="label-title">上限</b>: <code>{price_fmt % half_p}</code><br>
            <b class="label-title">下限</b>: <code>{price_fmt % user_lower}</code><br>
            <b class="label-title">本数</b>: <code>{h_buy_cnt} 本</code><br>
            <b class="label-title">SL</b>  : <code>{price_fmt % buy_stop}</code>
            </div>
            """, unsafe_allow_html=True)
        with hc2:
            st.markdown(f"""
            <div class="param-box param-box-sell">
            <b class="label-title">【売り (上半)】</b><br>
            <b class="label-title">上限</b>: <code>{price_fmt % user_upper}</code><br>
            <b class="label-title">下限</b>: <code>{price_fmt % half_p}</code><br>
            <b class="label-title">本数</b>: <code>{h_sell_cnt} 本</code><br>
            <b class="label-title">SL</b>  : <code>{price_fmt % sell_stop}</code>
            </div>
            """, unsafe_allow_html=True)

# 📈 Tab 3: チャート分析
with tab_chart:
    st.subheader(f"📈 {tf_label.split(' ')[0]} チャート")
    df_chart = data.tail(80).copy()
    if df_chart.index.tz is None: df_chart.index = df_chart.index.tz_localize("UTC")
    df_chart.index = df_chart.index.tz_convert("Asia/Tokyo")
    cx = df_chart.index.strftime("%m/%d %H:%M" if "足" in tf_label and "日" not in tf_label else "%Y-%m-%d")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
    
    # ローソク足
    fig.add_trace(go.Candlestick(
        x=cx, 
        open=clean_series(df_chart["Open"]), 
        high=clean_series(df_chart["High"]), 
        low=clean_series(df_chart["Low"]), 
        close=clean_series(df_chart["Close"]), 
        name="ローソク足"
    ), row=1, col=1)
    
    # 移動平均線
    if "SMA_20" in df_chart.columns: fig.add_trace(go.Scatter(x=cx, y=clean_series(df_chart["SMA_20"]), line=dict(color="orange", width=1.5), name="SMA20"), row=1, col=1)
    if "EMA_200" in df_chart.columns: fig.add_trace(go.Scatter(x=cx, y=clean_series(df_chart["EMA_200"]), line=dict(color="#3498db", width=2), name="EMA200"), row=1, col=1)
    
    # 選択レンジの描画
    fig.add_hline(y=user_upper, line_dash="dash", line_color="#ef4444", annotation_text="上限", row=1, col=1)
    fig.add_hline(y=user_lower, line_dash="dash", line_color="#22c55e", annotation_text="下限", row=1, col=1)
    
    # ==========================================
    # 💡 追加: 買い・売りサインのプロット処理
    # ==========================================
    
    # 【1】最新のAI判定シグナルをチャート右端（最新の足）に矢印付きで表示
    latest_x = cx[-1]
    latest_high = clean_series(df_chart["High"]).iloc[-1]
    latest_low = clean_series(df_chart["Low"]).iloc[-1]
    
    if market_status.startswith("BUY"):
        # ローソク足の下から上に向けて矢印を描画
        fig.add_annotation(
            x=latest_x, y=latest_low,
            text="AI: BUY", showarrow=True, arrowhead=1, arrowsize=2, arrowwidth=2,
            arrowcolor="#22c55e", ax=0, ay=40,
            font=dict(size=14, color="#22c55e", family="sans-serif", weight="bold"),
            row=1, col=1
        )
    elif market_status.startswith("SELL"):
        # ローソク足の上から下に向けて矢印を描画
        fig.add_annotation(
            x=latest_x, y=latest_high,
            text="AI: SELL", showarrow=True, arrowhead=1, arrowsize=2, arrowwidth=2,
            arrowcolor="#ef4444", ax=0, ay=-40,
            font=dict(size=14, color="#ef4444", family="sans-serif", weight="bold"),
            row=1, col=1
        )
        
    # 【2】過去のローソク足へのサイン描画（散布図マーカーを利用）
    buy_signals = (clean_series(df_chart["Close"]) > clean_series(df_chart["SMA_20"])) & (clean_series(df_chart["Close"]).shift(1) <= clean_series(df_chart["SMA_20"]).shift(1))
    sell_signals = (clean_series(df_chart["Close"]) < clean_series(df_chart["SMA_20"])) & (clean_series(df_chart["Close"]).shift(1) >= clean_series(df_chart["SMA_20"]).shift(1))
    
    if buy_signals.any():
        fig.add_trace(go.Scatter(
            x=cx[buy_signals], 
            y=clean_series(df_chart["Low"])[buy_signals] - (latest_atr * 0.2), # 安値の少し下に表示
            mode='markers', marker=dict(symbol='triangle-up', size=12, color='#22c55e'),
            name='過去の買サイン'
        ), row=1, col=1)
        
    if sell_signals.any():
        fig.add_trace(go.Scatter(
            x=cx[sell_signals], 
            y=clean_series(df_chart["High"])[sell_signals] + (latest_atr * 0.2), # 高値の少し上に表示
            mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ef4444'),
            name='過去の売サイン'
        ), row=1, col=1)
    # ==========================================

    # RSI
    if "RSI" in df_chart.columns: fig.add_trace(go.Scatter(x=cx, y=clean_series(df_chart["RSI"]), line=dict(color="#9b59b6", width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# 🛡️ Tab 4: リスク管理
with tab_risk:
    st.subheader("🛡️ 想定最大ドローダウン (リピート時)")
    order_units = int(quantity_wan * 10000)
    usd_rate = float(clean_series(yf.download("USDJPY=X", period="1d", progress=False)["Close"]).iloc[-1]) if not is_jpy else 1.0
    pip_val = (order_units / 10000.0) * 100 if is_jpy else (order_units * 0.0001 * usd_rate)
    
    max_loss = 0.0
    chk_count = grid_count if mode != "ハーフ＆ハーフ" else max(h_buy_cnt, h_sell_cnt)
    for i in range(chk_count):
        drop_pips = (i * trap_width_pips)
        max_loss += drop_pips * pip_val
        
    margin_req = (latest_price * order_units) / 25.0 if is_jpy else (latest_price * usd_rate * order_units) / 25.0
    total_margin = margin_req * chk_count
    loss_ratio = (max_loss / account_balance) * 100 if account_balance > 0 else 0
    
    rc1, rc2 = st.columns(2)
    rc1.metric("想定最大含み損", f"約 {int(max_loss):,} 円", f"資金比 {loss_ratio:.1f}%" , delta_color="inverse")
    rc2.metric("最大時 必要証拠金", f"約 {int(total_margin):,} 円")
    
    if loss_ratio > 50: st.error("🚨 含み損が資金の50%を超過します。数量(万通貨)を下げるか、レンジ幅を狭めてください。")
    else: st.success("🟢 リスクは許容範囲内です。")

if auto_refresh and HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_interval * 1000, limit=200, key="auto_refresh_timer")
