import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
# 0. 画面基本設定
# ==========================================
st.set_page_config(
    page_title="Pro FX Analyzer & Signal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 800 !important; }
    
    .badge-buy { background-color: #16a34a; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.1rem; display: inline-block; }
    .badge-sell { background-color: #dc2626; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.1rem; display: inline-block; }
    .badge-wait { background-color: #475569; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.1rem; display: inline-block; }
    
    .param-box { background-color: rgba(30, 41, 59, 0.8) !important; border-left: 5px solid #3b82f6; padding: 12px; border-radius: 6px; font-family: monospace; line-height: 1.8; color: #f8fafc !important; }
    .param-box code { font-size: 1.0rem !important; font-weight: 700 !important; color: #38bdf8 !important; background-color: rgba(51, 65, 85, 0.9) !important; }
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
    "selected_pair_label": "米ドル / 円 (USD/JPY)",
    "selected_tf_label": "15分足 (デイトレエントリー用)",
    "grid_density": "標準",
}

PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}

TIMEFRAMES = {
    "5分足 (スキャル用)": {"period": "7d", "interval": "5m"},
    "15分足 (デイトレエントリー用)": {"period": "1mo", "interval": "15m"},
    "1時間足 (デイトレメイン用)": {"period": "6mo", "interval": "1h"},
    "4時間足 (中期・リピート用)": {"period": "2y", "interval": "1h"},
}

def clean_series(s): return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s

# ==========================================
# 1. データ処理 & バックテスト付きAIモデル
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
        new_cols["EMA_200"] = c.ewm(span=200, adjust=False).mean()
        hl = h - l
        new_cols["ATR"] = hl.rolling(14).mean()
        
        new_cols["Return_1"] = c.diff(1) / (c.shift(1) + 1e-10)
        new_cols["Return_5"] = c.diff(5) / (c.shift(5) + 1e-10)
        new_cols["Dev_SMA20"] = (c - new_cols["SMA_20"]) / (new_cols["SMA_20"] + 1e-10)
        new_cols["Dev_EMA200"] = (c - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Dev_EMA20_200"] = (c.ewm(span=20, adjust=False).mean() - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
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
        new_cols["ATR_Ratio"] = new_cols["ATR"] / c

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # 正解ラベル（Target）: 5本先までに1.0*ATRの利益が出るか
        lookahead = 5
        f_high = pd.concat([h.shift(-i) for i in range(1, lookahead + 1)], axis=1).max(axis=1) - c
        f_low = c - pd.concat([l.shift(-i) for i in range(1, lookahead + 1)], axis=1).min(axis=1)
        tp_t, sl_t = new_cols["ATR"] * 1.0, new_cols["ATR"] * 0.5
        cond_buy = (f_high >= tp_t) & (f_low < sl_t)
        cond_sell = (f_low >= tp_t) & (f_high < sl_t)
        df["Target"] = np.select([cond_buy, cond_sell], [1, -1], default=0)
        
        return df.dropna(subset=[col for col in df.columns if col != "Target"])
    except Exception: return None

@st.cache_data(ttl=60, show_spinner=False)
def analyze_signal_with_backtest(df_current, df_htf):
    if df_current is None or len(df_current) < 200:
        return "WAIT (データ不足)", 0.0, 0.0, "不明"

    try:
        avail = [f for f in FEATURE_COLUMNS if f in df_current.columns]
        X, y = df_current[avail], df_current["Target"]
        
        # 直近の学習データ
        train_mask = ~y.isna()
        X_train, y_train = X[train_mask].iloc[:-100], y[train_mask].iloc[:-100]
        X_test, y_test = X[train_mask].iloc[-100:], y[train_mask].iloc[-100:]

        if len(np.unique(y_train)) < 2: return "WAIT", 0.0, 0.0, "判定不可"

        model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
        model.fit(X_train, y_train)

        # 簡易バックテスト勝率計算
        preds = model.predict(X_test)
        valid_eval = (preds != 0) & (y_test != 0)
        win_rate = (preds[valid_eval] == y_test[valid_eval]).mean() * 100 if valid_eval.sum() > 0 else 50.0

        # 最新足の予測
        probs = dict(zip(model.classes_, model.predict_proba(X.iloc[[-1]])[0]))
        prob_up, prob_down = probs.get(1.0, 0.0), probs.get(-1.0, 0.0)
        conf = max(prob_up, prob_down) * 100

        # 上位足（日足）トレンド判定（上位足EMA200）
        htf_close = clean_series(df_htf["Close"]).iloc[-1]
        htf_ema200 = clean_series(df_htf["EMA_200"]).iloc[-1]
        htf_uptrend = htf_close > htf_ema200

        # 厳格なフィルター（確信度70%以上 & 上位足と同方向のみ採用）
        if prob_up >= 0.70 and htf_uptrend:
            status = "BUY (買い)"
        elif prob_down >= 0.70 and not htf_uptrend:
            status = "SELL (売り)"
        else:
            status = "WAIT (様子見)"

        adx_val = float(clean_series(df_current["ADX"]).iloc[-1])
        m_type = "トレンド相場" if adx_val > 22 else "レンジ相場"

        return status, conf, win_rate, m_type
    except Exception:
        return "WAIT (エラー)", 0.0, 0.0, "エラー"

# ==========================================
# 2. UI構築
# ==========================================
selected_label = st.selectbox("通貨ペア", list(PAIRS.keys()), key="selected_pair_label")
tf_label = st.selectbox("時間足", list(TIMEFRAMES.keys()), key="selected_tf_label")

ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]
is_jpy = "JPY" in ticker
pip_unit = 0.01 if is_jpy else 0.0001
price_fmt = "%.3f" if is_jpy else "%.5f"

# サイドバー
account_balance = st.sidebar.number_input("口座資金 (円)", min_value=10000, value=500000, step=50000)
quantity_wan = st.sidebar.number_input("1注文の数量 (万通貨)", min_value=0.01, value=0.10, step=0.01)

# データ取得
data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
data_htf = load_and_process_data(ticker, "2y", "1d", "日足")

if data is None or data_htf is None:
    st.error("データの取得に失敗しました。時間足を変更してください。")
    st.stop()

latest_price = float(clean_series(data["Close"]).iloc[-1])
latest_atr = float(clean_series(data["ATR"]).iloc[-1])

status, conf, win_rate, m_type = analyze_signal_with_backtest(data, data_htf)

# --- 1. サマリーダッシュボード ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("現在レート", price_fmt % latest_price)

badge_html = f'<div class="badge-buy">{status}</div>' if "BUY" in status else \
             f'<div class="badge-sell">{status}</div>' if "SELL" in status else \
             f'<div class="badge-wait">{status}</div>'
m2.markdown("**AI 推奨アクション**")
m2.markdown(badge_html, unsafe_allow_html=True)

m3.metric("直近バックテスト勝率", f"{win_rate:.1f}%", f"確信度: {conf:.1f}%")
m4.metric("相場環境", m_type, f"ATR: {latest_atr/pip_unit:.1f} pips")

st.markdown("---")

# --- 2. リピート想定レンジの設定 ---
swing_high = float(clean_series(data["High"]).iloc[-100:].max())
swing_low = float(clean_series(data["Low"]).iloc[-100:].min())

with st.expander("⚙️ リピート自動売買のレンジ調整", expanded=False):
    rc1, rc2 = st.columns(2)
    user_lower = rc1.number_input("レンジ下限", value=swing_low, step=0.1 if is_jpy else 0.001, format=price_fmt)
    user_upper = rc2.number_input("レンジ上限", value=swing_high, step=0.1 if is_jpy else 0.001, format=price_fmt)
    user_half = (user_upper + user_lower) / 2.0

# --- 3. メインチャート表示 ---
df_chart = data.tail(120).copy()
if df_chart.index.tz is None: df_chart.index = df_chart.index.tz_localize("UTC")
df_chart.index = df_chart.index.tz_convert("Asia/Tokyo")
cx = df_chart.index

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)

# ローソク足
fig.add_trace(go.Candlestick(
    x=cx, open=clean_series(df_chart["Open"]), high=clean_series(df_chart["High"]),
    low=clean_series(df_chart["Low"]), close=clean_series(df_chart["Close"]), name="価格"
), row=1, col=1)

# EMA200
if "EMA_200" in df_chart.columns:
    fig.add_trace(go.Scatter(x=cx, y=clean_series(df_chart["EMA_200"]), line=dict(color="#38bdf8", width=1.5), name="EMA200"), row=1, col=1)

# ★ チャート上へのリピートレンジ可視化 (カラー帯＆破線)
fig.add_hrect(y0=user_lower, y1=user_upper, fillcolor="rgba(56, 189, 248, 0.05)", line_width=0, row=1, col=1)
fig.add_hline(y=user_upper, line_dash="dash", line_color="#ef4444", annotation_text="リピート上限", row=1, col=1)
fig.add_hline(y=user_half, line_dash="dot", line_color="#a855f7", annotation_text="ハーフライン", row=1, col=1)
fig.add_hline(y=user_lower, line_dash="dash", line_color="#22c55e", annotation_text="リピート下限", row=1, col=1)

# RSI
if "RSI" in df_chart.columns:
    fig.add_trace(go.Scatter(x=cx, y=clean_series(df_chart["RSI"]), line=dict(color="#a855f7", width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=550,
    margin=dict(l=10, r=60, t=10, b=10),
    template="plotly_dark",
    hovermode="x unified",
    dragmode="pan",
    showlegend=False
)
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showspikes=True)
fig.update_yaxes(side="right")

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})

# --- 4. 注文パラメータ生成 (アコーディオン化) ---
with st.expander("📋 松井証券 リピート注文設定値（コピー用）"):
    trap_width_pips = max(15, int(round(latest_atr / pip_unit)))
    range_pips = abs(user_upper - user_lower) / pip_unit
    grid_count = max(2, int(range_pips // trap_width_pips) + 1)
    
    st.markdown(f"""
    <div class="param-box">
    <b>【ハーフ＆ハーフ推奨設定】</b><br>
    ・買い設定（下半）: <code>{price_fmt % user_lower}</code> ～ <code>{price_fmt % user_half}</code><br>
    ・売り設定（上半）: <code>{price_fmt % user_half}</code> ～ <code>{price_fmt % user_upper}</code><br>
    ・注文/益出し幅: <code>{trap_width_pips} pips</code> | 注文本数: 約 <code>{grid_count} 本</code>
    </div>
    """, unsafe_allow_html=True)
