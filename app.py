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
# 0. 画面基本設定 & CSS
# ==========================================
st.set_page_config(
    page_title="Pro FX Analyzer & Signal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 800 !important; }
    
    .badge-buy { background-color: #16a34a; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.0rem; display: inline-block; }
    .badge-sell { background-color: #dc2626; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.0rem; display: inline-block; }
    .badge-wait { background-color: #475569; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.0rem; display: inline-block; }
    
    .param-box { background-color: rgba(30, 41, 59, 0.8) !important; border-left: 5px solid #3b82f6; padding: 14px; border-radius: 6px; font-family: monospace; line-height: 1.8; color: #f8fafc !important; }
    .param-box code { font-size: 0.95rem !important; font-weight: 700 !important; color: #38bdf8 !important; background-color: rgba(51, 65, 85, 0.9) !important; padding: 2px 6px; border-radius: 4px; }
    .update-time { font-size: 0.85rem; color: #94a3b8; text-align: right; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

FEATURE_COLUMNS = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200", "Dev_EMA20_200", "Vol_Ratio",
    "RSI", "RSI_Diff", "MACD_Hist_Ratio", "BB_PctB", "ADX",
    "ATR_Ratio", "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Stoch_K"
]

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
    "4時間足 (中期・リピート用)": {"period": "1y", "interval": "1h"},
}

LOOKAHEAD_BARS = 5  # Target予測の先読み足数

def clean_series(s):
    """DataFrameやMultiIndexから確実に単一のpd.Seriesを取り出す関数"""
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s

def safe_to_tokyo_tz(df):
    """タイムゾーンの安全変換処理"""
    df_out = df.copy()
    if df_out.index.tz is None:
        df_out.index = df_out.index.tz_localize("UTC")
    df_out.index = df_out.index.tz_convert("Asia/Tokyo")
    return df_out

# クロス通貨リスク計算用のドル円レート取得・キャッシュ関数
@st.cache_data(ttl=60, show_spinner=False)
def get_usdjpy_rate():
    try:
        df = yf.download("USDJPY=X", period="1d", progress=False)
        if not df.empty:
            c = clean_series(df["Close"])
            val = float(c.iloc[-1])
            if not np.isnan(val) and val > 0:
                return val
    except Exception:
        pass
    return 155.0  # 通信失敗時の標準フォールバック値

# ==========================================
# 1. データ処理 & バックテスト付きAIモデル
# ==========================================
@st.cache_data(ttl=30, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        
        if "4時間足" in tf_name and interval == "1h":
            tz_before = df.index.tz
            df = df.resample("4h", closed="left", label="left").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last"
            }).dropna()
            if tz_before is not None and df.index.tz is None: 
                df.index = df.index.tz_localize(tz_before)
            
        if len(df) < 50: return None
        
        # 確実に1次元のpd.Seriesとして抽出
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

        # Target作成ロジック（未来データの欠損による学習ノイズを防止）
        f_high = pd.concat([h.shift(-i) for i in range(1, LOOKAHEAD_BARS + 1)], axis=1).max(axis=1) - c
        f_low = c - pd.concat([l.shift(-i) for i in range(1, LOOKAHEAD_BARS + 1)], axis=1).min(axis=1)
        tp_t, sl_t = new_cols["ATR"] * 1.0, new_cols["ATR"] * 0.5
        cond_buy = (f_high >= tp_t) & (f_low < sl_t)
        cond_sell = (f_low >= tp_t) & (f_high < sl_t)
        
        target_series = pd.Series(np.nan, index=df.index)
        valid_future = f_high.notna() & f_low.notna()
        target_series[valid_future & cond_buy] = 1
        target_series[valid_future & cond_sell] = -1
        target_series[valid_future & ~cond_buy & ~cond_sell] = 0
        df["Target"] = target_series
        
        return df.dropna(subset=[col for col in df.columns if col != "Target"])
    except Exception: 
        return None

@st.cache_data(ttl=30, show_spinner=False)
def analyze_signal_with_backtest(df_current, df_htf):
    empty_res = ("WAIT (データ不足)", 0.0, 0.0, "不明", {"buy": 0.0, "sell": 0.0, "wait": 100.0})
    if df_current is None or len(df_current) < 200:
        return empty_res

    try:
        avail = [f for f in FEATURE_COLUMNS if f in df_current.columns]
        
        df_valid = df_current.dropna(subset=["Target"])
        test_size = 100
        min_required = test_size + LOOKAHEAD_BARS + 50
        
        if len(df_valid) < min_required:
            return ("WAIT (学習データ不足)", 0.0, 0.0, "判定不可", {"buy": 0.0, "sell": 0.0, "wait": 100.0})

        X = df_valid[avail]
        y = df_valid["Target"]
        
        # データリーク防止用ギャップ設置
        X_train = X.iloc[: -(test_size + LOOKAHEAD_BARS)]
        y_train = y.iloc[: -(test_size + LOOKAHEAD_BARS)]
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        if len(np.unique(y_train)) < 2: 
            return ("WAIT (データ偏り)", 0.0, 0.0, "判定不可", {"buy": 0.0, "sell": 0.0, "wait": 100.0})

        model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        valid_eval = (preds != 0) & (y_test != 0)
        win_rate = (preds[valid_eval] == y_test[valid_eval]).mean() * 100 if valid_eval.sum() > 0 else 50.0

        # 最新バー（リアルタイム足）の予測
        latest_X = df_current[avail].iloc[[-1]]
        prob_array = model.predict_proba(latest_X)[0]
        class_prob_map = dict(zip(model.classes_, prob_array))
        
        prob_up = float(class_prob_map.get(1.0, 0.0))
        prob_down = float(class_prob_map.get(-1.0, 0.0))
        prob_wait = float(class_prob_map.get(0.0, 0.0))
        conf = max(prob_up, prob_down) * 100

        prob_dict = {
            "buy": round(prob_up * 100, 1),
            "sell": round(prob_down * 100, 1),
            "wait": round(prob_wait * 100, 1)
        }

        # 日足上位足のトレンド確認
        htf_close = float(clean_series(df_htf["Close"]).iloc[-1])
        if "EMA_200" in df_htf.columns and not np.isnan(clean_series(df_htf["EMA_200"]).iloc[-1]):
            htf_ema200 = float(clean_series(df_htf["EMA_200"]).iloc[-1])
            htf_uptrend = htf_close > htf_ema200
        else:
            htf_uptrend = htf_close > float(clean_series(df_htf["SMA_20"]).iloc[-1])

        if prob_up >= 0.70 and htf_uptrend:
            status = "BUY (買い)"
        elif prob_down >= 0.70 and not htf_uptrend:
            status = "SELL (売り)"
        else:
            status = "WAIT (様子見)"

        adx_val = float(clean_series(df_current["ADX"]).iloc[-1])
        m_type = "トレンド相場" if adx_val > 22 else "レンジ相場"

        return status, conf, win_rate, m_type, prob_dict
    except Exception:
        return ("WAIT (エラー)", 0.0, 0.0, "エラー", {"buy": 0.0, "sell": 0.0, "wait": 100.0})

# ==========================================
# 2. UI構築
# ==========================================
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

# サイドバー設定
st.sidebar.header("⚙️ 資金 & リスク設定")
account_balance = st.sidebar.number_input("口座資金 (円)", min_value=10000, value=500000, step=50000, help="運用予定の口座残高を入力してください。")
quantity_wan = st.sidebar.number_input("1注文の数量 (万通貨)", min_value=0.01, value=0.10, step=0.01, help="1回の注文あたりの数量です。松井証券では100通貨(0.01万)単位で指定可能です。")

st.sidebar.markdown("---")
st.sidebar.header("🔄 更新設定")
if HAS_AUTOREFRESH:
    auto_refresh = st.sidebar.checkbox("60秒ごとに自動更新", value=False)
    if auto_refresh:
        st_autorefresh(interval=60000, key="datarefresh")

if st.sidebar.button("🔄 最新データに手動更新"):
    st.cache_data.clear()
    st.rerun()

# データ取得
data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
data_4h = load_and_process_data(ticker, "1y", "1h", "4時間足 (中期・リピート用)")
data_htf = load_and_process_data(ticker, "3y", "1d", "日足")

if data_4h is None and data is not None:
    data_4h = data

if data is None or data_htf is None:
    st.error("データの取得に失敗しました。時間足または通貨ペアを変更してください。")
    st.stop()

now_jst = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f'<div class="update-time">最終データ取得日時: <b>{now_jst} JST</b></div>', unsafe_allow_html=True)

latest_price = float(clean_series(data["Close"]).iloc[-1])
latest_atr = float(clean_series(data["ATR"]).iloc[-1])

status, conf, win_rate, m_type, prob_dict = analyze_signal_with_backtest(data, data_htf)

# --- サマリーダッシュボード ---
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

# --- 4時間足基準のリピートレンジ計算 ---
swing_high_4h = float(clean_series(data_4h["High"]).iloc[-100:].max())
swing_low_4h = float(clean_series(data_4h["Low"]).iloc[-100:].min())
atr_4h = float(clean_series(data_4h["ATR"]).iloc[-1])

# レンジ設定UI
with st.expander("⚙️ リピート自動売買のレンジ調整", expanded=False):
    rc1, rc2 = st.columns(2)
    in_lower = rc1.number_input("レンジ下限", value=swing_low_4h, step=0.1 if is_jpy else 0.001, format=price_fmt)
    in_upper = rc2.number_input("レンジ上限", value=swing_high_4h, step=0.1 if is_jpy else 0.001, format=price_fmt)
    
    user_lower = min(in_lower, in_upper)
    user_upper = max(in_lower, in_upper)
    if user_lower == user_upper:
        user_upper += pip_unit * 100.0
    user_half = (user_upper + user_lower) / 2.0

# 運用停止ライン（SL）の算出
stop_buffer_pips = max(10.0, round((atr_4h / pip_unit) * 1.5, 1))
stop_buffer_val = stop_buffer_pips * pip_unit
buy_stop_loss = user_lower - stop_buffer_val
sell_stop_loss = user_upper + stop_buffer_val

# --- タブ構造によるUI構築 ---
tab_chart, tab_repeat, tab_ai = st.tabs(["📈 メインチャート", "📋 松井証券 リピート設定 & リスク管理", "🤖 AIモデル分析詳細"])

# --- タブ1: メインチャート ---
with tab_chart:
    df_chart = safe_to_tokyo_tz(data.tail(120))
    # 週末ギャップ排除のためインデックスを年月日・時間を含む一意な文字列カテゴリーに変換
    x_labels = df_chart.index.strftime('%Y-%m-%d %H:%M')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=x_labels,
        open=clean_series(df_chart["Open"]),
        high=clean_series(df_chart["High"]),
        low=clean_series(df_chart["Low"]),
        close=clean_series(df_chart["Close"]),
        increasing_line_color='#22c55e', increasing_fillcolor='#22c55e',
        decreasing_line_color='#ef4444', decreasing_fillcolor='#ef4444',
        name="価格"
    ), row=1, col=1)

    # EMA200
    if "EMA_200" in df_chart.columns:
        fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["EMA_200"]), line=dict(color="#38bdf8", width=1.5), name="EMA200"), row=1, col=1)

    # リピートレンジ & 運用停止ライン描画
    fig.add_hrect(y0=user_lower, y1=user_upper, fillcolor="rgba(56, 189, 248, 0.05)", line_width=0, row=1, col=1)
    fig.add_hline(y=sell_stop_loss, line_dash="dashdot", line_color="#b91c1c", annotation_text="売 運用停止", row=1, col=1)
    fig.add_hline(y=user_upper, line_dash="dash", line_color="#ef4444", annotation_text="リピート上限", row=1, col=1)
    fig.add_hline(y=user_half, line_dash="dot", line_color="#a855f7", annotation_text="ハーフライン", row=1, col=1)
    fig.add_hline(y=user_lower, line_dash="dash", line_color="#22c55e", annotation_text="リピート下限", row=1, col=1)
    fig.add_hline(y=buy_stop_loss, line_dash="dashdot", line_color="#15803d", annotation_text="買 運用停止", row=1, col=1)

    # RSI
    if "RSI" in df_chart.columns:
        fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["RSI"]), line=dict(color="#a855f7", width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=560,
        margin=dict(l=10, r=70, t=10, b=10),
        template="plotly_dark",
        hovermode="x unified",
        dragmode="pan",
        showlegend=False
    )
    # カテゴリー軸化により週末の隙間を自動スキップ
    fig.update_xaxes(type='category', nticks=12, showspikes=True)
    fig.update_yaxes(side="right")

    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})

# --- タブ2: 松井証券 リピート設定 & リスク管理 ---
with tab_repeat:
    trap_width_pips = max(15, int(round((atr_4h / pip_unit))))
    half_range_pips = abs(user_upper - user_half) / pip_unit
    
    # 松井証券の実仕様に合わせた片側格子数の厳格計算
    half_grid_count = max(1, int(np.floor(half_range_pips / trap_width_pips)))
    total_grid_count = half_grid_count * 2

    # リスク計算
    order_units = int(quantity_wan * 10000)
    usd_rate = latest_price if ticker == "USDJPY=X" else get_usdjpy_rate()
    pip_value_yen = (order_units / 10000.0) * 100 if is_jpy else (order_units * 0.0001 * usd_rate)
    
    # 片側全トラップ保持状態で運用停止ライン(SL)に達した際の最大想定含み損
    max_loss_yen = 0.0
    for i in range(1, half_grid_count + 1):
        dist_pips = (i * trap_width_pips) + stop_buffer_pips
        max_loss_yen += dist_pips * pip_value_yen
        
    margin_per_order = (latest_price * order_units) / 25.0 if is_jpy else (latest_price * usd_rate * order_units) / 25.0
    total_margin_yen = margin_per_order * half_grid_count
    risk_ratio = (max_loss_yen / account_balance) * 100 if account_balance > 0 else 0.0

    st.markdown(f"""
    <div class="param-box">
    <b>【ハーフ＆ハーフ推奨設定値】</b><br>
    ・<b>買い設定（下半）</b>: レンジ <code>{price_fmt % user_lower}</code> ～ <code>{price_fmt % user_half}</code> | <b>運用停止(SL)</b>: <code>{price_fmt % buy_stop_loss}</code> (-{stop_buffer_pips}pips)<br>
    ・<b>売り設定（上半）</b>: レンジ <code>{price_fmt % user_half}</code> ～ <code>{price_fmt % user_upper}</code> | <b>運用停止(SL)</b>: <code>{price_fmt % sell_stop_loss}</code> (+{stop_buffer_pips}pips)<br>
    ・<b>注文幅 / 利確幅</b>: <code>{trap_width_pips} pips</code> | <b>片側注文本数</b>: 約 <code>{half_grid_count} 本</code> (全 {total_grid_count}本)
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("▼ 松井証券の自動売買設定画面へそのままコピー＆ペーストしてご使用ください")
    st.code(f"""[松井証券リピート注文 設定値]
通貨ペア: {selected_label.split(' ')[0]}
注文種別: ハーフ＆ハーフ
買いレンジ: {price_fmt % user_lower} - {price_fmt % user_half} (SL: {price_fmt % buy_stop_loss})
売りレンジ: {price_fmt % user_half} - {price_fmt % user_upper} (SL: {price_fmt % sell_stop_loss})
注文幅 / 利確幅: {trap_width_pips} pips
1本あたりの数量: {quantity_wan:.2f} 万通貨""", language="text")

    st.markdown("##### 🛡️ リスク・資金シミュレーション")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("想定最大含み損", f"約 {int(max_loss_yen):,} 円")
    rc2.metric("片側最大 必要証拠金", f"約 {int(total_margin_yen):,} 円")
    rc3.metric("資金リスク比率", f"{risk_ratio:.1f}%")

    if risk_ratio > 40.0:
        st.error("🚨 警告: 撤退時の最大損失が口座資金の40%を超えています。数量(万通貨)を減らすか口座資金を増やしてください。")
    else:
        st.success("🟢 資金管理チェック: 適切なリスク範囲内です。")

# --- タブ3: AIモデル分析詳細 ---
with tab_ai:
    st.markdown("##### 🤖 AI予測モデル（Random Forest）の評価と内訳")
    
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**最新バーの分類判定確率**")
        
        # 安全クランプ処理（0~100に制限）
        val_buy = max(0, min(100, int(prob_dict['buy'])))
        val_sell = max(0, min(100, int(prob_dict['sell'])))
        val_wait = max(0, min(100, int(prob_dict['wait'])))

        st.write(f"🟢 **BUY (買い)**: {prob_dict['buy']}%")
        st.progress(val_buy)
        
        st.write(f"🔴 **SELL (売り)**: {prob_dict['sell']}%")
        st.progress(val_sell)
        
        st.write(f"⚪ **WAIT (様子見)**: {prob_dict['wait']}%")
        st.progress(val_wait)

    with pcol2:
        st.markdown("**アウトオブサンプル検証（リーク防止対策済み）**")
        st.metric("直近テスト100足の方向勝率", f"{win_rate:.1f}%")
        st.caption("※ 先読みデータ（Lookahead Leak）を排除した厳格なバックテスト精度です。70%以上の確率スコアと日足トレンドが一致した場合のみ推奨シリアルが発動します。")
