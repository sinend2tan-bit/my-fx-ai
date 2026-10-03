import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. ページ基本設定 & スタイル
# ==========================================
st.set_page_config(
    page_title="FX 構造分析 & 松井証券リピート自動計算 Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .param-box {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2a2e39;
        margin-bottom: 12px;
    }
    .param-box-buy { border-left: 5px solid #26a69a; }
    .param-box-sell { border-left: 5px solid #ef5350; }
    .label-title { color: #90a4ae; font-size: 0.9em; }
    .val-highlight { color: #ffffff; font-weight: bold; font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 通貨ペア設定
# ==========================================
PAIRS = {
    "米ドル/円 (USD/JPY)": "USDJPY=X",
    "ユーロ/円 (EUR/JPY)": "EURJPY=X",
    "ポンド/円 (GBP/JPY)": "GBPJPY=X",
    "豪ドル/円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ/米ドル (EUR/USD)": "EURUSD=X",
    "ポンド/米ドル (GBP/USD)": "GBPUSD=X"
}

# ==========================================
# 3. データ取得 & 構造計算関数
# ==========================================
@st.cache_data(ttl=180, show_spinner=False)
def load_market_data(symbol: str):
    try:
        # 1時間足（構造・デイトレ用）と 日足（長期・ATR用）を取得
        df_1h = yf.download(symbol, period="2mo", interval="1h", progress=False)
        df_1d = yf.download(symbol, period="1y", interval="1d", progress=False)

        for df in [df_1h, df_1d]:
            if df.empty: return None, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        # 1時間足指標
        df_1h['SMA_20'] = df_1h['Close'].rolling(20).mean()
        df_1h['EMA_200'] = df_1h['Close'].ewm(span=200, adjust=False).mean()
        
        # 14期間 RSI
        delta = df_1h['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df_1h['RSI'] = 100 - (100 / (1 + rs))

        # 日足指標 (200日線 & 日足ATR)
        df_1d['EMA_200'] = df_1d['Close'].ewm(span=200, adjust=False).mean()
        tr = pd.concat([
            df_1d['High'] - df_1d['Low'],
            (df_1d['High'] - df_1d['Close'].shift()).abs(),
            (df_1d['Low'] - df_1d['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        df_1d['ATR'] = tr.rolling(14).mean()

        return df_1h.dropna(), df_1d.dropna()
    except Exception:
        return None, None

def find_swing_points(df, window=5):
    """構造高値・構造安値を自動抽出"""
    highs, lows = df['High'].values, df['Low'].values
    swing_highs, swing_lows = [], []
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            swing_highs.append((df.index[i], highs[i]))
        if lows[i] == min(lows[i - window : i + window + 1]):
            swing_lows.append((df.index[i], lows[i]))
    return swing_highs, swing_lows

# ==========================================
# 4. サイドバー（統一設定）
# ==========================================
st.sidebar.title("⚙️ 設定パネル")

selected_label = st.sidebar.selectbox("分析通貨ペア", list(PAIRS.keys()))
symbol = PAIRS[selected_label]
is_jpy = "JPY" in symbol
pip_unit = 0.01 if is_jpy else 0.0001
price_fmt = "{:.3f}" if is_jpy else "{:.5f}"

st.sidebar.markdown("---")
st.sidebar.subheader("💰 松井証券リピート資金設定")
account_balance = st.sidebar.number_input("口座資金 (円)", min_value=10000, value=1000000, step=50000)
quantity_wan = st.sidebar.number_input("1注文あたり数量 (万通貨)", min_value=0.001, value=0.1, step=0.01, format="%.3f")
custom_quantity = int(round(quantity_wan * 10000))

st.sidebar.markdown("---")
st.sidebar.subheader("📐 相場構造パラメータ")
swing_window = st.sidebar.slider("高安値の検出感度", min_value=3, max_value=12, value=5, help="大きいほど主要な波の高安値を拾います")

# ==========================================
# 5. メイン処理 & データ準備
# ==========================================
df_1h, df_1d = load_market_data(symbol)

if df_1h is None or df_1d is None or len(df_1h) < 30:
    st.error("データの取得に失敗しました。時間をおいて再試行してください。")
    st.stop()

# 基本数値の計算
latest_price = float(df_1h['Close'].iloc[-1])
daily_atr = float(df_1d['ATR'].iloc[-1])
daily_atr_pips = daily_atr / pip_unit

daily_close = float(df_1d['Close'].iloc[-1])
daily_ema200 = float(df_1d['EMA_200'].iloc[-1])
daily_trend = "上昇傾向 📈 (200日線上)" if daily_close > daily_ema200 else "下降傾向 📉 (200日線下)"

# スイング構造分析
swing_highs, swing_lows = find_swing_points(df_1h, window=swing_window)
recent_high = max([h[1] for h in swing_highs[-3:]]) if swing_highs else float(df_1h['High'].tail(50).max())
recent_low = min([l[1] for l in swing_lows[-3:]]) if swing_lows else float(df_1h['Low'].tail(50).min())

range_width = recent_high - recent_low
equilibrium = recent_low + (range_width * 0.5)

if latest_price > (equilibrium + range_width * 0.05):
    zone_status = "Premium Zone (高値圏 / 売り検討エリア)"
    zone_color = "off"
elif latest_price < (equilibrium - range_width * 0.05):
    zone_status = "Discount Zone (安値圏 / 買い検討エリア)"
    zone_color = "normal"
else:
    zone_status = "Equilibrium (中間領域 / 静観推奨)"
    zone_color = "off"

# ヘッダー情報表示
st.title(f"🔍 {selected_label} 総合環境認識")

head1, head2, head3, head4 = st.columns(4)
head1.metric("現在レート", price_fmt.format(latest_price))
head2.metric("日足1日平均値幅 (ATR)", f"{daily_atr_pips:.1f} pips")
head3.metric("日足トレンド", daily_trend)
head4.metric("現在ゾーン", zone_status)

st.markdown("---")

# ==========================================
# 6. タブ切り替え（リピート vs デイトレ）
# ==========================================
tab_repeat, tab_daytrade = st.tabs([
    "📋 【松井証券】リピートFX 自動算出", 
    "📈 【デイトレ】構造分析 & チャート"
])

# --------------------------------------------------
# TAB 1: 松井証券リピートFX 自動算出
# --------------------------------------------------
with tab_repeat:
    st.subheader("📋 松井証券FX 自動売買（リピート注文）設定ジェネレーター")
    st.caption("現在の相場構造（主要高安値）と日足ボラティリティ（ATR）から最適なレンジと注文幅を自動計算します。")

    # リピート計算ロジック
    # ATRに基づく推奨注文値幅 (例: 日足ATRの20%〜25%程度、最低15pips)
    rec_spacing_pips = max(15, int(round(daily_atr_pips * 0.20)))
    rec_spacing_val = rec_spacing_pips * pip_unit

    # バッファーを持たせたレンジ上限・下限の設定
    buffer_val = daily_atr * 0.5
    range_upper = round(recent_high + buffer_val, 3 if is_jpy else 5)
    range_lower = round(recent_low - buffer_val, 3 if is_jpy else 5)

    # ロスカットライン (レンジ外側に1.5 ATR離す)
    buy_sl = round(range_lower - (daily_atr * 1.5), 3 if is_jpy else 5)
    sell_sl = round(range_upper + (daily_atr * 1.5), 3 if is_jpy else 5)

    # 格子（注文）本数の算出
    total_range_pips = (range_upper - range_lower) / pip_unit
    grid_count = int(total_range_pips / rec_spacing_pips) + 1

    # 必要証拠金計算（レバレッジ25倍）
    # クロス円以外のドルストレート等は簡易的にドル円155円換算で計算
    uj_rate = latest_price if is_jpy else 155.0
    required_margin_per_order = (latest_price * custom_quantity * (1.0 if is_jpy else uj_rate)) / 25.0
    total_margin_required = required_margin_per_order * grid_count
    margin_usage_pct = (total_margin_required / account_balance) * 100

    # サマリー表示
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("推定カバーレンジ幅", f"{total_range_pips:.0f} pips")
    r_col2.metric("推奨注文値幅 / 益出し幅", f"{rec_spacing_pips} pips")
    r_col3.metric("カバー注文本数", f"{grid_count} 本")
    r_col4.metric("想定証拠金使用率", f"{margin_usage_pct:.1f} %", delta="危険" if margin_usage_pct > 70 else "正常", delta_color="inverse")

    if margin_usage_pct > 70:
        st.error("⚠️ 警告: 証拠金使用率が高すぎます。資金を増やすか、「1注文あたり数量」を下げてください。")

    st.markdown("---")
    st.markdown("#### 📱 松井証券 注文画面入力用パラメータ")

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("##### 🟢 【買い】リピート設定")
        st.markdown(f"""
        <div class="param-box param-box-buy">
            <span class="label-title">通貨ペア</span>: <span class="val-highlight">{selected_label}</span><br>
            <span class="label-title">売買区分</span>: <span class="val-highlight">買</span><br>
            <span class="label-title">レンジ上限</span>: <code class="val-highlight">{price_fmt.format(range_upper)}</code><br>
            <span class="label-title">レンジ下限</span>: <code class="val-highlight">{price_fmt.format(range_lower)}</code><br>
            <span class="label-title">数量</span>: <span class="val-highlight">{quantity_wan} 万通貨</span> ({custom_quantity:,}通貨)<br>
            <span class="label-title">注文値幅</span>: <span class="val-highlight">{rec_spacing_pips} pips</span><br>
            <span class="label-title">益出し幅</span>: <span class="val-highlight">{rec_spacing_pips} pips</span><br>
            <span class="label-title">運用停止 / SL</span>: <code class="val-highlight">{price_fmt.format(buy_sl)}</code>
        </div>
        """, unsafe_allow_html=True)

    with p_col2:
        st.markdown("##### 🔴 【売り】リピート設定")
        st.markdown(f"""
        <div class="param-box param-box-sell">
            <span class="label-title">通貨ペア</span>: <span class="val-highlight">{selected_label}</span><br>
            <span class="label-title">売買区分</span>: <span class="val-highlight">売</span><br>
            <span class="label-title">レンジ上限</span>: <code class="val-highlight">{price_fmt.format(range_upper)}</code><br>
            <span class="label-title">レンジ下限</span>: <code class="val-highlight">{price_fmt.format(range_lower)}</code><br>
            <span class="label-title">数量</span>: <span class="val-highlight">{quantity_wan} 万通貨</span> ({custom_quantity:,}通貨)<br>
            <span class="label-title">注文値幅</span>: <span class="val-highlight">{rec_spacing_pips} pips</span><br>
            <span class="label-title">益出し幅</span>: <span class="val-highlight">{rec_spacing_pips} pips</span><br>
            <span class="label-title">運用停止 / SL</span>: <code class="val-highlight">{price_fmt.format(sell_sl)}</code>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# TAB 2: デイトレ構造分析 & チャート
# --------------------------------------------------
with tab_daytrade:
    st.subheader("📈 相場構造チャート & デイトレ環境認識")
    
    # 壁（距離）の計算
    dist_high_pips = (recent_high - latest_price) / pip_unit
    dist_low_pips = (latest_price - recent_low) / pip_unit

    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric("🔴 上の壁（直近高値）まで", f"{price_fmt.format(recent_high)}", f"あと {dist_high_pips:.1f} pips 上", delta_color="inverse")
    d_col2.metric("🟢 下の壁（直近安値）まで", f"{price_fmt.format(recent_low)}", f"あと {dist_low_pips:.1f} pips 下", delta_color="normal")
    d_col3.metric("⚖️ 平衡価格 (Equilibrium 50%)", price_fmt.format(equilibrium))

    # Plotlyチャート構築
    df_plot = df_1h.tail(90).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'], name="価格"
    ), row=1, col=1)

    # 移動平均線
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_20'], mode='lines', name='20 SMA', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_200'], mode='lines', name='200 EMA', line=dict(color='#3498db', width=2)), row=1, col=1)

    # 構造ライン
    fig.add_hline(y=recent_high, line_dash="dash", line_color="#ef5350", annotation_text="高値 (Resist)", row=1, col=1)
    fig.add_hline(y=recent_low, line_dash="dash", line_color="#26a69a", annotation_text="安値 (Support)", row=1, col=1)
    fig.add_hline(y=equilibrium, line_dash="dot", line_color="#ffb74d", annotation_text="50% 中心線", row=1, col=1)

    # 背景ゾーン色付け
    fig.add_hrect(y0=equilibrium, y1=recent_high, fillcolor="rgba(239, 83, 80, 0.06)", layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=recent_low, y1=equilibrium, fillcolor="rgba(38, 166, 154, 0.06)", layer="below", line_width=0, row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], mode='lines', name='RSI', line=dict(color='#9b59b6', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=2, col=1)

    fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # デイトレセルフチェックリスト
    st.markdown("#### ✅ デイトレ実行前 セルフチェック")
    chk1, chk2 = st.columns(2)
    with chk1:
        st.checkbox("日足の200日線（長期的流れ）に逆らっていないか？")
        st.checkbox("現在地は「壁（高値・安値）」に十分引きつけているか？（中途半端な場所でないか）")
    with chk2:
        st.checkbox("利確までの期待値幅は、損切り幅に対して1.5倍以上取れるか？")
        st.checkbox("直近で大きな指標発表（米CPIやFOMC等）が控えていないか？")
