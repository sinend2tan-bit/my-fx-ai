import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ==========================================
# 1. ページ基本設定
# ==========================================
st.set_page_config(
    page_title="FX デイトレ環境認識 & 構造分析ツール",
    layout="wide",
    initial_sidebar_state="expanded"
)

# UIスタイル定義
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #2a2e39;
    }
    .zone-premium {
        color: #ef5350;
        font-weight: bold;
    }
    .zone-discount {
        color: #26a69a;
        font-weight: bold;
    }
    .zone-eq {
        color: #ffb74d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 通貨ペア・タイムフレーム定義
# ==========================================
PAIRS = {
    "USD/JPY": "USDJPY=X",
    "EUR/USD": "EURUSD=X",
    "GBP/JPY": "GBPJPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "AUD/JPY": "AUDJPY=X"
}

TIMEFRAMES = {
    "15分足 (デイトレ実行足)": {"period": "10d", "interval": "15m"},
    "1時間足 (メイントレンド)": {"period": "1mo", "interval": "1h"},
    "4時間足 (上位構造)": {"period": "3mo", "interval": "1h"},  # 1hから合成
    "日足 (長期環境)": {"period": "1y", "interval": "1d"}
}

# ==========================================
# 3. データ取得・分析ロジック
# ==========================================
@st.cache_data(ttl=180, show_spinner=False)
def load_market_data(symbol: str, period: str, interval: str):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 4時間足合成処理
        if interval == "4h_resample":
            df = df.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

        # テクニカル指標計算
        # SMA / EMA
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # ATR (14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        return df.dropna()
    except Exception:
        return None

def find_swing_points(df, window=5):
    """構造高値（Swing High）と構造安値（Swing Low）を抽出"""
    highs = df['High'].values
    lows = df['Low'].values
    
    swing_highs = []
    swing_lows = []

    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            swing_highs.append((df.index[i], highs[i]))
        if lows[i] == min(lows[i - window : i + window + 1]):
            swing_lows.append((df.index[i], lows[i]))

    return swing_highs, swing_lows

# ==========================================
# 4. サイドバー設定
# ==========================================
st.sidebar.title("🔍 相場構造アナライザー")

selected_pair_label = st.sidebar.selectbox("分析通貨ペア", list(PAIRS.keys()))
selected_tf_label = st.sidebar.selectbox("実行時間足", list(TIMEFRAMES.keys()))

symbol = PAIRS[selected_pair_label]
tf_config = TIMEFRAMES[selected_tf_label]
is_jpy = "JPY" in symbol
pip_unit = 0.01 if is_jpy else 0.0001
price_fmt = ":.3f" if is_jpy else ":.5f"

st.sidebar.markdown("---")
st.sidebar.subheader("📐 分析パラメーター")
swing_window = st.sidebar.slider("スイング検出感度", min_value=3, max_value=15, value=5, help="数値が大きいほど長期の重要な高安値を抽出します")

# ==========================================
# 5. メイン表示処理
# ==========================================
st.title(f"📊 {selected_pair_label} - デイトレ環境認識ボード")

df_current = load_market_data(symbol, tf_config["period"], tf_config["interval"])
df_daily = load_market_data(symbol, "6mo", "1d")

if df_current is None or len(df_current) < 50:
    st.error("データの取得に失敗しました。時間足を変更するか、時間をおいて再試行してください。")
    st.stop()

# 最新値取得
latest_close = float(df_current['Close'].iloc[-1])
latest_atr = float(df_current['ATR'].iloc[-1])
latest_rsi = float(df_current['RSI'].iloc[-1])
ema_200 = float(df_current['EMA_200'].iloc[-1])
atr_pips = latest_atr / pip_unit

# 構造高安値・Zone分析
swing_highs, swing_lows = find_swing_points(df_current, window=swing_window)

# 直近の最重要高値・安値
recent_high = max([h[1] for h in swing_highs[-3:]]) if swing_highs else float(df_current['High'].tail(30).max())
recent_low = min([l[1] for l in swing_lows[-3:]]) if swing_lows else float(df_current['Low'].tail(30).min())

# Premium / Discount / Equilibrium 計算
swing_range = recent_high - recent_low
equilibrium = recent_low + (swing_range * 0.5)

current_zone = "Premium (高値圏 / 売り検討)" if latest_close > (equilibrium + swing_range * 0.05) else \
               "Discount (安値圏 / 買い検討)" if latest_close < (equilibrium - swing_range * 0.05) else \
               "Equilibrium (中立 / レンジ中央)"

# 距離計算
dist_to_high_pips = (recent_high - latest_close) / pip_unit
dist_to_low_pips = (latest_close - recent_low) / pip_unit

# 日足の方向性
daily_close = float(df_daily['Close'].iloc[-1])
daily_ema200 = float(df_daily['EMA_200'].iloc[-1]) if 'EMA_200' in df_daily.columns else daily_close
daily_trend = "上昇（200EMA上）" if daily_close > daily_ema200 else "下降（200EMA下）"

# --- メトリクス表示 ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("現在レート", f"{latest_close:.3f}" if is_jpy else f"{latest_close:.5f}")
m2.metric("14日 ATR (期待変動幅)", f"{atr_pips:.1f} pips")
m3.metric("直近構造高値まで", f"{dist_to_high_pips:.1f} pips", delta_color="inverse")
m4.metric("直近構造安値まで", f"{dist_to_low_pips:.1f} pips", delta_color="normal")

st.markdown("---")

# ==========================================
# 6. インタラクティブ構造チャート
# ==========================================
tab_chart, tab_checklist, tab_levels = st.tabs(["📈 相場構造チャート", "✅ エントリー環境認識チェック", "📍 主要サポレジ一覧"])

with tab_chart:
    st.subheader("相場構造 & ゾーン可視化")
    
    # チャートデータの準備（直近80本）
    df_plot = df_current.tail(80).copy()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'],
        name="価格"
    ), row=1, col=1)

    # 移動平均線
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_20'], mode='lines', name='20 SMA', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_200'], mode='lines', name='200 EMA (トレンド分け目)', line=dict(color='#3498db', width=2)), row=1, col=1)

    # 直近の構造高値・安値・Equilibriumライン描画
    fig.add_hline(y=recent_high, line_dash="dash", line_color="#ef5350", annotation_text=f"構造高値 (Resist): {recent_high:.3f}" if is_jpy else f"Resist: {recent_high:.5f}", row=1, col=1)
    fig.add_hline(y=recent_low, line_dash="dash", line_color="#26a69a", annotation_text=f"構造安値 (Support): {recent_low:.3f}" if is_jpy else f"Support: {recent_low:.5f}", row=1, col=1)
    fig.add_hline(y=equilibrium, line_dash="dot", line_color="#ffb74d", annotation_text=f"50% 平衡価格 (Equilibrium): {equilibrium:.3f}" if is_jpy else f"50%: {equilibrium:.5f}", row=1, col=1)

    # Premium / Discount ゾーン帯の背景描画
    fig.add_hrect(y0=equilibrium, y1=recent_high, fillcolor="rgba(239, 83, 80, 0.08)", layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=recent_low, y1=equilibrium, fillcolor="rgba(38, 166, 154, 0.08)", layer="below", line_width=0, row=1, col=1)

    # スイングポイントのプロット
    for idx, val in swing_highs:
        if idx in df_plot.index:
            fig.add_trace(go.Scatter(x=[idx], y=[val], mode="markers+text", marker=dict(symbol="triangle-down", size=9, color="red"), text=["SH"], textposition="top center", showlegend=False), row=1, col=1)
    for idx, val in swing_lows:
        if idx in df_plot.index:
            fig.add_trace(go.Scatter(x=[idx], y=[val], mode="markers+text", marker=dict(symbol="triangle-up", size=9, color="green"), text=["SL"], textposition="bottom center", showlegend=False), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], mode='lines', name='RSI', line=dict(color='#9b59b6', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=2, col=1)

    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_checklist:
    st.subheader("🎯 デイトレ実行前 環境認識セルフチェック")
    st.caption("トレードに入る前に、根拠が重複しているか客観的に確認してください。")

    col_chk1, col_chk2 = st.columns(2)

    with col_chk1:
        st.markdown("#### 1. 価格の位置関係 (Zone)")
        if "Premium" in current_zone:
            st.error(f"🔴 **現在地: Premium Zone (高値圏)**\n\n* 安易な『買い』は高値掴みの危険度が高いエリアです。\n* 売り根拠（トリガー）を探すか、押し目を待つのが定石です。")
        elif "Discount" in current_zone:
            st.success(f"🟢 **現在地: Discount Zone (安値圏)**\n\n* 安易な『売り』は突っ込み売りの危険度が高いエリアです。\n* 買い根拠（トリガー）を探すか、戻りを待つのが定石です。")
        else:
            st.warning(f"🟡 **現在地: Equilibrium (中間値)**\n\n* レンジの中央付近です。方向感がなく、最も騙されやすい危険帯です。")

        st.markdown("#### 2. 上位足トレンドとの一致")
        st.info(f"🌐 **日足トレンド状態**: {daily_trend}")
        if latest_close > ema_200:
            st.write(f"・{selected_tf_label}では **200EMAの上に位置（買い優勢）**")
        else:
            st.write(f"・{selected_tf_label}では **200EMAの下に位置（売り優勢）**")

    with col_chk2:
        st.markdown("#### 3. エントリーフィルタ（自己点検リスト）")
        st.checkbox("上位足（日足・4時間足）のトレンド方向に沿ったエントリーか？")
        st.checkbox("重要なサポート/レジスタンスラインに十分に引きつけているか？")
        st.checkbox("直近高値/安値までの値幅（利確幅）は、リスク（損切幅）に対して1:1.5以上あるか？")
        st.checkbox("下位足（1分足/5分足など）で明確な反転シグナル（包み足・ピンバー等）が出たか？")
        st.checkbox("今後2時間以内に主要国の経済指標発表が控えていないか？")

with tab_levels:
    st.subheader("📍 意識されている主要価格帯 (スイングポイント抽出)")
    
    highs_df = pd.DataFrame(swing_highs, columns=["日時", "価格（レジスタンス）"]).sort_values(by="日時", ascending=False).head(5)
    lows_df = pd.DataFrame(swing_lows, columns=["日時", "価格（サポート）"]).sort_values(by="日時", ascending=False).head(5)

    c_lvl1, c_lvl2 = st.columns(2)
    with c_lvl1:
        st.markdown("##### 🔴 直近のレジスタンス高値")
        if not highs_df.empty:
            highs_df["現在地からの距離"] = [(f"+{(v - latest_close)/pip_unit:.1f} pips") for v in highs_df["価格（レジスタンス）"]]
            st.dataframe(highs_df, use_container_width=True)
        else:
            st.write("検出された高値はありません")

    with c_lvl2:
        st.markdown("##### 🟢 直近のサポート安値")
        if not lows_df.empty:
            lows_df["現在地からの距離"] = [(f"-{(latest_close - v)/pip_unit:.1f} pips") for v in lows_df["価格（サポート）"]]
            st.dataframe(lows_df, use_container_width=True)
        else:
            st.write("検出された安値はありません")
