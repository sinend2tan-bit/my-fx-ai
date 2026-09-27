import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ==========================================
# 1. ページ設定・定数定義
# ==========================================
st.set_page_config(
    page_title="FX AI Assistant (Simple Edition)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主要時間軸（5分、15分、1時間、4時間、日足）
TIMEFRAMES = {
    "15分足 (デイトレ用)": {"period": "1mo", "interval": "15m"},
    "5分足 (超短期)": {"period": "7d", "interval": "5m"},
    "1時間足 (メイン)": {"period": "6mo", "interval": "1h"},
    "4時間足 (中期)": {"period": "2y", "interval": "1h"},
    "日足 (長期)": {"period": "2y", "interval": "1d"},
}

PAIR_MAP = {
    "USD/JPY": "JPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/USD": "EURUSD=X"
}

# ==========================================
# 2. データ取得 & テクニカル指標計算
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()

        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Tokyo')
        else:
            df.index = df.index.tz_convert('Asia/Tokyo')

        if interval == "1h" and period == "2y":
            df = df.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

        return df
    except Exception:
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    
    df = df.copy()
    close, high, low = df['Close'], df['High'], df['Low']

    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    # MACD
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # ボリンジャーバンド
    std20 = close.rolling(20).std()
    df['BB_Upper'] = df['EMA_20'] + (std20 * 2)
    df['BB_Lower'] = df['EMA_20'] - (std20 * 2)

    # ATR (14) - ボラティリティ自動算出用
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    return df.dropna()

# ==========================================
# 3. AI分析 & マルチタイムフレーム判定
# ==========================================
def predict_signal(df: pd.DataFrame):
    if len(df) < 100:
        return "HOLD", 0.5

    df_ml = df.copy()
    df_ml['Target'] = np.where(df_ml['Close'].shift(-3) > df_ml['Close'], 1, 0)
    features = ['EMA_20', 'EMA_50', 'RSI_14', 'MACD', 'MACD_Hist', 'ATR_14']
    
    X = df_ml[features].iloc[:-3]
    y = df_ml['Target'].iloc[:-3]

    if len(X) < 50:
        return "HOLD", 0.5

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    prob = model.predict_proba(df_ml[features].iloc[[-1]])[0][1]
    
    if prob >= 0.58:
        return "BUY", prob
    elif prob <= 0.42:
        return "SELL", prob
    else:
        return "HOLD", prob

def get_mtf_summary(symbol: str) -> dict:
    summary = {}
    for name, params in TIMEFRAMES.items():
        d = fetch_stock_data(symbol, params["period"], params["interval"])
        if len(d) >= 50:
            d = add_indicators(d)
            last = d.iloc[-1]
            if last['Close'] > last['EMA_20'] and last['EMA_20'] > last['EMA_50']:
                summary[name.split(" ")[0]] = "上昇 📈"
            elif last['Close'] < last['EMA_20'] and last['EMA_20'] < last['EMA_50']:
                summary[name.split(" ")[0]] = "下降 📉"
            else:
                summary[name.split(" ")[0]] = "レンジ ➡️"
        else:
            summary[name.split(" ")[0]] = "-"
    return summary

# ==========================================
# 4. サイドバー（最小限の入力画面）
# ==========================================
st.sidebar.title("⚙️ 設定 (入力は2つだけ)")

selected_pair = st.sidebar.selectbox("通貨ペア", list(PAIR_MAP.keys()), index=0)
selected_tf = st.sidebar.selectbox("時間軸", list(TIMEFRAMES.keys()), index=0)

st.sidebar.markdown("---")
account_balance = st.sidebar.number_input("① 口座資金 (円)", min_value=10000, value=1000000, step=50000)
order_lots = st.sidebar.number_input("② 注文ロット (Lot)", min_value=0.01, value=0.10, step=0.01, help="1.0 Lot = 10万通貨 / 0.1 Lot = 1万通貨")

symbol = PAIR_MAP[selected_pair]
tf_params = TIMEFRAMES[selected_tf]

# データ取得
df = fetch_stock_data(symbol, tf_params["period"], tf_params["interval"])

if df.empty or len(df) < 50:
    st.error("データの取得に失敗しました。しばらく待ってから再読み込みしてください。")
    st.stop()

df = add_indicators(df)

# ==========================================
# 5. メイン画面表示
# ==========================================
st.title(f"🤖 FX AI 自動最適化アシスタント ({selected_pair})")

# トレンド一括パネル
mtf_data = get_mtf_summary(symbol)
cols_mtf = st.columns(len(mtf_data))
for idx, (tf_k, trend_v) in enumerate(mtf_data.items()):
    cols_mtf[idx].metric(tf_k, trend_v)

st.markdown("---")

tab_main, tab_repeat, tab_scan = st.tabs(["🎯 自動判定・最適値", "🔄 松井証券リピート自動算出", "🔍 全ペア一括スキャン"])

# ------------------------------------------
# タブ 1: AI分析 & 自動算出
# ------------------------------------------
with tab_main:
    col_chart, col_ai = st.columns([2, 1])

    with col_chart:
        # チャート描画
        is_daily = "日足" in selected_tf
        chart_x = df.index.strftime("%Y-%m-%d" if is_daily else "%m-%d %H:%M")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=chart_x, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="価格"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_x, y=df['EMA_20'], line=dict(color='orange', width=1), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_x, y=df['EMA_50'], line=dict(color='blue', width=1), name="EMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_x, y=df['RSI_14'], line=dict(color='purple', width=1), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        fig.update_layout(height=550, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_ai:
        st.subheader("💡 AI判定 & 自動算出結果")

        # 1. AI判定
        signal, buy_prob = predict_signal(df)
        if signal == "BUY":
            st.success(f"### 判定: 買 (BUY) 📈\n上昇確率: **{buy_prob*100:.1f}%**")
        elif signal == "SELL":
            st.error(f"### 判定: 売 (SELL) 📉\n上昇確率: **{buy_prob*100:.1f}%**")
        else:
            st.info(f"### 判定: 静観 (HOLD) ⏸️\n上昇確率: **{buy_prob*100:.1f}%**")

        st.markdown("---")

        # 2. ボラティリティ(ATR)から損切・利確を自動計算
        latest_close = df['Close'].iloc[-1]
        latest_atr = df['ATR_14'].iloc[-1]

        # 対円通貨とドルストレートの単位調整
        is_jpy = "JPY" in selected_pair
        pip_unit = 100.0 if is_jpy else 10000.0

        sl_pips = latest_atr * 1.5 * pip_unit  # 1.5 ATR
        tp_pips = sl_pips * 2.0                 # リスクリワード 1:2

        # 注文ロットから損失額とリスク比率を自動逆算
        units = order_lots * 100000.0
        max_loss_yen = (sl_pips / pip_unit) * units
        risk_pct = (max_loss_yen / account_balance) * 100.0

        st.markdown("#### 🛡️ 入力ロットでのリスク判定")
        st.write(f"* 想定最大損失額: **{max_loss_yen:,.0f} 円**")
        st.write(f"* 口座資金リスク率: **{risk_pct:.1f} %**")

        # リスク度の自動警告表示
        if risk_pct <= 2.5:
            st.success("安心サイズ: 適切なリスク管理ができています（2.5%以下）")
        elif risk_pct <= 5.0:
            st.warning("注意サイズ: ややリスクが高めです（2.5%〜5%）")
        else:
            st.error("危険サイズ: ロットが大きすぎます。数量を減らすことを推奨します")

        st.markdown("---")
        st.markdown("#### 🎯 AI推奨の自動目安値")
        st.write(f"* **現在価格:** `{latest_close:.3f}`")
        st.write(f"* **推奨 損切り幅 (SL):** `{sl_pips:.1f} pips` ({latest_close - (sl_pips/pip_unit):.3f})")
        st.write(f"* **推奨 利確幅 (TP):** `{tp_pips:.1f} pips` ({latest_close + (tp_pips/pip_unit):.3f})")

# ------------------------------------------
# タブ 2: 松井証券リピート注文（完全自動算出）
# ------------------------------------------
with tab_repeat:
    st.subheader("🔄 松井証券 リピート設定 (AI自動提案)")
    st.write("直近の価格範囲とボラティリティから、最適なリピート設定を自動計算しました。")

    latest_close = df['Close'].iloc[-1]
    latest_atr = df['ATR_14'].iloc[-1]
    is_jpy = "JPY" in selected_pair

    # AIが最適なレンジ上限・下限、本数、利確幅を自動計算
    auto_range_margin = latest_atr * 20.0  # 過去の変動から広めのレンジを設定
    auto_range_min = round(latest_close - auto_range_margin, 1)
    auto_range_max = round(latest_close + auto_range_margin, 1)
    
    # ロット数に応じた1本あたりの通貨数 (入力を生かす)
    auto_quantity = max(int((order_lots * 100000) / 20), 100) # 20分割
    auto_tp_width = round(latest_atr * 2.0, 1) if is_jpy else round(latest_atr * 2.0, 4)
    auto_grids = 20

    col_r1, col_r2 = col1, col2 = st.columns(2)
    with col_r1:
        st.markdown("#### 🤖 AIが弾き出した最適パラメータ")
        st.write(f"* **想定レンジ下限:** `{auto_range_min}`")
        st.write(f"* **想定レンジ上限:** `{auto_range_max}`")
        st.write(f"* **仕掛け本数:** `{auto_grids} 本`")
        st.write(f"* **1本あたりの注文数量:** `{auto_quantity:,} 通貨`")
        st.write(f"* **推奨 利確幅:** `{auto_tp_width} 円`")

    with col_r2:
        st.markdown("#### 💰 資金・収益目安")
        total_units = auto_quantity * auto_grids
        req_margin = (latest_close * total_units) / 25.0
        profit_per_hit = auto_quantity * auto_tp_width

        st.write(f"* **合計運用数量:** `{total_units:,} 通貨`")
        st.write(f"* **概算必要証拠金:** `{req_margin:,.0f} 円`")
        st.write(f"* **1回リピートあたりの利益:** `{profit_per_hit:,.0f} 円`")
        
        if req_margin > account_balance:
            st.error("⚠️ 証拠金が資金を超えています。サイドバーの「注文ロット」を下げてください。")
        else:
            st.success("✅ 口座資金の範囲内で運用可能な設定です。")

# ------------------------------------------
# タブ 3: 全ペア一括スキャン
# ------------------------------------------
with tab_scan:
    st.subheader("🔍 全ペア AI一括スキャン")
    st.write("設定された資金とロットに基づき、全通貨ペアをスキャンします。")

    if st.button("全通貨ペアをスキャン"):
        results = []
        bar = st.progress(0)
        for idx, (p_name, p_sym) in enumerate(PAIR_MAP.items()):
            p_df = fetch_stock_data(p_sym, tf_params["period"], tf_params["interval"])
            if len(p_df) >= 50:
                p_df = add_indicators(p_df)
                sig, prob = predict_signal(p_df)
                price = p_df['Close'].iloc[-1]
            else:
                sig, prob, price = "-", 0.5, 0.0

            results.append({
                "通貨ペア": p_name,
                "現在価格": f"{price:.3f}",
                "AI判定": sig,
                "上昇確率": f"{prob*100:.1f}%"
            })
            bar.progress((idx + 1) / len(PAIR_MAP))

        st.dataframe(pd.DataFrame(results), use_container_width=True)
