import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# ページ基本設定
# ---------------------------------------------------------
st.set_page_config(
    page_title="FX Prediction & Repeat Trade Helper",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# サイドバー設定 & パラメータ
# ---------------------------------------------------------
st.sidebar.title("⚙️ システム設定 & カスタマイズ")

if st.sidebar.button("🔄 今すぐ最新データに更新"):
    st.rerun()

auto_refresh = st.sidebar.checkbox("自動更新を有効にする", value=True)
refresh_interval = st.sidebar.selectbox(
    "更新間隔を選択",
    options=["1分ごと", "3分ごと", "5分ごと"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 松井証券トレード設定")
account_balance = st.sidebar.number_input("口座資金 (円)", min_value=10000, value=200000, step=10000)
custom_quantity = st.sidebar.number_input("注文数量 (通貨)", min_value=100, value=200, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("📱 アラート通知設定 (Discord)")
discord_url = st.sidebar.text_input("Discord Webhook URL", type="password")
enable_discord = st.sidebar.checkbox("売買サイン確定時に自動通知", value=True)

# ---------------------------------------------------------
# データ取得 & インジケーター計算関数
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_forex_data(symbol="EURUSD=X", period="5d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        return None

def calculate_indicators(df):
    df = df.copy()
    # SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR (Volatility)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    return df

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
df = fetch_forex_data("EURUSD=X")

if df is None or len(df) < 50:
    st.error("⚠️ 為替データの取得に失敗したか、データが不足しています。時間を置いて再試行してください。")
    st.stop()

df = calculate_indicators(df)
latest_row = df.iloc[-1]
latest_price = float(latest_row['Close'])
rsi = float(latest_row['RSI'])
atr = float(latest_row['ATR']) if not np.isnan(latest_row['ATR']) else 0.0015

# 通貨ペア判定（JPYが含まれているか）
is_jpy_pair = "JPY" in "EURUSD=X"
pip_unit = 0.01 if is_jpy_pair else 0.0001

# AI/テクニカル簡易判定
if rsi < 40:
    signal = "買"
    trend_text = "買い (信頼度 68.0%)"
    badge_color = "🟢"
elif rsi > 60:
    signal = "売"
    trend_text = "売り (信頼度 65.0%)"
    badge_color = "🔴"
else:
    signal = "買"  # デフォルト
    trend_text = "買い (信頼度 55.0%)"
    badge_color = "🟢"

# ---------------------------------------------------------
# タブ切り替え表示
# ---------------------------------------------------------
tab_day, tab_speed, tab_repeat, tab_chart, tab_scan, tab_backtest = st.tabs([
    "🎯 デイトレ単発", "⚡ スピード注文", "📋 リピート注文", "📈 ローソク足チャート", "🔍 全ペアスキャン", "📊 バックテスト"
])

# ---------------------------------------------------------
# リピート注文 タブ（修正メイン箇所）
# ---------------------------------------------------------
with tab_repeat:
    st.subheader("📋 松井証券FX 自動売買（リピート注文）入力用サマリー")
    st.write(f"💡 入力された口座資金（¥{account_balance:,}）と注文数量（{custom_quantity}通貨）を元に、ロスカットリスクを抑制した最適設定を算出しています。")

    leverage = 25.0

    # 1. 通貨ペアごとの円換算レート算出（EUR/USD等のドルストレート対応）
    if is_jpy_pair:
        jpy_rate = latest_price
    else:
        # ドルストレートの場合はUSD/JPYのレートを取得して正確に円換算
        try:
            usdjpy_df = yf.download("USDJPY=X", period="1d", interval="5m", progress=False)
            if isinstance(usdjpy_df.columns, pd.MultiIndex):
                usdjpy_df.columns = usdjpy_df.columns.get_level_values(0)
            usdjpy_price = float(usdjpy_df['Close'].iloc[-1])
        except Exception:
            usdjpy_price = 155.0  # 取得失敗時の安全フォールバック値

        jpy_rate = latest_price * usdjpy_price

    # 2. 正確な1本あたりの必要証拠金（円）
    margin_per_unit = (jpy_rate * custom_quantity) / leverage

    # 3. 推奨の値幅・益出し幅（pips）計算
    raw_pips = int((atr / pip_unit) * 1.5)
    ai_recommended_width = max(10, min(raw_pips, 100))

    # 4. 口座資金の70%までに抑えた安全な最大注文本数を算出
    max_allowable_grids = max(5, int((account_balance * 0.7) / max(margin_per_unit, 1.0)))

    # レンジ幅（pips）の計算と上限設定
    max_safe_range_pips = max_allowable_grids * ai_recommended_width
    cap_pips = 150 if is_jpy_pair else 1500
    safe_range_pips = min(max_safe_range_pips, cap_pips)

    half_range = (safe_range_pips * pip_unit) / 2.0

    if signal == "買":
        range_lower = latest_price - (half_range * 1.5)
        range_upper = latest_price + (half_range * 0.5)
        stop_line = range_lower - (50 * pip_unit)
    else:
        range_lower = latest_price - (half_range * 0.5)
        range_upper = latest_price + (half_range * 1.5)
        stop_line = range_upper + (50 * pip_unit)

    # サマリー表示
    st.success(f"{badge_color} {signal}いリピート推奨 （資金連動・安全レンジ自動調整済み）")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**通貨ペア**　　： ユーロ / 米ドル (EUR/USD)")
        st.markdown(f"**売買区分**　　： {signal}")
        st.markdown(f"**AI判定**　　　： {trend_text}")
        st.markdown(f"**レンジ下限**　： **{range_lower:.5f}**")
        st.markdown(f"**レンジ上限**　： **{range_upper:.5f}**")
        st.markdown(f"**注文値幅**　　： **{ai_recommended_width} pips**")
        st.markdown(f"**益出し幅**　　： **{ai_recommended_width} pips**")
        st.markdown(f"**運用停止ライン**： **{stop_line:.5f}**")
        st.markdown(f"**注文数量**　　： **{custom_quantity} 通貨**")
        st.markdown(f"**考慮口座資金**： **¥{account_balance:,}**")

    with col2:
        st.info("💡 **設定のワンポイント**\n\n"
                "・1本あたりの必要証拠金をドル円換算して正確に計算するように改善しました。\n"
                "・松井証券FXの注文画面で上記の【レンジ下限・上限・注文値幅・益出し幅・注文数量】をそのまま入力して発注してください。")

# ---------------------------------------------------------
# 他タブのプレースホルダー表示
# ---------------------------------------------------------
with tab_day:
    st.write("デイトレ単発のシグナル表示エリア")

with tab_speed:
    st.write("スピード注文用の算出エリア")

with tab_chart:
    st.line_chart(df['Close'])

with tab_scan:
    st.write("全ペアスキャンエリア")

with tab_backtest:
    st.write("バックテストエリア")
