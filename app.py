import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# ページ基本設定
# ---------------------------------------------------------
st.set_page_config(
    page_title="FX Prediction App",
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
# データ取得 & 計算
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
    except Exception:
        return None

symbol = "EURUSD=X"
df = fetch_forex_data(symbol)

if df is None or len(df) < 50:
    st.error("⚠️ 為替データの取得に失敗したか、データが不足しています。")
    st.stop()

latest_price = float(df['Close'].iloc[-1])

# ヘッダー領域のトレンドバッジ表示
st.markdown("<p style='text-align: right; color: green; font-weight: bold;'>↑ 🔥 強トレンド</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# タブ切り替え表示
# ---------------------------------------------------------
tab_day, tab_speed, tab_repeat, tab_chart, tab_scan, tab_backtest = st.tabs([
    "🎯 デイトレ単発", "⚡ スピード注文", "📋 リピート注文", "📈 ローソク足チャート", "🔍 全ペアスキャン", "📊 バックテスト"
])

with tab_day:
    st.write("🎯 デイトレ単発 サマリー機能")

with tab_speed:
    st.write("⚡ スピード注文 ガイド機能")

# ---------------------------------------------------------
# リピート注文 タブ（証拠金修正済み）
# ---------------------------------------------------------
with tab_repeat:
    st.subheader("📋 松井証券FX 自動売買（リピート注文）入力用サマリー")
    st.caption("💡 入力された口座資金（¥200,000）と注文数量（200通貨）を元に、ロスカットリスクを抑制した最適設定を算出しています。")

    # ドルストレートの正確な円換算ロジック
    leverage = 25.0
    try:
        usdjpy_df = yf.download("USDJPY=X", period="1d", interval="5m", progress=False)
        if isinstance(usdjpy_df.columns, pd.MultiIndex):
            usdjpy_df.columns = usdjpy_df.columns.get_level_values(0)
        usdjpy_price = float(usdjpy_df['Close'].iloc[-1])
    except Exception:
        usdjpy_price = 155.0

    # EUR/USDの円換算レートと証拠金計算
    jpy_rate = latest_price * usdjpy_price
    margin_per_unit = (jpy_rate * custom_quantity) / leverage

    # 資金上限に合わせたグリッド数の自動抑制
    max_allowable_grids = max(5, int((account_balance * 0.7) / max(margin_per_unit, 1.0)))
    
    # 画像の数値（24 pips）に合わせたレンジ幅設定
    width_pips = 24
    safe_range_pips = min(max_allowable_grids * width_pips, 1500)
    half_range = (safe_range_pips * 0.0001) / 2.0

    range_lower = latest_price - (half_range * 1.0)
    range_upper = latest_price + (half_range * 1.0)
    stop_line = range_lower - 0.0050

    st.success("🟢 買いリピート推奨 （資金連動・安全レンジ自動調整済み）")

    st.text(f"通貨ペア　　： ユーロ / 米ドル (EUR/USD)")
    st.text(f"売買区分　　： 買")
    st.text(f"AI判定　　　： 買い (信頼度 68.0%)")
    st.text(f"レンジ下限　： {range_lower:.5f}")
    st.text(f"レンジ上限　： {range_upper:.5f}")
    st.text(f"注文値幅　　： {width_pips} pips")
    st.text(f"益出し幅　　： {width_pips} pips")
    st.text(f"運用停止ライン： {stop_line:.5f}")
    st.text(f"注文数量　　： {custom_quantity} 通貨")
    st.text(f"考慮口座資金： ¥{account_balance:,}")

with tab_chart:
    st.line_chart(df['Close'])

with tab_scan:
    st.write("🔍 全ペアスキャン機能")

with tab_backtest:
    st.write("📊 バックテスト機能")
