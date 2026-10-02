import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 画面設定
st.set_page_config(page_title="シンプルFX環境認識", layout="centered")

# 通貨ペア
PAIRS = {
    "米ドル/円 (USD/JPY)": "USDJPY=X",
    "ユーロ/円 (EUR/JPY)": "EURJPY=X",
    "ポンド/円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ/米ドル (EUR/USD)": "EURUSD=X"
}

st.title("🔰 シンプルFX デイトレ環境認識")

# 通貨ペア選択
selected_label = st.selectbox("通貨ペアを選択してください", list(PAIRS.keys()))
symbol = PAIRS[selected_label]
is_jpy = "JPY" in symbol
pip_unit = 0.01 if is_jpy else 0.0001

# データ取得
@st.cache_data(ttl=180)
def get_simple_data(ticker):
    df_1h = yf.download(ticker, period="1mo", interval="1h", progress=False)
    df_1d = yf.download(ticker, period="6mo", interval="1d", progress=False)
    
    for df in [df_1h, df_1d]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
    return df_1h.dropna(), df_1d.dropna()

df_1h, df_1d = get_simple_data(symbol)

if df_1h.empty or df_1d.empty:
    st.error("データを取得できませんでした。しばらく置いてから再度お試しください。")
    st.stop()

# 計算処理
latest_price = float(df_1h['Close'].iloc[-1])

# 日足トレンド（200日移動平均線）
daily_sma200 = float(df_1d['Close'].rolling(200).mean().iloc[-1]) if len(df_1d) >= 200 else float(df_1d['Close'].mean())
if latest_price > daily_sma200:
    trend_status = "買い有利 📈"
    trend_detail = "日足レベルで上昇トレンド中です。押し目買いを検討しやすい状態です。"
    status_color = "success"
else:
    trend_status = "売り有利 📉"
    trend_detail = "日足レベルで下降トレンド中です。戻り売りを検討しやすい状態です。"
    status_color = "error"

# 1時間足の直近高値・安値（壁）
recent_high = float(df_1h['High'].tail(48).max())  # 過去2日間の最高値
recent_low = float(df_1h['Low'].tail(48).min())    # 過去2日間の最安値

p_to_high = (recent_high - latest_price) / pip_unit
p_to_low = (latest_price - recent_low) / pip_unit

# 1日の変動幅目安 (ATR)
high_low = df_1d['High'] - df_1d['Low']
atr_daily = float(high_low.tail(14).mean()) / pip_unit

# --------------------------------------------------
# UI描画 (上から順に見るだけ)
# --------------------------------------------------
st.markdown("---")

# 1. 現在価格と目線
st.subheader("1. 今日の基本方針（目線）")
col1, col2 = st.columns([1, 2])
col1.metric("現在レート", f"{latest_price:.3f}" if is_jpy else f"{latest_price:.5f}")

if status_color == "success":
    col2.success(f"**【{trend_status}】**\n\n{trend_detail}")
else:
    col2.error(f"**【{trend_status}】**\n\n{trend_detail}")

st.markdown("---")

# 2. 意識すべき価格（壁）
st.subheader("2. 意識すべき「壁（注目価格）」")
st.caption("※直近2日間で最も反発した高値・安値です。この付近での逆張りや、抜けた方向への追随を検討します。")

w_col1, w_col2 = st.columns(2)
w_col1.metric(
    label="🔴 上の壁（高値・レジスタンス）",
    value=f"{recent_high:.3f}" if is_jpy else f"{recent_high:.5f}",
    delta=f"あと {p_to_high:.1f} pips 上"
)
w_col2.metric(
    label="🟢 下の壁（安値・サポート）",
    value=f"{recent_low:.3f}" if is_jpy else f"{recent_low:.5f}",
    delta=f"あと {p_to_low:.1f} pips 下",
    delta_color="normal"
)

st.markdown("---")

# 3. 今日の値幅の目安
st.subheader("3. 今日の値幅目安")
st.info(f"💡 1日の平均変動幅は **約 {atr_daily:.1f} pips** です。\n\nすでに大きく動いている場合は深追いを避け、壁付近まで引きつけるのが無難です。")
