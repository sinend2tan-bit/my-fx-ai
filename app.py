import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="高度AI FXシグナル", layout="wide")

st.title("🤖 高度AI FX売買シグナル分析")
st.write("マルチテクニカル指標と周期性を学習したAIが、リアルタイムに売買判定を出力します。")

# 通貨ペア選択
PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}

selected_label = st.selectbox("通貨ペアを選択してください", list(PAIRS.keys()))
ticker = PAIRS[selected_label]

@st.cache_data(ttl=300)
def load_and_process_data(symbol):
    df = yf.download(symbol, period="2y", interval="1d")
    if df.empty:
        return None
    
    # マルチインデックスの解除（yfinanceの仕様対策）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. リターン（騰落率）
    df['Return'] = df['Close'].pct_change()

    # 2. 移動平均線（SMA）と乖離率
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Dev_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Dev_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

    # 3. RSI (14日)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4. ボラティリティ（10日標準偏差）
    df['Volatility'] = df['Return'].rolling(window=10).std()

    # 5. 曜日データ (0:月曜 ~ 4:金曜)
    df['DayOfWeek'] = df.index.dayofweek

    # 目的変数: 翌日の価格が上がるか（1）下がるか（0）
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()

data = load_and_process_data(ticker)

if data is None or len(data) < 50:
    st.error("データの取得に失敗したか、データ数が不足しています。しばらく待ってから再度お試しください。")
else:
    # 学習用の特徴量（AIに教える情報）
    features = ['Return', 'Dev_SMA20', 'Dev_SMA50', 'RSI', 'Volatility', 'DayOfWeek']
    X = data[features]
    y = data['Target']

    # 最新データ（予測用）とそれ以前のデータ（学習用）に分割
    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_latest = X.iloc[[-1]]

    # AIモデルの学習 (RandomForest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 予測
    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    confidence = max(prob) * 100

    latest_price = data['Close'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    latest_date = data.index[-1].strftime('%Y-%m-%d')

    # 表示セクション
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("現在レート", f"{latest_price:.3f}")
    col2.metric("RSI (14日)", f"{latest_rsi:.1f}")
    col3.metric("最新データ日付", latest_date)

    st.subheader("🤖 AI判定結果")
    if pred == 1 and confidence >= 60:
        st.success(f"🟢 **買い (BUY)** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 複数インジケーターおよび価格傾向から、上昇トレンドの可能性が高いと判断しました。")
    elif pred == 0 and confidence >= 60:
        st.error(f"🔴 **売り (SELL)** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 複数インジケーターおよび価格傾向から、下降トレンドの可能性が高いと判断しました。")
    else:
        st.warning(f"🟡 **様子見 (HOLD)** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 明確なトレンドが検出されませんでした。静観を推奨します。")

    st.divider()
    with st.expander("📊 AIが考慮したテクニカルデータ詳細"):
        st.dataframe(data[features].tail(10).style.highlight_max(axis=0))
