import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="高度AI FXシグナル (市場間分析対応)", layout="wide")

st.title("🤖 高度AI FX売買シグナル分析 (金利・株価連動モデル)")
st.write("FXテクニカル指標に加え、米10年債利回りや主要株価指数の動向を多角的に学習したAIが判定します。")

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
    # 対象通貨ペアデータの取得
    df = yf.download(symbol, period="2y", interval="1d")
    if df.empty:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 外部市場データの取得（米10年債利回り、日経平均、S&P500）
    tnx = yf.download("^TNX", period="2y", interval="1d")['Close']
    n225 = yf.download("^N225", period="2y", interval="1d")['Close']
    spx = yf.download("^GSPC", period="2y", interval="1d")['Close']

    if isinstance(tnx, pd.DataFrame): tnx = tnx.iloc[:, 0]
    if isinstance(n225, pd.DataFrame): n225 = n225.iloc[:, 0]
    if isinstance(spx, pd.DataFrame): spx = spx.iloc[:, 0]

    # データ結合
    df['US_10Y_Yield'] = tnx
    df['Nikkei225'] = n225
    df['SP500'] = spx

    # 1. 通貨ペア自身の指標
    df['Return'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Dev_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. 外部市場の変化率（ファンダメンタルズ要因）
    df['TNX_Return'] = df['US_10Y_Yield'].pct_change()
    df['N225_Return'] = df['Nikkei225'].pct_change()
    df['SPX_Return'] = df['SP500'].pct_change()

    # 3. 曜日データ
    df['DayOfWeek'] = df.index.dayofweek

    # 目的変数: 翌日の価格上昇（1）/ 下落（0）
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # 欠損値処理 (外部市場との休日ズレ等を補完)
    df = df.ffill().dropna()
    return df

data = load_and_process_data(ticker)

if data is None or len(data) < 50:
    st.error("データの取得に失敗しました。時間をおいて再試行してください。")
else:
    # 学習用の特徴量（金利・株価変化率をプラス）
    features = ['Return', 'Dev_SMA20', 'RSI', 'TNX_Return', 'N225_Return', 'SPX_Return', 'DayOfWeek']
    X = data[features]
    y = data['Target']

    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_latest = X.iloc[[-1]]

    # AIモデルの学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    confidence = max(prob) * 100

    latest_price = data['Close'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    latest_tnx = data['US_10Y_Yield'].iloc[-1]
    latest_date = data.index[-1].strftime('%Y-%m-%d')

    # メイン表示
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("現在レート", f"{latest_price:.3f}")
    col2.metric("RSI (14日)", f"{latest_rsi:.1f}")
    col3.metric("米10年債利回り", f"{latest_tnx:.2f}%")
    col4.metric("データ日付", latest_date)

    st.subheader("🤖 AI判定結果 (社会・金利情勢加味)")
    if pred == 1 and confidence >= 60:
        st.success(f"🟢 **買い (BUY)** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 米国金利や株式市場の変動トレンド、および各種テクニカル指標から**上昇優位**と判定しました。")
    elif pred == 0 and confidence >= 60:
        st.error(f"🔴 **売り (SELL)** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 米国金利や株式市場の変動トレンド、および各種テクニカル指標から**下落優位**と判定しました。")
    else:
        st.warning(f"🟡 **様子見 (HOLD)** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 市場環境およびテクニカル指標に明確な方向性が見られません。静観を推奨します。")

    st.divider()
    with st.expander("📊 学習に使用した最新の入力データ（金利・株価含む）"):
        st.dataframe(data[features].tail(10))
