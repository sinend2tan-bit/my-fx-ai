import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="プロ版 AI FXデイトレアナライザー", layout="wide")

st.title("⚡ AI FXデイトレアナライザー (マルチタイムフレーム & 勝率検証)")
st.write("15分足/1時間足/日足の切り替え、テクニカル＋外部市場のAI学習、過去勝率の検証機能を搭載したデイトレモデルです。")

# 選択オプション
PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}

TIMEFRAMES = {
    "15分足 (デイトレエントリー用)": {"period": "1mo", "interval": "15m"},
    "1時間足 (デイトレメイン用)": {"period": "6mo", "interval": "1h"},
    "日足 (スイング・環境認識用)": {"period": "2y", "interval": "1d"},
}

col_s1, col_s2 = st.columns(2)
with col_s1:
    selected_label = st.selectbox("通貨ペアを選択", list(PAIRS.keys()))
with col_s2:
    tf_label = st.selectbox("時間軸（タイムフレーム）を選択", list(TIMEFRAMES.keys()))

ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]

@st.cache_data(ttl=180)
def load_and_process_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 外部市場指標（米10年債利回り）の取得・結合
    try:
        tnx = yf.download("^TNX", period=period, interval=interval)['Close']
        if isinstance(tnx, pd.DataFrame): tnx = tnx.iloc[:, 0]
        df['US_10Y_Yield'] = tnx
    except:
        df['US_10Y_Yield'] = np.nan

    # 1. 基礎指標
    df['Return'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Dev_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']

    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 4. ボリンジャーバンド (%B)
    std20 = df['Close'].rolling(window=20).std()
    upper_band = df['SMA_20'] + (std20 * 2)
    lower_band = df['SMA_20'] - (std20 * 2)
    df['BB_PctB'] = (df['Close'] - lower_band) / (upper_band - lower_band + 1e-10)

    # 5. ATR (値幅)
    high_low = df['High'] - df['Low']
    df['ATR'] = high_low.rolling(window=14).mean()

    # 目的変数: 翌足の価格上昇（1）/ 下落（0）
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.ffill().dropna()
    return df

data = load_and_process_data(ticker, tf_config['period'], tf_config['interval'])

if data is None or len(data) < 60:
    st.error("データの取得に失敗したか、指定時間軸のデータ数が不足しています。しばらく待ってから再試行してください。")
else:
    features = ['Return', 'Dev_SMA20', 'RSI', 'MACD_Hist', 'BB_PctB']
    if 'US_10Y_Yield' in data.columns and not data['US_10Y_Yield'].isna().all():
        data['TNX_Return'] = data['US_10Y_Yield'].pct_change()
        features.append('TNX_Return')

    data = data.dropna()
    X = data[features]
    y = data['Target']

    # 最新足を除く過去データで学習
    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_latest = X.iloc[[-1]]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- 過去勝率のバックテスト検証（直近50足分） ---
    test_len = min(50, len(X_train) - 50)
    if test_len > 10:
        X_test_hist = X_train.iloc[-test_len:]
        y_test_hist = y_train.iloc[-test_len:]
        preds_hist = model.predict(X_test_hist)
        correct_count = int((preds_hist == y_test_hist).sum())
        win_rate = (correct_count / test_len) * 100
    else:
        win_rate = 0.0
        correct_count = 0
        test_len = 0

    # 最新足の予測
    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    confidence = max(prob) * 100

    latest_price = data['Close'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    latest_atr = data['ATR'].iloc[-1]
    latest_time = data.index[-1].strftime('%Y-%m-%d %H:%M')

    # 表示セクション
    st.divider()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("現在レート", f"{latest_price:.3f}")
    m_col2.metric("RSI (14)", f"{latest_rsi:.1f}")
    m_col3.metric("直近AI予測勝率", f"{win_rate:.1f}%", f"{correct_count}/{test_len} 回的中")
    m_col4.metric("データ日時", latest_time)

    st.subheader("🤖 AI判定結果 & デイトレターゲット")

    fmt = ".5f" if "USD" in selected_label and not "USD/JPY" in selected_label else ".3f"

    if pred == 1 and confidence >= 60:
        entry_price = latest_price
        tp_price = entry_price + (latest_atr * 1.0)
        sl_price = entry_price - (latest_atr * 0.5)

        st.success(f"🟢 **買い (BUY)** （AI信頼度: {confidence:.1f}%）")
        
        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("新規買い目安 (Entry)", f"{entry_price:{fmt}}")
        t_col2.metric("利確目標 (Take Profit)", f"{tp_price:{fmt}}", f"+{latest_atr*1.0:{fmt}}")
        t_col3.metric("損切り目安 (Stop Loss)", f"{sl_price:{fmt}}", f"-{latest_atr*0.5:{fmt}}")
        
        st.write(f"💡 **デイトレアドバイス**: 選択された【{tf_label}】で上昇サイン点灯。押し目買いの指値・逆指値を設定してエントリーを検討できます。")

    elif pred == 0 and confidence >= 60:
        entry_price = latest_price
        tp_price = entry_price - (latest_atr * 1.0)
        sl_price = entry_price + (latest_atr * 0.5)

        st.error(f"🔴 **売り (SELL)** （AI信頼度: {confidence:.1f}%）")

        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("新規売り目安 (Entry)", f"{entry_price:{fmt}}")
        t_col2.metric("利確目標 (Take Profit)", f"{tp_price:{fmt}}", f"-{latest_atr*1.0:{fmt}}")
        t_col3.metric("損切り目安 (Stop Loss)", f"{sl_price:{fmt}}", f"+{latest_atr*0.5:{fmt}}")

        st.write(f"💡 **デイトレアドバイス**: 選択された【{tf_label}】で下落サイン点灯。戻り売りの指値・逆指値を設定してエントリーを検討できます。")

    else:
        st.warning(f"🟡 **様子見 (HOLD)** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 方向性が不明確、または確率が不十分です。ポジションの保有は見送り（静観）を推奨します。")

    st.divider()
    with st.expander("📊 テクニカル指標・学習データの詳細"):
        st.dataframe(data[features + ['ATR']].tail(10))
