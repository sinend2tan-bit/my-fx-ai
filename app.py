import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime

# 画面表示の設定
st.set_page_config(page_title="AI FX Signal App", page_icon="🤖", layout="centered")

st.title("🤖 自動学習型 AI FXシグナル")
st.caption("最新データをリアルタイム取得し、AIが自動再学習を行って売買判定を出力します。")

# 通貨ペア選択
symbol = st.selectbox("通貨ペアを選択", ["USDJPY=X", "EURJPY=X", "GBPJPY=X", "EURUSD=X"], index=0)

# 手動再学習（キャッシュクリア）ボタン
if st.button("🔄 最新データでAIを再学習させる"):
    st.cache_resource.clear()
    st.success("AIの再学習が完了しました！")

# AI学習 ＆ 予測関数
@st.cache_resource(ttl=3600)  # 1時間ごとに全自動で再学習・更新
def train_and_get_model(sym):
    # 過去2年分のデータを取得
    df = yf.download(tickers=sym, period="2y", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'][sym]
    else:
        close = df['Close']
    
    data = pd.DataFrame({"Close": close})
    
    # テクニカル指標（特徴量）の計算
    data['Return'] = data['Close'].pct_change()
    data['SMA5_Ratio'] = data['Close'] / data['Close'].rolling(5).mean() - 1
    data['SMA20_Ratio'] = data['Close'] / data['Close'].rolling(20).mean() - 1
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # 未来の判定基準（ターゲット）作成
    future_return = data['Close'].pct_change(3).shift(-3)
    data['Target'] = 0
    data.loc[future_return > 0.002, 'Target'] = 1   # 上昇予想
    data.loc[future_return < -0.002, 'Target'] = -1  # 下落予想
    
    df_clean = data.dropna()
    features = ['Return', 'SMA5_Ratio', 'SMA20_Ratio', 'RSI']
    X = df_clean[features][:-3]
    y = df_clean['Target'][:-3]
    
    # 機械学習（AI）の学習実行
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    latest_x = df_clean[features].iloc[[-1]]
    latest_price = float(df_clean['Close'].iloc[-1])
    
    return model, latest_x, latest_price

# ライブラリ設定ファイル用（yfinance/scikit-learn等）
try:
    with st.spinner("AIが最新相場データを学習中..."):
        model, latest_x, latest_price = train_and_get_model(symbol)
    
    # 予測出力
    action_code = model.predict(latest_x)[0]
    probs = model.predict_proba(latest_x)[0]
    confidence = np.max(probs) * 100
    
    st.metric(label=f"現在レート ({symbol})", value=f"{latest_price:.3f}")
    
    if action_code == 1:
        st.success(f"🟢 **買い (BUY) シグナル** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 直近のパターンから見て上昇の確率が高い局面です。")
    elif action_code == -1:
        st.error(f"🔴 **売り (SELL) シグナル** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 直近のパターンから見て下落の確率が高い局面です。")
    else:
        st.warning(f"🟡 **様子見 (HOLD)** （AI信頼度: {confidence:.1f}%）")
        st.write("AI判定: 明確なトレンドがありません。静観を推奨します。")
        
    st.caption(f"最終分析・自動更新日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    st.error("データの取得処理中にエラーが発生しました。時間をおいて再試行してください。")
