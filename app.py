import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import time

st.set_page_config(page_title="プロ版 AI FXデイトレアナライザー Ultimate", layout="wide")

# ==========================================
# 1. 簡易パスワード認証機能
# ==========================================
PASSWORD = "aMV3#nRDfhZUa5"  # 👈 お好みのパスワードに変更してください

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    if st.session_state.get("password_input") == PASSWORD:
        st.session_state.authenticated = True
        del st.session_state["password_input"]
    else:
        st.error("🔒 パスワードが正しくありません。")

if not st.session_state.authenticated:
    st.title("🔒 ログイン")
    st.write("このアプリを利用するにはパスワードを入力してください。")
    st.text_input("パスワードを入力", type="password", key="password_input", on_change=check_password)
    st.button("ログイン", on_click=check_password)
    st.stop()

# ==========================================
# 2. 通知ヘルパー関数 (Discord / Telegram)
# ==========================================
def send_discord_notification(webhook_url, message):
    if not webhook_url:
        return False
    try:
        res = requests.post(webhook_url, json={"content": message}, timeout=5)
        return res.status_code == 204
    except:
        return False

# ==========================================
# 3. メイン画面 & サイドバー設定
# ==========================================
st.title("⚡ Pro AI FX デイトレアナライザー (Ultimate Edition)")

# サイドバー設定
st.sidebar.header("⚙️ システム設定 & カスタマイズ")

# ログアウト
if st.sidebar.button("🔒 ログアウト"):
    st.session_state.authenticated = False
    st.rerun()

# 手動更新
if st.sidebar.button("🔄 今すぐ最新データに更新"):
    st.cache_data.clear()
    st.rerun()

# 自動更新
st.sidebar.subheader("⏱️ 自動更新")
auto_refresh = st.sidebar.checkbox("自動更新を有効にする", value=False)
refresh_interval = st.sidebar.selectbox(
    "更新間隔を選択",
    options=[60, 180, 300],
    format_func=lambda x: f"{x // 60}分ごと",
    index=1
)

# 利確・損切り（ATR倍率）カスタマイズ
st.sidebar.subheader("🎯 ターゲット設定 (リスクリワード)")
tp_atr_mult = st.sidebar.slider("利確目標 (ATR倍率)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)
sl_atr_mult = st.sidebar.slider("損切り目安 (ATR倍率)", min_value=0.3, max_value=2.0, value=0.6, step=0.1)

# Discord通知設定
st.sidebar.subheader("📱 Discord通知設定 (オプション)")
discord_url = st.sidebar.text_input("Discord Webhook URL", type="password", help="DiscordのチャンネルWebhook URLを入力すると、売買サインを通知できます。")
enable_notify = st.sidebar.checkbox("売買サイン確定時に通知", value=False)

# 通貨ペアと時間軸
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

# ==========================================
# 4. データ取得 & 指算処理（安定化版）
# ==========================================
@st.cache_data(ttl=60)
def load_and_process_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 外部市場指標（米10年債利回り）
        try:
            tnx = yf.download("^TNX", period=period, interval=interval, progress=False)['Close']
            if isinstance(tnx, pd.DataFrame): tnx = tnx.iloc[:, 0]
            df['US_10Y_Yield'] = tnx
        except:
            df['US_10Y_Yield'] = np.nan

        # 1. リターンと移動平均乖離
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

        # 5. ATR
        high_low = df['High'] - df['Low']
        df['ATR'] = high_low.rolling(window=14).mean()

        # 6. ADX (トレンドの強さ)
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr = np.maximum(high_low, np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        atr14 = pd.Series(tr).rolling(14).mean()

        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / (atr14 + 1e-10))
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / (atr14 + 1e-10))
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        df['ADX'] = dx.rolling(14).mean()

        # 目的変数
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        df = df.ffill().dropna()
        return df
    except Exception as e:
        return None

data = load_and_process_data(ticker, tf_config['period'], tf_config['interval'])

# 注意喚起アラート
st.info("💡 **トレード前のチェック**: 雇用統計やFOMCなど主要経済指標の発表前後はテクニカル分析が不向きになります。重要指標発表直前のエントリーは控えましょう。")

# ==========================================
# 5. AI学習 & 予測エンジン
# ==========================================
if data is None or len(data) < 60:
    st.error("データの取得に失敗したか、指定時間軸のデータ数が不足しています。しばらく待ってから『今すぐ最新データに更新』ボタンを押してください。")
else:
    features = ['Return', 'Dev_SMA20', 'RSI', 'MACD_Hist', 'BB_PctB', 'ADX']
    if 'US_10Y_Yield' in data.columns and not data['US_10Y_Yield'].isna().all():
        data['TNX_Return'] = data['US_10Y_Yield'].pct_change()
        features.append('TNX_Return')

    data = data.dropna()
    X = data[features]
    y = data['Target']

    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_latest = X.iloc[[-1]]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # バックテスト検証
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

    # 最新足予測
    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    confidence = max(prob) * 100

    latest_price = data['Close'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    latest_atr = data['ATR'].iloc[-1]
    latest_adx = data['ADX'].iloc[-1]
    latest_time = data.index[-1].strftime('%Y-%m-%d %H:%M')

    # メトリクス表示
    st.divider()
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("現在レート", f"{latest_price:.3f}")
    m_col2.metric("RSI (14)", f"{latest_rsi:.1f}")
    m_col3.metric("ADX (トレンド強度)", f"{latest_adx:.1f}", "強トレンド" if latest_adx > 25 else "レンジ傾向")
    m_col4.metric("直近AI予測勝率", f"{win_rate:.1f}%", f"{correct_count}/{test_len} 回的中")
    m_col5.metric("データ日時", latest_time)

    # ==========================================
    # 6. AI判定結果 & 松井証券向け注文パラメータ UI
    # ==========================================
    st.subheader("🤖 AI判定結果 & エントリーパラメータ (松井証券連携用)")

    fmt = ".5f" if "USD" in selected_label and not "USD/JPY" in selected_label else ".3f"
    pip_unit = 0.0001 if "USD" in selected_label and not "USD/JPY" in selected_label else 0.01

    if pred == 1 and confidence >= 60 and latest_adx >= 18:
        entry_price = latest_price
        tp_price = entry_price + (latest_atr * tp_atr_mult)
        sl_price = entry_price - (latest_atr * sl_atr_mult)

        tp_pips = (tp_price - entry_price) / pip_unit
        sl_pips = (entry_price - sl_price) / pip_unit

        st.success(f"🟢 **買い (BUY)** （AI信頼度: {confidence:.1f}% / トレンド確認済）")

        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.metric("新規買い目安 (Entry)", f"{entry_price:{fmt}}")
            st.code(f"{entry_price:{fmt}}", language="text")
        with t_col2:
            st.metric("利確目標 (Take Profit)", f"{tp_price:{fmt}}", f"+{tp_pips:.1f} pips")
            st.code(f"{tp_price:{fmt}}", language="text")
        with t_col3:
            st.metric("損切り目安 (Stop Loss)", f"{sl_price:{fmt}}", f"-{sl_pips:.1f} pips")
            st.code(f"{sl_price:{fmt}}", language="text")

        st.caption("※グレーの枠内の数値をタップすると簡単にコピーできます。松井証券のOCO注文時などにご活用ください。")

        # 通知実行
        if enable_notify and discord_url:
            msg = f"【🟢 買いサイン点灯】\n通貨ペア: {selected_label}\n時間軸: {tf_label}\n現在値: {entry_price:{fmt}}\n利確目安: {tp_price:{fmt}}\n損切目安: {sl_price:{fmt}}"
            send_discord_notification(discord_url, msg)

    elif pred == 0 and confidence >= 60 and latest_adx >= 18:
        entry_price = latest_price
        tp_price = entry_price - (latest_atr * tp_atr_mult)
        sl_price = entry_price + (latest_atr * sl_atr_mult)

        tp_pips = (entry_price - tp_price) / pip_unit
        sl_pips = (sl_price - entry_price) / pip_unit

        st.error(f"🔴 **売り (SELL)** （AI信頼度: {confidence:.1f}% / トレンド確認済）")

        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.metric("新規売り目安 (Entry)", f"{entry_price:{fmt}}")
            st.code(f"{entry_price:{fmt}}", language="text")
        with t_col2:
            st.metric("利確目標 (Take Profit)", f"{tp_price:{fmt}}", f"-{tp_pips:.1f} pips")
            st.code(f"{tp_price:{fmt}}", language="text")
        with t_col3:
            st.metric("損切り目安 (Stop Loss)", f"{sl_price:{fmt}}", f"+{sl_pips:.1f} pips")
            st.code(f"{sl_price:{fmt}}", language="text")

        st.caption("※グレーの枠内の数値をタップすると簡単にコピーできます。松井証券のOCO注文時などにご活用ください。")

        # 通知実行
        if enable_notify and discord_url:
            msg = f"【🔴 売りサイン点灯】\n通貨ペア: {selected_label}\n時間軸: {tf_label}\n現在値: {entry_price:{fmt}}\n利確目安: {tp_price:{fmt}}\n損切目安: {sl_price:{fmt}}"
            send_discord_notification(discord_url, msg)

    else:
        st.warning(f"🟡 **様子見 (HOLD)** （AI信頼度: {confidence:.1f}%）")
        if latest_adx < 18:
            st.write("判定理由: ADXが低くレンジ相場（もみ合い）の傾向が強いため、騙しを避けるため静観を推奨します。")
        else:
            st.write("判定理由: 方向性が不鮮明、または信頼度が基準値（60%）に達していません。")

    st.divider()
    with st.expander("📊 テクニカル指標・学習データの詳細"):
        st.dataframe(data[features + ['ATR']].tail(10))

# --- 自動更新ループ処理 ---
if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
