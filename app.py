import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import time

st.set_page_config(page_title="プロ版 AI FXデイトレアナライザー Ultimate", layout="wide")

# ==========================================
# 1. 通知ヘルパー関数 (Discord)
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
# 2. メイン画面 & サイドバー設定
# ==========================================
st.title("⚡ Pro AI FX デイトレアナライザー (Ultimate Edition)")

st.sidebar.header("⚙️ システム設定 & カスタマイズ")

if st.sidebar.button("🔄 今すぐ最新データに更新"):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("自動更新を有効にする", value=False)
refresh_interval = st.sidebar.selectbox(
    "更新間隔を選択",
    options=[60, 180, 300],
    format_func=lambda x: f"{x // 60}分ごと",
    index=1
)

st.sidebar.subheader("🎯 ターゲット設定 (リスクリワード)")
tp_atr_mult = st.sidebar.slider("利確目標 (ATR倍率)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)
sl_atr_mult = st.sidebar.slider("損切り目安 (ATR倍率)", min_value=0.3, max_value=2.0, value=0.6, step=0.1)

st.sidebar.subheader("📋 松井証券リピート注文設定")
custom_order_width = st.sidebar.number_input("注文値幅 (pips)", min_value=5, max_value=200, value=30, step=5)
custom_profit_width = st.sidebar.number_input("益出し幅 (pips)", min_value=5, max_value=200, value=30, step=5)
custom_quantity = st.sidebar.number_input("注文数量 (通貨)", min_value=1, max_value=100000, value=100, step=100)

st.sidebar.subheader("📱 Discord通知設定 (オプション)")
discord_url = st.sidebar.text_input("Discord Webhook URL", type="password")
enable_notify = st.sidebar.checkbox("売買サイン確定時に通知", value=False)

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
# 3. データ取得 & 指標処理（堅牢化バージョン）
# ==========================================
@st.cache_data(ttl=60)
def load_and_process_data(symbol, period, interval):
    df = pd.DataFrame()
    
    # 複数回リトライしてデータを取得
    for attempt in range(3):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if not df.empty and len(df) >= 20:
                break
        except Exception:
            time.sleep(1)

    # 取得失敗時は日足へフォールバック
    if df.empty or len(df) < 20:
        try:
            df = yf.download(symbol, period="1y", interval="1d", progress=False)
        except Exception:
            return None

    if df.empty or len(df) < 20:
        return None

    try:
        # マルチインデックスの解除（yfinanceの仕様変更対策）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 必要なカラムの存在確認
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            return None

        # 1. リターンと移動平均乖離
        df['Return'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['Dev_SMA20'] = (df['Close'] - df['SMA_20']) / (df['SMA_20'] + 1e-10)

        # 2. RSI (14)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

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
        df['BB_PctB'] = (df['Close'] - lower_band) / ((upper_band - lower_band) + 1e-10)

        # 5. ATR
        high_low = df['High'] - df['Low']
        df['ATR'] = high_low.rolling(window=14).mean()

        # 6. ADX (トレンドの強さ)
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr = np.maximum(high_low, np.maximum((df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()))
        atr14 = pd.Series(tr).rolling(14).mean()

        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / (atr14 + 1e-10))
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / (atr14 + 1e-10))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        df['ADX'] = dx.rolling(14).mean()

        # 目的変数
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # 欠損値の穴埋め（データが全消えするのを防ぐため ffill を優先）
        df = df.ffill().bfill().dropna()
        return df if not df.empty else None
    except Exception as e:
        return None

data = load_and_process_data(ticker, tf_config['period'], tf_config['interval'])

st.info("💡 **トレード前のチェック**: 雇用統計やFOMCなど主要経済指標の発表前後はテクニカル分析が不向きになります。重要指標発表直前のエントリーは控えましょう。")

# ==========================================
# 4. AI学習 & 予測エンジン
# ==========================================
if data is None or len(data) < 20:
    st.error("データの取得に失敗したか、データ数が不足しています。しばらく待ってからサイドバーの『今すぐ最新データに更新』ボタンを押してください。")
else:
    features = ['Return', 'Dev_SMA20', 'RSI', 'MACD_Hist', 'BB_PctB', 'ADX']
    
    # 確実に存在する特徴量だけに絞る
    available_features = [f for f in features if f in data.columns]

    X = data[available_features]
    y = data['Target']

    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_latest = X.iloc[[-1]]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    test_len = min(30, len(X_train) - 10)
    if test_len > 5:
        X_test_hist = X_train.iloc[-test_len:]
        y_test_hist = y_train.iloc[-test_len:]
        preds_hist = model.predict(X_test_hist)
        correct_count = int((preds_hist == y_test_hist).sum())
        win_rate = (correct_count / test_len) * 100
    else:
        win_rate = 0.0
        correct_count = 0
        test_len = 0

    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    confidence = max(prob) * 100

    latest_price = data['Close'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    latest_atr = data['ATR'].iloc[-1]
    latest_adx = data['ADX'].iloc[-1] if 'ADX' in data.columns else 20.0
    latest_time = data.index[-1].strftime('%Y-%m-%d %H:%M')

    st.divider()
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("現在レート", f"{latest_price:.3f}")
    m_col2.metric("RSI (14)", f"{latest_rsi:.1f}")
    m_col3.metric("ADX (トレンド強度)", f"{latest_adx:.1f}", "強トレンド" if latest_adx > 25 else "レンジ傾向")
    m_col4.metric("直近AI予測勝率", f"{win_rate:.1f}%", f"{correct_count}/{test_len} 回的中")
    m_col5.metric("データ日時", latest_time)

    # ==========================================
    # 5. AI判定結果 & 松井証券向け注文パラメータ UI
    # ==========================================
    st.subheader("🤖 AI判定結果 & エントリーパラメータ (松井証券連携用)")

    fmt = ".5f" if "USD" in selected_label and not "USD/JPY" in selected_label else ".3f"
    pip_unit = 0.0001 if "USD" in selected_label and not "USD/JPY" in selected_label else 0.01

    if pred == 1 and confidence >= 50:
        entry_price = latest_price
        tp_price = entry_price + (latest_atr * tp_atr_mult)
        sl_price = entry_price - (latest_atr * sl_atr_mult)

        tp_pips = (tp_price - entry_price) / pip_unit
        sl_pips = (entry_price - sl_price) / pip_unit
        op_stop_line = sl_price - 0.400

        st.success(f"🟢 **買い (BUY)** （AI信頼度: {confidence:.1f}%）")

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

        st.markdown("### 📋 松井証券FX 自動売買（リピート注文）入力用サマリー")
        st.code(
            f"通貨ペア　　: {selected_label}\n"
            f"売買区分　　: 買\n"
            f"レンジ下限　: {sl_price:{fmt}}\n"
            f"レンジ上限　: {tp_price:{fmt}}\n"
            f"注文値幅　　: {custom_order_width} pips\n"
            f"益出し幅　　: {custom_profit_width} pips\n"
            f"運用停止ライン: {op_stop_line:{fmt}}\n"
            f"注文数量　　: {custom_quantity} 通貨",
            language="text"
        )
        st.caption("※上のグレーの枠内をそのままコピーして松井証券アプリの入力にお使いいただけます。")

        if enable_notify and discord_url:
            msg = f"【🟢 買いサイン点灯】\n通貨ペア: {selected_label}\n時間軸: {tf_label}\n現在値: {entry_price:{fmt}}\n利確目安: {tp_price:{fmt}}\n損切目安: {sl_price:{fmt}}"
            send_discord_notification(discord_url, msg)

    elif pred == 0 and confidence >= 50:
        entry_price = latest_price
        tp_price = entry_price - (latest_atr * tp_atr_mult)
        sl_price = entry_price + (latest_atr * sl_atr_mult)

        tp_pips = (entry_price - tp_price) / pip_unit
        sl_pips = (sl_price - entry_price) / pip_unit
        op_stop_line = sl_price + 0.400

        st.error(f"🔴 **売り (SELL)** （AI信頼度: {confidence:.1f}%）")

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

        st.markdown("### 📋 松井証券FX 自動売買（リピート注文）入力用サマリー")
        st.code(
            f"通貨ペア　　: {selected_label}\n"
            f"売買区分　　: 売\n"
            f"レンジ下限　: {tp_price:{fmt}}\n"
            f"レンジ上限　: {sl_price:{fmt}}\n"
            f"注文値幅　　: {custom_order_width} pips\n"
            f"益出し幅　　: {custom_profit_width} pips\n"
            f"運用停止ライン: {op_stop_line:{fmt}}\n"
            f"注文数量　　: {custom_quantity} 通貨",
            language="text"
        )
        st.caption("※上のグレーの枠内をそのままコピーして松井証券アプリの入力にお使いいただけます。")

        if enable_notify and discord_url:
            msg = f"【🔴 売りサイン点灯】\n通貨ペア: {selected_label}\n時間軸: {tf_label}\n現在値: {entry_price:{fmt}}\n利確目安: {tp_price:{fmt}}\n損切目安: {sl_price:{fmt}}"
            send_discord_notification(discord_url, msg)

    else:
        st.warning(f"🟡 **様子見 (HOLD)** （AI信頼度: {confidence:.1f}%）")
        st.write("判定理由: 方向性が不鮮明、または信頼度が基準値に達していません。")

    st.divider()
    with st.expander("📊 テクニカル指標・学習データの詳細"):
        st.dataframe(data[available_features + ['ATR']].tail(10))

if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
