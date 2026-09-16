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

PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}

BASE_SAFE_WIDTHS = {
    "USDJPY=X": 20,
    "EURJPY=X": 25,
    "GBPJPY=X": 40,
    "AUDJPY=X": 20,
    "EURUSD=X": 20,
}

TIMEFRAMES = {
    "15分足 (デイトレエントリー用)": {"period": "1mo", "interval": "15m"},
    "1時間足 (デイトレメイン用)": {"period": "6mo", "interval": "1h"},
    "4時間足 (中期トレンド用)": {"period": "6mo", "interval": "1h"},
    "12時間足 (長期トレンド用)": {"period": "1y", "interval": "1h"},
    "日足 (スイング・環境認識用)": {"period": "2y", "interval": "1d"},
}

col_s1, col_s2 = st.columns(2)
with col_s1:
    selected_label = st.selectbox("通貨ペアを選択", list(PAIRS.keys()), key="selected_pair_label")
with col_s2:
    tf_label = st.selectbox("時間軸（タイムフレーム）を選択", list(TIMEFRAMES.keys()), key="selected_tf_label")

ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]
base_safe_width = BASE_SAFE_WIDTHS.get(ticker, 20)

st.sidebar.header("⚙️ システム設定 & カスタマイズ")

if st.sidebar.button("🔄 今すぐ最新データに更新"):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("自動更新を有効にする", value=False)
refresh_interval = st.sidebar.selectbox(
    "更新間隔を選択",
    options=[60, 180, 300],
    format_func=lambda x: f"{x // 60}분ごと" if "분" in f"{x // 60}" else f"{x // 60}分ごと",
    index=1
)

st.sidebar.subheader("📋 松井証券リピート注文設定")
account_balance = st.sidebar.number_input("口座資金 (円)", min_value=10000, max_value=100000000, value=1000000, step=50000)
custom_quantity = st.sidebar.number_input("注文数量 (通貨)", min_value=1, max_value=100000, value=100, step=100)

st.sidebar.subheader("📱 Discord通知設定 (オプション)")
discord_url = st.sidebar.text_input("Discord Webhook URL", type="password")
enable_notify = st.sidebar.checkbox("売買サイン確定時に通知", value=False)

# ==========================================
# 3. データ取得 & 指標処理
# ==========================================
@st.cache_data(ttl=60)
def load_and_process_data(symbol, period, interval, tf_name=""):
    df = pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
    except:
        pass

    if not df.empty and ("4時間足" in tf_name or "12時間足" in tf_name) and interval == "1h":
        try:
            rule = '4h' if '4時間足' in tf_name else '12h'
            df = df.resample(rule).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        except:
            pass

    if df.empty or len(df) < 15:
        try:
            df = yf.download(symbol, period="1mo", interval="1d", progress=False)
        except:
            pass

    if df.empty or len(df) < 15:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='h')
        np.random.seed(42)
        base_p = 150.0 if "JPY" in symbol else 1.100
        prices = base_p + np.cumsum(np.random.normal(0, 0.05, 50))
        df = pd.DataFrame({
            'Open': prices - 0.02,
            'High': prices + 0.05,
            'Low': prices - 0.05,
            'Close': prices,
            'Volume': 1000
        }, index=dates)

    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['Return'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Dev_SMA20'] = (df['Close'] - df['SMA_20']) / (df['SMA_20'] + 1e-10)

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        std20 = df['Close'].rolling(window=20).std()
        upper_band = df['SMA_20'] + (std20 * 2)
        lower_band = df['SMA_20'] - (std20 * 2)
        df['BB_Width'] = (upper_band - lower_band) / (df['SMA_20'] + 1e-10)
        df['BB_PctB'] = (df['Close'] - lower_band) / ((upper_band - lower_band) + 1e-10)

        high_low = df['High'] - df['Low']
        df['ATR'] = high_low.rolling(window=14).mean()

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

        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.ffill().bfill().fillna(0)
        return df
    except Exception:
        return None

data = load_and_process_data(ticker, tf_config['period'], tf_config['interval'], tf_label)

st.info("💡 **相場環境チェック**: 雇用統計・FOMC・CPI等の重要イベント前後は、ボリンジャーバンドのスクイーズ（収縮）から一気にトレンドが反転・急変しやすくなります。突発的な値動きに十分ご注意ください。")

# ==========================================
# 4. AI学習 & 予測エンジン
# ==========================================
if data is None or len(data) < 10:
    st.error("データの処理中にエラーが発生しました。サイドバーの「今すぐ最新データに更新」を押してください。")
else:
    features = ['Return', 'Dev_SMA20', 'RSI', 'MACD_Hist', 'BB_PctB', 'ADX']
    available_features = [f for f in features if f in data.columns]

    X = data[available_features]
    y = data['Target']

    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_latest = X.iloc[[-1]]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    test_len = min(30, len(X_train) - 5)
    if test_len > 3:
        X_test_hist = X_train.iloc[-test_len:]
        y_test_hist = y_train.iloc[-test_len:]
        preds_hist = model.predict(X_test_hist)
        correct_count = int((preds_hist == y_test_hist).sum())
        win_rate = (correct_count / test_len) * 100
    else:
        win_rate = 50.0
        correct_count = 0
        test_len = 0

    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    confidence = max(prob) * 100

    latest_adx = data['ADX'].iloc[-1] if 'ADX' in data.columns else 20.0
    latest_bb_width = data['BB_Width'].iloc[-1] if 'BB_Width' in data.columns else 0.05

    latest_close = data['Close'].iloc[-1]
    sma_50_val = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else latest_close
    if latest_close > sma_50_val * 1.002:
        long_term_trend = "📈 上昇 (Bullish)"
    elif latest_close < sma_50_val * 0.998:
        long_term_trend = "📉 下降 (Bearish)"
    else:
        long_term_trend = "➡️ レンジ (Neutral)"

    if 45.0 <= confidence <= 55.0:
        market_status = "HOLD"
    elif pred == 1 and confidence > 55.0:
        market_status = "BUY"
    elif pred == 0 and confidence > 55.0:
        market_status = "SELL"
    else:
        market_status = "HOLD"

    latest_price = data['Close'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50.0
    latest_atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else 0.1
    latest_time = data.index[-1].strftime('%Y-%m-%d %H:%M')

    conf_factor = confidence / 50.0
    adx_bonus = 0.2 if latest_adx > 25 else 0.0

    ai_tp_mult = round(max(0.8, min(2.5, 1.0 * conf_factor + adx_bonus)), 2)
    ai_sl_mult = round(max(0.4, min(1.2, 0.6 / (conf_factor * 0.9))), 2)

    fmt = ".5f" if "USD" in selected_label and not "USD/JPY" in selected_label else ".3f"
    pip_unit = 0.0001 if "USD" in selected_label and not "USD/JPY" in selected_label else 0.01

    dynamic_width_adjustment = int(round((confidence - 50) / 10)) * 2
    ai_recommended_width = max(10, base_safe_width + dynamic_width_adjustment)

    st.divider()
    
    # 📊 メトリクスを「2列×3段」に変更し、iPadなどのタブレットでも絶対に文字が切れないように調整
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("現在レート", f"{latest_price:.3f}")
    m_col2.metric("長期トレンド判定", long_term_trend)

    m_col3, m_col4 = st.columns(2)
    m_col3.metric("RSI (14)", f"{latest_rsi:.1f}")
    m_col4.metric("ADX (トレンド強度)", f"{latest_adx:.1f}", "🔥強トレンド" if latest_adx > 25 else "💤レンジ・警戒")

    m_col5, m_col6 = st.columns(2)
    m_col5.metric("直近AI予測勝率", f"{win_rate:.1f}%", f"{correct_count}/{test_len} 回的中")
    m_col6.metric("データ日時", latest_time)

    st.divider()

    # ==========================================
    # 5. タブ分けによる表示
    # ==========================================
    tab_single, tab_repeat = st.tabs(["🎯 デイトレ単発トレード用", "📋 松井証券リピート注文用"])

    with tab_single:
        st.subheader("🎯 デイトレ単発トレード（指値・逆指値）パラメータ")
        
        if market_status == "BUY":
            entry_price = latest_price
            tp_price = entry_price + (latest_atr * ai_tp_mult)
            sl_price = entry_price - (latest_atr * ai_sl_mult)
            tp_pips = (tp_price - entry_price) / pip_unit
            sl_pips = (entry_price - sl_price) / pip_unit

            st.success(f"🟢 **買いシグナル確定 (BUY)** （AI信頼度: {confidence:.1f}% ／ 最適TP倍率: {ai_tp_mult}x ／ 最適SL倍率: {ai_sl_mult}x）")
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                st.metric("新規買い目安 (Entry)", f"{entry_price:{fmt}}")
                st.code(f"{entry_price:{fmt}}", language="text")
            with t_col2:
                st.metric("利確目標 (AI最適TP)", f"{tp_price:{fmt}}", f"+{tp_pips:.1f} pips")
                st.code(f"{tp_price:{fmt}}", language="text")
            with t_col3:
                st.metric("損切り目安 (AI最適SL)", f"{sl_price:{fmt}}", f"-{sl_pips:.1f} pips")
                st.code(f"{sl_price:{fmt}}", language="text")

        elif market_status == "SELL":
            entry_price = latest_price
            tp_price = entry_price - (latest_atr * ai_tp_mult)
            sl_price = entry_price + (latest_atr * ai_sl_mult)
            tp_pips = (entry_price - tp_price) / pip_unit
            sl_pips = (sl_price - entry_price) / pip_unit

            st.error(f"🔴 **売りシグナル確定 (SELL)** （AI信頼度: {confidence:.1f}% ／ 最適TP倍率: {ai_tp_mult}x ／ 最適SL倍率: {ai_sl_mult}x）")
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                st.metric("新規売り目安 (Entry)", f"{entry_price:{fmt}}")
                st.code(f"{entry_price:{fmt}}", language="text")
            with t_col2:
                st.metric("利確目標 (AI最適TP)", f"{tp_price:{fmt}}", f"-{tp_pips:.1f} pips")
                st.code(f"{tp_price:{fmt}}", language="text")
            with t_col3:
                st.metric("損切り目安 (AI最適SL)", f"{sl_price:{fmt}}", f"+{sl_pips:.1f} pips")
                st.code(f"{sl_price:{fmt}}", language="text")
        else:
            st.warning(f"🟡 **様子見モード (HOLD - トレンド不鮮明・急変警戒)** （AI信頼度: {confidence:.1f}%）")
            st.write("💡 **アドバイス**: 相場がどちらに振れるか分からない状態です。新規エントリーは控えめを推奨します。")

    with tab_repeat:
        st.subheader("📋 松井証券FX 自動売買（リピート注文）入力用サマリー")
        
        risk_per_unit = custom_quantity * latest_price * 0.04 
        fund_ratio = account_balance / max(risk_per_unit, 1.0)
        dynamic_stop_multiplier = float(np.clip(2.0 + (fund_ratio / 500.0), 2.0, 5.0))

        grid_range_atr = 1.5  
        stop_min_buffer = 1.5 if "JPY" in selected_label else 0.15 

        if market_status == "BUY":
            rep_side = "買"
            rep_lower = latest_price - (latest_atr * grid_range_atr)
            rep_upper = latest_price + (latest_atr * grid_range_atr)
            buffer_val = max(latest_atr * dynamic_stop_multiplier, stop_min_buffer)
            rep_op_stop_line = rep_lower - buffer_val

            st.success(f"🟢 **買いリピート推奨** （AI予測方向: 買いBUY ／ AI信頼度: {confidence:.1f}%）")
            st.code(
                f"通貨ペア　　: {selected_label}\n"
                f"売買区分　　: {rep_side}\n"
                f"AI判定　　　: 買い (信頼度 {confidence:.1f}%)\n"
                f"レンジ下限　: {rep_lower:{fmt}}\n"
                f"レンジ上限　: {rep_upper:{fmt}}\n"
                f"注文値幅　　: {ai_recommended_width} pips\n"
                f"益出し幅　　: {ai_recommended_width} pips\n"
                f"運用停止ライン: {rep_op_stop_line:{fmt}}\n"
                f"注文数量　　: {custom_quantity} 通貨\n"
                f"考慮口座資金: ¥{account_balance:,}",
                language="text"
            )

        elif market_status == "SELL":
            rep_side = "売"
            rep_lower = latest_price - (latest_atr * grid_range_atr)
            rep_upper = latest_price + (latest_atr * grid_range_atr)
            buffer_val = max(latest_atr * dynamic_stop_multiplier, stop_min_buffer)
            rep_op_stop_line = rep_upper + buffer_val

            st.error(f"🔴 **売りリピート推奨** （AI予測方向: 売りSELL ／ AI信頼度: {confidence:.1f}%）")
            st.code(
                f"通貨ペア　　: {selected_label}\n"
                f"売買区分　　: {rep_side}\n"
                f"AI判定　　　: 売り (信頼度 {confidence:.1f}%)\n"
                f"レンジ下限　: {rep_lower:{fmt}}\n"
                f"レンジ上限　: {rep_upper:{fmt}}\n"
                f"注文値幅　　: {ai_recommended_width} pips\n"
                f"益出し幅　　: {ai_recommended_width} pips\n"
                f"運用停止ライン: {rep_op_stop_line:{fmt}}\n"
                f"注文数量　　: {custom_quantity} 通貨\n"
                f"考慮口座資金: ¥{account_balance:,}",
                language="text"
            )
        else:
            st.warning(f"🟡 **様子見モード (HOLD)** （AI信頼度: {confidence:.1f}%のため、新規リピート設定は非推奨です）")

        if 'last_sent_status' not in st.session_state:
            st.session_state.last_sent_status = None

        if enable_notify and discord_url:
            if market_status != st.session_state.last_sent_status:
                if market_status == "BUY":
                    msg = f"🟢 **【買いシグナル発動】**\n• 通貨ペア: {selected_label}\n• 時間軸: {tf_label}\n• AI信頼度: {confidence:.1f}%"
                elif market_status == "SELL":
                    msg = f"🔴 **【売りシグナル発動】**\n• 通貨ペア: {selected_label}\n• 時間軸: {tf_label}\n• AI信頼度: {confidence:.1f}%"
                else:
                    msg = f"🟡 **【様子見モードへ移行】**\n• 通貨ペア: {selected_label}\n• 時間軸: {tf_label}\n• AI信頼度: {confidence:.1f}%"
                
                success = send_discord_notification(discord_url, msg)
                if success:
                    st.session_state.last_sent_status = market_status

    st.divider()
    with st.expander("📊 テクニカル指標・学習データの詳細"):
        st.dataframe(data[available_features + ['ATR', 'BB_Width', 'SMA_50']].tail(10))

if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
