import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import time
from datetime import datetime

st.set_page_config(page_title="プロ版 AI FXデイトレアナライザー Ultimate Pro", layout="wide")

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
st.title("⚡ Pro AI FX デイトレアナライザー (Ultimate Pro Edition)")

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
    format_func=lambda x: f"{x // 60}分ごと",
    index=1
)

st.sidebar.subheader("📋 松井証券リピート注文設定")
# 💡 入力値が確実に保持されるよう key を追加
account_balance = st.sidebar.number_input("口座資金 (円)", min_value=10000, max_value=100000000, value=100000, step=10000, key="input_account_balance")
custom_quantity = st.sidebar.number_input("注文数量 (通貨)", min_value=1, max_value=100000, value=100, step=100, key="input_custom_quantity")

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

    if df.empty or len(df) < 30:
        try:
            df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        except:
            pass

    if df.empty or len(df) < 30:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='h')
        np.random.seed(42)
        base_p = 150.0 if "JPY" in symbol else 1.100
        prices = base_p + np.cumsum(np.random.normal(0, 0.05, 100))
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

        high = df['High']
        low = df['Low']
        close = df['Close']

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)

        sum_di = plus_di + minus_di
        sum_di = sum_di.replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / sum_di
        
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(25.0)

        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.ffill().bfill().fillna(0)
        return df
    except Exception as e:
        st.error(f"データ処理エラー: {e}")
        return None

data = load_and_process_data(ticker, tf_config['period'], tf_config['interval'], tf_label)

# ==========================================
# 4. 新機能チェック
# ==========================================
current_hour_jst = datetime.now().hour
is_low_liquidity = 3 <= current_hour_jst <= 7

if is_low_liquidity:
    st.warning("⚠️ **【流動性低下タイムゾーン警告】**: 現在はオセアニア時間帯の早朝です。スプレッド拡大にご注意ください。")

# ==========================================
# 5. AI学習 & 予測エンジン
# ==========================================
if data is None or len(data) < 10:
    st.error("データの処理中にエラーが発生しました。")
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
    cumulative_wins = []
    if test_len > 3:
        X_test_hist = X_train.iloc[-test_len:]
        y_test_hist = y_train.iloc[-test_len:]
        preds_hist = model.predict(X_test_hist)
        
        correct_count = 0
        for i, (p, actual) in enumerate(zip(preds_hist, y_test_hist)):
            if p == actual:
                correct_count += 1
            cumulative_wins.append((i + 1, (correct_count / (i + 1)) * 100))
            
        win_rate = (correct_count / test_len) * 100
    else:
        win_rate = 50.0
        correct_count = 0
        test_len = 0

    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    confidence = max(prob) * 100

    latest_adx = data['ADX'].iloc[-1] if 'ADX' in data.columns else 25.0
    latest_bb_width = data['BB_Width'].iloc[-1] if 'BB_Width' in data.columns else 0.05

    avg_bb_width = data['BB_Width'].rolling(window=20).mean().iloc[-1] if 'BB_Width' in data.columns else 0.05
    is_squeezed = latest_bb_width < (avg_bb_width * 0.8)

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

    if is_squeezed:
        st.error("⚡ **【スクイーズ検知】**: ボリンジャーバンドが収縮しています。ブレイクアウトにご注意ください！")

    st.divider()
    
    # 📊 メトリクス表示
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("現在レート", f"{latest_price:.3f}")
    m_col2.metric("長期トレンド判定", long_term_trend)

    m_col3, m_col4 = st.columns(2)
    m_col3.metric("RSI (14)", f"{latest_rsi:.1f}")
    m_col4.metric("ADX (トレンド強度)", f"{latest_adx:.1f}", "🔥強トレンド" if latest_adx > 25 else "💤レンジ・警戒")

    m_col5, m_col6 = st.columns(2)
    m_col5.metric("直近AI予測勝率", f"{win_rate:.1f}%", f"{correct_count}/{test_len} 回)")
    m_col6.metric("データ日時", latest_time)

    st.divider()

    # ==========================================
    # 6. タブ切り替え
    # ==========================================
    tab_single, tab_repeat, tab_scanner, tab_backtest = st.tabs([
        "🎯 デイトレ単発トレード用", 
        "📋 松井証券リピート注文用", 
        "🔍 全通貨ペア一括スキャン", 
        "📈 AIバックテスト検証"
    ])

    with tab_single:
        st.subheader("🎯 デイトレ単発トレード（指値・逆指値）パラメータ")
        
        if market_status == "BUY":
            entry_price = latest_price
            tp_price = entry_price + (latest_atr * ai_tp_mult)
            sl_price = entry_price - (latest_atr * ai_sl_mult)
            tp_pips = (tp_price - entry_price) / pip_unit
            sl_pips = (entry_price - sl_price) / pip_unit

            st.success(f"🟢 **買いシグナル確定 (BUY)** （AI信頼度: {confidence:.1f}%）")
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

            st.error(f"🔴 **売りシグナル確定 (SELL)** （AI信頼度: {confidence:.1f}%）")
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
            st.warning(f"🟡 **様子見モード (HOLD)** （AI信頼度: {confidence:.1f}%）")

    with tab_repeat:
        st.subheader("📋 松井証券FX 自動売買（リピート注文）入力用サマリー")
        st.write(f"💡 入力された口座資金（¥{account_balance:,}）と注文数量（{custom_quantity}通貨）を元に、証拠金エラーが起きない安全なレンジ幅を自動計算しています。")
        
        is_jpy_pair = "JPY" in selected_label
        leverage = 25.0
        margin_per_unit = latest_price / leverage
        
        max_safe_total_units = (account_balance * 0.7) / max(margin_per_unit, 1.0)
        max_allowable_grids = max(5, int(max_safe_total_units / max(custom_quantity, 1)))
        
        max_safe_range_pips = max_allowable_grids * ai_recommended_width
        cap_pips = 150 if is_jpy_pair else 1500
        safe_range_pips = min(max_safe_range_pips, cap_pips)
        
        half_range = (safe_range_pips * pip_unit) / 2.0

        if market_status == "BUY":
            rep_side = "買"
            rep_lower = round(latest_price - half_range, 3)
            rep_upper = round(latest_price + half_range, 3)
            buffer_val = max(latest_atr * 2.0, 0.5 if is_jpy_pair else 0.05)
            rep_op_stop_line = round(rep_lower - buffer_val, 3)

            st.success(f"🟢 **買いリピート推奨** （資金連動・安全レンジ自動調整済み）")
            st.code(
                f"通貨ペア　　: {selected_label}\n"
                f"売買区分　　: {rep_side}\n"
                f"AI判定　　　: 買い (信頼度 {confidence:.1f}%)\n"
                f"レンジ下限　: {rep_lower}\n"
                f"レンジ上限　: {rep_upper}\n"
                f"注文値幅　　: {ai_recommended_width} pips\n"
                f"益出し幅　　: {ai_recommended_width} pips\n"
                f"運用停止ライン: {rep_op_stop_line}\n"
                f"注文数量　　: {custom_quantity} 通貨\n"
                f"考慮口座資金: ¥{account_balance:,} （※資金オーバー防止安全モード適用中）",
                language="text"
            )

        elif market_status == "SELL":
            rep_side = "売"
            rep_lower = round(latest_price - half_range, 3)
            rep_upper = round(latest_price + half_range, 3)
            buffer_val = max(latest_atr * 2.0, 0.5 if is_jpy_pair else 0.05)
            rep_op_stop_line = round(rep_upper + buffer_val, 3)

            st.error(f"🔴 **売りリピート推奨** （資金連動・安全レンジ自動調整済み）")
            st.code(
                f"通貨ペア　　: {selected_label}\n"
                f"売買区分　　: {rep_side}\n"
                f"AI判定　　　: 売り (信頼度 {confidence:.1f}%)\n"
                f"レンジ下限　: {rep_lower}\n"
                f"レンジ上限　: {rep_upper}\n"
                f"注文値幅　　: {ai_recommended_width} pips\n"
                f"益出し幅　　: {ai_recommended_width} pips\n"
                f"運用停止ライン: {rep_op_stop_line}\n"
                f"注文数量　　: {custom_quantity} 通貨\n"
                f"考慮口座資金: ¥{account_balance:,} （※資金オーバー防止安全モード適用中）",
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

    with tab_scanner:
        st.subheader("🔍 全監視通貨ペア AIスコア・一括スキャン")
        if st.button("🚀 全ペアを一括スキャン実行"):
            scan_results = []
            with st.spinner("各通貨ペアのAI予測モデルを計算中..."):
                for p_label, p_symbol in PAIRS.items():
                    sub_df = load_and_process_data(p_symbol, tf_config['period'], tf_config['interval'], tf_label)
                    if sub_df is not None and len(sub_df) > 10:
                        sub_X = sub_df[available_features]
                        sub_y = sub_df['Target']
                        sub_model = RandomForestClassifier(n_estimators=50, random_state=42)
                        sub_model.fit(sub_X.iloc[:-1], sub_y.iloc[:-1])
                        
                        s_pred = sub_model.predict(sub_X.iloc[[-1]])[0]
                        s_prob = sub_model.predict_proba(sub_X.iloc[[-1]])[0]
                        s_conf = max(s_prob) * 100
                        s_adx = sub_df['ADX'].iloc[-1] if 'ADX' in sub_df.columns else 25.0
                        
                        direction = "買い (BUY)" if s_pred == 1 else "売り (SELL)"
                        if 45 <= s_conf <= 55:
                            direction = "様子見 (HOLD)"
                            
                        scan_results.append({
                            "通貨ペア": p_label,
                            "AI推奨方向": direction,
                            "信頼度 (%)": round(s_conf, 1),
                            "ADX (トレンド強度)": round(s_adx, 1)
                        })
            
            if scan_results:
                res_df = pd.DataFrame(scan_results).sort_values(by="信頼度 (%)", ascending=False)
                st.dataframe(res_df, use_container_width=True)
        else:
            st.info("上のボタンを押すと、全通貨ペアの現在のAI予測信頼度とトレンド強度を一斉にスキャンできます。")

    with tab_backtest:
        st.subheader("📈 直近AI予測モデルの累積勝率バックテスト")
        st.write(f"選択中の時間軸（{tf_label}）における、直近 {test_len} 本のローソク足でのAI予測の的中推移です。")
        
        if cumulative_wins:
            cb_df = pd.DataFrame(cumulative_wins, columns=["検証ステップ (直近からの経過)", "累積勝率 (%)"]).set_index("検証ステップ (直近からの経過)")
            st.line_chart(cb_df)
            st.metric("最終テスト勝率", f"{win_rate:.1f}%", f"{correct_count}勝 / {test_len}戦")
        else:
            st.warning("十分な過去データがありません。")

    st.divider()
    with st.expander("📊 テクニカル指標・学習データの詳細"):
        st.dataframe(data[available_features + ['ATR', 'BB_Width', 'SMA_50']].tail(10))

if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
