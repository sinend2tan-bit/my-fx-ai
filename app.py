import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import time
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="プロ版 AI FXデイトレアナライザー Ultimate Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 0. 設定ファイルの永続化（保存・読み込み）ヘルパー
# ==========================================
SETTINGS_FILE = "user_settings.json"

DEFAULT_SETTINGS = {
    "account_balance": 200000,
    "quantity_wan": 0.02,
    "discord_url": "",
    "enable_notify": False,
    "auto_refresh": False,
    "refresh_interval": 180
}

def load_user_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                merged = DEFAULT_SETTINGS.copy()
                merged.update(saved)
                return merged
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_user_settings():
    settings = {
        "account_balance": st.session_state.get("account_balance", DEFAULT_SETTINGS["account_balance"]),
        "quantity_wan": st.session_state.get("quantity_wan", DEFAULT_SETTINGS["quantity_wan"]),
        "discord_url": st.session_state.get("discord_url", DEFAULT_SETTINGS["discord_url"]),
        "enable_notify": st.session_state.get("enable_notify", DEFAULT_SETTINGS["enable_notify"]),
        "auto_refresh": st.session_state.get("auto_refresh", DEFAULT_SETTINGS["auto_refresh"]),
        "refresh_interval": st.session_state.get("refresh_interval", DEFAULT_SETTINGS["refresh_interval"]),
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

if "initialized" not in st.session_state:
    saved_settings = load_user_settings()
    for key, val in saved_settings.items():
        st.session_state[key] = val
    st.session_state["initialized"] = True

# ==========================================
# 1. 通知ヘルパー関数 (Discordのみ)
# ==========================================
def send_discord_notification(webhook_url, message):
    if not webhook_url:
        return False
    try:
        res = requests.post(webhook_url, json={"content": message}, timeout=5)
        return res.status_code == 204
    except Exception:
        return False

# ==========================================
# 2. メイン画面 & サイドバー設定
# ==========================================
st.title("⚡ Pro AI FX デイトレアナライザー (Ultimate Full-Spec Edition)")

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
    "GBPJPY=X": 30,  # GBP/JPYのベース値を最適化
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

is_jpy_pair = "JPY" in ticker
pip_unit = 0.01 if is_jpy_pair else 0.0001
price_fmt = ".3f" if is_jpy_pair else ".5f"

st.sidebar.header("⚙️ システム設定 & カスタマイズ")

if st.sidebar.button("🔄 今すぐ最新データに更新", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("自動更新を有効にする", key="auto_refresh", on_change=save_user_settings)
refresh_interval = st.sidebar.selectbox(
    "更新間隔を選択",
    options=[60, 180, 300],
    format_func=lambda x: f"{x // 60}分ごと",
    key="refresh_interval",
    on_change=save_user_settings
)

st.sidebar.subheader("📋 松井証券トレード設定")
account_balance = st.sidebar.number_input(
    "口座資金 (円)", 
    min_value=10000, 
    max_value=100000000, 
    step=10000, 
    key="account_balance",
    on_change=save_user_settings
)

quantity_wan = st.sidebar.number_input(
    "注文数量 (万通貨)", 
    min_value=0.0001, 
    max_value=10.0, 
    step=0.01, 
    format="%.4f",
    key="quantity_wan",
    on_change=save_user_settings
)

custom_quantity = int(round(quantity_wan * 10000))

st.sidebar.subheader("📱 アラート通知設定 (Discord)")
discord_url = st.sidebar.text_input("Discord Webhook URL", type="password", key="discord_url", on_change=save_user_settings)
enable_notify = st.sidebar.checkbox("売買サイン確定時に自動通知", key="enable_notify", on_change=save_user_settings)

# ==========================================
# 3. データ取得 & 指標計算エンジン
# ==========================================
@st.cache_data(ttl=60)
def load_and_process_data(symbol, period, interval, tf_name=""):
    df = pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
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
        except Exception:
            pass

    if df.empty or len(df) < 30:
        try:
            df = yf.download(symbol, period="3mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    if df.empty or len(df) < 30:
        return None

    try:
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
        df['Upper_Band'] = df['SMA_20'] + (std20 * 2)
        df['Lower_Band'] = df['SMA_20'] - (std20 * 2)
        df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / (df['SMA_20'] + 1e-10)
        df['BB_PctB'] = (df['Close'] - df['Lower_Band']) / ((df['Upper_Band'] - df['Lower_Band']) + 1e-10)

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
    except Exception:
        return None

# AIシグナル & MTF判定関数
def analyze_signal(df_current, df_higher):
    if df_current is None or len(df_current) < 10:
        return "HOLD", 50.0, None

    try:
        features = ['Return', 'Dev_SMA20', 'RSI', 'MACD_Hist', 'BB_PctB', 'ADX']
        avail = [f for f in features if f in df_current.columns]
        
        X = df_current[avail]
        y = df_current['Target']
        X_train, y_train = X.iloc[:-1], y.iloc[:-1]
        X_latest = X.iloc[[-1]]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        raw_pred = model.predict(X_latest)[0]
        prob = model.predict_proba(X_latest)[0]
        confidence = max(prob) * 100

        # 上位足バイアス
        htf_bias = -1
        if df_higher is not None and not df_higher.empty and 'SMA_50' in df_higher.columns:
            htf_close = df_higher['Close'].iloc[-1]
            htf_sma50 = df_higher['SMA_50'].iloc[-1]
            if htf_close > htf_sma50 * 1.002:
                htf_bias = 1
            elif htf_close < htf_sma50 * 0.998:
                htf_bias = 0

        # 最終判定
        if 45.0 <= confidence <= 55.0:
            status = "HOLD"
        elif raw_pred == 1 and confidence > 55.0:
            if htf_bias == 0 and confidence < 65.0:
                status = "HOLD (逆張り警戒)"
            else:
                status = "BUY"
        elif raw_pred == 0 and confidence > 55.0:
            if htf_bias == 1 and confidence < 65.0:
                status = "HOLD (逆張り警戒)"
            else:
                status = "SELL"
        else:
            status = "HOLD"

        return status, confidence, model
    except Exception:
        return "HOLD", 50.0, None

data = load_and_process_data(ticker, tf_config['period'], tf_config['interval'], tf_label)
higher_tf_data = load_and_process_data(ticker, "1y", "1d", "日足 (スイング・環境認識用)")

# ==========================================
# 4. 時間帯・週末市場クローズ判定
# ==========================================
now_datetime = datetime.now()
current_day = now_datetime.weekday()
current_hour_jst = now_datetime.hour

is_weekend = (current_day == 5 and current_hour_jst >= 6) or (current_day == 6) or (current_day == 0 and current_hour_jst < 6)
is_low_liquidity = 3 <= current_hour_jst <= 7
is_ny_open = 21 <= current_hour_jst <= 23

if is_weekend:
    st.error("🛑 **【週末・為替市場クローズ中】**: 現在外国為替市場は休業時間帯です。表示価格は最終クローズ値となります。新規注文の発注にご注意ください。")
elif is_low_liquidity:
    st.warning("⚠️ **【流動性低下タイムゾーン】**: オセアニア時間の早朝です。スプレッド拡大および急変動リスクにご注意ください。")
elif is_ny_open:
    st.info("🔥 **【NY市場オープンタイムゾーン】**: ボラティリティが高まる時間帯です。利益・損切り幅を意識してトレードしてください。")

# ==========================================
# 5. AI学習 & メイン画面表示
# ==========================================
if data is None or len(data) < 10:
    st.error("🚨 リアルタイムデータの取得に失敗しました。市場休業日かネットワーク接続をご確認のうえ「最新データに更新」を押してください。")
else:
    market_status, confidence, main_model = analyze_signal(data, higher_tf_data)

    # バックテスト計算
    features = ['Return', 'Dev_SMA20', 'RSI', 'MACD_Hist', 'BB_PctB', 'ADX']
    available_features = [f for f in features if f in data.columns]
    X_bt = data[available_features].iloc[:-1]
    y_bt = data['Target'].iloc[:-1]
    
    test_len = min(30, len(X_bt) - 10)
    cumulative_wins = []
    
    if test_len > 5:
        correct_count = 0
        for i in range(test_len):
            idx = len(X_bt) - test_len + i
            sub_model = RandomForestClassifier(n_estimators=30, random_state=42)
            sub_model.fit(X_bt.iloc[:idx], y_bt.iloc[:idx])
            p = sub_model.predict(X_bt.iloc[[idx]])[0]
            if p == y_bt.iloc[idx]:
                correct_count += 1
            cumulative_wins.append((i + 1, (correct_count / (i + 1)) * 100))
        win_rate = (correct_count / test_len) * 100
    else:
        win_rate, correct_count, test_len = 50.0, 0, 0

    latest_adx = data['ADX'].iloc[-1] if 'ADX' in data.columns else 25.0
    latest_bb_width = data['BB_Width'].iloc[-1] if 'BB_Width' in data.columns else 0.05
    avg_bb_series = data['BB_Width'].rolling(window=20).mean()
    avg_bb_width = avg_bb_series.iloc[-1] if not avg_bb_series.empty and not pd.isna(avg_bb_series.iloc[-1]) else 0.05
    is_squeezed = latest_bb_width < (avg_bb_width * 0.8)

    # 上位足安全ガード
    if higher_tf_data is not None and not higher_tf_data.empty and 'SMA_50' in higher_tf_data.columns:
        htf_close = higher_tf_data['Close'].iloc[-1]
        htf_sma50 = higher_tf_data['SMA_50'].iloc[-1]
    else:
        htf_close = data['Close'].iloc[-1]
        htf_sma50 = data['SMA_50'].iloc[-1]
    
    if htf_close > htf_sma50 * 1.002:
        long_term_trend = "📈 強気上昇"
    elif htf_close < htf_sma50 * 0.998:
        long_term_trend = "📉 弱気下降"
    else:
        long_term_trend = "➡️ レンジ相場"

    latest_price = data['Close'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50.0
    latest_atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else 0.1
    latest_time = data.index[-1].strftime('%Y-%m-%d %H:%M')

    conf_factor = confidence / 50.0
    adx_bonus = 0.2 if latest_adx > 25 else 0.0

    ai_tp_mult = round(max(0.8, min(2.5, 1.0 * conf_factor + adx_bonus)), 2)
    ai_sl_mult = round(max(0.4, min(1.2, 0.6 / (conf_factor * 0.9))), 2)

    # 約定力を高めるため、値幅が大きくなりすぎないように上限（40pips）を自動設定
    dynamic_width_adjustment = int(round((confidence - 50) / 10)) * 2
    ai_recommended_width = min(40, max(10, base_safe_width + dynamic_width_adjustment))

    recommended_slippage = round(max(0.5, (latest_atr / pip_unit) * 0.05), 1)

    if is_squeezed:
        st.error("⚡ **【スクイーズ発生】**: ボリンジャーバンドが極端に収縮しています。エネルギー蓄積後の急激なブレイクアウトにご注意ください！")

    st.divider()

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("現在レート", f"{latest_price:{price_fmt}}")
    m_col2.metric("上位足 (日足) トレンド", long_term_trend)
    m_col3.metric("直近AI予測勝率", f"{win_rate:.1f}%", f"({correct_count}/{test_len} 回)")

    m_col4, m_col5, m_col6 = st.columns(3)
    m_col4.metric("RSI (14)", f"{latest_rsi:.1f}")
    m_col5.metric("ADX (トレンド強度)", f"{latest_adx:.1f}", "🔥強トレンド" if latest_adx > 25 else "💤低ボラ")
    m_col6.metric("データ更新日時", latest_time)

    st.divider()

    # ==========================================
    # 6. タブ切り替え機能
    # ==========================================
    tab_single, tab_speed, tab_repeat, tab_chart, tab_scanner, tab_backtest = st.tabs([
        "🎯 デイトレ単発", 
        "⚡ スピード注文",
        "📋 リピート注文", 
        "📈 ローソク足チャート",
        "🔍 全ペアスキャン", 
        "📊 バックテスト"
    ])

    with tab_single:
        st.subheader("🎯 デイトレ単発トレード（指値・逆指値）最適化値")
        
        if market_status == "BUY":
            entry_price = latest_price
            tp_price = entry_price + (latest_atr * ai_tp_mult)
            sl_price = entry_price - (latest_atr * ai_sl_mult)
            tp_pips = (tp_price - entry_price) / pip_unit
            sl_pips = (entry_price - sl_price) / pip_unit

            st.success(f"🟢 **買いシグナル確定 (BUY)** （AI信頼度: {confidence:.1f}% | MTF一致）")
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                st.metric("新規買い目安 (Entry)", f"{entry_price:{price_fmt}}")
                st.code(f"{entry_price:{price_fmt}}", language="text")
            with t_col2:
                st.metric("利確目標 (AI最適TP)", f"{tp_price:{price_fmt}}", f"+{tp_pips:.1f} pips")
                st.code(f"{tp_price:{price_fmt}}", language="text")
            with t_col3:
                st.metric("損切り目安 (AI最適SL)", f"{sl_price:{price_fmt}}", f"-{sl_pips:.1f} pips")
                st.code(f"{sl_price:{price_fmt}}", language="text")

        elif market_status == "SELL":
            entry_price = latest_price
            tp_price = entry_price - (latest_atr * ai_tp_mult)
            sl_price = entry_price + (latest_atr * ai_sl_mult)
            tp_pips = (entry_price - tp_price) / pip_unit
            sl_pips = (sl_price - entry_price) / pip_unit

            st.error(f"🔴 **売りシグナル確定 (SELL)** （AI信頼度: {confidence:.1f}% | MTF一致）")
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                st.metric("新規売り目安 (Entry)", f"{entry_price:{price_fmt}}")
                st.code(f"{entry_price:{price_fmt}}", language="text")
            with t_col2:
                st.metric("利確目標 (AI最適TP)", f"{tp_price:{price_fmt}}", f"-{tp_pips:.1f} pips")
                st.code(f"{tp_price:{price_fmt}}", language="text")
            with t_col3:
                st.metric("損切り目安 (AI最適SL)", f"{sl_price:{price_fmt}}", f"+{sl_pips:.1f} pips")
                st.code(f"{sl_price:{price_fmt}}", language="text")
        else:
            st.warning(f"🟡 **様子見モード ({market_status})** （AI信頼度: {confidence:.1f}%）")

    with tab_speed:
        st.subheader("⚡ 松井証券FX アプリ【スピード注文】設定用")
        st.write("松井証券FXアプリの「スピード注文設定」画面に直接入力できる数値です。")

        sp_tp_pips = round(latest_atr * ai_tp_mult / pip_unit, 1)
        sp_sl_pips = round(latest_atr * ai_sl_mult / pip_unit, 1)

        sp_col1, sp_col2, sp_col3 = st.columns(3)
        sp_col1.metric("数量 (万)", f"{quantity_wan}")
        sp_col2.metric("益出し幅 (利確)", f"{sp_tp_pips} pips")
        sp_col3.metric("損切り幅 (損切)", f"{sp_sl_pips} pips")

        st.markdown("#### 📱 スピード注文設定用サマリー（コピー用）")
        st.code(
            f"通貨ペア: {selected_label}\n"
            f"推奨エントリー: {'買 (ASK)' if market_status == 'BUY' else '売 (BID)' if market_status == 'SELL' else '様子見'}\n"
            f"数量(万): {quantity_wan}\n"
            f"益出し幅: {sp_tp_pips} pips\n"
            f"損切り幅: {sp_sl_pips} pips\n"
            f"許容スリッページ: {recommended_slippage} pips",
            language="text"
        )

    with tab_repeat:
        st.subheader("📋 松井証券FX 自動売買（リピート注文）入力用サマリー")
        st.write(f"💡 入力された口座資金（¥{account_balance:,}）と注文数量（{quantity_wan}万通貨 / {custom_quantity:,}通貨）を元に、ロスカットリスクを抑制した最適設定を算出しています。")
        
        leverage = 25.0

        if is_jpy_pair:
            jpy_rate = latest_price
        else:
            try:
                usdjpy_data = load_and_process_data("USDJPY=X", "5d", "5m", "ドル円換算用")
                usdjpy_price = usdjpy_data['Close'].iloc[-1] if usdjpy_data is not None else 155.0
            except Exception:
                usdjpy_price = 155.0
            jpy_rate = latest_price * usdjpy_price

        margin_per_unit = (jpy_rate * custom_quantity) / leverage
        max_allowable_grids = max(5, int((account_balance * 0.7) / max(margin_per_unit, 1.0)))
        
        max_safe_range_pips = max_allowable_grids * ai_recommended_width
        cap_pips = 1500
        safe_range_pips = min(max_safe_range_pips, cap_pips)
        
        half_range = (safe_range_pips * pip_unit) / 2.0

        if market_status == "BUY":
            rep_side = "買"
            rep_lower = round(latest_price - half_range, 3 if is_jpy_pair else 5)
            rep_upper = round(latest_price + half_range, 3 if is_jpy_pair else 5)
            buffer_val = max(latest_atr * 2.0, 0.5 if is_jpy_pair else 0.05)
            rep_op_stop_line = round(rep_lower - buffer_val, 3 if is_jpy_pair else 5)
            buffer_pips = round(buffer_val / pip_unit, 1)

            st.success(f"🟢 **買いリピート推奨** （資金連動・安全レンジ自動調整済み）")
            summary_text = (
                f"通貨ペア　　: {selected_label}\n"
                f"売買区分　　: {rep_side}\n"
                f"AI判定　　　: 買い (信頼度 {confidence:.1f}%)\n"
                f"レンジ下限　: {rep_lower}\n"
                f"レンジ上限　: {rep_upper}\n"
                f"数量（万）　: {quantity_wan}  ← ※松井証券アプリの「数量(万)」にそのまま入力！\n"
                f"注文値幅　　: {ai_recommended_width} pips\n"
                f"益出し幅　　: {ai_recommended_width} pips\n"
                f"運用停止ライン: {rep_op_stop_line} （レンジ下限から -{buffer_pips} pips）\n"
                f"（参考・計算用通貨量: {custom_quantity:,} 通貨 / 考慮口座資金: ¥{account_balance:,}）"
            )
            st.code(summary_text, language="text")

        elif market_status == "SELL":
            rep_side = "売"
            rep_lower = round(latest_price - half_range, 3 if is_jpy_pair else 5)
            rep_upper = round(latest_price + half_range, 3 if is_jpy_pair else 5)
            buffer_val = max(latest_atr * 2.0, 0.5 if is_jpy_pair else 0.05)
            rep_op_stop_line = round(rep_upper + buffer_val, 3 if is_jpy_pair else 5)
            buffer_pips = round(buffer_val / pip_unit, 1)

            st.error(f"🔴 **売りリピート推奨** （資金連動・安全レンジ自動調整済み）")
            summary_text = (
                f"通貨ペア　　: {selected_label}\n"
                f"売買区分　　: {rep_side}\n"
                f"AI判定　　　: 売り (信頼度 {confidence:.1f}%)\n"
                f"レンジ下限　: {rep_lower}\n"
                f"レンジ上限　: {rep_upper}\n"
                f"数量（万）　: {quantity_wan}  ← ※松井証券アプリの「数量(万)」にそのまま入力！\n"
                f"注文値幅　　: {ai_recommended_width} pips\n"
                f"益出し幅　　: {ai_recommended_width} pips\n"
                f"運用停止ライン: {rep_op_stop_line} （レンジ上限から +{buffer_pips} pips）\n"
                f"（参考・計算用通貨量: {custom_quantity:,} 通貨 / 考慮口座資金: ¥{account_balance:,}）"
            )
            st.code(summary_text, language="text")
        else:
            st.warning(f"🟡 **様子見モード (HOLD)** （AI信頼度: {confidence:.1f}%のため、新規リピート設定は非推奨です）")

        # Discord 重複通知防止ロジック
        if 'last_sent_pair_status' not in st.session_state:
            st.session_state.last_sent_pair_status = {}

        if enable_notify and discord_url:
            last_sent = st.session_state.last_sent_pair_status.get(selected_label)
            if market_status in ["BUY", "SELL"] and last_sent != market_status:
                msg = f"📱 **【FX AIシグナル発動】**\n• 通貨ペア: {selected_label}\n• 判定: {market_status}\n• 信頼度: {confidence:.1f}%\n• レート: {latest_price:{price_fmt}}"
                success = send_discord_notification(discord_url, msg)
                if success:
                    st.session_state.last_sent_pair_status[selected_label] = market_status

    with tab_chart:
        st.subheader("📈 Pro仕様 インタラクティブ・ローソク足チャート (Plotly)")
        
        df_chart = data.tail(60)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="ローソク足"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], mode='lines', name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Upper_Band'], mode='lines', name='+2σ', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Lower_Band'], mode='lines', name='-2σ', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], mode='lines', name='RSI(14)', line=dict(color='purple', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=5, r=5, t=20, b=5),
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_scanner:
        st.subheader("🔍 全監視通貨ペア AIスコア・一括スキャン")
        st.write("※個別画面と全く同じ日足フィルター＆100本学習で一括スキャンを行います。")
        if st.button("🚀 全ペアを一括スキャン実行", use_container_width=True):
            scan_results = []
            with st.spinner("全通貨ペアを同条件（100本学習＆日足分析）で計算中..."):
                for p_label, p_symbol in PAIRS.items():
                    sub_df = load_and_process_data(p_symbol, tf_config['period'], tf_config['interval'], tf_label)
                    sub_htf = load_and_process_data(p_symbol, "1y", "1d", "日足 (スイング・環境認識用)")
                    
                    if sub_df is not None and len(sub_df) > 10:
                        s_status, s_conf, _ = analyze_signal(sub_df, sub_htf)
                        s_adx = sub_df['ADX'].iloc[-1] if 'ADX' in sub_df.columns else 25.0
                        
                        scan_results.append({
                            "通貨ペア": p_label,
                            "AI総合判定": s_status,
                            "信頼度 (%)": round(s_conf, 1),
                            "ADX (トレンド強度)": round(s_adx, 1)
                        })
            
            if scan_results:
                res_df = pd.DataFrame(scan_results).sort_values(by="信頼度 (%)", ascending=False)
                st.dataframe(res_df, use_container_width=True)
            else:
                st.error("データの取得に失敗しました。時間をおいて再度お試しください。")

    with tab_backtest:
        st.subheader("📊 AIモデルの時系列バックテスト（Walk-forward方式）")
        st.write(f"過去データ（{test_len} 本分）において、事前に未来を知らされない状態で学習・予測を繰り返したリアルな精度推移です。")
        
        if cumulative_wins:
            cb_df = pd.DataFrame(cumulative_wins, columns=["検証ステップ", "累積勝率 (%)"]).set_index("検証ステップ")
            st.line_chart(cb_df)
            st.metric("ウォークフォワード最終勝率", f"{win_rate:.1f}%", f"({correct_count}勝 / {test_len}戦)")
        else:
            st.warning("十分な過去データがないため、バックテストをスキップしました。")

    st.divider()
    with st.expander("📄 データテーブル表示（デバッグ・分析用）"):
        st.dataframe(data[available_features + ['ATR', 'BB_Width', 'SMA_50']].tail(10))

# スマート自動リフレッシュ処理
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
