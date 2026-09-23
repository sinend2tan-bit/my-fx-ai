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

# ==========================================
# 0. 画面基本設定
# ==========================================
st.set_page_config(
    page_title="プロ版 AI FXデイトレ & リピートアナライザー Pro v5.0", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. 設定ファイルの永続化（保存・読み込み）
# ==========================================
SETTINGS_FILE = "user_settings.json"

DEFAULT_SETTINGS = {
    "account_balance": 200000,
    "quantity_wan": 0.20,
    "discord_url": "",
    "enable_notify": False,
    "auto_refresh": False,
    "refresh_interval": 180,
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

# Discord通知関数
def send_discord_notification(webhook_url, title, message, color=0x00ff00):
    if not webhook_url:
        return False, "URL未設定"
    
    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat()
        }]
    }
    try:
        res = requests.post(webhook_url, json=payload, timeout=5)
        if res.status_code in [200, 204]:
            return True, "送信成功"
        else:
            return False, f"ステータスコード: {res.status_code}"
    except Exception as e:
        return False, str(e)

# ==========================================
# 2. メイン画面 & サイドバー設定
# ==========================================
st.title("⚡ Pro AI FX デイトレ & リピートアナライザー (v5.0)")

PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}

TIMEFRAMES = {
    "15分足 (デイトレエントリー用)": {"period": "1mo", "interval": "15m"},
    "1時間足 (デイトレメイン用)": {"period": "6mo", "interval": "1h"},
    "4時間足 (中期トレンド用)": {"period": "6mo", "interval": "1h"},
    "12時間足 (長期トレンド用)": {"period": "1y", "interval": "1h"},
    "日足 (スイング・環境認識用)": {"period": "2y", "interval": "1d"},
}

PAIR_ATR_CONFIG = {
    "USDJPY=X": {"atr_mult": 0.20, "min_pips": 15},
    "GBPJPY=X": {"atr_mult": 0.25, "min_pips": 20},
    "EURJPY=X": {"atr_mult": 0.20, "min_pips": 15},
    "AUDJPY=X": {"atr_mult": 0.18, "min_pips": 12},
    "EURUSD=X": {"atr_mult": 0.18, "min_pips": 12},
}

col_s1, col_s2 = st.columns(2)
with col_s1:
    selected_label = st.selectbox("通貨ペアを選択", list(PAIRS.keys()), key="selected_pair_label")
with col_s2:
    tf_label = st.selectbox("時間軸（タイムフレーム）を選択", list(TIMEFRAMES.keys()), key="selected_tf_label")

ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]
atr_cfg = PAIR_ATR_CONFIG.get(ticker, {"atr_mult": 0.20, "min_pips": 15})

is_jpy_pair = "JPY" in ticker
pip_unit = 0.01 if is_jpy_pair else 0.0001
price_fmt = ".3f" if is_jpy_pair else ".5f"

st.sidebar.header("⚙️ システム設定 & 口座管理")

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

st.sidebar.subheader("📋 松井証券トレード資金設定")
account_balance = st.sidebar.number_input(
    "口座資金 (円)", 
    min_value=10000, 
    max_value=100000000, 
    step=50000, 
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

st.sidebar.subheader("🔔 Discord 通知設定")
discord_url = st.sidebar.text_input(
    "Webhook URL", 
    type="password", 
    key="discord_url", 
    on_change=save_user_settings
)
enable_notify = st.sidebar.checkbox(
    "AI売買シグナル時に通知する", 
    key="enable_notify", 
    on_change=save_user_settings
)

if st.sidebar.button("🧪 Discord テスト送信"):
    if discord_url:
        test_msg = f"選択中の通貨ペア: **{selected_label}**\nDiscord連携は正常に動作しています！"
        ok, msg = send_discord_notification(discord_url, "🧪 テスト通知成功", test_msg, color=0x3498db)
        if ok:
            st.sidebar.success("テスト通知を送信しました！")
        else:
            st.sidebar.error(f"送信失敗: {msg}")
    else:
        st.sidebar.warning("Webhook URLを入力してください。")

# ==========================================
# 3. データ取得 & 高精度インジケーター計算エンジン
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

    if df.empty or len(df) < 50:
        try:
            df = yf.download(symbol, period="3mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    if df.empty or len(df) < 50:
        return None

    try:
        pip_unit_local = 0.01 if "JPY" in symbol else 0.0001
        
        # 移動平均線
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # ATR & ボラティリティ変化率
        high_low = df['High'] - df['Low']
        df['ATR'] = high_low.rolling(window=14).mean()
        df['ATR_SMA20'] = df['ATR'].rolling(window=20).mean()
        df['ATR_Ratio'] = df['ATR'] / (df['ATR_SMA20'] + 1e-10)

        # ローソク足ヒゲ率
        total_range = df['High'] - df['Low'] + 1e-10
        df['Upper_Wick_Ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / total_range
        df['Lower_Wick_Ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / total_range

        # 特徴量 (相対化・騰落率)
        df['Return_1'] = df['Close'].pct_change(1)
        df['Return_5'] = df['Close'].pct_change(5)
        df['Dev_SMA20'] = (df['Close'] - df['SMA_20']) / (df['SMA_20'] + 1e-10)
        df['Dev_EMA200'] = (df['Close'] - df['EMA_200']) / (df['EMA_200'] + 1e-10)
        df['Vol_Ratio'] = df['ATR'] / (df['Close'] + 1e-10)

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
        df['RSI_Diff'] = df['RSI'].diff(1)

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_Ratio'] = df['MACD_Hist'] / (df['Close'] + 1e-10)

        # ボリンジャーバンド
        std20 = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['SMA_20'] + (std20 * 2)
        df['Lower_Band'] = df['SMA_20'] - (std20 * 2)
        df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / (df['SMA_20'] + 1e-10)
        df['BB_PctB'] = (df['Close'] - df['Lower_Band']) / ((df['Upper_Band'] - df['Lower_Band']) + 1e-10)

        # ADX
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
        sum_di = (plus_di + minus_di).replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / sum_di
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(25.0)

        # タイムフレーム別のTarget動的閾値設定
        if "15分" in tf_name or interval == "15m":
            target_pips_val = 8.0
        elif "1時間" in tf_name or interval == "1h":
            target_pips_val = 15.0
        else:
            target_pips_val = 25.0

        target_pips = target_pips_val * pip_unit_local
        future_max_up = df['High'].shift(-3).rolling(3).max() - df['Close']
        df['Target'] = np.where(future_max_up >= target_pips, 1, 0)

        # 欠損値の厳格穴埋め処理
        df = df.ffill().bfill().fillna(0)
        return df
    except Exception:
        return None

# AIシグナル & 防御重視型フィルター判定関数
def analyze_signal(df_current, df_higher, usdjpy_df=None, current_symbol=""):
    if df_current is None or len(df_current) < 50:
        return "HOLD", 50.0, None, "不明"

    try:
        features = [
            'Return_1', 'Return_5', 'Dev_SMA20', 'Dev_EMA200', 'Vol_Ratio', 
            'RSI', 'RSI_Diff', 'MACD_Hist_Ratio', 'BB_PctB', 'ADX',
            'ATR_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio'
        ]
        avail = [f for f in features if f in df_current.columns]
        
        X = df_current[avail]
        y = df_current['Target']
        
        X_train = X.iloc[-1000:-3] if len(X) > 1000 else X.iloc[:-3]
        y_train = y.iloc[-1000:-3] if len(y) > 1000 else y.iloc[:-3]
        X_latest = X.iloc[[-1]]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_latest)[0]
        confidence = max(prob) * 100

        latest_price = df_current['Close'].iloc[-1]
        latest_ema200 = df_current['EMA_200'].iloc[-1] if 'EMA_200' in df_current.columns else latest_price
        latest_sma20 = df_current['SMA_20'].iloc[-1] if 'SMA_20' in df_current.columns else latest_price
        latest_adx = df_current['ADX'].iloc[-1] if 'ADX' in df_current.columns else 25.0
        latest_rsi = df_current['RSI'].iloc[-1] if 'RSI' in df_current.columns else 50.0
        latest_atr = df_current['ATR'].iloc[-1] if 'ATR' in df_current.columns else 0.1
        upper_band = df_current['Upper_Band'].iloc[-1] if 'Upper_Band' in df_current.columns else latest_price
        lower_band = df_current['Lower_Band'].iloc[-1] if 'Lower_Band' in df_current.columns else latest_price
        
        latest_bb_width = df_current['BB_Width'].iloc[-1] if 'BB_Width' in df_current.columns else 0.05
        avg_bb_series = df_current['BB_Width'].rolling(window=20).mean()
        avg_bb_width = avg_bb_series.iloc[-1] if not avg_bb_series.empty and not pd.isna(avg_bb_series.iloc[-1]) else 0.05
        is_squeezed = latest_bb_width < (avg_bb_width * 0.75)

        htf_trend = "FLAT"
        if df_higher is not None and not df_higher.empty and 'EMA_200' in df_higher.columns:
            htf_close = df_higher['Close'].iloc[-1]
            htf_ema = df_higher['EMA_200'].iloc[-1]
            if htf_close > htf_ema:
                htf_trend = "UP"
            elif htf_close < htf_ema:
                htf_trend = "DOWN"

        # クロス円ストッパー判定
        is_cross_jpy = "JPY" in current_symbol and current_symbol != "USDJPY=X"
        usdjpy_strong_up = False
        usdjpy_strong_down = False
        
        if is_cross_jpy and usdjpy_df is not None and not usdjpy_df.empty and 'EMA_200' in usdjpy_df.columns:
            uj_close = usdjpy_df['Close'].iloc[-1]
            uj_ema = usdjpy_df['EMA_200'].iloc[-1]
            uj_rsi = usdjpy_df['RSI'].iloc[-1] if 'RSI' in usdjpy_df.columns else 50.0
            
            if uj_close > uj_ema and uj_rsi > 58:
                usdjpy_strong_up = True
            elif uj_close < uj_ema and uj_rsi < 42:
                usdjpy_strong_down = True

        recent_50_high = df_current['High'].iloc[-50:-1].max()
        recent_50_low = df_current['Low'].iloc[-50:-1].min()

        is_far_from_sma = abs(latest_price - latest_sma20) > (latest_atr * 1.5)
        space_to_support = latest_price - recent_50_low
        space_to_resistance = recent_50_high - latest_price
        is_near_support = space_to_support < (latest_atr * 0.8)
        is_near_resistance = space_to_resistance < (latest_atr * 0.8)

        HIGH_THRESHOLD = 0.70

        if latest_adx > 22.0:
            market_type = "トレンド相場"
            if prob[1] >= HIGH_THRESHOLD and htf_trend != "DOWN":
                if usdjpy_strong_down:
                    status = "HOLD (ストッパー: ドル円急落中)"
                elif latest_price <= latest_ema200:
                    status = "HOLD (逆張り警戒: 200EMA下)"
                elif is_far_from_sma:
                    status = "HOLD (高値掴み回避)"
                elif is_near_resistance:
                    status = "HOLD (抵抗線直前)"
                else:
                    status = "BUY"
            elif prob[0] >= HIGH_THRESHOLD and htf_trend != "UP":
                if usdjpy_strong_up:
                    status = "HOLD (ストッパー: ドル円急騰中)"
                elif latest_price >= latest_ema200:
                    status = "HOLD (逆張り警戒: 200EMA上)"
                elif is_far_from_sma:
                    status = "HOLD (安値掴み回避)"
                elif is_near_support:
                    status = "HOLD (支持線直前)"
                else:
                    status = "SELL"
            else:
                status = f"HOLD (確信度不足: {confidence:.1f}%)"
        else:
            market_type = "レンジ相場"
            if is_squeezed:
                status = "HOLD (ブレイクアウト警戒)"
            else:
                if (latest_price <= lower_band or latest_rsi <= 32.0) and prob[1] >= 0.65:
                    status = "BUY (レンジ逆張り)" if not usdjpy_strong_down else "HOLD (ストッパー: ドル円逆行)"
                elif (latest_price >= upper_band or latest_rsi >= 68.0) and prob[0] >= 0.65:
                    status = "SELL (レンジ逆張り)" if not usdjpy_strong_up else "HOLD (ストッパー: ドル円逆行)"
                else:
                    status = "HOLD (レンジ内静観)"

        return status, confidence, model, market_type
    except Exception:
        return "HOLD", 50.0, None, "不明"

# ドル円データの事前読み込み
usdjpy_data = load_and_process_data("USDJPY=X", tf_config['period'], tf_config['interval'], tf_label)
data = load_and_process_data(ticker, tf_config['period'], tf_config['interval'], tf_label)
higher_tf_data = load_and_process_data(ticker, "1y", "1d", "日足 (スイング・環境認識用)")

# ==========================================
# 4. 時間帯・指標・週末市場クローズ判定
# ==========================================
now_datetime = datetime.now()
current_day = now_datetime.weekday()
current_hour_jst = now_datetime.hour
current_minute_jst = now_datetime.minute

is_weekend = (current_day == 5 and current_hour_jst >= 6) or (current_day == 6) or (current_day == 0 and current_hour_jst < 6)
is_low_liquidity = 3 <= current_hour_jst <= 7
is_ny_open = 21 <= current_hour_jst <= 23

# 主要指標発表帯 (21:15〜22:45)
is_econ_indicator_time = (current_hour_jst == 21 and current_minute_jst >= 15) or (current_hour_jst == 22 and current_minute_jst <= 45)

if is_weekend:
    st.error("🛑 **【週末・為替市場クローズ中】**: 現在外国為替市場は休業時間帯です。表示価格は最終クローズ値となります。")
elif is_econ_indicator_time:
    st.error("🚨 **【重要経済指標 警戒タイムゾーン】**: 突発的乱高下の危険がある時間帯です（21:15〜22:45）。新規エントリーは自重をお勧めします。")
elif is_low_liquidity:
    st.warning("⚠️ **【流動性低下タイムゾーン】**: オセアニア時間の早朝です。スプレッド拡大にご注意ください。")
elif is_ny_open:
    st.info("🔥 **【NY市場オープンタイムゾーン】**: ボラティリティが高まる時間帯です。")

# ==========================================
# 5. AI学習 & メイン画面表示
# ==========================================
if data is None or len(data) < 10:
    st.error("🚨 リアルタイムデータの取得に失敗しました。「最新データに更新」を押してください。")
else:
    market_status, confidence, main_model, market_type = analyze_signal(
        data, higher_tf_data, usdjpy_df=usdjpy_data, current_symbol=ticker
    )

    if is_econ_indicator_time and ("BUY" in market_status or "SELL" in market_status):
        market_status = "HOLD (指標発表警戒時間帯)"

    # Discord 自動通知処理（シグナル発生時）
    if enable_notify and discord_url and ("BUY" in market_status or "SELL" in market_status):
        last_sig_key = f"last_notified_{ticker}"
        if st.session_state.get(last_sig_key) != market_status:
            color_val = 0x2ecc71 if "BUY" in market_status else 0xe74c3c
            msg_body = (
                f"**通貨ペア**: {selected_label}\n"
                f"**現在レート**: {data['Close'].iloc[-1]:{price_fmt}}\n"
                f"**AIシグナル**: {market_status}\n"
                f"**確信度**: {confidence:.1f}%\n"
                f"**相場環境**: {market_type}"
            )
            ok, _ = send_discord_notification(discord_url, f"🚨 AI FXシグナル通知 [{selected_label}]", msg_body, color=color_val)
            if ok:
                st.session_state[last_sig_key] = market_status

    features = [
        'Return_1', 'Return_5', 'Dev_SMA20', 'Dev_EMA200', 'Vol_Ratio', 
        'RSI', 'RSI_Diff', 'MACD_Hist_Ratio', 'BB_PctB', 'ADX',
        'ATR_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio'
    ]
    available_features = [f for f in features if f in data.columns]
    X_bt = data[available_features].iloc[:-3]
    y_bt = data['Target'].iloc[:-3]
    
    test_len = min(30, len(X_bt) - 10)
    cumulative_wins = []
    
    if test_len > 5:
        correct_count = 0
        for i in range(test_len):
            idx = len(X_bt) - test_len + i
            sub_model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)
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
    is_squeezed = latest_bb_width < (avg_bb_width * 0.75)

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

    recent_high_20 = data['High'].iloc[-20:].max()
    recent_low_20 = data['Low'].iloc[-20:].min()
    buffer_margin = 10 * pip_unit

    structural_buy_sl = round(recent_low_20 - buffer_margin, 3 if is_jpy_pair else 5)
    structural_sell_sl = round(recent_high_20 + buffer_margin, 3 if is_jpy_pair else 5)

    conf_factor = confidence / 50.0
    adx_bonus = 0.2 if latest_adx > 25 else 0.0

    if "レンジ" in market_type:
        ai_tp_mult = 0.8
        ai_sl_mult = 0.6
    else:
        ai_tp_mult = round(max(1.0, min(2.5, 1.2 * conf_factor + adx_bonus)), 2)
        ai_sl_mult = round(max(0.6, min(1.5, 0.8 / (conf_factor * 0.8))), 2)

    # ATR連動型の動的リピート注文値幅計算
    raw_atr_pips = (latest_atr / pip_unit)
    calc_dynamic_width = round(raw_atr_pips * atr_cfg["atr_mult"], 1)
    ai_recommended_width = int(max(calc_dynamic_width, atr_cfg["min_pips"]))

    recommended_slippage = round(max(0.5, (latest_atr / pip_unit) * 0.05), 1)

    # 単発適正数量計算
    sl_distance_pips = round((latest_atr * ai_sl_mult) / pip_unit, 1)
    if sl_distance_pips <= 0:
        sl_distance_pips = 20.0
        
    allowed_loss_jpy = account_balance * 0.02
    pip_value_per_unit = 0.01 if is_jpy_pair else (0.0001 * 155.0)
    
    safe_single_units = int(allowed_loss_jpy / (sl_distance_pips * pip_value_per_unit))
    safe_single_units = max(100, min(safe_single_units, 50000))
    safe_single_wan = round(safe_single_units / 10000.0, 4)

    if is_squeezed:
        st.error("⚡ **【スクイーズ発生】**: ボリンジャーバンドが収縮中です。ブレイクアウトにご注意ください。")

    st.divider()

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("現在レート", f"{latest_price:{price_fmt}}")
    m_col2.metric("AI識別・現在の相場環境", market_type, "🔥トレンド状態" if latest_adx > 22 else "💤レンジ・揉み合い")
    m_col3.metric("直近AI検証適合率", f"{win_rate:.1f}%", f"({correct_count}/{test_len} 回)")

    m_col4, m_col5, m_col6 = st.columns(3)
    m_col4.metric("RSI (14)", f"{latest_rsi:.1f}")
    m_col5.metric("ADX (トレンド強度)", f"{latest_adx:.1f}")
    m_col6.metric("上位足 (日足) トレンド", long_term_trend)

    st.divider()

    # ==========================================
    # 6. タブ機能 (単発・リピート・チャート等)
    # ==========================================
    tab_repeat, tab_single, tab_speed, tab_chart, tab_scanner, tab_backtest = st.tabs([
        "📋 リピート注文 (松井証券専用)", 
        "🎯 デイトレ単発 (高精度AI)", 
        "⚡ スピード注文",
        "📈 ローソク足チャート",
        "🔍 全ペアスキャン", 
        "📊 バックテスト"
    ])

    with tab_repeat:
        st.subheader("📋 松井証券FX 自動売買（リピート注文）入力用パラメータ")
        st.write(f"💡 **直近の相場変動幅 (ATR = {raw_atr_pips:.1f} pips) に連動し、値幅 `{ai_recommended_width} pips` を自動設定しました。**")
        
        leverage = 25.0
        if is_jpy_pair:
            jpy_rate = latest_price
        else:
            uj_price = usdjpy_data['Close'].iloc[-1] if usdjpy_data is not None else 155.0
            jpy_rate = latest_price * uj_price

        margin_per_unit = (jpy_rate * custom_quantity) / leverage
        
        # 安全のために資金の70%までを証拠金として使える計算にする
        max_allowable_grids = max(2, int((account_balance * 0.7) / max(margin_per_unit, 1.0)))
        
        max_half_grids = max(1, max_allowable_grids // 2)
        safe_half_range_pips = max_half_grids * ai_recommended_width
        safe_half_range_val = safe_half_range_pips * pip_unit

        rep_lower = round(latest_price - safe_half_range_val, 3 if is_jpy_pair else 5)
        rep_upper = round(latest_price + safe_half_range_val, 3 if is_jpy_pair else 5)
        
        buffer_val = max(latest_atr * 1.5, 0.4 if is_jpy_pair else 0.04)
        rep_buy_stop = round(rep_lower - buffer_val, 3 if is_jpy_pair else 5)
        rep_sell_stop = round(rep_upper + buffer_val, 3 if is_jpy_pair else 5)
        buffer_pips = round(buffer_val / pip_unit, 1)

        total_est_units = custom_quantity * max_allowable_grids
        total_est_wan = round(total_est_units / 10000.0, 2)
        est_margin_needed = int(margin_per_unit * max_allowable_grids)

        st.info(f"🛡️ **証拠金管理**: 口座資金 **{account_balance:,}円** に対し、1本あたり **{quantity_wan}万通貨 ({custom_quantity:,}通貨)** で同時設定できる安全上限は **{max_allowable_grids}本** です。（想定最大必要証拠金: 約{est_margin_needed:,}円）")

        rep_c1, rep_c2 = st.columns(2)
        with rep_c1:
            st.markdown("#### 🟢 買いリピート設定 (BUY)")
            st.code(
                f"通貨ペア　　: {selected_label}\n"
                f"売買区分　　: 買\n"
                f"レンジ上限　: {rep_upper}\n"
                f"レンジ下限　: {rep_lower}\n"
                f"数量（万）　: {quantity_wan}  (※松井アプリ用)\n"
                f"注文値幅　　: {ai_recommended_width} pips  (ATR自動連動)\n"
                f"益出し幅　　: {ai_recommended_width} pips  (ATR自動連動)\n"
                f"運用停止ライン: {rep_buy_stop} (-{buffer_pips}pips)\n"
                f"----------------------------------------\n"
                f"【構成案内】最大仕掛け本数: {max_allowable_grids}本 ({total_est_wan}万通貨分)",
                language="text"
            )

        with rep_c2:
            st.markdown("#### 🔴 売りリピート設定 (SELL)")
            st.code(
                f"通貨ペア　　: {selected_label}\n"
                f"売買区分　　: 売\n"
                f"レンジ上限　: {rep_upper}\n"
                f"レンジ下限　: {rep_lower}\n"
                f"数量（万）　: {quantity_wan}  (※松井アプリ用)\n"
                f"注文値幅　　: {ai_recommended_width} pips  (ATR自動連動)\n"
                f"益出し幅　　: {ai_recommended_width} pips  (ATR自動連動)\n"
                f"運用停止ライン: {rep_sell_stop} (+{buffer_pips}pips)\n"
                f"----------------------------------------\n"
                f"【構成案内】最大仕掛け本数: {max_allowable_grids}本 ({total_est_wan}万通貨分)",
                language="text"
            )

    with tab_single:
        st.subheader("🎯 デイトレ単発トレード（高精度ノイズ除去モデル）")
        
        st.info(f"🛡️ **口座資金 ({account_balance:,}円) に基づく単発適正数量ガイド**: 1回のリスクを資金2%（{int(allowed_loss_jpy):,}円）以下に抑える推奨注文数量は **`{safe_single_wan}万通貨` ({safe_single_units:,}通貨)** です。")
        
        if market_status == "BUY" or "BUY (" in market_status:
            entry_price = latest_price
            tp_price = entry_price + (latest_atr * ai_tp_mult)
            sl_price = min(entry_price - (latest_atr * ai_sl_mult), structural_buy_sl)
            tp_pips = (tp_price - entry_price) / pip_unit
            sl_pips = (entry_price - sl_price) / pip_unit

            st.success(f"🟢 **高確信 買いシグナル確定 ({market_status})** （確信度: {confidence:.1f}% | 厳格基準クリア）")
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                st.metric("新規買い目安 (Entry)", f"{entry_price:{price_fmt}}")
                st.code(f"{entry_price:{price_fmt}}", language="text")
            with t_col2:
                st.metric("利確目標 (AI最適TP)", f"{tp_price:{price_fmt}}", f"+{tp_pips:.1f} pips")
                st.code(f"{tp_price:{price_fmt}}", language="text")
            with t_col3:
                st.metric("防衛SL (構造的安値外側)", f"{sl_price:{price_fmt}}", f"-{sl_pips:.1f} pips")
                st.code(f"{sl_price:{price_fmt}}", language="text")

        elif market_status == "SELL" or "SELL (" in market_status:
            entry_price = latest_price
            tp_price = entry_price - (latest_atr * ai_tp_mult)
            sl_price = max(entry_price + (latest_atr * ai_sl_mult), structural_sell_sl)
            tp_pips = (entry_price - tp_price) / pip_unit
            sl_pips = (sl_price - entry_price) / pip_unit

            st.error(f"🔴 **高確信 売りシグナル確定 ({market_status})** （確信度: {confidence:.1f}% | 厳格基準クリア）")
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                st.metric("新規売り目安 (Entry)", f"{entry_price:{price_fmt}}")
                st.code(f"{entry_price:{price_fmt}}", language="text")
            with t_col2:
                st.metric("利確目標 (AI最適TP)", f"{tp_price:{price_fmt}}", f"-{tp_pips:.1f} pips")
                st.code(f"{tp_price:{price_fmt}}", language="text")
            with t_col3:
                st.metric("防衛SL (構造的高値外側)", f"{sl_price:{price_fmt}}", f"+{sl_pips:.1f} pips")
                st.code(f"{sl_price:{price_fmt}}", language="text")
        else:
            st.warning(f"🟡 **静観フィルター発動中 ({market_status})**")
            st.info("💡 **解説:** ノイズ除去・ドル円連動ストッパー・指標発表警戒帯等の安全装置により、騙しリスクが高い場面では自動で「HOLD」判定になります。")

    with tab_speed:
        st.subheader("⚡ 松井証券FX アプリ【スピード注文】設定用")
        sp_tp_pips = round(latest_atr * ai_tp_mult / pip_unit, 1)
        if "SELL" in market_status:
            sp_sl_pips = round((structural_sell_sl - latest_price) / pip_unit, 1)
        elif "BUY" in market_status:
            sp_sl_pips = round((latest_price - structural_buy_sl) / pip_unit, 1)
        else:
            sp_sl_pips = round(latest_atr * ai_sl_mult / pip_unit, 1)

        sp_col1, sp_col2, sp_col3 = st.columns(3)
        sp_col1.metric("資金ベース推奨数量 (万)", f"{safe_single_wan}万 ({safe_single_units:,}通貨)")
        sp_col2.metric("益出し幅 (利確)", f"{sp_tp_pips} pips")
        sp_col3.metric("防衛損切り幅 (損切)", f"{sp_sl_pips} pips")

        rec_dir = "買い (ASK)" if "BUY" in market_status else "売り (BID)" if "SELL" in market_status else "様子見 (静観)"

        st.code(
            f"通貨ペア: {selected_label}\n"
            f"推奨エントリー: {rec_dir}\n"
            f"【推奨安全数量】: {safe_single_wan} 万通貨 ({safe_single_units:,} 通貨)\n"
            f"益出し幅: {sp_tp_pips} pips\n"
            f"損切り幅: {sp_sl_pips} pips\n"
            f"許容スリッページ: {recommended_slippage} pips",
            language="text"
        )

    with tab_chart:
        st.subheader("📈 Pro仕様 インタラクティブ・ローソク足チャート (Plotly)")
        df_chart = data.tail(60).copy()
        chart_x = [str(x) for x in df_chart.index]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=chart_x, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="ローソク足"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=chart_x, y=df_chart['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_x, y=df_chart['SMA_50'], mode='lines', name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
        if 'EMA_200' in df_chart.columns:
            fig.add_trace(go.Scatter(x=chart_x, y=df_chart['EMA_200'], mode='lines', name='EMA 200', line=dict(color='white', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_x, y=df_chart['Upper_Band'], mode='lines', name='+2σ', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_x, y=df_chart['Lower_Band'], mode='lines', name='-2σ', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)

        fig.add_trace(go.Scatter(x=chart_x, y=df_chart['RSI'], mode='lines', name='RSI(14)', line=dict(color='purple', width=1.5)), row=2, col=1)
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
        st.subheader("🔍 全監視通貨ペア AI防衛スキャン")
        if st.button("🚀 全ペアを一括スキャン実行", use_container_width=True):
            scan_results = []
            with st.spinner("全通貨ペアを分析中..."):
                for p_label, p_symbol in PAIRS.items():
                    sub_df = load_and_process_data(p_symbol, tf_config['period'], tf_config['interval'], tf_label)
                    sub_htf = load_and_process_data(p_symbol, "1y", "1d", "日足 (スイング・環境認識用)")
                    if sub_df is not None and len(sub_df) > 10:
                        s_status, s_conf, _, s_mtype = analyze_signal(
                            sub_df, sub_htf, usdjpy_df=usdjpy_data, current_symbol=p_symbol
                        )
                        s_adx = sub_df['ADX'].iloc[-1] if 'ADX' in sub_df.columns else 25.0
                        scan_results.append({
                            "通貨ペア": p_label,
                            "相場環境": s_mtype,
                            "AI総合判定": s_status,
                            "確信度 (%)": round(s_conf, 1),
                            "ADX (強度)": round(s_adx, 1)
                        })
            if scan_results:
                res_df = pd.DataFrame(scan_results).sort_values(by="確信度 (%)", ascending=False)
                st.dataframe(res_df, use_container_width=True)

    with tab_backtest:
        st.subheader("📊 改良型モデルの時系列ウォークフォワード検証")
        if cumulative_wins:
            cb_df = pd.DataFrame(cumulative_wins, columns=["検証ステップ", "累積適合率 (%)"]).set_index("検証ステップ")
            st.line_chart(cb_df)
            st.metric("時系列検証の適合率", f"{win_rate:.1f}%", f"({correct_count}回適合 / {test_len}ステップ)")

    st.divider()
    with st.expander("📄 学習データテーブル確認（相対化済みの特徴量）"):
        st.dataframe(data[available_features + ['ATR', 'BB_Width']].tail(10))

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
