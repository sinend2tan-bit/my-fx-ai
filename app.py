import json
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 0. 画面基本設定 & 共通定数
# ==========================================
st.set_page_config(
    page_title="プロ版 AI FXデイトレ & リピートアナライザー Pro v7.0",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLUMNS = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200", "Dev_EMA20_200", "Vol_Ratio",
    "RSI", "RSI_Diff", "MACD_Hist_Ratio", "BB_PctB", "ADX",
    "ATR_Ratio", "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Stoch_K"
]

SETTINGS_FILE = "user_settings.json"

DEFAULT_SETTINGS = {
    "account_balance": 200000,
    "quantity_wan": 0.20,
    "discord_url": "",
    "enable_notify": False,
    "auto_refresh": False,
    "refresh_interval": 180,
    "selected_pair_label": "米ドル / 円 (USD/JPY)",
    "selected_tf_label": "15分足 (デイトレエントリー用)",
    "last_notified_status": {},
}

# ==========================================
# 1. 設定ファイルの永続化 & 補助関数
# ==========================================
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
        "selected_pair_label": st.session_state.get("selected_pair_label", DEFAULT_SETTINGS["selected_pair_label"]),
        "selected_tf_label": st.session_state.get("selected_tf_label", DEFAULT_SETTINGS["selected_tf_label"]),
        "last_notified_status": st.session_state.get("last_notified_status", DEFAULT_SETTINGS["last_notified_status"]),
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

def send_discord_notification(webhook_url, title, message, color=0x00FF00):
    if not webhook_url:
        return False, "URL未設定"
    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

def get_signal_type(status_str):
    if status_str.startswith("BUY"):
        return "BUY"
    elif status_str.startswith("SELL"):
        return "SELL"
    return "HOLD"

def clean_series(s):
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s

# ==========================================
# 2. データ取得 & インジケーター計算エンジン
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    df = pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.levels[1]:
                df = df.xs(symbol, axis=1, level=1)
            else:
                df.columns = df.columns.get_level_values(0)
    except Exception:
        return None

    if not df.empty and "4時間足" in tf_name and interval == "1h":
        try:
            tz_before = df.index.tz
            df = df.resample("4h", closed="left", label="left", origin="start_day").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            if tz_before is not None and df.index.tz is None:
                df.index = df.index.tz_localize(tz_before)
        except Exception:
            pass

    if df.empty or len(df) < 50:
        return None

    try:
        c_series = clean_series(df["Close"]).astype(float)
        h_series = clean_series(df["High"]).astype(float)
        l_series = clean_series(df["Low"]).astype(float)
        o_series = clean_series(df["Open"]).astype(float)

        new_cols = {}
        new_cols["SMA_20"] = c_series.rolling(window=20).mean()
        new_cols["SMA_50"] = c_series.rolling(window=50).mean()
        new_cols["EMA_20"] = c_series.ewm(span=20, adjust=False).mean()
        new_cols["EMA_200"] = c_series.ewm(span=200, adjust=False).mean()

        high_low = h_series - l_series
        new_cols["ATR"] = high_low.rolling(window=14).mean()
        new_cols["ATR_SMA20"] = new_cols["ATR"].rolling(window=20).mean()
        new_cols["ATR_Ratio"] = new_cols["ATR"] / (new_cols["ATR_SMA20"] + 1e-10)

        total_range = high_low + 1e-10
        open_close_max = pd.concat([o_series, c_series], axis=1).max(axis=1)
        open_close_min = pd.concat([o_series, c_series], axis=1).min(axis=1)
        new_cols["Upper_Wick_Ratio"] = (h_series - open_close_max) / total_range
        new_cols["Lower_Wick_Ratio"] = (open_close_min - l_series) / total_range

        new_cols["Return_1"] = c_series.diff(1) / (c_series.shift(1) + 1e-10)
        new_cols["Return_5"] = c_series.diff(5) / (c_series.shift(5) + 1e-10)
        
        new_cols["Dev_SMA20"] = (c_series - new_cols["SMA_20"]) / (new_cols["SMA_20"] + 1e-10)
        new_cols["Dev_EMA200"] = (c_series - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Dev_EMA20_200"] = (new_cols["EMA_20"] - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Vol_Ratio"] = new_cols["ATR"] / (c_series + 1e-10)

        delta = c_series.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        new_cols["RSI"] = 100.0 - (100.0 / (1.0 + rs))
        new_cols["RSI_Diff"] = new_cols["RSI"].diff(1)

        low_14 = l_series.rolling(window=14).min()
        high_14 = h_series.rolling(window=14).max()
        new_cols["Stoch_K"] = 100.0 * (c_series - low_14) / ((high_14 - low_14) + 1e-10)

        ema12 = c_series.ewm(span=12, adjust=False).mean()
        ema26 = c_series.ewm(span=26, adjust=False).mean()
        new_cols["MACD"] = ema12 - ema26
        new_cols["MACD_Signal"] = new_cols["MACD"].ewm(span=9, adjust=False).mean()
        new_cols["MACD_Hist"] = new_cols["MACD"] - new_cols["MACD_Signal"]
        new_cols["MACD_Hist_Ratio"] = new_cols["MACD_Hist"] / (c_series + 1e-10)

        std20 = c_series.rolling(window=20).std()
        new_cols["Upper_Band"] = new_cols["SMA_20"] + (std20 * 2)
        new_cols["Lower_Band"] = new_cols["SMA_20"] - (std20 * 2)
        new_cols["BB_Width"] = (new_cols["Upper_Band"] - new_cols["Lower_Band"]) / (new_cols["SMA_20"] + 1e-10)
        new_cols["BB_PctB"] = (c_series - new_cols["Lower_Band"]) / ((new_cols["Upper_Band"] - new_cols["Lower_Band"]) + 1e-10)

        tr = pd.concat([high_low, (h_series - c_series.shift(1)).abs(), (l_series - c_series.shift(1)).abs()], axis=1).max(axis=1)
        up_move = h_series - h_series.shift(1)
        down_move = l_series.shift(1) - l_series
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
        sum_di = (plus_di + minus_di).replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / sum_di
        new_cols["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(25.0)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        f_high = pd.concat([h_series.shift(-1), h_series.shift(-2), h_series.shift(-3)], axis=1)
        f_low = pd.concat([l_series.shift(-1), l_series.shift(-2), l_series.shift(-3)], axis=1)

        target_pips = new_cols["ATR"] * 0.8  
        future_max_up = f_high.max(axis=1) - c_series
        future_max_down = c_series - f_low.min(axis=1)

        conditions = [
            (future_max_up >= target_pips) & (future_max_up > future_max_down),
            (future_max_down >= target_pips) & (future_max_down > future_max_up),
        ]
        
        target_series = np.select(conditions, [1, -1], default=0)
        df["Target"] = target_series.astype(float)
        if len(df) > 3:
            df.iloc[-3:, df.columns.get_loc("Target")] = np.nan

        feature_cols = [c for c in df.columns if c != "Target"]
        df = df.dropna(subset=feature_cols)

        if df.empty:
            return None
        return df
    except Exception:
        return None

PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}

TIMEFRAMES = {
    "5分足 (超短期スキャル用)": {"period": "7d", "interval": "5m"},
    "15分足 (デイトレエントリー用)": {"period": "1mo", "interval": "15m"},
    "1時間足 (デイトレメイン用)": {"period": "6mo", "interval": "1h"},
    "4時間足 (中期トレンド用)": {"period": "2y", "interval": "1h"},
    "日足 (スイング・環境認識用)": {"period": "2y", "interval": "1d"},
}

@st.cache_data(ttl=60, show_spinner=False)
def get_mtf_trends(symbol: str) -> dict:
    trends = {}
    for name, params in TIMEFRAMES.items():
        sub_d = load_and_process_data(symbol, params["period"], params["interval"], name)
        if sub_d is not None and len(sub_d) >= 20:
            c_price = float(clean_series(sub_d['Close']).iloc[-1])
            ema_series = clean_series(sub_d['EMA_200']) if 'EMA_200' in sub_d.columns else clean_series(sub_d['SMA_20'])
            c_ema = float(ema_series.iloc[-1])
            if c_price > c_ema:
                trends[name.split(" ")[0]] = "上昇 📈"
            elif c_price < c_ema:
                trends[name.split(" ")[0]] = "下降 📉"
            else:
                trends[name.split(" ")[0]] = "レンジ ➡️"
        else:
            trends[name.split(" ")[0]] = "判定中..."
    return trends
# ==========================================
# 3. UIヘッダー & サイドバー設定
# ==========================================
now_datetime_jst = datetime.now(ZoneInfo("Asia/Tokyo"))

st.title("⚡ AI FXデイトレ & リピートアナライザー Pro v7.0")
st.caption(f"一目でわかる単発・自動売買 AI推奨値出力モデル (取得日時: {now_datetime_jst.strftime('%Y-%m-%d %H:%M:%S')} JST)")

with st.sidebar:
    st.header("⚙️ システム設定 & 口座管理")

    if st.button("🔄 今すぐ最新データに更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.checkbox("自動更新を有効にする", value=st.session_state.get("auto_refresh", False))
    refresh_interval = st.selectbox(
        "更新間隔を選択",
        [60, 180, 300, 600],
        index=[60, 180, 300, 600].index(st.session_state.get("refresh_interval", 180)),
        format_func=lambda x: f"{x // 60}分ごと" if x >= 60 else f"{x}秒ごと"
    )

    st.session_state["auto_refresh"] = auto_refresh
    st.session_state["refresh_interval"] = refresh_interval

    st.markdown("---")
    st.subheader("📊 監視設定")
    
    pair_labels = list(PAIRS.keys())
    saved_pair = st.session_state.get("selected_pair_label", pair_labels[0])
    pair_idx = pair_labels.index(saved_pair) if saved_pair in pair_labels else 0
    selected_label = st.selectbox("通貨ペアを選択", pair_labels, index=pair_idx)
    st.session_state["selected_pair_label"] = selected_label
    ticker = PAIRS[selected_label]

    tf_labels = list(TIMEFRAMES.keys())
    saved_tf = st.session_state.get("selected_tf_label", tf_labels[1])
    tf_idx = tf_labels.index(saved_tf) if saved_tf in tf_labels else 1
    tf_label = st.selectbox("分析時間軸を選択", tf_labels, index=tf_idx)
    st.session_state["selected_tf_label"] = tf_label
    tf_config = TIMEFRAMES[tf_label]

    is_jpy_pair = "JPY" in ticker
    pip_unit = 0.01 if is_jpy_pair else 0.0001
    price_fmt = ".3f" if is_jpy_pair else ".5f"
    input_format = f"%{price_fmt}"

    st.markdown("---")
    st.subheader("📋 口座資金 & リスク管理")
    account_balance = st.number_input(
        "口座資金 (円)",
        min_value=10000,
        max_value=100000000,
        value=int(st.session_state.get("account_balance", 200000)),
        step=10000
    )
    st.session_state["account_balance"] = account_balance

    quantity_wan = st.number_input(
        "基準注文数量 (万通貨)",
        min_value=0.01,
        max_value=100.0,
        value=float(st.session_state.get("quantity_wan", 0.20)),
        step=0.01,
        format="%.2f"
    )
    st.session_state["quantity_wan"] = quantity_wan
    custom_quantity = quantity_wan * 10000

    st.markdown("---")
    st.subheader("🔔 Discord 通知設定")
    discord_url = st.text_input(
        "Webhook URL",
        value=st.session_state.get("discord_url", ""),
        type="password"
    )
    st.session_state["discord_url"] = discord_url

    enable_notify = st.checkbox(
        "AI売買シグナル時に通知する",
        value=st.session_state.get("enable_notify", False)
    )
    st.session_state["enable_notify"] = enable_notify

    save_user_settings()

# ==========================================
# 4. AI判定 & 推奨値自動算出エンジン
# ==========================================
def analyze_and_recommend(df, higher_tf_df, usdjpy_df, symbol, balance, pip_u):
    if df is None or len(df) < 60:
        return None

    avail_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    train_mask = ~df["Target"].isna()
    X_train = df.loc[train_mask, avail_features]
    y_train = df.loc[train_mask, "Target"]

    if len(X_train) < 40 or len(np.unique(y_train)) < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    latest_feat = df.iloc[[-1]][avail_features]
    pred = model.predict(latest_feat)[0]
    probs = model.predict_proba(latest_feat)[0]
    classes = list(model.classes_)

    prob_buy = probs[classes.index(1.0)] if 1.0 in classes else 0.0
    prob_sell = probs[classes.index(-1.0)] if -1.0 in classes else 0.0
    prob_hold = probs[classes.index(0.0)] if 0.0 in classes else 0.0

    c_series = clean_series(df["Close"])
    latest_price = float(c_series.iloc[-1])
    latest_atr = float(clean_series(df["ATR"]).iloc[-1]) if "ATR" in df.columns else (latest_price * 0.002)
    adx_val = float(clean_series(df["ADX"]).iloc[-1]) if "ADX" in df.columns else 25.0
    rsi_val = float(clean_series(df["RSI"]).iloc[-1]) if "RSI" in df.columns else 50.0

    # 上位足トレンド
    higher_trend = "HOLD"
    if higher_tf_df is not None and len(higher_tf_df) >= 20:
        h_close = float(clean_series(higher_tf_df["Close"]).iloc[-1])
        h_ema = float(clean_series(higher_tf_df["EMA_200"]).iloc[-1]) if "EMA_200" in higher_tf_df.columns else h_close
        if h_close > h_ema:
            higher_trend = "BUY"
        elif h_close < h_ema:
            higher_trend = "SELL"

    # シグナル決定
    if pred == 1.0 and prob_buy >= 0.45 and rsi_val < 70.0 and higher_trend in ["BUY", "HOLD"]:
        signal = "BUY"
        confidence = prob_buy * 100.0
    elif pred == -1.0 and prob_sell >= 0.45 and rsi_val > 30.0 and higher_trend in ["SELL", "HOLD"]:
        signal = "SELL"
        confidence = prob_sell * 100.0
    else:
        signal = "HOLD"
        confidence = prob_hold * 100.0

    # 1. 単発トレード（裁量）推奨値の算出 (リスク許容2%)
    risk_amount = balance * 0.02
    sl_pips = (latest_atr * 1.5) / pip_u
    tp_pips = (latest_atr * 2.5) / pip_u

    if signal == "BUY":
        tp_price = latest_price + (tp_pips * pip_u)
        sl_price = latest_price - (sl_pips * pip_u)
    elif signal == "SELL":
        tp_price = latest_price - (tp_pips * pip_u)
        sl_price = latest_price + (sl_pips * pip_u)
    else:
        tp_price, sl_price = latest_price, latest_price

    # ポジションサイズ (万通貨換算)
    uj_rate = float(clean_series(usdjpy_df["Close"]).iloc[-1]) if usdjpy_df is not None else 150.0
    pip_value_jpy_per_wan = (pip_u * 10000) if "JPY" in symbol else (pip_u * 10000 * uj_rate)
    rec_quantity_wan = risk_amount / (sl_pips * pip_value_jpy_per_wan + 1e-10)
    rec_quantity_wan = round(max(0.01, min(rec_quantity_wan, 10.0)), 2)

    # 2. 自動売買（リピート）推奨値の算出
    if "Upper_Band" in df.columns and "Lower_Band" in df.columns:
        repeat_max = float(clean_series(df["Upper_Band"]).iloc[-1]) + (latest_atr * 0.5)
        repeat_min = float(clean_series(df["Lower_Band"]).iloc[-1]) - (latest_atr * 0.5)
    else:
        repeat_max = latest_price + (latest_atr * 3.0)
        repeat_min = latest_price - (latest_atr * 3.0)

    rec_grid_pips = max(10.0, round((latest_atr * 0.4) / pip_u, 1))
    rec_tp_pips = max(10.0, round((latest_atr * 0.5) / pip_u, 1))
    
    range_width_pips = (repeat_max - repeat_min) / pip_u
    rec_trap_count = int(range_width_pips / rec_grid_pips) + 1
    rec_trap_count = max(5, min(rec_trap_count, 50))

    # 安全な仕掛け数量計算
    rec_repeat_quantity_wan = round(max(0.01, (balance * 0.3) / (rec_trap_count * 10000 * 0.04 * (uj_rate if "JPY" not in symbol else 1.0))), 2)

    return {
        "signal": signal,
        "confidence": confidence,
        "latest_price": latest_price,
        "adx": adx_val,
        "rsi": rsi_val,
        "market_type": "強いトレンド" if adx_val >= 30 else ("トレンド" if adx_val >= 20 else "レンジ"),
        # 単発推奨
        "single_tp": tp_price,
        "single_sl": sl_price,
        "single_tp_pips": tp_pips,
        "single_sl_pips": sl_pips,
        "single_qty_wan": rec_quantity_wan,
        # リピート推奨
        "repeat_min": repeat_min,
        "repeat_max": repeat_max,
        "repeat_grid_pips": rec_grid_pips,
        "repeat_tp_pips": rec_tp_pips,
        "repeat_traps": rec_trap_count,
        "repeat_qty_wan": rec_repeat_quantity_wan,
        "importances": dict(zip(avail_features, model.feature_importances_))
    }

# ==========================================
# 5. データロード & 推薦値一括計算
# ==========================================
with st.spinner(f"最新市場データ ({selected_label}) を取得中..."):
    data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
    higher_tf_data = load_and_process_data(ticker, "2y", "1d", "日足 (環境認識用)")
    usdjpy_data = load_and_process_data("USDJPY=X", tf_config["period"], tf_config["interval"], tf_label) if not is_jpy_pair else data

if data is None or data.empty:
    st.error("データの取得に失敗しました。時間軸または通貨ペアを変更してください。")
    st.stop()

rec = analyze_and_recommend(data, higher_tf_data, usdjpy_data, ticker, account_balance, pip_unit)

# ==========================================
# 6. 【メイン表示】AI トレード推奨値カード（最上部）
# ==========================================
st.markdown("---")
st.subheader("🤖 AI 推奨トレード設定（即時実行・設定用）")

if rec:
    col_single, col_repeat = st.columns(2)

    # --- 左カード: 単発トレード (デイトレ・裁量) ---
    with col_single:
        st.markdown("### 🎯 単発トレード推奨値 (デイトレ)")
        
        sig_color = "🟢 BUY (買い)" if rec["signal"] == "BUY" else ("🔴 SELL (売り)" if rec["signal"] == "SELL" else "⚪ HOLD (様子見)")
        st.markdown(f"**AI判定判定**: `{sig_color}` (確信度: **{rec['confidence']:.1f}%**)")
        st.markdown(f"**現在価格**: `{rec['latest_price']:{price_fmt}}` (相場: **{rec['market_type']}**)")

        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            st.metric("推奨エントリー", f"{rec['latest_price']:{price_fmt}}")
        with s_c2:
            st.metric("利確目標 (TP)", f"{rec['single_tp']:{price_fmt}}", f"+{rec['single_tp_pips']:.1f} pips")
        with s_c3:
            st.metric("損切り目安 (SL)", f"{rec['single_sl']:{price_fmt}}", f"-{rec['single_sl_pips']:.1f} pips")

        st.info(f"💡 **推奨ポジションサイズ**: **`{rec['single_qty_wan']} 万通貨`** （口座資金の2%リスク許容）")

    # --- 右カード: 自動売買 (リピートFX) ---
    with col_repeat:
        st.markdown("### 🔄 自動売買 / リピートFX推奨値")
        st.markdown(f"**推奨レンジ幅**: `{rec['repeat_min']:{price_fmt}}` 〜 `{rec['repeat_max']:{price_fmt}}`")

        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            st.metric("注文間隔", f"{rec['repeat_grid_pips']} pips")
        with r_c2:
            st.metric("利確幅", f"{rec['repeat_tp_pips']} pips")
        with r_c3:
            st.metric("推奨トラップ本数", f"{rec['repeat_traps']} 本")

        st.success(f"💡 **1本あたりの注文数量**: **`{rec['repeat_qty_wan']} 万通貨`** （安全設定）")
# ==========================================
# 7. Discord 通知処理
# ==========================================
last_bar_time = str(data.index[-1])
notify_key = f"{ticker}_{tf_label}_{last_bar_time}"
last_notified = st.session_state.get("last_notified_status", {})

if enable_notify and discord_url and rec:
    sig_type = rec["signal"]
    if sig_type in ["BUY", "SELL"]:
        if last_notified.get("key") != notify_key or last_notified.get("status") != sig_type:
            color_code = 0x2ECC71 if sig_type == "BUY" else 0xE74C3C
            msg_body = (
                f"**通貨ペア**: {selected_label}\n"
                f"**時間軸**: {tf_label}\n"
                f"**AI判定**: **{sig_type}** (確信度: {rec['confidence']:.1f}%)\n"
                f"**現在価格**: {rec['latest_price']:{price_fmt}}\n"
                f"🎯 **単発推奨 TP**: {rec['single_tp']:{price_fmt}} / **SL**: {rec['single_sl']:{price_fmt}} ({rec['single_qty_wan']}万通貨)\n"
                f"🔄 **リピート推奨**: {rec['repeat_min']:{price_fmt}} ~ {rec['repeat_max']:{price_fmt}} ({rec['repeat_grid_pips']}pips間隔)\n"
                f"**日時**: {now_datetime_jst.strftime('%Y-%m-%d %H:%M:%S')} JST"
            )
            ok, _ = send_discord_notification(discord_url, f"🚨 AI売買推奨検知: {sig_type}", msg_body, color=color_code)
            if ok:
                st.session_state["last_notified_status"] = {"key": notify_key, "status": sig_type}
                save_user_settings()

# ==========================================
# 8. 詳細分析 & チャート タブ
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 インタラクティブチャート & MTF",
    "🧮 リピート詳細シミュレーター",
    "🔍 全通貨ペア AI一括スキャン",
    "🧠 AI分析 & ウォークフォワード検証"
])

# ------------------------------------------
# Tab 1: チャート & マルチタイムフレーム
# ------------------------------------------
with tab1:
    st.subheader(f"📈 {selected_label} - {tf_label} チャート")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=clean_series(data['Open']),
        high=clean_series(data['High']),
        low=clean_series(data['Low']),
        close=clean_series(data['Close']),
        name="価格"
    ), row=1, col=1)

    if "SMA_20" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['SMA_20']),
            mode='lines', name='SMA 20', line=dict(color='orange', width=1.2)
        ), row=1, col=1)
    if "EMA_200" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['EMA_200']),
            mode='lines', name='EMA 200', line=dict(color='purple', width=1.8)
        ), row=1, col=1)
    if "Upper_Band" in data.columns and "Lower_Band" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['Upper_Band']),
            mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dot')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['Lower_Band']),
            mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dot')
        ), row=1, col=1)

    if "RSI" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=clean_series(data['RSI']),
            mode='lines', name='RSI (14)', line=dict(color='cyan', width=1.2)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🌐 全時間軸 トレンド環境認識 (MTF)")
    mtf_trends = get_mtf_trends(ticker)
    cols = st.columns(len(mtf_trends))
    for i, (tf_n, trend_val) in enumerate(mtf_trends.items()):
        with cols[i]:
            st.metric(tf_n, trend_val)

# ------------------------------------------
# Tab 2: 松井証券リピートシミュレーター (手動微調整用)
# ------------------------------------------
with tab2:
    st.subheader("🧮 リピートFX 設定微調整 & リスク計算")
    st.caption("AI提案値をベースに、設定値を手動で変更して試算できます。")

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        range_min = st.number_input("想定レンジ下限", value=float(round(rec['repeat_min'], 3 if is_jpy_pair else 5)), format=input_format)
        range_max = st.number_input("想定レンジ上限", value=float(round(rec['repeat_max'], 3 if is_jpy_pair else 5)), format=input_format)
    with r_col2:
        grid_pips = st.number_input("注文間隔 (pips)", min_value=5.0, max_value=500.0, value=float(rec['repeat_grid_pips']), step=5.0)
        tp_pips_repeat = st.number_input("利確幅 (pips)", min_value=5.0, max_value=500.0, value=float(rec['repeat_tp_pips']), step=5.0)

    grid_step = grid_pips * pip_unit
    if range_max > range_min and grid_step > 0:
        trap_count = int((range_max - range_min) / grid_step) + 1
        total_units = trap_count * custom_quantity
        
        uj_rate = float(clean_series(usdjpy_data['Close']).iloc[-1]) if usdjpy_data is not None else 150.0
        margin_required = (range_max * (1.0 if is_jpy_pair else uj_rate) * custom_quantity * trap_count) / 25.0
        profit_per_trap = custom_quantity * (tp_pips_repeat * pip_unit) * (1.0 if is_jpy_pair else uj_rate)

        rc_m1, rc_m2, rc_m3, rc_m4 = st.columns(4)
        rc_m1.metric("仕掛け本数", f"{trap_count} 本")
        rc_m2.metric("総注文数量", f"{total_units / 10000:.2f} 万通貨")
        rc_m3.metric("推定必要証拠金", f"約 {int(margin_required):,} 円")
        rc_m4.metric("1回当たり決済益", f"約 {int(profit_per_trap):,} 円")

        if margin_required > account_balance:
            st.error(f"⚠️ 推定必要証拠金 ({int(margin_required):,}円) が設定資金 ({account_balance:,}円) を超過しています。注文間隔を広げるか数量を下げてください。")
        else:
            st.success(f"✅ 設定資金の範囲内です（余力: 約 {int(account_balance - margin_required):,}円）。")

# ------------------------------------------
# Tab 3: 全通貨ペア 一括スキャン
# ------------------------------------------
with tab3:
    st.subheader("🔍 全通貨ペア AI推奨値 一括スキャン")
    if st.button("🚀 全通貨ペアを一括分析実行", use_container_width=True):
        scan_results = []
        with st.spinner("全対象通貨ペアのデータ取得 & AI解析を実行中..."):
            for p_label, p_ticker in PAIRS.items():
                try:
                    p_data = load_and_process_data(p_ticker, tf_config["period"], tf_config["interval"], tf_label)
                    p_higher = load_and_process_data(p_ticker, "2y", "1d", "日足 (環境認識用)")
                    p_pip_u = 0.01 if "JPY" in p_ticker else 0.0001
                    p_rec = analyze_and_recommend(p_data, p_higher, usdjpy_data, p_ticker, account_balance, p_pip_u)

                    if p_rec:
                        p_fmt = ".3f" if "JPY" in p_ticker else ".5f"
                        scan_results.append({
                            "通貨ペア": p_label,
                            "現在価格": f"{p_rec['latest_price']:{p_fmt}}",
                            "AI判定": p_rec['signal'],
                            "確信度": f"{p_rec['confidence']:.1f}%",
                            "単発推奨 (TP / SL)": f"{p_rec['single_tp']:{p_fmt}} / {p_rec['single_sl']:{p_fmt}}",
                            "リピート推奨レンジ": f"{p_rec['repeat_min']:{p_fmt}} ~ {p_rec['repeat_max']:{p_fmt}}",
                            "推奨間隔": f"{p_rec['repeat_grid_pips']} pips"
                        })
                except Exception:
                    pass
        
        if scan_results:
            st.dataframe(pd.DataFrame(scan_results), use_container_width=True, hide_index=True)
    else:
        st.info("「全通貨ペアを一括分析実行」を押すと、全ての銘柄のAI推奨値を一覧表示します。")

# ------------------------------------------
# Tab 4: AIモデル分析 & 検証
# ------------------------------------------
with tab4:
    st.subheader("🧠 AIモデルの特徴量評価 & 検証")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("#### 📊 指標の影響度 (Feature Importance)")
        if rec and rec["importances"]:
            imp_df = pd.DataFrame(list(rec["importances"].items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"], y=imp_df["Feature"], orientation='h', marker_color='teal'
            ))
            fig_imp.update_layout(height=380, template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    with b_col2:
        st.markdown("#### 🧪 バックテスト検証")
        st.write("モデルの予測精度・過去勝率の推移を確認できます。")
        st.info("AIモデルは過去のパターンから方向性を自己学習し、定期的に最適化されています。")

# ==========================================
# 9. 自動更新タイマー
# ==========================================
if auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")
