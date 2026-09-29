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
# 0. 画面基本設定 & CSSデザイン定義
# ==========================================
st.set_page_config(
    page_title="AI FX 環境認識 & リピートアナライザー Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1280px; }
    
    label[data-testid="stWidgetLabel"] p { font-size: 1.05rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 1.0rem !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 800 !important; }
    [data-testid="stCaptionContainer"], .stCaption p { font-size: 0.95rem !important; font-weight: 600 !important; }
    
    .status-badge-buy { background-color: #15803d; color: #ffffff; border: 1px solid #16a34a; padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 1.15rem; display: inline-block; }
    .status-badge-sell { background-color: #b91c1c; color: #ffffff; border: 1px solid #dc2626; padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 1.15rem; display: inline-block; }
    .status-badge-hold { background-color: #854d0e; color: #ffffff; border: 1px solid #ca8a04; padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 1.15rem; display: inline-block; }
    
    .param-box { background-color: rgba(30, 41, 59, 0.8) !important; border-left: 6px solid #3b82f6; padding: 18px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 1.05rem; line-height: 1.8; color: #f8fafc !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .param-box b.label-title { color: #94a3b8 !important; }
    .param-box code { font-size: 1.1rem !important; font-weight: 700 !important; padding: 2px 8px !important; background-color: rgba(51, 65, 85, 0.8) !important; color: #38bdf8 !important; border: 1px solid #475569; border-radius: 4px; }
    .param-box-buy { border-left-color: #22c55e !important; }
    .param-box-sell { border-left-color: #ef4444 !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 18px; font-size: 1.0rem !important; font-weight: 700; border-radius: 6px 6px 0 0; }
    .stTabs [aria-selected="true"] { color: #0284c7 !important; border-bottom-color: #0284c7 !important; }
</style>
""",
    unsafe_allow_html=True,
)

FEATURE_COLUMNS = [
    "Return_1",
    "Return_5",
    "Dev_SMA20",
    "Dev_EMA200",
    "Dev_EMA20_200",
    "Vol_Ratio",
    "RSI",
    "RSI_Diff",
    "MACD_Hist_Ratio",
    "BB_PctB",
    "ADX",
    "ATR_Ratio",
    "Upper_Wick_Ratio",
    "Lower_Wick_Ratio",
    "Stoch_K",
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
            pass
    return DEFAULT_SETTINGS.copy()


def save_user_settings():
    settings = {
        "account_balance": st.session_state.get(
            "account_balance", DEFAULT_SETTINGS["account_balance"]
        ),
        "quantity_wan": st.session_state.get(
            "quantity_wan", DEFAULT_SETTINGS["quantity_wan"]
        ),
        "discord_url": st.session_state.get(
            "discord_url", DEFAULT_SETTINGS["discord_url"]
        ),
        "enable_notify": st.session_state.get(
            "enable_notify", DEFAULT_SETTINGS["enable_notify"]
        ),
        "auto_refresh": st.session_state.get(
            "auto_refresh", DEFAULT_SETTINGS["auto_refresh"]
        ),
        "refresh_interval": st.session_state.get(
            "refresh_interval", DEFAULT_SETTINGS["refresh_interval"]
        ),
        "selected_pair_label": st.session_state.get(
            "selected_pair_label", DEFAULT_SETTINGS["selected_pair_label"]
        ),
        "selected_tf_label": st.session_state.get(
            "selected_tf_label", DEFAULT_SETTINGS["selected_tf_label"]
        ),
        "last_notified_status": st.session_state.get(
            "last_notified_status", DEFAULT_SETTINGS["last_notified_status"]
        ),
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
        "embeds": [
            {
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
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
# 2. 事前警戒イベント＆タイムゾーン計算
# ==========================================
def get_upcoming_market_events(now_jst, is_summer):
    """現在のJST時刻から、直近の警戒イベント（指標発表・市場オープン）までの残り時間を算出"""
    h, m = now_jst.hour, now_jst.minute
    current_total_min = h * 60 + m

    london_open = 16 * 60 if is_summer else 17 * 60
    us_econ_indicator = (21 * 60 + 30) if is_summer else (22 * 60 + 30)
    ny_open = (22 * 60 + 30) if is_summer else (23 * 60 + 30)
    early_morning = 3 * 60

    events = [
        {
            "name": "欧州市場オープン (ロンドン)",
            "time_str": "16:00" if is_summer else "17:00",
            "min": london_open,
        },
        {
            "name": "米国 主要経済指標発表",
            "time_str": "21:30" if is_summer else "22:30",
            "min": us_econ_indicator,
        },
        {
            "name": "米国市場オープン (NY)",
            "time_str": "22:30" if is_summer else "23:30",
            "min": ny_open,
        },
        {
            "name": "早朝・広スプレッド警戒帯",
            "time_str": "03:00",
            "min": early_morning,
        },
    ]

    upcoming = []
    for ev in events:
        diff = ev["min"] - current_total_min
        if diff < -120:
            diff += 24 * 60

        upcoming.append(
            {
                "event": ev["name"],
                "time": ev["time_str"],
                "left_min": diff,
            }
        )

    upcoming.sort(key=lambda x: x["left_min"])
    return upcoming


# ==========================================
# 3. データ取得 & インジケーター計算エンジン
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    df = pd.DataFrame()
    try:
        df = yf.download(
            symbol, period=period, interval=interval, progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if isinstance(col, tuple) else col for col in df.columns
            ]
    except Exception:
        pass

    if not df.empty and "4時間足" in tf_name and interval == "1h":
        try:
            tz_before = df.index.tz
            df = (
                df.resample("4h", closed="left", label="left")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna()
            )
            if tz_before is not None and df.index.tz is None:
                df.index = df.index.tz_localize(tz_before)
        except Exception:
            pass

    if df.empty or len(df) < 50:
        return None

    try:
        c_series = clean_series(df["Close"])
        h_series = clean_series(df["High"])
        l_series = clean_series(df["Low"])
        o_series = clean_series(df["Open"])

        new_cols = {}
        new_cols["SMA_20"] = c_series.rolling(window=20).mean()
        new_cols["SMA_50"] = c_series.rolling(window=50).mean()
        new_cols["EMA_20"] = c_series.ewm(span=20, adjust=False).mean()
        new_cols["EMA_200"] = c_series.ewm(span=200, adjust=False).mean()

        high_low = h_series - l_series
        new_cols["ATR"] = high_low.rolling(window=14).mean()
        new_cols["ATR_SMA20"] = new_cols["ATR"].rolling(window=20).mean()
        new_cols["ATR_Ratio"] = new_cols["ATR"] / (
            new_cols["ATR_SMA20"] + 1e-10
        )

        total_range = high_low + 1e-10
        open_close_max = pd.concat([o_series, c_series], axis=1).max(axis=1)
        open_close_min = pd.concat([o_series, c_series], axis=1).min(axis=1)
        new_cols["Upper_Wick_Ratio"] = (
            h_series - open_close_max
        ) / total_range
        new_cols["Lower_Wick_Ratio"] = (
            open_close_min - l_series
        ) / total_range

        new_cols["Return_1"] = c_series.diff(1) / (c_series.shift(1) + 1e-10)
        new_cols["Return_5"] = c_series.diff(5) / (c_series.shift(5) + 1e-10)

        new_cols["Dev_SMA20"] = (c_series - new_cols["SMA_20"]) / (
            new_cols["SMA_20"] + 1e-10
        )
        new_cols["Dev_EMA200"] = (c_series - new_cols["EMA_200"]) / (
            new_cols["EMA_200"] + 1e-10
        )
        new_cols["Dev_EMA20_200"] = (
            new_cols["EMA_20"] - new_cols["EMA_200"]
        ) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Vol_Ratio"] = new_cols["ATR"] / (c_series + 1e-10)

        delta = c_series.diff()
        gain = (
            delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
        )
        loss = (
            (-delta.where(delta < 0, 0.0))
            .ewm(alpha=1 / 14, adjust=False)
            .mean()
        )
        rs = gain / (loss + 1e-10)
        new_cols["RSI"] = 100.0 - (100.0 / (1.0 + rs))
        new_cols["RSI_Diff"] = new_cols["RSI"].diff(1)

        low_14 = l_series.rolling(window=14).min()
        high_14 = h_series.rolling(window=14).max()
        new_cols["Stoch_K"] = (
            100.0 * (c_series - low_14) / ((high_14 - low_14) + 1e-10)
        )

        ema12 = c_series.ewm(span=12, adjust=False).mean()
        ema26 = c_series.ewm(span=26, adjust=False).mean()
        new_cols["MACD"] = ema12 - ema26
        new_cols["MACD_Signal"] = (
            new_cols["MACD"].ewm(span=9, adjust=False).mean()
        )
        new_cols["MACD_Hist"] = new_cols["MACD"] - new_cols["MACD_Signal"]
        new_cols["MACD_Hist_Ratio"] = new_cols["MACD_Hist"] / (
            c_series + 1e-10
        )

        std20 = c_series.rolling(window=20).std()
        new_cols["Upper_Band"] = new_cols["SMA_20"] + (std20 * 2)
        new_cols["Lower_Band"] = new_cols["SMA_20"] - (std20 * 2)
        new_cols["BB_Width"] = (
            new_cols["Upper_Band"] - new_cols["Lower_Band"]
        ) / (new_cols["SMA_20"] + 1e-10)
        new_cols["BB_PctB"] = (c_series - new_cols["Lower_Band"]) / (
            (new_cols["Upper_Band"] - new_cols["Lower_Band"]) + 1e-10
        )

        tr = (
            pd.concat(
                [
                    high_low,
                    (h_series - c_series.shift(1)).abs(),
                    (l_series - c_series.shift(1)).abs(),
                ],
                axis=1,
            )
            .max(axis=1)
        )
        up_move = h_series - h_series.shift(1)
        down_move = l_series.shift(1) - l_series
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where(
            (down_move > up_move) & (down_move > 0), down_move, 0.0
        )
        atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
        plus_di = (
            100
            * pd.Series(plus_dm, index=df.index)
            .ewm(alpha=1 / 14, adjust=False)
            .mean()
            / (atr14 + 1e-10)
        )
        minus_di = (
            100
            * pd.Series(minus_dm, index=df.index)
            .ewm(alpha=1 / 14, adjust=False)
            .mean()
            / (atr14 + 1e-10)
        )
        sum_di = (plus_di + minus_di).replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / sum_di
        new_cols["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(25.0)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        f_high = pd.concat(
            [h_series.shift(-1), h_series.shift(-2), h_series.shift(-3)],
            axis=1,
        )
        f_low = pd.concat(
            [l_series.shift(-1), l_series.shift(-2), l_series.shift(-3)],
            axis=1,
        )

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
# ==========================================
# 4. 機械学習（AI）モデルの学習と判定エンジン
# ==========================================
def train_rf_model(df):
    """RandomForestを使用したAI分類モデルの学習関数"""
    if df is None or len(df) < 100:
        return None, 0.0

    train_data = df.dropna(subset=["Target"])
    if len(train_data) < 50:
        return None, 0.0

    X = train_data[FEATURE_COLUMNS]
    y = train_data["Target"].astype(int)

    # 単一クラスのみの場合は学習をスキップ
    if len(np.unique(y)) < 2:
        return None, 0.0

    # 時系列分割を考慮（直近20%を検証用データに固定）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy


def predict_ai_signal(model, latest_df):
    """学習済みモデルを用いて最新足のシグナルと信頼度を予測"""
    if model is None or latest_df is None or latest_df.empty:
        return "HOLD", 0.0, 0.0, 0.0

    latest_features = latest_df[FEATURE_COLUMNS].iloc[-1:]

    if latest_features.isnull().any().any():
        return "HOLD", 0.0, 0.0, 0.0

    probs = model.predict_proba(latest_features)[0]
    classes = model.classes_

    prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
    buy_prob = prob_dict.get(1, 0.0)
    sell_prob = prob_dict.get(-1, 0.0)
    hold_prob = prob_dict.get(0, 0.0)

    # シグナル判定閾値（勝率重視：55%以上で発火）
    if buy_prob >= 0.55 and buy_prob > sell_prob:
        signal = "BUY"
        confidence = buy_prob
    elif sell_prob >= 0.55 and sell_prob > buy_prob:
        signal = "SELL"
        confidence = sell_prob
    else:
        signal = "HOLD"
        confidence = hold_prob

    return signal, buy_prob, sell_prob, confidence


# ==========================================
# 5. リピート系IFDOCO注文 バックテストシミュレータ
# ==========================================
def run_repeat_backtest(
    df, trap_width_pips, tp_pips, sl_pips, lot_wan, is_jpy_pair=True
):
    """グリッド型リピートトレードの簡易バックテスト計算"""
    if df is None or len(df) < 50:
        return None

    pips_unit = 0.01 if is_jpy_pair else 0.0001
    trap_dist = trap_width_pips * pips_unit
    tp_dist = tp_pips * pips_unit
    sl_dist = sl_pips * pips_unit if sl_pips > 0 else None
    units = lot_wan * 10000

    close_prices = df["Close"].values
    high_prices = df["High"].values
    low_prices = df["Low"].values
    timestamps = df.index

    positions = []
    trade_history = []
    total_realized_pnl = 0.0
    equity_curve = []

    for i in range(len(df)):
        curr_time = timestamps[i]
        curr_high = high_prices[i]
        curr_low = low_prices[i]
        curr_close = close_prices[i]

        # 1. 保有ポジションの利確・損切り判定
        remaining_positions = []
        for pos in positions:
            entry = pos["entry_price"]
            # 利確 (TP)
            if curr_high >= entry + tp_dist:
                pnl = tp_dist * units
                total_realized_pnl += pnl
                trade_history.append(
                    {
                        "time": curr_time,
                        "type": "BUY_TP",
                        "entry": entry,
                        "exit": entry + tp_dist,
                        "pnl": pnl,
                    }
                )
            # 損切り (SL)
            elif sl_dist and curr_low <= entry - sl_dist:
                pnl = -sl_dist * units
                total_realized_pnl += pnl
                trade_history.append(
                    {
                        "time": curr_time,
                        "type": "BUY_SL",
                        "entry": entry,
                        "exit": entry - sl_dist,
                        "pnl": pnl,
                    }
                )
            else:
                remaining_positions.append(pos)

        positions = remaining_positions

        # 2. 新規エントリー判定（トラップ間隔の条件充足時）
        if len(positions) < 10:  # 最大10ポジションに制限
            if not positions:
                positions.append({"entry_price": curr_close, "type": "BUY"})
            else:
                last_entry = positions[-1]["entry_price"]
                if abs(curr_close - last_entry) >= trap_dist:
                    positions.append({"entry_price": curr_close, "type": "BUY"})

        # 含み損益および評価残高の記録
        unrealized_pnl = sum(
            (curr_close - p["entry_price"]) * units for p in positions
        )
        total_equity = total_realized_pnl + unrealized_pnl
        equity_curve.append(total_equity)

    equity_series = pd.Series(equity_curve, index=timestamps)
    max_equity = equity_series.cummax()
    drawdown = equity_series - max_equity
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    win_trades = [t for t in trade_history if t["pnl"] > 0]
    win_rate = (
        (len(win_trades) / len(trade_history) * 100) if trade_history else 0.0
    )

    return {
        "total_pnl": total_realized_pnl,
        "final_equity": equity_curve[-1] if equity_curve else 0.0,
        "max_drawdown": max_dd,
        "total_trades": len(trade_history),
        "win_rate": win_rate,
        "active_positions": len(positions),
        "trade_history": pd.DataFrame(trade_history),
        "equity_curve": equity_series,
    }


# ==========================================
# 6. マルチタイムフレーム環境認識 & AI総合判定
# ==========================================
def analyze_mtf_environment(df_daily, df_4h, df_15m):
    """日足・4時間足・15分足の多角分析による環境認識ロジック"""
    res = {
        "daily_trend": "レンジ",
        "h4_trend": "レンジ",
        "m15_trend": "レンジ",
        "overall_status": "HOLD",
        "risk_score": 0,
        "recommended_strategy": "静観・ボラティリティ低下待ち",
        "grid_spacing": 20,
        "tp_pips": 20,
        "sl_pips": 100,
    }

    if df_daily is None or df_4h is None or df_15m is None:
        return res

    # 1. 日足トレンド評価
    d_close = df_daily["Close"].iloc[-1]
    d_ema20 = df_daily["EMA_20"].iloc[-1]
    d_ema200 = df_daily["EMA_200"].iloc[-1]
    d_adx = df_daily["ADX"].iloc[-1]

    if d_close > d_ema20 > d_ema200 and d_adx > 20:
        res["daily_trend"] = "強気上昇トレンド"
    elif d_close < d_ema20 < d_ema200 and d_adx > 20:
        res["daily_trend"] = "強気下降トレンド"
    elif d_close > d_ema200:
        res["daily_trend"] = "緩やかな上昇（レンジ傾向）"
    else:
        res["daily_trend"] = "緩やかな下降（レンジ傾向）"

    # 2. 4時間足トレンド評価
    h4_close = df_4h["Close"].iloc[-1]
    h4_ema20 = df_4h["EMA_20"].iloc[-1]
    h4_ema200 = df_4h["EMA_200"].iloc[-1]
    h4_rsi = df_4h["RSI"].iloc[-1]

    if h4_close > h4_ema20 > h4_ema200:
        res["h4_trend"] = "上昇トレンド"
    elif h4_close < h4_ema20 < h4_ema200:
        res["h4_trend"] = "下降トレンド"
    else:
        res["h4_trend"] = "ボックスレンジ"

    # 3. 15分足トレンド & モメンタム
    m15_atr = df_15m["ATR"].iloc[-1]
    m15_bb_width = df_15m["BB_Width"].iloc[-1]

    # リスクスコア計算
    risk = 0
    if m15_bb_width > 0.008:
        risk += 30  # ボラティリティ急増警戒
    if h4_rsi > 70 or h4_rsi < 30:
        risk += 20  # 加熱警戒

    if "上昇" in res["daily_trend"] and "上昇" in res["h4_trend"]:
        res["overall_status"] = "BUY (買い推奨)"
        res["recommended_strategy"] = "買いリピート展開 / 押し目買い運用"
        res["grid_spacing"] = max(15, int(m15_atr * 100 * 0.8))
        res["tp_pips"] = int(res["grid_spacing"] * 1.2)
        res["sl_pips"] = int(res["grid_spacing"] * 5)
    elif "下降" in res["daily_trend"] and "下降" in res["h4_trend"]:
        res["overall_status"] = "SELL (売り推奨)"
        res["recommended_strategy"] = "売りリピート展開 / 戻り売り運用"
        res["grid_spacing"] = max(15, int(m15_atr * 100 * 0.8))
        res["tp_pips"] = int(res["grid_spacing"] * 1.2)
        res["sl_pips"] = int(res["grid_spacing"] * 5)
    else:
        res["overall_status"] = "HOLD (様子見・レンジ運用)"
        res["recommended_strategy"] = "レンジ内逆張り または 調整待ち静観"
        res["grid_spacing"] = max(20, int(m15_atr * 100))
        res["tp_pips"] = res["grid_spacing"]
        res["sl_pips"] = int(res["grid_spacing"] * 4)

    res["risk_score"] = min(100, risk)
    return res
# ==========================================
# 7. Plotly インタラクティブチャート描画エンジン
# ==========================================
def create_chart_figure(df, title="チャート"):
    """ローソク足、移動平均線、ボリンジャーバンド、MACDを組み合わせたサブプロット描画"""
    if df is None or df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(title, "MACD & ヒストグラム"),
        row_width=[0.25, 0.75],
    )

    # 1. ローソク足
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="価格",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    # 2. 移動平均線 (EMA20, EMA200)
    if "EMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_20"],
                line=dict(color="#3b82f6", width=1.5),
                name="EMA 20",
            ),
            row=1,
            col=1,
        )

    if "EMA_200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_200"],
                line=dict(color="#f59e0b", width=2),
                name="EMA 200",
            ),
            row=1,
            col=1,
        )

    # 3. ボリンジャーバンド
    if "Upper_Band" in df.columns and "Lower_Band" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Upper_Band"],
                line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
                name="BB Upper",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Lower_Band"],
                line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.05)",
                name="BB Lower",
            ),
            row=1,
            col=1,
        )

    # 4. MACD & ヒストグラム (サブチャート)
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                line=dict(color="#38bdf8", width=1.5),
                name="MACD",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                line=dict(color="#f43f5e", width=1.5),
                name="Signal",
            ),
            row=2,
            col=1,
        )

        colors = np.where(df["MACD_Hist"] >= 0, "#22c55e", "#ef4444")
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Hist"],
                marker_color=colors,
                name="Hist",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    return fig


# ==========================================
# 8. Streamlit メインアプリ & 画面レイアウト
# ==========================================
def main():
    st.title("📈 AI FX 環境認識 & リピートアナライザー Pro")

    # サイドバー設定エリア
    with st.sidebar:
        st.header("⚙️ 動作コントロール")

        pair_dict = {
            "米ドル / 円 (USD/JPY)": "USDJPY=X",
            "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
            "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
            "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
            "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
        }

        selected_pair_label = st.selectbox(
            "通貨ペアの選択",
            list(pair_dict.keys()),
            key="selected_pair_label",
            on_change=save_user_settings,
        )
        selected_pair_symbol = pair_dict[selected_pair_label]
        is_jpy = "JPY" in selected_pair_symbol

        tf_dict = {
            "15分足 (デイトレエントリー用)": ("60d", "15m", "15分足"),
            "1時間足 (スイング・トレンド確認)": ("730d", "1h", "1時間足"),
            "4時間足 (中期トレンド分析)": ("730d", "1h", "4時間足"),
        }

        selected_tf_label = st.selectbox(
            "基準タイムフレーム",
            list(tf_dict.keys()),
            key="selected_tf_label",
            on_change=save_user_settings,
        )
        period, interval, tf_name = tf_dict[selected_tf_label]

        st.markdown("---")
        st.subheader("💰 資金管理 & トレード設定")

        account_balance = st.number_input(
            "運用元本 (円)",
            min_value=10000,
            max_value=100000000,
            step=50000,
            key="account_balance",
            on_change=save_user_settings,
        )

        quantity_wan = st.number_input(
            "注文数量 (万通貨)",
            min_value=0.01,
            max_value=100.0,
            step=0.05,
            key="quantity_wan",
            on_change=save_user_settings,
        )

        st.markdown("---")
        st.subheader("🔔 Discord通知設定")
        discord_url = st.text_input(
            "Webhook URL",
            key="discord_url",
            on_change=save_user_settings,
            type="password",
        )
        enable_notify = st.checkbox(
            "シグナル変化時に自動通知",
            key="enable_notify",
            on_change=save_user_settings,
        )

        if st.button("テスト通知を送信"):
            ok, msg = send_discord_notification(
                discord_url,
                "🔔 FXアナライザー テスト通知",
                "Discordとの連携が正常に動作しています。",
            )
            if ok:
                st.success("送信完了")
            else:
                st.error(f"送信失敗: {msg}")

        st.markdown("---")
        auto_refresh = st.checkbox(
            "自動更新を有効化",
            key="auto_refresh",
            on_change=save_user_settings,
        )
        refresh_interval = st.slider(
            "更新間隔 (秒)",
            min_value=60,
            max_value=600,
            step=30,
            key="refresh_interval",
            on_change=save_user_settings,
        )

    # 自動再読み込みタイマー設定
    if st.session_state.get("auto_refresh", False):
        st_autorefresh(
            interval=st.session_state.get("refresh_interval", 180) * 1000,
            key="data_autorefresh",
        )

    # マルチタイムフレーム用データの読み込み
    with st.spinner("リアルタイムデータ取得 & 分析中..."):
        df_daily = load_and_process_data(
            selected_pair_symbol, "730d", "1d", "日足"
        )
        df_4h = load_and_process_data(
            selected_pair_symbol, "730d", "1h", "4時間足"
        )
        df_main = load_and_process_data(
            selected_pair_symbol, period, interval, tf_name
        )

    if df_main is None or df_main.empty:
        st.error(
            "データの取得に失敗しました。市場が休業中か、一時的な通信エラーの可能性があります。"
        )
        return

    # タイムゾーン変換 (JST)
    now_utc = datetime.now(timezone.utc)
    now_jst = now_utc.astimezone(ZoneInfo("Asia/Tokyo"))
    is_summer = True  # 夏時間判定ロジック用（簡略化）

    # 環境認識 & AI分析の実行
    mtf_res = analyze_mtf_environment(df_daily, df_4h, df_main)
    rf_model, accuracy = train_rf_model(df_main)
    signal, buy_p, sell_p, conf = predict_ai_signal(rf_model, df_main)

    # Discord自動通知チェック
    if enable_notify and discord_url:
        last_status = st.session_state.get("last_notified_status", {})
        pair_last_sig = last_status.get(selected_pair_label, "")

        if signal != pair_last_sig and signal in ["BUY", "SELL"]:
            msg = (
                f"**【通貨ペア】**: {selected_pair_label}\n"
                f"**【新シグナル】**: {signal} (信頼度: {conf * 100:.1f}%)\n"
                f"**【推奨戦略】**: {mtf_res['recommended_strategy']}\n"
                f"**【トラップ幅】**: {mtf_res['grid_spacing']} pips\n"
                f"**【現在価格】**: {df_main['Close'].iloc[-1]:.3f}"
            )
            color = 0x22C55E if signal == "BUY" else 0xEF4444
            ok, _ = send_discord_notification(
                discord_url,
                f"🚨 【{signal}シグナル発火】{selected_pair_label}",
                msg,
                color,
            )
            if ok:
                st.session_state["last_notified_status"][
                    selected_pair_label
                ] = signal
                save_user_settings()

    # ==========================================
    # 9. メインパネルレイアウト描画
    # ==========================================

    # 1. 警戒イベント通知バー
    upcoming_events = get_upcoming_market_events(now_jst, is_summer)
    if upcoming_events:
        next_ev = upcoming_events[0]
        if next_ev["left_min"] <= 60 and next_ev["left_min"] >= 0:
            st.warning(
                f"⚠️ **市場警戒**: 60分以内に 【{next_ev['event']}】 ({next_ev['time']}) が控えています。突発的な価格変動にご注意ください。"
            )

    # 2. サマリー指標 (Metrics)
    col1, col2, col3, col4 = st.columns(4)
    curr_price = df_main["Close"].iloc[-1]
    prev_price = df_main["Close"].iloc[-2]
    price_diff = curr_price - prev_price

    with col1:
        st.metric(
            "現在価格",
            f"{curr_price:.3f}" if is_jpy else f"{curr_price:.5f}",
            delta=f"{price_diff:+.3f}" if is_jpy else f"{price_diff:+.5f}",
        )

    with col2:
        st.metric(
            "日足トレンド",
            mtf_res["daily_trend"],
        )

    with col3:
        st.metric(
            "AI 判定シグナル",
            f"{signal} ({conf * 100:.1f}%)",
            delta=f"勝率予想 {accuracy * 100:.1f}%",
        )

    with col4:
        st.metric(
            "推奨トラップ幅",
            f"{mtf_res['grid_spacing']} pips",
            delta=f"リスクスコア: {mtf_res['risk_score']}/100",
        )

    st.markdown("---")

    # 3. タブコンテンツ (チャート・分析・バックテスト)
    tab1, tab2, tab3 = st.tabs(
        [
            "📊 インタラクティブ・チャート",
            "🔍 MTF環境認識 & 推奨パラメーター",
            "🧪 リピート系IFDOCO バックテスト",
        ]
    )

    with tab1:
        fig = create_chart_figure(
            df_main, f"{selected_pair_label} - {selected_tf_label}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🎯 推奨エントリーパラメータ & 方針")

        box_class = (
            "param-box-buy"
            if "BUY" in mtf_res["overall_status"]
            else (
                "param-box-sell"
                if "SELL" in mtf_res["overall_status"]
                else ""
            )
        )

        st.markdown(
            f"""
        <div class="param-box {box_class}">
            <b>総合環境判定:</b> <code>{mtf_res['overall_status']}</code><br>
            <b>推奨運用方針:</b> <code>{mtf_res['recommended_strategy']}</code><br><br>
            <b>【リピート注文 推奨値】</b><br>
            ・<b>注文仕掛け幅 (Trap Width):</b> <code>{mtf_res['grid_spacing']} pips</code><br>
            ・<b>利確幅 (Take Profit):</b> <code>{mtf_res['tp_pips']} pips</code><br>
            ・<b>損切り幅 (Stop Loss):</b> <code>{mtf_res['sl_pips']} pips</code><br>
            ・<b>推奨発注数量:</b> <code>{quantity_wan} 万通貨 / トラップ</code>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🌐 マルチタイムフレーム詳細")
            st.write(f"- **日足 トレンド:** {mtf_res['daily_trend']}")
            st.write(f"- **4時間足 トレンド:** {mtf_res['h4_trend']}")
            st.write(
                f"- **15分足 ATR:** {df_main['ATR'].iloc[-1]:.3f}"
                if is_jpy
                else f"- **15分足 ATR:** {df_main['ATR'].iloc[-1]:.5f}"
            )

        with col_b:
            st.markdown("#### 🤖 AIモデル確率内訳")
            st.write(f"- **BUY 確率:** {buy_p * 100:.1f}%")
            st.write(f"- **SELL 確率:** {sell_p * 100:.1f}%")
            st.write(f"- **検証データ精度 (Accuracy):** {accuracy * 100:.1f}%")

    with tab3:
        st.subheader("🧪 直近データを用いたリピート運用シミュレーション")

        bt_col1, bt_col2, bt_col3 = st.columns(3)
        with bt_col1:
            trap_input = st.number_input(
                "テスト仕掛け幅 (pips)",
                value=int(mtf_res["grid_spacing"]),
                step=5,
            )
        with bt_col2:
            tp_input = st.number_input(
                "テスト利確幅 (pips)",
                value=int(mtf_res["tp_pips"]),
                step=5,
            )
        with bt_col3:
            sl_input = st.number_input(
                "テスト損切り幅 (0で損切無)",
                value=int(mtf_res["sl_pips"]),
                step=10,
            )

        bt_res = run_repeat_backtest(
            df_main, trap_input, tp_input, sl_input, quantity_wan, is_jpy
        )

        if bt_res:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("総実現損益", f"{bt_res['total_pnl']:,.0f} 円")
            m2.metric("総総トレード数", f"{bt_res['total_trades']} 回")
            m3.metric("勝率", f"{bt_res['win_rate']:.1f} %")
            m4.metric("最大ドローダウン", f"{bt_res['max_drawdown']:,.0f} 円")

            if not bt_res["equity_curve"].empty:
                st.markdown("#### 📈 資産推移（エクイティカーブ）")
                st.line_chart(bt_res["equity_curve"])


if __name__ == "__main__":
    main()
