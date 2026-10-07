import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

# Streamlitのページ設定（必ず最初のStreamlitコマンドとして実行）
st.set_page_config(
    page_title="Pro FX Analyzer & Signal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ==========================================
# 0. 設定ファイルの読み書き & 永続化処理
# ==========================================
SETTINGS_FILE = "user_settings.json"

def load_settings():
    """ローカルのJSONファイルから設定を読み込み"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "account_balance": 500000,
        "quantity_wan": 0.10,
        "auto_refresh": False,
        "risk_percent": 1.0,
        "rr_ratio": 1.5,
        "ranges": {}
    }

def save_settings():
    """現在の Session State を JSON ファイルへ保存"""
    settings = {
        "account_balance": st.session_state.get("account_balance", 500000),
        "quantity_wan": st.session_state.get("quantity_wan", 0.10),
        "auto_refresh": st.session_state.get("auto_refresh", False),
        "risk_percent": st.session_state.get("risk_percent", 1.0),
        "rr_ratio": st.session_state.get("rr_ratio", 1.5),
        "ranges": st.session_state.get("ranges", {})
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"設定の保存に失敗しました: {e}")

# 初期設定のロード
saved_config = load_settings()

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 800 !important; }
    
    .badge-buy { background-color: #16a34a; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.0rem; display: inline-block; }
    .badge-sell { background-color: #dc2626; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.0rem; display: inline-block; }
    .badge-wait { background-color: #475569; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.0rem; display: inline-block; }
    
    .param-box { background-color: rgba(30, 41, 59, 0.8) !important; border-left: 5px solid #3b82f6; padding: 14px; border-radius: 6px; font-family: monospace; line-height: 1.8; color: #f8fafc !important; }
    .param-box code { font-size: 0.95rem !important; font-weight: 700 !important; color: #38bdf8 !important; background-color: rgba(51, 65, 85, 0.9) !important; padding: 2px 6px; border-radius: 4px; }
    
    .ai-card { background-color: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    .update-time { font-size: 0.85rem; color: #94a3b8; text-align: right; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# Session State の初期化
if "account_balance" not in st.session_state:
    st.session_state["account_balance"] = saved_config.get("account_balance", 500000)
if "quantity_wan" not in st.session_state:
    st.session_state["quantity_wan"] = saved_config.get("quantity_wan", 0.10)
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = saved_config.get("auto_refresh", False)
if "risk_percent" not in st.session_state:
    st.session_state["risk_percent"] = saved_config.get("risk_percent", 1.0)
if "rr_ratio" not in st.session_state:
    st.session_state["rr_ratio"] = saved_config.get("rr_ratio", 1.5)
if "ranges" not in st.session_state:
    st.session_state["ranges"] = saved_config.get("ranges", {})

FEATURE_COLUMNS = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200", "Dev_EMA20_200", "Vol_Ratio",
    "RSI", "RSI_Diff", "MACD_Hist_Ratio", "BB_PctB", "ADX",
    "ATR_Ratio", "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Stoch_K"
]

PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}

TIMEFRAMES = {
    "5分足 (スキャル用)": {"period": "7d", "interval": "5m"},
    "15分足 (デイトレエントリー用)": {"period": "1mo", "interval": "15m"},
    "1時間足 (デイトレメイン用)": {"period": "6mo", "interval": "1h"},
    "4時間足 (中期・リピート用)": {"period": "1y", "interval": "1h"},
}

LOOKAHEAD_BARS = 5

def clean_series(s):
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0].squeeze()
    return s

def safe_to_tokyo_tz(df):
    df_out = df.copy()
    try:
        if df_out.index.tz is None:
            df_out.index = df_out.index.tz_localize("UTC")
        df_out.index = df_out.index.tz_convert("Asia/Tokyo")
    except Exception:
        pass
    return df_out

@st.cache_data(ttl=120, show_spinner=False)
def get_usdjpy_rate():
    try:
        df = yf.download("USDJPY=X", period="1d", progress=False, timeout=10)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            cols = {str(c).lower(): c for c in df.columns}
            close_col = cols.get("close", df.columns[0])
            c = clean_series(df[close_col])
            val = float(c.iloc[-1]) if len(c) > 0 else 155.0
            if not np.isnan(val) and val > 0:
                return val
    except Exception:
        pass
    return 155.0

# ==========================================
# 1. データ処理 & バックテスト付きAIモデル
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, timeout=10)
        if df is None or df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        
        df = df.loc[:, ~df.columns.duplicated()]
        df.columns = [str(c).capitalize() for c in df.columns]

        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            return None
        
        if "4時間足" in tf_name and interval == "1h":
            tz_before = df.index.tz
            df = df.resample("4h", closed="left", label="left").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last"
            }).dropna()
            if tz_before is not None and df.index.tz is None: 
                df.index = df.index.tz_localize(tz_before)
            
        if len(df) < 50:
            return None
        
        c = clean_series(df["Close"])
        h = clean_series(df["High"])
        l = clean_series(df["Low"])
        o = clean_series(df["Open"])

        new_cols = {}
        new_cols["SMA_20"] = c.rolling(20, min_periods=1).mean()
        new_cols["EMA_200"] = c.ewm(span=min(200, len(c)), adjust=False).mean()
        hl = h - l
        new_cols["ATR"] = hl.rolling(14, min_periods=1).mean()
        
        new_cols["Return_1"] = c.diff(1) / (c.shift(1) + 1e-10)
        new_cols["Return_5"] = c.diff(5) / (c.shift(5) + 1e-10)
        new_cols["Dev_SMA20"] = (c - new_cols["SMA_20"]) / (new_cols["SMA_20"] + 1e-10)
        new_cols["Dev_EMA200"] = (c - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Dev_EMA20_200"] = (c.ewm(span=20, adjust=False).mean() - new_cols["EMA_200"]) / (new_cols["EMA_200"] + 1e-10)
        new_cols["Vol_Ratio"] = new_cols["ATR"] / (c + 1e-10)

        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        new_cols["RSI"] = 100 - (100 / (1 + rs))
        new_cols["RSI_Diff"] = new_cols["RSI"].diff(1)
        
        l14 = l.rolling(14, min_periods=1).min()
        h14 = h.rolling(14, min_periods=1).max()
        new_cols["Stoch_K"] = 100 * (c - l14) / ((h14 - l14) + 1e-10)

        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        new_cols["MACD_Hist_Ratio"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c + 1e-10)

        std20 = c.rolling(20, min_periods=1).std().fillna(0)
        new_cols["Upper_Band"] = new_cols["SMA_20"] + (std20 * 2)
        new_cols["Lower_Band"] = new_cols["SMA_20"] - (std20 * 2)
        new_cols["BB_PctB"] = (c - new_cols["Lower_Band"]) / ((new_cols["Upper_Band"] - new_cols["Lower_Band"]) + 1e-10)
        
        tr = pd.concat([hl, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        up, down = h - h.shift(1), l.shift(1) - l
        
        up_cond = (up > down) & (up > 0)
        down_cond = (down > up) & (down > 0)
        
        plus_di_raw = pd.Series(np.where(up_cond, up, 0.0), index=df.index)
        minus_di_raw = pd.Series(np.where(down_cond, down, 0.0), index=df.index)
        
        tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean() + 1e-10
        plus_di = 100 * plus_di_raw.ewm(alpha=1/14, adjust=False).mean() / tr_smooth
        minus_di = 100 * minus_di_raw.ewm(alpha=1/14, adjust=False).mean() / tr_smooth
        
        di_sum = (plus_di + minus_di).replace(0, 1e-10)
        new_cols["ADX"] = (100 * (plus_di - minus_di).abs() / di_sum).ewm(alpha=1/14, adjust=False).mean().fillna(25.0)

        total_range = hl + 1e-10
        open_close_max = pd.concat([o, c], axis=1).max(axis=1)
        open_close_min = pd.concat([o, c], axis=1).min(axis=1)
        new_cols["Upper_Wick_Ratio"] = (h - open_close_max) / total_range
        new_cols["Lower_Wick_Ratio"] = (open_close_min - l) / total_range
        new_cols["ATR_Ratio"] = new_cols["ATR"] / c

        df_feat = pd.DataFrame(new_cols, index=df.index).replace([np.inf, -np.inf], np.nan)
        df = pd.concat([df, df_feat], axis=1)

        f_high = pd.concat([h.shift(-i) for i in range(1, LOOKAHEAD_BARS + 1)], axis=1).max(axis=1) - c
        f_low = c - pd.concat([l.shift(-i) for i in range(1, LOOKAHEAD_BARS + 1)], axis=1).min(axis=1)
        tp_t, sl_t = new_cols["ATR"] * 1.0, new_cols["ATR"] * 0.5
        
        cond_buy_only = (f_high >= tp_t) & (f_low < sl_t)
        cond_sell_only = (f_low >= tp_t) & (f_high < sl_t)
        
        target_series = pd.Series(np.nan, index=df.index)
        valid_future = f_high.notna() & f_low.notna()
        
        target_series[valid_future & cond_buy_only] = 1.0
        target_series[valid_future & cond_sell_only] = -1.0
        target_series[valid_future & ~cond_buy_only & ~cond_sell_only] = 0.0
        df["Target"] = target_series
        
        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        return df.dropna(subset=feat_cols)
    except Exception: 
        return None

@st.cache_data(ttl=60, show_spinner=False)
def analyze_signal_with_backtest(df_current, df_htf):
    empty_res = ("WAIT (データ不足)", 0.0, 0.0, "不明", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, None)
    if df_current is None or len(df_current) < 150:
        return empty_res

    try:
        avail = [f for f in FEATURE_COLUMNS if f in df_current.columns]
        df_valid = df_current.dropna(subset=["Target"])
        test_size = 80
        gap = LOOKAHEAD_BARS
        min_required = test_size + gap + 40
        
        if len(df_valid) < min_required:
            return ("WAIT (学習データ不足)", 0.0, 0.0, "判定不可", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, None)

        X = df_valid[avail].fillna(0)
        y = df_valid["Target"]
        
        X_train = X.iloc[: -(test_size + gap)]
        y_train = y.iloc[: -(test_size + gap)]
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        if len(np.unique(y_train)) < 2: 
            return ("WAIT (データ偏り)", 0.0, 0.0, "判定不可", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, None)

        model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        valid_eval = (preds != 0) & (y_test != 0)
        win_rate = float((preds[valid_eval] == y_test[valid_eval]).mean() * 100) if valid_eval.sum() > 0 else 50.0

        latest_X = df_current[avail].iloc[[-1]].fillna(0)
        prob_array = model.predict_proba(latest_X)[0]
        
        class_prob_map = {float(k): v for k, v in zip(model.classes_, prob_array)}
        
        prob_up = float(class_prob_map.get(1.0, 0.0))
        prob_down = float(class_prob_map.get(-1.0, 0.0))
        prob_wait = float(class_prob_map.get(0.0, 0.0))
        
        total_p = prob_up + prob_down + prob_wait
        if total_p > 0:
            prob_up, prob_down, prob_wait = prob_up/total_p, prob_down/total_p, prob_wait/total_p

        conf = max(prob_up, prob_down) * 100
        prob_dict = {
            "buy": round(prob_up * 100, 1),
            "sell": round(prob_down * 100, 1),
            "wait": round(prob_wait * 100, 1)
        }

        importances = pd.Series(model.feature_importances_, index=avail).sort_values(ascending=True)

        htf_close = float(clean_series(df_htf["Close"]).iloc[-1])
        htf_uptrend = True
        if "EMA_200" in df_htf.columns and not np.isnan(clean_series(df_htf["EMA_200"]).iloc[-1]):
            htf_ema200 = float(clean_series(df_htf["EMA_200"]).iloc[-1])
            htf_uptrend = htf_close > htf_ema200
        elif "SMA_20" in df_htf.columns and not np.isnan(clean_series(df_htf["SMA_20"]).iloc[-1]):
            htf_uptrend = htf_close > float(clean_series(df_htf["SMA_20"]).iloc[-1])

        threshold = 0.50
        if prob_up >= threshold and prob_up > prob_down and htf_uptrend:
            status = "BUY (買い)"
        elif prob_down >= threshold and prob_down > prob_up and not htf_uptrend:
            status = "SELL (売り)"
        else:
            status = "WAIT (様子見)"

        adx_val = float(clean_series(df_current["ADX"]).iloc[-1])
        m_type = "トレンド相場" if adx_val > 22 else "レンジ相場"

        return status, conf, win_rate, m_type, prob_dict, importances
    except Exception:
        return ("WAIT (エラー)", 0.0, 0.0, "エラー", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, None)

# ==========================================
# 2. UI構築 & 状態永続化連動
# ==========================================
st.title("Pro FX Analyzer & Signal Pro")

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    selected_label = st.selectbox("通貨ペア", list(PAIRS.keys()), key="selected_pair_label")
with col_sel2:
    tf_label = st.selectbox("時間足", list(TIMEFRAMES.keys()), key="selected_tf_label")

ticker = PAIRS[selected_label]
tf_config = TIMEFRAMES[tf_label]
is_jpy = "JPY" in ticker
pip_unit = 0.01 if is_jpy else 0.0001
price_fmt = "%.3f" if is_jpy else "%.5f"

# サイドバー設定
st.sidebar.header("⚙️ 資金 & リスク設定")
st.sidebar.number_input(
    "口座資金 (円)", 
    min_value=10000, 
    step=50000, 
    key="account_balance",
    on_change=save_settings
)
st.sidebar.number_input(
    "1注文の基本数量 (万通貨)", 
    min_value=0.01, 
    step=0.01, 
    key="quantity_wan",
    on_change=save_settings
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 単発トレード リスク設定")
st.sidebar.number_input(
    "1取引あたりの許容リスク (%)",
    min_value=0.1,
    max_value=10.0,
    step=0.1,
    key="risk_percent",
    on_change=save_settings,
    help="1回の単発トレードで損切になった際に許容する口座資金の割合"
)
st.sidebar.number_input(
    "デフォルト リスクリワード比 (RR)",
    min_value=0.5,
    max_value=5.0,
    step=0.1,
    key="rr_ratio",
    on_change=save_settings,
    help="損切幅に対して狙う利確幅の比率 (例: 1.5 = 損切10pipsに対し利確15pips)"
)

st.sidebar.markdown("---")
st.sidebar.header("🔄 更新設定")
if HAS_AUTOREFRESH:
    st.sidebar.checkbox("60秒ごとに自動更新", key="auto_refresh", on_change=save_settings)
    if st.session_state["auto_refresh"]:
        st_autorefresh(interval=60000, key="datarefresh")
else:
    st.sidebar.caption("💡 `pip install streamlit-autorefresh` で自動更新が有効になります")

if st.sidebar.button("💾 設定を即時手動保存"):
    save_settings()
    st.sidebar.success("設定を物理ファイルに保存しました！")

if st.sidebar.button("🔄 最新データに更新"):
    st.cache_data.clear()
    st.rerun()

# データ取得
with st.spinner("最新相場データを取得中..."):
    data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
    data_4h = load_and_process_data(ticker, "1y", "1h", "4時間足 (中期・リピート用)")
    data_htf = load_and_process_data(ticker, "3y", "1d", "日足")

if data_4h is None and data is not None:
    data_4h = data

if data is None or data_htf is None:
    st.error("データの取得に失敗しました。Yahoo Financeからの応答が遅延しているか、取引時間外の可能性があります。時間を置いて再度更新してください。")
    st.stop()

now_jst = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f'<div class="update-time">最終データ取得日時: <b>{now_jst} JST</b></div>', unsafe_allow_html=True)

latest_price = float(clean_series(data["Close"]).iloc[-1])
latest_atr = float(clean_series(data["ATR"]).iloc[-1])

status, conf, win_rate, m_type, prob_dict, feature_importances = analyze_signal_with_backtest(data, data_htf)

# サマリー表示
m1, m2, m3, m4 = st.columns(4)
m1.metric("現在レート", price_fmt % latest_price)

badge_html = f'<div class="badge-buy">{status}</div>' if "BUY" in status else \
             f'<div class="badge-sell">{status}</div>' if "SELL" in status else \
             f'<div class="badge-wait">{status}</div>'
m2.markdown("**AI 推奨アクション**")
m2.markdown(badge_html, unsafe_allow_html=True)

m3.metric("直近バックテスト勝率", f"{win_rate:.1f}%", f"確信度: {conf:.1f}%")
m4.metric("相場環境", m_type, f"ATR: {latest_atr/pip_unit:.1f} pips")

st.markdown("---")

# ==========================================
# 3. 通貨ペア毎のレンジ管理 & 安全な自動補正ロジック
# ==========================================
swing_high_4h = float(clean_series(data_4h["High"]).iloc[-100:].max())
swing_low_4h = float(clean_series(data_4h["Low"]).iloc[-100:].min())
atr_4h = float(clean_series(data_4h["ATR"]).iloc[-1])

def is_valid_range(r_low, r_up, current_p):
    """レンジ範囲が有効かチェック"""
    if r_low <= 0 or r_up <= 0 or r_low >= r_up:
        return False
    if r_low < current_p * 0.5 or r_up > current_p * 1.5:
        return False
    return True

need_reset = False
if ticker not in st.session_state["ranges"]:
    need_reset = True
else:
    existing_low = float(st.session_state["ranges"][ticker].get("lower", 0))
    existing_up = float(st.session_state["ranges"][ticker].get("upper", 0))
    if not is_valid_range(existing_low, existing_up, latest_price):
        need_reset = True

if need_reset:
    st.session_state["ranges"][ticker] = {
        "lower": float(swing_low_4h),
        "upper": float(swing_high_4h)
    }
    save_settings()

current_range = st.session_state["ranges"][ticker]

# Session State の同期（安全処理）
if "active_ticker" not in st.session_state or st.session_state["active_ticker"] != ticker:
    st.session_state["active_ticker"] = ticker
    st.session_state["in_lower"] = float(current_range["lower"])
    st.session_state["in_upper"] = float(current_range["upper"])

# レンジ調整UI
with st.expander("⚙️ リピート自動売買のレンジ調整", expanded=False):
    rc1, rc2 = st.columns(2)
    
    def update_range():
        st.session_state["ranges"][ticker] = {
            "lower": float(st.session_state["in_lower"]),
            "upper": float(st.session_state["in_upper"])
        }
        save_settings()

    in_lower = rc1.number_input(
        "レンジ下限", 
        step=0.1 if is_jpy else 0.001, 
        format=price_fmt, 
        key="in_lower", 
        on_change=update_range
    )
    in_upper = rc2.number_input(
        "レンジ上限", 
        step=0.1 if is_jpy else 0.001, 
        format=price_fmt, 
        key="in_upper", 
        on_change=update_range
    )
    
    # 修正: ボタン押下時にStateを直接修正し、安全に同期してリロード
    if st.button("✨ 4時間足高値・安値からレンジを自動計算"):
        st.session_state["ranges"][ticker] = {
            "lower": float(swing_low_4h),
            "upper": float(swing_high_4h)
        }
        if "active_ticker" in st.session_state:
            del st.session_state["active_ticker"]
        save_settings()
        st.rerun()

    user_lower = min(in_lower, in_upper)
    user_upper = max(in_lower, in_upper)
    if user_lower == user_upper:
        user_upper += pip_unit * 100.0
    user_half = (user_upper + user_lower) / 2.0

# 運用停止ライン（SL）
stop_buffer_pips = max(10.0, round((atr_4h / pip_unit) * 1.5, 1))
stop_buffer_val = stop_buffer_pips * pip_unit
buy_stop_loss = user_lower - stop_buffer_val
sell_stop_loss = user_upper + stop_buffer_val

# ==========================================
# 4. タブ描画
# ==========================================
tab_chart, tab_single, tab_repeat, tab_ai = st.tabs([
    "📈 メインチャート", 
    "⚡ 単発トレード (AI全自動アシスト)", 
    "📋 松井証券 リピート設定 & リスク管理", 
    "🤖 AIモデル分析詳細"
])

# --- タブ1: メインチャート ---
with tab_chart:
    df_chart = safe_to_tokyo_tz(data.tail(120))
    fmt_str = '%Y-%m-%d' if '日足' in tf_label else '%m-%d %H:%M'
    x_labels = df_chart.index.strftime(fmt_str)

    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.75, 0.25], 
        vertical_spacing=0.05
    )

    # 1. ローソク足描画
    fig.add_trace(go.Candlestick(
        x=x_labels,
        open=clean_series(df_chart["Open"]),
        high=clean_series(df_chart["High"]),
        low=clean_series(df_chart["Low"]),
        close=clean_series(df_chart["Close"]),
        increasing_line_color='#22c55e', increasing_fillcolor='#22c55e',
        decreasing_line_color='#ef4444', decreasing_fillcolor='#ef4444',
        name="価格"
    ), row=1, col=1)

    # 2. 移動平均線
    if "SMA_20" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=x_labels, y=clean_series(df_chart["SMA_20"]),
            line=dict(color="#f59e0b", width=1.2), name="SMA20"
        ), row=1, col=1)

    if "EMA_200" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=x_labels, y=clean_series(df_chart["EMA_200"]),
            line=dict(color="#38bdf8", width=1.8), name="EMA200"
        ), row=1, col=1)

    # 3. リピート範囲・運用停止ラインの描画
    fig.add_hrect(
        y0=user_lower, y1=user_upper, 
        fillcolor="rgba(56, 189, 248, 0.06)", line_width=0, 
        row=1, col=1
    )
    
    fig.add_hline(y=sell_stop_loss, line_dash="dashdot", line_color="#dc2626", annotation_text="売 SL", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=user_upper, line_dash="dash", line_color="#f87171", annotation_text="上限", annotation_position="bottom right", row=1, col=1)
    fig.add_hline(y=user_half, line_dash="dot", line_color="#c084fc", annotation_text="中央", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=user_lower, line_dash="dash", line_color="#4ade80", annotation_text="下限", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=buy_stop_loss, line_dash="dashdot", line_color="#16a34a", annotation_text="買 SL", annotation_position="bottom right", row=1, col=1)

    # 4. RSI (サブチャート)
    if "RSI" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=x_labels, y=clean_series(df_chart["RSI"]),
            line=dict(color="#a855f7", width=1.5), name="RSI"
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#94a3b8", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#94a3b8", row=2, col=1)

    # 5. Y軸可視化範囲決定
    chart_high = float(clean_series(df_chart["High"]).max())
    chart_low = float(clean_series(df_chart["Low"]).min())
    price_span = chart_high - chart_low
    if price_span <= 0:
        price_span = chart_high * 0.002
    
    y_min_fit = chart_low - (price_span * 0.08)
    y_max_fit = chart_high + (price_span * 0.08)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=10, r=80, t=20, b=10),
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(type='category', nticks=12, tickangle=-30, showspikes=True)
    
    fig.update_yaxes(
        range=[y_min_fit, y_max_fit],
        fixedrange=False,
        side="right",
        tickformat=".3f" if is_jpy else ".5f",
        row=1, col=1
    )
    fig.update_yaxes(range=[0, 100], side="right", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})

# --- タブ2: 単発トレード (AI全自動アシスト) ---
with tab_single:
    st.markdown("##### ⚡ 単発トレード（スキャル・デイトレ）AI発注アシスタント")
    st.caption("AIが現在の相場・ボラティリティ・口座資金を統合解析し、最適シグナルと注文数値をリアルタイムで自動計算します。")

    usd_rate = latest_price if ticker == "USDJPY=X" else get_usdjpy_rate()
    pip_val_per_wan = 100.0 if is_jpy else (0.0001 * usd_rate * 10000.0)

    account_bal = st.session_state["account_balance"]
    risk_pct = st.session_state["risk_percent"]
    max_risk_yen = account_bal * (risk_pct / 100.0)

    ai_is_buy = "BUY" in status
    ai_is_sell = "SELL" in status
    
    if ai_is_buy:
        ai_direction = "BUY (買い)"
    elif ai_is_sell:
        ai_direction = "SELL (売り)"
    else:
        ai_direction = "BUY (買い)" if prob_dict["buy"] >= prob_dict["sell"] else "SELL (売り)"

    ai_sl_pips = round((latest_atr / pip_unit) * 1.5, 1)
    ai_rr_ratio = float(st.session_state["rr_ratio"])
    ai_tp_pips = round(ai_sl_pips * ai_rr_ratio, 1)

    ai_entry = latest_price
    ai_sl_price = ai_entry - (ai_sl_pips * pip_unit) if "BUY" in ai_direction else ai_entry + (ai_sl_pips * pip_unit)
    ai_tp_price = ai_entry + (ai_tp_pips * pip_unit) if "BUY" in ai_direction else ai_entry - (ai_tp_pips * pip_unit)

    ai_loss_per_wan = ai_sl_pips * pip_val_per_wan
    ai_rec_wan = (max_risk_yen / ai_loss_per_wan) if ai_loss_per_wan > 0 else 0.01
    ai_rec_wan = max(0.01, round(ai_rec_wan, 2))

    ai_profit_yen = ai_tp_pips * pip_val_per_wan * ai_rec_wan
    ai_loss_yen = ai_sl_pips * pip_val_per_wan * ai_rec_wan

    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown("#### 🤖 AI提案トレードプラン")
    
    if "WAIT" in status:
        st.warning("⚠️ **現在AIシグナルは【WAIT (様子見)】です。** 明確なトレンドが出るまで見送りを推奨しますが、仮にトレードする場合の数値を下部に提示しています。")
    else:
        st.success(f"🎯 **AI判定: 【{status}】（確信度: {conf:.1f}% / 勝率目安: {win_rate:.1f}%）**")

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("推奨売買方向", ai_direction)
    a2.metric("想定エントリー", price_fmt % ai_entry)
    a3.metric("推奨損切 (SL)", price_fmt % ai_sl_price, f"-{ai_sl_pips} pips")
    a4.metric("推奨利確 (TP)", price_fmt % ai_tp_price, f"+{ai_tp_pips} pips")
    a5.metric("最適数量", f"{ai_rec_wan:.2f} 万通貨", f"許容リスク {risk_pct}%")

    st.markdown(f"""
    <div class="param-box">
    <b>【AI自動提示 注文コピー用パラメータ】</b><br>
    ・<b>通貨ペア</b>: <code>{selected_label}</code> | <b>売買方向</b>: <code>{"買い (BUY)" if "BUY" in ai_direction else "売り (SELL)"}</code><br>
    ・<b>新規成行価格</b>: <code>{price_fmt % ai_entry}</code><br>
    ・<b>決済利確(TP)</b>: <code>{price_fmt % ai_tp_price}</code> (+{ai_tp_pips} pips / ＋{int(ai_profit_yen):,}円)<br>
    ・<b>決済損切(SL)</b>: <code>{price_fmt % ai_sl_price}</code> (-{ai_sl_pips} pips / －{int(ai_loss_yen):,}円)<br>
    ・<b>推奨発注数量</b>: <code>{ai_rec_wan:.2f} 万通貨</code> ({int(ai_rec_wan * 10000):,} 通貨)
    </div>
    """, unsafe_allow_html=True)
    
    # 改善機能: AI提案をフォームへワンクリック反映
    if st.button("🤖 AIの提案値を手動設定に反映"):
        st.session_state[f"side_{ticker}"] = "BUY (買い)" if "BUY" in ai_direction else "SELL (売り)"
        st.session_state[f"entry_{ticker}"] = float(ai_entry)
        st.session_state[f"slpips_{ticker}"] = float(ai_sl_pips)
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🛠️ 注文条件の手動微調整")
    st.caption("AIの算出結果をベースに、エントリー価格や損切幅・リスクリワード比を自由に変更できます。")

    sc1, sc2 = st.columns(2)
    default_side_idx = 0 if "BUY" in ai_direction else 1
    
    with sc1:
        trade_side = st.radio("売買方向", ["BUY (買い)", "SELL (売り)"], index=default_side_idx, horizontal=True, key=f"side_{ticker}")
        entry_price = st.number_input("エントリー想定レート", value=latest_price, format=price_fmt, step=0.01 if is_jpy else 0.0001, key=f"entry_{ticker}")

    with sc2:
        sl_mode = st.radio("損切(SL)の決め方", ["ATRベース (推奨)", "固定 pips"], horizontal=True, key=f"slmode_{ticker}")
        if sl_mode == "ATRベース (推奨)":
            atr_multiplier = st.slider("ATR倍率 (1.0 = 現在のボラティリティ相当)", min_value=0.5, max_value=3.0, value=1.5, step=0.1, key=f"atrmul_{ticker}")
            sl_pips = round((latest_atr / pip_unit) * atr_multiplier, 1)
            st.info(f"現在のATR: {latest_atr/pip_unit:.1f} pips ➔ 損切幅: **{sl_pips} pips**")
        else:
            sl_pips = st.number_input("損切幅 (pips)", min_value=1.0, value=15.0, step=1.0, key=f"slpips_{ticker}")

    rr_value = st.slider("リスクリワード比 (RR)", min_value=0.5, max_value=4.0, value=float(st.session_state["rr_ratio"]), step=0.1, key=f"rr_{ticker}")
    tp_pips = round(sl_pips * rr_value, 1)

    is_buy = "BUY" in trade_side
    sl_price = entry_price - (sl_pips * pip_unit) if is_buy else entry_price + (sl_pips * pip_unit)
    tp_price = entry_price + (tp_pips * pip_unit) if is_buy else entry_price - (tp_pips * pip_unit)

    loss_per_wan = sl_pips * pip_val_per_wan
    recommended_wan = (max_risk_yen / loss_per_wan) if loss_per_wan > 0 else 0.01
    recommended_wan = max(0.01, round(recommended_wan, 2))

    expected_profit_yen = tp_pips * pip_val_per_wan * recommended_wan
    actual_loss_yen = sl_pips * pip_val_per_wan * recommended_wan

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("調整後の手動ロット数", f"{recommended_wan:.2f} 万通貨", f"許容リスク {risk_pct}%")
    res_col2.metric("損切(SL) レート", price_fmt % sl_price, f"-{sl_pips} pips")
    res_col3.metric("利確(TP) レート", price_fmt % tp_price, f"+{tp_pips} pips")
    res_col4.metric("想定損益", f"+{int(expected_profit_yen):,}円", f"最大損失 -{int(actual_loss_yen):,}円")

# --- タブ3: リピート設定 & リスク管理 ---
with tab_repeat:
    trap_width_pips = max(15, int(round((atr_4h / pip_unit))))
    half_range_pips = abs(user_upper - user_half) / pip_unit
    
    half_grid_count = max(1, int(np.floor(half_range_pips / trap_width_pips)))
    total_grid_count = half_grid_count * 2

    quantity_wan_val = st.session_state["quantity_wan"]
    account_balance_val = st.session_state["account_balance"]

    order_units = int(quantity_wan_val * 10000)
    usd_rate = latest_price if ticker == "USDJPY=X" else get_usdjpy_rate()
    pip_value_yen = (order_units / 10000.0) * 100 if is_jpy else (order_units * 0.0001 * usd_rate)
    
    max_loss_yen = sum((k * trap_width_pips + stop_buffer_pips) * pip_value_yen for k in range(half_grid_count))
        
    margin_per_order = (latest_price * order_units) / 25.0 if is_jpy else (latest_price * usd_rate * order_units) / 25.0
    total_margin_yen = margin_per_order * half_grid_count
    risk_ratio = (max_loss_yen / account_balance_val) * 100 if account_balance_val > 0 else 0.0

    st.markdown(f"""
    <div class="param-box">
    <b>【ハーフ＆ハーフ推奨設定値】</b><br>
    ・<b>買い設定（下半）</b>: レンジ <code>{price_fmt % user_lower}</code> ～ <code>{price_fmt % user_half}</code> | <b>運用停止(SL)</b>: <code>{price_fmt % buy_stop_loss}</code> (-{stop_buffer_pips}pips)<br>
    ・<b>売り設定（上半）</b>: レンジ <code>{price_fmt % user_half}</code> ～ <code>{price_fmt % user_upper}</code> | <b>運用停止(SL)</b>: <code>{price_fmt % sell_stop_loss}</code> (+{stop_buffer_pips}pips)<br>
    ・<b>注文幅 / 利確幅</b>: <code>{trap_width_pips} pips</code> | <b>片側注文本数</b>: 約 <code>{half_grid_count} 本</code> (全 {total_grid_count}本)
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("▼ 松井証券の自動売買設定画面へそのままコピー＆ペーストしてご使用ください")
    st.code(f"""[松井証券リピート注文 設定値]
通貨ペア: {selected_label}
注文種別: ハーフ＆ハーフ
買いレンジ: {price_fmt % user_lower} - {price_fmt % user_half} (SL: {price_fmt % buy_stop_loss})
売りレンジ: {price_fmt % user_half} - {price_fmt % user_upper} (SL: {price_fmt % sell_stop_loss})
注文幅 / 利確幅: {trap_width_pips} pips
1本あたりの数量: {quantity_wan_val:.2f} 万通貨""", language="text")

    st.markdown("##### 🛡️ リスク・資金シミュレーション")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("想定最大含み損", f"約 {int(max_loss_yen):,} 円")
    rc2.metric("片側最大 必要証拠金", f"約 {int(total_margin_yen):,} 円")
    rc3.metric("資金リスク比率", f"{risk_ratio:.1f}%")

    if risk_ratio > 40.0:
        st.error("🚨 警告: 撤退時の最大損失が口座資金の40%を超えています。数量(万通貨)を減らすか口座資金を増やしてください。")
    else:
        st.success("🟢 資金管理チェック: 適切なリスク範囲内です。")

# --- タブ4: AIモデル分析詳細 ---
with tab_ai:
    st.markdown("##### 🤖 AI予測モデル（Random Forest）の評価と内訳")
    
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**最新バーの分類判定確率**")
        p_buy = max(0.0, min(1.0, prob_dict['buy'] / 100.0))
        p_sell = max(0.0, min(1.0, prob_dict['sell'] / 100.0))
        p_wait = max(0.0, min(1.0, prob_dict['wait'] / 100.0))

        st.write(f"🟢 **BUY (買い)**: {prob_dict['buy']}%")
        st.progress(p_buy)
        
        st.write(f"🔴 **SELL (売り)**: {prob_dict['sell']}%")
        st.progress(p_sell)
        
        st.write(f"⚪ **WAIT (様子見)**: {prob_dict['wait']}%")
        st.progress(p_wait)

    with pcol2:
        st.markdown("**アウトオブサンプル検証（リーク防止対策済み）**")
        st.metric("直近テスト80足の方向勝率", f"{win_rate:.1f}%")
        st.caption("※ 先読みデータ（Lookahead Leak）を防止するためのGap（ギャップ）を設定した厳格な時系列検証精度です。")

    if feature_importances is not None:
        st.markdown("---")
        st.markdown("##### 📊 AIの判断根拠（特徴量重要度 TOP 10）")
        st.caption("AIが『買い・売り・様子見』を判断する際に、どの指標を重視したかを示す貢献度ランキングです。")
        
        top10_imp = feature_importances.tail(10)
        
        fig_imp = go.Figure(go.Bar(
            x=top10_imp.values,
            y=top10_imp.index,
            orientation='h',
            marker_color='#38bdf8'
        ))
        fig_imp.update_layout(
            height=320,
            margin=dict(l=10, r=20, t=10, b=30),
            template="plotly_dark",
            xaxis_title="重要度スコア",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
