import html
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

# Streamlitのページ設定
st.set_page_config(
    page_title="Pro FX Analyzer & Signal Pro",
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

def convert_to_builtin_type(obj):
    """NumPy型などをPython組み込み型へ変換してJSONシリアライズエラーを防止"""
    if isinstance(obj, dict):
        return {str(k): convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def load_settings():
    """ローカルのJSONファイルから設定を読み込み"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
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
        "account_balance": float(st.session_state.get("account_balance", 500000)),
        "quantity_wan": float(st.session_state.get("quantity_wan", 0.10)),
        "auto_refresh": bool(st.session_state.get("auto_refresh", False)),
        "risk_percent": float(st.session_state.get("risk_percent", 1.0)),
        "rr_ratio": float(st.session_state.get("rr_ratio", 1.5)),
        "ranges": st.session_state.get("ranges", {})
    }
    try:
        clean_data = convert_to_builtin_type(settings)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=2)
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
    .news-card { background-color: rgba(30, 41, 59, 0.5); border: 1px solid #475569; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
    .update-time { font-size: 0.85rem; color: #94a3b8; text-align: right; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# Session State の初期化
if "account_balance" not in st.session_state:
    st.session_state["account_balance"] = int(saved_config.get("account_balance", 500000))
if "quantity_wan" not in st.session_state:
    st.session_state["quantity_wan"] = float(saved_config.get("quantity_wan", 0.10))
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = bool(saved_config.get("auto_refresh", False))
if "risk_percent" not in st.session_state:
    st.session_state["risk_percent"] = float(saved_config.get("risk_percent", 1.0))
if "rr_ratio" not in st.session_state:
    st.session_state["rr_ratio"] = float(saved_config.get("rr_ratio", 1.5))
if "ranges" not in st.session_state:
    st.session_state["ranges"] = saved_config.get("ranges", {})

FEATURE_COLUMNS = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200", "Dev_EMA20_200", "Vol_Ratio",
    "RSI", "RSI_Diff", "MACD_Hist_Ratio", "BB_PctB", "ADX",
    "ATR_Ratio", "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Stoch_K"
]

FEATURE_LABELS_JA = {
    "Return_1": "直近1足変化率",
    "Return_5": "直近5足変化率",
    "Dev_SMA20": "SMA20乖離率",
    "Dev_EMA200": "EMA200乖離率",
    "Dev_EMA20_200": "EMA20/200乖離率",
    "Vol_Ratio": "ボラティリティ比率",
    "RSI": "RSI(14)",
    "RSI_Diff": "RSI変化幅",
    "MACD_Hist_Ratio": "MACDヒストグラム比",
    "BB_PctB": "ボリンジャー%B",
    "ADX": "ADX(トレンド強度)",
    "ATR_Ratio": "ATR比率",
    "Upper_Wick_Ratio": "上ヒゲ比率",
    "Lower_Wick_Ratio": "下ヒゲ比率",
    "Stoch_K": "ストキャスティクス%K"
}

PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}

DEFAULT_SPREAD_PIPS = {
    "USDJPY=X": 0.2,
    "EURUSD=X": 0.4,
    "GBPJPY=X": 1.0,
    "EURJPY=X": 0.5,
    "AUDJPY=X": 0.6,
}

TIMEFRAMES = {
    "5分足 (スキャル用)": {"period": "7d", "interval": "5m"},
    "15分足 (デイトレエントリー用)": {"period": "1mo", "interval": "15m"},
    "1時間足 (デイトレメイン用)": {"period": "60d", "interval": "1h"},
    "4時間足 (中期・リピート用)": {"period": "60d", "interval": "1h"},
}

LOOKAHEAD_BARS = 5

def clean_series(s):
    """DataFrameやSeriesから安全に1次元Seriesを取り出す"""
    if isinstance(s, pd.DataFrame):
        if s.shape[1] > 0:
            return s.iloc[:, 0]
        return pd.Series(dtype=float)
    elif isinstance(s, pd.Series):
        return s
    return pd.Series(s)

def safe_to_tokyo_tz(df):
    """タイムゾーンの有無を判定して安全にAsia/Tokyoに変換"""
    if df is None or df.empty:
        return df
    df_out = df.copy()
    try:
        if isinstance(df_out.index, pd.DatetimeIndex):
            if df_out.index.tz is None:
                df_out.index = df_out.index.tz_localize("UTC")
            df_out.index = df_out.index.tz_convert("Asia/Tokyo")
    except Exception:
        pass
    return df_out

def flatten_yf_df(df):
    """yfinanceのレスポンス（MultiIndex対応）を安全に単一層のDataFrameに平坦化"""
    if df is None or df.empty:
        return df
    df_out = df.copy()
    if isinstance(df_out.columns, pd.MultiIndex):
        try:
            l0 = [str(x).lower() for x in df_out.columns.get_level_values(0)]
            if any(c in l0 for c in ["close", "open", "high", "low"]):
                df_out.columns = df_out.columns.get_level_values(0)
            else:
                df_out.columns = df_out.columns.get_level_values(1)
        except Exception:
            df_out.columns = [str(c[0]) for c in df_out.columns]
            
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    df_out.columns = [str(c).capitalize() for c in df_out.columns]
    
    # 重複インデックスの除去
    df_out = df_out.loc[~df_out.index.duplicated(keep='last')]
    return df_out

@st.cache_data(ttl=120, show_spinner=False)
def get_usdjpy_rate():
    """週末・休日等の取得失敗を防ぐためperiod='5d'で最新レートを取得"""
    try:
        df = yf.download("USDJPY=X", period="5d", progress=False, timeout=10)
        df = flatten_yf_df(df)
        if df is not None and not df.empty and "Close" in df.columns:
            c = clean_series(df["Close"]).dropna()
            val = float(c.iloc[-1]) if len(c) > 0 else 155.0
            if not np.isnan(val) and val > 0:
                return val
    except Exception:
        pass
    return 155.0

# ==========================================
# ニュース＆インパクト解析モジュール
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_news_and_impact(symbol):
    """Yahoo Financeからニュースを取得し、キーワード解析で為替影響を自動予測（新旧API構造両対応）"""
    try:
        tk = yf.Ticker(symbol)
        news_list = tk.news
        if not news_list or not isinstance(news_list, list):
            return []
        
        parsed_news = []
        for item in news_list[:10]:
            if not isinstance(item, dict):
                continue
            
            # yfinanceの新旧API構造に柔軟対応
            content = item.get("content", {}) if isinstance(item.get("content"), dict) else item
            title = content.get("title") or item.get("title", "No Title")
            if not title or title == "No Title":
                continue

            provider = content.get("provider", {}) if isinstance(content.get("provider"), dict) else {}
            publisher = provider.get("displayName") or item.get("publisher", "市場ニュース")

            # リンクURLの多重フォールバック取得
            click_url = "#"
            canonical = content.get("canonicalUrl")
            click_through = content.get("clickThroughUrl")
            
            if isinstance(canonical, dict) and canonical.get("url"):
                click_url = canonical.get("url")
            elif isinstance(click_through, dict) and click_through.get("url"):
                click_url = click_through.get("url")
            elif isinstance(canonical, str):
                click_url = canonical
            elif "link" in item:
                click_url = str(item["link"])

            if click_url.startswith("/"):
                click_url = f"https://finance.yahoo.com{click_url}"

            pub_time = content.get("pubDate") or item.get("providerPublishTime", None)
            if isinstance(pub_time, (int, float)):
                dt_str = datetime.fromtimestamp(pub_time, tz=ZoneInfo("Asia/Tokyo")).strftime("%m/%d %H:%M")
            elif isinstance(pub_time, str):
                try:
                    dt = datetime.fromisoformat(pub_time.replace("Z", "+00:00")).astimezone(ZoneInfo("Asia/Tokyo"))
                    dt_str = dt.strftime("%m/%d %H:%M")
                except Exception:
                    dt_str = "直近"
            else:
                dt_str = "直近"

            t_lower = str(title).lower()
            impact_level = "ℹ️ 普通"
            direction = "↔️ 中立 / ボラティリティ注意"
            reason = "全般的な市況ニュース"

            if any(k in t_lower for k in ["fed", "powell", "cpi", "gdp", "pmi", "retail", "inflation", "rate", "payroll", "jobs", "fomc", "yield"]):
                impact_level = "🔥 高（米金融政策・大変動）"
                if any(k in t_lower for k in ["hike", "high", "rise", "strong", "inflation", "above", "beat"]):
                    direction = "📈 ドル高 / 通貨ペア上昇要因" if "USD" in symbol else "⚡ USD強含み"
                    reason = "米金利高止まり・経済指標好調によるドル買い圧力"
                elif any(k in t_lower for k in ["cut", "fall", "cool", "drop", "weak", "below", "miss"]):
                    direction = "📉 ドル安 / 通貨ペア下落要因" if "USD" in symbol else "⚡ USD弱含み"
                    reason = "米利下げ観測・経済減速に伴うドル売り圧力"
                else:
                    direction = "⚡ 急変動警戒"
                    reason = "米国の重要指標・FRB発言に伴う相場変動"

            elif any(k in t_lower for k in ["boj", "ueda", "yen", "japan", "bank of japan"]):
                impact_level = "🔥 高（円相場直接影響）"
                if any(k in t_lower for k in ["hike", "normalize", "taper", "intervention", "strong"]):
                    direction = "📉 円高（USD/JPY等 下落）要因"
                    reason = "日銀利上げ・政策正常化観測に伴う円買い"
                else:
                    direction = "📈 円安（USD/JPY等 上昇）要因"
                    reason = "日銀金融緩和維持観測に伴う円売り"

            elif any(k in t_lower for k in ["ecb", "lagarde", "euro", "europe"]):
                impact_level = "⚡ 中〜高（ユーロ影響）"
                direction = "🇪🇺 ユーロ急変動注意"
                reason = "ECB理事会・欧州経済指標の動き"

            parsed_news.append({
                "title": title,
                "publisher": publisher,
                "link": click_url,
                "time": dt_str,
                "impact": impact_level,
                "direction": direction,
                "reason": reason
            })
        return parsed_news
    except Exception:
        return []

# ==========================================
# 1. データ処理 & バックテスト付きAIモデル
# ==========================================
@st.cache_data(ttl=60, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, timeout=10)
        df = flatten_yf_df(df)
        if df is None or df.empty:
            return None

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

        df_feat = pd.DataFrame(new_cols, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0)
        df = pd.concat([df, df_feat], axis=1)

        # Target算出 (厳密順序判定)
        tp_t = new_cols["ATR"] * 1.0
        sl_t = new_cols["ATR"] * 0.5
        
        target_values = np.zeros(len(df), dtype=int)
        n = len(df)
        
        c_vals = c.values
        h_vals = h.values
        l_vals = l.values
        tp_vals = tp_t.values
        sl_vals = sl_t.values

        for i in range(n - LOOKAHEAD_BARS):
            entry_p = c_vals[i]
            tp_dist = tp_vals[i]
            sl_dist = sl_vals[i]
            
            if np.isnan(tp_dist) or np.isnan(sl_dist) or tp_dist <= 0 or sl_dist <= 0:
                continue

            outcome = 0
            for j in range(1, LOOKAHEAD_BARS + 1):
                idx = i + j
                curr_h = h_vals[idx]
                curr_l = l_vals[idx]
                
                buy_tp_hit = (curr_h - entry_p) >= tp_dist
                buy_sl_hit = (entry_p - curr_l) >= sl_dist
                sell_tp_hit = (entry_p - curr_l) >= tp_dist
                sell_sl_hit = (curr_h - entry_p) >= sl_dist
                
                # 買い優勢
                if buy_tp_hit and not buy_sl_hit:
                    outcome = 1
                    break
                # 売り優勢
                elif sell_tp_hit and not sell_sl_hit:
                    outcome = -1
                    break
                # 双方損切または不確実
                elif buy_sl_hit or sell_sl_hit:
                    outcome = 0
                    break
            
            target_values[i] = outcome

        target_series = pd.Series(target_values, index=df.index, dtype="float64")
        target_series.iloc[-LOOKAHEAD_BARS:] = np.nan
        df["Target"] = target_series

        feat_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
        return df.dropna(subset=feat_cols)
    except Exception: 
        return None

@st.cache_data(ttl=60, show_spinner=False)
def analyze_signal_with_backtest(df_current, df_htf):
    empty_res = ("WAIT (データ不足)", 0.0, 0.0, "不明", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, pd.Series(dtype=float))
    if df_current is None or len(df_current) < 150:
        return empty_res

    try:
        avail = [f for f in FEATURE_COLUMNS if f in df_current.columns]
        df_valid = df_current.dropna(subset=["Target"]).copy()
        df_valid["Target"] = df_valid["Target"].astype(int)
        
        test_size = 80
        gap = LOOKAHEAD_BARS
        min_required = test_size + gap + 40
        
        if len(df_valid) < min_required:
            return ("WAIT (学習データ不足)", 0.0, 0.0, "判定不可", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, pd.Series(dtype=float))

        X = df_valid[avail].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df_valid["Target"]
        
        X_train = X.iloc[: -(test_size + gap)]
        y_train = y.iloc[: -(test_size + gap)]
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        if len(np.unique(y_train)) < 2: 
            return ("WAIT (データ偏り)", 0.0, 0.0, "判定不可", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, pd.Series(dtype=float))

        model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        preds_series = pd.Series(preds, index=y_test.index)
        
        trade_signals = preds_series != 0
        if trade_signals.sum() > 0:
            win_count = (preds_series[trade_signals] == y_test[trade_signals]).sum()
            win_rate = float((win_count / trade_signals.sum()) * 100)
        else:
            win_rate = 50.0

        latest_X = df_current[avail].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
        prob_array = model.predict_proba(latest_X)[0]
        
        class_prob_map = {int(cls_val): float(p) for cls_val, p in zip(model.classes_, prob_array)}
        prob_up = class_prob_map.get(1, 0.0)
        prob_down = class_prob_map.get(-1, 0.0)
        prob_wait = class_prob_map.get(0, 0.0)
        
        total_p = prob_up + prob_down + prob_wait
        if total_p > 0:
            prob_up, prob_down, prob_wait = prob_up / total_p, prob_down / total_p, prob_wait / total_p

        prob_dict = {
            "buy": round(prob_up * 100, 1),
            "sell": round(prob_down * 100, 1),
            "wait": round(prob_wait * 100, 1)
        }

        ja_index = [FEATURE_LABELS_JA.get(col, col) for col in avail]
        importances = pd.Series(model.feature_importances_, index=ja_index).sort_values(ascending=True)

        htf_uptrend = True
        curr_close = float(clean_series(df_current["Close"]).iloc[-1])
        
        if df_htf is not None and not df_htf.empty and "Close" in df_htf.columns:
            htf_close = float(clean_series(df_htf["Close"]).iloc[-1])
            if "EMA_200" in df_htf.columns and not np.isnan(clean_series(df_htf["EMA_200"]).iloc[-1]):
                htf_ema200 = float(clean_series(df_htf["EMA_200"]).iloc[-1])
                htf_uptrend = htf_close > htf_ema200
            elif "SMA_20" in df_htf.columns and not np.isnan(clean_series(df_htf["SMA_20"]).iloc[-1]):
                htf_uptrend = htf_close > float(clean_series(df_htf["SMA_20"]).iloc[-1])
        else:
            if "EMA_200" in df_current.columns and not np.isnan(clean_series(df_current["EMA_200"]).iloc[-1]):
                htf_uptrend = curr_close > float(clean_series(df_current["EMA_200"]).iloc[-1])

        rsi_val = float(clean_series(df_current["RSI"]).iloc[-1]) if "RSI" in df_current.columns else 50.0

        # シグナル判定ロジック
        threshold = 0.38
        if prob_up > prob_wait and prob_up >= threshold and prob_up > prob_down and htf_uptrend and rsi_val < 75.0:
            status = "BUY (買い)"
            conf = prob_up * 100
        elif prob_down > prob_wait and prob_down >= threshold and prob_down > prob_up and not htf_uptrend and rsi_val > 25.0:
            status = "SELL (売り)"
            conf = prob_down * 100
        else:
            status = "WAIT (様子見)"
            conf = max(prob_up, prob_down, prob_wait) * 100

        adx_val = float(clean_series(df_current["ADX"]).iloc[-1]) if "ADX" in df_current.columns else 20.0
        m_type = "トレンド相場" if adx_val > 22 else "レンジ相場"

        return status, conf, win_rate, m_type, prob_dict, importances
    except Exception:
        return ("WAIT (エラー)", 0.0, 0.0, "エラー", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, pd.Series(dtype=float))

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
spread_pips = DEFAULT_SPREAD_PIPS.get(ticker, 0.3)

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
    st.toast("設定を保存しました！", icon="💾")

if st.sidebar.button("🔄 最新データに更新"):
    st.cache_data.clear()
    st.rerun()

# データ取得
with st.spinner("最新相場データ & ニュースを取得中..."):
    data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
    data_4h = load_and_process_data(ticker, "60d", "1h", "4時間足 (中期・リピート用)")
    data_htf = load_and_process_data(ticker, "2y", "1d", "日足")
    news_items = fetch_news_and_impact(ticker)

if data_4h is None and data is not None:
    data_4h = data

if data is None:
    st.error("データの取得に失敗しました。Yahoo Financeからの応答が遅延しているか、取引時間外の可能性があります。時間を置いて再度更新してください。")
    st.stop()

now_jst = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f'<div class="update-time">最終データ取得日時: <b>{now_jst} JST</b></div>', unsafe_allow_html=True)

latest_price = float(clean_series(data["Close"]).iloc[-1])
latest_atr = float(clean_series(data["ATR"]).iloc[-1]) if "ATR" in data.columns else (latest_price * 0.005)

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
swing_high_4h = float(clean_series(data_4h["High"]).iloc[-100:].max()) if data_4h is not None else latest_price * 1.02
swing_low_4h = float(clean_series(data_4h["Low"]).iloc[-100:].min()) if data_4h is not None else latest_price * 0.98
atr_4h = float(clean_series(data_4h["ATR"]).iloc[-1]) if data_4h is not None and "ATR" in data_4h.columns else (latest_price * 0.005)

def is_valid_range(r_low, r_up, current_p):
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

key_lower = f"in_lower_{ticker}"
key_upper = f"in_upper_{ticker}"

current_range = st.session_state["ranges"][ticker]

# ranges と Widget キーの初期同期
if key_lower not in st.session_state:
    st.session_state[key_lower] = float(current_range["lower"])
if key_upper not in st.session_state:
    st.session_state[key_upper] = float(current_range["upper"])

# レンジ調整UI
with st.expander("⚙️ リピート自動売買のレンジ調整", expanded=False):
    rc1, rc2 = st.columns(2)
    
    def update_range_callback():
        st.session_state["ranges"][ticker] = {
            "lower": float(st.session_state[key_lower]),
            "upper": float(st.session_state[key_upper])
        }
        save_settings()

    def trigger
