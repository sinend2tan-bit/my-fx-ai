# ==========================================
# Pro FX Analyzer & Signal Pro - Fixed版
# 松井証券FX 運用支援 / 免責: 投資判断は自己責任で
# ==========================================
import html
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

# 高速化用 (なくても動く)
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(func):
        return func

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

st.markdown("""
<style>
   .main.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
    [data-testid="stMetricValue"] { font-size: 1.4rem!important; font-weight: 800!important; }
   .badge-buy { background-color: #16a34a; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.0rem; display: inline-block; }
   .badge-sell { background-color: #dc2626; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.0rem; display: inline-block; }
   .badge-wait { background-color: #475569; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 1.0rem; display: inline-block; }
   .alert-box { background: linear-gradient(135deg, rgba(220, 38, 38, 0.2), rgba(245, 158, 11, 0.2)); border-left: 6px solid #f59e0b; padding: 14px; border-radius: 6px; margin-bottom: 16px; color: #f8fafc; }
   .param-box { background-color: rgba(30, 41, 59, 0.8)!important; border-left: 5px solid #3b82f6; padding: 14px; border-radius: 6px; font-family: monospace; line-height: 1.8; color: #f8fafc!important; }
   .param-box code { font-size: 0.95rem!important; font-weight: 700!important; color: #38bdf8!important; background-color: rgba(51, 65, 85, 0.9)!important; padding: 2px 6px; border-radius: 4px; }
   .ai-card { background-color: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
   .news-card { background-color: rgba(30, 41, 59, 0.5); border: 1px solid #475569; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
   .update-time { font-size: 0.85rem; color: #94a3b8; text-align: right; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# Session State 初期化 (ファイル保存を廃止)
DEFAULTS = {
    "account_balance": 500000,
    "quantity_wan": 0.10,
    "auto_refresh": False,
    "risk_percent": 1.0,
    "rr_ratio": 1.5,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "ranges" not in st.session_state:
    st.session_state["ranges"] = {}

FEATURE_COLUMNS = [
    "Return_1", "Return_5", "Dev_SMA20", "Dev_EMA200", "Dev_EMA20_200", "Vol_Ratio",
    "RSI", "RSI_Diff", "MACD_Hist_Ratio", "BB_PctB", "ADX",
    "ATR_Ratio", "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Stoch_K"
]
FEATURE_LABELS_JA = {
    "Return_1": "直近1足変化率", "Return_5": "直近5足変化率", "Dev_SMA20": "SMA20乖離率",
    "Dev_EMA200": "EMA200乖離率", "Dev_EMA20_200": "EMA20/200乖離率", "Vol_Ratio": "ボラティリティ比率",
    "RSI": "RSI(14)", "RSI_Diff": "RSI変化幅", "MACD_Hist_Ratio": "MACDヒストグラム比",
    "BB_PctB": "ボリンジャー%B", "ADX": "ADX(トレンド強度)", "ATR_Ratio": "ATR比率",
    "Upper_Wick_Ratio": "上ヒゲ比率", "Lower_Wick_Ratio": "下ヒゲ比率", "Stoch_K": "ストキャス%K"
}
PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
}
DEFAULT_SPREAD_PIPS = {"USDJPY=X": 0.2, "EURUSD=X": 0.4, "GBPJPY=X": 1.0, "EURJPY=X": 0.5, "AUDJPY=X": 0.6}
TIMEFRAMES = {
    "5分足 (スキャル用)": {"period": "7d", "interval": "5m"},
    "15分足 (デイトレエントリー用)": {"period": "30d", "interval": "15m"},
    "1時間足 (デイトレメイン用)": {"period": "60d", "interval": "1h"},
    "4時間足 (中期・リピート用)": {"period": "60d", "interval": "1h"},
}
LOOKAHEAD_BARS = 8

def clean_series(s):
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0] if s.shape[1] > 0 else pd.Series(dtype=float)
    return s if isinstance(s, pd.Series) else pd.Series(s)

def safe_to_tokyo_tz(df):
    if df is None or df.empty: return df
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
    if df is None or df.empty: return df
    df_out = df.copy()
    if isinstance(df_out.columns, pd.MultiIndex):
        try:
            l0 = [str(x).lower() for x in df_out.columns.get_level_values(0)]
            df_out.columns = df_out.columns.get_level_values(0) if any(c in l0 for c in ["close","open","high","low"]) else df_out.columns.get_level_values(1)
        except Exception:
            df_out.columns = [str(c[0]) for c in df_out.columns]
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    df_out.columns = [str(c).capitalize() for c in df_out.columns]
    return df_out.loc[~df_out.index.duplicated(keep='last')]

@st.cache_data(ttl=300, show_spinner=False)
def get_usdjpy_rate():
    try:
        df = yf.download("USDJPY=X", period="5d", progress=False, timeout=10)
        df = flatten_yf_df(df)
        if df is not None and not df.empty and "Close" in df.columns:
            c = clean_series(df["Close"]).dropna()
            val = float(c.iloc[-1]) if len(c) > 0 else 155.0
            if not np.isnan(val) and val > 0: return val
    except Exception:
        pass
    return 155.0

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news_and_impact(symbol):
    try:
        tk = yf.Ticker(symbol)
        news_list = getattr(tk, "news", None)
        if not news_list or not isinstance(news_list, list): return []
        parsed = []
        for item in news_list[:10]:
            if not isinstance(item, dict): continue
            content = item.get("content", {}) if isinstance(item.get("content"), dict) else item
            title = content.get("title") or item.get("title", "No Title")
            if not title or title == "No Title": continue
            provider = content.get("provider", {}) if isinstance(content.get("provider"), dict) else {}
            publisher = provider.get("displayName") or item.get("publisher", "市場ニュース")
            click_url = "#"
            canonical = content.get("canonicalUrl")
            click_through = content.get("clickThroughUrl")
            if isinstance(canonical, dict) and canonical.get("url"): click_url = canonical.get("url")
            elif isinstance(click_through, dict) and click_through.get("url"): click_url = click_through.get("url")
            elif isinstance(canonical, str): click_url = canonical
            elif "link" in item: click_url = str(item["link"])
            if click_url.startswith("/"): click_url = f"https://finance.yahoo.com{click_url}"
            pub_time = content.get("pubDate") or item.get("providerPublishTime", None)
            if isinstance(pub_time, (int, float)):
                dt_str = datetime.fromtimestamp(pub_time, tz=ZoneInfo("Asia/Tokyo")).strftime("%m/%d %H:%M")
            elif isinstance(pub_time, str):
                try:
                    dt = datetime.fromisoformat(pub_time.replace("Z", "+00:00")).astimezone(ZoneInfo("Asia/Tokyo"))
                    dt_str = dt.strftime("%m/%d %H:%M")
                except Exception: dt_str = "直近"
            else: dt_str = "直近"
            t_lower = str(title).lower()
            impact, direction, reason = "ℹ️ 普通", "↔️ 中立 / ボラティリティ注意", "全般的な市況ニュース"
            if any(k in t_lower for k in ["fed","powell","cpi","gdp","pmi","retail","inflation","rate","payroll","jobs","fomc","yield"]):
                impact = "🔥 高（米金融政策・大変動）"
                if any(k in t_lower for k in ["hike","high","rise","strong","inflation","above","beat"]):
                    direction = "📈 ドル高 / 通貨ペア上昇要因" if "USD" in symbol else "⚡ USD強含み"
                    reason = "米金利高止まり・経済指標好調によるドル買い圧力"
                elif any(k in t_lower for k in ["cut","fall","cool","drop","weak","below","miss"]):
                    direction = "📉 ドル安 / 通貨ペア下落要因" if "USD" in symbol else "⚡ USD弱含み"
                    reason = "米利下げ観測・経済減速に伴うドル売り圧力"
                else:
                    direction, reason = "⚡ 急変動警戒", "米国の重要指標・FRB発言に伴う相場変動"
            elif any(k in t_lower for k in ["boj","ueda","yen","japan","bank of japan"]):
                impact = "🔥 高（円相場直接影響）"
                if any(k in t_lower for k in ["hike","normalize","taper","intervention","strong"]):
                    direction, reason = "📉 円高（USD/JPY等 下落）要因", "日銀利上げ・政策正常化観測に伴う円買い"
                else:
                    direction, reason = "📈 円安（USD/JPY等 上昇）要因", "日銀金融緩和維持観測に伴う円売り"
            elif any(k in t_lower for k in ["ecb","lagarde","euro","europe"]):
                impact, direction, reason = "⚡ 中〜高（ユーロ影響）", "🇪🇺 ユーロ急変動注意", "ECB理事会・欧州経済指標の動き"
            parsed.append({"title":title,"publisher":publisher,"link":click_url,"time":dt_str,"impact":impact,"direction":direction,"reason":reason})
        return parsed
    except Exception:
        return []

@njit
def _compute_targets_fast(c_vals, h_vals, l_vals, tp_vals, sl_vals, n, lookahead):
    target = np.zeros(n, dtype=np.int64)
    for i in range(n - lookahead):
        entry_p = c_vals[i]
        tp_dist = tp_vals[i]
        sl_dist = sl_vals[i]
        if np.isnan(tp_dist) or np.isnan(sl_dist) or tp_dist <= 0 or sl_dist <= 0:
            continue
        outcome = 0
        for j in range(1, lookahead + 1):
            idx = i + j
            if idx >= n: break
            curr_h = h_vals[idx]
            curr_l = l_vals[idx]
            buy_tp_hit = (curr_h - entry_p) >= tp_dist
            buy_sl_hit = (entry_p - curr_l) >= sl_dist
            sell_tp_hit = (entry_p - curr_l) >= tp_dist
            sell_sl_hit = (curr_h - entry_p) >= sl_dist
            # 同時ヒットは判定不能として引き分け扱い
            if (buy_tp_hit and buy_sl_hit) or (sell_tp_hit and sell_sl_hit) or (buy_tp_hit and sell_tp_hit):
                outcome = 0
                break
            if buy_tp_hit and not buy_sl_hit:
                outcome = 1
                break
            if sell_tp_hit and not sell_sl_hit:
                outcome = -1
                break
            if buy_sl_hit or sell_sl_hit:
                outcome = 0
                break
        target[i] = outcome
    return target

@st.cache_data(ttl=300, show_spinner=False)
def load_and_process_data(symbol, period, interval, tf_name=""):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, timeout=10)
        df = flatten_yf_df(df)
        if df is None or df.empty: return None
        if not all(col in df.columns for col in ["Open","High","Low","Close"]): return None
        if "4時間足" in tf_name and interval == "1h":
            tz_before = df.index.tz
            df = df.resample("4h", closed="left", label="left").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
            if tz_before is not None and df.index.tz is None:
                df.index = df.index.tz_localize(tz_before)
        if len(df) < 50: return None
        c = clean_series(df["Close"]); h = clean_series(df["High"]); l = clean_series(df["Low"]); o = clean_series(df["Open"])
        # True RangeベースのATR
        hl = h - l
        hc = (h - c.shift(1)).abs()
        lc = (l - c.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        new_cols = {}
        new_cols["SMA_20"] = c.rolling(20, min_periods=1).mean()
        new_cols["EMA_200"] = c.ewm(span=min(200, len(c)), adjust=False).mean()
        new_cols["ATR"] = tr.rolling(14, min_periods=1).mean()
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
        l14 = l.rolling(14, min_periods=1).min(); h14 = h.rolling(14, min_periods=1).max()
        new_cols["Stoch_K"] = 100 * (c - l14) / ((h14 - l14) + 1e-10)
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        new_cols["MACD"] = macd; new_cols["MACD_Signal"] = macd_signal
        new_cols["MACD_Hist"] = macd - macd_signal
        new_cols["MACD_Hist_Ratio"] = new_cols["MACD_Hist"] / (c + 1e-10)
        std20 = c.rolling(20, min_periods=1).std().fillna(0)
        new_cols["Upper_Band"] = new_cols["SMA_20"] + (std20 * 2)
        new_cols["Lower_Band"] = new_cols["SMA_20"] - (std20 * 2)
        new_cols["BB_PctB"] = (c - new_cols["Lower_Band"]) / ((new_cols["Upper_Band"] - new_cols["Lower_Band"]) + 1e-10)
        up, down = h - h.shift(1), l.shift(1) - l
        up_cond = (up > down) & (up > 0); down_cond = (down > up) & (down > 0)
        plus_di_raw = pd.Series(np.where(up_cond, up, 0.0), index=df.index)
        minus_di_raw = pd.Series(np.where(down_cond, down, 0.0), index=df.index)
        tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean() + 1e-10
        plus_di = 100 * plus_di_raw.ewm(alpha=1/14, adjust=False).mean() / tr_smooth
        minus_di = 100 * minus_di_raw.ewm(alpha=1/14, adjust=False).mean() / tr_smooth
        di_sum = (plus_di + minus_di).replace(0, 1e-10)
        new_cols["ADX"] = (100 * (plus_di - minus_di).abs() / di_sum).ewm(alpha=1/14, adjust=False).mean()
        total_range = hl + 1e-10
        open_close_max = pd.concat([o, c], axis=1).max(axis=1)
        open_close_min = pd.concat([o, c], axis=1).min(axis=1)
        new_cols["Upper_Wick_Ratio"] = (h - open_close_max) / total_range
        new_cols["Lower_Wick_Ratio"] = (open_close_min - l) / total_range
        new_cols["ATR_Ratio"] = new_cols["ATR"] / c

        df_feat = pd.DataFrame(new_cols, index=df.index)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        # 中立値で補完 (0埋めをやめる)
        fill_defaults = {
            "Return_1":0, "Return_5":0, "Dev_SMA20":0, "Dev_EMA200":0, "Dev_EMA20_200":0, "Vol_Ratio":0,
            "RSI":50, "RSI_Diff":0, "MACD_Hist_Ratio":0, "BB_PctB":0.5, "ADX":25,
            "ATR_Ratio":0.005, "Upper_Wick_Ratio":0, "Lower_Wick_Ratio":0, "Stoch_K":50,
            "SMA_20": np.nan, "EMA_200": np.nan, "ATR": np.nan, "MACD":0, "MACD_Signal":0, "MACD_Hist":0, "Upper_Band": np.nan, "Lower_Band": np.nan
        }
        df_feat = df_feat.ffill()
        for col, default in fill_defaults.items():
            if col in df_feat.columns and not np.isnan(default):
                df_feat[col] = df_feat[col].fillna(default)
        df_feat["SMA_20"] = df_feat["SMA_20"].fillna(c); df_feat["EMA_200"] = df_feat["EMA_200"].fillna(c)
        df_feat["ATR"] = df_feat["ATR"].fillna(c * 0.005)
        df_feat["Upper_Band"] = df_feat["Upper_Band"].fillna(c); df_feat["Lower_Band"] = df_feat["Lower_Band"].fillna(c)
        df = pd.concat([df, df_feat], axis=1)

        tp_t = new_cols["ATR"] * 0.9; sl_t = new_cols["ATR"] * 0.6
        c_vals = c.values.astype(float); h_vals = h.values.astype(float); l_vals = l.values.astype(float)
        tp_vals = tp_t.values.astype(float); sl_vals = sl_t.values.astype(float)
        n = len(df)
        target_values = _compute_targets_fast(c_vals, h_vals, l_vals, tp_vals, sl_vals, n, LOOKAHEAD_BARS)
        target_series = pd.Series(target_values, index=df.index, dtype="float64")
        target_series.iloc[-LOOKAHEAD_BARS:] = np.nan
        df["Target"] = target_series
        feat_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
        return df.dropna(subset=feat_cols)
    except Exception as e:
        print(e)
        return None

@st.cache_data(ttl=60, show_spinner=False)
def analyze_signal_with_backtest(df_current, df_htf):
    empty_res = ("WAIT (データ不足)", 0.0, 0.0, "不明", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, pd.Series(dtype=float))
    if df_current is None or len(df_current) < 150: return empty_res
    try:
        avail = [f for f in FEATURE_COLUMNS if f in df_current.columns]
        df_valid = df_current.dropna(subset=["Target"]).copy()
        df_valid["Target"] = df_valid["Target"].astype(int)
        test_size, gap = 80, LOOKAHEAD_BARS
        if len(df_valid) < test_size + gap + 40: return ("WAIT (学習データ不足)", 0.0, 0.0, "判定不可", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, pd.Series(dtype=float))
        X = df_valid[avail].replace([np.inf, -np.inf], np.nan).fillna(0); y = df_valid["Target"]
        X_train = X.iloc[: -(test_size + gap)]; y_train = y.iloc[: -(test_size + gap)]
        X_test = X.iloc[-test_size:]; y_test = y.iloc[-test_size:]
        if len(np.unique(y_train)) < 2: return ("WAIT (データ偏り)", 0.0, 0.0, "判定不可", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, pd.Series(dtype=float))
        model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_series = pd.Series(preds, index=y_test.index)
        trade_signals = preds_series!= 0
        win_rate = float((preds_series[trade_signals] == y_test[trade_signals]).sum() / trade_signals.sum() * 100) if trade_signals.sum() > 0 else 50.0
        latest_X = df_current[avail].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
        prob_array = model.predict_proba(latest_X)[0]
        class_prob_map = {int(cls_val): float(p) for cls_val, p in zip(model.classes_, prob_array)}
        prob_up_raw = class_prob_map.get(1, 0.0); prob_down_raw = class_prob_map.get(-1, 0.0); prob_wait_raw = class_prob_map.get(0, 0.0)
        # 確率はモデルそのままを保持
        prob_dict = {"buy": round(prob_up_raw*100,1), "sell": round(prob_down_raw*100,1), "wait": round(prob_wait_raw*100,1)}
        # フィルター用に別途指標取得
        curr_close = float(clean_series(df_current["Close"]).iloc[-1])
        rsi_val = float(clean_series(df_current["RSI"]).iloc[-1]) if "RSI" in df_current.columns else 50.0
        adx_val = float(clean_series(df_current["ADX"]).iloc[-1]) if "ADX" in df_current.columns else 20.0
        htf_bullish, htf_bearish = True, False
        if df_htf is not None and not df_htf.empty and "Close" in df_htf.columns:
            htf_close = float(clean_series(df_htf["Close"]).iloc[-1])
            if "EMA_200" in df_htf.columns and not np.isnan(clean_series(df_htf["EMA_200"]).iloc[-1]):
                htf_ema200 = float(clean_series(df_htf["EMA_200"]).iloc[-1])
                htf_bullish = htf_close > htf_ema200; htf_bearish = htf_close < htf_ema200
        ja_index = [FEATURE_LABELS_JA.get(col, col) for col in avail]
        importances = pd.Series(model.feature_importances_, index=ja_index).sort_values(ascending=True)
        threshold = 0.42
        # 基本判定は確率最大値
        if prob_up_raw > prob_down_raw and prob_up_raw > prob_wait_raw and prob_up_raw >= threshold:
            status, conf = "BUY (買い)", prob_up_raw*100
        elif prob_down_raw > prob_up_raw and prob_down_raw > prob_wait_raw and prob_down_raw >= threshold:
            status, conf = "SELL (売り)", prob_down_raw*100
        else:
            status, conf = "WAIT (様子見)", max(prob_up_raw, prob_down_raw, prob_wait_raw)*100
        # トレンドフィルターでブロック (確率には加算しない)
        if "BUY" in status:
            if htf_bearish or rsi_val > 72: status = f"WAIT (上位足逆行/RSI高 {status}見送り)"
        if "SELL" in status:
            if htf_bullish or rsi_val < 28: status = f"WAIT (上位足逆行/RSI低 {status}見送り)"
        m_type = "トレンド相場" if adx_val > 22.0 else "レンジ相場"
        return status, conf, win_rate, m_type, prob_dict, importances
    except Exception:
        return ("WAIT (エラー)", 0.0, 0.0, "エラー", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, pd.Series(dtype=float))

# --- UI ---
st.title("Pro FX Analyzer & Signal Pro - Fixed")
col_sel1, col_sel2 = st.columns(2)
with col_sel1: selected_label = st.selectbox("通貨ペア", list(PAIRS.keys()), key="selected_pair_label")
with col_sel2: tf_label = st.selectbox("時間足", list(TIMEFRAMES.keys()), key="selected_tf_label")
ticker = PAIRS[selected_label]; tf_config = TIMEFRAMES[tf_label]
is_jpy = "JPY" in ticker; pip_unit = 0.01 if is_jpy else 0.0001; price_fmt = "%.3f" if is_jpy else "%.5f"
spread_pips = DEFAULT_SPREAD_PIPS.get(ticker, 0.3)

st.sidebar.header("⚙️ 資金 & リスク設定")
st.sidebar.number_input("口座資金 (円)", min_value=10000, step=50000, key="account_balance")
st.sidebar.number_input("1注文の基本数量 (万通貨)", min_value=0.01, step=0.01, key="quantity_wan")
st.sidebar.markdown("---")
st.sidebar.header("🎯 単発トレード リスク設定")
st.sidebar.number_input("1取引あたりの許容リスク (%)", min_value=0.1, max_value=10.0, step=0.1, key="risk_percent", help="1回の損切で許容する口座資金の割合")
st.sidebar.number_input("デフォルト リスクリワード比 (RR)", min_value=0.5, max_value=5.0, step=0.1, key="rr_ratio")
st.sidebar.markdown("---")
st.sidebar.header("🔄 更新設定")
if HAS_AUTOREFRESH:
    st.sidebar.checkbox("60秒ごとに自動更新", key="auto_refresh")
    if st.session_state["auto_refresh"]: st_autorefresh(interval=60000, key="datarefresh")
else:
    st.sidebar.caption("`pip install streamlit-autorefresh` で自動更新有効")
if st.sidebar.button("🔄 最新データに更新"):
    st.cache_data.clear(); st.rerun()

with st.spinner("最新相場データ & ニュースを取得中..."):
    data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
    data_4h = load_and_process_data(ticker, "60d", "1h", "4時間足 (中期・リピート用)")
    data_htf = load_and_process_data(ticker, "2y", "1d", "日足")
    news_items = fetch_news_and_impact(ticker)
if data_4h is None and data is not None: data_4h = data
if data is None:
    st.error("データ取得に失敗しました。取引時間外かYahoo Financeの遅延です。"); st.stop()

now_jst = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f'<div class="update-time">最終データ取得日時: <b>{now_jst} JST</b> | yfinance価格は松井証券配信価格と乖離する場合があります</div>', unsafe_allow_html=True)
latest_price = float(clean_series(data["Close"]).iloc[-1])
latest_atr = float(clean_series(data["ATR"]).iloc[-1]) if "ATR" in data.columns else (latest_price * 0.005)
status, conf, win_rate, m_type, prob_dict, feature_importances = analyze_signal_with_backtest(data, data_htf)

bb_pct = float(clean_series(data["BB_PctB"]).iloc[-1]) if "BB_PctB" in data.columns else 0.5
adx_val_curr = float(clean_series(data["ADX"]).iloc[-1]) if "ADX" in data.columns else 20.0
vol_ratio_curr = float(clean_series(data["Vol_Ratio"]).iloc[-1]) if "Vol_Ratio" in data.columns else 0.01
is_volatility_expanding = vol_ratio_curr > (clean_series(data["Vol_Ratio"]).mean() * 1.2)
is_squeeze = bb_pct < 0.15 or bb_pct > 0.85
vol_alert_msg = None
if is_volatility_expanding or adx_val_curr > 28.0:
    if prob_dict["buy"] > prob_dict["sell"]:
        vol_alert_msg = f"⚠️ **【急変動・上方ブレイク警戒】** ボラ拡大。上方向（買い優勢・確率 {prob_dict['buy']}%）へ動くエネルギーが高まっています！"
    else:
        vol_alert_msg = f"⚠️ **【急変動・下方ブレイク警戒】** ボラ拡大。下方向（売り優勢・確率 {prob_dict['sell']}%）へ動くエネルギーが高まっています！"
elif is_squeeze:
    vol_alert_msg = f"⚡ **【エネルギー蓄積中（スクイーズ）】** バンド収縮中。まもなく大きく跳ねる前兆です。"
if vol_alert_msg: st.markdown(f'<div class="alert-box">{vol_alert_msg}</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("現在レート", price_fmt % latest_price)
badge_html = f'<div class="badge-buy">{status}</div>' if "BUY" in status and "WAIT" not in status else f'<div class="badge-sell">{status}</div>' if "SELL" in status and "WAIT" not in status else f'<div class="badge-wait">{status}</div>'
m2.markdown("**AI 推奨アクション**"); m2.markdown(badge_html, unsafe_allow_html=True)
m3.metric("直近バックテスト勝率", f"{win_rate:.1f}%", f"確信度: {conf:.1f}%")
m4.metric("相場環境", m_type, f"ATR: {latest_atr/pip_unit:.1f} pips")
st.markdown("---")

# レンジ管理
swing_high_4h = float(clean_series(data_4h["High"]).iloc[-100:].max()) if data_4h is not None else latest_price * 1.02)
swing_low_4h = float(clean_series(data_4h["Low"]).iloc[-100:].min() if data_4h is not None else latest_price * 0.98)
atr_4h = float(clean_series(data_4h["ATR"]).iloc[-1]) if data_4h is not None and "ATR" in data_4h.columns else (latest_price * 0.005)
def is_valid_range(r_low, r_up, current_p):
    if r_low <= 0 or r_up <= 0 or r_low >= r_up: return False
    if r_low < current_p * 0.5 or r_up > current_p * 1.5: return False
    return True
need_reset = False
if ticker not in st.session_state["ranges"]: need_reset = True
else:
    ex_low = float(st.session_state["ranges"][ticker].get("lower", 0)); ex_up = float(st.session_state["ranges"][ticker].get("upper", 0))
    if not is_valid_range(ex_low, ex_up, latest_price): need_reset = True
if need_reset: st.session_state["ranges"][ticker] = {"lower": float(swing_low_4h), "upper": float(swing_high_4h)}
key_lower = f"in_lower_{ticker}"; key_upper = f"in_upper_{ticker}"
current_range = st.session_state["ranges"][ticker]
if key_lower not in st.session_state: st.session_state[key_lower] = float(current_range["lower"])
if key_upper not in st.session_state: st.session_state[key_upper] = float(current_range["upper"])
with st.expander("⚙️ リピート自動売買のレンジ調整", expanded=False):
    rc1, rc2 = st.columns(2)
    def update_range_callback():
        st.session_state["ranges"][ticker] = {"lower": float(st.session_state[key_lower]), "upper": float(st.session_state[key_upper])}
    def trigger_auto_calc_callback():
        st.session_state["ranges"][ticker] = {"lower": float(swing_low_4h), "upper": float(swing_high_4h)}
        st.session_state[key_lower] = float(swing_low_4h); st.session_state[key_upper] = float(swing_high_4h)
    rc1.number_input("レンジ下限", step=0.1 if is_jpy else 0.001, format=price_fmt, key=key_lower, on_change=update_range_callback)
    rc2.number_input("レンジ上限", step=0.1 if is_jpy else 0.001, format=price_fmt, key=key_upper, on_change=update_range_callback)
    st.button("✨ 4時間足高値・安値からレンジを自動計算", on_click=trigger_auto_calc_callback)
    user_lower = min(st.session_state[key_lower], st.session_state[key_upper])
    user_upper = max(st.session_state[key_lower], st.session_state[key_upper])
    if user_lower == user_upper: user_upper += pip_unit * 100.0
    user_half = (user_upper + user_lower) / 2.0
else:
    user_lower = min(st.session_state[key_lower], st.session_state[key_upper])
    user_upper = max(st.session_state[key_lower], st.session_state[key_upper])
    if user_lower == user_upper: user_upper += pip_unit * 100.0
    user_half = (user_upper + user_lower) / 2.0

stop_buffer_pips = max(10.0, round((atr_4h / pip_unit) * 1.5, 1))
stop_buffer_val = stop_buffer_pips * pip_unit
buy_stop_loss = user_lower - stop_buffer_val; sell_stop_loss = user_upper + stop_buffer_val

tab_chart, tab_single, tab_news, tab_repeat, tab_ai = st.tabs(["📈 メインチャート","⚡ 単発トレード (AI全自動アシスト)","📰 経済指標＆為替ニュースAI予測","📋 松井証券 リピート設定 & リスク管理","🤖 AIモデル分析詳細"])
with tab_chart:
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    with ctrl_col1: bars_count = st.slider("表示本数", 30, 300, 90, 10, key="chart_bars_slider")
    with ctrl_col2: show_bb = st.checkbox("ボリンジャーバンド(±2σ)", True, key="show_bb_check")
    with ctrl_col3: show_repeat_lines = st.checkbox("リピートレンジ表示", "4時間足" in tf_label, key="show_repeat_lines_check")
    with ctrl_col4: sub_indicator = st.selectbox("サブ指標", ["RSI (14)", "MACD", "なし"], 0, key="sub_indicator_select")
    df_chart = safe_to_tokyo_tz(data.tail(bars_count)).ffill().loc[lambda d: ~d.index.duplicated(keep='last')]
    fmt_str = '%m/%d %H:%M' if "分" in tf_label or "時間" in tf_label else '%Y/%m/%d'
    x_labels = df_chart.index.strftime(fmt_str) if isinstance(df_chart.index, pd.DatetimeIndex) else df_chart.index.astype(str)
    has_sub = sub_indicator!= "なし"; row_heights = [0.75, 0.25] if has_sub else [1.0]; rows_num = 2 if has_sub else 1
    fig = make_subplots(rows=rows_num, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.06)
    fig.add_trace(go.Candlestick(x=x_labels, open=clean_series(df_chart["Open"]), high=clean_series(df_chart["High"]), low=clean_series(df_chart["Low"]), close=clean_series(df_chart["Close"]), increasing_line_color='#22c55e', increasing_fillcolor='#22c55e', decreasing_line_color='#ef4444', decreasing_fillcolor='#ef4444', name="価格"), row=1, col=1)
    if "SMA_20" in df_chart.columns: fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["SMA_20"]), line=dict(color="#f59e0b", width=1.5), name="SMA20"), row=1, col=1)
    if "EMA_200" in df_chart.columns: fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["EMA_200"]), line=dict(color="#38bdf8", width=2.0), name="EMA200"), row=1, col=1)
    if show_bb and "Upper_Band" in df_chart.columns:
        fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["Upper_Band"]), line=dict(color="rgba(148, 163, 184, 0.4)", width=1, dash="dot"), name="BB +2σ"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["Lower_Band"]), line=dict(color="rgba(148, 163, 184, 0.4)", width=1, dash="dot"), fill='tonexty', fillcolor='rgba(148, 163, 184, 0.05)', name="BB -2σ"), row=1, col=1)
    if show_repeat_lines:
        if not np.isnan(user_lower) and not np.isnan(user_upper): fig.add_hrect(y0=user_lower, y1=user_upper, fillcolor="rgba(56, 189, 248, 0.05)", line_width=0, row=1, col=1)
        fig.add_hline(y=sell_stop_loss, line_dash="dashdot", line_color="#dc2626", annotation_text="売SL", row=1, col=1)
        fig.add_hline(y=user_upper, line_dash="dash", line_color="#f87171", annotation_text="上限", row=1, col=1)
        fig.add_hline(y=user_half, line_dash="dot", line_color="#c084fc", annotation_text="中央", row=1, col=1)
        fig.add_hline(y=user_lower, line_dash="dash", line_color="#4ade80", annotation_text="下限", row=1, col=1)
        fig.add_hline(y=buy_stop_loss, line_dash="dashdot", line_color="#16a34a", annotation_text="買SL", row=1, col=1)
    if sub_indicator == "RSI (14)" and "RSI" in df_chart.columns:
        fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["RSI"]), line=dict(color="#a855f7", width=1.8), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=2, col=1); fig.add_hline(y=30, line_dash="dot", line_color="#22c55e", row=2, col=1)
        fig.update_yaxes(range=[0, 100], side="right", row=2, col=1)
    elif sub_indicator == "MACD" and "MACD" in df_chart.columns:
        fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["MACD"]), line=dict(color="#38bdf8", width=1.5), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["MACD_Signal"]), line=dict(color="#f59e0b", width=1.5), name="Signal"), row=2, col=1)
        hist_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in clean_series(df_chart["MACD_Hist"])]
        fig.add_trace(go.Bar(x=x_labels, y=clean_series(df_chart["MACD_Hist"]), marker_color=hist_colors, name="Hist"), row=2, col=1)
        fig.update_yaxes(side="right", row=2, col=1)
    chart_high = float(clean_series(df_chart["High"]).max()); chart_low = float(clean_series(df_chart["Low"]).min())
    y_max_bound, y_min_bound = max(chart_high, chart_low), min(chart_high, chart_low)
    price_span = y_max_bound - y_min_bound if y_max_bound!= y_min_bound else y_max_bound * 0.002
    fig.update_layout(xaxis_rangeslider_visible=False, height=580, margin=dict(l=10, r=70, t=25, b=10), template="plotly_dark", hovermode="x unified", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    fig.update_xaxes(type='category', nticks=min(len(x_labels), 12), tickangle=-25, showspikes=True)
    fig.update_yaxes(range=[y_min_bound - price_span*0.05, y_max_bound + price_span*0.05], fixedrange=False, side="right", tickformat=".3f" if is_jpy else ".5f", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})

with tab_single:
    st.markdown("##### ⚡ 単発トレード（スキャル・デイトレ）AI発注アシスタント")
    usd_rate = latest_price if ticker == "USDJPY=X" else get_usdjpy_rate()
    pip_val_per_wan = 100.0 if is_jpy else (0.0001 * usd_rate * 10000.0)
    account_bal = float(st.session_state["account_balance"]); risk_pct = float(st.session_state["risk_percent"])
    max_risk_yen = account_bal * (risk_pct / 100.0)
    ai_is_buy = "BUY" in status and "WAIT" not in status; ai_is_sell = "SELL" in status and "WAIT" not in status
    ai_direction = "BUY (買い)" if ai_is_buy else "SELL (売り)" if ai_is_sell else ("BUY (買い)" if prob_dict["buy"] >= prob_dict["sell"] else "SELL (売り)")
    ai_sl_pips = round((latest_atr / pip_unit) * 1.5, 1); ai_rr_ratio = float(st.session_state["rr_ratio"]); ai_tp_pips = round(ai_sl_pips * ai_rr_ratio, 1)
    ai_entry = latest_price
    ai_sl_price = ai_entry - (ai_sl_pips * pip_unit) if "BUY" in ai_direction else ai_entry + (ai_sl_pips * pip_unit)
    ai_tp_price = ai_entry + (ai_tp_pips * pip_unit) if "BUY" in ai_direction else ai_entry - (ai_tp_pips * pip_unit)
    ai_loss_per_wan = (ai_sl_pips + spread_pips) * pip_val_per_wan
    ai_rec_wan = max(0.01, round((max_risk_yen / ai_loss_per_wan) if ai_loss_per_wan > 0 else 0.01, 2))
    ai_profit_yen = (ai_tp_pips - spread_pips) * pip_val_per_wan * ai_rec_wan
    ai_loss_yen = (ai_sl_pips + spread_pips) * pip_val_per_wan * ai_rec_wan
    req_margin_yen = (ai_entry * (ai_rec_wan * 10000)) / 25.0 if is_jpy else (ai_entry * usd_rate * (ai_rec_wan * 10000)) / 25.0
    entry_key = f"entry_{ticker}"; side_key = f"side_{ticker}"; slpips_key = f"slpips_{ticker}"; slmode_key = f"slmode_{ticker}"
    if side_key not in st.session_state: st.session_state[side_key] = "BUY (買い)" if "BUY" in ai_direction else "SELL (売り)"
    if entry_key not in st.session_state: st.session_state[entry_key] = float(latest_price)
    if slpips_key not in st.session_state: st.session_state[slpips_key] = float(ai_sl_pips)
    if slmode_key not in st.session_state: st.session_state[slmode_key] = "ATRベース (推奨)"
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown("#### 🤖 AI提案トレードプラン")
    if "WAIT" in status: st.warning(f"⚠️ **現在AIシグナルは【{status}】です。** 根拠不十分または上位足と逆行のため見送り推奨。モデル純確率 BUY {prob_dict['buy']}% / SELL {prob_dict['sell']}%")
    else: st.success(f"🎯 **高確信度AIシグナル: 【{status}】（確信度: {conf:.1f}% / 勝率目安: {win_rate:.1f}%）**")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("推奨売買方向", ai_direction); a2.metric("想定エントリー", price_fmt % ai_entry)
    a3.metric("推奨損切 (SL)", price_fmt % ai_sl_price, f"-{ai_sl_pips} pips")
    a4.metric("推奨利確 (TP)", price_fmt % ai_tp_price, f"+{ai_tp_pips} pips")
    a5.metric("最適数量", f"{ai_rec_wan:.2f} 万通貨", f"必要証拠金: 約{int(req_margin_yen):,}円")
    st.markdown(f"""
    <div class="param-box">
    <b>【AI自動提示 注文コピー用パラメータ】</b> (想定スプレッド: <code>{spread_pips:.1f} pips</code> 含む)<br>
    ・<b>通貨ペア</b>: <code>{selected_label}</code> | <b>売買方向</b>: <code>{"買い (BUY)" if "BUY" in ai_direction else "売り (SELL)"}</code><br>
    ・<b>新規成行価格</b>: <code>{price_fmt % ai_entry}</code><br>
    ・<b>決済利確(TP)</b>: <code>{price_fmt % ai_tp_price}</code> (+{ai_tp_pips} pips / 純益 ＋{int(ai_profit_yen):,}円)<br>
    ・<b>決済損切(SL)</b>: <code>{price_fmt % ai_sl_price}</code> (-{ai_sl_pips} pips / 損失 －{int(ai_loss_yen):,}円)<br>
    ・<b>推奨発注数量</b>: <code>{ai_rec_wan:.2f} 万通貨</code> ({int(ai_rec_wan * 10000):,} 通貨) | <b>必要証拠金</b>: 約 <code>{int(req_margin_yen):,}円</code>
    </div>
    """, unsafe_allow_html=True)
    def apply_ai_proposal():
        st.session_state[side_key] = "BUY (買い)" if "BUY" in ai_direction else "SELL (売り)"
        st.session_state[entry_key] = float(ai_entry); st.session_state[slpips_key] = float(ai_sl_pips); st.session_state[slmode_key] = "固定 pips"
    def reset_entry_to_latest(): st.session_state[entry_key] = float(latest_price)
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1: st.button("🤖 AIの提案値を手動設定に反映", on_click=apply_ai_proposal)
    with c_btn2: st.button("📍 手動設定のレートを現在価格へ更新", on_click=reset_entry_to_latest)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("##### 🛠️ 注文条件の手動微調整")
    sc1, sc2 = st.columns(2)
    with sc1:
        trade_side = st.radio("売買方向", ["BUY (買い)", "SELL (売り)"], key=side_key, horizontal=True)
        entry_price = st.number_input("エントリー想定レート", key=entry_key, format=price_fmt, step=0.01 if is_jpy else 0.0001)
    with sc2:
        sl_mode = st.radio("損切(SL)の決め方", ["ATRベース (推奨)", "固定 pips"], horizontal=True, key=slmode_key)
        if sl_mode == "ATRベース (推奨)":
            atr_multiplier = st.slider("ATR倍率", 0.5, 3.0, 1.5, 0.1, key=f"atrmul_{ticker}")
            sl_pips = round((latest_atr / pip_unit) * atr_multiplier, 1); st.info(f"現在のATR: {latest_atr/pip_unit:.1f} pips ➔ 損切幅: **{sl_pips} pips**")
        else: sl_pips = st.number_input("損切幅 (pips)", 1.0, step=1.0, key=slpips_key)
    rr_value = st.slider("リスクリワード比 (RR)", 0.5, 4.0, float(st.session_state["rr_ratio"]), 0.1, key=f"rr_{ticker}")
    tp_pips = round(sl_pips * rr_value, 1)
    is_buy = "BUY" in trade_side
    sl_price = entry_price - (sl_pips * pip_unit) if is_buy else entry_price + (sl_pips * pip_unit)
    tp_price = entry_price + (tp_pips * pip_unit) if is_buy else entry_price - (tp_pips * pip_unit)
    loss_per_wan = (sl_pips + spread_pips) * pip_val_per_wan
    recommended_wan = max(0.01, round((max_risk_yen / loss_per_wan) if loss_per_wan > 0 else 0.01, 2))
    expected_profit_yen = (tp_pips - spread_pips) * pip_val_per_wan * recommended_wan
    actual_loss_yen = (sl_pips + spread_pips) * pip_val_per_wan * recommended_wan
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric("調整後の手動ロット数", f"{recommended_wan:.2f} 万通貨", f"許容リスク {risk_pct}%")
    res_col2.metric("損切(SL) レート", price_fmt % sl_price, f"-{sl_pips} pips")
    res_col3.metric("利確(TP) レート", price_fmt % tp_price, f"+{tp_pips} pips")
    res_col4.metric("想定純損益", f"+{int(expected_profit_yen):,}円", f"最大損失 -{int(actual_loss_yen):,}円")

with tab_news:
    st.markdown("##### 📰 リアルタイム為替ニュース & AI事前影響予測")
    with st.expander("🔔 注目すべき主要経済指標と為替への影響パターン", True):
        st.markdown("| 経済指標 | 注目ポイント | ドル円への影響予測 |\n| :--- | :--- | :--- |\n| **FOMC** | 政策金利 & パウエル発言 | **利上げ/タカ派** ➔ 📈 ドル高 / **利下げ** ➔ 📉 ドル安 |\n| **米雇用統計** | NFP & 平均時給 | **予想上回る** ➔ 📈 ドル高 |\n| **米CPI** | インフレ率 | **加速** ➔ 📈 ドル高 / **鈍化** ➔ 📉 ドル安 |\n| **日銀会合** | 植田総裁会見 | **利上げ** ➔ 📉 円高 / **緩和維持** ➔ 📈 円安 |")
    st.markdown(f"##### 🌐 `{selected_label}` 関連の最新ニュース速報 & AI予測")
    if not news_items: st.info("現在関連ニュースが取得できないか、市場が落ち着いています。")
    else:
        for news in news_items:
            st.markdown(f"""
            <div class="news-card">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span style="font-weight:bold;color:#38bdf8;font-size:0.9rem;">{html.escape(news.get('publisher',''))} ({html.escape(news.get('time',''))})</span>
                    <span style="font-weight:bold;font-size:0.85rem;">{html.escape(news.get('impact',''))}</span>
                </div>
                <div style="font-size:1.05rem;font-weight:700;margin-bottom:8px;">
                    <a href="{html.escape(news.get('link','#'))}" target="_blank" style="color:#f8fafc;text-decoration:none;">{html.escape(news.get('title',''))} 🔗</a>
                </div>
                <div style="background-color:rgba(15,23,42,0.6);padding:8px 12px;border-radius:4px;font-size:0.9rem;">
                    🎯 <b>AI予測影響</b>: <span style="color:#facc15;font-weight:bold;">{html.escape(news.get('direction',''))}</span><br>
                    💡 <b>判定理由</b>: {html.escape(news.get('reason',''))}
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab_repeat:
    default_trap_pips = max(15, int(round((atr_4h / pip_unit))))
    trap_width_key = f"trap_width_{ticker}"
    if trap_width_key not in st.session_state: st.session_state[trap_width_key] = default_trap_pips
    trap_width_pips = st.number_input("注文幅 / 利確幅 (pips)", 5, 500, key=trap_width_key, step=5)
    half_range_pips = abs(user_upper - user_half) / pip_unit
    half_grid_count = max(1, int(np.floor(half_range_pips / max(1.0, float(trap_width_pips)))))
    total_grid_count = half_grid_count * 2
    quantity_wan_val = float(st.session_state["quantity_wan"]); account_balance_val = float(st.session_state["account_balance"])
    order_units = int(quantity_wan_val * 10000); usd_rate_r = latest_price if ticker == "USDJPY=X" else get_usdjpy_rate()
    pip_value_yen = (order_units / 10000.0) * 100 if is_jpy else (order_units * 0.0001 * usd_rate_r)
    max_loss_yen = sum((k * trap_width_pips + stop_buffer_pips) * pip_value_yen for k in range(half_grid_count))
    margin_per_order = (latest_price * order_units) / 25.0 if is_jpy else (latest_price * usd_rate_r * order_units) / 25.0
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
    st.code(f"""[松井証券リピート注文 設定値]
通貨ペア: {selected_label}
注文種別: ハーフ＆ハーフ
買いレンジ: {price_fmt % user_lower} - {price_fmt % user_half} (SL: {price_fmt % buy_stop_loss})
売りレンジ: {price_fmt % user_half} - {price_fmt % user_upper} (SL: {price_fmt % sell_stop_loss})
注文幅 / 利確幅: {trap_width_pips} pips
1本あたりの数量: {quantity_wan_val:.2f} 万通貨""", language="text")
    st.markdown("##### 🛡️ リスク・資金シミュレーション")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("想定最大含み損", f"約 {int(max_loss_yen):,} 円"); rc2.metric("片側最大 必要証拠金", f"約 {int(total_margin_yen):,} 円"); rc3.metric("資金リスク比率", f"{risk_ratio:.1f}%")
    if risk_ratio > 40.0: st.error("🚨 警告: 撤退時の最大損失が口座資金の40%超。数量を減らすか資金を増やしてください。")
    else: st.success("🟢 資金管理チェック: 適切なリスク範囲内です。")
    st.markdown("---")
    estimated_monthly_turns = max(5, int((latest_atr / (trap_width_pips * pip_unit)) * 15))
    est_monthly_profit = estimated_monthly_turns * (trap_width_pips * pip_value_yen)
    est_monthly_roi = (est_monthly_profit / account_balance_val) * 100 if account_balance_val > 0 else 0.0
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("推定月間利確回数", f"約 {estimated_monthly_turns} 回"); sc2.metric("推定月間利益額", f"約 +{int(est_monthly_profit):,} 円"); sc3.metric("推定月間利回り (ROI)", f"約 {est_monthly_roi:.1f}% / 月")

with tab_ai:
    st.markdown("##### 🤖 AI予測モデル（MTF多重時間足 × 機械学習）の評価と内訳")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**最新バーの分類判定確率（モデル純出力）**")
        st.write(f"🟢 **BUY (買い)**: {prob_dict['buy']}%"); st.progress(max(0.0, min(1.0, prob_dict['buy']/100.0)))
        st.write(f"🔴 **SELL (売り)**: {prob_dict['sell']}%"); st.progress(max(0.0, min(1.0, prob_dict['sell']/100.0)))
        st.write(f"⚪ **WAIT (様子見)**: {prob_dict['wait']}%"); st.progress(max(0.0, min(1.0, prob_dict['wait']/100.0)))
    with pcol2:
        st.markdown("**アウトオブサンプル検証（同時ヒットを除外）**")
        st.metric("直近テスト80足の実効勝率", f"{win_rate:.1f}%")
        st.caption("※ 未来データの先読みを排除し、到達順序不明なケースは引き分けとして除外した検証精度です。")
    if feature_importances is not None and not feature_importances.empty:
        st.markdown("---")
        st.markdown("##### 📊 AIの判断根拠（特徴量重要度 TOP 10）")
        top10_imp = feature_importances.tail(10)
        fig_imp = go.Figure(go.Bar(x=top10_imp.values, y=top10_imp.index, orientation='h', marker_color='#38bdf8'))
        fig_imp.update_layout(height=340, margin=dict(l=10, r=20, t=10, b=30), template="plotly_dark", xaxis_title="重要度スコア", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
