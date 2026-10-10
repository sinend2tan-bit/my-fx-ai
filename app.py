# ==========================================
# Pro FX Analyzer & Signal Pro - Fixed v3 Final
# 修正: SyntaxError解消 / 同時ヒットを引分け扱い / numba削除
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

st.set_page_config(page_title="Pro FX Analyzer & Signal Pro", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.markdown("""
<style>
   .main.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
   .badge-buy { background-color: #16a34a; color: #fff; padding: 6px 14px; border-radius: 6px; font-weight: bold; }
   .badge-sell { background-color: #dc2626; color: #fff; padding: 6px 14px; border-radius: 6px; font-weight: bold; }
   .badge-wait { background-color: #475569; color: #fff; padding: 6px 14px; border-radius: 6px; font-weight: bold; }
   .param-box { background-color: rgba(30,41,59,0.8)!important; border-left: 5px solid #3b82f6; padding: 14px; border-radius: 6px; font-family: monospace; color: #f8fafc!important; }
   .param-box code { color: #38bdf8!important; background-color: rgba(51,65,85,0.9)!important; padding: 2px 6px; border-radius: 4px; }
   .news-card { background-color: rgba(30,41,59,0.5); border: 1px solid #475569; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
   .update-time { font-size: 0.85rem; color: #94a3b8; text-align: right; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

DEFAULTS = {"account_balance": 500000, "quantity_wan": 0.10, "auto_refresh": False, "risk_percent": 1.0, "rr_ratio": 1.5}
for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v
if "ranges" not in st.session_state: st.session_state["ranges"] = {}

FEATURE_COLUMNS = ["Return_1","Return_5","Dev_SMA20","Dev_EMA200","Dev_EMA20_200","Vol_Ratio","RSI","RSI_Diff","MACD_Hist_Ratio","BB_PctB","ADX","ATR_Ratio","Upper_Wick_Ratio","Lower_Wick_Ratio","Stoch_K"]
FEATURE_LABELS_JA = {"Return_1":"直近1足変化率","Return_5":"直近5足変化率","Dev_SMA20":"SMA20乖離率","Dev_EMA200":"EMA200乖離率","Dev_EMA20_200":"EMA20/200乖離率","Vol_Ratio":"ボラティリティ比率","RSI":"RSI(14)","RSI_Diff":"RSI変化幅","MACD_Hist_Ratio":"MACDヒストグラム比","BB_PctB":"ボリンジャー%B","ADX":"ADX(トレンド強度)","ATR_Ratio":"ATR比率","Upper_Wick_Ratio":"上ヒゲ比率","Lower_Wick_Ratio":"下ヒゲ比率","Stoch_K":"ストキャス%K"}
PAIRS = {"米ドル / 円 (USD/JPY)": "USDJPY=X","ポンド / 円 (GBP/JPY)": "GBPJPY=X","ユーロ / 円 (EUR/JPY)": "EURJPY=X","豪ドル / 円 (AUD/JPY)": "AUDJPY=X","ユーロ / 米ドル (EUR/USD)": "EURUSD=X"}
DEFAULT_SPREAD_PIPS = {"USDJPY=X": 0.2,"EURUSD=X": 0.4,"GBPJPY=X": 1.0,"EURJPY=X": 0.5,"AUDJPY=X": 0.6}
TIMEFRAMES = {"5分足 (スキャル用)": {"period": "7d", "interval": "5m"},"15分足 (デイトレエントリー用)": {"period": "30d", "interval": "15m"},"1時間足 (デイトレメイン用)": {"period": "60d", "interval": "1h"},"4時間足 (中期・リピート用)": {"period": "60d", "interval": "1h"}}
LOOKAHEAD_BARS = 8

def clean_series(s):
    if isinstance(s, pd.DataFrame): return s.iloc[:, 0] if s.shape[1] > 0 else pd.Series(dtype=float)
    return s if isinstance(s, pd.Series) else pd.Series(s)

def safe_to_tokyo_tz(df):
    if df is None or df.empty: return df
    df_out = df.copy()
    try:
        if isinstance(df_out.index, pd.DatetimeIndex):
            if df_out.index.tz is None: df_out.index = df_out.index.tz_localize("UTC")
            df_out.index = df_out.index.tz_convert("Asia/Tokyo")
    except Exception: pass
    return df_out

def flatten_yf_df(df):
    if df is None or df.empty: return df
    df_out = df.copy()
    if isinstance(df_out.columns, pd.MultiIndex):
        try:
            l0 = [str(x).lower() for x in df_out.columns.get_level_values(0)]
            df_out.columns = df_out.columns.get_level_values(0) if any(c in l0 for c in ["close","open","high","low"]) else df_out.columns.get_level_values(1)
        except Exception: df_out.columns = [str(c[0]) for c in df_out.columns]
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
    except Exception: pass
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
            if isinstance(canonical, dict) and canonical.get("url"): click_url = canonical.get("url")
            elif isinstance(canonical, str): click_url = canonical
            elif "link" in item: click_url = str(item["link"])
            if click_url.startswith("/"): click_url = f"https://finance.yahoo.com{click_url}"
            pub_time = content.get("pubDate") or item.get("providerPublishTime", None)
            if isinstance(pub_time, (int, float)):
                dt_str = datetime.fromtimestamp(pub_time, tz=ZoneInfo("Asia/Tokyo")).strftime("%m/%d %H:%M")
            else: dt_str = "直近"
            t_lower = str(title).lower()
            impact, direction, reason = "ℹ️ 普通", "↔️ 中立", "全般的な市況ニュース"
            if any(k in t_lower for k in ["fed","powell","cpi","gdp","pmi","inflation","rate","payroll","jobs","fomc","yield"]):
                impact = "🔥 高（米金融政策）"; direction = "⚡ 急変動警戒"; reason = "米指標・FRB発言による変動"
            elif any(k in t_lower for k in ["boj","ueda","yen","japan"]):
                impact = "🔥 高（円相場）"; direction = "📉📈 円急変動注意"; reason = "日銀関連"
            parsed.append({"title":title,"publisher":publisher,"link":click_url,"time":dt_str,"impact":impact,"direction":direction,"reason":reason})
        return parsed
    except Exception: return []

def compute_targets(c_vals, h_vals, l_vals, tp_vals, sl_vals, n, lookahead):
    target = np.zeros(n, dtype=int)
    for i in range(n - lookahead):
        entry_p = c_vals[i]; tp_dist = tp_vals[i]; sl_dist = sl_vals[i]
        if np.isnan(tp_dist) or np.isnan(sl_dist) or tp_dist <= 0 or sl_dist <= 0: continue
        outcome = 0
        for j in range(1, lookahead + 1):
            idx = i + j
            if idx >= n: break
            curr_h = h_vals[idx]; curr_l = l_vals[idx]
            buy_tp_hit = (curr_h - entry_p) >= tp_dist
            buy_sl_hit = (entry_p - curr_l) >= sl_dist
            sell_tp_hit = (entry_p - curr_l) >= tp_dist
            sell_sl_hit = (curr_h - entry_p) >= sl_dist
            if (buy_tp_hit and buy_sl_hit) or (sell_tp_hit and sell_sl_hit) or (buy_tp_hit and sell_tp_hit):
                outcome = 0; break
            if buy_tp_hit and not buy_sl_hit: outcome = 1; break
            if sell_tp_hit and not sell_sl_hit: outcome = -1; break
            if buy_sl_hit or sell_sl_hit: outcome = 0; break
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
            if tz_before is not None and df.index.tz is None: df.index = df.index.tz_localize(tz_before)
        if len(df) < 50: return None
        c = clean_series(df["Close"]); h = clean_series(df["High"]); l = clean_series(df["Low"]); o = clean_series(df["Open"])
        hl = h - l; hc = (h - c.shift(1)).abs(); lc = (l - c.shift(1)).abs()
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
        df_feat = pd.DataFrame(new_cols, index=df.index).replace([np.inf, -np.inf], np.nan).ffill()
        df_feat["RSI"] = df_feat["RSI"].fillna(50); df_feat["ADX"] = df_feat["ADX"].fillna(25)
        df_feat["BB_PctB"] = df_feat["BB_PctB"].fillna(0.5); df_feat["Stoch_K"] = df_feat["Stoch_K"].fillna(50)
        df_feat["SMA_20"] = df_feat["SMA_20"].fillna(c); df_feat["EMA_200"] = df_feat["EMA_200"].fillna(c)
        df_feat["ATR"] = df_feat["ATR"].fillna(c * 0.005)
        df_feat["Upper_Band"] = df_feat["Upper_Band"].fillna(c); df_feat["Lower_Band"] = df_feat["Lower_Band"].fillna(c)
        df_feat = df_feat.fillna(0)
        df = pd.concat([df, df_feat], axis=1)
        tp_t = new_cols["ATR"] * 0.9; sl_t = new_cols["ATR"] * 0.6
        c_vals = c.values.astype(float); h_vals = h.values.astype(float); l_vals = l.values.astype(float)
        tp_vals = tp_t.values.astype(float); sl_vals = sl_t.values.astype(float)
        target_values = compute_targets(c_vals, h_vals, l_vals, tp_vals, sl_vals, len(df), LOOKAHEAD_BARS)
        target_series = pd.Series(target_values, index=df.index, dtype="float64")
        target_series.iloc[-LOOKAHEAD_BARS:] = np.nan
        df["Target"] = target_series
        feat_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
        return df.dropna(subset=feat_cols)
    except Exception: return None

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
        prob_dict = {"buy": round(prob_up_raw*100,1), "sell": round(prob_down_raw*100,1), "wait": round(prob_wait_raw*100,1)}
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
        if prob_up_raw > prob_down_raw and prob_up_raw > prob_wait_raw and prob_up_raw >= threshold:
            status, conf = "BUY (買い)", prob_up_raw*100
        elif prob_down_raw > prob_up_raw and prob_down_raw > prob_wait_raw and prob_down_raw >= threshold:
            status, conf = "SELL (売り)", prob_down_raw*100
        else:
            status, conf = "WAIT (様子見)", max(prob_up_raw, prob_down_raw, prob_wait_raw)*100
        if "BUY" in status and (htf_bearish or rsi_val > 72): status = f"WAIT (上位足逆行/RSI高 {status}見送り)"
        if "SELL" in status and (htf_bullish or rsi_val < 28): status = f"WAIT (上位足逆行/RSI低 {status}見送り)"
        m_type = "トレンド相場" if adx_val > 22.0 else "レンジ相場"
        return status, conf, win_rate, m_type, prob_dict, importances
    except Exception: return ("WAIT (エラー)", 0.0, 0.0, "エラー", {"buy": 0.0, "sell": 0.0, "wait": 100.0}, pd.Series(dtype=float))

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
st.sidebar.number_input("1取引あたりの許容リスク (%)", min_value=0.1, max_value=10.0, step=0.1, key="risk_percent")
st.sidebar.number_input("デフォルト リスクリワード比 (RR)", min_value=0.5, max_value=5.0, step=0.1, key="rr_ratio")
st.sidebar.markdown("---")
if HAS_AUTOREFRESH:
    st.sidebar.checkbox("60秒ごとに自動更新", key="auto_refresh")
    if st.session_state["auto_refresh"]: st_autorefresh(interval=60000, key="datarefresh")
if st.sidebar.button("🔄 最新データに更新"): st.cache_data.clear(); st.rerun()

with st.spinner("最新相場データ & ニュースを取得中..."):
    data = load_and_process_data(ticker, tf_config["period"], tf_config["interval"], tf_label)
    data_4h = load_and_process_data(ticker, "60d", "1h", "4時間足 (中期・リピート用)")
    data_htf = load_and_process_data(ticker, "2y", "1d", "日足")
    news_items = fetch_news_and_impact(ticker)
if data_4h is None and data is not None: data_4h = data
if data is None: st.error("データ取得に失敗しました。取引時間外かYahoo Finance遅延です。"); st.stop()

now_jst = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f'<div class="update-time">最終取得: <b>{now_jst} JST</b> | yfinance価格は松井証券価格と乖離する場合があります</div>', unsafe_allow_html=True)
latest_price = float(clean_series(data["Close"]).iloc[-1])
latest_atr = float(clean_series(data["ATR"]).iloc[-1]) if "ATR" in data.columns else (latest_price * 0.005)
status, conf, win_rate, m_type, prob_dict, feature_importances = analyze_signal_with_backtest(data, data_htf)

key_lower = f"in_lower_{ticker}"; key_upper = f"in_upper_{ticker}"
if data_4h is not None and not data_4h.empty:
    swing_high_4h = float(clean_series(data_4h["High"]).iloc[-100:].max())
    swing_low_4h = float(clean_series(data_4h["Low"]).iloc[-100:].min())
    atr_4h = float(clean_series(data_4h["ATR"]).iloc[-1]) if "ATR" in data_4h.columns else latest_price * 0.005
else:
    swing_high_4h = latest_price * 1.02
    swing_low_4h = latest_price * 0.98
    atr_4h = latest_price * 0.005

if ticker not in st.session_state["ranges"]:
    st.session_state["ranges"][ticker] = {"lower": float(swing_low_4h), "upper": float(swing_high_4h)}
if key_lower not in st.session_state:
    st.session_state[key_lower] = float(st.session_state["ranges"][ticker]["lower"])
if key_upper not in st.session_state:
    st.session_state[key_upper] = float(st.session_state["ranges"][ticker]["upper"])

with st.expander("⚙️ リピート自動売買のレンジ調整", expanded=False):
    c1, c2 = st.columns(2)
    new_lower = c1.number_input("レンジ下限", value=float(st.session_state[key_lower]), step=0.1 if is_jpy else 0.001, format=price_fmt)
    new_upper = c2.number_input("レンジ上限", value=float(st.session_state[key_upper]), step=0.1 if is_jpy else 0.001, format=price_fmt)
    if st.button("✨ 4時間足高値・安値から自動計算"):
        new_lower = swing_low_4h
        new_upper = swing_high_4h
    st.session_state[key_lower] = float(new_lower)
    st.session_state[key_upper] = float(new_upper)
    st.session_state["ranges"][ticker] = {"lower": float(new_lower), "upper": float(new_upper)}
    st.success(f"保存: {new_lower:.3f} - {new_upper:.3f}")

user_lower = min(float(st.session_state[key_lower]), float(st.session_state[key_upper]))
user_upper = max(float(st.session_state[key_lower]), float(st.session_state[key_upper]))
if user_lower == user_upper: user_upper += pip_unit * 100.0
user_half = (user_upper + user_lower) / 2.0
stop_buffer_pips = max(10.0, round((atr_4h / pip_unit) * 1.5, 1))
stop_buffer_val = stop_buffer_pips * pip_unit
buy_stop_loss = user_lower - stop_buffer_val
sell_stop_loss = user_upper + stop_buffer_val

m1, m2, m3, m4 = st.columns(4)
m1.metric("現在レート", price_fmt % latest_price)
badge_html = f'<div class="badge-buy">{status}</div>' if "BUY" in status and "WAIT" not in status else f'<div class="badge-sell">{status}</div>' if "SELL" in status and "WAIT" not in status else f'<div class="badge-wait">{status}</div>'
m2.markdown("**AI 推奨アクション**"); m2.markdown(badge_html, unsafe_allow_html=True)
m3.metric("直近バックテスト勝率", f"{win_rate:.1f}%", f"確信度: {conf:.1f}%")
m4.metric("相場環境", m_type, f"ATR: {latest_atr/pip_unit:.1f} pips")
st.markdown("---")

tab_chart, tab_single, tab_news, tab_repeat, tab_ai = st.tabs(["📈 メインチャート","⚡ 単発トレード","📰 ニュースAI予測","📋 松井証券 リピート設定","🤖 AI詳細"])
with tab_chart:
    bars_count = st.slider("表示本数", 30, 300, 90, 10)
    df_chart = safe_to_tokyo_tz(data.tail(bars_count)).ffill().loc[lambda d: ~d.index.duplicated(keep='last')]
    fmt_str = '%m/%d %H:%M' if "分" in tf_label or "時間" in tf_label else '%Y/%m/%d'
    x_labels = df_chart.index.strftime(fmt_str) if isinstance(df_chart.index, pd.DatetimeIndex) else df_chart.index.astype(str)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=x_labels, open=clean_series(df_chart["Open"]), high=clean_series(df_chart["High"]), low=clean_series(df_chart["Low"]), close=clean_series(df_chart["Close"]), increasing_line_color='#22c55e', decreasing_line_color='#ef4444', name="価格"))
    if "SMA_20" in df_chart.columns: fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["SMA_20"]), line=dict(color="#f59e0b", width=1.5), name="SMA20"))
    if "EMA_200" in df_chart.columns: fig.add_trace(go.Scatter(x=x_labels, y=clean_series(df_chart["EMA_200"]), line=dict(color="#38bdf8", width=2.0), name="EMA200"))
    fig.add_hline(y=user_upper, line_dash="dash", line_color="#f87171", annotation_text="上限")
    fig.add_hline(y=user_lower, line_dash="dash", line_color="#4ade80", annotation_text="下限")
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark", margin=dict(l=10, r=50, t=20, b=10))
    fig.update_xaxes(type='category', nticks=10, tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

with tab_single:
    st.markdown("##### ⚡ 単発トレード AIアシスト")
    usd_rate = latest_price if ticker == "USDJPY=X" else get_usdjpy_rate()
    pip_val_per_wan = 100.0 if is_jpy else (0.0001 * usd_rate * 10000.0)
    account_bal = float(st.session_state["account_balance"]); risk_pct = float(st.session_state["risk_percent"])
    max_risk_yen = account_bal * (risk_pct / 100.0)
    ai_direction = "BUY (買い)" if prob_dict["buy"] >= prob_dict["sell"] else "SELL (売り)"
    ai_sl_pips = round((latest_atr / pip_unit) * 1.5, 1); ai_tp_pips = round(ai_sl_pips * float(st.session_state["rr_ratio"]), 1)
    ai_sl_price = latest_price - (ai_sl_pips * pip_unit) if "BUY" in ai_direction else latest_price + (ai_sl_pips * pip_unit)
    ai_tp_price = latest_price + (ai_tp_pips * pip_unit) if "BUY" in ai_direction else latest_price - (ai_tp_pips * pip_unit)
    ai_loss_per_wan = (ai_sl_pips + spread_pips) * pip_val_per_wan
    ai_rec_wan = max(0.01, round((max_risk_yen / ai_loss_per_wan) if ai_loss_per_wan > 0 else 0.01, 2))
    st.markdown(f"""<div class="param-box"><b>{status} (確信度 {conf:.1f}% / 勝率 {win_rate:.1f}%)</b><br>方向: <code>{ai_direction}</code> | エントリー: <code>{price_fmt % latest_price}</code><br>SL: <code>{price_fmt % ai_sl_price}</code> (-{ai_sl_pips}pips) | TP: <code>{price_fmt % ai_tp_price}</code> (+{ai_tp_pips}pips)<br>推奨: <code>{ai_rec_wan:.2f}万通貨</code></div>""", unsafe_allow_html=True)

with tab_news:
    st.markdown("##### 📰 ニュース & 影響予測")
    if not news_items: st.info("ニュース取得できませんでした")
    else:
        for news in news_items:
            st.markdown(f"""<div class="news-card"><b>{html.escape(news['publisher'])} ({html.escape(news['time'])}) - {html.escape(news['impact'])}</b><br><a href="{html.escape(news['link'])}" target="_blank" style="color:#f8fafc">{html.escape(news['title'])} 🔗</a><br><span style="color:#facc15">🎯 {html.escape(news['direction'])}</span> - {html.escape(news['reason'])}</div>""", unsafe_allow_html=True)

with tab_repeat:
    trap_width_pips = st.number_input("注文幅 / 利確幅 (pips)", 5, 500, value=max(15, int(round((atr_4h / pip_unit)))), step=5)
    st.code(f"""通貨ペア: {selected_label}
買い: {price_fmt % user_lower} - {price_fmt % user_half} (SL {price_fmt % buy_stop_loss})
売り: {price_fmt % user_half} - {price_fmt % user_upper} (SL {price_fmt % sell_stop_loss})
幅: {trap_width_pips}pips
数量: {float(st.session_state['quantity_wan']):.2f}万通貨""")

with tab_ai:
    st.write(f"BUY: {prob_dict['buy']}%"); st.progress(prob_dict['buy']/100.0)
    st.write(f"SELL: {prob_dict['sell']}%"); st.progress(prob_dict['sell']/100.0)
    st.write(f"WAIT: {prob_dict['wait']}%"); st.progress(prob_dict['wait']/100.0)
    if feature_importances is not None and not feature_importances.empty:
        top10 = feature_importances.tail(10)
        fig_imp = go.Figure(go.Bar(x=top10.values, y=top10.index, orientation='h', marker_color='#38bdf8'))
        fig_imp.update_layout(height=340, template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)
