# Pro FX Analyzer - v3 Stable (requirementsエラー対策版)
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

st.set_page_config(page_title="Pro FX Analyzer", page_icon="📈", layout="wide")

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# --- 共通関数 ---
def clean_series(s):
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0] if s.shape[1] > 0 else pd.Series(dtype=float)
    return s if isinstance(s, pd.Series) else pd.Series(s)

def flatten_yf_df(df):
    if df is None or df.empty: return df
    df_out = df.copy()
    if isinstance(df_out.columns, pd.MultiIndex):
        df_out.columns = df_out.columns.get_level_values(0)
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    df_out.columns = [str(c).capitalize() for c in df_out.columns]
    return df_out.loc[~df_out.index.duplicated(keep='last')]

@st.cache_data(ttl=300)
def get_rate(symbol="USDJPY=X"):
    try:
        df = yf.download(symbol, period="5d", progress=False, timeout=15)
        df = flatten_yf_df(df)
        if df is not None and not df.empty and "Close" in df.columns:
            v = float(clean_series(df["Close"]).dropna().iloc[-1])
            if v > 0: return v
    except Exception as e:
        st.warning(f"レート取得失敗 {symbol}: {e}")
    return 155.0

@st.cache_data(ttl=300)
def load_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, timeout=15)
        df = flatten_yf_df(df)
        if df is None or df.empty or len(df) < 50: return None
        c = clean_series(df["Close"]); h = clean_series(df["High"]); l = clean_series(df["Low"])
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        df["SMA_20"] = c.rolling(20).mean()
        df["EMA_200"] = c.ewm(span=200, adjust=False).mean()
        df["ATR"] = tr.rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (c.diff().where(lambda x: x>0, 0).ewm(alpha=1/14, adjust=False).mean() / ( -c.diff().where(lambda x: x<0, 0).ewm(alpha=1/14, adjust=False).mean() + 1e-10))))
        df["Target"] = np.random.choice([-1,0,1], len(df)) # 簡易版: 本番は前回のcompute_targetsを使用
        return df.dropna()
    except Exception as e:
        st.error(f"データ処理エラー: {e}")
        return None

# --- UI ---
st.title("Pro FX Analyzer - 起動確認版")
PAIRS = {"USD/JPY": "USDJPY=X", "GBP/JPY": "GBPJPY=X", "EUR/JPY": "EURJPY=X", "EUR/USD": "EURUSD=X"}
pair_label = st.selectbox("通貨ペア", list(PAIRS.keys()))
ticker = PAIRS[pair_label]

with st.spinner("取得中..."):
    df = load_data(ticker, "60d", "1h")

if df is None:
    st.error("データが取れませんでした。時間をおいて Reboot してください。Logsを確認してください。")
    st.stop()

st.success(f"起動成功！ {pair_label} {len(df)}本取得")
st.line_chart(df["Close"].tail(100))

st.write("ここまで表示されれば requirements.txt の修正は成功です。")
st.write("この確認が取れたら、前回のフル機能版を戻してください。")
