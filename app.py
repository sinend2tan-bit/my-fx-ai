import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ==========================================
# 0. 画面基本設定 & デザインCSS
# ==========================================
st.set_page_config(
    page_title="松井証券 リピートFX アナライザー & 資金管理 Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1280px; }
    label[data-testid="stWidgetLabel"] p { font-size: 1.05rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 1.0rem !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 800 !important; }
    
    .param-box {
        background-color: #1e293b !important;
        border-left: 6px solid #3b82f6;
        padding: 20px;
        border-radius: 8px;
        font-family: monospace, sans-serif;
        font-size: 1.05rem;
        line-height: 1.9;
        color: #f8fafc !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .param-box b.label-title { color: #94a3b8 !important; font-family: sans-serif; }
    .param-box code {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 3px 8px !important;
        background-color: #334155 !important;
        color: #38bdf8 !important;
        border: 1px solid #475569;
        border-radius: 4px;
    }
    .param-box-buy { border-left-color: #22c55e !important; }
    .param-box-sell { border-left-color: #ef4444 !important; }
    .param-box-half { border-left-color: #a855f7 !important; }

    .risk-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 設定・定数
SETTINGS_FILE = "user_settings.json"
DEFAULT_SETTINGS = {
    "account_balance": 500000,
    "quantity_wan": 0.10,
    "selected_pair_label": "米ドル / 円 (USD/JPY)",
    "auto_refresh": False,
    "refresh_interval": 180,
    "grid_density": "標準 (ATR1.0倍)",
}

PAIRS = {
    "米ドル / 円 (USD/JPY)": "USDJPY=X",
    "豪ドル / 円 (AUD/JPY)": "AUDJPY=X",
    "ユーロ / 円 (EUR/JPY)": "EURJPY=X",
    "ポンド / 円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ / 米ドル (EUR/USD)": "EURUSD=X",
    "豪ドル / 米ドル (AUD/USD)": "AUDUSD=X",
    "ニュージーランドドル / 円 (NZD/JPY)": "NZDJPY=X",
}

PAIR_CONFIG = {
    "USDJPY=X": {"min_width_pips": 10, "default_stop_buffer_pips": 100},
    "AUDJPY=X": {"min_width_pips": 10, "default_stop_buffer_pips": 80},
    "EURJPY=X": {"min_width_pips": 15, "default_stop_buffer_pips": 120},
    "GBPJPY=X": {"min_width_pips": 15, "default_stop_buffer_pips": 150},
    "EURUSD=X": {"min_width_pips": 10, "default_stop_buffer_pips": 80},
    "AUDUSD=X": {"min_width_pips": 10, "default_stop_buffer_pips": 80},
    "NZDJPY=X": {"min_width_pips": 10, "default_stop_buffer_pips": 80},
}

# ==========================================
# 1. 補助関数 & データ処理
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
        "account_balance": st.session_state.get("account_balance", DEFAULT_SETTINGS["account_balance"]),
        "quantity_wan": st.session_state.get("quantity_wan", DEFAULT_SETTINGS["quantity_wan"]),
        "selected_pair_label": st.session_state.get("selected_pair_label", DEFAULT_SETTINGS["selected_pair_label"]),
        "auto_refresh": st.session_state.get("auto_refresh", DEFAULT_SETTINGS["auto_refresh"]),
        "refresh_interval": st.session_state.get("refresh_interval", DEFAULT_SETTINGS["refresh_interval"]),
        "grid_density": st.session_state.get("grid_density", DEFAULT_SETTINGS["grid_density"]),
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

if "initialized" not in st.session_state:
    saved = load_user_settings()
    for k, v in saved.items():
        st.session_state[k] = v
    if st.session_state.get("selected_pair_label") not in PAIRS:
        st.session_state["selected_pair_label"] = DEFAULT_SETTINGS["selected_pair_label"]
    st.session_state["initialized"] = True

def clean_series(s):
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s

@st.cache_data(ttl=120, show_spinner=False)
def fetch_market_data(symbol):
    """日足および4時間足データを取得・加工"""
    try:
        df_daily = yf.download(symbol, period="1y", interval="1d", progress=False)
        df_4h_raw = yf.download(symbol, period="6mo", interval="1h", progress=False)

        if df_daily.empty:
            return None, None

        for df in [df_daily, df_4h_raw]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        # 4時間足へリサンプル
        df_4h = df_4h_raw.resample("4h", closed="left", label="left").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()

        return df_daily, df_4h
    except Exception:
        return None, None

def calculate_pivot_and_levels(df_daily, df_4h):
    """ピボットポイント & Swing サポレジ算出"""
    c_d = clean_series(df_daily["Close"])
    h_d = clean_series(df_daily["High"])
    l_d = clean_series(df_daily["Low"])

    # 日足ピボット（直近日足確定値）
    prev_close = c_d.iloc[-2]
    prev_high = h_d.iloc[-2]
    prev_low = l_d.iloc[-2]

    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = (2 * pivot) - prev_low
    s1 = (2 * pivot) - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)

    # 直近スイングハイ・ロー (過去60日 & 4時間足100本)
    swing_high_d = float(h_d.iloc[-60:].max())
    swing_low_d = float(l_d.iloc[-60:].min())

    h_4h = clean_series(df_4h["High"])
    l_4h = clean_series(df_4h["Low"])
    swing_high_4h = float(h_4h.iloc[-100:].max())
    swing_low_4h = float(l_4h.iloc[-100:].min())

    # 14日ATR
    tr = pd.concat([
        h_d - l_d,
        (h_d - c_d.shift(1)).abs(),
        (l_d - c_d.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14).mean().iloc[-1])

    return {
        "pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2,
        "swing_high_d": swing_high_d, "swing_low_d": swing_low_d,
        "swing_high_4h": swing_high_4h, "swing_low_4h": swing_low_4h,
        "atr_14": atr_14, "latest_price": float(c_d.iloc[-1])
    }

# ==========================================
# 2. メイン画面レイアウト
# ==========================================
st.title("松井証券 リピートFX アナライザー Pro")
st.caption("AI単発シグナルを排除し、日足・4時間足構造分析と自動売買パラメータ計算に特化した専門ツール")

# サイドバー設定
st.sidebar.header("⚙️ 運用基本設定")

selected_label = st.sidebar.selectbox(
    "分析通貨ペア", list(PAIRS.keys()), key="selected_pair_label", on_change=save_user_settings
)
symbol = PAIRS[selected_label]
is_jpy = "JPY" in symbol
pip_unit = 0.01 if is_jpy else 0.0001

# Streamlitのnumber_input形式（Printfフォーマット）に変更
price_fmt = "%.3f" if is_jpy else "%.5f"

st.sidebar.markdown("---")
st.sidebar.subheader("💰 松井証券 資金・注文設定")
account_balance = st.sidebar.number_input(
    "口座資金 (円)", min_value=10000, max_value=100000000, step=50000, key="account_balance", on_change=save_user_settings
)
quantity_wan = st.sidebar.number_input(
    "1注文あたりの数量 (万通貨)", min_value=0.0001, max_value=10.0, step=0.01, format="%.4f", key="quantity_wan", on_change=save_user_settings
)
order_units = int(round(quantity_wan * 10000))

grid_density = st.sidebar.select_slider(
    "トラップ密度 (注文幅)",
    options=["広め (ATR1.5倍)", "標準 (ATR1.0倍)", "狭め (ATR0.7倍)"],
    key="grid_density",
    on_change=save_user_settings
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 データ再読み込み", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("自動更新を有効化", key="auto_refresh", on_change=save_user_settings)
refresh_interval = st.sidebar.selectbox("更新間隔", [60, 180, 300], format_func=lambda x: f"{x//60}分ごと", key="refresh_interval", on_change=save_user_settings)

# データ取得
with st.spinner("市場構造データ取得中..."):
    df_daily, df_4h = fetch_market_data(symbol)

if df_daily is None or df_4h is None:
    st.error("市場データの取得に失敗しました。しばらく時間を置いてからお試しください。")
    st.stop()

analysis = calculate_pivot_and_levels(df_daily, df_4h)
latest_price = analysis["latest_price"]
atr_14 = analysis["atr_14"]
atr_pips = atr_14 / pip_unit

# トラップ幅の倍率設定
density_mult = 1.5 if "広め" in grid_density else (0.7 if "狭め" in grid_density else 1.0)
recommended_width_pips = max(
    PAIR_CONFIG.get(symbol, {}).get("min_width_pips", 10),
    int(round(atr_pips * density_mult))
)
recommended_width_val = recommended_width_pips * pip_unit

# ==========================================
# 3. リピート帯・サポレジ解析結果
# ==========================================
# 予想レンジの初期値計算
range_lower = min(analysis["swing_low_4h"], analysis["s1"])
range_upper = max(analysis["swing_high_4h"], analysis["r1"])

# レンジの微調整UI
st.markdown("### 🎯 予想レンジ帯 確定・調整")
r_col1, r_col2, r_col3 = st.columns([1.5, 1.5, 1])

with r_col1:
    user_lower = st.number_input(
        "レンジ下限 (サポート)",
        value=float(round(range_lower, 3 if is_jpy else 5)),
        step=0.1 if is_jpy else 0.001,
        format=price_fmt
    )
with r_col2:
    user_upper = st.number_input(
        "レンジ上限 (レジスタンス)",
        value=float(round(range_upper, 3 if is_jpy else 5)),
        step=0.1 if is_jpy else 0.001,
        format=price_fmt
    )
with r_col3:
    st.metric("現在レート", price_fmt % latest_price)
    st.caption(f"日足14日ATR: **{atr_pips:.1f} pips**")

# レンジと本数の計算
range_pips = abs(user_upper - user_lower) / pip_unit
grid_count = int(range_pips // recommended_width_pips) + 1

st.markdown("---")

# Tab構成
tab_matsui, tab_chart, tab_risk = st.tabs([
    "📋 松井証券 注文設定出力", "📈 相場構造 & チャート", "🛡️ 資金管理 & ドローダウンシミュレーション"
])

# ==========================================
# Tab 1: 松井証券 注文設定出力
# ==========================================
with tab_matsui:
    st.subheader("📋 松井証券FX 自動売買 転記用パラメータ")
    st.caption("以下の設定値を松井証券の自動売買（リピート）注文画面にそのまま入力してください。")

    stop_buffer_pips = PAIR_CONFIG.get(symbol, {}).get("default_stop_buffer_pips", 100)
    buy_stop_price = round(user_lower - (stop_buffer_pips * pip_unit), 3 if is_jpy else 5)
    sell_stop_price = round(user_upper + (stop_buffer_pips * pip_unit), 3 if is_jpy else 5)

    mode = st.radio("注文タイプを選択", ["買いリピート", "売りリピート", "ハーフ＆ハーフ"], horizontal=True)

    if mode == "買いリピート":
        st.markdown(f"""
        <div class="param-box param-box-buy">
            <b class="label-title">【松井証券 注文設定画面用】</b><br>
            ・<b>通貨ペア</b>　　: <code>{selected_label.split(' ')[0]}</code><br>
            ・<b>売買区分</b>　　: <code>買</code><br>
            ・<b>レンジ上限</b>　: <code>{price_fmt % user_upper}</code><br>
            ・<b>レンジ下限</b>　: <code>{price_fmt % user_lower}</code><br>
            ・<b>注文本数</b>　　: <code>{grid_count} 本</code><br>
            ・<b>注文数量</b>　　: <code>{quantity_wan} 万通貨</code> ({order_units:,} 通貨)<br>
            ・<b>注文値幅</b>　　: <code>{recommended_width_pips} pips</code><br>
            ・<b>益出し幅</b>　　: <code>{recommended_width_pips} pips</code><br>
            ・<b>運用停止(SL)</b>: <code>{price_fmt % buy_stop_price}</code> (-{stop_buffer_pips}pips下落時)
        </div>
        """, unsafe_allow_html=True)

    elif mode == "売りリピート":
        st.markdown(f"""
        <div class="param-box param-box-sell">
            <b class="label-title">【松井証券 注文設定画面用】</b><br>
            ・<b>通貨ペア</b>　　: <code>{selected_label.split(' ')[0]}</code><br>
            ・<b>売買区分</b>　　: <code>売</code><br>
            ・<b>レンジ上限</b>　: <code>{price_fmt % user_upper}</code><br>
            ・<b>レンジ下限</b>　: <code>{price_fmt % user_lower}</code><br>
            ・<b>注文本数</b>　　: <code>{grid_count} 本</code><br>
            ・<b>注文数量</b>　　: <code>{quantity_wan} 万通貨</code> ({order_units:,} 通貨)<br>
            ・<b>注文値幅</b>　　: <code>{recommended_width_pips} pips</code><br>
            ・<b>益出し幅</b>　　: <code>{recommended_width_pips} pips</code><br>
            ・<b>運用停止(SL)</b>: <code>{price_fmt % sell_stop_price}</code> (+{stop_buffer_pips}pips上昇時)
        </div>
        """, unsafe_allow_html=True)

    else: # ハーフ＆ハーフ
        half_price = round((user_upper + user_lower) / 2.0, 3 if is_jpy else 5)
        half_buy_grids = max(1, int((half_price - user_lower) / (recommended_width_pips * pip_unit)))
        half_sell_grids = max(1, int((user_upper - half_price) / (recommended_width_pips * pip_unit)))

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown(f"""
            <div class="param-box param-box-buy">
                <b class="label-title">【買い設定 (下半分)】</b><br>
                ・<b>売買区分</b>: <code>買</code><br>
                ・<b>レンジ上限</b>: <code>{price_fmt % half_price}</code><br>
                ・<b>レンジ下限</b>: <code>{price_fmt % user_lower}</code><br>
                ・<b>注文本数</b>: <code>{half_buy_grids} 本</code><br>
                ・<b>注文/益出幅</b>: <code>{recommended_width_pips} pips</code><br>
                ・<b>運用停止(SL)</b>: <code>{price_fmt % buy_stop_price}</code>
            </div>
            """, unsafe_allow_html=True)
        with col_h2:
            st.markdown(f"""
            <div class="param-box param-box-sell">
                <b class="label-title">【売り設定 (上半分)】</b><br>
                ・<b>売買区分</b>: <code>売</code><br>
                ・<b>レンジ上限</b>: <code>{price_fmt % user_upper}</code><br>
                ・<b>レンジ下限</b>: <code>{price_fmt % half_price}</code><br>
                ・<b>注文本数</b>: <code>{half_sell_grids} 本</code><br>
                ・<b>注文/益出幅</b>: <code>{recommended_width_pips} pips</code><br>
                ・<b>運用停止(SL)</b>: <code>{price_fmt % sell_stop_price}</code>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# Tab 2: 相場構造 & チャート
# ==========================================
with tab_chart:
    st.subheader("📈 レンジ構造 ＆ サポレジ分析チャート")

    df_plot = df_4h.tail(120).copy()
    if df_plot.index.tz is None:
        df_plot.index = df_plot.index.tz_localize("UTC")
    df_plot.index = df_plot.index.tz_convert("Asia/Tokyo")
    chart_x = df_plot.index.strftime("%m/%d %H:%M")

    fig = make_subplots(rows=1, cols=1)

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=chart_x, open=clean_series(df_plot["Open"]), high=clean_series(df_plot["High"]),
        low=clean_series(df_plot["Low"]), close=clean_series(df_plot["Close"]), name="4時間足"
    ))

    # 設定レンジの描画
    fig.add_hline(y=user_upper, line_dash="solid", line_color="#ef4444", line_width=2, annotation_text="レンジ上限")
    fig.add_hline(y=user_lower, line_dash="solid", line_color="#22c55e", line_width=2, annotation_text="レンジ下限")

    # ピボットラインの描画
    fig.add_hline(y=analysis["pivot"], line_dash="dash", line_color="#eab308", line_width=1, annotation_text="日足Pivot")

    fig.update_layout(
        height=500, xaxis_rangeslider_visible=False, template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ピボット・サポレジ一覧表
    st.markdown("##### 📌 主要サポレジ・ピボット水準")
    lvl_col1, lvl_col2, lvl_col3 = st.columns(3)
    lvl_col1.metric("レジスタンス2 (R2)", price_fmt % analysis["r2"])
    lvl_col1.metric("レジスタンス1 (R1)", price_fmt % analysis["r1"])
    lvl_col2.metric("デイリーピボット (P)", price_fmt % analysis["pivot"])
    lvl_col3.metric("サポート1 (S1)", price_fmt % analysis["s1"])
    lvl_col3.metric("サポート2 (S2)", price_fmt % analysis["s2"])

# ==========================================
# Tab 3: 資金管理 & ドローダウンシミュレーション
# ==========================================
with tab_risk:
    st.subheader("🛡️ 想定最大ドローダウン・リスク評価")
    st.caption("設定したレンジの下限（またはロスカットライン）まで価格が完全逆行・全トラップ保有した際の最悪含み損額を事前計算します。")

    worst_price = buy_stop_price
    total_unrealized_loss = 0.0

    for i in range(grid_count):
        pos_price = user_upper - (i * recommended_width_val)
        if pos_price > worst_price:
            pips_drop = (pos_price - worst_price) / pip_unit
            usd_rate = latest_price if is_jpy else 155.0
            pip_value = 100 * (order_units / 10000.0) if is_jpy else (order_units * 0.0001 * usd_rate)
            total_unrealized_loss += pips_drop * pip_value

    single_margin = (latest_price * order_units) / 25.0 if is_jpy else (latest_price * 155.0 * order_units) / 25.0
    total_required_margin = single_margin * grid_count

    max_loss_ratio = (total_unrealized_loss / account_balance) * 100 if account_balance > 0 else 0
    margin_usage_ratio = (total_required_margin / account_balance) * 100 if account_balance > 0 else 0

    col_rk1, col_rk2, col_rk3 = st.columns(3)
    col_rk1.metric("想定最大含み損 (最悪時)", f"約 {int(total_unrealized_loss):,} 円", f"-{max_loss_ratio:.1f}%")
    col_rk2.metric("最大時 必要証拠金合計", f"約 {int(total_required_margin):,} 円", f"使用率 {margin_usage_ratio:.1f}%")
    col_rk3.metric("残存余力 (資金 - 最大損失)", f"約 {int(account_balance - total_unrealized_loss):,} 円")

    if max_loss_ratio > 70:
        st.error("🚨 **【ハイリスク警戒】** 最悪時の想定含み損が口座資金の70%を超えています！1注文の数量を減らすか、注文本数を少なく設定してください。")
    elif max_loss_ratio > 40:
        st.warning("⚠️ **【注意】** 最大含み損が資金の40%以上です。急激なトレンド発生時の追加入金余力を考慮してください。")
    else:
        st.success("🟢 **【安全圏】** 資金に対して健全なリスク範囲内で設計されています。")

# 自動更新
if auto_refresh and HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_interval * 1000, key="matsui_repeat_refresh")
