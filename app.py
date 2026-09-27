import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import requests
import json
from datetime import datetime, timedelta

# ==========================================
# 1. ページ設定・定数定義
# ==========================================
st.set_page_config(
    page_title="FX AI Trading Assistant v6.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 実用性を最優先した5つの時間軸構成
TIMEFRAMES = {
    "5分足 (超短期エントリー)": {"period": "7d", "interval": "5m"},
    "15分足 (デイトレエントリー)": {"period": "1mo", "interval": "15m"},
    "1時間足 (デイトレメイン)": {"period": "6mo", "interval": "1h"},
    "4時間足 (中期トレンド)": {"period": "2y", "interval": "1h"},  # 1hからリサンプリング生成
    "日足 (スイング・環境認識)": {"period": "2y", "interval": "1d"},
}

PAIR_MAP = {
    "USD/JPY": "JPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/USD": "EURUSD=X"
}

# ==========================================
# 2. データ取得・前処理関数
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """yfinanceからデータ取得・タイムゾーン調整および4時間足リサンプリング処理"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()

        # 日本時間に変換
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Tokyo')
        else:
            df.index = df.index.tz_convert('Asia/Tokyo')

        # 4時間足のリサンプリング処理
        if interval == "1h" and period == "2y":
            df = df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        return df
    except Exception as e:
        st.error(f"データ取得エラー ({symbol}): {e}")
        return pd.DataFrame()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """各種テクニカル指標の計算"""
    if len(df) < 50:
        return df
    
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']

    # 移動平均線
    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # ボリンジャーバンド (20, 2σ)
    std20 = close.rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + (std20 * 2)
    df['BB_Lower'] = df['SMA_20'] - (std20 * 2)

    # ATR (14) - SL/TP計算用
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    return df.dropna()

# ==========================================
# 3. マルチタイムフレーム (MTF) 分析関数
# ==========================================
def get_mtf_trends(symbol: str) -> dict:
    """すべての時間軸のトレンド方向を一括判定"""
    trends = {}
    for name, params in TIMEFRAMES.items():
        df = fetch_stock_data(symbol, params["period"], params["interval"])
        if len(df) >= 50:
            df = add_technical_indicators(df)
            last = df.iloc[-1]
            # EMA20 と EMA50 の関係および Close でトレンド判定
            if last['Close'] > last['EMA_20'] and last['EMA_20'] > last['EMA_50']:
                trends[name] = "上昇 📈"
            elif last['Close'] < last['EMA_20'] and last['EMA_20'] < last['EMA_50']:
                trends[name] = "下降 📉"
            else:
                trends[name] = "レンジ ➡️"
        else:
            trends[name] = "取得不可 ⚠️"
    return trends

# ==========================================
# 4. AI機械学習モデル・分析ロジック
# ==========================================
def train_and_predict(df: pd.DataFrame, n_estimators: int = 100):
    """RandomForestによる売買シグナルと確率の算出"""
    if len(df) < 100:
        return "HOLD", 0.5, 0.0

    df_ml = df.copy()
    # 目的変数: 3本後の終値が上がっているか (1: BUY, 0: SELL)
    df_ml['Target'] = np.where(df_ml['Close'].shift(-3) > df_ml['Close'], 1, 0)
    
    features = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Hist', 'ATR_14']
    X = df_ml[features].iloc[:-3]
    y = df_ml['Target'].iloc[:-3]

    if len(X) < 50:
        return "HOLD", 0.5, 0.0

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)

    # 最新データでの予測
    latest_x = df_ml[features].iloc[[-1]]
    prob = model.predict_proba(latest_x)[0]  # [Prob_SELL, Prob_BUY]
    buy_prob = prob[1]

    # モデルの直近精度評価 (Walk-Forward簡易検証)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    # シグナル判定しきい値
    if buy_prob >= 0.60:
        signal = "BUY"
    elif buy_prob <= 0.40:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, buy_prob, acc

def run_backtest(df: pd.DataFrame):
    """簡略バックテストの実行"""
    if len(df) < 150:
        return pd.DataFrame(), 0, 0

    df_bt = df.copy()
    features = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Hist', 'ATR_14']
    df_bt['Target'] = np.where(df_bt['Close'].shift(-3) > df_bt['Close'], 1, 0)

    signals = []
    # 過去50本分のローソク足でバックテスト
    step = 5
    test_range = range(100, len(df_bt) - 3, step)

    for i in test_range:
        X_train = df_bt[features].iloc[:i]
        y_train = df_bt['Target'].iloc[:i]
        
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)
        
        current_x = df_bt[features].iloc[[i]]
        p_buy = model.predict_proba(current_x)[0][1]
        
        if p_buy >= 0.60:
            sig = 1  # BUY
        elif p_buy <= 0.40:
            sig = -1 # SELL
        else:
            sig = 0  # HOLD
            
        signals.append((df_bt.index[i], sig, df_bt['Close'].iloc[i], df_bt['Close'].iloc[i+3]))

    bt_df = pd.DataFrame(signals, columns=['Time', 'Signal', 'EntryPrice', 'ExitPrice'])
    bt_df = bt_df[bt_df['Signal'] != 0].copy()

    if bt_df.empty:
        return bt_df, 0, 0.0

    # 損益計算 (BUY: Exit - Entry, SELL: Entry - Exit)
    bt_df['Pips'] = np.where(bt_df['Signal'] == 1, 
                             bt_df['ExitPrice'] - bt_df['EntryPrice'], 
                             bt_df['EntryPrice'] - bt_df['ExitPrice'])
    
    total_trades = len(bt_df)
    win_trades = len(bt_df[bt_df['Pips'] > 0])
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0

    return bt_df, total_trades, win_rate
# ==========================================
# 5. 外部連携・便利関数
# ==========================================
def send_discord_notification(webhook_url: str, message: str) -> bool:
    """Discord Webhookへのメッセージ送信"""
    if not webhook_url:
        return False
    try:
        data = {"content": message}
        response = requests.post(webhook_url, data=json.dumps(data), headers={"Content-Type": "application/json"})
        return response.status_code == 204
    except Exception as e:
        st.error(f"Discord送信エラー: {e}")
        return False

# ==========================================
# 6. Plotly チャート描画関数（改善版）
# ==========================================
def create_chart(df: pd.DataFrame, symbol: str, timeframe_name: str) -> go.Figure:
    """インタラクティブチャート作成 (日足時刻表示の自動最適化付き)"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{symbol} - {timeframe_name}", "RSI & MACD")
    )

    # 日足判定によるX軸フォーマットの切替
    is_daily = "日足" in timeframe_name
    if is_daily:
        chart_x = df.index.strftime("%Y-%m-%d")
    else:
        chart_x = df.index.strftime("%m-%d %H:%M")

    # 1. ローソク足
    fig.add_trace(go.Candlestick(
        x=chart_x, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="OHLC"
    ), row=1, col=1)

    # 2. 移動平均線
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=chart_x, y=df['EMA_20'], line=dict(color='orange', width=1.5), name="EMA 20"), row=1, col=1)
    if 'EMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=chart_x, y=df['EMA_50'], line=dict(color='blue', width=1.5), name="EMA 50"), row=1, col=1)

    # 3. ボリンジャーバンド
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(x=chart_x, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_x, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name="BB Lower"), row=1, col=1)

    # 4. RSI (サブチャート)
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=chart_x, y=df['RSI_14'], line=dict(color='purple', width=1.5), name="RSI(14)"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )
    return fig

# ==========================================
# 7. メインUIレイアウト & 設定
# ==========================================
st.title("📈 FX AI Trading Assistant v6.0")

# --- サイドバー設定 ---
st.sidebar.header("⚙️ トレード基本設定")

selected_pair_name = st.sidebar.selectbox("通貨ペア選択", list(PAIR_MAP.keys()), index=0)
symbol = PAIR_MAP[selected_pair_name]

selected_tf_name = st.sidebar.selectbox("時間軸選択", list(TIMEFRAMES.keys()), index=2)
tf_params = TIMEFRAMES[selected_tf_name]

account_balance = st.sidebar.number_input("口座残高 (円)", min_value=10000, value=1000000, step=50000)
risk_per_trade_pct = st.sidebar.slider("1トレード許容リスク (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🚨 リスク・手動制限")
manual_hold_override = st.sidebar.checkbox("手動で静観 (HOLD) に固定", value=False, help="経済指標発表前など、一時的にAIのエントリー判定を停止します。")

st.sidebar.markdown("---")
discord_webhook_url = st.sidebar.text_input("Discord Webhook URL", type="password")

# --- データ読み込み ---
df = fetch_stock_data(symbol, tf_params["period"], tf_params["interval"])

if df.empty or len(df) < 50:
    st.error("データの取得に失敗したか、十分なデータがありません。時間を置いて再試行してください。")
    st.stop()

df = add_technical_indicators(df)

# ==========================================
# 8. マルチタイムフレーム (MTF) パネル表示
# ==========================================
with st.expander("🌐 全時間軸 トレンド方向一覧 (マルチタイムフレーム分析)", expanded=True):
    col_mtf = st.columns(len(TIMEFRAMES))
    mtf_results = get_mtf_trends(symbol)
    
    for idx, (tf_name, trend_str) in enumerate(mtf_results.items()):
        short_name = tf_name.split(" ")[0]
        with col_mtf[idx]:
            st.metric(label=short_name, value=trend_str)

st.markdown("---")

# ==========================================
# 9. メインコンテンツ・タブ構成
# ==========================================
tab_main, tab_scanner, tab_repeat, tab_backtest = st.tabs([
    "🎯 単発デイトレ分析", 
    "🔍 全ペアAI一括スキャン", 
    "🔄 松井証券リピート計算", 
    "🧪 バックテスト・検証"
])

# ------------------------------------------
# タブ 1: 単発デイトレ分析
# ------------------------------------------
with tab_main:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        fig = create_chart(df, selected_pair_name, selected_tf_name)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🤖 AI売買判定 & 資金管理")

        # 時間帯チェック (東京/NY指標時間警戒)
        now_jst = datetime.now()
        is_indicator_time = (21 <= now_jst.hour <= 22) or (now_jst.hour == 23 and now_jst.minute <= 30)

        # AI分析の実行
        signal, buy_prob, accuracy = train_and_predict(df)

        # 手動オーバーライドまたは時間帯制限の適用
        if manual_hold_override:
            signal = "HOLD"
            st.warning("⚠️ 手動オーバーライドがONのため、HOLD判定となっています。")
        elif is_indicator_time:
            signal = "HOLD"
            st.warning("⚠️ 重要経済指標の発表時間帯（21:00-23:30）のため、自動静観モードです。")

        # シグナル表示
        if signal == "BUY":
            st.success(f"### 判定: 買 (BUY) 📈")
        elif signal == "SELL":
            st.error(f"### 判定: 売 (SELL) 📉")
        else:
            st.info(f"### 判定: 静観 (HOLD) ⏸️")

        st.metric("上昇予測確率", f"{buy_prob * 100:.1f}%")
        st.metric("AIモデル検証精度", f"{accuracy * 100:.1f}%")

        st.markdown("---")
        st.subheader("🛡️ リスク管理メーター")

        # 許容損失額とSL/TP算出
        max_loss_amount = account_balance * (risk_per_trade_pct / 100.0)
        latest_close = df['Close'].iloc[-1]
        latest_atr = df['ATR_14'].iloc[-1]

        # SL / TP の幅 (ATRベース)
        sl_pips = latest_atr * 1.5
        tp_pips = latest_atr * 3.0

        # 通貨ペアに応じた損益計算係数 (対円通貨かドルストレートか)
        pip_multiplier = 100.0 if "JPY" in selected_pair_name else 10000.0
        sl_pips_display = sl_pips * (100 if "JPY" in selected_pair_name else 10000)

        # 適正数量 (Lot) の計算
        if sl_pips > 0:
            recommended_units = max_loss_amount / sl_pips
            recommended_lots = recommended_units / 100000.0  # 1Lot = 10万通貨
        else:
            recommended_lots = 0.0

        st.write(f"1トレード想定許容損失額: **{max_loss_amount:,.0f} 円**")
        st.progress(min(risk_per_trade_pct / 5.0, 1.0))

        st.markdown(f"""
        * **推奨発注数量:** `{recommended_lots:.2f} Lot` ({int(recommended_lots * 100000):,} 通貨)
        * **損切り幅 (SL):** `{sl_pips_display:.1f} pips`
        * **利確幅 (TP):** `{sl_pips_display * 2.0:.1f} pips` (リスクリワード 1:2)
        """)

        # Discord通知ボタン
        if st.button("📢 この分析結果をDiscordに送信"):
            msg = (
                f"【FX AI Signal v6.0】\n"
                f"■ 通貨ペア: {selected_pair_name} ({selected_tf_name})\n"
                f"■ AI判定: {signal}\n"
                f"■ 上昇確率: {buy_prob * 100:.1f}%\n"
                f"■ 現在価格: {latest_close:.3f}\n"
                f"■ 推奨数量: {recommended_lots:.2f} Lot"
            )
            if send_discord_notification(discord_webhook_url, msg):
                st.success("Discordに通知を送信しました！")

# ------------------------------------------
# タブ 2: 全ペアAI一括スキャン
# ------------------------------------------
with tab_scanner:
    st.subheader("🔍 全主要通貨ペア 一括AIスキャン")
    st.write("現在設定されている時間軸において、全通貨ペアの売買シグナルを一括判定します。")

    if st.button("全通貨ペアをスキャン実行"):
        scan_results = []
        progress_bar = st.progress(0)
        
        for idx, (p_name, p_sym) in enumerate(PAIR_MAP.items()):
            try:
                # 高速化のためスキャン時はリクエストパラメータを最適化
                p_df = fetch_stock_data(p_sym, tf_params["period"], tf_params["interval"])
                if len(p_df) >= 50:
                    p_df = add_technical_indicators(p_df)
                    p_sig, p_prob, p_acc = train_and_predict(p_df, n_estimators=40)
                    p_close = p_df['Close'].iloc[-1]
                else:
                    p_sig, p_prob, p_acc, p_close = "データ不足", 0.5, 0.0, 0.0
            except Exception:
                p_sig, p_prob, p_acc, p_close = "エラー", 0.5, 0.0, 0.0

            scan_results.append({
                "通貨ペア": p_name,
                "現在価格": f"{p_close:.3f}",
                "AI判定": p_sig,
                "買確率": f"{p_prob * 100:.1f}%",
                "モデル精度": f"{p_acc * 100:.1f}%"
            })
            progress_bar.progress((idx + 1) / len(PAIR_MAP))

        scan_df = pd.DataFrame(scan_results)
        st.dataframe(scan_df, use_container_width=True)

# ------------------------------------------
# タブ 3: 松井証券リピート注文計算
# ------------------------------------------
with tab_repeat:
    st.subheader("🔄 松井証券FX リピート注文設定シミュレーター")

    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        range_min = st.number_input("想定レンジ下限", value=140.0, step=0.5)
        range_max = st.number_input("想定レンジ上限", value=155.0, step=0.5)
        grid_count = st.number_input("本数 (本)", min_value=5, max_value=100, value=30)
    with col_rep2:
        quantity_per_grid = st.number_input("1本あたりの数量 (通貨)", min_value=100, value=1000, step=100)
        tp_width = st.number_input("1本あたりの利確幅 (円)", value=0.5, step=0.1)

    grid_width = (range_max - range_min) / max((grid_count - 1), 1)
    total_units = grid_count * quantity_per_grid
    
    # 簡易必要証拠金計算 (レバレッジ25倍換算)
    avg_price = (range_min + range_max) / 2.0
    required_margin = (avg_price * total_units) / 25.0
    expected_profit_per_grid = quantity_per_grid * tp_width

    st.markdown("---")
    st.write(f"### 📊 設定試算結果")
    st.write(f"* **注文間隔:** `{grid_width:.2f} 円幅`")
    st.write(f"* **合計運用数量:** `{total_units:,} 通貨`")
    st.write(f"* **概算必要証拠金:** `{required_margin:,.0f} 円`")
    st.write(f"* **1回決済あたり見込み利益:** `{expected_profit_per_grid:,.0f} 円`")

# ------------------------------------------
# タブ 4: バックテスト・検証
# ------------------------------------------
with tab_backtest:
    st.subheader("🧪 過去データによるAI戦略のバックテスト")
    st.write("選択中の通貨ペア・時間軸データに対し、過去のシグナル精度と勝率を検証します。")

    if st.button("バックテストを実行"):
        with st.spinner("過去データを検証中..."):
            bt_df, trades, win_rate = run_backtest(df)

            if trades > 0:
                st.write(f"### 📈 検証結果")
                col_bt1, col_bt2 = st.columns(2)
                col_bt1.metric("総トレード回数", f"{trades} 回")
                col_bt2.metric("勝率", f"{win_rate:.1f} %")

                st.dataframe(bt_df.tail(20), use_container_width=True)
            else:
                st.warning("有効なトレードデータが検出されませんでした。")
