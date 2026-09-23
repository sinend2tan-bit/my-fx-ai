import streamlit as st
import yfinance as ticker_fetcher
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# 1. 画面初期設定
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FX 統合トレーディング・アシスタント v3.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ FX 統合トレーディング・アシスタント v3.0")
st.caption("単発AI予測 & 松井証券リピート自動売買パラメータ算出ツール")

# -----------------------------------------------------------------------------
# 2. サイドバー：口座・既存ポジション管理（含み損シミュレーション用）
# -----------------------------------------------------------------------------
st.sidebar.header("💰 口座・保有ポジション設定")

account_balance = st.sidebar.number_input(
    "口座残高（円）", value=1000000, step=50000, help="現在の口座資金を入力してください"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 既存ポジション（塩漬け等）")
has_position = st.sidebar.checkbox("含み損ポジションを保有中", value=True)

pos_pair = "GBPJPY=X"
pos_type = "SELL (売り)"
pos_price = 195.00
pos_lots = 0.1

if has_position:
    pos_pair_symbol = st.sidebar.selectbox(
        "ポジションの通貨ペア",
        ["GBPJPY=X", "USDJPY=X", "EURJPY=X", "AUDJPY=X"],
        index=0
    )
    pos_type = st.sidebar.radio("売買種別", ["SELL (売り)", "BUY (買い)"], index=0)
    pos_price = st.sidebar.number_input("平均取得単価", value=198.50, step=0.1)
    pos_lots = st.sidebar.number_input("保有数量 (万通貨)", value=1.0, step=0.1)

# -----------------------------------------------------------------------------
# 3. データ取得＆インジケーター計算関数
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_data(symbol, period="60d", interval="1h"):
    df = ticker_fetcher.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['Std'] = df['Close'].rolling(20).std()
    df['Upper_2Sigma'] = df['SMA20'] + (df['Std'] * 2)
    df['Lower_2Sigma'] = df['SMA20'] - (df['Std'] * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR (ボラティリティ指標)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    df.dropna(inplace=True)
    return df

# -----------------------------------------------------------------------------
# 4. メイン表示切り替え（タブ構成）
# -----------------------------------------------------------------------------
tab_repeat, tab_single, tab_risk = st.tabs([
    "🔄 リピート売買（松井証券向け）", 
    "🎯 単発トレード（AI判定）", 
    "📊 保有ポジション・ロスカット計算"
])

# 通貨ペア選択
symbol_map = {
    "米ドル/円 (USD/JPY)": "USDJPY=X",
    "ポンド/円 (GBP/JPY)": "GBPJPY=X",
    "ユーロ/円 (EUR/JPY)": "EURJPY=X",
    "豪ドル/円 (AUD/JPY)": "AUDJPY=X",
    "豪ドル/NZドル (AUD/NZD)": "AUDNZD=X"
}
selected_pair_label = st.selectbox("分析・計算対象の通貨ペアを選択", list(symbol_map.keys()), index=1)
symbol = symbol_map[selected_pair_label]

df = load_data(symbol)

if df.empty:
    st.error("データの取得に失敗しました。しばらく時間をおいて再試行してください。")
    st.stop()

current_price = df['Close'].iloc[-1]

# =============================================================================
# TAB 1: リピート売買（松井証券向けパラメータ提示）
# =============================================================================
with tab_repeat:
    st.header("🔄 松井証券FX リピート注文設定 パラメータ設定")
    st.info("※このモードでは「HOLD」判定は出ず、常時リピート運用に必要なレンジ帯を算出します。")

    # リピート計算ロジック（過去N日間の高値・安値・ボラティリティからレンジ設定）
    lookback_days = st.slider("想定レンジの算出期間（日）", min_value=10, max_value=60, value=30)
    recent_df = df.tail(lookback_days * 24)
    
    range_high = recent_df['High'].max()
    range_low = recent_df['Low'].min()
    atr_val = recent_df['ATR'].iloc[-1]
    
    # おすすめ注文値幅・益出し幅（pips計算）
    suggested_grid_pips = max(round(atr_val * 100 * 0.5, 0), 10.0)  # ATRに基づいた格子幅
    suggested_tp_pips = suggested_grid_pips * 1.5                   # 利確幅
    
    col1, col2, col3 = st.columns(3)
    col1.metric("現在価格", f"{current_price:.3f} 円")
    col2.metric(f"過去{lookback_days}日 最高値（上限目安）", f"{range_high:.3f} 円")
    col3.metric(f"過去{lookback_days}日 最安値（下限目安）", f"{range_low:.3f} 円")

    st.markdown("---")
    st.subheader("📋 松井証券アプリ入力用推移データ")
    
    # おすすめ設定の表示
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("#### 🔹 おすすめ推奨値（ハーフ＆ハーフまたは買いレンジ）")
        st.write(f"**注文種別:** 買い (BUY)")
        st.write(f"**レンジ上限:** `{(current_price + (range_high - current_price)*1.1):.2f}` 円")
        st.write(f"**レンジ下限:** `{range_low:.2f}` 円")
        st.write(f"**注文値幅（仕掛け幅）:** `{suggested_grid_pips:.0f}` pips")
        st.write(f"**益出し幅:** `{suggested_tp_pips:.0f}` pips")
        st.write(f"**推奨1注文あたりの数量:** `0.01万通貨（100通貨単位）` ※資金保護のため最小推奨")

    with rec_col2:
        st.markdown("#### ⚠️ リピート運用上の注意")
        st.warning(
            "現在の口座環境に既存の含み損ポジションがある場合、**リピート注文の同時稼働は証拠金維持率を急激に低下させる危険性**があります。\n\n"
            "必ず右側の「保有ポジション・ロスカット計算」タブで安全性を確認してから、一番小さな数量（0.01万通貨等）で開始してください。"
        )

# =============================================================================
# TAB 2: 単発トレード（AI判定・HOLDフィルター付き）
# =============================================================================
with tab_single:
    st.header("🎯 単発トレード AI未来予測（短期ブレイク狙い）")
    st.caption("厳格なフィルタリングを行っているため、条件が揃わない場合は「HOLD」が出ます。")
    
    # 簡易AI学習モデルの構築
    df_ai = df.copy()
    df_ai['Target'] = np.where(df_ai['Close'].shift(-3) > df_ai['Close'], 1, 0)
    
    features = ['SMA20', 'RSI', 'ATR', 'Upper_2Sigma', 'Lower_2Sigma']
    X = df_ai[features]
    y = df_ai['Target']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X[:-3], y[:-3])
    
    latest_X = X.iloc[[-1]]
    pred = model.predict(latest_X)[0]
    prob = model.predict_proba(latest_X)[0]
    
    # 安全フィルター（20SMA乖離率チェック等）
    sma20 = df['SMA20'].iloc[-1]
    dev_rate = abs(current_price - sma20) / sma20 * 100
    rsi_val = df['RSI'].iloc[-1]
    
    # 判定ロジック
    signal = "HOLD（静観）"
    color = "gray"
    reason = "明確な根拠がないか、高値掴みリスクがあります。"
    
    if prob[1] >= 0.65 and dev_rate < 0.8 and rsi_val < 65:
        signal = "BUY (買い)"
        color = "green"
        reason = "AI予測の確信度が高く、高価格乖離もありません。"
    elif prob[0] >= 0.65 and dev_rate < 0.8 and rsi_val > 35:
        signal = "SELL (売り)"
        color = "red"
        reason = "AI予測の下落確信度が高く、売りに適した条件です。"
    else:
        if dev_rate >= 0.8:
            reason = "価格が移動平均線から離れすぎており、高値掴み・安値売りの危険があります。"
        elif rsi_val >= 65 or rsi_val <= 35:
            reason = "RSIが買われすぎ/売られすぎゾーンにあり、反転リスクが高まっています。"

    # シグナル表示
    st.subheader(f"判定結果: :{color}[{signal}]")
    st.write(f"**判定理由:** {reason}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("AI 上昇予測確率", f"{prob[1]*100:.1f}%")
    m2.metric("20SMA 乖離率", f"{dev_rate:.2f}%", help="0.8%以上は高値掴み注意")
    m3.metric("RSI (14)", f"{rsi_val:.1f}")

# =============================================================================
# TAB 3: 保有ポジション・ロスカット計算機
# =============================================================================
with tab_risk:
    st.header("📊 塩漬けポジション＆ロスカット危険度シミュレーション")
    
    if not has_position:
        st.success("現在、登録されている既存の含み損ポジションはありません。")
    else:
        # 現在価格の取得
        pos_df = load_data(pos_pair_symbol)
        if not pos_df.empty:
            pos_current = pos_df['Close'].iloc[-1]
            
            # 含み損益の計算
            units = pos_lots * 10000
            if "SELL" in pos_type:
                pips_diff = (pos_price - pos_current) * 100
                unrealized_pnl = (pos_price - pos_current) * units
            else:
                pips_diff = (pos_current - pos_price) * 100
                unrealized_pnl = (pos_current - pos_price) * units
                
            effective_balance = account_balance + unrealized_pnl
            
            st.subheader(f"📌 対象ポジション: {pos_pair_symbol} [{pos_type}]")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("取得単価", f"{pos_price:.2f} 円")
            c2.metric("現在価格", f"{pos_current:.2f} 円")
            
            pnl_color = "normal" if unrealized_pnl >= 0 else "inverse"
            c3.metric("推定含み損益", f"{unrealized_pnl:,.0f} 円", f"{pips_diff:+.1f} pips", delta_color=pnl_color)

            st.markdown("---")
            st.subheader("🚨 ロスカット耐性（どこまで逆行に耐えられるか）")
            
            # 必要証拠金（レバレッジ25倍想定）
            margin_required = (pos_current * units) / 25
            
            # 限界逆行幅（概算）
            if "SELL" in pos_type:
                # 売りポジションの場合、上昇がリスク
                margin_call_price = pos_current + (effective_balance - margin_required) / units
                allowable_pips = (margin_call_price - pos_current) * 100
            else:
                # 買いポジションの場合、下落がリスク
                margin_call_price = pos_current - (effective_balance - margin_required) / units
                allowable_pips = (pos_current - margin_call_price) * 100

            rc1, rc2 = st.columns(2)
            rc1.metric("実質口座残高", f"{effective_balance:,.0f} 円")
            rc2.metric("強制ロスカット目安価格", f"{margin_call_price:.2f} 円")
            
            if allowable_pips < 300:
                st.error(f"⚠️ 危険: あと **{allowable_pips:.0f} pips** 逆行すると強制ロスカットの可能性が高まります！新規リピート注文は控えてください。")
            elif allowable_pips < 800:
                st.warning(f"⚡ 警戒: あと **{allowable_pips:.0f} pips** 耐えられます。リピート売買を行う場合は最少ロット（0.01万通貨）に限定してください。")
            else:
                st.success(f"✅ 余裕あり: あと **{allowable_pips:.0f} pips** 耐えられます。計画的なリピート運用が可能です。")
