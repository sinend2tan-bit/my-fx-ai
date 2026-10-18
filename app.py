# app.py
# 松井証券FX向け：AI分析・注文補助＋リピート注文設定サポート
# 発注・口座ログイン機能はありません。推奨値は参考情報で、利益を保証しません。

import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FX AI分析・リピート設定サポート",
    page_icon="📈",
    layout="centered",
)

# -----------------------------
# データ・指標
# -----------------------------
PAIR_TICKERS = {
    "米ドル/円 (USD/JPY)": "USDJPY=X",
    "ユーロ/円 (EUR/JPY)": "EURJPY=X",
    "英ポンド/円 (GBP/JPY)": "GBPJPY=X",
    "豪ドル/円 (AUD/JPY)": "AUDJPY=X",
}
PIP_SIZE = 0.01  # 円を含む通貨ペアの1 pip

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCデータから指標を作成。未来データは指標に使わない。"""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(out.columns):
        raise ValueError("価格データに Open / High / Low / Close がありません。")

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(14).mean()

    out["return_1"] = close.pct_change()
    out["return_3"] = close.pct_change(3)
    out["ema_gap"] = (out["ema20"] - out["ema50"]) / close.replace(0, np.nan)
    out["range_pct"] = (high - low) / close.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(ticker: str) -> pd.DataFrame:
    """Yahoo Financeの参考データを取得。松井証券のレートではない。"""
    import yfinance as yf

    raw = yf.download(
        ticker,
        period="6mo",
        interval="1h",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError("価格データを取得できませんでした。")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    needed = ["open", "high", "low", "close"]
    if not set(needed).issubset(raw.columns):
        raise ValueError("取得データの形式が想定と異なります。")
    raw = raw[needed].dropna().sort_index()
    # 最新足は未確定の可能性があるため、保守的に除外
    if len(raw) > 2:
        raw = raw.iloc[:-1]
    return raw


def train_and_predict(ind: pd.DataFrame):
    """
    時系列順に分割して分類器を検証。
    将来6本以内の変化をATRで分類する。実際の売買勝率ではない。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    feature_cols = [
        "rsi", "ema_gap", "return_1", "return_3", "range_pct",
        "atr",
    ]
    data = ind.copy()
    horizon = 6
    future = data["close"].shift(-horizon) - data["close"]
    threshold = data["atr"] * 0.8
    data["target"] = np.select(
        [future > threshold, future < -threshold],
        [1, -1],
        default=0,
    )
    # 未来データが存在しない末尾は学習・検証対象から除外
    labeled = data.iloc[:-horizon].dropna(subset=feature_cols + ["target"])
    current = data.dropna(subset=feature_cols)
    if len(labeled) < 120 or current.empty:
        raise ValueError("学習に使えるデータが不足しています。時間をおいて再試行してください。")

    split = int(len(labeled) * 0.8)
    split = max(60, min(split, len(labeled) - 30))
    train = labeled.iloc[:split]
    test = labeled.iloc[split:]
    # 境界付近のラベルが学習期間をまたがないように間隔を空ける
    if len(train) > horizon + 20:
        train = train.iloc[:-horizon]

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[feature_cols], train["target"].astype(int))
    validation_pred = model.predict(test[feature_cols])
    validation_accuracy = accuracy_score(test["target"].astype(int), validation_pred)

    latest = current.iloc[[-1]]
    probabilities = model.predict_proba(latest[feature_cols])[0]
    prob_map = {int(c): float(p) for c, p in zip(model.classes_, probabilities)}
    # クラスが学習データに存在しない場合も安全に扱う
    probs = {
        -1: prob_map.get(-1, 0.0),
        0: prob_map.get(0, 0.0),
        1: prob_map.get(1, 0.0),
    }
    return latest.iloc[0], probs, float(validation_accuracy), len(labeled)


def yen_risk_per_unit(stop_distance: float) -> float:
    # 円を含む通貨ペアの場合、1通貨あたりの損益は価格差（円）に概ね相当
    return max(0.0, float(stop_distance))


def safe_position_size(capital: float, risk_pct: float, stop_distance: float) -> tuple[int, float]:
    risk_budget = capital * risk_pct / 100.0
    per_unit = yen_risk_per_unit(stop_distance)
    if per_unit <= 0 or risk_budget <= 0:
        return 0, 0.0
    units = max(0, math.floor(risk_budget / per_unit))
    # 切り上げない。数量を切り下げて、想定損失が予算を超えないようにする
    estimated_loss = units * per_unit
    if estimated_loss > risk_budget:
        units = max(0, units - 1)
        estimated_loss = units * per_unit
    return units, estimated_loss


def fmt_price(value: float) -> str:
    return f"{value:,.3f}"


# -----------------------------
# 画面
# -----------------------------
st.title("📈 FX AI分析・リピート設定サポート")
st.caption("松井証券FX向けの注文補助ツール｜口座接続なし・自動発注なし")

with st.expander("最初にお読みください（重要）", expanded=True):
    st.warning(
        "このツールは参考情報を表示するもので、利益や勝率を保証しません。"
        "価格データはYahoo Finance由来で、松井証券のリアルタイムレートとは一致しない場合があります。"
        "必ず松井証券アプリのレート・スプレッド・注文条件を確認し、発注はご自身で行ってください。"
    )

st.subheader("① 基本設定")
c1, c2 = st.columns(2)
with c1:
    pair_name = st.selectbox("通貨ペア", list(PAIR_TICKERS.keys()), index=0)
    capital = st.number_input("運用資金（円）", min_value=10000, max_value=100000000, value=300000, step=10000)
with c2:
    style = st.selectbox("運用スタイル", ["安全重視", "標準", "積極運用"], index=0)
    risk_pct = st.selectbox("1回あたりの許容損失", [0.25, 0.5, 1.0], index=1, format_func=lambda x: f"{x:.2f}%")

risk_pct_value = float(risk_pct)
risk_budget = capital * risk_pct_value / 100.0
st.caption(f"設定上の許容損失：**{risk_budget:,.0f}円**（保証値ではなく、スプレッドや滑りは別途発生します）")

st.divider()
st.subheader("② AI相場分析・注文補助")
if st.button("相場データを取得して分析", type="primary", use_container_width=True):
    st.session_state["run_analysis"] = True

if st.session_state.get("run_analysis", False):
    try:
        with st.spinner("価格データと指標を計算しています…"):
            raw = fetch_market_data(PAIR_TICKERS[pair_name])
            ind = add_indicators(raw).dropna(subset=["ema20", "ema50", "atr"])
            latest, probs, val_acc, n_rows = train_and_predict(ind)

        price = float(latest["close"])
        atr = float(latest["atr"])
        rsi = float(latest["rsi"])
        ema20 = float(latest["ema20"])
        ema50 = float(latest["ema50"])
        buy_p = probs[1]
        sell_p = probs[-1]
        neutral_p = probs[0]
        direction = max(probs, key=probs.get)
        confidence = probs[direction]

        if style == "安全重視":
            min_confidence = 0.50
            sl_mult, tp_mult = 1.5, 2.0
        elif style == "積極運用":
            min_confidence = 0.45
            sl_mult, tp_mult = 1.3, 1.8
        else:
            min_confidence = 0.48
            sl_mult, tp_mult = 1.4, 2.0

        stop_distance = max(atr * sl_mult, PIP_SIZE * 5)
        take_distance = max(atr * tp_mult, PIP_SIZE * 5)
        units, estimated_loss = safe_position_size(capital, risk_pct_value, stop_distance)

        if direction == 1:
            signal_text = "買い方向の候補"
            entry = price
            stop = entry - stop_distance
            take = entry + take_distance
        elif direction == -1:
            signal_text = "売り方向の候補"
            entry = price
            stop = entry + stop_distance
            take = entry - take_distance
        else:
            signal_text = "見送り候補"
            entry = stop = take = price

        if confidence < min_confidence or direction == 0 or units < 1:
            verdict = "🟡 見送り・追加確認"
            signal_text = "見送り候補"
        else:
            verdict = "🟠 条件確認後に検討"

        st.markdown(f"### {verdict}")
        st.write(f"**AI判定：** {signal_text}")
        m1, m2, m3 = st.columns(3)
        m1.metric("参考価格", fmt_price(price))
        m2.metric("RSI（14）", f"{rsi:.1f}")
        m3.metric("ATR（14）", f"{atr:.3f}")

        st.write("**モデルの出力確率（売買の実際の勝率ではありません）**")
        st.write(f"- 買い方向：{buy_p:.1%}")
        st.write(f"- 見送り方向：{neutral_p:.1%}")
        st.write(f"- 売り方向：{sell_p:.1%}")
        st.caption(
            f"時系列で分けた検証データの分類正解率：{val_acc:.1%}。"
            f"対象データ {n_rows} 本。これは売買勝率・将来利益を意味しません。"
        )

        if direction != 0 and confidence >= min_confidence and units >= 1:
            st.markdown("#### 注文補助の目安")
            t1, t2, t3 = st.columns(3)
            t1.metric("数量（通貨）", f"{units:,}")
            t2.metric("損切り価格の目安", fmt_price(stop))
            t3.metric("利確価格の目安", fmt_price(take))
            st.write(f"- エントリー参考価格：{fmt_price(entry)}")
            st.write(f"- 損切り幅：約 {stop_distance / PIP_SIZE:.1f} pips")
            st.write(f"- 利確幅：約 {take_distance / PIP_SIZE:.1f} pips")
            st.write(f"- 数量から計算した想定損失：約 {estimated_loss:,.0f}円（コスト除く）")
        else:
            st.info("確率・方向・数量の条件が不十分なため、注文数量を推奨しません。")

        if (price > ema20 > ema50):
            trend = "上向きの並び"
        elif (price < ema20 < ema50):
            trend = "下向きの並び"
        else:
            trend = "方向が混在"
        st.write(f"**移動平均の状態：** {trend}")
        st.caption(
            "分析データの最終時刻："
            + str(ind.index[-1])
            + "（データ提供元の時刻表記。松井証券レートとの一致は保証されません）"
        )
        st.session_state["market_indicators"] = ind.tail(500).copy()

    except Exception as exc:
        st.error("分析を完了できませんでした。時間をおいて再試行してください。")
        st.caption(f"詳細：{type(exc).__name__}: {exc}")
        st.info("データ取得元の停止・通信エラー・利用ライブラリの問題などが考えられます。")

st.divider()
st.subheader("③ リピート注文設定サポート")
st.write(
    "直近の価格データの値幅・ATRから設定候補を計算します。"
    "これは利益が最大になる設定を保証するものではなく、実注文の約定・スプレッド・スワップを完全再現するものでもありません。"
)

if "market_indicators" in st.session_state:
    hist = st.session_state["market_indicators"].dropna(subset=["high", "low", "close", "atr"])
    if len(hist) >= 30:
        source_label = "直前の分析で取得したデータ"
    else:
        hist = pd.DataFrame()
        source_label = "手入力"
else:
    hist = pd.DataFrame()
    source_label = "手入力"

if not hist.empty:
    q_low = float(hist["low"].quantile(0.05))
    q_high = float(hist["high"].quantile(0.95))
    current_atr = float(hist["atr"].iloc[-1])
    st.caption(f"レンジ候補の出典：{source_label}")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        manual_price = st.number_input("現在価格（円）", min_value=1.0, max_value=1000.0, value=150.0, step=0.1)
    with c2:
        manual_range_pct = st.number_input("レンジ幅（%）", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
    with c3:
        manual_atr_pips = st.number_input("ATR相当（pips）", min_value=1.0, max_value=500.0, value=20.0, step=1.0)
    q_low = manual_price * (1 - manual_range_pct / 200)
    q_high = manual_price * (1 + manual_range_pct / 200)
    current_atr = manual_atr_pips * PIP_SIZE
    st.caption("価格データを取得していないため、入力値からの参考計算です。")

c1, c2 = st.columns(2)
with c1:
    repeat_side = st.selectbox("リピート方向", ["買い", "売り"], index=0)
with c2:
    max_risk_pct = st.selectbox("リピート設定全体の許容含み損", [1.0, 2.0, 3.0, 5.0], index=1, format_func=lambda x: f"資金の{x:.0f}%")

repeat_risk_budget = capital * float(max_risk_pct) / 100.0
# ATRと直近レンジ幅をもとに、複数候補を比較する
range_width = max(q_high - q_low, PIP_SIZE * 10)
spacing_candidates = sorted(set([
    max(PIP_SIZE * 5, current_atr * 0.5),
    max(PIP_SIZE * 5, current_atr * 0.75),
    max(PIP_SIZE * 5, current_atr * 1.0),
]))
take_candidates = sorted(set([
    max(PIP_SIZE * 5, current_atr * 0.5),
    max(PIP_SIZE * 5, current_atr * 0.75),
    max(PIP_SIZE * 5, current_atr * 1.0),
]))

candidate_rows = []
for spacing in spacing_candidates:
    count = max(1, int(math.floor(range_width / spacing)) + 1)
    # 想定したレンジ内に等間隔で置く価格。範囲外へ自動追加しない。
    levels = [q_low + i * spacing for i in range(count)]
    levels = [x for x in levels if x <= q_high + spacing * 1e-6]
    if not levels:
        levels = [q_low]
    if repeat_side == "買い":
        adverse_sum = sum(max(0.0, level - q_low) for level in levels)
    else:
        adverse_sum = sum(max(0.0, q_high - level) for level in levels)
    unit_loss = max(adverse_sum, PIP_SIZE)
    qty_per_order = max(0, math.floor(repeat_risk_budget / unit_loss))
    worst_case = qty_per_order * unit_loss
    for take in take_candidates:
        candidate_rows.append({
            "注文値幅(pips)": round(spacing / PIP_SIZE, 1),
            "益出し幅(pips)": round(take / PIP_SIZE, 1),
            "レンジ内注文数目安": len(levels),
            "1注文あたり数量(通貨)": qty_per_order,
            "単純想定含み損(円)": round(worst_case),
            "リスク予算内": worst_case <= repeat_risk_budget,
            "_spacing": spacing,
            "_take": take,
            "_unit_loss": unit_loss,
        })

candidates = pd.DataFrame(candidate_rows)
# リスク予算超過は除外。数量が0なら実用的候補とは扱わない。
valid_candidates = candidates[
    (candidates["リスク予算内"]) & (candidates["1注文あたり数量(通貨)"] > 0)
].copy()

st.write(f"**参考レンジ：** {fmt_price(q_low)} ～ {fmt_price(q_high)}")
st.write(f"**設定全体の含み損予算：** {repeat_risk_budget:,.0f}円")
if valid_candidates.empty:
    st.warning("この設定ではリスク予算内に収まる数量を計算できません。数量を減らすか、レンジ・値幅を見直してください。")
else:
    # 初期候補は、注文値幅が広めで、含み損を抑えやすいものを優先
    valid_candidates = valid_candidates.sort_values(
        by=["単純想定含み損(円)", "注文値幅(pips)", "益出し幅(pips)"],
        ascending=[True, False, True],
    )
    best = valid_candidates.iloc[0]
    st.markdown("#### 推奨候補（リスク優先の簡易計算）")
    a, b, c = st.columns(3)
    a.metric("注文値幅", f"{best['注文値幅(pips)']:.1f} pips")
    b.metric("益出し幅", f"{best['益出し幅(pips)']:.1f} pips")
    c.metric("1注文あたり数量", f"{int(best['1注文あたり数量(通貨)']):,} 通貨")
    st.write(f"- レンジ下限：{fmt_price(q_low)}")
    st.write(f"- レンジ上限：{fmt_price(q_high)}")
    st.write(f"- レンジ内の注文数目安：{int(best['レンジ内注文数目安'])} 本")
    st.write(f"- 単純想定含み損：約 {best['単純想定含み損(円)']:,.0f}円（コスト・窓開け・レンジ外への急変を除く）")
    st.caption("この候補は収益最大化ではなく、単純な含み損予算を優先した候補です。実際の注文数量・必要証拠金・注文間隔は松井証券アプリで必ず確認してください。")

    display_cols = [
        "注文値幅(pips)", "益出し幅(pips)", "レンジ内注文数目安",
        "1注文あたり数量(通貨)", "単純想定含み損(円)",
    ]
    st.write("**比較候補**")
    st.dataframe(
        valid_candidates[display_cols].head(9).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("④ 使い方")
st.markdown(
    """
1. 運用資金と運用スタイルを選びます。
2. 「相場データを取得して分析」を押します。
3. AIの方向、損切り・利確の参考価格、数量を確認します。
4. リピート注文の設定候補を確認します。
5. 松井証券アプリで最新レート・注文仕様・必要証拠金を確認してから、ご自身で発注します。
"""
)
st.info(
    "このアプリは注文を送信しません。価格データ取得に失敗した場合は無理に取引せず、"
    "松井証券の取引画面で最新情報を確認してください。"
)
st.caption("試作版｜最終更新：" + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
