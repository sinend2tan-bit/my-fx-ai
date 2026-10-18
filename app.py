# app.py
# 松井証券FX向け AI分析・注文補助 + リピート注文設定サポート
# 口座接続・自動発注は行いません。推奨値は参考情報であり、利益を保証しません。

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FX AI分析・注文補助", page_icon="📈", layout="centered")

PAIRS = {
    "米ドル/円 (USD/JPY)": "USDJPY=X",
    "ユーロ/円 (EUR/JPY)": "EURJPY=X",
    "英ポンド/円 (GBP/JPY)": "GBPJPY=X",
    "豪ドル/円 (AUD/JPY)": "AUDJPY=X",
}
PIP = 0.01


def add_indicators(raw):
    df = raw.copy()
    df.columns = [str(c).lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError("価格データの形式が想定と異なります。")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).sort_index()
    close, high, low = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["return_1"] = close.pct_change()
    df["return_3"] = close.pct_change(3)
    df["ema_gap"] = (df["ema20"] - df["ema50"]) / close.replace(0, np.nan)
    df["range_pct"] = (high - low) / close.replace(0, np.nan)
    return df.replace([np.inf, -np.inf], np.nan)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(ticker):
    import yfinance as yf
    data = yf.download(ticker, period="6mo", interval="1h", auto_adjust=False,
                        progress=False, threads=False, timeout=20)
    if data is None or data.empty:
        raise ValueError("価格データを取得できませんでした。")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).lower() for c in data.columns]
    if not {"open", "high", "low", "close"}.issubset(data.columns):
        raise ValueError("取得データの形式が想定と異なります。")
    data = data[["open", "high", "low", "close"]].dropna().sort_index()
    # 最新足は未確定の可能性があるため除外
    if len(data) > 2:
        data = data.iloc[:-1]
    return data


def ai_analysis(ind):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    features = ["rsi", "ema_gap", "return_1", "return_3", "range_pct", "atr"]
    horizon = 6
    work = ind.copy()
    future = work["close"].shift(-horizon) - work["close"]
    threshold = work["atr"] * 0.8
    work["target"] = np.select([future > threshold, future < -threshold], [1, -1], default=0)
    labeled = work.iloc[:-horizon].dropna(subset=features + ["target"]).copy()
    current = work.dropna(subset=features)
    if len(labeled) < 180 or current.empty:
        raise ValueError("分析データが不足しています。")
    split = int(len(labeled) * 0.8)
    if split < 100 or len(labeled) - split < 40:
        raise ValueError("時系列検証に必要なデータが不足しています。")
    train = labeled.iloc[:max(1, split - horizon)]
    test = labeled.iloc[split:]
    if train["target"].nunique() < 2:
        raise ValueError("学習データのパターンが不足しているため判定できません。")
    model = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=10,
                                   class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(train[features], train["target"].astype(int))
    accuracy = float(accuracy_score(test["target"].astype(int), model.predict(test[features])))
    latest = current.iloc[-1]
    raw_probs = model.predict_proba(current.iloc[[-1]][features])[0]
    mapping = {int(k): float(v) for k, v in zip(model.classes_, raw_probs)}
    probs = {-1: mapping.get(-1, 0.0), 0: mapping.get(0, 0.0), 1: mapping.get(1, 0.0)}
    return latest, probs, accuracy, len(labeled)


def price(value):
    return f"{value:,.3f}"


def units_for_risk(capital, risk_pct, stop_distance, cost_buffer):
    budget = capital * risk_pct / 100
    per_unit = max(0.0, stop_distance) + max(0.0, cost_buffer)
    if budget <= 0 or per_unit <= 0:
        return 0, 0.0, budget
    qty = max(0, math.floor(budget / per_unit))
    return qty, qty * per_unit, budget


st.title("📈 FX AI分析・注文補助")
st.caption("① AI分析・注文補助を優先｜② リピート注文設定も従来型の項目に戻した参考計算")
with st.expander("大切な注意事項", expanded=True):
    st.warning(
        "このツールは参考計算で、利益・勝率・損失上限を保証しません。"
        "価格データはYahoo Finance由来で、松井証券のリアルタイム価格と異なる場合があります。"
        "スプレッド拡大・約定ずれ・窓開け・急変・スワップ等により想定以上の損失が出る可能性があります。"
        "必ず松井証券アプリで価格・注文条件・必要証拠金を確認し、発注は手動で行ってください。"
    )

st.subheader("① 基本設定")
c1, c2 = st.columns(2)
with c1:
    pair = st.selectbox("通貨ペア", list(PAIRS.keys()))
    capital = st.number_input("運用資金（円）", min_value=10000, max_value=100000000,
                              value=300000, step=10000)
with c2:
    style = st.selectbox("運用スタイル", ["安全重視", "標準", "積極運用"], index=0)
    risk_pct = st.selectbox("1回あたりの許容損失", [0.25, 0.5, 1.0], index=0,
                            format_func=lambda x: f"{x:.2f}%")
cost_pips = st.number_input("AI注文補助：コスト余裕（pips）", min_value=1.0, max_value=20.0,
                             value=3.0, step=0.5)
st.caption(f"1回あたりの計算上の損失予算：{capital * risk_pct / 100:,.0f}円（実際の損失上限ではありません）")

st.subheader("② AI相場分析・注文補助")
if st.button("相場データを取得してAI分析", type="primary", use_container_width=True):
    st.session_state["run_ai"] = True
    st.session_state.pop("ai_error", None)

if st.session_state.get("run_ai", False):
    try:
        with st.spinner("価格データを取得して分析しています…"):
            raw = fetch_data(PAIRS[pair])
            ind = add_indicators(raw).dropna(subset=["ema20", "ema50", "atr"])
            latest, probs, acc, n = ai_analysis(ind)

        last_time = pd.Timestamp(ind.index[-1])
        if last_time.tzinfo is None:
            last_time = last_time.tz_localize("UTC")
        else:
            last_time = last_time.tz_convert("UTC")
        age = max(0.0, (pd.Timestamp.now(tz="UTC") - last_time).total_seconds() / 3600)
        p, atr, rsi = float(latest["close"]), float(latest["atr"]), float(latest["rsi"])
        ema20, ema50 = float(latest["ema20"]), float(latest["ema50"])
        direction = max(probs, key=probs.get)
        confidence = probs[direction]
        up = p > ema20 > ema50
        down = p < ema20 < ema50
        aligned = (direction == 1 and up) or (direction == -1 and down)
        fresh = age <= 36
        valid = np.isfinite([p, atr, rsi, ema20, ema50]).all() and atr > 0

        if style == "安全重視":
            min_conf, sl_mult, tp_mult = 0.55, 1.5, 2.0
        elif style == "積極運用":
            min_conf, sl_mult, tp_mult = 0.50, 1.3, 1.8
        else:
            min_conf, sl_mult, tp_mult = 0.53, 1.4, 2.0

        can_trade = direction in (-1, 1) and confidence >= min_conf and aligned and fresh and valid
        stop_dist = max(atr * sl_mult, PIP * 10)
        take_dist = max(atr * tp_mult, PIP * 10)
        qty, loss_est, budget = units_for_risk(capital, float(risk_pct), stop_dist, cost_pips * PIP)

        st.markdown("#### AI判定")
        if not fresh:
            st.error("データが古いため注文補助を停止しました。最新データが確認できるまで見送ってください。")
        elif not valid:
            st.error("価格または指標に異常があるため、注文補助を停止しました。")
        elif can_trade and qty > 0:
            st.info("参考シグナル条件を満たしました。発注前に松井証券の最新レートを再確認してください。")
        else:
            st.warning("条件がそろっていないため、今回は見送り候補です。")

        m1, m2, m3 = st.columns(3)
        m1.metric("参考価格", price(p))
        m2.metric("RSI（14）", f"{rsi:.1f}")
        m3.metric("ATR（14）", f"{atr:.3f}")
        st.write("**モデル出力確率（実際の売買勝率ではありません）**")
        st.write(f"- 買い方向：{probs[1]:.1%}")
        st.write(f"- 見送り方向：{probs[0]:.1%}")
        st.write(f"- 売り方向：{probs[-1]:.1%}")
        st.caption(f"時系列分割データの分類正解率：{acc:.1%}／対象 {n} 本。実際の勝率や将来利益を示すものではありません。")
        st.write(f"**移動平均の状態：** {'上向き' if up else '下向き' if down else '方向が混在'}")
        st.write(f"**データ最終時刻（UTC）：** {last_time.strftime('%Y-%m-%d %H:%M')}（取得から約{age:.1f}時間）")

        if can_trade and qty > 0:
            if direction == 1:
                side, stop, take = "買い方向", p - stop_dist, p + take_dist
            else:
                side, stop, take = "売り方向", p + stop_dist, p - take_dist
            st.markdown("#### 注文補助の参考値")
            st.write(f"**方向：** {side}")
            a, b, c = st.columns(3)
            a.metric("数量の計算値", f"{qty:,} 通貨")
            b.metric("損切り価格の目安", price(stop))
            c.metric("利確価格の目安", price(take))
            st.write(f"- 損切り幅：約 {stop_dist / PIP:.1f} pips")
            st.write(f"- 利確幅：約 {take_dist / PIP:.1f} pips")
            st.write(f"- コスト余裕を加えた想定損失：約 {loss_est:,.0f}円")
            st.caption("数量は円建て通貨ペアの単純概算です。証券会社の取引単位・必要証拠金・実コストに合わない場合は発注しないでください。")
        else:
            st.info("安全条件を満たさない場合は、数量・注文価格を提示しません。")
        st.session_state["market_indicators"] = ind.tail(500).copy()
        st.session_state["last_pair"] = pair
    except Exception as exc:
        st.session_state["ai_error"] = f"{type(exc).__name__}: {exc}"
        st.error("分析を完了できませんでした。データや分析に問題があるため、取引は見送ってください。")
        st.caption("技術情報：" + st.session_state["ai_error"])

st.divider()
st.subheader("③ リピート注文設定（従来型の設定項目）")
st.caption("AI分析とは分けて、注文レンジ・注文間隔・益出し幅・数量・運用停止ラインを確認できます。")

# Auto range if a fresh analysis for this pair exists; otherwise manual range
hist = st.session_state.get("market_indicators")
use_hist = isinstance(hist, pd.DataFrame) and len(hist) >= 30 and st.session_state.get("last_pair") == pair
if use_hist:
    default_low = float(hist["low"].tail(120).quantile(0.05))
    default_high = float(hist["high"].tail(120).quantile(0.95))
    # 価格がほぼ動かない等でレンジ幅がゼロになった場合の安全な初期値
    if not np.isfinite(default_low) or default_low <= 0:
        default_low = 150.0
    if not np.isfinite(default_high) or default_high <= default_low:
        default_high = default_low + max(0.5, default_low * 0.01)
    repeat_atr = float(hist["atr"].dropna().iloc[-1])
    st.caption("初期レンジは直近データの5～95パーセンタイルです。将来の値幅を保証しません。")
else:
    st.caption("AI分析データがない場合は、松井証券アプリの価格を参考に手入力してください。")
    default_price = st.number_input("現在価格（円）", min_value=1.0, max_value=1000.0, value=150.0, step=0.1)
    default_range_pct = st.number_input("レンジ幅（%）", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
    default_low = default_price * (1 - default_range_pct / 200)
    default_high = default_price * (1 + default_range_pct / 200)
    repeat_atr = st.number_input("ATR相当（pips）", min_value=1.0, max_value=500.0, value=20.0, step=1.0) * PIP

mode = st.selectbox("注文方式", ["ハーフ＆ハーフ（買い下がり＋売り上がり）", "買いのみ", "売りのみ"])
c1, c2 = st.columns(2)
with c1:
    range_low = st.number_input("注文レンジ下限（円）", min_value=1.0, max_value=1000.0,
                                value=float(round(default_low, 3)), step=0.01, format="%.3f")
    spacing_pips = st.number_input("注文値幅（pips）", min_value=1.0, max_value=1000.0,
                                   value=float(max(10, round(repeat_atr / PIP))), step=1.0)
    take_pips = st.number_input("益出し幅（pips）", min_value=1.0, max_value=1000.0,
                                value=float(max(10, round(repeat_atr / PIP))), step=1.0)
with c2:
    range_high = st.number_input("注文レンジ上限（円）", min_value=1.0, max_value=1000.0,
                                 value=float(round(default_high, 3)), step=0.01, format="%.3f")
    qty_per_order = st.number_input("1注文あたり数量（通貨）", min_value=1, max_value=1000000,
                                    value=1000, step=100)
    stop_line = st.number_input("運用停止ラインの目安（円・0なら未設定／自動停止なし）", min_value=0.0, max_value=1000.0,
                                value=0.0, step=0.1, format="%.3f")

margin_rate_pct = st.number_input("必要証拠金の概算率（%）", min_value=1.0, max_value=10.0,
                                  value=4.1, step=0.1,
                                  help="概算用の仮設定です。松井証券の実際の必要証拠金率を確認してください。")
max_orders_side = st.number_input("片側の最大注文本数（安全確認用）", min_value=1, max_value=100, value=10, step=1)
max_floating_pct = st.selectbox("許容する概算含み損（資金比）", [1.0, 2.0, 3.0, 5.0],
                                index=0, format_func=lambda x: f"{x:.0f}%")

if range_high <= range_low:
    st.error("レンジ上限は下限より大きい値にしてください。")
else:
    spacing = spacing_pips * PIP
    take = take_pips * PIP
    n_levels = int(math.floor((range_high - range_low) / spacing)) + 1
    n_levels = max(1, n_levels)
    half_mode = mode.startswith("ハーフ")
    buy_count = min(n_levels, max_orders_side) if mode != "売りのみ" else 0
    sell_count = min(n_levels, max_orders_side) if mode != "買いのみ" else 0
    if half_mode:
        mid = (range_low + range_high) / 2
        buy_count = min(max(1, int(math.floor((mid - range_low) / spacing)) + 1), max_orders_side)
        sell_count = min(max(1, int(math.floor((range_high - mid) / spacing)) + 1), max_orders_side)
    total_orders = buy_count + sell_count
    total_units = total_orders * int(qty_per_order)
    mid_price = (range_low + range_high) / 2
    margin_est = total_units * mid_price * margin_rate_pct / 100
    max_float_budget = capital * max_floating_pct / 100
    # Conservative simple estimate: accumulate losses from each grid level to the adverse range edge.
    buy_loss_per_unit = sum(max(0.0, (range_low + i * spacing) - range_low) for i in range(buy_count))
    sell_loss_per_unit = sum(max(0.0, range_high - (range_high - i * spacing)) for i in range(sell_count))
    # Combined estimate; excludes gap/fees/swap and is not a true broker margin simulation.
    float_est = (buy_loss_per_unit + sell_loss_per_unit) * qty_per_order
    st.markdown("#### 設定結果")
    st.write(f"**注文レンジ：** {price(range_low)} ～ {price(range_high)}")
    st.write(f"**注文値幅：** {spacing_pips:.1f} pips　｜　**益出し幅：** {take_pips:.1f} pips")
    a, b, c = st.columns(3)
    a.metric("片側の注文数目安", f"買い {buy_count} / 売り {sell_count}")
    b.metric("合計数量の目安", f"{total_units:,} 通貨")
    c.metric("必要証拠金の概算", f"{margin_est:,.0f}円")
    st.write(f"- 概算含み損（単純計算）：約 {float_est:,.0f}円")
    st.write(f"- 設定した含み損予算：{max_float_budget:,.0f}円")
    st.write(f"- 運用停止ラインの目安：{'未設定' if stop_line == 0 else price(stop_line) + ' 円'}")
    st.caption("重要：運用停止ラインは表示用の目安です。このツールは価格監視・自動停止・決済を行いません。")
    if float_est > max_float_budget:
        st.error("概算含み損が設定予算を超えています。数量を減らす、注文数を減らす、またはレンジを見直してください。")
    elif margin_est > capital * 0.5:
        st.warning("必要証拠金の概算が運用資金の50%を超えています。安全性を再確認してください。")
    else:
        st.info("計算上は設定した概算予算内ですが、実際の損失・必要証拠金を保証するものではありません。")
    st.warning(
        "この概算は、約定順序・既存ポジション・スプレッド・スワップ・ロスカット条件・"
        "レンジ外の急変・窓開けを完全に再現しません。停止ラインを設定しても指定価格で決済できる保証はありません。"
    )

st.divider()
st.subheader("④ 発注前チェック")
for i, item in enumerate([
    "松井証券アプリの最新レートとスプレッドを確認した",
    "注文数量・注文本数・必要証拠金を確認した",
    "買い／売り方向と損切り・停止ラインの目安を確認した（自動停止はされない）",
    "経済指標・週末・急変リスクを確認した",
]):
    st.checkbox(item, key=f"preflight_{i}")
st.info("本ツールは設定案を表示するだけで、注文を送信しません。迷いがある場合は発注を見送ってください。")
st.caption("試作版｜最終更新：" + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
