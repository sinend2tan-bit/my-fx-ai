"""フォワード評価の基準価格（次足始値）に関するテスト。

第1段階の変更（2026-10-30）の検証用。ネットワークは使わない。

確認すること:
- 評価の基準価格がシグナル足の終値（保存済み reference_price）ではなく、
  シグナル足の直後の足（評価ウィンドウの1本目）の始値であること。
- 結果に evaluation_reference_price（専用項目）が記録され、保存済みの
  reference_price が上書きされないこと。
- バリア判定の対象が「次足自身を含む8本」で、学習ラベルの範囲
  （シグナル後の1〜8本目）と一致すること（1本目・8本目の到達は数え、
  9本目の到達は数えない）。
- 評価ウィンドウより後の足が結果に影響しないこと（先読みなし）。
- 週末をまたぐ場合は再開後の最初の足の始値を使うこと。許容できない
  欠けや次足始値の異常では、従来どおり評価せず None（PENDING継続）。
- 現在の記録形式のまま評価でき、統計へ従来どおり反映されること。
"""

import pandas as pd
import pytest

import scanner

TZ = "Asia/Tokyo"
# 月曜 2026-10-26 10:00 JST をシグナル足とする（平日の連続した1時間足）。
SIGNAL = pd.Timestamp("2026-10-26 10:00", tz=TZ)


def make_frame(signal_time, bars):
    """シグナル足の後に続く足だけの DataFrame を作る。

    bars は (open, high, low, close) のリストで、時刻はシグナル後の
    連続した1時間足として割り当てる（週末ケースは make_frame_at で作る）。
    """
    index = pd.date_range(
        start=signal_time + pd.Timedelta(hours=1),
        periods=len(bars),
        freq="1h",
        tz=TZ,
    )
    return pd.DataFrame(
        bars, index=index, columns=["Open", "High", "Low", "Close"]
    ).astype(float)


def make_frame_at(timestamps, bars):
    index = pd.DatetimeIndex(list(timestamps)).tz_convert(TZ)
    return pd.DataFrame(
        bars, index=index, columns=["Open", "High", "Low", "Close"]
    ).astype(float)


def make_prediction(**overrides):
    prediction = {
        "direction": "買い",
        # 保存済み reference_price はシグナル足の終値という想定。
        "reference_price": 150.000,
        "atr": 0.100,
        "pip_size": 0.01,
        "spread_pips": 0.0,
        "signal_bar_time": SIGNAL.isoformat(),
    }
    prediction.update(overrides)
    return prediction


def quiet_bar(open_price=150.050, close=150.050):
    """バリア（次足始値±0.100）に触れない穏やかな足。"""
    return (open_price, open_price + 0.010, open_price - 0.010, close)


def test_anchor_is_next_bar_open_not_signal_close():
    # 次足始値 150.050 → 上バリア 150.150 / 下バリア 149.950。
    # 1本目の高値 150.120 は、旧基準（終値 150.000＋0.100＝150.100）なら
    # 到達だが、新基準では未到達。8本とも未到達なら NOT_REACHED。
    bars = [(150.050, 150.120, 150.040, 150.080)]
    bars += [quiet_bar() for _ in range(6)]
    bars += [quiet_bar(close=150.080)]
    frame = make_frame(SIGNAL, bars)

    outcome = scanner.evaluate_forward_prediction(make_prediction(), frame)

    assert outcome is not None
    assert outcome["status"] == "NOT_REACHED"
    assert outcome["evaluation_reference_price"] == pytest.approx(150.050)
    assert outcome["entry_bar_time"] == frame.index[0].isoformat()
    assert outcome["exit_price"] == pytest.approx(150.080)
    # 損益は次足始値から測る: (150.080 - 150.050) / 0.01 = 3.0 pips
    assert outcome["gross_pips"] == pytest.approx(3.0)


def test_anchor_next_open_below_close_hits_earlier():
    # 次足始値 149.950 → 上バリア 150.050。1本目の高値 150.060 で HIT。
    # 旧基準（150.100）では未到達になるケース。
    bars = [(149.950, 150.060, 149.940, 150.020)]
    bars += [quiet_bar(149.950) for _ in range(7)]
    frame = make_frame(SIGNAL, bars)

    prediction = make_prediction()
    outcome = scanner.evaluate_forward_prediction(prediction, frame)

    assert outcome is not None
    assert outcome["status"] == "HIT"
    assert outcome["exit_bar_offset"] == 1
    assert outcome["exit_price"] == pytest.approx(150.050)
    assert outcome["evaluation_reference_price"] == pytest.approx(149.950)
    # 保存済み reference_price は変更されず、結果にも含まれない（上書き防止）。
    assert prediction["reference_price"] == 150.000
    assert "reference_price" not in outcome


def test_window_includes_next_bar_and_eighth_bar():
    # 1本目（次足自身）の到達を数える。
    bars = [(150.050, 150.160, 150.040, 150.100)]
    bars += [quiet_bar() for _ in range(7)]
    frame = make_frame(SIGNAL, bars)
    outcome = scanner.evaluate_forward_prediction(make_prediction(), frame)
    assert outcome["status"] == "HIT"
    assert outcome["exit_bar_offset"] == 1

    # 8本目の到達を数える（売り方向・下バリア）。
    bars = [quiet_bar() for _ in range(7)]
    bars += [(150.050, 150.060, 149.940, 150.000)]
    frame = make_frame(SIGNAL, bars)
    outcome = scanner.evaluate_forward_prediction(
        make_prediction(direction="売り"), frame
    )
    assert outcome["status"] == "HIT"
    assert outcome["exit_bar_offset"] == 8
    assert outcome["exit_price"] == pytest.approx(149.950)


def test_window_excludes_ninth_bar():
    # 1〜8本目は未到達、9本目で初めて到達 → NOT_REACHED（8本目終値で決済）。
    bars = [quiet_bar() for _ in range(7)]
    bars += [quiet_bar(close=150.060)]
    bars += [(150.050, 150.500, 150.040, 150.400)]
    frame = make_frame(SIGNAL, bars)

    outcome = scanner.evaluate_forward_prediction(make_prediction(), frame)

    assert outcome["status"] == "NOT_REACHED"
    assert outcome["exit_bar_offset"] == 8
    assert outcome["exit_price"] == pytest.approx(150.060)


def test_no_lookahead_beyond_window():
    bars = [(150.050, 150.120, 150.040, 150.080)]
    bars += [quiet_bar() for _ in range(6)]
    bars += [quiet_bar(close=150.080)]
    base = scanner.evaluate_forward_prediction(
        make_prediction(), make_frame(SIGNAL, bars)
    )

    extended = bars + [(150.050, 160.000, 140.000, 155.000)] * 3
    outcome = scanner.evaluate_forward_prediction(
        make_prediction(), make_frame(SIGNAL, extended)
    )

    for key in (
        "status",
        "exit_price",
        "exit_bar_offset",
        "gross_pips",
        "net_pips",
        "mae_pips",
        "evaluation_reference_price",
        "entry_bar_time",
    ):
        assert outcome[key] == base[key]


def test_weekend_reopen_bar_open_is_anchor():
    # 金曜 2026-10-23 21:00 のシグナル → 次足は月曜 2026-10-26 00:00。
    friday = pd.Timestamp("2026-10-23 21:00", tz=TZ)
    monday = pd.date_range(
        start=pd.Timestamp("2026-10-26 00:00", tz=TZ), periods=8, freq="1h"
    )
    bars = [(150.050, 150.060, 150.040, 150.050)]
    bars += [(150.050, 150.160, 150.040, 150.100)]
    bars += [quiet_bar() for _ in range(6)]
    frame = make_frame_at(monday, bars)

    prediction = make_prediction(signal_bar_time=friday.isoformat())
    outcome = scanner.evaluate_forward_prediction(prediction, frame)

    assert outcome is not None
    assert outcome["status"] == "HIT"
    assert outcome["exit_bar_offset"] == 2
    assert outcome["evaluation_reference_price"] == pytest.approx(150.050)
    assert outcome["entry_bar_time"] == monday[0].isoformat()


def test_disallowed_gap_stays_pending():
    # 平日に3時間の欠けがある場合は評価しない。
    signal = pd.Timestamp("2026-10-28 10:00", tz=TZ)  # 水曜
    index = pd.date_range(
        start=signal + pd.Timedelta(hours=3), periods=8, freq="1h", tz=TZ
    )
    frame = make_frame_at(index, [quiet_bar() for _ in range(8)])

    prediction = make_prediction(signal_bar_time=signal.isoformat())
    assert scanner.evaluate_forward_prediction(prediction, frame) is None


def test_invalid_next_open_stays_pending():
    bars = [(float("nan"), 150.060, 150.040, 150.050)]
    bars += [quiet_bar() for _ in range(7)]
    frame = make_frame(SIGNAL, bars)
    assert scanner.evaluate_forward_prediction(make_prediction(), frame) is None

    bars = [(0.0, 150.060, 150.040, 150.050)]
    bars += [quiet_bar() for _ in range(7)]
    frame = make_frame(SIGNAL, bars)
    assert scanner.evaluate_forward_prediction(make_prediction(), frame) is None


def test_current_record_format_and_stats_flow():
    # 現在の記録形式（record_forward_predictions が保存する形）のまま評価でき、
    # 統計へ従来どおり反映されること。reference_price は上書きされないこと。
    prediction = {
        "id": "test-id",
        "recorded_at_jst": "2026-10-26T11:05:00+09:00",
        "name": "米ドル/円",
        "ticker": "USDJPY=X",
        "direction": "買い",
        "direction_sign": 1,
        "reference_price": 150.000,
        "atr": 0.100,
        "signal_bar_time": SIGNAL.isoformat(),
        "pip_size": 0.01,
        "spread_pips": 0.3,
        "status": "PENDING",
        "news_status_at_record": None,
        "news_verified": False,
        "model_version": scanner.MODEL_VERSION,
        "feature_snapshot": None,
    }
    bars = [(149.950, 150.060, 149.940, 150.020)]
    bars += [quiet_bar(149.950) for _ in range(7)]
    frame = make_frame(SIGNAL, bars)

    outcome = scanner.evaluate_forward_prediction(prediction, frame)
    assert outcome is not None
    assert outcome["status"] == "HIT"

    prediction.update(outcome)
    assert prediction["reference_price"] == 150.000
    assert prediction["evaluation_reference_price"] == pytest.approx(149.950)

    stats = scanner.normalize_forward_stats(None)
    scanner.update_forward_stats(stats, prediction, outcome)
    assert stats["evaluated_total"] == 1
    assert stats["hit"] == 1
    assert stats["buy_evaluated"] == 1
    expected_net = 10.0 - (
        0.3
        + scanner.SIMULATED_ROUND_TRIP_SLIPPAGE_PIPS
        + scanner.SIMULATED_SWAP_STRESS_COST_PIPS
    )
    assert stats["net_pips_total"] == pytest.approx(expected_net)


def label_direction_for_window(bars, entry_open, atr):
    """学習ラベルの規則（次足始値±1ATR、次足を含む8本、同足両到達は中立）を
    そのまま写した参照実装。フォワード評価の方向判定との一致確認用。"""
    upper = entry_open + atr
    lower = entry_open - atr
    for _, high, low, _ in bars[: scanner.LOOKAHEAD_BARS]:
        hit_up = high >= upper
        hit_down = low <= lower
        if hit_up and hit_down:
            return 0
        if hit_up:
            return 1
        if hit_down:
            return -1
    return 0


def test_direction_matches_label_rule():
    # 同足両到達を含まないケースで、ラベルの方向とフォワード評価の
    # HIT/MISS/NOT_REACHED が対応すること。
    cases = [
        # (bars, ラベルの方向, 買い予測の結果, 売り予測の結果)
        (
            [(150.050, 150.160, 150.040, 150.100)] + [quiet_bar()] * 7,
            1,
            "HIT",
            "MISS",
        ),
        (
            [quiet_bar()] * 3
            + [(150.050, 150.060, 149.940, 150.000)]
            + [quiet_bar()] * 4,
            -1,
            "MISS",
            "HIT",
        ),
        ([quiet_bar()] * 8, 0, "NOT_REACHED", "NOT_REACHED"),
    ]
    for bars, label, buy_result, sell_result in cases:
        frame = make_frame(SIGNAL, bars)
        assert label_direction_for_window(bars, 150.050, 0.100) == label
        buy = scanner.evaluate_forward_prediction(make_prediction(), frame)
        sell = scanner.evaluate_forward_prediction(
            make_prediction(direction="売り"), frame
        )
        assert buy["status"] == buy_result
        assert sell["status"] == sell_result
