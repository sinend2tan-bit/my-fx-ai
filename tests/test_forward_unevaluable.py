"""PENDING予測の評価不能（UNEVALUABLE）分類に関するテスト。

第2段階の変更（2026-10-31）の検証用。ネットワークは使わない。

確認すること:
- 記録から13日では分類されず、14日経過で分類されること（境界）。
- 理由コードごとに、模擬した条件で期待する code が返ること
  （invalid_record、insufficient_bars、bar_gap、invalid_values、
  unknown_timeout）。
- 14日経過していても、その実行で評価できる予測は分類されず
  通常に評価されること。
- 分類時に元のフィールドが不変であること
  （status と2つの新規フィールド以外）。
- UNEVALUABLE が pending_count に入らず、累積カウンタにも加算されず、
  unevaluable_count に反映されること。
- UNEVALUABLE が trim で保護されること。
- 新規フィールドの無い旧形式の記録を読み込んでも壊れないこと。
- フレーム取得自体が失敗した実行では分類されないこと。
"""

from datetime import timedelta

import pandas as pd
import pytest

import scanner

TZ = "Asia/Tokyo"
# 月曜 2026-10-26 10:00 JST をシグナル足とする（平日の連続した1時間足）。
SIGNAL = pd.Timestamp("2026-10-26 10:00", tz=TZ)


def make_frame(signal_time, bars):
    """実運用のフレームと同じ形（シグナル足を含む）を作る。

    bars はシグナル足の後に続く足のリスト。評価ウィンドウは
    シグナル足より後の足だけなので、シグナル足を先頭に入れても
    評価・診断の結果は変わらない。実運用では bars() が180日分を
    返すため、シグナル足は常にフレーム内に含まれる。
    """
    index = [signal_time] + list(pd.date_range(
        start=signal_time + pd.Timedelta(hours=1),
        periods=len(bars),
        freq="1h",
        tz=TZ,
    ))
    return pd.DataFrame(
        [quiet_bar()] + list(bars), index=index,
        columns=["Open", "High", "Low", "Close"],
    ).astype(float)


def make_frame_at(timestamps, bars):
    index = pd.DatetimeIndex(list(timestamps)).tz_convert(TZ)
    return pd.DataFrame(
        bars, index=index, columns=["Open", "High", "Low", "Close"]
    ).astype(float)


def gap_frame(signal_time):
    """シグナル足の直後が3時間空いたフレーム（平日の欠け）。

    シグナル足自体を先頭に含め、実運用のフレームと同じ形にする。
    """
    index = [signal_time] + list(pd.date_range(
        start=signal_time + pd.Timedelta(hours=3), periods=8, freq="1h",
        tz=TZ,
    ))
    return pd.DataFrame(
        [quiet_bar()] + [quiet_bar() for _ in range(8)], index=index,
        columns=["Open", "High", "Low", "Close"],
    ).astype(float)

def quiet_bar(open_price=150.050, close=150.050):
    """バリア（次足始値±0.100）に触れない穏やかな足。"""
    return (open_price, open_price + 0.010, open_price - 0.010, close)


def make_prediction(**overrides):
    """現在時刻から見て新しい（分類対象外の）PENDING記録を作る。"""
    prediction = {
        "id": "test-unevaluable",
        "recorded_at_jst": scanner.iso_now(),
        "ticker": "USDJPY=X",
        "direction": "買い",
        "reference_price": 150.000,
        "atr": 0.100,
        "pip_size": 0.01,
        "spread_pips": 0.0,
        "signal_bar_time": SIGNAL.isoformat(),
        "status": "PENDING",
        "model_version": scanner.MODEL_VERSION,
        "feature_snapshot": {"close": 150.000},
    }
    prediction.update(overrides)
    return prediction


def aged_prediction(days, **overrides):
    """記録日時を days 日前にした PENDING 記録を作る。"""
    recorded = (scanner.now_jst() - timedelta(days=days)).isoformat()
    return make_prediction(recorded_at_jst=recorded, **overrides)


def frame_with_attrs(ohlc):
    """bars() の戻り値の形（attrs に latest_features を持つ）を作る。"""
    outer = ohlc.copy()
    outer.attrs["latest_features"] = ohlc
    return outer


def run_process(monkeypatch, prediction, ohlc):
    """bars() を差し替えて process_forward_verification を1回回す。"""
    monkeypatch.setattr(
        scanner, "bars",
        lambda ticker, **kwargs: frame_with_attrs(ohlc),
    )
    previous_state = {
        "predictions": [prediction],
        "forward_stats": scanner.new_forward_stats(),
    }
    predictions, stats, evaluated_now = (
        scanner.process_forward_verification(previous_state)
    )
    return predictions[0], stats, evaluated_now


# =========================================================
# 境界：13日では分類されず、14日経過で分類される
# =========================================================

def test_13_days_not_classified(monkeypatch):
    # 平日に3時間の欠けがあるフレーム（評価できない条件）でも、
    # 13日では PENDING のまま。
    signal = pd.Timestamp("2026-10-28 10:00", tz=TZ)  # 水曜
    frame = gap_frame(signal)
    prediction = aged_prediction(
        13, signal_bar_time=signal.isoformat(), id="gap-13d",
    )
    result, stats, evaluated_now = run_process(monkeypatch, prediction, frame)
    assert result["status"] == "PENDING"
    assert "unevaluable_reason" not in result
    assert evaluated_now == 0
    assert stats["evaluated_total"] == 0


def test_14_days_classified(monkeypatch):
    signal = pd.Timestamp("2026-10-28 10:00", tz=TZ)  # 水曜
    frame = gap_frame(signal)
    prediction = aged_prediction(
        14, signal_bar_time=signal.isoformat(), id="gap-14d",
    )
    result, stats, evaluated_now = run_process(monkeypatch, prediction, frame)
    assert result["status"] == "UNEVALUABLE"
    assert result["unevaluable_reason"]["code"] == "bar_gap"
    assert result["status_changed_at_jst"]
    assert evaluated_now == 0
    # 統計の累積カウンタには入らない。
    assert stats["evaluated_total"] == 0


def test_13_days_23_hours_not_classified(monkeypatch):
    # 猶予は「14日×24時間の経過」で数える。13日23時間ではまだ分類しない。
    signal = pd.Timestamp("2026-10-28 10:00", tz=TZ)  # 水曜
    frame = gap_frame(signal)
    recorded = (
        scanner.now_jst() - timedelta(days=13, hours=23)
    ).isoformat()
    prediction = make_prediction(
        recorded_at_jst=recorded, signal_bar_time=signal.isoformat(),
        id="gap-13d23h",
    )
    result, _, _ = run_process(monkeypatch, prediction, frame)
    assert result["status"] == "PENDING"


def test_recorded_at_missing_falls_back_to_signal_time(monkeypatch):
    # recorded_at_jst が無い記録は signal_bar_time から数える。
    # シグナル足自体が現在から15日以上前なら分類される。
    signal = scanner.now_jst() - timedelta(days=15)
    signal = signal.replace(minute=0, second=0, microsecond=0)
    # pandas の tz 表現に統一する（ZoneInfo のまま混ぜると
    # DatetimeIndex 構築時に tz 不一致で失敗するため）。
    signal = pd.Timestamp(signal).tz_convert(TZ)
    frame = gap_frame(signal)
    prediction = make_prediction(
        signal_bar_time=signal.isoformat(), id="no-recorded-at",
    )
    prediction.pop("recorded_at_jst")
    result, _, _ = run_process(monkeypatch, prediction, frame)
    assert result["status"] == "UNEVALUABLE"
    assert result["unevaluable_reason"]["code"] == "bar_gap"


# =========================================================
# 理由コードの切り分け（diagnose_forward_pending の直接検査）
# =========================================================

def test_diagnose_invalid_record():
    frame = make_frame(SIGNAL, [quiet_bar() for _ in range(8)])
    assert scanner.diagnose_forward_pending(
        make_prediction(direction="不明"), frame
    )[0] == "invalid_record"
    assert scanner.diagnose_forward_pending(
        make_prediction(reference_price=0.0), frame
    )[0] == "invalid_record"
    assert scanner.diagnose_forward_pending(
        make_prediction(atr=None), frame
    )[0] == "invalid_record"
    assert scanner.diagnose_forward_pending(
        make_prediction(pip_size=0.0), frame
    )[0] == "invalid_record"
    assert scanner.diagnose_forward_pending(
        make_prediction(signal_bar_time="not-a-time"), frame
    )[0] == "invalid_record"
    # タイムゾーン無しの信号足時刻も不正。
    assert scanner.diagnose_forward_pending(
        make_prediction(signal_bar_time="2026-10-26 10:00"), frame
    )[0] == "invalid_record"


def test_diagnose_insufficient_bars():
    # シグナル後の足が5本しかない。
    frame = make_frame(SIGNAL, [quiet_bar() for _ in range(5)])
    code, detail = scanner.diagnose_forward_pending(make_prediction(), frame)
    assert code == "insufficient_bars"
    assert "5" in detail


def test_diagnose_insufficient_bars_signal_outside_frame():
    # シグナル足が取得フレームより前（200時間前の信号に8本の足が
    # あっても、シグナルと最初の足の間が大きく空いている）。
    signal = pd.Timestamp("2026-10-26 10:00", tz=TZ)
    index = pd.date_range(
        start=signal + pd.Timedelta(hours=200), periods=8, freq="1h", tz=TZ
    )
    frame = make_frame_at(index, [quiet_bar() for _ in range(8)])
    code, _ = scanner.diagnose_forward_pending(
        make_prediction(signal_bar_time=signal.isoformat()), frame
    )
    assert code == "insufficient_bars"


def test_diagnose_bar_gap():
    signal = pd.Timestamp("2026-10-28 10:00", tz=TZ)  # 水曜
    frame = gap_frame(signal)
    code, detail = scanner.diagnose_forward_pending(
        make_prediction(signal_bar_time=signal.isoformat()), frame
    )
    assert code == "bar_gap"
    assert "2026-10-28" in detail


def test_diagnose_invalid_values():
    # ウィンドウ内の高値が NaN。
    bars = [quiet_bar() for _ in range(8)]
    bars[3] = (150.050, float("nan"), 150.040, 150.050)
    frame = make_frame(SIGNAL, bars)
    code, _ = scanner.diagnose_forward_pending(make_prediction(), frame)
    assert code == "invalid_values"

    # 評価ウィンドウ1本目の始値が NaN。
    bars = [(float("nan"), 150.060, 150.040, 150.050)]
    bars += [quiet_bar() for _ in range(7)]
    frame = make_frame(SIGNAL, bars)
    code, _ = scanner.diagnose_forward_pending(make_prediction(), frame)
    assert code == "invalid_values"


def test_diagnose_unknown_timeout_when_no_cause_found():
    # 全ての前提条件が整っているフレームでは、原因は特定できない
    # （評価自体は成立するため、実運用でここに到達するのは
    # evaluate と diagnose の前提が将来ずれた場合に限る）。
    frame = make_frame(SIGNAL, [quiet_bar() for _ in range(8)])
    code, _ = scanner.diagnose_forward_pending(make_prediction(), frame)
    assert code == "unknown_timeout"


# =========================================================
# 評価できる予測は分類されず通常に評価される
# =========================================================

def test_evaluable_prediction_not_classified(monkeypatch):
    # 20日経過していても、評価できるフレームなら通常に評価される。
    bars = [(149.950, 150.060, 149.940, 150.020)]
    bars += [quiet_bar(149.950) for _ in range(7)]
    frame = make_frame(SIGNAL, bars)
    prediction = aged_prediction(20, id="evaluable-20d")
    result, stats, evaluated_now = run_process(monkeypatch, prediction, frame)
    assert result["status"] == "HIT"
    assert "unevaluable_reason" not in result
    assert evaluated_now == 1
    assert stats["evaluated_total"] == 1
    assert stats["hit"] == 1


# =========================================================
# フレーム取得失敗では分類しない
# =========================================================

def test_no_classification_when_frame_fetch_fails(monkeypatch):
    prediction = aged_prediction(20, id="fetch-fail-20d")
    monkeypatch.setattr(scanner, "bars", lambda ticker, **kwargs: None)
    previous_state = {
        "predictions": [prediction],
        "forward_stats": scanner.new_forward_stats(),
    }
    predictions, _, _ = scanner.process_forward_verification(previous_state)
    assert predictions[0]["status"] == "PENDING"
    assert "unevaluable_reason" not in predictions[0]


def test_no_classification_when_ohlc_empty(monkeypatch):
    prediction = aged_prediction(20, id="empty-ohlc-20d")
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    outer = empty.copy()
    outer.attrs["latest_features"] = empty
    monkeypatch.setattr(
        scanner, "bars", lambda ticker, **kwargs: outer,
    )
    previous_state = {
        "predictions": [prediction],
        "forward_stats": scanner.new_forward_stats(),
    }
    predictions, _, _ = scanner.process_forward_verification(previous_state)
    assert predictions[0]["status"] == "PENDING"
    assert "unevaluable_reason" not in predictions[0]


# =========================================================
# 分類時の不変性・集計・切り詰め・旧形式
# =========================================================

def test_classification_preserves_original_fields(monkeypatch):
    signal = pd.Timestamp("2026-10-28 10:00", tz=TZ)  # 水曜
    frame = gap_frame(signal)
    prediction = aged_prediction(
        15, signal_bar_time=signal.isoformat(), id="preserve-15d",
    )
    before = dict(prediction)
    result, _, _ = run_process(monkeypatch, prediction, frame)
    assert result["status"] == "UNEVALUABLE"
    for key, value in before.items():
        if key == "status":
            continue
        assert result[key] == value, key
    # 新規フィールドは status_changed_at_jst と unevaluable_reason のみ。
    assert set(result) - set(before) == {
        "status_changed_at_jst", "unevaluable_reason",
    }
    # 元の保存値が残っていること。
    assert result["reference_price"] == 150.000
    assert result["feature_snapshot"] == {"close": 150.000}
    assert result["model_version"] == scanner.MODEL_VERSION


def test_summary_counts_and_stats_exclusion():
    stats = scanner.new_forward_stats()
    predictions = [
        {"id": "p1", "status": "PENDING"},
        {"id": "u1", "status": "UNEVALUABLE",
         "unevaluable_reason": {"code": "bar_gap", "detail": "x"}},
        {"id": "u2", "status": "UNEVALUABLE",
         "unevaluable_reason": {"code": "invalid_record", "detail": "y"}},
        {"id": "h1", "status": "HIT"},
    ]
    view = scanner.build_forward_summary(
        stats, predictions, evaluated_now=0, recorded_now=0,
        unevaluable_now=1,
    )
    assert view["pending_count"] == 1
    assert view["unevaluable_count"] == 2
    assert view["unevaluable_this_run"] == 1
    assert view["stored_predictions"] == 4
    # UNEVALUABLE は累積カウンタに入らない（stats は空のまま）。
    assert view["evaluated_total"] == 0
    assert view["hit"] == 0


def test_trim_protects_unevaluable():
    mixed = [
        {"id": f"eval{i}", "status": "HIT"} for i in range(4)
    ]
    mixed.append({"id": "pend", "status": "PENDING"})
    mixed.append({"id": "unev", "status": "UNEVALUABLE"})
    kept = scanner.trim_forward_predictions(mixed, max_items=5)
    ids = [item["id"] for item in kept]
    assert "pend" in ids
    assert "unev" in ids
    assert len(kept) == 5


def test_old_format_record_without_new_fields(monkeypatch):
    # 新規フィールドの無い旧形式の記録でも壊れず、14日経過で分類され、
    # 分類時に2つの新規フィールドが付く。
    signal = pd.Timestamp("2026-10-28 10:00", tz=TZ)  # 水曜
    frame = gap_frame(signal)
    prediction = aged_prediction(
        15, signal_bar_time=signal.isoformat(), id="old-format-15d",
    )
    prediction.pop("feature_snapshot", None)
    result, _, _ = run_process(monkeypatch, prediction, frame)
    assert result["status"] == "UNEVALUABLE"
    assert result["unevaluable_reason"]["code"] == "bar_gap"
    assert result["status_changed_at_jst"]

    # 若い旧形式記録は PENDING のまま集計に入る。
    view = scanner.build_forward_summary(
        scanner.new_forward_stats(),
        [{"id": "old-young", "status": "PENDING"}],
        0, 0,
    )
    assert view["pending_count"] == 1
    assert view["unevaluable_count"] == 0
