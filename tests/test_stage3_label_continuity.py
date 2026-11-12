"""段階3（B-2／SYN-FX-018）の学習ラベル連続性判定のテスト。

承認済み仕様書「段階3-B2-仕様精査とテスト計画-最終版3-2026-11-12.md」
（SHA-256: 5762ca0428dfe37e6af5a215841d72d682824e08be906860479bbf874828dca1）
のテスト計画 T-L / T-P / T-F / T-D / T-B / T-A / T-V を実装する。
実データ・実ネットワークは使わない（yfinance はスタブに差し替え、
ソケット接続はガードで禁止する）。

対象は Scanner の1時間足の学習ラベル生成のみ:
- 窓（シグナル行＋未来8本）の連続性はフォワード評価と同じ正本
  _forward_bar_gap_allowed で判定する（正の時刻差の必須条件を含む）。
- 窓が無効な行は Target と Future_* をすべて NaN にして除外し、
  dropped_label_discontinuous に理由別（優先順位
  insufficient_future → discontinuous → other）で計上する。
- 監査フレームは training_df.attrs["label_audit"] に保持し、特徴量・
  予測記録・永続化データ・公開ステータスには混入させない。
- Scanner MODEL_VERSION は 21、App は 8 のまま。

注記:
- T-L10 は仕様書どおり T-D 系列へ移行したため、ここには置かない。
- T-L12 と T-A5 の "other" 行は、実経路（bars()）だけでは構成できない。
  入力検証で価格 > 0 が保証され、ATR は EWM で正の値を保つため、
  全特徴量が有限な行では基準価格・ATR が無効になることはなく、
  段階1の記録どおり other は通常 0 である。そのため、この2件に限り
  scanner.add_indicators の戻り値の ATR を対象行だけ 0 に差し替える
  注入を用いる（分類・計上・監査のコード自体は本物の番経路を通す）。
  注入なしで到達できる範囲の検証は、他のテストが実経路で行う。
- U-3（実データでの無効化頻度）・U-4（実データの重複有無）は
  未解決事項のままであり、このテストでは一切扱わない。
"""

import ast
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

import scanner

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app.py"
PAIR_NAME = "米ドル/円 (USD/JPY)"
TICKER = "USDJPY=X"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """実ネットワーク接続を禁止する（スタブ以外の取得経路を塞ぐ）。"""

    def guard(*args, **kwargs):
        raise AssertionError("real network access attempted during stage3 tests")

    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)


# ---------------------------------------------------------------------------
# 共通フィクスチャ
# ---------------------------------------------------------------------------

def stub_download(monkeypatch, frame):
    monkeypatch.setattr(yf, "download", lambda *a, **k: frame.copy())


def path_frame(index, calm_range=None):
    """決定的な価格パス（乱数なし）を index に載せた合成 OHLCV。

    calm_range=(開始位置, 終了位置) を指定すると、その範囲は値動きを
    ほぼ止めた「凪」区間になる（バリア到達パターンの合成用）。
    """
    n = len(index)
    i = np.arange(n, dtype=float)
    close = 150.0 + 0.2 * np.sin(i / 9.0) + 0.001 * i
    if calm_range is not None:
        start, end = calm_range
        base = close[start]
        close[start:end] = base + 0.0005 * np.arange(end - start)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.01
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=index,
    )


def continuous_index(n=340, start="2026-01-05 00:00"):
    return pd.date_range(start=start, periods=n, freq="1h", tz="UTC")


def fx_week_index(n=340, start="2026-01-05 00:00"):
    """土日を除いた平日のみの1時間足インデックス（市場休場の再現）。"""
    full = pd.date_range(start=start, periods=n * 2, freq="1h", tz="UTC")
    weekdays = full[full.weekday < 5]
    return weekdays[:n]


def run_bars(monkeypatch, frame):
    stub_download(monkeypatch, frame)
    out = scanner.bars(TICKER, interval="1h", period="180d")
    assert out is not None
    return out


def audit_of(df):
    return df.attrs["label_audit"]


def quality_of(df):
    return df.attrs["data_quality"]


def discontinuous_timestamps(df):
    audit = audit_of(df)
    return list(audit.index[audit["invalid_reason"] == "discontinuous"])


def legacy_training_df(raw):
    """段階3より前のラベル処理（位置ベース）の再実装（等価性比較用）。

    入力段階（鮮度除外まで）を bars() とそろえ、特徴量は無変更の
    scanner.add_indicators を使う。ラベルループと Future_* の構築は
    変更前のコードどおり（連続性判定なし）。
    """
    frame = raw.copy()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame[frame.index < frame.index.max()].copy()
    feature_df = scanner.add_indicators(frame)
    feature_df = feature_df.dropna(
        subset=scanner.FEATURE_COLUMNS + ["ATR", "Close"]
    )
    close_values = feature_df["Close"].to_numpy(dtype=float)
    open_values = feature_df["Open"].to_numpy(dtype=float)
    high_values = feature_df["High"].to_numpy(dtype=float)
    low_values = feature_df["Low"].to_numpy(dtype=float)
    atr_values = feature_df["ATR"].to_numpy(dtype=float)
    targets = np.full(len(feature_df), np.nan)
    for i in range(len(feature_df) - scanner.LOOKAHEAD_BARS):
        if i + 1 >= len(feature_df):
            continue
        price = open_values[i + 1]
        atr_value = atr_values[i]
        if (
            not np.isfinite(price)
            or not np.isfinite(atr_value)
            or price <= 0
            or atr_value <= 0
        ):
            continue
        upper = price + atr_value
        lower = price - atr_value
        label = 0
        for j in range(i + 1, i + scanner.LOOKAHEAD_BARS + 1):
            hit_up = high_values[j] >= upper
            hit_down = low_values[j] <= lower
            if hit_up and hit_down:
                label = 0
                break
            if hit_up:
                label = 1
                break
            if hit_down:
                label = -1
                break
        targets[i] = label
    training_df = feature_df.copy()
    training_df["Target"] = targets
    training_df["Future_Open_1"] = feature_df["Open"].shift(-1)
    training_df["Future_Close"] = feature_df["Close"].shift(
        -scanner.LOOKAHEAD_BARS
    )
    future_path_columns = []
    for step in range(1, scanner.LOOKAHEAD_BARS + 1):
        training_df[f"Future_High_{step}"] = feature_df["High"].shift(-step)
        training_df[f"Future_Low_{step}"] = feature_df["Low"].shift(-step)
        future_path_columns.extend(
            [f"Future_High_{step}", f"Future_Low_{step}"]
        )
    training_df = training_df.dropna(
        subset=scanner.FEATURE_COLUMNS
        + [
            "Target",
            "ATR",
            "Close",
            "Future_Open_1",
            "Future_Close",
            *future_path_columns,
        ]
    )
    training_df["Target"] = training_df["Target"].astype(int)
    return training_df


def gap_context(df, gap_ts):
    """欠損時刻 gap_ts に対する feature_df 上の位置関係を返す。

    戻り値: (prev_ts, next_ts, expected_discontinuous)
    - prev_ts: gap 直前の feature 行
    - next_ts: gap 直後の feature 行
    - expected_discontinuous: 窓に欠損ペアを含むシグナル行の時刻一覧
      （feature 位置で prev の7行前から prev まで）
    """
    feature_index = df.attrs["latest_features"].index
    prev_pos = feature_index.get_indexer([gap_ts], method="ffill")[0]
    prev_ts = feature_index[prev_pos]
    next_ts = feature_index[prev_pos + 1]
    expected = list(feature_index[prev_pos - 7 : prev_pos + 1])
    return prev_ts, next_ts, expected


# ---------------------------------------------------------------------------
# T-L: 学習ラベルの連続性判定
# ---------------------------------------------------------------------------

def test_tl1_clean_frame_unchanged(monkeypatch):
    """T-L1: 連続データでは変更前と完全に一致する。"""
    raw = path_frame(continuous_index(340))
    out = run_bars(monkeypatch, raw)
    legacy = legacy_training_df(raw)
    out_cmp = out.copy()
    out_cmp.attrs = {}
    pd.testing.assert_frame_equal(out_cmp, legacy)
    dq = quality_of(out)
    assert dq["dropped_label_discontinuous"] == 0
    assert dq["dropped_label_other"] == 0
    assert dq["dropped_label_insufficient_future"] == scanner.LOOKAHEAD_BARS
    assert dq["trainable_rows"] == len(legacy)


def test_tl2_provider_gap_invalidates_eight_signals(monkeypatch):
    """T-L2: 平日の欠損1本で、窓に欠損を含む8シグナル行が無効になる。"""
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    gapped = raw.drop(index=[gap_ts])
    out = run_bars(monkeypatch, gapped)
    prev_ts, next_ts, expected = gap_context(out, gap_ts)
    assert discontinuous_timestamps(out) == expected
    assert quality_of(out)["dropped_label_discontinuous"] == 8
    for ts in expected:
        assert ts not in out.index
    # 変更前は欠損直前の行にラベルが付いていた（詰めて8本を数えていた）。
    legacy = legacy_training_df(gapped)
    assert prev_ts in legacy.index
    assert prev_ts not in out.index


def test_tl3_feature_nan_drop_position_range(monkeypatch):
    """T-L3: 特徴量NaNで落ちた行が作る欠損も、位置どおりに無効化する。"""
    clean_raw = path_frame(continuous_index(340))
    clean = run_bars(monkeypatch, clean_raw)
    clean_feature_drops = quality_of(clean)["dropped_feature_nan"]

    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    pos = raw.index.get_loc(gap_ts)
    flat_price = float(raw["Close"].iloc[pos])
    raw.iloc[pos] = [flat_price, flat_price, flat_price, flat_price, 1_000_000.0]
    out = run_bars(monkeypatch, raw)
    # 平坦足（High == Low）はローソク足の上下ヒゲ比率が NaN になり、
    # 特徴量段階で落ちて feature_df に2時間の欠損を作る。
    assert quality_of(out)["dropped_feature_nan"] == clean_feature_drops + 1
    assert gap_ts not in out.attrs["latest_features"].index
    prev_ts, next_ts, expected = gap_context(out, gap_ts)
    assert discontinuous_timestamps(out) == expected
    assert quality_of(out)["dropped_label_discontinuous"] == 8
    # 欠損行より8行以上前のシグナルと、欠損直後のシグナルは影響を受けない。
    feature_index = out.attrs["latest_features"].index
    unaffected_before = feature_index[feature_index.get_loc(prev_ts) - 8]
    assert unaffected_before in out.index
    assert next_ts in out.index


def test_tl4_signal_to_first_bar_gap(monkeypatch):
    """T-L4: シグナル→1本目のペアで不連続なら、その行は無効になる。"""
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    out = run_bars(monkeypatch, raw.drop(index=[gap_ts]))
    prev_ts, next_ts, _ = gap_context(out, gap_ts)
    audit = audit_of(out)
    row = audit.loc[prev_ts]
    assert row["invalid_reason"] == "discontinuous"
    assert row["first_bad_pair_prev"] == prev_ts
    assert row["first_bad_pair_next"] == next_ts


def test_tl5_gap_outside_window_no_effect(monkeypatch):
    """T-L5: 欠損が窓の外にある行は無効化されない。"""
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    out = run_bars(monkeypatch, raw.drop(index=[gap_ts]))
    prev_ts, next_ts, expected = gap_context(out, gap_ts)
    feature_index = out.attrs["latest_features"].index
    before_window = feature_index[feature_index.get_loc(prev_ts) - 8]
    assert before_window in out.index
    assert audit_of(out).loc[before_window, "invalid_reason"] == ""
    assert next_ts in out.index


def test_tl6_weekend_window_valid(monkeypatch):
    """T-L6: 金曜→月曜の週末ジャンプを含む窓は有効。"""
    raw = path_frame(fx_week_index(340))
    out = run_bars(monkeypatch, raw)
    assert quality_of(out)["dropped_label_discontinuous"] == 0
    friday_evening = pd.Timestamp("2026-01-09 20:00", tz="UTC")
    assert friday_evening in out.index


def test_tl7_eighty_hour_boundary(monkeypatch):
    """T-L7: 80時間ちょうどのジャンプは有効、超えると無効。"""
    head_idx = pd.date_range(
        end="2026-01-02 00:00", periods=300, freq="1h", tz="UTC"
    )
    for resume, expect_valid in (
        ("2026-01-05 08:00", True),   # 金曜00:00 → 月曜08:00 = ちょうど80時間
        ("2026-01-05 09:00", False),  # 81時間
    ):
        tail_idx = pd.date_range(
            start=resume, periods=60, freq="1h", tz="UTC"
        )
        raw = path_frame(head_idx.append(tail_idx))
        out = run_bars(monkeypatch, raw)
        last_head = head_idx[-1]
        if expect_valid:
            assert last_head in out.index
            assert last_head not in discontinuous_timestamps(out)
        else:
            assert last_head in discontinuous_timestamps(out)


def test_tl8_friday_to_tuesday_invalid(monkeypatch):
    """T-L8: 月曜休場の金曜→火曜は、80時間以内でも無効。"""
    head_idx = pd.date_range(
        end="2026-01-09 23:00", periods=300, freq="1h", tz="UTC"
    )
    tail_idx = pd.date_range(
        start="2026-01-13 00:00", periods=60, freq="1h", tz="UTC"
    )
    raw = path_frame(head_idx.append(tail_idx))
    out = run_bars(monkeypatch, raw)
    assert head_idx[-1] in discontinuous_timestamps(out)


def test_tl9_saturday_to_monday_valid(monkeypatch):
    """T-L9: 土曜→月曜のジャンプを含む窓は有効。"""
    head_idx = pd.date_range(
        end="2026-01-03 12:00", periods=300, freq="1h", tz="UTC"
    )
    tail_idx = pd.date_range(
        start="2026-01-05 00:00", periods=60, freq="1h", tz="UTC"
    )
    raw = path_frame(head_idx.append(tail_idx))
    out = run_bars(monkeypatch, raw)
    assert head_idx[-1] in out.index
    assert quality_of(out)["dropped_label_discontinuous"] == 0


def test_tl11_tail_rows_are_insufficient_future(monkeypatch):
    """T-L11: 系列末尾の8行は insufficient_future で、discontinuous に混ぜない。"""
    out = run_bars(monkeypatch, path_frame(continuous_index(340)))
    audit = audit_of(out)
    feature_index = out.attrs["latest_features"].index
    tail = feature_index[-scanner.LOOKAHEAD_BARS :]
    for ts in tail:
        row = audit.loc[ts]
        assert row["invalid_reason"] == "insufficient_future"
        assert pd.isna(row["continuity_valid"])
        assert not row["label_trainable"]
    assert quality_of(out)["dropped_label_insufficient_future"] == 8
    assert not set(tail) & set(discontinuous_timestamps(out))


def _zero_atr_at(monkeypatch, target_ts):
    """add_indicators の戻り値の ATR を対象行だけ 0 に差し替える注入。

    "other" 分類は実経路の OHLCV からは到達できない（モジュール冒頭の
    注記参照）。分類順序そのものを検証するための最小限の注入であり、
    ラベル処理・計上・監査のコードはすべて実経路を通す。
    """
    original = scanner.add_indicators

    def wrapper(frame):
        out = original(frame)
        if target_ts in out.index:
            out.loc[target_ts, "ATR"] = 0.0
        return out

    monkeypatch.setattr(scanner, "add_indicators", wrapper)


def test_tl12_overlap_counted_once_as_discontinuous(monkeypatch):
    """T-L12: 窓不連続かつ ATR 無効の行は discontinuous にのみ1回計上する。"""
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    gapped = raw.drop(index=[gap_ts])
    # 欠損直前の行（窓が不連続になる行）の ATR を無効化して重なりを作る。
    probe = run_bars(monkeypatch, gapped)
    prev_ts, _, _ = gap_context(probe, gap_ts)
    _zero_atr_at(monkeypatch, prev_ts)
    out = run_bars(monkeypatch, gapped)
    dq = quality_of(out)
    assert dq["dropped_label_discontinuous"] == 8
    assert dq["dropped_label_other"] == 0
    assert audit_of(out).loc[prev_ts, "invalid_reason"] == "discontinuous"


def test_tl13_checksums_on_mixed_fixture(monkeypatch, capsys):
    """T-L13: 混在フィクスチャで C-1 の検算式が成立し、警告が出ない。"""
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    out = run_bars(monkeypatch, raw.drop(index=[gap_ts]))
    dq = quality_of(out)
    input_drops = (
        dq["dropped_invalid_timestamp"]
        + dq["dropped_missing_or_nonfinite"]
        + dq["dropped_duplicate_timestamp"]
        + dq["dropped_invalid_ohlc"]
        + dq["dropped_feature_nan"]
    )
    assert dq["feature_rows"] == dq["input_rows"] - input_drops
    label_drops = (
        dq["dropped_label_discontinuous"]
        + dq["dropped_label_insufficient_future"]
        + dq["dropped_label_other"]
    )
    assert dq["trainable_rows"] == dq["feature_rows"] - label_drops
    assert dq["trainable_rows"] == len(out)
    assert "[DATA QUALITY WARNING]" not in capsys.readouterr().out


def test_tl14_window_edge_positions(monkeypatch):
    """T-L14: 窓の8本目が欠損直後の行に当たる行は無効、欠損直前で窓が閉じる行は有効。"""
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    pos = raw.index.get_loc(gap_ts)
    flat_price = float(raw["Close"].iloc[pos])
    raw.iloc[pos] = [flat_price, flat_price, flat_price, flat_price, 1_000_000.0]
    out = run_bars(monkeypatch, raw)
    feature_index = out.attrs["latest_features"].index
    prev_pos = feature_index.get_indexer([gap_ts], method="ffill")[0]
    eighth_is_post_gap = feature_index[prev_pos - 7]
    eighth_is_prev = feature_index[prev_pos - 8]
    audit = audit_of(out)
    assert audit.loc[eighth_is_post_gap, "invalid_reason"] == "discontinuous"
    assert audit.loc[eighth_is_prev, "invalid_reason"] == ""
    assert eighth_is_prev in out.index


def test_tl15_consecutive_gap_counted_once_per_signal(monkeypatch):
    """T-L15: 連続欠損（3本）でも、影響するシグナル行は8行のまま二重に数えない。"""
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    dropped = [gap_ts + pd.Timedelta(hours=k) for k in range(3)]
    out = run_bars(monkeypatch, raw.drop(index=dropped))
    prev_ts, next_ts, expected = gap_context(out, gap_ts)
    assert discontinuous_timestamps(out) == expected
    assert quality_of(out)["dropped_label_discontinuous"] == 8
    audit = audit_of(out)
    assert audit.loc[prev_ts, "first_bad_pair_next"] == next_ts


def test_tl16_cause_independence(monkeypatch):
    """T-L16: 欠損の原因（生データ欠損／特徴量NaN落ち）によらず判定が同一。"""
    raw_removed = path_frame(continuous_index(340))
    gap_ts = raw_removed.index[260]
    out_removed = run_bars(monkeypatch, raw_removed.drop(index=[gap_ts]))

    raw_flat = path_frame(continuous_index(340))
    pos = raw_flat.index.get_loc(gap_ts)
    flat_price = float(raw_flat["Close"].iloc[pos])
    raw_flat.iloc[pos] = [
        flat_price, flat_price, flat_price, flat_price, 1_000_000.0
    ]
    out_flat = run_bars(monkeypatch, raw_flat)

    assert discontinuous_timestamps(out_removed) == discontinuous_timestamps(
        out_flat
    )
    assert list(out_removed.index) == list(out_flat.index)
    dq_r, dq_f = quality_of(out_removed), quality_of(out_flat)
    for key in (
        "dropped_label_discontinuous",
        "dropped_label_insufficient_future",
        "dropped_label_other",
        "trainable_rows",
    ):
        assert dq_r[key] == dq_f[key]


# ---------------------------------------------------------------------------
# T-P: Future_*・仮想損益・パージ
# ---------------------------------------------------------------------------

FUTURE_COLUMNS = ["Future_Open_1", "Future_Close"] + [
    col
    for step in range(1, scanner.LOOKAHEAD_BARS + 1)
    for col in (f"Future_High_{step}", f"Future_Low_{step}")
]


def test_tp1_valid_rows_future_columns_match_legacy(monkeypatch):
    """T-P1: 有効な行の Future_* は変更前と同一、無効な行は丸ごと除外される。"""
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    gapped = raw.drop(index=[gap_ts])
    out = run_bars(monkeypatch, gapped)
    legacy = legacy_training_df(gapped)
    _, _, expected = gap_context(out, gap_ts)
    for ts in expected:
        assert ts in legacy.index
        assert ts not in out.index
    common = out.index.intersection(legacy.index)
    assert len(common) == len(out)
    pd.testing.assert_frame_equal(
        out.loc[common, FUTURE_COLUMNS],
        legacy.loc[common, FUTURE_COLUMNS],
    )


def test_tp2_no_invalid_window_rows_in_training(monkeypatch):
    """T-P2: 学習データの全行について、窓の8ペアすべてが正本で連続と判定される。"""
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    out = run_bars(monkeypatch, raw.drop(index=[gap_ts]))
    feature_index = out.attrs["latest_features"].index
    positions = {ts: i for i, ts in enumerate(feature_index)}
    for ts in out.index:
        start = positions[ts]
        chain = feature_index[start : start + scanner.LOOKAHEAD_BARS + 1]
        assert len(chain) == scanner.LOOKAHEAD_BARS + 1
        for prev_ts, next_ts in zip(chain[:-1], chain[1:]):
            assert scanner._forward_bar_gap_allowed(prev_ts, next_ts)
    assert not set(out.index) & set(discontinuous_timestamps(out))


def _purge_timestamps(df, validation, fold_no):
    """fold の (a) 最終学習行時刻 (b) その8本目の未来足時刻 (c) テスト開始時刻。"""
    fold_size = len(df) // (scanner.N_WALK_FORWARD_FOLDS + 1)
    train_end = fold_size * fold_no
    train_cutoff = max(0, train_end - scanner.LOOKAHEAD_BARS)
    last_train_ts = df.index[train_cutoff - 1]
    feature_index = df.attrs["latest_features"].index
    eighth_future_ts = feature_index[
        feature_index.get_loc(last_train_ts) + scanner.LOOKAHEAD_BARS
    ]
    detail = next(d for d in validation["fold_details"] if d["fold"] == fold_no)
    test_start_ts = pd.Timestamp(detail["test_start"])
    return last_train_ts, eighth_future_ts, test_start_ts, df.index[train_end]


def test_tp3_purge_boundary_by_timestamps(monkeypatch):
    """T-P3: パージ境界を時刻で確認する（連続データと比較用の欠損データ）。"""
    raw = path_frame(continuous_index(500))
    out = run_bars(monkeypatch, raw)
    validation = scanner.walk_forward_validation(out, pair_name=PAIR_NAME)
    assert validation["fold_details"]
    for detail in validation["fold_details"]:
        fold_no = detail["fold"]
        last_train, eighth_future, test_start, expected_start = (
            _purge_timestamps(out, validation, fold_no)
        )
        # (c) は実際にテストへ渡されたデータの最初の時刻と一致する。
        assert test_start == expected_start
        # 8本目の未来足はテスト開始より前にある（漏洩なし）。
        assert eighth_future < test_start
        # 連続データでは8本目の未来足はテスト直前の学習行そのもの。
        assert eighth_future == out.index[
            (len(out) // (scanner.N_WALK_FORWARD_FOLDS + 1)) * fold_no - 1
        ]
        assert last_train < test_start

    # 欠損で間引かれたデータでも、同じ時刻関係が保たれる。
    gap_ts = raw.index[200]
    gapped_out = run_bars(monkeypatch, raw.drop(index=[gap_ts]))
    gapped_validation = scanner.walk_forward_validation(
        gapped_out, pair_name=PAIR_NAME
    )
    assert gapped_validation["fold_details"]
    for detail in gapped_validation["fold_details"]:
        last_train, eighth_future, test_start, expected_start = (
            _purge_timestamps(gapped_out, gapped_validation, detail["fold"])
        )
        assert test_start == expected_start
        assert eighth_future < test_start
        assert audit_of(gapped_out).loc[last_train, "invalid_reason"] == ""


def test_tp4_clean_data_validation_equivalence(monkeypatch):
    """T-P4: 連続データでは時系列検証の結果が変更前と一致する。"""
    raw = path_frame(continuous_index(500))
    out = run_bars(monkeypatch, raw)
    legacy = legacy_training_df(raw)
    new_validation = scanner.walk_forward_validation(out, pair_name=PAIR_NAME)
    legacy_validation = scanner.walk_forward_validation(
        legacy, pair_name=PAIR_NAME
    )
    assert new_validation["passed"] == legacy_validation["passed"]
    assert new_validation["simulated_pnl"] == legacy_validation["simulated_pnl"]
    new_folds = {
        d["fold"]: d for d in new_validation["fold_details"]
    }
    legacy_folds = {
        d["fold"]: d for d in legacy_validation["fold_details"]
    }
    assert new_folds.keys() == legacy_folds.keys()
    for fold_no in new_folds:
        for key in (
            "test_start",
            "test_rows",
            "train_rows",
            "balanced_accuracy",
            "simulated_pnl",
        ):
            assert new_folds[fold_no][key] == legacy_folds[fold_no][key]


# ---------------------------------------------------------------------------
# T-F: フォワード評価との整合
# ---------------------------------------------------------------------------

def _forward_prediction(signal_ts, direction="買い", atr=0.10):
    return {
        "direction": direction,
        "reference_price": 150.0,
        "atr": atr,
        "pip_size": 0.01,
        "spread_pips": 0.2,
        "signal_bar_time": signal_ts.isoformat(),
    }


def _forward_frame(timestamps, window_bars):
    """シグナル足＋評価窓8本の OHLC フレーム（window_bars は (O,H,L,C) の列）。"""
    rows = [(150.0, 150.05, 149.95, 150.0)]
    rows.extend(window_bars)
    return pd.DataFrame(
        {
            "Open": [r[0] for r in rows],
            "High": [r[1] for r in rows],
            "Low": [r[2] for r in rows],
            "Close": [r[3] for r in rows],
            "Volume": [1_000_000.0] * len(rows),
        },
        index=pd.DatetimeIndex(timestamps),
    )


def _window_index(signal_ts):
    return pd.date_range(
        start=signal_ts, periods=9, freq="1h", tz="UTC"
    )


# バリア: エントリー（1本目の始値）150.00、ATR 0.10 → 上 150.10 / 下 149.90。
UPPER_ONLY_BARS = [
    (150.00, 150.05, 149.95, 150.00),
    (150.00, 150.06, 149.96, 150.01),
    (150.01, 150.12, 149.97, 150.05),
    (150.05, 150.06, 149.98, 150.02),
    (150.02, 150.05, 149.97, 150.01),
    (150.01, 150.04, 149.96, 150.00),
    (150.00, 150.05, 149.95, 150.00),
    (150.00, 150.04, 149.96, 150.01),
]
LOWER_ONLY_BARS = [
    (150.00, 150.05, 149.95, 150.00),
    (150.00, 150.04, 149.85, 149.97),
    (149.97, 150.03, 149.93, 149.99),
    (149.99, 150.04, 149.94, 150.00),
    (150.00, 150.05, 149.95, 150.01),
    (150.01, 150.05, 149.96, 150.00),
    (150.00, 150.04, 149.95, 150.00),
    (150.00, 150.05, 149.96, 150.02),
]
NO_TOUCH_BARS = [
    (150.00, 150.05, 149.95, 150.00),
    (150.00, 150.06, 149.96, 150.01),
    (150.01, 150.05, 149.97, 150.02),
    (150.02, 150.06, 149.96, 150.01),
    (150.01, 150.05, 149.95, 150.00),
    (150.00, 150.04, 149.96, 150.01),
    (150.01, 150.05, 149.95, 150.02),
    (150.02, 150.06, 149.96, 150.03),
]
BOTH_TOUCH_BARS = [
    (150.00, 150.05, 149.95, 150.00),
    (150.00, 150.12, 149.85, 149.98),
    (149.98, 150.04, 149.94, 150.00),
    (150.00, 150.05, 149.95, 150.01),
    (150.01, 150.05, 149.96, 150.00),
    (150.00, 150.04, 149.95, 150.00),
    (150.00, 150.05, 149.96, 150.01),
    (150.01, 150.05, 149.95, 150.02),
]


def test_tf1_forward_evaluation_invariant():
    """T-F1: フォワード評価の既存挙動（週末・欠損・両触れ・コスト）が変わらない。"""
    # (i) 週末をまたぐ窓は評価される。
    friday = pd.Timestamp("2026-01-02 21:00", tz="UTC")
    monday = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    weekend_index = [friday] + list(
        pd.date_range(start=monday, periods=8, freq="1h", tz="UTC")
    )
    result = scanner.evaluate_forward_prediction(
        _forward_prediction(friday),
        _forward_frame(weekend_index, NO_TOUCH_BARS),
    )
    assert result is not None
    assert result["status"] == "NOT_REACHED"

    # (ii) 平日に欠けた足がある窓は評価されず PENDING のまま（None）。
    signal = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    gapped_index = [signal] + list(
        pd.date_range(start="2026-01-05 01:00", periods=4, freq="1h", tz="UTC")
    ) + list(
        pd.date_range(start="2026-01-05 06:00", periods=4, freq="1h", tz="UTC")
    )
    assert scanner.evaluate_forward_prediction(
        _forward_prediction(signal),
        _forward_frame(gapped_index, NO_TOUCH_BARS),
    ) is None

    # (iii) 同じ足で両バリアに触れたら不利側を先として MISS。
    index = _window_index(signal)
    result = scanner.evaluate_forward_prediction(
        _forward_prediction(signal),
        _forward_frame(index, BOTH_TOUCH_BARS),
    )
    assert result["status"] == "MISS"

    # (iv) コスト前提: 純損益 = 総損益 −（スプレッド＋スリッページ＋スワップストレス）。
    result = scanner.evaluate_forward_prediction(
        _forward_prediction(signal),
        _forward_frame(index, UPPER_ONLY_BARS),
    )
    assert result["gross_pips"] == pytest.approx(10.0)
    assert result["cost_pips"] == pytest.approx(0.2 + 0.5 + 0.2)
    assert result["net_pips"] == pytest.approx(10.0 - 0.9)


def test_tf2_same_function_and_same_chain(monkeypatch):
    """T-F2: ラベル生成とフォワード評価が同じ関数・同じ時刻配列を使う。"""
    calls = []
    original = scanner._forward_bar_gap_allowed

    def wrapper(prev_ts, next_ts):
        calls.append((prev_ts, next_ts))
        return original(prev_ts, next_ts)

    monkeypatch.setattr(scanner, "_forward_bar_gap_allowed", wrapper)
    raw = path_frame(continuous_index(340))
    out = run_bars(monkeypatch, raw)
    label_calls = list(calls)
    calls.clear()
    assert label_calls  # ラベル生成が正本関数を呼んでいる

    feature_index = out.attrs["latest_features"].index
    s_pos = 200
    signal_ts = feature_index[s_pos]
    expected_chain = [
        (feature_index[s_pos + k], feature_index[s_pos + k + 1])
        for k in range(scanner.LOOKAHEAD_BARS)
    ]
    # ラベル側の呼び出し列に、同じ9時刻の連鎖が順序どおり現れる。
    found = any(
        label_calls[i : i + len(expected_chain)] == expected_chain
        for i in range(len(label_calls) - len(expected_chain) + 1)
    )
    assert found

    # フォワード側の呼び出しは、その連鎖そのもの。
    window_frame = raw.iloc[
        raw.index.get_loc(signal_ts) : raw.index.get_loc(signal_ts) + 9
    ]
    result = scanner.evaluate_forward_prediction(
        _forward_prediction(signal_ts), window_frame
    )
    assert result is not None
    assert calls == expected_chain


def _calm_pattern_raw(case):
    """凪区間にバリア到達パターンを置いた bars() 用フィクスチャ。

    戻り値: (raw, signal_ts, atr, entry)。atr は変更のない
    scanner.add_indicators で算出したシグナル行の ATR（パターン合成用）。
    """
    index = continuous_index(340)
    s = 260
    raw = path_frame(index, calm_range=(s - 40, s + 12))
    probe = raw[raw.index < raw.index.max()]
    feats = scanner.add_indicators(probe)
    atr = float(feats.loc[index[s], "ATR"])
    entry = float(raw["Open"].iloc[s + 1])
    if case == "upper":
        raw.iloc[s + 3, raw.columns.get_loc("High")] = entry + atr + 0.5
    elif case == "lower":
        raw.iloc[s + 2, raw.columns.get_loc("Low")] = entry - atr - 0.5
    elif case == "both":
        raw.iloc[s + 2, raw.columns.get_loc("High")] = entry + atr + 0.5
        raw.iloc[s + 2, raw.columns.get_loc("Low")] = entry - atr - 0.5
    return raw, index[s], atr, entry


def test_tf3_forward_cases_with_concrete_values(monkeypatch):
    """T-F3: 3ケースすべてで具体値と時刻配列の一致を検証する。"""
    signal = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    index = _window_index(signal)
    cases = [
        ("upper", UPPER_ONLY_BARS, 1, "HIT", 3, 150.10),
        ("lower", LOWER_ONLY_BARS, -1, "MISS", 2, 149.90),
        ("none", NO_TOUCH_BARS, 0, "NOT_REACHED", 8, 150.03),
    ]
    for case, bars_spec, expected_label, status, offset, exit_price in cases:
        # フォワード側: 具体的なステータス・決済足・決済価格。
        frame = _forward_frame(index, bars_spec)
        result = scanner.evaluate_forward_prediction(
            _forward_prediction(signal), frame
        )
        assert result["status"] == status
        assert result["exit_bar_offset"] == offset
        assert result["exit_price"] == pytest.approx(exit_price)
        assert result["evaluation_reference_price"] == pytest.approx(150.00)
        assert result["entry_bar_time"] == index[1].isoformat()

        # ラベル側: 同じ9時刻の窓で、対応するラベルが付く。
        raw, signal_ts, _, _ = _calm_pattern_raw(
            "upper" if case == "upper" else "lower" if case == "lower" else "none"
        )
        out = run_bars(monkeypatch, raw)
        assert out.loc[signal_ts, "Target"] == expected_label
        feature_index = out.attrs["latest_features"].index
        pos = feature_index.get_loc(signal_ts)
        raw_index = raw.index
        raw_pos = raw_index.get_loc(signal_ts)
        # ラベル側の窓の時刻配列は raw の連続9時刻と一致する（= フォワード側と同一系列）。
        assert list(feature_index[pos : pos + 9]) == list(
            raw_index[raw_pos : raw_pos + 9]
        )


# ---------------------------------------------------------------------------
# T-D: バリア到達パターン（ラベル／フォワード／仮想損益）
# ---------------------------------------------------------------------------

def test_td1_upper_only(monkeypatch):
    """T-D1: 上方のみ到達 → ラベル +1、買い HIT、売り MISS（損切り）。"""
    raw, signal_ts, _, _ = _calm_pattern_raw("upper")
    out = run_bars(monkeypatch, raw)
    assert out.loc[signal_ts, "Target"] == 1

    signal = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    frame = _forward_frame(_window_index(signal), UPPER_ONLY_BARS)
    buy = scanner.evaluate_forward_prediction(
        _forward_prediction(signal, "買い"), frame
    )
    assert buy["status"] == "HIT"
    assert buy["exit_price"] == pytest.approx(150.10)
    sell = scanner.evaluate_forward_prediction(
        _forward_prediction(signal, "売り"), frame
    )
    assert sell["status"] == "MISS"
    assert sell["exit_price"] == pytest.approx(150.10)


def test_td2_lower_only(monkeypatch):
    """T-D2: 下方のみ到達 → ラベル −1、買い MISS（損切り）、売り HIT。"""
    raw, signal_ts, _, _ = _calm_pattern_raw("lower")
    out = run_bars(monkeypatch, raw)
    assert out.loc[signal_ts, "Target"] == -1

    signal = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    frame = _forward_frame(_window_index(signal), LOWER_ONLY_BARS)
    buy = scanner.evaluate_forward_prediction(
        _forward_prediction(signal, "買い"), frame
    )
    assert buy["status"] == "MISS"
    assert buy["exit_price"] == pytest.approx(149.90)
    sell = scanner.evaluate_forward_prediction(
        _forward_prediction(signal, "売り"), frame
    )
    assert sell["status"] == "HIT"
    assert sell["exit_price"] == pytest.approx(149.90)


def test_td3_both_touch(monkeypatch):
    """T-D3: 同足両触れ → ラベル中立（学習可能）、フォワードは買い・売りとも MISS。"""
    raw, signal_ts, _, _ = _calm_pattern_raw("both")
    out = run_bars(monkeypatch, raw)
    assert signal_ts in out.index
    assert out.loc[signal_ts, "Target"] == 0

    signal = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    frame = _forward_frame(_window_index(signal), BOTH_TOUCH_BARS)
    for direction in ("買い", "売り"):
        result = scanner.evaluate_forward_prediction(
            _forward_prediction(signal, direction), frame
        )
        assert result["status"] == "MISS"


def test_td4_no_touch(monkeypatch):
    """T-D4: 両バリア未到達 → ラベル中立、フォワードは NOT_REACHED（8本目終値）。"""
    raw, signal_ts, _, _ = _calm_pattern_raw("none")
    out = run_bars(monkeypatch, raw)
    assert out.loc[signal_ts, "Target"] == 0

    signal = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    frame = _forward_frame(_window_index(signal), NO_TOUCH_BARS)
    result = scanner.evaluate_forward_prediction(
        _forward_prediction(signal, "買い"), frame
    )
    assert result["status"] == "NOT_REACHED"
    assert result["exit_bar_offset"] == 8
    assert result["exit_price"] == pytest.approx(150.03)


def _validation_frame():
    """walk_forward_validation 用の合成フレーム（400行）。

    Return_1 の符号だけで Target が決まる学習区間を持ち、テスト区間は
    原則中立。パターン行を1行だけ置いて仮想損益を検証する。
    """
    n = 400
    index = pd.date_range(
        start="2026-01-05 00:00", periods=n, freq="1h", tz="UTC"
    )
    data = {col: np.zeros(n) for col in scanner.FEATURE_COLUMNS}
    ret = np.zeros(n)
    target = np.zeros(n, dtype=int)
    for i in range(200):
        if i % 3 == 0:
            ret[i] = 10.0
            target[i] = 1
        elif i % 3 == 1:
            ret[i] = -10.0
            target[i] = -1
    data["Return_1"] = ret
    df = pd.DataFrame(data, index=index)
    df["Target"] = target
    df["ATR"] = 0.10
    df["Future_Open_1"] = 150.0
    df["Future_Close"] = 150.0
    for step in range(1, scanner.LOOKAHEAD_BARS + 1):
        df[f"Future_High_{step}"] = 150.04
        df[f"Future_Low_{step}"] = 149.96
    return df


def _pnl_with_pattern(mutate):
    df = _validation_frame()
    pattern_ts = df.index[320]
    mutate(df, pattern_ts)
    result = scanner.walk_forward_validation(df, pair_name=PAIR_NAME)
    return result["simulated_pnl"]


def test_td_pnl_upper_only_buy_take_profit():
    """T-D（仮想損益）: 上方のみ到達の買い → 利確として +9.1 pips。"""
    def mutate(df, ts):
        df.loc[ts, "Return_1"] = 10.0
        df.loc[ts, "Future_High_3"] = 150.12

    pnl = _pnl_with_pattern(mutate)
    assert pnl["trades"] == 1
    assert pnl["net_pips"] == pytest.approx(9.1)


def test_td_pnl_upper_only_sell_stop_loss():
    """T-D（仮想損益）: 上方のみ到達の売り → 損切りとして −10.9 pips。"""
    def mutate(df, ts):
        df.loc[ts, "Return_1"] = -10.0
        df.loc[ts, "Future_High_2"] = 150.12

    pnl = _pnl_with_pattern(mutate)
    assert pnl["trades"] == 1
    assert pnl["net_pips"] == pytest.approx(-10.9)


def test_td_pnl_lower_only_buy_stop_loss():
    """T-D（仮想損益）: 下方のみ到達の買い → 損切りとして −10.9 pips。"""
    def mutate(df, ts):
        df.loc[ts, "Return_1"] = 10.0
        df.loc[ts, "Future_Low_2"] = 149.85

    pnl = _pnl_with_pattern(mutate)
    assert pnl["trades"] == 1
    assert pnl["net_pips"] == pytest.approx(-10.9)


def test_td_pnl_lower_only_sell_take_profit():
    """T-D（仮想損益）: 下方のみ到達の売り → 利確として +9.1 pips。"""
    def mutate(df, ts):
        df.loc[ts, "Return_1"] = -10.0
        df.loc[ts, "Future_Low_2"] = 149.85

    pnl = _pnl_with_pattern(mutate)
    assert pnl["trades"] == 1
    assert pnl["net_pips"] == pytest.approx(9.1)


def test_td_pnl_both_touch_is_stop_loss():
    """T-D（仮想損益）: 同足両触れの買い → 不利側を先として損切り −10.9 pips。"""
    def mutate(df, ts):
        df.loc[ts, "Return_1"] = 10.0
        df.loc[ts, "Future_High_2"] = 150.12
        df.loc[ts, "Future_Low_2"] = 149.85

    pnl = _pnl_with_pattern(mutate)
    assert pnl["trades"] == 1
    assert pnl["net_pips"] == pytest.approx(-10.9)


def test_td_pnl_no_touch_time_exit():
    """T-D（仮想損益）: 未到達の買い → 8本目終値で時間決済 +2.1 pips。"""
    def mutate(df, ts):
        df.loc[ts, "Return_1"] = 10.0
        df.loc[ts, "Future_Close"] = 150.03

    pnl = _pnl_with_pattern(mutate)
    assert pnl["trades"] == 1
    assert pnl["net_pips"] == pytest.approx(2.1)


# ---------------------------------------------------------------------------
# T-B: 時刻データの前提
# ---------------------------------------------------------------------------

def test_tb1_duplicates_through_real_bars_and_scalar_tz(monkeypatch):
    """T-B1: 実 bars() 経由の同一TZ重複（keep=last・後の行が残る）＋スカラー変換確認。

    異なるタイムゾーン表記の同一時刻は、個別 Timestamp の UTC 変換が
    同一の UTC 時刻に一致することでのみ確認する（混在タイムゾーンの
    DataFrame は実経路の入力として構成できないため、入力テストには
    しない）。
    """
    raw = path_frame(continuous_index(340))
    dup_ts = raw.index[300]
    original_close = float(raw.loc[dup_ts, "Close"])
    later = raw.loc[[dup_ts]].copy()
    later.iloc[0] = later.iloc[0] + 0.77
    combined = pd.concat([raw, later])
    out = run_bars(monkeypatch, combined)
    assert quality_of(out)["dropped_duplicate_timestamp"] == 1
    feature = out.attrs["latest_features"]
    assert feature.index.is_unique
    assert feature.index.is_monotonic_increasing
    assert str(feature.index.tz) == "UTC"
    # 重複排除は keep="last": 入力順で後の行（+0.77 の行）が残る。
    assert float(feature.loc[dup_ts, "Close"]) == pytest.approx(
        original_close + 0.77
    )
    # 異なるタイムゾーン表記の同一時刻は、UTC 変換で同一時刻になる。
    tokyo = pd.Timestamp("2026-01-05 09:00:00+09:00")
    assert tokyo.tz_convert("UTC") == pd.Timestamp(
        "2026-01-05 00:00:00+00:00"
    )
    new_york = pd.Timestamp("2026-01-04 19:00:00-05:00")
    assert new_york.tz_convert("UTC") == pd.Timestamp(
        "2026-01-05 00:00:00+00:00"
    )


def test_tb2_function_level_duplicate_pair():
    """T-B2: 同一時刻ペア（差ゼロ）は、正本関数でも不連続。"""
    monday = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    friday = pd.Timestamp("2026-01-02 00:00", tz="UTC")
    assert not scanner._forward_bar_gap_allowed(monday, monday)
    assert not scanner._forward_bar_gap_allowed(friday, friday)


def test_tb3_function_level_reversed_pairs():
    """T-B3: 逆順ペアは不連続（特に週末型の逆順は、正の時刻差条件で False）。"""
    saturday = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    previous_monday = pd.Timestamp("2025-12-29 00:00", tz="UTC")
    assert not scanner._forward_bar_gap_allowed(saturday, previous_monday)
    friday = pd.Timestamp("2026-01-02 00:00", tz="UTC")
    thursday = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    assert not scanner._forward_bar_gap_allowed(friday, thursday)
    # 正常な昇順ペアの判定は変わらない。
    monday = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    assert scanner._forward_bar_gap_allowed(friday, monday)
    assert scanner._forward_bar_gap_allowed(
        monday, monday + pd.Timedelta(hours=1)
    )


def test_tb4_tz_naive_input_stops(monkeypatch):
    """T-B4: tz-naive な入力は UTC と推定せず処理を停止する。"""
    naive_index = pd.date_range(
        start="2026-01-05 00:00", periods=340, freq="1h"
    )
    stub_download(monkeypatch, path_frame(naive_index))
    assert scanner.bars(TICKER, interval="1h", period="180d") is None


def test_tb5_non_utc_input_normalized_to_utc(monkeypatch):
    """T-B5: Asia/Tokyo 表記の入力は UTC に正規化され、曜日判定も UTC で行う。"""
    utc_index = fx_week_index(340)
    raw = path_frame(utc_index.tz_convert("Asia/Tokyo"))
    out = run_bars(monkeypatch, raw)
    feature = out.attrs["latest_features"]
    assert str(feature.index.tz) == "UTC"
    assert quality_of(out)["dropped_label_discontinuous"] == 0
    friday_evening_utc = pd.Timestamp("2026-01-09 20:00", tz="UTC")
    assert friday_evening_utc in out.index


def test_tb6_eighty_hour_function_boundary():
    """T-B6: 80時間ちょうどは連続、80時間＋1分は不連続（関数レベル）。"""
    friday = pd.Timestamp("2026-01-02 00:00", tz="UTC")
    exactly = pd.Timestamp("2026-01-05 08:00", tz="UTC")
    over = pd.Timestamp("2026-01-05 08:01", tz="UTC")
    assert scanner._forward_bar_gap_allowed(friday, exactly)
    assert not scanner._forward_bar_gap_allowed(friday, over)


# ---------------------------------------------------------------------------
# T-A: 監査フレーム
# ---------------------------------------------------------------------------

AUDIT_COLUMNS = {
    "continuity_valid",
    "label_trainable",
    "invalid_reason",
    "first_bad_pair_prev",
    "first_bad_pair_next",
}

EXPECTED_QUALITY_KEYS = {
    "input_rows",
    "dropped_invalid_timestamp",
    "dropped_missing_or_nonfinite",
    "dropped_duplicate_timestamp",
    "dropped_invalid_ohlc",
    "dropped_feature_nan",
    "feature_rows",
    "dropped_label_discontinuous",
    "dropped_label_insufficient_future",
    "dropped_label_other",
    "trainable_rows",
}


def _gap_out(monkeypatch):
    raw = path_frame(continuous_index(340))
    gap_ts = raw.index[260]
    out = run_bars(monkeypatch, raw.drop(index=[gap_ts]))
    return out, gap_ts


def test_ta1_audit_completeness(monkeypatch):
    """T-A1: 監査は feature_df の全行を保持し、除外行も欠落しない。"""
    out, gap_ts = _gap_out(monkeypatch)
    audit = audit_of(out)
    dq = quality_of(out)
    assert len(audit) == dq["feature_rows"]
    assert audit.index.equals(out.attrs["latest_features"].index)
    for ts in discontinuous_timestamps(out):
        assert ts in audit.index
        assert ts not in out.index


def test_ta2_audit_reason_and_first_bad_pair(monkeypatch):
    """T-A2: 監査の理由と最初の不良ペアが、カウンタと実欠損に対応する。"""
    out, gap_ts = _gap_out(monkeypatch)
    prev_ts, next_ts, expected = gap_context(out, gap_ts)
    audit = audit_of(out)
    for ts in expected:
        row = audit.loc[ts]
        assert row["invalid_reason"] == "discontinuous"
        assert row["first_bad_pair_prev"] == prev_ts
        assert row["first_bad_pair_next"] == next_ts
        assert bool(row["continuity_valid"]) is False
        assert not row["label_trainable"]


def test_ta3_audit_checksums_and_dtype(monkeypatch):
    """T-A3: 監査とカウンタの4検算が成立し、判定不能は pd.NA で区別される。"""
    for frame in (
        path_frame(continuous_index(340)),
        path_frame(continuous_index(340)).drop(
            index=[path_frame(continuous_index(340)).index[260]]
        ),
    ):
        out = run_bars(monkeypatch, frame)
        audit = audit_of(out)
        dq = quality_of(out)
        assert str(audit["continuity_valid"].dtype) == "boolean"
        assert (
            int(audit["continuity_valid"].eq(False).sum())
            == dq["dropped_label_discontinuous"]
        )
        assert int(audit["label_trainable"].sum()) == dq["trainable_rows"]
        for reason, key in (
            ("discontinuous", "dropped_label_discontinuous"),
            ("insufficient_future", "dropped_label_insufficient_future"),
            ("other", "dropped_label_other"),
        ):
            assert int((audit["invalid_reason"] == reason).sum()) == dq[key]
        assert len(audit) == dq["feature_rows"]
        na_rows = audit[audit["invalid_reason"] == "insufficient_future"]
        assert na_rows["continuity_valid"].isna().all()


def test_ta4_no_contamination(monkeypatch):
    """T-A4: 監査情報は特徴量・学習データ・公開ステータスの形に混入しない。"""
    out, _ = _gap_out(monkeypatch)
    assert not AUDIT_COLUMNS & set(out.columns)
    assert "label_audit" not in scanner.FEATURE_COLUMNS
    assert set(quality_of(out).keys()) == EXPECTED_QUALITY_KEYS
    assert set(audit_of(out).columns) == AUDIT_COLUMNS


def test_ta5_reason_separation(monkeypatch):
    """T-A5: insufficient_future / discontinuous / other が分離して記録される。"""
    out, gap_ts = _gap_out(monkeypatch)
    audit = audit_of(out)
    feature_index = out.attrs["latest_features"].index
    tail_row = audit.loc[feature_index[-1]]
    assert tail_row["invalid_reason"] == "insufficient_future"
    assert pd.isna(tail_row["continuity_valid"])
    prev_ts, _, _ = gap_context(out, gap_ts)
    disc_row = audit.loc[prev_ts]
    assert disc_row["invalid_reason"] == "discontinuous"
    assert bool(disc_row["continuity_valid"]) is False
    trainable_ts = next(
        ts for ts in feature_index if audit.loc[ts, "invalid_reason"] == ""
    )
    ok_row = audit.loc[trainable_ts]
    assert bool(ok_row["continuity_valid"]) is True
    assert bool(ok_row["label_trainable"]) is True

    # other 行は ATR 注入でのみ構成できる（モジュール冒頭の注記参照）。
    probe = run_bars(monkeypatch, path_frame(continuous_index(340)))
    probe_audit = audit_of(probe)
    other_ts = next(
        ts
        for ts in probe.attrs["latest_features"].index
        if probe_audit.loc[ts, "invalid_reason"] == ""
    )
    _zero_atr_at(monkeypatch, other_ts)
    out_other = run_bars(monkeypatch, path_frame(continuous_index(340)))
    other_row = audit_of(out_other).loc[other_ts]
    assert other_row["invalid_reason"] == "other"
    assert bool(other_row["continuity_valid"]) is True
    assert not other_row["label_trainable"]
    assert quality_of(out_other)["dropped_label_other"] == 1
    assert other_ts not in out_other.index


def test_ta6_attrs_retention_and_forward_path(monkeypatch):
    """T-A6: 監査は実経路（walk_forward_validation・analyze）を通しても保持される。"""
    out = run_bars(monkeypatch, path_frame(continuous_index(500)))
    audit_before = audit_of(out).copy()
    validation = scanner.walk_forward_validation(out, pair_name=PAIR_NAME)
    scanner.analyze(out, validation)
    pd.testing.assert_frame_equal(audit_of(out), audit_before)
    copied = out.copy()
    assert "label_audit" in copied.attrs
    pd.testing.assert_frame_equal(copied.attrs["label_audit"], audit_before)
    # フォワード評価の経路は監査を必要とせず、参照もしない。
    assert not AUDIT_COLUMNS & set(out.attrs["latest_features"].columns)
    signal = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    plain = _forward_frame(_window_index(signal), NO_TOUCH_BARS)
    assert not hasattr(plain, "attrs") or "label_audit" not in plain.attrs
    result = scanner.evaluate_forward_prediction(
        _forward_prediction(signal), plain
    )
    assert result is not None


# ---------------------------------------------------------------------------
# T-V: バージョン・系列分離
# ---------------------------------------------------------------------------

def _app_literal(name):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in app.py")


def test_tv1_model_versions():
    """T-V1: Scanner は 21、App は 8 のまま。"""
    assert scanner.MODEL_VERSION == 21
    assert _app_literal("MODEL_VERSION") == 8


def test_tv2_constants_and_features_unchanged():
    """T-V2: 特徴量・ゲート・閾値に関わる定数が変更されていない。"""
    assert scanner.FEATURE_COLUMNS == [
        "Return_1",
        "Return_5",
        "Dev_SMA20",
        "Dev_EMA20",
        "Dev_EMA200",
        "RSI",
        "RSI_Diff",
        "MACD_Hist_Ratio",
        "BB_PctB",
        "ADX",
        "ATR_Ratio",
        "Upper_Wick_Ratio",
        "Lower_Wick_Ratio",
        "Stoch_K",
        "CCI",
        "ROC",
    ]
    assert scanner.LOOKAHEAD_BARS == 8
    assert scanner.FORWARD_UNEVALUABLE_AFTER_DAYS == 14
    assert scanner.SIMULATED_ROUND_TRIP_SLIPPAGE_PIPS == 0.5
    assert scanner.SIMULATED_SWAP_STRESS_COST_PIPS == 0.2
    assert scanner.MIN_FEATURE_BARS == 220
    assert scanner.MIN_TRAIN_ROWS == 180
    assert scanner.N_WALK_FORWARD_FOLDS == 3
    spreads = {
        name: config["spread_pips"] for name, config in scanner.PAIRS.items()
    }
    assert spreads == {
        "米ドル/円 (USD/JPY)": 0.2,
        "英ポンド/円 (GBP/JPY)": 1.0,
        "ユーロ/円 (EUR/JPY)": 0.5,
        "豪ドル/円 (AUD/JPY)": 0.6,
        "ユーロ/米ドル (EUR/USD)": 0.4,
    }


def test_tv3_discontinuous_counter_is_real(monkeypatch):
    """T-V3: カウンタは固定値ではなく、データに応じて実際に数えられている。"""
    clean = run_bars(monkeypatch, path_frame(continuous_index(340)))
    assert quality_of(clean)["dropped_label_discontinuous"] == 0
    out, _ = _gap_out(monkeypatch)
    assert quality_of(out)["dropped_label_discontinuous"] == 8


# ---------------------------------------------------------------------------
# 適用範囲: label_audit は1時間足の経路のみ（再レビュー対応で追加）
# ---------------------------------------------------------------------------

BASE_ATTRS_KEYS = {
    "latest_data",
    "latest_index",
    "latest_history",
    "latest_features",
}


def test_audit_scope_1h_present_in_prescribed_form(monkeypatch):
    """適用範囲: 1時間足では label_audit が所定の形式で保持される。"""
    out, _ = _gap_out(monkeypatch)
    assert set(out.attrs) == BASE_ATTRS_KEYS | {"data_quality", "label_audit"}
    audit = out.attrs["label_audit"]
    assert set(audit.columns) == AUDIT_COLUMNS
    assert str(audit["continuity_valid"].dtype) == "boolean"
    assert audit.index.equals(out.attrs["latest_features"].index)
    assert len(audit) == quality_of(out)["feature_rows"]


def test_audit_scope_4h_absent_and_processing_unchanged(monkeypatch):
    """適用範囲: 4時間足では label_audit が付与されず、既存の集約処理も変わらない。"""
    hourly = path_frame(continuous_index(1500))
    stub_download(monkeypatch, hourly)
    out = scanner.bars(TICKER, interval="4h", period="90d")
    assert out is not None
    assert "label_audit" not in out.attrs
    assert "data_quality" not in out.attrs
    assert set(out.attrs) == BASE_ATTRS_KEYS
    feature = out.attrs["latest_features"]
    assert str(feature.index.tz) == "UTC"
    assert all(
        ts.hour % 4 == 0 and ts.minute == 0 for ts in feature.index
    )
    # 既存の集約（UTC の4時間区間、完了した区間のみ採用）の具体値で確認する。
    probe_ts = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    hour_rows = hourly.loc[probe_ts : probe_ts + pd.Timedelta(hours=3)]
    assert len(hour_rows) == 4
    row = feature.loc[probe_ts]
    assert row["Open"] == pytest.approx(hour_rows["Open"].iloc[0])
    assert row["High"] == pytest.approx(hour_rows["High"].max())
    assert row["Low"] == pytest.approx(hour_rows["Low"].min())
    assert row["Close"] == pytest.approx(hour_rows["Close"].iloc[-1])


def test_audit_scope_1d_absent_and_processing_unchanged(monkeypatch):
    """適用範囲: 日足では label_audit が付与されず、従来処理も変わらない。"""
    daily_index = pd.date_range(
        start="2024-01-01 00:00", periods=500, freq="1D", tz="UTC"
    )
    stub_download(monkeypatch, path_frame(daily_index))
    out = scanner.bars(TICKER, interval="1d", period="2y")
    assert out is not None
    assert "label_audit" not in out.attrs
    assert "data_quality" not in out.attrs
    assert set(out.attrs) == BASE_ATTRS_KEYS
    feature = out.attrs["latest_features"]
    # 日足は従来どおり、最後の取得足（未確定足）を除外する。
    assert daily_index[-1] not in feature.index
    assert daily_index[-2] in feature.index


def test_tl17_two_separated_gaps_first_bad_pair_and_single_count(monkeypatch):
    """T-L17: 1つの窓に離れた2つの欠損ペアがある場合も計上は一度だけで、
    first_bad_pair は時系列上最初の不良ペアを指す（ChatGPT 相談への対応）。"""
    index = fx_week_index(340)
    raw = path_frame(index)
    # 生の位置で4本間隔の2行を落とし、特徴量フレーム上に離れた2つの
    # 欠損ペアを作る（同じ8本窓に両方が入りうる近さにする）。
    raw = raw.drop(index=[index[200], index[204]])
    out = run_bars(monkeypatch, raw)
    dq = quality_of(out)
    audit = audit_of(out)
    feature_index = out.attrs["latest_features"].index

    # 特徴量フレーム上の不良ペア（連続性規則を満たさない隣接ペア）を特定する。
    bad_pairs = []
    for pos in range(len(feature_index) - 1):
        prev_ts, next_ts = feature_index[pos], feature_index[pos + 1]
        if not scanner._forward_bar_gap_allowed(prev_ts, next_ts):
            bad_pairs.append((pos, prev_ts, next_ts))
    assert len(bad_pairs) == 2
    (g1_pos, g1_prev, g1_next), (g2_pos, g2_prev, g2_next) = bad_pairs
    assert 0 < g2_pos - g1_pos <= 7

    # 各不良ペアを含む窓を持つシグナル行（位置 p-7..p）の和集合が無効範囲で、
    # 重なる行も一度だけ計上される。
    expected = set()
    for pos, _, _ in bad_pairs:
        for s in range(pos - 7, pos + 1):
            expected.add(feature_index[s])
    disc = audit[audit["invalid_reason"] == "discontinuous"]
    assert set(disc.index) == expected
    assert dq["dropped_label_discontinuous"] == len(expected)

    # 両方の不良ペアを含む窓では、時系列上最初の不良ペアが記録される。
    both = [feature_index[s] for s in range(g2_pos - 7, g1_pos + 1)]
    assert both
    for ts in both:
        row = audit.loc[ts]
        assert row["first_bad_pair_prev"] == g1_prev
        assert row["first_bad_pair_next"] == g1_next

    # 2番目のペアだけを含む窓では、2番目のペアが記録される。
    only_second = [feature_index[s] for s in range(g1_pos + 1, g2_pos + 1)]
    assert only_second
    for ts in only_second:
        row = audit.loc[ts]
        assert row["first_bad_pair_prev"] == g2_prev
        assert row["first_bad_pair_next"] == g2_next
