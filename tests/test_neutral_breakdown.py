"""中立クラス内訳の観測項目（AIFX-OBS-20261113-01）のテスト。

Cloud Scan の data_quality に、Target=0 の内訳（8本以内に上下バリア
未到達 = no_reach、同一足で上下両方到達 = both_touch）の件数と割合を
集計値として追加する。ラベル・特徴量・予測・ゲートは変更しない。

実データ・実ネットワークは使わない（yfinance はスタブに差し替え、
ソケット接続はガードで禁止する）。観測項目は集計値のみで、行レベル
データは保存・公開しない。
"""

import socket

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

import scanner

TICKER = "USDJPY=X"

PRE_EXISTING_KEYS = {
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

NEW_KEYS = {
    "neutral_total",
    "neutral_no_reach",
    "neutral_both_touch",
    "neutral_no_reach_pct",
    "neutral_both_touch_pct",
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """実ネットワーク接続を禁止する（スタブ以外の取得経路を塞ぐ）。"""

    def guard(*args, **kwargs):
        raise AssertionError("real network access attempted during tests")

    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)


def stub_download(monkeypatch, frame):
    monkeypatch.setattr(yf, "download", lambda *a, **k: frame.copy())


def continuous_index(n=340, start="2026-01-05 00:00"):
    return pd.date_range(start=start, periods=n, freq="1h", tz="UTC")


def path_frame(index):
    """決定的な価格パス（乱数なし）。足の値幅は ATR に対して小さい。"""
    n = len(index)
    i = np.arange(n, dtype=float)
    close = 150.0 + 0.2 * np.sin(i / 9.0) + 0.001 * i
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


def calm_frame(index):
    """小幅な往復パス。値幅全体が ATR 未満で、8本以内に到達しない。

    上昇・下落の両方がないと RSI が全行 NaN になる（ avg_loss=0 の
    実装上の帰結）ため、符号が交互になる小さなサイン波にする。
    振幅 0.004 の往復 + 足の上下 0.01 でも、到達幅は ATR（約0.024）を
    下回る。
    """
    n = len(index)
    i = np.arange(n, dtype=float)
    close = 150.0 + 0.004 * np.sin(i * np.pi / 3.0)
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


def ramp_frame(index):
    """強い上昇パス（小さな押し目入り）。全ラベル行が +1 になる。

    変化量のパターンは +1.2, +1.2, -0.4 の繰り返し。押し目（-0.4）は
    ATR（約0.95）を下回るため下側バリアには届かず、上側バリアには
    1〜2本で到達する。下落を含むのは RSI の NaN 化を避けるため。
    """
    n = len(index)
    steps = np.tile([1.2, 1.2, -0.4], n // 3 + 1)[: n - 1]
    close = 150.0 + np.concatenate([[0.0], np.cumsum(steps)])
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


def run_bars(monkeypatch, frame):
    stub_download(monkeypatch, frame)
    out = scanner.bars(TICKER, interval="1h", period="180d")
    assert out is not None
    return out


def quality_of(df):
    return df.attrs["data_quality"]


def test_neutral_keys_present_and_invariant(monkeypatch):
    out = run_bars(monkeypatch, path_frame(continuous_index()))
    dq = quality_of(out)

    assert NEW_KEYS <= set(dq)
    assert PRE_EXISTING_KEYS <= set(dq)

    neutral_in_frame = int((out["Target"] == 0).sum())
    assert dq["neutral_total"] == neutral_in_frame
    assert dq["neutral_no_reach"] + dq["neutral_both_touch"] == dq[
        "neutral_total"
    ]
    if dq["neutral_total"] > 0:
        assert dq["neutral_no_reach_pct"] is not None
        assert dq["neutral_both_touch_pct"] is not None
        total_pct = dq["neutral_no_reach_pct"] + dq["neutral_both_touch_pct"]
        assert abs(total_pct - 100.0) <= 0.2


def test_plain_frame_has_no_both_touch(monkeypatch):
    # 通常の値幅（ATR 未満の単一足）では同一足の両到達は起きない。
    out = run_bars(monkeypatch, path_frame(continuous_index()))
    dq = quality_of(out)
    assert dq["neutral_both_touch"] == 0
    assert dq["neutral_no_reach"] == dq["neutral_total"]


def test_wide_bar_counts_as_both_touch(monkeypatch):
    frame = path_frame(continuous_index())
    # 位置 320 の足だけ上下に大きく広げる。その直前のシグナル行では、
    # 窓の最初の足が上下両バリアへ到達するため both_touch になる。
    frame.iloc[320, frame.columns.get_loc("High")] = (
        frame.iloc[320]["Close"] + 5.0
    )
    frame.iloc[320, frame.columns.get_loc("Low")] = (
        frame.iloc[320]["Close"] - 5.0
    )
    out = run_bars(monkeypatch, frame)
    dq = quality_of(out)
    assert dq["neutral_both_touch"] >= 1
    assert dq["neutral_no_reach"] + dq["neutral_both_touch"] == dq[
        "neutral_total"
    ]
    assert dq["neutral_total"] == int((out["Target"] == 0).sum())


def test_calm_frame_all_neutral_no_reach(monkeypatch):
    out = run_bars(monkeypatch, calm_frame(continuous_index()))
    dq = quality_of(out)
    assert dq["trainable_rows"] > 0
    assert dq["neutral_total"] == dq["trainable_rows"]
    assert dq["neutral_no_reach"] == dq["trainable_rows"]
    assert dq["neutral_both_touch"] == 0
    assert dq["neutral_no_reach_pct"] == 100.0
    assert dq["neutral_both_touch_pct"] == 0.0


def test_no_neutral_percentages_are_none(monkeypatch):
    out = run_bars(monkeypatch, ramp_frame(continuous_index()))
    dq = quality_of(out)
    assert dq["neutral_total"] == 0
    assert dq["neutral_no_reach"] == 0
    assert dq["neutral_both_touch"] == 0
    assert dq["neutral_no_reach_pct"] is None
    assert dq["neutral_both_touch_pct"] is None


def test_labels_and_counters_unchanged_by_observation(monkeypatch):
    # 観測項目の追加が既存カウンタとラベル分布を変えないこと。
    out = run_bars(monkeypatch, path_frame(continuous_index()))
    dq = quality_of(out)
    assert scanner.MODEL_VERSION == 21
    label_check = dq["feature_rows"] - (
        dq["dropped_label_discontinuous"]
        + dq["dropped_label_insufficient_future"]
        + dq["dropped_label_other"]
    )
    assert label_check == dq["trainable_rows"] == len(out)
    counts = out["Target"].value_counts().to_dict()
    assert counts.get(0, 0) == dq["neutral_total"]
    assert sum(counts.values()) == len(out)
