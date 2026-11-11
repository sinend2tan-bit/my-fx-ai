"""段階1 追加レビュー対応（項目1）: 日足停止警告とキャッシュの AppTest。

確認すること:
- キャッシュミス時に、日足（tz-naive）のタイムゾーン停止警告が
  表示されること。
- キャッシュヒット時（再描画）にも、停止警告が表示されること。
  実装は警告表示をキャッシュ対象関数の外側（公開ラッパー）に置くため、
  Streamlit の要素リプレイ仕様に依存せず毎回表示される。ヒットは
  yfinance.download の呼び出し回数が増えないことで確認する。
- 日足の取得失敗（空データ）では停止警告が出ないこと（停止と取得
  失敗・データ不足を混同しないことの画面レベルの確認）。
- 日足が tz-aware の場合は停止警告が出ないこと（誤検出しないこと）。

実データ・実ネットワークは使わない（yfinance / requests はスタブ、
ソケットはガード）。streamlit が無い環境ではモジュールごとスキップ
する（test_app_smoke.py と同じ方式）。
"""

import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import yfinance as yf

pytest.importorskip("streamlit", reason="AppTest requires streamlit")
pytest.importorskip("plotly", reason="AppTest requires plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
TZ_STOP_MARKERS = ("タイムゾーン", "停止")


def _frame(n, tz, freq, start, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz=tz)
    close = 150.0 + np.cumsum(rng.normal(0, 0.15, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


class _FakeResponse:
    status_code = 200
    content = b""
    text = ""

    def json(self):
        return {}

    def raise_for_status(self):
        pass


class _Rig:
    """interval ごとに異なる合成フレームを返す yfinance スタブ。"""

    def __init__(self, daily_mode):
        # daily_mode: "naive"（段階0の実測どおり tz-naive で停止対象） /
        # "empty"（取得失敗） / "aware"（tz-aware で正常処理）
        self.daily_mode = daily_mode
        self.calls = {"1h": 0, "15m": 0, "1d": 0, "other": 0}
        self.real_attempts = []

    def guard(self, *args, **kwargs):
        self.real_attempts.append((args, kwargs))
        raise AssertionError("real network access attempted during AppTest")

    def fake_download(self, *args, **kwargs):
        interval = kwargs.get("interval")
        if interval == "1d":
            self.calls["1d"] += 1
            if self.daily_mode == "empty":
                return pd.DataFrame()
            if self.daily_mode == "aware":
                return _frame(500, "UTC", "1d", "2024-06-03")
            return _frame(500, None, "1d", "2024-06-03")
        if interval == "15m":
            self.calls["15m"] += 1
            return _frame(500, "UTC", "15min", "2026-01-05 00:00")
        if interval == "1h":
            self.calls["1h"] += 1
            return _frame(500, "UTC", "1h", "2026-01-05 00:00")
        self.calls["other"] += 1
        return _frame(500, "UTC", "1h", "2026-01-05 00:00")


@pytest.fixture()
def make_rig(monkeypatch):
    def _make(daily_mode):
        rig = _Rig(daily_mode)
        monkeypatch.setattr(socket, "create_connection", rig.guard)
        monkeypatch.setattr(socket.socket, "connect", rig.guard)
        monkeypatch.setattr(socket.socket, "connect_ex", rig.guard)
        monkeypatch.setattr(socket, "getaddrinfo", rig.guard)
        monkeypatch.setattr(socket, "gethostbyname", rig.guard)
        monkeypatch.setattr(socket, "gethostbyname_ex", rig.guard)
        monkeypatch.setattr(socket, "gethostbyaddr", rig.guard)
        monkeypatch.setattr(socket, "getnameinfo", rig.guard)
        monkeypatch.setattr(yf, "download", rig.fake_download)
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse())
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())
        import streamlit as st

        st.cache_data.clear()
        return rig

    return _make


def _tz_stop_warnings(at):
    return [
        w.value
        for w in at.warning
        if all(marker in str(w.value) for marker in TZ_STOP_MARKERS)
    ]


def _boot():
    return AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def test_daily_tz_stop_warning_on_cache_miss_and_hit(make_rig):
    rig = make_rig("naive")
    at = _boot()
    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"
    # キャッシュミス: 日足は1回だけ取得され、停止警告が表示される。
    assert rig.calls["1d"] == 1
    assert len(_tz_stop_warnings(at)) == 1
    # 再描画（キャッシュヒット）: 取得は繰り返されないが、停止警告は
    # キャッシュの外側で表示されるため再び表示される。
    at2 = at.run()
    assert not at2.exception
    assert rig.calls["1d"] == 1
    assert len(_tz_stop_warnings(at2)) == 1
    assert rig.real_attempts == []


def test_daily_download_failure_has_no_tz_stop_warning(make_rig):
    rig = make_rig("empty")
    at = _boot()
    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"
    assert rig.calls["1d"] == 1
    # 取得失敗は停止ではないため、タイムゾーン停止の警告は出ない。
    assert _tz_stop_warnings(at) == []


def test_daily_tz_aware_has_no_tz_stop_warning(make_rig):
    rig = make_rig("aware")
    at = _boot()
    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"
    assert rig.calls["1d"] == 1
    assert _tz_stop_warnings(at) == []
