"""app.py の起動スモークテスト（AppTest）。

目的は次の2点だけ。
1. app.py が合成データだけで例外なく起動すること。
2. 起動中に本物のネットワーク通信と Discord 送信が発生しないこと。

分析ロジック・予測判定・フォワード検証は対象にしない。

安全装置：
- ソケット層のガードが、実際の接続試行をすべて記録してから例外にする。
  対象は TCP 接続（create_connection / connect / connect_ex）、
  送信（sendto / sendmsg / send / sendall）、DNS 名前解決
  （getaddrinfo / gethostbyname / gethostbyname_ex / gethostbyaddr /
  getnameinfo）。
  アプリ側で例外が握りつぶされても試行の記録は残るため、実通信の
  試行があれば必ず検出できる。
- yfinance.download と requests.get / post は記録専用の偽物に差し替える。
- webhook_url は既定（空）のまま。秘密情報は使わない。
- streamlit が無い環境（スキャナー用 CI ジョブ）ではモジュールごと
  スキップする。このテストは AppTest 専用ジョブでのみ実行される。
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


def _synthetic_frame():
    """種を固定した合成 OHLCV（毎回同じ形・同じ値）。"""
    idx = pd.date_range("2024-01-01", periods=2000, freq="B")
    rng = np.random.default_rng(0)
    base = 150.0 + np.cumsum(rng.normal(0, 0.3, 2000))
    return pd.DataFrame(
        {
            "Open": base,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
            "Volume": [10**6] * 2000,
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


class _NetworkRig:
    """差し替え一式と、実通信試行の記録を持つ。"""

    def __init__(self):
        self.real_attempts = []
        self.yf_download_calls = 0
        self.requests_get_calls = 0
        self.requests_post_calls = 0

    def guard(self, *args, **kwargs):
        # ソケット層ガード：試行を記録してから例外にする。
        self.real_attempts.append((args, kwargs))
        raise AssertionError("real network access attempted during AppTest")

    def fake_download(self, *args, **kwargs):
        self.yf_download_calls += 1
        return _synthetic_frame()

    def fake_get(self, *args, **kwargs):
        self.requests_get_calls += 1
        return _FakeResponse()

    def fake_post(self, *args, **kwargs):
        self.requests_post_calls += 1
        return _FakeResponse()


@pytest.fixture()
def rig(monkeypatch):
    r = _NetworkRig()
    # TCP 接続
    monkeypatch.setattr(socket, "create_connection", r.guard)
    monkeypatch.setattr(socket.socket, "connect", r.guard)
    monkeypatch.setattr(socket.socket, "connect_ex", r.guard)
    # UDP 送信（接続なしで送れる経路）と、接続済み前提の送信
    monkeypatch.setattr(socket.socket, "sendto", r.guard)
    monkeypatch.setattr(socket.socket, "sendmsg", r.guard)
    monkeypatch.setattr(socket.socket, "send", r.guard)
    monkeypatch.setattr(socket.socket, "sendall", r.guard)
    # DNS 名前解決（解決そのものが外部通信になりうる）
    monkeypatch.setattr(socket, "getaddrinfo", r.guard)
    monkeypatch.setattr(socket, "gethostbyname", r.guard)
    monkeypatch.setattr(socket, "gethostbyname_ex", r.guard)
    monkeypatch.setattr(socket, "gethostbyaddr", r.guard)
    monkeypatch.setattr(socket, "getnameinfo", r.guard)
    # アプリが使う取得・送信の差し替え
    monkeypatch.setattr(yf, "download", r.fake_download)
    monkeypatch.setattr(requests, "get", r.fake_get)
    monkeypatch.setattr(requests, "post", r.fake_post)
    try:
        import streamlit as st

        st.cache_data.clear()
    except Exception:
        pass
    return r


def _boot():
    return AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def test_app_boots_without_exception(rig):
    at = _boot()
    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"
    assert [t.value for t in at.title], "title not rendered"


def test_no_real_network_during_boot(rig):
    # ガードが生きていることの陰性対照（各経路とも記録されて失敗する）。
    controls = [
        lambda: socket.create_connection(("example.com", 443), timeout=3),
        lambda: socket.socket().connect(("93.184.216.34", 80)),
        lambda: socket.socket().connect_ex(("93.184.216.34", 80)),
        lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(
            b"x", ("93.184.216.34", 53)
        ),
        lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendmsg(
            [b"x"], [], 0, ("93.184.216.34", 53)
        ),
        lambda: socket.socket().send(b"x"),
        lambda: socket.socket().sendall(b"x"),
        lambda: socket.getaddrinfo("example.com", 443),
        lambda: socket.gethostbyname("example.com"),
        lambda: socket.gethostbyname_ex("example.com"),
        lambda: socket.gethostbyaddr("93.184.216.34"),
        lambda: socket.getnameinfo(("93.184.216.34", 80), 0),
    ]
    for attempt in controls:
        with pytest.raises(AssertionError):
            attempt()
    assert len(rig.real_attempts) == len(controls)
    rig.real_attempts.clear()

    at = _boot()
    assert not at.exception
    # 実通信の試行が1件も記録されていないこと。
    assert rig.real_attempts == []
    # 起動時の取得が差し替え先で処理されたこと。
    assert rig.yf_download_calls >= 1
    assert rig.requests_get_calls >= 1


def test_no_discord_send_during_boot(rig):
    at = _boot()
    assert not at.exception
    assert rig.requests_post_calls == 0
