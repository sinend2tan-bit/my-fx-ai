"""画面簡素化（単発トレード／リピート注文の2構成）の回帰テスト。

確認すること:
- メイン画面に「単発トレード」と「リピート注文（ハーフ＆ハーフ）」だけが
  表示され、削除対象（口座資金入力、Discord Webhook設定、チャート画面、
  ニュース・経済指標画面、AI検証・予測履歴などの画面切替）が出ないこと。
- 単発トレードの方向表示: 4時間足・1時間足のトレンドが一致すれば
  買い方向／売り方向、データ鮮度が確認できなければ判断保留、
  時間足が不一致なら判断保留になること。判断に使った時間足が表示されること。
- リピート注文の設定表示（買いレンジ・売りレンジ・注文幅・数量）が残ること。

実データ・実ネットワークは使わない（yfinance / requests はスタブ、
ソケットはガード）。価格パスは決定的な周期パターン＋末尾トレンド区間で、
時間足ごとのトレンド方向を制御する。
streamlit が無い環境ではモジュールごとスキップする。
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


def _frame(n, freq, end, tail=0.0):
    """周期パターンの末尾に一方向区間（tail の符号×本数）を付けた価格パス。

    tail>0: 末尾30本が上昇、tail<0: 末尾30本が下降。EMA20/50 の向きが
    末尾区間に支配され、トレンド符号が決定的になる。
    """
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    i = np.arange(n)
    step = np.where((i // 12) % 2 == 0, 0.45, -0.45) + np.where(i % 2 == 0, 0.02, -0.02)
    if tail:
        step[-30:] = tail
    close = 150.0 + np.cumsum(step)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + 0.05,
            "Low": np.minimum(open_, close) - 0.05,
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


def _make_download(
    tail_1h=0.0,
    tail_4h_source=0.0,
    stale_main=False,
    stale_4h_source=False,
    missing_4h_source=False,
):
    now_h = pd.Timestamp.now(tz="UTC").floor("h")

    def fake_download(*args, **kwargs):
        interval = kwargs.get("interval")
        period = kwargs.get("period")
        if interval == "1d":
            return _frame(500, "1d", now_h.normalize())
        if interval == "15m":
            return _frame(500, "15min", now_h)
        # 1h 系: period=90d は4時間足の元データ、60d は1時間足（メイン含む）
        if period == "90d":
            if missing_4h_source:
                return pd.DataFrame()
            end4 = now_h - pd.Timedelta(days=2) if stale_4h_source else now_h
            return _frame(600, "1h", end4, tail=tail_4h_source)
        end = now_h - pd.Timedelta(days=10) if stale_main else now_h
        return _frame(600, "1h", end, tail=tail_1h)

    return fake_download


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    def guard(*args, **kwargs):
        raise AssertionError("real network access attempted during AppTest")

    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)
    monkeypatch.setattr(socket, "gethostbyname", guard)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())
    monkeypatch.chdir(tmp_path)
    import streamlit as st

    def boot(**fake_kwargs):
        monkeypatch.setattr(yf, "download", _make_download(**fake_kwargs))
        st.cache_data.clear()
        return AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    return boot


def _texts(at, kind):
    return [str(e.value) for e in getattr(at, kind)]


def _all_text(at):
    out = []
    for kind in ("markdown", "caption", "text", "code", "success", "error", "warning", "info"):
        out += _texts(at, kind)
    return "\n".join(out)


def test_two_sections_and_removed_displays(rig):
    at = rig()
    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"
    body = _all_text(at)
    assert "🎯 単発トレード" in body
    assert "📋 リピート注文（ハーフ＆ハーフ）" in body

    # 画面切替ラジオ（チャート／ニュース／AI検証など）は存在しない。
    for radio in at.radio:
        options = [str(o) for o in radio.options]
        assert not any(
            ("チャート" in o or "ニュース" in o or "AI検証" in o) for o in options
        ), options
    # 口座資金・リスク系の入力欄がない。
    number_labels = [str(w.label) for w in at.number_input]
    assert not any("口座資金" in lb for lb in number_labels), number_labels
    assert not any("許容リスク" in lb or "リスクリワード" in lb for lb in number_labels)
    # Discord Webhook の入力欄がない。
    text_labels = [str(w.label) for w in at.text_input]
    assert not any("Webhook" in lb for lb in text_labels), text_labels
    # AI検証の詳細などの削除対象 expander がない。
    expander_labels = [str(w.label) for w in at.expander]
    assert not any("AI検証" in lb for lb in expander_labels), expander_labels
    # モデル管理・メンテナンスは通常画面の見出しではなく折りたたみに整理され、
    # メイン上部の「分析モデルについて」説明は表示されない。
    assert any("AIモデル管理" in lb for lb in expander_labels), expander_labels
    assert any("更新・メンテナンス" in lb for lb in expander_labels), expander_labels
    assert "分析モデルについて" not in body

    # リピート注文の設定表示が残っている。
    assert any("ハーフ＆ハーフ" in c and "買いレンジ" in c for c in _texts(at, "code"))
    assert any("注文幅" in lb for lb in number_labels)
    # 判断に使った時間足が単発トレード節に表示される。
    assert any("4時間足" in m and "：" in m for m in _texts(at, "markdown"))
    assert any("1時間足" in m and "：" in m for m in _texts(at, "markdown"))


def test_direction_sell_when_trends_agree_down(rig):
    at = rig(tail_1h=-0.45, tail_4h_source=-0.45)
    assert not at.exception
    assert any("方向：売り方向" in e for e in _texts(at, "error")), _texts(at, "error")
    assert any("4時間足： 下降" in m for m in _texts(at, "markdown"))
    assert any("1時間足： 下降" in m for m in _texts(at, "markdown"))


def test_direction_buy_when_trends_agree_up(rig):
    at = rig(tail_1h=0.45, tail_4h_source=0.45)
    assert not at.exception
    assert any("方向：買い方向" in s for s in _texts(at, "success")), _texts(at, "success")


def test_direction_hold_when_only_1h_stale(rig):
    # 4時間足は新鮮でも、1時間足だけ古ければ判断保留（鮮度は系列ごとに検証）。
    at = rig(tail_1h=-0.45, tail_4h_source=-0.45, stale_main=True)
    assert not at.exception
    assert any("方向：判断保留" in w for w in _texts(at, "warning"))
    assert any("鮮度" in c for c in _texts(at, "caption"))
    assert not any("方向：売り方向" in e for e in _texts(at, "error"))


def test_direction_hold_when_only_4h_stale(rig):
    # 選択中の時間足（1時間足）は新鮮でも、4時間足だけ古ければ判断保留。
    # 修正前は選択中時間足の鮮度判定を流用していたため取りこぼしていた。
    at = rig(tail_1h=-0.45, tail_4h_source=-0.45, stale_4h_source=True)
    assert not at.exception
    assert any("方向：判断保留" in w for w in _texts(at, "warning"))
    assert any("鮮度" in c for c in _texts(at, "caption"))
    assert not any("方向：売り方向" in e for e in _texts(at, "error"))


def test_direction_hold_when_4h_data_is_missing(rig):
    # 1時間足が新鮮でも、4時間足の取得に失敗した場合は方向を補完せず保留する。
    at = rig(tail_1h=-0.45, missing_4h_source=True)
    assert not at.exception
    assert any("方向：判断保留" in w for w in _texts(at, "warning"))
    assert not any("方向：売り方向" in e for e in _texts(at, "error"))


def test_direction_hold_when_timeframes_disagree(rig):
    at = rig(tail_1h=0.45, tail_4h_source=-0.45)
    assert not at.exception
    assert any("方向：判断保留" in w for w in _texts(at, "warning"))
    assert any("一致していない" in c for c in _texts(at, "caption"))
