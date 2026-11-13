"""アプリのモデル保存失敗の可視化（AIFX-APP-PERSIST-20261113-01）の回帰テスト。

確認すること:
- joblib.dump が失敗しても例外で画面が壊れず、保存できなかった事実が
  警告として画面に表示されること（例外詳細・保存先パスは表示しない）。
- 保存成功時は従来どおりモデルファイルが書かれ、警告が出ないこと。
  保存される辞書のキー構成（形式）が従来と同じであること。
- 保存成否は session_state の model_save_status に、例外詳細やパスを
  含まない形で記録されること。

実データ・実ネットワークは使わない（yfinance / requests はスタブ、
ソケットはガード）。価格パスは学習ゲートを通過できる決定的な
周期パターン（12本上昇・12本下降の繰り返し）を使う。
streamlit が無い環境ではモジュールごとスキップする。
"""

import socket
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import requests
import yfinance as yf

pytest.importorskip("streamlit", reason="AppTest requires streamlit")
pytest.importorskip("plotly", reason="AppTest requires plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SAVE_FAILURE_MARKER = "保存できませんでした"
SECRET_MARKER = "simulated-save-failure-marker"


def _wave_frame(n, tz, freq, start):
    """12本上昇・12本下降を繰り返す決定的な価格パス（学習可能な周期性）。"""
    idx = pd.date_range(start=start, periods=n, freq=freq, tz=tz)
    i = np.arange(n)
    step = np.where((i // 12) % 2 == 0, 0.45, -0.45)
    noise = np.where(i % 2 == 0, 0.02, -0.02)
    close = 150.0 + np.cumsum(step) + noise
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


def _fake_download(*args, **kwargs):
    interval = kwargs.get("interval")
    if interval == "1d":
        return _wave_frame(500, "UTC", "1d", "2024-06-03")
    if interval == "15m":
        return _wave_frame(500, "UTC", "15min", "2026-01-05 00:00")
    return _wave_frame(600, "UTC", "1h", "2026-01-05 00:00")


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    def guard(*args, **kwargs):
        raise AssertionError("real network access attempted during AppTest")

    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)
    monkeypatch.setattr(socket, "gethostbyname", guard)
    monkeypatch.setattr(yf, "download", _fake_download)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())
    # モデルファイル等の相対パス出力は一時ディレクトリへ逃がす。
    monkeypatch.chdir(tmp_path)
    import streamlit as st

    st.cache_data.clear()
    return tmp_path


def _boot():
    return AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def _all_warning_texts(at):
    texts = [str(w.value) for w in at.warning]
    texts += [str(w.value) for w in at.sidebar.warning]
    return texts


def _save_failure_warnings(at):
    return [t for t in _all_warning_texts(at) if SAVE_FAILURE_MARKER in t]


def test_save_success_writes_file_without_warning(rig):
    at = _boot()
    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"

    status = at.session_state["model_save_status"]
    assert status["saved"] is True
    # 保存経路が実際に通ったこと（ゲート通過）を確認する。
    assert at.session_state["model_eval_metrics"]["quality_ok"] is True

    model_files = list(rig.glob("model_*.joblib"))
    assert len(model_files) == 1
    saved = joblib.load(model_files[0])
    assert set(saved) == {
        "version",
        "features",
        "model",
        "trained_at",
        "validation_balanced_accuracy",
        "baseline_balanced_accuracy",
        "feedback_applied",
    }
    assert saved["version"] == 8
    assert _save_failure_warnings(at) == []


def test_save_failure_is_visible_without_details(rig, monkeypatch):
    def failing_dump(*args, **kwargs):
        raise OSError(SECRET_MARKER)

    monkeypatch.setattr(joblib, "dump", failing_dump)
    at = _boot()
    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"

    status = at.session_state["model_save_status"]
    assert status["saved"] is False
    # 記録にも例外詳細・パスを含めない。
    assert SECRET_MARKER not in str(status)
    assert "path" not in status

    assert list(rig.glob("model_*.joblib")) == []

    failures = _save_failure_warnings(at)
    assert len(failures) >= 1
    # 画面のどこにも例外詳細・保存先パス・内部名を露出しない。
    visible = "\n".join(_all_warning_texts(at))
    assert SECRET_MARKER not in visible
    assert "joblib" not in visible
    assert str(rig) not in visible
