"""ネットワークを使わない最小限のスモークテスト。

目的は「構文エラーや import エラーで本番スキャンが起動直後に死ぬ」事故
(例: 2026-10-24 の SyntaxError) を CI で事前に止めること。
scanner.py は `if __name__ == "__main__":` ガードがあるため、
import してもスキャンは実行されない。
"""

import py_compile

import pandas as pd
import pytest

import scanner


def test_all_python_files_compile():
    for path in ("scanner.py", "fx_scanner_simple.py", "app.py"):
        py_compile.compile(path, doraise=True)


def test_safe_float():
    assert scanner.safe_float("1.5") == 1.5
    assert scanner.safe_float(2) == 2.0
    assert scanner.safe_float("not-a-number") is None
    assert scanner.safe_float(None, default=0.0) == 0.0
    assert scanner.safe_float(float("inf")) is None


def test_load_state_missing_file(tmp_path, monkeypatch):
    # 状態ファイルが無い (キャッシュの冷間スタート) でも落ちず、空 dict を返す。
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scanner, "STATE_FILE", str(tmp_path / "scan_state.json"))
    assert scanner.load_state() == {}


def test_load_state_broken_json(tmp_path, monkeypatch):
    state = tmp_path / "scan_state.json"
    state.write_text("{broken json", encoding="utf-8")
    monkeypatch.setattr(scanner, "STATE_FILE", str(state))
    assert scanner.load_state() == {}


def test_flatten_yf_columns_multiindex():
    columns = pd.MultiIndex.from_tuples(
        [("Open", "USDJPY=X"), ("Close", "USDJPY=X")]
    )
    df = pd.DataFrame([[1.0, 2.0]], columns=columns)
    out = scanner.flatten_yf_columns(df.copy())
    assert list(out.columns) == ["Open", "Close"]


def test_contains_news_keyword():
    assert scanner.contains_news_keyword("FOMC holds rates steady") is True
    assert scanner.contains_news_keyword("local weather is sunny") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
