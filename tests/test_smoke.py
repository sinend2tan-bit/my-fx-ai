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



# =========================================================
# フォワード検証（ネットワーク不要・合成データ）
# =========================================================

def _forward_frame(bars_after, signal_close=100.0):
    rows = [(100.0, 100.2, 99.8, signal_close)] + bars_after
    idx = pd.date_range("2026-01-05 00:00+00:00", periods=len(rows), freq="1h")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def _forward_prediction(frame, direction="買い"):
    return {
        "id": scanner.forward_prediction_id("USDJPY=X", frame.index[0].isoformat()),
        "ticker": "USDJPY=X",
        "direction": direction,
        "reference_price": 100.0,
        "atr": 1.0,
        "signal_bar_time": frame.index[0].isoformat(),
        "pip_size": 0.01,
        "spread_pips": 0.2,
        "status": "PENDING",
    }


def _flat_bars(n, close=100.3):
    return [(100.1, 100.6, 99.6, close)] * n


def test_forward_buy_hit():
    frame = _forward_frame(
        [(100.1, 100.4, 99.6, 100.2), (100.2, 101.3, 100.1, 101.0)] + _flat_bars(6)
    )
    out = scanner.evaluate_forward_prediction(_forward_prediction(frame), frame)
    assert out["status"] == "HIT"
    assert out["exit_price"] == 101.0
    assert out["gross_pips"] == 100.0
    # コスト（スプレッド0.2＋スリッページ0.5＋スワップストレス0.2）を控除した純損益
    assert out["net_pips"] == 99.1


def test_forward_buy_miss_and_both_hit_bar():
    frame = _forward_frame([(100.0, 100.3, 98.8, 99.1)] + _flat_bars(7))
    out = scanner.evaluate_forward_prediction(_forward_prediction(frame), frame)
    assert out["status"] == "MISS"
    assert out["net_pips"] == -100.9

    # 同じ足で上下両方に触れたら、足内の順序不明のため不利側（MISS）を先に扱う
    frame = _forward_frame([(100.0, 101.5, 98.5, 100.0)] + _flat_bars(7))
    out = scanner.evaluate_forward_prediction(_forward_prediction(frame), frame)
    assert out["status"] == "MISS"


def test_forward_sell_hit():
    frame = _forward_frame(
        [(99.9, 100.6, 99.5, 99.7), (99.7, 100.2, 98.7, 99.0)] + _flat_bars(6)
    )
    out = scanner.evaluate_forward_prediction(
        _forward_prediction(frame, "売り"), frame
    )
    assert out["status"] == "HIT"
    assert out["gross_pips"] == 100.0


def test_forward_not_reached_and_pending_when_short():
    bars = _flat_bars(8)
    bars[2] = (100.1, 100.6, 99.4, 100.2)
    bars[7] = (100.1, 100.6, 99.6, 100.3)
    frame = _forward_frame(bars)
    out = scanner.evaluate_forward_prediction(_forward_prediction(frame), frame)
    assert out["status"] == "NOT_REACHED"
    assert out["gross_pips"] == 30.0
    assert out["mae_pips"] == 60.0

    # 評価期間の足が8本そろっていなければ評価しない（PENDINGのまま残す）
    frame = _forward_frame(_flat_bars(5))
    assert scanner.evaluate_forward_prediction(_forward_prediction(frame), frame) is None


def test_forward_record_dedupes_and_skips_holds():
    predictions = []
    candidate = {
        "name": "米ドル/円 (USD/JPY)",
        "ticker": "USDJPY=X",
        "decision": "候補あり（手動確認のみ）",
        "direction": "買い",
        "price": 100.0,
        "atr": 1.0,
        "signal_bar_time": "2026-01-05T00:00:00+00:00",
    }
    hold = dict(candidate, decision="検証条件未達・保留")
    assert scanner.record_forward_predictions(predictions, [candidate, hold]) == 1
    # 同じシグナル足の候補は二重に記録しない
    assert scanner.record_forward_predictions(predictions, [candidate]) == 0
    assert len(predictions) == 1
    assert predictions[0]["status"] == "PENDING"


def test_forward_summary_hit_rate_definition():
    stats = scanner.new_forward_stats()
    for status in ("HIT", "MISS", "NOT_REACHED"):
        scanner.update_forward_stats(
            stats, {"direction": "買い"},
            {"status": status, "net_pips": 10.0, "mae_pips": 5.0},
        )
    view = scanner.build_forward_summary(
        stats, [], 3, 0, tracking_since="2026-01-05T00:00:00+09:00"
    )
    # 的中率は HIT / (HIT + MISS)。NOT_REACHED は分母に入れない
    assert view["hit_rate_pct"] == 50.0
    assert view["tracking_since"] == "2026-01-05T00:00:00+09:00"

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

