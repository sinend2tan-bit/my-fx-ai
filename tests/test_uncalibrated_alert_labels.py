"""Regression tests for honest labeling of uncalibrated model outputs."""
import ast
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _formatter():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "format_uncalibrated_direction_alert"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace["format_uncalibrated_direction_alert"]


def test_buy_alert_explicitly_says_uncalibrated_not_win_probability():
    message = _formatter()("buy", 63)
    assert "買い方向モデル出力値（未校正） 63%" in message
    assert "勝率・実際の発生確率ではありません" in message
    assert "買い方向確率" not in message


def test_sell_alert_explicitly_says_uncalibrated_not_win_probability():
    message = _formatter()("sell", 41)
    assert "売り方向モデル出力値（未校正） 41%" in message
    assert "勝率・実際の発生確率ではありません" in message


def test_alert_formatter_rejects_unknown_direction_and_out_of_range_values():
    import pytest

    formatter = _formatter()
    with pytest.raises(ValueError):
        formatter("wait", 50)
    with pytest.raises(ValueError):
        formatter("buy", 101)
