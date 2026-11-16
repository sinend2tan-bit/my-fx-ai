"""Appモデルのクラス不均衡補正が二重適用されないことを確認する。

RandomForestClassifier の class_weight と fit(sample_weight) は乗算される。
AppはScannerと同様に compute_sample_weight("balanced") の一経路だけを使う。
この静的テストは実市場データ・学習・ネットワークを使用しない。
予測性能の改善そのものを証明するテストではない。
"""
import ast
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _analysis_function():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "analyze_signal_with_backtest"
    )


def _call_name(node):
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_app_random_forest_uses_only_one_balancing_mechanism():
    function = _analysis_function()
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    estimators = [node for node in calls if _call_name(node) == "RandomForestClassifier"]
    assert len(estimators) == 2
    for estimator in estimators:
        assert not any(keyword.arg == "class_weight" for keyword in estimator.keywords)

    weight_calls = [node for node in calls if _call_name(node) == "compute_sample_weight"]
    assert len(weight_calls) == 2
    for weight_call in weight_calls:
        assert any(
            keyword.arg == "class_weight"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "balanced"
            for keyword in weight_call.keywords
        )

    fit_calls = [
        node for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"eval_model", "candidate_model"}
    ]
    assert len(fit_calls) == 2
    assert all(any(keyword.arg == "sample_weight" for keyword in call.keywords) for call in fit_calls)



def test_app_model_version_invalidates_models_trained_with_double_weighting():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    version = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MODEL_VERSION" for target in node.targets)
    )
    assert version == 9
def _extract_purge_helper():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_purged_train_mask"
    )
    namespace = {"pd": __import__("pandas")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace["_purged_train_mask"]


def test_embargo_uses_original_bar_positions_when_feature_rows_are_missing():
    import pandas as pd

    purge = _extract_purge_helper()
    original = pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    # Feature/label filtering removed original bars 4 and 7.
    eligible_train = original[[0, 1, 2, 3, 5, 6, 8, 9, 10]]
    test_start = original[12]
    keep = purge(original, eligible_train, test_start, lookahead=2)
    kept = eligible_train[keep]
    assert list(kept) == list(original[[0, 1, 2, 3, 5, 6, 8, 9]])
    # The final retained train label ends at original bar 11, before test bar 12.
    assert original.get_loc(kept[-1]) + 2 < original.get_loc(test_start)


def test_embargo_fails_closed_for_unknown_or_duplicate_original_timestamps():
    import pandas as pd
    import pytest

    purge = _extract_purge_helper()
    original = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    with pytest.raises(ValueError):
        purge(original, pd.DatetimeIndex([original[0], original[8]]), original[4], 2)
    duplicate = original.insert(2, original[1])
    with pytest.raises(Exception):
        purge(duplicate, pd.DatetimeIndex([original[0]]), original[4], 2)
