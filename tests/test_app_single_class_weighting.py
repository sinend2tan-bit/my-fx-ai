"""App model weighting and original-bar embargo regression tests.

Static/synthetic only: no live market data or model training.
These tests establish implementation contracts, not predictive improvement.
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
    assert version == 10


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
    eligible_train = original[[0, 1, 2, 3, 5, 6, 8, 9, 10]]
    test_start = original[12]
    keep = purge(original, eligible_train, test_start, lookahead=2)
    kept = eligible_train[keep]
    assert list(kept) == list(original[[0, 1, 2, 3, 5, 6, 8, 9]])
    assert original.get_loc(kept[-1]) + 2 < original.get_loc(test_start)


def test_embargo_fails_closed_for_unknown_or_duplicate_original_timestamps():
    import pandas as pd
    import pytest

    purge = _extract_purge_helper()
    original = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    with pytest.raises(ValueError):
        purge(original, pd.DatetimeIndex([original[0], original[-1] + pd.Timedelta(hours=1)]), original[4], 2)
    duplicate = original.insert(2, original[1])
    with pytest.raises(pd.errors.InvalidIndexError):
        purge(duplicate, pd.DatetimeIndex([original[0]]), original[4], 2)


def _extract_fixed_metrics_helper():
    import numpy as np

    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_fixed_class_metrics"
    )
    labels = next(
        item.value for item in tree.body
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "APP_CLASS_LABELS" for target in item.targets)
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[ast.Assign(targets=[ast.Name(id="APP_CLASS_LABELS", ctx=ast.Store())], value=labels), node], type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace["_fixed_class_metrics"]


def test_app_fixed_class_balanced_accuracy_requires_all_true_classes():
    metrics = _extract_fixed_metrics_helper()
    balanced, recalls, complete, supports = metrics(
        [-1, -1, 0, 0, 1, 1], [-1, 0, 0, 0, 1, -1]
    )
    assert complete is True
    assert supports == [2, 2, 2]
    assert recalls == [0.5, 1.0, 0.5]
    assert balanced == 2.0 / 3.0

    balanced, recalls, complete, supports = metrics(
        [-1, -1, 0, 0], [-1, 0, 0, 0]
    )
    assert complete is False
    assert supports == [2, 2, 0]
    assert recalls == [0.5, 1.0, None]
    assert balanced is None


def test_app_holds_signal_when_three_class_holdout_is_incomplete():
    function = _analysis_function()
    source = ast.unparse(function)
    assert "if not validation_complete" in source
    assert "return fallback" in source
