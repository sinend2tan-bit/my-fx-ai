"""Regression tests for complete three-class App holdout metrics."""
import ast
from pathlib import Path

import numpy as np

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _fixed_metrics():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    helper = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_fixed_class_metrics"
    )
    labels = next(
        item.value for item in tree.body
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "APP_CLASS_LABELS" for target in item.targets)
    )
    module = ast.Module(
        body=[
            ast.Assign(targets=[ast.Name(id="APP_CLASS_LABELS", ctx=ast.Store())], value=labels),
            helper,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace["_fixed_class_metrics"]


def test_balanced_accuracy_requires_all_true_classes():
    metrics = _fixed_metrics()
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
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    function = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "analyze_signal_with_backtest"
    )
    source = ast.unparse(function)
    assert "if not validation_complete" in source
    assert "st.warning" in source
    assert "return fallback" in source
    assert "classes_" in source
    assert "APP_CLASS_LABELS" in source
