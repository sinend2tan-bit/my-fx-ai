"""Appモデルのクラス不均衡補正が二重適用されないことを確認する。

RandomForestClassifier の class_weight と fit(sample_weight) は乗算される。
Appはscannerと同様に compute_sample_weight("balanced") の一経路だけを使う。
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
    calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]

    estimators = [
        node for node in calls
        if _call_name(node) == "RandomForestClassifier"
    ]
    assert len(estimators) == 2

    # Both the chronological evaluator and full-data candidate must avoid
    # combining class_weight with explicit balanced sample weights.
    for estimator in estimators:
        assert not any(keyword.arg == "class_weight" for keyword in estimator.keywords)

    weight_calls = [
        node for node in calls
        if _call_name(node) == "compute_sample_weight"
    ]
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
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit"
    ]
    assert len(fit_calls) == 2
    assert all(
        any(keyword.arg == "sample_weight" for keyword in fit_call.keywords)
        for fit_call in fit_calls
    )
