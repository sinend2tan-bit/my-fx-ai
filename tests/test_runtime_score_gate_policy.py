"""Synthetic regression tests for runtime score-gate vs argmax validation policy."""

import ast
import inspect

import numpy as np

import scanner


def test_argmax_walk_forward_prediction_can_be_rejected_by_live_score_gate():
    """A directional argmax below 55 raw-score points must be an abstention live.

    The ordinary walk-forward path currently uses model.predict() (argmax),
    while analyze() uses the 55% directional score threshold. This fixture
    isolates that policy difference without market data or model training.
    """
    # Scanner RandomForest classes are aligned as SELL, NEUTRAL, BUY.
    scores = np.array([[0.20, 0.30, 0.50]], dtype=float)
    class_labels = np.array([-1, 0, 1], dtype=int)

    walk_forward_argmax = int(class_labels[np.argmax(scores[0])])
    live_policy = scanner.directional_predictions_from_scores(scores)

    assert walk_forward_argmax == 1  # argmax selects BUY at 50 raw-score points
    assert live_policy.tolist() == [0]  # live 55% gate abstains


def test_live_score_gate_accepts_only_thresholded_dominant_direction():
    scores = np.array(
        [
            [0.10, 0.34, 0.56],  # BUY clears 55 and beats SELL
            [0.56, 0.34, 0.10],  # SELL clears 55 and beats BUY
            [0.20, 0.25, 0.55],  # exact threshold is accepted
            [0.55, 0.10, 0.55],  # directional tie abstains
            [0.20, 0.31, 0.54],  # below threshold abstains
        ],
        dtype=float,
    )

    assert scanner.directional_predictions_from_scores(scores).tolist() == [
        1, -1, 1, 0, 0
    ]


def test_walk_forward_validation_routes_model_predict_to_ungated_pnl():
    """Guard the real standard WF code path, not only a hand-written argmax."""
    source = inspect.getsource(scanner.walk_forward_validation)
    function = ast.parse(source).body[0]

    def is_test_x_model_predict(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "predict"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "test_X"
        )

    pred_assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pred"
            for target in node.targets
        )
        and is_test_x_model_predict(node.value)
    ]
    assert pred_assignments, (
        "standard WF must assign model.predict(test_X) directly to pred"
    )

    pnl_uses_same_pred = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "simulated_pnl"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "pred"
        for node in ast.walk(function)
    )
    assert pnl_uses_same_pred, (
        "standard WF PnL must receive the pred variable assigned from model.predict(test_X)"
    )
