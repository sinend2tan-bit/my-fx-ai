"""Walk-forward metrics must make missing test-fold classes explicit.

This test uses deterministic synthetic rows and a fixed estimator. It does not
download market data, fit a real model, or change validation thresholds.
"""

import numpy as np
import pandas as pd

import scanner


class _AlwaysSellClassifier:
    """Deterministic stand-in that always predicts SELL."""

    def __init__(self, **_kwargs):
        self.classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(np.asarray(y, dtype=int))
        self.feature_importances_ = np.full(
            X.shape[1], 1.0 / X.shape[1], dtype=float
        )
        return self

    def predict(self, X):
        return np.full(len(X), -1, dtype=int)

    def predict_proba(self, X):
        probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)
        sell_column = int(np.where(self.classes_ == -1)[0][0])
        probabilities[:, sell_column] = 1.0
        return probabilities


def _synthetic_walk_forward_frame():
    rows = 240
    index = pd.date_range("2026-01-05", periods=rows, freq="h", tz="UTC")
    data = {
        feature: np.linspace(0.0, 1.0, rows)
        for feature in scanner.FEATURE_COLUMNS
    }
    labels = np.tile(np.array([-1, 0, 1], dtype=int), rows // 3)

    # First test fold is rows 60..119 and deliberately has no NEUTRAL labels.
    labels[60:120] = np.tile(np.array([-1, 1], dtype=int), 30)
    data["Target"] = labels
    data["ATR"] = np.ones(rows, dtype=float)
    data["ADX"] = np.full(rows, 20.0, dtype=float)
    data["ATR_Ratio"] = np.full(rows, 0.01, dtype=float)
    data["Future_Open_1"] = np.full(rows, 150.0, dtype=float)
    data["Future_Close"] = np.full(rows, 150.0, dtype=float)

    for step in range(1, scanner.LOOKAHEAD_BARS + 1):
        data[f"Future_High_{step}"] = np.full(rows, 150.1, dtype=float)
        data[f"Future_Low_{step}"] = np.full(rows, 149.9, dtype=float)

    return pd.DataFrame(data, index=index)


def test_walk_forward_reports_missing_fold_class_without_hiding_recall(
    monkeypatch,
):
    monkeypatch.setattr(
        scanner, "RandomForestClassifier", _AlwaysSellClassifier
    )
    monkeypatch.setattr(scanner, "MIN_TRAIN_ROWS", 20)
    monkeypatch.setattr(scanner, "N_WALK_FORWARD_FOLDS", 3)

    result = scanner.walk_forward_validation(
        _synthetic_walk_forward_frame(),
        pair_name="unknown synthetic pair",
    )

    assert result["folds"] == 3
    first_fold = result["fold_details"][0]

    # The first test fold contains SELL and BUY only. sklearn's standard
    # balanced accuracy averages the classes present in y_true: (1.0 + 0.0)/2.
    assert first_fold["actual_class_counts"] == {
        "sell": 30,
        "neutral": 0,
        "buy": 30,
    }
    assert first_fold["balanced_accuracy"] == 0.5

    # Recall reporting is deliberately fixed to all three classes, so the
    # absent NEUTRAL class remains visible as zero rather than disappearing.
    assert first_fold["class_recall_pct"] == {
        "sell": 100.0,
        "neutral": 0.0,
        "buy": 0.0,
    }

    # The pooled evaluation contains all three classes; aggregate recall
    # therefore still reports NEUTRAL explicitly.
    assert result["class_counts"]["actual"]["neutral"] > 0
    assert result["class_recall_pct"]["neutral"] == 0.0
