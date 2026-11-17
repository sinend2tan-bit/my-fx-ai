"""Synthetic-only tests for fail-closed walk-forward fold completeness."""

import scanner


def _fold(fold, sell=5, neutral=5, buy=5):
    return {
        "fold": fold,
        "actual_class_counts": {
            "sell": sell,
            "neutral": neutral,
            "buy": buy,
        },
        "predicted_class_counts": {
            "sell": 4,
            "neutral": 7,
            "buy": 4,
        },
        "balanced_accuracy": 0.4 if sell and neutral and buy else None,
        "baseline_balanced_accuracy": 0.34 if sell and neutral and buy else None,
        "balanced_accuracy_reason": (
            "" if sell and neutral and buy
            else "test fold missing true class support"
        ),
        "confusion_matrix": [[3, 1, 1], [1, 3, 1], [1, 1, 3]],
        "class_recall_pct": {
            "sell": 60.0 if sell else None,
            "neutral": 60.0 if neutral else None,
            "buy": 60.0 if buy else None,
        },
    }


def test_fold_completeness_passes_only_when_every_configured_fold_is_complete():
    details = [_fold(1), _fold(2), _fold(3)]
    result = scanner._fold_completeness_diagnostics(details, required_folds=3)

    assert result["complete"] is True
    assert result["missing_folds"] == []
    assert result["incomplete_folds"] == []


def test_pooled_all_class_support_does_not_hide_incomplete_individual_fold():
    # Pooled actual labels contain all classes, but fold 2 has no BUY examples.
    details = [
        _fold(1, sell=5, neutral=5, buy=5),
        _fold(2, sell=5, neutral=5, buy=0),
        _fold(3, sell=5, neutral=5, buy=5),
    ]
    pooled = {
        name: sum(item["actual_class_counts"][name] for item in details)
        for name in ("sell", "neutral", "buy")
    }
    assert all(value > 0 for value in pooled.values())

    result = scanner._fold_completeness_diagnostics(details, required_folds=3)

    assert result["complete"] is False
    assert [item["fold"] for item in result["incomplete_folds"]] == [2]
    assert result["incomplete_folds"][0]["missing_true_classes"] == ["buy"]


def test_skipped_fold_cannot_be_hidden_by_other_successful_folds():
    details = [_fold(1), _fold(3)]

    result = scanner._fold_completeness_diagnostics(details, required_folds=3)

    assert result["complete"] is False
    assert result["missing_folds"] == [2]


def test_duplicate_or_unexpected_fold_ids_fail_closed():
    details = [_fold(1), _fold(2), _fold(2), _fold(4)]

    result = scanner._fold_completeness_diagnostics(details, required_folds=3)

    assert result["complete"] is False
    assert result["duplicate_folds"] == [2]
    assert result["unexpected_folds"] == [4]


def test_undefined_fold_or_baseline_metric_fails_closed_even_with_all_classes_present():
    details = [_fold(1), _fold(2), _fold(3)]
    details[1]["balanced_accuracy"] = None
    details[1]["baseline_balanced_accuracy"] = None

    result = scanner._fold_completeness_diagnostics(details, required_folds=3)

    assert result["complete"] is False
    assert [item["fold"] for item in result["incomplete_folds"]] == [2]
