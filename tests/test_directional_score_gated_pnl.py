"""Verify that forward PnL diagnostics apply the live directional score gate."""
import numpy as np
import pytest

import scanner


def test_score_gate_matches_live_buy_sell_threshold_and_tie_behavior():
    scores = np.array([
        [0.10, 0.20, 0.70],  # BUY above threshold
        [0.65, 0.20, 0.15],  # SELL above threshold
        [0.54, 0.10, 0.55],  # BUY qualifies and beats SELL
        [0.55, 0.10, 0.55],  # tie: no direction
        [0.54, 0.40, 0.06],  # SELL below threshold
        [0.20, 0.60, 0.20],  # neutral dominant
    ])
    assert scanner.directional_predictions_from_scores(scores).tolist() == [1, -1, 1, 0, 0, 0]


def test_score_gate_uses_same_configurable_threshold_for_both_sides():
    scores = np.array([[0.60, 0.20, 0.20], [0.20, 0.20, 0.60]])
    assert scanner.directional_predictions_from_scores(scores, 60).tolist() == [-1, 1]
    assert scanner.directional_predictions_from_scores(scores, 61).tolist() == [0, 0]


@pytest.mark.parametrize(
    "scores",
    [
        np.array([0.2, 0.8, 0.0]),
        np.array([[0.2, 0.8]]),
        np.array([[np.nan, 0.0, 1.0]]),
        np.array([[1.1, 0.0, 0.0]]),
    ],
)
def test_score_gate_rejects_malformed_or_nonfinite_scores(scores):
    with pytest.raises(ValueError):
        scanner.directional_predictions_from_scores(scores)


def test_score_gate_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        scanner.directional_predictions_from_scores(np.array([[0.2, 0.2, 0.6]]), 101)



def test_runtime_equivalent_pnl_gate_is_required_in_addition_to_argmax_gate():
    """A passing argmax PnL cannot compensate for a failing live-equivalent gate."""
    from scanner import _pnl_stability_gate, _combine_pnl_stability_gates

    argmax = _pnl_stability_gate(
        {"trades": 45, "net_pips": 18.0},
        [{"trades": 15, "net_pips": 8.0}, {"trades": 15, "net_pips": 6.0},
         {"trades": 15, "net_pips": 4.0}],
        required_folds=3,
    )
    runtime_equivalent = _pnl_stability_gate(
        {"trades": 0, "net_pips": 0.0},
        [{"trades": 0, "net_pips": 0.0}] * 3,
        required_folds=3,
    )
    combined = _combine_pnl_stability_gates(argmax, runtime_equivalent)

    assert argmax["passed"] is True
    assert runtime_equivalent["passed"] is False
    assert combined["passed"] is False
    assert any("55%スコアゲート" in reason for reason in combined["reasons"])
    # Public gate reasons must not leak private gated trade counts or PnL values.
    assert all("0件" not in reason and "pips" not in reason for reason in combined["reasons"])


def test_runtime_equivalent_pnl_gate_uses_same_conservative_stability_rules():
    """The live-equivalent path must satisfy folds, minimum trades, and positive net."""
    from scanner import _pnl_stability_gate

    passed = _pnl_stability_gate(
        {"trades": 36, "net_pips": 12.0},
        [{"trades": 12, "net_pips": 5.0}, {"trades": 12, "net_pips": 4.0},
         {"trades": 12, "net_pips": 3.0}],
        required_folds=3,
    )
    too_few_trades = _pnl_stability_gate(
        {"trades": 29, "net_pips": 12.0},
        [{"trades": 10, "net_pips": 5.0}, {"trades": 10, "net_pips": 4.0},
         {"trades": 9, "net_pips": 3.0}],
        required_folds=3,
    )

    assert passed["passed"] is True
    assert too_few_trades["passed"] is False
    assert any("取引件数不足" in reason for reason in too_few_trades["reasons"])



def test_directional_gate_requires_runtime_equivalent_pnl_for_each_side():
    from scanner import _combine_directional_pnl_gates

    argmax_passes = {"trades": 12, "net_pips": 4.0}
    runtime_equivalent_fails = {"trades": 2, "net_pips": 1.0}
    runtime_equivalent_passes = {"trades": 12, "net_pips": 2.0}

    assert _combine_directional_pnl_gates(argmax_passes, runtime_equivalent_fails, 10) is False
    assert _combine_directional_pnl_gates(argmax_passes, runtime_equivalent_passes, 10) is True
