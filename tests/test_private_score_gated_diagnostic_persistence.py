"""Ensure local-only 55% score-gated diagnostics never enter public scan state."""
import ast
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "scanner.py"


def _extract(name, globals_=None):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = dict(globals_ or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace[name]


def test_private_score_gated_diagnostic_is_opt_in(monkeypatch):
    import os

    enabled = _extract("_private_score_gated_pnl_diagnostic_enabled", {"os": os})
    monkeypatch.delenv("AIFX_PRIVATE_SCORE_GATED_PNL_DIAGNOSTIC", raising=False)
    assert enabled() is False
    monkeypatch.setenv("AIFX_PRIVATE_SCORE_GATED_PNL_DIAGNOSTIC", "1")
    assert enabled() is True
    monkeypatch.setenv("AIFX_PRIVATE_SCORE_GATED_PNL_DIAGNOSTIC", "true")
    assert enabled() is False


def test_private_diagnostic_fields_are_removed_recursively_without_mutating_input():
    strip = _extract("_strip_private_diagnostics")
    source = [{
        "name": "USDJPY=X",
        "validation": {
            "simulated_pnl": {"net_pips": 12.0},
            "simulated_pnl_after_55pct_score_gate_diagnostic": {"net_pips": 3.0},
            "fold_details": [{
                "simulated_pnl": {"net_pips": 4.0},
                "simulated_pnl_after_55pct_score_gate_diagnostic": {"net_pips": 1.0},
            }],
        },
    }]
    safe = strip(source)
    assert safe[0]["validation"]["simulated_pnl"] == {"net_pips": 12.0}
    assert "simulated_pnl_after_55pct_score_gate_diagnostic" not in safe[0]["validation"]
    assert "simulated_pnl_after_55pct_score_gate_diagnostic" not in safe[0]["validation"]["fold_details"][0]
    assert source[0]["validation"]["simulated_pnl_after_55pct_score_gate_diagnostic"]["net_pips"] == 3.0


def test_public_scan_state_uses_recursive_private_diagnostic_filter():
    source = APP_PATH.read_text(encoding="utf-8")
    assert '"pairs": _strip_private_diagnostics(pair_results)' in source
    assert '"simulated_pnl_after_55pct_score_gate_diagnostic"' not in source[source.index('compact_pairs = []'):source.index('run_record = {')]


def test_walk_forward_only_calculates_gated_pnl_when_private_opt_in_is_enabled():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    function = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "walk_forward_validation"
    )
    source = ast.unparse(function)
    assert "if private_diagnostic_enabled" in source
    assert "directional_predictions_from_scores" in source
