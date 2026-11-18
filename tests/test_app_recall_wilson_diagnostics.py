"""Wilson interval diagnostics for App class-recall reporting (no gate changes)."""
import ast
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _wilson():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    helper = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_wilson_interval"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace["_wilson_interval"]


def test_wilson_interval_for_small_samples_is_bounded_and_non_degenerate():
    interval = _wilson()
    low, high = interval(1, 1)
    assert 0.0 < low < 1.0
    assert high == 1.0
    low, high = interval(0, 1)
    assert low == 0.0
    assert 0.0 < high < 1.0


def test_wilson_interval_for_large_sample_contains_observed_rate():
    interval = _wilson()
    low, high = interval(90, 100)
    assert 0.0 <= low < 0.9 < high <= 1.0


def test_wilson_interval_returns_none_for_empty_class_support():
    assert _wilson()(0, 0) is None


def test_wilson_interval_rejects_invalid_counts():
    interval = _wilson()
    for successes, total in ((-1, 10), (11, 10), (0, -1)):
        try:
            interval(successes, total)
        except ValueError:
            pass
        else:
            raise AssertionError((successes, total))


def test_diagnostics_are_collapsed_and_do_not_change_validation_gate():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'with st.expander("📊 検証データの信頼度（参考）", expanded=False):' in source
    assert '"class_recall_wilson_pct": class_recall_wilson_pct' in source
    assert "if not validation_complete:" in source
    assert "model_quality_ok = bool(" in source
