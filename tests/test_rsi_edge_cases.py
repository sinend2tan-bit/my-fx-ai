import ast
from pathlib import Path

import numpy as np
import pandas as pd


def _load_pure_function(filename, function_name):
    source_path = Path(__file__).resolve().parents[1] / filename
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    namespace = {"np": np, "pd": pd}
    isolated_module = ast.Module(body=[function_node], type_ignores=[])
    exec(compile(isolated_module, str(source_path), "exec"), namespace)
    return namespace[function_name]


calculate_app_rsi = _load_pure_function("app.py", "calculate_rsi")
calculate_scanner_rsi = _load_pure_function("scanner.py", "calculate_rsi")


def test_both_rsi_implementations_handle_monotonic_rise_as_100():
    prices = pd.Series(np.arange(1.0, 41.0))
    assert calculate_app_rsi(prices).iloc[-1] == 100.0
    assert calculate_scanner_rsi(prices).iloc[-1] == 100.0


def test_both_rsi_implementations_handle_monotonic_fall_as_zero():
    prices = pd.Series(np.arange(40.0, 0.0, -1.0))
    assert calculate_app_rsi(prices).iloc[-1] == 0.0
    assert calculate_scanner_rsi(prices).iloc[-1] == 0.0


def test_both_rsi_implementations_handle_flat_prices_as_neutral_50():
    prices = pd.Series(np.full(40, 100.0))
    assert calculate_app_rsi(prices).iloc[-1] == 50.0
    assert calculate_scanner_rsi(prices).iloc[-1] == 50.0


def test_rsi_outputs_remain_bounded_for_mixed_prices():
    prices = pd.Series(
        [100, 101, 100.5, 102, 101.2, 103, 102.1, 101.8, 104, 103.2] * 5,
        dtype=float,
    )
    for result in (calculate_app_rsi(prices), calculate_scanner_rsi(prices)):
        finite = result.dropna()
        assert not finite.empty
        assert np.isfinite(finite).all()
        assert finite.between(0.0, 100.0).all()
