"""Characterization tests for App 1h label windows across missing timestamps.

Uses only deterministic synthetic OHLCV. This records the current behavior
before any production-code change; it does not alter labels or model logic.
"""

import pandas as pd

from test_stage1_input_validation import _app_namespace, make_frame


def test_app_1h_label_window_currently_counts_rows_across_weekday_gap():
    raw = make_frame(n=500, start="2026-01-05 00:00")
    gap_ts = raw.index[260]
    frame = raw.drop(index=[gap_ts])

    ns, _st = _app_namespace(lambda: frame.copy())
    result = ns["load_and_process_data"]("USDJPY=X", "60d", "1h")

    assert result is not None
    previous_ts = raw.index[259]
    next_ts = raw.index[261]
    assert next_ts - previous_ts == pd.Timedelta(hours=2)

    # Characterize the gap: App currently retains the signal row and gives it
    # a Target based on the next eight available rows, without invalidating
    # the window for the missing weekday hour.
    assert previous_ts in result.index
    assert result.loc[previous_ts, "Target"] in (-1, 0, 1)
    quality = result.attrs["data_quality"]
    assert quality["dropped_label_discontinuous"] == 0
