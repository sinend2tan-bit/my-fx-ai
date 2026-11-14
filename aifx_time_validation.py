"""Time-continuity checks for hourly FX forecast evaluation.

The rule intentionally matches scanner._forward_bar_gap_allowed:
- consecutive one-hour bars are accepted;
- a Friday/Saturday-to-Monday market-closure jump up to 80 hours is accepted;
- other gaps, duplicate timestamps, and reverse ordering are rejected.
"""

import pandas as pd


def forward_hourly_bar_gap_allowed(prev_ts, next_ts):
    """Return whether two adjacent hourly bars form an allowed time step."""
    diff = next_ts - prev_ts
    if diff <= pd.Timedelta(0):
        return False
    if diff == pd.Timedelta(hours=1):
        return True
    if prev_ts.weekday() in (4, 5) and next_ts.weekday() == 0:
        return diff <= pd.Timedelta(hours=80)
    return False


def has_contiguous_hourly_window(index):
    """Return True only when every adjacent pair in an hourly window is valid."""
    timestamps = pd.DatetimeIndex(index)
    return all(
        forward_hourly_bar_gap_allowed(timestamps[i], timestamps[i + 1])
        for i in range(len(timestamps) - 1)
    )
