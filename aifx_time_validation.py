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


def split_hourly_segments(frame):
    """Split a validated hourly DataFrame at every disallowed timestamp gap.

    The input must already be timestamp-normalized, timezone-aware, unique, and
    chronologically ordered. The existing continuity rule is the single source
    of truth, including the accepted weekend market closure.
    """
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("hourly segment input requires a timezone-aware DatetimeIndex")
    timestamps = frame.index
    if timestamps.hasnans:
        raise ValueError("hourly segment index contains NaT")
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("hourly segment index must be unique and increasing")
    if len(frame) == 0:
        return []

    segments = []
    start = 0
    for position in range(1, len(frame)):
        if not forward_hourly_bar_gap_allowed(
            timestamps[position - 1], timestamps[position]
        ):
            segments.append(frame.iloc[start:position].copy())
            start = position
    segments.append(frame.iloc[start:].copy())
    return segments
