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


def hourly_segment_slices(index):
    """Return positional slices split wherever hourly continuity is invalid.

    Input rows must already have passed timestamp validation, duplicate removal,
    and chronological sorting. A market-closure jump accepted by the shared
    continuity rule remains inside the same segment.
    """
    timestamps = pd.DatetimeIndex(index)
    if timestamps.hasnans:
        raise ValueError("hourly segment index contains NaT")
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("hourly segment index must be unique and increasing")
    if len(timestamps) == 0:
        return tuple()
    starts = [0]
    for position in range(1, len(timestamps)):
        if not forward_hourly_bar_gap_allowed(
            timestamps[position - 1], timestamps[position]
        ):
            starts.append(position)
    ends = starts[1:] + [len(timestamps)]
    return tuple(slice(start, end) for start, end in zip(starts, ends))


def split_hourly_segments(frame):
    """Split a validated hourly DataFrame without changing timestamps or order."""
    return [
        frame.iloc[segment].copy()
        for segment in hourly_segment_slices(frame.index)
    ]
