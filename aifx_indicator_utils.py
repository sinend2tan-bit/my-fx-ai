"""Shared helpers for minimum-history EMA eligibility."""


def ema_with_warmup(series, span):
    """Return EMA values only after at least span observations.

    Pandas otherwise exposes the recursively seeded EMA from the first row.
    min_periods hides early outputs, but it is an eligibility cutoff, not a
    convergence guarantee: with adjust=False, the first observation still
    influences the first visible output. For span=200, that seed contribution
    is about 13.67% at observation 200 and falls below 1% only after 462
    observations.
    """
    if isinstance(span, bool) or not isinstance(span, int) or span < 1:
        raise ValueError("span must be a positive integer")
    return series.ewm(span=span, min_periods=span, adjust=False).mean()
