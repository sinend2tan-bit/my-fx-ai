"""Shared indicator helpers for avoiding partially warmed long-period EMAs."""


def ema_with_warmup(series, span):
    """Return an EMA only after at least span observations.

    Pandas otherwise emits seeded EMA values from the first row. Those early
    values make a nominal EMA200 depend too strongly on the arbitrary start of
    the downloaded window and contaminate historical training rows.
    """
    if isinstance(span, bool) or not isinstance(span, int) or span < 1:
        raise ValueError("span must be a positive integer")
    return series.ewm(span=span, min_periods=span, adjust=False).mean()
