"""Small-sample descriptive statistics for model validation diagnostics."""

import math
from numbers import Integral


def wilson_interval(successes, total, z=1.959963984540054):
    """Return a two-sided Wilson score interval for a binomial proportion.

    Returns None when total is zero. Inputs are counts, not percentages.
    The interval is descriptive only; it assumes independent Bernoulli trials.
    """
    if isinstance(total, bool) or not isinstance(total, Integral) or total < 0:
        raise ValueError("total must be a non-negative integer")
    if (
        isinstance(successes, bool)
        or not isinstance(successes, Integral)
        or successes < 0
        or successes > total
    ):
        raise ValueError("successes must be an integer between 0 and total")
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0:
        raise ValueError("z must be a finite positive number")
    if total == 0:
        return None

    n = float(total)
    p = float(successes) / n
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = (
        float(z)
        * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
