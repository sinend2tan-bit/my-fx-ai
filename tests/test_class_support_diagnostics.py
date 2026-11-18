import pytest

from aifx_statistics import wilson_interval


def test_wilson_interval_returns_none_for_zero_support():
    assert wilson_interval(0, 0) is None


def test_wilson_interval_handles_zero_and_full_successes():
    low = wilson_interval(0, 10)
    high = wilson_interval(10, 10)
    assert low[0] == 0.0
    assert low[1] == pytest.approx(0.2775, abs=0.001)
    assert high[0] == pytest.approx(0.7225, abs=0.001)
    assert high[1] == pytest.approx(1.0)


def test_wilson_interval_is_symmetric_for_half_successes():
    lower, upper = wilson_interval(5, 10)
    assert lower == pytest.approx(0.2366, abs=0.001)
    assert upper == pytest.approx(0.7634, abs=0.001)


@pytest.mark.parametrize(
    ("successes", "total"),
    [(-1, 10), (11, 10), (0, -1), (1.5, 10), (True, 10), (0, True)],
)
def test_wilson_interval_rejects_invalid_counts(successes, total):
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def test_wilson_interval_rejects_invalid_z():
    with pytest.raises(ValueError):
        wilson_interval(1, 2, z=float("nan"))
