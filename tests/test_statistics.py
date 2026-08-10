"""The intervals are the product, so they get tested like the product."""
import pytest
from svgkit import wilson


def test_interval_brackets_the_estimate():
    lo, hi = wilson(0.6, 20)
    assert lo < 0.6 < hi


def test_matches_the_published_baseline_interval():
    """The README prints [0.39, 0.78] for baseline at 12/20. If this drifts, the
    figure and the table stop agreeing with each other."""
    lo, hi = wilson(0.6, 20)
    assert round(lo, 2) == 0.39
    assert round(hi, 2) == 0.78


def test_small_samples_give_wider_intervals():
    narrow = wilson(0.6, 500)
    wide = wilson(0.6, 20)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_degenerate_proportions_stay_in_range(p):
    """Wilson is used precisely because it does not produce a zero-width interval
    at 0 or 1 the way the normal approximation does."""
    lo, hi = wilson(p, 20)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi > lo


def test_interval_never_leaves_zero_to_one():
    for p in (0.05, 0.5, 0.95):
        lo, hi = wilson(p, 7)
        assert lo >= 0.0 and hi <= 1.0
