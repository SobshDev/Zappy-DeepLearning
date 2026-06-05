"""Regression tests for the speedrun driver's BEST comparison.

Pins the survivorship-free scorer: ``median_all`` is the median time-to-win
over ALL eval episodes with failures = +inf (finite iff rate > 50%), so it
compares across different win rates and balances reliability against speed.
The original bug compared winners-only medians and scored an 11.9%->92.0%
win-rate jump as "no improvement", stopping the campaign after two segments.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from speedrun_train import beats  # noqa: E402


def ev(rate, median_all, median=None):
    return {"rate": rate, "median": median, "median_all": median_all}


def test_majority_win_beats_minority_win_outright():
    # the exact campaign-stopping case: 11.9% baseline (median_all = inf)
    # vs 92.0% candidate — must be a NEW BEST regardless of winners-median
    assert beats(ev(0.920, 3600.0, median=3506.0),
                 ev(0.119, None, median=3491.0))


def test_minority_win_never_beats_majority_win():
    assert not beats(ev(0.40, None, median=2200.0),
                     ev(0.97, 3500.0, median=3476.0))


def test_median_all_decides_between_majority_win_runs():
    best = ev(0.969, 3500.0)
    # >=1% faster on the full distribution -> better
    assert beats(ev(0.960, 3450.0), best)
    # <1% faster -> not better
    assert not beats(ev(0.969, 3470.0), best)
    # slower -> not better
    assert not beats(ev(0.980, 3550.0), best)


def test_rate_floor_blocks_reliability_collapse():
    # much faster median_all cannot buy a >2pp win-rate drop
    assert not beats(ev(0.90, 3200.0), ev(0.969, 3500.0))
    # at exactly -2pp the floor does not trip; median_all decides
    assert beats(ev(0.949, 3400.0), ev(0.969, 3500.0))


def test_slow_win_degenerate_is_rejected():
    # higher rate but slower on the full distribution -> NOT better
    # (the old rate-first rule accepted this; review-pinned)
    assert not beats(ev(0.989, 3700.0), ev(0.969, 3500.0))


def test_minority_vs_minority_compares_rates():
    assert beats(ev(0.30, None), ev(0.10, None))
    assert not beats(ev(0.11, None), ev(0.10, None))
    assert not beats(ev(0.30, None), ev(0.29, None))  # inside +2pp band


def test_never_winning_candidate_never_beats_majority_win():
    assert not beats(ev(0.0, None), ev(0.969, 3500.0))
