"""Tests for the PFSP opponent pool (numpy/stdlib only — no jax).

Covers: add/load byte round-trip, the Beta-posterior win_rate math, the
50/35/15 schedule proportions, PFSP's p~0.5 preference, exploiter targeting the
worst matchup, persistence across reopen, and the empty-league guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from zappy_rl.algo.league import League


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---- snapshot storage -----------------------------------------------------


def test_add_load_round_trip_bytes(tmp_path):
    league = League(tmp_path)
    blob_a = b"\x00\x01params-A\xff\xfe"
    blob_b = bytes(range(256))  # every byte value, incl. NUL and 0xFF

    id_a = league.add_snapshot(blob_a, step=100, tag="first")
    id_b = league.add_snapshot(blob_b, step=200)

    assert id_a != id_b
    assert league.load_snapshot(id_a) == blob_a
    assert league.load_snapshot(id_b) == blob_b
    # Ids are monotonic, zero-padded, ascending.
    assert league.ids == [id_a, id_b]
    assert id_a == "000000" and id_b == "000001"


def test_load_unknown_snapshot_raises(tmp_path):
    league = League(tmp_path)
    with pytest.raises(KeyError):
        league.load_snapshot("000000")


# ---- win_rate posterior math ----------------------------------------------


def test_win_rate_unplayed_is_one_half(tmp_path):
    league = League(tmp_path)
    sid = league.add_snapshot(b"x", step=0)
    # (0 + 1) / (0 + 2) = 0.5
    assert league.win_rate(sid) == pytest.approx(0.5)


def test_win_rate_posterior_after_results(tmp_path):
    league = League(tmp_path)
    sid = league.add_snapshot(b"x", step=0)
    # 3 wins of 4 games -> (3 + 1) / (4 + 2) = 4/6.
    for won in (True, True, True, False):
        league.record_result(sid, won)
    assert league.win_rate(sid) == pytest.approx(4 / 6)

    # A single win reads 0.667, not a brittle 1.0.
    other = league.add_snapshot(b"y", step=1)
    league.record_result(other, True)
    assert league.win_rate(other) == pytest.approx(2 / 3)


def test_record_result_unknown_raises(tmp_path):
    league = League(tmp_path)
    with pytest.raises(KeyError):
        league.record_result("000000", True)


# ---- schedule proportions -------------------------------------------------


def test_sample_opponent_kind_proportions(tmp_path):
    league = League(tmp_path)
    # Need >1 snapshot so the three kinds resolve to distinguishable ids and the
    # schedule split is exercised independently of opponent selection.
    for i in range(5):
        league.add_snapshot(f"p{i}".encode(), step=i)

    rng = _rng(12345)
    n = 5000
    counts = {"pfsp": 0, "self": 0, "exploiter": 0}
    for _ in range(n):
        _, kind = league.sample_opponent(rng)
        counts[kind] += 1

    assert counts["pfsp"] / n == pytest.approx(0.50, abs=0.05)
    assert counts["self"] / n == pytest.approx(0.35, abs=0.05)
    assert counts["exploiter"] / n == pytest.approx(0.15, abs=0.05)


# ---- PFSP weighting -------------------------------------------------------


def test_pfsp_prefers_coin_flip_opponents(tmp_path):
    """f(p)=p(1-p) should pick the p~0.5 snapshot far more than p~0.05/0.95."""
    league = League(tmp_path)
    hard = league.add_snapshot(b"hard", step=0)   # target p ~ 0.5
    easy = league.add_snapshot(b"easy", step=1)    # target p ~ 0.95 (learner crushes)
    tough = league.add_snapshot(b"tough", step=2)  # target p ~ 0.05 (learner loses)

    # Drive the posteriors toward the intended win-rates with skewed records.
    for _ in range(50):  # hard: 25W/25L -> p ~ 0.5
        league.record_result(hard, True)
        league.record_result(hard, False)
    for _ in range(95):
        league.record_result(easy, True)
    for _ in range(5):
        league.record_result(easy, False)
    for _ in range(5):
        league.record_result(tough, True)
    for _ in range(95):
        league.record_result(tough, False)

    # Sanity on the posteriors themselves.
    assert league.win_rate(hard) == pytest.approx(0.5, abs=0.02)
    assert league.win_rate(easy) > 0.9
    assert league.win_rate(tough) < 0.1

    rng = _rng(7)
    n = 5000
    picks = {hard: 0, easy: 0, tough: 0}
    for _ in range(n):
        sid = league._sample_pfsp(rng)
        picks[sid] += 1

    # f(0.5)=0.25 vastly exceeds f(0.05)=f(0.95)=0.0475 -> hard dominates.
    assert picks[hard] / n > 0.6
    assert picks[hard] > picks[easy] * 3
    assert picks[hard] > picks[tough] * 3


def test_pfsp_uniform_fallback_when_all_weights_zero(tmp_path):
    """If every snapshot reads p in {0,1}, fall back to uniform (no crash)."""
    league = League(tmp_path)
    ids = [league.add_snapshot(f"p{i}".encode(), step=i) for i in range(3)]
    # Monkeypatch win_rate to force a degenerate all-zero weight vector.
    league.win_rate = lambda sid: 1.0  # type: ignore[assignment]

    rng = _rng(3)
    picks = {sid: 0 for sid in ids}
    for _ in range(3000):
        picks[league._sample_pfsp(rng)] += 1
    # Uniform over 3 -> each ~1/3.
    for sid in ids:
        assert picks[sid] / 3000 == pytest.approx(1 / 3, abs=0.05)


# ---- exploiter targeting --------------------------------------------------


def test_exploiter_picks_lowest_win_rate(tmp_path):
    league = League(tmp_path)
    a = league.add_snapshot(b"a", step=0)
    b = league.add_snapshot(b"b", step=1)
    c = league.add_snapshot(b"c", step=2)

    # Learner wins most vs a and c, loses most vs b -> b is the worst matchup.
    for _ in range(10):
        league.record_result(a, True)
        league.record_result(c, True)
    for _ in range(10):
        league.record_result(b, False)

    assert league.win_rate(b) < league.win_rate(a)
    assert league.win_rate(b) < league.win_rate(c)

    rng = _rng(0)
    # Force the exploiter schedule (u in [0.85, 1.0)) and check the target.
    seen = set()
    for _ in range(200):
        sid, kind = league.sample_opponent(rng)
        if kind == "exploiter":
            seen.add(sid)
    assert seen == {b}


def test_single_snapshot_always_returned(tmp_path):
    league = League(tmp_path)
    only = league.add_snapshot(b"solo", step=0)
    rng = _rng(1)
    kinds = set()
    for _ in range(300):
        sid, kind = league.sample_opponent(rng)
        assert sid == only  # all three schedules collapse to the lone snapshot
        kinds.add(kind)
    # but the schedule label still varies (telemetry stays meaningful).
    assert kinds == {"pfsp", "self", "exploiter"}


# ---- persistence ----------------------------------------------------------


def test_persistence_round_trip_after_reopen(tmp_path):
    league = League(tmp_path)
    blob0 = b"snapshot-zero"
    blob1 = b"snapshot-one\x00\x01"
    sid0 = league.add_snapshot(blob0, step=10, tag="alpha")
    sid1 = league.add_snapshot(blob1, step=20, tag="beta")
    for won in (True, False, True):
        league.record_result(sid0, won)
    league.record_result(sid1, False)

    # Reopen on the same dir — everything restores from disk.
    reopened = League(tmp_path)
    assert reopened.ids == [sid0, sid1]
    assert reopened.load_snapshot(sid0) == blob0
    assert reopened.load_snapshot(sid1) == blob1
    assert reopened.win_rate(sid0) == pytest.approx(3 / 5)  # 2W/3G -> 3/5
    assert reopened.win_rate(sid1) == pytest.approx(1 / 3)  # 0W/1G -> 1/3

    # next_id is preserved: a new snapshot keeps climbing, never reuses an id.
    sid2 = reopened.add_snapshot(b"snapshot-two", step=30)
    assert sid2 == "000002"


def test_empty_league_sample_raises(tmp_path):
    league = League(tmp_path)
    with pytest.raises(ValueError):
        league.sample_opponent(_rng(0))
