"""R2 control evidence. Synthetic data only."""
import inspect

import numpy as np
import pytest

from mtah import evidence as ev
from mtah import interfaces
from mtah.ball import ball_clean
from mtah.interfaces import Tracking
from mtah.synthetic import make_synthetic

VARIANTS = [ev.evidence, ev.evidence_weighted, ev.evidence_speed_aware]


def _duel(ball_x, n=30, home_x=0.0, away_x=10.0):
    """One home player at home_x, one away player at away_x, the ball standing at ball_x."""
    players = np.zeros((n, 2, 2))
    players[:, 0, 0], players[:, 1, 0] = home_x, away_x
    players[..., 1] = 30.0
    ball = np.column_stack([np.full(n, ball_x), np.full(n, 30.0)])
    return Tracking(ball, players, np.array([0, 1]))


@pytest.mark.parametrize("fn", VARIANTS)
def test_signature_matches_interface(fn):
    def shape(f):
        return [(p.name, p.kind, p.default) for p in inspect.signature(f).parameters.values()]

    assert shape(fn) == shape(interfaces.evidence)


@pytest.mark.parametrize("fn", VARIANTS)
def test_sign_follows_the_nearer_team(fn):
    assert (fn(_duel(0.5)) > 0.5).all()  # home 0.5 m away, away 9.5 m
    assert (fn(_duel(9.5)) < -0.5).all()


@pytest.mark.parametrize("fn", VARIANTS)
def test_flipping_the_team_array_flips_the_sign(fn):
    tr = _duel(3.0)
    flipped = Tracking(tr.ball, tr.players, 1 - tr.team)
    np.testing.assert_allclose(fn(flipped), -fn(tr))


@pytest.mark.parametrize("fn", VARIANTS)
def test_bounded_finite_and_deterministic(fn, syn):
    e = fn(syn.tracking)
    assert e.shape == (syn.tracking.n_frames,) and e.dtype == np.float64
    assert np.isfinite(e).all() and (np.abs(e) <= 1).all()
    np.testing.assert_array_equal(e, fn(syn.tracking))


@pytest.mark.parametrize("fn", VARIANTS)
def test_lost_ball_is_exactly_zero(fn, syn):
    _, lost = ball_clean(syn.tracking.ball)
    assert lost.any()
    assert (fn(syn.tracking)[lost] == 0).all()
    all_lost = Tracking(np.full((10, 2), np.nan), np.zeros((10, 2, 2)), np.array([0, 1]))
    assert (fn(all_lost) == 0).all()


@pytest.mark.parametrize("fn", VARIANTS)
def test_missing_players_are_ignored(fn):
    tr = _duel(0.5)
    tr.players[:, 1] = np.nan  # the away player is never tracked
    assert (fn(tr) > 0.5).all()
    tr.players[:, 0] = np.nan  # nobody tracked at all: no information
    assert (fn(tr) == 0).all()


@pytest.mark.parametrize("fn", VARIANTS)
def test_loose_ball_fades_to_zero(fn):
    assert (np.abs(fn(_duel(50.0, home_x=30.0, away_x=75.0))) < 0.05).all()  # nobody within 20 m


def test_fast_ball_is_no_information_only_in_the_speed_aware_version():
    n = 30
    tr = _duel(0.0, n=n)
    tr.ball[:, 0] = 0.5 + np.arange(n) * 0.04  # rolling past the home player at 1 m/s
    slow = ev.evidence_speed_aware(tr)
    tr.ball[:, 0] = 0.5 + np.arange(n) * 0.8  # flying at 20 m/s
    fast = ev.evidence_speed_aware(tr)
    assert (slow[5:-5] != 0).all() and (fast[5:-5] == 0).all()


def test_only_25_fps():
    tr = _duel(0.5)
    with pytest.raises(NotImplementedError):
        ev.evidence(Tracking(tr.ball, tr.players, tr.team, fps=30))


@pytest.mark.parametrize("fn", VARIANTS)
def test_sign_agrees_with_synthetic_possession(fn):
    # A whole synthetic match with pressing, dribbling and deflections: where the evidence is
    # confident, it should name the team that has the ball.
    syn = make_synthetic(50_000, seed=1)
    e = fn(syn.tracking)
    live = np.isin(syn.labels, ["H", "A"]) & (np.abs(e) > 0.5)
    agree = np.mean((e[live] > 0) == (syn.labels[live] == "H"))
    assert agree > 0.85, agree


def test_weighted_counts_a_player_tracked_twice_once():
    # Home player tracked in two slots at the same spot, away player as far on the other side.
    ball = np.tile([50.0, 30.0], (30, 1))
    players = np.zeros((30, 3, 2))
    players[:, 0] = players[:, 1] = [47.0, 30.0]
    players[:, 2] = [53.0, 30.0]
    e = ev.evidence_weighted(Tracking(ball, players, np.array([0, 0, 1])))
    np.testing.assert_allclose(e, 0.0, atol=1e-9)
