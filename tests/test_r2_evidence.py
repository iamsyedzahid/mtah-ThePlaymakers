"""Evidence (R2). Synthetic and hand-built data only."""
import numpy as np
import pytest

from mtah import evidence as ev
from mtah.ball import ball_clean
from mtah.interfaces import Tracking
from mtah.synthetic import make_synthetic

EVIDENCE = [ev.evidence, ev.evidence_reach]
IDS = [f.__name__ for f in EVIDENCE]


@pytest.fixture(scope="module")
def big():
    return make_synthetic(20_000, seed=1)


def static_scene(ball_xy, home_xy, away_xy, n=50):
    """n frames, one home and one away player standing still, the ball fixed."""
    players = np.tile(np.array([home_xy, away_xy], dtype=float), (n, 1, 1))
    return Tracking(np.tile(np.array(ball_xy, dtype=float), (n, 1)), players, np.array([0, 1]))


# -- every evidence function ------------------------------------------------------------------

@pytest.mark.parametrize("fn", EVIDENCE, ids=IDS)
def test_bounded_and_finite(syn, fn):
    e = fn(syn.tracking)
    assert e.shape == (syn.tracking.n_frames,) and e.dtype == np.float64
    assert np.isfinite(e).all()
    assert (np.abs(e) <= 1.0).all()
    assert (e != 0).mean() > 0.3  # it does say something on most trusted frames


@pytest.mark.parametrize("fn", EVIDENCE, ids=IDS)
def test_exactly_zero_on_lost_frames(syn, fn):
    _, lost = ball_clean(syn.tracking.ball)
    assert lost.any()
    assert (fn(syn.tracking)[lost] == 0.0).all()


@pytest.mark.parametrize("fn", EVIDENCE, ids=IDS)
def test_sign_follows_nearer_team_hand_built(fn):
    assert (fn(static_scene([11, 10], [10, 10], [20, 10])) > 0).all()
    assert (fn(static_scene([19, 10], [10, 10], [20, 10])) < 0).all()


@pytest.mark.parametrize("fn", EVIDENCE, ids=IDS)
def test_sign_follows_nearer_team_synthetic(big, fn):
    tr = big.tracking
    filled, lost = ball_clean(tr.ball)
    dist = np.linalg.norm(tr.players - filled[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    near_home, near_away = dist[:, tr.team == 0].min(axis=1), dist[:, tr.team == 1].min(axis=1)
    e = fn(tr)
    clear_home = ~lost & (near_home < 1.0) & (near_away > 4.0)
    clear_away = ~lost & (near_away < 1.0) & (near_home > 4.0)
    assert clear_home.sum() > 200 and clear_away.sum() > 200
    assert (e[clear_home] > 0).mean() > 0.97
    assert (e[clear_away] < 0).mean() > 0.97


@pytest.mark.parametrize("fn", EVIDENCE, ids=IDS)
def test_flipping_teams_flips_the_sign(syn, fn):
    tr = syn.tracking
    flipped = Tracking(tr.ball, tr.players, (1 - tr.team).astype(tr.team.dtype), tr.fps)
    np.testing.assert_array_equal(fn(flipped), -fn(tr))


@pytest.mark.parametrize("fn", EVIDENCE, ids=IDS)
def test_deterministic(fn):
    a, b = make_synthetic(2_000, seed=3), make_synthetic(2_000, seed=3)
    np.testing.assert_array_equal(fn(a.tracking), fn(b.tracking))


@pytest.mark.parametrize("fn", EVIDENCE, ids=IDS)
def test_rejects_other_fps(fn):
    tr = static_scene([11, 10], [10, 10], [20, 10])
    with pytest.raises(NotImplementedError):
        fn(Tracking(tr.ball, tr.players, tr.team, fps=30))


# -- evidence_reach ---------------------------------------------------------------------------

def test_reach_runner_beats_a_nearer_player_standing_still():
    # Home is 6 m away running at the ball at 6 m/s; away is 5 m away and still. Nearest-player
    # evidence says away; time to the ball says home.
    n = 50
    ball = np.tile([50.0, 30.0], (n, 1))
    home = np.column_stack([np.linspace(44.0, 44.0 + 6.0 * (n - 1) / 25, n), np.full(n, 30.0)])
    home -= home[n // 2] - [44.0, 30.0]  # 6 m from the ball at the middle frame
    away = np.tile([50.0, 35.0], (n, 1))
    tr = Tracking(ball, np.stack([home, away], axis=1), np.array([0, 1]))
    mid = n // 2
    assert ev.evidence(tr)[mid] < 0 < ev.evidence_reach(tr)[mid]


def test_reach_ignores_slot_order_and_duplicates(syn):
    tr = syn.tracking
    base = ev.evidence_reach(tr)
    order = np.random.default_rng(0).permutation(tr.team.size)
    np.testing.assert_array_equal(ev.evidence_reach(Tracking(tr.ball, tr.players[:, order], tr.team[order])), base)
    # the tracker duplicates a player into a spare slot of the same team
    dup = np.concatenate([tr.players, tr.players[:, :3]], axis=1)
    np.testing.assert_array_equal(ev.evidence_reach(Tracking(tr.ball, dup, np.r_[tr.team, tr.team[:3]])), base)


def test_reach_zero_when_a_team_is_not_tracked():
    tr = static_scene([11, 10], [10, 10], [np.nan, np.nan])
    assert (ev.evidence_reach(tr) == 0).all()


def test_reach_ignores_impossible_velocity():
    # A slot that teleports 30 m (an identity switch) must not be projected at 60 m/s.
    n = 50
    home = np.tile([10.0, 10.0], (n, 1))
    home[25:] = [40.0, 10.0]
    tr = Tracking(np.tile([41.0, 10.0], (n, 1)), np.stack([home, np.tile([45.0, 10.0], (n, 1))], axis=1),
                  np.array([0, 1]))
    vel = ev._velocity(tr.players)
    assert np.isfinite(vel).all() and (np.linalg.norm(vel, axis=-1) <= ev.MAX_PLAUSIBLE_MS).all()
    assert ev.evidence_reach(tr)[30] > 0  # home is next to the ball after the jump


def test_reach_blocks_do_not_change_the_result(syn, monkeypatch):
    whole = ev.evidence_reach(syn.tracking)
    monkeypatch.setattr(ev, "BLOCK_FRAMES", 97)  # many blocks, odd boundaries
    np.testing.assert_array_equal(ev.evidence_reach(syn.tracking), whole)


def test_velocity_matches_finite_difference():
    n = 40
    pos = np.zeros((n, 1, 2))
    pos[:, 0, 0] = 3.0 * np.arange(n) / 25  # 3 m/s along x
    pos[5, 0] = np.nan  # a hole: neighbours fall back to a one-sided difference
    vel = ev._velocity(pos)
    np.testing.assert_allclose(vel[[0, 10, 20, n - 1], 0, 0], 3.0)
    assert np.isfinite(vel).all()


# -- evidence_shape ---------------------------------------------------------------------------

def shape_scene(n_spells=20, spell=150, seed=0, ball_offset=(0.5, 0.0)):
    """Possession alternates every `spell` frames. The team with the ball spreads wide (y over
    5..63 m) and the other compacts (24..44 m); home plays on the left, so it attacks +x. The ball
    sits next to a player of the owning team and goes missing for 60 frames in the middle of each
    spell, so every lost stretch is surrounded by clear contacts. Returns (tracking, owner, lost),
    owner 0 for home and 1 for away."""
    rng = np.random.default_rng(seed)
    k, T = 10, n_spells * spell
    owner = np.repeat(np.arange(n_spells) % 2, spell)
    home_has = (owner == 0)[:, None]
    wide, tight = np.linspace(5.0, 63.0, k), np.linspace(24.0, 44.0, k)
    players = np.empty((T, 2 * k, 2))
    players[..., 0] = np.r_[np.linspace(10.0, 45.0, k), np.linspace(60.0, 95.0, k)]
    players[:, :k, 1] = np.where(home_has, wide, tight)
    players[:, k:, 1] = np.where(home_has, tight, wide)
    players += rng.normal(0.0, 0.3, size=players.shape)
    carrier = np.where(owner == 0, 3, k + 3)
    ball = players[np.arange(T), carrier] + np.array(ball_offset)
    lost = np.zeros(T, dtype=bool)
    for s in range(n_spells):
        lost[s * spell + 50 : s * spell + 110] = True
    ball[lost] = np.nan
    return Tracking(ball, players, np.repeat([0, 1], k)), owner, lost


def test_shape_reads_possession_from_team_shape():
    tr, owner, lost = shape_scene()
    e = ev.evidence_shape(tr)
    truth = np.where(owner == 0, 1.0, -1.0)
    assert (np.sign(e[lost]) == truth[lost]).mean() > 0.95
    model = ev.shape_logits(*_scan_and_lost(tr), ev.halves_for(tr.n_frames))[1][0]
    assert model.direction == 1 and model.weights is not None
    assert model.weights[ev.SHAPE_FEATURES.index("width")] > 0  # the wider team has the ball


def _scan_and_lost(tr):
    filled, lost = ball_clean(tr.ball)
    return ev.scan_teams(filled, tr.players, tr.team), lost


@pytest.mark.parametrize("scene", ["hand_built", "synthetic"])
def test_shape_only_on_lost_frames_bounded_finite(big, scene):
    tr = shape_scene()[0] if scene == "hand_built" else big.tracking
    _, lost = ball_clean(tr.ball)
    e = ev.evidence_shape(tr)
    assert e.shape == (tr.n_frames,) and e.dtype == np.float64
    assert np.isfinite(e).all() and (np.abs(e) <= 1.0).all()
    assert (e[~lost] == 0.0).all()  # the complement of evidence(): never on a trusted frame
    assert (e[lost] != 0).mean() > 0.9


@pytest.mark.parametrize("scene", ["hand_built", "synthetic"])
def test_shape_flipping_teams_flips_the_sign(big, scene):
    tr = shape_scene()[0] if scene == "hand_built" else big.tracking
    flipped = Tracking(tr.ball, tr.players, (1 - tr.team).astype(tr.team.dtype), tr.fps)
    e = ev.evidence_shape(tr)
    assert (e != 0).any()
    np.testing.assert_array_equal(ev.evidence_shape(flipped), -e)


def test_shape_deterministic_and_order_free(big):
    a, b = make_synthetic(20_000, seed=1), big
    e = ev.evidence_shape(a.tracking)
    assert (e != 0).any()
    np.testing.assert_array_equal(e, ev.evidence_shape(b.tracking))
    tr = big.tracking
    order = np.random.default_rng(1).permutation(tr.team.size)
    shuffled = Tracking(tr.ball, tr.players[:, order], tr.team[order])
    # sums over slots change order, so equal up to rounding, never in sign or size
    np.testing.assert_allclose(ev.evidence_shape(shuffled), ev.evidence_shape(tr), atol=1e-9)


def test_shape_blocks_do_not_change_the_result(big, monkeypatch):
    whole = ev.evidence_shape(big.tracking)
    assert (whole != 0).any()
    monkeypatch.setattr(ev, "BLOCK_FRAMES", 997)
    np.testing.assert_array_equal(ev.evidence_shape(big.tracking), whole)


def test_shape_silent_without_clear_contacts():
    tr = shape_scene(ball_offset=(0.0, 80.0))[0]  # the ball is never near anyone
    scan, lost = _scan_and_lost(tr)
    logit, (model,) = ev.shape_logits(scan, lost, ev.halves_for(tr.n_frames))
    assert (logit == 0).all()
    assert model.weights is None and model.n_home == model.n_away == 0
    assert (ev.evidence_shape(tr) == 0).all()


def test_shape_per_half_and_nothing_outside(big):
    scan, lost = _scan_and_lost(big.tracking)
    halves = ((100, 9_899), (10_100, 19_899))
    logit, models = ev.shape_logits(scan, lost, halves)
    assert [(m.first, m.last) for m in models] == list(halves)
    e = ev.logit_to_evidence(logit)
    outside = np.ones(big.tracking.n_frames, dtype=bool)
    for first, last in halves:
        outside[first : last + 1] = False
    assert all(m.weights is not None for m in models)
    assert (e[outside] == 0).all() and (e[~outside] != 0).any()


def test_halves_are_the_published_ones_on_our_match():
    from mtah.config import GRADED_HALVES, N_FRAMES

    assert ev.halves_for(N_FRAMES) == GRADED_HALVES
    assert ev.halves_for(3_000) == ((0, 2_999),)


def test_fit_logistic_recovers_known_weights():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20_000, 3))
    w_true = np.array([1.5, -0.5, 0.0])
    y = np.where(rng.random(20_000) < 1 / (1 + np.exp(-x @ w_true)), 1.0, -1.0)
    w = ev.fit_logistic(y[:, None] * x, ridge=1.0)
    np.testing.assert_allclose(w, w_true, atol=0.06)
    np.testing.assert_array_equal(ev.fit_logistic(-y[:, None] * -x, ridge=1.0), w)  # team swap


def test_smooth_nan_ignores_and_keeps_missing():
    x = np.array([[1.0], [2.0], [3.0], [np.nan], [5.0]])
    out = ev._smooth_nan(x, 3)[:, 0]
    np.testing.assert_allclose(out[[0, 1, 2, 4]], [1.5, 2.0, 2.5, 5.0])
    assert np.isnan(out[3])
