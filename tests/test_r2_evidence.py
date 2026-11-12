"""Evidence (R2). Synthetic and hand-built data only, never the match or the open games.

What the pipelines use: evidence() (the nearest player) in the rule-only pipeline, and for
mtah/fusion.py the time-to-ball lead and the per-half shape model, through mtah/r2_features.py.
"""
import numpy as np
import pytest

from mtah import evidence as ev
from mtah import r2_features as rf
from mtah.ball import ball_clean
from mtah.config import PITCH_M
from mtah.interfaces import AWAY, HOME, Tracking
from mtah.synthetic import make_synthetic


@pytest.fixture(scope="module")
def big():
    # 30,000 frames: the local synthetic match spends about a quarter of its time in stoppages, so
    # 20,000 left one team just under the 200 clear contacts the sign test needs.
    return make_synthetic(30_000, seed=1)


def static_scene(ball_xy, home_xy, away_xy, n=50):
    """n frames, one home and one away player standing still, the ball fixed."""
    players = np.tile(np.array([home_xy, away_xy], dtype=float), (n, 1, 1))
    return Tracking(np.tile(np.array(ball_xy, dtype=float), (n, 1)), players, np.array([0, 1]))


def _scan_and_lost(tr):
    filled, lost = ball_clean(tr.ball)
    return ev.scan_teams(filled, tr.players, tr.team), lost


def whole(tr):
    """The whole match as one half, inclusive."""
    return ((0, tr.n_frames - 1),)


def reach_lead(tr):
    """Home's time lead to the ball in seconds: the r2_reach_lead column."""
    scan, lost = _scan_and_lost(tr)
    return ev.time_lead(scan.t_best[:, HOME], scan.t_best[:, AWAY], lost)


def shape_lost(tr):
    """The shape model's logit on lost frames, 0 elsewhere: the r2_shape_lost column, uncapped."""
    return ev.shape_logits(*_scan_and_lost(tr), whole(tr))[0]


EVIDENCE = [ev.evidence, reach_lead]
IDS = [f.__name__ for f in EVIDENCE]
BOUND = {"evidence": 1.0, "reach_lead": ev.TIME_CAP_S}


# -- evidence() and the time-to-ball lead -------------------------------------------------------

@pytest.mark.parametrize("fn", EVIDENCE, ids=IDS)
def test_bounded_and_finite(syn, fn):
    e = fn(syn.tracking)
    assert e.shape == (syn.tracking.n_frames,) and e.dtype == np.float64
    assert np.isfinite(e).all()
    assert (np.abs(e) <= BOUND[fn.__name__]).all()
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
    # A ball at a player's feet (slow), not one in flight: a pass skimming past a player is on
    # its way to someone else, and reach rightly follows where the ball is going.
    tr = big.tracking
    filled, lost = ball_clean(tr.ball)
    dist = np.linalg.norm(tr.players - filled[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    near_home, near_away = dist[:, tr.team == 0].min(axis=1), dist[:, tr.team == 1].min(axis=1)
    e = fn(tr)
    slow = np.linalg.norm(ev.ball_velocity(filled), axis=1) < 3.0
    clear_home = ~lost & slow & (near_home < 1.0) & (near_away > 4.0)
    clear_away = ~lost & slow & (near_away < 1.0) & (near_home > 4.0)
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


def test_evidence_rejects_other_fps():
    tr = static_scene([11, 10], [10, 10], [20, 10])
    with pytest.raises(NotImplementedError):
        ev.evidence(Tracking(tr.ball, tr.players, tr.team, fps=30))


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
    assert ev.evidence(tr)[mid] < 0 < reach_lead(tr)[mid]


def test_reach_keeps_the_ball_with_a_dribbler():
    # Home dribbles at 5 m/s with the ball 0.5 m ahead; an away defender stands 2 m to the side.
    # With a still ball the dribbler would be projected 3.5 m past it and lose it to the defender.
    n = 50
    t = np.arange(n) / 25
    home = np.column_stack([30.0 + 5.0 * t, np.full(n, 30.0)])
    ball = home + [0.5, 0.0]
    away = np.tile([32.0 + 5.0 * t[n // 2], 32.0], (n, 1))
    tr = Tracking(ball, np.stack([home, away], axis=1), np.array([0, 1]))
    assert (reach_lead(tr)[10:40] > 0).all()


def test_reach_ignores_slot_order_and_duplicates(syn):
    tr = syn.tracking
    base = reach_lead(tr)
    order = np.random.default_rng(0).permutation(tr.team.size)
    np.testing.assert_array_equal(reach_lead(Tracking(tr.ball, tr.players[:, order], tr.team[order])), base)
    # the tracker duplicates a player into a spare slot of the same team
    dup = np.concatenate([tr.players, tr.players[:, :3]], axis=1)
    np.testing.assert_array_equal(reach_lead(Tracking(tr.ball, dup, np.r_[tr.team, tr.team[:3]])), base)


def test_reach_zero_when_a_team_is_not_tracked():
    tr = static_scene([11, 10], [10, 10], [np.nan, np.nan])
    assert (reach_lead(tr) == 0).all()


def test_reach_ignores_impossible_velocity():
    # A slot that teleports 30 m (an identity switch) must not be projected at 60 m/s.
    n = 50
    home = np.tile([10.0, 10.0], (n, 1))
    home[25:] = [40.0, 10.0]
    tr = Tracking(np.tile([41.0, 10.0], (n, 1)), np.stack([home, np.tile([45.0, 10.0], (n, 1))], axis=1),
                  np.array([0, 1]))
    vel = ev._velocity(tr.players)
    assert np.isfinite(vel).all() and (np.linalg.norm(vel, axis=-1) <= ev.MAX_PLAUSIBLE_MS).all()
    assert reach_lead(tr)[30] > 0  # home is next to the ball after the jump


def test_reach_blocks_do_not_change_the_result(syn, monkeypatch):
    whole_match = reach_lead(syn.tracking)
    monkeypatch.setattr(ev, "BLOCK_FRAMES", 97)  # many blocks, odd boundaries
    np.testing.assert_array_equal(reach_lead(syn.tracking), whole_match)


def test_velocity_matches_finite_difference():
    n = 40
    pos = np.zeros((n, 1, 2))
    pos[:, 0, 0] = 3.0 * np.arange(n) / 25  # 3 m/s along x
    pos[5, 0] = np.nan  # a hole: neighbours fall back to a one-sided difference
    vel = ev._velocity(pos)
    np.testing.assert_allclose(vel[[0, 10, 20, n - 1], 0, 0], 3.0)
    assert np.isfinite(vel).all()


def test_ball_velocity():
    n = 30
    ball = np.column_stack([np.arange(n) * 10.0 / 25, np.zeros(n)])  # 10 m/s along x
    np.testing.assert_allclose(ev.ball_velocity(ball), np.tile([10.0, 0.0], (n, 1)))


# -- the shape model ----------------------------------------------------------------------------

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
    logit, (model,) = ev.shape_logits(*_scan_and_lost(tr), whole(tr))
    truth = np.where(owner == 0, 1.0, -1.0)
    assert (np.sign(logit[lost]) == truth[lost]).mean() > 0.95
    assert model.direction == 1 and model.weights is not None
    assert model.weights[ev.SHAPE_FEATURES.index("width")] > 0  # the wider team has the ball


@pytest.mark.parametrize("scene", ["hand_built", "synthetic"])
def test_shape_only_on_lost_frames_and_finite(big, scene):
    tr = shape_scene()[0] if scene == "hand_built" else big.tracking
    _, lost = ball_clean(tr.ball)
    logit = shape_lost(tr)
    assert logit.shape == (tr.n_frames,) and logit.dtype == np.float64
    assert np.isfinite(logit).all()
    assert (logit[~lost] == 0.0).all()  # the complement of evidence(): never on a trusted frame
    assert (logit[lost] != 0).mean() > 0.9


@pytest.mark.parametrize("scene", ["hand_built", "synthetic"])
def test_shape_flipping_teams_flips_the_sign(big, scene):
    tr = shape_scene()[0] if scene == "hand_built" else big.tracking
    flipped = Tracking(tr.ball, tr.players, (1 - tr.team).astype(tr.team.dtype), tr.fps)
    logit = shape_lost(tr)
    assert (logit != 0).any()
    np.testing.assert_array_equal(shape_lost(flipped), -logit)


def test_shape_deterministic_and_order_free(big):
    a, b = make_synthetic(30_000, seed=1), big  # the same match as the big fixture
    logit = shape_lost(a.tracking)
    assert (logit != 0).any()
    np.testing.assert_array_equal(logit, shape_lost(b.tracking))
    tr = big.tracking
    order = np.random.default_rng(1).permutation(tr.team.size)
    shuffled = Tracking(tr.ball, tr.players[:, order], tr.team[order])
    # sums over slots change order, so equal up to rounding, never in sign or size
    np.testing.assert_allclose(shape_lost(shuffled), shape_lost(tr), atol=1e-9)


def test_shape_blocks_do_not_change_the_result(big, monkeypatch):
    whole_match = shape_lost(big.tracking)
    assert (whole_match != 0).any()
    monkeypatch.setattr(ev, "BLOCK_FRAMES", 997)
    np.testing.assert_array_equal(shape_lost(big.tracking), whole_match)


def test_shape_silent_without_clear_contacts():
    tr = shape_scene(ball_offset=(0.0, 80.0))[0]  # the ball is never near anyone
    logit, (model,) = ev.shape_logits(*_scan_and_lost(tr), whole(tr))
    assert (logit == 0).all()
    assert model.weights is None and model.n_home == model.n_away == 0


def test_shape_per_half_and_nothing_outside(big):
    scan, lost = _scan_and_lost(big.tracking)
    halves = ((100, 9_899), (10_100, 19_899))
    logit, models = ev.shape_logits(scan, lost, halves)
    assert [(m.first, m.last) for m in models] == list(halves)
    outside = np.ones(big.tracking.n_frames, dtype=bool)
    for first, last in halves:
        outside[first : last + 1] = False
    assert all(m.weights is not None for m in models)
    assert (logit[outside] == 0).all() and (logit[~outside] != 0).any()


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


def test_shape_feature_subsets_are_the_same_columns(big):
    scan, _ = _scan_and_lost(big.tracking)
    full = ev.shape_features(scan, 0, 9_999, 1, features=ev.SHAPE_COLUMNS)
    subset = ("speed", "width")
    np.testing.assert_array_equal(ev.shape_features(scan, 0, 9_999, 1, features=subset),
                                  full[:, [ev.SHAPE_COLUMNS.index(f) for f in subset]])


@pytest.mark.parametrize("features", [(), ("width", "width"), ("width", "height")], ids=["none", "twice", "unknown"])
def test_shape_features_refuse_bad_names(big, features):
    scan, _ = _scan_and_lost(big.tracking)
    with pytest.raises(ValueError, match="shape features"):
        ev.shape_features(scan, 0, 99, 1, features=features)


def test_shape_subset_model_flips_with_the_teams(big):
    tr = big.tracking
    flipped = Tracking(tr.ball, tr.players, (1 - tr.team).astype(tr.team.dtype), tr.fps)
    core = ("width", "forward", "speed")
    logit, (model,) = ev.shape_logits(*_scan_and_lost(tr), whole(tr), features=core)
    assert model.features == core and model.weights.shape == (3,) and (logit != 0).any()
    np.testing.assert_array_equal(ev.shape_logits(*_scan_and_lost(flipped), whole(tr), features=core)[0], -logit)


# -- fusion_features, for mtah/fusion.py --------------------------------------------------------

TWO_HALVES = [(0, 10_000), (10_000, 20_000)]  # half-open, as fusion passes them


def test_fusion_features_shape_names_and_finite(big):
    tr = big.tracking
    X, names = rf.fusion_features(tr.ball, tr.players, tr.team, PITCH_M, TWO_HALVES)
    assert X.shape == (tr.n_frames, len(names)) and X.dtype == np.float64
    assert names == list(rf.FEATURE_NAMES) and len(set(names)) == len(names)
    assert np.isfinite(X).all()  # no NaN and no infinity, ever
    assert (X != 0).any(axis=0).all()  # every column says something


def test_fusion_features_flip_exactly_with_the_teams(big):
    tr = big.tracking
    X, _ = rf.fusion_features(tr.ball, tr.players, tr.team, PITCH_M, TWO_HALVES)
    swapped, _ = rf.fusion_features(tr.ball, tr.players, 1 - tr.team, PITCH_M, TWO_HALVES)
    np.testing.assert_array_equal(swapped, -X)


def test_fusion_features_are_the_evidence_functions(big):
    tr = big.tracking
    X, names = rf.fusion_features(tr.ball, tr.players, tr.team, PITCH_M, [(0, tr.n_frames)])
    col = dict(zip(names, X.T))
    _, lost = ball_clean(tr.ball)
    np.testing.assert_array_equal(col["r2_nearest"], ev.evidence(tr))
    np.testing.assert_array_equal(col["r2_reach_lead"], reach_lead(tr))
    capped = np.clip(shape_lost(tr), -rf.SHAPE_LOGIT_CAP, rf.SHAPE_LOGIT_CAP)
    np.testing.assert_allclose(col["r2_shape_lost"], capped, atol=1e-8)
    assert (col["r2_shape_lost"][~lost] == 0).all() and (col["r2_shape"][~lost] != 0).any()
    np.testing.assert_array_equal(col["r2_shape"][lost], col["r2_shape_lost"][lost])
    assert (np.abs(col["r2_shape"]) <= rf.SHAPE_LOGIT_CAP).all()


def test_fusion_features_fit_each_half_on_its_own(big):
    tr = big.tracking
    X, _ = rf.fusion_features(tr.ball, tr.players, tr.team, PITCH_M, TWO_HALVES)
    players = tr.players.copy()
    players[10_100:, :, 1] += 1.0  # the second half changes, past the velocities' 12-frame reach
    Y, _ = rf.fusion_features(tr.ball, players, tr.team, PITCH_M, TWO_HALVES)
    np.testing.assert_array_equal(Y[:10_000], X[:10_000])
    assert not np.array_equal(Y[10_000:], X[10_000:])


def test_fusion_features_shape_only_inside_halves(big):
    tr = big.tracking
    X, names = rf.fusion_features(tr.ball, tr.players, tr.team, PITCH_M, [(1_000, 9_000), (11_000, 19_000)])
    col = dict(zip(names, X.T))
    outside = np.ones(tr.n_frames, dtype=bool)
    outside[1_000:9_000] = outside[11_000:19_000] = False
    assert (col["r2_shape"][outside] == 0).all() and (col["r2_shape_lost"][outside] == 0).all()
    assert (col["r2_shape"][~outside] != 0).any()


def test_fusion_features_treat_infinite_positions_as_missing(syn):
    tr = syn.tracking
    ball, players = tr.ball.copy(), tr.players.copy()
    ball[100:110] = np.inf
    players[200:300, :5] = -np.inf
    X, _ = rf.fusion_features(ball, players, tr.team, PITCH_M, [(0, tr.n_frames)])
    ball[100:110] = np.nan
    players[200:300, :5] = np.nan
    np.testing.assert_array_equal(X, rf.fusion_features(ball, players, tr.team, PITCH_M, [(0, tr.n_frames)])[0])
    assert np.isfinite(X).all()


def test_fusion_features_without_a_ball_say_nothing(syn):
    tr = syn.tracking
    X, _ = rf.fusion_features(np.full_like(tr.ball, np.nan), tr.players, tr.team, PITCH_M, [(0, tr.n_frames)])
    assert np.isfinite(X).all() and (X == 0).all()


@pytest.mark.parametrize("halves", [[], [(0, 0)], [(-1, 100)], [(0, 3_001)], [(0, 2_000), (1_500, 3_000)],
                                    [(2_000, 3_000), (0, 1_000)]],
                         ids=["none", "empty", "negative", "past_the_end", "overlap", "out_of_order"])
def test_fusion_features_refuse_bad_halves(syn, halves):
    tr = syn.tracking
    with pytest.raises(ValueError):
        rf.fusion_features(tr.ball, tr.players, tr.team, PITCH_M, halves)


def test_fusion_features_refuse_centre_origin_and_bad_pitch(syn):
    tr = syn.tracking
    centre = np.array(PITCH_M) / 2
    with pytest.raises(ValueError, match="corner-origin"):
        rf.fusion_features(tr.ball - centre, tr.players - centre, tr.team, PITCH_M, [(0, tr.n_frames)])
    with pytest.raises(ValueError, match="pitch"):
        rf.fusion_features(tr.ball, tr.players, tr.team, (0.0, 68.0), [(0, tr.n_frames)])
    with pytest.raises(ValueError):  # shapes are checked as for a Tracking
        rf.fusion_features(tr.ball[:-1], tr.players, tr.team, PITCH_M, [(0, 100)])
