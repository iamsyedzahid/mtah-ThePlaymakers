"""Evidence (R2). Synthetic and hand-built data only, never the match or the Metrica files."""
import itertools
import json

import numpy as np
import pytest

from mtah import evidence as ev
from mtah import evidence_fit as fit
from mtah import metrica_games as mg
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
    # A ball at a player's feet (slow), not one in flight: a pass skimming past a player is on
    # its way to someone else, and reach rightly follows where the ball is going.
    tr = big.tracking
    filled, lost = ball_clean(tr.ball)
    dist = np.linalg.norm(tr.players - filled[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    near_home, near_away = dist[:, tr.team == 0].min(axis=1), dist[:, tr.team == 1].min(axis=1)
    e = fn(tr)
    slow = ev.ball_speed(filled) < 3.0
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


def test_reach_keeps_the_ball_with_a_dribbler():
    # Home dribbles at 5 m/s with the ball 0.5 m ahead; an away defender stands 2 m to the side.
    # With a still ball the dribbler would be projected 3.5 m past it and lose it to the defender.
    n = 50
    t = np.arange(n) / 25
    home = np.column_stack([30.0 + 5.0 * t, np.full(n, 30.0)])
    ball = home + [0.5, 0.0]
    away = np.tile([32.0 + 5.0 * t[n // 2], 32.0], (n, 1))
    tr = Tracking(ball, np.stack([home, away], axis=1), np.array([0, 1]))
    assert (ev.evidence_reach(tr)[10:40] > 0).all()


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


# -- calibrated evidence ----------------------------------------------------------------------

def test_calibrated_contract(syn):
    e = ev.evidence_calibrated(syn.tracking)
    _, lost = ball_clean(syn.tracking.ball)
    assert np.isfinite(e).all() and (np.abs(e) <= 1).all() and (e[lost] == 0).all()
    tr = syn.tracking
    flipped = Tracking(tr.ball, tr.players, (1 - tr.team).astype(tr.team.dtype))
    np.testing.assert_array_equal(ev.evidence_calibrated(flipped), -e)
    assert (ev.evidence_calibrated(static_scene([11, 10], [10, 10], [20, 10])) > 0).all()


def test_calibrated_logit_never_favours_the_slower_team():
    lead = np.array([2.0, -2.0, 1.0])
    out = ev.calibrated_logit(lead, np.array([9.0, 9.0, 0.5]), np.array([30.0, 30.0, 0.0]), (1.0, -1.0, -1.0))
    assert (np.sign(out) * np.sign(lead) >= 0).all()  # the multiplier is clipped at 0


def test_ball_speed():
    n = 30
    ball = np.column_stack([np.arange(n) * 10.0 / 25, np.zeros(n)])  # 10 m/s
    np.testing.assert_allclose(ev.ball_speed(ball), 10.0)


# -- Metrica open games: labels, readers, ball mask (hand-built files only) ----------------------

def E(team, kind, start, end=None, sub=(), period=1):
    return mg.Event(team, kind, frozenset(sub), start, start if end is None else end, period)


def test_labels_from_events():
    events = [
        E(0, "SET PIECE", 10, sub=["KICK OFF"]), E(0, "PASS", 10, 20),
        E(1, "RECOVERY", 30), E(1, "BALL OUT", 30, 50),  # away clears it out at 50
        E(1, "CARD", 60, 61),  # booked while dead: does not move the dead-ball start
        E(0, "SET PIECE", 80, sub=["THROW IN"]), E(0, "PASS", 80, 90),
        E(0, "SHOT", 100, 110, sub=["ON TARGET", "GOAL"]),
        E(1, "CARRY", 120, 125),  # someone fetches the ball from the net
        E(1, "SET PIECE", 150, sub=["KICK OFF"]), E(1, "PASS", 150, 160),
        E(0, "FAULT RECEIVED", 165), E(0, "RECOVERY", 166),  # advantage: no restart, no D
        E(0, "BALL OUT", 170, 175), E(1, "RECOVERY", 190), E(1, "PASS", 190, 195, sub=["GOAL KICK"]),
        E(1, "BALL LOST", 196, 200, sub=["END HALF"]),
        E(0, "SET PIECE", 300, sub=["KICK OFF"], period=2), E(0, "PASS", 300, 320, period=2),
    ]
    lab = mg.labels_from_events(events, 400)
    expect = np.full(400, "X", dtype="<U1")
    expect[10:30], expect[30:50], expect[50:80] = "H", "A", "D"
    expect[80:110], expect[110:150] = "H", "D"  # from the goal to the kick-off, carry or not
    expect[150:166], expect[166:175], expect[175:190], expect[190:201] = "A", "H", "D", "A"
    expect[300:321] = "H"
    np.testing.assert_array_equal(lab, expect)


def test_contact_frames():
    events = [E(0, "PASS", 10, 20), E(1, "RECOVERY", 30), E(1, "PASS", 30, 60), E(0, "SHOT", 80, 90)]
    np.testing.assert_array_equal(mg.contact_frames(events), [20, 30, 60])


def test_mask_ball_is_seeded_and_shaped_like_our_match():
    contacts = np.arange(100, 199_000, 500)
    a, b = mg.mask_ball(199_319, contacts, seed=1), mg.mask_ball(199_319, contacts, seed=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, mg.mask_ball(199_319, contacts, seed=2))
    assert 0.3 < a.mean() < 0.5
    # over many gaps: the lost share and the gap quartiles R1 measured on our match
    from mtah.ball import _gaps

    long = mg.mask_ball(2_000_000, np.arange(100, 2_000_000, 500), seed=3)
    starts, ends = _gaps(long)
    quartiles = np.percentile((ends - starts) / 25, [25, 50, 75])
    np.testing.assert_allclose(quartiles, mg.GAP_QUARTILES_S, rtol=0.12)
    assert abs(long.mean() - mg.LOST_FRAC) < 0.02


def test_displace_ball_is_seeded_and_sized_like_our_ball():
    from mtah.synthetic import DRIBBLE_RANGE_M

    a, b = mg.displace_ball(10_000, seed=1), mg.displace_ball(10_000, seed=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, mg.displace_ball(10_000, seed=2))
    size = np.linalg.norm(a, axis=1)
    lo, hi = DRIBBLE_RANGE_M
    k = mg.DISPLACE_KNOT_FRAMES
    assert size.max() <= hi + 1e-9  # blending two offsets never makes a longer one
    knots = size[::k]
    assert knots.min() >= lo - 1e-9 and np.median(knots) > 1.0
    # slow: between knots the offset moves the ball by at most 2 * hi per knot interval
    assert np.abs(np.diff(a, axis=0)).max() <= 2 * hi / k + 1e-9
    assert 2 * hi / k * 25 < 2.0  # under 2 m/s, so a ball at rest still counts as slow


def test_masked_tracking_displaces_only_the_seen_ball():
    syn = make_synthetic(3_000, seed=0)
    game = mg.OpenGame(1, syn.tracking, ((0, 2_999),), syn.labels, np.array([100, 900]))
    plain, moved = mg.masked_tracking(game, 4), mg.masked_tracking(game, 4, displace=True)
    np.testing.assert_array_equal(np.isnan(plain.ball), np.isnan(moved.ball))  # same mask
    seen = ~np.isnan(plain.ball[:, 0])
    np.testing.assert_allclose(moved.ball[seen] - plain.ball[seen], mg.displace_ball(3_000, 4)[seen])
    np.testing.assert_array_equal(moved.players, syn.tracking.players)


def _write_metrica_csv(path, team_name, xy, period):
    """Metrica's CSV layout: two label rows, a header, then Period, Frame, Time and x, y pairs."""
    n, k = xy.shape[0], xy.shape[1] - 1
    names = [f"Player{i + 1}" for i in range(k)] + ["Ball"]
    lines = [",,," + "".join(f"{team_name},," for _ in range(k)) + ",",
             ",,," + "".join(f"{i + 1},," for i in range(k)) + ",",
             "Period,Frame,Time [s]," + ",".join(f"{nm}," for nm in names)]
    for f in range(n):
        vals = ",".join("NaN" if np.isnan(v) else f"{v:.5f}" for v in xy[f].ravel())
        lines.append(f"{period[f]},{f + 1},{(f + 1) / 25:.2f},{vals}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_csv_tracking(tmp_path):
    n = 30
    period = np.r_[np.ones(12, int), np.full(18, 2)]
    home = np.full((n, 3, 2), 0.25)
    home[:, 2] = [0.5, 0.5]  # the ball
    home[5, 1] = np.nan
    away = np.full((n, 3, 2), 0.75)
    away[:, 2] = [0.5, 0.5]
    _write_metrica_csv(tmp_path / "h.csv", "Home", home, period)
    _write_metrica_csv(tmp_path / "a.csv", "Away", away, period)
    tr, halves = mg.read_csv_tracking(tmp_path / "h.csv", tmp_path / "a.csv")
    assert tr.players.shape == (n, 4, 2) and tr.team.tolist() == [0, 0, 1, 1]
    np.testing.assert_allclose(tr.ball[0], [52.5, 34.0])
    np.testing.assert_allclose(tr.players[0, 0], [0.25 * 105, 0.25 * 68])
    assert np.isnan(tr.players[5, 1]).all()
    assert halves == ((0, 11), (12, 29))


def test_read_events_csv_and_json(tmp_path):
    (tmp_path / "e.csv").write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
        "Away,SET PIECE,KICK OFF,1,1,0.04,0,0,P1,,NaN,NaN,NaN,NaN\n"
        "Home,SHOT,ON TARGET-GOAL,2,50,2.0,60,2.4,P2,,0.9,0.5,1.0,0.5\n", encoding="utf-8")
    a, b = mg.read_csv_events(tmp_path / "e.csv")
    assert a == mg.Event(1, "SET PIECE", frozenset({"KICK OFF"}), 0, 0, 1)
    assert b == mg.Event(0, "SHOT", frozenset({"ON TARGET", "GOAL"}), 49, 59, 2)
    data = {"data": [
        {"team": {"id": "T1"}, "type": {"name": "PASS"}, "subtypes": None, "start": {"frame": 10}, "end": {"frame": 20}, "period": 1},
        {"team": {"id": "T2"}, "type": {"name": "SHOT"}, "subtypes": [{"name": "ON TARGET"}, {"name": "GOAL"}],
         "start": {"frame": 30}, "end": {"frame": None}, "period": 2},
    ]}
    (tmp_path / "e.json").write_text(json.dumps(data), encoding="utf-8")
    c, d = mg.read_json_events(tmp_path / "e.json", home_id="T1")
    assert c == mg.Event(0, "PASS", frozenset(), 9, 19, 1)
    assert d == mg.Event(1, "SHOT", frozenset({"ON TARGET", "GOAL"}), 29, 29, 2)


# -- decoder and leave-one-game-out (synthetic only) -------------------------------------------

def test_viterbi_is_the_exact_optimum():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        e = rng.normal(size=n) * rng.choice([0.1, 1.0, 3.0])
        cost = float(rng.choice([0.1, 1.0, 4.0]))

        def score(s):
            return float(np.dot(s, e)) - cost * int(np.count_nonzero(np.diff(s)))

        best = max(score(np.array(s)) for s in itertools.product((1.0, -1.0), repeat=n))
        path = fit.viterbi(e, cost)
        assert score(np.where(path == 0, 1.0, -1.0)) == pytest.approx(best)


def test_decode_keeps_dead_ball_and_switches_on_evidence():
    gt = np.array([3] * 5 + [0] * 50 + [2] * 10 + [1] * 50 + [3] * 5, dtype=np.int8)
    e = np.r_[np.zeros(5), np.ones(50), np.zeros(10), -np.ones(50), np.zeros(5)]
    pred = fit.decode(e, gt, [(5, 115)], cost=8.0)
    np.testing.assert_array_equal(pred[5:115], gt[5:115])
    assert (pred[:5] == 3).all() and (pred[115:] == 3).all()


def synthetic_game(number, n=6_000):
    syn = make_synthetic(n, seed=number)
    contacts = np.array([f for f, _ in syn.receptions], dtype=np.int64)
    return mg.OpenGame(number, syn.tracking, ((0, n - 1),), syn.labels, contacts)


def test_perfect_evidence_scores_one():
    d = fit.prepare(synthetic_game(1), seed=0)
    truth = np.where(d.gt == 0, 1.0, np.where(d.gt == 1, -1.0, 0.0))
    assert fit.spell_f1(d, fit.decode(truth, d.gt, d.runs, cost=4.0)) == 1.0


def test_calibration_is_deterministic_and_favours_the_faster_team():
    data = [fit.prepare(synthetic_game(k), seed=k) for k in (1, 2)]
    cal = fit.fit_calibration(data)
    assert cal == fit.fit_calibration([fit.prepare(synthetic_game(k), seed=k) for k in (1, 2)])
    assert cal.nearest_platt > 0 and cal.reach_platt[0] > 0
    assert np.isfinite(cal.reach_context).all() and np.isfinite(cal.shape_k)


def test_scores_come_from_the_official_scorer_file():
    import sys
    from pathlib import Path

    official = Path(__file__).resolve().parents[1] / "evaluation" / "possession_metrics.py"
    assert Path(fit.pm.__file__).resolve() == official
    assert str(official.parent) not in sys.path  # imported by path, nothing on sys.path to shadow it


def test_cache_with_pickled_objects_is_refused(tmp_path):
    # np.load would otherwise unpickle (run) whatever a planted cache holds
    np.savez(tmp_path / "metrica_game1.npz", ball=np.array([object()], dtype=object),
             players=np.zeros((1, 1, 2)), team=np.zeros(1, dtype=np.int8), halves=np.zeros((1, 2), dtype=np.int64))
    with pytest.raises(ValueError):
        mg._cached_tracking(1, tmp_path, tmp_path)


def test_evidence_to_logit_inverts_and_stays_finite():
    e = np.array([-1.0, -0.5, 0.0, 0.3, 1.0])
    logit = ev.evidence_to_logit(e)
    assert np.isfinite(logit).all()
    np.testing.assert_allclose(ev.logit_to_evidence(logit)[1:4], e[1:4])
    np.testing.assert_array_equal(ev.evidence_to_logit(-e), -logit)


def test_leave_one_game_out_runs_on_synthetic_games(monkeypatch):
    monkeypatch.setattr(fit, "COSTS", (8.0, 64.0))  # a smoke test: the grids' size is not the point
    monkeypatch.setattr(fit, "SHAPE_WEIGHTS", (1.0, 2.0))
    data = [fit.prepare(synthetic_game(k, n=4_000), seed=k) for k in (1, 2, 3)]
    results = fit.leave_one_game_out(data, log=lambda *_: None)
    assert set(results) == set(fit.CANDIDATES)
    for r in results.values():
        assert set(r["held_out"]) == {1, 2, 3}
        assert all(0.0 <= v <= 1.0 for v in r["held_out"].values())
    assert fit.choose(results, fit.BASES) in fit.BASES


# -- fusion_features, for mtah/fusion.py (synthetic only) --------------------------------------

from mtah import r2_features as rf  # noqa: E402
from mtah.config import PITCH_M  # noqa: E402

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
    np.testing.assert_array_equal(col["r2_nearest"], ev.nearest_evidence(tr.ball, lost, tr.players, tr.team))
    np.testing.assert_array_equal(ev.logit_to_evidence(col["r2_reach_lead"] / ev.CONTROL_SCALE_S), ev.evidence_reach(tr))
    np.testing.assert_allclose(ev.logit_to_evidence(col["r2_shape_lost"]), ev.evidence_shape(tr), atol=1e-8)
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


# -- choosing the shape features ----------------------------------------------------------------

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
    logit, (model,) = ev.shape_logits(*_scan_and_lost(tr), ev.halves_for(tr.n_frames), features=core)
    assert model.features == core and model.weights.shape == (3,) and (logit != 0).any()
    np.testing.assert_array_equal(ev.shape_logits(*_scan_and_lost(flipped), ev.halves_for(tr.n_frames), features=core)[0],
                                  -logit)


# -- the DFL reader (mtah/dfl_games.py), on a hand-written file in its format --------------------

DFL_INFO = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
  <Environment PitchX="100.00" PitchY="60.00" />
  <Teams>
    <Team TeamId="T-AWAY" Role="guest"><Players><Player PersonId="P-A1" /><Player PersonId="P-A2" /></Players></Team>
    <Team TeamId="T-HOME" Role="home"><Players>
      <Player PersonId="P-H2" /><Player PersonId="P-H1" /><Player PersonId="P-H9" /></Players></Team>
  </Teams>
</MatchInformation></PutDataRequest>
"""


def _frames(n_xy, extra=""):
    return "\n".join(f'<Frame N="{n}" X="{x}" Y="{y}" {extra}/>' for n, x, y in n_xy)


def _ball(rows):
    return "\n".join(f'<Frame N="{n}" X="{x}" Y="{y}" Z="0.00" BallPossession="{p}" BallStatus="{s}"/>'
                     for n, x, y, p, s in rows)


DFL_POSITIONS = f"""<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><Positions>
<FrameSet GameSection="firstHalf" TeamId="referee" PersonId="REF1">{_frames([(10000, 9, 9), (10001, 9, 9)])}</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B">{_ball([(10000, 0.0, 0.0, 1, 1), (10001, 0.5, 0.0, 2, 1),
                                                                      (10003, 1.0, -2.0, 2, 0)])}</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="T-HOME" PersonId="P-H1">{_frames([(9999, 7, 7), (10000, -1.0, 0.0),
                                                                             (10003, -2.0, 1.0)])}</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="T-AWAY" PersonId="P-A1">{_frames([(10000, 3.0, 4.0), (10001, 3.0, 4.0),
                                                                             (10002, 3.0, 4.0), (10003, 3.0, 4.0)])}</FrameSet>
<FrameSet GameSection="secondHalf" TeamId="BALL" PersonId="B">{_ball([(100000, 10.0, 0.0, 2, 1), (100001, 10.0, 0.0, 1, 1)])}</FrameSet>
<FrameSet GameSection="secondHalf" TeamId="T-HOME" PersonId="P-H2">{_frames([(100000, 10.5, 0.0), (100001, 10.5, 0.0)])}</FrameSet>
</Positions></PutDataRequest>
"""


def test_dfl_positions_read_as_kloppy_labels_them(tmp_path):
    from mtah import dfl_games as dg
    (tmp_path / "info.xml").write_text(DFL_INFO, encoding="utf-8")
    (tmp_path / "pos.xml").write_text(DFL_POSITIONS, encoding="utf-8")
    players, pitch = dg.read_match_information(tmp_path / "info.xml")
    assert pitch == (100.0, 60.0) and players == {"P-A1": 1, "P-A2": 1, "P-H1": 0, "P-H2": 0, "P-H9": 0}
    tr, halves, labels = dg.read_positions(tmp_path / "pos.xml", players, pitch)
    assert halves == ((0, 3), (4, 5))  # each period from its first to its last ball frame, back to back
    # D where BallStatus is not 1, else H where BallPossession is 1, else A; no ball record: X
    assert labels.tolist() == ["H", "A", "X", "D", "A", "H"]
    np.testing.assert_array_equal(tr.team, [0, 0, 1])  # home first, then away; squads without data are no slot
    np.testing.assert_allclose(tr.ball[[0, 3]], [[50.0, 30.0], [51.0, 28.0]])  # centre origin to corner origin
    assert np.isnan(tr.ball[2]).all()
    np.testing.assert_allclose(tr.players[0, :, :], [[49.0, 30.0], [np.nan, np.nan], [53.0, 34.0]])
    assert np.isnan(tr.players[1, 0]).all()  # P-H1 has no frame 10001; his frame 9999 is before the ball's
    assert np.isnan(tr.players[:4, 1]).all() and np.isnan(tr.players[4:, 2]).all()  # on the pitch in one half only
    assert not (tr.players == 9 + 50).any()  # the referee is no slot


def test_dfl_contacts_are_new_players_on_a_live_ball():
    from mtah import dfl_games as dg
    n = 12
    ball = np.tile([50.0, 30.0], (n, 1))
    far, near = [60.0, 30.0], [50.5, 30.0]
    p0 = np.array([near] * 3 + [far] * 9)  # p0 touches 0-2
    p1 = np.array([far] * 5 + [near] * 7)  # p1 touches from 5 on
    tr = Tracking(ball, np.stack([p0, p1], axis=1), np.array([0, 1]))
    labels = np.array(list("DHHAAAAAAXAA"))
    np.testing.assert_array_equal(dg.contact_frames(tr, labels), [1, 5])  # frame 0 is dead; 9 is X


def _fake_match(tmp_path, monkeypatch):
    from mtah import dfl_games as dg
    raw = tmp_path / "raw"
    raw.mkdir()
    m = dg.Match("TEST01", "000009", "", 0, "", 0)
    (raw / dg.info_name(m)).write_text(DFL_INFO, encoding="utf-8")
    (raw / dg.positions_name(m)).write_text(DFL_POSITIONS, encoding="utf-8")
    sizes = [(raw / name).stat().st_size for name in (dg.info_name(m), dg.positions_name(m))]
    m = m._replace(info_sha1=dg._digest(raw / dg.info_name(m), "sha1", sizes[0]), info_size=sizes[0],
                   positions_sha256=dg._digest(raw / dg.positions_name(m), "sha256", sizes[1]), positions_size=sizes[1])
    monkeypatch.setattr(dg, "MATCHES", (m,))
    return dg, raw, m


def test_dfl_info_hash_is_the_git_blob_id():
    import hashlib
    from mtah import dfl_games as dg
    body = b"hello\n"  # `git hash-object` gives ce013625030b... for this content
    assert hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest() == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert dg.MATCHES[0].info_sha1 == "8f2a82d84428c28e0e01f0a908c8395cc8d23900"  # as the mirror's revision lists it


def test_dfl_load_checks_hashes_and_caches_plain_arrays(tmp_path, monkeypatch):
    dg, raw, m = _fake_match(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    game = dg.load_match("TEST01", root=raw, cache_dir=cache)
    assert game.number == "TEST01" and game.pitch == (100.0, 60.0) and game.labels.tolist() == list("HAXDAH")
    again = dg.load_match("TEST01", root=raw, cache_dir=cache)  # from the cache now
    for field in ("ball", "players", "team"):
        np.testing.assert_array_equal(getattr(game.tracking, field), getattr(again.tracking, field))
    for field in ("halves", "labels", "contacts", "pitch"):
        np.testing.assert_array_equal(getattr(game, field), getattr(again, field))
    with np.load(cache / "dfl_TEST01.npz", allow_pickle=False) as saved:
        assert all(saved[k].dtype != object for k in saved.files)
    (raw / dg.positions_name(m)).write_text(DFL_POSITIONS.replace('X="0.5"', 'X="0.6"'), encoding="utf-8")
    with pytest.raises(ValueError, match="not the file"):
        dg.load_match("TEST01", root=raw, cache_dir=None)
    (raw / dg.info_name(m)).unlink()
    with pytest.raises(FileNotFoundError):
        dg.load_match("TEST01", root=raw, cache_dir=None)
    with pytest.raises(ValueError, match="DFL matches"):
        dg.load_match("NOPE", root=raw)


def test_dfl_reader_never_expands_external_entities(tmp_path):
    """The XML parser is the standard library's (expat): it refuses entities that point at files
    or URLs, and expat 2.4.1+ caps entity amplification. load_match only parses files whose hash
    matches the pinned revision, but the reader itself must not reach outside the file."""
    import pyexpat
    import xml.etree.ElementTree as ET
    from mtah import dfl_games as dg
    assert tuple(int(v) for v in pyexpat.EXPAT_VERSION.split("_")[1].split(".")) >= (2, 4, 1)
    secret = tmp_path / "secret.txt"
    secret.write_text("42", encoding="utf-8")
    hostile = DFL_POSITIONS.replace('<?xml version="1.0" encoding="UTF-8"?>',
                                    f'<?xml version="1.0"?><!DOCTYPE p [<!ENTITY s SYSTEM "{secret.as_uri()}">]>')
    hostile = hostile.replace('X="0.5"', 'X="&s;"')
    (tmp_path / "pos.xml").write_text(hostile, encoding="utf-8")
    with pytest.raises(ET.ParseError):
        dg.read_positions(tmp_path / "pos.xml", {"P-H1": 0, "P-H2": 0, "P-A1": 1}, (100.0, 60.0))


def test_dfl_reader_refuses_an_absurd_period(tmp_path):
    from mtah import dfl_games as dg
    broken = DFL_POSITIONS.replace('N="10003" X="1.0"', 'N="900000000" X="1.0"')
    (tmp_path / "pos.xml").write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="spans"):
        dg.read_positions(tmp_path / "pos.xml", {"P-H1": 0, "P-H2": 0, "P-A1": 1}, (100.0, 60.0))


def test_benchmark_lists_the_dfl_matches_with_their_own_seeds():
    numbers, load, seeds = fit.load_games("dfl")
    assert len(numbers) == 7 and len({seeds[n] for n in numbers}) == 7
    assert not set(seeds.values()) & set(fit.MASK_SEED.values())  # never Metrica's masks
    assert fit.load_games("metrica")[0] == list(mg.GAMES)
