"""Shapes, dtypes, 0-indexing, NaN handling and determinism. Synthetic data only."""
import inspect
from pathlib import Path

import numpy as np
import pytest

import run
from mtah import ball, config, deadball, evidence, interfaces, segment, touches, xpolicy
from mtah.interfaces import A, D, H, X, Touch, Tracking
from mtah.synthetic import make_synthetic

T = 3_000


# -- contract ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module, name",
    [
        (ball, "ball_clean"),
        (ball, "gap_stats"),
        (evidence, "evidence"),
        (touches, "touches"),
        (deadball, "dead_mask"),
        (deadball, "dead_intervals"),
        (deadball, "untagged_intervals"),
        (deadball, "slow_lost_intervals"),
        (deadball, "camera_still_intervals"),
        (deadball, "join_close"),
        (deadball, "restart_touches"),
        (segment, "segment"),
        (xpolicy, "to_x_policy"),
    ],
)
def test_signature_matches_interface(module, name):
    def shape(fn):
        return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

    assert shape(getattr(module, name)) == shape(getattr(interfaces, name))


def test_tracking_rejects_bad_shapes():
    good = dict(ball=np.zeros((5, 2)), players=np.zeros((5, 4, 2)), team=np.array([0, 0, 1, 1]))
    Tracking(**good)
    with pytest.raises(ValueError):
        Tracking(**{**good, "ball": np.zeros((5, 3))})
    with pytest.raises(ValueError):
        Tracking(**{**good, "players": np.zeros((4, 4, 2))})
    with pytest.raises(ValueError):
        Tracking(**{**good, "team": np.array([0, 1, 2, 1])})


def test_config_bounds():
    graded = sum(end - start + 1 for start, end in config.GRADED_HALVES)
    assert graded == 158_857
    (_, end1), (start2, end2) = config.GRADED_HALVES
    assert end1 < start2 and end2 < config.N_FRAMES
    d_start, d_end = config.SECOND_HALF_D_BLOCK
    assert start2 <= d_start <= d_end <= end2


def test_data_dir_env(monkeypatch):
    monkeypatch.delenv(config.DATA_DIR_ENV, raising=False)
    assert config.data_dir() == Path(__file__).resolve().parents[1] / "data"  # same as games.py
    monkeypatch.setenv(config.DATA_DIR_ENV, "somewhere")
    assert config.data_dir().name == "somewhere"


# -- synthetic --------------------------------------------------------------------------------

def test_synthetic_shapes_and_dtypes(syn):
    tr = syn.tracking
    assert tr.ball.shape == (T, 2) and tr.ball.dtype == np.float64
    assert tr.players.shape == (T, 46, 2) and tr.players.dtype == np.float64
    assert tr.team.shape == (46,) and tr.team.dtype.kind == "i"
    assert (tr.team == 0).sum() == 23 and (tr.team == 1).sum() == 23
    assert tr.fps == 25
    assert syn.labels.shape == (T,) and syn.labels.dtype == np.dtype("<U1")
    assert set(np.unique(syn.labels)) <= {H, A, D}


def test_synthetic_ball_gaps_match_the_real_atd():
    # Real ATD, graded frames: 40.7% lost, median gap 268 frames. Gaps are long, so only a
    # full-length match has stable statistics.
    lost = np.isnan(make_synthetic(config.N_FRAMES, seed=0).tracking.ball).any(axis=1)
    stats = ball.gap_stats(lost)
    assert 0.3 < stats["frac_lost"] < 0.5
    assert 150 < stats["median_gap_frames"] < 450


def test_synthetic_is_realistic_enough(syn):
    teams = syn.labels[syn.labels != D]
    assert np.count_nonzero(teams[1:] != teams[:-1]) >= 3  # a few possession changes
    assert np.isnan(syn.tracking.players).any()


def test_synthetic_slots_like_the_real_atd(syn):
    tracked = (~np.isnan(syn.tracking.players[..., 0])).sum(axis=1)
    assert tracked.max() <= 22
    assert 19.5 < tracked.mean() < 21.5  # DATA.md: 20.5
    used = (~np.isnan(syn.tracking.players[..., 0])).any(axis=0)
    assert used.sum() > 22  # tracks break and move to other slots


def test_synthetic_receptions_and_deflections():
    syn = make_synthetic(20_000, seed=0)
    for events in (syn.receptions, syn.deflections):
        frames = [f for f, _ in events]
        assert frames == sorted(frames) and all(0 <= f < 20_000 for f in frames)
    assert syn.receptions and syn.deflections
    # a reception hands the ball to its team; a deflection never changes possession
    for frame, side in syn.receptions:
        assert syn.labels[frame] in ({H, D} if side == 0 else {A, D})
    for frame, side in syn.deflections:
        assert syn.labels[frame] in ({A, D} if side == 0 else {H, D})


def test_synthetic_corner_origin(syn):
    players = syn.tracking.players
    seen = ~np.isnan(players[..., 0])
    assert (players[seen] >= 0).all()
    assert (players[seen] <= np.array(config.PITCH_M)).all()


# Ball cleaning and gap statistics (R1) are in tests/test_r1_ball.py.


# Control evidence (R2) is tested in tests/test_r2_evidence.py.


# Touches and Smart Tagging (R3) are tested in tests/test_r3_touches.py and test_r3_tagging.py.


# -- dead ball, segment, X policy (R4) --------------------------------------------------------

def test_dead_mask_inclusive():
    mask = deadball.dead_mask(10, [(0, 0), (3, 5), (9, 9)])
    assert mask.dtype == bool and mask.shape == (10,)
    np.testing.assert_array_equal(np.flatnonzero(mask), [0, 3, 4, 5, 9])
    with pytest.raises(ValueError):
        deadball.dead_mask(10, [(5, 10)])
    with pytest.raises(ValueError):
        deadball.dead_mask(10, [(-1, 2)])


def test_segment_hand_over_at_reception():
    n = 200
    labels = segment.segment(
        np.zeros(n), np.zeros(n, bool), np.zeros(n, bool),
        [Touch(0, 0, "reception"), Touch(100, 1, "reception")],
    )
    assert labels.dtype == np.dtype("<U1")
    assert (labels[:100] == H).all() and (labels[100:] == A).all()


def test_segment_on_synthetic(syn):
    filled, lost = ball.ball_clean(syn.tracking.ball)
    e = evidence.evidence(syn.tracking)
    events = touches.touches(syn.tracking.ball, syn.tracking.players, syn.tracking.team)
    dead = deadball.dead_mask(T, syn.set_piece_intervals)
    labels = segment.segment(e, lost, dead, events)
    assert labels.shape == (T,)
    assert set(np.unique(labels)) <= {H, A, D}
    assert (labels[dead] == D).all()


def test_segment_only_receptions_flip():
    n = 200
    e = np.r_[np.ones(120), -np.ones(80)]  # evidence turns to away, but nobody away receives
    events = [Touch(0, 0, "reception"), Touch(130, 1, "deflection")]
    labels = segment.segment(e, np.zeros(n, bool), np.zeros(n, bool), events)
    assert (labels == H).all()


def test_segment_ignores_a_reception_the_evidence_contradicts():
    # A presser's reception at 100, but home stays in control afterwards: no flip.
    n = 300
    e = np.ones(n)
    events = [Touch(0, 0, "reception"), Touch(100, 1, "reception")]
    labels = segment.segment(e, np.zeros(n, bool), np.zeros(n, bool), events)
    assert (labels == H).all()
    # The same reception followed by away control is a turnover.
    e[100:] = -1.0
    labels = segment.segment(e, np.zeros(n, bool), np.zeros(n, bool), events)
    assert (labels[:100] == H).all() and (labels[100:] == A).all()


def test_segment_change_after_a_lost_ball_moves_into_the_gap():
    n = 300
    lost = np.zeros(n, bool)
    lost[100:200] = True
    e = np.r_[np.ones(100), np.zeros(100), -np.ones(100)]
    events = [Touch(0, 0, "reception"), Touch(205, 1, "reception")]
    labels = segment.segment(e, lost, np.zeros(n, bool), events)
    first_away = int(np.flatnonzero(labels == A)[0])
    assert 100 < first_away < 200 and (labels[first_away:] == A).all()


def test_segment_d_only_from_dead_mask():
    n = 200
    lost = np.zeros(n, bool)
    lost[50:150] = True  # a long missing ball is not a dead ball
    dead = np.zeros(n, bool)
    dead[160:170] = True
    labels = segment.segment(np.zeros(n), lost, dead, [Touch(0, 1, "reception")])
    np.testing.assert_array_equal(np.flatnonzero(labels == D), np.arange(160, 170))
    assert (labels[labels != D] == A).all()


def test_segment_no_information_still_has_a_team():
    n = 50
    labels = segment.segment(np.zeros(n), np.ones(n, bool), np.zeros(n, bool), [])
    assert set(np.unique(labels)) <= {H, A}


def test_x_policy_only_mid_possession():
    labels = np.array([H] * 100 + [A] * 100 + [D] * 20 + [H] * 30, dtype="<U1")
    assert (xpolicy.to_x_policy(labels) == labels).all()

    out = xpolicy.to_x_policy(labels, confidence=np.zeros(labels.size), threshold=0.5, margin=10)
    assert (out[:10] == H).all() and (out[90:110] != X).all()  # run edges untouched
    assert (out[10:90] == X).all()
    assert (out[200:220] == D).all()  # D never becomes X
    assert (out[220:230] == H).all() and (out[240:] == H).all()
    changed = out != labels
    assert set(np.unique(labels[changed])) <= {H, A}


# -- end to end -------------------------------------------------------------------------------

def test_run_writes_zero_indexed_csv(tmp_path):
    out = tmp_path / "synthetic.csv"
    labels = run.main(out, n_frames=T)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "frame,label"
    assert len(lines) == T + 1
    frames = [int(line.split(",")[0]) for line in lines[1:]]
    assert frames == list(range(T))
    assert {line.split(",")[1] for line in lines[1:]} <= {H, A, D, X}
    assert labels.shape == (T,)


def test_determinism(tmp_path):
    a, b = make_synthetic(T, seed=0), make_synthetic(T, seed=0)
    np.testing.assert_array_equal(a.tracking.ball, b.tracking.ball)  # NaNs compare equal here
    np.testing.assert_array_equal(a.tracking.players, b.tracking.players)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.set_piece_intervals == b.set_piece_intervals

    run.main(tmp_path / "one.csv", n_frames=T)
    run.main(tmp_path / "two.csv", n_frames=T)
    assert (tmp_path / "one.csv").read_bytes() == (tmp_path / "two.csv").read_bytes()
