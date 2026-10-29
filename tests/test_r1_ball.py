"""R1 ball cleaning and gap statistics. Synthetic data only."""
import numpy as np
import pytest

from mtah import ball
from mtah.config import FPS, N_FRAMES
from mtah.synthetic import make_synthetic

T = 3_000


def test_ball_clean_on_synthetic(syn):
    raw_missing = np.isnan(syn.tracking.ball).any(axis=1)
    filled, lost = ball.ball_clean(syn.tracking.ball)
    assert filled.shape == (T, 2) and filled.dtype == np.float64
    assert lost.shape == (T,) and lost.dtype == bool
    assert not np.isnan(filled).any()
    assert lost.sum() <= raw_missing.sum() + 2 * ball.MAX_SPEED_PASSES  # short gaps are splined


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_impossible_speeds_after_cleaning(seed):
    # A whole synthetic match: between any two trusted frames the ball never beats the cap.
    filled, lost = ball.ball_clean(make_synthetic(N_FRAMES, seed).tracking.ball)
    speed = np.linalg.norm(np.diff(filled, axis=0), axis=1) * FPS
    trusted = np.logical_not(lost[1:] | lost[:-1])
    assert speed[trusted].max() <= ball.SPEED_CAP_MS


def _line(n=200):
    """A ball rolling at 10 m/s (0.4 m/frame) along x, on a corner-origin pitch."""
    return np.column_stack([10.0 + 0.4 * np.arange(n), np.full(n, 30.0)])


def test_ball_clean_short_gap_is_splined():
    truth = _line()
    b = truth.copy()
    b[100:110] = np.nan  # 0.4 s
    filled, lost = ball.ball_clean(b)
    assert not lost.any()
    np.testing.assert_allclose(filled, truth, atol=0.5)


def test_ball_clean_long_and_edge_gaps_are_lost():
    b = _line()
    b[:3] = np.nan  # leading gap: nothing to spline from
    b[100:140] = np.nan  # 1.6 s
    b[-2:] = np.nan  # trailing gap
    filled, lost = ball.ball_clean(b)
    np.testing.assert_array_equal(np.flatnonzero(lost), np.r_[0:3, 100:140, 198:200])
    assert not np.isnan(filled).any()


def test_ball_clean_speed_cap_removes_glitch():
    truth = _line()
    b = truth.copy()
    b[50] += [15.0, 0.0]  # a 15 m jump in one frame is 375 m/s
    filled, lost = ball.ball_clean(b)
    assert not lost.any()
    np.testing.assert_allclose(filled[50], truth[50], atol=0.5)


def test_ball_clean_all_nan():
    filled, lost = ball.ball_clean(np.full((4, 2), np.nan))
    assert lost.all()
    np.testing.assert_allclose(filled, [[52.65, 34.0]] * 4)  # centre spot


def test_gap_stats_known_mask():
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], dtype=bool)  # gaps of 2, 5, 1
    stats = ball.gap_stats(mask)
    assert stats["n_gaps"] == 3
    assert stats["n_lost"] == 8
    assert stats["longest_gap_frames"] == 5
    assert stats["median_gap_frames"] == 2.0
    assert stats["longest_gap_s"] == 5 / 25
    assert all(type(v) in (int, float) for v in stats.values())  # numbers only, not bools/arrays
    assert ball.gap_stats(np.zeros(0, dtype=bool))["n_gaps"] == 0


def test_ball_clean_keeps_a_hard_shot():
    # 38 m/s for 1.2 s: faster than any pass, but a real shot. It must stay trusted and in place.
    shot = np.column_stack([10.0 + 38.0 / FPS * np.arange(30), np.full(30, 34.0)])
    track = np.vstack([np.tile([10.0, 34.0], (20, 1)), shot, np.tile(shot[-1], (20, 1))])
    cleaned, lost = ball.ball_clean(track)
    assert not lost.any()
    assert np.abs(cleaned - track).max() < 1.0


def test_ball_clean_recovers_from_a_bad_anchor():
    # The first sample after a gap is a glitch 20 m off; the true ball after it must survive.
    track = np.tile([50.0, 30.0], (60, 1)) + np.column_stack([0.2 * np.arange(60), np.zeros(60)])
    track[:10] = np.nan
    track[10] = [70.0, 30.0]
    cleaned, lost = ball.ball_clean(track)
    assert lost[15:].sum() == 0
    assert np.abs(cleaned[20:] - track[20:]).max() < 0.5
