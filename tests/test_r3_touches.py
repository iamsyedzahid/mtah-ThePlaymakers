"""Tests for R3 touch detection and control spells (mtah/touches.py).

All tests run on synthetic data only. Never read real match data or ground truth here.
"""
import numpy as np
import pytest

from mtah.interfaces import Touch, Tracking
from mtah.synthetic import make_synthetic
from mtah.touches import touches, CONTROL_RADIUS_M, RECEPTION_MIN_FRAMES


@pytest.fixture
def syn_data():
    return make_synthetic(n_frames=3000, seed=42)


def test_touches_returns_list_of_touch(syn_data):
    tr = syn_data.tracking
    result = touches(tr.ball, tr.players, tr.team)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, Touch)
        assert item.kind in ("reception", "deflection")
        assert item.team in (0, 1)


def test_touches_sorted_by_frame(syn_data):
    tr = syn_data.tracking
    result = touches(tr.ball, tr.players, tr.team)
    assert len(result) > 0
    frames = [t.frame for t in result]
    assert frames == sorted(frames)
    # Check strict inequality: no two touches on the exact same frame
    assert len(frames) == len(set(frames))


def test_touches_frame_bounds(syn_data):
    tr = syn_data.tracking
    result = touches(tr.ball, tr.players, tr.team)
    T = tr.n_frames
    for t in result:
        assert 0 <= t.frame < T


def test_no_touch_on_nan_ball(syn_data):
    tr = syn_data.tracking
    result = touches(tr.ball, tr.players, tr.team)
    nan_mask = np.isnan(tr.ball).any(axis=1)
    touch_frames = [t.frame for t in result]
    for f in touch_frames:
        assert not nan_mask[f], f"Touch generated on frame {f} where ball is NaN!"


def test_no_touch_on_speed_glitch():
    # Construct a tracking array with a single severe jump
    T = 100
    ball = np.zeros((T, 2)) + 50.0  # stationary ball near centre
    players = np.zeros((T, 2, 2)) + 50.0  # 2 players right next to the ball
    team = np.array([0, 1])

    # Inject an impossible teleport glitch at frame 50
    ball[50] = [10.0, 10.0]  # jumps 56 metres in 1 frame (1400 m/s)

    result = touches(ball, players, team)
    touch_frames = [t.frame for t in result]
    assert 50 not in touch_frames, "Glitch frame should not have generated a touch"


def test_touches_deterministic(syn_data):
    tr = syn_data.tracking
    res1 = touches(tr.ball, tr.players, tr.team)
    res2 = touches(tr.ball, tr.players, tr.team)
    assert res1 == res2


def test_synthetic_receptions_detected(syn_data):
    tr = syn_data.tracking
    detected = touches(tr.ball, tr.players, tr.team)
    rec_detected = [t for t in detected if t.kind == "reception"]

    # syn_data.receptions has (frame, team) where a carrier takes control
    assert len(syn_data.receptions) > 0
    assert len(rec_detected) > 0

    # For each true reception where the ball wasn't lost at that moment,
    # we should detect a reception near it (within 10 frames = 0.4s)
    matches = 0
    for true_frame, true_team in syn_data.receptions:
        # Check if ball was visible near true_frame
        window = slice(max(0, true_frame - 2), min(tr.n_frames, true_frame + 5))
        if np.isnan(tr.ball[window]).all():
            continue  # ball was completely lost, cannot detect touch by design
        hit = any(abs(t.frame - true_frame) <= 10 and t.team == true_team for t in rec_detected)
        if hit:
            matches += 1

    assert matches > 0, "Expected at least some synthetic receptions to match detected ones"


def test_empty_and_all_nan_ball():
    empty_ball = np.empty((0, 2))
    empty_players = np.empty((0, 2, 2))
    empty_team = np.array([0, 1])
    assert touches(empty_ball, empty_players, empty_team) == []

    all_nan_ball = np.full((100, 2), np.nan)
    players = np.zeros((100, 2, 2))
    assert touches(all_nan_ball, players, empty_team) == []
