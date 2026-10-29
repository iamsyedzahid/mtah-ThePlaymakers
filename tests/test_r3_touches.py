"""R3 touch detection. Synthetic data only."""
import inspect

import numpy as np
import pytest

from mtah import interfaces, touches
from mtah.synthetic import make_synthetic


def _scene(n=80):
    """Home player at (30, 30), away player at (60, 30); the ball is filled in by each test."""
    players = np.zeros((n, 2, 2))
    players[:, 0] = [30.0, 30.0]
    players[:, 1] = [60.0, 30.0]
    return players, np.array([0, 1])


def test_signature_matches_interface():
    def shape(f):
        return [(p.name, p.kind, p.default) for p in inspect.signature(f).parameters.values()]

    assert shape(touches.touches) == shape(interfaces.touches)


def test_pass_and_trap_is_a_reception_by_the_receiver():
    players, team = _scene()
    ball = np.empty((80, 2))
    ball[:40] = np.column_stack([10.0 + 0.8 * np.arange(40) - 0.0, np.full(40, 30.0)])  # 20 m/s
    ball[:40, 0] = np.minimum(ball[:40, 0], 29.5)
    ball[40:] = [29.6, 30.0]  # stopped at the home player's feet
    events = touches.touches(ball, players, team)
    receptions = [t for t in events if t.kind == "reception"]
    assert receptions and receptions[0].team == 0
    assert abs(receptions[0].frame - 25) <= 3  # arrives at x = 29.5 on frame 24-25


def test_glancing_contact_is_a_deflection():
    players, team = _scene()
    # A fast ball flies at the away player, turns 60 degrees there and flies on at 20 m/s.
    t = np.arange(80)
    hit = 40
    heading_in, heading_out = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])
    contact = np.array([59.5, 30.0])
    ball = np.where((t < hit)[:, None], contact + (t - hit)[:, None] * 0.8 * heading_in,
                    contact + (t - hit)[:, None] * 0.8 * heading_out)
    events = touches.touches(ball, players, team)
    near = [e for e in events if abs(e.frame - hit) <= 3]
    assert near and near[0].team == 1 and near[0].kind == "deflection"


def test_jitter_near_a_player_is_not_a_touch():
    players, team = _scene()
    rng = np.random.default_rng(0)
    ball = np.array([30.5, 30.0]) + rng.normal(0.0, 0.05, size=(80, 2))
    assert touches.touches(ball, players, team) == []


def test_missing_and_glitch_frames_make_no_touch():
    players, team = _scene()
    ball = np.tile([30.5, 30.0], (80, 1))
    ball[20:30] = np.nan
    ball[50] = [45.0, 30.0]  # a single 15 m glitch
    events = touches.touches(ball, players, team)
    assert all(not (20 <= e.frame < 30) for e in events)
    assert not any(abs(e.frame - 50) <= 2 for e in events)


def test_output_is_sorted_and_in_range(syn):
    events = touches.touches(syn.tracking.ball, syn.tracking.players, syn.tracking.team)
    frames = [e.frame for e in events]
    assert frames == sorted(frames)
    assert all(isinstance(f, int) and 0 <= f < syn.tracking.n_frames for f in frames)
    assert {e.team for e in events} <= {0, 1} and {e.kind for e in events} <= {"reception", "deflection"}


@pytest.mark.parametrize("seed", [1, 2])
def test_against_synthetic_truth(seed):
    syn = make_synthetic(60_000, seed)
    events = touches.touches(syn.tracking.ball, syn.tracking.players, syn.tracking.team)
    frames = np.array([e.frame for e in events])
    kinds = np.array([e.kind for e in events])
    teams = np.array([e.team for e in events])
    seen = ~np.isnan(syn.tracking.ball).any(axis=1)

    def matched(truth):
        out = []
        for f, side in truth:
            if seen[max(f - 3, 0) : f + 4].all():  # only events the tracker saw
                j = np.flatnonzero(np.abs(frames - f) <= 3)
                out.append((int(j[0]) if j.size else -1, side))
        return out

    rec = matched(syn.receptions)
    found = [(j, side) for j, side in rec if j >= 0]
    assert len(found) / len(rec) > 0.85
    assert np.mean([kinds[j] == "reception" and teams[j] == side for j, side in found]) > 0.95
    defl = [j for j, _ in matched(syn.deflections) if j >= 0]
    assert defl and np.mean(kinds[defl] == "deflection") > 0.9
    # plausible volume: a few touches per possession, not one per frame
    assert len(events) < 0.02 * syn.tracking.n_frames
