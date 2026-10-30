"""R4 dead-ball finder. Synthetic data only."""
import numpy as np
import pytest

from mtah.ball import ball_clean
from mtah.config import FPS
from mtah.deadball import (
    MAX_DEAD_S,
    UNTAGGED_STILL_S,
    dead_intervals,
    dead_mask,
    join_close,
    restart_touches,
    slow_lost_intervals,
    untagged_intervals,
)
from mtah.interfaces import SetPiece
from mtah.synthetic import make_synthetic

N = 1000
RESTART = 500


def _rest_then_kick(rest_from=300):
    """The ball rolls, rests from rest_from until RESTART, then is kicked at 15 m/s."""
    ball = np.empty((N, 2))
    ball[:rest_from] = np.column_stack([5.0 + 0.1 * np.arange(rest_from), np.full(rest_from, 30.0)])
    ball[rest_from:RESTART] = ball[rest_from - 1]
    after = N - RESTART
    ball[RESTART:] = ball[rest_from - 1] + np.column_stack([0.6 * np.arange(after) % 40, np.zeros(after)])
    return ball


def test_d_runs_from_the_rest_to_the_restart_not_the_whole_tag():
    tag = SetPiece(RESTART - 60, RESTART + 60, 0, "Free Kick")
    (start, end), = dead_intervals([tag], _rest_then_kick(), np.zeros(N, bool))
    assert abs(start - 300) <= 6  # reaches back before the tag window
    assert abs(end - RESTART) <= 3  # stops at the kick: the rest of the window is live


def test_off_pitch_ball_is_quiet_even_while_moving():
    ball = _rest_then_kick()
    ball[300:RESTART, 1] = -2.0 + 0.04 * np.arange(RESTART - 300) % 1.0  # carried behind the line
    tag = SetPiece(RESTART - 60, RESTART + 60, None, "Throw-in")
    (start, end), = dead_intervals([tag], ball, np.zeros(N, bool))
    assert start <= 305 and abs(end - RESTART) <= 3


def test_a_long_rest_is_capped():
    long_rest = 2 * MAX_DEAD_S * FPS
    n = long_rest + 300
    ball = np.tile([50.0, 30.0], (n, 1))
    ball[long_rest:, 0] += 0.6 * np.arange(300)
    tag = SetPiece(long_rest - 60, long_rest + 60, 0, "Goal Kick")
    (start, end), = dead_intervals([tag], ball, np.zeros(n, bool))
    assert end - start + 1 <= MAX_DEAD_S * FPS


def test_a_lost_restart_falls_back_to_a_short_d():
    ball = np.column_stack([10.0 + 0.4 * np.arange(N), np.full(N, 30.0)])  # never rests
    tag = SetPiece(400, 520, None, "Throw-in")
    assert dead_intervals([tag], ball, np.zeros(N, bool)) == [(400, 460)]


def test_ball_vanishing_during_the_stoppage_keeps_its_start():
    ball = _rest_then_kick(rest_from=300)
    lost = np.zeros(N, bool)
    lost[420:600] = True  # gone from before the tag window until well after the restart
    (start, end), = dead_intervals([SetPiece(460, 560, 0, "Free Kick")], ball, lost)
    assert abs(start - 300) <= 6 and end == 510


def test_untagged_rest_counts_only_beyond_a_goalkeeper_hold():
    ball = _rest_then_kick(rest_from=RESTART - (UNTAGGED_STILL_S + 1) * FPS)
    assert len(untagged_intervals(ball, np.zeros(N, bool))) == 1
    held = _rest_then_kick(rest_from=RESTART - 4 * FPS)  # held 4 s, then played
    assert untagged_intervals(held, np.zeros(N, bool)) == []


def test_untagged_off_pitch_counts_and_a_missing_ball_alone_does_not():
    ball = np.column_stack([10.0 + 0.09 * np.arange(N), np.full(N, 30.0)])  # stays on the pitch
    assert untagged_intervals(ball, np.zeros(N, bool)) == []
    lost = np.zeros(N, bool)
    lost[200:700] = True
    assert untagged_intervals(ball, lost) == []
    ball[300:400, 1] = -3.0
    assert len(untagged_intervals(ball, np.zeros(N, bool))) == 1


def test_restart_gives_the_tagged_team_the_ball():
    tags = [SetPiece(0, 50, 1, "Free Kick"), SetPiece(100, 150, None, "Throw-in")]
    events = restart_touches(tags, [(10, 30), (110, 130)], N)
    assert [(e.frame, e.team, e.kind) for e in events] == [(31, 1, "reception")]


def _running(speed_ms, n=N, slots=4):
    """Players all moving at speed_ms along x, with a sleeping ghost track that must not matter."""
    players = np.zeros((n, slots, 2))
    players[:, :, 0] = 10.0 + np.arange(n)[:, None] * np.asarray(speed_ms).reshape(-1, 1) / FPS
    return players


def test_missing_ball_with_slow_players_is_d():
    lost = np.zeros(N, bool)
    lost[200:600] = True
    spans = slow_lost_intervals(_running(0.5), lost)
    assert spans == [(200, 599)]


def test_missing_ball_with_running_players_is_not_d():
    lost = np.zeros(N, bool)
    lost[200:600] = True
    assert slow_lost_intervals(_running(3.0), lost) == []


def test_slow_players_with_the_ball_seen_are_not_d():
    assert slow_lost_intervals(_running(0.5), np.zeros(N, bool)) == []


def test_join_close_merges_a_glimpse_inside_a_stoppage():
    mask = np.zeros(N, bool)
    mask[100:200] = mask[210:300] = mask[500:600] = True
    joined = join_close(mask)
    assert joined[100:300].all()  # 10 frames apart: one stoppage
    assert joined[300:500].sum() == 0  # 8 s apart: real play between them
    assert not joined[:100].any() and not joined[600:].any()  # never reaches the ends of the match


def test_outside_the_match_raises():
    with pytest.raises(ValueError):
        dead_intervals([SetPiece(10, N, 0, "Corner Kick")], np.zeros((N, 2)), np.zeros(N, bool))


def test_d_found_on_synthetic_is_mostly_right():
    syn = make_synthetic(60_000, 1)
    filled, lost = ball_clean(syn.tracking.ball)
    spans = dead_intervals(syn.set_pieces, filled, lost) + untagged_intervals(filled, lost)
    spans += slow_lost_intervals(syn.tracking.players, lost)
    found = dead_mask(len(lost), spans)
    truth = dead_mask(len(lost), syn.set_piece_intervals)
    hit = (found & truth).sum()
    assert hit / truth.sum() > 0.5 and hit / found.sum() > 0.7
