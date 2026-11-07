"""Possession evidence (R2).

Every function here has the signature of interfaces.evidence: Tracking in, e [T] float in [-1, 1]
out, positive for home, exactly 0 where ball_clean's lost_mask is True.

evidence_reach: time-to-ball control. Each player's time to reach the ball is a reaction time
    plus a straight run at MAX_RUN_MS from where their current velocity carries them during the
    reaction (a simple pitch-control model, after Spearman 2018). Each team keeps its best time,
    which makes duplicated slots harmless and slot order irrelevant, and the difference between
    the two teams' best times goes through a logistic: e = 2 * p_home - 1.

evidence: still the naive placeholder (nearest player, weighted by distance) until the
    leave-one-game-out comparison on the Metrica open games picks the combination to ship.
"""
import math

import numpy as np

from mtah.ball import ball_clean
from mtah.config import FPS
from mtah.interfaces import AWAY, HOME, FloatArray, Tracking

SIGMA_M = 2.0  # placeholder: evidence decays as exp(-distance / SIGMA_M)

# Time to reach the ball. Reaction time and running speed are Spearman's (2018) pitch-control
# values; the control uncertainty sets how fast a time advantage becomes certainty.
REACTION_S = 0.7
MAX_RUN_MS = 5.0
CONTROL_SIGMA_S = 0.45
CONTROL_SCALE_S = CONTROL_SIGMA_S * math.sqrt(3) / math.pi  # logistic scale with that std
TIME_CAP_S = 10.0  # beyond this a team is simply far away; caps keep the difference finite

# Velocity from positions over about 0.5 s: a central difference of +-VEL_HALF_FRAMES, one-sided
# where one end is missing, 0 where both are. A slot moving faster than any footballer runs is a
# track that jumped to another person, so its velocity is not trusted either.
VEL_HALF_FRAMES = 6  # 12 frames = 0.48 s
MAX_PLAUSIBLE_MS = 12.0

BLOCK_FRAMES = 16_384  # frames per block, so the full match never holds several [T, S, 2] copies


def evidence(tracking: Tracking) -> FloatArray:
    _check_fps(tracking)
    _, lost = ball_clean(tracking.ball)
    dist = np.linalg.norm(tracking.players - tracking.ball[:, None, :], axis=2)  # [T, S]
    e = np.zeros(tracking.n_frames)
    rows = np.flatnonzero(~np.isnan(dist).all(axis=1) & ~lost)  # ball trusted and a player seen
    if rows.size:
        nearest = np.nanargmin(dist[rows], axis=1)
        sign = np.where(tracking.team[nearest] == HOME, 1.0, -1.0)
        e[rows] = sign * np.exp(-dist[rows, nearest] / SIGMA_M)
    return e


def evidence_reach(tracking: Tracking) -> FloatArray:
    _check_fps(tracking)
    filled, lost = ball_clean(tracking.ball)
    t_home, t_away = team_times(filled, tracking.players, tracking.team)
    return reach_from_times(t_home, t_away, lost)


def reach_from_times(t_home: FloatArray, t_away: FloatArray, lost: np.ndarray) -> FloatArray:
    """e = 2 * p_home - 1 from each team's best time; 0 where lost or a team is not tracked."""
    ok = ~lost & np.isfinite(t_home) & np.isfinite(t_away)
    lead = np.minimum(t_away[ok], TIME_CAP_S) - np.minimum(t_home[ok], TIME_CAP_S)  # > 0: home first
    e = np.zeros(t_home.shape[0])
    # sign * tanh(|x|) rather than tanh(x), so swapping the teams negates e bit for bit.
    e[ok] = np.sign(lead) * np.tanh(np.abs(lead) / (2.0 * CONTROL_SCALE_S))
    return e


def team_times(ball: FloatArray, players: FloatArray, team: np.ndarray) -> tuple[FloatArray, FloatArray]:
    """Each team's best time to reach the ball, in seconds, [T] each. inf where a team has no
    tracked player. ball must be NaN-free (ball_clean's filled track)."""
    T = ball.shape[0]
    pad = 2 * VEL_HALF_FRAMES
    home, away = team == HOME, team == AWAY
    t_home, t_away = np.full(T, np.inf), np.full(T, np.inf)
    for a in range(0, T, BLOCK_FRAMES):
        b = min(a + BLOCK_FRAMES, T)
        lo, hi = max(0, a - pad), min(T, b + pad)
        vel = _velocity(players[lo:hi])[a - lo : b - lo]
        start = players[a:b] + vel * REACTION_S  # where each player is once they react
        dist = np.linalg.norm(ball[a:b, None, :] - start, axis=2)
        time = np.where(np.isnan(dist), np.inf, REACTION_S + dist / MAX_RUN_MS)
        if home.any():
            t_home[a:b] = time[:, home].min(axis=1)
        if away.any():
            t_away[a:b] = time[:, away].min(axis=1)
    return t_home, t_away


def _velocity(pos: FloatArray) -> FloatArray:
    """[T, S, 2] m/s from [T, S, 2] positions with NaN, as described at VEL_HALF_FRAMES."""
    k, n = VEL_HALF_FRAMES, pos.shape[0]
    vel = np.full(pos.shape, np.nan)
    if n > 2 * k:
        diff = (pos[2 * k :] - pos[: n - 2 * k]) / (2 * k / FPS)  # diff[j] spans frames j .. j + 2k
        vel[k : n - k] = diff  # central: t - k .. t + k
        back = vel[2 * k :]  # where the central one is NaN, try t - 2k .. t, then t .. t + 2k
        np.copyto(back, diff, where=np.isnan(back))
        fwd = vel[: n - 2 * k]
        np.copyto(fwd, diff, where=np.isnan(fwd))
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    return np.where(np.isnan(speed) | (speed > MAX_PLAUSIBLE_MS), 0.0, vel)


def _check_fps(tracking: Tracking) -> None:
    if tracking.fps != FPS:
        raise NotImplementedError(f"only {FPS} fps is supported, got {tracking.fps}")
