"""Seeded synthetic match for tests and dry runs. Never reads real data.

22 players drift around fixed anchors on a corner-origin pitch. The ball sits on a carrier, then
is passed at PASS_SPEED_MS to a teammate or, on a turnover, intercepted by the nearest opponent.
Some turnovers are preceded by a dead-ball spell. The ball's tracking drops out in contiguous gaps
covering about 40% of frames, a few ball samples are glitches, and about 7% of player positions
are missing. As in the real ATD (DATA.md), the 22 players are spread over 46 anonymous slots: a
track breaks now and then and the player reappears in a free slot of the same team.
"""
import math
from typing import NamedTuple

import numpy as np
from scipy.signal import lfilter

from mtah.config import FPS, PITCH_M
from mtah.interfaces import A, D, H, HOME, AWAY, LABEL_DTYPE, Interval, LabelArray, Tracking

PLAYERS_PER_TEAM = 11
PASS_SPEED_MS = 22.0  # below ball.SPEED_CAP_MS, so clean passes survive cleaning
BALL_NOISE_M = 0.15
TURNOVER_PROB = 0.15  # chance that a change of carrier goes to the other team
SET_PIECE_PROB = 0.4  # chance that a turnover is preceded by a dead-ball spell
# Ball gaps as measured on the real ATD (graded frames, aggregate only): 40.7% lost in 150 gaps,
# median 268 frames. An exponential with mean 430 has median 298, close enough.
BALL_LOST_FRAC = 0.407
MEAN_GAP_FRAMES = 430
GLITCH_FRAC = 0.001  # ball samples thrown metres off, as tracking glitches are
PLAYER_LOST_FRAC = 0.07  # DATA.md: 20.5 of 22 players tracked on average
SLOTS_PER_TEAM = 23  # 46 slots in all, as in the real ATD
MEAN_TRACK_S = 40  # a player's track breaks about this often


class Synthetic(NamedTuple):
    tracking: Tracking
    labels: LabelArray  # ground truth, H/A/D
    set_piece_intervals: list[Interval]


def make_synthetic(n_frames: int = 5_000, seed: int = 0) -> Synthetic:
    rng = np.random.default_rng(seed)
    T, S = n_frames, 2 * PLAYERS_PER_TEAM
    pitch = np.array(PITCH_M)
    half = pitch / 2
    frames = np.arange(T)
    team = np.repeat([HOME, AWAY], PLAYERS_PER_TEAM).astype(np.int8)

    # Players drift around a fixed anchor, home in the left half and away in the right.
    offsets = rng.uniform(-0.8, 0.8, size=(S, 2)) * half
    offsets[:, 0] = np.where(team == HOME, -1, 1) * np.abs(offsets[:, 0])
    drift = lfilter([1.0], [1.0, -0.9995], rng.normal(0.0, 0.15, size=(T, S, 2)), axis=0)
    players = np.clip(half + offsets + drift, 0.0, pitch)

    # Stints: a pass travels for travel[k] frames, then the carrier holds the ball for 1-10 s.
    # Turnovers are forced at the quarter marks so even a short match has a few.
    starts, carriers, travel = [0], [int(rng.integers(S))], [0]
    forced = [T // 4, T // 2, 3 * T // 4]
    while (start := starts[-1] + travel[-1] + int(rng.integers(FPS, 10 * FPS))) < T:
        prev = carriers[-1]
        flip = rng.random() < TURNOVER_PROB
        while forced and start >= forced[0]:
            forced.pop(0)
            flip = True
        if flip:  # the nearest opponent intercepts
            pool = np.flatnonzero(team != team[prev])
            new = int(pool[np.argmin(np.linalg.norm(players[start, pool] - players[start, prev], axis=1))])
        else:
            pool = np.flatnonzero((team == team[prev]) & (np.arange(S) != prev))
            new = int(rng.choice(pool))
        distance = float(np.linalg.norm(players[start, new] - players[start, prev]))
        starts.append(start)
        carriers.append(new)
        travel.append(max(1, math.ceil(distance / (PASS_SPEED_MS / FPS))))
    starts, carriers, travel = np.array(starts), np.array(carriers), np.array(travel)

    # The ball travels from the previous carrier to the new one, then sits on the new one.
    stint = np.searchsorted(starts, frames, side="right") - 1
    cur, prev = carriers[stint], carriers[np.maximum(stint - 1, 0)]
    alpha = np.clip((frames - starts[stint]) / np.maximum(travel[stint], 1), 0.0, 1.0)
    alpha[travel[stint] == 0] = 1.0
    ball = (1 - alpha)[:, None] * players[frames, prev] + alpha[:, None] * players[frames, cur]
    ball += rng.normal(0.0, BALL_NOISE_M, size=(T, 2))

    owner = np.where(alpha < 1.0, prev, cur)  # possession changes at the reception
    labels = np.where(team[owner] == HOME, H, A).astype(LABEL_DTYPE)

    # Some turnovers follow a dead-ball spell that ends when the restart is received. The spell
    # leaves at least 1 s of the previous possession visible, so it cannot swallow it.
    intervals: list[Interval] = []
    for i in np.flatnonzero(team[carriers[1:]] != team[carriers[:-1]]) + 1:
        if rng.random() < SET_PIECE_PROB:
            earliest = int(starts[i - 1] + travel[i - 1]) + FPS
            lo = max(int(starts[i]) - int(rng.integers(2 * FPS, 8 * FPS)), earliest)
            hi = min(int(starts[i] + travel[i]) - 1, T - 1)
            intervals.append((lo, hi))
            labels[lo : hi + 1] = D

    # Ball tracking alternates between seen and lost spells of exponential length.
    mean_seen = MEAN_GAP_FRAMES * (1 - BALL_LOST_FRAC) / BALL_LOST_FRAC
    n_pairs = T // MEAN_GAP_FRAMES + 20
    spells = np.ceil(rng.exponential(np.tile([mean_seen, MEAN_GAP_FRAMES], n_pairs))).astype(int)
    ends = np.cumsum(spells)
    assert ends[-1] >= T, "not enough spells drawn"
    ball[np.searchsorted(ends, frames, side="right") % 2 == 1] = np.nan

    glitch = rng.random(T) < GLITCH_FRAC
    ball[glitch] += rng.normal(0.0, 10.0, size=(int(glitch.sum()), 2))  # NaN rows stay NaN
    players[rng.random((T, S)) < PLAYER_LOST_FRAC] = np.nan

    slots, slot_team = _to_slots(players, team, rng)
    return Synthetic(Tracking(ball, slots, slot_team, FPS), labels, intervals)


def _to_slots(players: np.ndarray, team: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Spread each player's track over the anonymous slots of their team."""
    T, S = players.shape[:2]
    slot_team = np.repeat([HOME, AWAY], SLOTS_PER_TEAM).astype(np.int8)
    out = np.full((T, slot_team.size, 2), np.nan)

    breaks = np.cumsum(rng.exponential(MEAN_TRACK_S * FPS, size=(S, T // (MEAN_TRACK_S * FPS) * 2 + 5)), axis=1)
    events = sorted((int(frame), p) for p in range(S) for frame in breaks[p] if 0 < frame < T)
    current = np.where(team == HOME, np.arange(S), np.arange(S) - PLAYERS_PER_TEAM + SLOTS_PER_TEAM)
    since = np.zeros(S, dtype=int)
    for frame, p in events:
        out[since[p] : frame, current[p]] = players[since[p] : frame, p]
        free = np.setdiff1d(np.flatnonzero(slot_team == team[p]), current)
        current[p], since[p] = int(rng.choice(free)), frame
    for p in range(S):
        out[since[p] :, current[p]] = players[since[p] :, p]
    return out, slot_team
