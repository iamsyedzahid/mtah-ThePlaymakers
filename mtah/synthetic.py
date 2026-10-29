"""Seeded synthetic match for tests and dry runs. Never reads real data.

22 players drift around fixed anchors on a corner-origin pitch. The ball sits on a carrier, then
is passed at PASS_SPEED_MS to a teammate or, on a turnover, intercepted by the nearest opponent.
To keep it from being too easy, as the real match is not:
- the ball runs 0.3-1.2 m ahead of its carrier and an opponent presses most carriers from 1-3 m,
  so the nearest player is not always the owner;
- some passes clip an opponent on the way (a deflection: no control, possession does not change);
- many ball gaps start just before a reception, as the real ball is lost most in crowded moments.
Some turnovers are preceded by a dead-ball spell. About 40% of ball frames are lost, a few ball
samples are glitches, and about 7% of player positions are missing. As in the real ATD (DATA.md),
the 22 players are spread over 46 anonymous slots: a track breaks now and then and the player
reappears in a free slot of the same team.
"""
import math
from typing import NamedTuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

from mtah.config import FPS, PITCH_M
from mtah.interfaces import A, D, H, HOME, AWAY, LABEL_DTYPE, Interval, LabelArray, Tracking

PLAYERS_PER_TEAM = 11
PASS_SPEED_MS = 22.0  # below ball.SPEED_CAP_MS, so clean passes survive cleaning
BALL_NOISE_M = 0.15
TURNOVER_PROB = 0.15  # chance that a change of carrier goes to the other team
DEFLECT_PROB = 0.15  # chance that a pass clips an opponent on its way
PRESS_PROB = 0.7  # chance that an opponent presses the carrier
PRESS_RANGE_M = (1.0, 3.0)  # presser to carrier
DRIBBLE_RANGE_M = (0.3, 1.2)  # ball ahead of the carrier, so a presser is sometimes nearer
SET_PIECE_PROB = 0.4  # chance that a turnover is preceded by a dead-ball spell
# Ball gaps as measured on the real ATD (graded frames, aggregate only): 40.7% lost in 150 gaps,
# median 268 frames. An exponential with mean 430 has median 298, close enough.
BALL_LOST_FRAC = 0.407
MEAN_GAP_FRAMES = 430
CROWD_GAP_PROB = 0.6  # chance that a gap is moved to start just before the next reception
GLITCH_FRAC = 0.001  # ball samples thrown metres off, as tracking glitches are
PLAYER_LOST_FRAC = 0.07  # DATA.md: 20.5 of 22 players tracked on average
SLOTS_PER_TEAM = 23  # 46 slots in all, as in the real ATD
MEAN_TRACK_S = 40  # a player's track breaks about this often


class Synthetic(NamedTuple):
    tracking: Tracking
    labels: LabelArray  # ground truth, H/A/D
    set_piece_intervals: list[Interval]
    receptions: list[tuple[int, int]]  # (frame, team) where a carrier takes control
    deflections: list[tuple[int, int]]  # (frame, team) where an opponent clips a pass


def make_synthetic(n_frames: int = 5_000, seed: int = 0) -> Synthetic:
    rng = np.random.default_rng(seed)
    T, S = n_frames, 2 * PLAYERS_PER_TEAM
    pitch = np.array(PITCH_M)
    half = pitch / 2
    team = np.repeat([HOME, AWAY], PLAYERS_PER_TEAM).astype(np.int8)
    step = PASS_SPEED_MS / FPS  # metres per frame

    # Players drift around a fixed anchor, home in the left half and away in the right.
    offsets = rng.uniform(-0.8, 0.8, size=(S, 2)) * half
    offsets[:, 0] = np.where(team == HOME, -1, 1) * np.abs(offsets[:, 0])
    drift = lfilter([1.0], [1.0, -0.9995], rng.normal(0.0, 0.15, size=(T, S, 2)), axis=0)
    drift = uniform_filter1d(drift, size=FPS, axis=0)  # smooth paths, not 5 m/s frame-to-frame jitter
    players = np.clip(half + offsets + drift, 0.0, pitch)

    def nearest(pool: np.ndarray, frame: int, point: np.ndarray) -> int:
        return int(pool[np.argmin(np.linalg.norm(players[frame, pool] - point, axis=1))])

    # Stints: a pass travels (via a deflector, sometimes), then the carrier holds the ball 1-10 s.
    # Turnovers are forced at the quarter marks so even a short match has a few.
    starts, carriers, travel, leg1, deflector = [0], [int(rng.integers(S))], [0], [0], [-1]
    forced = [T // 4, T // 2, 3 * T // 4]
    while (start := starts[-1] + travel[-1] + int(rng.integers(FPS, 10 * FPS))) < T:
        prev = carriers[-1]
        flip = rng.random() < TURNOVER_PROB
        while forced and start >= forced[0]:
            forced.pop(0)
            flip = True
        opponents = np.flatnonzero(team != team[prev])
        if flip:  # the nearest opponent intercepts
            new = nearest(opponents, start, players[start, prev])
        else:
            new = int(rng.choice(np.flatnonzero((team == team[prev]) & (np.arange(S) != prev))))
        a, b = players[start, prev], players[start, new]
        defl = -1
        if not flip and rng.random() < DEFLECT_PROB:  # an opponent near the middle clips it
            defl = nearest(opponents, start, (a + b) / 2)
        if defl >= 0:
            first = max(1, math.ceil(np.linalg.norm(players[start, defl] - a) / step))
            second = max(1, math.ceil(np.linalg.norm(b - players[start, defl]) / step))
        else:
            first, second = max(1, math.ceil(np.linalg.norm(b - a) / step)), 0
        starts.append(start)
        carriers.append(new)
        travel.append(first + second)
        leg1.append(first)
        deflector.append(defl)
    n = len(starts)

    # Pressing: in most holds the nearest opponent closes to 1.5-4 m of the carrier over 1 s.
    for k in range(n):
        hold, end = starts[k] + travel[k], starts[k + 1] if k + 1 < n else T
        if hold >= end or rng.random() >= PRESS_PROB:
            continue
        cur = carriers[k]
        presser = nearest(np.flatnonzero(team != team[cur]), hold, players[hold, cur])
        angle = rng.uniform(0.0, 2 * np.pi)
        gap = rng.uniform(*PRESS_RANGE_M) * np.array([np.cos(angle), np.sin(angle)])
        span = np.arange(hold, end)
        w = np.clip((span - hold) / FPS, 0.0, 1.0)[:, None]
        target = players[span, cur] + gap
        players[span, presser] = np.clip((1 - w) * players[span, presser] + w * target, 0.0, pitch)

    # The ball: leg by leg from the previous carrier (via any deflector) to the new one, then on
    # the carrier. Possession changes only at the reception.
    ball = np.empty((T, 2))
    owner = np.empty(T, dtype=int)
    receptions: list[tuple[int, int]] = []
    deflections: list[tuple[int, int]] = []
    for k in range(n):
        s, end = starts[k], starts[k + 1] if k + 1 < n else T
        cur, prev = carriers[k], carriers[max(k - 1, 0)]
        hold = min(s + travel[k], end)
        if k > 0:
            legs = [(prev, deflector[k], leg1[k]), (deflector[k], cur, travel[k] - leg1[k])]
            if deflector[k] < 0:
                legs = [(prev, cur, travel[k])]
            f0 = s
            for src, dst, length in legs:
                span = np.arange(f0, min(f0 + length, end))
                if span.size:
                    alpha = ((span - f0) / length)[:, None]
                    ball[span] = (1 - alpha) * players[span, src] + alpha * players[span, dst]
                f0 += length
            owner[s:hold] = prev
            if deflector[k] >= 0 and s + leg1[k] < end:
                deflections.append((s + leg1[k], int(team[deflector[k]])))
        angle = rng.uniform(0.0, 2 * np.pi)
        carry = rng.uniform(*DRIBBLE_RANGE_M) * np.array([np.cos(angle), np.sin(angle)])
        ball[hold:end] = players[hold:end, cur] + carry
        owner[hold:end] = cur
        if k > 0 and hold < end:
            receptions.append((hold, int(team[cur])))
    ball += rng.normal(0.0, BALL_NOISE_M, size=(T, 2))
    labels = np.where(team[owner] == HOME, H, A).astype(LABEL_DTYPE)

    # Some turnovers follow a dead-ball spell that ends when the restart is received. The spell
    # leaves at least 1 s of the previous possession visible, so it cannot swallow it.
    intervals: list[Interval] = []
    for k in np.flatnonzero(team[carriers[1:]] != team[carriers[:-1]]) + 1:
        if rng.random() < SET_PIECE_PROB:
            earliest = starts[k - 1] + travel[k - 1] + FPS
            lo = max(starts[k] - int(rng.integers(2 * FPS, 8 * FPS)), earliest)
            hi = min(starts[k] + travel[k] - 1, T - 1)
            intervals.append((lo, hi))
            labels[lo : hi + 1] = D

    # Ball tracking: alternating seen and lost spells of exponential length; many gaps are moved
    # to start just before the next reception, where play is crowded.
    mean_seen = MEAN_GAP_FRAMES * (1 - BALL_LOST_FRAC) / BALL_LOST_FRAC
    rec_frames = np.array([f for f, _ in receptions], dtype=int)
    lost = np.zeros(T, dtype=bool)
    f = int(rng.exponential(mean_seen))
    while f < T:
        if rec_frames.size and rng.random() < CROWD_GAP_PROB:
            j = np.searchsorted(rec_frames, f)
            if j < rec_frames.size and rec_frames[j] - f < 5 * FPS:
                f = max(f, int(rec_frames[j]) - int(rng.integers(0, FPS)))
        length = max(1, math.ceil(rng.exponential(MEAN_GAP_FRAMES)))
        lost[f : f + length] = True
        f += length + max(1, math.ceil(rng.exponential(mean_seen)))
    ball[lost] = np.nan

    glitch = rng.random(T) < GLITCH_FRAC
    ball[glitch] += rng.normal(0.0, 10.0, size=(int(glitch.sum()), 2))  # NaN rows stay NaN
    players[rng.random((T, S)) < PLAYER_LOST_FRAC] = np.nan

    slots, slot_team = _to_slots(players, team, rng)
    return Synthetic(Tracking(ball, slots, slot_team, FPS), labels, intervals, receptions, deflections)


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
