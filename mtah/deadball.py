"""Dead ball (R4): D intervals from the ball, with Smart Tagging windows as evidence.

DATA.md: every set piece ends a dead-ball stretch, but the tags are automatic and have false
negatives, and a window is about 5 s around a restart and partly live play, so it is never D whole.
Before a restart the ball rests (placed for a free kick, corner, goal kick or kick-off) or is off
the pitch (throw-in, corner); the restart is the moment that "quiet" spell ends.

- quiet: a trusted ball that moved under STILL_M in the STILL_LAG frames before or after, which
  tracker jitter cannot break up and which ends sharply at a kick, or one beyond OFF_MARGIN_M
  outside the pitch. A spell needs QUIET_MIN_FRAMES. A lost stretch between two spells joins them,
  up to MAX_DEAD_S: the ball seen quiet on both sides. A missing ball alone is never D.

`dead_intervals`, one per tag:
- a spell ending inside the window: D is that spell, at most MAX_DEAD_S; of several the longest
  wins, as a trap after the restart is a short rest;
- else a spell ending before the window with only lost ball in between (the ball vanished during
  the stoppage): from that spell to the middle of the window;
- else a short D from the start of the window to its middle.

`untagged_intervals`: stoppages no tag marks. A spell counts with UNTAGGED_OFF_FRAMES off the pitch,
or UNTAGGED_STILL_S at rest: longer than the 6 s a goalkeeper may hold the ball, which is live.

`slow_lost_intervals`: stoppages with no ball to see. On the real match (aggregates only) the median
player moves at 1.9 m/s with a fast ball, 1.2 m/s in known D and 0.9 m/s with the ball missing. So a
missing ball while players stay under SLOW_MS for SLOW_FRAMES is D: there, 46% of known D counts as
slow and 6% of fast-ball frames does. The missing ball alone still never counts.
"""
from typing import Sequence

import numpy as np

from mtah.ball import _gaps
from mtah.config import FPS, PITCH_M
from mtah.interfaces import BoolArray, FloatArray, Interval, SetPiece, Touch

STILL_M = 0.5
STILL_LAG = FPS // 2
QUIET_MIN_FRAMES = FPS // 2
OFF_MARGIN_M = 1.0  # the tracker is not exact at the line
MAX_DEAD_S = 30
UNTAGGED_OFF_FRAMES = FPS
UNTAGGED_STILL_S = 7
SLOW_MS = 1.25
SLOW_FRAMES = 5 * FPS
SLOW_LOST_FRAMES = 3 * FPS  # a shorter overlap with the missing ball is not worth splitting a spell
SPEED_LAG = 12  # player speed over 2 * SPEED_LAG frames, about 1 s, so jitter is not running
CHUNK_FRAMES = 20_000  # bounds the memory of the per-frame median


def dead_mask(T: int, set_piece_intervals: Sequence[Interval]) -> BoolArray:
    mask = np.zeros(T, dtype=bool)
    for start, end in set_piece_intervals:
        if not 0 <= start <= end < T:
            raise ValueError(f"interval ({start}, {end}) is not inside [0, {T})")
        mask[start : end + 1] = True
    return mask


def dead_intervals(set_pieces: Sequence[SetPiece], ball: FloatArray, lost_mask: BoolArray) -> list[Interval]:
    ball = np.asarray(ball, dtype=float)
    lost = np.asarray(lost_mask, dtype=bool)
    T = len(ball)
    still, off = _quiet_frames(ball, lost)
    spells = _spells(still | off, lost)
    out: list[Interval] = []
    for sp in set_pieces:
        if not 0 <= sp.start <= sp.end < T:
            raise ValueError(f"set piece ({sp.start}, {sp.end}) is not inside [0, {T})")
        middle = (sp.start + sp.end) // 2
        inside = [(a, b) for a, b in spells if sp.start < b <= sp.end + 1]
        if inside:
            a, b = max(inside, key=lambda run: run[1] - run[0])
            out.append((max(a, b - MAX_DEAD_S * FPS), b - 1))
            continue
        before = [(a, b) for a, b in spells if b <= sp.start and lost[b : sp.start + 1].all()]
        if before and middle - before[-1][0] < MAX_DEAD_S * FPS:
            out.append((before[-1][0], middle))
        else:
            out.append((sp.start, middle))
    return out


def untagged_intervals(ball: FloatArray, lost_mask: BoolArray) -> list[Interval]:
    ball = np.asarray(ball, dtype=float)
    lost = np.asarray(lost_mask, dtype=bool)
    still, off = _quiet_frames(ball, lost)
    return [
        (max(a, b - MAX_DEAD_S * FPS), b - 1)
        for a, b in _spells(still | off, lost)
        if off[a:b].sum() >= UNTAGGED_OFF_FRAMES or still[a:b].sum() >= UNTAGGED_STILL_S * FPS
    ]


def slow_lost_intervals(players: FloatArray, lost_mask: BoolArray) -> list[Interval]:
    lost = np.asarray(lost_mask, dtype=bool)
    slow = np.zeros(len(lost), dtype=bool)
    for a, b in zip(*_gaps(_player_speed(np.asarray(players, dtype=float)) < SLOW_MS)):
        if b - a >= SLOW_FRAMES:
            slow[a:b] = True
    return [(int(a), int(b) - 1) for a, b in zip(*_gaps(slow & lost)) if b - a >= SLOW_LOST_FRAMES]


def _player_speed(players: FloatArray) -> FloatArray:
    """Per frame, the median speed of tracked players in m/s; NaN where nobody is tracked."""
    T = len(players)
    span = 2 * SPEED_LAG
    out = np.full(T, np.nan)
    for lo in range(0, max(T - span, 0), CHUNK_FRAMES):
        hi = min(lo + CHUNK_FRAMES, T - span)
        step = np.linalg.norm(players[lo + span : hi + span] - players[lo:hi], axis=2) * FPS / span
        seen = np.isfinite(step)
        count = seen.sum(axis=1)
        rows = np.flatnonzero(count)
        ordered = np.sort(np.where(seen, step, np.inf), axis=1)
        out[lo + SPEED_LAG + rows] = ordered[rows, (count[rows] - 1) // 2]
    return out


def restart_touches(set_pieces: Sequence[SetPiece], intervals: Sequence[Interval], T: int) -> list[Touch]:
    """CHALLENGE.md: a restart has a clear possessor. A tag that names its team gives it."""
    return [
        Touch(end + 1, sp.team, "reception")
        for sp, (_, end) in zip(set_pieces, intervals)
        if sp.team is not None and end + 1 < T
    ]


def _quiet_frames(ball: FloatArray, lost: BoolArray) -> tuple[BoolArray, BoolArray]:
    T = len(ball)
    moved_before, moved_after = np.full(T, np.inf), np.full(T, np.inf)
    if T > STILL_LAG:
        step = np.linalg.norm(ball[STILL_LAG:] - ball[:-STILL_LAG], axis=1)
        moved_before[STILL_LAG:], moved_after[:-STILL_LAG] = step, step
    trusted = np.logical_not(lost)
    still = trusted & ((moved_before < STILL_M) | (moved_after < STILL_M))
    off = trusted & (
        (ball[:, 0] < -OFF_MARGIN_M) | (ball[:, 0] > PITCH_M[0] + OFF_MARGIN_M)
        | (ball[:, 1] < -OFF_MARGIN_M) | (ball[:, 1] > PITCH_M[1] + OFF_MARGIN_M)
    )
    return still, off


def _spells(quiet: BoolArray, lost: BoolArray) -> list[tuple[int, int]]:
    """(start, end exclusive) of each quiet spell, joined across lost stretches between spells."""
    spells: list[tuple[int, int]] = []
    for a, b in zip(*_gaps(quiet)):
        if b - a < QUIET_MIN_FRAMES:
            continue
        if spells and a - spells[-1][1] <= MAX_DEAD_S * FPS and lost[spells[-1][1] : a].all():
            spells[-1] = (spells[-1][0], int(b))
        else:
            spells.append((int(a), int(b)))
    return spells
