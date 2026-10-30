"""Dead ball (R4): D intervals from Smart Tagging set-piece windows and the ball itself.

A tag window (SetPiece) is about 5 s around a restart and partly live play, so it is never D
whole. Inside it the restart is found from the ball: before a restart the ball rests (placed for
a free kick, corner, goal kick or kick-off) or is off the pitch (throw-in, corner); the restart is
the moment that "quiet" spell ends. D runs from the start of that spell, at most MAX_DEAD_S
earlier, to the frame before the restart.

- quiet: a trusted ball slower than STILL_MS, or outside the pitch. Lost stretches shorter than
  BRIDGE_FRAMES inside a quiet spell do not end it (a placed ball briefly hidden by players).
- the spell must last STILL_MIN_FRAMES and end inside the window; of several, the longest wins,
  as a trap after the restart is a short rest.
- no such spell (often the ball is lost right at the restart): a short D from the start of the
  window to its middle, as the tag says a restart happened around there.
"""
from typing import Sequence

import numpy as np

from mtah.ball import _gaps
from mtah.config import FPS, PITCH_M
from mtah.interfaces import BoolArray, FloatArray, Interval, SetPiece

STILL_MS = 1.0
STILL_MIN_FRAMES = FPS // 2
BRIDGE_FRAMES = 3 * FPS  # lost_mask gaps are at least 1 s, so a shorter bridge would never apply
MAX_DEAD_S = 30


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
    spells = _quiet_spells(ball, lost)
    out: list[Interval] = []
    for sp in set_pieces:
        if not 0 <= sp.start <= sp.end < T:
            raise ValueError(f"set piece ({sp.start}, {sp.end}) is not inside [0, {T})")
        inside = [(a, b) for a, b in spells if sp.start < b <= sp.end + 1]
        if not inside:
            out.append((sp.start, (sp.start + sp.end) // 2))
            continue
        a, b = max(inside, key=lambda run: run[1] - run[0])
        out.append((max(a, b - MAX_DEAD_S * FPS), b - 1))
    return out


def _quiet_spells(ball: FloatArray, lost: BoolArray) -> list[tuple[int, int]]:
    """(start, end exclusive) of each spell of resting or off-pitch ball."""
    speed = np.linalg.norm(np.gradient(ball, axis=0), axis=1) * FPS if len(ball) > 1 else np.zeros(len(ball))
    off = (ball[:, 0] < 0) | (ball[:, 0] > PITCH_M[0]) | (ball[:, 1] < 0) | (ball[:, 1] > PITCH_M[1])
    quiet = np.logical_not(lost) & ((speed < STILL_MS) | off)
    for start, end in zip(*_gaps(lost)):  # bridge short hidden stretches between quiet frames
        if end - start < BRIDGE_FRAMES and start > 0 and end < len(quiet) and quiet[start - 1] and quiet[end]:
            quiet[start:end] = True
    return [(int(a), int(b)) for a, b in zip(*_gaps(quiet)) if b - a >= STILL_MIN_FRAMES]
