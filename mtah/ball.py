"""Ball gap statistics and cleaning (R1).

Cleaning drops samples that imply a speed over SPEED_CAP_MS (a hard shot reaches 35-40 m/s), smooths the rest lightly and splines
across gaps shorter than SPLINE_GAP_FRAMES. Longer gaps, and gaps at either end of the track, are
flagged in lost_mask and filled linearly, so the output never holds NaN. A final check guarantees
that no two consecutive trusted frames are further apart than the speed cap allows.
"""
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from mtah.config import FPS, PITCH_M
from mtah.interfaces import BoolArray, FloatArray

SPEED_CAP_MS = 40.0
REANCHOR_FRAMES = 5  # a run this long that moves consistently is the ball, even far from the anchor
SMOOTH_FRAMES = 5  # centred moving average, 0.2 s
SPLINE_GAP_FRAMES = FPS  # gaps shorter than 1 s are splined; longer ones are lost
MAX_SPEED_PASSES = 5  # re-checks of the cleaned track before giving up on a too-fast frame


def ball_clean(ball: FloatArray) -> tuple[FloatArray, BoolArray]:
    ball = np.asarray(ball, dtype=float)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError(f"ball must be [T, 2], got {ball.shape}")
    seen = ~np.isnan(ball).any(axis=1)
    seen &= ~_speed_outliers(ball, seen)

    # Smoothing and filling can still create a jump next to a removed sample. Drop the samples on
    # both sides of any trusted too-fast step and fill again; whatever survives that is lost.
    for _ in range(MAX_SPEED_PASSES):
        filled, lost = _fill(ball, seen)
        fast = _too_fast(filled, lost)
        if not fast.any():
            return filled, lost
        seen[:-1] &= ~fast
        seen[1:] &= ~fast
    lost[:-1] |= fast
    lost[1:] |= fast
    return filled, lost


def _fill(ball: FloatArray, seen: BoolArray) -> tuple[FloatArray, BoolArray]:
    """Smooth the seen samples, spline short gaps, fill long gaps linearly and flag them lost."""
    T = len(ball)
    if not seen.any():
        return np.tile(np.array(PITCH_M) / 2, (T, 1)), np.ones(T, dtype=bool)  # centre spot

    smooth = _smooth(ball, seen)
    frames = np.arange(T)
    filled = np.column_stack([np.interp(frames, frames[seen], smooth[seen, k]) for k in range(2)])

    short = np.zeros(T, dtype=bool)
    for start, end in zip(*_gaps(~seen)):
        if start > 0 and end < T and end - start < SPLINE_GAP_FRAMES:
            short[start:end] = True
    if short.any() and seen.sum() >= 2:
        # PCHIP never overshoots its neighbouring samples, unlike a plain cubic spline.
        filled[short] = PchipInterpolator(frames[seen], smooth[seen], axis=0)(frames[short])
    return filled, ~seen & ~short


def _too_fast(filled: FloatArray, lost: BoolArray) -> BoolArray:
    """[T-1] True where a step between two trusted frames is faster than SPEED_CAP_MS."""
    speed = np.linalg.norm(np.diff(filled, axis=0), axis=1) * FPS
    return (speed > SPEED_CAP_MS) & ~lost[:-1] & ~lost[1:]


def gap_stats(lost_mask: BoolArray, fps: int = FPS) -> dict[str, float]:
    lost = np.asarray(lost_mask, dtype=bool)
    starts, ends = _gaps(lost)
    lengths = ends - starts
    longest = int(lengths.max()) if lengths.size else 0
    median = float(np.median(lengths)) if lengths.size else 0.0
    return {
        "n_frames": int(lost.size),
        "n_lost": int(lost.sum()),
        "frac_lost": float(lost.mean()) if lost.size else 0.0,
        "n_gaps": int(lengths.size),
        "longest_gap_frames": longest,
        "median_gap_frames": median,
        "longest_gap_s": longest / fps,
        "median_gap_s": median / fps,
    }


def _gaps(mask: BoolArray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) of each run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _speed_outliers(ball: FloatArray, seen: BoolArray) -> BoolArray:
    """Samples too far from the last accepted sample less than a gap's length earlier.

    A glitch jumps away and comes back. When REANCHOR_FRAMES rejected samples in a row instead
    move consistently with each other, the anchor was the odd one out (or the tracker moved on):
    they are accepted and become the new anchor, so one bad sample cannot wipe out a second.
    """
    max_step = SPEED_CAP_MS / FPS  # metres per frame
    xs, ys = ball[:, 0].tolist(), ball[:, 1].tolist()

    def fits(a: int, b: int) -> bool:
        return math.hypot(xs[b] - xs[a], ys[b] - ys[a]) <= max_step * (b - a)

    bad = np.zeros(len(ball), dtype=bool)
    last = -1
    chain: list[int] = []
    for f in np.flatnonzero(seen).tolist():
        if last >= 0 and f - last < SPLINE_GAP_FRAMES and not fits(last, f):
            bad[f] = True
            if chain and not (f - chain[-1] <= 2 and fits(chain[-1], f)):
                chain = []
            chain.append(f)
            if len(chain) >= REANCHOR_FRAMES:
                bad[chain] = False
                last, chain = f, []
        else:
            last, chain = f, []
    return bad


def _smooth(ball: FloatArray, seen: BoolArray) -> FloatArray:
    """Moving average over seen samples only; values on unseen frames are meaningless."""
    if len(ball) < SMOOTH_FRAMES:
        return ball.copy()
    kernel = np.ones(SMOOTH_FRAMES)
    weight = np.maximum(np.convolve(seen.astype(float), kernel, mode="same"), 1.0)
    return np.column_stack(
        [np.convolve(np.where(seen, ball[:, k], 0.0), kernel, mode="same") / weight for k in range(2)]
    )
