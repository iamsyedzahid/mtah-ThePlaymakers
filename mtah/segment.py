"""Possession segmentation (R4).

Naive: possession flips only on a reception, at the reception's own frame; deflections and
evidence never flip it. Before the first reception the sign of the summed evidence picks the team.
D comes only from dead_mask, and H/A runs shorter than a second are absorbed by a neighbour.
lost_mask is not used yet: possession simply holds through lost frames.
"""
from typing import Sequence

import numpy as np

from mtah.config import FPS
from mtah.interfaces import A, AWAY, D, H, HOME, LABEL_DTYPE, BoolArray, FloatArray, LabelArray, Touch

MIN_RUN_FRAMES = FPS  # shorter H/A runs are flicker


def run_bounds(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) of each run of equal values."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return np.r_[0, change], np.r_[change, labels.size]


def segment(
    e: FloatArray, lost_mask: BoolArray, dead_mask: BoolArray, touches: Sequence[Touch]
) -> LabelArray:
    e = np.asarray(e, dtype=float)
    lost = np.asarray(lost_mask, dtype=bool)
    dead = np.asarray(dead_mask, dtype=bool)
    T = e.shape[0]
    if e.ndim != 1 or lost.shape != (T,) or dead.shape != (T,):
        raise ValueError("e, lost_mask and dead_mask must all be [T]")
    if any(not 0 <= t.frame < T for t in touches):
        raise ValueError(f"touch frame outside [0, {T})")

    receptions = sorted((t.frame, t.team) for t in touches if t.kind == "reception")
    frames = np.array([f for f, _ in receptions], dtype=int)
    teams = np.array([k for _, k in receptions], dtype=int)

    # Before the first reception, go with the evidence; with no evidence either, guess home,
    # because while the ball is alive there is always a team.
    lead = e[: frames[0]].sum() if frames.size else e.sum()
    if lead != 0:
        first = HOME if lead > 0 else AWAY
    else:
        first = teams[0] if teams.size else HOME

    last = np.searchsorted(frames, np.arange(T), side="right") - 1  # latest reception at or before f
    state = np.where(last >= 0, teams[np.maximum(last, 0)] if teams.size else first, first)

    labels = np.where(state == HOME, H, A).astype(LABEL_DTYPE)
    labels[dead] = D
    return _absorb_short_runs(labels)


def _absorb_short_runs(labels: LabelArray) -> LabelArray:
    """Relabel H/A runs shorter than MIN_RUN_FRAMES with the neighbouring possession."""
    out = labels.copy()
    for start, end in zip(*run_bounds(labels)):
        if out[start] == D or end - start >= MIN_RUN_FRAMES:
            continue
        before = out[start - 1] if start > 0 else D
        after = out[end] if end < out.size else D
        if before != D:
            out[start:end] = before
        elif after != D:
            out[start:end] = after
    return out
