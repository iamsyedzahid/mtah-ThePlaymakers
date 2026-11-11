"""Possession segmentation (R4)."""
from typing import Sequence
import numpy as np

from mtah.config import FPS
from mtah.interfaces import A, AWAY, D, H, HOME, LABEL_DTYPE, BoolArray, FloatArray, LabelArray, Touch

MIN_RUN_FRAMES = 2 * FPS  # shorter H/A runs are flicker; 2 s chosen on open games, not on our match


def run_bounds(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) of each run of equal values."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return np.r_[0, change], np.r_[change, labels.size]


def _merge_short_runs(labels: np.ndarray) -> np.ndarray:
    """Absorb H/A runs shorter than MIN_RUN_FRAMES into a neighbouring H/A run."""
    labels = labels.copy()
    for _ in range(10):  # repeat until stable (bounded)
        starts, ends = run_bounds(labels)
        changed = False
        for i, (s, en) in enumerate(zip(starts, ends)):
            lab = labels[s]
            if lab == D or (en - s) >= MIN_RUN_FRAMES:
                continue
            left = labels[starts[i - 1]] if i > 0 else None
            right = labels[starts[i + 1]] if i + 1 < len(starts) else None
            left_ok = left is not None and left != D
            right_ok = right is not None and right != D
            if left_ok and right_ok:
                ll = ends[i - 1] - starts[i - 1]
                rl = ends[i + 1] - starts[i + 1]
                new = left if ll >= rl else right
            elif left_ok:
                new = left
            elif right_ok:
                new = right
            else:
                continue
            if new != lab:
                labels[s:en] = new
                changed = True
        if not changed:
            break
    return labels


def segment(
    e: FloatArray, lost_mask: BoolArray, dead_mask: BoolArray, touches: Sequence[Touch]
) -> LabelArray:
    """
    Turn evidence and touches into labels [T] in {H, A, D}. Never X.

    Possession flips only at a reception, at the reception's own frame, to that reception's team;
    deflections never flip. It holds through lost frames. D is exactly the dead_mask frames. Before
    the first reception the team is the sign of the evidence up to it. Short H/A runs are absorbed.
    """
    e = np.asarray(e, dtype=float)
    lost = np.asarray(lost_mask, dtype=bool)
    dead = np.asarray(dead_mask, dtype=bool)
    T = e.shape[0]

    if e.ndim != 1 or lost.shape != (T,) or dead.shape != (T,):
        raise ValueError("e, lost_mask and dead_mask must all be [T]")
    if any(not 0 <= t.frame < T for t in touches):
        raise ValueError(f"touch frame outside [0, {T})")

    if T == 0:
        return np.full(0, H, dtype=LABEL_DTYPE)

    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    usable = ~dead & (e != 0)
    receptions = sorted((t for t in touches if t.kind == "reception"), key=lambda t: t.frame)

    # who has the ball before the first reception
    first = receptions[0].frame if receptions else T
    pre = usable[:first]
    if pre.any():
        home = float(e[:first][pre].sum()) >= -1e-9
    elif receptions:
        home = receptions[0].team == HOME
    else:
        home = True

    is_home = np.empty(T, dtype=bool)
    prev = 0
    for r in receptions:
        new = r.team == HOME
        if new == home:
            continue
        is_home[prev : r.frame] = home
        prev, home = r.frame, new
    is_home[prev:] = home

    labels = np.where(is_home, H, A).astype(LABEL_DTYPE)
    labels[dead] = D
    return _merge_short_runs(labels)