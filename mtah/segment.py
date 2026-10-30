"""Possession segmentation (R4): labels H, A, D from evidence, receptions and dead ball.

spellF1 matches whole possessions by overlap, so a false flip costs a whole extra spell while a
few frames of lateness cost almost nothing. Hence:

1. Possession may change only at a reception (R3) by the other team, or freely after a dead
   ball. Evidence alone never flips it: a slow pass rolling past an opponent is not a turnover.
2. Which receptions to follow is one Viterbi pass over two states, H and A. Between candidate
   frames the cost of a state is EVIDENCE_WEIGHT * the evidence against it, summed; ignoring a
   reception by the other team costs IGNORE_RECEPTION. A presser's false reception, followed by
   evidence for the carrier's team, is ignored; one followed by evidence for the presser's team
   is not. Lost and dead frames carry no evidence, so possession holds through a lost ball.
3. A change found just after a lost stretch (the ball reappears with the other team) happened
   somewhere inside it; it is placed GAP_SHIFT of the way into the stretch.
4. dead_mask frames are D, and only those. H/A runs shorter than MIN_RUN_FRAMES are absorbed by
   a neighbour.
"""
from typing import Sequence

import numpy as np

from mtah.ball import _gaps
from mtah.config import FPS
from mtah.interfaces import A, AWAY, D, H, HOME, LABEL_DTYPE, BoolArray, FloatArray, LabelArray, Touch

MIN_RUN_FRAMES = FPS  # shorter H/A runs are flicker
EVIDENCE_WEIGHT = 1.0
IGNORE_RECEPTION = 1.0  # a tie-breaker: with no evidence follow the reception, else evidence decides
FREE_SWITCH = 1.0  # after a dead ball either team may restart, but with nothing to go on keep the last
GAP_SHIFT = 0.5
GAP_NEAR_FRAMES = FPS  # a reception this soon after a lost stretch may be its hidden change


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
    if T == 0:
        return np.zeros(0, dtype=LABEL_DTYPE)

    # Candidate frames: receptions (switch to their team only) and the end of each dead spell
    # (switch to anyone). The last reception on a frame wins.
    reception = {t.frame: t.team for t in sorted(touches) if t.kind == "reception" and not dead[t.frame]}
    free = {0} | {int(end) for _, end in zip(*_gaps(dead)) if end < T}
    bounds = np.array(sorted(set(reception) | free), dtype=int)
    ends = np.r_[bounds[1:], T]

    useful = np.where(lost | dead, 0.0, e)
    total = np.r_[0.0, np.cumsum(useful)]
    for_home = total[ends] - total[bounds]  # summed evidence of each stretch
    cost = EVIDENCE_WEIGHT * np.column_stack([-for_home, for_home]) / 2  # [stretch, HOME/AWAY]

    n = len(bounds)
    best = np.zeros((n, 2))
    back = np.zeros((n, 2), dtype=int)
    for i in range(n):
        b = int(bounds[i])
        for state in (HOME, AWAY):
            unfollowed = IGNORE_RECEPTION if b in reception and reception[b] != state else 0.0
            if i == 0:
                best[i, state] = cost[i, state] + unfollowed
                continue
            options = []
            for before in (HOME, AWAY):
                if b in free:
                    step = unfollowed + (0.0 if before == state else FREE_SWITCH)
                elif before == state:
                    step = IGNORE_RECEPTION if reception[b] != state else 0.0
                elif reception[b] == state:
                    step = 0.0
                else:
                    continue  # a switch to the team that did not receive
                options.append((best[i - 1, before] + step, before))
            best[i, state], back[i, state] = min(options)
            best[i, state] += cost[i, state]
    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmin(best[-1]))
    for i in range(n - 1, 0, -1):
        states[i - 1] = back[i, states[i]]

    team = np.repeat(states, ends - bounds)
    _move_changes_into_gaps(team, bounds, states, lost, dead, free)
    labels = np.where(team == HOME, H, A).astype(LABEL_DTYPE)
    labels[dead] = D
    return _absorb_short_runs(labels)


def _move_changes_into_gaps(team, bounds, states, lost, dead, free) -> None:
    """A change just after a lost stretch moves GAP_SHIFT of the way into the stretch."""
    gap_starts, gap_ends = _gaps(lost & np.logical_not(dead))
    for i in range(1, len(bounds)):
        b = int(bounds[i])
        if states[i] == states[i - 1] or b in free:
            continue
        j = np.searchsorted(gap_ends, b, side="right") - 1
        if j < 0 or b - gap_ends[j] > GAP_NEAR_FRAMES:
            continue
        start = max(int(gap_starts[j] + GAP_SHIFT * (gap_ends[j] - gap_starts[j])), int(bounds[i - 1]) + 1)
        if start < b and not dead[start:b].any():
            team[start:b] = states[i]


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
