"""R2's per-frame features for the learned layer in mtah/fusion.py.

    X, names = fusion_features(ball, players, team, pitch, halves)

Columns (FEATURE_NAMES), every one home-positive:

    r2_nearest      the nearest player's team, weighted by exp(-distance / 2 m): evidence() of
                    mtah/evidence.py on the ball given. [-1, 1]; 0 where ball_clean calls the ball
                    lost or nobody is tracked.
    r2_reach_lead   seconds by which home's best player beats away's to the ball, both moving
                    (evidence_reach's time lead). [-10, 10]; 0 where lost or a team is untracked.
    r2_shape        the half's own shape model: logit of p_home from team shape alone, fitted on
                    that half's clear contacts (pseudo-labels, no labels). On every frame of the
                    half with both teams tracked; 0 outside the halves, and in a half with fewer
                    than 100 clear contacts per team. Capped at +-SHAPE_LOGIT_CAP.
    r2_shape_lost   r2_shape on lost frames only, 0 elsewhere: what evidence_shape adds.

Calling again with 1 - team gives exactly -X, bit for bit, so the swapped orientation is free.
There is no NaN and no infinity: 0 means no information, as for evidence. (NaN would become -1 in
fusion, which for these columns reads as strong evidence for away.)

Assumes 25 fps and corner-origin metres (interfaces.py): x from 0 to the pitch length. Raises on
positions that look centre-origin, and on halves outside [0, T), empty or overlapping. No
randomness anywhere; about 5 s for a 199k-frame match.
"""
from typing import Sequence

import numpy as np

from mtah import evidence as ev
from mtah.ball import ball_clean
from mtah.config import FPS
from mtah.interfaces import AWAY, HOME, Interval, Tracking

FEATURE_NAMES = ("r2_nearest", "r2_reach_lead", "r2_shape", "r2_shape_lost")
SHAPE_LOGIT_CAP = 20.0  # p_home within 2e-9 of 0 or 1: beyond it a logit only adds outliers
ORIGIN_CHECK_STRIDE = FPS  # the origin check looks at one frame a second
OFF_PITCH_M = 5.0  # a corner-origin position this far outside the pitch is unusual...
OFF_PITCH_SHARE = 0.2  # ...and centre-origin data puts about half of all positions there


def fusion_features(ball: np.ndarray, players: np.ndarray, team: np.ndarray, pitch: Sequence[float],
                    halves: Sequence[Sequence[int]]) -> tuple[np.ndarray, list[str]]:
    """[T, k] float features plus k names, for the learned layer in mtah/fusion.py.

    ball [T, 2] and players [T, S, 2] in corner-origin metres, NaN where missing; team [S], 0 for
    home; pitch (length, width) in metres; halves [start, end) frame pairs. Per-half models are
    fitted inside each half only. See the module docstring for the columns.
    """
    ball, players = (_inf_as_missing(np.asarray(a, dtype=float)) for a in (ball, players))
    tracking = Tracking(ball, players, np.asarray(team), FPS)
    length, width = _check_pitch(pitch)
    inclusive = _check_halves(halves, tracking.n_frames)
    _check_corner_origin(tracking.players, length, width)

    filled, lost = ball_clean(tracking.ball)
    scan = ev.scan_teams(filled, tracking.players, tracking.team)
    nearest = ev.nearest_evidence(tracking.ball, lost, tracking.players, tracking.team)
    lead = ev.time_lead(scan.t_best[:, HOME], scan.t_best[:, AWAY], lost)
    shape, _ = ev.shape_logits(scan, lost, inclusive, everywhere=True, length=length)
    shape = np.clip(shape, -SHAPE_LOGIT_CAP, SHAPE_LOGIT_CAP)
    shape_lost = np.where(lost, shape, 0.0)
    return np.column_stack([nearest, lead, shape, shape_lost]), list(FEATURE_NAMES)


def _inf_as_missing(positions: np.ndarray) -> np.ndarray:
    """An infinite position is a missing one. Copies only when there is one: the players array of
    a full match is about 150 MB."""
    if np.isinf(positions).any():
        return np.where(np.isinf(positions), np.nan, positions)
    return positions


def _check_pitch(pitch: Sequence[float]) -> tuple[float, float]:
    length, width = (float(v) for v in pitch)
    if not (np.isfinite(length) and np.isfinite(width) and length > 0 and width > 0):
        raise ValueError(f"pitch must be (length, width) in metres, got {tuple(pitch)}")
    return length, width


def _check_halves(halves: Sequence[Sequence[int]], n_frames: int) -> tuple[Interval, ...]:
    """Half-open [start, end) pairs to inclusive (first, last), refusing anything that would have
    to be clipped or merged: a silently adjusted half hides an off-by-one."""
    pairs = [(int(start), int(end)) for start, end in halves]
    if not pairs:
        raise ValueError("halves is empty: there is nothing to fit a per-half model on")
    for (start, end), nxt in zip(pairs, pairs[1:] + [None]):
        if not 0 <= start < end <= n_frames:
            raise ValueError(f"half [{start}, {end}) is empty or outside [0, {n_frames})")
        if nxt is not None and nxt[0] < end:
            raise ValueError(f"halves [{start}, {end}) and [{nxt[0]}, {nxt[1]}) overlap or are out of order")
    return tuple((start, end - 1) for start, end in pairs)


def _check_corner_origin(players: np.ndarray, length: float, width: float) -> None:
    """Refuse positions that look centre-origin: the shape features measure x from a goal line."""
    xy = players[::ORIGIN_CHECK_STRIDE].reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not xy.size:
        return
    outside = ((xy < -OFF_PITCH_M) | (xy > np.array([length, width]) + OFF_PITCH_M)).any(axis=1)
    if outside.mean() > OFF_PITCH_SHARE:
        raise ValueError(f"{outside.mean():.0%} of player positions are over {OFF_PITCH_M:g} m outside a "
                         f"{length:g} x {width:g} m pitch: pass corner-origin metres (interfaces.py)")
