"""Possession evidence (R2).

evidence: the rule-only pipeline's team evidence (run.predict). Tracking in, e [T] float in [-1, 1]
    out, positive for home, always finite, exactly 0 where ball_clean's lost_mask is True: the
    nearest player's team, weighted by exp(-distance / SIGMA_M). R2 chose it by leave-one-game-out
    spellF1 on Metrica's open games, ball masked and displaced like ours: nearest 0.40-0.42 against
    0.39 for the best time-to-ball variant, and calibrating either was a tie, so the simpler one
    stays. (The tuning code, mtah/evidence_fit.py, and the variants that lost are in the history.)

For mtah/fusion.py, through mtah/r2_features.py:

time_lead (with scan_teams): seconds by which home's best player beats away's to the ball. Each
    player's time to reach it is a reaction time plus a straight run at MAX_RUN_MS from where their
    current velocity carries them during the reaction, to where the ball's own velocity carries it
    in the same time (a simple pitch-control model, after Spearman 2018). Each team keeps its best
    time, which makes duplicated slots harmless and slot order irrelevant.

shape_logits: per half, a logistic model on ball-free team-shape features (each a home-minus-away
    difference in each team's own attacking direction), trained on that half's own clear contacts:
    trusted frames where one team's nearest player is within SHAPE_NEAR_M of the ball and the
    other's beyond SHAPE_FAR_M. Those pseudo-labels are made by an algorithm from the tracking; no
    hand labels.
"""
from typing import NamedTuple, Sequence

import numpy as np
from scipy.special import expit

from mtah.ball import ball_clean
from mtah.config import FPS, PITCH_M
from mtah.interfaces import AWAY, HOME, BoolArray, FloatArray, Interval, Tracking

SIGMA_M = 2.0  # evidence decays as exp(-distance / SIGMA_M)

# Time to reach the ball: Spearman's (2018) pitch-control reaction time and running speed.
REACTION_S = 0.7
MAX_RUN_MS = 5.0
TIME_CAP_S = 10.0  # beyond this a team is simply far away; caps keep the difference finite

# Velocity from positions over about 0.5 s: a central difference of +-VEL_HALF_FRAMES, one-sided
# where one end is missing, 0 where both are. A slot moving faster than any footballer runs is a
# track that jumped to another person, so its velocity is not trusted either.
VEL_HALF_FRAMES = 6  # 12 frames = 0.48 s
MAX_PLAUSIBLE_MS = 12.0

BLOCK_FRAMES = 16_384  # frames per block, so the full match never holds several [T, S, 2] copies

# Team shape on lost frames.
SHAPE_NEAR_M = 1.5  # a clear contact: one team's nearest player this close to the trusted ball...
SHAPE_FAR_M = 3.0  # ...and the other team's nearest player at least this far
SHAPE_SMOOTH_FRAMES = 2 * FPS  # features are averaged over a centred 2 s window
SHAPE_RIDGE = 1.0  # L2 penalty on the standardised weights
SHAPE_MIN_LABELS = 100  # clear contacts needed per team in a half, else that half says nothing
SHAPE_LINE = (1, 5)  # sorted from the own goal: skip the deepest (the keeper), average the next 4
SHAPE_COLUMNS = ("block", "width", "length", "line", "forward", "speed")  # every feature shape_features knows
SHAPE_FEATURES = SHAPE_COLUMNS  # the ones the shipped model uses
FIT_DECIMALS = 6  # fitted weights are rounded, so last-bit BLAS differences cannot reach labels
BALL_VEL_HALF_FRAMES = 3  # ball velocity from a central difference over 6 frames (0.24 s)


class Scan(NamedTuple):
    """Per-frame, per-team summaries; column 0 is home and 1 is away."""

    t_best: FloatArray  # [T, 2] best time to the ball (s), inf where a team has no tracked slot
    near: FloatArray  # [T, 2] nearest tracked slot to the ball (m), inf likewise
    n: FloatArray  # [T, 2] tracked slots
    mean: FloatArray  # [T, 2, 2] team mean x, y (m); NaN where untracked
    std: FloatArray  # [T, 2, 2] team std x, y (m)
    vel: FloatArray  # [T, 2, 2] team mean velocity x, y (m/s)
    speed: FloatArray  # [T, 2] team mean speed (m/s)
    line: FloatArray  # [T, 2, 2] defensive line x when attacking towards +x [..., 0] or -x [..., 1]
    ball_speed: FloatArray  # [T] m/s, from the filled ball (meaningless on lost frames)


class ShapeModel(NamedTuple):
    """One half's shape model, or a half that had too few clear contacts (weights None)."""

    first: int
    last: int  # inclusive
    direction: int  # +1 if home attacks towards +x in this half, else -1
    n_home: int  # clear contacts per team
    n_away: int
    scale: FloatArray | None  # [F] RMS of each feature over the clear contacts
    weights: FloatArray | None  # [F] on the standardised features
    features: tuple[str, ...] = SHAPE_FEATURES  # what the F columns are


# -- evidence functions -------------------------------------------------------------------------

def evidence(tracking: Tracking) -> FloatArray:
    _check_fps(tracking)
    _, lost = ball_clean(tracking.ball)
    return nearest_evidence(tracking.ball, lost, tracking.players, tracking.team)


def nearest_evidence(ball: FloatArray, lost: BoolArray, players: FloatArray, team: np.ndarray) -> FloatArray:
    """The nearest player's team, weighted by exp(-distance / SIGMA_M), from the raw ball; 0
    where lost_mask is True or nobody is seen. In frame blocks, like scan_teams, so the full
    match never holds a [T, S, 2] difference array."""
    T = ball.shape[0]
    e = np.zeros(T)
    for a in range(0, T, BLOCK_FRAMES):
        b = min(a + BLOCK_FRAMES, T)
        dist = np.linalg.norm(players[a:b] - ball[a:b, None, :], axis=2)  # [B, S]
        rows = np.flatnonzero(~np.isnan(dist).all(axis=1) & ~lost[a:b])  # ball trusted, a player seen
        if rows.size:
            nearest = np.nanargmin(dist[rows], axis=1)
            sign = np.where(team[nearest] == HOME, 1.0, -1.0)
            e[a + rows] = sign * np.exp(-dist[rows, nearest] / SIGMA_M)
    return e


def ball_velocity(filled: FloatArray) -> FloatArray:
    """[T, 2] m/s from a NaN-free ball track: central difference over 2 * BALL_VEL_HALF_FRAMES,
    the end frames copying their neighbour; 0 for a track too short to difference."""
    k, n = BALL_VEL_HALF_FRAMES, filled.shape[0]
    vel = np.zeros((n, 2))
    if n > 2 * k:
        vel[k : n - k] = (filled[2 * k :] - filled[: n - 2 * k]) * FPS / (2 * k)
        vel[:k], vel[n - k :] = vel[k], vel[n - k - 1]
    return vel


# -- time to the ball ---------------------------------------------------------------------------

def time_lead(t_home: FloatArray, t_away: FloatArray, lost: BoolArray) -> FloatArray:
    """[T] seconds by which home's best player beats away's to the ball (capped at TIME_CAP_S);
    exactly 0 where the ball is lost or a team has no tracked slot."""
    ok = ~lost & np.isfinite(t_home) & np.isfinite(t_away)
    lead = np.zeros(t_home.shape[0])
    lead[ok] = np.minimum(t_away[ok], TIME_CAP_S) - np.minimum(t_home[ok], TIME_CAP_S)
    return lead


def scan_teams(ball: FloatArray, players: FloatArray, team: np.ndarray) -> Scan:
    """Everything the evidence functions need from the players, in one pass over frame blocks.

    ball must be NaN-free (ball_clean's filled track); on lost frames the ball-based columns are
    meaningless and the callers ignore them. Player velocities are computed once per block, with
    2 * VEL_HALF_FRAMES of context either side so blocks give the same result as one pass.
    """
    T = ball.shape[0]
    pad = 2 * VEL_HALF_FRAMES
    ball_vel = ball_velocity(ball)
    target = ball + ball_vel * REACTION_S  # where the ball is once the players react
    inf2, nan2, nan22 = np.full((T, 2), np.inf), np.full((T, 2), np.nan), np.full((T, 2, 2), np.nan)
    scan = Scan(inf2, inf2.copy(), np.zeros((T, 2)), nan22, nan22.copy(), nan22.copy(), nan2, nan22.copy(),
                np.linalg.norm(ball_vel, axis=1))
    sides = [(side, team == side) for side in (HOME, AWAY) if (team == side).any()]
    for a in range(0, T, BLOCK_FRAMES):
        b = min(a + BLOCK_FRAMES, T)
        lo, hi = max(0, a - pad), min(T, b + pad)
        vel = _velocity(players[lo:hi])[a - lo : b - lo]  # [B, S, 2], 0 where unknown
        pos = players[a:b]
        seen = ~np.isnan(pos[..., 0])
        start = pos + vel * REACTION_S  # where each player is once they react
        time = REACTION_S + np.linalg.norm(target[a:b, None, :] - start, axis=2) / MAX_RUN_MS
        time = np.where(seen, time, np.inf)
        dist = np.where(seen, np.linalg.norm(ball[a:b, None, :] - pos, axis=2), np.inf)
        speed = np.linalg.norm(vel, axis=2)
        for side, cols in sides:
            on = seen[:, cols]
            n = on.sum(axis=1)
            scan.t_best[a:b, side] = time[:, cols].min(axis=1)
            scan.near[a:b, side] = dist[:, cols].min(axis=1)
            scan.n[a:b, side] = n
            xy, v = pos[:, cols], vel[:, cols]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(on[..., None], xy, 0.0).sum(axis=1) / n[:, None]
                sq = np.where(on[..., None], xy * xy, 0.0).sum(axis=1) / n[:, None]
                scan.mean[a:b, side] = mean
                scan.std[a:b, side] = np.sqrt(np.maximum(sq - mean * mean, 0.0))
                scan.vel[a:b, side] = np.where(on[..., None], v, 0.0).sum(axis=1) / n[:, None]
                scan.speed[a:b, side] = np.where(on, speed[:, cols], 0.0).sum(axis=1) / n
            x = xy[..., 0]
            first, stop = SHAPE_LINE
            scan.line[a:b, side, 0] = _nanmean_rows(np.sort(x, axis=1)[:, first:stop])
            scan.line[a:b, side, 1] = -_nanmean_rows(np.sort(-x, axis=1)[:, first:stop])
    return scan


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


def _nanmean_rows(values: FloatArray) -> FloatArray:
    """Mean of the non-NaN entries of each row; NaN for a row with none."""
    ok = ~np.isnan(values)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(ok, values, 0.0).sum(axis=1) / n, np.nan)


# -- team shape ---------------------------------------------------------------------------------

def shape_features(scan: Scan, first: int, last: int, direction: int, length: float = PITCH_M[0],
                   features: Sequence[str] = SHAPE_FEATURES) -> FloatArray:
    """[n, F] smoothed home-minus-away shape features for frames first..last (inclusive), the
    columns named in `features`, in that order.

    Each is positive when home looks like the team with the ball: the block is in home's
    attacking half (block), home is wider (width) and longer (length), home's defensive line is
    higher up and away's deeper (line), everyone runs towards away's goal (forward), home runs
    more (speed). Swapping the teams negates every feature exactly, because the attacking
    direction swaps with them. `length` is the pitch length in metres, with x measured from one
    goal line (the corner origin of interfaces.py); only block and line use it.
    """
    sl = slice(first, last + 1)
    d = float(direction)
    own, other = (0, 1) if direction > 0 else (1, 0)  # which line column is each team's own goal side
    columns = {
        "block": lambda: d * (scan.mean[sl, HOME, 0] + scan.mean[sl, AWAY, 0] - length),
        "width": lambda: scan.std[sl, HOME, 1] - scan.std[sl, AWAY, 1],
        "length": lambda: scan.std[sl, HOME, 0] - scan.std[sl, AWAY, 0],
        "line": lambda: d * (scan.line[sl, HOME, own] + scan.line[sl, AWAY, other] - length),
        "forward": lambda: d * (scan.vel[sl, HOME, 0] + scan.vel[sl, AWAY, 0]),
        "speed": lambda: scan.speed[sl, HOME] - scan.speed[sl, AWAY],
    }
    if not features or set(features) - set(SHAPE_COLUMNS) or len(set(features)) != len(features):
        raise ValueError(f"shape features must be distinct names from {SHAPE_COLUMNS}, got {tuple(features)}")
    return _smooth_nan(np.column_stack([columns[name]() for name in features]), SHAPE_SMOOTH_FRAMES)


def attack_direction(scan: Scan, first: int, last: int) -> int:
    """+1 if home's average position in the half is nearer x = 0 (so home attacks towards +x)."""
    mx = scan.mean[first : last + 1, :, 0]
    both = ~np.isnan(mx).any(axis=1)
    if not both.any():
        return 1
    return 1 if float(np.mean(mx[both, AWAY] - mx[both, HOME])) >= 0 else -1


def clear_contacts(scan: Scan, lost: BoolArray) -> FloatArray:
    """[T] +1 where home clearly touches the trusted ball, -1 for away, 0 otherwise."""
    near_home, near_away = scan.near[:, HOME], scan.near[:, AWAY]
    home = ~lost & (near_home < SHAPE_NEAR_M) & (near_away > SHAPE_FAR_M)
    away = ~lost & (near_away < SHAPE_NEAR_M) & (near_home > SHAPE_FAR_M)
    return home.astype(float) - away.astype(float)


class ShapeFit(NamedTuple):
    """A shape model fitted once on other matches' clear contacts (fit_shape), for shape_logits."""

    scale: FloatArray  # [F] RMS of each feature over the clear contacts
    weights: FloatArray  # [F] on the standardised features


def _fit(x: FloatArray, y: FloatArray) -> ShapeFit:
    scale = np.sqrt(np.mean(x ** 2, axis=0))
    scale = np.array([float(f"{s:.6g}") if s > 0 else 1.0 for s in scale])
    return ShapeFit(scale, np.round(fit_logistic(y[:, None] * x / scale, SHAPE_RIDGE), FIT_DECIMALS))


def shape_contacts(scan: Scan, lost: BoolArray, halves: Sequence[Interval], length: float = PITCH_M[0],
                   features: Sequence[str] = SHAPE_FEATURES) -> tuple[FloatArray, FloatArray]:
    """(x [n, F], y [n] of +-1): every half's clear contacts as shape features, for fit_shape."""
    labels = clear_contacts(scan, lost)
    xs, ys = [np.zeros((0, len(tuple(features))))], [np.zeros(0)]
    for first, last in halves:
        x = shape_features(scan, first, last, attack_direction(scan, first, last), length, features)
        y = labels[first : last + 1]
        use = (y != 0) & np.isfinite(x).all(axis=1)
        xs.append(x[use])
        ys.append(y[use])
    return np.vstack(xs), np.concatenate(ys)


def fit_shape(contacts: Sequence[tuple[FloatArray, FloatArray]]) -> ShapeFit:
    """One shape model from the clear contacts of other matches (shape_contacts of each), pooled."""
    return _fit(np.vstack([x for x, _ in contacts]), np.concatenate([y for _, y in contacts]))


def shape_logits(scan: Scan, lost: BoolArray, halves: Sequence[Interval], everywhere: bool = False,
                 length: float = PITCH_M[0], features: Sequence[str] = SHAPE_FEATURES,
                 fitted: ShapeFit | None = None) -> tuple[FloatArray, list[ShapeModel]]:
    """([T] logit of p_home, one model per half). With fitted (a model fitted on other matches, for
    the submission), every half uses it; without, each half's model is trained on that half's own
    clear contacts only. The logit is set inside halves that have a model, on lost frames only
    (or on every frame with shape features, if everywhere), and is exactly 0 everywhere else.
    halves are inclusive (first, last) pairs; length and features as in shape_features."""
    features = tuple(features)
    labels = clear_contacts(scan, lost)
    logit = np.zeros(lost.shape[0])
    models = []
    for first, last in halves:
        direction = attack_direction(scan, first, last)
        x = shape_features(scan, first, last, direction, length, features)  # computed once: fit, then predict
        finite = np.isfinite(x).all(axis=1)
        y = labels[first : last + 1]
        use = (y != 0) & finite
        n_home, n_away = int((use & (y > 0)).sum()), int((use & (y < 0)).sum())
        scale = weights = None
        if fitted is not None:
            scale, weights = fitted
        elif min(n_home, n_away) >= SHAPE_MIN_LABELS:
            scale, weights = _fit(x[use], y[use])
        if weights is not None:
            on = finite if everywhere else lost[first : last + 1] & finite
            logit[first : last + 1][on] = (x[on] / scale) @ weights
        models.append(ShapeModel(first, last, direction, n_home, n_away, scale, weights, features))
    return logit, models


def fit_logistic(z: FloatArray, ridge: float, max_iter: int = 100, tol: float = 1e-10) -> FloatArray:
    """Weights w maximising sum(log sigmoid(z @ w)) - ridge / 2 * |w|^2, by Newton's method.

    z holds label * features (labels +-1), so there is no intercept and the problem is the same
    when every label and feature changes sign: swapping the teams leaves w unchanged. Concave, so
    Newton converges from 0; no randomness anywhere.
    """
    w = np.zeros(z.shape[1])
    for _ in range(max_iter):
        p = expit(z @ w)
        grad = z.T @ (1.0 - p) - ridge * w
        hess = (z * (p * (1.0 - p))[:, None]).T @ z + ridge * np.eye(z.shape[1])
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < tol:
            break
    return w


def _smooth_nan(x: FloatArray, width: int) -> FloatArray:
    """Centred moving average over `width` frames of each column, ignoring NaN; NaN where a
    frame itself is NaN, so smoothing never invents a value for an untracked frame."""
    ok = np.isfinite(x)
    vals = np.where(ok, x, 0.0)
    pad_lo, pad_hi = width // 2, width - 1 - width // 2

    def window_sum(a: FloatArray) -> FloatArray:
        c = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
        idx = np.arange(a.shape[0])
        hi = np.minimum(idx + pad_hi + 1, a.shape[0])
        lo = np.maximum(idx - pad_lo, 0)
        return c[hi] - c[lo]

    with np.errstate(invalid="ignore", divide="ignore"):
        out = window_sum(vals) / window_sum(ok.astype(float))
    return np.where(ok, out, np.nan)


def _check_fps(tracking: Tracking) -> None:
    if tracking.fps != FPS:
        raise NotImplementedError(f"only {FPS} fps is supported, got {tracking.fps}")
