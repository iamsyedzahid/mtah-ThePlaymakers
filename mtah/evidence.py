"""Control evidence (R2): which team looks in control of the ball, frame by frame.

Output e[T] in [-1, 1] (see interfaces.evidence): positive favours home, negative away, 0 means
no information. Each team gets a soft control score c from its players' distances to the ball,
and a fixed "nobody" score c0 stands for a loose ball:

    e = (c_home - c_away) / (c_home + c_away + c0)

When someone is close, c0 is negligible and e = 2 * p_home - 1. As both teams move away, e fades
smoothly to 0 instead of being cut off at a distance. The score is bounded by construction.

Main version (`evidence`): each team's nearest player, c = exp(-d^2 / (2 * SIGMA_M^2)).
SIGMA_M = 2 m because a player can play the ball within about one stride plus leg reach; beyond
about 4 m control is unlikely. c0 is the score of a player NOBODY_M = 5 m away.

Dwell: a short masked average over DWELL_FRAMES (0.2 s) rewards sustained proximity without
blurring hand-overs; R4 does the long-range smoothing. Frames where ball_clean's lost_mask is
True are exactly 0, so R4 holds the previous team instead of guessing.

Two alternatives with the same signature, for voting:
- `evidence_weighted`: every player within WEIGHT_RADIUS_M counts, weight 1 / (d + 0.5 m), so
  support and pressure show.
- `evidence_speed_aware`: as `evidence`, but 0 while the ball moves faster than FLIGHT_SPEED_MS.
  A pass or shot in flight is nobody's, so it never flips to an opponent it flies past.
"""
import numpy as np

from mtah.ball import ball_clean
from mtah.config import FPS
from mtah.interfaces import AWAY, HOME, BoolArray, FloatArray, Tracking

SIGMA_M = 2.0
NOBODY_M = 5.0
DWELL_FRAMES = 5
WEIGHT_RADIUS_M = 10.0
WEIGHT_SOFTENING_M = 0.5
FLIGHT_SPEED_MS = 12.0  # faster than any dribble; passes and shots are in flight


def evidence(tracking: Tracking) -> FloatArray:
    ball, lost = _prepare(tracking)
    return _dwell(_gaussian(tracking, ball), ~lost)


def evidence_weighted(tracking: Tracking) -> FloatArray:
    ball, lost = _prepare(tracking)
    dist = _distances(tracking, ball)
    near = dist <= WEIGHT_RADIUS_M  # False where dist is inf
    weight = np.where(near, 1.0 / (np.where(near, dist, 0.0) + WEIGHT_SOFTENING_M), 0.0)
    c_home = weight[:, tracking.team == HOME].sum(axis=1)
    c_away = weight[:, tracking.team == AWAY].sum(axis=1)
    c0 = 1.0 / (WEIGHT_RADIUS_M + WEIGHT_SOFTENING_M)
    return _dwell(_share(c_home, c_away, c0), ~lost)


def evidence_speed_aware(tracking: Tracking) -> FloatArray:
    ball, lost = _prepare(tracking)
    speed = np.linalg.norm(np.gradient(ball, axis=0), axis=1) * FPS if len(ball) > 1 else np.zeros(len(ball))
    return _dwell(_gaussian(tracking, ball), ~lost & (speed <= FLIGHT_SPEED_MS))


def _prepare(tracking: Tracking) -> tuple[FloatArray, BoolArray]:
    if tracking.fps != FPS:
        raise NotImplementedError(f"only {FPS} fps is supported, got {tracking.fps}")
    return ball_clean(tracking.ball)


def _distances(tracking: Tracking, ball: FloatArray) -> FloatArray:
    """[T, S] ball-to-slot distance in metres, inf where the slot is not tracked."""
    dist = np.linalg.norm(tracking.players - ball[:, None, :], axis=2)
    return np.where(np.isnan(dist), np.inf, dist)


def _gaussian(tracking: Tracking, ball: FloatArray) -> FloatArray:
    dist = _distances(tracking, ball)

    def score(side: int) -> FloatArray:
        nearest = dist[:, tracking.team == side].min(axis=1, initial=np.inf)
        return np.exp(-(nearest**2) / (2 * SIGMA_M**2))  # 0 where nobody of that team is tracked

    return _share(score(HOME), score(AWAY), np.exp(-(NOBODY_M**2) / (2 * SIGMA_M**2)))


def _share(c_home: FloatArray, c_away: FloatArray, c0: float) -> FloatArray:
    return (c_home - c_away) / (c_home + c_away + c0)


def _dwell(e: FloatArray, valid: BoolArray) -> FloatArray:
    """Centred average over DWELL_FRAMES of valid frames only; invalid frames are exactly 0."""
    if len(e) < DWELL_FRAMES:
        return np.where(valid, e, 0.0)
    kernel = np.ones(DWELL_FRAMES)
    total = np.convolve(np.where(valid, e, 0.0), kernel, mode="same")
    count = np.convolve(valid.astype(float), kernel, mode="same")
    return np.where(valid, total / np.maximum(count, 1.0), 0.0)


if __name__ == "__main__":
    # Real data: run this yourself. Prints shares over graded frames only, never per-frame values.
    from mtah.config import GRADED_HALVES
    from mtah.io_atd import default_cache, load_tracking

    tracking = load_tracking(default_cache())
    graded = np.zeros(tracking.n_frames, dtype=bool)
    for first, last in GRADED_HALVES:
        graded[first : last + 1] = True
    for name, fn in (("evidence", evidence), ("weighted", evidence_weighted), ("speed_aware", evidence_speed_aware)):
        e = fn(tracking)[graded]
        print(f"{name:>12}: zero {np.mean(e == 0):.3f}  home>0 {np.mean(e > 0):.3f}  away<0 {np.mean(e < 0):.3f}"
              f"  |e|>0.5 {np.mean(np.abs(e) > 0.5):.3f}  mean|e| {np.abs(e).mean():.3f}")
