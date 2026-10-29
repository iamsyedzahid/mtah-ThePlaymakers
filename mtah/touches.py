"""Touch detection (R3): frames where a player touches the ball, and whether they took control.

Works on the raw ball, with only R1's speed-outlier removal: R1's smoothing would flatten the
sharp speed change that marks a touch and shift it by about 2 frames. Filled-in positions never
produce a touch.

1. Velocity: a Savitzky-Golay derivative (SG_WINDOW frames = 0.28 s, quadratic), computed on each
   unbroken stretch of trusted ball separately.
2. Candidate: between 2 frames before and 2 after, the speed changes by SPEED_JUMP_MS or more,
   or the direction turns by TURN_DEG or more while the ball moves at TURN_MIN_SPEED_MS. Rolling
   friction changes speed by well under 1 m/s in 0.16 s, so only a foot or body does this. The
   strongest frame of each run of candidates is kept, and peaks closer than MERGE_FRAMES merge.
3. Who: the nearest player within TOUCH_RADIUS_M around that frame. Nobody that close: a bounce
   or tracker noise, dropped.
4. Control: "reception" when the ball then stays within CONTROL_RADIUS_M of a player of the same
   team for CONTROL_MIN of the next CONTROL_WINDOW trusted frames (sustained proximity plus a
   touch event). Too few trusted frames to tell: a slow ball after the touch (under
   SLOW_AFTER_MS) is a reception. Otherwise, or with nothing to go on, a "deflection", which
   never flips possession.
"""
import numpy as np
from scipy.signal import savgol_filter

from mtah.ball import _gaps, _speed_outliers
from mtah.config import FPS
from mtah.interfaces import FloatArray, IntArray, Touch

SG_WINDOW = 7
SG_ORDER = 2
LAG = 2  # compare the ball 2 frames before and after
SPEED_JUMP_MS = 4.0
TURN_DEG = 30.0
TURN_MIN_SPEED_MS = 5.0  # below this a direction is mostly tracker noise; deflections are fast
MERGE_FRAMES = 5  # 0.2 s
TOUCH_RADIUS_M = 1.5
CONTROL_RADIUS_M = 2.0
CONTROL_WINDOW = 15
CONTROL_MIN = 10
CONTROL_LOOKAHEAD = FPS  # the next CONTROL_WINDOW trusted frames must come within 1 s
SLOW_AFTER_MS = 8.0


def touches(ball: FloatArray, players: FloatArray, team: IntArray) -> list[Touch]:
    ball = np.asarray(ball, dtype=float)
    players = np.asarray(players, dtype=float)
    team = np.asarray(team)
    T = len(ball)
    seen = ~np.isnan(ball).any(axis=1)
    trusted = seen & ~_speed_outliers(ball, seen)

    velocity = np.full((T, 2), np.nan)
    for start, end in zip(*_gaps(trusted)):
        if end - start >= SG_WINDOW:
            velocity[start:end] = savgol_filter(ball[start:end], SG_WINDOW, SG_ORDER, deriv=1, axis=0) * FPS
    speed = np.linalg.norm(velocity, axis=1)

    score = _change_score(velocity, speed)
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist) | ~trusted[:, None], np.inf, dist)  # [T, S]

    events = []
    for frame in _peaks(score):
        lo, hi = max(frame - LAG, 0), min(frame + LAG + 1, T)
        window = dist[lo:hi]
        if not np.isfinite(window).any() or window.min() > TOUCH_RADIUS_M:
            continue
        slot = int(np.unravel_index(np.argmin(window), window.shape)[1])
        side = int(team[slot])
        kind = _control(frame, side, dist, team, trusted, speed)
        events.append(Touch(int(frame), side, kind))
    return events


def _change_score(velocity: FloatArray, speed: FloatArray) -> FloatArray:
    """Per frame, how strongly the ball's motion changes around it (1 = at a threshold)."""
    T = len(speed)
    score = np.zeros(T)
    if T <= 2 * LAG:
        return score
    v0, v1 = velocity[: -2 * LAG], velocity[2 * LAG :]
    s0, s1 = speed[: -2 * LAG], speed[2 * LAG :]
    ok = np.isfinite(s0) & np.isfinite(s1)
    jump = np.where(ok, np.abs(s1 - s0), 0.0) / SPEED_JUMP_MS
    moving = ok & (np.maximum(s0, s1) >= TURN_MIN_SPEED_MS) & (s0 > 0) & (s1 > 0)
    cos = np.where(moving, np.sum(v0 * v1, axis=1) / np.where(moving, s0 * s1, 1.0), 1.0)
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) / TURN_DEG
    score[LAG:-LAG] = np.maximum(jump, turn)
    return score


def _peaks(score: FloatArray) -> list[int]:
    """The strongest frame of each run of candidates (score >= 1), then merge close peaks."""
    peaks: list[int] = []
    for start, end in zip(*_gaps(score >= 1.0)):
        frame = start + int(np.argmax(score[start:end]))
        if peaks and frame - peaks[-1] < MERGE_FRAMES:
            if score[frame] > score[peaks[-1]]:
                peaks[-1] = frame
            continue
        peaks.append(frame)
    return peaks


def _control(frame: int, side: int, dist: FloatArray, team: IntArray, trusted, speed) -> str:
    after = np.flatnonzero(trusted[frame + 1 : frame + 1 + CONTROL_LOOKAHEAD]) + frame + 1
    after = after[:CONTROL_WINDOW]
    if len(after) >= CONTROL_MIN:
        own = dist[after][:, team == side].min(axis=1, initial=np.inf)
        return "reception" if np.sum(own <= CONTROL_RADIUS_M) >= CONTROL_MIN else "deflection"
    later = speed[frame + LAG : frame + 1 + CONTROL_LOOKAHEAD]
    later = later[np.isfinite(later)]
    if later.size and later[0] < SLOW_AFTER_MS:
        return "reception"
    return "deflection"


if __name__ == "__main__":
    # Real data: run this yourself. Prints counts only, for a plausibility check.
    from mtah.config import GRADED_HALVES
    from mtah.io_atd import default_cache, load_tracking

    tracking = load_tracking(default_cache())
    events = touches(tracking.ball, tracking.players, tracking.team)
    graded = [e for e in events if any(a <= e.frame <= b for a, b in GRADED_HALVES)]
    minutes = sum(b - a + 1 for a, b in GRADED_HALVES) / FPS / 60
    rec = sum(e.kind == "reception" for e in graded)
    print(f"graded touches {len(graded)} ({len(graded) / minutes:.1f} per minute): "
          f"receptions {rec}, deflections {len(graded) - rec}")
    print(f"receptions by home {sum(e.kind == 'reception' and e.team == 0 for e in graded)}, "
          f"by away {sum(e.kind == 'reception' and e.team == 1 for e in graded)}")
