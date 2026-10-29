"""Touch detection (R3): frames where a team touches the ball, and whether it took control.

Works on the raw ball, with only R1's speed-outlier removal: R1's smoothing would flatten the
sharp speed change that marks a touch and shift it by about 2 frames. Filled-in positions never
produce a touch.

1. Velocity: a Savitzky-Golay derivative (SG_WINDOW frames = 0.28 s, quadratic), computed on each
   unbroken stretch of trusted ball separately.
2. Candidate: the speed changes by SPEED_JUMP_MS or more, or the direction turns by TURN_DEG or
   more while the ball moves at TURN_MIN_SPEED_MS, between LAG frames before and after; or between
   WIDE_LAG frames, as the provider smooths the ball and a kick can bend it over 0.3 s. Rolling
   friction changes speed by well under 1 m/s in that time, so only a foot or body does this. The
   strongest frame of each run of candidates is kept, and peaks closer than MERGE_FRAMES merge.
3. Who: the nearest player within TOUCH_RADIUS_M around that frame. Nobody that close: a bounce
   or tracker noise, dropped. On the real match half the slow balls are over 1.2 m from the
   nearest tracked position (a body centre, not a foot), so the radius is wider than a stride.
4. Control: a frame is controlled by a team when its nearest player is within CONTROL_RADIUS_M
   and the ball moves with that player (relative speed under CONTROL_REL_SPEED_MS), so a pass
   flying past is not. A team holds the ball from a frame where it controls CONTROL_MIN of the
   next CONTROL_WINDOW frames; spells of one team less than EPISODE_JOIN apart are one spell.
5. Kind: the start of each spell is a "reception" by that team, at the first touch of the team
   within RECEPTION_LEAD frames before it (or at the start itself when no touch was seen, e.g. a
   smoothed-out trap, or the ball reappearing at a player's feet). Any other touch is a reception
   when the next spell within FOLLOW_FRAMES belongs to the toucher's team (an interception played
   on first time to a teammate), or, with too little ball afterwards to tell, when the ball is
   slow after it (under SLOW_AFTER_MS). Otherwise a "deflection", which never flips possession.
"""
import numpy as np
from scipy.signal import savgol_filter

from mtah.ball import _gaps, _speed_outliers
from mtah.config import FPS
from mtah.interfaces import FloatArray, IntArray, Touch

SG_WINDOW = 7
SG_ORDER = 2
LAG = 2  # compare the ball 2 frames before and after
WIDE_LAG = 4
SPEED_JUMP_MS = 4.0
TURN_DEG = 30.0
TURN_MIN_SPEED_MS = 5.0  # below this a direction is mostly tracker noise; deflections are fast
MERGE_FRAMES = 5  # 0.2 s
TOUCH_RADIUS_M = 2.5
CONTROL_RADIUS_M = 3.0
CONTROL_REL_SPEED_MS = 3.0
CONTROL_WINDOW = 15
CONTROL_MIN = 10
CONTROL_LOOKAHEAD = FPS  # the next CONTROL_WINDOW trusted frames must come within 1 s
EPISODE_JOIN = FPS
RECEPTION_LEAD = 10
FOLLOW_FRAMES = 3 * FPS
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

    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist) | ~trusted[:, None], np.inf, dist)  # [T, S]

    raw = []  # (frame, team) of each touch
    for frame in _peaks(_score(velocity, speed)):
        lo, hi = max(frame - LAG, 0), min(frame + LAG + 1, T)
        window = dist[lo:hi]
        if window.min() > TOUCH_RADIUS_M:
            continue
        slot = int(np.unravel_index(np.argmin(window), window.shape)[1])
        raw.append((int(frame), int(team[slot])))

    spells = _spells(_controller(velocity, players, team, dist))
    kinds: dict[tuple[int, int], str] = {}
    for start, side in spells:
        mine = [f for f, s in raw if s == side and start - RECEPTION_LEAD <= f <= start + LAG]
        kinds[(mine[0] if mine else start, side)] = "reception"
    spell_starts = np.array([s for s, _ in spells], dtype=int)
    for frame, side in raw:
        if (frame, side) in kinds:
            continue
        j = np.searchsorted(spell_starts, frame, side="right")
        if j < len(spells) and spells[j][0] - frame <= FOLLOW_FRAMES:
            kinds[(frame, side)] = "reception" if spells[j][1] == side else "deflection"
        else:
            kinds[(frame, side)] = _unclear(frame, trusted, speed)
    return [Touch(f, s, k) for (f, s), k in sorted(kinds.items())]


def _score(velocity: FloatArray, speed: FloatArray) -> FloatArray:
    return np.maximum(_change_score(velocity, speed, LAG), _change_score(velocity, speed, WIDE_LAG))


def _change_score(velocity: FloatArray, speed: FloatArray, lag: int = LAG) -> FloatArray:
    """Per frame, how strongly the ball's motion changes around it (1 = at a threshold)."""
    T = len(speed)
    score = np.zeros(T)
    if T <= 2 * lag:
        return score
    v0, v1 = velocity[: -2 * lag], velocity[2 * lag :]
    s0, s1 = speed[: -2 * lag], speed[2 * lag :]
    ok = np.isfinite(s0) & np.isfinite(s1)
    jump = np.where(ok, np.abs(s1 - s0), 0.0) / SPEED_JUMP_MS
    moving = ok & (np.maximum(s0, s1) >= TURN_MIN_SPEED_MS) & (s0 > 0) & (s1 > 0)
    cos = np.where(moving, np.sum(v0 * v1, axis=1) / np.where(moving, s0 * s1, 1.0), 1.0)
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) / TURN_DEG
    score[lag:-lag] = np.maximum(jump, turn)
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


def _controller(velocity: FloatArray, players: FloatArray, team: IntArray, dist: FloatArray) -> IntArray:
    """Per frame, the team whose nearest player has the ball at their feet, else -1."""
    T = len(dist)
    idx = np.arange(T)
    slot = np.argmin(dist, axis=1)
    near = dist[idx, slot]
    before = players[np.clip(idx - LAG, 0, T - 1), slot]
    after = players[np.clip(idx + LAG, 0, T - 1), slot]
    player_velocity = (after - before) * FPS / (2 * LAG)
    with np.errstate(invalid="ignore"):
        rel = np.linalg.norm(velocity - player_velocity, axis=1)
        ok = (near <= CONTROL_RADIUS_M) & (rel <= CONTROL_REL_SPEED_MS)  # NaN compares False
    return np.where(ok, team[slot], -1)


def _spells(controller: IntArray) -> list[tuple[int, int]]:
    """(start, team) of each spell in which one team holds the ball."""
    T = len(controller)
    holds = np.full(T, -1)
    best = np.zeros(T)
    for side in (0, 1):
        mine = controller == side
        total = np.concatenate([[0], np.cumsum(mine)])
        ahead = total[np.minimum(np.arange(T) + CONTROL_WINDOW, T)] - total[:T]
        on = mine & (ahead >= CONTROL_MIN) & (ahead > best)
        holds[on], best[on] = side, ahead[on]

    spells: list[tuple[int, int]] = []
    last_side, last_end = -1, -EPISODE_JOIN
    for start, end in zip(*_gaps(holds >= 0)):
        # a run of held frames may switch team inside; split it where it does
        cuts = np.flatnonzero(np.diff(holds[start:end])) + start + 1
        for a, b in zip(np.r_[start, cuts], np.r_[cuts, end]):
            side = int(holds[a])
            if side != last_side or a - last_end >= EPISODE_JOIN:
                spells.append((int(a), side))
            last_side, last_end = side, int(b)
    return spells


def _unclear(frame: int, trusted, speed) -> str:
    """Too little ball after the touch to see control: a slow ball means it was kept."""
    after = np.flatnonzero(trusted[frame + 1 : frame + 1 + CONTROL_LOOKAHEAD])
    if len(after) >= CONTROL_MIN:
        return "deflection"
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
    changes = 0
    last = -1
    for e in graded:
        if e.kind == "reception":
            changes += last >= 0 and e.team != last
            last = e.team
    print(f"changes of team between consecutive receptions: {changes} ({changes / minutes:.2f} per minute)")
