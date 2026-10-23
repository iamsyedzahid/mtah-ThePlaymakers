"""Touch detection (R3). Naive: the ball within reach of one player; long contact is a reception."""
import numpy as np

from mtah.interfaces import FloatArray, IntArray, Touch

TOUCH_RADIUS_M = 1.5
RECEPTION_MIN_FRAMES = 5  # 0.2 s with the same player counts as control


def touches(ball: FloatArray, players: FloatArray, team: IntArray) -> list[Touch]:
    ball = np.asarray(ball, dtype=float)
    players = np.asarray(players, dtype=float)
    team = np.asarray(team)
    T = ball.shape[0]
    if T == 0:
        return []

    # owner[f] is the player slot within reach of the ball at frame f, or -1.
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    owner = np.full(T, -1)
    rows = np.flatnonzero(~np.isnan(dist).all(axis=1))
    if rows.size:
        nearest = np.nanargmin(dist[rows], axis=1)
        close = dist[rows, nearest] <= TOUCH_RADIUS_M
        owner[rows[close]] = nearest[close]

    change = np.flatnonzero(owner[1:] != owner[:-1]) + 1
    events = []
    for start, end in zip(np.r_[0, change], np.r_[change, T]):
        if owner[start] < 0:
            continue
        kind = "reception" if end - start >= RECEPTION_MIN_FRAMES else "deflection"
        events.append(Touch(int(start), int(team[owner[start]]), kind))
    return events
