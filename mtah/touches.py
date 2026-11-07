"""Touch detection and control spell extraction (R3).

Detects ball contacts and distinguishes controlled receptions from deflections.
Input is the raw ball track with NaNs. Ball positions are never smoothed
or interpolated before touch detection to avoid creating synthetic contacts.
"""
from typing import Sequence
import numpy as np

from mtah.config import FPS
from mtah.interfaces import FloatArray, IntArray, Touch

# Thresholds set from physics and dataset characteristics:
# Tracking slots reflect body centres (not feet), and provider smoothing spreads kicks.
# A carrier typically has the ball within 0.3 - 2.5 m (see synthetic.DRIBBLE_RANGE_M).
CONTROL_RADIUS_M = 2.0
# 5 frames at 25 fps = 0.20 s. Sustained proximity to the same player signifies control.
RECEPTION_MIN_FRAMES = 5
# Micro-gap bridging: brief tracking loss of 1-2 frames while in control does not break the spell
MICRO_GAP_MAX_FRAMES = 2


def touches(ball: FloatArray, players: FloatArray, team: IntArray) -> list[Touch]:
    """Ball contacts, sorted by frame.

    Takes raw ball coordinates, with NaNs on missing frames.
    Missing values are NOT interpolated so that no touches are hallucinated across tracking gaps.
    Brief micro-gaps (<= 2 frames) where the same player maintains proximity are bridged so that
    continuous ball carries are not artificially fragmented into deflections.

    Args:
        ball: [T, 2] ball position in metres (NaN when lost/untracked).
        players: [T, S, 2] player positions in metres (NaN for untracked slots).
        team: [S] team indicator (0=home, 1=away) for each of the S slots.

    Returns:
        list of Touch(frame, team, kind) sorted ascending by frame index in [0, T).
    """
    ball = np.asarray(ball, dtype=float)
    players = np.asarray(players, dtype=float)
    team = np.asarray(team)

    T = ball.shape[0]
    if T == 0 or players.shape[0] != T or players.shape[1] == 0:
        return []

    # Valid frames are frames where ball coordinates are not NaN
    seen = ~np.isnan(ball).any(axis=1)
    valid_frames = np.flatnonzero(seen)
    if valid_frames.size == 0:
        return []

    # For each valid ball frame, find the nearest player slot
    ball_v = ball[valid_frames, None, :]             # [V, 1, 2]
    players_v = players[valid_frames, :, :]          # [V, S, 2]
    dist_v = np.linalg.norm(players_v - ball_v, axis=2) # [V, S]

    # Slot owner array across all T frames: -1 means no owner or ball missing
    owner = np.full(T, -1, dtype=int)

    # Ignore slots where player position is NaN
    has_tracked_player = ~np.isnan(dist_v).all(axis=1)
    if has_tracked_player.any():
        sub_frames = valid_frames[has_tracked_player]
        sub_dist = dist_v[has_tracked_player]
        nearest_slots = np.nanargmin(sub_dist, axis=1)
        nearest_dists = sub_dist[np.arange(len(sub_dist)), nearest_slots]

        close_mask = nearest_dists <= CONTROL_RADIUS_M
        owner[sub_frames[close_mask]] = nearest_slots[close_mask]

    # Bridge brief micro-gaps (<= 2 frames) where tracking blinked
    # If frame f-1 has slot S and frame f+k has slot S (where both belong to the same team and k <= 3)
    # and intermediate frames are -1, bridge them to preserve continuous control.
    if MICRO_GAP_MAX_FRAMES > 0 and T > 3:
        for gap_len in range(1, MICRO_GAP_MAX_FRAMES + 1):
            if T <= gap_len + 2:
                continue
            prev_slots = owner[:-gap_len - 1]
            next_slots = owner[gap_len + 1:]
            active_spans = (prev_slots >= 0) & (prev_slots == next_slots)
            if not active_spans.any():
                continue
            for idx in np.flatnonzero(active_spans):
                # Ensure intermediate frames had no other slot
                mid = owner[idx + 1 : idx + 1 + gap_len]
                if (mid == -1).all():
                    owner[idx + 1 : idx + 1 + gap_len] = prev_slots[idx]

    # Detect spells of consecutive contact with the same player slot
    change = np.flatnonzero(owner[1:] != owner[:-1]) + 1
    spell_starts = np.r_[0, change]
    spell_ends = np.r_[change, T]

    events: list[Touch] = []
    for start, end in zip(spell_starts, spell_ends):
        slot = owner[start]
        if slot < 0:
            continue
        duration = end - start
        kind = "reception" if duration >= RECEPTION_MIN_FRAMES else "deflection"
        events.append(Touch(frame=int(start), team=int(team[slot]), kind=kind))

    events.sort(key=lambda t: t.frame)
    return events
