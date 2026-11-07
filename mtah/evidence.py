"""Possession evidence (R2). Naive: the nearest player's team, weighted by how close they are."""
import numpy as np

from mtah.config import FPS
from mtah.interfaces import HOME, FloatArray, Tracking

SIGMA_M = 2.0  # evidence decays as exp(-distance / SIGMA_M)


def evidence(tracking: Tracking) -> FloatArray:
    if tracking.fps != FPS:
        raise NotImplementedError(f"only {FPS} fps is supported, got {tracking.fps}")
    dist = np.linalg.norm(tracking.players - tracking.ball[:, None, :], axis=2)  # [T, S]
    e = np.zeros(tracking.n_frames)
    rows = np.flatnonzero(~np.isnan(dist).all(axis=1))  # ball and at least one player seen
    if rows.size:
        nearest = np.nanargmin(dist[rows], axis=1)
        sign = np.where(tracking.team[nearest] == HOME, 1.0, -1.0)
        e[rows] = sign * np.exp(-dist[rows, nearest] / SIGMA_M)
    return e
