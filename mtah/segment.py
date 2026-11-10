"""Possession segmentation (R4)."""
from typing import Sequence
import numpy as np

from mtah.config import FPS
from mtah.interfaces import A, AWAY, D, H, HOME, LABEL_DTYPE, BoolArray, FloatArray, LabelArray, Touch

MIN_RUN_FRAMES = 2 * FPS  # shorter H/A runs are flicker

# ---- tunables (in units of "typical per-frame evidence magnitude") ----
RECEPTION_TOL = 3          # frames either side of a reception where a flip is cheap
SWITCH_COST_AT_RECEPTION = 2.0
SWITCH_COST_ELSEWHERE = 2.0 * FPS   # ~ two seconds of contrary evidence to flip with no touch


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
                # pick the longer neighbour
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
    Uses a 3-state Viterbi decoder with soft, reception-aware switching.
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

    # --- normalise evidence scale ---
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    usable = ~dead & (e != 0)
    scale = np.median(np.abs(e[usable])) if usable.any() else 1.0
    if scale <= 0:
        scale = 1.0
    ev = e / scale   # lost frames now carry shape evidence, so don't zero them

    # --- reception proximity (soft window) ---
    near_reception = np.zeros(T, dtype=bool)
    for t in touches:
        if t.kind == "reception":
            lo = max(0, t.frame - RECEPTION_TOL)
            hi = min(T, t.frame + RECEPTION_TOL + 1)
            near_reception[lo:hi] = True

    # States: 0=H, 1=A, 2=D
    NEG = -1e18  # finite "forbidden" so we never get all -inf paths
    V = np.full((T, 3), NEG)
    B = np.zeros((T, 3), dtype=int)

    def obs(t: int) -> np.ndarray:
        if dead[t]:
            return np.array([NEG, NEG, 0.0])
        return np.array([ev[t], -ev[t], NEG])

    V[0] = obs(0)

    for t in range(1, T):
        o = obs(t)
        sw = SWITCH_COST_AT_RECEPTION if near_reception[t] else SWITCH_COST_ELSEWHERE

        # trans[prev, curr] = additive cost
        trans = np.zeros((3, 3))
        trans[0, 1] = trans[1, 0] = -sw       # H<->A flip
        # D -> H/A (restart) and H/A -> D (dead ball) are free;
        # D membership itself is enforced through obs()

        cand = V[t - 1][:, None] + trans + o[None, :]   # [prev, curr]
        B[t] = np.argmax(cand, axis=0)
        V[t] = cand[B[t], np.arange(3)]

    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmax(V[-1]))
    for t in range(T - 1, 0, -1):
        states[t - 1] = B[t, states[t]]

    labels = np.full(T, H, dtype=LABEL_DTYPE)
    labels[states == 1] = A
    labels[states == 2] = D

    return _merge_short_runs(labels)