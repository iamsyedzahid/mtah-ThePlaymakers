"""Possession segmentation (R4)."""
from typing import Sequence
import numpy as np

from mtah.config import FPS
from mtah.interfaces import A, AWAY, D, H, HOME, LABEL_DTYPE, BoolArray, FloatArray, LabelArray, Touch

MIN_RUN_FRAMES = FPS  # shorter H/A runs are flicker

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

def segment(
    e: FloatArray, lost_mask: BoolArray, dead_mask: BoolArray, touches: Sequence[Touch]
) -> LabelArray:
    """
    Turn evidence and touches into labels [T] in {H, A, D}. Never X.
    Uses a 3-state Viterbi decoder with strict reception-only switching.
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

    # Only receptions allow a possession flip
    receptions = [t for t in touches if t.kind == "reception"]
    is_reception = np.zeros(T, dtype=bool)
    for r in receptions:
        is_reception[r.frame] = True

    # States: 0=H, 1=A, 2=D
    V = np.full((T, 3), -np.inf)
    B = np.zeros((T, 3), dtype=int)

    # Initialize based on evidence, but ensure valid starting numbers
    if dead[0]:
        V[0, 2] = 0.0
    else:
        V[0, 0] = e[0] if not lost[0] else 0.001 
        V[0, 1] = -e[0] if not lost[0] else -0.001

    for t in range(1, T):
        obs = np.zeros(3)
        if dead[t]:
            # Force D state
            obs[0], obs[1], obs[2] = -np.inf, -np.inf, 10.0 
        else:
            # Evidence dictates observation
            obs[0] = e[t] if not lost[t] else 0.0
            obs[1] = -e[t] if not lost[t] else 0.0
            obs[2] = -np.inf

        for curr_state in range(3):
            if obs[curr_state] == -np.inf:
                continue
                
            best_val = -np.inf
            best_prev = 0
            
            for prev_state in range(3):
                cost = 0.0
                
                # Rule: D only happens if dead_mask is True
                if prev_state != 2 and curr_state == 2 and not dead[t]:
                    cost = -np.inf
                
                # Rule: Possession flips ONLY on a reception
                if (prev_state == 0 and curr_state == 1) or (prev_state == 1 and curr_state == 0):
                    if not is_reception[t]:
                        # Absolutely forbidden to switch without a touch
                        cost = -np.inf  
                        
                val = V[t-1, prev_state] + cost + obs[curr_state]
                if val > best_val:
                    best_val = val
                    best_prev = prev_state
                    
            V[t, curr_state] = best_val
            B[t, curr_state] = best_prev

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(V[-1, :])
    
    for t in range(T - 1, 0, -1):
        states[t - 1] = B[t, states[t]]

    labels = np.full(T, H, dtype=LABEL_DTYPE)
    labels[states == 1] = A
    labels[states == 2] = D
    
    return labels