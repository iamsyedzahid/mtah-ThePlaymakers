"""Possession segmentation (R4)."""
from typing import Sequence
import numpy as np

from mtah.config import FPS
from mtah.interfaces import A, AWAY, D, H, HOME, LABEL_DTYPE, BoolArray, FloatArray, LabelArray, Touch

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
    Uses a 3-state Viterbi decoder.
    """
    e = np.asarray(e, dtype=float)
    lost = np.asarray(lost_mask, dtype=bool)
    dead = np.asarray(dead_mask, dtype=bool)
    T = e.shape[0]
    
    if e.ndim != 1 or lost.shape != (T,) or dead.shape != (T,):
        raise ValueError("e, lost_mask and dead_mask must all be [T]")
    if any(not 0 <= t.frame < T for t in touches):
        raise ValueError(f"touch frame outside [0, {T})")

    # Extract valid touches (receptions) that allow possession to flip
    receptions = [t for t in touches if t.kind == "reception"]
    is_touch = np.zeros(T, dtype=bool)
    for r in receptions:
        is_touch[r.frame] = True

    # Viterbi Initialization
    # States: 0=Home, 1=Away, 2=Dead
    V = np.full((T, 3), -np.inf)
    B = np.zeros((T, 3), dtype=int)
    
    # State Observation Costs at t=0
    # If dead_mask is True at t=0, enforce D. Otherwise, guess based on first evidence.
    if dead[0]:
        V[0, 2] = 0.0
    else:
        V[0, 0] = e[0] if not lost[0] else 0.0
        V[0, 1] = -e[0] if not lost[0] else 0.0

    # Switching Penalties
    SWITCH_PENALTY = -100.0  # Massive penalty for changing possession without a touch
    
    # Viterbi Forward Pass
    for t in range(1, T):
        # Observation Scores for this frame
        obs = np.zeros(3)
        if dead[t]:
            obs[0] = -np.inf
            obs[1] = -np.inf
            obs[2] = 0.0
        else:
            # If lost, observation is 0, so possession naturally carries forward
            obs[0] = e[t] if not lost[t] else 0.0
            obs[1] = -e[t] if not lost[t] else 0.0
            obs[2] = -np.inf

        # Calculate transitions from t-1 to t
        for curr_state in range(3):
            if obs[curr_state] == -np.inf:
                continue
                
            best_val = -np.inf
            best_prev = 0
            
            for prev_state in range(3):
                # Transition cost
                cost = 0.0
                
                # Rule: Cannot be Dead if dead_mask is False, and MUST be Dead if dead_mask is True
                if prev_state != 2 and curr_state == 2 and not dead[t]:
                    cost = -np.inf
                
                # Possession flip (H->A or A->H)
                if (prev_state == 0 and curr_state == 1) or (prev_state == 1 and curr_state == 0):
                    # Flipping is severely penalized unless there is a controlled reception
                    if not is_touch[t]:
                        cost = SWITCH_PENALTY
                        
                val = V[t-1, prev_state] + cost + obs[curr_state]
                if val > best_val:
                    best_val = val
                    best_prev = prev_state
                    
            V[t, curr_state] = best_val
            B[t, curr_state] = best_prev

    # Viterbi Backtrack (Decoding)
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(V[-1, :])
    
    for t in range(T - 1, 0, -1):
        states[t - 1] = B[t, states[t]]

    # Map state integers back to labels
    labels = np.full(T, "H", dtype=LABEL_DTYPE)
    labels[states == 1] = A
    labels[states == 2] = D
    
    return labels