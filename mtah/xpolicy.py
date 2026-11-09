"""Abstain policy (R4). X only inside a possession, never near a hand-over."""
import numpy as np

from mtah.config import FPS
from mtah.interfaces import A, H, LABEL_DTYPE, X, FloatArray, LabelArray
from mtah.segment import run_bounds


def to_x_policy(
    labels: LabelArray,
    confidence: FloatArray | None = None,
    threshold: float = 0.5,
    margin: int = FPS,
) -> LabelArray:
    out = np.asarray(labels).astype(LABEL_DTYPE)  # always a copy
    if confidence is None:
        return out
        
    confidence = np.asarray(confidence, dtype=float)
    
    # --- THE FIX: Smooth the confidence! ---
    # Raw evidence is too noisy. A 2-second moving average prevents 
    # momentary dips from splintering possessions into X.
    window_size = 2 * FPS
    kernel = np.ones(window_size) / window_size
    smooth_conf = np.convolve(confidence, kernel, mode='same')
    
    for start, end in zip(*run_bounds(out)):
        if out[start] not in (H, A):
            continue
        lo, hi = start + margin, end - margin
        if lo < hi:
            inner = out[lo:hi] 
            # Use the smoothed confidence instead of raw!
            inner[smooth_conf[lo:hi] < threshold] = X
            
    return out