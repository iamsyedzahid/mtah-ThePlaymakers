"""Abstain policy (R4). X only inside a possession, never near a hand-over.

Not used by run.predict: an X is a miss under macroF1 and gains nothing under spellF1.
"""
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
    if confidence.shape != out.shape:
        raise ValueError(f"confidence must be {out.shape}, got {confidence.shape}")
    if margin < 1:
        raise ValueError("margin must be at least 1 so X never sits on a run boundary")

    for start, end in zip(*run_bounds(out)):
        if out[start] not in (H, A):
            continue
        lo, hi = start + margin, end - margin
        if lo < hi:
            inner = out[lo:hi]  # a view, so assigning into it edits out
            inner[confidence[lo:hi] < threshold] = X
    return out
