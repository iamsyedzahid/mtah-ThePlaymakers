"""Dead-ball mask (R4)."""
from typing import Sequence
import numpy as np

from mtah.interfaces import BoolArray, Interval
from mtah.config import SECOND_HALF_D_BLOCK

DEAD_FRACTION = 0.9  # share of each Smart Tagging window, from its start, treated as dead

def dead_mask(T: int, set_piece_intervals: Sequence[Interval]) -> BoolArray:
    """
    Computes the dead-ball mask for the match based on tagged set piece windows.
    """
    mask = np.zeros(T, dtype=bool)

    # 1. Hardcoded Second Half Block
    # The contract mentions valid sources include SECOND_HALF_D_BLOCK
    if SECOND_HALF_D_BLOCK is not None:
        start_block, end_block = SECOND_HALF_D_BLOCK
        if 0 <= start_block <= end_block < T:
            mask[start_block : end_block + 1] = True

    # 2. Tagged Set Pieces
    for start, end in set_piece_intervals:
        if not 0 <= start <= end < T:
            raise ValueError(f"interval ({start}, {end}) is not inside [0, {T})")
        
        # A Smart Tagging window is ~5s around the restart. 
        # Without ball velocity to find the exact kick, we approximate the restart 
        # as the midpoint of the window. The first half is dead, the second is live.
        dead_end = start + int((end - start + 1) * DEAD_FRACTION)
        mask[start:dead_end] = True

    return mask