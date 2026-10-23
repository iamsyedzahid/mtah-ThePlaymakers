"""Dead-ball mask (R4)."""
from typing import Sequence

import numpy as np

from mtah.interfaces import BoolArray, Interval


def dead_mask(T: int, set_piece_intervals: Sequence[Interval]) -> BoolArray:
    mask = np.zeros(T, dtype=bool)
    for start, end in set_piece_intervals:
        if not 0 <= start <= end < T:
            raise ValueError(f"interval ({start}, {end}) is not inside [0, {T})")
        mask[start : end + 1] = True
    return mask
