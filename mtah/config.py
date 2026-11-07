"""Project-wide constants. Nothing in this module reads data.

Match facts come from games.py (official, never edited) so the two cannot drift apart.
"""
import os
from pathlib import Path

import games

GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]  # 25
PITCH_M = games.GAMES[GAME]["pitch"]  # (105.3, 68.0) metres, measured from corrected tracking
N_FRAMES = games.GAMES[GAME]["n_video_frames"]  # 199,319; frame ids run 0 .. N_FRAMES - 1

# Graded halves as inclusive (start, end) video frames: (11185, 90651), (110808, 190197).
# Together they are 158,857 frames.
GRADED_HALVES = tuple((first, last) for _, first, last in games.half_video_blocks(GAME))

# The second half is graded from 110,808 but kicks off at 113,742 (FAQ.md), so this block is D.
SECOND_HALF_D_BLOCK = (GRADED_HALVES[1][0], 113_741)

# Where the real data bundle is unpacked. Same rule as games.py: $MTAH_DATA_DIR, else ./data.
DATA_DIR_ENV = "MTAH_DATA_DIR"


def data_dir() -> Path:
    """The data bundle root: $MTAH_DATA_DIR if set, else data/ next to games.py."""
    return Path(os.environ.get(DATA_DIR_ENV) or Path(games.__file__).resolve().parent / "data")
