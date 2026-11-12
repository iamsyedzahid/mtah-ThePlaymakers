"""The contract between modules.

Every pipeline stage is declared here once, with its types and its rules. The modules in mtah/
implement these signatures, and tests/test_pipeline.py checks that they still match. To change a
signature, change it here first, then in the module.

Conventions everywhere:
- Frames are 0-indexed: frame f is row f of every [T, ...] array and row f of the output CSV. The
  raw tracking's frame_id starts at 1; the loader subtracts 1.
- Positions are in metres on a 105.3 x 68 pitch with the origin at a corner: x in [0, 105.3]
  along the length, y in [0, 68] across. The raw tracking is 0..1 and is scaled by PITCH_M.
- NaN means missing. Never 0, never -1.
- Team 0 is home, team 1 is away. Evidence is positive for home.
- Labels are one-character strings: H (home), A (away), D (dead ball), X (abstain).
- Intervals are inclusive (start, end) frame pairs.
"""
from dataclasses import dataclass
from typing import Literal, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from mtah.config import FPS

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.integer]
LabelArray = npt.NDArray[np.str_]

H, A, D, X = "H", "A", "D", "X"
HOME, AWAY = 0, 1
LABEL_DTYPE = "<U1"

TouchKind = Literal["reception", "deflection"]
Interval = tuple[int, int]


@dataclass(frozen=True, eq=False)
class Tracking:
    """Positions for one match, or a synthetic stand-in.

    ball     [T, 2] float, metres. NaN where the ball is not tracked.
    players  [T, S, 2] float, metres. NaN where a slot is empty or not tracked. The real data has
             46 anonymous slots, so most slots are NaN on any given frame; never assume S == 22.
    team     [S] int, 0 = home and 1 = away for each slot. The loader (R1) owns this mapping:
             the first team in the metadata is home. Nothing downstream may assume slot order.
    fps      frames per second. The pipeline is built for 25.
    """

    ball: FloatArray
    players: FloatArray
    team: IntArray
    fps: int = FPS

    def __post_init__(self) -> None:
        ball, players, team = self.ball, self.players, self.team
        if ball.ndim != 2 or ball.shape[1] != 2:
            raise ValueError(f"ball must be [T, 2], got {ball.shape}")
        if players.ndim != 3 or players.shape[0] != ball.shape[0] or players.shape[2] != 2:
            raise ValueError(f"players must be [T={ball.shape[0]}, S, 2], got {players.shape}")
        if team.shape != (players.shape[1],):
            raise ValueError(f"team must be [S={players.shape[1]}], got {team.shape}")
        if not np.isin(team, (HOME, AWAY)).all():
            raise ValueError("team values must be 0 (home) or 1 (away)")

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])


class Touch(NamedTuple):
    """A ball contact. `frame` is the first frame of contact, `team` the nearest player's team."""

    frame: int
    team: int
    kind: TouchKind


class SetPiece(NamedTuple):
    """A Smart Tagging set piece (R3), as evidence for R4, never as a label.

    start..end (inclusive frames) is the tagged window around the restart, about 5 s, not a
    dead-ball interval. team is the Team tag mapped to 0 (home) / 1 (away), or None where the tag
    says N/A (every throw-in). kind is the Type tag as written, e.g. "Throw-in".
    """

    start: int
    end: int
    team: int | None
    kind: str


def ball_clean(ball: FloatArray) -> tuple[FloatArray, BoolArray]:
    """Clean the ball track so downstream geometry never sees NaN.

    Steps: drop samples implying a speed over about 40 m/s, smooth lightly, spline across gaps
    shorter than about 1 s. Returns (ball_filled [T, 2] float without NaN, lost_mask [T] bool).
    lost_mask is True where the position is not trustworthy: gaps of about 1 s or longer, and gaps
    at either end of the track. Those frames are still filled (linearly), so consumers must
    respect lost_mask. A missing ball is never by itself a dead ball.
    """
    raise NotImplementedError("contract only; see mtah/ball.py")


def gap_stats(lost_mask: BoolArray, fps: int = FPS) -> dict[str, float]:
    """Summarise ball gaps as numbers only: counts, fractions and lengths. No frame lists."""
    raise NotImplementedError("contract only; see mtah/ball.py")


def evidence(tracking: Tracking) -> FloatArray:
    """Per-frame control evidence, e [T] float in [-1, 1].

    e = (c_home - c_away) / (c_home + c_away + c0), from each team's soft control score and a fixed
    "nobody" score c0: when someone is close this is 2 * p_home - 1, and it fades to 0 for a loose
    ball. A fixed, bounded scale, so R4's costs mean the same everywhere. Positive favours home,
    negative favours away, 0 means no information. Always finite. Frames where ball_clean's
    lost_mask is True are exactly 0.
    """
    raise NotImplementedError("contract only; see mtah/evidence.py")


def touches(ball: FloatArray, players: FloatArray, team: IntArray) -> list[Touch]:
    """Ball contacts, sorted by frame.

    Takes the raw ball, NaN included: a frame without a ball position cannot hold a touch, and
    interpolated positions would invent touches. Every frame is in [0, T).

    A "reception" means control: the ball staying near one team's player and moving with them, not
    distance alone. It may have no sharp touch, as the provider smooths the ball. A touch followed
    by its own team's control is also a reception (an interception played first time). A single
    touch, a deflected pass that is not intercepted and a won header without control are
    "deflection".
    """
    raise NotImplementedError("contract only; see mtah/touches.py")


def dead_mask(T: int, set_piece_intervals: Sequence[Interval]) -> BoolArray:
    """bool [T], True on every frame inside any inclusive (start, end) interval.

    Intervals outside [0, T) raise. A silently clipped interval hides an off-by-one.

    D follows the referee, not tracking loss. Valid sources for intervals: the second-half block
    before kick-off (config.SECOND_HALF_D_BLOCK), a long still or absent ball, and set pieces. A
    Smart Tagging set piece (SetPiece) is a window AROUND the restart and partly live play: only
    the part before the ball is put back in play is D, and R4 finds that moment inside the window.
    Never pass a tag window here whole. A missing ball alone is never a reason.
    """
    raise NotImplementedError("contract only; see mtah/deadball.py")


def dead_intervals(set_pieces: Sequence[SetPiece], ball: FloatArray, lost_mask: BoolArray) -> list[Interval]:
    """Inclusive D intervals found from set-piece tag windows and ball_clean's output.

    One interval per tag: the part of the window before the ball is put back in play, reaching
    back to when the ball went quiet, as dead_mask's rules require. Tags outside [0, T) raise.
    """
    raise NotImplementedError("contract only; see mtah/deadball.py")


def camera_still_intervals(matrices: npt.NDArray[np.float64], players: FloatArray, lost_mask: BoolArray) -> list[Interval]:
    """Inclusive D intervals where the ball is missing, the camera is still and players are slowish.

    matrices [T, 3, 3] is the per-frame homography, normalised pitch to pixels, NaN where invalid.
    """
    raise NotImplementedError("contract only; see mtah/deadball.py")


def join_close(mask: BoolArray) -> BoolArray:
    """A dead mask with live stretches under about 3 s between two dead ones filled in."""
    raise NotImplementedError("contract only; see mtah/deadball.py")


def restart_touches(set_pieces: Sequence[SetPiece], intervals: Sequence[Interval], T: int) -> list[Touch]:
    """A reception at each restart whose tag names a team. intervals are dead_intervals' output."""
    raise NotImplementedError("contract only; see mtah/deadball.py")


def untagged_intervals(ball: FloatArray, lost_mask: BoolArray) -> list[Interval]:
    """Inclusive D intervals for stoppages no tag marks, from ball_clean's output alone.

    The tags have false negatives (DATA.md). Only a ball seen off the pitch, or resting for longer
    than a goalkeeper may hold it, counts. A missing ball alone never does.
    """
    raise NotImplementedError("contract only; see mtah/deadball.py")


def slow_lost_intervals(players: FloatArray, lost_mask: BoolArray) -> list[Interval]:
    """Inclusive D intervals where the ball is missing and the players have slowed down.

    The missing ball alone never counts: it needs sustained slow players around it.
    """
    raise NotImplementedError("contract only; see mtah/deadball.py")


def segment(
    e: FloatArray, lost_mask: BoolArray, dead_mask: BoolArray, touches: Sequence[Touch]
) -> LabelArray:
    """Turn evidence and touches into labels [T] in {H, A, D}. Never X.

    Rules, because spellF1 matches whole possessions by overlap, so a false flip costs a spell:
    - dead_mask frames are D, and only those. D splits possessions: H D H is two possessions;
    - while the ball is alive there is always a team. A loose ball stays with the last team;
    - possession flips only at a reception, or after a dead ball, and the evidence after a
      reception decides whether it is believed. Evidence alone and deflections never flip;
    - possession holds through lost frames. A change just after a lost stretch is placed inside it;
    - no flicker: very short H/A runs are absorbed by a neighbour.
    """
    raise NotImplementedError("contract only; see mtah/segment.py")
