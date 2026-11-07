"""Metrica's open sample games as labelled data (R2). Other matches, never ours.

Three games from https://github.com/metrica-sports/sample-data at commit e706dd5, cloned into
outputs/metrica/sample (TEAM_PLAN.md, gitignored). Games 1-2 are Metrica's CSV format, one file
per team with 0..1 coordinates; game 3 is FIFA EPTS like our match and goes through R1's streamed
reader. The rules allow data about other games (RULES.md, FAQ.md); we use them to fit and to
choose, never our match's ground truth.

Labels come from Metrica's event data, the same way for all three games:
- H/A: the team of the latest PASS, CARRY, RECOVERY, SHOT, SET PIECE or BALL LOST to start at or
  before the frame (a BALL LOST says that team had the ball until then). Challenges, ball-outs and
  cards never change the team, so a loose ball stays with the last team.
- D: before every restart (a SET PIECE, or an event marked GOAL KICK) except a half's opening
  kick-off, from where the last live action before it ends (the latest event end frame, a foul's
  own frame) up to the restart. Before a kick-off that follows a goal, from the goal.
- X: before a half's first event and after its last.

The ball is almost never missing in these games, unlike ours (41% of graded frames). mask_ball
hides it the way R1 measured our match losing it: 40.8% of frames, in gaps with our quartiles
(6.1 / 10.7 / 21.1 s), many starting just before a player takes the ball.
And their corrected ball sits on its carrier, while ours is detected metres off the players:
displace_ball moves it off as R1 measured ours (masked_tracking(..., displace=True)).
"""
import json
import math
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from mtah.config import FPS
from mtah.interfaces import (A, D, H, HOME, AWAY, LABEL_DTYPE, X, BoolArray, FloatArray, Interval, IntArray,
                             LabelArray, Tracking)
from mtah.synthetic import CROWD_GAP_PROB, DRIBBLE_RANGE_M

REPO = Path(__file__).resolve().parents[1]
METRICA_DIR = REPO / "outputs" / "metrica" / "sample" / "data"
CACHE_DIR = REPO / "outputs" / "cache"
METRICA_PITCH_M = (105.0, 68.0)  # README: 105 x 68 m for games 1-2; game 3's metadata agrees
GAMES = (1, 2, 3)

DISPLACE_STREAM = 7  # displace_ball's random stream, apart from mask_ball's for the same seed
# A new ball offset every 3 s, blended linearly in between. Faster, the offset itself moves the
# ball: with 1 s knots a ball at rest drifts at about 2 m/s and the slow-ball share (Savitzky-Golay
# speed under 2 m/s) falls from 0.17-0.22 to 0.12-0.15; with 3 s it stays at 0.16-0.22.
DISPLACE_KNOT_FRAMES = 3 * FPS

# Our match's ball gaps, R1's measurement over the graded halves: 150 gaps; p25 6.1 s, p50 10.7 s,
# p75 21.1 s (p90 33.8 s, longest 221.5 s); 40.8% of graded frames lost after cleaning. The
# lognormal through the quartiles gives p90 34.8 s and a mean of 16.3 s.
GAP_QUARTILES_S = (6.1, 10.7, 21.1)
LOST_FRAC = 0.408
GAP_LOG_MU = math.log(GAP_QUARTILES_S[1] * FPS)  # in frames
GAP_LOG_SIGMA = (math.log(GAP_QUARTILES_S[2]) - math.log(GAP_QUARTILES_S[0])) / (2 * 0.6744897501960817)
GAP_MEAN_FRAMES = math.exp(GAP_LOG_MU + GAP_LOG_SIGMA**2 / 2)

POSSESSION_EVENTS = frozenset({"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST"})
NOT_LIVE = frozenset({"CARD"})  # booked while the ball is dead, so its frames say nothing about play


class Event(NamedTuple):
    team: int  # 0 home, 1 away
    type: str
    sub: frozenset[str]  # subtype words, e.g. {"ON TARGET", "GOAL"}
    start: int  # 0-indexed frames, inclusive
    end: int
    period: int


class OpenGame(NamedTuple):
    number: int | str  # Metrica's game number, or a DFL match id (mtah/dfl_games.py)
    tracking: Tracking  # the ball as Metrica gives it, before any masking
    halves: tuple[Interval, ...]  # the periods, 0-indexed inclusive
    labels: LabelArray  # H/A/D/X from the events
    contacts: IntArray  # frames where a player takes the ball: pass receptions and recoveries
    pitch: tuple[float, float] = METRICA_PITCH_M  # length, width (m)


def load_game(number: int, root: Path = METRICA_DIR, cache_dir: Path | None = CACHE_DIR) -> OpenGame:
    """One open game with event labels. The tracking is cached in cache_dir (gitignored, plain
    arrays loaded without pickle); delete outputs/cache/metrica_game*.npz after changing a reader."""
    if number not in GAMES:
        raise ValueError(f"Metrica sample games are {GAMES}, got {number}")
    tracking, halves = _cached_tracking(number, Path(root), cache_dir)
    events = read_events(number, Path(root))
    return OpenGame(number, tracking, halves, labels_from_events(events, tracking.n_frames), contact_frames(events))


# -- labels ---------------------------------------------------------------------------------------

def labels_from_events(events: Sequence[Event], n_frames: int) -> LabelArray:
    """H/A/D/X per frame from the events, as described in the module docstring."""
    labels = np.full(n_frames, X, dtype=LABEL_DTYPE)
    for period in sorted({e.period for e in events}):
        evs = sorted((e for e in events if e.period == period), key=lambda e: e.start)
        poss = [e for e in evs if e.type in POSSESSION_EVENTS]
        if not poss:
            continue
        first, last = poss[0].start, min(max(e.end for e in evs), n_frames - 1)
        starts = np.array([e.start for e in poss])
        teams = np.array([e.team for e in poss])
        latest = np.searchsorted(starts, np.arange(first, last + 1), side="right") - 1
        labels[first : last + 1] = np.where(teams[latest] == HOME, H, A)
        opening = poss[0]
        for r in evs:
            if r is opening or not _is_restart(r):
                continue
            dead_from = _dead_from(evs, r)
            if first <= dead_from < r.start:
                labels[dead_from : r.start] = D
    return labels


def _is_restart(e: Event) -> bool:
    return e.type == "SET PIECE" or (e.type in POSSESSION_EVENTS and "GOAL KICK" in e.sub)


def _dead_from(evs: Sequence[Event], restart: Event) -> int:
    """First dead frame before `restart`: where the last live action ends, or the goal."""
    before = [e for e in evs if e.start < restart.start and e.type not in NOT_LIVE]
    if not before:
        return restart.start
    if "KICK OFF" in restart.sub:
        kickoffs = [e.start for e in before if e.type == "SET PIECE" and "KICK OFF" in e.sub]
        since = kickoffs[-1] if kickoffs else -1
        goals = [e.end for e in before if e.type == "SHOT" and "GOAL" in e.sub and e.start > since]
        if goals:
            return goals[-1]
    return max(e.start if e.type == "FAULT RECEIVED" else e.end for e in before)


def contact_frames(events: Sequence[Event]) -> IntArray:
    """Sorted frames where a player takes the ball: the end of each pass and each recovery."""
    frames = [e.end for e in events if e.type == "PASS"] + [e.start for e in events if e.type == "RECOVERY"]
    return np.unique(np.array(frames, dtype=np.int64))


# -- the ball as our match loses it ------------------------------------------------------------

def mask_ball(n_frames: int, contacts: IntArray, seed: int) -> BoolArray:
    """[T] True where the ball is hidden, in gaps shaped like our match's (GAP_QUARTILES_S): a
    lognormal through R1's measured quartiles, LOST_FRAC of frames in all. Like R1's synthetic
    match, gaps are often moved to start just before a contact (CROWD_GAP_PROB), where play is
    crowded and the real ball is lost most."""
    rng = np.random.default_rng(seed)
    mean_seen = GAP_MEAN_FRAMES * (1 - LOST_FRAC) / LOST_FRAC
    contacts = np.asarray(contacts, dtype=np.int64)
    hidden = np.zeros(n_frames, dtype=bool)
    f = int(rng.exponential(mean_seen))
    while f < n_frames:
        if contacts.size and rng.random() < CROWD_GAP_PROB:
            j = np.searchsorted(contacts, f)
            if j < contacts.size and contacts[j] - f < 5 * FPS:
                f = max(f, int(contacts[j]) - int(rng.integers(0, FPS)))
        length = max(1, int(np.ceil(rng.lognormal(GAP_LOG_MU, GAP_LOG_SIGMA))))
        hidden[f : f + length] = True
        f += length + max(1, int(np.ceil(rng.exponential(mean_seen))))
    return hidden


def masked_tracking(game: OpenGame, seed: int, displace: bool = False) -> Tracking:
    """The game's tracking with the ball hidden as mask_ball(seed) says, and with displace also
    moved off the players as displace_ball(seed) says."""
    ball = game.tracking.ball.copy()
    if displace:
        ball += displace_ball(game.tracking.n_frames, seed)
    ball[mask_ball(game.tracking.n_frames, game.contacts, seed)] = np.nan
    return Tracking(ball, game.tracking.players, game.tracking.team, game.tracking.fps)


def displace_ball(n_frames: int, seed: int) -> FloatArray:
    """[T, 2] metres to add to Metrica's ball so it sits off the players as ours does.

    Metrica's corrected ball sits on its carrier: with a slow ball the nearest player is a median
    0.07-0.11 m away, 82-95% within 1 m. Ours is detected apart from the players: R1 measured a
    median 1.2 m and 44% within 1 m, and modelled it as the ball running DRIBBLE_RANGE_M from its
    carrier's tracked position (mtah/synthetic.py). Here every DISPLACE_KNOT_FRAMES gets an offset
    of that size in a random direction, blended linearly in between; with R1's definition (slow =
    Savitzky-Golay speed under 2 m/s) the open games then give 1.10-1.18 m and 39-45%. No per-frame
    jitter: it would add flicker that neither data set has. A separate random stream from
    mask_ball's for the same seed.
    """
    rng = np.random.default_rng([seed, DISPLACE_STREAM])
    knots = np.arange(0, n_frames + DISPLACE_KNOT_FRAMES, DISPLACE_KNOT_FRAMES)
    radius = rng.uniform(*DRIBBLE_RANGE_M, size=knots.size)
    angle = rng.uniform(0.0, 2 * np.pi, size=knots.size)
    frames = np.arange(n_frames)
    return np.column_stack([np.interp(frames, knots, radius * np.cos(angle)),
                            np.interp(frames, knots, radius * np.sin(angle))])


# -- reading ----------------------------------------------------------------------------------

def read_events(number: int, root: Path = METRICA_DIR) -> list[Event]:
    if number in (1, 2):
        return read_csv_events(root / f"Sample_Game_{number}" / f"Sample_Game_{number}_RawEventsData.csv")
    return read_json_events(root / "Sample_Game_3" / "Sample_Game_3_events.json",
                            home_id=_epts_home_id(root / "Sample_Game_3" / "Sample_Game_3_metadata.xml"))


def read_csv_events(path: Path) -> list[Event]:
    """Games 1-2: Team is Home/Away, Subtype joins words with '-', frames are 1-based."""
    import pandas as pd

    ev = pd.read_csv(path)
    out = []
    for team, kind, sub, period, start, end in ev[["Team", "Type", "Subtype", "Period", "Start Frame", "End Frame"]].itertuples(index=False):
        words = frozenset(str(sub).split("-")) if isinstance(sub, str) else frozenset()
        start0 = int(start) - 1
        out.append(Event(HOME if team == "Home" else AWAY, str(kind), words, start0, max(int(end) - 1, start0), int(period)))
    return out


def read_json_events(path: Path, home_id: str) -> list[Event]:
    """Game 3: Metrica's JSON; team ids match the EPTS metadata, frames are 1-based."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)["data"]
    out = []
    for e in data:
        subs = e.get("subtypes")
        subs = subs if isinstance(subs, list) else ([subs] if subs else [])
        start0 = int(e["start"]["frame"]) - 1
        end = e["end"]["frame"] if e.get("end") and e["end"].get("frame") is not None else e["start"]["frame"]
        out.append(Event(HOME if e["team"]["id"] == home_id else AWAY, e["type"]["name"],
                         frozenset(s["name"] for s in subs), start0, max(int(end) - 1, start0), int(e["period"])))
    return out


def read_csv_tracking(home_path: Path, away_path: Path, pitch_m: tuple = METRICA_PITCH_M) -> tuple[Tracking, tuple[Interval, ...]]:
    """Games 1-2: three header rows, then Period, Frame, Time and an x, y pair per player, the
    ball last. Frames must be 1..T in order in both files."""
    import pandas as pd

    parts = []
    for path in (home_path, away_path):
        df = pd.read_csv(path, skiprows=2)
        frames = df["Frame"].to_numpy()
        if not np.array_equal(frames, np.arange(1, len(df) + 1)):
            raise ValueError(f"{Path(path).name}: frames are not 1..{len(df)} in order")
        xy = df.iloc[:, 3:].to_numpy(dtype=float)
        if xy.shape[1] % 2:
            raise ValueError(f"{Path(path).name}: odd number of coordinate columns")
        parts.append((df["Period"].to_numpy(dtype=int), xy.reshape(len(df), -1, 2)))
    (period, home), (period_away, away) = parts
    if not np.array_equal(period, period_away):
        raise ValueError("home and away files disagree on frames or periods")
    scale = np.asarray(pitch_m, dtype=float)
    players = np.concatenate([home[:, :-1], away[:, :-1]], axis=1) * scale  # the ball is the last pair
    team = np.r_[np.full(home.shape[1] - 1, HOME), np.full(away.shape[1] - 1, AWAY)].astype(np.int8)
    return Tracking(home[:, -1] * scale, players, team, FPS), _runs(period)


def read_epts_tracking(meta_path: Path, raw_path: Path, pitch_m: tuple = METRICA_PITCH_M) -> tuple[Tracking, tuple[Interval, ...]]:
    """Game 3: the same EPTS format as our match, through R1's streamed reader."""
    from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata
    from kloppy.infra.serializers.tracking.metrica_epts.reader import read_raw_data

    from mtah.teams import slot_teams

    with open(meta_path, "rb") as handle:
        metadata = load_metadata(handle)
    with open(raw_path, "rb") as handle:
        n_frames = sum(1 for _ in handle)
    with open(raw_path, "rb") as handle:
        tracking = _rows_to_tracking(read_raw_data(handle, metadata, sample_rate=1.0, limit=0),
                                     slot_teams(metadata.teams), n_frames, pitch_m)
    halves = tuple(
        (round(p.start_timestamp.total_seconds() * FPS) - 1, min(round(p.end_timestamp.total_seconds() * FPS) - 1, n_frames - 1))
        for p in metadata.periods
    )
    return tracking, halves


def _rows_to_tracking(rows: Iterable[dict], slot_team: dict[str, int], n_frames: int, pitch_m: tuple) -> Tracking:
    """kloppy's raw EPTS rows to a Tracking, like R1's io_atd.tracking_from_rows but for a game
    with substitutions: a player's channel exists only while they are on the pitch, so a slot
    missing from a row is NaN there. Every frame_id 1..n_frames must still appear exactly once."""
    from mtah.io_atd import _sorted_slots

    slots = _sorted_slots(slot_team)
    keys = [(f"player_{s}_x", f"player_{s}_y") for s in slots]
    nan = float("nan")
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, len(slots), 2), np.nan)
    seen = np.zeros(n_frames, dtype=bool)
    for row in rows:
        f = int(row["frame_id"]) - 1
        if not 0 <= f < n_frames or seen[f]:
            raise ValueError(f"frame_id {f + 1} is outside 1..{n_frames} or repeated")
        seen[f] = True
        ball[f] = (row.get("ball_x", nan), row.get("ball_y", nan))
        players[f] = [(row.get(kx, nan), row.get(ky, nan)) for kx, ky in keys]
    if not seen.all():
        raise ValueError(f"{int((~seen).sum())} frame_ids of 1..{n_frames} never appear")
    scale = np.asarray(pitch_m, dtype=float)
    team = np.array([slot_team[s] for s in slots], dtype=np.int8)
    return Tracking(ball * scale, players * scale, team, FPS)


def _epts_home_id(meta_path: Path) -> str:
    from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata

    with open(meta_path, "rb") as handle:
        return str(load_metadata(handle).teams[0].team_id)  # the first team is home, as in R1's loader


def _cached_tracking(number: int, root: Path, cache_dir: Path | None) -> tuple[Tracking, tuple[Interval, ...]]:
    cache = None if cache_dir is None else Path(cache_dir) / f"metrica_game{number}.npz"
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:  # plain arrays only: a cache cannot run code
            halves = tuple((int(a), int(b)) for a, b in saved["halves"])
            return Tracking(saved["ball"], saved["players"], saved["team"], FPS), halves
    game = root / f"Sample_Game_{number}"
    if number in (1, 2):
        tracking, halves = read_csv_tracking(game / f"Sample_Game_{number}_RawTrackingData_Home_Team.csv",
                                             game / f"Sample_Game_{number}_RawTrackingData_Away_Team.csv")
    else:
        tracking, halves = read_epts_tracking(game / "Sample_Game_3_metadata.xml", game / "Sample_Game_3_tracking.txt")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ball=tracking.ball, players=tracking.players, team=tracking.team,
                            halves=np.array(halves, dtype=np.int64))
    return tracking, halves


def _runs(values: np.ndarray) -> tuple[Interval, ...]:
    """Inclusive (first, last) of each run of equal values."""
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(values)] - 1
    return tuple((int(a), int(b)) for a, b in zip(starts, ends))
