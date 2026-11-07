"""Metrica's three open sample games: the training data of the learned layer in mtah/fusion.py.

Other matches, with real tracking and Metrica's own event data, which the rules allow (RULES.md, FAQ.md):

    git clone https://github.com/metrica-sports/sample-data outputs/metrica/sample-data
    git -C outputs/metrica/sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996

Point $MTAH_OPEN_GAMES (or run.py --open-games) at the folder holding Sample_Game_1..3, which is the
clone's data folder. Every file is checked against its SHA-256 below, so the model is trained on
exactly the same data everywhere and the submission comes out byte for byte the same.

Labels come from the events (`labels_from_events`): H or A is the team of the latest pass, carry,
recovery, shot or set piece; D runs from a ball out, a foul or a goal to the next of those; X lies
outside the periods. Approximate, as event timestamps are set by people and a pass in flight stays with
the passer.
"""
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np

from mtah.interfaces import A, D, H, LABEL_DTYPE, X, Tracking

PITCH = (105.0, 68.0)  # both games, from the sample-data README
ENV = "MTAH_OPEN_GAMES"
DEFAULT_FOLDERS = ("outputs/metrica/sample-data/data", "outputs/metrica/sample/games")
SHA256 = {
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv": "803f312a07572912b9ca2dc9482d31a75e573eae825773253b4ca9472e7760c0",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv": "380e1b23affe778a7e1e8480983291798f118f71b1dbcf8f9729c06977669101",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv": "ed7289a72d41f280b5316647655b1934575d3506c34e181c343294ceafdcb280",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv": "c2296e8e0caf6f5bb832d5ecf9ebb52c13b18815b1fdbbf79a886238a01d6472",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv": "f887c6c9227e56c01d1bf62d8d5bc66cdbc1da0c3e9b397c0e9499a0bc6dcb20",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv": "92cc3ae1b5ac74530e21a899ab71cf12ee3a58c16592e87add6197f88a042b31",
    "Sample_Game_3/Sample_Game_3_events.json": "ce12d8bf2b8242825e343fe12a06a0731f2d35e0ffb130603ef746f79fbf0b09",
    "Sample_Game_3/Sample_Game_3_metadata.xml": "0011881426043ff4d404b3445381eafb69086a02d95ac3bcf213a062fae74502",
    "Sample_Game_3/Sample_Game_3_tracking.txt": "bd33b0a6f89d5617275eac373b3fd043247c581759484e8f566f7701d09975c9",
}
ON_BALL = {"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE"}

# (start frame, end frame, type, subtype, team 0/1, period); frames 0-indexed
Event = tuple[int, int, str, str, int, int]


def folder(path: str | os.PathLike | None = None) -> Path:
    """The folder holding Sample_Game_1..3: the argument, else $MTAH_OPEN_GAMES, else a default."""
    root = Path(__file__).resolve().parents[1]
    candidates = [Path(path)] if path else [Path(os.environ[ENV])] if os.environ.get(ENV) else \
        [root / p for p in DEFAULT_FOLDERS]
    for candidate in candidates:
        if (candidate / "Sample_Game_1").is_dir():
            return candidate
    raise FileNotFoundError(f"no Sample_Game_1 under {', '.join(map(str, candidates))}; see mtah/opengames.py")


def verify(games_folder: Path) -> None:
    """Raise unless every file matches the pinned SHA-256."""
    for name, expected in SHA256.items():
        digest = hashlib.sha256((games_folder / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"{name} differs from the pinned open games (commit e706dd5)")


def _csv_team(path: Path):
    import pandas as pd

    raw = pd.read_csv(path, header=None, skiprows=3, low_memory=False).to_numpy(dtype=float)
    frames, xy = raw[:, 1].astype(int), raw[:, 3:]
    return frames, xy[:, :-2].reshape(len(xy), -1, 2), xy[:, -2:]


def load_csv_game(games_folder: Path, g: int) -> tuple[Tracking, list[Event]]:
    """Games 1 and 2: one CSV per team, coordinates 0..1, the ball in the last two columns."""
    import pandas as pd

    base = games_folder / f"Sample_Game_{g}"
    frames, home, ball = _csv_team(base / f"Sample_Game_{g}_RawTrackingData_Home_Team.csv")
    frames_away, away, _ = _csv_team(base / f"Sample_Game_{g}_RawTrackingData_Away_Team.csv")
    if not (np.array_equal(frames, frames_away) and np.array_equal(frames, np.arange(1, len(frames) + 1))):
        raise ValueError(f"game {g}: home and away frames do not line up from 1")
    scale = np.array(PITCH)
    players = np.concatenate([home, away], axis=1) * scale
    team = np.repeat([0, 1], [home.shape[1], away.shape[1]]).astype(np.int8)
    ev = pd.read_csv(base / f"Sample_Game_{g}_RawEventsData.csv")
    events = [(int(r["Start Frame"]) - 1, int(r["End Frame"]) - 1, str(r["Type"]), str(r["Subtype"]),
               0 if r["Team"] == "Home" else 1, int(r["Period"])) for _, r in ev.iterrows()]
    return Tracking(ball * scale, players, team), events


def load_epts_game3(games_folder: Path) -> tuple[Tracking, list[Event]]:
    """Game 3: the same EPTS format as the competition match, read with kloppy."""
    from kloppy import metrica

    from mtah.io_atd import tracking_from_frame
    from mtah.teams import slot_teams

    base = games_folder / "Sample_Game_3"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = metrica.load_tracking_epts(meta_data=str(base / "Sample_Game_3_metadata.xml"),
                                             raw_data=str(base / "Sample_Game_3_tracking.txt"))
    df = dataset.to_df()
    tracking = tracking_from_frame(df, slot_teams(dataset.metadata.teams), n_frames=len(df), pitch_m=PITCH)
    home_id = dataset.metadata.teams[0].team_id
    data = json.loads((base / "Sample_Game_3_events.json").read_text(encoding="utf-8"))["data"]
    events = [(int(e["start"]["frame"]) - 1, int(e["end"]["frame"]) - 1, e["type"]["name"],
               (e["subtypes"] or {}).get("name", "") if isinstance(e["subtypes"], dict) else "",
               0 if e["team"]["id"] == home_id else 1, int(e["period"])) for e in data]
    return tracking, events


def labels_from_events(events: list[Event], T: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Labels [T] (H/A/D/X) and the periods as [start, end) pairs."""
    marks = []
    for start, end, kind, sub, team, _ in sorted(events):
        side = H if team == 0 else A
        if kind in ("BALL OUT", "FAULT RECEIVED"):
            marks.append((start, D))
        elif kind == "SHOT" and "GOAL" in sub and "OUT" not in sub and "SAVED" not in sub:
            marks += [(start, side), (max(end, start + 1), D)]
        elif kind in ON_BALL:
            marks.append((start, side))
    marks.sort()
    labels = np.full(T, X, dtype=LABEL_DTYPE)
    for (f, state), nxt in zip(marks, marks[1:] + [(T, None)]):
        labels[max(f, 0) : min(nxt[0], T)] = state
    periods: dict[int, tuple[int, int]] = {}
    for start, _, _, _, _, period in events:
        lo, hi = periods.get(period, (start, start))
        periods[period] = (min(lo, start), max(hi, start))
    halves = sorted((max(lo, 0), min(hi + 1, T)) for lo, hi in periods.values())
    graded = np.zeros(T, dtype=bool)
    for lo, hi in halves:
        graded[lo:hi] = True
    labels[np.logical_not(graded)] = X
    return labels, halves


def load_game(games_folder: Path, g: int) -> dict:
    tracking, events = load_csv_game(games_folder, g) if g < 3 else load_epts_game3(games_folder)
    labels, halves = labels_from_events(events, tracking.n_frames)
    codes = np.select([labels == H, labels == A, labels == D], [0, 1, 2], 3).astype(np.int8)
    return dict(name=f"G{g}", tracking=tracking, events=events, labels=labels, halves=halves, codes=codes,
                ball=tracking.ball, players=tracking.players, team=tracking.team, pitch=PITCH)


def training_games(path: str | os.PathLike | None = None) -> list[dict]:
    """All three games, checked against the pinned files."""
    games_folder = folder(path)
    verify(games_folder)
    return [load_game(games_folder, g) for g in (1, 2, 3)]
