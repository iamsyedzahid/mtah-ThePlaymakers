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
# Of each file with Windows line endings turned into Unix ones (digest), so a checkout on any system,
# with or without git's line-ending conversion, gives the same hashes.
SHA256 = {
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv": "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv": "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv": "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv": "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv": "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv": "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
    "Sample_Game_3/Sample_Game_3_events.json": "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0",
    "Sample_Game_3/Sample_Game_3_metadata.xml": "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1",
    "Sample_Game_3/Sample_Game_3_tracking.txt": "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658",
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


def digest(path: Path) -> str:
    """SHA-256 of a file with its line endings made Unix ones."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify(games_folder: Path) -> None:
    """Raise unless every file matches the pinned SHA-256."""
    for name, expected in SHA256.items():
        if digest(games_folder / name) != expected:
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
