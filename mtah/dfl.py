"""The DFL/IDSSE open data: seven Bundesliga 2022/23 matches with optical tracking, CC-BY 4.0.

Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific
Data (2025). The original is on figshare; we use the pysport/idsse-data mirror on Hugging Face at a
pinned revision, file by file (the events files are not needed):

    https://huggingface.co/datasets/pysport/idsse-data/resolve/a715a38dfbaf5f58e431727c2b78d174101a703c/<file>

for each <file> in SHA256 below, saved into outputs/idsse/. Every file is checked against its SHA-256,
so the model is trained on exactly the same data everywhere. Point $MTAH_DFL_GAMES (or run.py
--dfl-games) at another folder if needed.

Its labels are what makes it worth having: every ball frame carries BallStatus (1 alive, 0 dead) and
BallPossession (1 home, 2 away) from the provider, which is how the competition's ground truth is built
(CHALLENGE.md: D where the tracking marks the ball dead, else the team it names as owning it).

The positions files are about 400 MB of XML each, so they are streamed, and the arrays are cached in
outputs/cache/ after the first read.
"""
import os
from pathlib import Path

import numpy as np

from mtah.interfaces import A, D, H, LABEL_DTYPE, X, Tracking

ENV = "MTAH_DFL_GAMES"
DEFAULT_FOLDER = "outputs/idsse"
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SECTIONS = ("firstHalf", "secondHalf")  # no extra time in these matches
REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
SHA256 = {
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml": "a884728647a564cf8b20e5223a1d04a5da6f6725e1e506c8651505327c197ba9",
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml": "5490286436c86e6f80a538d2b1b379d0efd016e0defcaf594bc59d9ef5079f4c",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml": "f23b3ee13b988b1ff9a0bce438c416dac47e71355bbc7ee1dab21aaf3e33c18c",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml": "4c497fac683eee8292901c18ca02c0be1d9609d0b9d4bd8aa092c61f9e0b19e9",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml": "566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml": "e976aa0c5dba862cfa898ea57ec226efce9d1bc04b23b8d775908db86aa6fa41",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml": "429b584e89df3e3e8b81f4bd0d004b38bd0b4298bb390b8fb43efccb60963e01",
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml": "d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49",
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml": "56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml": "ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml": "a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml": "845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml": "cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml": "70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269",
}


def verify(games_folder: Path) -> None:
    """Raise unless every needed file matches the pinned SHA-256."""
    import hashlib

    for name, expected in SHA256.items():
        digest = hashlib.sha256()
        with open(games_folder / name, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{name} differs from the pinned DFL open data (revision {REVISION[:7]})")


def folder(path: str | os.PathLike | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    candidate = Path(path) if path else Path(os.environ[ENV]) if os.environ.get(ENV) else root / DEFAULT_FOLDER
    if not any(candidate.glob("DFL_04_03_positions_raw_observed_*.xml")):
        raise FileNotFoundError(f"no DFL positions files in {candidate}; see mtah/dfl.py")
    return candidate


def _file(games_folder: Path, kind: str, match_id: str) -> Path:
    found = sorted(games_folder.glob(f"DFL_{kind}_*_DFL-MAT-{match_id}.xml"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {kind} file for {match_id} in {games_folder}")
    return found[0]


def _match_info(path: Path):
    """Pitch size and each player's team (0 home, 1 away)."""
    from lxml import etree

    root = etree.parse(str(path)).getroot()
    env = root.find(".//Environment")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    team_of = {}
    for team_elm in root.iter("Team"):
        side = 0 if team_elm.get("Role") == "home" else 1
        for player in team_elm.iter("Player"):
            team_of[player.get("PersonId")] = side
    return pitch, team_of


def _read_positions(path: Path, team_of: dict):
    """{section: {object: (N, X, Y[, status, possession])}} streamed from the positions XML."""
    from lxml import etree

    out = {s: {} for s in SECTIONS}
    section = obj = None
    rows: list = []
    for event, elem in etree.iterparse(str(path), events=("start", "end"), tag=("FrameSet", "Frame")):
        if elem.tag == "FrameSet":
            if event == "start":
                section = elem.get("GameSection")
                obj = "ball" if elem.get("TeamId") == "BALL" else elem.get("PersonId")
                rows = []
            else:
                if section in out and (obj == "ball" or obj in team_of) and rows:
                    out[section][obj] = np.array(rows, dtype=float)
                elem.clear()
        elif event == "end":
            if section in out and (obj == "ball" or obj in team_of):
                row = [float(elem.get("N")), float(elem.get("X")), float(elem.get("Y"))]
                if obj == "ball":
                    row += [float(elem.get("BallStatus", 0)), float(elem.get("BallPossession", 0))]
                rows.append(row)
            elem.clear()
    return out


def _build(positions, team_of, pitch):
    """Arrays for both halves back to back, in pitch coordinates from a corner like our match."""
    shift = np.array(pitch) / 2  # DFL measures from the centre spot
    persons = sorted({p for sec in positions.values() for p in sec if p != "ball"})
    slot = {p: i for i, p in enumerate(persons)}
    balls, players, labels, halves, start = [], [], [], [], 0
    for section in SECTIONS:
        sec = positions[section]
        ball = sec["ball"]
        first, last = int(ball[:, 0].min()), int(ball[:, 0].max())
        n = last - first + 1
        b = np.full((n, 2), np.nan)
        idx = ball[:, 0].astype(int) - first
        b[idx] = ball[:, 1:3] + shift
        lab = np.full(n, X, dtype=LABEL_DTYPE)
        status, owner = ball[:, 3], ball[:, 4]
        lab[idx] = np.where(status != 1, D, np.where(owner == 1, H, np.where(owner == 2, A, X)))
        p = np.full((n, len(persons), 2), np.nan)
        for person, rows in sec.items():
            if person == "ball":
                continue
            keep = (rows[:, 0] >= first) & (rows[:, 0] <= last)
            p[rows[keep, 0].astype(int) - first, slot[person]] = rows[keep, 1:3] + shift
        balls.append(b)
        players.append(p)
        labels.append(lab)
        halves.append((start, start + n))
        start += n
    team = np.array([team_of[p] for p in persons], dtype=np.int8)
    return np.concatenate(balls), np.concatenate(players), team, np.concatenate(labels), halves


def load_game(games_folder: Path, match_id: str, cache: Path | None = None) -> dict:
    """One match in the shape of mtah/opengames.load_game, cached after the first read."""
    cache = cache or Path(__file__).resolve().parents[1] / "outputs" / "cache" / f"dfl_{match_id}.npz"
    positions_file = _file(games_folder, "04_03_positions_raw_observed", match_id)
    if cache.exists():
        z = np.load(cache)
        ball, players, team, labels = z["ball"], z["players"], z["team"], z["labels"]
        halves = [tuple(h) for h in z["halves"].tolist()]
        pitch = tuple(z["pitch"].tolist())
    else:
        pitch, team_of = _match_info(_file(games_folder, "02_01_matchinformation", match_id))
        ball, players, team, labels, halves = _build(_read_positions(positions_file, team_of), team_of, pitch)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ball=ball, players=players, team=team, labels=labels,
                            halves=np.array(halves), pitch=np.array(pitch))
    codes = np.select([labels == H, labels == A, labels == D], [0, 1, 2], 3).astype(np.int8)
    tracking = Tracking(ball, players, team)
    return dict(name=match_id, tracking=tracking, events=[], labels=labels, halves=halves, codes=codes,
                ball=ball, players=players, team=team, pitch=pitch)


def degrade(gm: dict, seed: int) -> dict:
    """The match as our feed would show it. DFL's optical tracking almost never loses the ball, while
    ours misses 41% of graded frames, so ball gaps follow mtah/synthetic.py's model (shaped by the real
    match's aggregates), starting more often just before a change of possession, plus glitches and
    missing players. Seeded, so it is the same everywhere."""
    import math

    from mtah import synthetic as syn
    from mtah.config import FPS

    rng = np.random.default_rng(seed)
    ball, players = gm["ball"].copy(), gm["players"].copy()
    T = len(ball)
    codes = gm["codes"]
    changes = np.flatnonzero((codes[1:] != codes[:-1]) & (codes[1:] < 2)) + 1
    mean_seen = syn.MEAN_GAP_FRAMES * (1 - syn.BALL_LOST_FRAC) / syn.BALL_LOST_FRAC
    lost = np.zeros(T, dtype=bool)
    f = int(rng.exponential(mean_seen))
    while f < T:
        if changes.size and rng.random() < syn.CROWD_GAP_PROB:
            j = np.searchsorted(changes, f)
            if j < changes.size and changes[j] - f < 5 * FPS:
                f = max(f, int(changes[j]) - int(rng.integers(0, FPS)))
        length = max(1, math.ceil(rng.exponential(syn.MEAN_GAP_FRAMES)))
        lost[f : f + length] = True
        f += length + max(1, math.ceil(rng.exponential(mean_seen)))
    ball[lost] = np.nan
    glitch = rng.random(T) < syn.GLITCH_FRAC
    ball[glitch] += rng.normal(0.0, 10.0, size=(int(glitch.sum()), 2))
    players[rng.random(players.shape[:2]) < syn.PLAYER_LOST_FRAC] = np.nan
    return dict(gm, ball=ball, players=players, tracking=Tracking(ball, players, gm["team"]))


def training_games(path: str | os.PathLike | None = None, degraded: bool = True) -> list[dict]:
    """All seven matches, checked against the pinned files, by default degraded to look like our feed
    (seed 7, 8, ... per match)."""
    games_folder = folder(path)
    verify(games_folder)
    games = [load_game(games_folder, m) for m in MATCHES]
    return [degrade(gm, 7 + k) for k, gm in enumerate(games)] if degraded else games
