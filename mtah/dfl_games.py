"""Seven DFL Bundesliga matches, the IDSSE open data, as labelled data (R2). Other matches, never ours.

Bassek et al. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer",
Scientific Data. figshare blocks automated downloads, so the files come from the Hugging Face
mirror pysport/idsse-data at revision HF_REVISION, each checked against the hash that revision
publishes for it:

    python -m mtah.dfl_games --download    # 14 files, 2.6 GB, into outputs/dfl/raw (gitignored)

Labels come from the tracking provider's own flags on the ball, read exactly as kloppy's
sportec.load_tracking(only_alive=False) reads them, which is how our ground truth is built: D where
BallStatus is not 1, else H where BallPossession is 1, else A. A frame with no ball record (kloppy
drops it) is X. The rules allow data about other games (RULES.md); we use these to choose, never
our match's ground truth.

The XML is read as a stream, every element cleared once read, into typed arrays: about 12 s and a
290 MB peak for a 400 MB file, where kloppy keeps a dict per object per frame. Positions are
centre-origin metres in the files and corner-origin here (interfaces.py); referees are left out,
and a player's slot is NaN while he is not on the pitch. Cached in outputs/cache as plain arrays
(no pickle).

The optical ball is almost never missing and sits on its carrier, so the benchmark hides it with
mtah.metrica_games.mask_ball and can move it with displace_ball, as for Metrica. mask_ball crowds
its gaps before contacts; with no event file read, a contact is a frame where a player other than
the last one comes within CONTACT_M of the true ball while it is in play.
"""
import argparse
import hashlib
import os
import urllib.request
import xml.etree.ElementTree as ET
from array import array
from pathlib import Path
from typing import NamedTuple

import numpy as np

from mtah.config import FPS
from mtah.interfaces import A, AWAY, D, H, HOME, LABEL_DTYPE, X, Interval, IntArray, LabelArray, Tracking
from mtah.metrica_games import CACHE_DIR, REPO, OpenGame

DFL_DIR = REPO / "outputs" / "dfl" / "raw"
HF_REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
HF_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/" + HF_REVISION + "/"
SECTIONS = ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra")  # kloppy's periods 1-4
CONTACT_M = 1.0
MASK_SEED_BASE = 100  # match i gets ball mask seed MASK_SEED_BASE + i
MAX_PERIOD_FRAMES = 2 * 3600 * FPS  # no period lasts two hours


class Match(NamedTuple):
    match_id: str
    competition: str
    info_sha1: str  # git blob id of the match information file (published by the mirror)
    info_size: int
    positions_sha256: str  # LFS object id of the positions file: the sha256 of its bytes
    positions_size: int


MATCHES = (
    Match("J03WMX", "000001", "8f2a82d84428c28e0e01f0a908c8395cc8d23900", 11930,
          "d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49", 418524833),
    Match("J03WN1", "000001", "45d2f112aed6c4212e66db8a91e8da1b8d728090", 11722,
          "56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec", 391722803),
    Match("J03WOH", "000002", "a693a0cacc2b597357e65a04eec5febfdaff79aa", 11745,
          "ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f", 348945715),
    Match("J03WOY", "000002", "bce56e5f518216e62d1e93a38322d4b9e357e5ab", 11952,
          "a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b", 362567859),
    Match("J03WPY", "000002", "46f1ffe15e1b96b12aab10594b12886c503c2f68", 11922,
          "845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1", 371688499),
    Match("J03WQQ", "000002", "7ec8acbff5a0c6167b632cc230d2a8b0e7b44b1c", 11932,
          "cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f", 356219423),
    Match("J03WR9", "000002", "7a8eb84881a1631da510982c01c90a3afd1b2e23", 11823,
          "70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269", 372934140),
)
GAMES = tuple(m.match_id for m in MATCHES)
MASK_SEED = {m.match_id: MASK_SEED_BASE + i for i, m in enumerate(MATCHES)}


def info_name(m: Match) -> str:
    return f"DFL_02_01_matchinformation_DFL-COM-{m.competition}_DFL-MAT-{m.match_id}.xml"


def positions_name(m: Match) -> str:
    return f"DFL_04_03_positions_raw_observed_DFL-COM-{m.competition}_DFL-MAT-{m.match_id}.xml"


# -- files ------------------------------------------------------------------------------------------

def _match(match_id: str) -> Match:
    for m in MATCHES:
        if m.match_id == match_id:
            return m
    raise ValueError(f"DFL matches are {GAMES}, got {match_id!r}")


def _digest(path: Path, kind: str, size: int) -> str:
    """sha256 of the bytes, or the git blob id (sha1 of a 'blob <size>' header and the bytes)."""
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(m: Match, root: Path) -> list[tuple[Path, str, str, int]]:
    return [(root / info_name(m), "sha1", m.info_sha1, m.info_size),
            (root / positions_name(m), "sha256", m.positions_sha256, m.positions_size)]


def check_files(match_id: str, root: Path = DFL_DIR) -> None:
    """Raise unless both of the match's files are exactly the ones the pinned revision publishes."""
    for path, kind, want, size in _files(_match(match_id), Path(root)):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: python -m mtah.dfl_games --download")
        if path.stat().st_size != size or _digest(path, kind, size) != want:
            raise ValueError(f"{path.name} is not the file revision {HF_REVISION[:7]} publishes (size or hash)")


def download(root: Path = DFL_DIR) -> None:
    """Fetch every missing file from the pinned revision, keeping it only if its hash matches."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for m in MATCHES:
        for path, kind, want, size in _files(m, root):
            if path.is_file() and path.stat().st_size == size:
                continue
            part = path.with_name(path.name + ".part")
            with urllib.request.urlopen(HF_URL + path.name, timeout=60) as response, open(part, "wb") as out:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    out.write(chunk)
            if part.stat().st_size != size or _digest(part, kind, size) != want:
                part.unlink()
                raise ValueError(f"{path.name}: the download does not match revision {HF_REVISION[:7]}")
            os.replace(part, path)
            print(f"ok {path.name}")


# -- reading ----------------------------------------------------------------------------------------

def read_match_information(path: Path) -> tuple[dict[str, int], tuple[float, float]]:
    """({PersonId: HOME or AWAY} for both squads, (pitch length, width) in metres)."""
    root = ET.parse(path).getroot()
    env = root.find("MatchInformation/Environment")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    players = {}
    for team in root.iterfind("MatchInformation/Teams/Team"):
        role = team.get("Role")
        if role not in ("home", "guest"):
            raise ValueError(f"{path.name}: team role {role!r}")
        for p in team.iterfind("Players/Player"):
            players[p.get("PersonId")] = HOME if role == "home" else AWAY
    return players, pitch


def read_positions(path: Path, players: dict[str, int], pitch: tuple[float, float]
                   ) -> tuple[Tracking, tuple[Interval, ...], LabelArray]:
    """The positions file as (tracking, halves, labels), periods one after another.

    A period's frames are its ball frames' N, first to last, so time runs at 25 fps throughout; a
    frame inside that range with no ball record has a NaN ball and label X. Players outside it are
    dropped, as kloppy drops frames without a ball.
    """
    sets: dict[str, dict[str, list[array]]] = {}  # section -> object -> [N, X, Y(, BallPossession, BallStatus)]
    rows = None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            if elem.tag == "FrameSet":
                section, team, person = elem.get("GameSection"), elem.get("TeamId"), elem.get("PersonId")
                obj = "ball" if team == "BALL" else person if person in players else None  # referees: None
                rows = None
                if obj is not None and section in SECTIONS:
                    kinds = ("l", "d", "d", "b", "b") if obj == "ball" else ("l", "d", "d")  # typed: 8 bytes a value
                    rows = sets.setdefault(section, {}).setdefault(obj, [array(k) for k in kinds])
            continue
        if elem.tag == "Frame":
            if rows is not None:
                a = elem.attrib
                rows[0].append(int(a["N"]))
                rows[1].append(float(a["X"]))
                rows[2].append(float(a["Y"]))
                if len(rows) == 5:
                    rows[3].append(int(a.get("BallPossession", 0)))
                    rows[4].append(int(a.get("BallStatus", 0)))
            elem.clear()
        elif elem.tag == "FrameSet":
            elem.clear()
    sections = [s for s in SECTIONS if s in sets and "ball" in sets[s]]
    if not sections:
        raise ValueError(f"{path.name}: no ball frames")
    slots = sorted({o for s in sections for o in sets[s] if o != "ball"}, key=lambda o: (players[o], o))
    index = {o: k for k, o in enumerate(slots)}
    offset = np.array(pitch) / 2.0  # centre origin to corner origin
    balls, people, labels, halves, start = [], [], [], [], 0
    for s in sections:
        n_ball = np.array(sets[s]["ball"][0])
        first, last = int(n_ball.min()), int(n_ball.max())
        n = last - first + 1
        if n > MAX_PERIOD_FRAMES:  # a corrupt frame number would otherwise allocate gigabytes
            raise ValueError(f"{path.name}: {s} spans {n} frames, more than {MAX_PERIOD_FRAMES}")
        ball = np.full((n, 2), np.nan)
        lab = np.full(n, X, dtype=LABEL_DTYPE)
        f = n_ball - first
        poss, status = np.array(sets[s]["ball"][3]), np.array(sets[s]["ball"][4])
        ball[f] = np.column_stack([sets[s]["ball"][1], sets[s]["ball"][2]]) + offset
        lab[f] = np.where(status != 1, D, np.where(poss == 1, H, A))
        pos = np.full((n, len(slots), 2), np.nan)
        for o, r in sets[s].items():
            if o == "ball":
                continue
            g = np.array(r[0]) - first
            ok = (g >= 0) & (g < n)
            pos[g[ok], index[o]] = np.column_stack([r[1], r[2]])[ok] + offset
        balls.append(ball)
        people.append(pos)
        labels.append(lab)
        halves.append((start, start + n - 1))
        start += n
    team = np.array([players[o] for o in slots], dtype=np.int64)
    return Tracking(np.concatenate(balls), np.concatenate(people), team, FPS), tuple(halves), np.concatenate(labels)


def contact_frames(tracking: Tracking, labels: LabelArray) -> IntArray:
    """Sorted live frames where a player other than the last one comes within CONTACT_M of the
    true ball: receptions and recoveries, as metrica_games.contact_frames takes from events."""
    d = np.linalg.norm(tracking.players - tracking.ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(d, axis=1)
    touch = np.flatnonzero((d[np.arange(len(d)), nearest] < CONTACT_M) & (labels != D) & (labels != X))
    who = nearest[touch]
    new = np.r_[True, who[1:] != who[:-1]]
    return touch[new].astype(np.int64)


def load_match(match_id: str, root: Path = DFL_DIR, cache_dir: Path | None = CACHE_DIR) -> OpenGame:
    """One DFL match as an OpenGame (its number is the match id). Read once, after checking the
    files' hashes, then cached; delete outputs/cache/dfl_*.npz after changing the reader."""
    m = _match(match_id)
    cache = None if cache_dir is None else Path(cache_dir) / f"dfl_{m.match_id}.npz"
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as s:  # plain arrays only: a cache cannot run code
            tracking = Tracking(s["ball"], s["players"], s["team"], FPS)
            halves = tuple((int(a), int(b)) for a, b in s["halves"])
            return OpenGame(m.match_id, tracking, halves, s["labels"], s["contacts"], tuple(float(v) for v in s["pitch"]))
    check_files(m.match_id, root)
    players, pitch = read_match_information(Path(root) / info_name(m))
    tracking, halves, labels = read_positions(Path(root) / positions_name(m), players, pitch)
    contacts = contact_frames(tracking, labels)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ball=tracking.ball, players=tracking.players, team=tracking.team,
                            halves=np.array(halves, dtype=np.int64), labels=labels, contacts=contacts,
                            pitch=np.array(pitch))
    return OpenGame(m.match_id, tracking, halves, labels, contacts, pitch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="fetch the missing files into outputs/dfl/raw")
    if parser.parse_args().download:
        download()
