"""Load the tactical ATD into a Tracking (R1).

Parsing is kloppy's own FIFA/EPTS line reader (the one behind starter/load_atd.py), but streamed:
each line goes straight into numpy arrays, about 150 MB in all. kloppy's full load builds a Python
object per player per frame and then a table, which runs out of memory on a laptop. To prove the
streamed reader is right without anyone looking at data, the first VERIFY_FRAMES frames are also
loaded the official way and compared automatically; any difference stops the load.

This module adds what the pipeline depends on: frame_id - 1 indexing, 0..1 coordinates to metres,
slots to home/away.

    python -m mtah.io_atd        # load the real ATD, cache it, print aggregate numbers only

The cache lives in outputs/cache/ (gitignored, and too big for scripts/check_repo.sh to let
through), so the data/ folder stays exactly as games.py expects. Delete the cache to rebuild it.
"""
from pathlib import Path
from typing import Any

import numpy as np

from mtah.ball import SPEED_CAP_MS, _gaps
from mtah.config import FPS, GRADED_HALVES, N_FRAMES, PITCH_M, data_dir
from mtah.interfaces import AWAY, HOME, Tracking
from mtah.teams import slot_teams

VERIFY_FRAMES = 300  # frames also loaded the official kloppy way, to check the streamed reader


def default_cache() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "cache" / "tactical_tracking.npz"


def tracking_from_frame(
    df: Any, slot_team: dict[str, int], n_frames: int = N_FRAMES, pitch_m: tuple = PITCH_M
) -> Tracking:
    """Build a Tracking from kloppy's to_df() layout (DATA.md).

    df has frame_id (1-based), ball_x, ball_y and <slot>_x, <slot>_y for each slot, all in 0..1.
    Every frame_id 1..n_frames must appear exactly once; anything else raises rather than
    silently shifting rows. Row f of the result is frame_id f + 1.
    """
    ids = df["frame_id"].to_numpy()
    if len(ids) != n_frames or not np.array_equal(np.sort(ids), np.arange(1, n_frames + 1)):
        raise ValueError(f"frame_id must be exactly 1..{n_frames}, got {len(ids)} records "
                         f"spanning {ids.min() if len(ids) else None}..{ids.max() if len(ids) else None}")
    slots = _sorted_slots(slot_team)
    missing = [s for s in slots if f"{s}_x" not in df.columns or f"{s}_y" not in df.columns]
    if missing:
        raise ValueError(f"no x/y columns for slots {missing[:5]}; columns start {list(df.columns[:8])}")

    order = np.argsort(ids, kind="stable")
    scale = np.asarray(pitch_m, dtype=float)

    def metres(prefix: str) -> np.ndarray:
        return df[[f"{prefix}_x", f"{prefix}_y"]].to_numpy(dtype=float)[order] * scale

    ball = metres("ball")
    players = np.stack([metres(s) for s in slots], axis=1)
    team = np.array([slot_team[s] for s in slots], dtype=np.int8)
    return Tracking(ball, players, team, FPS)


def tracking_from_rows(
    rows: Any, slot_team: dict[str, int], n_frames: int = N_FRAMES, pitch_m: tuple = PITCH_M
) -> Tracking:
    """Build a Tracking from kloppy's raw EPTS rows, streamed one dict per line.

    Each row has frame_id (1-based), ball_x, ball_y and player_<slot>_x, player_<slot>_y, all in
    0..1, NaN when missing. Every frame_id 1..n_frames must appear exactly once.
    """
    slots = _sorted_slots(slot_team)
    keys = [(f"player_{s}_x", f"player_{s}_y") for s in slots]
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, len(slots), 2), np.nan)
    seen = np.zeros(n_frames, dtype=bool)
    nan = float("nan")
    for row in rows:
        if not seen.any():
            missing = [s for s, (kx, ky) in zip(slots, keys) if kx not in row or ky not in row]
            if missing:
                raise ValueError(f"no x/y fields for slots {missing[:5]}; fields start {list(row)[:6]}")
        f = int(row["frame_id"]) - 1
        if not 0 <= f < n_frames or seen[f]:
            raise ValueError(f"frame_id {f + 1} is outside 1..{n_frames} or repeated")
        seen[f] = True
        ball[f] = (row.get("ball_x", nan), row.get("ball_y", nan))
        players[f] = [(row[kx], row[ky]) for kx, ky in keys]
    if not seen.all():
        raise ValueError(f"{int((~seen).sum())} frame_ids of 1..{n_frames} never appear")
    scale = np.asarray(pitch_m, dtype=float)
    team = np.array([slot_team[s] for s in slots], dtype=np.int8)
    return Tracking(ball * scale, players * scale, team, FPS)


def _sorted_slots(slot_team: dict[str, int]) -> list[str]:
    return sorted(slot_team, key=lambda s: (0, int(s), s) if s.isdigit() else (1, 0, s))


def load_tracking(cache: Path | None = None) -> Tracking:
    """The real tactical ATD as a Tracking. Reads `cache` if it exists, else loads and writes it."""
    if cache is not None and Path(cache).is_file():
        with np.load(cache) as saved:
            if "names" in saved.files:  # a cache from before team names were stored is rebuilt
                return Tracking(saved["ball"], saved["players"], saved["team"], FPS)
    tracking, names = _load_from_atd()
    if cache is not None:
        _save(cache, tracking, names)
    return tracking


def team_names(cache: Path) -> tuple[str, str]:
    """(home, away) club names as the ATD metadata spells them, read from the cache."""
    with np.load(cache) as saved:
        if "names" not in saved.files:
            raise RuntimeError("the cache predates team names; delete outputs/cache/ and reload")
        home, away = saved["names"].tolist()
    return str(home), str(away)


load = load_tracking  # the name used in the team plan


def _load_from_atd() -> tuple[Tracking, tuple[str, str]]:
    """(Tracking, (home name, away name)) streamed from the ATD files, checked against kloppy."""
    # kloppy is imported here so tests never need it
    from kloppy import metrica
    from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata
    from kloppy.infra.serializers.tracking.metrica_epts.reader import read_raw_data
    from starter.load_atd import atd_paths

    meta_path, raw_path = atd_paths(data_dir=str(data_dir()))
    with open(meta_path, "rb") as handle:
        metadata = load_metadata(handle)
    slot_team = slot_teams(metadata.teams)
    with open(raw_path, "rb") as handle:
        tracking = tracking_from_rows(read_raw_data(handle, metadata, sample_rate=1.0, limit=0), slot_team)

    head = metrica.load_tracking_epts(meta_data=meta_path, raw_data=raw_path, limit=VERIFY_FRAMES)
    _check_against(tracking, head.to_df(), slot_team)
    return tracking, (str(metadata.teams[0]), str(metadata.teams[1]))


def _check_against(tracking: Tracking, df: Any, slot_team: dict[str, int]) -> None:
    """Stop if the streamed arrays differ from kloppy's own load on the frames in df.

    The message names columns only, never values, so it is safe to share.
    """
    rows = df["frame_id"].to_numpy(dtype=int) - 1
    scale = np.asarray(PITCH_M, dtype=float)
    pairs = [("ball", tracking.ball)] + [
        (s, tracking.players[:, i]) for i, s in enumerate(_sorted_slots(slot_team))
    ]
    for prefix, ours in pairs:
        theirs = df[[f"{prefix}_x", f"{prefix}_y"]].to_numpy(dtype=float)
        if not np.allclose(ours[rows] / scale, theirs, equal_nan=True):
            raise RuntimeError(f"streamed reader disagrees with kloppy on {prefix}_x/{prefix}_y "
                               f"in the first {len(rows)} frames; do not use this cache")


def _save(cache: Path, tracking: Tracking, names: tuple[str, str]) -> None:
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ball=tracking.ball, players=tracking.players, team=tracking.team,
                        names=np.array(names))


def summary(tracking: Tracking) -> dict[str, float]:
    """Aggregate numbers only, comparable with the figures in DATA.md. No per-frame values."""
    graded = np.zeros(tracking.n_frames, dtype=bool)
    for first, last in GRADED_HALVES:
        graded[first : last + 1] = True
    ball_seen = ~np.isnan(tracking.ball).any(axis=1)
    tracked = (~np.isnan(tracking.players[..., 0])).sum(axis=1)

    def mean(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else float("nan")  # e.g. a short synthetic track

    def pct(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values, q)) if values.size else float("nan")

    ball_g = tracking.ball[graded & ball_seen]
    pitch = np.asarray(PITCH_M, dtype=float)
    step = np.linalg.norm(np.diff(tracking.ball, axis=0), axis=1) * FPS  # m/s, NaN across gaps
    pairs = graded[1:] & ball_seen[1:] & ball_seen[:-1]
    graded_stats = {
        "max_tracked_graded": int(tracked[graded].max()) if graded.any() else 0,
        "frac_over_22_tracked_graded": mean(tracked[graded] > 22),
        "ball_off_pitch_graded": mean(((ball_g < 0) | (ball_g > pitch)).any(axis=1)),
        "ball_x_p01_m": pct(ball_g[:, 0], 1),
        "ball_x_p99_m": pct(ball_g[:, 0], 99),
        "ball_y_p01_m": pct(ball_g[:, 1], 1),
        "ball_y_p99_m": pct(ball_g[:, 1], 99),
        "ball_over_speed_cap_graded": mean(step[pairs] > SPEED_CAP_MS),
    }

    # Coverage per half, and the gap-length distribution with gaps counted inside each half.
    half_gaps = []
    for half, (first, last) in enumerate(GRADED_HALVES, start=1):
        seen_half = ball_seen[first : last + 1]
        graded_stats[f"ball_present_half{half}"] = mean(seen_half)
        starts, ends = _gaps(~seen_half)
        half_gaps.append(ends - starts)
    gaps_s = np.concatenate(half_gaps) / FPS
    graded_stats["n_gaps_graded"] = int(gaps_s.size)
    for q in (25, 50, 75, 90):
        graded_stats[f"gap_p{q}_s"] = pct(gaps_s, q)
    graded_stats["gap_max_s"] = float(gaps_s.max()) if gaps_s.size else float("nan")

    # A team with more than 11 tracked objects means referees, duplicates or slots that swap
    # team: the per-slot team mapping would then be wrong for some frames.
    seen_slot = ~np.isnan(tracking.players[..., 0])
    for name, side in (("home", HOME), ("away", AWAY)):
        per_frame = seen_slot[:, tracking.team == side].sum(axis=1)[graded]
        graded_stats[f"max_{name}_tracked_graded"] = int(per_frame.max()) if per_frame.size else 0
        graded_stats[f"frac_{name}_over_11_graded"] = mean(per_frame > 11)

    return graded_stats | {
        "n_frames": int(tracking.n_frames),
        "n_slots": int(tracking.team.size),
        "home_slots": int((tracking.team == HOME).sum()),
        "away_slots": int((tracking.team == AWAY).sum()),
        "ball_present_all": mean(ball_seen),
        "ball_present_graded": mean(ball_seen[graded]),
        "mean_tracked_graded": mean(tracked[graded]),
        "max_tracked": int(tracked.max()),
        "ball_x_min_m": float(np.nanmin(tracking.ball[:, 0])),
        "ball_x_max_m": float(np.nanmax(tracking.ball[:, 0])),
        "ball_y_min_m": float(np.nanmin(tracking.ball[:, 1])),
        "ball_y_max_m": float(np.nanmax(tracking.ball[:, 1])),
    }


# What DATA.md says the real tactical ATD should give.
EXPECTED = {
    "n_frames": 199_319,
    "n_slots": 46,
    "ball_present_all": 0.569,
    "ball_present_graded": 0.593,
    "mean_tracked_graded": 20.5,
}


if __name__ == "__main__":
    from mtah.ball import ball_clean

    cache = default_cache()
    tracking = load_tracking(cache)  # loads the ATD (minutes) only when there is no usable cache
    home, away = team_names(cache)
    print(f"home = {home}, away = {away}   (DATA.md: home is Barcelona)")
    for key, value in summary(tracking).items():
        expected = f"   (DATA.md: {EXPECTED[key]})" if key in EXPECTED else ""
        shown = value if isinstance(value, int) else f"{value:.4g}"
        print(f"{key:>28} = {shown}{expected}")
    graded = np.zeros(tracking.n_frames, dtype=bool)
    for first, last in GRADED_HALVES:
        graded[first : last + 1] = True
    filled, lost = ball_clean(tracking.ball)
    speed = np.linalg.norm(np.diff(filled, axis=0), axis=1) * FPS
    trusted = graded[1:] & ~lost[1:] & ~lost[:-1]
    print(f"after cleaning, graded frames: lost {lost[graded].mean():.4f}, "
          f"trusted steps over {SPEED_CAP_MS:g} m/s: {int((speed[trusted] > SPEED_CAP_MS).sum())}")
