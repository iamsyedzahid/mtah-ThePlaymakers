"""Load the tactical ATD into a Tracking (R1).

File parsing is reused from starter/load_atd.py (kloppy's FIFA/EPTS reader). This module adds what
the pipeline depends on: frame_id - 1 indexing, 0..1 coordinates to metres, slots to home/away.

    python -m mtah.io_atd        # load the real ATD, cache it, print aggregate numbers only

kloppy takes minutes on the full match, so the check above writes a cache under outputs/cache/
(gitignored, and too big for scripts/check_repo.sh to let through). The data/ folder stays exactly
as games.py expects. Delete the cache file to rebuild it.
"""
from pathlib import Path
from typing import Any

import numpy as np

from mtah.ball import SPEED_CAP_MS, _gaps
from mtah.config import FPS, GRADED_HALVES, N_FRAMES, PITCH_M, data_dir
from mtah.interfaces import AWAY, HOME, Tracking
from mtah.teams import slot_teams


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
    slots = sorted(slot_team, key=lambda s: (0, int(s), s) if s.isdigit() else (1, 0, s))
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


def load_tracking(cache: Path | None = None) -> Tracking:
    """The real tactical ATD as a Tracking. Reads `cache` if it exists, else loads and writes it."""
    if cache is not None and Path(cache).is_file():
        with np.load(cache) as saved:
            return Tracking(saved["ball"], saved["players"], saved["team"], FPS)
    tracking, _ = _load_from_atd()
    if cache is not None:
        _save(cache, tracking)
    return tracking


load = load_tracking  # the name used in the team plan


def _load_from_atd() -> tuple[Tracking, Any]:
    """(Tracking, kloppy metadata.teams) straight from the ATD files. Takes minutes."""
    from starter.load_atd import load  # needs kloppy; imported here so tests never need it

    dataset = load(data_dir=str(data_dir()))
    teams = dataset.metadata.teams
    return tracking_from_frame(dataset.to_df(), slot_teams(teams)), teams


def _save(cache: Path, tracking: Tracking) -> None:
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ball=tracking.ball, players=tracking.players, team=tracking.team)


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
    if cache.is_file():
        tracking = load_tracking(cache)
        print("read the cache; delete outputs/cache/ to reload and re-check the team order")
    else:
        tracking, teams = _load_from_atd()
        print(f"home = {teams[0]}, away = {teams[1]}   (DATA.md: home is Barcelona)")
        _save(cache, tracking)
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
