"""Run the pipeline end to end and write a submission-shaped CSV.

    python run.py                                   # synthetic, 199,319 frames -> outputs/synthetic.csv
    python run.py --frames 5000 --out outputs/small.csv
    python run.py --real                            # our match -> outputs/The Playmakers.csv

Without --real nothing here reads $MTAH_DATA_DIR. With --real it reads the tactical tracking (through
mtah/io_atd.py, cached in outputs/cache/) and the Smart Tagging (mtah/tagging.py), writes the
submission and prints only totals and the file's SHA-256: never a frame, a row or a position.
Running it twice, or on another computer with the same data and requirements, must print the same
SHA-256 (SUBMISSION.md: the repository must regenerate the same file).
"""
import argparse
import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from mtah.ball import ball_clean, gap_stats
from mtah.config import GRADED_HALVES, N_FRAMES, SECOND_HALF_D_BLOCK
from mtah.deadball import dead_mask
from mtah.evidence import evidence
from mtah.interfaces import A, D, H, X, Interval, LabelArray, Tracking
from mtah.segment import segment
from mtah.synthetic import make_synthetic
from mtah.touches import touches
from mtah.xpolicy import to_x_policy

REPO = Path(__file__).resolve().parent
DEFAULT_OUT = REPO / "outputs" / "synthetic.csv"
TEAM_NAME = "The Playmakers"  # SUBMISSION.md: name the CSV after the team
REAL_OUT = REPO / "outputs" / f"{TEAM_NAME}.csv"


def predict(tracking: Tracking, set_piece_intervals: Sequence[Interval]) -> LabelArray:
    _, lost = ball_clean(tracking.ball)
    e = evidence(tracking)
    events = touches(tracking.ball, tracking.players, tracking.team)
    dead = dead_mask(tracking.n_frames, set_piece_intervals)
    labels = segment(e, lost, dead, events)
    # Abstain only where there is no evidence at all; to_x_policy keeps X away from hand-overs.
    return to_x_policy(labels, confidence=np.abs(e), threshold=1e-9)


def write_csv(labels: LabelArray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{frame},{label}\n" for frame, label in enumerate(labels))
    out.write_text("frame,label\n" + rows, encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out: Path = DEFAULT_OUT, n_frames: int = N_FRAMES, seed: int = 0) -> LabelArray:
    syn = make_synthetic(n_frames, seed)
    labels = predict(syn.tracking, syn.set_piece_intervals)
    write_csv(labels, out)

    stats = gap_stats(np.isnan(syn.tracking.ball).any(axis=1))
    lost_after = float(ball_clean(syn.tracking.ball)[1].mean())
    scored = labels != X
    accuracy = float(np.mean(labels[scored] == syn.labels[scored])) if scored.any() else 0.0
    print(f"wrote {out} ({n_frames} frames)")
    print("raw ball gaps: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
    print(f"lost after cleaning (gaps of 1 s or more, or at the ends): {lost_after:.3f}")
    print(f"X share {1 - scored.mean():.3f}, accuracy on non-X frames {accuracy:.3f}")
    return labels


# -- our match --------------------------------------------------------------------------------

def with_published_blocks(windows: Sequence[Interval]) -> list[Interval]:
    """The set-piece windows plus the organisers' published dead-ball block: the second half is
    graded from 110,808 but kicks off at 113,742 (FAQ.md, config.SECOND_HALF_D_BLOCK)."""
    return list(windows) + [SECOND_HALF_D_BLOCK]


def real_inputs() -> tuple[Tracking, list[Interval]]:
    """Our match's tracking and dead-ball intervals. Reads the data bundle: run it yourself, never
    inside an AI tool."""
    from mtah.io_atd import default_cache, load_tracking, team_names
    from mtah.tagging import load_tags, set_pieces

    cache = default_cache()
    tracking = load_tracking(cache)  # builds the cache from $MTAH_DATA_DIR the first time
    home, away = team_names(cache)
    windows = [(sp.start, sp.end) for sp in set_pieces(load_tags(home, away))]
    return tracking, with_published_blocks(windows)


def summary(labels: LabelArray, halves: Sequence[Interval] = GRADED_HALVES) -> dict[str, float]:
    """Totals only: label shares and possession counts on the graded frames. A possession is a
    maximal H or A run inside a half, with X filled by the previous label as the scorer does."""
    labels = np.asarray(labels)
    graded = np.zeros(labels.size, dtype=bool)
    n_poss = 0
    for first, last in halves:
        part = labels[first : last + 1]
        graded[first : last + 1] = True
        known = part[part != X]  # dropping X equals forward-filling it, for counting runs
        if known.size:
            starts = np.r_[True, known[1:] != known[:-1]]
            n_poss += int(np.sum(starts & ((known == H) | (known == A))))
    g = labels[graded]
    share = {k: float(np.mean(g == k)) if g.size else 0.0 for k in (H, A, D, X)}
    return {"graded_frames": int(g.size), **{f"share_{k}": v for k, v in share.items()}, "possessions": n_poss}


def main_real(out: Path = REAL_OUT, load: Callable[[], tuple[Tracking, list[Interval]]] = real_inputs,
              n_expected: int = N_FRAMES, halves: Sequence[Interval] = GRADED_HALVES) -> str:
    """Write the submission for our match and return its SHA-256. Prints totals only."""
    tracking, intervals = load()
    if tracking.n_frames != n_expected:
        raise ValueError(f"expected {n_expected} frames, got {tracking.n_frames}: wrong recording?")
    labels = predict(tracking, intervals)
    write_csv(labels, out)
    digest = sha256(out)
    print(f"wrote {out.name}: {labels.size} frames, {len(intervals)} dead-ball intervals given to dead_mask")
    for key, value in summary(labels, halves).items():
        print(f"{key:>14} = {value if isinstance(value, int) else f'{value:.4f}'}")
    print(f"sha256 {digest}")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true", help="our match from $MTAH_DATA_DIR (run it yourself)")
    parser.add_argument("--frames", type=int, default=N_FRAMES, help="synthetic only")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0, help="synthetic only")
    args = parser.parse_args()
    if args.real:
        main_real(args.out or REAL_OUT)
    else:
        main(args.out or DEFAULT_OUT, args.frames, args.seed)
