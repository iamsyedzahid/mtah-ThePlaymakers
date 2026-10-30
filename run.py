"""Run the pipeline end to end and write a submission-shaped CSV.

    python run.py --real                            # the real match -> outputs/real.csv
    python run.py --real --out <team_name>.csv      # the file to submit, named after the team
    python run.py                                   # synthetic, 199,319 frames -> outputs/synthetic.csv
    python run.py --frames 5000 --out outputs/small.csv

--real needs the data bundle: $MTAH_DATA_DIR, else data/ in this folder. The first run reads the ATD
and caches it in outputs/cache/, which takes a few minutes. Without --real nothing reads the data.
"""
import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from mtah.ball import ball_clean, gap_stats
from mtah.config import GRADED_HALVES, N_FRAMES, SECOND_HALF_D_BLOCK
from mtah.deadball import (
    camera_still_intervals,
    dead_intervals,
    dead_mask,
    join_close,
    restart_touches,
    slow_lost_intervals,
    untagged_intervals,
)
from mtah.evidence import evidence
from mtah.interfaces import A, D, H, Interval, LabelArray, SetPiece, Tracking, X
from mtah.segment import segment
from mtah.synthetic import make_synthetic
from mtah.touches import touches

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "outputs" / "synthetic.csv"
DEFAULT_REAL_OUT = ROOT / "outputs" / "real.csv"


def predict(tracking: Tracking, set_pieces: Sequence[SetPiece], extra_dead: Sequence[Interval] = ()) -> LabelArray:
    """Labels H/A/D, never X: the scorer fills an X with the previous label for spellF1 but
    counts it as a miss frame by frame, so abstaining can only lose."""
    filled, lost = ball_clean(tracking.ball)
    e = evidence(tracking)
    events = touches(tracking.ball, tracking.players, tracking.team)
    tagged = dead_intervals(set_pieces, filled, lost)
    intervals = [*tagged, *untagged_intervals(filled, lost), *slow_lost_intervals(tracking.players, lost), *extra_dead]
    events = [*events, *restart_touches(set_pieces, tagged, tracking.n_frames)]
    return segment(e, lost, join_close(dead_mask(tracking.n_frames, intervals)), events)


def predict_real() -> LabelArray:
    """The real match. It adds what synthetic matches lack: the homography's still-camera stoppages, and
    its one frame-specific fact, which is published: the second half kicks off late (FAQ.md)."""
    from mtah.config import data_dir
    from mtah.io_atd import default_cache, load_tracking, team_names
    from mtah.tagging import load_tags, set_pieces
    from starter.load_homography import load as load_homography

    tracking = load_tracking(default_cache())
    home, away = team_names(default_cache())
    tags = set_pieces(load_tags(home, away, data_dir=str(data_dir())))
    matrices = load_homography(data_dir=str(data_dir()))[0]
    still = camera_still_intervals(matrices, tracking.players, ball_clean(tracking.ball)[1])
    return predict(tracking, tags, extra_dead=[*still, SECOND_HALF_D_BLOCK])


def write_csv(labels: LabelArray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{frame},{label}\n" for frame, label in enumerate(labels))
    out.write_text("frame,label\n" + rows, encoding="utf-8", newline="\n")


def main_real(out: Path = DEFAULT_REAL_OUT) -> LabelArray:
    labels = predict_real()
    write_csv(labels, out)
    graded = np.concatenate([labels[a : b + 1] for a, b in GRADED_HALVES])
    print(f"wrote {out} ({len(labels)} frames)")
    print(f"graded frames: H {np.mean(graded == H):.3f}  A {np.mean(graded == A):.3f}  D {np.mean(graded == D):.3f}")
    print(f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return labels


def main(out: Path = DEFAULT_OUT, n_frames: int = N_FRAMES, seed: int = 0) -> LabelArray:
    syn = make_synthetic(n_frames, seed)
    labels = predict(syn.tracking, syn.set_pieces)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true", help="the real match instead of synthetic data")
    parser.add_argument("--frames", type=int, default=N_FRAMES, help="synthetic only")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0, help="synthetic only")
    args = parser.parse_args()
    if args.real:
        main_real(args.out or DEFAULT_REAL_OUT)
    else:
        main(args.out or DEFAULT_OUT, args.frames, args.seed)
