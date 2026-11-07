"""Run the stub pipeline end to end on synthetic data and write a submission-shaped CSV.

    python run.py                                   # 199,319 frames -> outputs/synthetic.csv
    python run.py --frames 5000 --out outputs/small.csv

Synthetic data only: nothing here reads $MTAH_DATA_DIR.
"""
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from mtah.ball import ball_clean, gap_stats
from mtah.config import N_FRAMES
from mtah.deadball import dead_mask
from mtah.evidence import evidence
from mtah.interfaces import X, Interval, LabelArray, Tracking
from mtah.segment import segment
from mtah.synthetic import make_synthetic
from mtah.touches import touches
from mtah.xpolicy import to_x_policy

DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "synthetic.csv"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.out, args.frames, args.seed)
