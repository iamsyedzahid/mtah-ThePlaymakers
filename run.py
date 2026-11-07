"""Run the pipeline end to end and write a submission-shaped CSV.

    python run.py --real                            # the real match -> outputs/real.csv
    python run.py --real --out "The Playmakers.csv" # the file to submit, named after the team
    python run.py --real --rules                    # the earlier rule-only pipeline, the fallback
    python run.py                                   # synthetic, rule-only, 199,319 frames -> outputs/synthetic.csv
    python run.py --frames 5000 --out outputs/small.csv

--real needs the data bundle ($MTAH_DATA_DIR, else data/ in this folder), Metrica's open sample games
and the DFL open matches, which train the learned layer ($MTAH_OPEN_GAMES or --open-games, see
mtah/opengames.py; $MTAH_DFL_GAMES or --dfl-games, see mtah/dfl.py). The
first run reads the ATD and caches it in outputs/cache/, which takes a few minutes. Without --real
nothing reads either.
"""
import argparse
import hashlib
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from mtah.ball import ball_clean, gap_stats
from mtah.config import FPS, GRADED_HALVES, N_FRAMES, PITCH_M, SECOND_HALF_D_BLOCK
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
from mtah.interfaces import A, D, H, LABEL_DTYPE, Interval, LabelArray, SetPiece, Tracking, X
from mtah.segment import run_bounds, segment
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


def predict_real(open_games: str | None = None, dfl_games: str | None = None) -> LabelArray:
    """The submission: mtah/fusion.py, its learned layer trained on Metrica's open games and the DFL
    open matches.

    Inside the graded halves only, X outside them, which is never scored. The one frame-specific fact is
    published: the second half kicks off late (FAQ.md). One thread, so every computer writes the same file.
    """
    from threadpoolctl import threadpool_limits

    from mtah import dfl, fusion, opengames
    from mtah.io_atd import default_cache, load_tracking

    def step(text):
        print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    with threadpool_limits(1):
        step("loading Metrica's open games (about a minute)")
        games = opengames.training_games(open_games)
        step("checking and loading the DFL open matches (about a minute)")
        games += dfl.training_games(dfl_games)
        rows = []
        for gm in games:
            step(f"features for {gm['name']}")
            rows += fusion.training_rows(gm)
        step(f"training the learned layer on {sum(len(y) for _, y in rows):,} frames (several minutes, one thread)")
        model = fusion.fit(rows)
        step("labelling our match")
        tracking = load_tracking(default_cache())
        known_dead = np.zeros(tracking.n_frames, dtype=bool)
        known_dead[SECOND_HALF_D_BLOCK[0] : SECOND_HALF_D_BLOCK[1] + 1] = True
        halves = [(a, b + 1) for a, b in GRADED_HALVES]
        codes = fusion.label(tracking.ball, tracking.players, tracking.team, PITCH_M, halves,
                             model=model, known_dead=known_dead)
    return np.array([H, A, D, X])[codes].astype(LABEL_DTYPE)


def predict_real_rules() -> LabelArray:
    """The earlier rule-only pipeline on the real match, kept as the fallback. It adds the homography's
    still-camera stoppages and the published second-half block (FAQ.md)."""
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


def main_real(out: Path = DEFAULT_REAL_OUT, rules: bool = False, open_games: str | None = None,
              dfl_games: str | None = None) -> LabelArray:
    labels = predict_real_rules() if rules else predict_real(open_games, dfl_games)
    write_csv(labels, out)
    print(f"wrote {out} ({len(labels)} frames)")
    # Totals per half too: the two halves should look alike, a sign it holds up over the whole match.
    for name, spans in (("graded", GRADED_HALVES), ("half 1", GRADED_HALVES[:1]), ("half 2", GRADED_HALVES[1:])):
        part = np.concatenate([labels[a : b + 1] for a, b in spans])
        runs = [(s, e) for a, b in spans for s, e in zip(*run_bounds(labels[a : b + 1])) if labels[a + s] in (H, A)]
        seconds = np.array([(e - s) / FPS for s, e in runs])
        minutes = len(part) / FPS / 60
        print(f"{name}: H {np.mean(part == H):.3f}  A {np.mean(part == A):.3f}  D {np.mean(part == D):.3f}; "
              f"possessions {len(runs)} ({len(runs) / minutes:.2f} a minute), median {np.median(seconds):.1f} s, "
              f"under 3 s {np.mean(seconds < 3):.3f}")
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
    parser.add_argument("--rules", action="store_true", help="with --real: the earlier rule-only pipeline")
    parser.add_argument("--open-games", default=None, help="with --real: the folder holding Sample_Game_1..3")
    parser.add_argument("--dfl-games", default=None, help="with --real: the folder holding the DFL open XML files")
    parser.add_argument("--frames", type=int, default=N_FRAMES, help="synthetic only")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0, help="synthetic only")
    args = parser.parse_args()
    if args.real:
        main_real(args.out or DEFAULT_REAL_OUT, rules=args.rules, open_games=args.open_games, dfl_games=args.dfl_games)
    else:
        main(args.out or DEFAULT_OUT, args.frames, args.seed)
