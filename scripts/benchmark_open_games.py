"""Open-games benchmark (R5): the whole pipeline, scored on other matches, never on ours.

    python scripts/benchmark_open_games.py                        # Metrica's three open games
    python scripts/benchmark_open_games.py --dfl                  # the seven DFL matches
    python scripts/benchmark_open_games.py --code outputs/wt-r4   # another branch's mtah/, same data

This is how the team chooses between versions (TEAM_PLAN.md, section 6) without touching our match.
The organisers' clarification allows training, tuning and choosing on external data only.

Per game, the same steps as the submission, on another match:
- the tracking is the open game's, with the ball hidden and moved off its carrier exactly as
  mtah/evidence_fit.py does (mtah.metrica_games.masked_tracking, fixed seeds);
- set-piece windows are made from the game's own restarts: every frame where a dead ball (D) turns
  live gets a window from WINDOW_BEFORE_S before it to WINDOW_AFTER_S after it, standing in for
  Smart Tagging's set pieces (about 5 s around the restart, interfaces.SetPiece). Every restart gets
  one, so this is kinder than Smart Tagging, which misses some. The window sizes are an assumption,
  not measured on our match;
- touches, dead_mask, segment and to_x_policy come from the code tree being scored (--code), so a
  branch is scored as it would ship. Only the evidence is swapped, between the variants below;
- the scorer is this repository's evaluation/possession_metrics.py, imported as is.

Evidence variants. All use the nearest player on trusted frames; they differ on lost frames:
- nearest:        nothing on lost frames (main's evidence()).
- shape_self:     plus the shape model trained per half on the game's own clear contacts
                  (evidence_shape_calibrated; R4's evidence()). On our match this would train on
                  our match, which the organisers' clarification may not allow.
- shape_external: plus a shape model trained only on the OTHER games' true labels (leave one game
                  out), so nothing is fitted on the game it is applied to.

mtah/deadball.py may add our match's second-half block (config.SECOND_HALF_D_BLOCK) to any match
long enough to contain it. On an open game that is a wrong D, so it is switched off here.

Differences under SPELL_NOISE_BAND (0.04 spellF1) are ties. Prints scores and counts only.
"""
import argparse
import contextlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, NamedTuple, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _code_root(argv: Sequence[str]) -> Path:
    """The folder whose mtah/ is benchmarked: --code DIR, else this repository."""
    for i, arg in enumerate(argv):
        if arg == "--code" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if arg.startswith("--code="):
            return Path(arg.split("=", 1)[1]).resolve()
    return REPO


CODE_ROOT = _code_root(sys.argv[1:]) if __name__ == "__main__" else REPO
if not (CODE_ROOT / "mtah" / "__init__.py").is_file():
    sys.exit(f"--code {CODE_ROOT}: no mtah/ package there")
sys.path.insert(0, str(CODE_ROOT))

import mtah  # noqa: E402
from mtah import deadball  # noqa: E402
from mtah import evidence as ev  # noqa: E402
from mtah import metrica_games as mg  # noqa: E402
from mtah.ball import ball_clean  # noqa: E402
from mtah.config import FPS  # noqa: E402
from mtah.deadball import dead_mask  # noqa: E402
from mtah.evidence_fit import MASK_SEED as METRICA_SEED  # noqa: E402  same masks as evidence_fit
from mtah.interfaces import A, D, H, Interval, LabelArray, Tracking  # noqa: E402
from mtah.segment import segment  # noqa: E402
from mtah.touches import touches  # noqa: E402
from mtah.xpolicy import to_x_policy  # noqa: E402


def _official_scorer():
    """evaluation/possession_metrics.py of this repository, imported from exactly that path."""
    path = REPO / "evaluation" / "possession_metrics.py"
    spec = importlib.util.spec_from_file_location("possession_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pm = _official_scorer()

NOISE = pm.SPELL_NOISE_BAND  # 0.04
WINDOW_BEFORE_S = 2.5  # set-piece window: this long before each restart...
WINDOW_AFTER_S = 2.5  # ...and this long after it, about 5 s in all (interfaces.SetPiece)
STRIDE = 5  # the external shape model trains on every 5th frame, as evidence_fit's calibration does
VARIANTS = ("nearest", "shape_self", "shape_external")

# The open data always comes from this repository, even when --code points at another branch.
METRICA_ROOT = REPO / "outputs" / "metrica" / "sample" / "data"
DFL_ROOT = REPO / "outputs" / "dfl" / "raw"
CACHE_DIR = REPO / "outputs" / "cache"


# -- set-piece windows --------------------------------------------------------------------------

def restart_frames(labels: LabelArray) -> np.ndarray:
    """Frames where a dead ball turns live: D on the frame before, H or A on this one."""
    labels = np.asarray(labels)
    live = (labels == H) | (labels == A)
    return np.flatnonzero((labels[:-1] == D) & live[1:]) + 1


def set_piece_windows(labels: LabelArray, before: int = round(WINDOW_BEFORE_S * FPS),
                      after: int = round(WINDOW_AFTER_S * FPS)) -> list[Interval]:
    """One inclusive window per restart, `before` frames before it to `after` frames from it,
    clipped to the match."""
    last = len(labels) - 1
    return [(int(max(0, r - before)), int(min(last, r + after - 1))) for r in restart_frames(labels)]


@contextlib.contextmanager
def without_our_match_constants() -> Iterator[bool]:
    """Switch off deadball's SECOND_HALF_D_BLOCK, if the code tree has one, then restore it.
    Yields whether there was one to switch off."""
    saved = getattr(deadball, "SECOND_HALF_D_BLOCK", None)
    if saved is None:
        yield False
        return
    deadball.SECOND_HALF_D_BLOCK = None
    try:
        yield True
    finally:
        deadball.SECOND_HALF_D_BLOCK = saved


# -- the pipeline -------------------------------------------------------------------------------

def decode(e: np.ndarray, lost: np.ndarray, events: list, windows: Sequence[Interval], n_frames: int) -> LabelArray:
    """run.predict after the evidence: dead ball, segmentation, X policy."""
    dead = dead_mask(n_frames, windows)
    labels = segment(e, lost, dead, events)
    return to_x_policy(labels, confidence=np.abs(e), threshold=1e-9)


def pipeline(tracking: Tracking, windows: Sequence[Interval], e: np.ndarray) -> LabelArray:
    """run.predict with the evidence passed in (tests check the two agree)."""
    _, lost = ball_clean(tracking.ball)
    events = touches(tracking.ball, tracking.players, tracking.team)
    return decode(e, lost, events, windows, tracking.n_frames)


class Prepared(NamedTuple):
    """One open game, ball hidden, reduced to what every variant needs."""

    number: object
    gt: np.ndarray  # scorer codes
    n_frames: int
    lost: np.ndarray
    events: list
    windows: list
    nearest: np.ndarray  # evidence on trusted frames, 0 on lost ones
    shape_self: np.ndarray  # evidence_shape_calibrated with the game's real halves, 0 off lost frames
    halves: tuple
    shape_x: list  # per half, [n, F] shape features (NaN where a team is untracked)
    sign: np.ndarray  # +1 where the true label is H, -1 where A, 0 otherwise


def prepare(game: mg.OpenGame, seed: int) -> Prepared:
    tr = mg.masked_tracking(game, seed, displace=True)
    filled, lost = ball_clean(tr.ball)
    scan = ev.scan_teams(filled, tr.players, tr.team)
    length = game.pitch[0]
    logit, _ = ev.shape_logits(scan, lost, game.halves, length=length)
    shape_x = [ev.shape_features(scan, first, last, ev.attack_direction(scan, first, last), length)
               for first, last in game.halves]
    labels = np.asarray(game.labels)
    return Prepared(
        number=game.number,
        gt=pm.label_codes(labels),
        n_frames=tr.n_frames,
        lost=lost,
        events=touches(tr.ball, tr.players, tr.team),
        windows=set_piece_windows(labels),
        nearest=ev.nearest_evidence(tr.ball, lost, tr.players, tr.team),
        shape_self=ev.logit_to_evidence(ev.SHAPE_TEMPERATURE * logit),
        halves=tuple(game.halves),
        shape_x=shape_x,
        sign=np.where(labels == H, 1.0, np.where(labels == A, -1.0, 0.0)),
    )


# -- the shape model trained on other games ----------------------------------------------------

class ShapeFit(NamedTuple):
    scale: np.ndarray  # [F] RMS of each feature over the training rows
    weights: np.ndarray  # [F] on the scaled features


def _training_rows(p: Prepared) -> tuple[np.ndarray, np.ndarray]:
    """Every STRIDE-th lost frame with a live true label and finite features: (x, sign)."""
    xs, ys = [], []
    for (first, last), x in zip(p.halves, p.shape_x):
        idx = np.arange(0, last - first + 1, STRIDE)
        frames = first + idx
        use = p.lost[frames] & (p.sign[frames] != 0) & np.isfinite(x[idx]).all(axis=1)
        xs.append(x[idx[use]])
        ys.append(p.sign[frames[use]])
    width = p.shape_x[0].shape[1] if p.shape_x else 0
    return (np.concatenate(xs) if xs else np.zeros((0, width))), (np.concatenate(ys) if ys else np.zeros(0))


def fit_external_shape(train: Sequence[Prepared]) -> ShapeFit | None:
    """A logistic model on the training games' lost frames and true labels; None without rows."""
    rows = [_training_rows(p) for p in train]
    if not rows:
        return None
    x = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    if not len(y) or (y > 0).all() or (y < 0).all():
        return None
    scale = np.sqrt(np.mean(x**2, axis=0))
    scale = np.array([float(f"{s:.6g}") if s > 0 else 1.0 for s in scale])
    weights = np.round(ev.fit_logistic(y[:, None] * x / scale, ev.SHAPE_RIDGE), ev.FIT_DECIMALS)
    return ShapeFit(scale, weights)


def external_shape(p: Prepared, fit: ShapeFit | None) -> np.ndarray:
    """Evidence from the externally trained model, on lost frames only; 0 elsewhere."""
    e = np.zeros(p.n_frames)
    if fit is None:
        return e
    for (first, last), x in zip(p.halves, p.shape_x):
        on = p.lost[first : last + 1] & np.isfinite(x).all(axis=1)
        e[first : last + 1][on] = ev.logit_to_evidence((x[on] / fit.scale) @ fit.weights)
    return e


def evidence_variant(name: str, p: Prepared, fit: ShapeFit | None) -> np.ndarray:
    """The two parts never overlap (nearest is 0 on lost frames), so the sum stays in [-1, 1]."""
    if name == "nearest":
        return p.nearest
    if name == "shape_self":
        return p.nearest + p.shape_self
    if name == "shape_external":
        return p.nearest + external_shape(p, fit)
    raise ValueError(f"evidence variants are {VARIANTS}, got {name!r}")


# -- benchmark ----------------------------------------------------------------------------------

def score(p: Prepared, labels: LabelArray) -> dict:
    pred = pm.label_codes(labels)
    result = pm.evaluate(p.gt, pred)
    graded = p.gt != pm.X
    return {"spell": result["spell_f1_mean"], "macro": result["macro_f1"],
            "n_pred": result["n_pred_spells"], "n_gt": result["n_gt_spells"],
            "dead_pred": float((pred[graded] == pm.D).mean()), "dead_gt": float((p.gt[graded] == pm.D).mean())}


def run_benchmark(prepared: Sequence[Prepared]) -> dict:
    """{variant: {game: {"spell", "macro", "n_pred", "n_gt"}}}, every game held out in turn."""
    out = {v: {} for v in VARIANTS}
    with without_our_match_constants():
        for p in prepared:
            fit = fit_external_shape([q for q in prepared if q is not p])
            for variant in VARIANTS:
                e = evidence_variant(variant, p, fit)
                out[variant][p.number] = score(p, decode(e, p.lost, p.events, p.windows, p.n_frames))
    return out


def game_loaders(dataset: str) -> list[tuple[object, Callable[[], mg.OpenGame], int]]:
    """(game id, loader, ball-mask seed) per game, reading the open data of this repository."""
    if dataset == "dfl":
        from mtah import dfl_games

        return [(n, lambda n=n: dfl_games.load_match(n, root=DFL_ROOT, cache_dir=CACHE_DIR), dfl_games.MASK_SEED[n])
                for n in dfl_games.GAMES]
    return [(n, lambda n=n: mg.load_game(n, root=METRICA_ROOT, cache_dir=CACHE_DIR), METRICA_SEED[n])
            for n in mg.GAMES]


def _git_version(root: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        dirty = git("status", "--porcelain", "--untracked-files=no")
        return f"{git('rev-parse', '--abbrev-ref', 'HEAD')} @ {git('rev-parse', '--short', 'HEAD')}" + \
               (" + uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "not a git checkout"


def _name(number: object) -> str:
    return f"game {number}" if isinstance(number, (int, np.integer)) else str(number)


def main(dataset: str = "metrica", mask_offset: int = 0) -> dict:
    t0 = time.perf_counter()
    print(f"code: {Path(mtah.__file__).resolve().parent} ({_git_version(CODE_ROOT)})")
    has_block = getattr(deadball, "SECOND_HALF_D_BLOCK", None) is not None
    print(f"deadball.SECOND_HALF_D_BLOCK: {'present, switched off for open games' if has_block else 'not used'}")

    prepared = []
    for number, load, seed in game_loaders(dataset):
        p = prepare(load(), seed + mask_offset)  # one full game in memory at a time
        alive = (p.gt == pm.H) | (p.gt == pm.A)
        print(f"{_name(number)}: {p.n_frames} frames, live {alive.mean():.3f}, lost on live {p.lost[alive].mean():.3f}, "
              f"restarts {len(p.windows)}, touches {len(p.events)}")
        prepared.append(p)
    results = run_benchmark(prepared)

    print(f"\nwhole-pipeline spellF1 per held-out game: {dataset}, mask offset {mask_offset}, set-piece windows "
          f"{WINDOW_BEFORE_S:g} s before to {WINDOW_AFTER_S:g} s after each restart; {time.perf_counter() - t0:.0f} s")
    games = [p.number for p in prepared]
    print(f"{'evidence':<16}" + "".join(f"{_name(g):>10}" for g in games)
          + f"{'mean':>9}{'macroF1':>9}  dead pred/true   spells pred/gt")
    means = {}
    for variant in VARIANTS:
        r = results[variant]
        means[variant] = float(np.mean([r[g]["spell"] for g in games]))
        macro = float(np.mean([r[g]["macro"] for g in games]))
        dead = f"{np.mean([r[g]['dead_pred'] for g in games]):.3f}/{np.mean([r[g]['dead_gt'] for g in games]):.3f}"
        print(f"{variant:<16}" + "".join(f"{r[g]['spell']:>10.4f}" for g in games) + f"{means[variant]:>9.4f}{macro:>9.4f}"
              + f"{dead:>17}   " + " ".join(f"{r[g]['n_pred']}/{r[g]['n_gt']}" for g in games))

    print()
    for variant in VARIANTS[1:]:
        diff = means[variant] - means["nearest"]
        verdict = "better" if diff >= NOISE else "worse" if diff <= -NOISE else f"a tie (under {NOISE})"
        print(f"{variant} vs nearest: {diff:+.4f}, {verdict}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dfl", action="store_true", help="the seven DFL matches instead of Metrica's three games")
    parser.add_argument("--mask-offset", type=int, default=0, help="added to every game's ball-mask seed")
    parser.add_argument("--code", metavar="DIR", default=str(REPO),
                        help="a folder holding another branch's mtah/ (e.g. a git worktree); default: this repository")
    args = parser.parse_args()
    main("dfl" if args.dfl else "metrica", args.mask_offset)
