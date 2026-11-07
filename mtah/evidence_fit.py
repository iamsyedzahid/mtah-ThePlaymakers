"""Calibration and choice of the evidence (R2), on Metrica's open games only.

    python -m mtah.evidence_fit            # leave-one-game-out table, then the calibration to ship

Never touches our match: it reads outputs/metrica/sample through mtah/metrica_games.py, where the
labels come from Metrica's event data and the ball is hidden the way our match loses it.

The comparison isolates the evidence. Every candidate goes through the same decoder: dead ball is
taken from the event labels (so every candidate gets the same D), and inside each live stretch a
two-state Viterbi chooses H or A to maximise sum(s_t * e_t) - cost * (number of switches). The
switch cost, and for candidates with shape the weight given to shape, are chosen on the two
training games; the held-out game is then scored with evaluation/possession_metrics.py (imported,
never edited). Calibration is fitted on the training games too, so nothing is fitted on the game
it is scored on.

Candidates, simplest first: nearest (the nearest player's team, weighted by distance: the
evidence() placeholder), nearest + Platt scaling, reach, reach + Platt scaling, reach + context
calibration, and each of those with shape added on lost frames. A candidate replaces a simpler
one only if its mean held-out spellF1 is higher by at least NOISE.

    python -m mtah.evidence_fit --mask-offset 10    # the same with other ball masks (robustness)
"""
import argparse
import importlib.util
import time
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

from mtah import evidence as ev
from mtah import metrica_games as mg
from mtah.ball import ball_clean


def _official_scorer():
    """evaluation/possession_metrics.py, imported as is from exactly that path: no sys.path change,
    so nothing else in evaluation/ (or anywhere) can stand in for it."""
    path = Path(__file__).resolve().parents[1] / "evaluation" / "possession_metrics.py"
    spec = importlib.util.spec_from_file_location("possession_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pm = _official_scorer()

NOISE = pm.SPELL_NOISE_BAND  # 0.04: smaller differences are ties
COSTS = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)  # switch cost, in frames of full evidence
SHAPE_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
STRIDE = 5  # calibration uses every 5th frame: neighbours 40 ms apart carry the same information
CALIBRATION_RIDGE = 1.0
BASES = ("nearest", "nearest_platt", "reach", "reach_platt", "reach_context")  # simplest first
CANDIDATES = BASES + tuple(f"{b}+shape" for b in BASES)
MASK_SEED = {1: 1, 2: 2, 3: 3}  # one fixed ball mask per game


class Prepared(NamedTuple):
    """One game, ball hidden, reduced to what the candidates and the decoder need."""

    number: int
    gt: np.ndarray  # scorer codes, H 0, A 1, D 2, X 3
    runs: list  # pm.valid_runs(gt)
    lost: np.ndarray
    lead: np.ndarray
    best: np.ndarray
    speed: np.ndarray
    nearest: np.ndarray  # evidence(): the nearest player's team, weighted by distance
    shape_logit: np.ndarray


class Calibration(NamedTuple):
    nearest_platt: float  # slope on the nearest-player logit
    reach_platt: tuple[float, float, float]
    reach_context: tuple[float, float, float]
    shape_k: float


def prepare(game: mg.OpenGame, seed: int) -> Prepared:
    tr = mg.masked_tracking(game, seed)
    filled, lost = ball_clean(tr.ball)
    scan = ev.scan_teams(filled, tr.players, tr.team)
    lead, best, speed = ev.reach_inputs(scan, lost)
    shape_logit, _ = ev.shape_logits(scan, lost, game.halves)
    gt = pm.label_codes(game.labels)
    nearest = ev.nearest_evidence(tr.ball, lost, tr.players, tr.team)  # what evidence() computes
    return Prepared(game.number, gt, pm.valid_runs(gt), lost, lead, best, speed, nearest, shape_logit)


# -- calibration ----------------------------------------------------------------------------------

def _alive_sign(d: Prepared) -> np.ndarray:
    return np.where(d.gt == pm.H, 1.0, np.where(d.gt == pm.A, -1.0, 0.0))


def fit_calibration(data: Sequence[Prepared]) -> Calibration:
    """Reach on trusted live frames, shape on lost live frames, pooled over `data`, every STRIDE-th
    frame. Rounded, so the shipped constants are exactly what was evaluated."""
    zn, z1, z3, zs = [], [], [], []
    for d in data:
        y = _alive_sign(d)
        idx = np.arange(0, len(y), STRIDE)
        near = idx[(y[idx] != 0) & ~d.lost[idx] & (d.nearest[idx] != 0)]
        zn.append(y[near, None] * ev.evidence_to_logit(d.nearest[near])[:, None])
        reach = idx[(y[idx] != 0) & ~d.lost[idx] & (d.lead[idx] != 0)]
        lead = d.lead[reach]
        z1.append(y[reach, None] * lead[:, None])
        z3.append(y[reach, None] * np.column_stack([lead, lead * d.best[reach], lead * d.speed[reach]]))
        shape = idx[(y[idx] != 0) & d.lost[idx] & (d.shape_logit[idx] != 0)]
        zs.append(y[shape, None] * d.shape_logit[shape, None])

    def fit(z):
        return tuple(float(v) for v in np.round(ev.fit_logistic(np.concatenate(z), CALIBRATION_RIDGE), ev.FIT_DECIMALS))

    (a,) = fit(z1)
    k = fit(zs)[0] if sum(len(z) for z in zs) else 1.0
    return Calibration(fit(zn)[0], (a, 0.0, 0.0), fit(z3), k)


def candidate_evidence(name: str, d: Prepared, cal: Calibration | None, shape_weight: float = 1.0) -> np.ndarray:
    base, _, extra = name.partition("+")
    if base == "nearest":
        e = d.nearest
    elif base == "nearest_platt":
        e = ev.logit_to_evidence(cal.nearest_platt * ev.evidence_to_logit(d.nearest))
    elif base == "reach":
        e = ev.logit_to_evidence(d.lead / ev.CONTROL_SCALE_S)
    else:
        e = ev.logit_to_evidence(ev.calibrated_logit(d.lead, d.best, d.speed, getattr(cal, base)))
    if extra == "shape":
        k = cal.shape_k if cal is not None else 1.0
        e = e + shape_weight * ev.logit_to_evidence(k * d.shape_logit)  # disjoint: base is 0 on lost frames
    return e


def needs_fold(name: str) -> bool:
    """Whether a candidate depends on what it was fitted on (so is recomputed per fold)."""
    return name.split("+")[0] not in ("nearest", "reach") or name.endswith("+shape")


# -- decoder and score --------------------------------------------------------------------------

def decode(e: np.ndarray, gt: np.ndarray, runs: Sequence[tuple[int, int]], cost: float) -> np.ndarray:
    """Scorer codes: D where the ground truth is D, X outside its runs, and inside every live
    stretch the H/A path maximising sum(s_t * e_t) - cost * switches."""
    pred = np.full(len(gt), pm.X, dtype=np.int8)
    for start, stop in runs:
        alive = gt[start:stop] != pm.D
        pred[start:stop][~alive] = pm.D
        edges = np.diff(np.r_[0, alive.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            pred[start + a : start + b] = viterbi(e[start + a : start + b], cost)
    return pred


def viterbi(e: np.ndarray, cost: float) -> np.ndarray:
    """Two states, +1 (H) and -1 (A): maximise sum(s_t * e_t) - cost * switches, exactly.

    With d = V_H - V_A the forward pass is d_t = clip(d_{t-1}, -cost, cost) + 2 e_t, and the best
    predecessor of H is H iff d_{t-1} >= -cost, of A is A iff d_{t-1} <= cost. Ties go to H.
    """
    n = len(e)
    twice = (2.0 * np.asarray(e, dtype=float)).tolist()
    d = [0.0] * n
    prev = d[0] = twice[0]
    for t in range(1, n):
        prev = (cost if prev > cost else -cost if prev < -cost else prev) + twice[t]
        d[t] = prev
    dprev = np.array(d[:-1])
    stay_h, stay_a = (dprev >= -cost).tolist(), (dprev <= cost).tolist()
    path = np.empty(n, dtype=np.int8)
    home = d[-1] >= 0
    path[-1] = pm.H if home else pm.A
    for t in range(n - 1, 0, -1):
        home = stay_h[t - 1] if home else not stay_a[t - 1]
        path[t - 1] = pm.H if home else pm.A
    return path


def spell_f1(d: Prepared, pred: np.ndarray) -> float:
    return float(pm.spell_scores(d.gt, pred, d.runs)["mean_f1"])


# -- leave one game out -----------------------------------------------------------------------

def grid(name: str, d: Prepared, cal: Calibration | None) -> dict[tuple[float, float], float]:
    """spellF1 of one candidate on one game for every (cost, shape weight)."""
    out = {}
    for w in SHAPE_WEIGHTS if name.endswith("+shape") else (1.0,):
        e = candidate_evidence(name, d, cal, w)  # depends on the weight only, not on the cost
        out.update({(c, w): spell_f1(d, decode(e, d.gt, d.runs, c)) for c in COSTS})
    return out


def leave_one_game_out(data: Sequence[Prepared], log=print) -> dict:
    """{candidate: {"held_out": {game: spellF1}, "chosen": {game: (cost, weight)}, "macro": {...}}}."""
    fixed = {name: {d.number: grid(name, d, None) for d in data} for name in CANDIDATES if not needs_fold(name)}
    out = {name: {"held_out": {}, "chosen": {}, "macro": {}, "n_pred": {}} for name in CANDIDATES}
    for test in data:
        train = [d for d in data if d is not test]
        cal = fit_calibration(train)
        log(f"fold: hold out game {test.number}; calibration fitted on games {[d.number for d in train]}: {cal}")
        for name in CANDIDATES:
            scores = fixed[name] if name in fixed else {d.number: grid(name, d, cal) for d in data}
            best = max(scores[train[0].number], key=lambda key: (np.mean([scores[d.number][key] for d in train]), -key[0]))
            out[name]["held_out"][test.number] = scores[test.number][best]
            out[name]["chosen"][test.number] = best
            result = pm.evaluate(test.gt, decode(candidate_evidence(name, test, cal, best[1]), test.gt, test.runs, best[0]))
            out[name]["macro"][test.number] = result["macro_f1"]
            out[name]["n_pred"][test.number] = (result["n_pred_spells"], result["n_gt_spells"])
    return out


def choose(results: dict, names: Sequence[str]) -> str:
    """The simplest of `names` (they are listed simplest first) unless a later one beats the
    current choice by at least NOISE on mean held-out spellF1."""
    chosen = names[0]
    for name in names[1:]:
        if np.mean(list(results[name]["held_out"].values())) >= np.mean(list(results[chosen]["held_out"].values())) + NOISE:
            chosen = name
    return chosen


def main(mask_offset: int = 0) -> None:
    t0 = time.perf_counter()
    data = []
    for number in mg.GAMES:
        game = mg.load_game(number)
        data.append(prepare(game, MASK_SEED[number] + mask_offset))
        d = data[-1]
        alive = (d.gt == pm.H) | (d.gt == pm.A)
        print(f"game {number}: {len(d.gt)} frames, live {alive.mean():.3f}, lost on live frames {d.lost[alive].mean():.3f}")
    results = leave_one_game_out(data)

    print(f"\nleave-one-game-out spellF1 (held-out game), ties under {NOISE}, mask offset {mask_offset}; "
          f"{time.perf_counter() - t0:.0f} s")
    print(f"{'candidate':<20}" + "".join(f"{'game ' + str(d.number):>9}" for d in data) + f"{'mean':>9}{'macroF1':>9}   chosen (cost, shape weight) | spells pred/gt")
    for name in CANDIDATES:
        r = results[name]
        held = [r["held_out"][d.number] for d in data]
        print(f"{name:<20}" + "".join(f"{v:>9.4f}" for v in held) + f"{np.mean(held):>9.4f}{np.mean(list(r['macro'].values())):>9.4f}   "
              + " ".join(f"{r['chosen'][d.number]}" for d in data) + " | " + " ".join(f"{p}/{g}" for p, g in r["n_pred"].values()))

    evidence_choice = choose(results, BASES)
    shape_choice = choose(results, (evidence_choice, f"{evidence_choice}+shape"))
    print(f"\nevidence(): {evidence_choice}  |  adding evidence_shape on lost frames: "
          f"{'yes' if shape_choice != evidence_choice else 'no (a tie or worse)'}")
    final = fit_calibration(data)
    print("calibration on all three games, to ship:")
    print(f"    NEAREST_CALIBRATION = {final.nearest_platt}")
    print(f"    REACH_CALIBRATION = {final.reach_context if evidence_choice == 'reach_context' else final.reach_platt}")
    print(f"    SHAPE_TEMPERATURE = {final.shape_k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask-offset", type=int, default=0, help="added to every game's ball-mask seed")
    main(parser.parse_args().mask_offset)
