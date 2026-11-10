"""Tune segment()'s constants on other games, never on ours (RULES.md allows other games).

    python tune_open.py            # Metrica's three open games (outputs/metrica, see TEAM_PLAN.md)
    python tune_open.py --dfl      # the seven DFL matches (outputs/dfl/raw)

Per game: ball hidden and displaced like ours, evidence = nearest + shape on lost frames (what
evidence() should return), touches from mtah.touches, dead = the game's own D labels. segment() is
run for every (no-touch cost, reception cost, reception window); the minimum run length is applied
afterwards, since it only post-processes the decoded labels. Scored with the official scorer.
"""
import argparse
import itertools
import time
from typing import NamedTuple

import numpy as np

import mtah.segment as seg
from mtah import evidence as ev
from mtah import evidence_fit as ef
from mtah import metrica_games as mg
from mtah.ball import ball_clean
from mtah.config import FPS
from mtah.touches import touches

pm = ef.pm  # evaluation/possession_metrics.py, imported as is
NOISE = pm.SPELL_NOISE_BAND

COST_ELSE_X = (1, 2, 4)        # switch cost with no reception nearby, in multiples of FPS
COST_AT = (2.0, 6.0, 12.0)     # switch cost near a reception
TOLERANCE = (3, 8)             # frames either side of a reception
MIN_RUN_S = (1.0, 2.0, 3.0)    # seconds; shorter H/A runs are absorbed
CONFIGS = [(x, c, t, r) for x, c, t, r in itertools.product(COST_ELSE_X, COST_AT, TOLERANCE, MIN_RUN_S)]
DEFAULT = (2, 2.0, 3, 1.0)     # the constants segment.py started from


class Prep(NamedTuple):
    number: object
    gt: np.ndarray      # scorer codes
    runs: list
    e: np.ndarray
    lost: np.ndarray
    dead: np.ndarray
    events: list


def prepare(game, seed, displace):
    tr = mg.masked_tracking(game, seed, displace=displace)
    filled, lost = ball_clean(tr.ball)
    scan = ev.scan_teams(filled, tr.players, tr.team)
    shape_logit, _ = ev.shape_logits(scan, lost, game.halves, length=game.pitch[0])
    nearest = ev.nearest_evidence(tr.ball, lost, tr.players, tr.team)
    e = nearest + ev.logit_to_evidence(ev.SHAPE_TEMPERATURE * shape_logit)  # disjoint supports
    gt = pm.label_codes(game.labels)
    return Prep(game.number, gt, pm.valid_runs(gt), e, lost, gt == pm.D,
                touches(tr.ball, tr.players, tr.team))


def score(p, labels):
    codes = pm.label_codes(labels)
    pred = np.full(len(p.gt), pm.X, dtype=np.int8)
    for a, b in p.runs:
        pred[a:b] = codes[a:b]
    spell = float(pm.spell_scores(p.gt, pred, p.runs)["mean_f1"])
    res = pm.evaluate(p.gt, pred)
    return spell, res["macro_f1"], res["n_pred_spells"], res["n_gt_spells"]


def main(dataset, mask_offset, displace, top):
    t0 = time.perf_counter()
    numbers, load, seeds = ef.load_games(dataset)
    preps = []
    for n in numbers:
        try:
            game = load(n)
        except FileNotFoundError as err:
            raise SystemExit(f"{err}\nThe open-game data is missing; see TEAM_PLAN.md for where it goes.")
        preps.append(prepare(game, seeds[n] + mask_offset, displace))
        p = preps[-1]
        alive = (p.gt == pm.H) | (p.gt == pm.A)
        print(f"{n}: {len(p.gt)} frames, live {alive.mean():.3f}, lost on live frames {p.lost[alive].mean():.3f}")

    results = {c: {} for c in CONFIGS}
    for x, cost_at, tol in itertools.product(COST_ELSE_X, COST_AT, TOLERANCE):
        seg.SWITCH_COST_ELSEWHERE, seg.SWITCH_COST_AT_RECEPTION, seg.RECEPTION_TOL = x * FPS, cost_at, tol
        for p in preps:
            seg.MIN_RUN_FRAMES = 0  # no merging yet: the raw decode
            raw = seg.segment(p.e, p.lost, p.dead, p.events)
            for run_s in MIN_RUN_S:
                seg.MIN_RUN_FRAMES = int(run_s * FPS)
                results[(x, cost_at, tol, run_s)][p.number] = score(p, seg._merge_short_runs(raw))
        print(f"  done else={x}*FPS at={cost_at} tol={tol}  ({time.perf_counter() - t0:.0f} s)")

    def mean_spell(c, games=numbers):
        return float(np.mean([results[c][n][0] for n in games]))

    ranked = sorted(CONFIGS, key=mean_spell, reverse=True)
    best = mean_spell(ranked[0])
    print(f"\n{dataset}, mask offset {mask_offset}; spellF1 per game, noise band {NOISE}")
    print(f"{'else':>5}{'at':>6}{'tol':>5}{'minrun':>8}" + "".join(f"{str(n):>10}" for n in numbers)
          + f"{'mean':>9}{'macro':>8}  spells pred/gt")
    shown = ranked[:top] + ([DEFAULT] if DEFAULT not in ranked[:top] else [])
    for c in shown:
        r = results[c]
        tag = "  <- started from" if c == DEFAULT else ""
        print(f"{c[0]:>4}x{c[1]:>6g}{c[2]:>5}{c[3]:>7g}s" + "".join(f"{r[n][0]:>10.4f}" for n in numbers)
              + f"{mean_spell(c):>9.4f}{np.mean([r[n][1] for n in numbers]):>8.4f}  "
              + " ".join(f"{r[n][2]}/{r[n][3]}" for n in numbers) + tag)
    print(f"\n{sum(mean_spell(c) >= best - NOISE for c in CONFIGS)} of {len(CONFIGS)} configs are within {NOISE} of the best")

    if len(numbers) > 1:
        held = []
        for test in numbers:
            train = [n for n in numbers if n != test]
            c = max(CONFIGS, key=lambda cfg: mean_spell(cfg, train))
            held.append((test, c, results[c][test][0], results[DEFAULT][test][0]))
            print(f"hold out {test}: chosen on the rest {c} -> {held[-1][2]:.4f} (started-from constants: {held[-1][3]:.4f})")
        print(f"leave-one-game-out mean: chosen {np.mean([h[2] for h in held]):.4f}, "
              f"started-from {np.mean([h[3] for h in held]):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dfl", action="store_true", help="the seven DFL matches instead of Metrica's three")
    parser.add_argument("--mask-offset", type=int, default=0, help="added to every game's ball-mask seed")
    parser.add_argument("--as-given", action="store_true", help="keep the ball on its carrier (no displacement)")
    parser.add_argument("--top", type=int, default=12, help="rows to print")
    a = parser.parse_args()
    main("dfl" if a.dfl else "metrica", a.mask_offset, not a.as_given, a.top)