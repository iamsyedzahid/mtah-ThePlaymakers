"""Compare a teammate's version against ours, role by role, on synthetic data only.

    python scripts/compare_versions.py path/to/their/checkout
    python scripts/compare_versions.py path/to/their/checkout --seeds 101 102 103 --frames 60000

For each role, their module files replace ours and everything else stays ours, so a row measures that
one role. The last row swaps every role they have. Scored with the official scorer's spellF1 and
macroF1 on synthetic matches, whose seeds were never used to choose any setting. It never touches the
match data or the public ground truth, so a result here is safe to act on. Differences under the
scorer's noise band are ties. Their code runs on this machine, so only point this at people you trust.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _official_scorer():
    """evaluation/possession_metrics.py, imported by its path: nothing added to sys.path can shadow it."""
    spec = importlib.util.spec_from_file_location("possession_metrics", ROOT / "evaluation" / "possession_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pm = _official_scorer()

from mtah.config import N_FRAMES  # noqa: E402
from mtah.synthetic import make_synthetic  # noqa: E402

LOAD_ORDER = ("ball", "evidence", "touches", "segment", "deadball")  # each only needs those before it
ROLES = {
    "R1 ball": ("ball",),
    "R2 evidence": ("evidence",),
    "R3 touches": ("touches",),
    "R4 segment and dead ball": ("segment", "deadball"),
}
SEEDS = (101, 102, 103)


def load_modules(other: Path | None, swapped: tuple[str, ...]) -> dict:
    """Fresh copies of the pipeline modules, with `swapped` taken from `other`."""
    keep = {f"mtah.{name}": sys.modules.get(f"mtah.{name}") for name in LOAD_ORDER}
    modules = {}
    try:
        for name in LOAD_ORDER:
            folder = other if name in swapped else ROOT
            spec = importlib.util.spec_from_file_location(f"mtah.{name}", folder / "mtah" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"mtah.{name}"] = module  # so later modules import this copy
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        for key, value in keep.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return modules


def predict(mods: dict, ours: dict, syn) -> np.ndarray:
    """The pipeline of run.predict. A step their dead-ball module lacks comes from ours."""
    def dead_step(name):
        return getattr(mods["deadball"], name, None) or getattr(ours["deadball"], name)

    tracking = syn.tracking
    filled, lost = mods["ball"].ball_clean(tracking.ball)
    e = mods["evidence"].evidence(tracking)
    events = mods["touches"].touches(tracking.ball, tracking.players, tracking.team)
    tagged = dead_step("dead_intervals")(syn.set_pieces, filled, lost)
    intervals = [*tagged, *dead_step("untagged_intervals")(filled, lost),
                 *dead_step("slow_lost_intervals")(tracking.players, lost)]
    events = [*events, *dead_step("restart_touches")(syn.set_pieces, tagged, tracking.n_frames)]
    dead = dead_step("join_close")(mods["deadball"].dead_mask(tracking.n_frames, intervals))
    return mods["segment"].segment(e, lost, dead, events)


def score(mods: dict, ours: dict, matches: list) -> tuple[float, float]:
    spell, macro = [], []
    for syn in matches:
        gt = pm.label_codes(syn.labels)
        pred = pm.label_codes(predict(mods, ours, syn))
        spell.append(pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"])
        macro.append(pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))))
    return float(np.mean(spell)), float(np.mean(macro))


def verdict(diff: float) -> str:
    if abs(diff) < pm.SPELL_NOISE_BAND:
        return "tie (within noise)"
    return "theirs better" if diff > 0 else "ours better"


def compare(other: Path, seeds=SEEDS, n_frames: int = N_FRAMES - 1) -> list[dict]:
    """One row per role, plus one for all their roles together. A role that fails says why."""
    other = Path(other)
    if not (other / "mtah").is_dir():
        raise SystemExit(f"{other} has no mtah/ folder")
    matches = [make_synthetic(n_frames, seed) for seed in seeds]
    ours = load_modules(None, ())
    base_spell, base_macro = score(ours, ours, matches)

    have = {name: (other / "mtah" / f"{name}.py").is_file() for name in LOAD_ORDER}
    rows = [{"role": "ours (baseline)", "spell": base_spell, "macro": base_macro, "diff": 0.0, "note": ""}]
    every = tuple(name for name in LOAD_ORDER if have[name])
    for role, files in [*ROLES.items(), ("all their roles", every)]:
        swapped = tuple(name for name in files if have[name])
        if not swapped:
            rows.append({"role": role, "note": "they have no such file"})
            continue
        try:
            spell, macro = score(load_modules(other, swapped), ours, matches)
        except Exception as err:  # their code, any failure: report it, carry on with the other roles
            rows.append({"role": role, "note": f"failed: {type(err).__name__}: {err}"})
            continue
        rows.append({"role": role, "spell": spell, "macro": macro, "diff": spell - base_spell,
                     "note": verdict(spell - base_spell)})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("other", type=Path, help="their checkout: the folder holding mtah/")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--frames", type=int, default=N_FRAMES - 1)
    args = parser.parse_args()
    print(f"synthetic only, seeds {args.seeds}, {args.frames} frames, noise band {pm.SPELL_NOISE_BAND}")
    print(f"{'swapped in':<28}{'spellF1':>9}{'macroF1':>9}{'vs ours':>9}  verdict")
    for row in compare(args.other, args.seeds, args.frames):
        if "spell" in row:
            print(f"{row['role']:<28}{row['spell']:>9.4f}{row['macro']:>9.4f}{row['diff']:>+9.4f}  {row['note']}")
        else:
            print(f"{row['role']:<28}{'':>27}  {row['note']}")
