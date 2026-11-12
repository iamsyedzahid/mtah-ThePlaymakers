"""R5 open-games benchmark. Synthetic data only: never reads real match data, open data or ground truth."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import run
from mtah import deadball
from mtah.config import PITCH_M
from mtah.evidence import evidence
from mtah.interfaces import A, D, H, X
from mtah.metrica_games import OpenGame
from mtah.synthetic import make_synthetic

REPO = Path(__file__).resolve().parents[1]


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_open_games", REPO / "scripts" / "benchmark_open_games.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark()


def _synthetic_games(n_games: int = 3, frames: int = 3_000) -> list[OpenGame]:
    games = []
    for seed in range(n_games):
        syn = make_synthetic(frames, seed=seed)
        contacts = np.array(sorted(f for f, _ in syn.receptions), dtype=np.int64)
        halves = ((0, frames // 2 - 1), (frames // 2, frames - 1))
        games.append(OpenGame(seed, syn.tracking, halves, syn.labels, contacts, pitch=PITCH_M))
    return games


@pytest.fixture(scope="module")
def prepared():
    return [bench.prepare(game, seed=int(game.number)) for game in _synthetic_games()]


# -- set-piece windows --------------------------------------------------------------------------

def test_restarts_are_dead_to_live_only():
    # X 0-1, H 2-11, D 12-16, A 17-26, D 27-29, H 30-34: restarts at 17 and 30, not at 2.
    labels = np.array([X] * 2 + [H] * 10 + [D] * 5 + [A] * 10 + [D] * 3 + [H] * 5, dtype="<U1")
    np.testing.assert_array_equal(bench.restart_frames(labels), [17, 30])
    assert bench.set_piece_windows(labels, before=2, after=3) == [(15, 19), (28, 32)]
    assert bench.set_piece_windows(labels, before=20, after=10) == [(0, 26), (10, 34)]  # clipped


def test_no_restart_without_a_live_frame_after():
    labels = np.array([H] * 5 + [D] * 5, dtype="<U1")
    assert bench.set_piece_windows(labels) == []


def test_default_windows_are_about_five_seconds():
    labels = np.array([H] * 500 + [D] * 100 + [A] * 500, dtype="<U1")
    ((start, end),) = bench.set_piece_windows(labels)
    before, after = round(bench.WINDOW_BEFORE_S * 25), round(bench.WINDOW_AFTER_S * 25)
    assert (start, end) == (600 - before, 600 + after - 1)
    assert 4.5 * 25 <= end - start + 1 <= 5.5 * 25


# -- our match's constants never reach an open game ----------------------------------------------

def test_second_half_block_is_switched_off_and_restored(monkeypatch):
    monkeypatch.setattr(deadball, "SECOND_HALF_D_BLOCK", (1, 2), raising=False)
    with bench.without_our_match_constants() as switched:
        assert switched and deadball.SECOND_HALF_D_BLOCK is None
    assert deadball.SECOND_HALF_D_BLOCK == (1, 2)


def test_nothing_to_switch_off(monkeypatch):
    monkeypatch.setattr(deadball, "SECOND_HALF_D_BLOCK", None, raising=False)
    with bench.without_our_match_constants() as switched:
        assert not switched


# -- the pipeline is the submission's ------------------------------------------------------------

def test_pipeline_is_what_run_predicts(syn):
    e = evidence(syn.tracking)
    np.testing.assert_array_equal(bench.pipeline(syn.tracking, syn.set_piece_intervals, e),
                                  run.predict(syn.tracking, syn.set_piece_intervals))


# -- evidence variants ---------------------------------------------------------------------------

def test_external_shape_only_on_lost_frames(prepared):
    fit = bench.fit_external_shape(prepared[1:])  # trained without game 0
    for p, f in ((prepared[0], fit), (prepared[0], None)):
        e = bench.external_shape(p, f)
        assert e.shape == (p.n_frames,) and np.isfinite(e).all()
        assert (np.abs(e) <= 1).all() and (e[~p.lost] == 0).all()
    assert bench.fit_external_shape([]) is None


def test_every_variant_stays_in_range(prepared):
    fit = bench.fit_external_shape(prepared[1:])
    for name in bench.VARIANTS:
        e = bench.evidence_variant(name, prepared[0], fit)
        assert np.isfinite(e).all() and (np.abs(e) <= 1).all()
    with pytest.raises(ValueError):
        bench.evidence_variant("nope", prepared[0], fit)


def test_benchmark_scores_every_variant_and_game_deterministically(prepared):
    results = bench.run_benchmark(prepared)
    assert set(results) == set(bench.VARIANTS)
    for variant, per_game in results.items():
        assert set(per_game) == {0, 1, 2}
        for r in per_game.values():
            assert 0.0 <= r["spell"] <= 1.0 and 0.0 <= r["macro"] <= 1.0
            assert r["n_gt"] > 0 and r["n_pred"] >= 0
    assert bench.run_benchmark(prepared) == results
