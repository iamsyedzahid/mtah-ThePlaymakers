"""The submission pipeline (mtah/fusion.py). Synthetic data only."""
import numpy as np
import pytest

from mtah import fusion
from mtah.config import PITCH_M
from mtah.interfaces import A, D, H
from mtah.synthetic import make_synthetic

pytest.importorskip("sklearn")


def _codes(labels):
    return np.select([labels == H, labels == A, labels == D], [0, 1, 2], 3).astype(np.int8)


def _game(n, seed):
    syn = make_synthetic(n, seed)
    tr = syn.tracking
    return dict(ball=tr.ball, players=tr.players, team=tr.team, pitch=PITCH_M,
                halves=[(0, n // 2), (n // 2, n)], codes=_codes(syn.labels))


@pytest.fixture(scope="module")
def model():
    return fusion.train([_game(8_000, 11), _game(8_000, 12)])


def test_codes_valid_and_x_only_outside_the_halves(syn):
    tr = syn.tracking
    halves = [(100, 1400), (1600, 2900)]
    codes = fusion.label(tr.ball, tr.players, tr.team, PITCH_M, halves)
    assert codes.shape == (tr.n_frames,) and set(np.unique(codes)) <= {0, 1, 2, 3}
    inside = np.zeros(tr.n_frames, dtype=bool)
    for a, b in halves:
        inside[a:b] = True
    assert (codes[np.logical_not(inside)] == 3).all() and (codes[inside] != 3).all()


def test_known_dead_frames_are_dead(syn):
    tr = syn.tracking
    known = np.zeros(tr.n_frames, dtype=bool)
    known[500:700] = True
    codes = fusion.label(tr.ball, tr.players, tr.team, PITCH_M, [(0, tr.n_frames)], known_dead=known)
    assert (codes[500:700] == 2).all()


def test_swapping_the_teams_swaps_the_labels():
    # 30,000 frames: on a few thousand R2's shape model has too little to fit and returns 0, which
    # leaves exact H/A ties on lost frames that the Viterbi breaks towards H in both orientations.
    tr = make_synthetic(30_000, seed=0).tracking
    straight = fusion.label(tr.ball, tr.players, tr.team, PITCH_M, [(0, tr.n_frames)])
    swapped = fusion.label(tr.ball, tr.players, 1 - tr.team, PITCH_M, [(0, tr.n_frames)])
    assert np.mean(fusion.swap_codes(straight) == swapped) > 0.99


def test_rules_follow_synthetic_possession():
    g = _game(20_000, 3)
    codes = fusion.label(g["ball"], g["players"], g["team"], g["pitch"], g["halves"])
    named = (g["codes"] < 2) & (codes < 2)  # synthetic players are slow, so some live frames read as dead
    assert np.mean(codes[named] == g["codes"][named]) > 0.75


def test_learned_layer_is_deterministic_and_team_symmetric(model):
    g = _game(4_000, 5)
    again = fusion.train([_game(8_000, 11), _game(8_000, 12)])
    def logp(m, team):
        f = fusion.frame_features(g["ball"], g["players"], team, g["pitch"])
        cols = fusion.team_columns(g["ball"], g["players"], team, g["pitch"], g["halves"])
        return fusion.learned_logp(m, f, g["players"], team, g["pitch"], g["halves"], cols)

    p1 = logp(model, g["team"])
    np.testing.assert_array_equal(p1, logp(again, g["team"]))
    np.testing.assert_allclose(logp(model, 1 - g["team"])[:, [1, 0, 2]], p1, atol=1e-9)


def test_teammate_columns_change_sign_with_the_teams():
    g = _game(4_000, 6)
    cols = fusion.team_columns(g["ball"], g["players"], g["team"], g["pitch"], g["halves"])
    swapped = fusion.team_columns(g["ball"], g["players"], 1 - g["team"], g["pitch"], g["halves"])
    assert cols.shape == (4_000, len(fusion.TEAM_COLUMNS)) and np.isfinite(cols).all()
    np.testing.assert_allclose(swapped, cols * fusion.TEAM_COLUMN_SIGNS, atol=1e-9)


def test_learned_features_have_no_missing_values(syn):
    tr = syn.tracking
    halves = [(0, tr.n_frames)]
    f = fusion.frame_features(tr.ball, tr.players, tr.team, PITCH_M)
    cols = fusion.team_columns(tr.ball, tr.players, tr.team, PITCH_M, halves)
    feats, names = fusion.learned_features(f, tr.players, tr.team, PITCH_M, halves, cols)
    assert np.isfinite(feats).all() and feats.shape == (tr.n_frames, len(names))
    assert names[-len(fusion.TEAM_COLUMNS):] == list(fusion.TEAM_COLUMNS)


def test_viterbi_does_not_flicker():
    score = np.zeros((200, 3))
    score[:, 0] = 1.0
    score[100:103, 1] = 3.0  # three frames of strong away evidence: not worth a switch
    assert (fusion.viterbi(score) == 0).all()
    score[100:150, 1] = 3.0  # two seconds of it: a switch
    assert (fusion.viterbi(score)[110:140] == 1).all()
