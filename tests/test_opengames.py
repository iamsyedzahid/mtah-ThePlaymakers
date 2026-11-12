"""The open-games loader (mtah/opengames.py). Fake events and folders only; never the real files."""
import numpy as np
import pytest

from mtah import opengames
from mtah.interfaces import A, D, H, X


def test_labels_follow_the_on_ball_team_and_dead_spells():
    events = [
        (10, 10, "SET PIECE", "KICK OFF", 0, 1),
        (10, 20, "PASS", "", 0, 1),
        (40, 40, "RECOVERY", "INTERCEPTION", 1, 1),
        (60, 60, "BALL OUT", "", 1, 1),
        (80, 80, "SET PIECE", "THROW IN", 0, 1),
        (90, 95, "SHOT", "ON TARGET-GOAL", 0, 1),
        (120, 120, "SET PIECE", "KICK OFF", 1, 1),
    ]
    labels, halves = opengames.labels_from_events(events, 200)
    assert halves == [(10, 121)]
    assert (labels[:10] == X).all() and (labels[121:] == X).all()
    assert (labels[10:40] == H).all() and (labels[40:60] == A).all()
    assert (labels[60:80] == D).all() and (labels[80:95] == H).all()
    assert (labels[95:120] == D).all() and labels[120] == A


def test_two_periods_are_two_halves():
    events = [(5, 5, "PASS", "", 0, 1), (50, 50, "PASS", "", 1, 1), (100, 100, "PASS", "", 1, 2), (150, 150, "PASS", "", 0, 2)]
    _, halves = opengames.labels_from_events(events, 200)
    assert halves == [(5, 51), (100, 151)]


def test_a_missing_folder_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="Sample_Game_1"):
        opengames.folder(tmp_path)


def test_files_that_differ_from_the_pinned_ones_are_refused(tmp_path):
    for name in opengames.SHA256:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not the real file", encoding="utf-8")
    assert opengames.folder(tmp_path) == tmp_path
    with pytest.raises(ValueError, match="differs from the pinned"):
        opengames.verify(tmp_path)


def test_every_pinned_file_has_a_sha256():
    assert len(opengames.SHA256) == 9
    assert all(len(v) == 64 and int(v, 16) >= 0 for v in opengames.SHA256.values())
    assert np.unique(list(opengames.SHA256.values())).size == 9
