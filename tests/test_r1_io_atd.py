"""R1 loader on fake data shaped like kloppy's output (DATA.md). Never touches the real ATD."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtah import io_atd, teams
from mtah.config import PITCH_M

N = 6


def _fake_teams(home=("1", "10"), away=("2",)):
    def side(ids):
        return SimpleNamespace(players=[SimpleNamespace(player_id=i) for i in ids])

    return [side(home), side(away)]


def _fake_frame(frame_ids=None):
    """kloppy's to_df() layout: frame_id from 1, ball_x/ball_y, <slot>_x/_y/_s, all 0..1."""
    ids = np.arange(1, N + 1) if frame_ids is None else np.asarray(frame_ids)
    df = pd.DataFrame({"frame_id": ids, "ball_x": ids / 10, "ball_y": ids / 20})
    for slot in ("1", "2", "10"):
        df[f"{slot}_x"] = int(slot) / 100 + ids / 1000
        df[f"{slot}_y"] = 0.5
        df[f"{slot}_s"] = 3.0  # speed, ignored
    df.loc[df["frame_id"] == 3, ["ball_x", "ball_y"]] = np.nan
    return df


def test_slot_teams_home_first():
    assert teams.slot_teams(_fake_teams()) == {"1": 0, "10": 0, "2": 1}


def test_slot_teams_rejects_bad_metadata():
    with pytest.raises(ValueError):
        teams.slot_teams(_fake_teams()[:1])
    with pytest.raises(ValueError):
        teams.slot_teams(_fake_teams(home=("1", "2"), away=("2",)))


def test_tracking_from_frame_indexing_units_and_teams():
    shuffled = _fake_frame().sample(frac=1.0, random_state=0)  # row order must not matter
    tr = io_atd.tracking_from_frame(shuffled, teams.slot_teams(_fake_teams()), n_frames=N)
    assert tr.ball.shape == (N, 2) and tr.players.shape == (N, 3, 2)
    # row 0 is frame_id 1, scaled to metres
    np.testing.assert_allclose(tr.ball[0], [0.1 * PITCH_M[0], 0.05 * PITCH_M[1]])
    assert np.isnan(tr.ball[2]).all() and not np.isnan(tr.ball[3]).any()  # frame_id 3 -> row 2
    # slots in numeric order 1, 2, 10, with their teams
    np.testing.assert_array_equal(tr.team, [0, 1, 0])
    np.testing.assert_allclose(tr.players[0, :, 0], np.array([0.011, 0.021, 0.101]) * PITCH_M[0])


@pytest.mark.parametrize(
    "frame_ids",
    [
        [0, 1, 2, 3, 4, 5],  # 0-based: would shift every row
        [1, 2, 3, 4, 5],  # one record missing
        [1, 2, 3, 4, 5, 5],  # duplicate
        [1, 2, 3, 4, 5, 7],  # gap
    ],
)
def test_tracking_from_frame_rejects_bad_frame_ids(frame_ids):
    with pytest.raises(ValueError):
        io_atd.tracking_from_frame(_fake_frame(frame_ids), {"1": 0, "2": 1}, n_frames=N)


def test_tracking_from_frame_rejects_missing_slot_columns():
    with pytest.raises(ValueError, match="no x/y columns"):
        io_atd.tracking_from_frame(_fake_frame(), {"1": 0, "99": 1}, n_frames=N)


def test_cache_round_trip(tmp_path):
    tr = io_atd.tracking_from_frame(_fake_frame(), teams.slot_teams(_fake_teams()), n_frames=N)
    cache = tmp_path / "cache" / "tracking.npz"
    io_atd._save(cache, tr, ("Home FC", "Away Club"))
    back = io_atd.load_tracking(cache)
    np.testing.assert_array_equal(back.ball, tr.ball)
    np.testing.assert_array_equal(back.players, tr.players)
    np.testing.assert_array_equal(back.team, tr.team)
    assert io_atd.team_names(cache) == ("Home FC", "Away Club")


def test_old_cache_without_names_is_flagged(tmp_path):
    tr = io_atd.tracking_from_frame(_fake_frame(), teams.slot_teams(_fake_teams()), n_frames=N)
    cache = tmp_path / "old.npz"
    np.savez_compressed(cache, ball=tr.ball, players=tr.players, team=tr.team)  # no names
    with pytest.raises(RuntimeError, match="predates team names"):
        io_atd.team_names(cache)


def test_summary_is_numbers_only(syn):
    stats = io_atd.summary(syn.tracking)
    assert all(type(v) in (int, float) for v in stats.values())
    assert stats["n_slots"] == 46
