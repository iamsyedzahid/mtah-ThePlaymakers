"""The DFL open-data reader (mtah/dfl.py). Small fake XML files in the DFL format; never the real ones."""
import numpy as np
import pytest

from mtah import dfl
from mtah.interfaces import A, D, H

MATCH = "TEST01"


def _write_match(folder, n_first=50, n_second=40):
    info = (
        '<PutDataRequest><MatchInformation><Environment PitchX="105.00" PitchY="68.00"/><Teams>'
        '<Team TeamId="T1" Role="home"><Players><Player PersonId="P1"/><Player PersonId="P3"/></Players></Team>'
        '<Team TeamId="T2" Role="guest"><Players><Player PersonId="P2"/></Players></Team>'
        "</Teams></MatchInformation></PutDataRequest>"
    )
    (folder / f"DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-{MATCH}.xml").write_text(info, encoding="utf-8")

    def frameset(section, team, person, first, n, ball=False):
        frames = []
        for k in range(n):
            extra = ""
            if ball:
                status = 0 if 10 <= k < 20 else 1
                owner = 1 if k < 30 else 2
                extra = f' BallStatus="{status}" BallPossession="{owner}"'
            frames.append(f'<Frame N="{first + k}" X="{k * 0.1:.2f}" Y="-1.00"{extra}/>')
        return f'<FrameSet GameSection="{section}" TeamId="{team}" PersonId="{person}">{"".join(frames)}</FrameSet>'

    body = "".join(
        frameset(section, team, person, first, n, ball=team == "BALL")
        for section, first, n in (("firstHalf", 10000, n_first), ("secondHalf", 100000, n_second))
        for team, person in (("T1", "P1"), ("T2", "P2"), ("T1", "P3"), ("referee", "R1"), ("BALL", "B1"))
    )
    positions = f"<PutDataRequest><Positions>{body}</Positions></PutDataRequest>"
    (folder / f"DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-{MATCH}.xml").write_text(positions, encoding="utf-8")


def test_reader_gives_halves_labels_and_corner_coordinates(tmp_path):
    _write_match(tmp_path)
    gm = dfl.load_game(tmp_path, MATCH, cache=tmp_path / "cache.npz")
    assert gm["halves"] == [(0, 50), (50, 90)]
    assert gm["players"].shape == (90, 3, 2)  # the referee is left out
    np.testing.assert_array_equal(gm["team"], [0, 1, 0])  # P1 home, P2 away, P3 home
    labels = gm["labels"]
    assert (labels[:10] == H).all() and (labels[10:20] == D).all() and (labels[20:30] == H).all()
    assert (labels[30:50] == A).all() and (labels[50:60] == H).all()
    np.testing.assert_allclose(gm["ball"][0], [52.5, 33.0])  # DFL measures from the centre spot
    np.testing.assert_array_equal(gm["codes"][10:20], 2)


def test_the_cache_gives_the_same_game(tmp_path):
    _write_match(tmp_path)
    first = dfl.load_game(tmp_path, MATCH, cache=tmp_path / "cache.npz")
    again = dfl.load_game(tmp_path, MATCH, cache=tmp_path / "cache.npz")
    np.testing.assert_array_equal(first["ball"], again["ball"])
    np.testing.assert_array_equal(first["players"], again["players"])
    np.testing.assert_array_equal(first["labels"], again["labels"])
    assert first["halves"] == again["halves"] and first["pitch"] == again["pitch"]


def test_degrade_loses_the_ball_like_our_feed_and_is_seeded(tmp_path):
    _write_match(tmp_path, n_first=20_000, n_second=20_000)
    gm = dfl.load_game(tmp_path, MATCH, cache=tmp_path / "cache.npz")
    one, two = dfl.degrade(gm, 3), dfl.degrade(gm, 3)
    np.testing.assert_array_equal(one["ball"], two["ball"])
    lost = np.isnan(one["ball"][:, 0]).mean()
    assert 0.2 < lost < 0.6
    assert np.isfinite(gm["ball"]).all()  # the original is untouched


def test_files_that_differ_from_the_pinned_ones_are_refused(tmp_path):
    for name in dfl.SHA256:
        (tmp_path / name).write_text("not the real file", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the pinned"):
        dfl.verify(tmp_path)


def test_a_folder_without_positions_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="DFL positions"):
        dfl.folder(tmp_path)
