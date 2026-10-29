"""R3 Smart Tagging on a tiny fake file in the official format (no real data)."""
import pytest

from mtah import tagging
from mtah.interfaces import SetPiece

# The structure the official starter/load_smart_tagging.py reads: <instance> with <start>, <end>,
# <code> and <label><group/><text/></label> pairs. Names and values here are made up.
FAKE = """<?xml version="1.0" encoding="utf-8"?>
<file><ALL_INSTANCES>
<instance><ID>1</ID><start>447.4</start><end>452.4</end><code>SET PIECES</code>
  <label><group>Team</group><text>Home FC</text></label>
  <label><group>Type</group><text>Kick Off (Start)</text></label>
  <label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>2</ID><start>500.0</start><end>505.0</end><code>Set Pieces</code>
  <label><group>Team</group><text>N/A</text></label>
  <label><group>Type</group><text>Throw-in</text></label></instance>
<instance><ID>3</ID><start>510.0</start><end>530.0</end><code>BUILD UP</code>
  <label><group>Team</group><text>away club</text></label></instance>
</ALL_INSTANCES></file>
"""


@pytest.fixture
def bundle(tmp_path):
    folder = tmp_path / "tactical"
    folder.mkdir()
    (folder / "BAR-ATH_tactical_smart_tagging.xml").write_text(FAKE, encoding="utf-8")
    return str(tmp_path)


def test_frames_are_rounded_and_inclusive(bundle):
    first = tagging.load_tags("Home FC", "Away Club", data_dir=bundle)[0]
    assert first.start == round(447.4 * 25)  # 11185
    assert first.end == round(452.4 * 25) - 1  # official end_frame is exclusive


def test_teams_map_case_insensitively_and_na_is_none(bundle):
    tags = tagging.load_tags("Home FC", "Away Club", data_dir=bundle)
    assert [t.team for t in tags] == [0, None, 1]


def test_unknown_club_raises(bundle):
    with pytest.raises(ValueError, match="neither"):
        tagging.load_tags("Somebody", "Else", data_dir=bundle)


def test_set_pieces_are_windows_with_team_or_none(bundle):
    pieces = tagging.set_pieces(tagging.load_tags("Home FC", "Away Club", data_dir=bundle))
    assert pieces == [SetPiece(11185, 11309, 0, "Kick Off (Start)"), SetPiece(12500, 12624, None, "Throw-in")]


def test_summary_counts_only(bundle):
    counts = tagging.summary(tagging.load_tags("Home FC", "Away Club", data_dir=bundle))
    assert counts["instances"] == 3 and counts["set_pieces"] == 2 and counts["set_pieces_team_none"] == 1
    assert all(type(v) is int for v in counts.values())
