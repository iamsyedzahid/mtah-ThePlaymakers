"""Tests for R3 Smart Tagging parser (mtah/tagging.py).

Tests run on synthetic/fake XML elements only. Never reads real match data here.
"""
import xml.etree.ElementTree as ET
import pytest

from mtah.interfaces import SetPiece
from mtah.tagging import parse_tags_from_element, set_pieces

FAKE_SPORTSCODE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<file>
  <ALL_INSTANCES>
    <instance>
      <ID>1</ID>
      <start>10.0</start>
      <end>15.2</end>
      <code>SET PIECES</code>
      <label><group>Team</group><text>Barcelona</text></label>
      <label><group>Half</group><text>1st Half</text></label>
      <label><group>Type</group><text>Corner Kick</text></label>
    </instance>
    <instance>
      <ID>2</ID>
      <start>20.0</start>
      <end>25.0</end>
      <code>SET PIECES</code>
      <label><group>Team</group><text>Atletic Club</text></label>
      <label><group>Half</group><text>1st Half</text></label>
      <label><group>Type</group><text>Free Kick</text></label>
    </instance>
    <instance>
      <ID>3</ID>
      <start>30.0</start>
      <end>34.0</end>
      <code>SET PIECES</code>
      <label><group>Team</group><text>N/A</text></label>
      <label><group>Half</group><text>2nd Half</text></label>
      <label><group>Type</group><text>Throw-in</text></label>
    </instance>
    <instance>
      <ID>4</ID>
      <start>40.0</start>
      <end>48.0</end>
      <code>BUILD UP</code>
      <label><group>Team</group><text>Barcelona</text></label>
    </instance>
    <instance>
      <ID>5</ID>
      <start>5.0</start>
      <end>8.0</end>
      <code>SET PIECES</code>
      <label><group>Team</group><text>barcelona</text></label>
      <label><group>Type</group><text>Goal Kick</text></label>
    </instance>
  </ALL_INSTANCES>
</file>
"""


@pytest.fixture
def fake_tags():
    root = ET.fromstring(FAKE_SPORTSCODE_XML)
    return parse_tags_from_element(root, home_name="Barcelona", away_name="Atletic Club", fps=25)


def test_tag_parsing_counts(fake_tags):
    assert len(fake_tags) == 5


def test_team_mapping(fake_tags):
    # Instance 1: Barcelona -> 0
    assert fake_tags[0]["team_id"] == 0
    # Instance 2: Atletic Club -> 1
    assert fake_tags[1]["team_id"] == 1
    # Instance 3: N/A -> None
    assert fake_tags[2]["team_id"] is None
    # Instance 5: case-insensitive "barcelona" -> 0
    assert fake_tags[4]["team_id"] == 0


def test_frame_bounds_inclusive(fake_tags):
    # Instance 1: start 10.0s -> 250, end 15.2s -> round(380.0) = 380 exclusive, inclusive 379
    tag0 = fake_tags[0]
    assert tag0["start_frame_inclusive"] == 250
    assert tag0["end_frame_inclusive"] == 379
    assert tag0["end_frame"] == 380


def test_set_pieces_filter_and_sorting(fake_tags):
    sps = set_pieces(fake_tags)
    # 4 of 5 instances are SET PIECES (instance 4 is BUILD UP)
    assert len(sps) == 4
    for sp in sps:
        assert isinstance(sp, SetPiece)
        assert sp.start <= sp.end

    # Must be sorted by start frame
    starts = [sp.start for sp in sps]
    assert starts == sorted(starts)


def test_set_pieces_values(fake_tags):
    sps = set_pieces(fake_tags)
    # Earliest is instance 5: start 5.0s (frame 125)
    assert sps[0].start == 125
    assert sps[0].team == 0
    assert sps[0].kind == "Goal Kick"

    # Throw-in must have team=None
    throw_ins = [sp for sp in sps if sp.kind == "Throw-in"]
    assert len(throw_ins) == 1
    assert throw_ins[0].team is None
