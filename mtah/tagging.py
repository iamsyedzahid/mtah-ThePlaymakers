"""Smart Tagging parser and set-piece window extractor (R3).

Loads Metrica's automatic tactical annotation XML (SportsCode format) and extracts
set-piece intervals as SetPiece NamedTuples.

Conventions:
- In the SportsCode XML, <start> and <end> are seconds from the beginning of tactical video.
  Video frame = round(seconds * 25).
- starter.load_smart_tagging.load() gives end_frame as EXCLUSIVE (the first frame after the phase).
  SetPiece.end is INCLUSIVE (start..end frames), so end = tag['end_frame'] - 1.
- Team tags name clubs (e.g. 'Barcelona', 'Atletic Club'). They are mapped to 0 (home) / 1 (away).
- 'N/A' Team tag (including all throw-ins) maps to None.
"""
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

from mtah.config import FPS, data_dir as default_data_dir
from mtah.interfaces import SetPiece

SET_PIECES_CODE = "set pieces"


def load_tags(
    home_name: str,
    away_name: str,
    data_dir: Path | str | None = None,
    fps: int = FPS,
) -> list[dict]:
    """Load Smart Tagging XML and resolve team IDs and frame bounds.

    Args:
        home_name: Name of home club (case-insensitive match against Team tag).
        away_name: Name of away club (case-insensitive match against Team tag).
        data_dir: Root directory of data bundle (defaults to mtah.config.data_dir()).
        fps: Video frame rate (default 25).

    Returns:
        List of dicts representing all tag instances in the XML with additional fields:
        - 'start_frame_inclusive': int
        - 'end_frame_inclusive': int
        - 'team_id': int | None (0 for home, 1 for away, None for N/A)
    """
    if data_dir is None:
        data_dir = default_data_dir()
    else:
        data_dir = Path(data_dir)

    xml_path = data_dir / "tactical" / "BAR-ATH_tactical_smart_tagging.xml"
    if not xml_path.is_file():
        # Fall back to checking directly in data_dir if not in tactical/ subdirectory
        alt_path = data_dir / "BAR-ATH_tactical_smart_tagging.xml"
        if alt_path.is_file():
            xml_path = alt_path
        else:
            raise FileNotFoundError(f"Smart Tagging XML not found at {xml_path}")

    return parse_tags_from_file(xml_path, home_name, away_name, fps=fps)


def parse_tags_from_file(
    file_path: Path | str,
    home_name: str,
    away_name: str,
    fps: int = FPS,
) -> list[dict]:
    """Parse a SportsCode XML file into tag dicts with inclusive frames and team IDs."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    return parse_tags_from_element(root, home_name, away_name, fps=fps)


def parse_tags_from_element(
    root: ET.Element,
    home_name: str,
    away_name: str,
    fps: int = FPS,
) -> list[dict]:
    """Parse SportsCode XML element tree into tag dicts with inclusive frames and team IDs."""
    home_lower = home_name.strip().lower()
    away_lower = away_name.strip().lower()

    out = []
    for instance in root.iter("instance"):
        start_s_text = instance.findtext("start")
        end_s_text = instance.findtext("end")
        if start_s_text is None or end_s_text is None:
            continue

        start_s = float(start_s_text)
        end_s = float(end_s_text)
        start_frame = int(round(start_s * fps))
        end_frame = int(round(end_s * fps))

        # Collect tags from <label> elements
        tags: dict[str, str] = {}
        for label in instance.findall("label"):
            group = label.findtext("group")
            text = label.findtext("text")
            if group is not None and text is not None:
                tags[group] = text

        team_val = tags.get("Team", "").strip()
        team_lower = team_val.lower()

        if team_lower == home_lower:
            team_id = 0
        elif team_lower == away_lower:
            team_id = 1
        else:
            team_id = None

        code = instance.findtext("code") or ""
        # The XML end_frame is exclusive; SetPiece and inclusive frame ranges use end_frame - 1
        end_frame_inclusive = max(start_frame, end_frame - 1)

        out.append({
            "code": code,
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_frame_inclusive": start_frame,
            "end_frame_inclusive": end_frame_inclusive,
            "team_id": team_id,
            "tags": tags,
        })

    return out


def set_pieces(tags: Sequence[dict]) -> list[SetPiece]:
    """Extract set-piece windows as SetPiece NamedTuples.

    Args:
        tags: Sequence of tag dicts (from load_tags).

    Returns:
        List of SetPiece(start, end, team, kind) sorted ascending by start frame.
    """
    out: list[SetPiece] = []
    for tag in tags:
        code = (tag.get("code") or "").strip().lower()
        if code == SET_PIECES_CODE:
            start = tag["start_frame_inclusive"]
            end = tag["end_frame_inclusive"]
            team = tag["team_id"]
            kind = tag["tags"].get("Type", "Unknown")
            out.append(SetPiece(start=start, end=end, team=team, kind=kind))

    out.sort(key=lambda sp: sp.start)
    return out
