"""Smart Tagging (R3): Metrica's automatic phase and set-piece tags, as evidence for R4.

Parsing is the official starter/load_smart_tagging.py (standard library). This module adds:
- frames: round(seconds * 25), and the official end_frame (first frame AFTER the interval) is
  turned into our inclusive end: end - 1;
- case-insensitive codes and tag values (codes are upper case, values title case);
- the Team tag mapped to 0 (home) / 1 (away) by the club names in the ATD metadata, or None where
  it says N/A (36 tags, including every throw-in). An unknown club name raises: never guessed.

Tags are inferred automatically, so they are evidence, not ground truth. A set piece's interval is
a ~5 s window AROUND the restart and partly live play (interfaces.SetPiece), never a D interval.
"""
from typing import NamedTuple

from mtah.config import FPS
from mtah.interfaces import AWAY, HOME, SetPiece

SET_PIECES_CODE = "set pieces"
NO_TEAM = "n/a"


class Tag(NamedTuple):
    start: int  # inclusive frames
    end: int
    code: str  # as written, e.g. "SET PIECES"
    team: int | None  # 0 home, 1 away, None for N/A or no Team tag
    half: str | None  # "1st Half" / "2nd Half"
    kind: str | None  # the Type tag, e.g. "Throw-in"
    tags: dict[str, str]  # every tag group, as written


def load_tags(home_name: str, away_name: str, data_dir: str | None = None) -> list[Tag]:
    """Every Smart Tagging instance, in file order. Club names come from io_atd.team_names()."""
    from starter.load_smart_tagging import load  # the official parser

    return [to_tag(event, home_name, away_name) for event in load(data_dir=data_dir, fps=FPS)]


def to_tag(event: dict, home_name: str, away_name: str) -> Tag:
    tags = event["tags"]
    start = int(event["start_frame"])
    return Tag(
        start=start,
        end=max(start, int(event["end_frame"]) - 1),
        code=event["code"] or "",
        team=team_of(tags.get("Team"), home_name, away_name),
        half=tags.get("Half"),
        kind=tags.get("Type"),
        tags=dict(tags),
    )


def team_of(value: str | None, home_name: str, away_name: str) -> int | None:
    if value is None or value.strip().casefold() == NO_TEAM:
        return None
    name = value.strip().casefold()
    if name == home_name.strip().casefold():
        return HOME
    if name == away_name.strip().casefold():
        return AWAY
    raise ValueError(f"Team tag {value!r} is neither {home_name!r} nor {away_name!r}")


def set_pieces(tags: list[Tag]) -> list[SetPiece]:
    """The set pieces, as windows around each restart. team None means it must be inferred."""
    return [SetPiece(t.start, t.end, t.team, t.kind or "")
            for t in tags if t.code.strip().casefold() == SET_PIECES_CODE]


def summary(tags: list[Tag]) -> dict[str, int]:
    """Counts only, comparable with DATA.md."""
    pieces = set_pieces(tags)
    counts = {
        "instances": len(tags),
        "set_pieces": len(pieces),
        "team_home": sum(t.team == HOME for t in tags),
        "team_away": sum(t.team == AWAY for t in tags),
        "team_none": sum(t.team is None for t in tags),
        "set_pieces_team_none": sum(p.team is None for p in pieces),
    }
    for kind in sorted({p.kind for p in pieces}):
        counts[f"type: {kind}"] = sum(p.kind == kind for p in pieces)
    return counts


if __name__ == "__main__":
    # Real data: run this yourself. Prints counts only.
    from mtah.config import data_dir
    from mtah.io_atd import default_cache, team_names
    from starter.load_smart_tagging import check_time_base, load

    home, away = team_names(default_cache())
    tags = load_tags(home, away, data_dir=str(data_dir()))
    for key, value in summary(tags).items():
        print(f"{key:>32} = {value}")
    print("DATA.md: 638 instances, 59 SET PIECES, Team N/A 36, Throw-in 26, Corner Kick 12, "
          "Goal Kick 10, Free Kick 5, Kick Off (After Goal) 4, Kick Off (Start) 2")
    print("kickoffs contain the real frames:", check_time_base(load(data_dir=str(data_dir())))["agrees"])
