"""Home/away mapping of tracking slots (R1).

The ATD has anonymous slots ("Track 1" .. "Track 46"), each belonging to one team. The first team
in the metadata is home (starter/load_atd.py prints it as H). Columns in the tracking frame are
named after each slot's player_id, so the mapping is keyed by player_id.
"""
from typing import Any

from mtah.interfaces import AWAY, HOME


def slot_teams(teams: Any) -> dict[str, int]:
    """{player_id: 0 for home, 1 for away} from kloppy's metadata.teams (home first).

    Takes anything shaped like kloppy's teams: a sequence of two objects, each with `.players`,
    each player with `.player_id`. A slot listed under both teams is an error, not a guess.
    """
    if len(teams) != 2:
        raise ValueError(f"expected 2 teams in the metadata, got {len(teams)}")
    mapping: dict[str, int] = {}
    for side, team in zip((HOME, AWAY), teams):
        for player in team.players:
            slot = str(player.player_id)
            if slot in mapping:
                raise ValueError(f"slot {slot!r} is listed under both teams")
            mapping[slot] = side
    return mapping
