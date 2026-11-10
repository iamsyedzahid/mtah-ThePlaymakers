from pathlib import Path
import numpy as np

from mtah.io_atd import load_tracking, default_cache, team_names
from mtah.tagging import load_tags, set_pieces
from mtah.ball import ball_clean
from mtah.evidence import evidence
from mtah.touches import touches
from mtah.deadball import dead_mask
from mtah.segment import segment
from mtah.xpolicy import to_x_policy
from mtah.interfaces import X
from run import write_csv

cache = default_cache()
tracking = load_tracking(cache)
home, away = team_names(cache)

tags = load_tags(home, away)
set_piece_intervals = [(sp.start, sp.end) for sp in set_pieces(tags)]
print("set pieces:", len(set_piece_intervals))

_, lost = ball_clean(tracking.ball)
e = evidence(tracking)
events = touches(tracking.ball, tracking.players, tracking.team)
dead = dead_mask(tracking.n_frames, set_piece_intervals)
print("lost:", lost.mean(), " dead:", dead.mean(), " e==0:", np.mean(e == 0))

labels = segment(e, lost, dead, events)
labels = to_x_policy(labels, confidence=np.abs(e), threshold=1e-9)
print("X share:", np.mean(labels == X))

write_csv(labels, Path("outputs/mine_real.csv"))