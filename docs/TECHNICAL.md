# Possession from automatic tracking: technical document

Task: label each of the 199,319 frames of the tactical video H, A or D, ranked on spellF1.

## Approach

A classical, rule-based pipeline with no learning and no randomness. We chose it because the one
labelled sample available is the public 5 minutes, which the rules exclude from training and tuning.
Every step can also be read and checked by hand. spellF1 matches whole possessions, so the design
favours getting the structure of a match right over per-frame accuracy: a false flip costs a whole
extra possession, while a few frames of lateness cost almost nothing.

```
ATD ------> ball cleaning ---> evidence (who is near the ball) ---+
    |                                                             +--> possession (Viterbi) --> CSV
    +-----> touches and spells of control (receptions) -----------+            ^
    |                                                                          |
    +-----> player speed --+                                                   |
Smart Tagging set pieces --+--> dead ball -------------------------------------+
```

**Ball cleaning** (`mtah/ball.py`). Samples implying over 40 m/s are dropped, as a hard shot reaches
35-40 m/s. Five rejected samples in a row that move consistently become the new reference, so one bad
sample cannot wipe out a second of ball. The track is smoothed over 5 frames and gaps under 1 s are
splined. Longer gaps are flagged lost and never treated as positions.

**Evidence** (`mtah/evidence.py`). A per-frame score in [-1, 1] from each team's nearest player,
Gaussian in distance (sigma 2 m), against a fixed "nobody" score at 5 m, so a loose ball fades to 0.
It is exactly 0 on lost frames.

**Touches and control** (`mtah/touches.py`). Sharp changes in ball speed or direction mark candidate
touches. The provider's ball is smooth and the tracked position is a body centre rather than a foot,
so sharp touches alone find far too few: 1.3 per minute on the real match in our first version. We
therefore also detect spells of control: the nearest player within 3 m with the ball moving with them
(relative speed under 3 m/s), for 10 of the next 15 frames. Each spell start is a reception. A touch
followed by its own team's spell within 3 s is also a reception, which catches an interception played
first time. Any other touch is a deflection and never flips possession.

**Dead ball** (`mtah/deadball.py`). A frame is D if any of these holds:

1. *A set-piece tag.* A tag window is about 5 s around a restart and partly live, so it is never D
   whole. D is the spell before the restart in which the ball rests or is off the pitch, up to 30 s.
   If the ball vanished during the stoppage, D runs from that spell to the middle of the window. With
   no such spell, D is the first half of the window.
2. *A stoppage with no tag*, as the tags have false negatives (DATA.md). The ball is seen off the pitch
   for 1 s, or rests for 7 s, which is longer than the 6 s a goalkeeper may hold it.
3. *Ball missing and players slow.* The median tracked player stays under 1.0 m/s for 5 s, overlapping
   a missing ball by at least 3 s.
4. *The second-half block before kick-off*, published in FAQ.md. This is the only frame-specific fact
   in the pipeline.

D stretches under 3 s apart are joined, as a glimpse of the ball inside a stoppage is not a possession.

**Possession** (`mtah/segment.py`). One Viterbi pass over H and A. Possession may change only at a
reception by the other team, or at the end of a dead ball. Between those frames a state's cost is the
summed evidence against it, and ignoring a reception costs a small tie-break. So evidence decides which
receptions to believe, but can never create a change on its own. A change found within 1 s of a lost
stretch is placed halfway into that stretch. H/A runs under 1 s are absorbed by a neighbour.

## Assets

| Asset | Use |
|---|---|
| Tactical ATD: ball and player positions | Everything above |
| Smart Tagging, `Set Pieces` only | Evidence for restarts: window and team |
| Tactical video | Not used |
| Broadcast video and ATD | Not used: no mapping to the tactical frame index |
| Homography | Not used: the ATD is already in pitch coordinates |
| Smart Tagging phases (Build Up, Transition and so on) | Not used: they are phases, not possessions (DATA.md) |
| ATD speed channel | Not used: speeds are computed from positions |

Tried and dropped:

- Two evidence variants, `evidence_weighted` and `evidence_speed_aware`. They remain in the code but
  are not used. On the real match the weighted variant blurs who has the ball, because pressers and
  supporting players count too.
- Abstaining with X. It counts as a miss under macroF1 and gains nothing under spellF1.
- A slow-players limit of 1.25 m/s. It gave similar spell counts to 1.0 m/s on the real match, and a
  false D costs a whole possession, so the stricter limit stayed.

## Training data

None. No labelled dataset of any kind is used, and nothing is trained.

**How settings were chosen.** Thresholds come from football reasoning, from aggregate statistics of
the real tracking, and from a seeded synthetic match generator in `mtah/synthetic.py`, shaped by those
same aggregates. Those statistics are counts and percentiles computed without any labels. Settings were
compared by the official scorer on synthetic truth only.

**The public 5 minutes** were scored once, on the frozen design, to check correctness. Nothing was
changed as a result.

## Possession rules

- **Loose ball:** stays with the last team that had it (CHALLENGE.md).
- **Change of possession:** needs control by the other team, seen as a reception that the evidence
  afterwards does not contradict.
- **Challenges and deflections:** a deflection or a challenge without control changes nothing. A touch
  by a player whose team then keeps the ball does change possession.
- **Restarts:** a dead ball may end with either team in possession. A tag naming a team counts as a
  reception by that team, because a restart has a clear possessor (CHALLENGE.md). Throw-in tags name no
  team, so there the evidence decides.
- **Dead ball:** always D, never owned by a team. `H D H` is two possessions.
- **Missing ball:** possession holds through it. A missing ball alone is never D.

## Limitations

- **Hidden changes.** About 41% of the ball is missing, and a change inside a gap can only be placed by
  estimate. On synthetic data this is the largest loss: spellF1 is 0.62 for the full pipeline, 0.67 with
  true dead-ball intervals, and over 0.9 with true receptions as well. Those are comparisons between
  versions, not estimates of the real score.
- **Dead ball is inferred.** Slow players can also mean slow live build-up, which the rule would call
  dead, and stoppages with quick players are missed. In the public 5 minutes only 8% of D frames were
  found, though 98% of the frames marked D were right. Our graded D share is 25.7%, so this window may
  be unusual, but one window cannot show that.
- **Most exact transitions are missed.** In the public 5 minutes 2 of 20 fell within 25 frames.
- **2D only.** There is no ball height, so a ball in the air passing over a player can look close.
- **Team, not player.** Pressers near the ball can cause false receptions. Evidence removes some of them.
- **Validated on synthetic data.** With no labels to tune on, the thresholds are checked against a
  generator we wrote, which cannot contain what we did not think of.

With more time we would use player formations around restarts to find stoppages, and estimate from
team shape when a change happened inside a ball gap.

### Results we can state

| | Value |
|---|---|
| Public 5 minutes, spellF1 | 0.4400 |
| Public 5 minutes, macroF1 | 0.5623 |
| Graded frames, H / A / D | 43.5% / 30.8% / 25.7% |
| Possessions in the graded halves | 265 |

The public window does not predict the full-match score (EVALUATION.md). The first two rows show only
that the pipeline is aligned and its labels mean what they should.

## Reproduction

Needs Python 3.15 and the data bundle. No GPU, and about a minute of runtime after the first run.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt

$env:MTAH_DATA_DIR = "path\to\bundle"   # Linux or macOS: export MTAH_DATA_DIR=path/to/bundle; skip if in ./data
python run.py --real --out <team_name>.csv
python evaluation/validate_submission.py <team_name>.csv
```

The first run reads the ATD and caches it in `outputs/cache/`, which takes a few minutes. The pipeline
has no randomness, so it writes the same file every time, and it prints the file's SHA-256 to compare.

`python -m pytest` runs the tests, on synthetic data only. `python run.py` runs the pipeline on a
synthetic match without the data bundle.
