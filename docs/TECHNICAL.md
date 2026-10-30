# Possession from automatic tracking: technical document

Team: The Playmakers

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
Homography (camera) -------+
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
4. *Ball missing, camera still and players slowish.* The homography gives where the camera looks. It
   mostly stops at a stoppage but also often in live play, so it counts only with the median player
   under 1.5 m/s, both held for 8 s, overlapping a missing ball by at least 3 s.
5. *The second-half block before kick-off*, published in FAQ.md. This is the only frame-specific fact
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
| Tactical homography | Only to tell when the camera is still, as a stoppage cue. The ATD is already in pitch coordinates |
| Smart Tagging phases (Build Up, Transition and so on) | Tested, not used: see below |
| ATD speed channel | Not used: speeds are computed from positions |

Tried and dropped:

- Two evidence variants, `evidence_weighted` and `evidence_speed_aware`. They remain in the code but
  are not used. On the real match the weighted variant blurs who has the ball, because pressers and
  supporting players count too.
- Abstaining with X. It counts as a miss under macroF1 and gains nothing under spellF1.
- A slow-players limit of 1.25 m/s. It gave similar spell counts to 1.0 m/s on the real match, and a
  false D costs a whole possession, so the stricter limit stayed.
- Team spread to find changes inside ball gaps. The team with the ball spreads out: against our own
  output, with no ground truth, the spread gap between the teams scored an AUC of 0.76-0.79 with the
  ball visible and 0.70-0.73 inside gaps. As a flip rule it was not enough. At its best setting, on 56
  gaps, it caught 6 of 16 changes but wrongly flipped 5 of 40 unchanged gaps, for 73% accuracy against
  71% for keeping the team.
- Looking up to 5 s after a gap for a hidden change, and treating clear evidence as the ball reappears
  as a reception. Both lowered spellF1 slightly on synthetic data.
- Smart Tagging phases as a vote across ball gaps. They agree with our output on 77% of the visible
  frames they cover, from 62% (`PLAYERS IN THE BOX`) to 89% (`PROGRESSION`). Flipping on all of them was
  worse than keeping the team (64% against 71%, on 56 gaps). Using only `BUILD UP` and `PROGRESSION` gave
  75%, but that was 3 changes caught and 1 wrong flip, on codes picked after seeing the data, and those
  changes were ones the pipeline already finds when the ball returns.
- The camera as a ball locator. It is further from the ball (median 24 m) than the players' average
  position (20 m).
- A still camera alone as a stoppage cue. It wrongly caught 5-38% of fast-ball frames, depending on the
  setting. Requiring slowish players as well brought the chosen setting to 1.5%.

## Training data

None. No labelled dataset of any kind is used, and nothing is trained.

**How settings were chosen.** Thresholds come from football reasoning, from aggregate statistics of
the real tracking, and from a seeded synthetic match generator in `mtah/synthetic.py`, shaped by those
same aggregates. Those statistics are counts and percentiles computed without any labels. Settings were
compared by the official scorer on synthetic truth only.

**The public 5 minutes** were scored once, to check correctness. Nothing was changed as a result. The
still-camera rule was added afterwards, on a pass-or-fail bar set in advance from label-free statistics
of the real tracking, and its version was not scored on the public window.

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
  estimate. On synthetic data this is the largest loss: spellF1 is 0.59 for the full pipeline, 0.66 with
  true dead-ball intervals, 0.83 with true receptions and 0.93 with both. Those are comparisons between
  versions, not estimates of the real score, and differences of about 0.03 are noise.
- **Dead ball is inferred.** Slow players can also mean slow live build-up, which the rules would call
  dead, and stoppages with quick players are missed. The still-camera rule cannot be scored at all, as
  synthetic matches have no real camera. In the public 5 minutes the version before that rule found only
  8% of D frames, though 98% of the frames marked D were right. Its graded D share was 25.7%, so this
  window may be unusual, but one window cannot show that.
- **Most exact transitions are missed.** In the public 5 minutes 2 of 20 fell within 25 frames.
- **2D only.** There is no ball height, so a ball in the air passing over a player can look close.
- **Team, not player.** Pressers near the ball can cause false receptions. Evidence removes some of them.
- **Validated on synthetic data.** With no labels to tune on, the thresholds are checked against a
  generator we wrote, which cannot contain what we did not think of.

With more time we would use player formations around restarts to find stoppages, and try team spread
as a soft signal in possession decisions rather than a flip rule. It carries real information, but we
had no labelled way to validate such a change.

### Results we can state

| | Value |
|---|---|
| Public 5 minutes, spellF1 (version before the still-camera rule) | 0.4400 |
| Public 5 minutes, macroF1 (same version) | 0.5623 |
| Graded frames, H / A / D | 42.4% / 29.6% / 28.0% |
| Possessions in the graded halves | 268 |

The public window does not predict the full-match score (EVALUATION.md). The first two rows show only
that the pipeline is aligned and its labels mean what they should.

## Reproduction

Needs Python 3.15 and the data bundle. No GPU.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt

$env:MTAH_DATA_DIR = "path\to\bundle"   # Linux or macOS: export MTAH_DATA_DIR=path/to/bundle; skip if in ./data
python run.py --real --out "The Playmakers.csv"
python evaluation/validate_submission.py "The Playmakers.csv"
```

The first run reads the ATD and caches it in `outputs/cache/`, which takes a few minutes. The pipeline
has no randomness, so it writes the same file every time, and it prints the file's SHA-256 to compare.

`python -m pytest` runs the tests, on synthetic data only. `python run.py` runs the pipeline on a
synthetic match without the data bundle.
