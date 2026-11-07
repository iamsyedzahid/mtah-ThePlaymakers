# Possession from automatic tracking: technical document

Team: The Playmakers

Task: label each of the 199,319 frames of the tactical video H, A or D, ranked on spellF1.

## Approach

Per-frame evidence from hand-written rules and from a learned model, combined and decoded into
possessions by a Viterbi pass. The learned model is trained on **other matches**: Metrica's three open
sample games, whose event data gives real labels. Those games are also the benchmark every design
choice was made on, because the public 5 minutes of our match may not be used for tuning, and synthetic
data cannot show how real players and a real ball behave.

```
ATD ----> features ----> rule evidence: team, shape (ball missing), dead ball --+
            |                                                                    +--> Viterbi per half --> merge short spells --> CSV
            +----------> learned layer (trained on Metrica's open games) -------+
```

The pipeline is `mtah/fusion.py`; `run.py --real` runs it.

1. **Features.** The ball, with samples that jump away faster than 45 m/s and back removed and gaps up
   to 0.5 s filled. Each team's nearest and second-nearest distance to the ball, ball speed, mean player
   speed, how far the ball is outside the pitch, and each team's centroid and spread.
2. **Team evidence.** The difference between the two teams' nearest distances, weighted down when
   nobody is near the ball and when it flies (over 10 m/s). With the ball missing, a **shape model** takes
   over: a logistic model on team shape, fitted separately in each half on that half's own clear contacts
   (one team within 1.5 m of the ball, the other over 3 m). These are pseudo-labels made by an algorithm
   (RULES.md), and refitting per half follows the change of ends and of tactics.
3. **Dead-ball evidence.** The ball more than 0.5 m outside the pitch; a ball at rest among slow
   players; a missing ball among slow players (a weak cue: a missing ball alone is never dead); and,
   per half, a two-group fit of collective player speed that separates slow phases from fast ones
   without a fixed threshold.
4. **Learned layer.** A gradient-boosted classifier (scikit-learn) on 32 features: the above, plus
   averages over windows of about 1, 3 and 8 s around each frame, and the ball's position and motion relative to
   each team's attacking direction. It is trained with home and away swapped as well, and applied both
   ways round and averaged, so it cannot favour either team.
5. **Decoding.** One Viterbi pass per half over H, A and D, with a cost for switching team and a higher
   one for switching to or from D. Then a possession under 2 s between two of the other team's, with no
   dead ball between, joins them.

The ideas were pooled from several public forks of the starter repository and rebuilt here with three
fixes: the published graded halves instead of halves guessed from the tracking, the learned layer seeing
missing values the same way in training and in use, and the published second-half block.

## Assets

| Asset | Use |
|---|---|
| Tactical ATD: ball and player positions | Everything above |
| Smart Tagging | Not used by the submission. The earlier rule-only pipeline uses its set pieces; see below |
| Tactical homography | Not used by the submission. The earlier pipeline uses it to tell when the camera is still |
| Tactical video, broadcast video and ATD | Not used: the broadcast has no mapping to the tactical frames |
| Metrica open sample games | Training data for the learned layer, and the benchmark (see below) |

## Training data

**Metrica Sports' open sample data**: three anonymised professional matches, other than ours, with
tracking and event data. Source `https://github.com/metrica-sports/sample-data`, commit
`e706dd506b360d69d9d123d5b8026e7294b13996`. `mtah/opengames.py` checks every file against a pinned SHA-256.

| Game | Frames | Graded (both periods) | Format |
|---|---|---|---|
| Sample Game 1 | 145,006 | 142,066 | CSV |
| Sample Game 2 | 141,156 | 140,076 | CSV |
| Sample Game 3 | 143,761 | 142,646 | EPTS, as our match |

**Labels.** Made by `mtah/opengames.py` from Metrica's own event data for those games, which their
analysts produced: H or A is the team of the latest pass, carry, recovery, shot or set piece; D runs
from a ball out, a foul or a goal to the next of those. No data of any kind about our match is labelled
or used for training, and no person labelled anything for this project.

**How settings were chosen.** Each open game held out in turn, the learned layer trained on the other
two, scored with the official scorer (`evaluation/possession_metrics.py`). The tracking was used as it
comes: its ball is already missing on 32-40% of graded frames, close to our match's 41%.

| Pipeline | spellF1 | macroF1 |
|---|---|---|
| Our earlier rule-only pipeline | 0.496 | 0.686 |
| Pooled rules, no learned layer | 0.507 | 0.726 |
| Pooled rules + learned layer | 0.654 | 0.779 |
| **+ merge possessions under 2 s (submitted)** | **0.615** | 0.776 |

Differences under 0.04 are noise (EVALUATION.md). The merge costs 0.039 here and was kept anyway: the
event labels are fragmented themselves (22-26% of their possessions last under 3 s), and FAQ.md warns
against submissions full of tiny possessions, so the merge guards against a ground truth that is less
fragmented than event data. Across the halves of the three games spellF1 ranges from 0.56 to 0.69, with
no systematic drop in either half.

**The public 5 minutes** were scored only as a correctness check: once on the earlier rule-only
pipeline, and once on the submitted file after it was chosen, with the decision to keep it whatever the
score unless something was broken. Nothing was chosen or tuned on them. The submitted file scored
spellF1 0.553 there (17 possessions against 17, team accuracy 0.90, D found on 21% of its frames, all of
them right); with 17 possessions, that shows alignment, not the size of any gain.

## Possession rules

- **Loose ball:** stays with the last team. The Viterbi switch costs keep a team until the evidence for
  the other team outweighs them, so a slow ball past an opponent does not flip it.
- **Changes of possession:** need sustained evidence for the other team, from proximity, from shape
  with the ball missing, and from the learned layer.
- **Short possessions:** under 2 s between two of the other team's are merged into them, so a touch by
  an opponent that is not control does not count.
- **Dead ball:** D, never owned by a team. `H D H` is two possessions. A missing ball alone is never D.
- **Before the second-half kick-off:** D, from the published block (FAQ.md). This is the only
  frame-specific fact in the pipeline.
- **Outside the graded halves:** X, which is never scored.

## Limitations

- **Labels made from events.** The learned layer learns the style of event-derived labels: possessions
  change at events, and a pass in flight belongs to the passer. Our match's ground truth comes from
  corrected tracking (CHALLENGE.md), which may be less fragmented. On our match the pipeline reports
  355 possessions, 3.35 a minute.
- **Dead ball is over-called.** On the open games it predicts up to 10 points more D than the labels
  (finding 74-91% of their dead frames, with 48-87% of what it marks right). On our match it marks
  34.8% of graded frames D: 28.2% in the first half and 41.4% in the second, of which 3.7 points are
  the published block. The earlier rule-only pipeline, built on different cues (set-piece tags, a still
  camera, slow players), also finds more D after the break: 24.4% and 31.6%. So the second half of this
  match likely has more stoppages, and the rest of the gap fits the over-calling seen on the open games.
- **Three training games.** A small base; a different league or tracking provider could behave
  differently.
- **2D only.** No ball height, so a ball in the air over a player can look close to him.

With more time we would label more open games, add set-piece and camera cues as learned features, and
calibrate the dead-ball evidence on more data.

### Tried and dropped

Our earlier rule-only pipeline remains in the repository (`run.py --real --rules`), with its own
evidence, touch detection, dead-ball rules (set-piece tags, untagged stoppages, slow players, a still
camera) and possession decoding. It scored 0.496 on the benchmark. Within it:

- Two evidence variants, `evidence_weighted` and `evidence_speed_aware`, did no better.
- Team spread inside ball gaps scored an AUC of 0.70-0.73 against the pipeline's own output, but as a
  flip rule caught 6 of 16 changes while wrongly flipping 5 of 40 unchanged gaps.
- Smart Tagging phases as a vote across ball gaps were worse than keeping the team (64% against 71%).
- The camera as a ball locator was further from the ball (24 m) than the players' average position
  (20 m).
- Forcing that pipeline's dead-ball cues into the submitted one changed spellF1 by -0.007 on the
  benchmark, so they were left out.

## Results we can state

| | Value |
|---|---|
| Benchmark, three open games held out, spellF1 | 0.615 |
| Graded frames, H / A / D | 38.3% / 26.9% / 34.8% |
| Possessions in the graded halves | 355 (179 and 176) |
| SHA-256 of the submission file | `61e0485fdbd8933e4b92f038f75025afdd7cdc872722cc4dbb475a160c248e41` |

## Reproduction

Needs Python 3.15, the competition data bundle and Metrica's open sample games. No GPU.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/metrica-sports/sample-data outputs/metrica/sample-data
git -C outputs/metrica/sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996

$env:MTAH_DATA_DIR = "path\to\bundle"   # Linux or macOS: export MTAH_DATA_DIR=path/to/bundle; skip if in ./data
python run.py --real --out "The Playmakers.csv"
python evaluation/validate_submission.py "The Playmakers.csv"
```

The open games are found in `outputs/metrica/sample-data/data`; elsewhere, set `$MTAH_OPEN_GAMES` or
pass `--open-games`. The first run reads the ATD and caches it in `outputs/cache/`, which takes a few
minutes. Everything fitted uses fixed seeds and one thread, so the run writes the same file every time
and prints its SHA-256 to compare with the one above.

`python -m pytest` runs the tests, on synthetic data and small fake files only.
