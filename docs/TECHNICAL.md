# Possession from automatic tracking: technical document

Team: The Playmakers

Task: label each of the 199,319 frames of the tactical video H, A or D, ranked on spellF1.

## Approach

Per-frame evidence from hand-written rules and from a learned model, combined and decoded into
possessions by a Viterbi pass. The learned model is trained on **other matches**: Metrica's three open
sample games and seven open Bundesliga matches from the DFL, which together give real labels in two
styles. Those ten matches are also the benchmark every design choice was made on, because the public 5
minutes of our match may not be used for tuning, and synthetic data cannot show how real players and a
real ball behave.

```
ATD ----> features ----> rule evidence: team, shape (ball missing), dead ball --+
            |                                                                    +--> Viterbi per half --> CSV
            +----------> learned layer (trained on 10 open matches) ------------+
```

The pipeline is `mtah/fusion.py`; `run.py --real` runs it.

1. **Features.** The ball, with samples that jump away faster than 45 m/s and back removed and gaps up
   to 0.5 s filled. Each team's nearest and second-nearest distance to the ball, ball speed, mean player
   speed, how far the ball is outside the pitch, and each team's centroid and spread.
2. **Team evidence.** The difference between the two teams' nearest distances, weighted down when
   nobody is near the ball and when it flies (over 10 m/s). With the ball missing, a **shape model** takes
   over: R2's logistic model on team shape (`mtah/evidence.py`, through `mtah/r2_features.py`), each
   feature measured in each team's own attacking direction. It is fitted once on the open matches'
   clear contacts (one team's nearest player within 1.5 m of the ball, the other's over 3 m) and
   applied unchanged to our match.
3. **Dead-ball evidence.** The ball more than 0.5 m outside the pitch; a ball at rest among slow
   players; a missing ball among slow players (a weak cue: a missing ball alone is never dead); and
   how likely the collective player speed is under a slow group against a fast one, the two groups
   fitted once on the open matches.

Nothing is fitted on our match: the learned layer, the shape model and the speed groups are all fitted
on the open matches (`fusion.fit_context`, `fusion.fit`), and a test checks that labelling our match
fits nothing (`tests/test_fusion.py`).
4. **Learned layer.** A gradient-boosted classifier (scikit-learn) on 39 features: the above, plus
   averages over windows of about 1, 3 and 8 s around each frame, the ball's position and motion
   relative to each team's attacking direction, R2's four evidence columns (nearest-player evidence,
   time-to-ball lead, and the shape model on every frame and on lost frames), and three from R3's
   receptions (`mtah/touches.py`: the team of the half's latest reception, seconds since it, and home
   minus away receptions over the last 10 s). It is trained with home and away swapped as well, and
   applied both ways round and averaged, so it cannot favour either team.
5. **Decoding.** One Viterbi pass per half over H, A and D, with a cost for switching team and a higher
   one for switching to or from D. Short possessions are kept (see "How settings were chosen").

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
| Metrica open sample games | Training data for the learned layer, and a benchmark |
| DFL open Bundesliga matches | Training data for the learned layer, and a benchmark |

## Training data

No data of any kind about our match is labelled or used for training, and no person labelled anything
for this project. Both sources are other matches, which the rules allow (RULES.md, FAQ.md).

**1. Metrica Sports' open sample data**: three anonymised professional matches with tracking and event
data. Source `https://github.com/metrica-sports/sample-data`, commit
`e706dd506b360d69d9d123d5b8026e7294b13996`. `mtah/opengames.py` checks every file against a pinned SHA-256.
Labels are made from Metrica's own event data for those games, which their analysts produced: H or A is
the team of the latest pass, carry, recovery, shot or set piece; D runs from a ball out, a foul or a goal
to the next of those. The tracking is used as it comes: its ball is already missing on 32-40% of graded
frames, close to our match's 41%.

| Game | Frames | Graded (both periods) | Format |
|---|---|---|---|
| Sample Game 1 | 145,006 | 142,066 | CSV |
| Sample Game 2 | 141,156 | 140,076 | CSV |
| Sample Game 3 | 143,761 | 142,646 | EPTS, as our match |

**2. The DFL/IDSSE open data**: seven Bundesliga matches from 2022/23 (first and second division), with
optical tracking at 25 fps, under the CC BY 4.0 licence. Bassek, M., Rein, R., Weber, H. et al. "An
integrated dataset of spatiotemporal and event data in elite soccer." Sci Data 12, 195 (2025),
`https://doi.org/10.1038/s41597-025-04505-y`. We use the `pysport/idsse-data` mirror on Hugging Face at
revision `a715a38dfbaf5f58e431727c2b78d174101a703c`, and `mtah/dfl.py` checks every file against a pinned
SHA-256. **Labels come straight from the tracking provider:** every ball frame carries `BallStatus`
(alive or dead) and `BallPossession` (home or away). That is how our match's ground truth is built (D
where the tracking marks the ball dead, else the team it names as owning it; CHALLENGE.md). The optical
ball is almost never missing, so it is given our feed's gaps with a fixed seed per match (`dfl.degrade`).

| Match | Frames |   | Match | Frames |
|---|---|---|---|---|
| J03WMX | 145,967 | | J03WPY | 146,211 |
| J03WN1 | 141,561 | | J03WQQ | 142,345 |
| J03WOH | 137,214 | | J03WR9 | 146,810 |
| J03WOY | 142,536 | | | |

In all, the learned layer trains on 2,854,864 rows: every graded frame of the ten matches, once as it is
and once with home and away swapped.

**How settings were chosen.** Each match held out in turn, the learned layer retrained without it,
scored with the official scorer (`evaluation/possession_metrics.py`). Differences under 0.04 are noise
(EVALUATION.md).

| Pipeline | Metrica (3 games) spellF1 / macroF1 | DFL (7 matches) spellF1 / macroF1 |
|---|---|---|
| Our earlier rule-only pipeline | 0.496 / 0.686 | |
| Pooled rules, no learned layer | 0.507 / 0.726 | 0.361 / 0.796 |
| Learned layer trained on Metrica only, short possessions merged | 0.615 / 0.776 | 0.417 / 0.801 |
| Learned layer trained on Metrica only | 0.654 / 0.779 | 0.438 / 0.802 |
| Learned layer trained on DFL only | 0.634 / 0.771 | 0.500 / 0.859 |
| **Learned layer trained on both (submitted)** | **0.652 / 0.781** | **0.496 / 0.856** |

Training on both ties the best single source on each benchmark, and beats the Metrica-only model by 0.06
to 0.08 on the matches whose labels are built like our ground truth.

**The team's modules as inputs.** R2's shape model in place of ours, plus R2's and R3's signals as
learned-layer inputs, were then tested the same way, held out in turn on all ten matches, and also with
the ball moved 0.3-2.5 m off its carrier the way ours sits (see Limitations). For an average over ten
matches, the 0.04 band for one match is too blunt, so each change was judged by its paired gain per
match, with a bootstrap range for the mean gain.

| Change against the row above (spellF1) | Gain, as trained | Gain, ball moved off the carrier |
|---|---|---|
| R2's shape model only | +0.010, better on 8 of 10 | +0.009, better on 7 of 10 |
| R2's shape model and R3's receptions | +0.017, 9 of 10 (range +0.010 to +0.024) | +0.011, 9 of 10 (+0.005 to +0.018) |
| R2's shape model, R2's and R3's inputs | +0.019, 10 of 10 (+0.011 to +0.028) | +0.020, 9 of 10 (+0.012 to +0.029) |
| **Then the learned layer's weight 0.5 → 0.7 (submitted)** | **+0.012, 8 of 10 (+0.006 to +0.017)** | **+0.006, 8 of 10 (+0.002 to +0.009)** |

With the new inputs in, the decoding settings were searched again (switch costs 10/14/20 and 20/30/45,
learned weight 0.5/0.7): weight 0.7 gained at almost every switch cost, and the switch costs stayed as
they were. The ten-match mean is 0.573 as trained and 0.505 with the ball displaced, against 0.543 and
0.479 before the team's modules went in.

**Short possessions are kept.** An earlier version merged possessions under 2 s into their neighbours,
as a guard in case our ground truth was less fragmented than event labels. The DFL labels answered that:
labels built from tracking are more fragmented, not less.

| Labels | Possessions a minute | Median | Under 3 s |
|---|---|---|---|
| Metrica, from events | 3.3-3.7 | about 7 s | 22-26% |
| DFL, from tracking ownership | 3.8-5.3 | 3.0-4.7 s | 38-50% |

The merge cost 0.02 on every one of the seven DFL matches and 0.039 on Metrica's, so it was dropped.

**The public 5 minutes** were scored only as correctness checks, never to choose or tune: on the earlier
rule-only pipeline and on the earlier submitted version (spellF1 0.553 there, 17 possessions against
17, team accuracy 0.90). With 17 possessions, that shows alignment, not the size of any gain.

## Possession rules

- **Loose ball:** stays with the last team. The Viterbi switch costs keep a team until the evidence for
  the other team outweighs them, so a slow ball past an opponent does not flip it.
- **Changes of possession:** need sustained evidence for the other team, from proximity, from shape
  with the ball missing, and from the learned layer.
- **Dead ball:** D, never owned by a team. `H D H` is two possessions. A missing ball alone is never D.
- **Before the second-half kick-off:** D, from the published block (FAQ.md). This is the only
  frame-specific fact in the pipeline.
- **Outside the graded halves:** X, which is never scored.

## Limitations

- **Two label styles, neither ours.** Metrica's labels come from events, the DFL's from another tracking
  provider's ownership flag. Ours come from Metrica's corrected tracking. The DFL labels are built the
  same way as ours, but the two providers may define "owning" differently.
- **The ball sits differently.** Metrica's open-games ball sits on its carrier: with a slow ball the
  nearest player is a median 0.04-0.09 m away, against 1.21 m on our match (moving it 0.3-2.5 m off
  brings Metrica's to 1.09-1.27 m). A model trained on balls at the players' feet may read our match's
  proximity as weaker than it is.
- **Dead ball.** On our match the pipeline marks 37.5% of graded frames D: 34.4% in the first half and
  40.5% in the second, of which 3.7 points are the published block. The earlier rule-only pipeline,
  built on different cues (set-piece tags, a still camera, slow players), also finds more D after the
  break (24.4% and 31.6%), so the second half of this match likely has more stoppages. On the DFL
  matches, whose labels are built like ours, D is 41-52% of frames.
- **Ten training matches.** Still a small base; another league or tracking provider could behave
  differently.
- **2D only.** No ball height, so a ball in the air over a player can look close to him.

With more time we would label more open matches (SkillCorner's open A-League tracking, for one), add
set-piece and camera cues as learned features, train on balls placed off the players the way ours are,
and try a sequence network over several seconds of context alongside the learned layer.

### Tried and dropped

Our earlier rule-only pipeline remains in the repository (`run.py --real --rules`), with R2's
evidence, R3's touch detection, dead-ball rules (set-piece tags, untagged stoppages, slow players, a
still camera) and possession decoding. It scored 0.496 on the Metrica benchmark and 0.351 on the DFL
matches. Within it:

- Two evidence variants, `evidence_weighted` and `evidence_speed_aware`, did no better.
- Team spread inside ball gaps scored an AUC of 0.70-0.73 against the pipeline's own output, but as a
  flip rule caught 6 of 16 changes while wrongly flipping 5 of 40 unchanged gaps.
- Smart Tagging phases as a vote across ball gaps were worse than keeping the team (64% against 71%).
- The camera as a ball locator was further from the ball (24 m) than the players' average position
  (20 m).
- Forcing that pipeline's dead-ball cues into the submitted one changed spellF1 by -0.007 on the
  Metrica benchmark, so they were left out.
- A simpler decoder (R4's: possession flips at every reception by the other team, possessions under
  2 s merged) gained 0.041 on Metrica's games and 0.064 with the ball displaced, but lost 0.027 on
  every one of the seven DFL matches. Most of that is the 2 s merge: ours with it gained 0.030 and
  0.068 on Metrica and lost 0.022 on the DFL matches (0 of 7), the same split as in the submitted
  pipeline, so both were dropped. Keeping R2's shape evidence on lost frames in the decoder gained
  0.005 on the DFL matches (6 of 7) but lost 0.014 on Metrica's (0 of 3).
- R2's time-to-ball evidence and calibrated versions of both tied with the nearest player on
  Metrica's games (0.39 against 0.40-0.42), so the simpler one stays here; the time-to-ball lead is
  a learned-layer input in the submitted pipeline instead.
- Abstaining (X) inside possessions: an X is a miss under macroF1 and gains nothing under spellF1,
  so neither pipeline writes one inside the graded halves.

In the submitted pipeline: training on Metrica only, training on the DFL matches only (0.634 on
Metrica), and merging possessions under 2 s (see above).

## Results we can state

| | Value |
|---|---|
| Benchmark, ten matches held out in turn, spellF1 | 0.580 (0.507 with the ball displaced) |
| Graded frames, H / A / D | 36.0% / 26.5% / 37.5% |
| Possessions in the graded halves | 387 (193 and 194), 3.65 a minute, median 6.2 s |
| SHA-256 of the submission file | `002c07e3329aab1474ae255deab9040ba254d9d3e4bb5df4c9c23b07c02e6b9c` |

## Rules and use of AI

We follow the organisers' clarification of the rules.

- **Training and tuning on external data only.** The learned layer, the team-shape model and the
  player-speed groups are fitted on the ten open matches, and every setting was chosen on them, each
  match held out in turn. Nothing is fitted on our match.
- **The ground truth.** The public five minutes were scored only as correctness checks of the final
  files, never to choose between versions or to tune. No other labels of our match were used.
- **No hand labels and no video checks.** Nothing was labelled or corrected by hand, and no output was
  checked against the video. The only frame-specific fact in the pipeline is the published
  second-half block.
- **Summary numbers of our match's tracking** (how often and for how long the ball is lost, how far
  it sits from the nearest player) were measured to make the open matches look like ours: the ball
  gaps given to the DFL matches and the displaced test view. They appear in code comments and
  contain no labels.
- **AI.** We used LLM coding assistants to help write and review code and documents. No competition
  video, data file or extract was given to a hosted AI service: team members ran everything on the
  competition data locally and shared only totals such as counts, file hashes and scores. The
  pipeline itself uses no AI service.
- **Licences.** The DFL/IDSSE matches are CC BY 4.0 (cited above). Metrica's sample data is published
  by Metrica Sports, an organiser of this competition, for public use; its repository states no
  separate licence.

## Reproduction

Needs Python 3.15, the competition data bundle and both open datasets. No GPU. About 5 minutes after the
first run, most of it training on one thread.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_open_data.py     # the open matches, about 450 MB, each file checked against its hash

$env:MTAH_DATA_DIR = "path\to\bundle"   # Linux or macOS: export MTAH_DATA_DIR=path/to/bundle; skip if in ./data
python run.py --real                # writes outputs/The Playmakers.csv and prints its SHA-256
python evaluation/validate_submission.py "outputs/The Playmakers.csv"
```

The Metrica games go to `outputs/metrica/sample-data/data` and the DFL files to `outputs/idsse`;
elsewhere, set `$MTAH_OPEN_GAMES` and `$MTAH_DFL_GAMES` (or pass `--open-games` and `--dfl-games`).
Metrica's files are checked with their line endings normalised, so a checkout on any system passes. The first run reads
the ATD and the DFL XML and caches them in `outputs/cache/`, which takes a few minutes. Everything fitted
uses fixed seeds and one thread, so the run writes the same file every time and prints its SHA-256 to
compare with the one above.

`python -m pytest` runs the tests, on synthetic data and small fake files only.
