# The Playmakers: frame-level ball possession

Technical document for *More than a Hack 2027*, Phase 1. Team: The Playmakers.
Repository: https://github.com/iamsyedzahid/mtah-ThePlaymakers, commit **[TBD: final commit]**.
Submission: `The Playmakers.csv`, SHA-256 **[TBD: final hash]**.

> Draft. Sections marked **[TBD]** are filled in after the final merge on November 14, 2026.

## 1. Approach

Possession is an interval, not a per-frame guess. spellF1 counts each possession once, so we built
a pipeline that holds possession steady and changes it only on strong evidence of a change. The ball
is missing on about 41% of graded frames, so the pipeline never needs a ball position to keep a
label. Every stage is a plain algorithm with fixed constants and no randomness, so the same data
always gives the same file.

```
tracking --> 1. load and clean the ball --> 2. evidence (who looks in control)  --+
         \-> 3. touches (receptions and deflections) -----------------------------+--> 5. segment --> 6. X policy --> CSV
Smart Tagging + published dead block --> 4. dead-ball mask -----------------------+
```

1. **Loading and ball cleaning** (`mtah/io_atd.py`, `mtah/ball.py`). The tactical tracking is read
   line by line (kloppy's EPTS reader), indexed from 0 (`frame_id - 1`) and scaled to metres
   (105.3 x 68 m). The first 300 frames are checked against kloppy's own loader. Ball cleaning drops
   samples implying more than 40 m/s, smooths over 5 frames, fills gaps under 1 s with a
   shape-preserving spline (PCHIP), and marks longer gaps as *lost*. Lost frames are filled for
   geometry but never trusted.
2. **Evidence** (`mtah/evidence.py`). On each trusted frame, the nearest tracked player's team,
   weighted by `exp(-distance / 2 m)`: a number in [-1, 1], positive for home. On lost frames it is 0.
   **[TBD: final evidence() — whether team-shape evidence on lost frames is kept, see section 6.]**
3. **Touches** (`mtah/touches.py`). From the raw ball, never interpolated: the nearest tracked slot
   within 2 m of the ball. A run of at least 5 frames (0.2 s) with the same slot is a *reception*;
   a shorter one is a *deflection*. Tracking blinks of up to 2 frames inside a run are bridged.
4. **Dead ball** (`mtah/deadball.py`). Smart Tagging's set-piece windows (`mtah/tagging.py`) plus
   the organisers' published block before the second-half kick-off (video frames 110,808-113,741).
   A missing ball is never, by itself, a dead ball. **[TBD: final dead-ball rules from R4.]**
5. **Segmentation** (`mtah/segment.py`). Labels each frame H, A or D. D comes only from the
   dead-ball mask. While the ball is alive there is always a team: possession changes at a
   reception by the other team and holds through lost frames, deflections and loose balls. Runs
   shorter than 1 s are absorbed by a neighbour. **[TBD: final segmenter from R4 (Viterbi
   decoder, switch costs and minimum run length tuned on open games).]**
6. **Abstention** (`mtah/xpolicy.py`). `X` only where there is no evidence at all, and only at least
   1 s from either end of a possession, so it never delays a hand-over. The official scorer fills an
   `X` with the previous label for spellF1, so this costs nothing on the ranking metric.

## 2. Assets used and dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (ATD) | Yes | Positions of players and ball, every stage |
| Smart Tagging | Yes | Set-piece windows for the dead-ball mask. Phase tags are not used |
| Published frame facts | Yes | Graded halves and the second-half block before kick-off (DATA.md, FAQ.md) |
| Tactical homography | No | A camera-movement cue was planned but not built **[TBD]** |
| Tactical video | No | Not read by the pipeline |
| Broadcast video, tracking, homography | No | Not synchronised to the tactical timeline; no mapping provided |
| Public 5-minute ground truth | Check only | Run once at the end to confirm the file is aligned and well formed. Never used to train, tune or choose |

## 3. Training and validation data

No data about the provided match was labelled by anyone. All learning, tuning and choosing between
versions used other matches only:

| Dataset | Source | Labels | Amount | Used for |
|---|---|---|---|---|
| Metrica sample data, games 1-3 | https://github.com/metrica-sports/sample-data, commit `e706dd5` | From Metrica's event data, by rule (`mtah/metrica_games.py`) | 429,923 frames | Evidence calibration and choice, segmenter constants, benchmark |
| IDSSE: 7 DFL Bundesliga matches | Bassek et al. (2025), *Scientific Data* 12, 195; CC BY 4.0; Hugging Face mirror `pysport/idsse-data` | The tracking provider's own ball status and possession flags | 1,002,644 frames | Benchmark and robustness checks |

**How the labels were made.** Metrica: H or A is the team of the latest pass, carry, recovery, shot,
set piece or ball loss; D runs from the end of the last live action to each restart. DFL: D where
the provider marks the ball dead, else the team it marks in possession. No person labelled any
frame for us.

**Making the open games look like ours.** The open games almost never lose the ball, and their ball
sits on its carrier. To test on realistic input we hide the ball in gaps and move it off the
carrier, with fixed random seeds (`mtah/metrica_games.py`). The gap lengths and lost share used for
this (about 41% of frames, median gap about 11 s) and the ball's typical distance from its carrier
(about 1.2 m) are summary statistics measured on the provided tracking. They describe the input,
not the answer: no possession label of the provided match was used.

**Choosing between versions.** `scripts/benchmark_open_games.py` runs the whole pipeline on every
open game, with set-piece windows made from that game's own restarts (2.5 s either side, standing
in for Smart Tagging), and scores it with the official `evaluation/possession_metrics.py`.
Differences under 0.04 spellF1 are treated as ties, and a tie keeps the simpler version.

## 4. Possession rules we implemented

- **Loose balls and challenges:** possession stays with the last team until the other team receives
  the ball. A touch counts as a reception only if the same player stays within 2 m for 0.2 s.
- **Deflections and won headers without control:** never change possession.
- **Lost ball:** possession holds. A missing ball is never a dead ball.
- **Restarts:** D inside Smart Tagging's set-piece windows and before the second-half kick-off.
  After a restart, the next reception decides the team.
- **Flicker:** a possession shorter than **[TBD: 1 s on main, 2 s in R4]** is absorbed by its
  neighbour.

## 5. Results

On the open games, ball masked like ours, whole pipeline, mean spellF1 over held-out games:

| Version | Metrica (3 games) | DFL (7 matches) |
|---|---|---|
| Current `main` | 0.291 | 0.249 |
| R4 segmenter (in progress) | 0.299 | 0.237 |
| Final submission | **[TBD]** | **[TBD]** |

Possession counts per game are close to the truth on Metrica (about 365 predicted against 382) but
low on DFL (about 342 against 437). The organisers' published floor on the provided match is
spellF1 0.2433 (`baseline_smooth50`); we have no score on the full match.

## 6. Limitations and what we would fix

- **Dead ball is the biggest gap.** We mark about 9% of open-game frames dead against 45% in DFL's
  labels. Smart Tagging covers only set pieces, in windows of about 5 s, while real stoppages are
  longer and include balls out of play without a tag. A detector for the ball off the pitch, a still
  ball and slow players would help most.
- **Possession changes while the ball is lost are hard to see.** In one Metrica game only 11 of
  1,301 detected receptions fall on lost-ball frames. A team-shape model scores the same as no model on spellF1
  on both datasets, so its value is unproven. **[TBD: kept or dropped; if kept, trained on other
  games only.]**
- **Receptions are a distance rule.** A classifier trained on open-game events was planned but not
  built.
- **Throw-ins have no team in Smart Tagging**, so restarts name no possessor.
- **The benchmark is small** (10 matches) and its labels are not the organisers' definition for
  Metrica, so close results are ties.

## 7. Reproduction

Tested on Windows 11 with Python 3.14.6. Dependencies are pinned in `requirements.txt`.

```
git clone https://github.com/iamsyedzahid/mtah-ThePlaymakers
cd mtah-ThePlaymakers
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                # synthetic data only
```

Unpack the competition bundle so that `<folder>/tactical/` holds the files listed in DATA.md, then:

```
set MTAH_DATA_DIR=<folder>         # PowerShell: $env:MTAH_DATA_DIR = "<folder>"; macOS/Linux: export
python run.py --real               # writes outputs/The Playmakers.csv, prints its SHA-256
python evaluation/validate_submission.py "outputs/The Playmakers.csv"
```

The printed SHA-256 must be **[TBD: final hash]**. It was reproduced after deleting the cache, from
a clean clone (`bash scripts/fresh_clone_check.sh --real`) and on a second computer **[TBD]**.

To regenerate the constants fitted on open games and the benchmark:

```
git clone https://github.com/metrica-sports/sample-data outputs/metrica/sample
git -C outputs/metrica/sample checkout e706dd5
python -m mtah.dfl_games --download     # 2.6 GB
python -m mtah.evidence_fit             # evidence calibration and choice
python scripts/benchmark_open_games.py  # add --dfl for the DFL matches
```

## 8. Use of AI tools

We used LLM coding assistants to help write and review code and documents. Following the Terms and
Conditions, no competition video, dataset, file or extract was ever given to a hosted LLM: the
assistants saw code and documentation, and team members ran everything on the competition data
locally, sharing only totals such as counts and file hashes. The pipeline itself uses no LLM.
