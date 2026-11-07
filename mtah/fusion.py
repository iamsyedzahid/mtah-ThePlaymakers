"""The submission pipeline: per-frame evidence from rules and a learned model, decoded by Viterbi.

Chosen on two labelled benchmarks of other matches, which the rules allow: Metrica's three open games
(mtah/opengames.py, labels from their events) and seven DFL Bundesliga matches (mtah/dfl.py, labels
from tracking ownership like our ground truth). Trained on both, with each match held out in turn, it
scores spellF1 0.652 on Metrica's games and 0.496 on the DFL matches, against 0.507 and 0.361 for the
rules alone. Built from ideas pooled across team forks; see docs/TECHNICAL.md.

Inputs are plain arrays in metres, x in [0, L], y in [0, W]: ball [T, 2] and players [T, S, 2] with NaN
where unseen, team [S] with 0 = home. Output codes [T]: 0 = H, 1 = A, 2 = D, 3 = X (outside the halves).

1. Features (`frame_features`): the ball with impossible jumps removed and gaps up to 0.5 s filled;
   each team's nearest and second-nearest distance to it; ball speed; mean player speed; how far the
   ball is outside the pitch; each team's centroid and spread.
2. Team evidence (`team_logodds`): the nearest-distance difference, weighted down when nobody is near
   and when the ball flies. With the ball missing, a shape model (`shape_logodds`) takes over: a
   logistic model on team shape, fitted per half on that half's own clear contacts (pseudo-labels).
3. Dead-ball evidence (`dead_llr`): ball off the pitch, a resting ball among slow players, a missing
   ball among slow players, and a two-group fit of collective player speed per half (slow and fast,
   with no fixed threshold). A missing ball alone is never dead.
4. Learned layer (`train`, `learned_logp`): a gradient-boosted classifier on windowed features, trained
   on the open games with home and away swapped as well, and applied both ways round and averaged, so it
   cannot favour either team. Missing values are filled the same way in training and in use.
5. Decoding (`label`): one Viterbi pass per half over H, A and D with switch costs. Short possessions
   are kept: tracking-derived labels are fragmented (DFL: 38-50% of possessions under 3 s), and merging
   those under 2 s cost 0.02 on all seven DFL matches. `absorb_short` stays for anyone who wants it.

Every fitted step is deterministic: fixed seeds, and callers run it on one thread (run.py).
"""
import numpy as np

from mtah.config import FPS

H, A, D, X = 0, 1, 2, 3
MIN_SPELL_FRAMES = 0  # no merge: 2 s cost 0.02 on the DFL matches and 0.039 on Metrica's
W_RULE, W_LEARNED = 1.0, 0.5
SWITCH_TEAM, SWITCH_DEAD = 14.0, 30.0
MISSING = -1.0  # how a missing feature value reaches the learned layer, in training and in use
SHAPE_WEIGHT = 0.6

# Teammates' signals (team_inputs). Held out on the ten open matches, R2's shape model plus both their
# input sets gained +0.019 spellF1 as trained (better on 10/10) and +0.020 with the ball displaced (9/10).
SHAPE_SOURCE = "r2"  # "r2": R2's per-half shape model on lost frames; "ours": the earlier one
EXTRA: tuple[str, ...] = ("r2", "r3")  # learned-layer inputs: "r2" (R2's four columns), "r3" (R3's receptions)
R2_SHAPE_COLUMN = 2  # "r2_shape" in mtah/r2_features.FEATURE_NAMES
TEAM_INPUT_SIGNS = {"r2": np.array([-1.0, -1.0, -1.0, -1.0]), "r3": np.array([-1.0, 1.0, -1.0])}
R3_SINCE_CAP_S = 30.0
R3_BALANCE_S = 10.0


def _roll_mean(x, w):
    """Centred rolling mean over 2w+1 frames, ignoring NaN."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    total = np.concatenate(([0], np.cumsum(np.where(ok, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    n = count[hi] - count[lo]
    return np.where(n > 0, (total[hi] - total[lo]) / np.maximum(n, 1), np.nan)


def clean_ball(ball, max_gap=12, spike_ms=45.0):
    """Drop samples that jump away faster than spike_ms and back, fill gaps up to max_gap frames."""
    b = np.array(ball, dtype=float)
    seen = np.isfinite(b[:, 0])
    idx = np.flatnonzero(seen)
    if len(idx) > 2:
        speed = np.linalg.norm(np.diff(b[idx], axis=0), axis=1) / np.maximum(np.diff(idx), 1) * FPS
        bad = np.flatnonzero((speed[:-1] > spike_ms) & (speed[1:] > spike_ms)) + 1
        b[idx[bad]] = np.nan
        seen = np.isfinite(b[:, 0])
    out = b.copy()
    idx = np.flatnonzero(seen)
    if len(idx) > 1:
        gaps = np.diff(idx)
        for k in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
            a, z = idx[k], idx[k + 1]
            t = np.arange(a + 1, z)
            out[t] = b[a] + ((t - a) / (z - a))[:, None] * (b[z] - b[a])
    return out, seen


def attack_dir(players, team, lo, hi, length):
    """+1 if home defends the left goal in [lo, hi), else -1: the side of home's most extreme player."""
    part = players[lo:hi][:, team == 0, 0]
    extremes = []
    for row in part[:: max(1, (hi - lo) // 2000)]:
        row = row[np.isfinite(row)]
        if len(row):
            extremes.append(row[np.argmax(np.abs(row - length / 2))] - length / 2)
    return 1.0 if np.nanmean(extremes) < 0 else -1.0


def frame_features(ball, players, team, pitch):
    length, width = pitch
    T = len(ball)
    bc, seen = clean_ball(ball)
    vis = np.isfinite(bc[:, 0])
    dist = np.linalg.norm(players - bc[:, None, :], axis=2)
    dist[np.logical_not(vis)] = np.nan
    big = 1e3
    dh = np.sort(np.where(np.isfinite(dist[:, team == 0]), dist[:, team == 0], big), axis=1)
    da = np.sort(np.where(np.isfinite(dist[:, team == 1]), dist[:, team == 1], big), axis=1)
    d1h, d1a, d2h, d2a = dh[:, 0].copy(), da[:, 0].copy(), dh[:, 1].copy(), da[:, 1].copy()
    for v in (d1h, d1a, d2h, d2a):
        v[(v >= big) | np.logical_not(vis)] = np.nan

    v = np.zeros(T)
    v[2:-2] = np.linalg.norm(bc[4:] - bc[:-4], axis=1) / 4 * FPS
    v[np.logical_not(vis)] = np.nan
    bspeed = _roll_mean(v, 2)

    k = 4  # player speed over +-k frames, so jitter does not count as running
    step = np.full(players.shape[:2], np.nan)
    step[k:-k] = np.linalg.norm(players[2 * k :] - players[: -2 * k], axis=2) / (2 * k) * FPS
    step[step > 12] = np.nan  # slot switches and glitches
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pspeed = _roll_mean(np.nanmean(step, axis=1), 12) if np.isfinite(step).any() else np.full(T, np.nan)

            def shape(side):
                px, py = players[:, team == side, 0], players[:, team == side, 1]
                return np.nanmean(px, axis=1), np.nanstd(px, axis=1), np.nanstd(py, axis=1)

            hx, hsx, hsy = shape(0)
            ax, asx, asy = shape(1)

    out_m = np.maximum.reduce([-bc[:, 0], bc[:, 0] - length, -bc[:, 1], bc[:, 1] - width])
    out_m[np.logical_not(vis)] = np.nan
    n_vis = np.isfinite(players[:, :, 0]).sum(1)
    return dict(bc=bc, seen=seen, vis=vis, d1h=d1h, d1a=d1a, d2h=d2h, d2a=d2a, bspeed=bspeed,
                pspeed=pspeed, out_m=out_m, hx=hx, hsx=hsx, hsy=hsy, ax=ax, asx=asx, asy=asy, n_vis=n_vis)


def _two_groups(x, iters=40):
    """A two-component 1-D Gaussian mixture fitted by EM: means, spreads and weights."""
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [20, 80])
    m, s, w = np.array([lo, hi]), np.array([x.std() / 2 + 1e-3] * 2), np.array([0.3, 0.7])
    for _ in range(iters):
        p = w * np.exp(-0.5 * ((x[:, None] - m) / s) ** 2) / s
        p /= p.sum(1, keepdims=True) + 1e-12
        w = p.mean(0)
        m = (p * x[:, None]).sum(0) / p.sum(0)
        s = np.sqrt((p * (x[:, None] - m) ** 2).sum(0) / p.sum(0)) + 1e-3
    return m, s, w


def team_logodds(f, scale=1.8, clip=3.0, fast=10.0):
    """Positive favours home: nearest-distance difference, weighted by how near, damped in flight."""
    lo = np.clip((f["d1a"] - f["d1h"]) / scale, -clip, clip)
    reach = np.exp(-np.nan_to_num(np.fmin(f["d1h"], f["d1a"]), nan=50.0) / 7.0)
    damp = np.where(np.nan_to_num(f["bspeed"], nan=0) > fast, 0.2, 1.0)
    return np.nan_to_num(lo * reach * damp, nan=0.0)


def shape_logodds(f, halves, weight=0.6):
    """Ball-free evidence: per half, a logistic model on team shape fitted on that half's clear contacts."""
    from sklearn.linear_model import LogisticRegression

    out = np.zeros(len(f["vis"]))
    for lo, hi in halves:
        part = slice(lo, hi)
        feats = np.column_stack([f["hx"], f["ax"], f["hsx"], f["asx"], f["hsy"], f["asy"], f["hx"] - f["ax"]])[part]
        d1h, d1a = f["d1h"][part], f["d1a"][part]
        y = np.where((d1h < 1.5) & (d1a > 3.0), 1, np.where((d1a < 1.5) & (d1h > 3.0), 0, -1))
        ok = np.isfinite(feats).all(1)
        train = ok & (y >= 0)
        if train.sum() < 200 or len(np.unique(y[train])) < 2:
            continue
        mu, sd = feats[train].mean(0), feats[train].std(0) + 1e-6
        model = LogisticRegression(C=0.5, max_iter=300).fit((feats[train] - mu) / sd, y[train])
        z = np.zeros(hi - lo)
        z[ok] = model.decision_function((feats[ok] - mu) / sd)
        out[part] = np.clip(z, -2, 2) * weight
    return out


def dead_llr(f, halves):
    """Positive favours dead ball."""
    T = len(f["vis"])
    llr = np.where(np.nan_to_num(f["out_m"], nan=-9) > 0.5, 3.0, 0.0)
    llr += np.where(np.nan_to_num(f["bspeed"], nan=0) > 6.0, -1.5, 0.0)  # a fast ball on the pitch is live
    ps = f["pspeed"]
    for lo, hi in halves:
        x = np.log(np.maximum(ps[lo:hi], 0.05))
        if np.isfinite(x).sum() < 1000:
            continue
        m, s, w = _two_groups(x)
        slow = int(np.argmin(m))
        p_slow = w[slow] * np.exp(-0.5 * ((x - m[slow]) / s[slow]) ** 2) / s[slow]
        p_fast = w[1 - slow] * np.exp(-0.5 * ((x - m[1 - slow]) / s[1 - slow]) ** 2) / s[1 - slow]
        llr[lo:hi] += np.clip(np.nan_to_num(np.log((p_slow + 1e-9) / (p_fast + 1e-9)), nan=0.0), -2.5, 2.5)
    still = (np.nan_to_num(f["bspeed"], nan=9) < 0.4) & (np.nan_to_num(ps, nan=9) < 2.0) & f["vis"]
    llr += np.where(still, 1.0, 0.0)
    missing_slow = np.logical_not(f["vis"]) & (np.nan_to_num(ps, nan=9) < 1.2)  # weak: missing alone is not dead
    llr += np.where(missing_slow, 0.8, 0.0)
    assert len(llr) == T
    return llr


def team_inputs(ball, players, team, pitch, halves):
    """The teammates' per-frame signals the options below ask for: {"r2": [T, 4], "r3": [T, 3]}.

    "r2": R2's fusion_features (mtah/r2_features.py): nearest-player evidence, time-to-ball lead, and
    the half's own shape-model logit on every frame and on lost frames. "r3": from R3's receptions
    (mtah/touches.py), see r3_columns. Every column is home-positive except r3's "since", so swapping
    the teams negates all but that one (TEAM_INPUT_SIGNS).
    """
    out = {}
    if SHAPE_SOURCE == "r2" or "r2" in EXTRA:
        from mtah.r2_features import fusion_features

        out["r2"] = np.asarray(fusion_features(ball, players, team, pitch, halves)[0], dtype=float)
    if "r3" in EXTRA:
        out["r3"] = r3_columns(ball, players, team, halves)
    return out


def r3_columns(ball, players, team, halves):
    """[T, 3] from R3's receptions (deflections never flip possession): +1/-1 for the team that made the
    half's latest reception (0 before the first), seconds since it (capped), and home minus away
    receptions over the last R3_BALANCE_S."""
    from mtah.touches import touches

    T = len(ball)
    receptions = [e for e in touches(ball, players, team) if e.kind == "reception"]
    signed = np.zeros(T)
    for e in receptions:
        signed[e.frame] += 1.0 if e.team == 0 else -1.0
    total = np.concatenate([[0.0], np.cumsum(signed)])
    idx = np.arange(T)
    balance = total[idx + 1] - total[np.maximum(idx + 1 - int(R3_BALANCE_S * FPS), 0)]
    last, since = np.zeros(T), np.full(T, R3_SINCE_CAP_S)
    for a, b in halves:
        current, at = 0.0, None
        made = {e.frame: (1.0 if e.team == 0 else -1.0) for e in receptions if a <= e.frame < b}
        for t in range(a, b):
            if t in made:
                current, at = made[t], t
            last[t] = current
            since[t] = R3_SINCE_CAP_S if at is None else min((t - at) / FPS, R3_SINCE_CAP_S)
    return np.column_stack([last, since, balance])


def extra_columns(inputs):
    """(columns, sign under a team swap) for EXTRA, in its order; None when EXTRA is empty."""
    if not EXTRA:
        return None
    cols = [inputs[name] for name in EXTRA]
    signs = [TEAM_INPUT_SIGNS[name] for name in EXTRA]
    return np.column_stack(cols), np.concatenate(signs)


def learned_features(f, players, team, pitch, halves, swap=False, extra=None):
    """The learned layer's inputs. swap=True exchanges home and away (labels must be swapped too).
    extra: optional (columns, signs) from extra_columns, appended after ours."""
    length, width = pitch
    T = len(f["vis"])
    d1h, d1a, d2h, d2a = (f["d1a"], f["d1h"], f["d2a"], f["d2h"]) if swap else (f["d1h"], f["d1a"], f["d2h"], f["d2a"])
    cols = {"d1h": d1h, "d1a": d1a, "d2h": d2h, "d2a": d2a, "dd": d1a - d1h, "bspeed": f["bspeed"],
            "pspeed": f["pspeed"], "out": f["out_m"], "vis": f["vis"].astype(float), "nvis": f["n_vis"].astype(float)}
    bx = f["bc"][:, 0]
    vx = np.gradient(np.where(np.isfinite(bx), bx, np.nan)) * FPS
    sign = np.ones(T)
    for lo, hi in halves:
        sign[lo:hi] = attack_dir(players, team, lo, hi, length) * (-1 if swap else 1)
    cols["vx_att"] = _roll_mean(vx * sign, 6)
    cols["x_att"] = np.where(sign > 0, bx, length - bx) / length
    cols["y"] = f["bc"][:, 1] / width
    for w in (12, 37, 100):
        cols[f"dd_m{w}"] = _roll_mean(cols["dd"], w)
        cols[f"ps_m{w}"] = _roll_mean(f["pspeed"], w)
        cols[f"bs_m{w}"] = _roll_mean(f["bspeed"], w)
        cols[f"vis_m{w}"] = _roll_mean(f["vis"].astype(float), w)
        cols[f"d1h_m{w}"] = _roll_mean(d1h, w)
        cols[f"d1a_m{w}"] = _roll_mean(d1a, w)
    cols["shape_dx"] = (f["ax"] - f["hx"]) * (-1 if swap else 1) * sign
    names = sorted(cols)
    feats = np.nan_to_num(np.column_stack([np.asarray(cols[n], float) for n in names]), nan=MISSING)
    if extra is not None:
        more, signs = extra
        feats = np.column_stack([feats, more * signs if swap else more])
        names = names + [f"extra{k}" for k in range(more.shape[1])]
    return feats, names


def swap_codes(codes):
    out = codes.copy()
    out[codes == H], out[codes == A] = A, H
    return out


def training_rows(gm):
    """One game's (features, codes) pairs, as it is and with home and away swapped. Graded frames only."""
    f = frame_features(gm["ball"], gm["players"], gm["team"], gm["pitch"])
    extra = extra_columns(team_inputs(gm["ball"], gm["players"], gm["team"], gm["pitch"], gm["halves"])) if EXTRA else None
    rows = []
    for swap in (False, True):
        feats, _ = learned_features(f, gm["players"], gm["team"], gm["pitch"], gm["halves"], swap=swap, extra=extra)
        y = swap_codes(gm["codes"]) if swap else gm["codes"]
        keep = y != X
        rows.append((feats[keep], y[keep]))
    return rows


def fit(rows):
    """Fit the learned layer on training_rows output, in the order given."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    return model.fit(np.vstack([x for x, _ in rows]), np.concatenate([y for _, y in rows]))


def train(games):
    """Fit the learned layer. games: dicts with ball, players, team, pitch, halves and codes (0-3)."""
    return fit([r for gm in games for r in training_rows(gm)])


def learned_logp(model, f, players, team, pitch, halves, extra=None):
    """Log-probabilities [T, 3] for H, A, D, averaged over both team orientations."""
    straight, _ = learned_features(f, players, team, pitch, halves, swap=False, extra=extra)
    swapped, _ = learned_features(f, players, team, pitch, halves, swap=True, extra=extra)
    p = 0.5 * (model.predict_proba(straight) + model.predict_proba(swapped)[:, [1, 0, 2]])
    return np.log(np.clip(p, 1e-4, 1))


def viterbi(score, switch_team=SWITCH_TEAM, switch_dead=SWITCH_DEAD):
    """Best H/A/D path for per-frame log-evidence score [T, 3] under switch costs."""
    T = len(score)
    trans = np.array([[0, switch_team, switch_dead], [switch_team, 0, switch_dead],
                      [switch_dead, switch_dead, 0]], float)
    cost = np.empty((T, 3))
    back = np.zeros((T, 3), np.int8)
    cost[0] = -score[0]
    for t in range(1, T):
        c = cost[t - 1][:, None] + trans
        back[t] = np.argmin(c, axis=0)
        cost[t] = c[back[t], np.arange(3)] - score[t]
    path = np.empty(T, np.int8)
    path[-1] = np.argmin(cost[-1])
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def absorb_short(path, min_frames=2 * FPS):
    """A possession under min_frames between two of the other team's, with no dead ball between, joins them."""
    p = path.copy()
    change = np.flatnonzero(np.diff(p)) + 1
    starts, ends = np.concatenate(([0], change)), np.concatenate((change, [len(p)]))
    for k in range(1, len(starts) - 1):
        s, e = starts[k], ends[k]
        cur, prev, nxt = p[s], p[starts[k - 1]], p[starts[k + 1]]
        if cur in (H, A) and prev == nxt and prev in (H, A) and prev != cur and e - s < min_frames:
            p[s:e] = prev
    return p


def label(ball, players, team, pitch, halves, model=None, known_dead=None, min_spell=MIN_SPELL_FRAMES):
    """Codes [T]: 0 H, 1 A, 2 D inside halves ([start, end) pairs), 3 X outside them.

    known_dead: optional bool [T], frames that are D by a published fact (the second-half block).
    """
    ball, players, team = np.asarray(ball, float), np.asarray(players, float), np.asarray(team)
    T = len(ball)
    f = frame_features(ball, players, team, pitch)
    inputs = team_inputs(ball, players, team, pitch, halves) if (SHAPE_SOURCE == "r2" or EXTRA) else {}
    if SHAPE_SOURCE == "r2":  # R2's per-half shape model, scaled exactly as ours is
        shape = np.clip(inputs["r2"][:, R2_SHAPE_COLUMN], -2, 2) * SHAPE_WEIGHT
    else:
        shape = shape_logodds(f, halves, SHAPE_WEIGHT)
    lo = team_logodds(f) + shape * np.logical_not(f["vis"])
    score = np.zeros((T, 3))
    score[:, H], score[:, A] = W_RULE * lo / 2, -W_RULE * lo / 2
    score[:, D] = W_RULE * dead_llr(f, halves)
    if model is not None:
        score += W_LEARNED * learned_logp(model, f, players, team, pitch, halves, extra_columns(inputs) if EXTRA else None)
    if known_dead is not None:
        score[known_dead] = [-50, -50, 50]
    codes = np.full(T, X, np.int8)
    for a, b in halves:
        path = viterbi(score[a:b])
        codes[a:b] = absorb_short(path, min_spell) if min_spell > 0 else path
    return codes
