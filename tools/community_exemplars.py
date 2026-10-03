"""community_exemplars.py -- which chunks represent a community, per sparsevec-lexsem-graph
(steps 7, 14, 14c, 17d; guards R20, R23, R24, R26, R28), file updated 2026-10-03.

NO GOVERNING SPEC beyond that skill file (repo root). Pure numpy/scipy, no I/O, so each rule is
testable on its own.

    RULE                                                                          GUARD
    resolution: inside the seed-ARI plateau, the one whose n is nearest serving   R20
    band: one-sided cut on distance-to-centroid, Box-Cox only where |skew| > .5   R23, R24
    pool: band AND a lexical-and-semantic kNN edge; fall back to band only        14c, 17d
    count n: Box-Cox of community size, min-max to [1, 3] (hnsw_communities)      R26
    picks: medoid, then targets z = k/n (k = 1..n-1) on the band's own axis       R26
    one MMR (lambda 0.5) pick per equal-width window |z - k/n| <= 1/(2n)          R26
    redundancy: product of percentile-scaled dense and lexical cosines            R28
    colours: 2D-neighbouring communities never share a hue                        step 7
"""
from __future__ import annotations

import numpy as np
from scipy import stats

SKEW_MAX = 0.5          # R24: report skew, transform only above this
MIN_BAND = 8            # below this many members mean/sd (and skew) are not estimable: the band is everyone
LAMBDA = 0.5            # R26: MMR trade-off, fixed by the skill


def percentile_scaler(null_cosines):
    """R28. Guarantee: f(c) = share of the null (random-pair) cosines <= c, in [0, 1], monotone."""
    s = np.sort(np.asarray(null_cosines, float))
    return lambda c: np.searchsorted(s, np.asarray(c, float), side="right") / len(s)


def band(t):
    """R23, R24. t = distance to the centroid per member (>= 0).
    Guarantee: (x, edge, info): x is t in its normal space (Box-Cox only where |skew| > SKEW_MAX),
    edge the ONE-SIDED cut (mean + 1 sd; median + 1.4826 MAD where skew remains; the max when the
    community is too small to estimate), and members with x <= edge are the band."""
    t = np.asarray(t, float)
    info = {"skew": float(stats.skew(t)) if len(t) > 2 else 0.0, "lambda": None, "edge_rule": "mean+sd"}
    x = t
    if len(t) >= MIN_BAND and abs(info["skew"]) > SKEW_MAX and np.ptp(t) > 0:
        x, lam = stats.boxcox(t + 1e-9)
        info["lambda"] = float(lam)
    if len(x) < MIN_BAND or np.ptp(x) == 0:
        info["edge_rule"] = "all members"
        return x, float(x.max()), info
    if abs(stats.skew(x)) > SKEW_MAX:
        info["edge_rule"] = "median+1.4826MAD"
        return x, float(np.median(x) + 1.4826 * stats.median_abs_deviation(x)), info
    return x, float(x.mean() + x.std()), info


def select_exemplars(t, core, n, redundancy):
    """R26, 17d. t: distance to the centroid per member; core: bool per member (lexical-and-semantic
    kNN edge to a same-community member); n: exemplar count (<= members); redundancy(cands, shown) ->
    array in [0, 1] of each candidate's redundancy against the shown members.
    Guarantee: [(role, member_index, z)], medoid first, never outside the band, fewer than n when the
    pool runs out, deterministic."""
    t = np.asarray(t, float)
    x, edge, info = band(t)
    in_band = x <= edge
    medoid = int(np.argmin(t))
    span = edge - x[medoid]
    z = (x - x[medoid]) / span if span > 0 else np.zeros_like(x)
    pool = in_band & np.asarray(core, bool)
    if pool.sum() < n:
        pool = in_band                                           # fallback: band only, never beyond
    shown, out = [medoid], [("medoid", medoid, 0.0)]
    for k in range(1, n):
        target, half = k / n, 1 / (2 * n)
        cand = np.array([i for i in np.where(pool)[0] if i not in shown], int)
        if len(cand) == 0:
            break
        win = cand[np.abs(z[cand] - target) <= half]
        if len(win) == 0:
            pick = int(cand[np.argmin(np.abs(z[cand] - target))])   # empty window -> nearest z
        else:
            rel = 1 - np.abs(z[win] - target) / half
            pick = int(win[np.argmax(LAMBDA * rel - (1 - LAMBDA) * redundancy(win, shown))])
        shown.append(pick)
        out.append(("z=%d/%d" % (k, n), pick, float(z[pick])))
    return out


def pick_resolution(res, n_comm, ari, target_n):
    """R20. res/n_comm/ari: one entry per swept resolution (ari = mean pairwise seed-ARI).
    Guarantee: (index of the chosen resolution, plateau mask). The plateau is the LONGEST CONTIGUOUS RUN of
    resolutions whose seed-ARI stays within the sweep's own point-to-point jitter, tol = 1.4826 * MAD of the
    successive differences (no constant; ties go to the higher mean ARI). It is a flat stretch, not the
    maximum: on the arxiv sweep the maximum (0.871) was n~4, a few giant communities that are trivially
    stable, flanked by a crash to 0.27, and anchoring on it picked a meaningless partition. Inside the
    plateau the choice is the resolution whose community count is nearest the serving size; n is an output."""
    ari, n_comm = np.asarray(ari, float), np.asarray(n_comm, float)
    tol = 1.4826 * stats.median_abs_deviation(np.diff(ari))
    best, best_key = (0, 1), (0, -np.inf)
    for i in range(len(ari)):
        j = i
        while j + 1 < len(ari) and ari[i:j + 2].max() - ari[i:j + 2].min() <= tol:
            j += 1
        key = (j - i + 1, float(ari[i:j + 1].mean()))
        if key > best_key:
            best, best_key = (i, j + 1), key
    plateau = np.zeros(len(ari), bool)
    plateau[best[0]:best[1]] = True
    cand = np.where(plateau)[0]
    return int(cand[np.argmin(np.abs(n_comm[cand] - target_n))]), plateau


def assign_colours(xy, sizes, n_colours, k=6, fixed_top=12):
    """Step 7. xy: 2D centroid per community. Guarantee: colour index per community such that no
    community shares a colour with any of its k nearest 2D neighbours, and the `fixed_top` largest
    communities all differ (they head the legend). Needs n_colours > k + 1."""
    xy = np.asarray(xy, float)
    d = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    nbr = [set(row.tolist()) for row in nn]
    for i, row in enumerate(nn):
        for j in row:
            nbr[j].add(i)
    col = -np.ones(len(xy), int)
    top_used = set()                                             # colours already given to the legend's communities
    for rank, i in enumerate(np.argsort(-np.asarray(sizes), kind="stable")):
        banned = {col[j] for j in nbr[i] if col[j] >= 0}
        if rank < fixed_top:
            banned |= top_used
        free = [c for c in range(n_colours) if c not in banned]
        assert free, "palette too small for this neighbourhood"
        col[i] = free[0]
        if rank < fixed_top:
            top_used.add(col[i])
    return col
