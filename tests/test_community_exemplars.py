"""Pins community_exemplars.py: the skill's band (R23/R24), exemplar rule (R26, 17d), redundancy
scale (R28), resolution pick (R20) and colour rule (step 7). Synthetic numbers, so every right
answer is known by construction.

Run:  pytest tests/test_community_exemplars.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from community_exemplars import (MIN_BAND, assign_colours, band, percentile_scaler, pick_resolution,
                                 select_exemplars)

NO_REDUNDANCY = lambda cands, shown: np.zeros(len(cands))


def test_percentile_scaler_is_monotone_and_spans_zero_to_one():
    f = percentile_scaler(np.linspace(0, 1, 1001))
    assert f(-1) == 0 and f(2) == 1 and f(0.5) == pytest.approx(0.5, abs=0.002)
    assert np.all(np.diff(f(np.linspace(0, 1, 50))) >= 0)


def test_band_is_one_sided_and_keeps_roughly_five_sixths_of_a_symmetric_community():
    rng = np.random.default_rng(0)
    t = np.abs(rng.normal(0.4, 0.05, 4000))                    # symmetric: no transform
    x, edge, info = band(t)
    assert info["lambda"] is None and info["edge_rule"] == "mean+sd"
    assert 0.80 < (x <= edge).mean() < 0.88                     # one-sided mean+1sd keeps ~84%


def test_band_transforms_a_skewed_distance_distribution_before_cutting():
    rng = np.random.default_rng(1)
    t = rng.lognormal(-1.5, 0.9, 4000)                         # strongly right-skewed distances
    x, edge, info = band(t)
    assert info["skew"] > 0.5 and info["lambda"] is not None
    assert abs(float(np.mean((x - x.mean()) ** 3) / x.std() ** 3)) < abs(info["skew"])


def test_a_community_too_small_to_estimate_is_all_band():
    t = np.array([0.1, 0.2, 0.9, 0.3])
    assert len(t) < MIN_BAND
    x, edge, info = band(t)
    assert (x <= edge).all() and info["edge_rule"] == "all members"


def _community(m=600, seed=2):
    rng = np.random.default_rng(seed)
    t = np.sort(np.abs(rng.normal(0.35, 0.08, m)))
    return t[rng.permutation(m)]


def test_one_exemplar_is_just_the_medoid():
    t = _community()
    assert select_exemplars(t, np.ones(len(t), bool), 1, NO_REDUNDANCY) == [("medoid", int(np.argmin(t)), 0.0)]


def test_three_exemplars_sit_near_one_third_and_two_thirds_of_the_band_never_the_edge():
    t = _community()
    out = select_exemplars(t, np.ones(len(t), bool), 3, NO_REDUNDANCY)
    assert [r for r, _, _ in out] == ["medoid", "z=1/3", "z=2/3"]
    assert out[0][1] == int(np.argmin(t))
    zs = [z for _, _, z in out]
    assert abs(zs[1] - 1 / 3) <= 1 / 6 and abs(zs[2] - 2 / 3) <= 1 / 6 and max(zs) < 1.0


def test_two_exemplars_use_the_band_midpoint():
    out = select_exemplars(_community(), np.ones(600, bool), 2, NO_REDUNDANCY)
    assert out[1][0] == "z=1/2" and abs(out[1][2] - 0.5) <= 0.25


def test_the_pool_is_the_core_and_falls_back_to_the_band_only_when_the_core_is_too_small():
    t = _community()
    core = np.zeros(len(t), bool)
    core[np.argsort(t)[:300]] = True                            # core = nearer half
    out = select_exemplars(t, core, 3, NO_REDUNDANCY)
    assert all(core[i] for _, i, _ in out[1:])                  # picks come from the core
    tiny_core = np.zeros(len(t), bool)
    tiny_core[np.argsort(t)[:2]] = True                         # fewer than n -> band is the pool
    out2 = select_exemplars(t, tiny_core, 3, NO_REDUNDANCY)
    assert len(out2) == 3


def test_it_never_reaches_outside_the_band_and_shows_fewer_when_the_pool_runs_out():
    t = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.95])    # one far outlier
    x, edge, _ = band(t)
    out = select_exemplars(t, np.ones(10, bool), 3, NO_REDUNDANCY)
    assert all(x[i] <= edge for _, i, _ in out) and 9 not in [i for _, i, _ in out]
    two = select_exemplars(np.array([0.1, 0.2]), np.ones(2, bool), 2, NO_REDUNDANCY)
    assert len(two) <= 2


def test_mmr_moves_the_pick_off_a_redundant_candidate_inside_the_window():
    t = _community()
    core = np.ones(len(t), bool)
    best = select_exemplars(t, core, 2, NO_REDUNDANCY)[1][1]          # closest to the target, no redundancy
    penalise_best = lambda cands, shown: np.where(np.asarray(cands) == best, 0.9, 0.0)
    moved = select_exemplars(t, core, 2, penalise_best)[1][1]
    assert moved != best                                              # an MMR-less selector would pick `best` again
    _, edge_z = band(t)[1], select_exemplars(t, core, 2, penalise_best)[1][2]
    assert abs(edge_z - 0.5) <= 0.25                                  # and it stayed inside the window


def test_selection_is_deterministic():
    t = _community()
    a = select_exemplars(t, np.ones(len(t), bool), 3, NO_REDUNDANCY)
    b = select_exemplars(t, np.ones(len(t), bool), 3, NO_REDUNDANCY)
    assert a == b


def test_pick_resolution_chooses_inside_the_plateau_by_serving_size():
    res = np.geomspace(0.1, 20, 10)
    n = np.array([5, 12, 30, 70, 130, 160, 240, 400, 700, 1200])
    ari = np.array([0.55, 0.80, 0.88, 0.90, 0.91, 0.91, 0.90, 0.74, 0.60, 0.50])      # flat top at 70..240
    i, plateau = pick_resolution(res, n, ari, target_n=135)
    assert plateau[4] and plateau[5] and not plateau[0] and not plateau[9]
    assert n[i] == 130                                           # nearest 135 among plateau members only


def test_pick_resolution_on_the_real_arxiv_sweep_ignores_the_degenerate_n4_spike():
    """Regression: the rule once anchored on max(ARI)=0.871 at n~4 and chose a 4-community partition."""
    res = np.geomspace(0.1, 20, 10)
    n = np.array([4.2, 5.0, 8.5, 17.2, 34.0, 57.2, 95.2, 151.5, 237.0, 404.5])
    ari = np.array([0.871, 0.273, 0.651, 0.635, 0.631, 0.680, 0.679, 0.678, 0.691, 0.665])
    i, plateau = pick_resolution(res, n, ari, target_n=135)
    assert not plateau[0] and not plateau[1]                      # the spike and the crash are not a plateau
    assert plateau[5:9].all() and not plateau[9]                  # the flat stretch: n 57 to 237
    assert n[i] == 151.5                                          # nearest the serving size inside it


def test_pick_resolution_never_leaves_the_plateau_even_when_a_better_n_lies_outside_it():
    n = np.array([10, 20, 40, 135, 300])
    ari = np.array([0.90, 0.91, 0.91, 0.50, 0.50])              # n=135 is unstable
    i, plateau = pick_resolution(np.arange(5), n, ari, target_n=135)
    assert plateau[i] and n[i] == 40


def test_neighbouring_communities_never_share_a_colour_and_the_legend_ones_all_differ():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 100, (135, 2))
    sizes = rng.integers(5, 2500, 135)
    col = assign_colours(xy, sizes, n_colours=16, k=6)
    d = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    for i in range(135):
        for j in np.argsort(d[i])[:6]:
            assert col[i] != col[j]
    top = np.argsort(-sizes, kind="stable")[:12]
    assert len(set(col[top])) == 12
