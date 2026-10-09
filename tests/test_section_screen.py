"""Pins src/section_screen.py's pure pieces: the whole-paper sample, the arm config, the same-heading share. The pipeline itself is run on real data.

Spec: approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md, task T161. Synthetic ids, no disk.

Run:  pytest tests/test_section_screen.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import section_screen as scr


def _docs():
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 40, 300)                                               # 300 papers of 1..39 sections
    return np.repeat(["p%03d" % i for i in range(300)], sizes), sizes


def test_the_sample_is_made_of_whole_papers_reaches_the_target_and_is_seeded():
    docs, sizes = _docs()
    rows = scr.pick_sample(docs, target=1500, seed=3)
    chosen = set(docs[rows])
    assert all((docs == p).sum() == (docs[rows] == p).sum() for p in chosen)        # every section of a chosen paper is in
    assert len(rows) >= 1500 and len(rows) < 1500 + 40                              # reaches the target and overshoots by less than one paper
    assert np.array_equal(rows, scr.pick_sample(docs, target=1500, seed=3)) and not np.array_equal(rows, scr.pick_sample(docs, target=1500, seed=4))
    assert (np.diff(rows) > 0).all()                                                # sorted, unique


def test_a_target_above_the_corpus_takes_everything():
    docs, _ = _docs()
    assert len(scr.pick_sample(docs, target=10**9)) == len(docs)


def test_arm_config_applies_overrides_and_refuses_a_typo():
    assert scr.arm_config({}) == scr.DEFAULTS
    assert scr.arm_config({"genre_k": 8})["genre_k"] == 8 and scr.arm_config({"genre_k": 8})["lam"] == scr.DEFAULTS["lam"]
    with pytest.raises(AssertionError):
        scr.arm_config({"genre_kk": 8})                                             # a typo must not silently run the baseline
    assert set(scr.ARMS["C_heading"]) <= set(scr.DEFAULTS) and all(set(v) <= set(scr.DEFAULTS) for v in scr.ARMS.values())


def test_parse_args_reads_a_fixed_resolution_and_refuses_an_unknown_arm():
    assert scr.parse_args(["baseline", "C_heading"]) == (["baseline", "C_heading"], None, scr.SAMPLE_SEED)
    assert scr.parse_args(["res=3.42", "baseline", "B_k8"]) == (["baseline", "B_k8"], 3.42, scr.SAMPLE_SEED)
    assert scr.parse_args(["res=1.5"]) == ([], 1.5, scr.SAMPLE_SEED)                 # no names: main runs every arm
    assert scr.parse_args(["seed=2", "res=3.42", "baseline"]) == (["baseline"], 3.42, 2)    # a different sample, for the noise floor
    with pytest.raises(AssertionError):
        scr.parse_args(["basline"])
    with pytest.raises(AssertionError):
        scr.parse_args(["res=1", "res=2", "baseline"])
    with pytest.raises(AssertionError):
        scr.parse_args(["seed=1", "seed=2", "baseline"])


def test_same_heading_share_against_the_chance_of_it():
    kind = np.array([0, 0, 1, 1, 2, -1])                                           # heading ids; -1 = no heading
    i, j = np.array([0, 2, 0, 4]), np.array([1, 3, 2, 5])
    share, base = scr.same_heading_share(i, j, kind)
    assert share == pytest.approx(2 / 3)                                           # pair (4,5) has an empty heading and is not counted; (0,1) and (2,3) match, (0,2) does not
    assert base == pytest.approx((2 / 5) ** 2 + (2 / 5) ** 2 + (1 / 5) ** 2)       # chance two sections with a heading match: sum of squared frequencies
    assert scr.same_heading_share(np.array([5]), np.array([5]), kind) == (0.0, base)   # only empty headings: share 0, not a crash
