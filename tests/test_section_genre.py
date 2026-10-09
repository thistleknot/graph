"""Pins src/section_genre.py on a planted world whose genre axes and subjects are known.

Spec: approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md, arm B, task T160. Synthetic vectors, no disk.

Run:  pytest tests/test_section_genre.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import section_genre as sg


def _world(n_paper=40, n_sec=8, d=24, seed=0):
    """Each section = a SUBJECT vector (the paper's subject, shared with the other papers on that subject) + a GENRE vector (one of 3 shared axes, strong)
    + noise. Subjects: 4, assigned per paper (p % 4), so within a paper only genre and noise vary. Genre: section j of every paper has genre j % 3."""
    rng = np.random.default_rng(seed)
    subj = rng.normal(size=(4, d))
    genre = np.linalg.qr(rng.normal(size=(d, 3)))[0].T * 6.0            # 3 orthogonal genre axes, strong
    rows, paper, subject, g = [], [], [], []
    for p in range(n_paper):
        for j in range(n_sec):
            s = p % 4
            rows.append(subj[s] * 1.5 + genre[j % 3] * rng.choice([-1, 1]) + rng.normal(size=d) * 0.3)
            paper.append(p); subject.append(s); g.append(j % 3)
    return np.array(rows, np.float32), np.array(paper), np.array(subject), np.array(g), genre


def test_the_top_axes_are_the_planted_genre_axes_and_carry_most_of_the_within_paper_variance():
    X, paper, _, _, genre = _world()
    V, share = sg.role_axes(X, paper, 3)
    assert V.shape == (24, 3) and np.allclose(V.T @ V, np.eye(3), atol=1e-8)
    assert share.sum() > 0.8 and (np.diff(share) <= 1e-12).all()                  # strongest first, most of the variance
    assert np.linalg.norm(V @ V.T @ genre.T - genre.T) < 0.2 * np.linalg.norm(genre)   # the recovered span contains the planted genre axes


def test_removing_the_axes_removes_genre_and_keeps_subject():
    X, paper, subject, genre_id, _ = _world()
    V, _ = sg.role_axes(X, paper, 3)
    Y = sg.remove_axes(X, V)
    assert np.allclose(np.linalg.norm(Y, axis=1), 1.0, atol=1e-5)
    def neighbours(Z):                                                             # each row's 10 nearest OTHER-PAPER rows (same-paper pairs are excluded in the real pass)
        S = Z @ Z.T
        S[paper[:, None] == paper[None, :]] = -np.inf
        return np.argsort(-S, axis=1)[:, :10]
    def share(nn, labels):
        return float((labels[nn] == labels[:, None]).mean())
    before, after = neighbours(X), neighbours(Y)
    assert share(before, genre_id) > 0.8                                           # before: a row's neighbours are of its own genre (genre dominates)
    assert share(after, genre_id) < 0.5                                            # after: neighbours come from every genre (chance is 1/3)
    assert share(after, subject) > 0.8                                             # and they are of the same subject


def test_k_zero_leaves_every_row_pointing_the_same_way():
    X, paper, _, _, _ = _world()
    V, share = sg.role_axes(X, paper, 0)
    assert V.shape == (24, 0) and share.shape == (0,)
    Y = sg.remove_axes(X, V)
    cos = (Y * (X / np.linalg.norm(X, axis=1, keepdims=True))).sum(1)
    assert np.allclose(cos, 1.0, atol=1e-5)


def test_the_axes_change_the_neighbour_lists_so_the_k_sweep_is_live():
    X, paper, _, _, _ = _world()
    X0 = sg.remove_axes(X, sg.role_axes(X, paper, 0)[0])
    X8 = sg.remove_axes(X, sg.role_axes(X, paper, 8)[0])
    def top(Z):
        S = Z @ Z.T
        np.fill_diagonal(S, -np.inf)
        return np.argsort(-S, axis=1)[:, :15]
    a, b = top(X0), top(X8)
    same = np.mean([len(set(a[i]) & set(b[i])) / 15 for i in range(len(a))])
    assert same < 0.9                                                              # k = 0 and k = 8 give different neighbours: the axis reaches the output


def test_a_paper_with_one_section_contributes_nothing_and_k_beyond_the_dimension_is_refused():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 6)).astype(np.float32)
    V, share = sg.role_axes(X, np.arange(5), 2)                                    # every paper has one section: all deviations are zero
    assert np.allclose(share, 0.0) and V.shape == (6, 2)
    with pytest.raises(AssertionError):
        sg.role_axes(X, np.arange(5), 7)


def test_genre_fraction_is_the_share_of_the_norm_inside_the_axes_and_genre_flags_average_it_per_community():
    V = np.eye(4)[:, :2]                                                              # the span of the first two coordinates
    X = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0.6, 0, 0.8, 0], [0, 0.8, 0.6, 0]])
    f = sg.genre_fraction(X, V)
    assert np.allclose(f, [1.0, 0.0, 0.36, 0.64])
    assert np.allclose(sg.genre_fraction(X, np.zeros((4, 0))), 0.0)                  # no axes: nothing is inside them
    lab = np.array([0, 0, 1, 1])
    score, flag = sg.genre_flags(lab, f, threshold=0.5, min_size=1)
    assert np.allclose(score, [0.5, 0.5]) and flag.tolist() == [True, True]          # the threshold is inclusive (>=)
    score, flag = sg.genre_flags(lab, f, threshold=0.51, min_size=1)
    assert flag.tolist() == [False, False]
    assert (sg.GENRE_K, sg.GENRE_THRESHOLD, sg.GENRE_MIN_SIZE) == (8, 0.533, 10)     # the pre-registered values


def test_a_community_with_an_empty_id_gets_a_zero_score_and_is_not_flagged():
    score, flag = sg.genre_flags(np.array([0, 0, 2]), np.array([0.9, 0.9, 0.9]), threshold=0.5, min_size=1)
    assert score.tolist() == [0.9, 0.0, 0.9] and flag.tolist() == [True, False, True]


def test_a_community_below_the_size_floor_is_never_flagged_however_high_its_score():
    lab = np.array([0] * 10 + [1] * 9)
    score, flag = sg.genre_flags(lab, np.full(19, 0.9))
    assert flag.tolist() == [True, False]                                            # 10 sections flagged, 9 not


def test_a_row_that_lies_wholly_in_the_removed_span_keeps_a_zero_vector_not_nan():
    V = np.eye(3)[:, :1]
    Y = sg.remove_axes(np.array([[2.0, 0, 0], [1.0, 1.0, 0]], np.float32), V)
    assert np.isfinite(Y).all() and np.allclose(Y[0], 0.0) and np.allclose(np.linalg.norm(Y[1]), 1.0)
