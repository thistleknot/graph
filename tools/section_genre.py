"""section_genre.py -- remove the GENRE directions from the dense view of sections, leaving what is shared across papers about a subject.

Spec: approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md, arm B, task T160 (playbook.md Layer 40). Operator 2026-10-06: "plan a proper fix".

Hypothesis under test (NOT a finding): the grab-bag communities (prompt templates, generic introduction/conclusion prose) are glued by a genre direction
that is shared across papers. A section's genre is what it IS within its paper (an introduction, a prompt, a proof), so the pooled deviation of every
section from its own paper's mean vector is dominated by genre. The top-k principal directions of that pooled deviation are the genre axes; removing
them from every section leaves the part that varies between papers, i.e. subject.

    role_axes(X, paper, k)    (V, share): the top-k eigenvectors of the pooled within-paper covariance and the share of its variance each carries
    remove_axes(X, V)         rows with the span of V projected out, L2-renormalised

k = 0 returns X unchanged (renormalised), so the k sweep has an honest zero. The sparse view has no such axis: arm B cleans only the dense side.
"""
from __future__ import annotations

import numpy as np


def role_axes(X: np.ndarray, paper: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Require: X (N, d) float rows, paper (N,) integer id per row, 0 <= k <= d. Guarantee: (V, share). V is (d, k) with orthonormal columns, the top-k
    eigenvectors of the covariance of the within-paper deviations (each row minus the mean of its paper's rows), strongest first. share[i] is the
    fraction of the total within-paper variance eigenvector i carries. A paper with one section contributes a zero deviation."""
    assert 0 <= k <= X.shape[1], "k must lie between 0 and the dimension"
    X64 = np.asarray(X, np.float64)
    codes = np.unique(paper, return_inverse=True)[1]
    sums = np.zeros((codes.max() + 1, X.shape[1]))
    np.add.at(sums, codes, X64)
    D = X64 - (sums / np.bincount(codes)[:, None])[codes]
    C = D.T @ D
    w, V = np.linalg.eigh(C)
    order = np.argsort(-w)[:k]
    return V[:, order], (w[order] / max(float(w.sum()), 1e-30))


GENRE_K, GENRE_THRESHOLD, GENRE_MIN_SIZE = 8, 0.533, 10   # the flag: a community of at least 10 sections whose sections lie at least this far inside the top-8 within-paper axes (mean squared-norm share) is a GENRE block


def genre_fraction(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Spec: plan task T168. Guarantee: per row of X (unit-length rows), the share of its squared norm inside the span of V, in [0, 1]; a zero-column V gives zeros."""
    return ((np.asarray(X, np.float64) @ V) ** 2).sum(1) if V.shape[1] else np.zeros(len(X))


def genre_flags(lab: np.ndarray, frac: np.ndarray, threshold: float = GENRE_THRESHOLD, min_size: int = GENRE_MIN_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Spec: plan task T168. Guarantee: (score, flag), one entry per community id 0..max: score = the mean of `frac` over the community's sections, flag = score >= threshold AND
    the community has at least `min_size` sections (every graded community had 10 or more; a smaller one's mean is one section's noise).
    THRESHOLD PRE-REGISTERED 2026-10-06 on 14 hand-graded communities (6 topic, max 0.531; 8 genre, min 0.535), then VALIDATED on 18 held-out ones graded blind: no topic or mixed
    community flagged (0 of 7), every genre community on the cross-paper map caught (4 of 4) but only 3 of 7 on the map with the genre axes removed, whose genre blocks are held together
    by what these axes cannot see."""
    n = int(lab.max()) + 1
    size = np.bincount(lab, minlength=n)
    score = np.bincount(lab, weights=frac, minlength=n) / np.maximum(size, 1)
    return score, (score >= threshold) & (size >= min_size)


def remove_axes(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Guarantee: X with the span of V projected out of every row, rows L2-normalised (float32). V with no columns leaves the direction of every row
    unchanged. A row left with no length keeps its zero vector."""
    Y = np.asarray(X, np.float64)
    if V.shape[1]:
        Y = Y - (Y @ V) @ V.T
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    return (Y / np.maximum(n, 1e-12)).astype(np.float32)
