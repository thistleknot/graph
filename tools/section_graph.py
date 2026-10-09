"""section_graph.py -- the significant correlations between sections as ONE graph, computed exactly and streamed (never n x n).

Spec: operator 2026-10-05 ("pure normalized embeddings/sparsevec -> inner dot product then a correlation matrix over both representations as
their own matrices, and identify edges between sections based on significant correlations ... take whatever SET of significant relations we
find as edges for one graph"; "I don't want to use hnsw anymore"); approved plan step B3. Task: playbook.md T153.

It is chunkgraph.py's pipeline (guard R7 every node keeps its nearest neighbours, R10 thresholded inside the block loop, `_bc` Box-Cox on a seeded
subsample, `_cut_pooled` z of the positive similarities, the union of the two spaces with provenance and strength = mean z) re-written so no
n x n matrix exists: chunkgraph's `_sparse_sim` allocates np.zeros((n, n)), 54 GB at 82,000 sections.

    fit_null(X)        a seeded sample of rows against all rows estimates the null: Box-Cox lambda, mean and sd of the transformed positive
                       similarities, kurtosis (the cut's z for any similarity s is (boxcox(s - min + 1e-3) - mu) / sd)
    stream_edges(X)    every pair exactly, blockwise. A significant EDGE is a pair with s >= the k-sigma threshold in which one end is among the
                       other's exact top-`topk` neighbours. The per-node cap replaces chunkgraph's global budget (guard R2): measured on the 82,542
                       sections, a global top-N by similarity is 91% same-paper near-duplicates and boilerplate (2026-10-05), while the cut alone passes
                       30.4 million pairs (0.9%). The NEGATIVE tail is counted (pairs with s <= -threshold), not assumed to mean antonyms.
    fuse(...)          the union of the representations' significant pairs and backbones as one symmetric weighted matrix with provenance
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy import stats

SIM_FLOOR, KURT_OK, KNN, K_SIGMA = 0.02, 0.5, 2, 2.0      # chunkgraph.py: R10 in-loop floor, R2 kurtosis gate (reported, no longer a switch), R7 backbone neighbours per space, default k-sigma
BC_FIT_SAMPLE, BC_FIT_SEED = 200_000, 0
EPS = 1e-3                                                # chunkgraph._bc: xs = x - x.min() + 1e-3
EDGE_FLOOR = 1e-3                                         # a backbone edge below the cut still carries this much weight, so no node is isolated


@dataclass
class Cut:
    lam: float
    xmin: float
    mu: float
    sd: float
    kurt: float
    n_pairs_seen: int           # ordered pairs the sample covered
    frac_positive: float

    def z(self, s: np.ndarray) -> np.ndarray:
        """Guarantee: z of similarities `s` under the null; s at or below the sample minimum maps to the bottom of the scale."""
        xs = np.maximum(np.asarray(s, np.float64) - self.xmin + EPS, EPS)
        y = np.log(xs) if abs(self.lam) < 1e-9 else (xs ** self.lam - 1.0) / self.lam
        return (y - self.mu) / self.sd

    def threshold(self, k: float) -> float:
        """Guarantee: the similarity whose z is k (the inverse of `z`, by bisection on the monotone transform); 1.0 when no cosine reaches k."""
        lo, hi = self.xmin, 1.0 + 1e-9
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if float(self.z(np.array([mid]))[0]) < k else (lo, mid)
        return float(hi)


def _transpose(X):
    """Guarantee: X.T in the layout a product with X[idx] wants (CSR for a sparse X, so the conversion happens once, not once per block)."""
    return X.T.tocsr() if sp.issparse(X) else X.T


def _rows(X, idx: np.ndarray, XT=None) -> np.ndarray:
    S = X[idx] @ (_transpose(X) if XT is None else XT)
    return (S.toarray() if sp.issparse(S) else np.asarray(S)).astype(np.float32)


def fit_null(X, rows: int = 2000, seed: int = BC_FIT_SEED) -> Cut:
    """Require: X rows L2-normalised (dense float32 array or CSR). Guarantee: the null of the positive similarities from `rows` seeded rows
    against every row (self excluded); Box-Cox as chunkgraph._bc: shifted to positive, lambda by MLE on a seeded subsample of BC_FIT_SAMPLE."""
    N = X.shape[0]
    R = np.sort(np.random.default_rng(seed).choice(N, min(rows, N), replace=False))
    vals, seen, pos = [], 0, 0
    XT = _transpose(X)
    for a in range(0, len(R), 256):
        sub = R[a:a + 256]
        S = _rows(X, sub, XT)
        S[np.arange(len(sub)), sub] = 0.0                                    # no self pairs
        seen += S.size - len(sub)
        v = S[S > SIM_FLOOR]
        pos += v.size
        vals.append(v)
    x = np.concatenate(vals).astype(np.float64)
    xs = x - x.min() + EPS
    fit = xs if xs.size <= BC_FIT_SAMPLE else np.random.default_rng(BC_FIT_SEED).choice(xs, BC_FIT_SAMPLE, replace=False)
    lam = float(stats.boxcox_normmax(fit))
    y = stats.boxcox(xs, lmbda=lam)
    return Cut(lam, float(x.min()), float(y.mean()), float(y.std()), float(stats.kurtosis(y)), seen, pos / max(seen, 1))


def stream_edges(X, cut: Cut, k_sigma: float = K_SIGMA, topk: int = 15, block: int = 512, log=None, group=None) -> dict:
    """Require: X rows L2-normalised; `cut` from fit_null; 1 <= topk <= N-1; `group` None or one integer per row (the paper). With `group`, a pair whose
    rows share a group is not a pair at all: not a neighbour, not an edge, not counted in either tail (measured 2026-10-06: half of the section
    communities were one paper's own outline, bound by that paper's coined vocabulary; a topic is what DIFFERENT papers say alike).
    Guarantee, every pair computed exactly, blockwise (guard R10):
      knn_idx, knn_sim   each row's exact `topk` neighbours, strongest first (self excluded)
      i, j, s            the significant edges (i < j, unique): s >= the k-sigma threshold and one end in the other's top-`topk`
      stats              threshold, kurtosis, the positive and negative tail counts over ALL pairs, the number of significant edges, nodes with
                         none, and the strongest negative pairs"""
    N = X.shape[0]
    k = min(topk, N - 1)
    thr = cut.threshold(k_sigma)
    knn_idx, knn_sim = np.zeros((N, k), np.int32), np.zeros((N, k), np.float32)
    n_pos, n_neg = 0, 0
    neg_i, neg_j, neg_s = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    cols = np.arange(N)
    XT = _transpose(X)
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        S = _rows(X, np.arange(i0, i1), XT)
        rr = np.arange(i1 - i0)
        S[rr, i0 + rr] = -np.inf                                             # no self pairs
        if group is not None:
            S[group[i0:i1, None] == group[None, :]] = -np.inf                # nor same-paper pairs
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(S, part, 1)
        order = np.argsort(-sims, axis=1)
        knn_idx[i0:i1] = np.take_along_axis(part, order, 1)
        knn_sim[i0:i1] = np.take_along_axis(sims, order, 1)
        upper = cols[None, :] > (i0 + rr)[:, None]
        n_pos += int((upper & (S >= thr)).sum())
        nr, nc = np.nonzero(upper & (S <= -thr))
        n_neg += nr.size
        if nr.size:
            neg_i = np.concatenate([neg_i, i0 + nr]); neg_j = np.concatenate([neg_j, nc]); neg_s = np.concatenate([neg_s, S[nr, nc]])
            if neg_s.size > 5000:
                keep = np.argsort(neg_s)[:1000]
                neg_i, neg_j, neg_s = neg_i[keep], neg_j[keep], neg_s[keep]
        if log and (i0 // block) % 20 == 0:
            log("block %d/%d" % (i0 // block + 1, -(-N // block)))
    a = np.repeat(np.arange(N), k)
    b = knn_idx.ravel().astype(np.int64)
    sig = knn_sim.ravel() >= thr
    lo, hi = np.minimum(a, b)[sig], np.maximum(a, b)[sig]
    uniq, first = np.unique(lo * N + hi, return_index=True)
    ei, ej, es = uniq // N, uniq % N, knn_sim.ravel()[sig][first]
    deg = np.bincount(np.concatenate([ei, ej]), minlength=N)
    keep = np.argsort(neg_s)[:50]
    return {"i": ei.astype(np.int64), "j": ej.astype(np.int64), "s": es.astype(np.float32), "knn_idx": knn_idx, "knn_sim": knn_sim,
            "stats": {"k_sigma": k_sigma, "threshold": thr, "kurtosis": cut.kurt, "kurtosis_ok": cut.kurt <= KURT_OK, "lam": cut.lam, "topk": k,
                      "positive_tail": n_pos, "negative_tail": n_neg, "neg_threshold": -thr, "significant_edges": int(es.size),
                      "nodes_without_edge": int((deg == 0).sum()), "mean_degree": float(2 * es.size / N),
                      "strongest_negative": [(int(neg_i[t]), int(neg_j[t]), float(neg_s[t])) for t in keep]}}


def pair_sim(X, i: np.ndarray, j: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Guarantee: the exact inner product of rows i[t] and j[t] for every pair t (dense or CSR rows)."""
    out = np.zeros(len(i), np.float32)
    for a in range(0, len(i), chunk):
        ii, jj = i[a:a + chunk], j[a:a + chunk]
        out[a:a + chunk] = (np.asarray(X[ii].multiply(X[jj]).sum(1)).ravel() if sp.issparse(X) else np.einsum("ij,ij->i", X[ii], X[jj]))
    return out


def fuse(N: int, reps: dict, backbone: int = KNN) -> dict:
    """Require: reps = {name: {"X": rows, "cut": Cut, "edges": stream_edges' output}}. Guarantee: the union of every representation's
    significant pairs and each node's `backbone` nearest neighbours (guard R7) as ONE symmetric weighted matrix:
      G        csr (N, N), weight = max(mean z over the representations, 0) + EDGE_FLOOR
      i, j     the union's pairs (i < j), `prov` a bitmask per pair: bit k set where representation k (in dict order) found it significant,
               `back` a bitmask where it is in that representation's backbone
      s, z     per representation, the pair's exact similarity and z (computed for every union pair, not only where it was significant)"""
    names = list(reps)
    key, flag_sig, flag_back = [], [], []
    for k, n in enumerate(names):
        e = reps[n]["edges"]
        key.append(e["i"] * N + e["j"]); flag_sig.append(np.full(len(e["i"]), 1 << k)); flag_back.append(np.zeros(len(e["i"]), np.int64))
        a = np.repeat(np.arange(N), backbone); b = e["knn_idx"][:, :backbone].ravel()
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key.append(lo.astype(np.int64) * N + hi); flag_sig.append(np.zeros(len(a), np.int64)); flag_back.append(np.full(len(a), 1 << k))
    key, fs, fb = np.concatenate(key), np.concatenate(flag_sig), np.concatenate(flag_back)
    uniq, inv = np.unique(key, return_inverse=True)
    prov, back = np.zeros(uniq.size, np.int64), np.zeros(uniq.size, np.int64)
    np.bitwise_or.at(prov, inv, fs); np.bitwise_or.at(back, inv, fb)
    i, j = uniq // N, uniq % N
    out = {"i": i, "j": j, "prov": prov, "back": back, "names": names, "s": {}, "z": {}}
    zsum = np.zeros(uniq.size)
    for n in names:
        s = pair_sim(reps[n]["X"], i, j)
        out["s"][n] = s
        z = reps[n]["cut"].z(s)
        out["z"][n] = z.astype(np.float32)
        zsum += z
    w = np.maximum(zsum / len(names), 0.0) + EDGE_FLOOR
    G = sp.csr_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(N, N), dtype=np.float32)
    out["G"], out["w"] = G, w.astype(np.float32)
    return out
