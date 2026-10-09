"""diag_section_sparse.py -- does the sparse representation of a section find it again, and does adjacency help?

Spec: operator 2026-10-05 (BPE tokenizer derived from the texts as the sparse tokenizer; "rerank sparsevec by adjacent tokens ... that's how we
should measure cosine"); approved plan step B2. Task: playbook.md T152. No other governing spec.

Battery (the dense one's protocol, src/diag_dense_arms.py --unit section): a seeded 20,000-section corpus, 200 queries, each ONE body sentence of
a sampled section, truth = that section (n_relevant = 1, so recall@k is hit@k), exact scoring, no index. Arms: tokens = words (stoplist.tokenize)
or BPE pieces (ingest_arxiv_sparsevec.bpe_docs, trained on the corpus's own surviving terms); adjacency weight lambda in the pre-registered grid
{0, 0.5, 1, 2}: score = <U, qU> + lambda <P, qP>, every row L2-normalised (section_sparse). Every line is paired with words, lambda = 0 (bootstrap
interval on the difference of hit@50), and the within-arm line is paired with the SAME arm at lambda = 0.

CAVEAT built into the battery: the query is a verbatim sentence of its section, so its adjacent pairs are really in the truth section; a gain from
adjacency here measures exact-phrase overlap, which is also what adjacency measures between two sections, but it overstates the gain for a query
that only paraphrases.

Run:  python -u src\\diag_section_sparse.py [--merges 10000] [--n 20000]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diag_dense_arms as dd
import section_sparse as s2
from ingest_arxiv_sparsevec import bpe_docs
from stoplist import tokenize

GRID = (0.0, 0.5, 1.0, 2.0)


def views(C: list[dict], Q: list, merges: int | None):
    """Guarantee: (U, P, QU, QP, seconds): the corpus views and the query matrices under the arm's tokenizer, statistics frozen on the corpus."""
    t0 = time.time()
    docs = [tokenize(c["text"]) for c in C]
    qtok = tokenize
    if merges:
        docs, to_pieces = bpe_docs(docs, merges)
        qtok = lambda q: to_pieces(tokenize(q))
    U, col, idf, whales, avgdl = s2.unigram_view(docs)
    P, pcol, pidf, pavg, tok = s2.pair_view(docs, min_df=2)
    rows = [s2.query_rows(qtok(q), col, idf, avgdl, pcol, pidf, pavg, tok) for _, q, _ in Q]
    QU = s2.to_matrix([u for u, _ in rows], U.shape[1])
    QP = s2.to_matrix([p for _, p in rows], P.shape[1])
    return U, P, QU, QP, time.time() - t0


def paired(a: np.ndarray, b: np.ndarray, k: int = 50) -> str:
    """Guarantee: 'hit@k a-minus-b [95% bootstrap interval]' over the same queries."""
    d = (a < k).astype(float) - (b < k).astype(float)
    boot = np.random.default_rng(0)
    lo, hi = np.percentile([d[boot.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])
    return "%+.3f [%+.3f, %+.3f]" % (d.mean(), lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=10_000)
    ap.add_argument("--n", type=int, default=dd.N_CORPUS)
    a = ap.parse_args()
    dd.N_CORPUS = a.n
    C = dd.section_corpus()
    Q = dd.queries(C)
    truth = [i for i, _, _ in Q]
    print("corpus %d sections (usable)  queries %d  n_relevant=1 each" % (len(C), len(Q)), flush=True)
    base = None
    for name, merges in (("words", None), ("bpe", a.merges)):
        U, P, QU, QP, secs = views(C, Q, merges)
        print("%s: U %d columns nnz/row %.0f | P %d columns nnz/row %.0f | built %.0fs" % (
            name, U.shape[1], U.nnz / U.shape[0], P.shape[1], P.nnz / P.shape[0], secs), flush=True)
        own0 = None
        for lam in GRID:
            t0 = time.time()
            r = s2.rank_of_truth(U, P, QU, QP, truth, lam)
            if base is None:
                base = r
            if own0 is None:
                own0 = r
            h = lambda k: float((r < k).mean())
            print("  %-6s lambda %-4s rec@10 %.3f  rec@50 %.3f | vs words lambda 0: %s | vs same arm lambda 0: %s  (%.0fs)" % (
                name, lam, h(10), h(50), paired(r, base), paired(r, own0), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
