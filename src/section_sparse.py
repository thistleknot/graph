"""section_sparse.py -- the sparse representation of a section: its tokens plus their ADJACENT PAIRS, as L2-normalised rows.

Spec: operator 2026-10-05 ("apply the bpe tokenizer for sparsevec/inner product", "rerank sparsevec by adjacent tokens ... n-grams ranked
higher than records that are [not] ... that's how we should measure cosine"; no HNSW, exact inner products); approved plan step B2.
Task: playbook.md T152. Guards of sparsevec-lexsem-graph: R6 (idf on the query side), R7 (doc-doc similarity from the symmetric L2 view).

    unigram view U  rows of idf * saturated-tf over the tokens (words, or BPE pieces), terms in more than half the sections dropped
                    (src/ingest_arxiv_sparsevec.full_views), each row L2-normalised, at most SPARSEVEC_MAX_NNZ entries
    pair view    P  the same weighting over ORDERED adjacent token pairs (a, b), pairs in fewer than `min_df` sections (they can match nothing)
                    or in more than half dropped, each row L2-normalised, at most SPARSEVEC_MAX_NNZ entries
    score(i, j) = <U_i, U_j> + lam * <P_i, P_j>        exactly the inner product of the rows [U_i, sqrt(lam) P_i]

A run of k shared tokens contributes k-1 shared pairs, so chains add up; a bag of tokens in another order shares none (the symptom of the
BPE precision loss measured 2026-10-04: 0.102 for BPE pieces against 0.410 for words). Whole sections, no index: the product is exact.
"""
from __future__ import annotations

import collections
import math
import os
import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sparsevec_store as ss
from salient_grams import bm25_matrix

K1, B = 1.5, 0.75
MAX_NNZ = ss.SPARSEVEC_MAX_NNZ


def _cap_rows(M: sp.csr_matrix, cap: int = MAX_NNZ) -> sp.csr_matrix:
    """Guarantee: M with each row's `cap` heaviest entries kept (stable on ties)."""
    M = M.tocsr().copy()
    for i in np.where(np.diff(M.indptr) > cap)[0]:
        s, e = M.indptr[i], M.indptr[i + 1]
        M.data[s + np.argsort(-M.data[s:e], kind="stable")[cap:]] = 0.0
    M.eliminate_zeros()
    return M


def _l2(M: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
    n[n == 0] = 1.0
    return (sp.diags(1.0 / n) @ M).tocsr()


def unigram_view(token_docs: list[list[str]], cap: int = MAX_NNZ):
    """Guarantee: (U, col, idf, whales, avgdl): the L2 view over the tokens, the vocabulary {token: column}, idf per column, the dropped
    whale columns, and the mean document length the saturation used (frozen statistics for featurising a query)."""
    BM, TF, terms, df = bm25_matrix(token_docs, K1, B)
    N = len(token_docs)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    whales = np.where(df * 2 > N)[0]
    U = BM.tocsr().copy()
    U.data = U.data * ~np.isin(U.indices, whales)
    U.eliminate_zeros()
    U = _l2(_cap_rows(U, cap))
    col = {str(t): j for j, t in enumerate(terms)}
    return U, col, idf, whales, float(np.mean([len(d) for d in token_docs if d]) if any(token_docs) else 1.0)


def _pair_codes(ids: np.ndarray, base: int) -> np.ndarray:
    return ids[:-1].astype(np.int64) * base + ids[1:] if len(ids) > 1 else np.zeros(0, np.int64)


def pair_view(token_docs: list[list[str]], min_df: int = 2, cap: int = MAX_NNZ):
    """Guarantee: (P, pair_col, pair_idf, avgdl, tok_id): the L2 view over ordered adjacent pairs, {(a_id, b_id): column}, idf per column,
    the mean token count, and the token-to-id map the pair codes use. A pair in fewer than `min_df` or in more than half the documents is
    dropped (it matches nothing, or everything)."""
    tok_id: dict[str, int] = {}
    for d in token_docs:
        for t in d:
            tok_id.setdefault(t, len(tok_id))
    base = max(len(tok_id), 1)
    doc_i, codes, counts = [], [], []
    for i, d in enumerate(token_docs):
        c = _pair_codes(np.fromiter((tok_id[t] for t in d), np.int64, len(d)), base)
        if c.size:
            u, n = np.unique(c, return_counts=True)
            doc_i.append(np.full(u.size, i, np.int64)); codes.append(u); counts.append(n)
    N = len(token_docs)
    if not codes:
        return sp.csr_matrix((N, 1), dtype=np.float32), {}, np.zeros(1), 1.0, tok_id
    doc_i, codes, counts = np.concatenate(doc_i), np.concatenate(codes), np.concatenate(counts)
    uniq, inv = np.unique(codes, return_inverse=True)
    df = np.bincount(inv, minlength=uniq.size)
    good = (df >= min_df) & (df * 2 <= N)
    newcol = np.cumsum(good) - 1
    keep = good[inv]
    idf = np.log(1 + (N - df[good] + 0.5) / (df[good] + 0.5))
    lens = np.fromiter((len(d) for d in token_docs), np.float64, N)
    avgdl = float(lens[lens > 0].mean()) if (lens > 0).any() else 1.0
    c = counts[keep].astype(np.float64)
    sat = c * (K1 + 1) / (c + K1 * (1 - B + B * lens[doc_i[keep]] / avgdl))
    P = sp.csr_matrix((sat * idf[newcol[inv[keep]]], (doc_i[keep], newcol[inv[keep]])), shape=(N, max(int(good.sum()), 1)))
    pair_col = {(int(u // base), int(u % base)): int(newcol[k]) for k, u in enumerate(uniq) if good[k]}
    return _l2(_cap_rows(P, cap)), pair_col, idf, avgdl, tok_id


def query_rows(tokens: list[str], col: dict, idf, avgdl: float, pair_col: dict, pair_idf, pair_avgdl: float, tok_id: dict):
    """Guarantee: (u, p): the query's unigram and pair rows as {column: weight}, each L2-normalised, featurised under the FROZEN statistics
    of the corpus (idf on the query side, as sparsevec-lexsem-graph R6 puts it). Tokens or pairs the corpus never kept are dropped."""
    qtf = collections.Counter(t for t in tokens if t in col)
    u = {col[t]: n * idf[col[t]] for t, n in qtf.items()}
    ids = [tok_id.get(t) for t in tokens]                                   # an unknown token BREAKS the chain: its neighbours are not adjacent
    pc = collections.Counter((a, b) for a, b in zip(ids[:-1], ids[1:]) if a is not None and b is not None)
    p = {pair_col[k]: n * pair_idf[pair_col[k]] for k, n in pc.items() if k in pair_col}

    def l2(d):
        s = math.sqrt(sum(v * v for v in d.values()))
        return {k: v / s for k, v in d.items()} if s > 0 else {}
    return l2(u), l2(p)


def to_matrix(rows: list[dict], width: int) -> sp.csr_matrix:
    r, c, v = [], [], []
    for i, d in enumerate(rows):
        for j, w in d.items():
            r.append(i); c.append(j); v.append(w)
    return sp.csr_matrix((v, (r, c)), shape=(len(rows), width))


def joined(U: sp.csr_matrix, P: sp.csr_matrix, lam: float) -> sp.csr_matrix:
    """Guarantee: the rows [U, sqrt(lam) P], each L2-normalised (a zero row stays zero), so every row is a unit vector and the inner product of two rows
    that both have pairs is (cos_tokens + lam cos_pairs) / (1 + lam): a weighted average of the two cosines. A row with NO kept pair is renormalised
    on its tokens alone, so it is not penalised for having no adjacency to offer (without this its self-similarity would be 1 / (1 + lam))."""
    return _l2(sp.hstack([U, np.sqrt(lam) * P]).tocsr()).astype(np.float32)


def build(records: list[dict], merges: int | None = 10_000, lam: float = 1.0, log=None, strip_heading: bool = False):
    """Guarantee: (Z, info): the joined sparse rows of every record under tokens = words (merges None) or BPE pieces trained on these texts, and
    the sizes. Statistics are fitted on these records (the section map's own corpus). With `strip_heading` the heading line is left out of what is
    tokenised (section_corpus.index_text; arm C of the plan); the default tokenises the whole text, as the baseline map did."""
    from stoplist import tokenize
    from section_corpus import index_text
    docs = [tokenize(index_text(r) if strip_heading else r["text"]) for r in records]
    if merges:
        from ingest_arxiv_sparsevec import bpe_docs
        docs, _ = bpe_docs(docs, merges)
        log and log("BPE: %d merges, %d tokens" % (merges, sum(map(len, docs))))
    U, col, idf, whales, avgdl = unigram_view(docs)
    log and log("unigram view %d columns, %.0f nnz/row" % (U.shape[1], U.nnz / U.shape[0]))
    P, pcol, pidf, pavg, tok = pair_view(docs, min_df=2)
    log and log("pair view %d columns, %.0f nnz/row" % (P.shape[1], P.nnz / P.shape[0]))
    return joined(U, P, lam), {"U": U.shape[1], "P": P.shape[1], "lam": lam, "merges": merges, "tokens": int(sum(map(len, docs)))}


def rank_of_truth(U, P, QU, QP, truth: list[int], lam: float) -> np.ndarray:
    """Guarantee: for each query, the number of sections scoring STRICTLY above its truth section under <U,QU> + lam <P,QP> (0-based rank;
    ties do not push the truth down, as diag_dense_arms.score does). QU, QP are the (n_queries, width) query matrices."""
    S = (QU @ U.T).toarray()
    if lam:
        S = S + lam * (QP @ P.T).toarray()
    t = S[np.arange(len(truth)), truth]
    return (S > t[:, None]).sum(1)
