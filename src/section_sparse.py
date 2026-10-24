"""section_sparse.py -- BPE BM25 unigram and adjacent-pair sparse rows for section mapping and query retrieval.

Spec: operator 2026-10-05 ("apply the bpe tokenizer for sparsevec/inner product", "rerank sparsevec by adjacent tokens ... n-grams ranked
higher than records that are [not] ... that's how we should measure cosine"; no HNSW, exact inner products); approved plan step B2.
Task: playbook.md T152. Guards of sparsevec-lexsem-graph: R6 (idf on the query side), R7 (doc-doc similarity from the symmetric L2 view).

    unigram view U  rows of idf * saturated-tf over the tokens (words, or BPE pieces), terms in more than half the sections dropped
                    (src/ingest_arxiv_sparsevec.full_views), each row L2-normalised, at most SPARSEVEC_MAX_NNZ entries
    pair view    P  the same weighting over ORDERED adjacent token pairs (a, b), pairs in fewer than `min_df` sections (they can match nothing)
                    or in more than half dropped, each row L2-normalised, at most SPARSEVEC_MAX_NNZ entries
    score(i, j) = <U_i, U_j> + lam * <P_i, P_j>        exactly the inner product of the rows [U_i, sqrt(lam) P_i]
    QUERY        encode map queries from the map corpus's frozen BPE vocabulary and unigram/pair statistics
    RERANK       reorder the top sparse pool by bounded within-word token alignment, within-word token pairs, and directed word pairs

Q1 Map query rows SHALL use the same tokenizer, feature columns, IDF values, and pair weighting as section rows.
Q2 The positional reranker SHALL preserve word boundaries for BPE token matches and count repeated query tokens/pairs up to query multiplicity.

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
from tokenizers import Tokenizer

from stoplist import tokenize

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
    """Build normalized section rows with word or BPE unigram and adjacent-pair features."""
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


def adjacency_scores(query_groups: list[str], documents: list[str], tokenizer: Tokenizer) -> list[dict]:
    """Score document order against query-word and query-pair structure.

    Require:
        Query groups and documents use the same word tokenizer as the sparse index; `tokenizer` is its frozen BPE tokenizer.
    Guarantee:
        One bounded score record per document, combining within-word BPE position, within-word adjacent-piece, and directed adjacent-word matches.
    Maintain:
        BPE pieces never match across word boundaries; repeated query items contribute up to their query multiplicity.
    Assert:
        Every component ratio and combined score is in [0, 1].
    """
    query_words = [tokenize(group) for group in query_groups]
    flat_query = [word for group in query_words for word in group]
    if not flat_query:
        return [{"score": 0.0, "within_word": 0.0, "piece_adjacency": 0.0, "term_adjacency": 0.0}
                for _ in documents]

    q_pieces = {word: tokenizer.encode(word).tokens for word in set(flat_query)}
    q_piece_pairs = collections.Counter(
        pair for word in flat_query for pair in zip(q_pieces[word], q_pieces[word][1:])
    )
    q_term_pairs = collections.Counter(
        pair for group in query_words for pair in zip(group, group[1:])
    )

    def overlap(left: collections.Counter, right: collections.Counter) -> int:
        return sum(min(count, right.get(item, 0)) for item, count in left.items())

    def position_match(query_tokens: list[str], document_tokens: list[str]) -> float:
        if not query_tokens:
            return 0.0
        return max(
            sum(query_tokens[i] == document_tokens[i + offset]
                for i in range(len(query_tokens))
                if 0 <= i + offset < len(document_tokens)) / len(query_tokens)
            for offset in (-1, 0, 1)
        )

    results = []
    for document in documents:
        doc_words = tokenize(document)
        doc_pieces = [tokenizer.encode(word).tokens for word in doc_words]
        used_doc_positions: set[int] = set()
        alignments = []
        for query_word in flat_query:
            qtokens = q_pieces[query_word]
            candidates = [
                (position_match(qtokens, pieces), index)
                for index, pieces in enumerate(doc_pieces)
                if index not in used_doc_positions
            ]
            if candidates:
                score, index = max(candidates, key=lambda pair: (pair[0], -pair[1]))
                if score > 0:
                    alignments.append(score)
                    used_doc_positions.add(index)
                else:
                    alignments.append(0.0)
            else:
                alignments.append(0.0)
        within_word = sum(alignments) / len(flat_query)

        d_piece_pairs = collections.Counter(
            pair for pieces in doc_pieces for pair in zip(pieces, pieces[1:])
        )
        piece_adjacency = (overlap(q_piece_pairs, d_piece_pairs) / sum(q_piece_pairs.values())
                           if q_piece_pairs else 0.0)
        d_term_pairs = collections.Counter(zip(doc_words, doc_words[1:]))
        term_adjacency = (overlap(q_term_pairs, d_term_pairs) / sum(q_term_pairs.values())
                          if q_term_pairs else 0.0)
        components = [within_word]
        if q_piece_pairs:
            components.append(piece_adjacency)
        if q_term_pairs:
            components.append(term_adjacency)
        score = float(sum(components) / len(components))
        assert all(0.0 <= value <= 1.0 for value in (within_word, piece_adjacency, term_adjacency, score))
        results.append({"score": score, "within_word": within_word,
                        "piece_adjacency": piece_adjacency, "term_adjacency": term_adjacency})
    return results


def rank_of_truth(U, P, QU, QP, truth: list[int], lam: float) -> np.ndarray:
    """Guarantee: for each query, the number of sections scoring STRICTLY above its truth section under <U,QU> + lam <P,QP> (0-based rank;
    ties do not push the truth down, as diag_dense_arms.score does). QU, QP are the (n_queries, width) query matrices."""
    S = (QU @ U.T).toarray()
    if lam:
        S = S + lam * (QP @ P.T).toarray()
    t = S[np.arange(len(truth)), truth]
    return (S > t[:, None]).sum(1)
