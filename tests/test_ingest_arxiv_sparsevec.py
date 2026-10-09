"""Pins ingest_arxiv_sparsevec.py: the doc/query split of BM25 (R6) and the nnz trim (V2).

The identity that makes the whole lane valid: sat[i, t] * idf[t] == the idf-baked BM25 the
earlier diagnostics scored with, so <query(idf*qtf), sat> reproduces the old numbers.
No database, no corpus.

Run:  pytest tests/test_ingest_arxiv_sparsevec.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest_arxiv_sparsevec import keep_heaviest, query_terms, saturated
from salient_grams import bm25_matrix

DOCS = [["graph", "edge", "graph", "node"], ["graph", "leiden", "community", "node", "node"],
        ["bm25", "sparse", "vector", "edge"], ["sparse", "graph", "community"]]


def test_saturated_times_idf_is_the_idf_baked_bm25():
    BM, TF, terms, df = bm25_matrix(DOCS)
    sat, idf = saturated(BM, df, len(DOCS))
    back = sat.multiply(idf[None, :]).toarray()
    assert np.allclose(back, BM.toarray(), rtol=1e-6)


def test_saturated_vectors_hold_no_idf():
    """R6: a term in every document and a rare term get the same saturation at equal tf."""
    docs = [["common", "rare"], ["common", "x"], ["common", "y"], ["common", "z"]]
    BM, TF, terms, df = bm25_matrix(docs)
    sat, idf = saturated(BM, df, len(docs))
    t = {str(w): j for j, w in enumerate(terms)}
    assert idf[t["common"]] < idf[t["rare"]]
    assert np.isclose(sat[0, t["common"]], sat[0, t["rare"]])


def test_keep_heaviest_keeps_the_top_cap_by_weight_per_row():
    M = sp.csr_matrix(np.array([[5.0, 1.0, 4.0, 2.0, 3.0], [1.0, 2.0, 0.0, 0.0, 0.0]]))
    kept = keep_heaviest(M, cap=3)
    row0 = M.indices[M.indptr[0]:M.indptr[1]][kept[M.indptr[0]:M.indptr[1]]]
    assert sorted(row0.tolist()) == [0, 2, 4]            # weights 5, 4, 3
    assert kept[M.indptr[1]:M.indptr[2]].all()           # a short row is untouched


def test_query_terms_weights_are_idf_times_qtf_over_known_terms_only():
    col = {"graph": 0, "node": 1}
    idf = np.array([2.0, 0.5])
    cols, w = query_terms("graph node node unknownword", col, idf)
    assert cols == [0, 1] and w == [2.0, 1.0]
    assert query_terms("zzz qqq", col, idf) == ([], [])


# ---- doc_row must reproduce a full build's rows from frozen statistics (V7, the incremental path)

CORPUS = DOCS + [["graph", "graph", "graph", "tensor", "rank", "node"], [], ["leiden", "bm25", "sparse", "vector", "graph"]]


def _frozen(cap=1000):
    from ingest_arxiv_sparsevec import B, K1, full_views
    BM, TF, terms, df = bm25_matrix(CORPUS, K1, B)
    sat_t, cos_t, idf, whales = full_views(BM, df, len(CORPUS), cap=cap)
    col = {str(t): j for j, t in enumerate(terms)}
    avgdl = float(np.mean([len(t) for t in CORPUS]))
    return sat_t, cos_t, idf, whales, col, avgdl


def _dense(d, dim):
    out = np.zeros(dim)
    for j, w in d.items():
        out[j] = w
    return out


def test_doc_row_equals_the_full_build_row_for_every_document_including_the_empty_one():
    from ingest_arxiv_sparsevec import doc_row
    sat_t, cos_t, idf, whales, col, avgdl = _frozen()
    dim = sat_t.shape[1]
    assert len(whales) >= 1                                           # "graph" is in most documents: a whale
    for i, toks in enumerate(CORPUS):
        sat, cos = doc_row(toks, col, idf, avgdl, whales)
        assert np.allclose(_dense(sat, dim), sat_t[i].toarray().ravel(), rtol=1e-9), i
        assert np.allclose(_dense(cos, dim), cos_t[i].toarray().ravel(), rtol=1e-9), i
    assert doc_row([], col, idf, avgdl, whales) == ({}, {})


def test_the_cosine_view_has_no_whale_columns_and_unit_rows_except_where_nothing_is_left():
    sat_t, cos_t, idf, whales, col, avgdl = _frozen()
    assert cos_t[:, whales].nnz == 0 and sat_t[:, whales].nnz > 0     # whales stay in the query view, not the doc-doc view
    norms = np.sqrt(np.asarray(cos_t.multiply(cos_t).sum(1)).ravel())
    nonempty = np.diff(cos_t.indptr) > 0
    assert np.allclose(norms[nonempty], 1.0) and norms[~nonempty].max() == 0.0


def test_doc_row_applies_the_same_cap_as_the_full_build():
    from ingest_arxiv_sparsevec import doc_row
    sat_t, cos_t, idf, whales, col, avgdl = _frozen(cap=2)
    dim = sat_t.shape[1]
    assert np.diff(sat_t.indptr).max() == 2
    for i, toks in enumerate(CORPUS):
        sat, cos = doc_row(toks, col, idf, avgdl, whales, cap=2)
        assert np.allclose(_dense(sat, dim), sat_t[i].toarray().ravel()), i
        assert np.allclose(_dense(cos, dim), cos_t[i].toarray().ravel()), i


def test_doc_row_drops_unknown_terms_but_their_length_still_counts():
    from ingest_arxiv_sparsevec import doc_row
    sat_t, cos_t, idf, whales, col, avgdl = _frozen()
    base, _ = doc_row(["leiden", "bm25"], col, idf, avgdl, whales)
    padded, _ = doc_row(["leiden", "bm25"] + ["neverseen"] * 20, col, idf, avgdl, whales)
    assert set(padded) == set(base) == {col["leiden"], col["bm25"]}           # the unknown term has no column
    assert all(padded[j] < base[j] for j in base)                            # but a longer document saturates less
    assert doc_row(["neverseen", "alsonew"], col, idf, avgdl, whales) == ({}, {})


# ------------------------------------------------------------------------------------------------- A6 (BPE vocabulary)
def _bpe_corpus():
    rng = np.random.default_rng(4)
    stems = ["transform", "attention", "quantiz", "gradient", "embedding", "optimi", "regulariz", "convolut"]
    ends = ["er", "ers", "ation", "ing", "ed", "s", "al"]
    return [[str(rng.choice(stems) + rng.choice(ends)) for _ in range(12)] for _ in range(60)]


def _join(pieces):
    return "".join(p[2:] if p.startswith("##") else p for p in pieces)


def test_bpe_docs_spell_every_word_from_pieces_that_concatenate_back_to_the_word_A6():
    from ingest_arxiv_sparsevec import bpe_docs
    docs = _bpe_corpus()
    out, to_pieces = bpe_docs(docs, merges=40)                                                      # a budget above the term count would merge every word whole
    assert len(out) == len(docs)
    for words, pieces in zip(docs[:10], out[:10]):
        assert _join(pieces) == "".join(words)                                                     # nothing lost, nothing added
        assert "[UNK]" not in pieces
    assert sum(len(o) for o in out) > sum(len(d) for d in docs)                                    # rare words split: more pieces than words
    assert not to_pieces(["transformer"])[0].startswith("##")                                      # a word starts with a bare piece


def test_bpe_docs_spell_a_word_never_seen_and_spell_queries_as_the_index_was_spelled_A6():
    from ingest_arxiv_sparsevec import bpe_docs
    docs = _bpe_corpus()
    out, to_pieces = bpe_docs(docs, merges=200)
    unseen = to_pieces(["attentionest"])                                                            # in no document, built from characters the corpus has
    assert _join(unseen) == "attentionest" and "[UNK]" not in unseen
    assert "[UNK]" in to_pieces(["kiwi"])                                                           # a character no document holds: unspellable, so it matches no column
    seen = docs[0][0]
    n = len(to_pieces([seen]))
    assert to_pieces([seen]) == out[0][:n]                                                          # same spelling in a document and in a query


def test_bpe_docs_keep_an_empty_document_empty_and_still_spell_whales_and_hapaxes_A6():
    from ingest_arxiv_sparsevec import bpe_docs
    out, _ = bpe_docs(_bpe_corpus() + [[]], merges=200)
    assert out[-1] == []
    docs2 = [["everyword", "common%d" % (i % 5), "common%d" % (i % 5)] for i in range(30)] + [["hapaxonly"]]
    _, to_pieces = bpe_docs(docs2, merges=100)                                                      # a whale and a hapax do not train the merges
    assert _join(to_pieces(["hapaxonly"])) == "hapaxonly" and _join(to_pieces(["everyword"])) == "everyword"


def test_a_bpe_query_is_spelled_from_words_not_from_characters_A6():
    """The live defect: query_terms passed the query STRING to to_pieces (which takes a list of words) and spelled it from single characters,
    so recall@50 on the BPE build was 0.001. The query side must be words first, then pieces: tok = lambda text: to_pieces(tokenize(text))."""
    from ingest_arxiv_sparsevec import bpe_docs
    from stoplist import tokenize
    docs = [tokenize("transformer attention gradient embedding"), tokenize("quantization regularization convolution optimizer")] * 12
    out, to_pieces = bpe_docs(docs, merges=60)
    BM, TF, terms, df = bm25_matrix(out)
    col = {str(t): j for j, t in enumerate(terms)}
    idf = np.ones(len(terms))
    tok = lambda text: to_pieces(tokenize(text))
    cols, _ = query_terms("transformer attention", col, idf, tok)
    assert cols == sorted({col[p] for p in to_pieces(["transformer", "attention"])})                # the pieces of those two words, nothing else
    wrong, _ = query_terms("transformer attention", col, idf, to_pieces)                            # the buggy wiring: a string iterated as characters
    assert wrong != cols
