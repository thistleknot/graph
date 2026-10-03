"""ingest_arxiv_sparsevec.py -- the arxiv section chunks as a pgvector sparsevec BM25 index, measured.

NO GOVERNING SPEC. Basis: operator instruction 2026-10-02 ("ingest my arxiv papers ... post
processed markdown ... use the chunking strategy outlined in my skills", with section names
tracked "so I can exclude them during retrieval"). Skill: sparsevec-lexsem-graph (R6, R8, R12,
R13), bpe-bm25 (rec@50, parity first). Chunks: domain_corpora.chunk_arxiv via
.tmp/run_arxiv_chunks.py -> .tmp/arxiv_section_chunks.pkl.

    STAGE     MECHANISM
    TOKENS    stoplist.tokenize, one tokenizer for index and queries (R2)
    BM25      salient_grams.bm25_matrix over RAW WORDS, all terms (A1)
    SPLIT     doc side = saturated tf (idf divided back out); idf lives in lex_vocab (R6, A2)
    TRIM      keep each row's 1,000 heaviest by idf*sat (pgvector HNSW limit, V2)
    STORE     lex_vocab, lex_chunk_arxiv_sect (sparsevec), lex_chunk_meta (text + key + flags),
              lex_cos_arxiv_sect (the symmetric doc-doc view, V7), lex_build (the frozen statistics)
    INDEX     HNSW sparsevec_ip_ops (queries) and sparsevec_cosine_ops (doc-doc), m=16 ef_construction=64
    MEASURE   200 title + 200 body-sentence queries, relevant = the document's chunks:
              numpy exact == psql exact (parity first), then recall@50 at ef 50/100/400

A1  Raw words, NOT BPE. Measured on the earlier 169,382-row arxiv+neop population (commits
    ae272f6, 9cea42b): BPE cost 8-17% rec@50 against raw at 40% more disk; HNSW 16/64 at
    ef=50 cost nothing at rec@50. Those rows came from the OLD chunking, so for the new
    chunks this is a hypothesis the measurement below re-tests.
A2  <q, d> = sum idf(t)*qtf(t) * sat(t, d) equals the idf-baked score the earlier
    diagnostics used, to float precision; asserted by parity before any HNSW number.
A3  is_reference chunks are STORED and flagged, never dropped (operator 2026-10-02).
A4  is_junk chunks (domain_corpora C10) are stored and flagged too, but enter the statistics as
    empty documents: no term, no vector row, no say in df. They are still counted in N and in
    avgdl (259 of 58,870 chunks, 0.44%), an approximation accepted here and not measured.
    The statistics (vocabulary, idf, avgdl, k1, b, the whale columns) are frozen in `lex_build`
    so tools/arxiv_graph_service.py can add later documents with doc_row() and get the row a
    full build would have given them.
A5  The stored cosine view is fitted on every non-junk chunk, references included; the community
    map's in-memory lexical view (tools/arxiv_community_map.py) is fitted on retrievable chunks
    only. Both are cosine over saturated tf*idf rows; their idf differs slightly.

Run:  python -u tools\\ingest_arxiv_sparsevec.py [--no-battery]
"""
from __future__ import annotations

import collections
import io
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as ss_sparse

import sparsevec_store as ss
from diag_dense_arms import pick_sentence
from domain_corpora import with_junk_flag
from salient_grams import bm25_matrix
from stoplist import tokenize

CACHE = ".tmp/arxiv_section_chunks.pkl"
LABEL = "arxiv_sect"
K, EFS, N_Q, SEED, M, EFC = 50, (50, 100, 400), 200, 0, 16, 64
K1, B = 1.5, 0.75                   # salient_grams.bm25_matrix's defaults, passed explicitly so a build records them
DENSE_DIM = 384                     # all-MiniLM-L6-v2


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def saturated(BM, df, N):
    """R6. Guarantee: (sat, idf) with sat[i, t] * idf[t] == BM[i, t]; sat is CSR."""
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    sat = BM.tocsr().copy()
    sat.data = sat.data / idf[sat.indices]
    return sat, idf


def keep_heaviest(BM, cap: int = ss.SPARSEVEC_MAX_NNZ):
    """V2. Guarantee: boolean mask over BM.data keeping each row's `cap` heaviest by BM weight."""
    mask = np.ones(BM.nnz, dtype=bool)
    for i in np.where(np.diff(BM.indptr) > cap)[0]:
        s, e = BM.indptr[i], BM.indptr[i + 1]
        mask[s + np.argsort(-BM.data[s:e], kind="stable")[cap:]] = False
    return mask


def full_views(BM, df, N, cap: int = ss.SPARSEVEC_MAX_NNZ):
    """V2, V7, R7. Guarantee: (sat_t, cos_t, idf, whales). sat_t: saturated tf with each row's `cap`
    heaviest terms by BM weight (the inner-product view, idf on the query side). cos_t: the symmetric
    doc-doc view -- those same kept terms as idf*sat, terms in more than half the documents dropped
    (`whales`, columns), rows L2-normalised. doc_row reproduces one row of each from frozen statistics."""
    sat, idf = saturated(BM, df, N)
    mask = keep_heaviest(BM, cap)
    sat_t = sat.copy()
    sat_t.data = sat.data * mask
    sat_t.eliminate_zeros()
    whales = np.where(df * 2 > N)[0]
    cos_t = BM.tocsr().copy()
    cos_t.data = cos_t.data * mask * ~np.isin(cos_t.indices, whales)
    cos_t.eliminate_zeros()
    norms = np.sqrt(np.asarray(cos_t.multiply(cos_t).sum(1)).ravel())
    norms[norms == 0] = 1.0
    return sat_t, (ss_sparse.diags(1.0 / norms) @ cos_t).tocsr(), idf, whales


def doc_row(tokens: list[str], col: dict, idf, avgdl: float, whales=(), cap: int = ss.SPARSEVEC_MAX_NNZ):
    """Guarantee: ({col0: saturated tf}, {col0: L2-normalised idf*sat}) for ONE document under FROZEN
    statistics (vocabulary `col`, idf, avgdl, whales), equal to the row full_views gives it in the
    build that fixed them. Terms outside the vocabulary are dropped; `tokens` is every token of the
    document, because document length counts them all."""
    cnt = collections.Counter(t for t in tokens if t in col)
    if not cnt:
        return {}, {}
    norm_len = 1 - B + B * len(tokens) / avgdl
    sat = {col[t]: c * (K1 + 1) / (c + K1 * norm_len) for t, c in cnt.items()}
    kept = {j: sat[j] for j in sorted(sat, key=lambda j: (-sat[j] * idf[j], j))[:cap]}      # ties go to the lower column, as keep_heaviest's stable sort does
    bm = {j: s * idf[j] for j, s in kept.items() if j not in set(whales)}
    n = math.sqrt(sum(w * w for w in bm.values()))
    return kept, ({j: w / n for j, w in bm.items()} if n > 0 else {})


def gold(recs: list[dict]):
    """Guarantee: ([(title, relevant_ords)], [(sentence, relevant_ords)]), seeded, n=N_Q each."""
    by_doc = collections.defaultdict(set)
    first = {}
    for i, r in enumerate(recs):
        by_doc[r["doc_id"]].add(i)
        first.setdefault(r["doc_id"], r["section_title"])
    rng = np.random.default_rng(SEED)
    docs = sorted(d for d, t in first.items() if not t.startswith("arxiv/") and len(t.split()) >= 3)
    titles = [(first[d], by_doc[d]) for d in rng.permutation(docs)[:N_Q]]
    body = []
    for i in rng.permutation(len(recs)):
        if recs[i]["is_reference"]:
            continue
        got = pick_sentence(recs[i]["text"], rng)
        if got:
            body.append((got[0], by_doc[recs[i]["doc_id"]]))
        if len(body) == N_Q:
            break
    return titles, body


def query_terms(text: str, col: dict, idf: np.ndarray):
    """R2, R6. Guarantee: (cols, weights = idf * qtf) over the terms the vocabulary holds."""
    qtf = collections.Counter(t for t in tokenize(text) if t in col)
    cols = sorted(col[t] for t in qtf)
    inv = {col[t]: n for t, n in qtf.items()}
    return cols, [float(idf[c] * inv[c]) for c in cols]


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    cached = pickle.load(open(CACHE, "rb"))
    recs = with_junk_flag(cached["records"])
    log("chunks %d (reference-flagged %d, junk-flagged %d)"
        % (len(recs), sum(r["is_reference"] for r in recs), sum(r["is_junk"] for r in recs)))

    t0 = time.time()
    token_docs = [[] if r["is_junk"] else tokenize(r["text"]) for r in recs]       # A4: a junk chunk is an empty document
    BM, TF, terms, df = bm25_matrix(token_docs, K1, B)
    BM = BM.tocsr()
    N, dim = BM.shape
    avgdl = float(np.mean([len(t) for t in token_docs]))
    sat_t, cos_t, idf, whales = full_views(BM, df, N)
    n_cut = int((np.diff(BM.indptr) > ss.SPARSEVEC_MAX_NNZ).sum())
    log("bm25 %.0fs  rows=%d dim=%d nnz=%d  rows over %d nnz trimmed: %d  whale terms dropped from the cosine view: %d"
        % (time.time() - t0, N, dim, BM.nnz, ss.SPARSEVEC_MAX_NNZ, n_cut, len(whales)))

    conn = ss.connect()
    ss.reset_label_tables(conn, LABEL)                          # the vocabulary changed, and with it the sparse dimension
    ss.ensure_schema(conn, LABEL, dim)
    ss.ensure_chunk_meta(conn)
    ss.ensure_view_schema(conn, LABEL, dim, DENSE_DIM)
    ss.ensure_build_schema(conn)
    t0 = time.time()
    ss.write_vocab(conn, LABEL, [str(t) for t in terms], idf=idf)
    log("vocab written  %.0fs" % (time.time() - t0))
    log("meta written: %d rows" % ss.write_chunk_meta(conn, LABEL, recs))
    build = ss.new_build(conn, LABEL, len(recs), {
        "bm25": {"k1": K1, "b": B, "avgdl": avgdl, "n_docs": N, "dim": dim, "whale_cols": whales.tolist()},
        "fit": cached.get("fit"), "dense_dim": DENSE_DIM})
    log("build %d opened (statistics frozen here until the next full build)" % build)

    def rows(M_):
        def gen():
            for i in range(N):
                s, e = M_.indptr[i], M_.indptr[i + 1]
                yield i, recs[i]["doc_id"], "arxiv", dict(zip(M_.indices[s:e].tolist(), M_.data[s:e].tolist()))
        return gen()
    t0 = time.time()
    w = ss.write_chunks(conn, LABEL, dim, rows(sat_t))
    log("vectors written %d rows (skipped %d empty)  %.0fs" % (w["written"], N - w["written"], time.time() - t0))
    log("hnsw build %.0fs (m=%d, ef_construction=%d)  %s" % (ss.create_hnsw(conn, LABEL, M, EFC), M, EFC, ss.counts(conn, LABEL)))
    t0 = time.time()
    wc = ss.write_chunks(conn, LABEL, dim, rows(cos_t), kind="cos")
    log("cosine view written %d rows  %.0fs; hnsw build %.0fs" % (wc["written"], time.time() - t0,
                                                                  ss.create_hnsw(conn, LABEL, M, EFC, kind="cos")))

    if "--no-battery" in sys.argv:
        log("--no-battery: recall measurements skipped")
        conn.close()
        return
    col = {str(t): j for j, t in enumerate(terms)}
    sat_c = sat_t.tocsc()
    titles, body = gold(recs)
    log("gold: title %d  body %d" % (len(titles), len(body)))
    hdr = "  %-24s %8s %9s"
    for gname, gq in (("title -> own paper", titles), ("body sentence -> own paper", body)):
        qs = [(query_terms(q, col, idf), rel) for q, rel in gq]
        qs = [(c, w_, rel) for (c, w_), rel in qs if c]
        print("\n  [%s]  n=%d (queries with no vocabulary term dropped: %d)" % (gname, len(qs), len(gq) - len(qs)))
        print(hdr % ("arm", "rec@50", "ms/q"))

        def recall(top):
            return float(np.mean([len(set(t) & rel) / len(rel) for t, (_, _, rel) in zip(top, qs)]))
        np_top, np_scores = [], []
        t0 = time.time()
        for cols, wts, _ in qs:
            s = sat_c[:, cols] @ np.asarray(wts)
            o = np.argsort(-s, kind="stable")[:K]
            np_top.append(o.tolist())
            np_scores.append(s[o])
        print(hdr % ("numpy exact", "%.3f" % recall(np_top), "%.0f" % ((time.time() - t0) / len(qs) * 1000)))
        t0 = time.time()
        pg = [ss.query(conn, LABEL, dim, c, K, exact=True, weights=w_) for c, w_, _ in qs]
        ms = (time.time() - t0) / len(qs) * 1000
        print(hdr % ("psql exact scan", "%.3f" % recall([[o for o, _ in r] for r in pg]), "%.0f" % ms))
        same = np.mean([np.allclose(np.sort(a), np.sort([s for _, s in r]), rtol=1e-3, atol=1e-4) for a, r in zip(np_scores, pg)])
        print("  parity: psql exact top-%d SCORE set == numpy on %.1f%% of queries%s"
              % (K, 100 * same, "" if same >= 0.99 else "   <-- PARITY DEFECT, read no HNSW number below"))
        for ef in EFS:
            t0 = time.time()
            hn = [ss.query(conn, LABEL, dim, c, K, ef_search=ef, weights=w_) for c, w_, _ in qs]
            ms = (time.time() - t0) / len(qs) * 1000
            assert all(len(r) == min(K, ef) for r in hn), "HNSW returned fewer rows than min(k, ef)"
            print(hdr % ("psql hnsw ef=%d" % ef, "%.3f" % recall([[o for o, _ in r] for r in hn]), "%.0f" % ms))
    conn.close()


if __name__ == "__main__":
    main()
