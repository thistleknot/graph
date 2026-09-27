"""diag_sparsevec_arms.py -- the 2x2: {raw words, BPE pieces} x {exact, HNSW}, one
population, one gold, index bytes beside recall.

NO GOVERNING SPEC. Basis: operator instruction 2026-09-27 -- "raw bm25 / raw bm25 as
sparsevec with 16/64 hnsw (ef_search at 40) / our filtered token bpe as sparsevec with
16/64 hnsw (ef_search at 40) -- recall for each? think about this, don't just agree
with me". Task: playbook.md T119. Skill: ~/.skills/bpe-bm25 (psql lane).

Why a square and not three arms: measured on the pieces index, HNSW at ef=40 costs
-11% hit@10 against exact while the vocabulary costs -0.5% MRR against full BM25. A
single-ef three-arm design is dominated by the index knob and would read the
vocabulary as losing 11%. The four cells separate the two factors:

    [B]-[A]  HNSW cost on the RAW vocabulary        (is HNSW-on-raw even viable?)
    [D]-[B]  vocabulary effect at MATCHED HNSW       (the operator's question)
    [D]-[C]  HNSW cost on pieces, full corpus        (was -11% / -1.5% on the sample)

Rules baked in, each from a measurement or a verified fact (2026-09-27):

    RAW      unnormalised salient_grams.bm25_matrix rows over the FULL word vocabulary
             (df >= 1, no masks). NOT chunkgraph._sparse_sim, whose rows are L2-normalised
             (chunkgraph.py:692-693): under sparsevec_ip_ops that turns <#> into
             cosine-BM25, a different ranking.
    POP      arxiv full (168,794) + neop (588) = 169,382 rows in EVERY cell. Chunk
             sampling fragments gold sets -- measured 0.776 vs 0.897 hit@10 on the same
             vocabulary -- so the sample is used ONLY to derive the BPE vocabulary.
    EF CAP   pgvector's HNSW scan returns at most ef_search rows (verified: ef=40 with
             LIMIT 100/50/40 -> 40 rows). At ef=40 only k <= 10 is computed. Every
             HNSW query asserts len(rows) == min(k, ef).
    NNZ      rows keep their 1,000 heaviest pieces. SPARSEVEC_MAX_NNZ is 16,000 in the
             installed header; 1,000 is the documented HNSW cap, unverified here.
             Truncation count reported per arm; exact is computed on the truncated
             matrix (what psql holds) AND untruncated (true BM25), so the cost is visible.
    PARITY   200-query psql exact scan vs numpy per arm BEFORE any HNSW number is read.
             Proven on pieces; NOT yet proven on the raw vocabulary. A raw parity
             failure means L2/cosine leaked in: stop.
    BYTES    pg_total_relation_size per arm is a first-class column -- the operator's
             motive is a smaller index, and [B] vs [D] is 1.49M dims vs 8.8k.

Deterministic: fixed seed, no model call. `--reuse` skips COPY + HNSW when a table
already holds the full population, so a rerun is queries only.

Run (cmd):  set PYTHONPATH=.  &  python -u tools\\diag_sparsevec_arms.py
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from salient_grams import bm25_matrix
from stoplist import tokenize
import sparsevec_store as ss
from diag_domain_recall import body_queries, neop_gold
from diag_subset_sparsevec import (allocation, arxiv_full, derive_vocab,
                                   metrics_from_ranked, piece_fn, truncate_rows)

SEED = 0
N_TITLE = 500
N_BODY = 200
SAMPLE_TOTAL = 30_000            # the vocabulary-derivation sample, as measured 2026-09-26
EF_SEARCH = (40, 400)
KS_FULL = (1, 5, 10, 50)
PARITY_N = 200
HNSW_M, HNSW_EFC = 16, 64


def _utf8():
    # line_buffering=True: a fresh TextIOWrapper block-buffers on its own even under
    # `python -u`, so without this every stage line lands in one burst at exit and a
    # monitor on the log sees nothing until then (observed 2026-09-27).
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)


def ks_for(ef: int):
    """EF CAP: only k <= ef is a real top-k."""
    return tuple(k for k in KS_FULL if k <= ef)


def fmt(r, ks):
    def g(d, k):
        return "%6.3f" % d[k] if k in d else "     -"
    return "%s %s %s  %s  %s %6.3f" % (g(r["hit"], 1), g(r["hit"], 5), g(r["hit"], 10),
                                         g(r["recall"], 10), g(r["recall"], 50), r["mrr"])


HDR = "  %-34s %9s %4s   hit@1  hit@5 hit@10  rec@10  rec@50    MRR   %s" % (
    "cell", "vocab", "nq", "ms/q")


def rank_numpy(BMc, qcols_list, K):
    out = []
    for cols in qcols_list:
        s = np.asarray(BMc[:, cols].sum(axis=1)).ravel()
        out.append(np.argsort(-s, kind="stable")[:K].tolist())
    return out


def prep_queries(gq, rel_of, qtok, col):
    qcols, rels = [], []
    for q, target in gq:
        cols = sorted({col[p] for p in qtok(q) if p in col})
        rel = rel_of(target)
        if cols and rel:
            qcols.append(cols)
            rels.append(rel)
    return qcols, rels


def run_arm(conn, name, label, BM, terms, qtok, golds, reuse, rows_meta):
    """One vocabulary through all four measurements. Returns rows for the report."""
    dim = len(terms)
    col = {str(t): j for j, t in enumerate(terms)}
    n_rows = BM.shape[0]
    print("\n== ARM %s  dim=%d rows=%d ==" % (name, dim, n_rows))

    BM_t, n_trunc = truncate_rows(BM)
    print("  rows over %d nnz truncated: %d (%.2f%%)"
          % (ss.SPARSEVEC_MAX_NNZ, n_trunc, 100.0 * n_trunc / n_rows))
    BMc, BMc_t = BM.tocsc(), BM_t.tocsc()

    # ---- psql table: write unless --reuse finds the full population
    t = ss.ensure_schema(conn, label, dim)
    have = ss.counts(conn, label)["rows"]
    if reuse and have == n_rows - 0:            # all rows present (empty-vector rows are skipped below)
        print("  reuse: %s already holds %d rows; skipping COPY + HNSW" % (t, have))
        build_s = None
    else:
        t0 = time.time()
        csr = BM.tocsr()

        def rows():
            for i in range(csr.shape[0]):
                s, e = csr.indptr[i], csr.indptr[i + 1]
                yield (i, rows_meta[0][i], rows_meta[1][i],
                       dict(zip(csr.indices[s:e].tolist(), csr.data[s:e].tolist())))
        w = ss.write_chunks(conn, label, dim, rows())
        print("  COPY %d rows (%d truncated) %.0fs" % (w["written"], w["truncated"], time.time() - t0))
        build_s = ss.create_hnsw(conn, label, m=HNSW_M, ef_construction=HNSW_EFC)
        print("  HNSW m=%d ef_construction=%d build %.0fs" % (HNSW_M, HNSW_EFC, build_s))
    c = ss.counts(conn, label)
    print("  table+index bytes: %s  (%.1f MB)  max_nnz=%s" % (
        "{:,}".format(c["bytes"]), c["bytes"] / 1e6, c["max_nnz"]))

    report = []
    K = max(KS_FULL)
    for gname, gq, rel_of in golds:
        qcols, rels = prep_queries(gq, rel_of, qtok, col)
        if not rels:
            continue
        # exact, untruncated (true BM25) and truncated (what psql holds)
        r_true = metrics_from_ranked(rank_numpy(BMc, qcols, K), rels, KS_FULL)
        np_t = rank_numpy(BMc_t, qcols, K)
        r_trunc = metrics_from_ranked(np_t, rels, KS_FULL)
        report.append((gname, "exact numpy, untruncated", dim, r_true, None, KS_FULL))
        report.append((gname, "exact numpy, top-1000 nnz", dim, r_trunc, None, KS_FULL))

        # parity on the first PARITY_N queries
        n_par = min(PARITY_N, len(qcols))
        t0 = time.time()
        pg = [[o for o, _ in ss.query(conn, label, dim, cols, K, exact=True)]
              for cols in qcols[:n_par]]
        lat = (time.time() - t0) / n_par * 1000
        agree = float(np.mean([set(a[:10]) == set(b[:10]) for a, b in zip(np_t[:n_par], pg)]))
        report.append((gname, "psql exact scan (%d q parity)" % n_par, dim,
                       metrics_from_ranked(pg, rels[:n_par], KS_FULL), lat, KS_FULL))
        print("  [%s] parity: psql exact top-10 == numpy on %.1f%% of %d%s"
              % (gname, 100 * agree, n_par, "" if agree >= 0.99 else "   <-- PARITY DEFECT, HNSW numbers below are suspect"))

        for ef in EF_SEARCH:
            ks = ks_for(ef)
            k = max(ks)
            t0 = time.time()
            ranked = []
            for cols in qcols:
                rows_ = [o for o, _ in ss.query(conn, label, dim, cols, k, ef_search=ef)]
                if len(rows_) != min(k, ef):      # EF CAP invariant
                    print("  !! ef=%d k=%d returned %d rows" % (ef, k, len(rows_)))
                ranked.append(rows_)
            lat = (time.time() - t0) / len(qcols) * 1000
            report.append((gname, "psql hnsw ef_search=%d" % ef, dim,
                           metrics_from_ranked(ranked, rels, ks), lat, ks))
    del BMc, BMc_t, BM_t
    return report, c, build_s


def main():
    _utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--titles", type=int, default=N_TITLE)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------ population
    A = arxiv_full()
    ntexts, ncids, nq_head, _ = neop_gold()
    texts = A["texts"] + ntexts
    cids = list(A["cids"]) + list(ncids)
    srcs = list(A["srcs"]) + ["neop"] * len(ntexts)
    n_arxiv = len(A["texts"])
    print("population rows=%d (arxiv %d + neop %d)" % (len(texts), n_arxiv, len(ntexts)))

    by_doc = {}
    for k, c in enumerate(cids):
        by_doc.setdefault(c, set()).add(k)
    rel_doc = lambda d: by_doc.get(d, set())
    titles = [(t, d) for d, t in A["titles"].items() if t]
    pick = rng.choice(len(titles), size=min(args.titles, len(titles)), replace=False)
    q_title = [titles[i] for i in sorted(pick)]
    q_body = body_queries(A["texts"], A["cids"], np.random.default_rng(SEED), n=N_BODY)
    q_neop = [(h, {n_arxiv + k for k in rel}) for h, rel in nq_head]   # offset into population
    golds = (("arxiv title", q_title, rel_doc),
             ("arxiv body", q_body, rel_doc),
             ("neop heading", q_neop, lambda rel: rel))
    print("gold: titles=%d body=%d neop=%d" % (len(q_title), len(q_body), len(q_neop)))

    conn = ss.connect()
    meta = (cids, srcs)
    all_reports = {}

    # ------------------------------------------------------------ RAW arm
    t0 = time.time()
    BMw, TFw, termsw, dfw = bm25_matrix([tokenize(t) for t in texts])
    print("[raw] words=%d  bm25 %.0fs  (df>=1, no masks, rows NOT normalised)" % (len(termsw), time.time() - t0))
    rep, cnt, build = run_arm(conn, "RAW words", "arms_raw", BMw, termsw, tokenize, golds, args.reuse, meta)
    all_reports["raw"] = (rep, cnt, build)
    del BMw, TFw

    # ------------------------------------------------------------ BPE arm
    al = allocation(len(ntexts), n_arxiv, SAMPLE_TOTAL)
    a_pick = np.sort(np.random.default_rng(SEED).choice(n_arxiv, size=al["take_arxiv"], replace=False))
    sample = ntexts + [A["texts"][i] for i in a_pick]
    v = derive_vocab(sample, "subset")
    pf = piece_fn(v)
    t0 = time.time()
    BMp, TFp, termsp, dfp = bm25_matrix([pf(t) for t in texts])
    print("[bpe] pieces=%d  bm25 over pieces %.0fs" % (len(termsp), time.time() - t0))
    rep, cnt, build = run_arm(conn, "BPE pieces", "arms_bpe", BMp, termsp, pf, golds, args.reuse, meta)
    all_reports["bpe"] = (rep, cnt, build)
    del BMp, TFp
    conn.close()

    # ------------------------------------------------------------ THE SQUARE
    print("\n" + "=" * 100)
    print("THE 2x2 -- population %d rows; gold: %d titles, %d body, %d neop headings"
          % (len(texts), len(q_title), len(q_body), len(q_neop)))
    for arm in ("raw", "bpe"):
        rep, cnt, build = all_reports[arm]
        print("\n-- %s: table+index %.1f MB, hnsw build %s --"
              % ("RAW words" if arm == "raw" else "BPE pieces", cnt["bytes"] / 1e6,
                 ("%.0fs" % build) if build is not None else "reused"))
        cur = None
        for gname, cell, dim, r, lat, ks in rep:
            if gname != cur:
                print("  [%s]" % gname)
                print(HDR)
                cur = gname
            print("  %-34s %9d %4d  %s   %s" % (cell, dim, r["n_queries"], fmt(r, ks),
                                               ("%5.0f" % lat) if lat is not None else "    -"))


if __name__ == "__main__":
    main()
