"""diag_subset_sparsevec.py -- derive the vocabulary on a 100k-chunk subset, prove
it transfers to the full corpus, then index it as pgvector sparsevec and measure
recall there.

NO GOVERNING SPEC. Basis: operator instruction 2026-09-26 -- "apply this to a
subset of the data ... select 100k records ... then do all the work we did, then
use this vocab size for sparsevec in psql and test recall with it".
Task: playbook.md T114 (subset + transfer), T115 (psql lane).
Skill: ~/.skills/bpe-bm25.

Three questions, three measurements, in order:

  ALLOCATION  how many records per corpus. The operator's log-ratio rule is
              computed and printed, then the FEASIBLE rule is applied: a corpus
              contributes min(its size, its quota). Measured: neop has 588 chunks
              and the log ratio would ask it for 34,631 -- sampling with
              replacement, which duplicates df. So: all of neop + seeded arxiv.
  TRANSFER    (numpy) vocabulary derived on the 100k sample vs on the full corpus,
              both evaluated on the FULL arxiv gold. If the subset vocab holds
              recall, the procedure scales to corpora that do not fit in memory.
  PSQL        the sample indexed as sparsevec. Exact scan asserted equal to numpy
              on the same rows (same top-1000 truncation), THEN HNSW recall and
              latency at each ef_search -- the only two things psql adds.

Deterministic: fixed seed for the sample and the body queries, no model call.
The 585 s full-corpus load is cached to .tmp/ after the first run.

Run (cmd):  set PYTHONPATH=.  &  python tools\\diag_subset_sparsevec.py
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import sparse

from domain_corpora import chunk_corpus, derive_pack_target, load_arxiv, load_neop
from salient_grams import anomaly_mask, bm25_matrix
from stoplist import tokenize
import sparsevec_store as ss
from diag_domain_recall import (KS, body_queries, evaluate, first_heading,
                                neop_gold, soft_keep)

CACHE = ".tmp/arxiv_full_chunks.pkl"
SEED = 0
TOTAL = 100_000
N_MERGES = 10_000
EF_SEARCH = (40, 100, 400)


def _utf8():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ----------------------------------------------------------------- corpora ----
def arxiv_full():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            d = pickle.load(fh)
        print("[arxiv] cached  chunks=%d docs=%d" % (len(d["texts"]), len(d["titles"])))
        return d
    t0 = time.time()
    ids, docs, srcs = load_arxiv()
    fit = derive_pack_target(docs)
    cids, texts, csrcs = chunk_corpus(ids, docs, srcs, fit["target"])
    titles = {did: first_heading(doc) for did, doc in zip(ids, docs)}
    d = {"cids": cids, "texts": texts, "srcs": csrcs, "titles": titles, "fit": fit}
    os.makedirs(".tmp", exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(d, fh)
    print("[arxiv] built   chunks=%d docs=%d  %.0fs (cached)"
          % (len(texts), len(titles), time.time() - t0))
    return d


def allocation(n_neop: int, n_arxiv: int, total: int) -> dict:
    lr = np.log(n_neop) / (np.log(n_neop) + np.log(n_arxiv))
    quota_neop = int(round(total * lr))
    take_neop = min(n_neop, quota_neop)                 # feasible: never exceed
    take_arxiv = min(n_arxiv, total - take_neop)
    return {"log_ratio_neop": lr, "quota_neop": quota_neop, "have_neop": n_neop,
            "take_neop": take_neop, "take_arxiv": take_arxiv,
            "infeasible": quota_neop > n_neop}


# ------------------------------------------------------------ vocabulary ----
def derive_vocab(texts: list[str], label: str) -> dict:
    """bpe-bm25 steps 2-8 on `texts`: soft band f=0 survivors -> BPE pieces."""
    from tokenizers import Tokenizer, models, trainers

    t0 = time.time()
    BM, TF, terms, df = bm25_matrix([tokenize(t) for t in texts])
    n_chunks = TF.shape[0]
    score = BM.tocsc().max(0).toarray().ravel() * np.sqrt(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        ls = np.log2(np.where(score > 0, score, np.nan))
    elig = (df >= 2) & anomaly_mask(terms, texts) & np.isfinite(ls)
    keep = soft_keep(ls, np.where(elig)[0], df, n_chunks, 0.0)
    survivors = [str(t) for t in terms[np.where(keep)[0]]]

    alphabet = len({c for t in survivors for c in t})
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tr = trainers.BpeTrainer(vocab_size=alphabet + N_MERGES + 1, min_frequency=2,
                             special_tokens=["[UNK]"], show_progress=False)
    tok.train_from_iterator(survivors, trainer=tr)
    pieces = sorted(p for p in tok.get_vocab() if p != "[UNK]")
    print("[vocab %-6s] chunks=%d words=%d eligible=%d survivors=%d pieces=%d  %.0fs"
          % (label, n_chunks, len(terms), int(elig.sum()), len(survivors),
             len(pieces), time.time() - t0))
    return {"tok": tok, "pieces": pieces, "survivors": survivors, "cache": {}}


def piece_fn(v: dict):
    tok, cache = v["tok"], v["cache"]

    def f(text_or_words):
        words = tokenize(text_or_words) if isinstance(text_or_words, str) else text_or_words
        out = []
        for w in words:
            p = cache.get(w)
            if p is None:
                p = cache[w] = tok.encode(w).tokens
            out.extend(p)
        return out
    return f


def piece_bm25(texts: list[str], v: dict):
    pf = piece_fn(v)
    BM, TF, terms, df = bm25_matrix([pf(t) for t in texts])
    return BM, terms, df, pf


# ------------------------------------------------------------------ report ----
def row(name, vocab, r):
    print("  %-26s %7d %4d  %6.3f %6.3f %6.3f  %6.3f  %6.3f %6.3f"
          % (name, vocab, r["n_queries"], r["hit"][1], r["hit"][5], r["hit"][10],
             r["recall"][10], r["recall"][50], r["mrr"]))


HDR = "  %-26s %7s %4s  %6s %6s %6s  %6s  %6s %6s" % (
    "arm", "vocab", "nq", "hit@1", "hit@5", "hit@10", "rec@10", "rec@50", "MRR")


def truncate_rows(BM, cap=ss.SPARSEVEC_MAX_NNZ):
    """Zero all but the `cap` heaviest entries per row -- what psql will hold (V2)."""
    BM = BM.tocsr(copy=True)
    nnz = np.diff(BM.indptr)
    heavy = np.where(nnz > cap)[0]
    for i in heavy:
        s, e = BM.indptr[i], BM.indptr[i + 1]
        d = BM.data[s:e]
        drop = np.argsort(-d)[cap:]
        d[drop] = 0.0
    BM.eliminate_zeros()
    return BM, int(heavy.size)


def metrics_from_ranked(ranked_ords_per_query, rels, ks=KS):
    hits = {k: 0 for k in ks}
    rec = {k: [] for k in ks}
    rr = []
    for order, rel in zip(ranked_ords_per_query, rels):
        first = next((r for r, o in enumerate(order, 1) if o in rel), None)
        rr.append(1.0 / first if first else 0.0)
        for k in ks:
            inter = len(set(order[:k]) & rel)
            hits[k] += 1 if inter else 0
            rec[k].append(inter / len(rel))
    n = max(len(rels), 1)
    return {"n_queries": len(rels), "hit": {k: hits[k] / n for k in ks},
            "recall": {k: float(np.mean(rec[k])) if rec[k] else 0.0 for k in ks},
            "mrr": float(np.mean(rr)) if rr else 0.0}


# -------------------------------------------------------------------- main ----
def main():
    _utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--no-psql", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    A = arxiv_full()
    ntexts, ncids, nq_head, _ = neop_gold()
    a_texts, a_cids = A["texts"], A["cids"]

    # ---------------------------------------------------------- ALLOCATION
    al = allocation(len(ntexts), len(a_texts), args.total)
    print("\n== ALLOCATION for %d ==" % args.total)
    print("  log ratio: neop share %.4f -> quota %d, neop HAS %d -> %s"
          % (al["log_ratio_neop"], al["quota_neop"], al["have_neop"],
             "INFEASIBLE, capped" if al["infeasible"] else "feasible"))
    print("  take: neop %d (all) + arxiv %d of %d (%.1f%%)"
          % (al["take_neop"], al["take_arxiv"], len(a_texts),
             100.0 * al["take_arxiv"] / len(a_texts)))
    a_pick = np.sort(rng.choice(len(a_texts), size=al["take_arxiv"], replace=False))
    sub_texts = ntexts + [a_texts[i] for i in a_pick]
    sub_cids = list(ncids) + [a_cids[i] for i in a_pick]
    sub_src = ["neop"] * len(ntexts) + ["arxiv"] * len(a_pick)
    print("  sample chunks=%d" % len(sub_texts))

    # ---------------------------------------------------------- GOLD (full)
    a_by_doc = {}
    for k, c in enumerate(a_cids):
        a_by_doc.setdefault(c, set()).add(k)
    q_title = [(t, d) for d, t in A["titles"].items() if t]
    q_body = body_queries(a_texts, a_cids, np.random.default_rng(SEED), n=200)
    a_rel = lambda did: a_by_doc.get(did, set())
    print("  gold: arxiv titles=%d  arxiv body=%d  neop headings=%d"
          % (len(q_title), len(q_body), len(nq_head)))

    # ---------------------------------------------------------- VOCABULARIES
    print("\n== VOCABULARY: subset-derived vs full-derived ==")
    v_sub = derive_vocab(sub_texts, "subset")
    v_full = derive_vocab(a_texts + ntexts, "full")
    inter = len(set(v_sub["pieces"]) & set(v_full["pieces"]))
    print("  piece overlap: %d shared / %d subset / %d full  (Jaccard %.3f)"
          % (inter, len(v_sub["pieces"]), len(v_full["pieces"]),
             inter / len(set(v_sub["pieces"]) | set(v_full["pieces"]))))

    # ---------------------------------------------------------- TRANSFER (numpy, FULL arxiv)
    print("\n== TRANSFER: evaluated on the FULL arxiv corpus (%d chunks) ==" % len(a_texts))
    # Build each full-corpus matrix ONCE and score both gold sets against it.
    # The first version rebuilt the 169k-chunk piece matrix inside the gold loop
    # (twice per vocabulary) while the word matrix stayed resident -- and the
    # run was reaped for system memory. Peak is one matrix at a time now.
    import gc
    golds = (("title -> own paper", q_title), ("body sentence -> own paper", q_body))
    results = {g: [] for g, _ in golds}

    BMw, TFw, termsw, dfw = bm25_matrix([tokenize(t) for t in a_texts])
    allw = np.ones(len(termsw), dtype=bool)
    for gname, gq in golds:
        results[gname].append(("words, full vocab (baseline)", len(termsw),
                               evaluate(BMw, termsw, allw, gq, a_rel)))
    del BMw, TFw, allw
    gc.collect()

    for vname, v in (("pieces, FULL-derived", v_full), ("pieces, SUBSET-derived", v_sub)):
        BMp, termsp, dfp, pf = piece_bm25(a_texts, v)
        allp = np.ones(len(termsp), dtype=bool)
        for gname, gq in golds:
            results[gname].append((vname, len(termsp),
                                   evaluate(BMp, termsp, allp, gq, a_rel, qtok=pf)))
        del BMp, allp
        gc.collect()

    for gname, _ in golds:
        print("  [%s]" % gname)
        print(HDR)
        for name, vocab, r in results[gname]:
            row(name, vocab, r)

    if args.no_psql:
        return

    # ---------------------------------------------------------- PSQL (the sample)
    print("\n== PSQL sparsevec: the %d-chunk sample, SUBSET-derived pieces ==" % len(sub_texts))
    label = "sub%dk" % (args.total // 1000)
    BMs, termss, dfs, pfs = piece_bm25(sub_texts, v_sub)
    dim = len(termss)
    BMs_t, n_trunc = truncate_rows(BMs)
    print("  dim=%d  rows over %d nnz truncated: %d" % (dim, ss.SPARSEVEC_MAX_NNZ, n_trunc))

    # gold restricted to docs in the sample, ords are sample row indices
    s_by_doc = {}
    for k, c in enumerate(sub_cids):
        s_by_doc.setdefault(c, set()).add(k)
    picked_docs = {a_cids[i] for i in a_pick}
    gq_title = [(t, d) for t, d in q_title if d in picked_docs]
    gq_body = [(t, d) for t, d in q_body if d in picked_docs]
    # neop headings: relevant sets were neop row indices; neop rows are first in sample, unchanged
    gq_neop = [(h, rel) for h, rel in nq_head]
    golds = (("arxiv title", gq_title, lambda d: s_by_doc.get(d, set())),
             ("arxiv body", gq_body, lambda d: s_by_doc.get(d, set())),
             ("neop heading", gq_neop, lambda rel: rel))
    print("  sample gold: title=%d body=%d neop=%d" % (len(gq_title), len(gq_body), len(gq_neop)))

    conn = ss.connect()
    t = ss.ensure_schema(conn, label, dim)
    ss.write_vocab(conn, label, [str(x) for x in termss],
                   idf=np.log(1 + (BMs.shape[0] - dfs + 0.5) / (dfs + 0.5)))
    t0 = time.time()
    csr = BMs.tocsr()
    def rows():
        for i in range(csr.shape[0]):
            s, e = csr.indptr[i], csr.indptr[i + 1]
            yield i, sub_cids[i], sub_src[i], dict(zip(csr.indices[s:e].tolist(), csr.data[s:e].tolist()))
    w = ss.write_chunks(conn, label, dim, rows())
    print("  wrote %d rows (%d truncated) in %.0fs -> %s" % (w["written"], w["truncated"], time.time() - t0, t))
    print("  hnsw build %.0fs (m=16, ef_construction=64)" % ss.create_hnsw(conn, label))
    print("  " + str(ss.counts(conn, label)))

    col = {str(x): j for j, x in enumerate(termss)}
    BMs_tc = BMs_t.tocsc()
    K = max(KS)
    for gname, gq, rel_of in golds:
        qcols_list, rels = [], []
        for q, target in gq:
            cols = sorted({col[p] for p in pfs(q) if p in col})
            rel = rel_of(target)
            if cols and rel:
                qcols_list.append(cols); rels.append(rel)
        if not rels:
            continue
        print("\n  [%s] n=%d" % (gname, len(rels)))
        print(HDR)
        # numpy on the truncated matrix -- what psql holds
        np_ranked = []
        for cols in qcols_list:
            s = np.asarray(BMs_tc[:, cols].sum(axis=1)).ravel()
            np_ranked.append(np.argsort(-s, kind="stable")[:K].tolist())
        row("numpy exact (truncated)", dim, metrics_from_ranked(np_ranked, rels))
        # psql exact scan
        t0 = time.time()
        pg_ranked = [[o for o, _ in ss.query(conn, label, dim, cols, K, exact=True)]
                     for cols in qcols_list]
        lat = (time.time() - t0) / len(rels) * 1000
        r = metrics_from_ranked(pg_ranked, rels)
        row("psql exact scan  %5.0fms/q" % lat, dim, r)
        agree = np.mean([set(a[:10]) == set(b[:10]) for a, b in zip(np_ranked, pg_ranked)])
        print("  exact-scan top-10 set == numpy top-10 set on %.1f%% of queries%s"
              % (100 * agree, "" if agree >= 0.99 else "   <-- PARITY DEFECT"))
        for ef in EF_SEARCH:
            t0 = time.time()
            h_ranked = [[o for o, _ in ss.query(conn, label, dim, cols, K, ef_search=ef)]
                        for cols in qcols_list]
            lat = (time.time() - t0) / len(rels) * 1000
            row("psql hnsw ef=%-4d %5.0fms/q" % (ef, lat), dim, metrics_from_ranked(h_ranked, rels))
    conn.close()


if __name__ == "__main__":
    main()
