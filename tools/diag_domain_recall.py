"""diag_domain_recall.py -- retrieval recall, and the sigma factor tuned against it.

NO GOVERNING SPEC. Basis: operator instruction 2026-09-26 -- "factor should be set
by best recall / that's the whole point of the 'tuning'". Task: playbook.md T111.

The soft-band sigma factor is a hyperparameter. Diagnostic term coverage cannot
tune it: measured, every factor from 0.0 to 1.0 yields an IDENTICAL top-1000, so
that metric is blind to the knob. This module supplies the metric that is not.

GOLD, deterministic and free -- no judge, no model, no hand labelling:

    query    = the paper's title (first ATX heading of its docling markdown)
    relevant = every chunk belonging to that paper
    neop     = the Lewy book's section headings, relevant = that section's chunks

The title line itself is dropped by domain_corpora.md_units (it is a heading), so
the query text is NOT verbatim in any chunk. This is a retrieval task, not a
string lookup.

WHY BM25-WITH-FULL-VOCABULARY IS THE BASELINE, not embeddings: the only thing the
arms change is which terms may contribute to a score. Holding the chunks, the
scorer and the queries fixed and varying only the allowed columns attributes the
effect to the vocabulary. Dense retrieval changes the scorer AND the space, so it
answers a different question ("is sparse or dense better here"), which needs its
own experiment.

Scoring is exact and vectorised: a chunk's score for a query is the sum of its
BM25 weights over the query's terms, i.e. `BM[:, qcols].sum(1)`. Restricting
`qcols` to the selected vocabulary IS the index restriction.

Run (cmd):
    set PYTHONPATH=.
    python tools\\diag_domain_recall.py --arxiv 200
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _utf8_stdout():
    """Greek and math glyphs need utf-8 on a cp1252 console.

    Called from main() ONLY. Rewrapping at import time replaced pytest's captured
    stdout and every test that imported this module died with
    "ValueError: I/O operation on closed file".
    """
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

import numpy as np
from scipy import sparse, stats

from domain_corpora import (chunk_corpus, derive_pack_target, load_arxiv,
                            load_neop, md_sections)
from salient_grams import anomaly_mask, bm25_matrix
from stoplist import tokenize

FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0)
KS = (1, 5, 10, 50)
_ATX = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(\S.*)$")


def first_heading(text: str) -> str | None:
    """The docling title: the first ATX heading with enough words to be a query."""
    for ln in text.split("\n"):
        m = _ATX.match(ln)
        if m:
            t = m.group(1).strip()
            if len(t.split()) >= 4 and not t.lower().startswith("abstract"):
                return t
    return None


def arxiv_gold(n_docs):
    """(queries, relevant-doc-id) from paper titles."""
    ids, docs, srcs = load_arxiv(n_docs=n_docs)
    fit = derive_pack_target(docs)
    cids, texts, _ = chunk_corpus(ids, docs, srcs, fit["target"])
    queries = []
    for did, text in zip(ids, docs):
        h = first_heading(text)
        if h:
            queries.append((h, did))
    return texts, np.array(cids), queries, fit


def heading_sections(text: str) -> list[tuple[str, list[str]]]:
    """(heading, its paragraphs) pairs, kept ALIGNED.

    An earlier version zipped a FILTERED heading list against md_sections' output,
    which drops zero-paragraph sections -- so every heading was paired with the
    wrong section and the gold set was garbage (hit@1 = 0.000). Build the pairs in
    one pass instead; never re-pair two independently filtered lists.
    """
    def paras(buf):
        # STRIP every paragraph. Unstripped, paras[0][:120] carries a leading
        # newline into the substring probe below and only matches by accident
        # (chunks are joined with "\n\n", which happens to contain it).
        return [p.strip() for p in re.split(r"\n[ \t]*\n", "\n".join(buf))
                if p.strip()]

    out, head, buf = [], None, []
    for ln in text.split("\n"):
        m = _ATX.match(ln)
        if m:
            if head is not None and buf:
                out.append((head, paras(buf)))
            head, buf = m.group(1).strip(), []
        else:
            buf.append(ln)
    if head is not None and buf:
        out.append((head, paras(buf)))
    return [(h, ps) for h, ps in out if ps]


def body_queries(texts, cids, rng, n=200, min_words=8, max_words=24):
    """Known-item queries drawn from BODY text, to make the tail vocabulary live.

    Title queries cannot tune the soft band: measured, three sigma factors gave
    byte-identical recall because title words are central, high-BM25-mass terms
    and the band only ever trims the bottom of the distribution. A sentence from
    mid-document uses whatever vocabulary is actually there, tail included.
    """
    by_doc = {}
    for k, c in enumerate(cids):
        by_doc.setdefault(c, []).append(k)
    out = []
    for did in sorted(by_doc):
        ks = by_doc[did]
        k = ks[len(ks) // 2]                     # mid-document chunk
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", texts[k])
                 if min_words <= len(s.split()) <= max_words]
        if not sents:
            continue
        out.append((sents[rng.integers(len(sents))], did))
        if len(out) >= n:
            break
    return out


def neop_gold():
    """(queries, relevant-chunk-index-set) from the Lewy book's section headings.

    Thin -- one book, ~30 usable sections -- and reported as such.
    """
    ids, docs, srcs = load_neop()
    fit = derive_pack_target([d for i, d in zip(ids, docs) if "Lewy" in i])
    cids, texts, _ = chunk_corpus(ids, docs, srcs, fit["target"])

    lewy_id, lewy_text = [(i, d) for i, d in zip(ids, docs) if "Lewy" in i][0]
    lewy_chunks = [k for k, c in enumerate(cids) if c == lewy_id]
    queries = []
    for head, paras in heading_sections(lewy_text):
        if len(head.split()) < 2:
            continue
        probe = paras[0][:120]
        rel = {k for k in lewy_chunks if probe in texts[k]}
        if rel:
            queries.append((head, rel))
    return texts, cids, queries, fit


def soft_keep(ls, idx, df, n_chunks, factor, df_max_frac=0.50):
    """The operator's soft band, applied across the whole distribution."""
    v = ls[idx]
    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    thr = min(float(np.mean(v)) - factor * sd,
              float(np.median(v)) - factor * 1.4826
              * float(stats.median_abs_deviation(v)))
    keep = np.zeros(ls.shape, dtype=bool)
    keep[idx[v >= thr]] = True
    keep &= df <= df_max_frac * n_chunks
    return keep


def evaluate(BM, terms, keep, queries, rel_of, ks=KS, qtok=tokenize):
    """Recall and MRR for one vocabulary. `rel_of` maps a query to relevant rows.

    `qtok` turns a query string into index units -- words by default, or a
    piece tokenizer when BM is built over BPE pieces. One scorer for both, so a
    word-vs-piece comparison cannot differ in anything but the vocabulary.
    """
    col = {str(t): j for j, t in enumerate(terms)}
    allowed = keep
    BMc = BM.tocsc()
    hits = {k: 0 for k in ks}
    rec = {k: [] for k in ks}
    rr = []
    scored = 0
    for q, target in queries:
        cols = [col[t] for t in set(qtok(q))
                if t in col and allowed[col[t]]]
        if not cols:
            continue
        scored += 1
        s = np.asarray(BMc[:, cols].sum(axis=1)).ravel()
        order = np.argsort(-s)
        rel = rel_of(target)
        if not rel:
            scored -= 1
            continue
        first = None
        for rank, r in enumerate(order[:max(ks)], start=1):
            if r in rel:
                first = rank
                break
        rr.append(1.0 / first if first else 0.0)
        for k in ks:
            top = set(order[:k].tolist())
            inter = len(top & rel)
            hits[k] += 1 if inter else 0
            rec[k].append(inter / len(rel))
    n = max(scored, 1)
    return {"n_queries": scored,
            "hit": {k: hits[k] / n for k in ks},
            "recall": {k: float(np.mean(rec[k])) if rec[k] else 0.0 for k in ks},
            "mrr": float(np.mean(rr)) if rr else 0.0}


def bpe_arms(tag, texts, queries, rel_of, selected_terms, merge_budgets=(1000, 3000, 10000)):
    """The claim actually under test: does a SUBWORD index approximate BM25?

    Every arm above keeps whole terms. This re-tokenises chunks AND queries into
    BPE pieces trained on the surviving terms only (T7), builds BM25 over the
    pieces, and scores the same gold. No OOV is possible -- every word decomposes
    -- so this is the "smaller corpus (vocab)" the operator asked for, measured
    on retrieval rather than on piece statistics.
    """
    from tokenizers import Tokenizer, models, trainers

    print("\n  -- BPE subword index over the SELECTED terms (%d), same gold --"
          % len(selected_terms))
    print("  arm                   vocab   nq   hit@1  hit@5 hit@10  rec@10  rec@50    MRR")
    word_docs = [tokenize(t) for t in texts]
    alphabet = len({c for t in selected_terms for c in t})
    for nm in merge_budgets:
        t0 = time.time()
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tr = trainers.BpeTrainer(vocab_size=alphabet + nm + 1, min_frequency=2,
                                 special_tokens=["[UNK]"], show_progress=False)
        tok.train_from_iterator(selected_terms, trainer=tr)
        cache = {}

        def pieces(words):
            out = []
            for w in words:
                p = cache.get(w)
                if p is None:
                    p = cache[w] = tok.encode(w).tokens
                out.extend(p)
            return out

        piece_docs = [pieces(d) for d in word_docs]
        BM, TF, terms, df = bm25_matrix(piece_docs)
        col = {str(t): j for j, t in enumerate(terms)}
        BMc = BM.tocsc()
        ks = KS
        hits = {k: 0 for k in ks}
        rec = {k: [] for k in ks}
        rr, scored = [], 0
        for q, target in queries:
            cols = [col[p] for p in set(pieces(tokenize(q))) if p in col]
            rel = rel_of(target)
            if not cols or not rel:
                continue
            scored += 1
            s = np.asarray(BMc[:, cols].sum(axis=1)).ravel()
            order = np.argsort(-s)
            first = next((r for r, i in enumerate(order[:max(ks)], 1) if i in rel), None)
            rr.append(1.0 / first if first else 0.0)
            for k in ks:
                inter = len(set(order[:k].tolist()) & rel)
                hits[k] += 1 if inter else 0
                rec[k].append(inter / len(rel))
        n = max(scored, 1)
        print("  BPE merges=%-6d %8d %4d  %6.3f %6.3f %6.3f  %6.3f  %6.3f %6.3f   (%.0fs)"
              % (nm, len(terms), scored, hits[1] / n, hits[5] / n, hits[10] / n,
                 float(np.mean(rec[10])), float(np.mean(rec[50])),
                 float(np.mean(rr)), time.time() - t0))


def sweep(tag, texts, queries, rel_of):
    print("\n" + "=" * 78)
    print("%s  chunks=%d  queries=%d" % (tag, len(texts), len(queries)))
    t0 = time.time()
    BM, TF, terms, df = bm25_matrix([tokenize(t) for t in texts])
    n_chunks = TF.shape[0]
    score = BM.tocsc().max(0).toarray().ravel() * np.sqrt(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        ls = np.log2(np.where(score > 0, score, np.nan))
    elig = (df >= 2) & anomaly_mask(terms, texts) & np.isfinite(ls)
    idx = np.where(elig)[0]
    print("full vocab=%d  eligible=%d   bm25 %.1fs"
          % (len(terms), idx.size, time.time() - t0))

    arms = [("full (no selection)", np.ones(len(terms), dtype=bool), None)]
    for f in FACTORS:
        arms.append(("soft band f=%.2f" % f,
                     soft_keep(ls, idx, df, n_chunks, f), f))

    print("\n  arm                   vocab   nq   hit@1  hit@5 hit@10  rec@10  rec@50    MRR")
    rows = []
    for name, keep, f in arms:
        r = evaluate(BM, terms, keep, queries, rel_of)
        rows.append((name, f, int(keep.sum()), r))
        print("  %-20s %7d %4d  %6.3f %6.3f %6.3f  %6.3f  %6.3f %6.3f"
              % (name, int(keep.sum()), r["n_queries"],
                 r["hit"][1], r["hit"][5], r["hit"][10],
                 r["recall"][10], r["recall"][50], r["mrr"]))

    band_rows = [x for x in rows if x[1] is not None]
    best = max(band_rows, key=lambda x: (x[3]["recall"][10], x[3]["mrr"]))
    base = rows[0]
    print("\n  BEST soft-band factor by recall@10: f=%.2f  (recall@10=%.3f MRR=%.3f)"
          % (best[1], best[3]["recall"][10], best[3]["mrr"]))
    print("  vs full vocabulary:                        recall@10=%.3f MRR=%.3f"
          % (base[3]["recall"][10], base[3]["mrr"]))
    d = best[3]["recall"][10] - base[3]["recall"][10]
    print("  delta recall@10 = %+.3f  -> selection %s"
          % (d, "HELPS" if d > 0.005 else
             ("HURTS" if d < -0.005 else "is NEUTRAL (within 0.005)")))

    # The subword claim, on the f=0.00 survivors (the shipped factor).
    keep0 = soft_keep(ls, idx, df, n_chunks, 0.0)
    bpe_arms(tag, texts, queries, rel_of,
             [str(t) for t in terms[np.where(keep0)[0]]])
    return best[1]


def main():
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(0)               # determinism: fixed seed
    texts, cids, queries, _ = arxiv_gold(args.arxiv)
    by_doc = {}
    for k, c in enumerate(cids):
        by_doc.setdefault(c, set()).add(k)
    rel_of = lambda did: by_doc.get(did, set())

    sweep("ARXIV (title -> own paper)", texts, queries, rel_of)
    sweep("ARXIV (body sentence -> own paper)", texts,
          body_queries(texts, cids, rng, n=args.arxiv), rel_of)

    ntexts, ncids, nqueries, _ = neop_gold()
    sweep("NEOP (Lewy section heading -> that section)",
          ntexts, nqueries, lambda rel: rel)
    nby = {}
    for k, c in enumerate(ncids):
        nby.setdefault(c, set()).add(k)
    sweep("NEOP (body sentence -> own document)", ntexts,
          body_queries(ntexts, ncids, rng, n=200),
          lambda did: nby.get(did, set()))


if __name__ == "__main__":
    main()
