"""diag_domain_terms.py -- the domain term-analysis report.

NO GOVERNING SPEC. Basis: operator instruction 2026-09-26, planned in
C:\\Users\\user\\.claude\\plans\\i-ve-been-thinking-about-quiet-cray.md.
Task: playbook.md T106.

Deterministic: no model call anywhere, so every number reruns identically.
A rerun that differs is a defect, not variance.

Run (cmd):
    set PYTHONPATH=.
    python tools\\diag_domain_terms.py            REM 200 arxiv docs
    python tools\\diag_domain_terms.py --arxiv 0  REM all of them
"""
from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

from domain_corpora import chunk_corpus, derive_pack_target, load_arxiv, load_neop
from domain_terms import (bpe_by_merges, bpe_over_terms, probe_at_cuts,
                          probe_recall, select)


def _pct(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * q))]


def build(domain: str, n_arxiv: int):
    t0 = time.time()
    if domain == "neop":
        ids, docs, srcs = load_neop()
        fit = derive_pack_target([d for i, d in zip(ids, docs) if "Lewy" in i])
    else:
        ids, docs, srcs = load_arxiv(n_docs=n_arxiv or None)
        fit = derive_pack_target(docs)
    cids, texts, csrcs = chunk_corpus(ids, docs, srcs, fit["target"])
    print("[%s] docs=%d chunks=%d target=%.0f lam=%.4f  load+chunk %.1fs"
          % (domain, len(docs), len(texts), fit["target"], fit["lam"],
             time.time() - t0))
    return texts, fit


def band_table(res):
    print("  band   n     lo      hi    mean      sd  median     MAD    thr   binds     adm   frac")
    for r in res["bands"]:
        print("  %4d %4d %6.2f %7.2f %7.2f %7.3f %7.2f %7.3f %6.2f %-10s %5d %5.1f%%"
              % (r["band"], r["n"], r["lo"], r["hi"], r["mean"], r["sd"],
                 r["median"], r["mad"], r["threshold"], r["binds"],
                 r["admitted"], 100.0 * r["frac"]))
    fr = [r["frac"] for r in res["bands"]]
    if fr:
        print("  admitted-fraction spread: min=%.1f%% p50=%.1f%% max=%.1f%%  (flat => the stage is inert)"
              % (100 * min(fr), 100 * statistics.median(fr), 100 * max(fr)))
        binds = [r["binds"] for r in res["bands"]]
        print("  branch that binds: parametric=%d robust=%d"
              % (binds.count("parametric"), binds.count("robust")))


def outlier_report(res, length_k):
    """T6: the check that can overturn the no-compound-detector decision."""
    from salient_grams import anomaly_mask
    terms, df = res["terms"], res["df"]
    alpha = np.array([t.isalpha() for t in terms])
    lens = np.array([len(t) for t in terms], float)
    a = lens[alpha]
    from scipy import stats as st
    bound = np.median(a) + length_k * 1.4826 * st.median_abs_deviation(a)
    flagged = np.where(alpha & (lens > bound))[0]
    print("  alpha term length: median=%.0f MAD=%.0f -> bound=%.1f chars"
          % (np.median(a), st.median_abs_deviation(a), bound))
    print("  length-flagged terms: %d of %d" % (flagged.size, len(terms)))
    if flagged.size:
        dfs = df[flagged]
        order = flagged[np.argsort(-df[flagged])]
        print("  their df: p50=%d max=%d   df==1: %d of %d (%.0f%%)"
              % (int(np.median(dfs)), int(dfs.max()),
                 int((dfs == 1).sum()), flagged.size,
                 100.0 * (dfs == 1).sum() / flagged.size))
        print("  top by df: " + ", ".join(
            "%s(df=%d)" % (terms[i], df[i]) for i in order[:8]))
        print("  --> if these df values sit well above 1, the hapax reasoning")
        print("      behind 'no compound detector' is FALSIFIED here.")
    print("  df==1 terms removed by df_min=2: %d" % int((df == 1).sum()))


def run_domain(domain: str, n_arxiv: int, n_bands: int):
    texts, fit = build(domain, n_arxiv)

    t0 = time.time()
    res = select(texts, n_bands=n_bands)
    print("  select %.1fs  stages=%s" % (time.time() - t0, res["stages"]))
    print("  n_chunks=%d  df_max_frac cut at df>%.0f"
          % (res["n_chunks"], res["params"]["df_max_frac"] * res["n_chunks"]))

    print("\n-- bands (log2 BM25, equal-count, min-8 merge) --")
    band_table(res)

    print("\n-- outlier mask --")
    outlier_report(res, res["params"]["length_k"])

    print("\n-- probe recall at matched size (the gate's test) --")
    banded = probe_recall(res["selected_terms"], domain)
    glob_res = select(texts, n_bands=n_bands, use_bands=False)
    # match size: take the top-N global by score, N = banded count
    n = len(res["selected"])
    ls = res["log_score"]
    elig = np.where(np.isfinite(ls) & (res["df"] >= 2))[0]
    topn = elig[np.argsort(-ls[elig])][:n]
    top_terms = [str(res["terms"][i]) for i in topn]
    tp = probe_recall(top_terms, domain)
    print("  banded      n_vocab=%5d  probe %2d/%2d recall=%.2f"
          % (n, banded["n_hit"], banded["n_probe"], banded["recall"]))
    print("  global-topN n_vocab=%5d  probe %2d/%2d recall=%.2f"
          % (len(top_terms), tp["n_hit"], tp["n_probe"], tp["recall"]))
    print("  global-thr  n_vocab=%5d" % len(glob_res["selected"]))
    verdict = ("BANDING EARNS IT" if banded["recall"] > tp["recall"]
               else "BANDING DOES NOT EARN IT at matched size")
    print("  --> %s" % verdict)
    print("  banded missed: %s" % ", ".join(banded["missed"][:12]))

    print("\n-- diagnostic recall at matched cardinality cuts (rank binds here) --")
    rows = probe_at_cuts(res, domain)
    print("  cut     " + "".join("%7d" % r["cut"] for r in rows))
    print("  hit     " + "".join("%7d" % r["hit"] for r in rows)
          + "   (of %d)" % rows[0]["of"])

    print("\n-- BPE over survivors, both parameterisations (T107) --")
    print("  scale_factor (vocab = n_terms * f):")
    for sf in (0.2, 0.33, 0.5, 2, 3, 5):
        b = bpe_over_terms(res["selected_terms"], sf)
        print("    f=%-5s target=%6d actual=%6d  single-token %5.1f%%  pieces mean=%.2f max=%d"
              % (sf, b["target_vocab"], b["actual_vocab"],
                 100 * b["whole_frac"], b["mean_pieces"], b["max_pieces"]))
    print("  merge budget (trigram.md:58, min pair freq 2):")
    for nm in (300, 1000, 3000, 10000):
        b = bpe_by_merges(res["selected_terms"], nm)
        print("    merges=%-6d alphabet=%3d actual=%6d  single-token %5.1f%%  pieces mean=%.2f max=%d"
              % (nm, b["alphabet"], b["actual_vocab"],
                 100 * b["whole_frac"], b["mean_pieces"], b["max_pieces"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv", type=int, default=200,
                    help="arxiv docs; 0 = all 2233")
    ap.add_argument("--bands", type=int, default=10)
    args = ap.parse_args()

    out = {}
    for dom in ("neop", "arxiv"):
        print("\n" + "=" * 78)
        out[dom] = run_domain(dom, args.arxiv, args.bands)

    print("\n" + "=" * 78)
    print("CROSS-DOMAIN DIFF -- the deliverable")
    a = set(out["neop"]["selected_terms"])
    b = set(out["arxiv"]["selected_terms"])
    print("neop=%d arxiv=%d  shared=%d  neop-only=%d  arxiv-only=%d"
          % (len(a), len(b), len(a & b), len(a - b), len(b - a)))

    for dom, only in (("neop", a - b), ("arxiv", b - a)):
        res = out[dom]
        idx = {str(t): i for i, t in enumerate(res["terms"])}
        ranked = sorted(only, key=lambda t: -res["log_score"][idx[t]])
        print("\ntop 30 %s-ONLY terms (by log2 BM25):" % dom)
        for i in range(0, min(30, len(ranked)), 6):
            print("   " + "  ".join("%-16s" % t for t in ranked[i:i + 6]))


if __name__ == "__main__":
    main()
