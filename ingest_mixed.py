"""Populate the jsonb graph from a mixed multi-source corpus (Brown + quotes + wiki).

R20 body (design.md sec 6.14, lines 1089-1095): mixing heterogeneous document
sources into one ChunkGraph run so retrieval sees genre spread beyond Brown
alone -- Brown (structured prose, genre-coded), abirate/english_quotes (short,
aphoristic, high chunk-density), EleutherAI/wikitext_document_level (long,
discursive). Each source is tagged on ingest so a consumer can reproduce the
source of any doc_id from its prefix alone; retrieval itself never sees or
weights the source (forbidden by R20 -- that's not this file's job).

Usage:
    python ingest_mixed.py [label] [--brown N] [--brown-stride S]
                                     [--quotes N] [--quotes-stride S]
                                     [--wiki N] [--wiki-stride S]

Set CHUNKGRAPH_MODEL_DIR to a local sentence-transformer directory to exercise
the DENSE arm as well; unset, the run stays sparse-only per R5.

Spec: .spec/specs/graph-explorer/design.md sec 6.14 R20, sec 6.15 R22 - Task: playbook.md T2, T8, T6
"""
from __future__ import annotations

import argparse
import inspect
import sys
import time

import ingest_brown
from chunkgraph import ChunkGraph
import os
import pg_store

_START = time.time()


def _ckpt(stage, detail=""):
    """T8: stdout is buffered when redirected to a file, and plain prints
    carry no timestamp -- this is the launch-verification and staleness
    signal for the detached mixed-full run."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elapsed = time.time() - _START
    line = f"CKPT {ts} +{elapsed:.1f}s {stage}"
    if detail:
        line += f" {detail}"
    print(line, flush=True)

QUOTES_DATASET = "abirate/english_quotes"   # field: quote
WIKI_DATASET = "EleutherAI/wikitext_document_level"
WIKI_CONFIG = "wikitext-103-raw-v1"         # field: page
SOURCES = ("brown", "quotes", "wiki")


def _hf_rows(path, config, split, field):
    """Single point of network contact -- lazy import so the test suite never
    needs `datasets` installed."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(f"`datasets` package required to load {path}") from e
    ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
    return list(ds[field])


def _take(items, n, stride):
    """Drops blank/whitespace-only texts, then strides and caps at n.

    Returns (original_index, text) pairs so ids keep the pre-stride position
    (D3: quotes/wiki ids are stable under n, not under stride)."""
    non_blank = [(i, t) for i, t in enumerate(items) if t and t.strip()]
    return non_blank[::stride][:n]


def load_brown(n_docs, stride):
    """(doc_ids, docs, sources) -- delegates to ingest_brown.load_docs, the
    incumbent Brown loader, and re-prefixes its ids."""
    if n_docs <= 0:
        return [], [], []
    fids, docs = ingest_brown.load_docs(n_docs, stride)
    doc_ids = [f"brown/{fid}" for fid in fids]
    sources = ["brown"] * len(docs)
    return doc_ids, docs, sources


def load_quotes(n_docs, stride):
    """(doc_ids, docs, sources) for abirate/english_quotes."""
    if n_docs <= 0:
        return [], [], []
    rows = _hf_rows(QUOTES_DATASET, None, "train", "quote")
    taken = _take(rows, n_docs, stride)
    doc_ids = [f"quotes/{i}" for i, _ in taken]
    docs = [t for _, t in taken]
    sources = ["quotes"] * len(docs)
    return doc_ids, docs, sources


def wiki_title(page: str) -> str | None:
    """The article's own title, parsed once at the boundary from the
    '= Title =' head wikitext documents open with. Consumers select by
    EQUALITY on this -- never by scanning text with patterns (the same law
    as R20's explicit `source`: parse once, no downstream parsing)."""
    head = page.lstrip()[:200]
    if not head.startswith("="):
        return None
    rest = head[1:].lstrip()          # between the opening '=' and the next '='
    j = rest.find("=")
    if j <= 0:
        return None
    return rest[:j].strip() or None


def load_wiki(n_docs, stride, titles=None):
    """(doc_ids, docs, sources) for EleutherAI/wikitext_document_level.

    titles: article titles to include REGARDLESS of the stride, matched by
    case-insensitive equality on the parsed title. A strided sample is
    representative, but an anchor article the operator asks about must not
    be a stride victim (measured 2026-09-02: "Battle of Midway" sits at
    index 28410, 28410 % 4 == 2, absent from every stride-4 run; the
    answer was built from adjacent evidence)."""
    want = {t.strip().casefold() for t in (titles or []) if t.strip()}
    if n_docs <= 0 and not want:
        return [], [], []
    rows = _hf_rows(WIKI_DATASET, WIKI_CONFIG, "train", "page")
    taken = _take(rows, n_docs, stride) if n_docs > 0 else []
    if want:
        have = {i for i, _ in taken}
        taken += [(i, t) for i, t in enumerate(rows)
                  if i not in have and (wiki_title(t) or "").casefold() in want]
        taken.sort(key=lambda it: it[0])
    doc_ids = [f"wiki/{i}" for i, _ in taken]
    docs = [t for _, t in taken]
    sources = ["wiki"] * len(docs)
    return doc_ids, docs, sources


def load_mixed(brown, quotes, wiki, brown_stride, quotes_stride, wiki_stride,
               wiki_titles=None):
    """Concatenates the three arms in SOURCES order. Returns
    (doc_ids, docs, sources, titles). Titles are computed HERE rather than
    inside each loader (R22): loader arity stays stable for brown/quotes (they
    have no notion of a title, so their entries are all None) and the parse
    still happens exactly once, at this boundary, via wiki_title()."""
    b_ids, b_docs, b_src = load_brown(brown, brown_stride)
    q_ids, q_docs, q_src = load_quotes(quotes, quotes_stride)
    w_ids, w_docs, w_src = load_wiki(wiki, wiki_stride, titles=wiki_titles)

    doc_ids = b_ids + q_ids + w_ids
    docs = b_docs + q_docs + w_docs
    sources = b_src + q_src + w_src
    titles = [None] * len(b_docs) + [None] * len(q_docs) + [wiki_title(d) for d in w_docs]

    assert len(doc_ids) == len(docs) == len(sources) == len(titles)
    for did, src in zip(doc_ids, sources):
        assert did.startswith(f"{src}/")
    assert docs, "load_mixed produced zero docs"

    return doc_ids, docs, sources, titles


def build_argparser():
    p = argparse.ArgumentParser(description="Ingest a mixed multi-source corpus into ChunkGraph.")
    p.add_argument("label", nargs="?", default="mixed")
    p.add_argument("--brown", type=int, default=50)
    p.add_argument("--brown-stride", type=int, default=10)
    p.add_argument("--quotes", type=int, default=500)
    p.add_argument("--quotes-stride", type=int, default=1)
    p.add_argument("--wiki", type=int, default=200)
    p.add_argument("--wiki-stride", type=int, default=1)
    p.add_argument("--wiki-title", action="append", default=None, metavar="TITLE",
                   help="always include the wiki article with this exact title, "
                        "matched against the parsed article head (repeatable; "
                        "stride-proof anchors)")
    return p


def _fit(cg, docs, doc_ids, sources, titles=None):
    """The D1 seam: passes sources= (and, per R22, titles=) to fit() iff the
    parameter exists on the ChunkGraph in play, so this module needs zero
    edits when a newer fit() signature lands."""
    params = inspect.signature(cg.fit).parameters
    kw = {}
    if "sources" in params:
        print("sources: passed to fit")
        kw["sources"] = sources
    else:
        print("sources: carried in doc_id prefix only (pre-T4 fit)")
    if "titles" in params:
        print("titles: passed to fit")
        kw["titles"] = titles
    else:
        print("titles: not carried (pre-R22 fit)")
    return cg.fit(docs, doc_ids=doc_ids, **kw)


def main(argv=None):
    args = build_argparser().parse_args(argv if argv is not None else sys.argv[1:])
    _ckpt("start", f"label={args.label}")

    print(f"DSN   : {pg_store.DSN}")
    _ckpt("load:start")
    doc_ids, docs, sources, titles = load_mixed(
        args.brown, args.quotes, args.wiki,
        args.brown_stride, args.quotes_stride, args.wiki_stride,
        wiki_titles=args.wiki_title,
    )
    _ckpt("load:done", f"docs={len(docs)} chars={sum(len(d) for d in docs)} "
          f"titled={sum(1 for t in titles if t is not None)}")
    for src in SOURCES:
        print(f"{src:6s}: {sources.count(src)} docs")
    print(f"corpus: {len(docs)} docs, {sum(len(d) for d in docs):,} chars")

    model_dir = os.environ.get("CHUNKGRAPH_MODEL_DIR")   # R5: absent -> sparse-only
    print(f"dense : {model_dir or 'DISABLED (sparse-only, R5)'}")
    cg = ChunkGraph(model_dir=model_dir) if model_dir else ChunkGraph(embed_fn=None)
    _ckpt("fit:start", f"docs={len(docs)} dense={'on' if model_dir else 'off'}")
    _fit(cg, docs, doc_ids, sources, titles=titles)
    _ckpt("fit:done", f"n={cg.n}")
    print(f"fitted: {cg.n} chunks, blend_mode={cg.blend_mode}", flush=True)

    edges = cg.edges()
    comms = cg.communities(min_size=5)
    _ckpt("graph:done", f"edges={len(edges)} comms={len(comms)}")
    print(f"graph : {len(edges)} edges, {len(comms)} communities (min_size=5)")

    run_id = pg_store.save(cg, args.label)
    _ckpt("save:done", f"run_id={run_id} label={args.label}")
    print(f"run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
