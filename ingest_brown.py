"""Populate the jsonb graph from a real corpus.

Mirrors the SAMPLE step in graph3d.py (Brown files, strided for genre spread —
the same corpus chunkgraph.py:50 cites for its validation), fits a sparse-only
ChunkGraph (embed_fn=None, so no model2vec/torch), and persists via pg_store.

Usage:
    python ingest_brown.py [label] [n_docs]

Swap `load_docs()` for your own corpus; everything downstream is unchanged.
"""
import sys, os
from nltk.corpus import brown

from chunkgraph import ChunkGraph
import pg_store

N_DOCS = 50               # graph3d.py default
STRIDE = 10               # strided for genre spread


def load_docs(n_docs=N_DOCS):
    """Returns (doc_ids, docs). Replace this to point at your own corpus."""
    fids = brown.fileids()[::STRIDE][:n_docs]
    docs = []
    for fid in fids:
        paras = ['\n'.join(' '.join(s) for s in p)
                 for p in brown.paras(fileids=fid)]
        docs.append('\n\n'.join(paras))
    return fids, docs


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "brown-50"
    n_docs = int(sys.argv[2]) if len(sys.argv) > 2 else N_DOCS

    print(f"DSN   : {pg_store.DSN}")
    doc_ids, docs = load_docs(n_docs)
    print(f"corpus: {len(docs)} docs, {sum(len(d) for d in docs):,} chars")

    cg = ChunkGraph(embed_fn=None)          # sparse-only: no dense arm
    cg.fit(docs, doc_ids=doc_ids)
    print(f"fitted: {cg.n} chunks, blend_mode={cg.blend_mode}")

    edges = cg.edges()
    comms = cg.communities(min_size=5)
    print(f"graph : {len(edges)} edges, {len(comms)} communities (min_size=5)")

    run_id = pg_store.save(cg, label)
    print(f"run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
