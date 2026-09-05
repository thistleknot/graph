"""Populate the jsonb graph from a real corpus.

Mirrors the SAMPLE step in graph3d.py (Brown files, strided for genre spread —
the same corpus chunkgraph.py:50 cites for its validation), fits a sparse-only
ChunkGraph (embed_fn=None, so no model2vec/torch), and persists via pg_store.

Usage:
    python ingest_brown.py [label] [n_docs] [stride]   # stride defaults to 1 when n_docs > 50

Set CHUNKGRAPH_MODEL_DIR to a local sentence-transformer directory to exercise
the DENSE arm as well; unset, the run stays sparse-only per R5. The dense arm is
what makes edge provenance discriminating -- without it every edge is `sparse`
and the provenance column carries no information. The walker's default model
dir is ~/models/m2v-minilm-l6-256 (config.MODEL_DIR); this script deliberately
does NOT fall back to it -- absent env here means sparse-only, not a default.

Swap `load_docs()` for your own corpus; everything downstream is unchanged.
"""
import sys, os
from nltk.corpus import brown

from chunkgraph import ChunkGraph
import pg_store

N_DOCS = 50               # graph3d.py default
STRIDE = 10               # strided for genre spread


def load_docs(n_docs=N_DOCS, stride=STRIDE):
    """Returns (doc_ids, docs). Replace this to point at your own corpus.

    Brown has 500 files; stride=10 spreads a small sample across genres and
    caps n_docs at 50 SILENTLY (500/10). Pass stride=1 for the whole corpus."""
    fids = brown.fileids()[::stride][:n_docs]
    docs = []
    for fid in fids:
        paras = ['\n'.join(' '.join(s) for s in p)
                 for p in brown.paras(fileids=fid)]
        docs.append('\n\n'.join(paras))
    return fids, docs


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "brown-50"
    n_docs = int(sys.argv[2]) if len(sys.argv) > 2 else N_DOCS
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else (1 if n_docs > 50 else STRIDE)

    print(f"DSN   : {pg_store.DSN}")
    doc_ids, docs = load_docs(n_docs, stride)
    print(f"corpus: {len(docs)} docs, {sum(len(d) for d in docs):,} chars")

    model_dir = os.environ.get("CHUNKGRAPH_MODEL_DIR")   # R5: absent -> sparse-only
    print(f"dense : {model_dir or 'DISABLED (sparse-only, R5)'}")
    cg = ChunkGraph(model_dir=model_dir) if model_dir else ChunkGraph(embed_fn=None)
    cg.fit(docs, doc_ids=doc_ids)
    print(f"fitted: {cg.n} chunks, blend_mode={cg.blend_mode}", flush=True)

    edges = cg.edges()
    comms = cg.communities(min_size=5)
    print(f"graph : {len(edges)} edges, {len(comms)} communities (min_size=5)")

    run_id = pg_store.save(cg, label)
    print(f"run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    main()
