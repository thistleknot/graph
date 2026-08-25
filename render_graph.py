"""Render a ChunkGraph PNG from the Brown sample — the 2D counterpart to graph3d.py.

Exists because this render was previously an inline `python -c` invocation: it
was lost between sessions and its background run left a 0-byte log (unbuffered
stdout, killed before flush). Run this with `python -u` for a live log.

Sparse-only (embed_fn=None), so no model2vec/torch and no Postgres — the draw
path reads the in-memory graph, not pg_store.

Usage:
    python -u render_graph.py [n_docs] [out.png]      # default 50 docs

Cost note: DRAW is spring_layout(iterations=200) over every node; it dominates
runtime superlinearly. 10 docs is seconds, 50 docs is minutes.
"""
import sys
import time

from chunkgraph import ChunkGraph
from ingest_brown import load_docs, N_DOCS


def main():
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else N_DOCS
    out = sys.argv[2] if len(sys.argv) > 2 else f"graph_brown{n_docs}.png"

    t0 = time.time()
    doc_ids, docs = load_docs(n_docs)
    print(f"corpus: {len(docs)} docs, {sum(len(d) for d in docs):,} chars", flush=True)

    cg = ChunkGraph(embed_fn=None)
    cg.fit(docs, doc_ids=doc_ids)
    print(f"fitted: {cg.n} chunks, blend_mode={cg.blend_mode} "
          f"[{time.time() - t0:.1f}s]", flush=True)

    comms = cg.communities(min_size=5)
    print(f"graph : {int(cg.A.sum() // 2)} edges, {len(comms)} communities "
          f"[{time.time() - t0:.1f}s]", flush=True)

    print(f"draw  : spring_layout over {cg.n} nodes, 200 iterations...", flush=True)
    cg.draw(out)
    print(f"wrote : {out} [{time.time() - t0:.1f}s total]", flush=True)
    return out


if __name__ == "__main__":
    main()
