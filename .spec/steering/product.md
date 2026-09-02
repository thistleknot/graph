---
inclusion: always
---

# Product

## Purpose

A **dual-space retrieval graph over arbitrary text**. Chunks become nodes; edges
are drawn from two independent similarity spaces (sparse BM25 and dense
embedding), normalized, thresholded, fused with provenance, and persisted to
Postgres for retrieval.

The organizing commitment: **graph construction is fully deterministic.** No LLM
participates in chunking, edge formation, or community assignment. Every edge
can name the documents it came from (R8) and every run is reproducible from
stored params. The corpus may be several corpora — Brown, quotes, and wikitext
ingest side by side into one graph, each fitted on its own scale and each chunk
tagged with its source (R19–R21).

## Users

- Primary: single operator/developer running ingest and query locally. *(inferred
  from repo shape — no auth, no multi-tenancy, DSN defaults to localhost)*
- Planned: an **SME reviewer** who confirms/rejects/renames Louvain communities.
  Not built. See the community-labelling spec.

## Key capabilities

Staged pipeline, per `chunkgraph.py`:

| Stage | Mechanism |
|---|---|
| CHUNK | one node per document; a document splits only above its SOURCE's Box-Cox hi (lines per document, one fit per source — R19/R17), at paragraph boundaries, short tail merged back; source-prefixed doc_id retained on every node (R8, R20) |
| PHRASE | Dunning-LLR + NPMI merged into tokens pre-BM25 (R9, optional) |
| SPARSE | BM25-weighted CSR, L2 rows, blockwise `X@X.T`, in-loop threshold (R10) |
| DENSE | pluggable `embed_fn`, default local MiniLM mean-pool (R5, optional) |
| NORMAL | Box-Cox per source-pair block (3 intra + 3 cross), estimator-pair gate per block, z values reassembled (R21) |
| EDGES | k-sigma tail cut, budget-match fallback when kurtosis high (R2); ONE global k-sigma cut over the reassembled z, never per block (R21) |
| BACKBONE | per-node top-KNN unioned in, both spaces (R7) |
| FUSE | union + provenance `{sparse,dense,both}`, BC-z strength, rank-blend fallback (R3) |
| COMMUNITY | Louvain on union (weight=strength), tf\*idf keywords, blend medoids |
| SERVE | `query()`: validated anchors → H-hop frontier → GIST-walk (R1, R4) |
| DRAW / EXPORT | spring layout, provenance-colored edges; bitemporal edge rows |

Persistence (`pg_store.py`): one `graph_run` plus nodes/edges/communities/
embeddings, committed atomically. Prior live runs under the same label are
**superseded, never deleted.**

## What success means

- Retrieval quality is attributable — every result traces to source docs.
- Reruns are reproducible; `_PARAM_ATTRS` captures what varies a run.
- Degradation is graceful: no embed model ⇒ sparse-only still works (R5).
- Categorization stays falsifiable by a human without trusting model prose.
