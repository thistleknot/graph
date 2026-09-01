<!-- Spec: top-level e2e pipeline description. Sources: steering/{product,structure,tech}.md,
     specs/graph-explorer/design.md 6.1-6.11, specs/graph-term-selection/requirements.md.
     Guard ids cited here live in module docstrings (R=chunkgraph, S=sampler, W=graph_tools,
     I=interpret, X=export_neo4j) and each is pinned by a test. Operator request 2026-08-31. -->

# The pipeline, end to end

One prompt against one corpus, in five phases. Phases A–C are **fully
deterministic** — no model anywhere; every number is recomputable from the
stored run. Phase D is the only place a language model exists, and it is
walled off: it can rank and phrase, never retrieve, and every citation it
emits is checked against the evidence it was shown.

```
A INGEST    corpus -> chunks -> two similarity spaces -> fused edges -> communities   (offline, once)
B QUERY     prompt -> lexical anchors -> self-stopping walk -> one-degree ring        (per prompt, ~1 s)
C ANALYZE   walked subgraph -> terms, medoids, shape, pathways                        (computed, no model)
D INTERPRET evidence -> Reason / Judge -> checked citations                           (the only model calls)
E RESULTS   answer first; all evidence one expander deep                              (walker UI)
```

---

## Phase A — INGEST (build a run)

`ingest_brown.py` → `chunkgraph.py` → `pg_store.py`. Offline, deterministic,
reproducible from stored params (`_PARAM_ATTRS`).

| step | what happens | guard |
|---|---|---|
| chunk | the DOCUMENT is the unit. Lines per document → Box-Cox; a doc splits only above `hi = median + 2·MAD`, at paragraph bounds, short tail merged back. Never inside a word. `doc_id` kept | R17, R8 |
| phrases | NPMI welds collocations into single tokens (`white_citizens`, `self_help`) before any scoring; Dunning-LLR reported as diagnostic only | R9 |
| stoplist | one stoplist for build and serve, contraction-aware, read straight from the NLTK english file | R18 |
| sparse arm | BM25-weighted term matrix, L2 rows, blockwise cosine chunk×chunk. BM25 over PPMI because BM25 corrects for length; PPMI would make long chunks hubs | R10 |
| dense arm | pluggable embedder; default model2vec static vectors (MiniLM-distilled, 256d). Absent model ⇒ sparse-only still works | R5, R14 |
| normalize | Box-Cox each similarity distribution so the two spaces are commensurable |  |
| edges | k-sigma tail cut per space; budget-match fallback when kurtosis is high | R2 |
| backbone | per-node top-KNN unioned in from both spaces, so significance ranks but never isolates | R7 |
| fuse | union with **provenance** `{sparse, dense, both}` on every edge; strength persisted bounded (`1 − D`) so path products cannot amplify | R3, R15 |
| communities | Louvain (fixed seed) on the fused graph; per-community tf·idf keywords and a global medoid chunk | — |
| persist | one `graph_run` + nodes/edges/communities/embeddings, atomic; re-ingest **supersedes, never deletes** (bitemporal `valid_from`/`valid_to`) | pg_store contract |

What a run guarantees downstream: every edge names its space(s), every chunk
names its document, and the whole thing rebuilds byte-identically from params.

## Phase B — QUERY (the walk)

`sampler.py` over the read-only tool surface `graph_tools.py`. Per prompt,
deterministic, ~1 s.

1. **Anchors** — BM25 search of the prompt against a per-run cached inverted
   index (Python, ~4 ms); top-k chunks seed the walk. No lexical hit ⇒ the walk
   says so rather than walking nothing.
2. **Expansion** — HNSW-style ef-search: a frontier expands along stored edges,
   path score = product of edge strengths, and it **stops itself** when the best
   frontier candidate cannot beat the worst held result at `ef` width (tuned
   ef=64). Depth is emergent, not an operator dial. Boltzmann sampling steers
   the frontier, never the output. | S9–S12
3. **Ring** — one degree out from the top-3 chunks: their strongest unseen
   edges enter at `parent score × strength`, ≤8 per parent, never outranking
   the parent, never seeding further expansion. | S13
4. **Bundle** — the frozen evidence: ordinals, per-chunk walk scores, anchors,
   telemetry. Everything downstream derives from this one object. | S6/S7

## Phase C — ANALYZE (the walked subgraph)

All computed on the Bundle, no model. `graph_tools.py`.

| feature | what it answers | guard |
|---|---|---|
| communities touched | which stored Louvain communities the walk landed in, ranked by presence | — |
| query-conditioned terms | the community's OWN BM25 vocabulary re-ranked by the prompt (lexical first, model2vec cosine second) | 6.1 |
| unsupervised concept | community-as-document BM25 terms, prompt-independent, cached per run | — |
| medoids | per community: the **local** medoid (walk-weighted centrality — what the walk found here) and the **global** medoid (what the community is). Any ordinal can serve as an anchor; medoids are defaults, not privileged | 6.2 |
| salient titling | a chunk is titled by its own salient terms: stoplist → BM25 vs corpus → log-normal gate keep ≥ min(median−1.4826·MAD, mean−sd) | W14 |
| in between | retrieved chunks whose walked edges reach a different retrieved community — where the concepts meet | — |
| shape | WCC count and sizes, largest-component share, density, conductance of the walk against the rest of the run (how leaky the neighbourhood is) | W15 |
| pathways | idea-to-idea **DWPC**: enumerate simple paths ≤3 edges between anchor pairs, score each `Π strength × Π deg(node)^-0.4` (global degrees), sum per pair, keep the best chain for display. Hub correction is the point: many specific paths beat one path through a hub | W15, 6.11 |

## Phase D — INTERPRET (the only model phase)

`interpret.py`. OpenRouter (qwen, reasoning off, throughput-routed) with an
Ollama fallback; urllib transport; bounded at ~90 s and two attempts.

- **Rendering is deterministic** (I3): community briefs (terms + both medoids,
  excerpted by the prompt) or the full chunk list, each chunk tagged `[id=n]`
  against a VALID IDS header. Excerpts are hit-centred, never document openings.
- **Reason** — one call: hypotheses → chosen → premises with cited ids →
  per-premise verdict (supports/contradicts/insufficient; analogy is
  insufficient) → answer built only from supported premises. | 6.6
- **Judge** — one verdict per shown chunk (entails/contradicts/neutral with a
  reason), then an answer from the entailed set only. Stricter by design: it
  asks whether a chunk literally answers the prompt. | 6.4
- **Checks, not trust** — invented ids discarded (I9), answers citing outside
  their supported/entailed set flagged (I10), every stage kept raw for
  inspection (I11). The walk's stop rule owns sufficiency, so the model is not
  asked what the evidence "does not settle".

## Phase E — RESULTS (walker UI)

`walker_app.py`, one page: prompt → answer(s) at top → a Reason-vs-Judge
disagreement note when both ran and differ → ONE collapsed **Evidence**
expander holding the community graph, query terms, ranked communities with
salient-titled medoid cards and walk scores, in-between chunks, and the
pathways block with the subgraph shape line. Sidebar is the run picker;
diagnostics only surface when a signal is genuinely degraded.

## Periphery

`export_neo4j.py` (X1–X5) exports nodes/CONTAINS/SIMILAR CSVs with provenance;
`label_communities.py` drafts model-authored community labels — drafts over a
frozen cluster, run-scoped, never a join key. `graph3d.py`/`render_graph.py`
render outputs.

## Where the letters live

R = `chunkgraph.py` build guards · S = `sampler.py` walk guards ·
W = `graph_tools.py` tool-surface guards · I = `interpret.py` model-boundary
guards · X = `export_neo4j.py` export guards. Each guard is pinned by a test
in `tests/`; §6.x references are `specs/graph-explorer/design.md`. This file
describes; the guards govern. If they disagree, fix this file.
