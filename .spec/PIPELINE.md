<!-- Spec: top-level e2e pipeline description. Sources: steering/{product,structure,tech}.md,
     specs/graph-explorer/design.md 6.1-6.11, specs/graph-term-selection/requirements.md.
     Guard ids cited here live in module docstrings (R=chunkgraph, S=sampler, W=graph_tools,
     I=interpret, X=export_neo4j, E=entities) and each is pinned by a test. Operator request 2026-08-31. -->

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

`ingest_mixed.py` → `chunkgraph.py` → `pg_store.py`. Offline, deterministic,
reproducible from stored params (`_PARAM_ATTRS`).

| step | what happens | guard |
|---|---|---|
| chunk | the DOCUMENT is the unit. Lines per document → Box-Cox, fitted **per source**; a doc splits only above its own source's `hi = median + 2·MAD`, at paragraph bounds, short tail merged back. Never inside a word. `doc_id` is source-prefixed | R17, R19, R8, R20 |
| phrases | NPMI welds collocations into single tokens (`white_citizens`, `self_help`) before any scoring; Dunning-LLR reported as diagnostic only | R9 |
| stoplist | one stoplist for build and serve, contraction-aware, read straight from the NLTK english file | R18 |
| sparse arm | BM25-weighted term matrix, L2 rows, blockwise cosine chunk×chunk. BM25 over PPMI because BM25 corrects for length; PPMI would make long chunks hubs | R10 |
| dense arm | pluggable embedder; default model2vec static vectors (MiniLM-distilled, 256d). Absent model ⇒ sparse-only still works — both shipped mixed runs (mixed-smoke, mixed-full) exercised exactly this path: sparse-only, dense disabled, R5 degradation observed live | R5, R14 |
| normalize | Box-Cox per source-pair block (3 intra + 3 cross), so the two spaces AND the source pairs are commensurable | R21 |
| edges | k-sigma tail cut per space; budget-match fallback when kurtosis is high; one global cut over the reassembled z (never per block) | R2, R21 |
| backbone | per-node top-KNN unioned in from both spaces, so significance ranks but never isolates | R7 |
| fuse | union with **provenance** `{sparse, dense, both}` on every edge; strength persisted bounded (`1 − D`) so path products cannot amplify | R3, R15 |
| communities | Louvain (fixed seed) on the fused graph; per-community tf·idf keywords and a global medoid chunk | — |
| persist | one `graph_run` + nodes/edges/communities/embeddings, atomic, every node payload carries `source` **and `title` where the document has one** (absent, never null); re-ingest **supersedes, never deletes** (bitemporal `valid_from`/`valid_to`) | pg_store contract, R20, R22 |

What a run guarantees downstream: every edge names its space(s), every chunk
names its document, every chunk names its source corpus, and the whole thing
rebuilds byte-identically from params.

**Measured — run `mixed-full`, 2026-09-01.** 10,369 docs → 10,826 chunks
(wiki 7,818 / quotes 2,508 / brown 500), 373,215 edges, 27 communities, 669 s
wall / 549 s fit, peak RSS 7.1 GB; `--wiki-stride 4` chosen against a
12,000-chunk memory-derived target; intra-block edge-rate ratio 1.156 inside
the 10x bound, all six blocks `fit=own`; sparse-only. Full per-source params
and rationale: `specs/graph-explorer/design.md` §6.14 "Measured".

## Phase B — QUERY (the walk)

`sampler.py` over the read-only tool surface `graph_tools.py`. Per prompt,
deterministic, ~1 s.

1. **Anchors** — BM25 search of the prompt against a per-run cached inverted
   index (Python, ~4 ms); top-k chunks seed the walk. No lexical hit ⇒ the walk
   says so rather than walking nothing. Anchors are the plain global BM25 top-k,
   plus at most one EXTRA anchor per source absent from that top-k when its best
   hit clears 0.5x the k-th score — additive, never reallocated (S15/S16).
2. **Expansion** — HNSW-style ef-search: a frontier expands along stored edges,
   path score = product of edge strengths, and it **stops itself** when the best
   frontier candidate cannot beat the worst held result at `ef` width (tuned
   ef=64). Depth is emergent, not an operator dial. Boltzmann sampling steers
   the frontier, never the output. | S9–S12
3. **Ring** — one degree out from the top-3 chunks: their strongest unseen
   edges enter at `parent score × strength`, ≤8 per parent, never outranking
   the parent, never seeding further expansion. Where the anchor list spans
   >1 source the ring budget is split by the anchor source mix, largest-remainder,
   shortfall forfeited to global strength order (S17). | S13
4. **Bridges (discovery)** — for each pair among the top-3 chunks, the single
   strongest degree-damped path over the WHOLE run (Dijkstra on −log strength
   with a deg^-0.4 interior penalty). Chunks that path crosses which the walk
   never retrieved enter the evidence at ≤ min(endpoint scores). This is the
   2-hop generalization of the ring: scoring stays on the subgraph, discovery
   is allowed to search everything. | S14, W16
5. **Bundle** — the frozen evidence: ordinals, per-chunk walk scores, anchors,
   **origin** (walk | ring | bridge), telemetry. Everything downstream derives
   from this one object. | S6/S7

Display key, everywhere evidence is listed: **bold #id** = a bridge chunk
(discovered, not walked); its *salient terms are italicised*. Unmarked ids came
from the walk or its one-degree ring.

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
| second-order terms | domain token ↔ semantic nomen affinity: Dunning-LLR gate → Schütze context-centroid cosine → Mann-Whitney AUC re-rank only when the skew diagnostic trips | W17 |

## Phase D — INTERPRET (the only model phase)

The structure computed in Phase C is not UI-only: Reason's briefs carry the
same pathways object as an attached image (walked subgraph + global community
map) or as a numbers block (`structure="text"`), and Reason+Judge run as ONE
serialized call (`judge=True`) whose single JSON reply carries both channels
(I12/I13, design 6.12).

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

`export_neo4j.py` (X1–X10) emits the 4-file neo4j-admin layout
(chunks/terms/contains/similar.csv) with provenance, a 256-dim embedding column
plus `CREATE VECTOR INDEX`, and `C<cid>` community labels; it also writes walks
back into a live neo4j as `(:Walk)-[:ANCHORS]->(:Chunk)` and
`(:Chunk)-[:PATHWAY]->(:Chunk)`. `entities.py` (E1–E5) builds the run-scoped
bipartite entity store from the existing salient/phrase vocabulary.
`label_communities.py` drafts model-authored community labels — drafts over a
frozen cluster, run-scoped, never a join key. `graph3d.py`/`render_graph.py`
render outputs.

## Where the letters live

R = `chunkgraph.py` build guards · S = `sampler.py` walk guards ·
W = `graph_tools.py` tool-surface guards · I = `interpret.py` model-boundary
guards · X = `export_neo4j.py` export guards · E = `entities.py` entity-store
guards. Each guard is pinned by a test
in `tests/`; §6.x references are `specs/graph-explorer/design.md`. This file
describes; the guards govern. If they disagree, fix this file. R19–R21 govern
multi-source ingest and are pinned in `tests/test_chunk.py` /
`tests/test_pg_store.py`.


---

## FAQ — operator questions, answered once

**Does the model see the path/subgraph information?** Yes, since I12: Reason's
input carries the pathways and shape as an image by default (the same two-panel
map the UI shows) or as a numbers block; failure of the image path falls back
to numbers, noted in `structure_note`. The Judge channel deliberately sees
none of it — its contract is chunk-by-chunk entailment on content alone, and
structural hints would contaminate that.

**Should paths be derived over the subgraph or the whole graph?** Both, split
by job. *Scoring* the connectedness of retrieved ideas runs on the walked
subgraph (W15) — a path through a chunk you never retrieved is not evidence
you can show. *Discovery* runs on the whole graph (S14/W16): the strongest
degree-damped path between two retrieved ideas may cross a chunk the walk
missed, and that chunk is pulled into the evidence, bolded, exactly like the
ring at one hop.

**How does this scale? Can samples generalize?** Yes, with the standard split.
Precompute what is query-independent: global degrees, community medoids, and a
DWPC matrix over medoids can all be built at ingest — Hetionet ships DWPC at
millions of nodes exactly this way, offline [empirical:cited — Himmelstein's
Rephetio computes DWPC features in batch]. Sample what is query-dependent:
DWPC is a sum over paths, so Monte-Carlo random walks converge to the same
*ranking* of pairs long before the values converge, and ranking is all we use;
Personalized PageRank is the mature sublinear version of the same "many
hub-free routes" notion [empirical:cited — PPR push/local methods,
Andersen–Chung–Lang]. Conductance and density estimate fine from sampled
neighbourhoods. Caveat: samples generalize for rankings and shapes, not
exhibits — the displayed chain always comes from an exact bounded search.
Brown at 500 nodes needs none of this; the split is written down for the
corpus that does.

**Does DWPC find paths?** No — it scores them. Enumeration is plain traversal
(DFS on the subgraph in W15, Dijkstra on the whole graph in W16); DWPC turns
the enumerated set into one connectivity number per pair. Finding and scoring
are the two halves of `pathways()`.

**Doesn't DWPC need typed metapaths?** Moot here: the serve-time graph is
homogeneous (chunk↔chunk), so only the degree-damping transfers, and that is
the part in the code. Metapath machinery becomes relevant only if the
term↔chunk bipartite layer is built (term-chunk-term paths).

**Where is WCC?** It was always there under another name: the graph is
undirected, so weakly connected components are just components. `pathways()`
reports the count and per-component sizes (`wcc_sizes`), and the UI labels
them WCC.

**NPMI vs BM25 — who does what?** NPMI where the datum is co-occurrence of a
pair (phrase welding, and term–term edges when that layer lands); BM25 where
the datum is a term's rate inside something with a length (term–chunk);
cosine where the datum is two profiles (chunk–chunk). Community salience by
BM25 mass is `community_terms`; the term → chunk_ids inverted index exists and
also feeds the Neo4j CONTAINS export. Length-normalized chunk–term PMI is
log-ratio keyness and has a queued A/B against BM25.

**Louvain or Leiden?** Open, queued: same API, usually better partitions, and
the real motivation is the too-coarse 7-community document-level partition —
to be tried at the next re-ingest.
