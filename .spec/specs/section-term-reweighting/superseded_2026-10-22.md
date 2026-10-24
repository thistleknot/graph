# Superseded section-term-reweighting proposal — 2026-10-22

Preserved because the user replaced HNSW retrieval with exact vector operations and narrow vector tables; this history is not the active proposal.

## Requirements

---
description: 'Build and serve complete section-keyed BPE BM25 retrieval with optional community Dunning reweighting.'

***definitions***

- :SectionCorpus: is the live `sect_node` population for one graph build.
- :BM25Index: is a versioned pgvector sparsevec build in the BM25 database, keyed by paper and section.
- :BM25Sample: is the persisted reproducible sample used to train BPE and estimate document frequency.
- :SelectedTerm: is a community Dunning term supplied by the Open WebUI multiselect.

***implementation reqs***

- `CHUNKGRAPH_DSN` addresses the graph database; `ARXIV_BM25_DSN` addresses the BM25 database and defaults to the `langchain_postgres` service on port 5432.
- The BM25 watcher polls every 60 seconds by default and rebuilds only when its recorded source graph build differs from the live `xpa` build.
- `ARXIV_BM25_SAMPLE_SIZE` defaults to 80000 section-keyed chunks; populations below the limit use all rows.
- BPE merge count defaults to 10000; `k1=1.5`, `b=0.75`, sparse candidate pool=144, and HNSW `m=16`, `ef_construction=64` remain recorded build settings.
- The Dunning dropdown is optional and defaults empty; community boost defaults to 0.25.
- Dependencies: PostgreSQL with pgvector sparsevec, `tokenizers`, `wordfreq`, numpy, scipy, and the graph database's active section build.

***test reqs***

- Synthetic fixtures cover fewer-than-sample, equal-to-sample, and larger-than-sample populations; common, singleton, unseen, and overlong non-dictionary terms; repeated/adjacent BPE and word matches; and graph/BM25 key joins.
- Live acceptance checks verify sample persistence, complete source-key coverage, API startup against both databases, and opt-in term reweighting before top-k.

***functional specs***

- When an index build starts, the system shall read every eligible whole section from the active :SectionCorpus:.
- When an index build starts, the system shall preserve each section's `(doc_id, section_idx)` key through chunking.
- When the eligible chunk population exceeds `ARXIV_BM25_SAMPLE_SIZE`, the system shall persist one deterministic :BM25Sample: materialized view.
- When the eligible chunk population exceeds `ARXIV_BM25_SAMPLE_SIZE`, the system shall derive BPE merges and document frequencies from the persisted sample.
- If the eligible population does not exceed the configured sample size, then the system shall use every eligible row as :BM25Sample:.
- When sampled document frequency is zero for a corpus term, the system shall assign that term BM25 `df=1`.
- When a term has sample `df=1` or is an overlong non-dictionary word, the system shall exclude it from BPE merge training.
- When a term is excluded from BPE merge training, the system shall retain its encoded occurrences in BM25 scoring.
- When the API starts, the system shall validate the active graph build through the graph database connection.
- When the API starts, the system shall validate the matching :BM25Index: build, tokenizer, vocabulary, and IDF through the BM25 database connection.
- When the live `xpa` graph build changes, the system shall rebuild the BM25 index from that graph.
- When the BM25 watcher finishes rebuilding, the system shall verify the published source graph build ID.
- When retrieval runs, the system shall retrieve at most 144 chunks by BM25 sparsevec.
- When BM25 chunks are retrieved, the system shall rerank them by within-word token alignment and adjacent token/term counts before collapsing them to graph section keys.
- When selected terms are empty, the system shall leave the original query's dense and sparse inputs unchanged.
- When selected terms are empty, the system shall apply no community boost.
- When selected terms are supplied, the system shall add them only to the sparse query.
- When selected terms are supplied, the system shall apply normalized community Dunning affinity after RRF fusion and before top-k selection.
- When incremental ingestion embeds a new chunk, the system shall complete embedding and assignment calculations before mutating persisted paper rows.
- When incremental ingestion writes a paper, the system shall persist replacement deletion, chunk metadata, sparse vectors, dense vectors, assignments, and entity mentions in one transaction.
- When a previously persisted paper is missing a required dense vector or nearest-centroid assignment, the system shall classify that paper as incomplete and retry it instead of skipping it as stored.
- If embedding or a persistence step fails, then the system shall leave the prior paper rows unchanged and leave no partial rows for the attempted paper.
- When the incremental graph service runs from the command line, the system shall accept an explicit Markdown root for one-shot and watch modes.
- If a BM25 chunk cannot join an active graph section key, then the system shall exclude it and report the unmatched count.
- If a BM25 chunk cannot join an active graph section key, then the system shall not substitute legacy corpus rows.
- The system shall store BM25 metadata for every eligible graph section key.

## Design

# Section Term Reweighting — Design

**Spec layer:** Structural and Behavioral. This document fixes data shape and stage ordering for `requirements.md`.

## Data flow

```text
active graph sect_node rows
  -> 60-second watcher compares live graph build ID with BM25 source build ID
  -> rebuild only when the graph build changes
  -> section-preserving chunks keyed by (doc_id, section_idx, chunk_idx)
  -> langchain_postgres BM25 source table
  -> deterministic 80k sampled materialized view (or all rows if population <= 80k)
  -> BPE merges and document frequency estimated from the same sample
  -> frozen BPE tokenizer, IDF, sparsevec index, and chunk metadata in langchain_postgres
  -> GET /community-terms -> dynamic Open WebUI multiselect
  -> selected terms in POST /agent
  -> BM25 BPE sparsevec chunk retrieval(question + selected terms, top 144)
  -> within-word token/adjacency and directed term-pair reranking over chunk text
  -> collapse ranked chunks to graph section ords
  -> dense rank(original question) + reranked sparse section rank
  -> RRF over the union of dense and BPE sparse arms
  -> community affinity from selected-term G² values
  -> adjusted fused scores
  -> top-k seeds
  -> graph DB section-key join, existing graph expansion, agent, and evidence assembly
```

## Scoring contract

For each selected term `t`, normalize the stored score by the largest stored G² for `t` over all communities:

```text
term_affinity(c, t) = g2(c, t) / max_c(g2(c, t))
community_affinity(c) = mean_t(term_affinity(c, t))
adjusted_score(section) = rrf_score(section) * (1 + 0.25 * community_affinity(section.community))
```

The boost is bounded to 25%. Missing term/community pairs contribute zero. Dense embedding remains based on the user's question; only the sparse query receives the added terms. BPE training eligibility is a tokenizer-only concern: dictionary membership and the robust long-word cutoff shape the training corpus, while BM25 term counts include singleton terms and unseen query terms use df=1. The persisted 80k materialized sample is the provenance source for both BPE and sampled DF; sample DFs are scaled by population/sample size and clamped to [1, N]. The frozen tokenizer, sample identity, scale, IDF parameters, and source graph build are recorded with the BM25 build. The top 144 chunk hits are reranked before mapping to graph sections and RRF. Within each source word, BPE token alignment permits an equal or one-position offset; internal adjacent BPE pairs and ordered adjacent word pairs are counted with multiplicity, capped by query multiplicity. The available component ratios are averaged; BM25 score then breaks adjacency ties. The ordered section results join dense results in RRF, and Dunning affinity is applied to the fused list before top-k.

## Storage and compatibility

The graph database owns live sections, communities, Dunning terms, dense vectors, and graph edges. The BM25 database owns the section-keyed chunk source, 80k sample materialized view, vocabulary/IDF, sparse vectors, metadata, and frozen build settings. Both connections are required by the API. The ingester reads all sections from the graph build and rebuilds the BM25 lane from that source; old `arxiv_chunks` and `.tmp` cache rows are not accepted as current source because their key coverage is incomplete.

`sect_community.terms` remains JSONB and stores objects with `term` and `g2`. The existing renderer artifact is the source for the term/G² pairs, so updating the live term payload does not require rebuilding vectors, edges, or communities. Readers accept the new object form and the old string form during migration; old string rows have no usable G² and receive zero affinity until refreshed from the artifact.

## UI contract

The filter's `UserValves` exposes a dynamic multiselect whose options come from `/community-terms`. Values are distinct term strings. Option labels include the term and its top community labels. The selected list is sent unchanged to `/agent`; it is not interpolated into model instructions or treated as a hard community filter.

## Failure behavior

An empty or unavailable term inventory yields no selectable options. Unknown selected terms are ignored by scoring and disclosed in the report. A missing BPE chunk build, vocabulary/IDF, frozen tokenizer metadata, source build ID, or either database connection fails API startup; no `tsvector` fallback is allowed. Build coverage records the number of distinct graph section keys represented and the unmatched key count.

Incremental paper ingestion reads and embeds first, with no writes. It then deletes any replaced version and writes metadata, both sparse views, dense rows, assignments, and entity mentions inside one PostgreSQL transaction. Before selecting pending markdown files, the watcher detects stored papers whose retrievable rows lack a dense vector or assignment and reloads their current source file as a replacement. An embedding or write failure therefore preserves the prior complete paper or leaves the newly attempted paper absent, allowing the next cycle to retry.

## Approval state before the revised proposal

```json
{
  "feature": "section-term-reweighting",
  "kind": "feature",
  "phase": "implement",
  "approvals": {
    "requirements": "2026-10-21",
    "design": "2026-10-21",
    "tasks": "2026-10-21"
  },
  "scope": [
    "README.md",
    "F:\\Python\\README.md",
    "interpretations.md",
    "src/config.py",
    "src/sparsevec_store.py",
    "src/arxiv_bm25_worker.py",
    "src/arxiv_graph_service.py",
    "tests/test_arxiv_graph_service.py",
    "src/arxiv_community_map.py",
    "src/hnsw_communities.py",
    "scripts/install_arxiv_bm25_service.ps1",
    "src/ingest_arxiv_sparsevec.py",
    "src/section_graphrag.py",
    "src/section_sparse.py",
    "src/section_store.py",
    "src/arxiv_rag_api.py",
    "src/owui_filter.py",
    "tests/test_section_sparse.py",
    "tests/test_ingest_arxiv_sparsevec.py",
    "tests/test_arxiv_bm25_worker.py",
    "tests/test_section_graphrag.py",
    "tests/test_arxiv_rag_api.py",
    "tests/test_owui_filter.py"
  ],
  "created": "2026-10-21T00:00:00Z"
}
```
