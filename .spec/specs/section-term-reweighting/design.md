# Exact Vector Retrieval and Community Terms — Proposed Design

## Current applied critique — 2026-10-23

Behavioral layer: extend the incumbent pipeline prompt and Filter; no storage, ranking, or Function-type changes.

- `answer_prompt` inventories canonical document IDs, preserving every section key and source passage; methods aliases count once in that inventory.
- Final synthesis compares on source-defined criteria, names relevant additional approaches outside the principal table, and separates results from proposals.
- The pipeline's selected model generates the answer with a 2048-token final allowance; actions/summaries retain 1024 and the shared deadline.
- Inlet preserves model/term forwarding and request-local metadata, then requests only a short acknowledgement.
- Outlet displays `report.final` as Model commentary with citation links, eliminating a second interpretation pass.
- Exact passages live in one expandable Markdown group per canonical API URL; absent URLs use document ID without `_methods`.
- Newline-compliant details syntax uses the installed tokenizer; adaptive source fences preserve literal Markdown.
- TOON entities/communities remain visible, overview/subgraph Mermaid diagrams render directly, and unused seed tables are hidden.

Verification chain: three browser disclosure fixtures → canonical grouping/source/output checks → three synthetic comparison inputs → frozen 19-passage coverage replay → hosted query and actual browser. Preserve existing answer branches, activation, toggle, and deployment valves.

Validated: 110 focused tests passed (three skipped, two deselected); three clear-source fixtures and frozen replay passed. Function 0.9.0 is deployed with settings preserved; the hosted query completed in 142.8s with 19 passages/eight canonical groups and a cited distillation boundary. The actual browser opened all eight groups, exposed 19 passage headings, verified literal source/TOON/provenance and two Mermaid SVGs, and closed groups again. Current deployment record and screenshot close the rendered acceptance gate.

Rejected prompt-only and second-rewrite approaches, earlier measurements, and superseded display contracts: [2026-10-23 history](superseded_context_2026-10-23.md).

## Current timeout repair — 2026-10-22

Behavioral layer: keep the complete retrieval, cross-encoder gate, graph traversal, and evidence response; repair the internal OpenRouter calls before changing any retrieval parameters.

`summarize_clusters.chat(model, prompt, *, deadline=None)` owns textual-completion validation and request-budget-aware retries. `_request` accepts a network timeout and deadline; response reads check the deadline even when OpenRouter sends keepalive bytes. Optional reasoning is disabled for these structured task calls; the user's final Open WebUI generation retains its model configuration.

`AgentRequest.timeout_seconds` configures a request budget (default 180 seconds, accepted 5–480). `agent_ep` establishes `monotonic() + timeout_seconds` before resolving history. It passes that same deadline through `agent_answer` and `global_view`; every internal model call consumes the same budget. Exhaustion becomes HTTP 504; upstream HTTP and malformed/unfinished model output become HTTP 502. Neither case silently generates an answer. Log model identity, completion status, token counts, and duration without prompt text or secrets.

Validation before deployment: response battery (empty/unfinished/stopped), deadline battery (expired/short/sufficient), retry-budget exhaustion, API HTTP error mapping, existing retrieval/Filter tests, then the exact failing query through the hosted Function with persisted nonempty assistant text and evidence cards. Archive the former completion-only disposition rather than treating it as universal reliability evidence.

Validated implementation: 111 focused tests passed; strengthened deadline propagation passed 13 tests; the three live task calls completed; the exact hosted query completed in 122.8s with 19 passage cards and nonempty final text. The existing Function is unchanged; API PID 461272 serves the repaired code. No new helper duplicates the incumbent model client.

**Storage disposition — 2026-10-22; exact retrieval and record_id-only vector bridge rows are implemented and verified against the active graph/BM25 generations.**

**Entity DF write disposition — 2026-10-22:** entity mention persistence materializes per-entity document frequencies once and disables nested-loop joins locally for that update. The live 81,473-section refresh showed PostgreSQL estimating uncommitted COPY rows as one row; a materialized CTE alone still produced a nested loop over the entity table. The local planner setting forces a one-pass hash/merge join without changing session defaults.
**Spec layer:** Structural, with Behavioral contracts for migration, querying, and publication.
Earlier HNSW settings are superseded by the user's explicit instruction to use direct vector operations.
Previous requirements, design, and approval state: [planning history](superseded_2026-10-22.md).

## Evidence and open assumptions

- Observed: dense retrieval ranks sections; sparse retrieval ranks chunks and joins their paper/section keys to the graph.
- Observed: the attempted build published its live flag before loading finished; the HNSW operation was canceled.
- Observed: the active section dense and sparse query helpers use exact operators; legacy shared helpers still expose HNSW options outside this lane.
- Observed: the shared BM25 database has a cmetadata GIN index on a different table; the active retrieval lane does not query it.
- CLOSED: the harness-design query's fused retrieval completed in 1.92 seconds on the active corpus; its later timeout was reproduced in internal model output.
- CLOSED: xpa build 1163 and matching BM25 build 3 include the refreshed Markdown corpus; the earlier build-32 absence is superseded by live coverage and new-paper retrieval checks.

## Storage contract

| Owner / incumbent | Proposed responsibility | Identity |
|---|---|---|
| Graph DB / sect_node | Section text, paper/section key, community and layout; vector columns retire after validated cutover | Existing (build_id, ord) |
| Graph DB / sect_embedding | Narrow section-vector bridge containing only record_id, emb and optional genre-projected emb_b | Globally unique record_id; section metadata owns build_id and ord |
| BM25 DB / lex_sparse_<label> | Narrow sparse-vector bridge containing only record_id and vec | Globally unique record_id; chunk metadata owns build_id, doc_id, section/chunk key |
| Graph/BM25 metadata tables | Own text, build membership, business keys, flags, and diagnostics | record_id maps each vector to exactly one metadata row |
| BM25 DB / lex_build, lex_vocab, sampled view | Generation-specific tokenizer, DF/IDF, source graph build, sample provenance and completion counts | Build identity |

Current ordinals are build-local and cannot be reused as the vector-table key while generations coexist. Add a globally unique record_id to each owning metadata row; preserve `(build_id, ord)` as the logical key, and make record_id the one-to-one vector bridge. Chunk and section IDs remain separate namespaces.
Existing paper/section keys remain the cross-database bridge; joins are validated because PostgreSQL cannot enforce a cross-database foreign key.
Vector tables carry only record_id and vector data, with a primary-key B-tree index for the join. Metadata tables own build and business fields.
No HNSW or IVFFlat index is introduced. Keep an existing metadata index only if an active query predicate uses it; do not add field-based metadata indexes speculatively.
Keep metadata indexes on their owning tables; inspect actual query predicates before retaining or relocating one.
Without ANN, vec can use untyped sparsevec to hold build-specific dimensions in one incumbent table; validate dimensions against lex_build.
Preserve emb_b and its matching query projection; table separation must not change either dense space.

## Ordered implementation work

1. **Repair the governing contract.** Reconcile requirements, module front matter and README defaults to exact retrieval; invalidate the old HNSW approval.
2. **Migrate identity and tables.** Add globally unique record_id to section/chunk metadata, preserve build-local ordinals and business keys, and backfill vector bridge tables containing record_id plus vectors only.
3. **Make builds atomic.** Open a non-live generation, load vectors and provenance, check coverage/dimensions, then publish one live pointer in a transaction; keep the previous complete generation readable on failure.
4. **Change query helpers.** Resolve the pinned generation through metadata joins, then rank dense vectors by emb/emb_b `<=>` and sparse vectors by vec `<#>`; break ties by record_id.
5. **Complete the retrieval handoff.** Join the top 144 sparse hits to chunk metadata, rerank adjacency, map to whole sections, fuse dense/sparse ranks, apply selected-term community affinity, then take seed top-k.
6. **Change ingestion and automation.** Remove HNSW creation and ef_search assumptions from this active lane; migrate legacy build readiness before graph-watcher reads; make the BM25 worker validate completion/schema/query mode and record_id coverage before skipping a build; fail API startup on incomplete or mismatched generations.
7. **Unify serving.** Route `/search` and `/agent` through the exact split-store `SectionGraph` retriever; preserve `/search` response fields and sparse/dense/hybrid modes. Remove its legacy HNSW/old-schema dependency.\n8. **Validate, then cut over.** Run the bounded gates below, migrate the active API to the F: interpreter, and verify the community-term endpoint and existing Open WebUI dropdown against the published generation.

Extend the incumbent store/query/ingest helpers; no second retrieval service or parallel store API.
Audit legacy callers of shared sparsevec_store helpers before changing signatures; unrelated database applications are outside this migration.
Delay removal of old vector columns/tables until the new reads pass validation; preserve a rollback route and build provenance.
The separate xpa refresh must close its source-membership gate before the newer papers can count as end-to-end ingested.

## Acceptance gates, in dependency order

| Gate | Required evidence | Bound |
|---|---|---|
| Identity and storage | Three fixtures cover repeated ordinals across builds, multiple chunks in one section, and an empty-vector chunk; no orphan/wrong-section joins | Under 5 minutes |
| Exact ranking and handoff | Three varied queries match direct dot-product/cosine controls, preserve deterministic ties, and retain adjacency/RRF/community-boost ordering | Under 5 minutes |
| Failure and publication | Failed embedding/load leaves the previous build live; an interrupted build cannot be served; vector omissions are reconciled with explicitly empty encodings | Under 5 minutes |
| Active corpus | Compare current exact reads and proposed narrow-table reads for three queries, record timings/plans and payload parity; confirm vector operations use no ANN index | 60-second statement timeout per query |
| Serving | API startup validates both databases; `/search` supports all three modes on the exact retriever with its existing response shape; `/community-terms` returns options; selected terms reach pre-top-k community adjustment | After predecessor gates pass |

No full rebuild is used to discover a failed identity, scoring or publication assumption.
A slow exact scan triggers query/storage diagnosis and a reported measurement; it does not silently restore ANN.
Retain the 80k sample, tokenizer-only df/long-word gates, BM25 defaults, evidence passthrough and optional selected-term UI.

## Spec validation and disposition

Structural roles, record granularity, build lifetime, failure boundaries and acceptance conditions are explicit.
The proposed requirements use one SHALL response per condition; existing feature requirements remain in force.
CLOSED: automatic HNSW creation in this retrieval lane, superseded by the user's 2026-10-22 instruction.
Sprawl review: nothing to collapse; the plan extends existing storage and query boundaries.



