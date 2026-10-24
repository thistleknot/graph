---
description: 'Build and serve complete section-keyed BPE BM25 retrieval with optional community Dunning reweighting.'

## Current applied critique — 2026-10-23

Behavioral layer: deterministic source grouping/rendering and ordered evidence interpretation. The user explicitly requests applying the identified fixes. The source corpus/builds remain unchanged.

- When main and `_methods` records share a canonical arXiv URL, the Function SHALL count and group them as one paper while retaining every evidence record and its original section key.
- When source metadata lacks a URL, the Function SHALL use its document ID without the `_methods` suffix as the display identity.
- When a methods record is displayed, the Function SHALL label its source variant explicitly rather than treating it as independent corroboration.
- When evidence is rendered, the Function SHALL use one expandable Markdown section per canonical paper with exact passage text inside adaptively sized fences.
- When a details section is serialized, the Function SHALL put the opening details tag and summary tag on separate lines, as required by the installed client tokenizer.
- When retrieval diagrams are rendered, the overview and subgraph SHALL be visible native Mermaid blocks; trace and gate details MAY remain expandable.
- When no Dunning term is selected, the Function SHALL omit the unused seed-reweighting debug table.
- When relevant evidence provides explicit comparison criteria, pipeline synthesis SHALL organize interpretation around those criteria and contrast alternatives on shared axes; outlet displays that answer directly.
- Comparison-axis labels SHALL remain criteria rather than substitute method names; unsupported table cells SHALL say not reported rather than infer a value.
- When relevant evidence spans distinct change targets or boundary-crossing methods, the model SHALL preserve those distinctions rather than force one definition onto every method.
- A comparison table from one source SHALL NOT define the whole answer's scope; the interpretation SHALL explicitly name supplied boundary methods outside that table and cite their source before making an all-methods claim.
- The final synthesis prompt SHALL enumerate canonical source records and require an additional-approaches row for each relevant paper absent from the principal comparison.
- The Function SHALL display the API pipeline's final answer as Model commentary with citation-link rewriting only; Open WebUI's mandatory generation step SHALL request a short acknowledgement rather than another interpretation.
- When a source proposes future work, the model SHALL attribute it as a proposal rather than report it as an established finding or universal requirement.
- When the pipeline generates its final comparative synthesis, it SHALL allow 2048 completion tokens; action and community-summary calls retain their existing 1024-token default and shared deadline.
- The Function SHALL preserve source text, citations, truncation notices, TOON entities/communities, opt-in activation, and synchronized content/output.

Acceptance: three grouping cases (shared URL, URL-less aliases, distinct papers); three literal source variants; browser expand/collapse checks with newline-compliant Markdown, visible Mermaid SVGs and retained TOON; three varied synthetic comparison prompts before the live query. The live harness query SHALL count eight canonical papers for its 19 evidence records, retain the `_methods` passage with a variant label, and compare the source-defined axes including the distillation boundary. Each experiment is bounded below 15 minutes; no full corpus rebuild is involved.

Spec validation: extends incumbent `_details`, `_render_visible_retrieval`, inlet status/context, and existing graph answer/agent prompts. Storage IDs remain source-record IDs under C11; canonical display identity uses the API's existing arXiv URL. The installed client parser `ChODYpE7.js` matches `^<details...>\\n` and `^<summary>...</summary>\\n`; this supersedes the prior assumption that arbitrary details markup cannot render.

Acceptance closed: 110 focused tests passed (three skipped, two deselected), three clear-source model fixtures passed, and the frozen 19-passage replay retained the distillation boundary. Deployed Function 0.9.0 is AST-equal to the repository. The original hosted query completed in 142.8 seconds with 19 passages/eight canonical paper groups, source-defined comparison and the cited HARNESS-ZERO boundary, synchronized output/content, methods provenance, and unused seed diagnostics hidden. Browser verification opened all eight groups and exposed all 19 passages, confirmed literal source text/TOON and two visible Mermaid SVGs, then verified collapse. Evidence: `docs/owui_refs/deployment_current.md`; original chat assistant `7f9c5a8d-2441-4a77-b024-a9dbb123426c`.

Superseded context-display contracts and browser conclusions: [2026-10-23 history](superseded_context_2026-10-23.md); newline-compliant disclosures replace the earlier inline-only display, with canonical paper counts.

Current timeout repair 2026-10-22: the user's harness-design query retrieved its fused candidates in 4.75 seconds, but an internal Qwen action request spent 1024 completion tokens and returned empty content with finish_reason=length after 31.62 seconds. The matched non-reasoning request returned a valid action in 2.52 seconds. The former completed-chat check did not cover this failure.

- When the graph pipeline requests a textual model response, the system SHALL disable optional reasoning for that internal call.
- If a model returns empty content or an unfinished completion, the system SHALL fail explicitly before consuming that response as an agent action or accepted summary.
- When `/agent` starts, the system SHALL establish one monotonic deadline shared by history resolution, partial summaries, reduction, agent actions, and final synthesis.
- While a request deadline is active, the model client SHALL limit each network wait to the lesser of 60 seconds and the remaining budget.
- If no request budget remains, the system SHALL start no additional model call or retry.
- When `/agent` exceeds its budget or encounters an upstream model failure, the system SHALL return a descriptive HTTP error rather than a traceback or an ordinary answer without retrieved evidence.
- The request budget SHALL default to 180 seconds and remain configurable from 5 to 480 seconds in `AgentRequest.timeout_seconds`.
- The service SHALL log retrieval and model-call durations without credentials or prompt contents.
- Given empty, length-limited, and valid stopped model responses, When the client receives them, Then only the valid stopped response SHALL reach the agent parser.
- Given expired, nearly exhausted, and sufficient budgets, When a call or retry starts, Then its wait SHALL respect the remaining budget and an expired budget SHALL perform no network request.
- Given the user's harness-design query, When the real hosted Function executes it, Then the saved assistant message SHALL contain final text and passage cards within the Function's 600-second limit.

Spec validation: Behavioral sequencing owns the single request-scoped deadline; the model client's optional `deadline` parameter preserves existing callers; explicit failure preserves the existing no-fallback contract. The criteria cover three response classes and three deadline states before the hosted regression.

Acceptance closure 2026-10-22: 111 focused checks passed (3 skipped, 5 deselected); 13 deadline/propagation checks passed after strengthening assertions. All three live task prompts returned stopped nonempty responses. The exact hosted harness-design query completed in 122.8 seconds with final text, 19 passage cards, nine papers, and persisted done=true. Evidence: `docs/owui_refs/deployment_current.md` and `.tmp/arxiv_rag_api_timeout.stderr.log`.

Storage disposition 2026-10-22: exact vector scans and record-ID-only vector bridges are live; graph build 1163 and matching BM25 build 3 passed corpus coverage checks. Build identity and searchable fields remain in metadata.

Entity document-frequency updates shall aggregate the newly written mention rows once per persistence transaction, even when planner statistics do not yet describe uncommitted rows.

***definitions***

- :SectionCorpus: is the live `sect_node` population for one graph build.
- :BM25Index: is a versioned pgvector sparsevec build in the BM25 database, keyed by chunk record ID and bridged to paper and section.
- :BM25Sample: is the persisted reproducible sample used to train BPE and estimate document frequency.
- :SelectedTerm: is a community Dunning term supplied by the Open WebUI multiselect.

***implementation reqs***

- `CHUNKGRAPH_DSN` addresses the graph database; `ARXIV_BM25_DSN` addresses the BM25 database and defaults to the `langchain_postgres` service on port 5432.
- The BM25 watcher polls every 60 seconds by default and rebuilds only when its recorded source graph build differs from the live `xpa` build.
- `ARXIV_BM25_SAMPLE_SIZE` defaults to 80000 section-keyed chunks; populations below the limit use all rows.
- BPE merge count defaults to 10000; `k1=1.5`, `b=0.75`, and sparse candidate pool=144 remain recorded build settings; retrieval uses exact vector operations without ANN indexes.
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
- When the incremental graph watcher starts a cycle, the system shall add the legacy build readiness column before reading the live build.
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
- When an active BPE build is written, the system shall store sparse vectors in a vector-only bridge table keyed by globally unique record ID.
- When an active section build is written, the system shall store dense vectors in a vector-only bridge table keyed by globally unique record ID.
- When a vector record is queried, the system shall resolve its build membership and business key through the owning metadata table.
- The system shall keep retrieval vectors separate from text and metadata and keep build identity in the owning metadata/build tables.
- When a chunk has no BPE tokens, the system shall store a zero sparsevec for its record ID and count it as a complete vector row.
- When an exact query runs, the system shall rank dense vectors with cosine distance and BM25 vectors with inner product against the pinned build.
- When the active arXiv retrieval lane reads vector records, the system shall rank them with exact pgvector operations.
- When the active arXiv retrieval lane creates vector tables, the system shall omit HNSW and IVFFlat indexes.
- When /search receives a request, the system shall read graph sections and BM25 chunks from their owning databases.
- When /search receives sparse mode, the system shall rank through exact BM25 chunk vectors and map hits to sections.
- When /search receives dense mode, the system shall rank through exact section vectors.
- When /search receives hybrid mode, the system shall fuse exact sparse and dense section ranks by reciprocal rank.
- When /search returns a section, the system shall preserve its documented whole-section response fields.
- When candidate scores tie, the system shall order records by ascending record ID.
- When a BM25 build starts, the system shall keep it non-live and leave the previous ready build live.
- When a BM25 build passes vector, metadata, sample, vocabulary, and source coverage checks, the system shall publish it atomically.
- If a BM25 build fails or is interrupted, then the system shall preserve the previous complete live generation.






