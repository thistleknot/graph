<!-- Spec: .spec/steering/product.md · Task: playbook.md T1 (doc pointer, operator instruction 2026-08-31) -->

# chunkgraph

A dual-space retrieval graph over arbitrary text. Chunks become nodes; edges are
drawn from two independent similarity spaces (sparse BM25 and dense embedding),
normalized, thresholded, fused with provenance, and persisted to Postgres.
Graph construction is fully deterministic — no LLM touches chunking, edge
formation, or community assignment, and every edge names the documents it came
from.

The pipeline and its numbered EARS guards (`R1`…`R21`) live in the module
docstrings of [chunkgraph.py](src/chunkgraph.py) and
[salient_grams.py](src/salient_grams.py); the docstrings ARE the spec for the code
they sit above.

## Repository layout

```
src/            every Python module, flat (modules import each other by bare name; tests and scripts put src/ on the path)
tests/          pytest suite; pytest.ini sets pythonpath = src
docs/           notes and reference: section_map/ (example PNG + markdown), cookbook/ (query snippets), trigram/, psql_graph.md, plan.md, chunking.md
sql/            Postgres init migration, mounted by docker-compose.yml
.spec/          steering and specs (the module docstrings carry the guards)
playbook.md     task ledger
third_party/    ignored: vendored book companion code, re-download from upstream
.tmp/           ignored: every intermediate and output of a run
```

## arXiv section map — start here

Python and MCP runtime ownership, including the graph service and `arxiv_id_lists` paths, is tracked in `F:\Python\README.md`.

The current work is a map over whole **sections** of 2,564 arXiv papers (81,473 usable sections), kept in Postgres
(`chunkgraph-pg`, host port 5433, user/db/password `graph`). Each stage is its own module; run them in this order
from the repo root (`python -u src\<file>.py`). `.tmp/` holds every intermediate and output.

| Step | Entrypoint | What it does |
|---|---|---|
| 1 | `src/section_corpus.py --root D:\c\arxiv_id_lists\papers\post_processed` | cut the selected Markdown root into sections and keep usable ones; default root remains in `domain_corpora.py` |
| 2 | `src/section_embed.py`, `src/section_sparse.py` | dense (jina v5 nano through model2vec) and sparse (BPE + adjacent pair) vectors |
| 3 | `src/section_map.py` | exact correlation edges in both spaces, one fused graph, Leiden communities, exemplars, UMAP layout |
| 4 | `src/section_store.py --tag xpa` | write the build to Postgres: `sect_node`, `sect_edge`, `sect_community` |
| 4b | `src/section_store.py --tag xpa --entities` | match the frozen entity inventory onto the sections into `sect_mention` (about 6 minutes) |
| 5 | `src/summarize_clusters.py --all` | LLM draft summary per community (resumes from its JSON) |
| 6 | `src/arxiv_titles.py` | paper titles into `paper_title`, prefilled from the CSVs in `C:/Users/user/arxiv_id_lists`, the arXiv API for gaps |
| 7 | `src/section_render.py --tag xpa` | **the picture**: labelled community PNG and markdown (Dunning terms, entities, paper titles) |
| 8 | `src/section_query_panel.py` | ask a question: hybrid retrieval, community view, and the search-planning ReAct agent, drawn as a strip (`--hops 1-3`, `--queries 1-3`) |

Library modules, imported not run: `section_graphrag.py` (hybrid GraphRAG and the agent), `section_graph.py` (edges),
`arxiv_community_map.py` (exemplars and the card/markdown writers), `term_salience.py` (Dunning terms), `section_genre.py`.
`entity_derive.py` (repo root) derives entities with no tagger: n-grams filtered by NPMI, closed-class, nesting and residual-IDF rules.

Three layers stay apart. **Vector edges** join section to section (`sect_edge`; communities come from these).
**Mentions** join a section to an entity (`sect_mention`). **Co-mention edges** join entity to entity
(`sect_entity_edge`, `kind = 'co_mention'`, NPMI > 0 over at least 5 shared sections). Co-mention edges feed no
retrieval step yet. The agent's answer quality is not yet measured against plain retrieval.

### How a question is answered (`section_graphrag.py`)

1. **Global view.** The cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` scores (question, section) for the 50 best sections of the hybrid ranking (`CE_POOL`);
   a section at or above `CE_TAU` (0.0, the model's own relevant/irrelevant boundary) passes. All scores are collected before the generation model writes a partial for each community that holds a passing section,
   best section first; a partial opens with `RELEVANT: YES|NO`, a NO is kept in the JSON (`partials_rejected`) and never reaches the reducer. **There is no fallback:** when no section passes the cut the question is rejected and the strip says so (`question_rejected`). A question with a passing section is never rejected because the partials said NO. Each partial is given the community summary plus that community's 3 best passing sections (`PARTIAL_SECTIONS`): a summary is a topical abstract and cannot hold the detail a question asks for. With the summary alone, the memory-taxonomy question's six passing sections all sat in one community (c21, "Long-term memory for AI agents") whose summary answered NO and the question got an empty global view; with the sections the same community answers YES, and the safety question's global view grew from two communities to four.
   This replaced the earlier Jev-Style community-summary gate, which rejected 29 of 60 in-corpus questions at its 0.50 cut.
2. **The question's subgraph.** The 3 best seeds plus their vector neighbours out to `HOPS` hops (default 3, tunable 1 to 3; at most 15
   sections per hop, strongest edge first); the 18 nearest the question by dense similarity are shown (`PER_QUERY`).
3. **The agent** sees that subgraph, the global answer and a Box-Cox-weighted sample of the communities (with their top entities), and may
   make up to `AGENT_QUERIES` more searches (default 3, tunable 1 to 3), each its own subgraph kept apart from the earlier ones: `topup`
   (default: its natural subgraph minus what is already shown, plus as many more as overlapped, hopped from the whole natural subgraph)
   or `mask`. It can READ a section whole and may stop at any reply. It never extends the graph by hops itself. A question rejected in step 1 gets no agent, no evidence and no answer.
4. **The final answer** interprets every section the traversal saw, in the light of the global answer: sections the agent read, the typical
   section of each top community (always included), then all the rest nearest the question first; each cut to fit a 270,000-character budget, and every cut says so. For methods questions, the prompt asks the agent to track both what approaches do and how sources classify their variants, while preserving any other distinctions the question asks for. When sources classify by separate axes, such as function versus storage form, the synthesis keeps those axes distinct; overloaded terms are grounded in the relevant field.

The cross-encoder runs on CPU from `~/models/ce-ms-marco-minilm-l6` (download `cross-encoder/ms-marco-MiniLM-L-6-v2` there; no service is needed). Each query-panel JSON stores every section score (`section_scores`) and the pass score (`section_threshold`).
Cross-encoder gate (`docs/experiments/ce_gate_probe_result.txt`, the 60 in-corpus gold questions and 8 off-topic ones, no language model in the loop): at 0.0 it rejects 7 of 60 in-corpus questions and accepts 2 of 8 off-topic; at 2.0 it rejects 11 and accepts 1; at 5.116 (the 25th percentile of the gold sections' own scores) it rejects 23 and accepts 0. Score alone does not separate a broad survey question from an off-topic one: with the cut at 2.0 the memory-taxonomy and safety panel questions were rejected (best sections 2.38 and 1.66) while an off-topic football question scored 4.36. The cut is 0.0 because all three panel questions pass there, not because it was tuned.

Measured (`.tmp/hop_grid_stage_a.log`, `.tmp/mask_vs_topup.log`, 200 queries each): the graph walk does **not** beat plain hybrid retrieval on
finding the source section of a sentence (0.445 to 0.450 against 0.555; second query 0.41 against 0.505), hops 1, 2 and 3 score the same, and
`topup` ties `mask`. Stage B (60 questions, `docs/experiments/stage_b_result.txt`): the walk is worse than a same-size hybrid list at finding the source section at every hop setting and search count (0.48 against 0.67 with 3 searches at hops 2, paired 2 against 13), hops 1-3 are not distinguishable, and cross-encoder precision is a small loss. Answer quality is not measured. The replaced Jev gate (`docs/experiments/jev_gate_probe_result.txt`, run before it was removed; 60 in-corpus questions, all 415 summaries scored solo): at 0.50 no community passes for 29 of 60 (0.48), so the reject rule rejects them; the source section's own community passes in 13 of 60 and ranks within the top 10 in 24; raising or lowering the cut only trades rejections against that. With the gate in the stage B loop (`stage_b_gate_result.txt`) 35 of 60 are rejected, and on the 25 answered the walk finds the source section in 9 against 15 for a same-size hybrid list; over all 60 the whole system finds it in 0.15-0.17 against 0.52 for a plain hybrid list (known-item sentences, source-section recall, not answer quality). A model-judged blind comparison of four answer arms on the 25 answered questions (`docs/experiments/blind_cmp_result.txt`) ranks them 2.70 / 2.48 / 2.48 / 2.34 (hybrid only / + subgraph / + global / + agent; random is 2.50), no difference; judge and writer are the same small model and no human read the answers. That test cannot see whether hops improve an answer; answer quality is not measured.

### PostgreSQL AGE retrieval path

The live PostgreSQL image runs Apache AGE 1.7.0 on the same PostgreSQL 18 / pgvector 0.8.1 base. The
existing `graph_graphdata` volume is preserved, and the live `xpa` build 1163 has an AGE graph. For arXiv
retrieval, AGE mirrors each `sect_build` section vertex and `sect_edge` similarity relationship; it
handles neighbor expansion only. Dense and sparse ranking, BM25, section text, metadata, communities,
and answer synthesis remain in their existing relational/API path. `graph_backend` defaults to `sql`;
AGE must be selected explicitly and reports the backend actually used. A missing or mismatched AGE graph
returns an error rather than silently switching back to SQL.

`src/owui_filter.py` remains the SQL-path reference and calls `/agent`; it does not query JSONB itself.
`src/owui_filter_age.py` is the separate opt-in Function and requires the API to confirm `graph_backend=age`.
The JSONB fields in the general ChunkGraph schema remain for their current term and metadata consumers;
this retrieval change does not convert arbitrary payload fields into graph relationships.

### What the map looks like

One card per community, largest first, with its Dunning terms, its top entities, its exemplar sections and the
paper each came from. [`docs/section_map/community_map.png`](docs/section_map/community_map.png) is the
picture, with the three example questions drawn as a strip under the cards (every section each query saw, hops, subgraph size, entity composition per query); [`docs/section_map/communities.md`](docs/section_map/communities.md) contains every exemplar section with links to arXiv, followed by the three answers and traversal receipts. The community cards and text are written by `src/section_render.py`; the query strip is rendered by `src/section_query_panel.py`.

![community map](docs/section_map/community_map.png)

### How edges are decided (section to section)

Two representations of every section, compared **exactly**, all pairs, no approximate index:

| Space | Vector | Reads as |
|---|---|---|
| dense | jina-embeddings-v5-text-nano through model2vec (256 dims), centred on the corpus mean | meaning |
| sparse | BPE pieces plus adjacent pairs, unit rows | shared wording (this is where terms enter the edges) |

1. For each space, the cosine of every pair is turned into a z-score against a Box-Cox-normalised null of all pairs.
2. A pair is an edge in that space when **z >= 2.0 and it is in the node's exact top 15**.
3. Every node also keeps its top 2 neighbours per space (the backbone), at a tiny floor weight, so none is isolated.
4. The two spaces fuse by union; the edge weight is the mean z over the spaces that saw the pair (clipped at 0) plus the floor.
5. Leiden runs over the fused matrix: a resolution sweep, a plateau pick, three-run consensus, and a seed-stability gate (ARI >= 0.8).
   Communities of one section are dropped and left as unassigned nodes.

Terms never make an edge on their own. The Dunning terms on each card are **labels**: for each community, the words and
word pairs whose share inside it is most over-represented against the whole corpus (Dunning log-likelihood G2).

### How entities and entity edges are decided

Entities are derived from the text by information theory, with no tagger, no NER and no pretrained model
(`entity_derive.py`, guards AE1 to AE15):

1. Candidates are n-grams of up to 4 tokens inside one sentence, never across math, a citation or a table cell.
2. A candidate must occur in enough **papers** (not chunks), so one broken book cannot create entities.
3. It scores zero, and is dropped, if its weakest split has NPMI <= 0; if it starts or ends with a closed-class word
   (derived from the corpus, no stop list); if a longer gram accounts for most of its occurrences; if its residual IDF is
   low (spread evenly, so vocabulary rather than an entity); or if every token is a single character (math left outside `$...$`).
4. Acronym and expansion pairs (`LLM`, `large language model`) are merged into one entity.
5. The survivors form a frozen inventory of about 40,000 entities, built on the chunk map and matched, longest first with no
   overlap, onto the section bodies. That is `sect_mention` (section to entity, with a count).

Entity edges (`sect_entity_edge`, `kind = 'co_mention'`) join two entities that occur together in at least 5 sections, both
in at most 2,000 sections, kept only when NPMI = ln(p_ab / (p_a p_b)) / -ln(p_ab) is positive. The `kind` column keeps this
evidence apart from vector edges: entities are never nodes of `sect_edge`, and an entity vector is never an edge.
`rel:<class>` (DIRT-style relation edges) is reserved and not built.

Entities per community on a card are chosen by Dunning G2 against the corpus; the entities shown for a question's subgraph
are ranked by sections x ln(N / df), with hubs over 2,000 sections out and subset or superset names merged.

`louvain_pg.py` and `psql_graph.md` are the reference for running Louvain over a Postgres property graph; the section
map follows their idea (graph and communities in Postgres) but uses Leiden over `sect_edge`.

## Where things are documented

| Topic | Where |
|---|---|
| Product intent, pipeline stage table | [.spec/steering/product.md](.spec/steering/product.md) |
| Repo layout, conventions, architectural decisions | [.spec/steering/structure.md](.spec/steering/structure.md) |
| End-to-end pipeline (ingest → query → analyze → interpret) | [.spec/PIPELINE.md](.spec/PIPELINE.md) |
| Open WebUI user-toggleable arXiv Filter Function | [src/owui_filter.py](src/owui_filter.py), [src/arxiv_rag_api.py](src/arxiv_rag_api.py) |
| Cross-cutting design and rationale | [.spec/specs/graph-explorer/design.md](.spec/specs/graph-explorer/design.md) |
| **Multi-source ingest** (Brown + quotes + wikitext in one graph): per-source chunk fits (R19), source metadata (R20), per-source-pair block normalization with one global cut (R21), and the distribution-alignment rationale — why per-block Box-Cox-to-z handles register *and* size imbalance with no quotas | [design.md §6.14](.spec/specs/graph-explorer/design.md), "Multi-source ingest" |

## Quickstart — ask the corpus a question

Windows (`cmd`, the primary shell here). **`PYTHONPATH=src python ...` is bash-only
and fails in cmd with `ModuleNotFoundError: No module named 'config'`** — `set` it
on its own line instead:

```bat
docker compose up -d graphdb
set PYTHONPATH=src
set OPENROUTER_API_KEY=...
python -m streamlit run src/walker_app.py --server.port 8501
```

PowerShell uses `$env:` and bash uses the inline form:

```powershell
$env:PYTHONPATH="src"; $env:OPENROUTER_API_KEY="..."
python -m streamlit run src/walker_app.py --server.port 8501
```

```bash
PYTHONPATH=src OPENROUTER_API_KEY=... python -m streamlit run src/walker_app.py --server.port 8501
```

Open <http://localhost:8501>, pick a run in the dropdown, type a question.

`config.MODEL_DIR` must point at a model2vec artifact for the dense space (R14);
without it the system still runs sparse-only (R5).

**Two things that will otherwise look like bugs:**

- **The walker caches by (run, prompt).** Asking the same question twice replays
  the first answer rather than re-walking. Restart streamlit to clear it, or
  change the wording. (This produced a stale screenshot during T96 and cost an
  hour of misdiagnosis.)
- **Answers are not deterministic.** The agentic loop calls a model, so the same
  question can land differently run to run. Measured on `ab-section`, "who is the
  most famous musician of the 1990's?": the right artist is named in **3 of 4**
  runs; the fourth returns no answer (`entailed=0`). A blank answer is the known
  failure mode, not a regression — ask again. Everything BEFORE the model call is
  deterministic and reruns identically.

## arXiv BM25 index and defaults

The graph PostgreSQL database owns the active section map and Dunning communities (`chunkgraph-pg`, port
5433, database/user `graph`). The lexical retrieval index lives separately in `langchain_postgres` (port
5432, database/user `langchain`). The API reads graph sections/communities from the first connection and
BM25 vectors, vocabulary, tokenizer, sample, and chunk keys from the second. Keep both DSNs reachable
from the API host.

| Setting | Default | Purpose |
|---|---|---|
| `CHUNKGRAPH_DSN` | `postgresql://graph:graph@localhost:5433/graph` | Active section graph database |
| `ARXIV_BM25_DSN` | `postgresql://langchain:langchain@localhost:5432/langchain` | BM25 index database |
| `ARXIV_BM25_SAMPLE_SIZE` | `80000` | Maximum deterministic source chunks in the persisted sample materialized view |
| `ARXIV_BM25_SAMPLE_SEED` | `arxiv-bm25-sample-2026-10` | Stable sample ordering seed |
| BPE merge budget | `10000` | BPE tokenizer merges trained from the persisted sample |
| BM25 `k1`, `b` | `1.5`, `0.75` | Term-frequency saturation and length normalization |
| BM25 vector scan | Exact `<#>` inner-product scan, no HNSW/IVFFlat | Rank narrow record_id-only sparse vectors joined to build-scoped metadata |
| Sparse candidate pool | `144` | Chunks retrieved before adjacency reranking |
| Adjacency | within-word BPE alignment ±1, adjacent BPE pairs, directed neighboring words | Rerank signal, with BM25 score as tie-break |
| Community term boost | `0.25` maximum | Selected Dunning terms reweight fused section ranks before seed top-k |

Run a one-time full rebuild after the active `xpa` section graph is updated:

```powershell
$env:CHUNKGRAPH_DSN="postgresql://graph:graph@localhost:5433/graph"
$env:ARXIV_BM25_DSN="postgresql://langchain:langchain@localhost:5432/langchain"
python -u src/ingest_arxiv_sparsevec.py --from-live-graph --label arxiv_sect_bpe --bpe 10000 --no-battery
```

The ingester reads every active `sect_node` section, chunks each section independently, and writes its
stable `(doc_id, section_idx, chunk_idx)` key into the BM25 database. It creates the deterministic sample
view `lex_sample_arxiv_sect_bpe`; when the population is below 80,000 rows, that view contains every row.
The same sample trains BPE and estimates document frequency. Sample DFs scale to the full population;
terms absent from the sample receive df=1. The df=1 and robust long-word gates apply only to BPE merge
training, so all corpus terms remain in BM25 vectors. Build metadata records the graph build, sample
view/seed/scale, tokenizer JSON, and BM25 parameters. The sparse-vector bridge contains only a globally
unique `record_id` and `vec`; chunk text, build membership, and paper/section keys stay in `lex_chunk_meta`.
A generation is built as non-live and becomes live only after metadata, vocabulary, and one vector per chunk validate. Empty
vectors are retained as zero sparsevec rows. Retrieval uses exact vector operators without ANN indexes.
The default command without `--from-live-graph`
continues to use the old cache for legacy diagnostics; it is not the live BM25 source.

For automatic refresh, `src/arxiv_bm25_worker.py --watch` polls the active graph build ID every 60 seconds
and rebuilds only when that ID changes or the live generation is incomplete. Register and start the current-user
logon task with `scripts/register_arxiv_bm25_task.ps1`; it uses
`F:\Python\mcp\graph\Scripts\python.exe` and appends output to `.tmp\arxiv_bm25_worker.log`. The task uses
the current user's `CHUNKGRAPH_DSN` and `ARXIV_BM25_DSN` environment variables when set, otherwise the defaults
in `src/config.py`. It runs while that user is logged on and starts automatically at the next logon.
Do not point the worker at `C:\Users\user\py310` or
`.tmp\arxiv_section_chunks.pkl`.

## Open WebUI arXiv Filter Function

The in-process Filter calls the arXiv API's `POST /agent` route from its `inlet`. The API returns the agent's final
answer separately from `evidence[]`; each accepted evidence row carries its own bounded `text` and `truncated` flag.
The selected Open WebUI model performs synthesis inside the API pipeline. The Filter inserts the exact returned passages
before that pipeline answer under **Model commentary** in its `outlet`; a mandatory Open WebUI generation call requests
only an acknowledgement, which the outlet replaces, avoiding a second rewrite of the research answer.
Expand a paper to read its passages in Markdown `text` blocks with paper/section
links and explicit truncation notices; their text comes directly from the API rather than model transcription. Source
backticks are preserved using a longer outer fence, and HTML is displayed literally as source text. Repeated outlet
calls preserve that source text, including literal citations, HTML, and code fences. The outlet synchronizes both
`content` and the final structured `output` message because this Open WebUI client displays `output`; saving only
`content` can leave the evidence invisible. Disclosure syntax requires `<details>` and `<summary>...</summary>` on
separate lines, with a newline after each; putting both tags on one line bypasses this client's custom Markdown
tokenizer. Main and `_methods` source records sharing a canonical arXiv URL form one paper group and count once;
their original passages and citation keys remain distinct, with methods variants labeled. Entity/community TOON
tables remain visible; the overview and subgraph are visible Mermaid diagrams after commentary. Agent trace and
gate scores remain expandable, and the seed-reweighting table appears only when terms were applied. Reasoning
and tool records are preserved. The selected Open WebUI model in `body["model"]` is forwarded to the API pipeline.
The cross-encoder checks relevance, not entailment; rejected questions have no
accepted evidence. Background prompts beginning `### Task:` bypass retrieval.

For live Filter checks, leave `params.max_tokens` unset so the selected model's configured defaults apply. In the
2026-10-22 Qwen control, a 100-token cap ended with `finish_reason=length` and no final content; the model-default
request completed, and the Filter's saved response contained all 18 passage cards.

Run the API on the graph host (`192.168.3.122`) with the graph database, `OPENROUTER_API_KEY`, and section-model assets available:

```powershell
python -u src/arxiv_rag_api.py --serve --host 0.0.0.0 --port 8780
```

The API exposes `POST /agent` alongside whole-section `GET /search`. At startup it checks the live build and warms the
dense query encoder and cross-encoder before reporting ready; restart it after updating `src/arxiv_rag_api.py`.
In Open WebUI, import `src/owui_filter.py` under **Admin Settings → Functions** and leave it active but non-global.
Attach it to the intended model under **Workspace → Models → Edit → Filters**. The Filter sets `self.toggle = True`,
so you can turn it on for a chat from the Integrations menu or its input-bar chip; the model's **Start enabled in new
chats** setting controls its initial state. Global filters are forced on, so leave the Global switch off for opt-in use.
The default `rag_api_url` is `http://192.168.3.122:8780`, where the graph service runs; make sure that address is
reachable from the Open WebUI server at `192.168.3.18`. `api_key` forwards a bearer token for an API deployment
protected by a reverse proxy.

The filter shows **Searching arXiv graph…**, then the retrieved evidence/paper counts or an explicit error.
An active Function attached to a model is available for selection; that alone does not prove it ran for a chat.
Check for the retrieval status and **Retrieved graph evidence** before treating an answer as corpus-grounded.
In direct chat API requests, include `"filter_ids": ["raxiv_graphrag"]`; in the UI, enable **raxiv graphrag** under
Integrations before sending or regenerating. On 2026-10-23, an answer proposing a Python harness had neither
retrieval status nor injected evidence; the same query with the Function explicitly selected retrieved 19 passages.
`timeout_seconds` defaults to 600; retrieval failures stop the chat before model generation.
The graph API's separate `AgentRequest.timeout_seconds` defaults to 180 seconds (range 5–480), shared across
history resolution, community summaries, agent actions, and final synthesis. Internal OpenRouter task calls disable
optional reasoning and reject empty or unfinished output; each network wait is capped at 60 seconds or the remaining
budget. Final pipeline synthesis allows 2048 completion tokens; action and summary calls retain 1024. Pipeline
synthesis compares methods on source-defined criteria and checks a canonical-paper inventory for additional approaches
outside the principal comparison table. Instructions preserve parameter-changing extensions,
attribute recommendations/proposals, and mark unsupported table values as not reported. These instructions guide
interpretation; source text remains independently inspectable and is not guaranteed correct by citation-key checks.
The mandatory acknowledgement generation keeps the selected model's configured limits. Budget exhaustion returns
HTTP 504; upstream model failures return HTTP 502. Increasing the Filter timeout does not fix empty internal actions.
This repairs a reproduced Qwen failure: default reasoning consumed all 1,024 completion tokens and returned no
action after 31.62 seconds; the matched task with reasoning disabled returned a valid action in 2.52 seconds.
Restart the graph API after updating its Python modules; this server-side repair requires no Function re-upload.
The exact query `recursive self improvement agentic harness design` subsequently completed through `.18` in
122.8 seconds, with final text and 19 passage cards from nine papers; the Filter reported 77.1 seconds for the
graph response. [Open the verified chat](http://192.168.3.18:8080/c/70e15c7e-c6a3-431a-bf04-a951a0b0c1b5).
On 2026-10-22, the hosted Function was synced from `src/owui_filter.py` through the authenticated Functions API; its
source AST matches the repository, and it remains active, non-global, and toggleable. Its dynamic `selected_terms`
multiselect loads live Dunning terms. A completed Qwen chat called `/agent`, retrieved 18 evidence sections from 17
papers, and persisted one graph summary card, 18 passage cards, and arXiv links. That memory-topic result did not
cover the later harness-design timeout; its internal action-response failure is documented above. The first headless attempt capped
generation tokens and returned no final answer; the corrected model-default request completed normally. Current
deployment state and prior verification history are in [the deployment record](docs/owui_refs/deployment_current.md).
The authenticated `/api/v1/tasks/queries/completions` endpoint is not used by the Filter.

### BM25/community-term implementation status

The checked-in API now exposes `GET /community-terms`, accepts `selected_terms` on `POST /agent`, and the Filter
loads those terms into an optional multiselect. Selected terms are added to the sparse query; their community G²
affinity is applied after RRF and before seed top-k. Sparse candidates default to 144 chunks and use within-word
BPE alignment, adjacent BPE pairs, and directed adjacent-word pairs before collapsing to section keys.

As of 2026-10-22, the API at `192.168.3.122:8780` serves graph build 1163 and BM25 build 3, with matching source
IDs. `/community-terms` returns 1,719 live multiselect options. Exact sparse search retrieved the newly ingested
paper `arxiv/2610.02163` for each of three title/body queries; the hybrid top-three query did not rank it high
enough, so the corpus check uses the sparse arm. The Filter's option callback returned the same live options.
The authenticated Qwen `/agent` route returned HTTP 200; selected-term ranking was verified against the live
retriever, and the completed Open WebUI chat persisted its graph evidence cards. The current code defaults are
`CHUNKGRAPH_DSN=postgresql://graph:graph@localhost:5433/graph` and
`ARXIV_BM25_DSN=postgresql://langchain:langchain@localhost:5432/langchain`.

The data lanes are separate: `arxiv_graph_service.py` incrementally updates the older `arxiv_sect` chunk graph;
the section map (`xpa`) is the source for this BM25 index. The 2026-10-22 refresh read the extracted Markdown root;
the active build now includes 81,473 sections from 2,564 papers. Its matching BM25 build contains 82,512 chunks,
including every graph section key, with an 80,000-row sample. The section cache builder accepts an explicit root with
`python -u src/section_corpus.py --root D:\c\arxiv_id_lists\papers\post_processed`. The active `xpa` and BM25
builds have been refreshed from those records. Incremental graph-paper writes embed
before writing and commit replacement, metadata, vectors, assignments, and mentions atomically. Later cycles retry
stored papers missing vectors or assignments, repairing partial rows left by a failed encoder load.

For the legacy chunk lane, the source is `D:\c\arxiv_id_lists\papers\post_processed`; run
`F:\Python\mcp\graph\Scripts\python.exe -u src\arxiv_graph_service.py --once --root D:\c\arxiv_id_lists\papers\post_processed`
for one pass or add `--watch` for the existing 30-minute watcher. The new `--root` option applies to both modes.
Its frozen encoder is `sentence-transformers/all-MiniLM-L6-v2`, CUDA, 250-token windows, at most 32 windows per
chunk. `C:\Users\user\py310` still has CPU-only Torch paired with CUDA torchvision and fails model import; the
F: graph environment is verified and scheduled tasks now use it for API and legacy graph ingestion. The BM25
watcher is registered as a current-user logon task; setup details are in
[`F:\Python\README.md`](F:\Python\README.md).

## Checking it works

With `PYTHONPATH` already set (see above):

```bat
REM deterministic: does the gold term survive into the evidence the judge reads?
REM no model call, zero variance, rerunning gives the same number (6.29)
python src\diag_evidence.py ab-section

REM the frozen 20-row do-no-harm gate
REM expect: PASS 18/20 | FAIL ['E3'] | KNOWN-FAIL ['A2']
python src\diag_rerun.py mixed-full-dual

REM the 9-row gold lane: runs the full agentic loop and scores the ANSWER.
REM carries the model's sampling variance -- read 6.29 before comparing two
REM configurations with it; its repeat noise (0.139) exceeds the effects it is
REM usually pointed at (0.083)
python src\diag_agentic.py ab-section
```

## Ingest

`CHUNKGRAPH_MODEL_DIR` is needed for ingest only (it turns the dense space on):

```bat
for /f %i in ('python -c "import config;print(config.MODEL_DIR)"') do set CHUNKGRAPH_MODEL_DIR=%i
python src/ingest_brown.py <label>
python src/ingest_mixed.py <label> --brown N --quotes N --wiki N
python src/ingest_mixed.py <label> --chunk-mode section
pytest -q
```

`--chunk-mode section` fits chunk size on paragraph counts per heading-delimited
section, per source (R23); `document` is the older whole-document rule (R17/R19)
and remains the default so existing runs stay reproducible. Changing the
tokenizer or the chunk mode requires a re-ingest — the index stores the tokens.
