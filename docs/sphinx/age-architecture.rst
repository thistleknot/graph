Markdown to AGE Retrieval
=========================

Scope
-----

The build path starts from the selected arXiv Markdown root and persists a
section build to PostgreSQL. The AGE snapshot mirrors section adjacency. A
release record pins one validated section build and BM25 generation. The active
``raxiv_graphrag`` Open WebUI Filter calls the graph API; AGE is its default
neighbor backend, SQL remains selectable, and the generated three-query UMAP
map can be sent to final synthesis and shown in the answer.

Sphinx-Needs components and workflow
------------------------------------

Each component need records the code or store that owns a handoff. Typed links
label the transfer in the generated graph, so a link change updates the
workflow diagram from the same records.

.. comp:: Selected Markdown root
   :id: C_MARKDOWN
   :feeds: C_SECTION_CORPUS

   Source input: the operator-selected directory of arXiv Markdown files.

.. comp:: Section corpus builder
   :id: C_SECTION_CORPUS
   :produces: C_FEATURES

   ``src/section_corpus.py`` extracts usable sections and preserves document
   identity and section text.

.. comp:: Section representations
   :id: C_FEATURES
   :feeds: C_GRAPH_MAP

   ``src/section_embed.py`` and ``src/section_sparse.py`` create the dense and
   sparse representations consumed by graph construction.

.. comp:: Section graph builder
   :id: C_GRAPH_MAP
   :persists: C_SECTION_STORE

   ``src/section_map.py`` builds the fused similarity graph, communities, and
   saved build artifact.

.. comp:: Section store
   :id: C_SECTION_STORE
   :writes_to: C_SECTION_DB, C_AGE_GRAPH

   ``src/section_store.py`` commits the relational section build and its
   build-scoped AGE graph before publishing the live build.

.. comp:: Relational section database
   :id: C_SECTION_DB

   PostgreSQL owns ``sect_build``, ``sect_node``, ``sect_embedding``,
   ``sect_edge``, and ``sect_community``. It remains authoritative for section
   text, metadata, ranking inputs, and community data.

.. comp:: Apache AGE graph
   :id: C_AGE_GRAPH
   :returns_neighbors: C_GRAPH_API

   ``arxiv_section_<build_id>`` stores ``Section`` vertices and
   ``SIMILAR_TO`` relationships. AGE supplies neighbor expansion only.

.. comp:: Serving release pointer
   :id: C_RELEASE
   :reads: C_SECTION_DB, C_BM25_DB

   ``sect_serving_release`` pins the graph build and ready BM25 build used by
   requests.

.. comp:: Unified Open WebUI Filter
   :id: C_OWUI_FILTER
   :requests: C_GRAPH_API

   ``src/owui_filter.py`` sends the question, selected backend, and optional
   ``include_community_map`` flag to ``POST /agent``. It checks backend and map
   status, then displays evidence and the generated UMAP image.

.. comp:: Graph API
   :id: C_GRAPH_API
   :reads: C_SECTION_DB, C_BM25_DB
   :queries: C_AGE_GRAPH
   :calls_model: C_MODEL
   :returns_response: C_OWUI_FILTER

   ``src/arxiv_rag_api.py`` resolves one release pair, coordinates retrieval
   and synthesis, and returns retrieval evidence or an aggregate-job receipt.

.. comp:: BM25 index database
   :id: C_BM25_DB

   The separate ``langchain_postgres`` database provides sparse BPE/BM25
   candidates; it is not stored in the AGE graph database.

.. comp:: BM25 index builder
   :id: C_BM25_BUILDER
   :reads: C_SECTION_DB
   :writes_to: C_BM25_DB, C_RELEASE

   ``src/ingest_arxiv_sparsevec.py`` validates a candidate generation before
   moving the serving release pointer.

.. comp:: Selected synthesis model
   :id: C_MODEL

   The API forwards the selected model to the answer pipeline after retrieval.

.. needflow:: Build and retrieval handoffs
   :filter: id in ("C_MARKDOWN", "C_SECTION_CORPUS", "C_FEATURES", "C_GRAPH_MAP", "C_SECTION_STORE", "C_SECTION_DB", "C_AGE_GRAPH", "C_RELEASE", "C_OWUI_FILTER", "C_GRAPH_API", "C_BM25_DB", "C_BM25_BUILDER", "C_MODEL")
   :link_types: feeds, produces, persists, writes_to, requests, reads, queries, returns_neighbors, calls_model, returns_response
   :show_link_names:
   :direction: right
   :alt: Markdown section build and AGE-backed Open WebUI retrieval component flow

Class-style component view
--------------------------

The boxes summarize the data and operations owned by each component. The
direction of the arrows shows the build-time or request-time handoff; the
``returns`` edge marks the response path back to Open WebUI.

.. graphviz::
   :caption: Component responsibilities and runtime interfaces
   :alt: Class-style view of Markdown ingest, PostgreSQL and AGE storage, and Open WebUI retrieval

   digraph age_architecture {
       graph [rankdir=TB, compound=true, fontname="Arial", nodesep=0.4, ranksep=0.55];
       node [shape=record, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", fontname="Arial", fontsize=10, margin="0.12,0.08"];
       edge [color="#475569", fontname="Arial", fontsize=9, arrowsize=0.75];

       markdown [label="{Markdown root|*.md files|selected source directory}"];
       corpus [label="{section_corpus.py|extract_sections()|section text + source keys}"];
       representations [label="{section_embed.py + section_sparse.py|encode_dense()|build_sparse()}"];
       mapper [label="{section_map.py|build_edges()|assign_communities()}"];
       store [label="{section_store.py|persist_build()|build_age_graph()}"];

       section_db [label="{PostgreSQL section store|sect_build / sect_node|sect_embedding / sect_edge|sect_community}", fillcolor="#dbeafe"];
       age [label="{Apache AGE graph|Section vertices|SIMILAR_TO edges|neighbors(seed_ids)}", fillcolor="#dbeafe"];
       bm25 [label="{langchain_postgres|BPE sparse vectors|BM25 candidate ranking}", fillcolor="#dbeafe"];
       release [label="{sect_serving_release|graph_build_id|bm25_label + bm25_build_id}", fillcolor="#dbeafe"];
       bm25_builder [label="{ingest_arxiv_sparsevec.py|build + validate|publish matched release}"];

       filter [label="{Unified Open WebUI Filter|inlet(question)|AGE default; SQL option|community map + evidence}", fillcolor="#dcfce7"];
       api [label="{arxiv_rag_api.py|POST /agent|coordinate retrieval + synthesis}", fillcolor="#dcfce7"];
       retrieval [label="{Section retrieval|dense ranking|BM25 ranking|AGE neighbor expansion}", fillcolor="#dcfce7"];
       model [label="{Selected model|answer synthesis|citation-key checks}", fillcolor="#fef3c7"];

       markdown -> corpus [label="Markdown"];
       corpus -> representations [label="sections"];
       representations -> mapper [label="vectors"];
       mapper -> store [label="build artifact"];
       store -> section_db [label="relational snapshot"];
       store -> age [label="same-transaction graph snapshot"];
       section_db -> bm25_builder [label="live graph rows"];
       bm25_builder -> bm25 [label="candidate generation"];
       bm25_builder -> release [label="atomic serving switch"];
       release -> api [label="pinned build IDs"];
       section_db -> release [label="graph build ID"];
       bm25 -> release [label="BM25 build ID"];

       filter -> api [label="HTTP request: AGE required"];
       api -> retrieval [label="question + selected terms"];
       retrieval -> section_db [label="dense seeds + section data", dir=both, constraint=false];
       retrieval -> bm25 [label="sparse candidates", dir=both, constraint=false];
       retrieval -> age [label="neighbor expansion", dir=both, constraint=false];
       retrieval -> model [label="retrieved evidence"];
       model -> api [label="final + evidence", constraint=false];
       api -> filter [label="aggregate job receipt (202)", constraint=false];
       api -> filter [label="returns", constraint=false];
   }

Data ownership
--------------

* **Relational PostgreSQL** owns the section build, exact dense ranking inputs,
  section text, communities, and metadata.
* **The BM25 PostgreSQL database** supplies sparse candidates through its own
  BPE/BM25 index.
* **Apache AGE** traverses section-to-section ``SIMILAR_TO`` neighbors; it does
  not replace either ranking store or own answer synthesis.
* **The unified Open WebUI Filter** selects AGE by default, keeps SQL available,
  checks the reported backend, and renders the repository map with returned evidence.
* **The serving release pointer** is the commit point for a matched graph and
  BM25 pair; local ``live`` markers remain build-tool state.

.. note::

   The unified filter keeps both backend choices under the existing
   ``raxiv_graphrag`` identity. API retrieval defaults remain SQL for callers
   outside this Open WebUI function.
