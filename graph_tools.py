"""
graph_tools.py — deterministic, index-safe traversal primitives over a persisted
ChunkGraph run. This is the tool surface an LLM caller is handed; it is NOT an
LLM itself. Every function here is a plain parameterized SELECT.

WHY THIS SHAPE
The graph is built deterministically -- chunking, edges, fusion and Louvain all
run at ingest with no model in the loop. A caller that walks it at query time
therefore chooses only the ROUTE; the substrate and every community label it
reports were fixed up front. That is what keeps an agentic walk auditable: the
sequence of calls below fully reconstructs what was seen and why.

The builder exists so a tool-caller CANNOT express an unsafe query. There is no
sql() primitive and no string interpolation of identifiers. Guardrail, not sugar.

TOOL SURFACE
| tool                | question it answers                          | bound  |
|---------------------|----------------------------------------------|--------|
| list_runs           | which runs exist                             | all    |
| get_run             | identity + capability of one run             | 1 row  |
| search              | where do I start, lexically                  | k      |
| node                | what is this chunk, who owns it              | 1 row  |
| neighbors           | where can I step from here                   | limit  |
| community           | what is this cid about                       | 1 row  |
| communities_touched | what does this node set partake in           | n cids |
| bridges             | which edges actually connect two communities | limit  |
| quotient            | how communities interconnect, corpus-wide    | limit  |
| subgraph_edges      | induced edges among a visited set (drawing)  | set    |
| walk                | reach + WHY: prov_path/doc_path per result   | cap    |
| term_stats          | which query terms exist, and who carries them| terms  |
| community_terms     | top-k BM25 terms, community as the document  | k/cid  |
| query_terms         | those terms re-ranked by the prompt          | k/cid  |
| chunk_salient       | a chunk's own BM25 terms, gated (W14), top-k | ord    |
| pathways            | idea-to-idea DWPC over the walked subgraph (W15) | ords, anchors |
| chunk_terms         | the top-k of that, for titles                | k/ord  |
| local_medoid        | most central retrieved chunk in a community  | 1/cid  |
| cross_community     | retrieved chunks bridging retrieved cids     | set    |
| second_order         | which terms keep this term's company (W17)   | k/term |
| node_metrics        | betweenness/pagerank/triangles per node (W18) | run   |
| community_metrics   | density + conductance per stored cid (W19)   | cids  |

GUARDS (EARS)
W1  Every statement SHALL be run-scoped: run_id is the first predicate, so the
    (run_id, ...) index prefixes are usable and cross-run bleed is impossible.
W2  Traversal SHALL read edge / edge_sym via the indexed integer src/dst columns
    only. attrs is read for payload and lexical scoring, NEVER as a traversal or
    join key (steering hard constraint 3).
W3  The connection SHALL be opened read-only at the server. A write raises
    psycopg.errors.ReadOnlySqlTransaction rather than relying on convention.
W4  Every tool returning rows SHALL take an explicit bound.
W5  WHERE a run has no dense space (embed_dim NULL or zero embedding rows),
    capability SHALL report sparse-only rather than failing.
W6  Live rows only: valid_to IS NULL on edges, superseded_at IS NULL on runs
    (the live_run view). Note graph_run has NO valid_to column.
W8  term_stats() SHALL report a query term the corpus does NOT contain, with
    df 0, rather than dropping it. A term absent from the run's vocabulary is
    the single most common reason a retrieval looks wrong, and silently
    omitting it makes the query look like it asked for less than it did.
W9  community_terms() SHALL score over EVERY member of the community, never
    only the retrieved ones. The ranked terms are the implied evidence -- the
    concept the community holds -- and scoring them on the retrieved subset
    collapses that back into a search result.
W10 query_terms() SHALL choose only from the community's OWN vocabulary
    (community_terms pool). The prompt re-ranks; it never imports a term the
    community does not carry. Communities stay unsupervised -- only the
    three words shown for each are conditioned on the prompt.
W11 local_medoid() SHALL weight BOTH the chunk and its neighbours by walk
    score: centrality(o) = w(o) * sum_j w(j) * strength(o, j). Neighbour
    weighting alone tracks structural centrality and returned the global
    medoid at 25 of 110 retrieved; the chunk's own relevance must multiply in.
W12 search() SHALL be BM25 -- tf saturation (k1) and length normalisation (b)
    on the stored n_tok -- not tf*idf. Measured at document-level nodes: plain
    tf*idf anchored on long essays dense in common query words and missed 19
    of 23 documents carrying the rare, decisive term.
W14 Which of a chunk's terms are SALIENT is a gate, not a top-k: log-normalise
    the BM25 scores and keep all at or above min(median - 1.4826*MAD, mean -
    sd). Both branches are one sigma below a centre on the same scale; the
    disjunction keeps the upper half and a little more (trigram.md's dual
    measure, measured 24/24 probe recall there vs 20-21/24 for a lone median).
    Titles show the top-k OF the kept set; the kept set is the vocabulary.
W7  walk() SHALL return, per reached node, the EDGE PROVENANCE and SOURCE DOC of
    every hop that reached it -- not merely the node ids traversed. A route that
    cannot name its own justification is a browser, not an evidence instrument.
    The arrays ride the recursive CTE at no extra join: edge_sym already exposes
    provenance and both endpoint docs.
W15 Evidence pathways are computed on the induced subgraph only, with GLOBAL
    degrees for damping and conductance; a pair's DWPC sums simple paths of
    <= max_len edges, each scored prod(strength) * prod(deg^-damp) over every
    node on the path, so no path outranks the same path through smaller hubs.
W16 Bridge discovery searches the WHOLE live edge set: best_path maximises
    prod(strength) x prod(deg^-damp) over interior nodes (Dijkstra on the
    negative log). It finds evidence the walk missed; pathways (W15) only
    scores what was already retrieved. Two jobs, two functions.
W17 second_order_terms() SHALL apply the ladder as a gate-then-fallback chain,
    in order: (a) Dunning-LLR co-occurrence gate (rung "llr") admits only
    candidates whose 2x2 association with the target clears g2_gate, and
    min_df floors BOTH the target and every candidate before anything else
    runs; (b) Schutze context-centroid cosine (rung "centroid") ranks the
    LLR survivors by shared company, not raw co-occurrence; (c) a skew
    diagnostic over the survivors' cosine scores decides whether the
    centroid ranking is trustworthy -- ONLY when it trips (|skew| >
    skew_trip, and only past min_skew_n survivors) does Mann-Whitney AUC
    re-rank the top_k (rung "auc"); otherwise the centroid ordering from (b)
    stands untouched. WHERE a run has no dense space (W5), the ladder
    degrades to the LLR ordering (rung "llr", cos/auc None) rather than
    failing -- rungs (b)/(c) are a dense-only refinement, never a
    requirement. The v0 nomen pool is a df-band + stoplist floor (df in
    [min_df, max_df_frac*n], len > 2, not stoplisted); true PPMI demotion of
    the high-frequency band is LATER, once entities v0's ppmi table exists.
W18 node_metrics() SHALL compute the centrality lane over the WHOLE run and cache
    it by (run_id, k, seed); `ords` filters the returned view, never the
    computation, so two callers asking about different chunks get numbers off the
    same graph. Betweenness is exact at or below BETWEENNESS_EXACT_MAX nodes and
    the k-sample estimator above it, with k and seed PINNED as constants -- an
    unseeded sample would make the number a coin flip and the cache a lie.
    Betweenness and clustering are computed unweighted: networkx reads `weight`
    as a DISTANCE and our strengths are similarities, so passing them would make
    the strongest edges the longest. PageRank, where higher weight genuinely is
    closer, uses strength, with alpha and tol pinned. Per-provenance degree
    (sparse/dense/both) is read from edge_sym's provenance column and SHALL sum
    to the node's degree.
W19 community_metrics() SHALL score the STORED partition -- density and
    conductance per cid read from community.members, never a re-partitioning of
    an induced subgraph -- and SHALL report the whole-run WCC (component count
    and sizes) beside it, because a modularity partition of a shattered graph is
    a number without a shape. The run-wide summary is median and p90, not a mean:
    community sizes are heavy-tailed and a mean reports the tail.
W20 Personalized PageRank from the walk anchors is an ADDITIVE named column in
    pathways(): every pair keeps its dwpc, the pair ordering stays DWPC's, and
    `ppr` rides beside it. Two measures of connectedness, named separately, so a
    later swap is a measured decision rather than a silent one. Nothing consumes
    these metrics yet -- surfacing is T22's job.
W21 Alias expansion SHALL be opt-in (`expand_aliases`, default OFF) and ADDITIVE:
    a query token that is itself an entity surface form SHALL contribute the other
    names sharing its `canonical_id` as extra OR-terms at full BM25 weight, and NO
    original query term SHALL be dropped or reweighted. Matching SHALL be exact
    against `entities.name` -- `tokenize` and the entity vocabulary are the same
    vocabulary, and a looser match reopens the false merges E7 exists to close.
    WHERE resolution has not run for a run -- no `entities` table, no rows, or
    `canonical_id` NULL -- the map SHALL read back empty and search SHALL be
    byte-identical to the unexpanded call, never an error.

NOT HERE, DELIBERATELY
- No LLM. No prompt, no model call, no NL->query translation.
- No re-partitioning. cid is read from community, never recomputed over an
  induced subgraph (subgraph Louvain does not restrict global Louvain).
- No writes of any kind.

SOURCE (R20)
`source` is read from `attrs->>'source'` and is metadata only -- it never
filters, weights, or steers a traversal (W2 still holds). A run predating R20
has no `source` key, so it reads back as `None` and renders unlabelled rather
than failing (design.md §6.14 R20).

MODULE MAP (T37 split -- this file is a re-export shim, no def, no SQL)
This module used to be 1571 lines holding three unrelated concerns. It is now
an index over three sibling modules, each owning one concern and stating its
own guard subset (their docstrings name which W-ids they implement):
  gt_sql.py     -- parameterized SELECTs, run identity, provenance narration
                   (W1-W13, W21 search-half, R20)
  gt_terms.py   -- vocabulary, salience, the disk-cache layer (W8, W12
                   tokenize-half, W14, W17, W21 expand-half)
  gt_metrics.py -- adjacency numerics: centrality, pathways, partition
                   metrics (W15, W16, W18, W19, W20)
Every cross-module call (e.g. gt_sql.search calling gt_terms.tokenize) is
resolved at CALL TIME through this shim via a `_gt()` helper defined
identically in each sibling -- see gt_terms.py's `_gt` docstring for why.
This file remains the single place all 21 guards are STATED; the siblings
only IMPLEMENT them.
"""
from __future__ import annotations

# Spec: .spec/specs/graph-explorer/design.md (module split, no behaviour change)
# Task: playbook.md T37

from gt_sql import (
    DSN, RunHandle, connect, list_runs, get_run, search, dense_search, node, neighbors,
    community, communities_touched, run_sources, bridges, quotient,
    subgraph_edges, walk, why, source_of, source_mix, format_source_mix,
    term_stats, community_terms, local_medoid, cross_community, query_terms,
    _decorate, _CT_CACHE,
)
from gt_terms import (
    _STOP, tokenize, expand_terms, corpus_index, corpus_df, alias_map,
    salient_gate, chunk_salient, chunk_terms, dendrite_sort,
    subgraph_embeddings, llr, second_order_terms, second_order,
    CACHE_DIR, _disk, _disk_put, _DF_CACHE, _ALIAS_CACHE,
)
from gt_metrics import (
    BETWEENNESS_EXACT_MAX, BETWEENNESS_K, BETWEENNESS_SEED,
    PR_ALPHA, PR_TOL, PR_MAX_ITER, PPR_ALPHA, PPR_TOL, PPR_MAX_ITER,
    degrees, pathways, full_adjacency, best_path, graph_metrics,
    partition_metrics, ppr, node_metrics, community_metrics,
    _pct, _ADJ_CACHE, _METRIC_CACHE, _COMM_METRIC_CACHE,
)

__all__ = [
    "DSN", "RunHandle", "connect", "list_runs", "get_run", "search", "node",
    "neighbors", "community", "communities_touched", "run_sources", "bridges",
    "quotient", "subgraph_edges", "walk", "why", "source_of", "source_mix",
    "format_source_mix", "term_stats", "community_terms", "local_medoid",
    "cross_community", "query_terms",
    "tokenize", "expand_terms", "corpus_index", "corpus_df", "alias_map",
    "salient_gate", "chunk_salient", "chunk_terms", "dendrite_sort",
    "subgraph_embeddings", "llr", "second_order_terms", "second_order",
    "CACHE_DIR",
    "BETWEENNESS_EXACT_MAX", "BETWEENNESS_K", "BETWEENNESS_SEED",
    "PR_ALPHA", "PR_TOL", "PR_MAX_ITER", "PPR_ALPHA", "PPR_TOL", "PPR_MAX_ITER",
    "degrees", "pathways", "full_adjacency", "best_path", "graph_metrics",
    "partition_metrics", "ppr", "node_metrics", "community_metrics",
]
