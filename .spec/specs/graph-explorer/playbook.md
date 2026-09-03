# Walk tab playbook -- design §6

One line each. Status is the only thing that changes in this file.

- [DONE] community-as-document BM25 top-k terms          graph_tools.community_terms
- [DONE] local medoid over retrieved chunks per community graph_tools.local_medoid
- [DONE] in-between exemplars by cross-community edges   graph_tools.cross_community
- [DONE] Walk tab: community graph, terms as labels       walker_app.py
- [DONE] Walk tab: ranked list, presence, top-3, 2 medoids walker_app.py
- [DONE] Walk tab: expander -> retrieved chunks w/ prov    walker_app.py
- [DONE] Walk tab: in-between exemplars section           walker_app.py
- [DONE] tests: the three primitives                      tests/test_graph_tools.py
- [DONE] test: Walk renders communities from a prompt     tests/test_walker_render.py
- [DONE] commit
- [LATER] MMR: >1 exemplar per community (same similarity, diversified)
- [LATER] chunk-in-many-communities -- needs term-node layer (graph-term-selection)

## §6.1 query-conditioned terms
- [DONE] graph_tools.query_terms: lexical-first, cosine re-rank of community pool (W10)
- [DONE] walker_app: embed loader from CHUNKGRAPH_MODEL_DIR; labels use query_terms
- [DONE] walker_app: expander shows unsupervised "community concept" alongside
- [DONE] tests: pool invariant, lexical-first, cosine argmax with a fake embed
- [DONE] commit
- [TODO] stamp model_dir into graph_run.params at ingest (reproducibility)

## §6.2 / §6.3 medoid weighting, chunk boundaries
- [DONE] graph_tools.local_medoid takes walk-score weights (W11); Bundle.scores
- [DONE] chunkgraph._chunk recursive, never intra-word (R16)
- [DONE] walker_app: word-boundary clip for previews; medoid uses weights
- [DONE] tests: chunk battery (3 shapes); weighted medoid differs from unweighted
- [DONE] commit
- [DONE] re-ingest brown-50 and brown-50-dual under R16
- [DONE] re-pin brown-50 shape constants in tests; commit

## after the colonial-power walk (2026-08-29)
- [DONE] local_medoid multiplies the chunk's OWN walk score in (W11 amended)
- [DONE] draft labels apply only when community_labels.json names THIS run_id
- [DONE] walker restarted with CHUNKGRAPH_MODEL_DIR so the dense signal is on
- [TODO] dense anchoring: anchors are BM25-only, so `power` anchored a physics
         lecture (c1 hydrogen/peas/planets, 17 of 24) and the fused edges could
         not repair a bad anchor. Anchor set := BM25 top-k UNION dense ANN top-k
         over node_embedding (cookbook #4), each anchor tagged with which space
         found it; the walk's provenance then starts at the anchor, not one hop in.
- [TODO] stamp model_dir into graph_run.params; app reads it from the run, not env

## §6.4 interpretation layer
- [DONE] interpret.render_bundle: deterministic evidence text with #ord ids
- [DONE] interpret.answer: OpenRouter qwen/qwen3.5-9b default, Ollama fallback think=False (I5)
- [DONE] interpret: entailment JSON per chunk; answer from ENTAILED only (I6); I1 on both
- [DONE] interpret: rerank hook (pylate ColBERT) engaged only when RERANK_MODEL is cached (I7)
- [DONE] walker_app: Interpret pane, draft-marked, foreign citations surfaced
- [DONE] tests: render is deterministic and complete; citation check; live call skips without Ollama
- [DONE] commit

## §6.5 corpus-derived chunking (R17)
- [DONE] chunkgraph.derive_chunk_params: Box-Cox m/hi over lines-per-DOCUMENT, chars fallback (6.5a)
- [DONE] chunkgraph._chunk: one chunk per document; split above hi at paragraph bounds; tail merge; conservation
- [DONE] fit() derives params over the corpus, records them in diagnostics and run params
- [DONE] tests: battery -- params, split at paragraph bounds, tail merge, conservation, chars fallback (29)
- [DONE] re-ingest brown-50 and brown-50-dual; re-pin constants (51 nodes); commit
- [DONE] 500 docs -> 500 nodes (8 MB dense, not 7.4 GB); brown-500-dual ingested

## after the timeout screenshot
- [DONE] interpret: urllib client in rl_V2's shape (Avast TLS proxy stalled httpx); 3 attempts, backoff
- [TODO] walker: label the prompt box as the retrieval query; separate 'ask about this walk' box
- [TODO] interpret: EXPLAIN mode -- prose over the same rendered bundle, same citation check, no verdicts
- [TODO] re-measure ef on the 500-document graph (sweep was on 1,789 chunks)
- [DONE] evidence pathways: idea-to-idea DWPC + subgraph shape over the walk (design 6.11, W15)
  _Files:_ graph_tools.py, walker_app.py, tests/test_sampler.py  _Verify:_ pytest tests/test_sampler.py -q -k w15
- [TODO] reach test: ten prompts with hand-listed topic docs; score = topic docs reached by the walk. Gates the next two.
  _Files:_ tests/test_reach.py  _Verify:_ pytest tests/test_reach.py -q
- [TODO] walk-time DWPC: damp the path score by deg(intermediate)^-0.4 in ef_search and ring; keep only if the reach test does not drop
  _Files:_ sampler.py, tests/test_sampler.py  _Verify:_ pytest tests/test_sampler.py tests/test_reach.py -q
- [TODO] anchor expansion: decode the prompt's model2vec vector to its nearest corpus vocabulary rows (CSLS, salient-gated) and add them as BM25 anchors; keep only if the reach test improves on >=7/10
  _Files:_ sampler.py, graph_tools.py  _Verify:_ pytest tests/test_reach.py -q
- [DONE] structural evidence to the model: image (default) or numbers, same pathways object; fallback noted (I12, design 6.12)
  _Files:_ interpret.py, walker_app.py  _Verify:_ pytest tests/test_interpret.py -q -k i12
- [DONE] reason+judge as ONE serialized call with a verdicts key; walker single button (I13)
  _Files:_ interpret.py, walker_app.py, tests/test_interpret.py, tests/test_walker_render.py  _Verify:_ pytest tests/test_interpret.py -q -k i13
- [DONE] bridge discovery: whole-graph best-path between top chunks pulls unretrieved chunks into evidence, bolded with italic terms + key (S14/W16, design 6.13)
  _Files:_ sampler.py, graph_tools.py, walker_app.py, tests/test_sampler.py  _Verify:_ pytest tests/test_sampler.py -q -k 's14 or w16'
- [DONE] PIPELINE.md FAQ: model visibility, subgraph-vs-global, scale/samples, DWPC scope, WCC, NPMI/BM25 roles, Leiden
  _Files:_ .spec/PIPELINE.md  _Verify:_ read
- [TODO] synset-style word collapse: word-column correlation + knee/anomaly candidate detection + dual-measure certification (term-selection amendment 2026-08-31)
  _Files:_ chunkgraph.py, salient_grams.py  _Verify:_ 4-pair battery incl. antonym pair that must not merge
- [TODO] A/B chunk-term weighting: BM25 vs length-normalized PMI (log-ratio keyness) + Dunning-G2 gate, brown-50, same edge pipeline; compare community coherence and walk reach
  _Evidence (operator comparison, 2026-09-01):_ the quadrants are settled -- low-BM25/high-PPMI terms are garbage for a retrieval vocabulary (PMI's rare-event bias: tight one-context collocates, names, fragments); high-BM25/low-PPMI is where the real domain terms live. Direction: PMI does not replace BM25 as the term weight; at most PPMI is a DEMOTION filter on the vocabulary (drop the low-BM25/high-PPMI quadrant). The A/B's remaining open question is only whether that filter moves community coherence / walk reach.
  _Files:_ chunkgraph.py, tests/test_chunk.py  _Verify:_ side-by-side ingest report
- [DONE] walk map speaks terms + bipartite translation layer (operator, 2026-09-01, option C): each subgraph node shows its top-3 salient terms as a small spring-dithered word cloud (deterministic per-ord jitter), id kept tiny for citation cross-reference; global map communities show their top-3 keywords and plain red DWPC hops (midpoint term labels tried and rejected -- illegible in the walked cluster). NEW third panel: bipartite translation layer -- pathway chunks (top, named by top term) <-> their salient terms drawn as nodes (below, barycentric x, deterministic dither); red hop arcs along the chunk row, DASHED when the two chunks share no drawn term = semantic bridge (dense-space hop made visible). I12 determinism preserved, pinned by byte-identical double render.
  _Files:_ interpret.py, tests/test_interpret.py  _Verify:_ pytest tests/test_interpret.py -q
- [DONE] term graph sub-tab: the walk re-expressed with TERMS as nodes (operator, 2026-09-02): tri-state membership colour (prompt-conditioned BM25 set / global unsupervised concept / both -- the provenance idiom applied to vocabulary, W10's two sets finally drawn against each other), edges = co-occurrence within walked chunks (>=2 shared), size = subgraph df. Deterministic (seed 7).
  _Files:_ walker_app.py, tests/test_walker_render.py  _Verify:_ pytest tests/test_walker_render.py -q
- [DONE] dendrite sorting + 3D layered view (operator design, correlation sorting.md, 2026-09-02): gt.dendrite_sort -- correlation-chain decomposition (signed log1p -> Yeo-Johnson -> MAD-z per column, Pearson with significance at n=observations, master = highest mean significant r, chains grow from the tail, terminate when no significant unassigned candidate; min_support floor). Applied to BOTH planes of one term-document matrix: chunks over rows (dense-cosine profiles, n=walked chunks), terms over columns (tf*idf profiles over the walked chunks' own salient vocabulary). New walker sub-tab '3D layers': chunk plane z=0 (community colour, source symbol), term plane z=1, each spring-settled in its own slice, chain backbones red/blue, membership edges falling between the planes. Measured on the midway walk: 74 chunks -> chains 68+6; 64 terms -> 7 chains, longest a 42-term naval thread (converted_minelayer ... ark_royal ... kamikaze ... yamato).
  _Files:_ graph_tools.py, walker_app.py, tests/test_graph_tools.py, tests/test_walker_render.py  _Verify:_ pytest tests/test_graph_tools.py tests/test_walker_render.py -q
- [DONE] structure digest: pregrouped partitions as compact text for the model (operator, 2026-09-02: "if a VLM can't interpret the image a human can't either" -- the model reasons over logically pregrouped data, never pixels). interpret.render_digest serializes communities+source mix, both dendrite chain sets (chunk ids elided past 20), tri-state term sets, top-quartile chunk<->term bindings, and DWPC pathways in a TOON-style pipe-packed form (~1k tokens on the midway walk); reason(digest=...) appends it to the structure channel in BOTH image and numbers modes. Deterministic, pinned.
  _Files:_ interpret.py, walker_app.py, tests/test_interpret.py  _Verify:_ pytest tests/test_interpret.py -q
- [DONE] title as persisted node metadata (operator principle, 2026-09-03: parse once at the boundary, no downstream pattern-scanning -- R20's law extended from `source` to `title`): carry wiki_title() through fit into the node payload jsonb so the walker, digest, and exports name articles instead of doc ids, and any future pinning/selection is metadata equality. Needs a re-ingest to take effect; payload contract amendment in design.md 6.14 first.
  _Files:_ ingest_mixed.py, chunkgraph.py, pg_store.py, .spec/specs/graph-explorer/design.md  _Verify:_ pytest tests/test_pg_store.py tests/test_ingest_mixed.py -q
  — R22 shipped (design 6.15 B): title carried through fit() into node payload jsonb, absent-not-null; walker + digest degrade to doc_id on pre-R22 runs; 106 passed / 2 skipped. Re-ingest still pending (LATER).
- [DONE] neo4j mirror keeps the analysis layer (operator, 2026-09-03: "dont forget the subgraph analysis -- dwpc, conductance, all that jazz"): walks become first-class -- (:Walk {prompt, n, edges, wcc, density, conductance})-[:ANCHORS]->(:Chunk), (:Chunk)-[:PATHWAY {dwpc, of}]->(:Chunk) -- written from gt.pathways at retrieval time (demonstrated live on the midway walk, 2026-09-03: density 0.67, conductance 0.80, top DWPC 0.138 3439<->3552). Also: ship 256-dim embeddings as a chunks.csv column + CREATE VECTOR INDEX for neo4j-native vector-topk -> Cypher-expansion GraphRAG; C<cid> dynamic labels carry the canonical ingest Louvain into Cypher (done live, 10,820 labeled).
  _Files:_ export_neo4j.py, tests/test_export_neo4j.py  _Verify:_ pytest tests/test_export_neo4j.py -q
  — X9/X10 shipped (design 6.16 B): write_walk MERGEs Walk/ANCHORS/PATHWAY over urllib, idempotent on re-write, live round-trip against :7474 verified; 33 tests.
- [DONE] export_neo4j: emit the modern 4-file neo4j-admin layout (measured 2026-09-03: the dual-ID single-file header `id:ID(Chunk)`+`id:ID(Term)` is rejected by current neo4j-admin as a duplicate property; the scratch splitter in the 2026-09-03 session imported 286,158 nodes / 6,462,576 rels in 42s from chunks/terms/contains/similar.csv with --multiline-fields=true). Fold the split into the exporter, update import.sh, pin with a header-shape test.
  _Files:_ export_neo4j.py, tests/test_export_neo4j.py  _Verify:_ pytest tests/test_export_neo4j.py -q
  — X6/X7/X8 shipped (design 6.16 A): chunks/terms/contains/similar.csv + ;-separated embedding column + vector_index.cypher + C<cid> :LABEL column; header-shape test pinned; 24 tests.
- [DONE] second-order term co-occurrence lane (operator design, 2026-09-02): domain token <-> semantic nomen affinity via a three-rung ladder -- (1) Dunning-LLR gate on chunk-level co-occurrence (candidates must beat chance at first order; also floors context counts), (2) score survivors by Schutze context-centroid cosine (term = mean embedding of the chunks carrying it; second-order signal, shared company without shared occurrence), (3) re-rank by Mann-Whitney AUC (t-chunks vs non-t-chunks against the nomen centroid) ONLY when the centroid ranking's skew diagnostic trips -- the R2/R6 gate-then-fallback pattern at term level. Probe first in .tmp/second_order_probe.py vs mixed-full-dual; promote to salient_grams/graph_tools with an R-guard on measured lift.
  _Files:_ .tmp/second_order_probe.py (probe), then TBD  _Verify:_ probe prints per-rung rankings for >=3 domain tokens
  — W17 shipped (design 6.16 C): promoted out of .tmp/second_order_probe.py into graph_tools; 9 planted + 1 live test, 92/92 green.
- [DONE] entities v0: bipartite entity store beside the chunk graph (E1-E5, design 6.15 C) — entities/mentions/entity_edges, run-scoped, MIN_JOINT_CHUNKS=5 floor, npmi/ppmi/bm25 co-resident on one row; 13 tests against live Postgres :5433.
- [BLOCKED] B-class register dilution: quotes share on the dual run — HANDED TO OPERATOR 2026-09-03. Two structural scopes spent (sqrt anchors -> measured harmful; additive anchors + S17 source-aware ring share -> 8-29%, gate is 40%). E3 also unrestored, failure mode now upstream in ring/PPR steering. Ladder law: no third re-scope. Evidence: design 6.15 A outcome block, diagnostic-prompts.md 2026-09-03 T7c entry.
- [TODO] dual-run probe battery as process, not chat advice (operator, 2026-09-01): (1) cross-register prompts whose walk must cross a dense-only bridge (e.g. a quotes-anchored prompt reaching Brown fiction), asserting the source-changing hop has provenance=dense; (2) community source-mix audit -- per-cid node counts by source separating integrated cross-register topics from single-source islands (dual run collapsed 27 comms -> 16; the merge is the claim to falsify); (3) a `both`-provenance cross-source edge as the corroboration exhibit
  _Files:_ tests/test_mixed_acceptance.py, cookbook/  _Verify:_ pytest tests/test_mixed_acceptance.py -q
- [LATER] Leiden in place of Louvain at ingest (same API, better partitions; try on the next re-ingest against the coarse-7 problem)
- [LATER] build-time DWPC term-mediated edges X.diag(deg^-0.4).X^T (graph-term-selection spec, approved, unbuilt)
- [DONE] I8 excerpt by prompt: lexical+dense, paragraph cap at budget/3, anchors x4
- [DONE] I6 verdict = answers-or-partly-answers; dedupe; two-stage answer over entailed
- [DONE] dual-run fixtures -> brown-500-dual; sampler stability pinned there, n<=24; render hub derived

## §6.6 reason over community evidence
- [DONE] interpret.community_briefs: per community terms x2, local + global medoid excerpts (deterministic)
- [DONE] interpret.reason: hypothesis -> premises -> evaluate -> answer; I9-I11 checks; stage texts kept
- [DONE] walker_app: 'Reason about this walk' -- hypothesis, premises with verdicts, answer, briefs
- [DONE] tests: briefs deterministic + both medoids; staged pipeline with fake backends; I9/I10 checks
- [DONE] live run on 'how did communities respond to school desegregation'; commit
- [DONE] W12 search() is BM25 (was tf*idf); question words are stopwords; excerpt clips from the hit

## §6.7 / §6.8 answer first; medoid titles; salient gate; one stoplist
- [DONE] walker: answer under the prompt; ONE Evidence expander, collapsed once an answer exists
- [DONE] graph_tools.chunk_salient / chunk_terms / corpus_df (cached): medoid titled by its own terms
- [DONE] graph_tools.salient_gate: min(median-1.4826*MAD, mean-sd) on log BM25 (W14); measured 84% kept
- [DONE] R18 one stoplist (NLTK + extras) at ingest and query; `didn` gone
- [TODO] re-ingest brown-50, brown-50-dual, brown-500-dual under R18; re-pin; commit
- [LATER] tune the salient gate against a probe set (84% kept vs the stated 'upper half and a little more')
- [LATER] keyness prior for register words (got/knew/looked) -- salient_grams R9

## §6.9 latency
- [DONE] search() in Python over a cached per-run index (7.3 s -> 4 ms); one search per walk
- [DONE] community term pool cached per run (9.2 s -> 0.4 s); index + pool pickled by run_id
- [DONE] stoplist.py: NLTK english file read directly (import 11.6 s -> 0.5 s)
- [DONE] reason() one-shot by default (4 calls -> 1); provider sort=throughput; 86 s -> 9 s total

## book-adoption campaign (T18-T22, design 6.17-6.19, reconciled T23 2026-09-03)
- [DONE] named metrics lane (W18-W20): betweenness/PageRank/triangles/per-provenance
  degree, community density+conductance+WCC, PPR beside DWPC in pathways() -- additive,
  local_medoid untouched; 105/105 test_graph_tools.py; surfaced into digest/mirror/walker (T22)
- [DONE] entity resolution v1 (E6-E8): string-similarity + co-occurrence corroboration +
  union-find -> canonical_id, supersede-never-delete; 23/23 test_entities.py
- [DONE] text2cypher serve path (Q1-Q7): read-only Cypher over the mirror, cookbook as
  few-shot bank, retry-on-error (max 3); 13 tests, live example "What did the walk anchor
  on, and where did the anchors sit?" -> Cypher -> 8 rows, 1 attempt
- [DONE] alias-aware search (W21/S19): opt-in expand_aliases (default OFF), additive
  OR-terms at full BM25 weight, byte-identical when no entities table; 11 new tests,
  smoke-scale evidence 905 alias groups, "aboard"->"board" surfaced 5 new chunks
- [OPEN] E6 scale bound: full-vocab resolve on mixed-full-dual is out of Article VII's
  budget -- entity_edges pair enumeration hits 2.47e9 candidate pairs at the default
  MIN_JOINT_CHUNKS=5 floor (1.75e9 at df>=50, 1.45e9 at df>=100, 1.09e9 at df>=200; no
  floor terminates in minutes on this corpus). Open spec question for entities.py, own
  ledger row, not this campaign's.
- [OPEN] alias expansion default stays OFF until full-dual entities are populated --
  flag-ON on mixed-full-dual was vacuous (no entities table, map reads back {}, W21's
  degrade guard makes it a no-op: 18/20 byte-identical, unmeasured-vacuous not
  measured-neutral). Real behavior only demonstrated on mixed-smoke (905 alias groups,
  "aboard"->"board" surfaced 5 new chunks) -- flip the default once mixed-full-dual (or
  its successor) carries a resolved entities table.
