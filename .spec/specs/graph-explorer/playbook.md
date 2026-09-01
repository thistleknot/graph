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
