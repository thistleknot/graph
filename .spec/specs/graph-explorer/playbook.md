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
