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
