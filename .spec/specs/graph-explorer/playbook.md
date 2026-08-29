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
