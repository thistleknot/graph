Lexical index and corpus updates
================================

Scope
-----

The lexical index stores the keyword vectors that the retrieval service ranks
chunks with. A full build fixes the vocabulary and term weights for its
generation, and an incremental cycle adds newly extracted papers under those
frozen settings, so each added chunk equals the chunk a full build would give.
The sources are ``src/ingest_arxiv_sparsevec.py``, ``src/sparsevec_store.py``
and ``src/arxiv_graph_service.py``. The full build is the BM25 builder and
database described in ``age-architecture``.

Sphinx-Needs components and workflow
------------------------------------

.. comp:: Update cycle
   :id: C_LX_CYCLE
   :feeds: C_LX_INCREMENT

   Starts a cycle every thirty minutes at a fixed minute, and skips the cycle
   when the previous one is still running, its input is held, or the graphics
   memory is more than half used.

.. comp:: Incremental ingest
   :id: C_LX_INCREMENT
   :reads: C_SECTION_DB
   :persists: C_BM25_DB
   :requests: C_BM25_BUILDER

   Finds papers not yet indexed, chunks them with the frozen settings, computes
   their vectors, places each paper in the nearest community, and stores the
   results together. Requests a full build once the added chunks reach the
   rebuild fraction of the build's size.

.. comp:: Serving pair publisher
   :id: C_LX_PUBLISH
   :produces: C_RELEASE

   After a full build is validated, publishes the graph generation and the
   keyword generation together as the pair the retrieval service serves.

Update handoffs
---------------

.. needflow:: Lexical index update handoffs
   :filter: id.startswith("C_LX_")
   :link_types: feeds, reads, persists, requests, produces
   :show_link_names:
   :direction: right
   :alt: Lexical index update cycle, incremental ingest against the section store and keyword database, requesting the full keyword build, and publishing the served pair
