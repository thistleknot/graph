Chunk graph build
=================

Scope
-----

The chunk graph is the general build over any corpus text. A run keeps its
chunks, the connections between them, their communities and their embeddings
under one label, and it supersedes the earlier live run under that label. No
language model takes part in the build. The source is ``src/chunkgraph.py`` and
``src/pg_store.py``, with the corpus drivers ``src/ingest_brown.py`` and
``src/ingest_mixed.py``.

Sphinx-Needs components and workflow
------------------------------------

.. comp:: Corpus source
   :id: C_CG_CORPUS
   :feeds: C_CG_CHUNKER

   Text documents with identifiers: a sample of the Brown corpus, a mixed set of
   sources, or the operator's own documents.

.. comp:: Chunker
   :id: C_CG_CHUNKER
   :feeds: C_CG_SPARSE, C_CG_DENSE

   Splits each document at paragraph boundaries, never inside a word, and keeps
   the document identifier on every chunk.

.. comp:: Wording similarity
   :id: C_CG_SPARSE
   :produces: C_CG_EDGES

   Scores every chunk pair by shared wording and keeps each chunk's nearest
   neighbours whatever their significance.

.. comp:: Meaning similarity
   :id: C_CG_DENSE
   :produces: C_CG_EDGES

   Scores every chunk pair by meaning when a local embedding model is configured.
   Without one the run continues with wording only.

.. comp:: Connection builder
   :id: C_CG_EDGES
   :produces: C_CG_COMMUNITY

   Keeps the significant pairs, merges both kinds into one graph, and records
   which kind of similarity found each connection.

.. comp:: Community builder
   :id: C_CG_COMMUNITY
   :persists: C_CG_STORE

   Groups the chunks into communities and names each by its distinctive terms
   and a representative chunk.

.. comp:: Graph store
   :id: C_CG_STORE

   Saves the run, its chunks, connections, communities and embeddings in one
   transaction and supersedes the earlier live run under the label.

Build handoffs
--------------

.. needflow:: Chunk graph build handoffs
   :filter: id.startswith("C_CG_")
   :link_types: feeds, produces, persists
   :show_link_names:
   :direction: right
   :alt: Chunk graph build from corpus source through chunker, similarity arms, connection builder, community builder and graph store
