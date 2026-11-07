Section map enrichment and review
=================================

Scope
-----

After a section build is live, three steps add what a reader needs to recognise
a community: the entities its sections mention, the titles of its papers, and a
drafted summary. A fourth step renders the labelled map. The build itself is
described in ``age-architecture``; this page covers the steps after it. The
sources are ``src/section_store.py`` (entity matching), ``src/arxiv_titles.py``,
``src/summarize_clusters.py`` and ``src/section_render.py``.

Sphinx-Needs components and workflow
------------------------------------

.. comp:: Frozen entity inventory
   :id: C_SM_INVENTORY
   :feeds: C_SM_ENTITIES

   The entity inventory derived from the corpus when the chunk build was made.

.. comp:: Entity matcher
   :id: C_SM_ENTITIES
   :reads: C_SECTION_DB
   :persists: C_SECTION_DB

   Records, for each section of the live build, which entities it mentions and
   how often, and keeps the entity links whose entities co-occur more often than
   chance.

.. comp:: arXiv metadata service
   :id: C_SM_ARXIV

   External service that returns the title of a paper when no local listing has it.

.. comp:: Title resolver
   :id: C_SM_TITLES
   :reads: C_SM_ARXIV
   :persists: C_SECTION_DB

   Fills in paper titles from the local listings first, then from the arXiv
   metadata service, and stores each title once.

.. comp:: Community summary drafter
   :id: C_SM_SUMMARY
   :calls_model: C_MODEL
   :persists: C_SECTION_DB

   Drafts one summary per community from its exemplar sections, in full, and
   stores only complete replies. Each stored summary is marked a machine draft.

.. comp:: Labelled map renderer
   :id: C_SM_RENDER
   :reads: C_SECTION_DB

   Draws one card per community, largest first, with its terms, entities and
   exemplar sections, and writes the markdown listing.

Enrichment handoffs
-------------------

.. needflow:: Section map enrichment handoffs
   :filter: id.startswith("C_SM_")
   :link_types: feeds, reads, persists, calls_model
   :show_link_names:
   :direction: right
   :alt: Section map enrichment: entity matcher, title resolver with the arXiv metadata service, community summary drafter calling the model, and the labelled map renderer
