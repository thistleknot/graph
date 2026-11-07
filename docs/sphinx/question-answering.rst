Question answering pipeline
===========================

Scope
-----

A question from a chat, or a request from a calling program, is answered from
the live section build and its keyword index. The retrieval service ranks
candidate sections by meaning and by keyword, weights them by community, and
expands them through the map. It then keeps only the communities whose sections
pass the relevance gate, runs a bounded research loop, and writes the answer
from the retained evidence. The filter shows the evidence before the model's
commentary. The sources are ``src/section_graphrag.py``, ``src/arxiv_rag_api.py``
and ``src/owui_filter.py``.

Sphinx-Needs components and workflow
------------------------------------

.. comp:: Candidate retrieval
   :id: C_QA_RETRIEVE
   :reads: C_SECTION_DB
   :queries: C_BM25_DB

   Ranks sections by meaning and by keyword, combines the two rankings, weights
   candidates by the chosen community terms, and expands the strongest sections
   through their neighbours.

.. comp:: Relevance gate
   :id: C_QA_GATE
   :calls_model: C_MODEL

   Scores candidates against the question and keeps the communities that hold a
   passing section. A question with no passing section is rejected.

.. comp:: Research loop
   :id: C_QA_AGENT
   :reads: C_SECTION_DB
   :calls_model: C_MODEL

   Lets a model read sections, make at most three further searches, or stop.
   Each search is kept apart from the earlier ones.

.. comp:: Answer synthesis
   :id: C_QA_SYNTH
   :calls_model: C_MODEL
   :returns_response: C_OWUI_FILTER

   Writes the final answer from the retained evidence, tagging each claim with
   its community. A truncated reply is retried once with another provider and
   otherwise fails.

Request handoffs
----------------

.. needflow:: Question answering handoffs
   :filter: id.startswith("C_QA_")
   :link_types: reads, queries, calls_model, returns_response
   :show_link_names:
   :direction: right
   :alt: Question answering from candidate retrieval through the relevance gate and research loop to answer synthesis returned to the filter
