ChunkGraph architecture
=======================

This page documents the arXiv section build and AGE-backed Open WebUI retrieval
paths from the current source contracts. It uses Sphinx-Needs component records
for the directed workflow and Graphviz class-style boxes for component
responsibilities and interfaces.

Build with ``python -m pip install -r docs/sphinx/requirements.txt`` and a
Graphviz ``dot`` executable, then run
``python -m sphinx -W -b html docs/sphinx docs/diagrams`` from the repository
root. The rendered entry page is ``docs/diagrams/age-architecture.html``.

.. toctree::
   :maxdepth: 2

   requirements
   uml-class-view
   chunk-graph
   section-map-enrichment
   lexical-index
   question-answering
   global-aggregate
   age-architecture
