# Entity Model

## Modules

| Module | Description | Entities | Depends on |
|---|---|---|---|
| [Chunk Graph](entity_model/chunk-graph.md) | Chunk graphs built from a corpus: runs, chunks, connections, communities and chunk embeddings | GRAPH_RUN, NODE, EDGE, COMMUNITY, NODE_EMBEDDING | - |
| [Lexical Index](entity_model/lexical-index.md) | Keyword index of the corpus chunks: builds, vocabulary, chunk records, communities, placements, summaries and entity mentions | LEX_BUILD, LEX_BUILD_VOCAB, LEX_VOCAB, LEX_CHUNK_META, ARXIV_SECT_BPE, LEX_COMMUNITY, LEX_ASSIGN, LEX_EXEMPLAR, LEX_SUMMARY, LEX_ENTITY, LEX_MENTION | - |
| [Section Map](entity_model/section-map.md) | Section builds of the papers: sections, similarity connections, communities, entities, titles and the served release | SECT_BUILD, SECT_NODE, SECT_EMBEDDING, SECT_EDGE, SECT_COMMUNITY, SECT_ENTITY, SECT_MENTION, SECT_ENTITY_EDGE, SECT_SERVING_RELEASE, PAPER_TITLE | - |
| [Question Answering](entity_model/question-answering.md) | Answers to questions, built from the stored indexes; owns no tables | none | - |
| [Global Aggregate](entity_model/global-aggregate.md) | Per-community term counts for corpus-wide questions, and the durable jobs that classify sampled papers | GLOBAL_QA_BUILD, GLOBAL_QA_DOC, GLOBAL_QA_COMMUNITY, GLOBAL_QA_TERM_DF, GLOBAL_QA_SIGNATURE, AGGREGATE_JOB, AGGREGATE_JOB_CALL | - |
