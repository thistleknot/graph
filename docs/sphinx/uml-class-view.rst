UML class view of the entity model
==================================

Class diagrams generated from ``docs/entity_model/*.md`` by
``tools/entity_to_uml.py``. Primary and foreign keys are stereotyped. Each
association is a foreign key from the entity model's relationship diagram.

This view documents the entity model. It does not yet inform it: the
conceptual class model that the sw-design flow puts before the entity model
has not been authored from the requirements.

Chunk graph
-----------

.. needuml:: Chunk graph classes

   @startuml
   class GRAPH_RUN {
     run_id : String <<PK>>
     label : String
     ingested_at : DateTime
     n_chunks : Integer
     embed_dim : Integer
     params : String
     diagnostics : String
     superseded_at : DateTime
   }
   class NODE {
     run_id : String <<PK,FK>>
     ord : Integer <<PK>>
     doc_id : String
     chunk_hash : String
     body : String
     attrs : String
   }
   class EDGE {
     run_id : String <<PK,FK>>
     src : Integer <<PK,FK>>
     dst : Integer <<PK,FK>>
     provenance : String
     strength : Decimal
     sim_sparse : Decimal
     sim_dense : Decimal
     src_doc : String
     dst_doc : String
     attrs : String
     valid_from : DateTime
     valid_to : DateTime
     ingested_at : DateTime
   }
   class COMMUNITY {
     run_id : String <<PK,FK>>
     cid : Integer <<PK>>
     members : String
     size : Integer
     keywords : String
     medoid : Integer <<FK>>
     medoid_text : String
     attrs : String
   }
   class NODE_EMBEDDING {
     run_id : String <<PK,FK>>
     ord : Integer <<PK,FK>>
     embedding : String
   }
   GRAPH_RUN "1" -- "0..*" NODE : holds
   GRAPH_RUN "1" -- "0..*" COMMUNITY : holds
   NODE "1" -- "0..*" EDGE : is the source of
   NODE "1" -- "0..*" EDGE : is the destination of
   NODE "1" -- "0..*" COMMUNITY : is the medoid of
   NODE "1" -- "0..1" NODE_EMBEDDING : has
   @enduml

Lexical index
-------------

.. needuml:: Lexical index classes

   @startuml
   class LEX_BUILD {
     build_id : Long <<PK>>
     label : String
     created : DateTime
     live : Boolean
     ready : Boolean
     n_chunks : Integer
     params : String
   }
   class LEX_VOCAB {
     label : String <<PK>>
     col : Integer <<PK>>
     piece : String
     idf : Decimal
   }
   class LEX_CHUNK_META {
     label : String <<PK>>
     build_id : Long <<PK>>
     ord : Integer <<PK>>
     record_id : Long
     doc_id : String
     section_idx : Integer
     chunk_idx : Integer
     section_title : String
     is_reference : Boolean
     nnz : Integer
     truncated : Boolean
     text : String
   }
   class ARXIV_SECT_BPE {
     ord : Integer <<PK>>
     doc_id : String
     source : String
     nnz : Integer
     truncated : Boolean
     vec : String
   }
   class LEX_BUILD_VOCAB {
     build_id : Long <<PK,FK>>
     col : Integer <<PK>>
     piece : String
     idf : Decimal
   }
   class LEX_COMMUNITY {
     build_id : Long <<PK,FK>>
     cid : Integer <<PK>>
     size : Integer
     centroid : String
   }
   class LEX_ASSIGN {
     build_id : Long <<PK,FK>>
     ord : Integer <<PK>>
     cid : Integer
     how : String
     x : Decimal
     y : Decimal
   }
   class LEX_EXEMPLAR {
     build_id : Long <<PK,FK>>
     cid : Integer <<PK>>
     rank : Integer <<PK>>
     role : String
     z : Decimal
     ord : Integer
   }
   class LEX_SUMMARY {
     build_id : Long <<PK,FK>>
     cid : Integer <<PK>>
     status : String
     title : String
     summary : String
     model : String
     provider : String
     prompt_tokens : Integer
     completion_tokens : Integer
     cost : Decimal
     chunk_keys : String
   }
   class LEX_ENTITY {
     build_id : Long <<PK,FK>>
     ent_id : Integer <<PK>>
     fold : String
     surface : String
     n : Integer
     score : Decimal
     df_papers : Integer
     df_chunks : Integer
     tf : Integer
     canonical : Integer
   }
   class LEX_MENTION {
     build_id : Long <<PK,FK>>
     ord : Integer <<PK>>
     ent_id : Integer <<PK>>
     cnt : Integer
   }
   LEX_BUILD "1" -- "0..*" LEX_BUILD_VOCAB : has
   LEX_BUILD "1" -- "0..*" LEX_COMMUNITY : has
   LEX_BUILD "1" -- "0..*" LEX_ASSIGN : places
   LEX_BUILD "1" -- "0..*" LEX_EXEMPLAR : has
   LEX_BUILD "1" -- "0..*" LEX_SUMMARY : has
   LEX_BUILD "1" -- "0..*" LEX_ENTITY : has
   LEX_BUILD "1" -- "0..*" LEX_MENTION : has
   @enduml

Section map
-----------

.. needuml:: Section map classes

   @startuml
   class SECT_BUILD {
     build_id : Long <<PK>>
     tag : String
     created_at : DateTime
     live : Boolean
     n_nodes : Integer
     n_edges : Integer
     n_communities : Integer
     params : String
   }
   class SECT_NODE {
     build_id : Long <<PK,FK>>
     ord : Integer <<PK>>
     record_id : Long
     doc_id : String
     section_idx : Integer
     section_title : String
     community : Integer
     x : Decimal
     y : Decimal
     genre_frac : Decimal
     text : String
     tsv : String
   }
   class SECT_EMBEDDING {
     record_id : Long <<PK,FK>>
     emb : String
     emb_b : String
   }
   class SECT_EDGE {
     build_id : Long <<PK,FK>>
     a : Integer <<PK>>
     b : Integer <<PK>>
     weight : Decimal
     dense_sim : Decimal
     dense_z : Decimal
     sparse_sim : Decimal
     sparse_z : Decimal
     sig : Integer
     backbone : Integer
   }
   class SECT_COMMUNITY {
     build_id : Long <<PK,FK>>
     cid : Integer <<PK>>
     size : Integer
     genre_score : Decimal
     genre_flag : Boolean
     title : String
     summary : String
     status : String
     terms : String
     emb : String
     entities : String
   }
   class SECT_ENTITY {
     build_id : Long <<PK,FK>>
     ent_id : Integer <<PK>>
     surface : String
     n : Integer
     df : Integer
   }
   class SECT_MENTION {
     build_id : Long <<PK,FK>>
     ord : Integer <<PK>>
     ent_id : Integer <<PK>>
     cnt : Integer
   }
   class SECT_ENTITY_EDGE {
     build_id : Long <<PK,FK>>
     a : Integer <<PK>>
     b : Integer <<PK>>
     kind : String <<PK>>
     n_shared : Integer
     npmi : Decimal
   }
   class SECT_SERVING_RELEASE {
     tag : String <<PK>>
     graph_build_id : Long
     bm25_label : String
     bm25_build_id : Long
     published_at : DateTime
   }
   class PAPER_TITLE {
     arxiv_id : String <<PK>>
     title : String
     thesis : String
     source : String
     loaded_at : DateTime
   }
   SECT_BUILD "1" -- "0..*" SECT_NODE : holds
   SECT_BUILD "1" -- "0..*" SECT_EDGE : holds
   SECT_BUILD "1" -- "0..*" SECT_COMMUNITY : holds
   SECT_BUILD "1" -- "0..*" SECT_ENTITY : holds
   SECT_BUILD "1" -- "0..*" SECT_MENTION : holds
   SECT_BUILD "1" -- "0..*" SECT_ENTITY_EDGE : holds
   SECT_NODE "1" -- "0..1" SECT_EMBEDDING : has
   @enduml

Global aggregate
----------------

.. needuml:: Global aggregate classes

   @startuml
   class GLOBAL_QA_BUILD {
     qa_build_id : Long <<PK>>
     source_build_id : Long
     tokenizer_version : String
     signature_top_k : Integer
     n_docs : Integer
     complete : Boolean
     created_at : DateTime
   }
   class GLOBAL_QA_DOC {
     qa_build_id : Long <<PK,FK>>
     paper_id : String <<PK>>
     community_id : Integer
     text_sha256 : String
   }
   class GLOBAL_QA_COMMUNITY {
     qa_build_id : Long <<PK,FK>>
     community_id : Integer <<PK>>
     n_docs : Integer
   }
   class GLOBAL_QA_TERM_DF {
     qa_build_id : Long <<PK,FK>>
     community_id : Integer <<PK>>
     term : String <<PK>>
     df : Integer
   }
   class GLOBAL_QA_SIGNATURE {
     qa_build_id : Long <<PK,FK>>
     community_id : Integer <<PK>>
     term : String <<PK>>
     keyness : Decimal
     df : Integer
   }
   class AGGREGATE_JOB {
     job_id : String <<PK>>
     status : String
     manifest_json : String
     progress_json : String
     result_json : String
     error : String
     created_at : DateTime
     updated_at : DateTime
   }
   class AGGREGATE_JOB_CALL {
     job_id : String <<PK,FK>>
     call_key : String <<PK>>
     status : String
     response_json : String
     error : String
     started_at : DateTime
     completed_at : DateTime
   }
   GLOBAL_QA_BUILD "1" -- "0..*" GLOBAL_QA_DOC : has
   GLOBAL_QA_BUILD "1" -- "0..*" GLOBAL_QA_COMMUNITY : has
   GLOBAL_QA_BUILD "1" -- "0..*" GLOBAL_QA_TERM_DF : has
   GLOBAL_QA_BUILD "1" -- "0..*" GLOBAL_QA_SIGNATURE : has
   AGGREGATE_JOB "1" -- "0..*" AGGREGATE_JOB_CALL : makes
   @enduml

