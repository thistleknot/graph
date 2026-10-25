# playbook.md

Spec of record: `community_tally.py` (pipeline table, guards R1-R26, CLOSED).
Decision log: `interpretations.md`. Code and spec disagree -> fix the spec first, then the code.
One task per session. Each task is bounded (<15 min to disposition) and closes before the next opens.

## Current disposition (2026-10-25)

CLASSIFICATION, INTERVALS, and RESUME are closed. Source-backed classifier sanity passed 12/12; the adaptive PPR
confidence sequence passed its 200-repetition all-look test; API restart recovery reused four committed calls
with zero provider replay. Neyman reallocation is promoted by a matched 10-question, 100-seed run on all 152
eval strata: mean absolute error fell 25.8%, measured coverage stayed 100% in both arms, and both used 200 calls.
These semantic labels are closed-world aliases, not corpus truth. The matched ten-question route eval used the same 152
labeled communities and top-10 width: lexical scored 100% hit / 23.85% positive-community recall, summary embeddings
90% / 9.90%, four exemplars 80% / 11.93%, and lexical+embedding RRF 100% / 19.94%. Keep embedding, exemplars, and
RRF off. CORPUS INDEX is complete as QA build 1 for build 1163: 2,242 papers, 207 communities, 1,558,405 term-DF
rows, 20,700 top-100 signatures, and passing exact-lookup/eval audits. SUMMARY, exact lexical API serving, full AGE
Function live execution, and completed WORKLOADS remain open. `answer_kind=lexical_term` and `summary`
currently return HTTP 501 rather than silently falling back.

## Chain
BUILD (partition, signature index, exemplars) -> ROUTE -> TALLY (lexical exact, sampled) -> SUMMARY -> SERVE
The full index follows its index-specific Layer 1 gates; summary and served workloads follow the complete index.

## Tasks (ordered; root first)
1. [x] EVAL SET (root). Build >=20 labeled aggregate questions with ground-truth tallies, mixed lexical and
       semantic predicates, on a deterministic slice of <=2k canonical papers. Group `_methods` records
       with their main paper before sampling and counting. Closed by `global_eval_set.jsonl`: build 1163,
       1,000 papers, 20 questions (10 lexical, 10 closed-world semantic alias predicates); manifest stores
       per-paper labels, source IDs, primary community, and text digest. Main/method IDs are deduplicated.
2. [x] B2 + T1: exact t3 per-community paper DF is stored separately from the top-k signature. Three-community
       synthetic tests confirm ten-term DF identities, main/method grouping, tie-breaking, digit identifiers,
       and exact lookup for a term omitted from every signature. The 1,000-paper build-1163 eval slice matches
       all stored text hashes, primary communities, and ten lexical gold totals; PostgreSQL schema and
       transactional publication were checked in rollback-scoped live tests.
3. [x] T4 on synthetic: the original fixed-sample normal baseline produced 189/200 coverage. Adaptive stopping
       now uses a hypergeometric PPR confidence sequence; 200 repetitions of the actual allocation/stopping loop
       pass the test's >=94% all-look coverage condition. Non-census p=0/1 endpoints retain positive variance,
       census sampling variance is zero, and unsure labels widen only the upper endpoint. No LLM involved.
4. [x] ROUTE: lexical-only routing on the eval set, comparing signature widths 100 and 500. Both widths
       hit at least one positive community for all ten lexical questions; strict top-5 covered only 5.66%
       of positive communities. Tie expansion selected all 152 communities for every 100-term query and
       nine of ten 500-term queries. Keep width 100; lexical routing is not promoted as a pruning path.
5. [x] T2/T3: deterministic sample from eval community 5 (85 papers), S01 contrastive-learning predicate,
       seed 20261025, 12 papers, one OpenRouter request each. With a 60,000-character cap: 7 no, 5 unsure,
       0 yes; the 5 truncated papers' raw `no` labels were overridden to `unsure`. One alias-positive paper
       was truncated and remained unsure, so this pilot validates the response contract, not semantic recall.
       A first 12,000-character trial also returned `no` on truncated inputs; it motivated the deterministic guard.
6. [x] EVAL CHECKPOINT: SQLite key includes source build, tokenizer version, and artifact SHA-256. The live
       1,000-paper verifier built the cache, then reused it on a second pass while rechecking source IDs,
       text hashes, primary communities, all ten lexical totals, and both route variants; the two-pass gate
       passed in 308 seconds. Cache size was 157,159,424 bytes; it stores counts and metadata, not source text.
7. [x] TALLY LOOP: synthetic three-stratum gates verify all-strata pilot, uniform draws without replacement,
       budget limits, stop behavior, census, cache reuse, truncation-to-unsure, and auditable yes/no/unsure estimates.
8. [x] RESUME/API: SQLite jobs pin source/population/model/prompt/seed/draw order/budgets, commit each response,
       expose submit/status/resume, and mark interrupted calls uncertain. A restart test resumes from four committed
       responses and proves zero provider replays; repository tests reject uncertain calls and budget changes.
       The AGE Function polls the job endpoint in mocked tests; hosted provider-backed execution is still open.
9. [x] Neyman reallocation comparison: 10 semantic questions, 100 matched seeds, actual 152 eval strata,
       200-call budget, equal coverage and call count; average MAE 22.639 -> 16.804. Promoted on.
    [x] Remaining route flags: matched on ten semantic questions at the same 152 candidate communities and top-10
       width. Each optional arm had lower positive-community recall than lexical; keep all three off.
10. [x] CORPUS INDEX: QA build 1 is complete for build 1163; full paper, community, term, source-build, tokenizer,
    and marker invariants passed. The index precedes summary because tally-backed claims require it.
11. [ ] SUMMARY: run global_summary on 5 questions against the complete index; assert MMR, citation provenance,
    numeric tally replacement (R19), and reported coverage (R20).
12. [ ] SERVE/WORKLOADS: implement exact lexical dispatch, complete a bounded AGE Function job through rendering,
    and verify real served application workloads.

## Primitives named in the pseudocode but NOT defined (implement or bind to a library)
document_frequency, keyness_scores (weighted log-odds with informative prior preferred), global_df,
top_k_by_value, kmeans_medoids, mmr, rrf, include_ties, lookup_signature, df_lookup, doc_ids, n_docs,
top_relevant_docs, llm_summarize, llm_combine, extract_numeric_claims, parse, normalize_terms
(must not alter digit-bearing identifiers, R4).

## Open items
- Measured `top_k_terms=100` versus 500: strict top-k recall did not improve at 500, and tie expansion still selected nearly every community; retain 100 as the lower-storage choice. Four exemplars were evaluated and not promoted; alternative exemplar counts, `pilot_n`, `target_halfwidth`, and `summary_*` remain unmeasured.
- `tie_margin=0.1` has high tie frequency (100% at 100 signature terms); it confirms lexical route cannot prune this eval set safely.
- Predicate homogeneity inside communities is unmeasured; high per-stratum variance means no convergence (R15).
- Signature and exact DF ownership is PostgreSQL. The runnable B2/T1 implementation uses `entity_derive`
  tokenizer t3 because it is the tokenizer recorded by the eval artifact; English `tsvector` is not equivalent.
  This is an ownership choice, not a measured speed comparison with Neo4j.

## Sources behind CLOSED and design choices
- AGGBench (aggregation over unstructured text): https://arxiv.org/html/2602.01355v1
- GlobalQA (global RAG benchmark): https://arxiv.org/html/2510.26205v2
- OLLA (stratified sampling, Neyman allocation, LLM aggregates): https://arxiv.org/html/2603.08443
- InQuest (proxy-stratified aggregation, defensive sampling): https://www.vldb.org/pvldb/vol16/p2897-russo.pdf
- GraphRAG paper: https://arxiv.org/abs/2404.16130
- DRIFT search: https://www.microsoft.com/en-us/research/?p=1098888
- BenchmarkQED: https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/
- ALT landmarks: https://arxiv.org/pdf/1603.01607
- HyperMinHash: https://arxiv.org/pdf/1710.08436
- pgfaceting / roaring bitmaps: https://github.com/cybertec-postgresql/pgfaceting
- Postgres index-only count for bitmap indexes (2017 thread): https://www.postgresql.org/message-id/239a8955-c0fc-f506-026d-c837e86c827b%40postgrespro.ru
- Postgres ts_stat: https://www.postgresql.org/docs/current/textsearch-features.html
