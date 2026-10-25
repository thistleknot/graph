# Global aggregate QA design

## Current approved delivery contract

The 2026-10-25 approved plan governs the remaining delivery: source-backed classifier quality, finite-population sequential intervals, durable resumable jobs, measured optional flags, cited MMR summaries with trusted numeric rendering, existing API/AGE Function delivery, then the complete production index.

For each binary endpoint (yes and yes-or-unsure), use the hypergeometric prior-posterior-ratio confidence sequence with a uniform beta-binomial working prior (a=b=1). At observed n,s and community population N, candidate success totals K in [s,N-n+s] survive when prior(K)/posterior(K) < 1/alpha_endpoint. Allocate alpha_endpoint=.05/(2*C), intersect sets across looks, and sum community lower-yes and upper-possible bounds. Keep the old normal interval only as a fixed-size comparison diagnostic. Sampling intervals are conditional on classifier outcomes, not guarantees of model semantic accuracy.

Current implementation status: `AggregateJobs` persists manifests, call reservations/responses, progress, and results in the existing tally SQLite file; startup marks interrupted calls uncertain and running jobs incomplete. `/aggregate` submits, GET reads, and POST resume preserves budgets. `/agent` defaults to retrieval; `answer_kind=auto` dispatches explicit count-question patterns and `semantic` submits the supplied predicate. `lexical_term` and `summary` remain HTTP 501 until their exact-index and cited-summary paths exist. The AGE Function polls aggregate jobs; a real provider-backed run remains unverified.

Checkpoint one validated classification immediately, not at batch completion. Durable jobs pin population, build, model/prompt, seed, draw order, budgets and observations. Cache hits do not increase an evaluation's frozen sample cap; production job progress records sample count, cached labels and actual attempts separately. Resume preserves this manifest and never implicitly extends total budgets.

API additions: `/agent` answer_kind defaults to retrieval and accepts retrieval/auto/lexical_term/semantic/summary. Auto routes explicit count patterns; summary will return a structured coverage/citation result and lexical_term remains HTTP 501 pending exact-index service. POST /aggregate submits a build-pinned job, GET /aggregate/{job_id} returns progress/results, POST /aggregate/{job_id}/resume continues persisted work. The AGE Function opts into auto and polls/resumes jobs; legacy Function is unchanged.

### Summary mapping and reduction

`SummaryCitation [value]` carries canonical paper ID and source document IDs. `SummaryPart [value]` carries community ID, population, selected-paper count, narrative, and citations. `NumericClaim [value]` carries draft claim text, a tally predicate, and the exact/semantic tally result. `SummaryResult [value]` carries rendered text, selected-community parts, covered and total population, coverage fraction, uncovered community IDs, and tally-backed claims.

`global_summary(conn, source_build_id, question, config, embed, tally_claim) -> SummaryResult` is a stateless service function in `global_aggregate.py`; it owns no durable state. It requires the source build's section vectors, community terms, canonical-paper identity, configured map/reduce provider, and a tally callback. Map/reduce uses the existing `interpret._via_openrouter` JSON-mode transport; no second provider client is introduced.

Behavioral order:

1. Normalize query terms with t3 and rank communities using the measured lexical control; keep route ranking separate from the corpus population.
2. For each selected community, retrieve a bounded relevance pool of canonical papers, choose up to `summary_sample_n` with MMR over query relevance and pairwise redundancy, and preserve the selected paper IDs.
3. Map each community evidence set to JSON containing narrative and structured numeric-claim requests; accept only citation IDs from that evidence set.
4. Reduce validated community narratives and citations into one answer; retain per-community population and read counts for coverage.
5. For every structured numeric claim, call `tally_claim`; replace the draft claim with the returned exact count or estimate and interval. If tallying fails, fail the summary instead of returning the draft number.
6. Compute coverage as the sum of selected communities' canonical-paper populations divided by the total canonical-paper population; return uncovered community IDs.

Each map response and the reduce response must parse as the declared JSON object. Invalid JSON, unknown citations, a missing tally result, or a build/index mismatch fails the summary. The five-question Layer 1 battery uses controlled tally callbacks to validate replacement and citation mechanics; a provider-backed API run remains a separate SERVE gate. Defaults for route width, candidate pool, MMR sample count, and lambda remain unmeasured until the five-question battery records coverage and output evidence.

### AggregateJobs [repository]

No incumbent aggregate job repository exists. `aggregate_jobs.py` is the persistence/scheduling seam; numerical and QA behavior remain in `global_aggregate.py`. The repository owns SQLite jobs, call reservations, durable responses, worker leases and bounded execution. `create(request, manifest) -> job_id` pins immutable inputs; `get(job_id) -> snapshot` reports state; `call(job_id, key, callback) -> response` reserves an attempt before execution and commits the response immediately; `start(job_id, runner)` claims a lease; `resume(job_id, runner, additional_calls=0)` preserves inputs and changes only an explicitly increased budget. Recovery requeues expired leases and reuses every completed call. Unconfirmed in-flight calls are explicit failures, not silent free retries. Job boundaries record exceptions and expose failures; they never substitute successful results.

Summary generation emits structured qualitative statements with citations and typed quantitative requests; only exact/estimated tally results supply quantitative text. Each part records community population and papers read; output reports covered population and uncovered communities. Unsupported quantitative language fails validation.

## Pipeline

`section build -> canonical-paper stream -> deterministic primary community -> t3 token sets -> full per-community DF -> separate keyness signature -> exact lexical tally`

The first delivery implements B2/T1 and the T4 estimator gate. Routing, LLM sampling, flags, and summaries remain blocked until their preceding gates pass.

## Ownership and boundaries

- `sect_node` and `sect_build` remain authoritative source rows.
- `global_aggregate.py` owns deterministic canonical grouping, t3 token-set conversion, DF construction, keyness ranking, and exact lexical lookup.
- PostgreSQL persists a versioned complete-index marker, paper/community assignments, community sizes, every observed term DF, and a separately truncated routing signature.
- Tally reads only the full DF table; signature rows never determine whether a term exists.
- No JSONB field is used for graph traversal or exact term counts.
- The implementation exposes pure transformations independently from PostgreSQL persistence so synthetic checks do not require the live corpus.

## Canonical source query

For one pinned section build, group `doc_id` after removing only a terminal `_methods` suffix. Count its section rows by source community, choose the highest count, and break ties by ascending community ID. Stream grouped records in stable `(canonical_id, source_doc_id, section_idx)` order. Tokenize each section with `entity_derive.tokenize_text` t3; union folded tokens per canonical paper so repeated terms count once per paper. Preserve t3 segment handling and do not concatenate sections across boundaries.

## Persistence and publication

- Allocate a QA build record with source build ID, tokenizer version, expected paper count, and `complete=false`.
- In one transaction, write the canonical paper assignments, community counts, full term DF rows, and top-k positive keyness signature rows.
- Verify each paper has one community, community sizes sum to the indexed paper count, and each term's community DF sum equals its direct corpus DF.
- Mark the QA build complete only after all checks pass; rollback leaves no complete marker or partial rows.
- Exact tally first resolves a complete marker matching both source build and tokenizer version, then sums the requested token across every indexed community. Missing term rows count as zero only after marker validation.

## Signature score

Use a two-population weighted log-odds z score over canonical-paper presence counts: the selected community versus all other papers, with an informative prior proportional to global document frequency. Retain only positive scores for routing and truncate after scoring. This score affects routing only; it does not alter exact DF or tally results.

## Stratified estimator

For each community, compute lower prevalence `yes/n` and upper prevalence `(yes+unsure)/n`; weight each by its population `N`. Expand both endpoint totals by `z * sqrt(sum(N^2 * p_smooth * (1-p_smooth) / n * n/(n-1) * (1-n/N)))`. The `n/(n-1)` term is one for `n=1`; a census stratum (`n=N`) has zero sampling variance. `p_smooth` adds 0.5 successes and failures to each endpoint to avoid zero variance for non-census all-zero/all-one observations. Report the midpoint of the unexpanded endpoints as the point estimate and expose per-community `p_yes`, `p_possible`, and unsure counts.

## Semantic tally loop

The runnable tally receives a complete canonical-paper-to-primary-community population, one predicate, a text loader, and a one-paper classifier. It draws a deterministic uniform pilot in every requested stratum before estimating. Each additional round allocates at most a bounded batch from the remaining call budget; measured Neyman reallocation is the default and uses population times the larger smoothed endpoint standard deviation with an evenly spread defensive floor. A matched 100-repetition, ten-question eval comparison on all 152 eval strata reduced average absolute error from 22.639 to 16.804 papers at equal 100% observed coverage and equal call count (200). The eval gold is closed-world alias truth and does not establish unrestricted semantic accuracy. Each selected ID is removed from its stratum's remaining pool. After each round it stops on the relative interval half-width target, total budget exhaustion, or census of every stratum. Budget exhaustion returns `converged=false`; census reports zero sampling variance. Malformed classifier responses and loader omissions fail the tally rather than being counted as negative.

Validated labels are checkpointed in SQLite by source build, predicate digest, exact classifier-input digest, truncation state, and caller-supplied classifier version. Cache hits reuse the same outcome without spending a model-call attempt; raw text is never persisted.

## Lexical routing

For each community, sum positive keyness for query tokens present in that community's top-k signature. Rank by descending score and ascending community ID. When the relative score gap at the cutoff is below `tie_margin`, include all following communities whose scores remain within the margin. Report both the strict top-k result and tie-expanded result. Routing is ranking metadata only; a global tally still reads every community.

The route evaluation compares signature widths 100 and 500 against the same exact community DF and labeled lexical questions. The wider signature is promoted only if it improves positive-community recall without expanding every query to every community; otherwise keep the lower storage cost and record the route limitation.

## Predicate classification pilot

Sample paper IDs uniformly without replacement inside one selected community using a fixed local seed. Retrieve section text in stable source order and preserve section headers as boundaries. Cap prompt text at 60,000 characters and mark truncation explicitly. Send one fixed yes/no/unsure classification prompt per paper through the existing OpenRouter JSON-mode transport with retries disabled so request counts are auditable. Accept only JSON with one `label` property in the three-label set; malformed output aborts the pilot. A truncated input without positive evidence must be labeled unsure, not no.

## Validation order

1. Synthetic three-community fixture: main/method grouping, modal assignment and tie, t3 digit-bearing token, duplicate mentions, missing token, top-k exclusion, and build/tokenizer mismatch.
2. Ten-term identity: sum community DF and compare with direct canonical-paper count.
3. Eval slice: rebuild the deterministic 1,000-paper slice and match all ten lexical manifest counts.
4. T4: run 200 seeded SRS repetitions over three known finite populations; require coverage within six percentage points of 95%, positive non-census endpoint variance, and unsure-only upper-bound contribution.
5. ROUTE: measure top-k hit rate, positive-community recall, and tie expansion on lexical eval questions.
6. After the index-specific Layer 1 gates pass (schema, source identity, canonical grouping, t3 parity, synthetic invariants, lexical gold, and route-flag disposition), create the full build-1163 index and report row counts and duration. This index precedes SUMMARY and SERVE because exact lexical tallies and tally-backed summary claims require a matching complete marker.
7. Run the five-question summary battery on the complete index; verify MMR selection, source citations, population coverage, and tally replacement.
8. Run exact lexical dispatch and a real provider-backed AGE Function job through rendered completion; then record completed application workloads.

## Eval checkpoint

The 1,000-paper t3 pass is a measured 3.5-minute operation, so persist its verified exact DF and paper hashes in `.tmp/global_aggregate_eval.sqlite`. Key entries by source build ID, tokenizer version, and SHA-256 of the eval artifact. A cache hit must match the manifest paper IDs, per-paper text hashes, and primary communities before route variants run; current source identities and hashes are rechecked without tokenization. Cache data contains counts and IDs, not source text. A key mismatch is a cache miss; a malformed matching cache is an error, not a silent fallback.

Each gate is bounded to 15 minutes. The full build is production work, not a discovery test.

## Failure behavior

- Missing live source build, incomplete index marker, tokenization mismatch, non-unique paper assignment, or failed count invariant aborts the build or exact tally.
- No fallback to `tsvector`, truncated signatures, JSONB, or sampled estimates is allowed for a single-token exact tally.
