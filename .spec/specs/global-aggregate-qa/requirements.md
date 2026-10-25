---
description: 'Exact lexical and sampled semantic paths for global aggregate QA'
import: []
---

***definitions***

- :CanonicalPaper: is one paper formed by grouping the main source record with its terminal `_methods` record, when present.
- :PrimaryCommunity: is the mode of section community across a paper's grouped source records, with ties assigned to the lowest community ID.
- :FullTermDF: is the exact count of canonical papers in one primary community containing a normalized token.
- :RoutingSignature: is a separately stored, optionally truncated ranking of community-associated terms.
- :CompleteIndex: is a marker binding persisted counts to one source build and tokenizer version.
- :StratifiedEstimate: is a point estimate and conservative interval over yes/no/unsure labels, weighted by community population.
- :RouteResult: is a lexical ranking of communities with any near-cutoff ties included.
- :AggregateJob: is a durable, source-build-pinned semantic tally with immutable sampling inputs and budget.
- :CallLedger: is the per-job record of reserved, completed, and uncertain provider calls.
- :SummaryPart: is one community's bounded evidence set, generated narrative, and canonical paper citations.
- :NumericClaim: is a structured predicate request paired with the tally result that replaces its draft claim.

***implementation reqs***

- The builder must use the current canonical-paper grouping and primary-community rule recorded in `docs/cookbook/community_tally.py` R1.
- Lexical matching must use the `entity_derive.tokenize_text` t3 normalization contract recorded in `global_eval_set.jsonl`; it must preserve digit-bearing tokens and segment boundaries.
- The persistent FullTermDF and CompleteIndex must be scoped to a source build and tokenizer version. The RoutingSignature must be stored separately.
- PostgreSQL owns the persisted DF and routing tables because the source paper and community records already live there; this is an ownership decision, not a performance claim.
- SQLite job rows and the provider-call ledger share the existing semantic tally checkpoint database; raw source paper text is never persisted there.
- The API's auto dispatcher recognizes explicit corpus count phrasing only; unsupported lexical-term and summary modes fail with HTTP 501 rather than silently using retrieval.
- When a global summary is generated, the system shall rank communities with the measured lexical control.
- When a community enters a summary, the system shall select a bounded relevant-paper pool with MMR.
- When a summary response is generated, the system shall report population coverage and uncovered communities.
- When a community summary is generated, the system shall accept citations only for canonical papers in that community's selected evidence set.
- Where model output contains a quantitative claim, the system shall replace the claim with its exact or estimated tally result before serving.
- If model output cites a paper outside the selected evidence set, then the system shall remove that citation before serving the summary.

***test reqs***

- Use at least three varied synthetic communities with repeated tokens, digit-bearing identifiers, a missing token, and terms deliberately excluded from the top-k signature.
- For ten selected terms, the sum of per-community FullTermDF values must equal the direct canonical-paper document frequency.
- A term omitted from RoutingSignature must retain an exact positive FullTermDF lookup; an absent term may return zero only for a matching CompleteIndex.
- Build and query paths must reject a source-build or tokenizer-version mismatch.
- On the deterministic 1,000-paper evaluation slice, the ten lexical question totals must equal the manifest gold counts.
- On a three-stratum synthetic population, 200 SRS-without-replacement repetitions must achieve coverage within six percentage points of the nominal 95% interval.
- Non-census strata with all-zero or all-one sampled labels must have positive estimated variance; census strata may have zero sampling variance.
- An unsure label must contribute to the upper prevalence endpoint and remain visible in the per-stratum audit.
- The predicate classifier must request a JSON object with exactly one `label` in `{yes,no,unsure}` and reject malformed or additional output fields.
- When the supplied paper text is truncated and does not show sufficient evidence, the predicate classifier must return `unsure` rather than infer `no`.
- On the eval set, lexical routing must report top-k relevant-community hit rate, positive-community recall, and tie-expansion frequency.
- The eval route gate must compare the current 100-term signature with a 500-term signature on the same exact DF and query set.
- When optional semantic route flags are evaluated, the system shall compare the same ten questions, labeled candidate communities, and top-k width; report hit rate and positive-community recall; and keep an option off when it does not improve recall over the lexical control. The gold limit is the eval artifact's closed-world alias truth.
- A durable job manifest must pin source build, population digest, model/prompt identity, seed, full per-community draw order, and total budgets before classification.
- A completed provider response must be committed under a stable call key before it is returned to the tally loop.
- On process restart, in-flight calls must become uncertain and running jobs must become incomplete; uncertain calls must never be silently repeated.
- The API must expose submit, inspect, and resume operations; resume must retain the original manifest and call budget.
- Open WebUI auto dispatch must poll aggregate jobs and show sampling bounds without claiming that AGE performed the tally.
- When the five-question summary battery runs, the system shall report selected and uncovered communities, coverage, MMR-selected canonical-paper citations, and tally-backed quantitative claims.

***functional specs***

- When the index builder processes a source build, it must count each canonical paper once in exactly one PrimaryCommunity.
  - Given main and `_methods` records share a canonical ID, then their sections contribute to one paper before assignment and counting.
- When the index builder persists term counts, it must store every observed term in FullTermDF independently of RoutingSignature truncation.
  - Given a term is outside the configured signature top-k but present in papers, then its exact FullTermDF remains queryable.
- When a single lexical term is tallied against a matching CompleteIndex, the system must return the sum of its per-community FullTermDF values as an exact count.
  - Given no community row exists for the term and the CompleteIndex matches the source build and tokenizer version, then that community contributes zero.
- If the CompleteIndex is absent or its source build or tokenizer version differs, then the system must reject the exact tally rather than infer zero from missing rows.
- When the builder receives tokenized text, it must use the evaluation set's t3 tokenization behavior, including digit-bearing tokens and tokenizer-defined segment boundaries.
- When the evaluation slice is indexed, the ten lexical question totals must match their manifest gold counts exactly.
- When semantic labels contain yes/no/unsure, the estimator must compute lower prevalence from yes/n and upper prevalence from (yes+unsure)/n, weighted by each community's population.
- When the estimator expands either endpoint, it must use a finite-population correction and the smoothed `(successes+0.5)/(n+1)` variance at non-census sample endpoints.
- When routing receives lexical terms, it must sum matching signature keyness by community and include communities tied within the configured relative cutoff margin.
- When a tally has global scope, the route result must not remove any community from the tally population.
- The labeled eval index must checkpoint exact counts in SQLite by source build, tokenizer version, and eval-artifact digest, then reuse only an exact key match.
- A matching cache hit must revalidate current source paper IDs, text hashes, and primary communities without tokenizing the paper text again.
- A matching but incomplete or internally inconsistent cache must fail explicitly rather than return partial counts.
- A global semantic tally must pilot every community in scope before estimating prevalence.
- Every semantic sample must be uniform without replacement within its primary-community stratum.
- The semantic tally must stop when its relative interval half-width meets the target or its call budget is exhausted, and must report whether it converged.
- Neyman reallocation must remain within the remaining call budget and never sample a paper twice; its default is on after matched eval evidence showed lower absolute error at equal measured coverage.
- Every semantic tally result must report per-community population, sample size, yes/unsure counts, endpoint proportions, call count, interval, and convergence state.
- Validated semantic labels must be checkpointed by source build, predicate, exact classifier input, truncation state, and classifier version so resumed tallies can reuse paid calls safely.
- Given an auto request asking how many papers discuss a named topic, when the API recognizes the count pattern, then it must return HTTP 202 with a durable job ID and pinned manifest.
- Given an ordinary research question, when auto dispatch cannot identify explicit count phrasing, then it must continue through the existing AGE retrieval path.
- Given a completed job, when Open WebUI polls its ID, then it must display the stored estimate, sequential interval, convergence state, and classifier-accuracy limitation.
- When an `answer_kind=summary` request arrives, the system shall return community summaries with population-weighted coverage and citations drawn only from selected evidence.
  - Where a model proposes a quantitative claim, the system shall replace it with the corresponding exact or estimated tally result before returning the summary.
  - If a model cites a paper outside the selected evidence set, then the system shall remove that citation before serving.
- Given `answer_kind=lexical_term` before exact lexical dispatch exists, when the API receives the request, then it must return HTTP 501 instead of reporting a retrieval answer as an aggregate result.
