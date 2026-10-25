"""
community_tally.py  --  PSEUDOCODE (python-flavored, not guaranteed to run)

Goal: answer corpus-wide aggregate questions (tally / prevalence of a predicate per entity or overall)
by using communities as strata, a signature index for routing + exact lexical tallies, and
stratified random sampling + an LLM predicate for semantic tallies.

Spec layer entered: Behavioral (ordered, stateful pipeline) with a Structural block for stored tables.
Why: the work is a staged pipeline with persistence (signature index) and a sampling loop.

STAGED PIPELINE TABLE
| Stage  | Step                                                                 | Tag         | Guards          |
|--------|----------------------------------------------------------------------|-------------|-----------------|
| EVAL   | Freeze a <=2k-document labeled aggregate QA set before tuning.        | [mandatory] | E1              |
| BUILD  | B1 community partition given (Leiden or equivalent)                  | [contract]  | R1              |
| BUILD  | B2 full community-term df table plus separate top-k keyness signature | [mandatory] | R2 R3 R4        |
| BUILD  | B3 diverse medoid vectors per community                              | [opt] flag  | R5              |
| ROUTE  | R1 lexical score = sum of keyness over matched query terms           | [mandatory] | R6              |
| ROUTE  | R2 embedding score = max cosine to community vectors                 | [opt] flag  | R5 R6           |
| ROUTE  | R3 fuse R1+R2 by reciprocal rank fusion                              | [opt] flag  | R6              |
| ROUTE  | R4 tie margin -> widen the set (fan out)                             | [mandatory] | R7              |
| TALLY  | T1 lexical single-term exact tally from df                           | [mandatory] | R3 R8           |
| TALLY  | T2 pilot random sample in every community in scope                   | [mandatory] | R9 R10          |
| TALLY  | T3 LLM predicate on sampled docs                                     | [contract]  | R11             |
| TALLY  | T4 stratified estimator + confidence interval                        | [mandatory] | R12 R13         |
| TALLY  | T5 Neyman reallocation with defensive floor                          | [opt] flag  | R14             |
| TALLY  | T6 stop rule: interval half-width or budget                          | [mandatory] | R15             |
| JOBS   | J1 pin manifest and checkpoint each logical classifier call          | [mandatory] | R23 R24 R25    |
| SUMMARY| M1 per-community read set: relevance pool, then MMR for coverage    | [mandatory] | R17             |
| SUMMARY| M2 map: LLM summarizes each community's read set                     | [contract]  | R18             |
| SUMMARY| M3 reduce: combine community summaries, weighted by N_c, cited       | [mandatory] | R18 R20         |
| SUMMARY| M4 numeric claims replaced by tally-path results                     | [mandatory] | R19             |
| SERVE  | S1 return estimate, interval, mode (exact|estimated), provenance     | [mandatory] | R16 R26         |

Every [opt] stage is a flag; defaults start OFF and change only after a matched measurement against the labeled
aggregate eval set. Neyman reallocation is now promoted ON by R14 evidence. Parameter defaults marked [unmeasured] remain guesses.
The live 100/500 lexical signature comparison (10 questions) had 100% question hit rate, 100% tie expansion,
and 5.66% top-k positive-community recall; it does not evaluate embedding routing, exemplars, or RRF.
The matched semantic route comparison (10 questions, same 152 candidate communities, top-10 width) scored
lexical 100% hit / 23.85% positive-community recall, summary embeddings 90% / 9.90%, four exemplars 80% /
11.93%, and lexical+embedding RRF 100% / 19.94%. Keep embedding, exemplars, and RRF off; labels are closed-world aliases.

EARS GUARDS (only what the table cannot carry)
E1  The evaluation set SHALL contain at least 20 questions, a <=2,000-canonical-paper corpus slice, exact
    ground-truth labels and tallies, and both lexical and semantic predicates. Every label SHALL have
    reproducible provenance. Semantic gold generated from curated aliases SHALL state that its scope is
    closed-world (those aliases), not free-form human semantic truth.
R1  A paper's canonical ID SHALL remove only a terminal `_methods` suffix; main and methods records for
    one paper SHALL be grouped before sampling or counting. The partition SHALL be disjoint (each paper
    in exactly one community). When source communities are section-level, assign each paper to its most
    frequent section community across its grouped records; break ties by lowest community ID. This
    primary assignment SHALL NOT rewrite the source section partition.
R2  The full df table SHALL store exact canonical-paper document frequencies for every t3 token and n_docs per
    community. The routing signature MAY truncate to top_k_terms but SHALL be stored separately from full df.
R3  WHEN the predicate is a single lexical term, the system SHALL compute the tally as sum_c df[c, term]
    from the full df table. An absent row means zero only when the index complete marker, source build ID,
    and tokenizer version match the queried population.
    IF the predicate is a conjunction, THEN df alone SHALL NOT be reported as exact; it is an upper bound
    min over terms of df, and the system SHALL fall back to T2-T4 or to a bridge-table / bitmap intersection.
R4  Index and query normalization SHALL use the same `entity_derive` t3 tokenization contract and SHALL NOT
    alter digit-bearing identifiers (part numbers, case numbers).
R5  WHERE use_exemplars is on, community vectors SHALL be diverse exemplars scored by max cosine, never
    averaged into one vector (averaging diverse exemplars lands between modes).
R6  Routing SHALL rank communities; it SHALL NOT be used to drop communities from a GLOBAL tally (see R9).
R7  IF the gap between the k-th and (k+1)-th route score is below tie_margin, THEN include the tied
    communities.
R8  Lexical exact tallies SHALL be labeled mode="exact"; sampled tallies mode="estimated".
R9  WHEN scope is global, the system SHALL draw a pilot sample in EVERY community (strata = all), because
    pruned communities contribute unknown mass and bias the total low.
R10 Sampling inside a community SHALL be uniform random without replacement. MMR/diversity selection
    SHALL NOT be used for tally samples (it biases prevalence).
R11 The LLM predicate SHALL use a fixed prompt and JSON schema with exactly one `label` in {yes, no, unsure};
    malformed output SHALL fail the call. "Unsure" counts as a separate stratum outcome and is reported, never
    silently folded into "no". Truncated text without sufficient positive evidence SHALL be labeled unsure.
R12 For yes/no/unsure labels, estimate lower prevalence from yes/n and upper prevalence from
    (yes+unsure)/n. Bound each endpoint with a hypergeometric prior-posterior-ratio confidence sequence
    using a uniform beta-binomial working prior. Split alpha=.05 across both endpoints and all communities,
    intersect bounds across every look, and sum lower-yes and upper-possible bounds across strata. Report
    the raw midpoint as point estimate; expose p_yes, p_possible, and unsure per stratum. Sampling coverage
    is conditional on classifier labels and SHALL NOT be described as semantic accuracy.
R13 IF either endpoint sample proportion is 0 or 1 and n<N_c, THEN its variance SHALL use a smoothed p
    (add 0.5 successes and failures); a census stratum has zero sampling variance.
R14 WHERE reallocation is on, remaining budget SHALL be split by Neyman allocation (N_c * s_c) after
    reserving defensive_floor of the budget evenly across strata.
R15 The loop SHALL stop when the sequential interval half-width <= target OR either call/sample budget is spent; IF budget is spent first, THEN
    the result SHALL carry converged=False.
R16 The result SHALL include per-community n_sampled, p_hat, N_c so the number can be audited.
R17 WHEN kind is summary, the read set per community SHALL be chosen by MMR over relevance to the
    question (coverage of modes), not uniform random. Summaries are not estimators, so R10 does not apply.
R18 Each community summary SHALL carry N_c and n_read; the reduce step SHALL weight by N_c and cite the
    communities each statement came from.
R19 Any numeric claim in a summary (counts, rates, "most", "majority") SHALL be replaced by a tally-path
    result (T1 exact or T2-T4 estimated with interval). The summarizer SHALL NOT be the source of numbers.
R20 Summary routing MAY prune communities below the relevance floor. IF it prunes, THEN the result SHALL
    report coverage = N_covered / N_total and list the uncovered communities.

R21 The eval index checkpoint SHALL key exact counts by source build, tokenizer version, and eval-artifact
    digest. WHEN a matching checkpoint is loaded, THEN source paper IDs, text digests, and primary communities
    SHALL be revalidated without tokenization; a malformed matching checkpoint SHALL fail explicitly.
R22 Semantic classification results SHALL be checkpointed by source build, predicate, text digest, truncation
    state, and classifier version. A matching cached label SHALL be reused without an external model call.
R23 A job SHALL persist its immutable source build, population digest, model/prompt, seed, per-community draw order,
    and total budgets before sampling; every successful provider response SHALL be committed immediately.
R24 WHEN a process restarts with a job marked running, THEN mark it incomplete; IF a call was in-flight without
    a committed response, THEN mark it uncertain and block silent replay. Resume SHALL preserve completed calls and budgets.
R25 The API SHALL expose POST /aggregate, GET /aggregate/{job_id}, and POST /aggregate/{job_id}/resume.
    The AGE Function SHALL poll accepted jobs; the legacy SQL Function SHALL remain unchanged.
R26 Tally output SHALL separate label-cache hits, logical requests, actual provider attempts, and durable
    response reuse. Aggregate counting SHALL report PostgreSQL tally provenance and SHALL NOT claim AGE traversal.

CLOSED (rejected on evidence; do not re-litigate without new data)
- Hop distance from a centroid as a source of counts. Landmark distances give bounds on pairwise distance
  (triangle inequality), used to steer shortest-path search, not counts. [ALT, arXiv 1603.01607]
- GIN count(*) as a fast path. GIN has no index-only scan; counts go through a bitmap heap scan.
  [pgsql-hackers 2017 thread on index-only count for bitmap indexes; patch outcome unknown]
- Graph DB as a magnitudes-faster tally engine than Postgres. No head-to-head found; for single-term
  tallies both are inverted-index-shaped. Neo4j helps on multi-hop, a different query class.
- MMR for tally sampling. Diversity-selected samples are not probability samples. (reasoning, no paper)
- Averaging diverse exemplars into one routing vector. (reasoning; evaluate before reopening)
- Community summaries as the tally source. Summaries answer thematic questions; AGGBench/GlobalQA
  show graph-flavored RAG and standard RAG fail corpus-wide counting. [arXiv 2602.01355, 2510.26205]
- PostgreSQL owns the signature/df index because the corpus and exact document counts already live there.
  This is an ownership choice; no speed advantage over Neo4j is claimed.
- English `tsvector` is closed for this index because the eval artifact is labeled with `entity_derive` t3;
  mixing tokenizers would break exact lexical parity.

PRECONDITIONS AND FAILURE MODES
- Communities must be reasonably homogeneous on the predicate, or sample variance stays high and the
  budget is spent without converging (R15). Measure by per-stratum s_c after the pilot.
- Embedding similarity must predict the LLM label for adaptive reallocation to help (OLLA assumption).
- The lexical proxy (token presence) is not the semantic predicate: negation and synonyms are miscounted.
  T1 is exact for the token, not for the meaning.
- Stale index: df and n_docs drift as docs are added. Rebuild or incrementally update on a schedule.

EVAL (promotion/cut criterion for every [opt] flag)
- Labeled aggregate set: >=20 questions with ground-truth tallies, mixed lexical and semantic predicates.
- The initial canonical-paper slice is selected deterministically from one pinned live build by MD5 order
  of canonical ID. Each sampled paper
  stores one primary community by R1, a source-text digest, and per-question binary labels; question
  definitions store predicate kind, exact matching rule, alias list if applicable, and gold count.
- Metrics: absolute relative error of the total, CI coverage (does the true value fall inside), LLM calls.
- Promote a flag only if it lowers error or LLM calls at equal coverage. Run each flag on/off, one at a time.
"""

from dataclasses import dataclass
from math import sqrt
import random

# ----------------------------------------------------------------------------------------------
# Structural block: stored tables
# ----------------------------------------------------------------------------------------------
# global_qa_build(qa_build_id PK, source_build_id, tokenizer_version, signature_top_k, n_docs, complete BOOL)
# global_qa_doc(qa_build_id, paper_id, community_id, text_sha256)      PK(qa_build_id, paper_id)
# global_qa_community(qa_build_id, community_id, n_docs)               PK(qa_build_id, community_id)
# global_qa_term_df(qa_build_id, community_id, term, df INT)           PK(...), every observed t3 token
# global_qa_signature(qa_build_id, community_id, term, keyness, df)   PK(...), top-k routing rows only
# global_qa_vec(qa_build_id, community_id, vec_id, vector)             diverse medoids only

@dataclass
class Config:
    top_k_terms: int = 100            # [unmeasured] signature terms kept per community
    n_exemplars: int = 4              # [unmeasured] medoids per community
    use_exemplars: bool = False       # flag, default off
    use_embedding_route: bool = False # flag, default off
    use_rrf_fusion: bool = False      # flag, default off
    use_reallocation: bool = True     # promoted by matched 10-question/100-seed eval; see .tmp/reallocation_eval_result.json
    route_top_k: int = 5              # [unmeasured]
    tie_margin: float = 0.1           # [unmeasured] relative score gap
    pilot_budget: int = 1000          # [unmeasured] global calls; must cover at least one per stratum
    pilot_per_stratum_cap: int = 30   # [unmeasured] cap after the one-per-stratum floor
    budget: int = 2000                # total LLM calls
    alpha: float = 0.05               # familywise sequential sampling error bound; split across endpoints and strata
    defensive_floor: float = 0.10     # fraction of remaining budget spread evenly (InQuest default)
    target_halfwidth: float = 0.02    # fraction of the estimated total, [unmeasured]
    z: float = 1.96
    summary_pool: int = 50            # [unmeasured] most-relevant docs per community before MMR
    summary_sample_n: int = 8         # [unmeasured] docs actually read per community
    mmr_lambda: float = 0.7           # [unmeasured] relevance vs diversity

# ----------------------------------------------------------------------------------------------
# BUILD
# ----------------------------------------------------------------------------------------------
def build_signature_index(docs, partition, cfg):
    """
    Require: partition is disjoint (R1); docs carry normalized tokens that preserve digit-bearing ids (R4).
    Guarantee: community(n_docs) and community_term(keyness, df) written; df exact per (community, term).
    Maintain: sum over communities of n_docs == total docs.
    Assert: no community has n_docs == 0.
    """
    for c in partition.communities():
        members = partition.docs_in(c)
        n_docs = len(members)
        assert n_docs > 0
        df = document_frequency(members)                 # every term -> exact docs in c containing term
        # keyness vs. rest of corpus; weighted log-odds with informative prior preferred for rare terms
        keyness = keyness_scores(df, n_docs, background=global_df())
        top = top_k_by_value(keyness, cfg.top_k_terms)
        store_community(c, n_docs)
        for term, count in df.items():
            store_community_df(c, term, count)            # complete table; exact tally never uses top-k
        for term in top:
            store_community_signature(c, term, keyness[term], df[term])

def build_exemplars(docs, partition, embed, cfg):
    """
    Require: cfg.use_exemplars is True.
    Guarantee: per community, n_exemplars "medoid" vectors (real records).
    Maintain: medoids come from k-means with k = n_exemplars inside the community (diversity by construction).
    """
    if not cfg.use_exemplars:
        return
    for c in partition.communities():
        X = [embed(d.text) for d in partition.docs_in(c)]
        for medoid in kmeans_medoids(X, k=cfg.n_exemplars):
            store_vec(c, "medoid", medoid)

# ----------------------------------------------------------------------------------------------
# ROUTE
# ----------------------------------------------------------------------------------------------
def route(query_terms, query_vec, cfg):
    """
    Require: query_terms normalized the same way as BUILD (R4).
    Guarantee: ranked list of (community, score); ties within tie_margin are all included (R7).
    Maintain: ranking only; never used to drop communities from a global tally (R6, R9).
    """
    lex = {}
    for c, term, keyness, _df in lookup_signature(query_terms):
        lex[c] = lex.get(c, 0.0) + keyness

    ranked = sort_desc(lex)
    if cfg.use_embedding_route and query_vec is not None:
        emb = {c: max(cosine(query_vec, v) for v in vectors_of(c)) for c in all_communities()}
        ranked_emb = sort_desc(emb)
        ranked = rrf(ranked, ranked_emb) if cfg.use_rrf_fusion else ranked_emb

    return include_ties(ranked, k=cfg.route_top_k, margin=cfg.tie_margin)

# ----------------------------------------------------------------------------------------------
# TALLY
# ----------------------------------------------------------------------------------------------
def lexical_tally(term):
    """
    Require: predicate is ONE t3 lexical term and a complete matching source-build/tokenizer marker exists (R3).
    Guarantee: exact canonical-paper count from the full term-DF table, mode="exact".
    Assert: absent rows are zero only after marker validation; routing signatures are not queried.
    """
    # Routing is truncated; exact counts always query the independent full df table.
    validate_complete_index(source_build_id, tokenizer_version)
    total = 0
    for c in all_communities():
        df = df_lookup(c, term)          # absent term -> 0 only after complete-index validation
        total += df
    return Result(estimate=total, interval=(total, total), mode="exact")

def stratified_estimate(strata, z):
    """
    Require: strata[c] = dict(N, n, yes, unsure) with N docs in community and n sampled.
    Guarantee: fixed-sample normal comparison interval; it never drives adaptive stopping (R12).
    Assert: every stratum has n >= 1.
    """
    lower, upper = 0.0, 0.0
    var_lower, var_upper, population = 0.0, 0.0, 0
    for c, s in strata.items():
        assert s["n"] >= 1
        p_lo = s["yes"] / s["n"]
        p_hi = (s["yes"] + s["unsure"]) / s["n"]
        fpc = 1.0 - s["n"] / s["N"]
        smooth_lo = (s["yes"] + 0.5) / (s["n"] + 1.0)          # R13: variance only
        smooth_hi = (s["yes"] + s["unsure"] + 0.5) / (s["n"] + 1.0)
        var_lower += (s["N"] ** 2) * smooth_lo * (1.0-smooth_lo) / s["n"] * fpc
        var_upper += (s["N"] ** 2) * smooth_hi * (1.0-smooth_hi) / s["n"] * fpc
        lower += s["N"] * p_lo
        upper += s["N"] * p_hi
        population += s["N"]
    lo = max(0.0, lower - z * sqrt(var_lower))
    hi = min(float(population), upper + z * sqrt(var_upper))
    return (lower + upper) / 2.0, (lo, hi)

def sequential_interval(strata, alpha, previous_bounds):
    """
    Require: strata contain current SRSWOR counts; alpha is split across two endpoints and every stratum.
    Guarantee: simultaneous finite-population bounds remain valid across adaptive looks (R12).
    Maintain: each endpoint bound is intersected with its previous look.
    """
    endpoint_alpha = alpha / (2 * len(strata))
    lower_total, upper_total = 0, 0
    for c, s in strata.items():
        for endpoint, successes in [("yes", s["yes"]), ("possible", s["yes"] + s["unsure"])]:
            current = hypergeometric_ppr_bounds(s["N"], s["n"], successes, endpoint_alpha)
            previous_bounds[c, endpoint] = intersect(previous_bounds[c, endpoint], current)
        lower_total += previous_bounds[c, "yes"][0]
        upper_total += previous_bounds[c, "possible"][1]
    return sum(s["N"] * (s["yes"] + (s["yes"] + s["unsure"])) / (2 * s["n"])
               for s in strata.values()), (lower_total, upper_total)

def neyman_allocation(strata, remaining, floor_frac):
    """
    Require: strata carry N and a smoothed stdev s_c.
    Guarantee: integer allocation summing to <= remaining; floor_frac of remaining spread evenly (R14).
    """
    floor_total = int(remaining * floor_frac)
    per_floor = floor_total // len(strata)
    weight = {c: s["N"] * s["sd"] for c, s in strata.items()}
    wsum = sum(weight.values()) or 1.0
    rest = remaining - per_floor * len(strata)
    return {c: per_floor + int(rest * weight[c] / wsum) for c in strata}

def semantic_tally(predicate, scope_communities, llm, cfg, query_terms=None, query_vec=None):
    """
    Require: scope_communities is ALL communities when the question is global (R9); pilot_budget covers one draw per stratum.
    Guarantee: Result(mode="estimated", interval, converged, per-community audit rows) (R15, R16).
    Maintain: samples are uniform random without replacement per community (R10);
              LLM calls used <= cfg.budget.
    Assert: pilot drawn in every community in scope before any reallocation.
    """
    strata = {}
    used = 0
    confidence_bounds = initialize_all_endpoint_bounds(scope_communities)
    if cfg.pilot_budget > cfg.budget or cfg.pilot_budget < len(scope_communities):
        raise ValueError("pilot budget must fit the query budget and cover every stratum")
    # T2 pilot includes every stratum first, then spends the remaining pilot budget evenly up to the cap.
    pilot_sizes = allocate_uniform_pilot(scope_communities, cfg.pilot_budget, cfg.pilot_per_stratum_cap)
    for c in scope_communities:
        N = n_docs(c)
        n0 = min(pilot_sizes[c], N)
        sample = random.sample(doc_ids(c), n0)                        # R10, no MMR
        labels = [llm(predicate, d) for d in sample]                  # R11: validated yes/no/unsure
        strata[c] = dict(N=N, n=n0, yes=labels.count("yes"), unsure=labels.count("unsure"),
                         seen=set(sample))
        used += n0

    while True:
        total, interval = sequential_interval(strata, cfg.alpha, confidence_bounds)
        hw = (interval[1] - interval[0]) / 2.0
        if hw <= cfg.target_halfwidth * max(total, 1.0):
            return finish(strata, total, interval, converged=True)
        if used >= cfg.budget:
            return finish(strata, total, interval, converged=False)   # R15

        remaining = cfg.budget - used
        for c, s in strata.items():
            p_lo = (s["yes"] + 0.5) / (s["n"] + 1.0)
            p_hi = (s["yes"] + s["unsure"] + 0.5) / (s["n"] + 1.0)
            s["sd"] = max(sqrt(p_lo*(1-p_lo)), sqrt(p_hi*(1-p_hi)))
        if cfg.use_reallocation:
            plan = neyman_allocation(strata, min(remaining, 200), cfg.defensive_floor)
        else:
            plan = {c: min(remaining, 200) // len(strata) for c in strata}   # proportional baseline

        for c, extra in plan.items():
            pool = [d for d in doc_ids(c) if d not in strata[c]["seen"]]
            take = random.sample(pool, min(extra, len(pool)))
            for d in take:
                lab = llm(predicate, d)
                strata[c]["n"] += 1
                strata[c]["yes"] += (lab == "yes")
                strata[c]["unsure"] += (lab == "unsure")
                strata[c]["seen"].add(d)
            used += len(take)
        if all(not [d for d in doc_ids(c) if d not in strata[c]["seen"]] for c in strata):
            total, interval = sequential_interval(strata, cfg.alpha, confidence_bounds)
            return finish(strata, total, interval, converged=(interval[1]-interval[0])/2 <= cfg.target_halfwidth*max(total,1.0))

def finish(strata, total, interval, converged):
    """Guarantee: audit rows (N, n, p_yes, p_possible, unsure) and a conservative interval are returned (R16)."""
    rows = [dict(community=c, N=s["N"], n=s["n"], p_yes=s["yes"] / s["n"],
                 p_possible=(s["yes"]+s["unsure"]) / s["n"], unsure=s["unsure"])
            for c, s in strata.items()]
    return Result(estimate=total, interval=interval, mode="estimated",
                  converged=converged, audit=rows)

# ----------------------------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------------------------
def global_summary(question, cfg, llm, embed):
    """
    Require: question is thematic or mixed (kind == "summary").
    Guarantee: Summary(text, coverage, uncovered, parts, numbers) where every number comes from the
               tally path (R19) and coverage is reported (R20).
    Maintain: per-community read set size <= cfg.summary_sample_n; LLM calls = communities routed + 1
              (reduce) + numeric claims tallied.
    Assert: every part carries N and n_read (R18).
    """
    qterms = normalize_terms(question)                       # R4 preserved
    qvec = embed(question)
    ranked = route(qterms, qvec, cfg)                        # pruning allowed here (R20)

    total_N = sum(n_docs(c) for c in all_communities())
    covered_N = sum(n_docs(c) for c, _ in ranked)

    # MAP: read a coverage-oriented set from each routed community
    parts = []
    for c, _score in ranked:
        pool = top_relevant_docs(c, qvec, m=cfg.summary_pool)
        picked = mmr(pool, qvec, k=cfg.summary_sample_n, lam=cfg.mmr_lambda)      # R17
        parts.append(dict(community=c, N=n_docs(c), n_read=len(picked),
                          text=llm_summarize(question, picked)))

    # REDUCE: largest communities first; summarizer must cite community ids (R18)
    parts.sort(key=lambda p: -p["N"])
    draft = llm_combine(question, parts)

    # NUMBERS: replace every quantitative claim with a tally-path answer (R19)
    numbers = []
    for claim in extract_numeric_claims(draft):
        res = answer(claim.as_question(), cfg, llm, embed)   # exact or estimated with interval
        draft = draft.replace(claim.text, render(claim, res))
        numbers.append(res)

    uncovered = [c for c in all_communities() if c not in {p["community"] for p in parts}]
    return Summary(text=draft, coverage=covered_N / total_N, uncovered=uncovered,
                   parts=parts, numbers=numbers)

# ----------------------------------------------------------------------------------------------
# SERVE
# ----------------------------------------------------------------------------------------------
def answer(question, cfg, llm, embed):
    """
    Require: question parsed to (predicate, kind in {"lexical_term","semantic","summary"}, scope in {"global","routed"}).
    Guarantee: a Result; lexical single-term -> exact (R3, R8); otherwise estimated with interval.
    """
    predicate, kind, scope = parse(question)
    if kind == "summary":
        return global_summary(question, cfg, llm, embed)
    if kind == "lexical_term":
        return lexical_tally(predicate)
    qterms = normalize_terms(predicate)                     # R4 preserved
    qvec = embed(predicate) if cfg.use_embedding_route else None
    if scope == "global":
        communities = all_communities()                      # R9
    else:
        communities = [c for c, _ in route(qterms, qvec, cfg)]
    return semantic_tally(predicate, communities, llm, cfg, qterms, qvec)
