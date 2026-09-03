"""
sampler.py — deterministic evidence walker over a persisted run.

    anchors -> induced subgraph -> top-n by decayed score -> group by stored cid
            -> rank communities by hit count -> frozen evidence bundle

No model is called anywhere in this file, and by default no randomness either:
the walk is top-n by strength-decayed score, so the whole retrieval is a pure
function of stored data plus n. A disputed answer can be recomputed rather than
taken on trust.

The module is named for the Boltzmann sampler it was designed around. That
sampler is still here behind T > 0, but it lost to plain argmax on every query
measured (see MEASURED below), so it is not the default and the name is now
historical.

Replaces the LLM tool-caller in walk_agent.py, which was built, measured, and
retired: 14 steps, 26 chunks, `finish` never called, community saturation flat
from step 8. Route selection is a sampling problem; an LLM asked "have I seen
enough" has no calibrated notion of enough, and a sample count does.

The operator's framing, which is the right one: sampling the graph is looking a
word up in the index, and the communities you land in are the chapters.

GUARDS (EARS)
S1  Sampling SHALL be a pure function of (candidate set, scores, n, T, seed).
S2  Scores SHALL be z-scored over the candidate set before exponentiation.
    BM25 is unbounded and corpus-scaled; exp() of a raw score saturates to a
    one-hot distribution and T stops meaning anything.
S3  Sampling SHALL be without replacement -- with replacement, one dominant
    chunk consumes the budget and the community histogram degenerates.
S4  IF n >= |C| THEN the sampler SHALL return C and report enumerated=True
    rather than pretending to have sampled.
S5  Sampled chunks SHALL be grouped by their STORED cid. No partitioning
    happens at query time (subgraph Louvain does not restrict global Louvain).
S6  The evidence bundle SHALL be frozen before any model sees it.
S7  Every bundle SHALL carry the parameters that produced it, so it can be
    re-derived exactly.

S8  The default SHALL be T=0 (argmax). Boltzmann sampling is retained as an
    opt-in and is documented as MEASURED WORSE on this corpus.

S9  ef_search() SHALL NOT take a hop count. Depth is EMERGENT: expansion stops
    when the best remaining candidate cannot beat the worst result already
    held, exactly as HNSW terminates. A fixed `hops` is the operator guessing
    how far is far enough; the stopping rule measures it per query.
S10 The tuned dial SHALL be `ef` -- the width of the result set, and hence of
    the candidate frontier. It is HNSW's `ef` in both name and role: larger ef
    explores longer before the stop condition bites, and depth follows from it.
S11 WHERE T > 0, Boltzmann sampling SHALL choose WHICH neighbours to expand,
    not which results to return. Sampling the frontier is exploration; sampling
    the output is noise, and S8's measurement is about the latter.
S12 ef_search() SHALL report the depth it reached and why it stopped
    (converged / hop cap / frontier exhausted). A search that always hits the
    cap is not converging and its ef is mis-tuned.

S15 (AMENDED 2026-09-03, supersedes the sqrt allocation below) BM25 anchors SHALL
    start from the SAME unconditional global gt.search(k=k_anchor) run today, byte-
    identical -- allocation never shrinks or reorders the base anchors. A source
    absent from the base MAY add at most one extra: its single best hit from an
    over-fetched pool, granted only when that hit's score is >= COMPETITIVE_FRAC of
    the base's k-th (weakest) score. Anchors are do-no-harm ADDITIVE-ONLY: a global
    reallocation cannot fix register starvation without breaking something else --
    MEASURED 2026-09-02, run bfa594df, the sqrt per-source split (k_anchor 3 -> 9)
    lifted the quotes share on register-explicit prompts from 14-18% to 23-29% but
    dropped C3's brown floor from 30% to 19%, flipping A1 and E3 PASS -> FAIL. Every
    selected anchor, base or extra, SHALL still pass R1 validation, unchanged:
    competitiveness decides who MAY add, never whether a weak one is admitted.
    [SUPERSEDED sqrt text retained for history: anchors were to be allocated per
    source in proportion to sqrt(n_s), split by largest-remainder rounding and
    selected by a separate per-source search. Superseded because it caps the
    majority source regardless of its true share, which is what broke A1/E3.]

S16 (AMENDED 2026-09-03) At most ONE extra per eligible source, chosen as that
    source's single best hit from the over-fetched pool (pool score order, first
    match). The extras cap is `max(0, ef - len(base))`, applied to the extras BEFORE
    they are merged with base -- base anchors are never displaced by a later
    truncation, so a tie on score with a lower ordinal cannot bump a base row out.
    The merged anchor list SHALL be deduplicated by ordinal and ordered by BM25
    score descending, ties by ordinal ascending; when no extra is granted the
    return value SHALL be the base list itself (the same object), not a rebuilt
    equal one. WHERE the run reports fewer than two source labels -- including
    R19's literal "default" and a pre-R20 run carrying no labels -- the anchor list
    SHALL be identical, element for element and in order, to today's single
    gt.search(k=k_anchor).

S17 (new 2026-09-03; promotes the queued source-aware ring share. Trigger: T7's
    ring evidence, run bfa594df -- B1-B5 ring origin carries 8/24, 4/24, 0/24,
    8/24, 4/24 quotes chunks per walk while walk-only quotes sits at 4-17%, so the
    ring top-off, not the walk, is where register share is won or lost.) WHERE the
    run reports more than one source label AND the anchor list spans more than one
    source, the ring top-off budget (RING_TOP x RING_PER, S13) SHALL be allocated
    across sources in proportion to the anchor list's source composition by
    largest-remainder rounding, each source's slots filled by its strongest ring
    candidates -- S13's candidate pool and strength ordering unchanged, only which
    slots go to which source changes. A source with fewer candidates than slots
    SHALL forfeit the shortfall to the remaining candidates in global strength
    order, so the walk never shrinks. WHERE every anchor shares one source -- or
    the run reports one label -- allocation SHALL be skipped and the ring fill
    SHALL be identical to today's, element for element and in order.

S18 (new 2026-09-03; promotes the probe-validated ring router. Trigger: T7c stopped
    the B lane at 8-29% with S17's anchor-mix share, and a 4-round steering probe
    measured that the ring POOL never contains the minority register's on-topic
    chunks, while every corpus-score allocation signal is flat or sign-wrong.)
    WHERE the run reports more than one source label, the S13/S17 ring candidate
    pool SHALL be UNIONed with a per-source BM25 top-INJECT_K, scored by the same
    formula and postings gt.search uses and bucketed by source BEFORE the cut,
    rescaled into the walk pool's own score range so an injected row never enters
    above the pool's strongest member. Injection is ADDITIVE: no pooled candidate is
    removed, no member of W is re-entered, and select_ring still performs the cut.
    The ring allocation SHALL come from the QUERY, not from the corpus scores:
    w_s = softmax_ROUTER_TAU( mean over query tokens t of
    log( p_s(t) / p_corpus(t) ) ), with p_s(t) = (occ_s(t) + 0.5) / (tokens_s + 0.5)
    counted in OCCURRENCES per token of corpus, never in document frequency -- df is
    length-confounded across registers. The mix passed to select_ring SHALL be
    max(w_s, ROUTER_EPS) renormalized, REPLACING the anchor mix: the anchor-mix
    floor is MEASURED as the cap (same router, anchor floor on -> B 31-38%, floor
    off -> B 52-56%), so it SHALL NOT be applied. WHERE the query shares no token
    with the run's postings the router SHALL return no weights and the ring SHALL
    fall back to S17's anchor mix unchanged. WHERE the run reports one source label
    -- including R19's literal "default" and a pre-R20 run carrying no labels -- the
    router degenerates to a single weight of 1.0, injection is SKIPPED, and the ring
    fill SHALL be identical to today's, element for element and in order, reached by
    the same early return.

S19 (new 2026-09-03) The sampler SHALL pass `expand_aliases` through the ANCHOR path
    only (ef_evidence -> ef_search -> anchor_hits -> gt.search, and candidate_scores's
    legacy call), defaulting OFF, and SHALL record it in Bundle.params. The S18 ring
    injection (source_topk) SHALL NOT expand: two term sets changing in one
    measurement makes a diagnostic delta unattributable.

MEASURED (brown-50, 6 queries, 2026-08-26) -- the sampling premise did not hold
------------------------------------------------------------------------------
Stability (mean pairwise Jaccard of top-3 community sets across seeds) is NOT
monotonic in n. It is U-shaped: 0.884 at frac=0.1/T=0.3, dipping to 0.70 at
frac=0.3, recovering to 0.80 at frac=0.7. High at both ends because low n with
low T is nearly argmax and high frac is nearly enumeration; maximum variance
sits in the middle. Consequence: maximising stability drives T -> 0, i.e. it
rewards not sampling. Stability alone is the wrong objective.

Against a deterministic top-n baseline, sampling surfaced only noise. Every
community Boltzmann found that argmax missed was off-topic:
    jury      -> +Freddy Motors, +Medical misinformation, +Regretful Relationships
    molecular -> +Orioles baseball game report
    election  -> +Orioles, +Ryan Ekstrohm dialogue, +Faculty workload
while argmax's top-3 was correct on all six queries (jury -> Due Process,
c1 [jury/election/department/mayor], Hengesbach murder trial).

Under argmax the #1 community is identical at every n in {8,12,16,24,40} for
all six queries. Positions 2-3 populate by n=24; at n=40 noise reaches position
3. Hence DEFAULT_N=24, DEFAULT_T=0.

T limits, kept because the opt-in path still honours them:
    T -> 0    argmax; top-n by score -- THE DEFAULT, measured best
    T  = 1    z-scores used as-is
    T -> inf  uniform over C; score ignored, pure structure

An earlier caveat that bit once: with n >= |C| the sampler enumerates (S4), and
an enumeration has Jaccard 1.0 by construction. A stability sweep that does not
exclude those cases reports a plateau that is an artifact of not sampling.
Candidate sets here run 25-102, so n=40 enumerates three of five queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import graph_tools as gt

DEFAULT_N = 24             # measured: top-3 communities populated and clean
DEFAULT_T = 0.0            # argmax. Boltzmann measured worse -- see MEASURED
DEFAULT_HOPS = 2
DEFAULT_EF  = 64           # HNSW's ef: result width, the tuned dial (S10)
                           # MEASURED brown-50-dual, 6 queries, T=0.7, m=3,
                           # ef in {8..128}: top-3 community stability vs 2*ef
                           # 0.61 0.64 0.58 0.64 0.67 0.75 0.75 -- first holds
                           # >= 0.75 at 64. NO KNEE: expanded == ef-1 at every
                           # ef, so the stop rule fires only after every held
                           # result is expanded and cost is linear in ef. On
                           # this corpus ef is a cost dial more than a quality
                           # dial. 6 queries x Jaccard-of-3 is underpowered
                           # (~20 needed for a 5% delta); treat 64 as a floor,
                           # not a tuned optimum.
DEFAULT_M   = 3            # neighbours expanded per pop. MEASURED: candidate
                           # pools after seen-filtering have median 2, max 10,
                           # so m=8 enumerates and T never applies (S11 inert).
                           # m must sit BELOW the pool for sampling to exist.
DEFAULT_K_ANCHOR = 3       # BM25 anchors seeded. MEASURED (diagnostic set,
                           # 2026-09-02, run bfa594df): k=9 lifted quotes share
                           # on register-explicit prompts (14-18% -> 23-29%)
                           # but broke C3's brown floor (30% -> 19%), and the
                           # frozen set's rule is do-no-harm -- so 3 stands.
                           # AMENDED 2026-09-03: a GLOBAL anchor count cannot
                           # fix the B-class finding -- and neither can a global
                           # per-source REALLOCATION (the superseded scheme
                           # capped the majority source and flipped A1/E3, see
                           # S15). Anchors are
                           # do-no-harm only: additive minority extras (S15/S16).
                           # The measured B lever is the source-aware ring
                           # share, not anchor allocation. See diagnostic-prompts.md.
COMPETITIVE_FRAC  = 0.5    # S15 amended: a minority source's best hit is granted
                           # an extra anchor only when its score >= this fraction
                           # of the base k-th score. Spec-fixed threshold, not
                           # tuned -- MEASURED 2026-09-02, run bfa594df, basis
                           # for the additive-only amendment.
ANCHOR_FETCH_MIN  = 64     # S15/S16: floor on the over-fetch used to find each
                           # absent source's best hit (no longer per-source quotas)
ANCHOR_FETCH_MULT = 16     # multiplier applied to k before escalation
ANCHOR_FETCH_CAP  = 4096   # ceiling on the over-fetch -- stop escalating past this
MAX_HOPS    = 8            # runaway guard, NOT a depth policy (S9)
DEFAULT_RING_TOP = 3       # S13: one degree out from the top-3 of W (design 6.10)
DEFAULT_RING_PER = 8       # S13: strongest 8 new neighbours per parent
DEFAULT_BRIDGE_PAIRS = 3   # S14: best whole-graph path between the top-3 chunks, pairwise
MAX_BRIDGE = 8             # S14: cap on discovered bridge chunks per walk
DEFAULT_CAP = 150          # R4.4 neighbourhood bound, 2-hop p99 is 111
INJECT_K    = 24           # S18: per-source BM25 top-k UNIONed into the ring
                           # pool. MEASURED (probe R3a): injection alone, with
                           # no router re-quota, is INERT -- B unchanged from
                           # baseline -- it works only paired with the router.
ROUTER_TAU  = 0.2          # S18: softmax temperature on the LM log-odds.
                           # MEASURED: 0.1/0.2/0.5 all plateau at B 48-56%;
                           # 1.0 flattens to 44-46%. 0.2 chosen mid-plateau.
ROUTER_EPS  = 0.05         # S18: constant floor per source -- NOT the anchor
                           # mix (the anchor-mix floor is the measured cap).
                           # MEASURED: eps 0.0 -> B 52-56%, 0.05 -> 48-52%,
                           # 0.1 -> 44-48%. 0.05 buys the no-starvation
                           # guarantee for ~3 points.


@dataclass(frozen=True)
class Bundle:
    """S6/S7: frozen evidence plus everything needed to re-derive it."""
    query: str
    run_id: str
    anchors: list
    candidates: int
    sampled: list                       # ordinals, sample order
    communities: list                   # [{cid, hits, keywords, medoid_text, size}]
    params: dict = field(default_factory=dict)
    enumerated: bool = False            # S4
    scores: dict = field(default_factory=dict)   # ord -> walk score (ef_evidence)
    origin: dict = field(default_factory=dict)   # ord -> walk | ring | bridge (S13/S14)

    @property
    def top_cids(self) -> list:
        return [c["cid"] for c in self.communities]

    def concentration(self) -> float:
        """Share of the sample sitting in the single largest community.
        1.0 means one chapter; near 1/len(communities) means no head at all."""
        if not self.communities:
            return 0.0
        return self.communities[0]["hits"] / max(
            sum(c["hits"] for c in self.communities), 1)


def select_anchors(base: list, pool: list, sources: dict, ef: int,
                    frac: float = COMPETITIVE_FRAC) -> list:
    """S16 (amended 2026-09-03): base is do-no-harm -- a source absent from base
    may ADD at most one extra, its best pool hit, when competitive.

    Require:  ef >= 1; base ordered (-score, ord); pool a superset of base or
              empty; sources is {label: n_s} restricted to n_s > 0.
    Guarantee: every element of base is in the result, in a result ordered
               (-score, ord), deduped by ord, len(result) <= ef and
               <= len(base) + number of missing eligible sources. Returns
               `base` itself (same object) when nothing is granted. Source
               labels are read only to decide who MAY add an extra -- never to
               admit or reject on strength (R1 still decides that, unchanged).

    Deliberate asymmetry: base is source-agnostic, so an unlabelled or
    non-eligible row CAN be a base anchor. Only the extras pass is source-gated.
    """
    if not base:
        return base
    threshold = frac * base[-1]["score"]
    present = {h.get("source") for h in base}
    missing = [s for s in sources if s not in present]
    if not missing:
        return base

    best_by_source: dict = {}
    for h in pool:
        src = h.get("source")
        if src in missing and src not in best_by_source:
            best_by_source[src] = h

    granted = [h for s, h in best_by_source.items() if h["score"] >= threshold]
    granted.sort(key=lambda h: (-h["score"], h["ord"]))       # before the cap
    cap = max(0, ef - len(base))
    granted = granted[:cap]
    if not granted:
        return base

    base_ords = {h["ord"] for h in base}
    merged = list(base) + [h for h in granted if h["ord"] not in base_ords]
    merged.sort(key=lambda h: (-h["score"], h["ord"]))
    return merged


def anchor_hits(conn, run: gt.RunHandle, query: str, k_anchor: int, ef: int,
                expand_aliases: bool = False) -> list:
    """S15 (amended 2026-09-03) entry point: today's global gt.search top-k,
    unconditionally, plus at most one competitive extra per absent source.

    Require:  ef >= 1, k_anchor >= 1.
    Guarantee: returns gt.search-shaped rows, ordered (-score, ord), len <= ef.
               WHERE the run reports fewer than 2 source labels the return value is
               EXACTLY `gt.search(conn, run, query, k=min(k_anchor, ef))` -- the same
               call, the same object, no re-sort, no re-wrap.

    S19: `expand_aliases` (default OFF) is passed to BOTH gt.search calls below
    (the base call and the over-fetch loop) -- passing it to only one splits the
    term set between the two and corrupts select_anchors.
    """
    k = min(k_anchor, ef)
    # Load-bearing: today's call, made FIRST and unconditionally. Identity with
    # today comes from making the same call, not from assuming the top-k of a
    # larger fetch is prefix-stable.
    base = gt.search(conn, run, query, k=k, expand_aliases=expand_aliases)

    sources = gt.run_sources(conn, run)
    elig = {s: n for s, n in sources.items() if n > 0}
    if len(elig) < 2:                                       # S16 degenerate path
        return base

    present = {h.get("source") for h in base}
    if all(s in present for s in elig) or len(base) >= ef:
        return base                                         # no extra possible: no over-fetch

    fetch = max(k * ANCHOR_FETCH_MULT, ANCHOR_FETCH_MIN)
    rows: list = []
    while True:
        rows = gt.search(conn, run, query, k=fetch, expand_aliases=expand_aliases)
        found = {h.get("source") for h in rows} & set(elig)
        if elig.keys() <= found or len(rows) < fetch or fetch >= ANCHOR_FETCH_CAP:
            break
        fetch *= 4

    return select_anchors(base, rows, elig, ef)


def candidate_scores(conn, run: gt.RunHandle, query: str, k_anchor: int = DEFAULT_K_ANCHOR,
                     hops: int = DEFAULT_HOPS, min_strength: float = 0.0,
                     cap: int = DEFAULT_CAP, expand_aliases: bool = False) -> tuple[dict, list]:
    """Induced subgraph around the query's lexical anchors.

    Score is the strength-decayed path score: an anchor starts at its
    normalised BM25 score, and each hop multiplies by the edge strength. Best
    path per node wins, which mirrors the cookbook expansion.

    Traversal is graph_tools.neighbors -- indexed src/dst only, never attrs.

    S19: `expand_aliases` (default OFF) is threaded to this legacy call too,
    for consistency with the ef_evidence anchor path.
    """
    # Legacy fixed-hop path behind evidence(). Allocation (S15/S16) is
    # ef_search's serve path only -- this call stays a plain global search.
    hits = gt.search(conn, run, query, k=k_anchor, expand_aliases=expand_aliases)
    if not hits:
        return {}, []
    top = max(h["score"] for h in hits) or 1.0
    anchors = [h["ord"] for h in hits]

    scores = {h["ord"]: h["score"] / top for h in hits}
    frontier = dict(scores)
    for _ in range(hops):
        nxt: dict = {}
        for src, sc in frontier.items():
            for nb in gt.neighbors(conn, run, src, limit=40,
                                   min_strength=min_strength):
                cand = sc * nb["strength"]
                if cand > scores.get(nb["ord"], 0.0):
                    scores[nb["ord"]] = cand
                    nxt[nb["ord"]] = cand
        frontier = nxt
        if not frontier:
            break

    if len(scores) > cap:                      # R4.4, strongest kept
        keep = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:cap]
        scores = dict(keep)
    return scores, anchors


def boltzmann_sample(scores: dict, n: int = DEFAULT_N, T: float = DEFAULT_T,
                     seed: int = 0) -> tuple[list, bool]:
    """S1-S4. Returns (ordinals in draw order, enumerated).

    Without replacement: draw, remove, renormalise. Implemented via the Gumbel
    top-k trick, which is exactly equivalent to sequential sampling without
    replacement and needs no renormalisation loop.
    """
    if not scores:
        return [], False
    ords = sorted(scores)                       # deterministic ordering
    s = np.array([scores[o] for o in ords], float)

    if n >= len(ords):                          # S4
        # Score-ordered even when enumerating: `sampled` is documented as draw
        # order, and returning ordinal order here would silently make it mean
        # something different in the one branch that takes everything.
        return sorted(ords, key=lambda o: (-scores[o], o)), True

    if T <= 0.0:                                # argmax -- the default, S8
        pick = np.argsort(-s, kind="stable")[:n]
        return [ords[i] for i in pick], False

    sd = s.std()
    z = (s - s.mean()) / sd if sd > 1e-12 else np.zeros_like(s)   # S2
    logits = z / T

    rng = np.random.default_rng(seed)           # S1
    g = rng.gumbel(size=len(logits))            # S3 via Gumbel top-k
    pick = np.argsort(-(logits + g))[:n]
    return [ords[i] for i in pick], False


def ef_search(conn, run: gt.RunHandle, query: str, ef: int = DEFAULT_EF,
              T: float = DEFAULT_T, m: int = DEFAULT_M,
              k_anchor: int = DEFAULT_K_ANCHOR, max_hops: int = MAX_HOPS,
              seed: int = 0, expand_aliases: bool = False) -> tuple[dict, dict]:
    """Best-first expansion with an HNSW stop condition (S9-S12).

    ef IS the evidence budget (S10), so anchors never exceed it:
    k_anchor is clamped to ef -- a wide default anchor count must not
    inflate a deliberately tight walk.

    HNSW does not take a hop count. It holds a result set W of width `ef` and a
    candidate frontier C, and stops when the best remaining candidate cannot
    beat the worst member of W -- nothing reachable from here can improve the
    answer. Depth is whatever that took.

    The same rule applies to this graph, with BM25 anchors as entry points and
    strength-decayed path score in place of distance:

        W  <- anchors                       result set, capped at ef
        C  <- anchors                       frontier, best-first
        while C:
            c = best of C
            if score(c) <= worst(W) and |W| = ef:  STOP    <- S9
            expand c's neighbours, keep any that improve W

    WHERE T > 0 the neighbours of c are Boltzmann-sampled rather than all
    expanded (S11). That is exploration of the frontier, which is a different
    question from S8's measurement -- S8 found sampling the OUTPUT surfaced
    off-topic communities, and this does not touch the output.

    Require:  ef >= 1; m >= 1; max_hops >= 1 as a runaway guard, not a policy.
    Guarantee: (results, telemetry) where results is {ord: score} of size <= ef
              and telemetry names the depth reached and the stop reason (S12).

    S15/S16 (amended 2026-09-03): on a run reporting >= 2 source labels,
    anchor_hits starts from the same global gt.search(k=k_anchor) as a single-
    source run and may add at most one competitive extra per absent source. The
    budget exceeds k_anchor only by those granted extras, never by a
    reallocation; S10 still wins -- anchors never exceed ef, since the extras
    cap is applied inside anchor_hits before it returns.

    S19: `expand_aliases` (default OFF) is passed through to anchor_hits only
    -- the ring injection (source_topk) does not expand.
    """
    if ef < 1 or m < 1 or max_hops < 1:
        raise ValueError(f"ef, m, max_hops must all be >= 1 "
                         f"(got {ef}, {m}, {max_hops})")
    hits = anchor_hits(conn, run, query, k_anchor, ef, expand_aliases=expand_aliases)
    if not hits:
        return {}, {"stop": "no_anchor", "depth": 0, "expanded": 0, "seen": 0,
                    "hop_capped": 0, "pool_median": 0, "pools_over_m": 0,
                    "pools": 0, "ef": ef, "T": T, "m": m, "anchors": []}
    top = max(h["score"] for h in hits) or 1.0

    W = {h["ord"]: h["score"] / top for h in hits}          # result set
    C = {h["ord"]: (h["score"] / top, 0) for h in hits}     # frontier: ord -> (score, hop)
    hop_of = {h["ord"]: 0 for h in hits}
    seen = set(W)
    expanded = 0
    capped = 0
    pools = []
    stop = "frontier_exhausted"

    while C:
        c = max(C, key=lambda o: C[o][0])
        c_score, c_hop = C.pop(c)

        worst = min(W.values()) if W else 0.0
        if len(W) >= ef and c_score <= worst:               # S9: HNSW stop
            stop = "converged"
            break
        if c_hop >= max_hops:
            capped += 1                                     # do NOT overwrite stop
            continue

        nbrs = gt.neighbors(conn, run, c, limit=max(m * 3, 20))
        expanded += 1
        if not nbrs:
            continue

        cand = {n["ord"]: c_score * n["strength"] for n in nbrs
                if n["ord"] not in seen}
        pools.append(len(cand))
        if not cand:
            continue
        chosen, _ = boltzmann_sample(cand, n=min(m, len(cand)), T=T, seed=seed)

        for o in chosen:
            seen.add(o)
            sc = cand[o]
            worst = min(W.values()) if W else 0.0
            if len(W) < ef or sc > worst:
                W[o] = sc
                C[o] = (sc, c_hop + 1)
                hop_of[o] = c_hop + 1
                if len(W) > ef:
                    W.pop(min(W, key=lambda x: W[x]))

    # Depth is the deepest hop that SURVIVED into the result set, not the
    # deepest one popped -- a candidate expanded and then evicted did not
    # contribute to the answer and should not inflate the reported depth.
    depth = max((hop_of[o] for o in W), default=0)
    n_sampled = sum(1 for p in pools if p > m)              # where T could act
    return W, {"stop": stop, "depth": depth, "expanded": expanded,
               "seen": len(seen), "hop_capped": capped,
               "pool_median": (sorted(pools)[len(pools) // 2] if pools else 0),
               "pools_over_m": n_sampled, "pools": len(pools),
               "ef": ef, "T": T, "m": m, "anchors": [h["ord"] for h in hits],
               "anchor_mix": gt.source_mix(hits)}


def select_ring(cands: list, mix: dict, budget: int) -> list:
    """S17 pure selector: largest-remainder split of `budget` slots across
    `mix`'s sources, filled from `cands` in strength order, shortfall forfeit
    to global strength order.

    Require:  cands a list of row-dicts each with ord, score, source (source
              may be None), no duplicate ords; mix {label: anchor_count} with
              positive counts; budget >= 0.
    Guarantee: returns a subset of cands of length min(budget, len(cands)),
              ordered (-score, ord); a pure function of its arguments,
              independent of input list order; no source receives more than
              its quota unless reached through the forfeit pass; every source
              with candidates and a quota >= 1 receives at least one slot.
    """
    if budget <= 0 or not cands:
        return []
    ordered = sorted(cands, key=lambda r: (-r["score"], r["ord"]))

    A = sum(mix.values())
    if A <= 0:
        quotas = {}
    else:
        exact = {s: budget * n / A for s, n in mix.items()}
        floors = {s: int(exact[s]) for s in mix}
        remainder = budget - sum(floors.values())
        remainders = sorted(mix, key=lambda s: (-(exact[s] - floors[s]), -mix[s], s))
        quotas = dict(floors)
        for s in remainders[:max(remainder, 0)]:
            quotas[s] += 1

    by_source: dict = {}
    for r in ordered:
        by_source.setdefault(r.get("source"), []).append(r)

    chosen: list = []
    taken_ords = set()
    for s in sorted(quotas, key=lambda s: (-quotas[s], s)):
        for r in by_source.get(s, [])[:quotas[s]]:
            chosen.append(r)
            taken_ords.add(r["ord"])

    if len(chosen) < budget:
        for r in ordered:
            if len(chosen) >= budget:
                break
            if r["ord"] in taken_ords:
                continue
            chosen.append(r)
            taken_ords.add(r["ord"])

    chosen.sort(key=lambda r: (-r["score"], r["ord"]))
    return chosen[:budget]


_LM_CACHE: dict = {}    # run_id -> (ntok, post, smap), S18 query-side router
_INJ_CACHE: dict = {}   # (run_id, query, k) -> {source: [(ord, score), ...]}


def source_lm_stats(conn, run: gt.RunHandle) -> tuple[dict, dict, dict]:
    """S18: per-source unigram language model stats -- (ntok, post, smap).

    Require:  run has a persisted corpus_index (gt.corpus_index).
    Guarantee: pure per run; sum(ntok.values()) > 0 on any non-empty run;
               the keys of ntok are the source labels actually PERSISTED on
               nodes, never gt.run_sources's keys -- a labelled node with no
               tf rows must not mint a zero-token source.

    Cached in-process by run_id, one indexed `node` scan. Deliberately NOT
    persisted via gt._disk: corpus_index is already disk-cached and the only
    added cost here is one cheap query per process, so a second disk cache
    would be a speculative abstraction (Article II) for a cost that does not
    exist yet.
    """
    key = str(run.run_id)
    if key in _LM_CACHE:
        return _LM_CACHE[key]
    with conn.cursor() as cur:
        cur.execute("SELECT ord, attrs->>'source' AS s FROM node "
                    "WHERE run_id=%s", (run.run_id,))
        smap = {r["ord"]: r["s"] for r in cur.fetchall()}
    ix = gt.corpus_index(conn, run)
    ntok: dict = {}
    for o, dl in ix["dl"].items():
        s = smap.get(o)
        if s is not None:
            ntok[s] = ntok.get(s, 0.0) + float(dl)
    _LM_CACHE[key] = (ntok, ix["post"], smap)
    return _LM_CACHE[key]


def router_weights(conn, run: gt.RunHandle, query: str, tau: float = ROUTER_TAU) -> dict:
    """S18: query-side source router, w_s = softmax_tau(mean over query
    tokens of log(p_s(t) / p_corpus(t))), p_s and p_corpus counted in
    OCCURRENCES of `corpus_index["post"]` (tf), never document frequency --
    df is length-confounded across registers (a 15-token quotes chunk vs a
    200-token wiki chunk).

    Require:  run has a persisted corpus_index.
    Guarantee: returns a softmax distribution over ntok's sources, or {}
               when the query shares no token with the run's postings (S18's
               deliberate fallback: the ring then falls back to S17's anchor
               mix rather than a uniform ring).
    """
    import math
    ntok, post, smap = source_lm_stats(conn, run)
    tot = sum(ntok.values()) or 1.0
    lo = {s: 0.0 for s in ntok}
    nt = 0
    for t in set(gt.tokenize(query)):
        p = post.get(t)
        if not p:
            continue
        occ: dict = {}
        for o, f in p.items():
            s = smap.get(o)
            if s is not None:
                occ[s] = occ.get(s, 0.0) + float(f)
        gtot = sum(occ.values()) or 1.0
        pg = gtot / tot
        for s in ntok:
            ps = (occ.get(s, 0.0) + 0.5) / (ntok[s] + 0.5)
            lo[s] += math.log(ps / max(pg, 1e-12))
        nt += 1
    if nt == 0:
        return {}
    lo = {s: v / nt for s, v in lo.items()}
    if not lo:
        return {}
    mx = max(lo.values())
    e = {s: math.exp((v - mx) / max(tau, 1e-6)) for s, v in lo.items()}
    z = sum(e.values()) or 1.0
    return {s: v / z for s, v in e.items()}


def source_topk(conn, run: gt.RunHandle, query: str, k: int = INJECT_K) -> dict:
    """S18: source-bucketed BM25 top-k. No incumbent: gt.search takes its
    top-k GLOBALLY, before source is known, so a minority corpus can never be
    observed through it; gt.search is load-bearing for S15/S16 byte-identity
    and is deliberately not parameterized. This is the same scorer's formula
    (same postings, same K1=1.5/B=0.75, same idf) with two cut points instead
    of one -- bucketed by source BEFORE the cut.

    Returns {source: [(ord, score), ...]}, each list sorted (-score, ord) and
    cut at k; None-source rows are dropped.

    S19: deliberately NOT alias-expanded. It is a separate, separately-measured
    lane (S18); changing the anchor terms and the ring-injection terms in one
    task makes any diagnostic delta unattributable. If a later task does thread
    it, `_INJ_CACHE`'s key `(run_id, query, k)` must grow the flag -- a silently
    shared cache across two term sets would be a correctness bug, not a perf one.
    """
    import math
    ck = (str(run.run_id), query, k)
    if ck in _INJ_CACHE:
        return _INJ_CACHE[ck]
    if len(_INJ_CACHE) > 256:
        _INJ_CACHE.clear()
    terms = gt.tokenize(query)
    ix = gt.corpus_index(conn, run)
    N, avgdl = ix["n"], ix["avgdl"]
    score: dict = {}
    for t in set(terms):
        post = ix["post"].get(t)
        if not post:
            continue
        d = len(post)
        idf = math.log(1 + (N - d + 0.5) / (d + 0.5))
        for o, f in post.items():
            dl = ix["dl"].get(o, 1.0)
            score[o] = score.get(o, 0.0) + idf * f * (1.5 + 1) / (
                f + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    _, _, smap = source_lm_stats(conn, run)
    buckets: dict = {}
    for o, sc in score.items():
        buckets.setdefault(smap.get(o), []).append((o, sc))
    out = {s: sorted(v, key=lambda p: (-p[1], p[0]))[:k]
           for s, v in buckets.items() if s is not None}
    _INJ_CACHE[ck] = out
    return out


def router_mix(pool_sources, anchor_mix: dict | None, weights: dict) -> dict:
    """S18: pure mix builder -- eps-floors `weights`, REPLACING the anchor
    mix (the anchor-mix floor is the measured cap). No DB; the piece the unit
    tests pin without a live run.

    Require:  `weights` a non-empty dict when a router mix is wanted.
    Guarantee: every key gets a count >= 1; output is a pure function of its
               three arguments, independent of dict insertion order.
    """
    use_mix = dict(anchor_mix) if anchor_mix and len(anchor_mix) >= 2 \
        else {s: 1 for s in pool_sources if s}
    keys = {s for s in pool_sources if s} | set(use_mix)
    w = {s: max(weights.get(s, 0.0), ROUTER_EPS) for s in keys}
    tot = sum(w.values()) or 1.0
    return {s: max(int(round(1000 * v / tot)), 1) for s, v in w.items()}


def ring(conn, run: gt.RunHandle, W: dict, top: int = DEFAULT_RING_TOP,
         per: int = DEFAULT_RING_PER, mix: dict | None = None,
         query: str | None = None) -> dict:
    """S13: one degree out from the strongest `top` members of W, without a
    walk. Each parent's strongest `per` edges not already in W enter at
    score = parent score x edge strength (never above the parent). Edge table
    only (W2), deterministic. Returns {ord: score} for the new members.

    S17: WHERE `mix` (the anchor source composition) spans >= 2 sources, the
    top*per budget is allocated across sources by select_ring instead of
    filled per-parent. WHERE mix is absent or spans < 1 source, this is
    today's loop, unmoved, reached by an early return -- byte-identical dict,
    not a re-derivation that happens to agree.

    S18: WHERE `query` is given AND the run reports >= 2 source labels, the
    S17 pool is UNIONed with a per-source BM25 top-INJECT_K (source_topk) and
    allocated by the query-side router (router_weights/router_mix) instead of
    the anchor mix. WHERE `query` is None, behaviour is UNCHANGED from S17 --
    this is the S17 contract, preserved verbatim. WHERE the run reports fewer
    than 2 source labels, the ring fill is identical to today's even with a
    query present (the S16-era invariant, checked at the run level rather
    than via `len(mix)` so a query-only call still degenerates correctly)."""
    if query is None:
        if not mix or len(mix) < 2:
            out = {}
            parents = sorted(W, key=lambda o: (-W[o], o))[:max(top, 0)]
            for p in parents:
                added = 0
                for nb in gt.neighbors(conn, run, p, limit=per * 3):
                    o = nb["ord"]
                    if o in W or o in out:
                        continue
                    out[o] = W[p] * max(min(nb["strength"], 1.0), 0.0)
                    added += 1
                    if added >= per:
                        break
            return out

        # S17 allocated path. Pool fetch is S13's unchanged limit=per*3 per
        # parent; the per-parent `added >= per` early break is NOT carried
        # here -- that was a *fill* rule, not the pool, and keeping it would
        # cap the pool at the same rows today's fill already picks, leaving
        # allocation nothing to steer with.
        budget = max(top, 0) * max(per, 0)
        pool: dict = {}
        parents = sorted(W, key=lambda o: (-W[o], o))[:max(top, 0)]
        for p in parents:
            for nb in gt.neighbors(conn, run, p, limit=per * 3):
                o = nb["ord"]
                if o in W or o in pool:
                    continue
                pool[o] = {"ord": o, "score": W[p] * max(min(nb["strength"], 1.0), 0.0),
                           "source": gt.source_of(nb)}

        chosen = select_ring(list(pool.values()), mix, budget)
        return {r["ord"]: r["score"] for r in chosen}

    # query is not None: run-level single-source check, S16-era invariant.
    elig = {s: n for s, n in gt.run_sources(conn, run).items() if n > 0}
    if len(elig) < 2:
        out = {}
        parents = sorted(W, key=lambda o: (-W[o], o))[:max(top, 0)]
        for p in parents:
            added = 0
            for nb in gt.neighbors(conn, run, p, limit=per * 3):
                o = nb["ord"]
                if o in W or o in out:
                    continue
                out[o] = W[p] * max(min(nb["strength"], 1.0), 0.0)
                added += 1
                if added >= per:
                    break
        return out

    # S18: build the S17 pool (same fetch, same skip rule, no per-parent cap),
    # then UNION in the per-source injection and allocate by the router.
    budget = max(top, 0) * max(per, 0)
    pool: dict = {}
    parents = sorted(W, key=lambda o: (-W[o], o))[:max(top, 0)]
    for p in parents:
        for nb in gt.neighbors(conn, run, p, limit=per * 3):
            o = nb["ord"]
            if o in W or o in pool:
                continue
            pool[o] = {"ord": o, "score": W[p] * max(min(nb["strength"], 1.0), 0.0),
                       "source": gt.source_of(nb)}

    inj = source_topk(conn, run, query)
    gmax = max((sc for lst in inj.values() for _, sc in lst), default=1.0) or 1.0
    scale = max((r["score"] for r in pool.values()), default=1.0) or 1.0
    for s, lst in inj.items():
        for o, sc in lst:
            if o in W:
                continue
            ns = (sc / gmax) * scale
            if o in pool:
                pool[o]["score"] = max(pool[o]["score"], ns)
            else:
                pool[o] = {"ord": o, "score": ns, "source": s}

    pool_srcs = {r["source"] for r in pool.values()}
    w = router_weights(conn, run, query)
    use_mix = router_mix(pool_srcs, mix, w) if w else (mix or {s: 1 for s in pool_srcs if s})

    chosen = select_ring(list(pool.values()), use_mix, budget)
    return {r["ord"]: r["score"] for r in chosen}


def bridges(conn, run: gt.RunHandle, W: dict, pairs: int = DEFAULT_BRIDGE_PAIRS,
            damp: float = 0.4) -> tuple[dict, list]:
    """S14 (design 6.13): DISCOVERY, not scoring. For each pair among the top
    `pairs` chunks of W, find the single strongest degree-damped path over the
    WHOLE run (gt.best_path). Nodes the winning path crosses that the walk never
    retrieved enter the evidence at score = min(endpoint scores) x path weight
    -- never above either endpoint. Returns ({ord: score}, [paths])."""
    top = sorted(W, key=lambda o: (-W[o], o))[:max(pairs, 0)]
    found, paths = {}, []
    for i, a_ in enumerate(top):
        for b_ in top[i + 1:]:
            r = gt.best_path(conn, run, a_, b_, damp=damp)
            if r is None:
                continue
            path, wgt = r
            fresh = [o for o in path if o not in W]
            if not fresh:
                continue
            paths.append(path)
            sc = min(W[a_], W[b_]) * min(wgt, 1.0)
            for o in fresh:
                found[o] = max(found.get(o, 0.0), sc)
    if len(found) > MAX_BRIDGE:
        keep = sorted(found, key=lambda o: (-found[o], o))[:MAX_BRIDGE]
        found = {o: found[o] for o in keep}
        paths = [p for p in paths if all(o in W or o in found for o in p)]
    return found, paths


def ef_evidence(conn, run: gt.RunHandle, query: str, ef: int = DEFAULT_EF,
                T: float = DEFAULT_T, m: int = DEFAULT_M, k_anchor: int = DEFAULT_K_ANCHOR,
                seed: int = 0, k_comm: int = None,
                ring_top: int = DEFAULT_RING_TOP,
                ring_per: int = DEFAULT_RING_PER,
                bridge_pairs: int = DEFAULT_BRIDGE_PAIRS,
                expand_aliases: bool = False) -> tuple["Bundle", dict]:
    """evidence() with ef_search in place of fixed-hop expansion.

    Same Bundle shape so community_histogram, term_stats and the walker's
    rendering are unchanged; the second return is ef_search's telemetry (S12).
    Everything in W is the evidence -- there is no second sampling step, because
    ef IS the evidence budget and W is already the best ef reachable.

    S19: `expand_aliases` (default OFF) rides the anchor path only (ef_search
    -> anchor_hits -> gt.search) and is recorded in Bundle.params so a bundle
    says which term set produced it. The ring (source_topk) does not expand.
    """
    W, tele = ef_search(conn, run, query, ef=ef, T=T, m=m,
                        k_anchor=k_anchor, seed=seed, expand_aliases=expand_aliases)
    anchors = list(tele.get("anchors", []))
    extra = (ring(conn, run, W, top=ring_top, per=ring_per, mix=tele.get("anchor_mix"),
                  query=query)
             if W and ring_top else {})
    tele["ring"] = len(extra)                                   # S13
    if W and len({s: n for s, n in gt.run_sources(conn, run).items() if n > 0}) >= 2:
        tele["router_mix"] = router_weights(conn, run, query)   # S18 telemetry
    br, br_paths = (bridges(conn, run, W, pairs=bridge_pairs)
                    if W and bridge_pairs else ({}, []))
    br = {o: sc for o, sc in br.items() if o not in extra}
    tele["bridge"] = len(br)                                    # S14
    tele["bridge_paths"] = br_paths
    origin = {**{o: "walk" for o in W}, **{o: "ring" for o in extra},
              **{o: "bridge" for o in br}}
    W = {**W, **extra, **br}
    sampled = sorted(W, key=lambda o: (-W[o], o))
    comms = community_histogram(conn, run, sampled, k_comm=k_comm)
    b = Bundle(query=query, run_id=str(run.run_id), anchors=anchors,
               candidates=tele["seen"], sampled=sampled, communities=comms,
               enumerated=False, scores=dict(W),
               origin=origin,
               params={"ef": ef, "T": T, "m": m, "seed": seed,
                       "k_anchor": k_anchor, "k_comm": k_comm,
                       "ring_top": ring_top, "ring_per": ring_per,
                       "bridge_pairs": bridge_pairs,
                       "expand_aliases": expand_aliases, **tele})
    return b, tele


def community_histogram(conn, run: gt.RunHandle, sampled: list,
                        k_comm: int = None) -> list:
    """S5: group the sample by STORED cid, rank by hits."""
    if not sampled:
        return []
    rows = gt.communities_touched(conn, run, sampled)
    out = [{"cid": r["cid"], "hits": r["hits"], "size": r["size"],
            "keywords": list(r["keywords"][:5]),
            "medoid_text": (r["medoid_text"] or "")[:200]} for r in rows]
    return out[:k_comm] if k_comm else out


def evidence(conn, run: gt.RunHandle, query: str, n: int = DEFAULT_N,
             T: float = DEFAULT_T, seed: int = 0, k_anchor: int = DEFAULT_K_ANCHOR,
             hops: int = DEFAULT_HOPS, min_strength: float = 0.0,
             cap: int = DEFAULT_CAP, k_comm: int = None) -> Bundle:
    """The whole deterministic path, end to end. No model call."""
    scores, anchors = candidate_scores(conn, run, query, k_anchor=k_anchor,
                                       hops=hops, min_strength=min_strength,
                                       cap=cap)
    sampled, enumerated = boltzmann_sample(scores, n=n, T=T, seed=seed)
    comms = community_histogram(conn, run, sampled, k_comm=k_comm)
    return Bundle(
        query=query, run_id=str(run.run_id), anchors=anchors,
        candidates=len(scores), sampled=sampled, communities=comms,
        enumerated=enumerated,
        params={"n": n, "T": T, "seed": seed, "k_anchor": k_anchor,
                "hops": hops, "min_strength": min_strength, "cap": cap,
                "k_comm": k_comm})


def stability(conn, run: gt.RunHandle, query: str, n: int, T: float,
              k_comm: int = 3, seeds: range = range(8)) -> float:
    """Mean pairwise Jaccard of the top-k_comm community sets across seeds.

    DIAGNOSTIC ONLY -- do NOT tune against this. It was written as the tuning
    quantity and measurement showed it is not one: stability is U-shaped in n,
    maximised at both ends (near-argmax and near-enumeration), so maximising it
    selects T -> 0, i.e. it rewards not sampling at all. It also reads 1.0 for
    any query where n >= |C|, because an enumeration is trivially stable.

    Meaningless at the default T=0: argmax has no seed dependence, so this
    returns 1.0 by construction. Kept for evaluating the opt-in T > 0 path.
    """
    sets = []
    for s in seeds:
        b = evidence(conn, run, query, n=n, T=T, seed=s, k_comm=k_comm)
        sets.append(frozenset(b.top_cids))
    if len(sets) < 2:
        return 1.0
    js = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            js.append(len(sets[i] & sets[j]) / len(u) if u else 1.0)
    return sum(js) / len(js)
