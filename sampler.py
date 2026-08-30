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

import math
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
MAX_HOPS    = 8            # runaway guard, NOT a depth policy (S9)
DEFAULT_RING_TOP = 3       # S13: one degree out from the top-3 of W (design 6.10)
DEFAULT_RING_PER = 8       # S13: strongest 8 new neighbours per parent
DEFAULT_CAP = 150          # R4.4 neighbourhood bound, 2-hop p99 is 111


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


def candidate_scores(conn, run: gt.RunHandle, query: str, k_anchor: int = 3,
                     hops: int = DEFAULT_HOPS, min_strength: float = 0.0,
                     cap: int = DEFAULT_CAP) -> tuple[dict, list]:
    """Induced subgraph around the query's lexical anchors.

    Score is the strength-decayed path score: an anchor starts at its
    normalised BM25 score, and each hop multiplies by the edge strength. Best
    path per node wins, which mirrors the cookbook expansion.

    Traversal is graph_tools.neighbors -- indexed src/dst only, never attrs.
    """
    hits = gt.search(conn, run, query, k=k_anchor)
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
              k_anchor: int = 3, max_hops: int = MAX_HOPS,
              seed: int = 0) -> tuple[dict, dict]:
    """Best-first expansion with an HNSW stop condition (S9-S12).

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
    """
    if ef < 1 or m < 1 or max_hops < 1:
        raise ValueError(f"ef, m, max_hops must all be >= 1 "
                         f"(got {ef}, {m}, {max_hops})")
    hits = gt.search(conn, run, query, k=k_anchor)
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
               "ef": ef, "T": T, "m": m, "anchors": [h["ord"] for h in hits]}


def ring(conn, run: gt.RunHandle, W: dict, top: int = DEFAULT_RING_TOP,
         per: int = DEFAULT_RING_PER) -> dict:
    """S13: one degree out from the strongest `top` members of W, without a
    walk. Each parent's strongest `per` edges not already in W enter at
    score = parent score x edge strength (never above the parent). Edge table
    only (W2), deterministic. Returns {ord: score} for the new members."""
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


def ef_evidence(conn, run: gt.RunHandle, query: str, ef: int = DEFAULT_EF,
                T: float = DEFAULT_T, m: int = DEFAULT_M, k_anchor: int = 3,
                seed: int = 0, k_comm: int = None,
                ring_top: int = DEFAULT_RING_TOP,
                ring_per: int = DEFAULT_RING_PER) -> tuple["Bundle", dict]:
    """evidence() with ef_search in place of fixed-hop expansion.

    Same Bundle shape so community_histogram, term_stats and the walker's
    rendering are unchanged; the second return is ef_search's telemetry (S12).
    Everything in W is the evidence -- there is no second sampling step, because
    ef IS the evidence budget and W is already the best ef reachable.
    """
    W, tele = ef_search(conn, run, query, ef=ef, T=T, m=m,
                        k_anchor=k_anchor, seed=seed)
    anchors = list(tele.get("anchors", []))
    extra = ring(conn, run, W, top=ring_top, per=ring_per) if W and ring_top else {}
    tele["ring"] = len(extra)                                   # S13
    W = {**W, **extra}
    sampled = sorted(W, key=lambda o: (-W[o], o))
    comms = community_histogram(conn, run, sampled, k_comm=k_comm)
    b = Bundle(query=query, run_id=str(run.run_id), anchors=anchors,
               candidates=tele["seen"], sampled=sampled, communities=comms,
               enumerated=False, scores=dict(W),
               params={"ef": ef, "T": T, "m": m, "seed": seed,
                       "k_anchor": k_anchor, "k_comm": k_comm,
                       "ring_top": ring_top, "ring_per": ring_per, **tele})
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
             T: float = DEFAULT_T, seed: int = 0, k_anchor: int = 3,
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
