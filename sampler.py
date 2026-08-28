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
