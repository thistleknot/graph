"""entities.py -- Entities v0: bipartite entity/mention store beside the chunk
graph, built over the run's EXISTING salient vocabulary (node.attrs->'tf'),
plus resolution v1: deterministic canonical ids over that population (E6-E8),
plus class assignment: a co-mention Louvain partition over the SAME
population (E15/E17/E19).

Spec: .spec/specs/graph-explorer/design.md §6.15 blocks C + D, guards E1-E9;
      §6.24 E15 (class graph), E17 (consumers), E19 (cost bound)
Task: playbook.md T4 (v0), T19 (resolution v1), T24 (E9 scale bound), T70 (classes)

E1  Entities v0 SHALL be populated from the run's EXISTING salient/phrase vocabulary
    (the terms already persisted in node.attrs->'tf'), typed "term_v0", by
    deterministic NLP only -- no LLM call, no NER dependency. v0 exists to fix the
    SHAPE, a bipartite entity-mention store beside the chunk graph, not to improve
    extraction; real NER is v1 and SHALL replace the population without changing
    these tables.

E2  Every entities table SHALL be run-scoped with run_id as the first column and the
    leading key predicate (W1), and rows SHALL cascade with graph_run. Chunk identity
    SHALL be the run-local node ordinal `ord`, the key node, community.members and
    the walker already use; no new chunk id space SHALL be minted (X1's lesson one
    level down).

E3  The builder SHALL be READ-ONLY against node, edge and community -- it reads
    node.attrs and writes only entities, mentions and entity_edges -- and SHALL be
    re-runnable: a second build over the same run REPLACES that run's rows and leaves
    every other run's rows untouched.

E4  WHERE a pair of entities co-occurs in fewer than MIN_JOINT_CHUNKS (= 5) chunks,
    NO entity_edges row SHALL be written. NPMI on a joint count of one or two is
    dominated by rare-event bias -- a pair occurring exactly once, together, scores
    1.0 by construction -- so an unfloored ranking is a list of hapax coincidences
    rather than of associations.

E5  entity_edges SHALL carry npmi, ppmi and bm25 on the SAME row, computed over the
    same chunk-level co-occurrence counts, so choosing between weightings is a column
    choice at read time and never a rebuild. bm25 SHALL be the symmetric sum of both
    directions and SHALL use graph_tools.search's constants (K1 = 1.5, B = 0.75), so
    entity weighting and retrieval weighting share one ruler. a < b SHALL hold on
    every row: one row per undirected pair, the same law `edge` carries.

E6  Candidate pairs SHALL come from a NORMALIZED string similarity over
    entities.name -- casefold, separators collapsed to single spaces, surrounding
    punctuation stripped -- scored by difflib.SequenceMatcher.ratio(), stdlib only,
    NO new dependency. A pair scoring below SIM_THRESHOLD (= 0.90) SHALL NOT be a
    candidate. Candidate generation SHALL be blocked by a character-3-gram
    inverted index plus the lossless length bound 2*min(la,lb)/(la+lb) >=
    SIM_THRESHOLD: an unblocked pass is O(V^2) over a vocabulary that reaches six
    figures on a full wiki run, which is not a tuning choice but the difference
    between a pass that terminates and one that does not.

E7  A candidate pair SHALL be merged ONLY WHERE it also shares at least
    MIN_SHARED_NEIGHBORS (= 2) co-occurrence neighbors, taken as each entity's top
    NEIGHBOR_TOP_K (= 25) entity_edges partners by bm25 among rows with ppmi > 0.
    String similarity alone merges "united" into "untied"; requiring a SECOND,
    independent signal -- that the two names keep the same company in the corpus --
    is what makes a false merge cost two coincidences rather than one. An entity
    with no qualifying edges SHALL merge with nothing.

E8  Resolution SHALL be additive and idempotent. canonical_id SHALL be added by
    idempotent DDL (ALTER TABLE ... ADD COLUMN IF NOT EXISTS), SHALL be non-NULL
    on every row of the run -- an unmerged entity is its own canonical -- and SHALL
    be chosen per connected component as the LONGEST name, ties broken by ascending
    name, so the value does not depend on row order, dict order, or the order pairs
    were discovered. No row and no surface form SHALL be deleted or rewritten: the
    alias set of a canonical is exactly the rows carrying its canonical_id. The
    embedding signal (model2vec, fixed threshold, deterministic) SHALL be reachable
    only through an explicit flag defaulting to OFF, and where the model or the
    dependency is absent it SHALL contribute no candidates rather than raise.

E9  WHERE the run's df-eligible pool (E4's floor) exceeds VOCAB_BOUND terms, pair
    enumeration SHALL restrict further to the top-VOCAB_BOUND terms of that pool BY
    DOCUMENT FREQUENCY, ties broken by ascending term name -- a deterministic cut
    on the pool's cardinality, never a sample. Measured motivation: unbounded
    pair_counts on mixed-full-dual (10,830 chunks) enumerated 2.47e9 pair slots,
    and raising MIN_JOINT_CHUNKS barely helps (1.75e9 at df>=50, 1.09e9 at df>=200)
    because the explosion concentrates in the highest-df terms themselves, which a
    floor cannot remove -- only a cap on how MANY are eligible bounds it.
    VOCAB_BOUND (= 65, halved once from 130 after the first bounded live build
    still overran the 10-minute budget) keeps the worst case at
    n_chunks * C(VOCAB_BOUND, 2) ~ 2.3e7 slots on this corpus size. Scoped to
    entity_edges candidate generation
    only: entities and mentions still cover the FULL vocabulary (E1, E2
    unchanged). WHERE the pool is at or below VOCAB_BOUND the cut SHALL be a
    no-op, byte-identical to today's enumeration.
    (amended 2026-09-03.) The bound covers pair enumeration but NOT resolution:
    string_candidates enumerates over every entity NAME, and the first bounded
    live mixed-full-dual build committed (275,328 entities / 5,813,717 mentions
    / 2,080 entity_edges) then hung in resolve_entities, killed at 23 min.
    resolve_entities SHALL therefore restrict its CANDIDATE population to the
    entities appearing in entity_edges for the run (either endpoint) -- 275,328
    names down to 65 here, 22.9 s -- while canonical_id assignment stays over the FULL
    population (E8 unchanged). Lossless by E7, not a behavior change: a merge
    needs MIN_SHARED_NEIGHBORS shared entity_edges neighbors, so an edge-less
    entity can never merge and its candidate pairs are only ever discarded.

E15 class_id SHALL be assigned additively (idempotent DDL, same discipline as
    canonical_id) over a co-mention graph: for each pair of entities sharing
    at least CLASS_MIN_JOINT (= 3) chunks, an edge weighted by npmi_ppmi's
    npmi (E5's one ruler, no second implementation), floored to npmi > 0.0.
    Louvain (LOUVAIN_SEED = 7, the house convention already used by
    chunkgraph.py, graph3d.py and walker_core.subgraph_louvain) partitions
    that graph; each member's class_id is the MIN entity_id in its partition,
    exactly parallel to canonical_ids picking a representative. Every entity
    with no qualifying edge is its own class (the singleton fallback) --
    class_id is non-NULL over the FULL population, never a dropped row.

E19 (class cost bound) Pair enumeration for E15 SHALL be bounded PER CHUNK,
    not corpus-wide: CLASS_CHUNK_TOPK (= 32) is the top-k entities by mention
    count kept per chunk (ties by ascending entity_id) before pairs are
    formed. Measured motivation: the committed live receipt in the E9
    docstring is 275,328 entities / 5,813,717 mentions over 10,830 chunks,
    ~537 distinct entities per chunk mean, so an unbounded `sum C(k, 2)` is
    ~1.5e9 pair slots in pure Python -- the same wall E9 hit (2.47e9 slots,
    killed at 23 min; Article VII bounds any stage to 15 minutes). E9's
    VOCAB_BOUND is corpus-wide and therefore wrong here -- it would class only
    the 65 bound-surviving terms corpus-wide. The per-chunk cut instead keeps
    the FULL population eligible: top-32 by (-cnt, entity_id) gives
    10,830 * C(32, 2) ~ 5.4e6 slots. Determinism: the cut sorts by
    (-cnt, entity_id); every downstream intermediate (pairs, edges, the
    partition) is sorted, so the same rows produce the same classes.

Require:  a live run exists for the given label (graph_tools.get_run).
Guarantee: build_entities() is read-only against node/edge/community and
          idempotent per run -- a rebuild replaces exactly that run's three
          tables' rows, byte-for-byte reproducible (deterministic entity ids),
          leaving every other run untouched (E3).

DDL lives HERE, not in sql/001_schema.sql: 001_schema.sql is an init migration
that runs once on an empty volume, so it cannot create tables on the existing
populated dev database. The three tables below are therefore created
idempotently by the builder itself (`ensure_schema`, CREATE TABLE IF NOT
EXISTS) -- design.md block C already states this is the correct call. Do not
"fix" this by moving the DDL into sql/.
"""
from __future__ import annotations

import difflib
import itertools
import math
import os
import sys
import time

import psycopg
from psycopg.rows import dict_row
from psycopg.rows import tuple_row as _tuple_row

import graph_tools as gt
import config

DSN = config.DSN
MIN_JOINT_CHUNKS = 5           # E4
ENTITY_TYPE = "term_v0"        # E1
K1, B = 1.5, 0.75              # E5: graph_tools.search's constants, one ruler

SIM_THRESHOLD = 0.90        # E6: SequenceMatcher.ratio floor for a candidate pair
MIN_SHARED_NEIGHBORS = 2    # E7: two coincidences, not one
NEIGHBOR_TOP_K = 25         # E7: neighbors per entity, by bm25 among ppmi > 0
GRAM_N = 3                  # E6: blocking key width
MAX_GRAM_BLOCK = 2000       # E6: a 3-gram in more blocks than this is a stopword-
                            # grade key and is skipped; declared lossy, bounded
EMB_THRESHOLD = 0.85        # E8: cosine floor for the OFF-by-default third signal
VOCAB_BOUND = 65            # E9: cap on the df-eligible pool's cardinality before pair
                            # enumeration. Measured: unbounded pair_counts on
                            # mixed-full-dual (10,830 chunks) hit 2.47e9 pair slots, and
                            # raising MIN_JOINT_CHUNKS barely helped (design.md §6.19) --
                            # a cap on eligible COUNT (not on df value) is what bounds
                            # the worst case, at n_chunks * C(N, 2) ~ 9.1e7 slots at 130.
                            # 130 still ran past the 10-minute budget live on
                            # mixed-full-dual (Article VII), so halved once per plan to
                            # 65 (~2.3e7 worst-case slots) -- Article VI: the pure-Python
                            # per-chunk loop, not the DB write, is the live bottleneck.

CLASS_CHUNK_TOPK = 32       # E19: per-chunk cut for co-mention enumeration.
                            # Measured basis: 5,813,717 mentions / 10,830 chunks
                            # ~ 537 entities/chunk -> ~1.5e9 unbounded pair slots
                            # (E9 killed at 2.47e9). Top-32 by mention cnt, ties by
                            # ascending entity_id, gives 10,830 * C(32,2) ~ 5.4e6
                            # slots. Corpus-wide population is UNCUT: an entity that
                            # never makes a chunk's top-32 is simply its own class
                            # (E15's singleton fallback), never a dropped row.
CLASS_MIN_JOINT = 3         # E15 support floor: >= 3 shared chunks before NPMI
LOUVAIN_SEED = 7            # house convention: chunkgraph.py:732, graph3d.py:168,
                            # walker_core.LOUVAIN_SEED

_DDL = (
    """CREATE TABLE IF NOT EXISTS entities (
        run_id    uuid    NOT NULL REFERENCES graph_run ON DELETE CASCADE,
        entity_id integer NOT NULL,
        name      text    NOT NULL,
        type      text    NOT NULL DEFAULT 'term_v0',
        PRIMARY KEY (run_id, entity_id),
        UNIQUE (run_id, name)
    )""",
    """CREATE TABLE IF NOT EXISTS mentions (
        run_id    uuid    NOT NULL,
        ord       int     NOT NULL,
        entity_id integer NOT NULL,
        cnt       integer NOT NULL,
        PRIMARY KEY (run_id, ord, entity_id),
        FOREIGN KEY (run_id, ord)       REFERENCES node (run_id, ord)          ON DELETE CASCADE,
        FOREIGN KEY (run_id, entity_id) REFERENCES entities (run_id, entity_id) ON DELETE CASCADE
    )""",
    """CREATE INDEX IF NOT EXISTS mentions_entity ON mentions (run_id, entity_id)""",
    """CREATE TABLE IF NOT EXISTS entity_edges (
        run_id uuid    NOT NULL,
        a      integer NOT NULL,
        b      integer NOT NULL,
        npmi   real    NOT NULL,
        ppmi   real    NOT NULL,
        bm25   real    NOT NULL,
        PRIMARY KEY (run_id, a, b),
        CHECK (a < b),
        FOREIGN KEY (run_id, a) REFERENCES entities (run_id, entity_id) ON DELETE CASCADE,
        FOREIGN KEY (run_id, b) REFERENCES entities (run_id, entity_id) ON DELETE CASCADE
    )""",
    """CREATE INDEX IF NOT EXISTS entity_edges_npmi ON entity_edges (run_id, npmi DESC)""",
    # No FK on canonical_id: Postgres has no ADD CONSTRAINT IF NOT EXISTS, so a
    # self-referential composite FK cannot be added idempotently in this pattern.
    """ALTER TABLE entities ADD COLUMN IF NOT EXISTS canonical_id integer""",
    """CREATE INDEX IF NOT EXISTS entities_canonical ON entities (run_id, canonical_id)""",
    # No FK on class_id either, same reason as canonical_id above.
    """ALTER TABLE entities ADD COLUMN IF NOT EXISTS class_id integer""",
    """CREATE INDEX IF NOT EXISTS entities_class ON entities (run_id, class_id)""",
)


def ensure_schema(conn) -> None:
    """Runtime-idempotent: every statement is CREATE ... IF NOT EXISTS, safe
    to run on every build (§0).

    ALTER TABLE ... ADD COLUMN IF NOT EXISTS still takes an ACCESS EXCLUSIVE
    lock on `entities` even when the column already exists (Postgres locks
    the relation before checking), which stalls behind any concurrent reader
    holding even an AccessShare lock on that table. Skipping the ALTER once
    the column is already present avoids re-acquiring that lock on every
    build/resolve/classify call -- the common case after the first run. Now
    covers both canonical_id (E8) and class_id (E15/E19).
    """
    with conn.cursor(row_factory=_tuple_row) as cur:
        cur.execute("""SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'entities'
                          AND column_name IN ('canonical_id', 'class_id')""")
        present = {r[0] for r in cur.fetchall()}
        for stmt in _DDL:
            if "ADD COLUMN IF NOT EXISTS canonical_id" in stmt and "canonical_id" in present:
                continue
            if "ADD COLUMN IF NOT EXISTS class_id" in stmt and "class_id" in present:
                continue
            cur.execute(stmt)


def read_vocab(conn, run) -> dict:
    """Wraps graph_tools.corpus_index -- reuse the cached per-run inverted
    index rather than re-querying. Transposes the term-major `post` map into
    a chunk-major `tf` map, which is the shape the builder needs.

    Returns {"tf": {ord: {term: cnt}}, "df": {term: int}, "dl": {ord: float},
             "avgdl": float, "n": int}.
    """
    ix = gt.corpus_index(conn, run)
    tf_by_ord: dict = {}
    for term, post in ix["post"].items():
        for ord_, cnt in post.items():
            tf_by_ord.setdefault(ord_, {})[term] = cnt
    return {"tf": tf_by_ord, "df": ix["df"], "dl": ix["dl"],
            "avgdl": ix["avgdl"], "n": ix["n"]}


def bm25_weight(tf: float, dl: float, df: int, n: int, avgdl: float,
                K1: float = K1, B: float = B) -> float:
    """One term's BM25 weight in one chunk (E5's w(y,c))."""
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))


def npmi_ppmi(df_a: int, df_b: int, joint: int, n: int) -> tuple[float, float]:
    """pmi = log(p(a,b) / (p(a)*p(b))); npmi = pmi / -log(p(a,b)); ppmi =
    max(pmi, 0).

    Degenerate case (decided, §5): joint == n means both terms are in every
    chunk, so p(a,b) == 1.0 and -log(p(a,b)) == 0 -- return (0.0, 0.0) rather
    than divide by zero. A pair present in every chunk distinguishes nothing
    [colloquial: convention chosen here, not a derived result].
    `joint >= MIN_JOINT_CHUNKS` guarantees p(a,b) > 0, so the low side needs
    no guard.
    """
    if joint >= n:
        return 0.0, 0.0
    p_a, p_b, p_ab = df_a / n, df_b / n, joint / n
    pmi = math.log(p_ab / (p_a * p_b))
    npmi = pmi / -math.log(p_ab)
    ppmi = max(pmi, 0.0)
    return npmi, ppmi


def pair_bm25(a, b, joint_ords: list, tf_by_ord: dict, df: dict, dl: dict,
             n: int, avgdl: float) -> float:
    """E5: symmetric sum of both directions. bm25_dir(a->b) sums w(b,c) over
    chunks containing a, and w(b,c) == 0 wherever tf(b,c) == 0, so only the
    joint chunks contribute -- this closed form over `joint_ords` IS
    bm25_dir(a->b) + bm25_dir(b->a), not a divergence from the spec."""
    df_a, df_b = df[a], df[b]
    total = 0.0
    for c in joint_ords:
        tf_a = tf_by_ord[c][a]
        tf_b = tf_by_ord[c][b]
        dl_c = dl[c]
        total += bm25_weight(tf_a, dl_c, df_a, n, avgdl)   # w(a,c)
        total += bm25_weight(tf_b, dl_c, df_b, n, avgdl)   # w(b,c)
    return total


def pair_counts(tf_by_ord: dict, df: dict, min_joint: int = MIN_JOINT_CHUNKS,
                vocab_bound: int | None = VOCAB_BOUND) -> dict:
    """Maps canonical (a, b), a < b -> list of joint ords, floor already
    applied (E4, inclusive boundary at min_joint).

    Prefilter to df >= min_joint before enumerating pairs: lossless, since
    joint(a, b) <= min(df(a), df(b)) -- a term below the floor can never
    reach it, so dropping it first is the only reason the O(V^2) pair
    enumeration stays bounded per chunk.

    E9: WHERE that floor-eligible pool still exceeds vocab_bound terms, cut
    further to the top-vocab_bound BY DOCUMENT FREQUENCY, ties broken by
    ascending term name -- deterministic, a cap on pool cardinality rather
    than on df value (a floor cannot bound the worst case; §6.19 measured
    that raising MIN_JOINT_CHUNKS barely moves the pair count). A no-op
    when the pool is already at or below the bound.
    """
    floor_ok = {t for t, d in df.items() if d >= min_joint}
    if vocab_bound is not None and len(floor_ok) > vocab_bound:
        floor_ok = set(sorted(floor_ok, key=lambda t: (-df[t], t))[:vocab_bound])
    pairs: dict = {}
    for ord_, terms in tf_by_ord.items():
        present = sorted(t for t in terms if t in floor_ok)
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                pairs.setdefault((a, b), []).append(ord_)
    return {pair: ords for pair, ords in pairs.items() if len(ords) >= min_joint}


def normalize_name(name: str) -> str:
    """E6: casefold; collapse `_ - / .` and any whitespace run to a single
    space; strip leading/trailing punctuation and spaces. Deterministic, no
    regex state."""
    s = name.casefold()
    for ch in "_-/.":
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s.strip(" \t\n\r\f\v.,;:!?'\"()[]{}")


def name_similarity(a: str, b: str) -> float:
    """E6: difflib ratio over normalized names. Callers always pass
    `(names[x], names[y])` with x < y, so symmetry is by construction."""
    return difflib.SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def _grams(s: str, n: int) -> set:
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def string_candidates(names_by_id: dict, threshold: float = SIM_THRESHOLD,
                      n: int = GRAM_N, max_block: int = MAX_GRAM_BLOCK) -> list:
    """E6: blocked candidate generation. Returns a sorted list of (a, b), a < b."""
    norm = {i: normalize_name(name) for i, name in names_by_id.items()}
    index: dict = {}
    for i, s in norm.items():
        for g in _grams(s, n):
            index.setdefault(g, []).append(i)

    seen = set()
    out = []
    for g, ids in index.items():
        if len(ids) > max_block:
            continue
        ids_sorted = sorted(ids)
        for pi, a in enumerate(ids_sorted):
            for b in ids_sorted[pi + 1:]:
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                la, lb = len(norm[a]), len(norm[b])
                if la == 0 and lb == 0:
                    bound = 1.0
                elif la + lb == 0:
                    bound = 0.0
                else:
                    bound = 2 * min(la, lb) / (la + lb)
                if bound < threshold:
                    continue
                if name_similarity(names_by_id[a], names_by_id[b]) >= threshold:
                    out.append((a, b))
    return sorted(out)


def neighbor_sets(edge_rows, top_k: int = NEIGHBOR_TOP_K) -> dict:
    """Folds already-ranked-and-truncated (x, y) rows (§5's SQL) into
    {entity_id: set(entity_id)}."""
    out: dict = {}
    for x, y in edge_rows:
        out.setdefault(x, set()).add(y)
    return out


def corroborated(candidates: list, neighbors: dict,
                 min_shared: int = MIN_SHARED_NEIGHBORS) -> list:
    """E7: keep (a, b) where the two neighbor sets share >= min_shared ids,
    excluding a and b themselves. Order preserved."""
    out = []
    for a, b in candidates:
        shared = (neighbors.get(a, set()) & neighbors.get(b, set())) - {a, b}
        if len(shared) >= min_shared:
            out.append((a, b))
    return out


def components(pairs: list, ids) -> dict:
    """Stdlib union-find: union by smaller id becomes root (deterministic,
    rank-free). Returns {entity_id: root_id} for every id in `ids`."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    return {i: find(i) for i in ids}


def canonical_ids(comp_root: dict, names_by_id: dict) -> dict:
    """E8: per connected component, canonical = longest name, ties by
    ascending name. Returns {entity_id: canonical_id}, non-NULL for every id."""
    groups: dict = {}
    for i, root in comp_root.items():
        groups.setdefault(root, []).append(i)

    out = {}
    for root, members in groups.items():
        canon = min(members, key=lambda i: (-len(names_by_id[i]), names_by_id[i]))
        for i in members:
            out[i] = canon
    return out


def comention_pairs(mention_rows, top_k: int = CLASS_CHUNK_TOPK) -> "Counter":
    """E15/E19: DB-free. `mention_rows` is an iterable of (ord, entity_id,
    cnt) ordered by ord -- exactly `SELECT ord, entity_id, cnt FROM mentions
    WHERE run_id = %s ORDER BY ord`.

    Per chunk, keeps only the top-`top_k` entities by (-cnt, entity_id) --
    the E19 per-chunk cut that bounds `sum C(k, 2)` without dropping any
    entity from the corpus-wide population (an entity that never makes a
    chunk's top-k is simply its own class downstream, E15's singleton
    fallback). Pairs come out canonical a < b. No floor applied here -- that
    is class_graph's job, so a caller can inspect the raw joint counts.
    """
    from collections import Counter

    pairs: Counter = Counter()
    for _ord, group in itertools.groupby(mention_rows, key=lambda r: r[0]):
        ranked = sorted(((-cnt, eid) for _o, eid, cnt in group))[:top_k]
        present = sorted(eid for _neg_cnt, eid in ranked)
        for a, b in itertools.combinations(present, 2):
            pairs[(a, b)] += 1
    return pairs


def class_graph(pairs: dict, df_by_id: dict, n_chunks: int,
                min_joint: int = CLASS_MIN_JOINT) -> dict:
    """E15: DB-free. Applies the support floor BEFORE NPMI, then reuses the
    incumbent npmi_ppmi (E5's one ruler) rather than a second implementation.

    `npmi > 0.0` is a stated design choice, not spec text -- a negative- or
    zero-association edge is not evidence of a shared class [colloquial:
    convention chosen here, matching npmi_ppmi's own note].
    """
    out: dict = {}
    for (a, b), joint in pairs.items():
        if joint < min_joint:
            continue
        npmi, _ppmi = npmi_ppmi(df_by_id[a], df_by_id[b], joint, n_chunks)
        if npmi > 0.0:
            out[(a, b)] = npmi
    return out


def class_partition(edges: dict, all_ids, seed: int = LOUVAIN_SEED) -> dict:
    """E15: DB-free (lazy-imports networkx/community, mirroring
    walker_core.subgraph_louvain's style verbatim). Every id in `all_ids` not
    present in the graph maps to itself (the singleton fallback) -- the
    return is non-NULL for every id in `all_ids`, exactly parallel to
    canonical_ids returning an entity_id for every entity.

    Fewer than 2 graph nodes or no edges short-circuits to the all-singleton
    map without importing networkx at all (mirrors subgraph_louvain's early
    return).
    """
    all_ids = list(all_ids)
    node_ids = sorted({i for pair in edges for i in pair})
    if len(node_ids) < 2 or not edges:
        return {i: i for i in all_ids}

    import networkx as nx
    import community as community_louvain

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for (a, b) in sorted(edges):
        G.add_edge(a, b, weight=edges[(a, b)])
    part = community_louvain.best_partition(G, weight="weight", random_state=seed)

    groups: dict = {}
    for i, label in part.items():
        groups.setdefault(label, []).append(i)

    class_of: dict = {}
    for members in groups.values():
        rep = min(members)
        for i in members:
            class_of[i] = rep

    return {i: class_of.get(i, i) for i in all_ids}


def assign_classes(conn, run, top_k: int = CLASS_CHUNK_TOPK,
                   min_joint: int = CLASS_MIN_JOINT, seed: int = LOUVAIN_SEED) -> dict:
    """The only writer of class_id (E15/E19). Structure mirrors
    resolve_entities line-for-line: read, DB-free compute stages, then a
    single write pass.

    E19: prints per-stage wall-clock timings, the same discipline E14
    requires of relations.py's builder -- a live receipt of where the time
    goes, since this stage is the one CLASS_CHUNK_TOPK exists to bound.
    """
    ensure_schema(conn)
    rid = run.run_id
    timings: dict = {}

    t0 = time.perf_counter()
    with conn.cursor(row_factory=_tuple_row) as cur:
        cur.execute("SELECT entity_id FROM entities WHERE run_id = %s", (rid,))
        all_ids = [r[0] for r in cur.fetchall()]

        cur.execute("SELECT count(DISTINCT ord) FROM mentions WHERE run_id = %s", (rid,))
        n_chunks = cur.fetchone()[0]

        cur.execute("SELECT entity_id, count(*) FROM mentions WHERE run_id = %s "
                   "GROUP BY entity_id", (rid,))
        df_by_id = {r[0]: r[1] for r in cur.fetchall()}

        cur.execute("SELECT ord, entity_id, cnt FROM mentions WHERE run_id = %s "
                   "ORDER BY ord", (rid,))
        mention_rows = cur           # psycopg cursors are iterable; no fetchall
        print(f"  entities={len(all_ids)} n_chunks={n_chunks}")
        pairs = comention_pairs(mention_rows, top_k=top_k)
        print(f"  mentions rows consumed, pairs={len(pairs)}")
    timings["read+pairs"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    edges = class_graph(pairs, df_by_id, n_chunks, min_joint=min_joint)
    print(f"  edges={len(edges)}")
    timings["graph"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    classes = class_partition(edges, all_ids, seed=seed)
    timings["louvain"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("UPDATE entities SET class_id = entity_id WHERE run_id = %s", (rid,))
        rows = sorted((cid, rid, eid) for eid, cid in classes.items() if cid != eid)
        if rows:
            cur.executemany(
                "UPDATE entities SET class_id = %s WHERE run_id = %s AND entity_id = %s",
                rows)
    conn.commit()
    timings["write"] = time.perf_counter() - t0

    for stage, dt in timings.items():
        print(f"  {stage:<18} {dt:7.2f}s")

    n_classes = len(set(classes.values()))
    n_classed = sum(1 for eid, cid in classes.items() if cid != eid)
    return {"entities": len(all_ids), "n_chunks": n_chunks, "pairs": len(pairs),
            "edges": len(edges), "classes": n_classes, "classed": n_classed,
            "timings": timings}


def embedding_candidates(names_by_id: dict, model_dir: str | None = None,
                         threshold: float = EMB_THRESHOLD,
                         n: int = GRAM_N, max_block: int = MAX_GRAM_BLOCK) -> list:
    """OFF-by-default third signal (E8). Any failure -- missing dependency,
    missing model, bad model_dir -- returns [] rather than raising."""
    if not model_dir:
        return []
    try:
        import numpy as np
        from model2vec import StaticModel

        ids = sorted(names_by_id)
        model = StaticModel.from_pretrained(model_dir)
        vecs = model.encode([names_by_id[i] for i in ids])
        vecs = np.asarray(vecs, dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        norm = {i: normalize_name(names_by_id[i]) for i in ids}
        index: dict = {}
        for i in ids:
            for g in _grams(norm[i], n):
                index.setdefault(g, []).append(i)

        pos = {i: k for k, i in enumerate(ids)}
        seen = set()
        out = []
        for g, gid in index.items():
            if len(gid) > max_block:
                continue
            gid_sorted = sorted(gid)
            for pi, a in enumerate(gid_sorted):
                for b in gid_sorted[pi + 1:]:
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    cos = float(np.dot(vecs[pos[a]], vecs[pos[b]]))
                    if cos >= threshold:
                        out.append((a, b))
        return sorted(out)
    except Exception:
        return []


def resolve_entities(conn, run, threshold: float = SIM_THRESHOLD,
                     min_shared: int = MIN_SHARED_NEIGHBORS,
                     top_k: int = NEIGHBOR_TOP_K,
                     use_embeddings: bool = False,
                     model_dir: str | None = None) -> dict:
    """The only writer of canonical_id (E8). Deterministic: every
    intermediate is sorted, so two runs over the same rows return an equal
    dict.

    E9 (amended): candidate generation runs over the EDGE-BEARING entities
    only -- lossless, because E7 needs >= min_shared shared entity_edges
    neighbors and an entity with no entity_edges row has none. canonical_id is
    still assigned over the full population.
    """
    ensure_schema(conn)
    rid = run.run_id

    with conn.cursor(row_factory=_tuple_row) as cur:
        cur.execute("SELECT entity_id, name FROM entities WHERE run_id = %s", (rid,))
        names_by_id = {row[0]: row[1] for row in cur.fetchall()}

        # E9 (amended): the candidate population is the edge-bearing entities
        # only. Lossless via E7 -- no entity_edges row means no shared
        # neighbor, so such a pair could never survive corroboration anyway.
        cur.execute("""
            SELECT e.entity_id, e.name FROM entities e
             WHERE e.run_id = %(rid)s
               AND EXISTS (SELECT 1 FROM entity_edges x
                            WHERE x.run_id = %(rid)s
                              AND (x.a = e.entity_id OR x.b = e.entity_id))
        """, {"rid": rid})
        cand_names = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("""
            WITH und AS (
                SELECT a AS x, b AS y, bm25 FROM entity_edges WHERE run_id = %(rid)s AND ppmi > 0
                UNION ALL
                SELECT b AS x, a AS y, bm25 FROM entity_edges WHERE run_id = %(rid)s AND ppmi > 0
            ), rk AS (
                SELECT x, y, ROW_NUMBER() OVER (PARTITION BY x ORDER BY bm25 DESC, y ASC) AS rn
                  FROM und
            )
            SELECT x, y FROM rk WHERE rn <= %(k)s ORDER BY x, rn
        """, {"rid": rid, "k": top_k})
        edge_rows = cur.fetchall()

    candidates = string_candidates(cand_names, threshold=threshold)
    if use_embeddings:
        emb = embedding_candidates(cand_names, model_dir=model_dir)
        candidates = sorted(set(candidates) | set(emb))
    neighbors = neighbor_sets(edge_rows, top_k=top_k)
    merged_pairs = corroborated(candidates, neighbors, min_shared=min_shared)
    comp_root = components(merged_pairs, names_by_id.keys())
    canon = canonical_ids(comp_root, names_by_id)

    groups: dict = {}
    for i, root in comp_root.items():
        groups.setdefault(root, []).append(i)

    with conn.cursor() as cur:
        cur.execute("UPDATE entities SET canonical_id = entity_id WHERE run_id = %s", (rid,))
        merged_rows = sorted((cid, rid, eid) for eid, cid in canon.items() if cid != eid)
        if merged_rows:
            cur.executemany(
                "UPDATE entities SET canonical_id = %s WHERE run_id = %s AND entity_id = %s",
                merged_rows)
    conn.commit()

    n_aliases = sum(1 for eid, cid in canon.items() if cid != eid)
    return {"entities": len(names_by_id), "candidates": len(candidates),
            "merged_pairs": len(merged_pairs), "clusters": len(groups),
            "aliases": n_aliases}


def build_entities(conn, run, min_joint: int = MIN_JOINT_CHUNKS, resolve: bool = True,
                   classes: bool = True) -> dict:
    """The only writer. `conn` must be a WRITABLE connection -- gt.connect()
    is read-only at the server (W3) and raises on the first CREATE TABLE.

    Read-only law (E3): the only statements issued against node/edge/community
    are the SELECTs inside gt.corpus_index. No UPDATE, INSERT or DDL on those
    tables.
    """
    ensure_schema(conn)
    v = read_vocab(conn, run)
    names = sorted(v["df"])
    eid = {name: i for i, name in enumerate(names)}   # ascending, zero-based:
                                                        # deterministic (E3)
    df_by_id = {eid[name]: v["df"][name] for name in names}
    tf_by_ord_id: dict = {
        ord_: {eid[t]: cnt for t, cnt in terms.items()}
        for ord_, terms in v["tf"].items()
    }
    pairs = pair_counts(tf_by_ord_id, df_by_id, min_joint)

    rid = run.run_id
    with conn.cursor() as cur:
        # E3: REPLACES this run's rows only -- FK order, never TRUNCATE, the
        # run_id predicate is mandatory on every DELETE.
        cur.execute("DELETE FROM entity_edges WHERE run_id = %s", (rid,))
        cur.execute("DELETE FROM mentions WHERE run_id = %s", (rid,))
        cur.execute("DELETE FROM entities WHERE run_id = %s", (rid,))

        with cur.copy("COPY entities (run_id, entity_id, name, type) FROM STDIN") as cp:
            for name in names:
                cp.write_row((rid, eid[name], name, ENTITY_TYPE))

        n_mentions = 0
        with cur.copy("COPY mentions (run_id, ord, entity_id, cnt) FROM STDIN") as cp:
            for ord_, terms in tf_by_ord_id.items():
                for tid, cnt in terms.items():
                    cp.write_row((rid, ord_, tid, int(cnt)))
                    n_mentions += 1

        with cur.copy("COPY entity_edges (run_id, a, b, npmi, ppmi, bm25) FROM STDIN") as cp:
            for (a, b), joint_ords in pairs.items():
                npmi, ppmi = npmi_ppmi(df_by_id[a], df_by_id[b], len(joint_ords), v["n"])
                bm25 = pair_bm25(a, b, joint_ords, tf_by_ord_id, df_by_id, v["dl"],
                                 v["n"], v["avgdl"])
                cp.write_row((rid, a, b, npmi, ppmi, bm25))
    conn.commit()

    result = {"n_chunks": v["n"], "entities": len(names), "mentions": n_mentions,
              "edges": len(pairs)}
    if resolve:
        result["resolution"] = resolve_entities(conn, run)
    if classes:
        # order matters: classes run AFTER resolve so a later E17 consumer
        # can join canonical_id and class_id off the same committed rows.
        result["classes"] = assign_classes(conn, run)
    return result


def main(argv: list) -> int:
    """usage: python entities.py <label> [--classes-only]

    --classes-only skips the ~20-minute entity build entirely and runs
    assign_classes alone against an already-built run -- reads only
    entities/mentions, so it is standalone-callable. A separate classes.py
    would duplicate DSN/get_run/ensure_schema/npmi_ppmi imports for one
    function, and class_id lives on the entities table whose DDL this file
    already owns -- one flag on the incumbent CLI is the smaller change
    (Article III / anti-sprawl).
    """
    if len(argv) < 2:
        print("usage: python entities.py <label> [--classes-only]", file=sys.stderr)
        return 2
    label = argv[1]
    classes_only = "--classes-only" in argv[2:]
    conn = psycopg.connect(DSN, row_factory=dict_row)
    try:
        try:
            run = gt.get_run(conn, label)
        except LookupError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if classes_only:
            cls = assign_classes(conn, run)
            print(f"{label} {run.run_id} entities={cls['entities']} "
                  f"pairs={cls['pairs']} edges={cls['edges']} "
                  f"classes={cls['classes']} classed={cls['classed']}")
            return 0
        result = build_entities(conn, run)
    finally:
        conn.close()
    res = result.get("resolution", {})
    cls = result.get("classes", {})
    print(f"{label} {run.run_id} N={result['n_chunks']} "
          f"entities={result['entities']} mentions={result['mentions']} "
          f"edges={result['edges']} "
          f"canonical={res.get('clusters', 0)} aliases={res.get('aliases', 0)} "
          f"classes={cls.get('classes', 0)} classed={cls.get('classed', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
