"""entities.py -- Entities v0: bipartite entity/mention store beside the chunk
graph, built over the run's EXISTING salient vocabulary (node.attrs->'tf'),
plus resolution v1: deterministic canonical ids over that population (E6-E8).

Spec: .spec/specs/graph-explorer/design.md §6.15 blocks C + D, guards E1-E9
Task: playbook.md T4 (v0), T19 (resolution v1), T24 (E9 scale bound)

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
import math
import os
import sys

import psycopg
from psycopg.rows import dict_row
from psycopg.rows import tuple_row as _tuple_row

import graph_tools as gt

DSN = os.environ.get("CHUNKGRAPH_DSN",
                     "postgresql://graph:graph@localhost:5433/graph")
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
)


def ensure_schema(conn) -> None:
    """Runtime-idempotent: every statement is CREATE ... IF NOT EXISTS, safe
    to run on every build (§0).

    ALTER TABLE ... ADD COLUMN IF NOT EXISTS still takes an ACCESS EXCLUSIVE
    lock on `entities` even when the column already exists (Postgres locks
    the relation before checking), which stalls behind any concurrent reader
    holding even an AccessShare lock on that table. Skipping the ALTER once
    the column is already present avoids re-acquiring that lock on every
    build/resolve call -- the common case after the first run.
    """
    with conn.cursor() as cur:
        cur.execute("""SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'entities' AND column_name = 'canonical_id'""")
        has_canonical_id = cur.fetchone() is not None
        for stmt in _DDL:
            if "ADD COLUMN IF NOT EXISTS canonical_id" in stmt and has_canonical_id:
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


def build_entities(conn, run, min_joint: int = MIN_JOINT_CHUNKS, resolve: bool = True) -> dict:
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
    return result


def main(argv: list) -> int:
    if len(argv) < 2:
        print("usage: python entities.py <label>", file=sys.stderr)
        return 2
    label = argv[1]
    conn = psycopg.connect(DSN, row_factory=dict_row)
    try:
        try:
            run = gt.get_run(conn, label)
        except LookupError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        result = build_entities(conn, run)
    finally:
        conn.close()
    res = result.get("resolution", {})
    print(f"{label} {run.run_id} N={result['n_chunks']} "
          f"entities={result['entities']} mentions={result['mentions']} "
          f"edges={result['edges']} "
          f"canonical={res.get('clusters', 0)} aliases={res.get('aliases', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
