"""section_store.py -- the section map in Postgres/pgvector: nodes (text, vector, tsvector), fused significant edges, communities with summaries, and the SQL that seeds, expands and ranks.

Spec: operator 2026-10-07 ("look at louvain_pg.py and psql_graph.md; we should be leveraging these capabilities since we have pgvector hosted locally in docker"; "yes do the
obvious based on my inputs"), the persist step of playbook.md T165 (operator go 2026-10-07), approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md. Task: playbook.md T172.
Tables are NEW (sect_*); `arxiv_sect` and every lex_* table are read by nothing here and written by nothing.

    sect_build      one row per persisted map: tag, live flag, params (jsonb, incl. the centring mean the query side needs)
    sect_node       (build_id, ord) -> doc_id, section_idx, title, community, x, y, genre_frac, text (whole), emb vector(256), tsv tsvector of the body (heading stripped)
    sect_edge       (build_id, a < b) -> fused weight, dense/sparse similarity and z, significance bitmask, backbone bitmask; sect_edge_sym is the undirected view
    sect_community  (build_id, cid) -> size, genre score and flag, title, summary, status, Dunning terms (jsonb), emb vector(256) of the summary

D1  A persist is one transaction; a failure leaves nothing. A new build for a tag marks the tag's earlier builds not live and deletes nothing (supersede, never delete).
D2  Section text is stored whole. Nothing is cut on the way in or out; the 191M characters of 80,642 sections are about 200 MB before compression.
D3  The lexical arm indexes the BODY (heading stripped, section_corpus.index_text): a heading is a stop-phrase here as in the embedding.
D4  Seeds and community ranking are exact (`<=>` scan, no ANN index): 80,642 x 256 is a sequential scan, and the operator dropped HNSW for the graph. An index can be added when a scan is measured too slow.
D6  The lexical arm weights each question word by idf = ln(N / df) times its ts_rank_cd per section (document frequencies measured 2026-10-07: models 52%, evaluate 26%, language 19%, safety 2.1%).
D7  The dense arm may seed from `emb_b`, the vectors with the 8 within-paper genre axes removed (section_genre.role_axes), the question projected the same way. On the 400-query known-item
    battery (2026-10-07) hybrid recall@50 went .690 -> .748, and dense alone .427 -> .578; the paired counts are 79 against 19 and 30 against 7.
D8  A community's centrepoint is the member nearest the mean of its members' vectors (the representative, global view of that community); a section's similarity to a question is a
    cosine in the same space.
D9  Entity hops (operator 2026-10-07: "3 hops out from the initial subgraph using the entities we identified"): sections are bridged by the frozen 40,000-entity inventory of the live chunk build
    (arxiv_graph_service.frozen_inventory), re-matched on the section bodies into sect_mention; a hop is cut to n sections ranked by idf-weighted shared entities, and bridges over max_df sections
    are ignored. Spec'd in playbook T173; the recall test against a same-length RRF control is in .tmp/entity_hop_recall.py.
D10 Entity edges are a second layer with entities as nodes: `co_mention` = two entities in at least min_shared sections with positive NPMI, in sect_entity_edge with a `kind` column that keeps
    this evidence apart from the section vector edges of sect_edge (operator 2026-10-08, "entities and their edges vs vector edges"; RL_V2 measured the two rank different neighbours,
    top-5 Jaccard 0.036 against 0.008 by chance). Entities are never nodes of sect_edge and an entity vector is never an edge.
D11 A community's entities are the ones a Dunning G2 against the whole corpus marks as over-represented in it, stored beside (not inside) its Dunning words.
D5  A neighbour is ranked by edge WEIGHT (mean z over the spaces that saw the pair, plus the edge floor), the significance the map's communities were built from; the weight is returned, not hidden.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

DIM = 256
DDL = [
    """CREATE TABLE IF NOT EXISTS sect_build (
        build_id bigserial PRIMARY KEY, tag text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
        live boolean NOT NULL DEFAULT true, n_nodes int, n_edges int, n_communities int, params jsonb NOT NULL DEFAULT '{}')""",
    """CREATE TABLE IF NOT EXISTS sect_node (
        build_id bigint NOT NULL REFERENCES sect_build ON DELETE CASCADE, ord int NOT NULL, doc_id text NOT NULL, section_idx int NOT NULL,
        section_title text, community int NOT NULL, x real, y real, genre_frac real, text text NOT NULL, emb vector(%d), tsv tsvector,
        PRIMARY KEY (build_id, ord))""" % DIM,
    "ALTER TABLE sect_node ADD COLUMN IF NOT EXISTS emb_b vector(%d)" % DIM,
    "CREATE INDEX IF NOT EXISTS sect_node_doc ON sect_node (build_id, doc_id, section_idx)",
    "CREATE INDEX IF NOT EXISTS sect_node_comm ON sect_node (build_id, community)",
    "CREATE INDEX IF NOT EXISTS sect_node_tsv ON sect_node USING gin (tsv)",
    """CREATE TABLE IF NOT EXISTS sect_edge (
        build_id bigint NOT NULL REFERENCES sect_build ON DELETE CASCADE, a int NOT NULL, b int NOT NULL, weight real NOT NULL,
        dense_sim real, dense_z real, sparse_sim real, sparse_z real, sig smallint NOT NULL, backbone smallint NOT NULL,
        PRIMARY KEY (build_id, a, b), CONSTRAINT sect_edge_canonical CHECK (a < b))""",
    "CREATE INDEX IF NOT EXISTS sect_edge_b ON sect_edge (build_id, b)",
    """CREATE OR REPLACE VIEW sect_edge_sym AS
        SELECT build_id, a AS src, b AS dst, weight, dense_sim, sparse_sim, sig, backbone FROM sect_edge
        UNION ALL SELECT build_id, b, a, weight, dense_sim, sparse_sim, sig, backbone FROM sect_edge""",
    """CREATE TABLE IF NOT EXISTS sect_entity (
        build_id bigint NOT NULL REFERENCES sect_build ON DELETE CASCADE, ent_id int NOT NULL, surface text NOT NULL, n smallint NOT NULL, df int NOT NULL,
        PRIMARY KEY (build_id, ent_id))""",
    """CREATE TABLE IF NOT EXISTS sect_mention (
        build_id bigint NOT NULL REFERENCES sect_build ON DELETE CASCADE, ord int NOT NULL, ent_id int NOT NULL, cnt int NOT NULL,
        PRIMARY KEY (build_id, ord, ent_id))""",
    "CREATE INDEX IF NOT EXISTS sect_mention_ent ON sect_mention (build_id, ent_id)",
    """CREATE TABLE IF NOT EXISTS sect_community (
        build_id bigint NOT NULL REFERENCES sect_build ON DELETE CASCADE, cid int NOT NULL, size int NOT NULL, genre_score real, genre_flag boolean NOT NULL DEFAULT false,
        title text, summary text, status text, terms jsonb, emb vector(%d), PRIMARY KEY (build_id, cid))""" % DIM,
    "ALTER TABLE sect_community ADD COLUMN IF NOT EXISTS entities jsonb",
    """CREATE TABLE IF NOT EXISTS paper_title (
        arxiv_id text PRIMARY KEY, title text NOT NULL, thesis text, source text NOT NULL, loaded_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS sect_entity_edge (
        build_id bigint NOT NULL REFERENCES sect_build ON DELETE CASCADE, a int NOT NULL, b int NOT NULL, kind text NOT NULL, n_shared int NOT NULL, npmi real NOT NULL,
        PRIMARY KEY (build_id, a, b, kind), CONSTRAINT sect_entity_edge_canonical CHECK (a < b))""",
    "CREATE INDEX IF NOT EXISTS sect_entity_edge_b ON sect_entity_edge (build_id, b)",
]


def vec(a: np.ndarray) -> str:
    """Guarantee: the pgvector text literal of a 1-D array."""
    return "[" + ",".join("%.6g" % x for x in a) + "]"


def or_query(question: str) -> str:
    """D3. Guarantee: a to_tsquery string that ORs the question's alphanumeric words (a natural question ANDed would match nothing); '' when it has none."""
    return " | ".join(dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]{1,}", question)))


def ensure_schema(conn) -> None:
    """Guarantee: the sect_* tables, indexes and view exist (the vector extension is already installed)."""
    for stmt in DDL:
        conn.execute(stmt)


def persist(conn, tag: str, recs: list[dict], X: np.ndarray, lab: np.ndarray, XY: np.ndarray, frac: np.ndarray, fused: dict, comm: dict, S: np.ndarray, params: dict) -> int:
    """D1, D2, D3. Require: recs/X/lab/XY/frac aligned (N rows); `fused` section_graph.fuse's output; `comm` {cid: {size, genre_score, genre_flag, title, summary, status, terms}};
    S (C, DIM) summary embeddings (a zero row = none). Guarantee: the new build_id; every row written in ONE transaction (the shared connection is autocommit, so it opens its own;
    inside the caller's transaction it is a savepoint), earlier builds of `tag` marked not live; any failure leaves nothing."""
    with conn.transaction():
        return _persist(conn, tag, recs, X, lab, XY, frac, fused, comm, S, params)


def _persist(conn, tag, recs, X, lab, XY, frac, fused, comm, S, params) -> int:
    import section_corpus as sc
    ensure_schema(conn)
    N = len(recs)
    assert X.shape == (N, DIM) and lab.shape == (N,) and XY.shape == (N, 2) and frac.shape == (N,), (X.shape, lab.shape, XY.shape, frac.shape)
    names = fused["names"]
    assert names == ["dense", "sparse"], names
    build = conn.execute("INSERT INTO sect_build (tag, live, n_nodes, n_edges, n_communities, params) VALUES (%s, false, %s, %s, %s, %s) RETURNING build_id",
                         (tag, N, len(fused["i"]), len(comm), json.dumps(params))).fetchone()[0]
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS pg_temp.sect_stage")                  # a second persist in the same transaction (the tests do this) finds the first one's stage
        cur.execute("CREATE TEMP TABLE sect_stage (ord int, body text) ON COMMIT DROP")
        with cur.copy("COPY sect_node (build_id, ord, doc_id, section_idx, section_title, community, x, y, genre_frac, text, emb) FROM STDIN") as cp:
            for o, r in enumerate(recs):
                cp.write_row((build, o, r["doc_id"], r["section_idx"], r["section_title"], int(lab[o]), float(XY[o, 0]), float(XY[o, 1]), float(frac[o]), r["text"], vec(X[o])))
        with cur.copy("COPY sect_stage (ord, body) FROM STDIN") as cp:
            for o, r in enumerate(recs):
                cp.write_row((o, sc.index_text(r)))
        cur.execute("UPDATE sect_node n SET tsv = to_tsvector('english', s.body) FROM sect_stage s WHERE n.build_id = %s AND n.ord = s.ord", (build,))
        s_d, s_s, z_d, z_s = fused["s"]["dense"], fused["s"]["sparse"], fused["z"]["dense"], fused["z"]["sparse"]
        with cur.copy("COPY sect_edge (build_id, a, b, weight, dense_sim, dense_z, sparse_sim, sparse_z, sig, backbone) FROM STDIN") as cp:
            for k in range(len(fused["i"])):
                cp.write_row((build, int(fused["i"][k]), int(fused["j"][k]), float(fused["w"][k]), float(s_d[k]), float(z_d[k]), float(s_s[k]), float(z_s[k]),
                              int(fused["prov"][k]), int(fused["back"][k])))
        with cur.copy("COPY sect_community (build_id, cid, size, genre_score, genre_flag, title, summary, status, terms, emb) FROM STDIN") as cp:
            for c, d in sorted(comm.items()):
                has = bool(np.any(S[c]))
                cp.write_row((build, c, d["size"], d.get("genre_score"), bool(d.get("genre_flag")), d.get("title"), d.get("summary"), d.get("status"),
                              json.dumps(d.get("terms") or []), vec(S[c]) if has else None))
    conn.execute("UPDATE sect_build SET live = false WHERE tag = %s", (tag,))
    conn.execute("UPDATE sect_build SET live = true WHERE build_id = %s", (build,))
    return build


# ------------------------------------------------------------------- queries ----
def live_build(conn, tag: str) -> tuple[int, dict] | None:
    """Guarantee: (build_id, params) of the tag's live build, None when there is none."""
    r = conn.execute("SELECT build_id, params FROM sect_build WHERE tag = %s AND live ORDER BY build_id DESC LIMIT 1", (tag,)).fetchone()
    return (r[0], r[1]) if r else None


def set_genre_vectors(conn, build: int, XB: np.ndarray, V: np.ndarray) -> int:
    """D7. Require: XB (N, DIM) the section vectors with the within-paper genre axes V (DIM, k) projected out and re-normalised (section_genre.remove_axes), row = ord.
    Guarantee: sect_node.emb_b holds XB for the build and sect_build.params["genre_axes"] holds V, in one transaction; returns rows updated."""
    with conn.transaction():
        ensure_schema(conn)
        conn.execute("DROP TABLE IF EXISTS pg_temp.sect_stage_b")
        conn.execute("CREATE TEMP TABLE sect_stage_b (ord int, emb vector(%d)) ON COMMIT DROP" % DIM)
        with conn.cursor() as cur:
            with cur.copy("COPY sect_stage_b (ord, emb) FROM STDIN") as cp:
                for o in range(len(XB)):
                    cp.write_row((o, vec(XB[o])))
            cur.execute("UPDATE sect_node n SET emb_b = s.emb FROM sect_stage_b s WHERE n.build_id = %s AND n.ord = s.ord", (build,))
            n = cur.rowcount
        conn.execute("UPDATE sect_build SET params = params || %s::jsonb WHERE build_id = %s", (json.dumps({"genre_axes": V.tolist()}), build))
    return n


def seeds_dense(conn, build: int, q: np.ndarray, n: int, col: str = "emb") -> list[int]:
    """D4, D7. Require: q centred and L2-normalised like the stored vectors of `col` ('emb', or 'emb_b' with the genre axes removed from q as well). Guarantee: ords of the n
    nearest sections by cosine, best first."""
    assert col in ("emb", "emb_b"), col
    return [r[0] for r in conn.execute("SELECT ord FROM sect_node WHERE build_id = %s ORDER BY " + col + " <=> %s::vector LIMIT %s", (build, vec(q), n)).fetchall()]


def word_df(conn, build: int, question: str) -> list[tuple[str, int]]:
    """D6. Guarantee: [(word, document frequency)] of the question's words that the index holds (a stop word or an unseen word has none), rarest first, ties in question order."""
    df = []
    for w in dict.fromkeys(or_query(question).split(" | ")):
        if w:
            d = conn.execute("SELECT count(*) FROM sect_node WHERE build_id = %s AND tsv @@ to_tsquery('english', %s)", (build, w)).fetchone()[0]
            if d:
                df.append((w, d))
    return sorted(df, key=lambda t: t[1])


def seeds_lexical(conn, build: int, question: str, n: int) -> list[int]:
    """D3, D6. Guarantee: ords of up to n sections matching any word of the question, best first by the sum over the matched words of idf x ts_rank_cd (idf = ln(N / df)); [] when no
    word is indexed. ts_rank_cd alone has no idf: "models" (in 52% of sections) buried "safety" (2.1%) under reference lists that repeat it. A hard cut on rare words was tried and
    lost: on "long-term memory for LLM agents" every topic word is common in this corpus, the cut kept the wrong ones, and the arm returned MCMC sections."""
    words = word_df(conn, build, question)
    if not words:
        return []
    N = conn.execute("SELECT n_nodes FROM sect_build WHERE build_id = %s", (build,)).fetchone()[0]
    vals = ", ".join(["(%s, %s::float8)"] * len(words))
    args = [x for w, d in words for x in (w, float(np.log(N / d)))]
    return [r[0] for r in conn.execute(
        "SELECT n.ord FROM sect_node n JOIN (VALUES " + vals + ") t(w, idf) ON n.tsv @@ to_tsquery('english', t.w) WHERE n.build_id = %s "
        "GROUP BY n.ord ORDER BY sum(t.idf * ts_rank_cd(n.tsv, to_tsquery('english', t.w), 1)) DESC, n.ord LIMIT %s", (*args, build, n)).fetchall()]


def neighbours(conn, build: int, seeds: list[int], per_seed: int) -> dict[int, list[dict]]:
    """D5. Guarantee: {seed: [{ord, weight, dense_sim, sparse_sim, sig}]}: each seed's `per_seed` strongest edges to a section that is not itself a seed, strongest first."""
    rows = conn.execute(
        """SELECT seed, nb, weight, dense_sim, sparse_sim, sig FROM (
             SELECT s.seed, e.dst AS nb, e.weight, e.dense_sim, e.sparse_sim, e.sig,
                    row_number() OVER (PARTITION BY s.seed ORDER BY e.weight DESC, e.dst) AS rn
             FROM unnest(%s::int[]) AS s(seed) JOIN sect_edge_sym e ON e.build_id = %s AND e.src = s.seed
             WHERE e.dst <> ALL(%s::int[])) t WHERE rn <= %s ORDER BY seed, rn""", (seeds, build, seeds, per_seed)).fetchall()
    out: dict[int, list[dict]] = {s: [] for s in seeds}
    for seed, nb, w, ds, ss_, sig in rows:
        out[seed].append({"ord": nb, "weight": w, "dense_sim": ds, "sparse_sim": ss_, "sig": sig})
    return out


def communities_by_summary(conn, build: int, q: np.ndarray, n: int, exclude_genre: bool = False) -> list[int]:
    """D4. Guarantee: cids of the n communities whose summary embedding is nearest q, best first; communities without a summary, and communities of a single section, are never returned."""
    return [r[0] for r in conn.execute(
        "SELECT cid FROM sect_community WHERE build_id = %s AND emb IS NOT NULL AND status = 'draft' AND size > 1 AND (NOT %s OR NOT genre_flag) ORDER BY emb <=> %s::vector LIMIT %s",
        (build, exclude_genre, vec(q), n)).fetchall()]


def persist_mentions(conn, build: int, surfaces: list[tuple[str, int]], rows: list[tuple[int, int, int]]) -> tuple[int, int]:
    """D9. Require: surfaces[e] = (surface, words) of entity e; rows = (section ord, entity e, count). Guarantee: the build's sect_mention holds exactly `rows` and sect_entity the entities that occur,
    each with df = the number of sections that mention it, in one transaction (an earlier set for the build is replaced); returns (mention rows, entities that occur)."""
    with conn.transaction():
        ensure_schema(conn)
        conn.execute("DELETE FROM sect_mention WHERE build_id = %s", (build,))
        conn.execute("DELETE FROM sect_entity WHERE build_id = %s", (build,))
        with conn.cursor() as cur:
            with cur.copy("COPY sect_mention (build_id, ord, ent_id, cnt) FROM STDIN") as cp:
                for o, e, c in rows:
                    cp.write_row((build, int(o), int(e), int(c)))
        occurs = sorted({int(e) for _, e, _ in rows})
        with conn.cursor() as cur:
            with cur.copy("COPY sect_entity (build_id, ent_id, surface, n, df) FROM STDIN") as cp:
                for e in occurs:
                    cp.write_row((build, e, surfaces[e][0], surfaces[e][1], 0))
        conn.execute("UPDATE sect_entity e SET df = d.df FROM (SELECT ent_id, count(*) AS df FROM sect_mention WHERE build_id = %s GROUP BY ent_id) d "
                     "WHERE e.build_id = %s AND e.ent_id = d.ent_id", (build, build))
    return len(rows), len(occurs)


def entity_hop(conn, build: int, frontier: list[int], reached: list[int], max_df: int, n: int) -> list[dict]:
    """D9. Guarantee: up to n sections not in `reached` that share an entity with a section of `frontier`, best first by the sum over the shared entities of ln(N / df), counting only entities
    that occur in at most `max_df` sections (a hub such as "Fig" or "language model" would bridge a third of the corpus): [{ord, score, shared}]."""
    N = conn.execute("SELECT n_nodes FROM sect_build WHERE build_id = %s", (build,)).fetchone()[0]
    rows = conn.execute(
        """WITH ent AS (SELECT DISTINCT m.ent_id FROM sect_mention m JOIN sect_entity e ON e.build_id = m.build_id AND e.ent_id = m.ent_id
                        WHERE m.build_id = %s AND m.ord = ANY(%s) AND e.df <= %s)
           SELECT m2.ord, sum(ln(%s::float8 / e.df)) AS score, count(*) AS shared
           FROM ent JOIN sect_mention m2 ON m2.build_id = %s AND m2.ent_id = ent.ent_id
                JOIN sect_entity e ON e.build_id = m2.build_id AND e.ent_id = m2.ent_id
           WHERE m2.ord <> ALL(%s) GROUP BY m2.ord ORDER BY score DESC, m2.ord LIMIT %s""", (build, frontier, max_df, N, build, reached, n)).fetchall()
    return [{"ord": o, "score": float(s), "shared": int(k)} for o, s, k in rows]


def entity_walk(conn, build: int, subgraph: list[int], hops: int = 3, n: int = 30, max_df: int = 100) -> list[list[dict]]:
    """D9. Guarantee: `hops` lists, hop h holding up to n sections first reached at hop h by entity hops from the previous hop's sections (hop 1 starts from `subgraph`, the seeds and their edge
    neighbours). The frontier of a hop is only what the previous hop added, never everything reached, and each hop is cut to n: depth alone is no bound (measured: unbounded entity hops reach
    99% of the chunks in two hops)."""
    reached, frontier, out = list(subgraph), list(subgraph), []
    for _ in range(hops):
        got = entity_hop(conn, build, frontier, reached, max_df, n) if frontier else []
        out.append(got)
        frontier = [g["ord"] for g in got]
        reached += frontier
    return out


def arxiv_id_of(doc_id: str) -> str | None:
    """D12. Guarantee: the arXiv id of a doc id ('arxiv/2504_07891' and its '_methods' extract -> '2504.07891', 'arxiv/hep-th_9901001' -> 'hep-th/9901001'), None for a book. Reuses
    domain_corpora.arxiv_url, the one place the id shapes are known."""
    import domain_corpora as dc
    url = dc.arxiv_url(doc_id)
    return url.removeprefix("https://arxiv.org/abs/") if url else None


def save_titles(conn, rows: list[tuple[str, str, str | None, str]]) -> int:
    """D12 (operator 2026-10-08: "pull all those arxiv id titles ... prefilled so we don't have to do it fresh every run"). Require: rows = (arxiv_id, title, thesis or None, source). Guarantee: each
    arXiv id the table does not hold yet is stored, in one transaction; one it already holds is left as it is (the first source to give a title wins, so the order of the calls is the priority);
    returns how many were new."""
    new = 0
    with conn.transaction():
        for r in rows:
            new += conn.execute("INSERT INTO paper_title (arxiv_id, title, thesis, source) VALUES (%s, %s, %s, %s) ON CONFLICT (arxiv_id) DO NOTHING", r).rowcount
    return new


def paper_titles(conn, doc_ids: list[str]) -> dict[str, str]:
    """D12. Guarantee: {doc_id: the paper's title} for each doc id whose paper the table holds. A book, or a paper no source gave a title for, is absent: it is shown by its key alone."""
    ids = {d: arxiv_id_of(d) for d in doc_ids}
    got = dict(conn.execute("SELECT arxiv_id, title FROM paper_title WHERE arxiv_id = ANY(%s)", ([i for i in set(ids.values()) if i],)).fetchall())
    return {d: got[i] for d, i in ids.items() if i in got}


def community_catalogue(conn, build: int) -> list[tuple[int, int, str]]:
    """Guarantee: [(community, size, title)] of every community of more than one section that has a draft summary, largest first (ties by community)."""
    return [(c, n, t) for c, n, t in conn.execute(
        "SELECT cid, size, title FROM sect_community WHERE build_id = %s AND status = 'draft' AND size > 1 ORDER BY size DESC, cid", (build,)).fetchall()]


def subgraph_entities(conn, build: int, ords: list[int], top: int = 6, max_df: int = 2000) -> list[tuple[str, int]]:
    """D11. Guarantee: up to `top` (surface, sections of `ords` that mention it), the entities that most characterise this set of sections: ranked by (sections in the set) x ln(N / df) so a hub
    such as "model" does not lead, counting only entities in at most `max_df` sections; ties by surface. An entity whose words are a subset or a superset of an already chosen one is skipped
    (measured 2026-10-08: "speculative decoding 20 . speculative 11 . speculative decoding methods 7" is one entity three times). The mini composition of a retrieved subgraph."""
    N = conn.execute("SELECT n_nodes FROM sect_build WHERE build_id = %s", (build,)).fetchone()[0]
    rows = conn.execute(
        """SELECT e.surface, count(*) AS n FROM sect_mention m JOIN sect_entity e ON e.build_id = m.build_id AND e.ent_id = m.ent_id AND e.df <= %s
           WHERE m.build_id = %s AND m.ord = ANY(%s) GROUP BY e.ent_id, e.surface, e.df ORDER BY count(*) * ln(%s::float8 / e.df) DESC, e.surface LIMIT %s""",
        (max_df, build, ords, N, 8 * top)).fetchall()
    return distinct_entities(rows, top)


def distinct_entities(ranked: list[tuple[str, int]], top: int) -> list[tuple[str, int]]:
    """D11. Guarantee: the first `top` of `ranked` [(surface, count)] after skipping any surface whose words are a subset or a superset of the words of one already kept."""
    out: list[tuple[str, int]] = []
    for surface, n in ranked:
        words = set(surface.lower().split())
        if not any(words <= set(s.lower().split()) or set(s.lower().split()) <= words for s, _ in out):
            out.append((surface, int(n)))
    return out[:top]


def subgraph_entities_by_community(conn, build: int, ords: list[int], top_comms: int = 6, top: int = 4, max_df: int = 2000) -> list[dict]:
    """D11 (operator 2026-10-08: aggregate entity stats per community within the subqueries). Guarantee: [{cid, n, entities}] for the `top_comms` communities that hold the most of `ords`
    (ties by community): `n` the sections of `ords` in it, `entities` its `top` characteristic entities as (surface, sections of `ords` in that community that mention it), ranked as
    subgraph_entities ranks (sections x ln(N / df), hubs over `max_df` out, subset or superset surfaces skipped)."""
    N = conn.execute("SELECT n_nodes FROM sect_build WHERE build_id = %s", (build,)).fetchone()[0]
    held = conn.execute("SELECT community, count(*) FROM sect_node WHERE build_id = %s AND ord = ANY(%s) GROUP BY community ORDER BY count(*) DESC, community LIMIT %s",
                        (build, ords, top_comms)).fetchall()
    by: dict[int, list[tuple[float, str, int]]] = {c: [] for c, _ in held}
    for c, surface, n, df in conn.execute(
            """SELECT n.community, e.surface, count(*), e.df FROM sect_mention m JOIN sect_node n ON n.build_id = m.build_id AND n.ord = m.ord AND n.community = ANY(%s)
               JOIN sect_entity e ON e.build_id = m.build_id AND e.ent_id = m.ent_id AND e.df <= %s
               WHERE m.build_id = %s AND m.ord = ANY(%s) GROUP BY n.community, e.ent_id, e.surface, e.df""", ([c for c, _ in held], max_df, build, ords)).fetchall():
        by[c].append((-n * float(np.log(N / df)), surface, int(n)))
    return [{"cid": c, "n": int(k), "entities": distinct_entities([(s, n) for _, s, n in sorted(by[c])], top)} for c, k in held]


def community_entity_stats(conn, build: int, cids: list[int] | None = None) -> dict[int, dict]:
    """D11 (operator 2026-10-08: aggregate entity stats per community within the 12 shown). Guarantee: {cid: {size, covered, mentions, distinct}} for `cids` (every community when None): the
    sections of the community, how many of them mention at least one entity, the mentions in all and the distinct entities. A community none of whose sections mention an entity has zeros."""
    out = {c: {"size": n, "covered": 0, "mentions": 0, "distinct": 0} for c, n in conn.execute(
        "SELECT cid, size FROM sect_community WHERE build_id = %s AND (%s::int[] IS NULL OR cid = ANY(%s))", (build, cids, cids)).fetchall()}
    for c, covered, mentions, distinct in conn.execute(
            """SELECT n.community, count(DISTINCT m.ord), sum(m.cnt), count(DISTINCT m.ent_id) FROM sect_mention m JOIN sect_node n ON n.build_id = m.build_id AND n.ord = m.ord
               WHERE m.build_id = %s AND (%s::int[] IS NULL OR n.community = ANY(%s)) GROUP BY n.community""", (build, cids, cids)).fetchall():
        out[c].update(covered=int(covered), mentions=int(mentions), distinct=int(distinct))
    return out


def community_entity_lines(conn, build: int, top: int = 3) -> dict[int, str]:
    """D11. Guarantee: {cid: 'surface sections . surface sections ...'} the first `top` of each community's stored characteristic entities (community_entities), for the communities that have any."""
    return {c: " · ".join("%s %d" % (e[0], e[1]) for e in ents[:top]) for c, ents in conn.execute(
        "SELECT cid, entities FROM sect_community WHERE build_id = %s AND entities IS NOT NULL", (build,)).fetchall()}


def shared_entities(conn, build: int, a: int, others: list[int], max_df: int = 100, top: int = 3) -> dict[int, list[str]]:
    """D9. Guarantee: {other: up to `top` surfaces of the entities that section `a` and that section both mention, rarest first}, counting only entities in at most `max_df` sections (the
    same bridge rule as entity_hop); an `other` sharing none is absent. This is what lets an agent see WHY an entity bridge connects two sections."""
    out: dict[int, list[str]] = {}
    for o, surface in conn.execute(
            """SELECT m2.ord, e.surface FROM sect_mention m1 JOIN sect_mention m2 ON m2.build_id = m1.build_id AND m2.ent_id = m1.ent_id AND m2.ord = ANY(%s)
               JOIN sect_entity e ON e.build_id = m1.build_id AND e.ent_id = m1.ent_id AND e.df <= %s
               WHERE m1.build_id = %s AND m1.ord = %s ORDER BY m2.ord, e.df, e.surface""", (others, max_df, build, a)).fetchall():
        if len(out.setdefault(o, [])) < top:
            out[o].append(surface)
    return out


def entity_edges(conn, build: int, min_shared: int = 5, max_df: int = 2000) -> int:
    """D10. Guarantee: the build's `co_mention` rows of sect_entity_edge are replaced by every entity pair (a < b) that occurs together in at least `min_shared` sections, both entities in at most
    `max_df` sections, kept only when NPMI = ln(p_ab / (p_a p_b)) / -ln(p_ab) is positive (p = sections / N); returns the number of edges. The `kind` column is what keeps this evidence apart
    from the section vector edges (sect_edge) and from any later relation kind."""
    N = conn.execute("SELECT n_nodes FROM sect_build WHERE build_id = %s", (build,)).fetchone()[0]
    with conn.transaction():
        conn.execute("DELETE FROM sect_entity_edge WHERE build_id = %s AND kind = 'co_mention'", (build,))
        conn.execute(
            """INSERT INTO sect_entity_edge (build_id, a, b, kind, n_shared, npmi)
               SELECT %s, p.a, p.b, 'co_mention', p.n, p.npmi FROM (
                   SELECT q.a, q.b, q.n, ln(q.n::float8 * %s / (ea.df::float8 * eb.df)) / -ln(q.n::float8 / %s) AS npmi
                   FROM (SELECT m1.ent_id AS a, m2.ent_id AS b, count(*) AS n
                         FROM sect_mention m1 JOIN sect_entity x1 ON x1.build_id = m1.build_id AND x1.ent_id = m1.ent_id AND x1.df <= %s
                              JOIN sect_mention m2 ON m2.build_id = m1.build_id AND m2.ord = m1.ord AND m2.ent_id > m1.ent_id
                              JOIN sect_entity x2 ON x2.build_id = m2.build_id AND x2.ent_id = m2.ent_id AND x2.df <= %s
                         WHERE m1.build_id = %s GROUP BY m1.ent_id, m2.ent_id HAVING count(*) >= %s AND count(*) < %s) q
                        JOIN sect_entity ea ON ea.build_id = %s AND ea.ent_id = q.a JOIN sect_entity eb ON eb.build_id = %s AND eb.ent_id = q.b) p
               WHERE p.npmi > 0""", (build, N, N, max_df, max_df, build, min_shared, N, build, build))
    return conn.execute("SELECT count(*) FROM sect_entity_edge WHERE build_id = %s AND kind = 'co_mention'", (build,)).fetchone()[0]


def community_entities(conn, build: int, top: int = 8, min_in: int = 3) -> int:
    """D11. Guarantee: sect_community.entities holds, per community, its `top` most characteristic entities as [[surface, sections in the community, sections overall, G2]], best first. G2 is the
    Dunning log-likelihood of the 2x2 table (in/out of the community x mentions/does not mention the entity); only entities over-represented in the community (a/size > df/N) and mentioned in at
    least `min_in` of its sections qualify. Returns the number of communities that got a list."""
    N = conn.execute("SELECT n_nodes FROM sect_build WHERE build_id = %s", (build,)).fetchone()[0]
    with conn.transaction():
        conn.execute("UPDATE sect_community SET entities = NULL WHERE build_id = %s", (build,))
        conn.execute(
            """WITH c AS (SELECT community AS cid, count(*)::float8 AS size FROM sect_node WHERE build_id = %(b)s GROUP BY community),
                    o AS (SELECT n.community AS cid, m.ent_id, count(*)::float8 AS a FROM sect_mention m JOIN sect_node n ON n.build_id = m.build_id AND n.ord = m.ord
                          WHERE m.build_id = %(b)s GROUP BY n.community, m.ent_id HAVING count(*) >= %(min_in)s),
                    t AS (SELECT o.cid, e.surface, o.a, e.df::float8 AS df, o.a AS k11, c.size - o.a AS k12, e.df - o.a AS k21, %(N)s - c.size - e.df + o.a AS k22, c.size
                          FROM o JOIN c USING (cid) JOIN sect_entity e ON e.build_id = %(b)s AND e.ent_id = o.ent_id WHERE o.a * %(N)s > e.df * c.size),
                    g AS (SELECT cid, surface, a, df, 2 * (k11 * ln(k11 * %(N)s / (size * df))
                              + CASE WHEN k12 > 0 THEN k12 * ln(k12 * %(N)s / (size * (%(N)s - df))) ELSE 0 END
                              + CASE WHEN k21 > 0 THEN k21 * ln(k21 * %(N)s / ((%(N)s - size) * df)) ELSE 0 END
                              + CASE WHEN k22 > 0 THEN k22 * ln(k22 * %(N)s / ((%(N)s - size) * (%(N)s - df))) ELSE 0 END) AS g2 FROM t),
                    r AS (SELECT *, row_number() OVER (PARTITION BY cid ORDER BY g2 DESC, surface) AS rk FROM g)
               UPDATE sect_community sc SET entities = x.ents
               FROM (SELECT cid, jsonb_agg(jsonb_build_array(surface, a::int, df::int, round(g2::numeric, 1)) ORDER BY rk) AS ents FROM r WHERE rk <= %(top)s GROUP BY cid) x
               WHERE sc.build_id = %(b)s AND sc.cid = x.cid""", {"b": build, "N": float(N), "min_in": min_in, "top": top})
    return conn.execute("SELECT count(*) FROM sect_community WHERE build_id = %s AND entities IS NOT NULL", (build,)).fetchone()[0]


def centrepoints(conn, build: int, cids: list[int], col: str = "emb_b") -> dict[int, int]:
    """D8. Guarantee: {cid: ord} -- for each community, the member section nearest the community's own centroid (the mean of its members' `col` vectors): the typical section,
    not the one nearest any question. A community with no vectors in `col` is absent."""
    assert col in ("emb", "emb_b"), col
    rows = conn.execute(
        "WITH cen AS (SELECT community AS cid, avg(" + col + ") AS c FROM sect_node WHERE build_id = %s AND community = ANY(%s) GROUP BY community) "
        "SELECT DISTINCT ON (n.community) n.community, n.ord FROM sect_node n JOIN cen ON cen.cid = n.community WHERE n.build_id = %s AND n." + col + " IS NOT NULL "
        "ORDER BY n.community, n." + col + " <=> cen.c, n.ord", (build, cids, build)).fetchall()
    return dict(rows)


def similarity(conn, build: int, ords: list[int], q: np.ndarray, col: str = "emb_b") -> dict[int, float]:
    """D8. Guarantee: {ord: cosine similarity to q} in the space of `col` (q projected the same way as the stored vectors)."""
    assert col in ("emb", "emb_b"), col
    return {o: s for o, s in conn.execute("SELECT ord, 1 - (" + col + " <=> %s::vector) FROM sect_node WHERE build_id = %s AND ord = ANY(%s)", (vec(q), build, ords)).fetchall()}


def community_of(conn, build: int, ords: list[int]) -> dict[int, int]:
    """Guarantee: {ord: community}."""
    return dict(conn.execute("SELECT ord, community FROM sect_node WHERE build_id = %s AND ord = ANY(%s)", (build, ords)).fetchall())


def summaries(conn, build: int, cids: list[int]) -> dict[int, dict]:
    """Guarantee: {cid: {community, size, title, summary, genre_flag}} for the cids that have a draft summary."""
    return {c: {"community": c, "size": n, "title": t, "summary": s, "genre_flag": g} for c, n, t, s, g in conn.execute(
        "SELECT cid, size, title, summary, genre_flag FROM sect_community WHERE build_id = %s AND cid = ANY(%s) AND status = 'draft'", (build, cids)).fetchall()}


def sections(conn, build: int, ords: list[int]) -> dict[int, dict]:
    """D2. Guarantee: {ord: {doc_id, section_idx, section_title, community, text}} with the whole text."""
    return {o: {"doc_id": d, "section_idx": i, "section_title": t, "community": c, "text": x} for o, d, i, t, c, x in conn.execute(
        "SELECT ord, doc_id, section_idx, section_title, community, text FROM sect_node WHERE build_id = %s AND ord = ANY(%s)", (build, ords)).fetchall()}


# ------------------------------------------------------------------- persisting a saved map ----
def main() -> None:
    """Persist the saved arm A map (`--tag xpa`): rebuilds the fused graph from the cached significant edges (deterministic), embeds the summaries, writes one build.
    Run:  python -u tools\\section_store.py --tag xpa"""
    import argparse
    import scipy.sparse as sp
    import section_corpus as sc
    import section_embed as se
    import section_genre as sgn
    import section_graph as sg
    import section_map as smap
    import sparsevec_store as ss
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xpa")
    ap.add_argument("--genre-vectors", action="store_true", help="only add emb_b and the genre axes to the tag's live build (D7)")
    ap.add_argument("--entities", action="store_true", help="only match the live chunk build's frozen entity inventory onto the section bodies into sect_mention (D9)")
    ap.add_argument("--reuse-mentions", action="store_true", help="with --entities: read the matches saved by the last run instead of matching again (the matching takes about 6 minutes)")
    args = ap.parse_args()
    tag = args.tag
    t0 = time.time()
    log = lambda m: print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)
    os.chdir(ROOT)
    recs = sc.load()
    if args.entities:
        import arxiv_graph_service as ags
        import entity_derive as ed
        conn = ss.connect()
        build, _ = live_build(conn, tag)
        live = ss.live_build(conn, ags.LABEL)
        meta = conn.execute("SELECT params FROM lex_build WHERE build_id = %s", (live[0],)).fetchone()[0].get("entities")
        inv = ags.frozen_inventory(conn, live[0], meta)
        assert inv is not None, "the live chunk build %r has no usable frozen entity inventory" % ags.LABEL
        cache = ".tmp/sections_%s_mentions.npy" % tag
        rows, step, skipped = [], 2000, []

        def match(lo: int, hi: int) -> list[tuple[int, int, int]]:
            chunk = recs[lo:hi]
            c, e, n = ed.match_frozen(inv, ed.tokenize_corpus([sc.index_text(r) for r in chunk], [r["doc_id"] for r in chunk]))
            return [(lo + int(a), int(b), int(k)) for a, b, k in zip(c, e, n)]

        if args.reuse_mentions:
            rows = [tuple(int(x) for x in r) for r in np.load(cache)]
            log("read %d saved mentions from %s" % (len(rows), cache))
        for s in range(0, 0 if args.reuse_mentions else len(recs), step):
            try:
                rows += match(s, s + step)
            except MemoryError:                              # one section tokenised to more than memory held (2026-10-07, between sections 22,000 and 32,000): find it, say which, go on
                log("batch %d out of memory: matching its sections one at a time" % s)
                for o in range(s, min(s + step, len(recs))):
                    try:
                        rows += match(o, o + 1)
                    except MemoryError:
                        skipped.append(o)
                        log("section %d (%s#%s, %d characters) out of memory: skipped, no mentions" % (o, recs[o]["doc_id"], recs[o]["section_idx"], len(recs[o]["text"])))
            if (s // step) % 5 == 0:
                log("matched %d of %d sections, %d mentions" % (min(s + step, len(recs)), len(recs), len(rows)))
        if skipped:
            log("SKIPPED %d sections: %s" % (len(skipped), skipped))
        if not args.reuse_mentions:
            np.save(cache, np.array(rows, np.int64))                  # saved BEFORE the write: a database that is down must not cost the 6 minutes of matching again (2026-10-07)
        nm, ne = persist_mentions(conn, build, [(inv.surface[i], int(inv.n[i])) for i in range(len(inv.surface))], rows)
        log("build %d: %d mentions of %d entities in %d sections (%.0fs)" % (build, nm, ne, len({o for o, _, _ in rows}), time.time() - t0))
        return
    E = np.load(smap.DENSE_NPY)
    X, mu = se.center(E)
    if args.genre_vectors:
        paper = np.unique([r["doc_id"] for r in recs], return_inverse=True)[1]
        V, _ = sgn.role_axes(X, paper, sgn.GENRE_K)
        conn = ss.connect()
        build, _ = live_build(conn, tag)
        n = set_genre_vectors(conn, build, sgn.remove_axes(X, V), V)
        log("build %d: emb_b written for %d sections, %d genre axes stored (%.0fs)" % (build, n, V.shape[1], time.time() - t0))
        return
    Z = sp.load_npz(smap.SPARSE_NPZ).tocsr()
    z = np.load(".tmp/sections_%s_map_state.npz" % tag)
    lab, XY = z["lab"], z["XY"]
    N = len(recs)
    assert X.shape[0] == Z.shape[0] == lab.shape[0] == N, (X.shape, Z.shape, lab.shape, N)
    paper = np.unique([r["doc_id"] for r in recs], return_inverse=True)[1]
    V, _ = sgn.role_axes(X, paper, sgn.GENRE_K)
    frac = sgn.genre_fraction(X, V)
    score, flag = sgn.genre_flags(lab, frac)
    cd, ed = smap.edges_for("dense", X, smap.EDGES["dense"], group=paper)
    cs, es = smap.edges_for("sparse", Z, smap.EDGES["sparse"], group=paper)
    fused = sg.fuse(N, {"dense": {"X": X, "cut": cd, "edges": ed}, "sparse": {"X": Z, "cut": cs, "edges": es}})
    log("fused graph rebuilt: %d edges" % len(fused["i"]))
    G = fused["G"]
    assert G.shape == (N, N)
    size = np.bincount(lab)
    summ = {d["community"]: d for d in json.load(open(".tmp/sections_%s_summaries.json" % tag, encoding="utf-8"))}
    terms = {int(c): [t for t, _ in v] for c, v in json.load(open(".tmp/sections_%s_dunning_terms.json" % tag, encoding="utf-8")).items()}
    from model2vec import StaticModel
    model = StaticModel.from_pretrained(os.path.expanduser("~/models/m2v-jina-v5-nano-256"))
    texts = ["%s. %s" % (summ[c]["title"], summ[c]["summary"]) if c in summ and summ[c].get("status") == "draft" else "" for c in range(len(size))]
    S, _ = se.center(se.pool(model, texts), mu)
    S[[not t for t in texts]] = 0.0
    comm = {c: {"size": int(size[c]), "genre_score": float(score[c]), "genre_flag": bool(flag[c]), "title": summ.get(c, {}).get("title"),
                "summary": summ.get(c, {}).get("summary"), "status": summ.get(c, {}).get("status"), "terms": terms.get(c, [])} for c in range(len(size))}
    params = {"mu": [float(x) for x in mu], "dim": DIM, "genre_threshold": sgn.GENRE_THRESHOLD, "genre_min_size": sgn.GENRE_MIN_SIZE, "k_sigma": smap.K_SIGMA,
              "resolution": float(z["chosen"]), "seed_ari": float(z["cons_ari"]), "source_state": ".tmp/sections_%s_map_state.npz" % tag, "model": "m2v-jina-v5-nano-256"}
    conn = ss.connect()
    build = persist(conn, tag, recs, X, lab, XY, frac, fused, comm, S, params)
    log("persisted build %d: %d sections, %d edges, %d communities (%.0fs)" % (build, N, len(fused["i"]), len(comm), time.time() - t0))


if __name__ == "__main__":
    main()
