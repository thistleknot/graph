"""Exercises pg_store.save()/load_edges() against a live Postgres.

Uses a stub that duck-types the exact ChunkGraph surface save() touches, so the
COPY statements, Jsonb adaptation, pgvector literal format, dimension stamping
and supersede ordering get tested without loading the embedding/Louvain stack.

Run:  pytest tests/test_pg_store.py -v      (needs `docker compose up -d`)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

import pg_store

LABEL = "pytest_pg_store"
N = 12          # schema enforces n_chunks >= 10
DIM = 8

# ---------------------------------------------------------------- the stub


class FakeGraph:
    """Duck-types ChunkGraph post-fit(). Attribute names mirror chunkgraph.py."""

    def __init__(self, dim: int = DIM, with_embeddings: bool = True):
        self.n = N
        self.chunks = [f"chunk number {i} about topic {i % 3}" for i in range(N)]
        self.docs_tok = [c.split() for c in self.chunks]
        self.doc_id = [f"doc{i // 4}" for i in range(N)]
        self.tfs = [{t: 1 for t in toks} for toks in self.docs_tok]
        self.qterms = [{"topic", f"{i % 3}"} for i in range(N)]
        self.disc = [{f"term{i}", "shared"} for i in range(N)]
        self.ingested_at = datetime.now(timezone.utc)
        self.diagnostics = {"note": "stub", "div_warn": False, "kept": N}

        # _params() reads these
        self.vocab = None
        self.avgdl = 6.0
        self.k_sigma, self.M, self.H, self.K = 1.0, 8, 4, 5
        self.LAM, self.EPS, self.phrases = 0.5, 1e-6, True

        if with_embeddings:
            rng = np.random.default_rng(0)
            E = rng.normal(size=(N, dim))
            # save()'s HNSW uses vector_ip_ops, which only ranks like cosine
            # when rows are L2-normalized -- match that precondition.
            self.E = E / np.linalg.norm(E, axis=1, keepdims=True)
            self.sim_dense = np.eye(N)
        else:
            self.sim_dense = None

    def edges(self) -> list[dict]:
        """Canonical src < dst, exactly as np.triu emits."""
        now = self.ingested_at
        # sim_sparse is always a float -- provenance='dense' means the dense
        # channel carried the edge over threshold, not that sim_sparse is
        # absent. sim_dense is all-or-nothing across the run (None only when
        # embed_fn was None), exactly as ChunkGraph.edges() emits it.
        spec = [
            (0, 1, "both", 0.90, 0.40, 0.80),
            (0, 2, "sparse", 0.50, 0.50, 0.10),
            (1, 2, "dense", 0.70, 0.05, 0.70),
            (3, 4, "both", 0.60, 0.30, 0.55),
            (5, 9, "sparse", 0.45, 0.45, 0.02),
        ]
        return [
            {
                "src": s, "dst": d, "provenance": p, "strength": st,
                "sim_sparse": sp,
                "sim_dense": dn if self.sim_dense is not None else None,
                "src_doc": self.doc_id[s], "dst_doc": self.doc_id[d],
                "valid_from": now, "valid_to": None, "ingested_at": now,
            }
            for s, d, p, st, sp, dn in spec
        ]

    def communities(self, min_size: int = 5) -> list[dict]:
        return [
            {"id": 0, "members": [0, 1, 2, 3, 4, 5],
             "keywords": ["topic", "chunk"], "medoid": 1,
             "medoid_text": self.chunks[1]},
            {"id": 1, "members": [6, 7, 8, 9, 10, 11],
             "keywords": ["number", "about"], "medoid": 7,
             "medoid_text": self.chunks[7]},
        ]


# ---------------------------------------------------------------- fixtures


TEST_DB = "graph_test"          # this suite owns a database, not just a label


def _dsn() -> str:
    """The suite's OWN database, not the dev one.

    node_embedding's dimension is stamped on the COLUMN, so it is global to a
    database. This suite's stub is 8-dim; a real ingest is 256. They cannot
    coexist -- steering hard constraint 4 from the test side. Label isolation is
    not enough; the schema itself has to be separate.
    """
    return pg_store.DSN.rsplit("/", 1)[0] + "/" + TEST_DB


def _provision() -> None:
    """Create the test database and apply sql/ migrations. Idempotent."""
    admin = pg_store.DSN
    with psycopg.connect(admin, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s",
                              (TEST_DB,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    ddl = sorted((Path(__file__).resolve().parent.parent / "sql").glob("*.sql"))
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        has = conn.execute("SELECT to_regclass('node_embedding')").fetchone()[0]
        if has is None:
            for f in ddl:
                conn.execute(f.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def db():
    try:
        _provision()
        with psycopg.connect(_dsn(), connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"no Postgres at {_dsn()}: {exc}")
    yield _dsn()
    # Teardown: drop our runs, then un-pin the embedding column so a later real
    # ingest into this dev database is not blocked by the stub's dim.
    #
    # Un-pinning is ONLY safe once no embedding rows remain. This suite is not
    # the only writer: a real run (brown-50-dual, 256-dim) leaves rows behind,
    # and un-pinning over them produces a bare `vector` column holding 256-dim
    # data -- so the NEXT stamp fails with "expected N dimensions, not 256",
    # pointing at the stub rather than at this teardown. Steering hard
    # constraint 4 is the same hazard from the other side.
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        conn.execute("DELETE FROM graph_run WHERE label LIKE %s", (LABEL + "%",))
        conn.execute("DROP INDEX IF EXISTS node_embedding_hnsw")
        if conn.execute("SELECT count(*) FROM node_embedding").fetchone()[0] == 0:
            conn.execute("ALTER TABLE node_embedding "
                         "ALTER COLUMN embedding TYPE vector")


@pytest.fixture(scope="module")
def first_run(db):
    return pg_store.save(FakeGraph(), LABEL, dsn=db)


# ---------------------------------------------------------------- tests


def test_save_returns_run_id(first_run):
    assert len(first_run) == 36 and first_run.count("-") == 4


def test_nodes_and_jsonb_attrs(db, first_run):
    with psycopg.connect(db) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM node WHERE run_id = %s", (first_run,))
        assert cur.fetchone()[0] == N

        cur.execute("""SELECT attrs->>'n_tok', attrs->'tf', attrs->'qterms',
                              length(chunk_hash), body
                         FROM node WHERE run_id = %s AND ord = 0""", (first_run,))
        n_tok, tf, qterms, hash_len, body = cur.fetchone()
        assert int(n_tok) == 6
        assert tf["chunk"] == 1
        assert set(qterms) == {"topic", "0"}
        assert hash_len == 16                     # blake2b digest_size=16
        assert body.startswith("chunk number 0")

        # the jsonb containment path the cookbook queries rely on
        cur.execute("""SELECT count(*) FROM node
                        WHERE run_id = %s AND attrs->'tf' ? 'topic'""",
                    (first_run,))
        assert cur.fetchone()[0] == N


def test_edges_roundtrip_shape(db, first_run):
    out = pg_store.load_edges(LABEL, dsn=db)
    src = FakeGraph().edges()
    assert len(out) == len(src)

    got = {(e["src"], e["dst"]): e for e in out}
    for e in src:
        r = got[(e["src"], e["dst"])]
        assert r["provenance"] == e["provenance"]
        assert r["strength"] == pytest.approx(e["strength"])
        assert r["src_doc"] == e["src_doc"] and r["dst_doc"] == e["dst_doc"]
        assert r["valid_to"] is None
        # NULL sims must round-trip as None, not 0.0
        if e["sim_dense"] is None:
            assert r["sim_dense"] is None
        else:
            assert r["sim_dense"] == pytest.approx(e["sim_dense"])


def test_edge_sym_view_doubles(db, first_run):
    with psycopg.connect(db) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM edge_sym WHERE run_id = %s", (first_run,))
        assert cur.fetchone()[0] == 2 * 5


def test_communities_generated_size(db, first_run):
    with psycopg.connect(db) as conn, conn.cursor() as cur:
        cur.execute("""SELECT cid, size, cardinality(members), keywords, medoid
                         FROM community WHERE run_id = %s ORDER BY cid""",
                    (first_run,))
        rows = cur.fetchall()
    assert len(rows) == 2
    for cid, size, card, keywords, medoid in rows:
        assert size == card == 6              # GENERATED column agrees
        assert isinstance(keywords, list) and keywords


def test_embedding_dim_stamped_and_indexed(db, first_run):
    with psycopg.connect(db) as conn, conn.cursor() as cur:
        cur.execute("""SELECT format_type(a.atttypid, a.atttypmod)
                         FROM pg_attribute a
                        WHERE a.attrelid = 'node_embedding'::regclass
                          AND a.attname = 'embedding'""")
        assert cur.fetchone()[0] == f"vector({DIM})"

        cur.execute("SELECT count(*) FROM node_embedding WHERE run_id = %s",
                    (first_run,))
        assert cur.fetchone()[0] == N

        cur.execute("SELECT count(*) FROM pg_indexes "
                    "WHERE indexname = 'node_embedding_hnsw'")
        assert cur.fetchone()[0] == 1


def test_inner_product_ranks_self_first(db, first_run):
    """L2-normalized rows: a vector's nearest neighbour under <#> is itself."""
    cg = FakeGraph()
    probe = pg_store._vec_literal(cg.E[3])
    with psycopg.connect(db) as conn, conn.cursor() as cur:
        cur.execute("""SELECT ord FROM node_embedding
                        WHERE run_id = %s
                        ORDER BY embedding <#> %s::vector LIMIT 1""",
                    (first_run, probe))
        assert cur.fetchone()[0] == 3


def test_resave_supersedes_without_deleting(db, first_run):
    second = pg_store.save(FakeGraph(), LABEL, dsn=db)
    assert second != first_run

    with psycopg.connect(db) as conn, conn.cursor() as cur:
        # both runs survive; supersede is history, not deletion
        cur.execute("SELECT count(*) FROM graph_run WHERE label = %s", (LABEL,))
        assert cur.fetchone()[0] == 2

        # exactly one live run, and it is the new one
        cur.execute("SELECT run_id FROM live_run WHERE label = %s", (LABEL,))
        live = cur.fetchall()
        assert len(live) == 1 and str(live[0][0]) == second

        # supersede_label closes both halves: the run row and its edge intervals
        cur.execute("""SELECT superseded_at IS NOT NULL FROM graph_run
                        WHERE run_id = %s""", (first_run,))
        assert cur.fetchone()[0] is True

        cur.execute("""SELECT count(*) FROM edge
                        WHERE run_id = %s AND valid_to IS NULL""", (first_run,))
        assert cur.fetchone()[0] == 0, "old edges must be closed, not deleted"

        cur.execute("SELECT count(*) FROM edge WHERE run_id = %s", (first_run,))
        assert cur.fetchone()[0] == 5

    # load_edges follows live_run, so it now reads the new run
    assert len(pg_store.load_edges(LABEL, dsn=db)) == 5


def test_mismatched_embedding_dim_is_rejected(db, first_run):
    """Documents a real hazard: _fix_embedding_dim() stamps the column once and
    never re-checks, so a later run with a different width fails at COPY."""
    with pytest.raises(Exception):
        pg_store.save(FakeGraph(dim=DIM * 2), LABEL + "_widedim", dsn=db)

    with psycopg.connect(db, autocommit=True) as conn:
        conn.execute("DELETE FROM graph_run WHERE label = %s", (LABEL + "_widedim",))


def test_save_without_embeddings(db):
    """embed_fn=None: no E, so embed_dim/sim_dense stay NULL and no vectors
    are written. save() must not touch the embedding column at all."""
    label = LABEL + "_noembed"
    run = pg_store.save(FakeGraph(with_embeddings=False), label, dsn=db)

    with psycopg.connect(db) as conn, conn.cursor() as cur:
        cur.execute("SELECT embed_dim, params->>'has_dense' FROM graph_run "
                    "WHERE run_id = %s", (run,))
        embed_dim, has_dense = cur.fetchone()
        assert embed_dim is None
        assert has_dense == "false"

        cur.execute("SELECT count(*) FROM node_embedding WHERE run_id = %s", (run,))
        assert cur.fetchone()[0] == 0

        cur.execute("""SELECT count(*) FROM edge
                        WHERE run_id = %s AND sim_dense IS NOT NULL""", (run,))
        assert cur.fetchone()[0] == 0

        # sim_sparse is still NOT NULL-satisfiable without any dense channel
        cur.execute("""SELECT count(*) FROM edge
                        WHERE run_id = %s AND sim_sparse IS NOT NULL""", (run,))
        assert cur.fetchone()[0] == 5
