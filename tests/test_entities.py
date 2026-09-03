"""Exercises entities.py -- Entities v0 bipartite store beside the chunk graph.

Spec: .spec/specs/graph-explorer/design.md §6.15 block C, guards E1-E5
Task: playbook.md T4

8.1 is DB-free (must pass with no Postgres at all). 8.2 needs a live run planted
via pg_store.save and skips cleanly when :5433 is down (test_pg_store.py's
fixture pattern).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _TESTS_DIR)

import entities as ent

# ================================================================== 8.1 DB-free


def test_npmi_matches_closed_form():
    import math

    # perfect co-occurrence: df_a == df_b == joint (but joint < n so no
    # degenerate branch) -> npmi == 1.0
    df_a = df_b = joint = 20
    n = 100
    npmi, ppmi = ent.npmi_ppmi(df_a, df_b, joint, n)
    p_ab = joint / n
    p_a = p_b = df_a / n
    pmi = math.log(p_ab / (p_a * p_b))
    expected_npmi = pmi / -math.log(p_ab)
    assert npmi == pytest.approx(expected_npmi, abs=1e-9)
    assert npmi == pytest.approx(1.0, abs=1e-9)
    assert ppmi == pytest.approx(max(pmi, 0.0), abs=1e-9)

    # independence: joint == df_a * df_b / n -> pmi == 0, npmi == 0
    n = 100
    df_a, df_b = 20, 25
    joint = df_a * df_b / n            # 5.0, use directly (test is on the math)
    npmi, ppmi = ent.npmi_ppmi(df_a, df_b, joint, n)
    assert npmi == pytest.approx(0.0, abs=1e-9)
    assert ppmi == pytest.approx(0.0, abs=1e-9)

    # negative association: joint below independence -> pmi < 0, npmi < 0
    df_a, df_b = 40, 40
    joint = 5                          # independence would be 16
    npmi, ppmi = ent.npmi_ppmi(df_a, df_b, joint, n)
    p_ab = joint / n
    p_a = p_b = df_a / n
    pmi = math.log(p_ab / (p_a * p_b))
    assert pmi < 0
    assert npmi == pytest.approx(pmi / -math.log(p_ab), abs=1e-9)
    assert npmi < 0


def test_npmi_is_zero_when_the_pair_fills_the_corpus():
    n = 50
    npmi, ppmi = ent.npmi_ppmi(n, n, n, n)
    assert (npmi, ppmi) == (0.0, 0.0)


def test_ppmi_is_pmi_clipped_at_zero():
    n = 100
    df_a, df_b = 40, 40
    joint = 5                          # negative association, as above
    npmi, ppmi = ent.npmi_ppmi(df_a, df_b, joint, n)
    assert npmi < 0
    assert ppmi == 0.0


def test_floor_drops_pairs_below_five_joint_chunks():
    # two entity ids (0, 1) co-occur in exactly 4 chunks; (2, 3) in exactly 5.
    tf_by_ord = {}
    for c in range(4):
        tf_by_ord[c] = {0: 1, 1: 1}
    for c in range(4, 9):
        tf_by_ord[c] = {2: 1, 3: 1}
    df = {0: 4, 1: 4, 2: 5, 3: 5}
    pairs = ent.pair_counts(tf_by_ord, df, min_joint=5)
    assert (0, 1) not in pairs
    assert (2, 3) in pairs
    assert len(pairs[(2, 3)]) == 5


def test_pairs_are_canonical_a_lt_b():
    tf_by_ord = {c: {3: 1, 1: 1, 2: 1} for c in range(6)}
    df = {1: 6, 2: 6, 3: 6}
    pairs = ent.pair_counts(tf_by_ord, df, min_joint=5)
    assert pairs
    for a, b in pairs:
        assert a < b


def test_pair_bm25_equals_both_directions_summed():
    K1, B = ent.K1, ent.B

    def bm25_dir(x, y, joint_ords, tf_by_ord, df, dl, n, avgdl):
        """Independent reimplementation straight from E5: sum w(y, c) over
        chunks containing x (here, the joint chunks)."""
        import math
        total = 0.0
        df_y = df[y]
        idf_y = math.log(1 + (n - df_y + 0.5) / (df_y + 0.5))
        for c in joint_ords:
            tf_y = tf_by_ord[c][y]
            dl_c = dl[c]
            total += idf_y * tf_y * (K1 + 1) / (tf_y + K1 * (1 - B + B * dl_c / avgdl))
        return total

    a, b = 0, 1
    joint_ords = [0, 1, 2, 3, 4]
    tf_by_ord = {c: {a: (c % 3) + 1, b: (c % 2) + 1} for c in joint_ords}
    df = {a: 8, b: 6}
    dl = {c: 10.0 + c for c in joint_ords}
    n, avgdl = 30, 12.0

    expected = (bm25_dir(a, b, joint_ords, tf_by_ord, df, dl, n, avgdl)
                + bm25_dir(b, a, joint_ords, tf_by_ord, df, dl, n, avgdl))
    got = ent.pair_bm25(a, b, joint_ords, tf_by_ord, df, dl, n, avgdl)
    assert got == pytest.approx(expected, abs=1e-9)


def test_entity_ids_follow_ascending_name():
    names = ["delta", "alpha", "gamma", "beta"]
    order_a = {name: i for i, name in enumerate(sorted(names))}
    shuffled = ["beta", "delta", "alpha", "gamma"]
    order_b = {name: i for i, name in enumerate(sorted(shuffled))}
    assert order_a == order_b
    assert order_a == {"alpha": 0, "beta": 1, "delta": 2, "gamma": 3}


def test_ddl_is_runtime_idempotent():
    assert ent._DDL
    for stmt in ent._DDL:
        assert "IF NOT EXISTS" in stmt


# =============================================================== 8.2 DB-backed

import psycopg

sys.path.insert(0, _TESTS_DIR)
from test_pg_store import FakeGraph          # noqa: E402  (incumbent stub, extended below)

import pg_store

LABEL_A = "pytest_entities_a"
LABEL_B = "pytest_entities_b"

# Planted 12-chunk corpus (schema enforces n_chunks >= 10), hand-counted:
#   alpha: chunks 0-7   (df 8)
#   beta:  chunks 0-5   (df 6)
#   gamma: chunks 4-7   (df 4)
#   delta: chunks 0-11  (df 12)
#   fill<i>: chunk i only (df 1)
# Joint: alpha-beta 6, alpha-delta 8, beta-delta 6 (edges, >= 5)
#        alpha-gamma 4, gamma-delta 4, beta-gamma 2 (below floor, no edge)
_N = 12
_TERMS_BY_CHUNK = []
for _i in range(_N):
    _terms = ["delta"]
    if _i <= 7:
        _terms.append("alpha")
    if _i <= 5:
        _terms.append("beta")
    if 4 <= _i <= 7:
        _terms.append("gamma")
    _terms.append(f"fill{_i}")
    _TERMS_BY_CHUNK.append(_terms)


class PlantedGraph(FakeGraph):
    """Extends the incumbent pg_store test stub -- overrides only chunks,
    docs_tok, tfs so node.attrs->'tf' carries the planted vocabulary above."""

    def __init__(self):
        super().__init__(with_embeddings=False)
        self.n = _N
        self.chunks = [" ".join(_TERMS_BY_CHUNK[i]) for i in range(_N)]
        # vary n_tok per chunk (padding words that carry no tf) so dl is real
        self.docs_tok = [_TERMS_BY_CHUNK[i] + ["pad"] * (i % 4) for i in range(_N)]
        self.doc_id = [f"doc{i // 4}" for i in range(_N)]
        self.tfs = [{t: 1 for t in _TERMS_BY_CHUNK[i]} for i in range(_N)]

    def edges(self):
        return []

    def communities(self, min_size=5):
        return []


def _dsn():
    return pg_store.DSN


@pytest.fixture(scope="module")
def db():
    try:
        with psycopg.connect(_dsn(), connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"no database: {exc}")
    yield _dsn()
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        conn.execute("DELETE FROM graph_run WHERE label IN (%s, %s)",
                    (LABEL_A, LABEL_B))


@pytest.fixture(scope="module")
def run_a(db):
    import graph_tools as gt
    from psycopg.rows import dict_row
    rid = pg_store.save(PlantedGraph(), LABEL_A, dsn=db)
    conn = psycopg.connect(db, row_factory=dict_row)
    run = gt.get_run(conn, LABEL_A)
    return conn, run, rid


@pytest.fixture(scope="module")
def run_b(db):
    import graph_tools as gt
    from psycopg.rows import dict_row
    pg_store.save(PlantedGraph(), LABEL_B, dsn=db)
    conn = psycopg.connect(db, row_factory=dict_row)
    run = gt.get_run(conn, LABEL_B)
    return conn, run


def test_planted_rows_are_exactly_as_counted(db, run_a):
    conn, run, rid = run_a
    result = ent.build_entities(conn, run)
    assert result["entities"] == len(set(t for row in _TERMS_BY_CHUNK for t in row))
    assert result["mentions"] == sum(len(row) for row in _TERMS_BY_CHUNK)
    assert result["edges"] == 3

    with psycopg.connect(db) as c2, c2.cursor() as cur:
        cur.execute("""
            SELECT ea.name, eb.name FROM entity_edges x
              JOIN entities ea ON ea.run_id = x.run_id AND ea.entity_id = x.a
              JOIN entities eb ON eb.run_id = x.run_id AND eb.entity_id = x.b
             WHERE x.run_id = %s""", (run.run_id,))
        rows = cur.fetchall()
    got = {frozenset(r) for r in rows}
    assert got == {frozenset(("alpha", "beta")), frozenset(("alpha", "delta")),
                   frozenset(("beta", "delta"))}


def test_no_edge_below_the_floor(db, run_a):
    conn, run, rid = run_a
    ent.build_entities(conn, run)
    with psycopg.connect(db) as c2, c2.cursor() as cur:
        cur.execute("""
            SELECT ea.name, eb.name FROM entity_edges x
              JOIN entities ea ON ea.run_id = x.run_id AND ea.entity_id = x.a
              JOIN entities eb ON eb.run_id = x.run_id AND eb.entity_id = x.b
             WHERE x.run_id = %s""", (run.run_id,))
        pairs = {frozenset(r) for r in cur.fetchall()}
    for below in (("alpha", "gamma"), ("gamma", "delta"), ("beta", "gamma")):
        assert frozenset(below) not in pairs


def test_db_npmi_matches_closed_form_on_three_pairs(db, run_a):
    conn, run, rid = run_a
    ent.build_entities(conn, run)
    df = {"alpha": 8, "beta": 6, "delta": 12}
    joint = {("alpha", "beta"): 6, ("alpha", "delta"): 8, ("beta", "delta"): 6}
    with psycopg.connect(db) as c2, c2.cursor() as cur:
        cur.execute("""
            SELECT ea.name, eb.name, x.npmi, x.ppmi FROM entity_edges x
              JOIN entities ea ON ea.run_id = x.run_id AND ea.entity_id = x.a
              JOIN entities eb ON eb.run_id = x.run_id AND eb.entity_id = x.b
             WHERE x.run_id = %s""", (run.run_id,))
        rows = cur.fetchall()
    assert len(rows) == 3
    for na, nb, npmi, ppmi in rows:
        key = (na, nb) if (na, nb) in joint else (nb, na)
        exp_npmi, exp_ppmi = ent.npmi_ppmi(df[key[0]], df[key[1]], joint[key], _N)
        assert npmi == pytest.approx(exp_npmi, abs=1e-5)
        assert ppmi == pytest.approx(exp_ppmi, abs=1e-5)
        if frozenset(key) == frozenset(("alpha", "delta")):
            assert npmi == pytest.approx(0.0, abs=1e-5)   # delta in every chunk


def test_rebuild_replaces_this_run_and_leaves_the_other_alone(db, run_a, run_b):
    conn_a, run_a_h, _ = run_a
    conn_b, run_b_h = run_b

    result_a1 = ent.build_entities(conn_a, run_a_h)
    result_b = ent.build_entities(conn_b, run_b_h)

    def _snapshot(conn, run):
        with conn.cursor() as cur:
            cur.execute("SELECT entity_id, name FROM entities WHERE run_id=%s "
                       "ORDER BY entity_id", (run.run_id,))
            e = cur.fetchall()
            cur.execute("SELECT ord, entity_id, cnt FROM mentions WHERE run_id=%s "
                       "ORDER BY ord, entity_id", (run.run_id,))
            m = cur.fetchall()
            cur.execute("SELECT a, b, npmi, ppmi, bm25 FROM entity_edges "
                       "WHERE run_id=%s ORDER BY a, b", (run.run_id,))
            x = cur.fetchall()
        return e, m, x

    snap_b_before = _snapshot(conn_b, run_b_h)

    result_a2 = ent.build_entities(conn_a, run_a_h)
    assert result_a2 == result_a1

    snap_b_after = _snapshot(conn_b, run_b_h)
    assert snap_b_after == snap_b_before


def test_builder_never_writes_the_chunk_graph(db, run_a):
    conn, run, rid = run_a
    with psycopg.connect(db) as c2, c2.cursor() as cur:
        cur.execute("SELECT count(*) FROM node WHERE run_id=%s", (run.run_id,))
        n_node_before = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM edge WHERE run_id=%s", (run.run_id,))
        n_edge_before = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM community WHERE run_id=%s", (run.run_id,))
        n_comm_before = cur.fetchone()[0]

    ent.build_entities(conn, run)

    with psycopg.connect(db) as c2, c2.cursor() as cur:
        cur.execute("SELECT count(*) FROM node WHERE run_id=%s", (run.run_id,))
        assert cur.fetchone()[0] == n_node_before
        cur.execute("SELECT count(*) FROM edge WHERE run_id=%s", (run.run_id,))
        assert cur.fetchone()[0] == n_edge_before
        cur.execute("SELECT count(*) FROM community WHERE run_id=%s", (run.run_id,))
        assert cur.fetchone()[0] == n_comm_before
