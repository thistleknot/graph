"""Exercises relations.py -- Relations v0 sentence-window relation detection.

Spec: .spec/specs/graph-explorer/design.md §6.20, guards E10-E14
Task: playbook.md T28

DB-free tests (1-7) must pass with no Postgres at all. Test 8 needs a live run
planted via pg_store.save and skips cleanly when :5433 is down (test_pg_store.py's
fixture pattern).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

import graph_tools as gt
import relations as rel
from conftest import require_dsn_db, require_gt_conn, require_run

# ================================================================== DB-free


def test_possessive_unifies_across_both_surfaces():
    vocab = {"kennedy", "plantation"}
    canon = {"kennedy": 1, "plantation": 2}

    toks_a = rel.rel_tokenize("Kennedy's plantation was large.")
    occs_a = rel.anchor(toks_a, vocab, canon)
    cands_a = rel.candidates(occs_a, toks_a)
    keys_a = {(c.src, c.dst, c.template) for c in cands_a}
    assert (1, 2, "GEN") in keys_a

    toks_b = rel.rel_tokenize("The plantation of Kennedy was large.")
    occs_b = rel.anchor(toks_b, vocab, canon)
    cands_b = rel.candidates(occs_b, toks_b)
    keys_b = {(c.src, c.dst, c.template) for c in cands_b}
    assert (1, 2, "GEN") in keys_b   # kennedy is src (possessor) in BOTH surfaces


def test_plural_possessive_and_the_case_bit():
    toks = rel.rel_tokenize("Workers' Union")
    assert toks[0] == rel.Tok("workers", True, False)
    assert toks[1] == rel.Tok(rel.POSS, False, True)
    assert toks[2] == rel.Tok("union", True, False)

    toks2 = rel.rel_tokenize("the union")
    assert toks2[-1] == rel.Tok("union", False, False)   # lowercase surface -> cap=False


def test_splitter_guards_abbreviations_initials_and_decimals():
    body = ("Mr. Smith met J. Davis near 3.5 million people today. "
            "Today they discussed plans.")
    abbrev_with_mr = rel.derive_abbreviations([body])
    assert "mr" in abbrev_with_mr

    body_bare_mr = "mr can you help me. Thanks."
    abbrev_without_mr = rel.derive_abbreviations([body, body_bare_mr])
    assert "mr" not in abbrev_without_mr

    splits_guarded = rel.split_sentences(body, None, abbrev_with_mr)
    splits_unguarded = rel.split_sentences(body, None, abbrev_without_mr)

    assert len(splits_guarded) == 2
    assert splits_guarded[0].startswith("Mr. Smith")
    assert len(splits_unguarded) == 2
    assert not splits_unguarded[0].startswith("Mr.")   # guard derived, not typed -- proven by the flip


def test_phrase_spans_stopwords():
    vocab = {"state_union"}
    canon = {"state_union": 42}
    toks = rel.rel_tokenize("The State of the Union address was long.")
    occs = rel.anchor(toks, vocab, canon)
    assert len(occs) == 1
    occ = occs[0]
    assert occ.eid == 42
    # spans from "State" through "Union", not two separate occurrences
    assert toks[occ.start].low == "state"
    assert toks[occ.end].low == "union"


def test_direction_follows_surface_order():
    vocab = {"kennedy", "oswald"}
    canon = {"kennedy": 1, "oswald": 2}

    toks = rel.rel_tokenize("Kennedy killed Oswald yesterday.")
    occs = rel.anchor(toks, vocab, canon)
    cands = rel.candidates(occs, toks)
    non_gen = [c for c in cands if c.template != "GEN"]
    assert non_gen and non_gen[0].src == 1 and non_gen[0].dst == 2

    toks_rev = rel.rel_tokenize("Oswald killed Kennedy yesterday.")
    occs_rev = rel.anchor(toks_rev, vocab, canon)
    cands_rev = rel.candidates(occs_rev, toks_rev)
    non_gen_rev = [c for c in cands_rev if c.template != "GEN"]
    assert non_gen_rev and non_gen_rev[0].src == 2 and non_gen_rev[0].dst == 1


def test_template_collapses_content_to_w():
    toks = rel.rel_tokenize("was killed in")
    assert rel.template_of(toks) == "was w in"

    src = Path(rel.__file__).read_text(encoding="utf-8")
    assert "CONNECTOR_CLOSED =" not in src   # no such constant defined; is_content is the only gate (§0)


def test_llr_gate_and_support_floor():
    events = []
    # LOW_N: n=2, dropped by MIN_REL_SUPPORT regardless of llr
    for i in range(2):
        events.append(rel.Cand(1, 2, "A", "connA", 1, i))
    # LOW_LLR: n=3 but engineered near-independence -> llr < gate
    for i in range(3):
        events.append(rel.Cand(3, 4, "B", "connB", 2, i))
    for i in range(6):
        events.append(rel.Cand(3, 4, "A2", "connA2", 3, i))
    for i in range(6):
        events.append(rel.Cand(7, 8, "B", "connB2", 4, i))
    # PASS: n=5, isolated pair/template -> high llr
    for i in range(5):
        events.append(rel.Cand(5, 6, "C", "connC", 5, i))

    sent_df = {1: 2, 2: 2, 3: 9, 4: 3, 5: 5, 6: 5, 7: 6, 8: 6}
    pair_sent_df = {frozenset((1, 2)): 2, frozenset((3, 4)): 3,
                   frozenset((7, 8)): 6, frozenset((5, 6)): 5}
    n_sent = len(events)

    rows = rel.score(events, sent_df, pair_sent_df, n_sent)
    by_key = {(r["src"], r["dst"], r["template"]): r for r in rows}

    assert (1, 2, "A") not in by_key     # support floor
    assert (3, 4, "B") not in by_key     # llr gate
    assert (5, 6, "C") in by_key

    row = by_key[(5, 6, "C")]
    assert row["n"] == 5

    key = (5, 6, "C")
    by_pair = Counter((e.src, e.dst) for e in events)
    by_tmpl = Counter(e.template for e in events)
    k11 = sum(1 for e in events if (e.src, e.dst, e.template) == key)
    k12 = by_pair[(5, 6)] - k11
    k21 = by_tmpl["C"] - k11
    k22 = len(events) - k11 - k12 - k21
    expected_llr = gt.llr(k11, k12, k21, k22)
    assert row["llr"] == pytest.approx(expected_llr, abs=1e-5)


def test_pairsets_split_gen_keys_by_connector():
    rows = [
        {"template": "GEN", "connector": "'s", "src": 1, "dst": 2},
        {"template": "GEN", "connector": "of the", "src": 3, "dst": 4},
        {"template": "in the", "connector": "in the", "src": 5, "dst": 6},
    ]
    default = rel.template_pairsets(rows)
    assert set(default) == {"GEN", "in the"}
    assert default["GEN"] == {(1, 2), (3, 4)}

    split = rel.template_pairsets(rows, split_gen=True)
    assert set(split) == {"GEN/'s", "GEN/of the", "in the"}
    assert split["GEN/'s"] == {(1, 2)}
    assert split["GEN/of the"] == {(3, 4)}


def test_similar_templates_coclass_and_floor_holds_the_rest():
    pairs = [(i, i + 1) for i in range(10)]
    pairsets = {
        "A": set(pairs),
        "B": set(pairs[:8]),                       # shares 8/10 with A -> cos 0.8
        "C": set(pairs[:2]) | {(100, 101)},         # shares 2 with A -> below MIN_SHARED
    }
    sim = rel.template_similarity(pairsets)
    assert ("A", "B") in sim
    expected_cos = 8 / (10 * 8) ** 0.5   # shared / sqrt(|A|*|B|) = 8/sqrt(80)
    assert sim[("A", "B")][0] == pytest.approx(expected_cos, abs=1e-6)
    assert ("A", "C") not in sim and ("B", "C") not in sim   # shared floor, not threshold, excludes C

    classes = rel.rel_class_partition({k: v[0] for k, v in sim.items()}, sorted(pairsets))
    assert classes["A"] == classes["B"]
    assert classes["C"] != classes["A"]


def test_singleton_template_gets_its_own_class():
    all_templates = ["A", "B", "C"]
    classes = rel.rel_class_partition({}, all_templates)
    assert classes == {"A": "A", "B": "B", "C": "C"}
    assert all(t in classes and classes[t] is not None for t in all_templates)


def test_rel_class_partition_is_deterministic_and_order_free():
    edges = {("A", "B"): 0.5, ("B", "C"): 0.4, ("D", "E"): 0.9}
    all_templates = ["A", "B", "C", "D", "E"]

    forward = rel.rel_class_partition(edges, all_templates)
    reversed_edges = {(b, a): v for (a, b), v in edges.items()}
    shuffled = {k: reversed_edges[k] for k in reversed(list(reversed_edges))}
    backward = rel.rel_class_partition(shuffled, list(reversed(all_templates)))

    assert forward == backward
    assert forward["A"] == "A" and forward["B"] == "A" and forward["C"] == "A"
    assert forward["D"] == "D" and forward["E"] == "D"


def test_gen_connector_variants_coclass():
    base = [(i, i + 1) for i in range(10)]
    pairsets = {
        "GEN/'s": set(base),
        "GEN/of": set(base[:9]),
        "GEN/of the": set(base[:8]),
        "in the": {(200, 201), (202, 203), (204, 205)},
    }
    sim = rel.template_similarity(pairsets)
    edges = {k: v[0] for k, v in sim.items()}
    classes = rel.rel_class_partition(edges, sorted(pairsets))
    gen_keys = [k for k in pairsets if k.startswith("GEN/")]
    assert len({classes[k] for k in gen_keys}) == 1
    assert classes["in the"] not in {classes[k] for k in gen_keys}


# =============================================================== DB-backed

import psycopg

from test_pg_store import FakeGraph          # noqa: E402  (incumbent stub, extended below)

import entities as ent
import pg_store

LABEL_A = "pytest_relations_a"
LABEL_B = "pytest_relations_b"

_N = 12
_SENTENCE = "Alpha's beta was near delta today."


class RelGraph(FakeGraph):
    """Extends the incumbent pg_store test stub -- overrides only chunks,
    docs_tok, doc_id, tfs so node.body/node.attrs->'tf' carry a planted
    sentence containing a genitive (Alpha's beta) and a plain connector
    (beta ... near ... delta), repeated across every chunk so the relation
    is the corpus's only pattern and easily clears both E13 floors."""

    def __init__(self):
        super().__init__(with_embeddings=False)
        self.n = _N
        self.chunks = [_SENTENCE for _ in range(_N)]
        self.docs_tok = [["alpha", "beta", "delta"] for _ in range(_N)]
        self.doc_id = [f"doc{i // 4}" for i in range(_N)]
        self.tfs = [{"alpha": 1, "beta": 1, "delta": 1} for _ in range(_N)]

    def edges(self):
        return []

    def communities(self, min_size=5):
        return []


def _dsn():
    return pg_store.DSN


@pytest.fixture(scope="module")
def db():
    require_dsn_db(_dsn())
    yield _dsn()
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        conn.execute("DELETE FROM graph_run WHERE label IN (%s, %s)", (LABEL_A, LABEL_B))


def test_rebuild_replaces_this_run_and_leaves_the_other_alone(db):
    from psycopg.rows import dict_row

    pg_store.save(RelGraph(), LABEL_A, dsn=db)
    pg_store.save(RelGraph(), LABEL_B, dsn=db)

    conn_a = psycopg.connect(db, row_factory=dict_row)
    conn_b = psycopg.connect(db, row_factory=dict_row)
    run_a = gt.get_run(conn_a, LABEL_A)
    run_b = gt.get_run(conn_b, LABEL_B)

    ent.build_entities(conn_a, run_a)
    ent.build_entities(conn_b, run_b)

    result_a1 = rel.build_relations(conn_a, run_a)
    result_b = rel.build_relations(conn_b, run_b)
    assert result_a1["relations"] >= 1

    def _snapshot(conn, run):
        with conn.cursor() as cur:
            cur.execute("SELECT src, dst, template, connector, n, llr, npmi, example_ord "
                       "FROM relations WHERE run_id=%s ORDER BY src, dst, template",
                       (run.run_id,))
            return cur.fetchall()

    snap_b_before = _snapshot(conn_b, run_b)

    result_a2 = rel.build_relations(conn_a, run_a)
    assert {k: v for k, v in result_a2.items() if k != "timings"} == \
           {k: v for k, v in result_a1.items() if k != "timings"}

    snap_b_after = _snapshot(conn_b, run_b)
    assert snap_b_after == snap_b_before

    # E16/E19: same idempotence law, at class grain.
    def _snapshot_classes(conn, run):
        with conn.cursor() as cur:
            cur.execute("SELECT template, rel_class FROM relation_classes "
                       "WHERE run_id=%s ORDER BY template", (run.run_id,))
            return cur.fetchall()

    cls_a1 = rel.assign_rel_classes(conn_a, run_a)
    rel.assign_rel_classes(conn_b, run_b)
    snap_b_classes_before = _snapshot_classes(conn_b, run_b)

    cls_a2 = rel.assign_rel_classes(conn_a, run_a)
    assert {k: v for k, v in cls_a1.items() if k != "timings"} == \
           {k: v for k, v in cls_a2.items() if k != "timings"}

    snap_b_classes_after = _snapshot_classes(conn_b, run_b)
    assert snap_b_classes_after == snap_b_classes_before

    conn_a.close()
    conn_b.close()


@pytest.mark.live_db
def test_live_smoke_mixed_full_dual():
    """T29 pin (2026-09-05): the live build on mixed-full-dual produced 328,825
    rows; the GEN class the layer was built for is present and well-formed
    (e.g. season -[GEN/'of the']-> end, war -[GEN/'of the']-> end). Asserted
    structurally against whatever run is live; skips without it."""
    conn = require_gt_conn()
    run = require_run(conn, "mixed-full-dual")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relations WHERE run_id=%s",
                    (run.run_id,))
        total = cur.fetchone()["count"]
        if total == 0:
            pytest.skip("relations not built for this run")
        cur.execute(
            "SELECT r.template, r.connector, ea.name AS src, eb.name AS dst, "
            "r.n, r.llr FROM relations r "
            "JOIN entities ea ON ea.run_id=r.run_id AND ea.entity_id=r.src "
            "JOIN entities eb ON eb.run_id=r.run_id AND eb.entity_id=r.dst "
            "WHERE r.run_id=%s AND r.template='GEN' "
            "ORDER BY r.llr DESC LIMIT 20", (run.run_id,))
        gen = cur.fetchall()
    assert total > 100_000                      # measured: 328,825
    assert gen, "GEN class must be present"
    assert all(g["llr"] >= 10.83 and g["n"] >= 3 for g in gen)   # E13 floors
    assert any(g["connector"] in ("'s",) or g["connector"].startswith("of")
               for g in gen)                    # genitive surface forms


@pytest.mark.live_db
def test_live_rel_classes_mixed_full_dual():
    """T71 pin (2026-09-07): live `python relations.py mixed-full-dual
    --rel-classes-only` produced templates=11,328 rows=328,825 edges=396
    classes=11,027 classed=301 (largest classes: '' mass=1,096,925 (96,839
    rows), GEN mass=152,609, w mass=141,764, the/the w/the w w/the w w w
    mass=108,881); similarity stage 65.21s, well inside Article VII's 15 min
    (T=11,328 measured, not the "dozens" first assumed -- template includes
    the free-text connector shape). Asserted structurally against whatever
    run is live; skips without it.

    gt.connect() (require_gt_conn) opens read-only at the server (W3) --
    assign_rel_classes writes, so this test needs its own writable
    connection, gated the same way (require_dsn_db skips cleanly with no
    Postgres)."""
    from psycopg.rows import dict_row
    require_dsn_db(rel.DSN)
    conn = psycopg.connect(rel.DSN, row_factory=dict_row)
    run = require_run(conn, "mixed-full-dual")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM relations WHERE run_id=%s", (run.run_id,))
        total = cur.fetchone()["count"]
        if total == 0:
            conn.close()
            pytest.skip("relations not built for this run")

    result = rel.assign_rel_classes(conn, run)

    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT template FROM relations WHERE run_id=%s", (run.run_id,))
        rel_templates = {r["template"] for r in cur.fetchall()}
        cur.execute("SELECT template, rel_class FROM relation_classes WHERE run_id=%s",
                    (run.run_id,))
        class_rows = cur.fetchall()

    class_templates = {r["template"] for r in class_rows}
    assert class_templates == rel_templates          # every template classed, none unclassed
    assert len(class_rows) == len(class_templates)    # no dupes -- PK (run_id, template) already enforces this
    rel_classes = {r["rel_class"] for r in class_rows}
    assert rel_classes <= rel_templates               # rel_class values are a subset of the templates

    sizes = Counter(r["rel_class"] for r in class_rows)
    has_multi_member_class = any(n >= 2 for n in sizes.values())
    # Either at least one class has >= 2 members, or the run's own census
    # (classed=0) says none qualified -- a reportable threshold result, not
    # a failure to hide (subplan §3.6).
    assert has_multi_member_class == (result["classed"] > 0)
    conn.close()
