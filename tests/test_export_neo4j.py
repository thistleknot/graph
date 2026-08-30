"""Exercises export_neo4j against a live run.

Integration by design, same as test_graph_tools.py: the export contract is about
what neo4j-admin will accept, and the interesting failures (dangling endpoints,
merged id-spaces, unescaped chunk bodies) only exist against real data.

Run:  pytest tests/test_export_neo4j.py -v      (needs `docker compose up -d`)
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

import export_neo4j
import graph_tools as gt

LABEL = "brown-500-dual"          # the dual-space run: SIMILAR carries 3 provenances


@pytest.fixture(scope="module")
def conn():
    try:
        c = gt.connect()
    except psycopg.OperationalError as e:                # pragma: no cover
        pytest.skip(f"no database: {e}")
    yield c
    c.close()


@pytest.fixture(scope="module")
def run(conn):
    try:
        return gt.get_run(conn, LABEL)
    except Exception as e:                               # pragma: no cover
        pytest.skip(f"no run {LABEL}: {e}")


@pytest.fixture(scope="module")
def exported(conn, run, tmp_path_factory):
    out = tmp_path_factory.mktemp("neo4j")
    stats = export_neo4j.export(conn, run, out)
    nodes = list(csv.DictReader((out / "nodes.csv").open(encoding="utf-8")))
    edges = list(csv.DictReader((out / "edges.csv").open(encoding="utf-8")))
    return out, stats, nodes, edges


def _ids(nodes):
    return ({r["id:ID(Chunk)"] for r in nodes if r[":LABEL"] == "Chunk"},
            {r["id:ID(Term)"] for r in nodes if r[":LABEL"] == "Term"})


# ------------------------------------------------------------------ contract


def test_emits_the_three_import_files(exported):
    out, *_ = exported
    for name in ("nodes.csv", "edges.csv", "import.sh"):
        assert (out / name).is_file(), f"{name} missing"


def test_import_sh_names_the_run(exported):
    """X5: an imported database must be able to say where it came from."""
    out, stats, *_ = exported
    body = (out / "import.sh").read_text(encoding="utf-8")
    assert stats["run_id"] in body
    assert stats["label"] in body


def test_stats_match_the_files(exported):
    _, stats, nodes, edges = exported
    chunk_ids, term_ids = _ids(nodes)
    assert len(chunk_ids) == stats["chunks"]
    assert len(term_ids) == stats["terms"]
    kinds = Counter(r[":TYPE"] for r in edges)
    assert kinds["CONTAINS"] == stats["contains"]
    assert kinds["SIMILAR"] == stats["similar"]


# ------------------------------------------------------------------ guards


def test_x1_chunk_and_term_ids_are_separate_spaces(exported):
    """A Term literally named "42" must not merge with record_id 42.

    The header declares :ID(Chunk) and :ID(Term), so neo4j keeps them apart.
    This pins that every node populates exactly ONE of the two id columns --
    a row filling both would be ambiguous and the importer would not complain.
    """
    _, _, nodes, _ = exported
    for r in nodes:
        filled = bool(r["id:ID(Chunk)"]) + bool(r["id:ID(Term)"])
        assert filled == 1, f"node fills {filled} id columns: {r}"


def test_x2_each_term_is_one_node(exported):
    """Terms are corpus-scoped; a term in 900 chunks is still one Term node."""
    _, _, nodes, _ = exported
    terms = [r["id:ID(Term)"] for r in nodes if r[":LABEL"] == "Term"]
    assert len(terms) == len(set(terms))


def test_x3_no_dangling_endpoints(exported):
    """neo4j-admin fails the WHOLE import on the first unresolvable id."""
    _, _, nodes, edges = exported
    chunk_ids, term_ids = _ids(nodes)
    dangling = []
    for r in edges:
        if r[":TYPE"] == "CONTAINS":
            if (r[":START_ID(Chunk)"] not in chunk_ids
                    or r[":END_ID(Term)"] not in term_ids):
                dangling.append(r)
        else:
            if (r[":START_ID(Chunk)"] not in chunk_ids
                    or r[":END_ID(Chunk)"] not in chunk_ids):
                dangling.append(r)
    assert not dangling, f"{len(dangling)} dangling, first {dangling[0]}"


def test_x4_chunk_text_survives_the_csv_round_trip(exported, conn, run):
    """Chunk bodies carry commas, quotes and newlines. Hand-quoting loses them."""
    _, _, nodes, _ = exported
    by_id = {r["id:ID(Chunk)"]: r["text"] for r in nodes if r[":LABEL"] == "Chunk"}
    with conn.cursor() as cur:
        cur.execute("""SELECT ord, body FROM node
                        WHERE run_id = %s AND (body LIKE '%%,%%' OR body LIKE '%%"%%')
                        ORDER BY ord LIMIT 25""", (run.run_id,))
        rows = cur.fetchall()
    assert rows, "fixture should contain punctuated bodies"
    for r in rows:
        assert by_id[str(r["ord"])] == r["body"], f"body mangled at ord {r['ord']}"


def test_contains_pairs_are_unique(exported):
    """One CONTAINS per (chunk, term); tf carries the count."""
    _, _, _, edges = exported
    pairs = [(r[":START_ID(Chunk)"], r[":END_ID(Term)"])
             for r in edges if r[":TYPE"] == "CONTAINS"]
    assert len(pairs) == len(set(pairs))


def test_contains_edges_carry_positive_tf(exported):
    _, _, _, edges = exported
    tfs = [int(r["tf:int"]) for r in edges if r[":TYPE"] == "CONTAINS"]
    assert tfs and min(tfs) >= export_neo4j.MIN_TF


# ------------------------------------------------------- the dense-space point


def test_similar_edges_carry_all_three_provenances(exported):
    """The reason SIMILAR is exported at all.

    Sparse SIMILAR is redundant with the CONTAINS pattern -- it summarises it.
    Dense is not: embedding proximity is not a function of shared terms, so a
    contains-only export is sparse-only by construction.
    """
    _, _, _, edges = exported
    provs = Counter(r["prov"] for r in edges if r[":TYPE"] == "SIMILAR")
    assert set(provs) == {"sparse", "dense", "both"}, provs
    assert provs["dense"] > 0


def test_edges_contains_drops_similar_entirely(conn, run, tmp_path):
    stats = export_neo4j.export(conn, run, tmp_path, edges="contains")
    assert stats["similar"] == 0 and stats["contains"] > 0
    rows = list(csv.DictReader((tmp_path / "edges.csv").open(encoding="utf-8")))
    assert all(r[":TYPE"] == "CONTAINS" for r in rows)


def test_edges_similar_drops_contains_entirely(conn, run, tmp_path):
    stats = export_neo4j.export(conn, run, tmp_path, edges="similar")
    assert stats["contains"] == 0 and stats["similar"] > 0


def test_min_tf_prunes_the_inverted_index(conn, run, tmp_path):
    lo = export_neo4j.export(conn, run, tmp_path / "lo", min_tf=1)
    hi = export_neo4j.export(conn, run, tmp_path / "hi", min_tf=3)
    assert hi["contains"] < lo["contains"]
    assert hi["terms"] <= lo["terms"]


# ------------------------------------------- term-mediated hop is reachable


def test_a_shared_term_connects_two_chunks_without_a_similar_edge(exported):
    """(:Chunk)-[:CONTAINS]->(:Term)<-[:CONTAINS]-(:Chunk) -- the whole point.

    Finds a term shared by two chunks that have NO SIMILAR edge between them,
    proving the export carries connections the chunk-chunk edge list does not.
    """
    _, _, _, edges = exported
    holders = {}
    for r in edges:
        if r[":TYPE"] == "CONTAINS":
            holders.setdefault(r[":END_ID(Term)"], set()).add(r[":START_ID(Chunk)"])
    sim = {frozenset((r[":START_ID(Chunk)"], r[":END_ID(Chunk)"]))
           for r in edges if r[":TYPE"] == "SIMILAR"}
    for term, cs in holders.items():
        if len(cs) < 2:
            continue
        cs = sorted(cs)
        for i, a in enumerate(cs[:20]):
            for b in cs[i + 1:21]:
                if frozenset((a, b)) not in sim:
                    return          # found one; the pattern reaches further than SIMILAR
    pytest.fail("no term-mediated pair outside the SIMILAR edge set")
