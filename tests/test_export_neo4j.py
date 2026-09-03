"""Exercises export_neo4j against a live run.

Integration by design, same as test_graph_tools.py: the export contract is about
what neo4j-admin will accept, and the interesting failures (dangling endpoints,
merged id-spaces, unescaped chunk bodies, ragged embeddings) only exist against
real data.

Run:  pytest tests/test_export_neo4j.py -v      (needs `docker compose up -d`)
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
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
    chunks = list(csv.DictReader((out / "chunks.csv").open(encoding="utf-8")))
    terms = list(csv.DictReader((out / "terms.csv").open(encoding="utf-8")))
    contains = list(csv.DictReader((out / "contains.csv").open(encoding="utf-8")))
    similar = list(csv.DictReader((out / "similar.csv").open(encoding="utf-8")))
    return out, stats, chunks, terms, contains, similar


def _ids(chunks, terms):
    return ({r["id:ID(Chunk)"] for r in chunks},
            {r["name:ID(Term)"] for r in terms})


# ------------------------------------------------------------------ contract


def test_emits_the_three_import_files(exported):
    out, *_ = exported
    for name in ("chunks.csv", "terms.csv", "contains.csv", "similar.csv",
                 "import.sh"):
        assert (out / name).is_file(), f"{name} missing"


def test_import_sh_names_the_run(exported):
    """X5: an imported database must be able to say where it came from."""
    out, stats, *_ = exported
    body = (out / "import.sh").read_text(encoding="utf-8")
    assert stats["run_id"] in body
    assert stats["label"] in body


def test_stats_match_the_files(exported):
    _, stats, chunks, terms, contains, similar = exported
    chunk_ids, term_ids = _ids(chunks, terms)
    assert len(chunk_ids) == stats["chunks"]
    assert len(term_ids) == stats["terms"]
    assert len(contains) == stats["contains"]
    assert len(similar) == stats["similar"]


# ------------------------------------------------------------------ guards


def test_x1_chunk_and_term_ids_are_separate_spaces(exported):
    """A Term literally named "42" must not merge with record_id 42.

    chunks.csv and terms.csv each carry exactly one id column, in separate
    files -- X6 makes this structural rather than a per-row invariant.
    """
    _, _, chunks, terms, _, _ = exported
    for r in chunks:
        assert "id:ID(Chunk)" in r and r["id:ID(Chunk)"] != ""
    for r in terms:
        assert "name:ID(Term)" in r and r["name:ID(Term)"] != ""


def test_x2_each_term_is_one_node(exported):
    """Terms are corpus-scoped; a term in 900 chunks is still one Term node."""
    _, _, _, terms, _, _ = exported
    names = [r["name:ID(Term)"] for r in terms]
    assert len(names) == len(set(names))


def test_x3_no_dangling_endpoints(exported):
    """neo4j-admin fails the WHOLE import on the first unresolvable id."""
    _, _, chunks, terms, contains, similar = exported
    chunk_ids, term_ids = _ids(chunks, terms)
    dangling = []
    for r in contains:
        if r[":START_ID(Chunk)"] not in chunk_ids or r[":END_ID(Term)"] not in term_ids:
            dangling.append(r)
    for r in similar:
        if r[":START_ID(Chunk)"] not in chunk_ids or r[":END_ID(Chunk)"] not in chunk_ids:
            dangling.append(r)
    assert not dangling, f"{len(dangling)} dangling, first {dangling[0]}"


def test_x4_chunk_text_survives_the_csv_round_trip(exported, conn, run):
    """Chunk bodies carry commas, quotes and newlines. Hand-quoting loses them."""
    _, _, chunks, _, _, _ = exported
    by_id = {r["id:ID(Chunk)"]: r["text"] for r in chunks}
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
    _, _, _, _, contains, _ = exported
    pairs = [(r[":START_ID(Chunk)"], r[":END_ID(Term)"]) for r in contains]
    assert len(pairs) == len(set(pairs))


def test_contains_edges_carry_positive_tf(exported):
    _, _, _, _, contains, _ = exported
    tfs = [int(r["tf:int"]) for r in contains]
    assert tfs and min(tfs) >= export_neo4j.MIN_TF


# ------------------------------------------------------- the dense-space point


def test_similar_edges_carry_all_three_provenances(exported):
    """The reason SIMILAR is exported at all.

    Sparse SIMILAR is redundant with the CONTAINS pattern -- it summarises it.
    Dense is not: embedding proximity is not a function of shared terms, so a
    contains-only export is sparse-only by construction.
    """
    _, _, _, _, _, similar = exported
    provs = Counter(r["prov"] for r in similar)
    assert set(provs) == {"sparse", "dense", "both"}, provs
    assert provs["dense"] > 0


def test_edges_contains_drops_similar_entirely(conn, run, tmp_path):
    stats = export_neo4j.export(conn, run, tmp_path, edges="contains")
    assert stats["similar"] == 0 and stats["contains"] > 0
    rows = list(csv.DictReader((tmp_path / "similar.csv").open(encoding="utf-8")))
    assert rows == []


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
    _, _, _, _, contains, similar = exported
    holders = {}
    for r in contains:
        holders.setdefault(r[":END_ID(Term)"], set()).add(r[":START_ID(Chunk)"])
    sim = {frozenset((r[":START_ID(Chunk)"], r[":END_ID(Chunk)"])) for r in similar}
    for term, cs in holders.items():
        if len(cs) < 2:
            continue
        cs = sorted(cs)
        for i, a in enumerate(cs[:20]):
            for b in cs[i + 1:21]:
                if frozenset((a, b)) not in sim:
                    return          # found one; the pattern reaches further than SIMILAR
    pytest.fail("no term-mediated pair outside the SIMILAR edge set")


# ---- 4-file layout, embeddings, vector index, C<cid> labels (T3)


CHUNKS_HEADER = ["id:ID(Chunk)", "doc_id", "source", "text", "cid:int",
                  "embedding:float[]", ":LABEL"]
TERMS_HEADER = ["name:ID(Term)", ":LABEL"]
CONTAINS_HEADER = [":START_ID(Chunk)", ":END_ID(Term)", ":TYPE", "tf:int"]
SIMILAR_HEADER = [":START_ID(Chunk)", ":END_ID(Chunk)", ":TYPE", "strength:float",
                   "prov", "sim_sparse:float", "sim_dense:float"]


def test_header_shape_is_the_four_file_layout(exported):
    out, *_ = exported
    headers = {}
    for name in ("chunks.csv", "terms.csv", "contains.csv", "similar.csv"):
        with (out / name).open(encoding="utf-8") as f:
            headers[name] = next(csv.reader(f))
    assert headers["chunks.csv"] == CHUNKS_HEADER
    assert headers["terms.csv"] == TERMS_HEADER
    assert headers["contains.csv"] == CONTAINS_HEADER
    assert headers["similar.csv"] == SIMILAR_HEADER
    for name, header in headers.items():
        id_cols = [c for c in header if c.startswith(":ID(") or ":ID(" in c]
        assert len(id_cols) <= 1, f"{name} carries >1 :ID( column: {header}"


def test_import_sh_uses_multiline_fields_and_all_four_files(exported):
    out, *_ = exported
    body = (out / "import.sh").read_text(encoding="utf-8")
    assert "--nodes=chunks.csv" in body
    assert "--nodes=terms.csv" in body
    assert "--relationships=contains.csv" in body
    assert "--relationships=similar.csv" in body
    assert "--multiline-fields=true" in body
    assert "nodes.csv" not in body.replace("chunks.csv", "").replace("terms.csv", "")
    assert "edges.csv" not in body


def _fake_fetch_with_embeddings(chunks):
    def _fetch(conn, run, min_tf):
        return chunks, [(0, "term", 2)], []
    return _fetch


def test_embedding_column_is_semicolon_array_of_uniform_width(monkeypatch, tmp_path):
    # (a) all embedded
    chunks_a = [
        (0, "d0", "brown", "body a", None, ["0.1", "0.2"]),
        (1, "d1", "brown", "body b", None, ["0.3", "0.4"]),
    ]
    monkeypatch.setattr(export_neo4j, "_fetch", _fake_fetch_with_embeddings(chunks_a))
    stats = export_neo4j.export(None, _FakeRun(), tmp_path / "a")
    rows = list(csv.DictReader((tmp_path / "a" / "chunks.csv").open(encoding="utf-8")))
    for r in rows:
        toks = r["embedding:float[]"].split(";")
        assert len(toks) == stats["embed_dim"] == 2
        for t in toks:
            float(t)
    assert stats["embed_dim"] == 2
    assert (tmp_path / "a" / "vector_index.cypher").is_file()

    # (b) some embedded, some not
    chunks_b = [
        (0, "d0", "brown", "body a", None, ["0.1", "0.2"]),
        (1, "d1", "brown", "body b", None, None),
    ]
    monkeypatch.setattr(export_neo4j, "_fetch", _fake_fetch_with_embeddings(chunks_b))
    stats_b = export_neo4j.export(None, _FakeRun(), tmp_path / "b")
    rows_b = list(csv.DictReader((tmp_path / "b" / "chunks.csv").open(encoding="utf-8")))
    by_doc = {r["doc_id"]: r for r in rows_b}
    assert by_doc["d0"]["embedding:float[]"] == "0.1;0.2"
    assert by_doc["d1"]["embedding:float[]"] == ""
    assert stats_b["embed_dim"] == 2
    assert stats_b["embeddings"] == 1

    # (c) none embedded
    chunks_c = [
        (0, "d0", "brown", "body a", None, None),
        (1, "d1", "brown", "body b", None, None),
    ]
    monkeypatch.setattr(export_neo4j, "_fetch", _fake_fetch_with_embeddings(chunks_c))
    stats_c = export_neo4j.export(None, _FakeRun(), tmp_path / "c")
    assert stats_c["embed_dim"] is None
    assert stats_c["embeddings"] == 0
    assert not (tmp_path / "c" / "vector_index.cypher").exists()


def test_vector_index_cypher_matches_the_measured_dim(exported):
    out, stats, *_ = exported
    if not stats["embed_dim"]:
        pytest.skip("run has no embeddings")
    body = (out / "vector_index.cypher").read_text(encoding="utf-8")
    assert "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS" in body
    assert "FOR (c:Chunk) ON (c.embedding)" in body
    assert str(stats["embed_dim"]) in body
    assert "'cosine'" in body


def test_c_cid_labels_ride_the_label_column(monkeypatch, tmp_path):
    chunks = [
        (0, "d0", "brown", "body a", 3, None),
        (1, "d1", "brown", "body b", None, None),
        (2, "d2", "brown", "body c", 3, None),
    ]
    monkeypatch.setattr(export_neo4j, "_fetch", _fake_fetch_with_embeddings(chunks))
    stats = export_neo4j.export(None, _FakeRun(), tmp_path)
    rows = list(csv.DictReader((tmp_path / "chunks.csv").open(encoding="utf-8")))
    by_doc = {r["doc_id"]: r for r in rows}
    assert by_doc["d0"][":LABEL"] == "Chunk;C3"
    assert by_doc["d2"][":LABEL"] == "Chunk;C3"
    assert by_doc["d1"][":LABEL"] == "Chunk"
    assert stats["labeled"] == 2


def test_c_cid_labels_ride_the_label_column_live(exported):
    _, stats, chunks, *_ = exported
    labeled = 0
    for r in chunks:
        cid = r["cid:int"]
        labels = r[":LABEL"].split(";")
        if cid:
            assert labels == ["Chunk", f"C{cid}"]
            labeled += 1
        else:
            assert labels == ["Chunk"]
    assert labeled == stats["labeled"]
    assert stats["labeled"] > 0


# ---- source column (R20)
# DB-free: export() touches `conn` ONLY through `_fetch`, so `conn=None` is
# safe once `_fetch` is monkeypatched.


class _FakeRun:
    def __init__(self, label="fake-run", run_id="00000000-0000-0000-0000-000000000000"):
        self.label = label
        self.run_id = run_id


def _fake_fetch(chunks):
    def _fetch(conn, run, min_tf):
        return chunks, [(0, "term", 2)], []
    return _fetch


def test_source_column_all_sourced(monkeypatch, tmp_path):
    chunks = [(0, "brown/ca01", "brown", "body a", None, None),
              (1, "wiki/7", "wiki", "body b", None, None),
              (2, "brown/ca02", "brown", "body c", None, None)]
    monkeypatch.setattr(export_neo4j, "_fetch", _fake_fetch(chunks))
    stats = export_neo4j.export(None, _FakeRun(), tmp_path)
    chunk_rows = {r["doc_id"]: r for r in csv.DictReader(
        (tmp_path / "chunks.csv").open(encoding="utf-8"))}
    assert chunk_rows["brown/ca01"]["source"] == "brown"
    assert chunk_rows["wiki/7"]["source"] == "wiki"
    assert chunk_rows["brown/ca02"]["source"] == "brown"
    assert stats["sources"] == {"brown": 2, "wiki": 1}
    body = (tmp_path / "import.sh").read_text(encoding="utf-8")
    assert "brown 2" in body


def test_source_column_none_sourced(monkeypatch, tmp_path):
    """Today's shape: source=None, unprefixed doc_ids -- the pre-R20 degrade."""
    chunks = [(0, "ca01", None, "body a", None, None),
              (1, "ca02", None, "body b", None, None)]
    monkeypatch.setattr(export_neo4j, "_fetch", _fake_fetch(chunks))
    stats = export_neo4j.export(None, _FakeRun(), tmp_path)
    chunk_rows = list(csv.DictReader((tmp_path / "chunks.csv").open(encoding="utf-8")))
    assert all(r["source"] == "" for r in chunk_rows)
    assert stats["sources"] == {}
    body = (tmp_path / "import.sh").read_text(encoding="utf-8")
    assert "sources:" not in body


def test_source_column_mixed(monkeypatch, tmp_path):
    chunks = [(0, "brown/ca01", "brown", "body a", None, None),
              (1, "wiki/7", "wiki", "body b", None, None),
              (2, "ca03", None, "body c", None, None)]
    monkeypatch.setattr(export_neo4j, "_fetch", _fake_fetch(chunks))
    stats = export_neo4j.export(None, _FakeRun(), tmp_path)
    chunk_rows = {r["doc_id"]: r for r in csv.DictReader(
        (tmp_path / "chunks.csv").open(encoding="utf-8"))}
    assert chunk_rows["brown/ca01"]["source"] == "brown"
    assert chunk_rows["wiki/7"]["source"] == "wiki"
    assert chunk_rows["ca03"]["source"] == ""
    assert stats["sources"] == {"brown": 1, "wiki": 1}


# DB-backed, skip-tolerant: the degrade path against real pre-R20 data.


def test_source_column_empty_on_real_pre_r20_export(exported):
    _, stats, chunks, _, _, _ = exported
    assert chunks
    assert all(r["source"] == "" for r in chunks)
    assert stats["sources"] == {}


# ---------------------------------------------- live walk writer


class _FakeBundle:
    def __init__(self, query="what connects brown and wiki", run_id="run-1",
                 anchors=None):
        self.query = query
        self.run_id = run_id
        self.anchors = anchors if anchors is not None else [3439, {"ord": 3552}]


_PATHS = {
    "n": 12, "edges": 18, "components": 2, "wcc_sizes": [10, 2],
    "largest_component_frac": 0.83, "density": 0.67, "conductance": 0.80,
    "pairs": [
        {"a": 3439, "b": 3552, "dwpc": 0.138, "n_paths": 4,
         "path": [3439, 100, 3552]},
        {"a": 200, "b": 50, "dwpc": 0.02, "n_paths": 1, "path": None},
    ],
}


def _capturing_post():
    """Fake `post=` transport: records statements, returns matching counts."""
    captured = []

    def post(statements, url=None, auth=None, db=None):
        captured.append(list(statements))
        results = []
        for s in statements:
            cypher = s["statement"]
            params = s["parameters"]
            if "MATCH (c:Chunk {id: cid})" in cypher:
                n = len(params["ids"])
            elif "MATCH (s:Chunk {id: r.a})" in cypher:
                n = len(params["rows"])
            else:
                n = 0
            results.append({"columns": ["n"], "data": [{"row": [n]}]})
        return results

    post.captured = captured
    return post


def test_walk_merge_is_keyed_on_prompt():
    post = _capturing_post()
    write_walk_stats = export_neo4j.write_walk(_FakeBundle(), _PATHS, post=post)
    assert "walk_prompt" in post.captured[0][0]["statement"]      # constraint, own tx
    stmts = post.captured[-1]
    walk_stmt = stmts[0]
    assert "MERGE (w:Walk {prompt: $prompt})" in walk_stmt["statement"]
    p = walk_stmt["parameters"]
    assert p["n"] == _PATHS["n"]
    assert p["edges"] == _PATHS["edges"]
    assert p["wcc"] == _PATHS["components"]
    assert p["density"] == _PATHS["density"]
    assert p["conductance"] == _PATHS["conductance"]
    assert write_walk_stats["prompt"] == _FakeBundle().query


def test_anchors_match_chunks_and_never_merge_them():
    post = _capturing_post()
    export_neo4j.write_walk(_FakeBundle(), _PATHS, post=post)
    anchor_stmt = next(s for s in post.captured[-1]
                        if "UNWIND $ids AS cid" in s["statement"])
    assert "MATCH (c:Chunk {id: cid})" in anchor_stmt["statement"]
    assert "MERGE (c:Chunk" not in anchor_stmt["statement"]


def test_chunk_ids_go_over_the_wire_as_strings():
    post = _capturing_post()
    export_neo4j.write_walk(_FakeBundle(), _PATHS, post=post)
    for stmt in post.captured[-1]:
        params = stmt["parameters"]
        for cid in params.get("ids", []):
            assert isinstance(cid, str)
        for row in params.get("rows", []):
            assert isinstance(row["a"], str) and isinstance(row["b"], str)


def test_pathway_pairs_are_normalized_low_to_high():
    post = _capturing_post()
    export_neo4j.write_walk(_FakeBundle(), _PATHS, post=post)
    pathway_stmt = next(s for s in post.captured[-1]
                         if "UNWIND $rows AS r" in s["statement"])
    assert "MERGE (s:Chunk {id: r.a})->" not in pathway_stmt["statement"]  # sanity: not a typo pattern
    assert "MERGE (s)-[p:PATHWAY {of: $prompt}]->(d)" in pathway_stmt["statement"]
    rows = pathway_stmt["parameters"]["rows"]
    reversed_pair = next(r for r in rows if r["a"] == "50")
    assert reversed_pair["b"] == "200"
    for r in rows:
        assert r["a"] <= r["b"] or int(r["a"]) <= int(r["b"])


def test_two_writes_of_the_same_walk_emit_identical_statements():
    post1 = _capturing_post()
    export_neo4j.write_walk(_FakeBundle(), _PATHS, post=post1)
    post2 = _capturing_post()
    export_neo4j.write_walk(_FakeBundle(), _PATHS, post=post2)
    assert post1.captured == post2.captured


def test_batching_splits_rows_at_the_batch_size():
    bundle = _FakeBundle(anchors=[1, 2, 3, 4, 5])
    paths = dict(_PATHS)
    paths["pairs"] = [{"a": i, "b": i + 1, "dwpc": 0.1, "n_paths": 1, "path": None}
                       for i in range(5)]
    post = _capturing_post()
    export_neo4j.write_walk(bundle, paths, post=post, batch=2)
    stmts = post.captured[-1]
    anchor_stmts = [s for s in stmts if "UNWIND $ids AS cid" in s["statement"]]
    pathway_stmts = [s for s in stmts if "UNWIND $rows AS r" in s["statement"]]
    assert len(anchor_stmts) == 3
    assert len(pathway_stmts) == 3


def test_transport_errors_raise_rather_than_degrade(monkeypatch):
    class _FakeResponse:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    body = json.dumps({
        "results": [],
        "errors": [{"code": "Neo.ClientError.Schema.ConstraintValidationFailed",
                     "message": "boom"}],
    }).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(body)

    monkeypatch.setattr(export_neo4j.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="ConstraintValidationFailed"):
        export_neo4j.write_walk(_FakeBundle(), _PATHS)


def test_anchor_shortfall_raises():
    def post(statements, url=None, auth=None, db=None):
        results = []
        for s in statements:
            if "UNWIND $ids AS cid" in s["statement"]:
                results.append({"columns": ["n"], "data": [{"row": [1]}]})
            elif "UNWIND $rows AS r" in s["statement"]:
                results.append({"columns": ["n"],
                                 "data": [{"row": [len(s["parameters"]["rows"])]}]})
            else:
                results.append({"columns": [], "data": [{"row": [0]}]})
        return results

    with pytest.raises(ValueError, match="anchors"):
        export_neo4j.write_walk(_FakeBundle(), _PATHS, post=post)


@pytest.fixture(scope="module")
def neo4j_up():
    import urllib.error as _uerr
    try:
        urllib.request.urlopen(export_neo4j.NEO4J_HTTP, timeout=2).read()
    except (_uerr.URLError, OSError) as e:                # pragma: no cover
        pytest.skip(f"no neo4j at {export_neo4j.NEO4J_HTTP}: {e}")


def test_write_walk_round_trips_against_live_neo4j(neo4j_up):
    picked = export_neo4j._tx(
        [{"statement": "MATCH (c:Chunk) RETURN c.id AS id LIMIT 2",
          "parameters": {}}])
    ids = [row["row"][0] for row in picked[0]["data"]]
    if len(ids) < 2:
        pytest.skip("no Chunk nodes imported (mixed-full-dual import absent)")

    prompt = "__t8_selftest__"
    paths = {
        "n": 2, "edges": 1, "components": 1, "wcc_sizes": [2],
        "largest_component_frac": 1.0, "density": 1.0, "conductance": 0.5,
        "pairs": [{"a": int(ids[0]), "b": int(ids[1]), "dwpc": 0.1,
                    "n_paths": 1, "path": [int(ids[0]), int(ids[1])]}],
    }
    bundle = _FakeBundle(query=prompt, run_id="t8-selftest", anchors=ids)
    try:
        export_neo4j.write_walk(bundle, paths, prompt=prompt)
        r1 = export_neo4j._tx([
            {"statement": "MATCH (w:Walk {prompt:$p})-[:ANCHORS]->(c) "
                          "RETURN count(c) AS n", "parameters": {"p": prompt}},
            {"statement": "MATCH ()-[p:PATHWAY {of:$p}]->() RETURN count(p) AS n",
             "parameters": {"p": prompt}},
        ])
        anchors_n1 = r1[0]["data"][0]["row"][0]
        pathways_n1 = r1[1]["data"][0]["row"][0]
        assert anchors_n1 == 2
        assert pathways_n1 == 1

        export_neo4j.write_walk(bundle, paths, prompt=prompt)          # X10
        r2 = export_neo4j._tx([
            {"statement": "MATCH (w:Walk {prompt:$p})-[:ANCHORS]->(c) "
                          "RETURN count(c) AS n", "parameters": {"p": prompt}},
            {"statement": "MATCH ()-[p:PATHWAY {of:$p}]->() RETURN count(p) AS n",
             "parameters": {"p": prompt}},
        ])
        anchors_n2 = r2[0]["data"][0]["row"][0]
        pathways_n2 = r2[1]["data"][0]["row"][0]
        assert anchors_n2 == anchors_n1
        assert pathways_n2 == pathways_n1
    finally:
        export_neo4j._tx([
            {"statement": "MATCH (w:Walk {prompt:$p}) DETACH DELETE w",
             "parameters": {"p": prompt}},
            {"statement": "MATCH ()-[p:PATHWAY {of:$p}]->() DELETE p",
             "parameters": {"p": prompt}},
        ])
