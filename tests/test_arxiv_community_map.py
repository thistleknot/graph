"""Pins arxiv_community_map.py: the reproducible kNN build (A2), the exact lexical kNN (R7), the
derivation plumbing (tiers, exemplars, provenance), the stale-summary guard and the no-clipping render.

Synthetic clusters, no GPU, no model, no corpus.

Run:  pytest tests/test_arxiv_community_map.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arxiv_community_map as m
import hnsw_communities as hc


def _unit(X):
    X = np.asarray(X, np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_snippet_skips_the_header_and_collapses_whitespace():
    s = m.snippet({"text": "## Title\n\nfirst   line\nsecond line " + "x" * 400}, 40)
    assert s.startswith("first line second line") and s.endswith("...") and len(s) <= 40


def test_the_single_thread_knn_build_is_reproducible():
    """A2: multi-thread insertion gave 148, 155, 135 communities on identical embeddings."""
    E = _unit(np.random.default_rng(0).normal(size=(3000, 16)))
    (i1, d1), (i2, d2) = m.knn_det(E), m.knn_det(E)
    assert (i1 == i2).all() and np.allclose(d1, d2)


def test_the_lexical_knn_is_the_exact_top_k_by_cosine():
    rng = np.random.default_rng(1)
    X = sp.random(300, 60, density=0.15, random_state=2, format="csr", dtype=np.float32)
    X = sp.diags(1 / np.maximum(np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()), 1e-9)).dot(X).tocsr()
    got = m.lexical_knn(X, k=5, block=64, cache=None)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, -1)
    for i in rng.integers(0, 300, 40):
        want = set(np.argsort(-S[i])[:5].tolist())
        assert S[i, sorted(got[i])].min() >= np.sort(S[i])[-5] - 1e-6        # same score floor even through ties
        assert len(set(got[i].tolist()) & want) >= 4                         # (ties may swap one member)


def _world(tmp_path, monkeypatch):
    """Three separated clusters (120, 100, 80 chunks) in a dense and a lexical view."""
    rng = np.random.default_rng(3)
    sizes, dim = [120, 100, 80], 12
    centres = np.eye(dim)[:3] * 4
    E = _unit(np.concatenate([centres[c] + rng.normal(size=(s, dim)) for c, s in enumerate(sizes)]))
    E = hc.center(E, E.mean(0))
    lab = np.repeat(np.arange(3), sizes)
    N = len(lab)
    X = np.zeros((N, 90), np.float32)
    for i in range(N):
        X[i, lab[i] * 30 + rng.choice(30, 6, replace=False)] = rng.random(6) + 0.5
    X = sp.csr_matrix(_unit(X))
    idx, dist = m.knn_det(E)
    recs = [{"doc_id": "arxiv/p%d" % (lab[i] if i % 4 else 9), "section_idx": i, "chunk_idx": 0,
             "section_title": "T%d" % i, "text": "## T%d\n\nbody %d" % (i, i)} for i in range(N)]
    monkeypatch.setattr(m, "OUT_EX", str(tmp_path / "ex.json"))
    return recs, E, X, {"lab": lab, "idx": idx, "dist": dist}, m.lexical_knn(X, k=10, block=128, cache=None)


def test_derive_gives_every_community_a_medoid_first_exemplars_by_size_and_provenance(tmp_path, monkeypatch):
    recs, E, X, st, lidx = _world(tmp_path, monkeypatch)
    ex = m.derive(recs, E, X, st, lidx)
    assert set(ex) == {0, 1, 2}
    n_by_size = hc.n_per_community(np.array([120, 100, 80]))
    for c in ex:
        items = ex[c]["exemplars"]
        assert items[0]["role"] == "medoid" and items[0]["z"] == 0.0
        assert len(items) <= n_by_size[c] and ex[c]["n"] == n_by_size[c]
        rows = [it["row"] for it in items]
        assert len(set(rows)) == len(rows) and all(st["lab"][r] == c for r in rows)          # distinct, own community
        assert items[0]["key"] == [recs[rows[0]]["doc_id"], recs[rows[0]]["section_idx"], 0]
        assert ex[c]["n_papers"] >= 1 and 0 < ex[c]["top_share"] <= 1
    assert json.load(open(tmp_path / "ex.json"))["0"]["size"] == 120


def _ex(keys_by_comm):
    return {c: {"exemplars": [{"key": k} for k in keys]} for c, keys in keys_by_comm.items()}


def test_a_summary_is_shown_only_when_its_chunks_are_the_communitys_exemplars(tmp_path, monkeypatch):
    base = {"status": "draft", "title": "t", "summary": "s", "model": "x"}
    path = tmp_path / "s.json"
    path.write_text(json.dumps([
        {**base, "community": 0, "chunk_keys": [["arxiv/0", 0, 0], ["arxiv/1", 1, 0]]},     # its own exemplars
        {**base, "community": 1, "chunk_keys": [["arxiv/3", 3, 0]]},                        # written for other chunks: stale
        {"community": 2, "status": "failed", "chunk_keys": [], "reason": "x"}]), encoding="utf-8")
    monkeypatch.setattr(m, "SUMMARIES", str(path))
    ex = _ex({0: [["arxiv/0", 0, 0], ["arxiv/1", 1, 0]], 1: [["arxiv/2", 2, 0]]})
    assert set(m.load_summaries(ex)) == {0}
    monkeypatch.setattr(m, "SUMMARIES", str(tmp_path / "missing.json"))
    assert m.load_summaries(ex) == {}


def _rec(text, ref=False, junk=None):
    r = {"doc_id": "arxiv/x", "section_idx": 0, "chunk_idx": 0, "section_title": "T", "is_reference": ref, "text": text}
    if junk is not None:
        r["is_junk"] = junk
    return r


def test_load_keeps_only_retrievable_chunks_and_their_positions_among_all_chunks(tmp_path, monkeypatch):
    import pickle
    prose = "## A\n\nOrdinary words about graphs and communities in text."
    soup = "## S\n\n" + "\n\n".join("abcdefghij"[i % 10] for i in range(30))
    allrecs = [_rec(prose + " 0"), _rec("## References\n\n[1] x", ref=True), _rec(soup),                 # soup: flag absent, judged from text
               _rec(prose + " 3"), _rec(prose + " 4", junk=True), _rec(prose + " 5")]                  # flagged junk
    pkl = tmp_path / "c.pkl"
    pickle.dump({"records": allrecs}, open(pkl, "wb"))
    monkeypatch.setattr(m, "CACHE", str(pkl))
    recs, ords = m.load()
    assert ords == [0, 3, 5] and [r["text"] for r in recs] == [allrecs[i]["text"] for i in ords]


def test_a_cache_written_for_other_texts_is_purged_even_when_the_row_count_is_the_same(tmp_path):
    caches = tuple(str(tmp_path / n) for n in ("emb.npy", "state.npz", "lex.npz"))
    fp = str(tmp_path / "fp.txt")
    mk = lambda *t: [_rec(x) for x in t]
    old = mk("alpha", "beta", "gamma")
    assert m.purge_stale_caches(old, fp, caches) is True                 # no fingerprint yet: whatever is there predates it
    for c in caches:
        Path(c).write_bytes(b"cached")
    assert m.purge_stale_caches(old, fp, caches) is False                # same texts, same order: kept
    assert all(Path(c).exists() for c in caches)
    assert m.purge_stale_caches(mk("alpha", "beta", "DELTA"), fp, caches) is True       # one text replaced, count unchanged
    assert not any(Path(c).exists() for c in caches)
    for c in caches:
        Path(c).write_bytes(b"cached")
    assert m.purge_stale_caches(mk("beta", "alpha", "DELTA"), fp, caches) is True       # same texts, new order
    assert not any(Path(c).exists() for c in caches)


def test_fingerprint_is_stable_and_sensitive_to_one_character():
    a = [_rec("one"), _rec("two")]
    assert m.fingerprint(a) == m.fingerprint([_rec("one"), _rec("two")])
    assert m.fingerprint(a) != m.fingerprint([_rec("one"), _rec("twp")])


def test_persist_writes_vectors_communities_assignments_and_exemplars_on_the_live_build_and_can_run_twice():
    import psycopg
    import sparsevec_store as ss
    test_db = "graph_sparsevec_test"
    try:
        with psycopg.connect(ss.DSN, autocommit=True) as admin:
            if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_db,)).fetchone():
                admin.execute(f'CREATE DATABASE "{test_db}"')
        conn = psycopg.connect(ss.DSN.rsplit("/", 1)[0] + "/" + test_db, autocommit=True)
    except Exception as exc:                                              # pragma: no cover
        pytest.skip("no Postgres: %s" % exc)
    label = "zz_map"
    ss.ensure_build_schema(conn)
    conn.execute("DELETE FROM lex_build WHERE label = %s", (label,))
    ss.reset_label_tables(conn, label)
    build = ss.new_build(conn, label, 10, {"bm25": {"dim": 12}})
    rng = np.random.default_rng(0)
    E = _unit(rng.normal(size=(6, 4)))
    ords = [0, 2, 3, 5, 6, 8]
    st = {"lab": np.array([0, 0, 1, 1, 1, 2]), "XY": rng.normal(size=(6, 2)), "chosen": 6.162, "cons_ari": 0.885}
    ex = {0: {"exemplars": [{"role": "medoid", "row": 1, "z": 0.0}]},
          1: {"exemplars": [{"role": "medoid", "row": 3, "z": 0.0}, {"role": "z=1/2", "row": 4, "z": 0.48}]},
          2: {"exemplars": [{"role": "medoid", "row": 5, "z": 0.0}]}}
    mean = np.array([0.1, 0.2, 0.3, 0.4])
    for _ in range(2):                                                     # a second run replaces, never doubles
        assert m.persist(ords, mean, E, st, ex, label=label, conn=conn) == build
    count = lambda t: conn.execute(f"SELECT count(*) FROM {t} WHERE build_id = %s", (build,)).fetchone()[0]
    assert (count("lex_community"), count("lex_assign"), count("lex_exemplar")) == (3, 6, 4)
    assert conn.execute(f"SELECT count(*) FROM {ss._table(label, 'dense')}").fetchone()[0] == 6
    assert conn.execute("SELECT ord, cid, how FROM lex_assign WHERE build_id = %s ORDER BY ord", (build,)).fetchall() == [
        (0, 0, "consensus"), (2, 0, "consensus"), (3, 1, "consensus"), (5, 1, "consensus"), (6, 1, "consensus"), (8, 2, "consensus")]
    assert conn.execute("SELECT ord FROM lex_exemplar WHERE build_id = %s AND cid = 1 ORDER BY rank", (build,)).fetchall() == [(5,), (6,)]
    p = conn.execute("SELECT params FROM lex_build WHERE build_id = %s", (build,)).fetchone()[0]
    assert p["dense"]["mean"] == [0.1, 0.2, 0.3, 0.4] and p["communities"]["n"] == 3 and p["bm25"] == {"dim": 12}
    cen = conn.execute("SELECT centroid FROM lex_community WHERE build_id = %s AND cid = 1", (build,)).fetchone()[0]
    assert abs(float(np.linalg.norm(np.array(json.loads(cen)))) - 1.0) < 1e-4          # centroids are stored unit length
    ss.reset_label_tables(conn, label)
    conn.execute("DELETE FROM lex_build WHERE label = %s", (label,))
    with pytest.raises(AssertionError, match="no live build"):
        m.persist(ords, mean, E, st, ex, label=label, conn=conn)


def _chunk(doc, si, ci, text, junk=False, ref=False):
    return {"doc_id": doc, "section_idx": si, "chunk_idx": ci, "section_title": "Long results" if doc.endswith("0001") else "Other",
            "is_reference": ref, "is_junk": junk, "text": text}


def test_section_groups_orders_chunks_by_chunk_idx_and_leaves_junk_out():
    recs = [_chunk("arxiv/a", 0, 2, "c2"), _chunk("arxiv/a", 0, 0, "c0"), _chunk("arxiv/a", 0, 1, "c1", junk=True),
            _chunk("arxiv/a", 1, 0, "other"), _chunk("arxiv/b", 0, 0, "b0")]
    assert m.section_groups(recs) == {("arxiv/a", 0): [(0, "c0"), (2, "c2")], ("arxiv/a", 1): [(0, "other")], ("arxiv/b", 0): [(0, "b0")]}


def test_the_companion_file_holds_the_whole_section_of_every_selected_chunk_not_a_snippet(tmp_path):
    P, Q = "arxiv/2601_0001", "arxiv/2601_0002"
    allrecs = [_chunk(P, 0, 0, "## Long results\n\nfirst paragraph alpha"),
               _chunk(P, 0, 1, "## Long results\n\nsecond paragraph beta"),
               _chunk(P, 0, 2, "## Long results\n\nthird paragraph gamma ~~~ a fence inside and `` ticks"),
               _chunk(P, 0, 3, "x\n\ny\n\nz", junk=True),
               _chunk(Q, 1, 0, "## Other\n\nonly body"),
               _chunk(Q, 2, 0, "## References\n\n[1] cited", ref=True)]
    recs = [allrecs[i] for i in (0, 1, 2, 4)]                              # what the map clusters
    ex = {0: {"size": 3, "exemplars": [{"role": "medoid", "row": 1, "z": 0.0}]},         # the MIDDLE chunk was selected
          1: {"size": 2, "exemplars": [{"role": "medoid", "row": 3, "z": 0.0}]}}          # two sections: a community of one is not listed (A15)
    path = str(tmp_path / "c.md")
    assert m.write_sections_md(allrecs, recs, ex, {0: {"title": "Long results topic", "summary": "A draft summary."}}, path) == (2, 0)
    text = Path(path).read_text(encoding="utf-8")
    assert all(p in text for p in ("first paragraph alpha", "second paragraph beta", "third paragraph gamma"))   # the whole section
    assert text.count("## Long results") == 1                                # the continuation headers were dropped
    assert "x\n\ny\n\nz" not in text and "[1] cited" not in text             # junk and references are not part of it
    assert text.index("community 0") < text.index("community 1") and "A draft summary." in text and "Long results topic" in text
    assert "https://arxiv.org/abs/2601.0001" in text and "https://arxiv.org/abs/2601.0002" in text
    fences = [ln for ln in text.splitlines() if ln and set(ln) == {"~"}]
    assert len(fences) == 4 and all(len(f) >= 4 for f in fences[:2])        # longer than the ~~~ run inside the section


def test_a_caller_that_rebinds_OUT_MD_is_honoured_and_the_chunk_maps_file_is_left_alone(tmp_path, monkeypatch):
    """Regression, 2026-10-06: a default `path=OUT_MD` was bound at definition, so src/section_map.py (which rebinds m.OUT_MD) overwrote the chunk
    map's .tmp/arxiv_communities.md twice."""
    chunk_md = tmp_path / "chunk_map.md"
    chunk_md.write_text("the chunk map's own file", encoding="utf-8")
    other = tmp_path / "section_map.md"
    allrecs = [_chunk("arxiv/2601_0001", 0, 0, "## Long results\n\nonly body")]
    ex = {0: {"size": 2, "exemplars": [{"role": "medoid", "row": 0, "z": 0.0}]}}
    monkeypatch.setattr(m, "OUT_MD", str(other))
    assert m.write_sections_md(allrecs, allrecs, ex, {}) == (1, 0)
    assert "only body" in other.read_text(encoding="utf-8")
    assert chunk_md.read_text(encoding="utf-8") == "the chunk map's own file"


def test_a_known_paper_title_follows_the_section_heading_in_the_markdown_and_gets_a_line_on_the_card_A16(tmp_path):
    P = "arxiv/2601_0001"
    allrecs = [_chunk(P, 0, 0, "## Long results\n\nshared body"), _chunk("arxiv/2601_0002", 1, 0, "## Other\n\nother body")]
    ex = {0: {"size": 2, "n_papers": 2, "top_paper": "arxiv/2601_0001", "top_share": 0.5, "exemplars": [{"role": "medoid", "row": 0, "z": 0.0}, {"role": "z=1/2", "row": 1, "z": 0.5}]}}
    titles = {P: "A Survey of Speculative Decoding"}
    path = str(tmp_path / "t.md")
    m.write_sections_md(allrecs, allrecs, ex, {}, path, titles=titles)
    text = Path(path).read_text(encoding="utf-8")
    assert "### medoid, z=0.00: Long results — A Survey of Speculative Decoding [2601_0001](https://arxiv.org/abs/2601.0001)" in text
    assert "### z=1/2, z=0.50: Other [2601_0002]" in text                                              # no title for this paper: the heading is as before
    items, tot = m.card_plan(0, 0, ex[0], None, allrecs, None, None, None, titles)
    plain, tot0 = m.card_plan(0, 0, ex[0], None, allrecs)
    assert "A Survey of Speculative Decoding" in [t for t, *_ in items] and "A Survey of Speculative Decoding" not in [t for t, *_ in plain] and tot > tot0


def test_a_community_of_one_section_is_not_listed_and_the_markdown_says_how_many_were_left_out_A15(tmp_path):
    P = "arxiv/2601_0001"
    allrecs = [_chunk(P, 0, 0, "## Long results\n\nshared body"), _chunk("arxiv/2601_0002", 1, 0, "## Other\n\nlone body"), _chunk("arxiv/2601_0003", 1, 0, "## Other\n\nsecond lone body")]
    ex = {0: {"size": 2, "exemplars": [{"role": "medoid", "row": 0, "z": 0.0}]}, 1: {"size": 1, "exemplars": [{"role": "medoid", "row": 1, "z": 0.0}]},
          2: {"size": 1, "exemplars": [{"role": "medoid", "row": 2, "z": 0.0}]}}
    path = str(tmp_path / "one.md")
    assert m.write_sections_md(allrecs, allrecs, ex, {}, path) == (1, 0)                              # one section written: the community of two
    text = Path(path).read_text(encoding="utf-8")
    assert "shared body" in text and "lone body" not in text and "community 1" not in text and "community 2" not in text
    assert "2 communities of a single section are not listed" in text and m.MIN_COMMUNITY == 2


def test_a_section_longer_than_the_cap_is_cut_with_a_note_that_says_how_much_and_how_to_read_the_rest(tmp_path):
    big = "## Everything\n\n" + "\n\n".join("paragraph %04d of a book that came out as one section" % i for i in range(400))
    small = "## Small\n\nshort body"
    allrecs = [_chunk("arxiv/Book", 0, 0, big), _chunk("arxiv/2601_0001", 1, 0, small)]
    ex = {0: {"size": 2, "exemplars": [{"role": "medoid", "row": 0, "z": 0.0}, {"role": "z=1/2", "row": 1, "z": 0.5}]}}
    path = str(tmp_path / "c.md")
    n, cut = m.write_sections_md(allrecs, allrecs, ex, {}, path, cap=1000)
    text = Path(path).read_text(encoding="utf-8")
    assert (n, cut) == (2, 1)
    assert "paragraph 0000" in text and "paragraph 0399" not in text           # only the first `cap` characters of the long one
    assert "SECTION CUT: 1000 of %d characters shown" % len(big) in text
    assert "GET /section?doc_id=arxiv/Book&section_idx=0&max_chars=20000000" in text
    assert "short body" in text                                                 # a section under the cap is whole
    assert len(text) < 4000


def test_windows_cut_every_text_into_consecutive_token_windows_with_a_cap_and_an_owner_A10():
    import re
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m.MINILM)
    short = "graphs and communities in text"
    long = " ".join("word%d" % i for i in range(600))                        # several windows, under the cap
    huge = " ".join("w%d" % i for i in range(m.WIN * m.MAX_WIN * 3))          # more than the cap allows
    pieces, owner = m.windows([short, "", long, huge], tok)
    assert owner == sorted(owner) and set(owner) == {0, 1, 2, 3}              # every text, including the empty one, has a piece
    assert [p for p, o in zip(pieces, owner) if o == 0] == [short] and [p for p, o in zip(pieces, owner) if o == 1] == [""]
    long_pieces = [p for p, o in zip(pieces, owner) if o == 2]
    assert len(long_pieces) > 1 and all(len(tok(p, add_special_tokens=False)["input_ids"]) <= m.WIN for p in long_pieces)
    nonws = lambda s: re.sub(r"\s+", "", s)
    assert nonws("".join(long_pieces)) == nonws(long)                          # windows are consecutive: nothing lost, nothing repeated
    assert sum(1 for o in owner if o == 3) == m.MAX_WIN                        # the cap


def test_the_embedding_and_state_caches_have_their_own_names_so_first_window_caches_are_never_read_A10():
    assert m.EMB.endswith("_win.npy") and m.STATE.endswith("_win.npz") and m.KNN_SWEEP.endswith("_win.npz")
    assert {m.EMB, m.STATE, m.KNN_SWEEP} <= set(m.CACHES)                      # and the text-fingerprint purge covers them


def _marks_world():
    """Three communities of 20 points: 1, 2 and 3 selected exemplars."""
    rng = np.random.default_rng(0)
    N = 60
    XY = rng.normal(size=(N, 2))
    st = {"lab": np.repeat([0, 1, 2], 20), "XY": XY, "chosen": 3.0, "cons_ari": 0.9}
    pick = lambda c, k: [{"role": "medoid" if i == 0 else "z=%d/%d" % (i, k), "row": c * 20 + i, "z": i / k} for i in range(k)]
    ex = {0: {"exemplars": pick(0, 1)}, 1: {"exemplars": pick(1, 2)}, 2: {"exemplars": pick(2, 3)}}
    return XY, st, ex


def test_map_marks_gives_every_community_one_star_at_its_medoid_and_a_pick_per_other_exemplar_A9():
    XY, st, ex = _marks_world()
    stars, picks = m.map_marks(XY, ex)
    assert [c for c, _, _ in stars] == [0, 1, 2]
    assert all((x, y) == tuple(XY[ex[c]["exemplars"][0]["row"]]) for c, x, y in stars)           # the star sits on the medoid
    assert [c for c, *_ in picks] == [1, 2, 2]                                                    # 0 + 1 + 2 others: one exemplar draws no pick
    for c, x, y, sx, sy in picks:
        assert (sx, sy) == tuple(XY[ex[c]["exemplars"][0]["row"]]) and (x, y) != (sx, sy)         # each pick is joined to its own star


def test_the_map_draws_every_community_as_one_square_text_free_figure_A9(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(m, "OUT_PNG", str(tmp_path / "x.png"))
    XY, st, ex = _marks_world()
    m.render(st, ex)
    assert Image.open(tmp_path / "x.png").size == (m.FIG_IN * 80, m.FIG_IN * 80)               # no card grid to grow it: the size is fixed
    big = {c: {"exemplars": [{"role": "medoid", "row": c % 60, "z": 0.0}]} for c in range(253)}   # the live community count
    stars, picks = m.map_marks(XY, big)
    assert len(stars) == 253 and picks == []                                                   # every community, none dropped


def test_section_lines_drop_the_header_keep_paragraphs_wrap_and_say_when_text_was_cut_A11():
    rec = {"text": "## Heading\n\n" + "alpha " * 40 + "\n\nsecond paragraph here"}
    lines, cut = m.section_lines(rec, width=40, chars=10_000)
    assert not cut and "Heading" not in " ".join(lines)
    assert "" in lines and lines[-1] == "second paragraph here"                         # a blank line separates the paragraphs, none trails
    assert all(len(ln) <= 40 for ln in lines)
    short, cut = m.section_lines(rec, width=40, chars=50)
    assert cut and sum(len(ln) for ln in short) <= 50 + len(short)
    assert m.section_lines({"text": "header only"}) == ([], False)                   # no body, nothing drawn


def _card_world(n_comm=14, per=6, seed=1):
    rng = np.random.default_rng(seed)
    N = n_comm * per
    lab = np.repeat(np.arange(n_comm), per)
    recs = [{"doc_id": "arxiv/p%d" % (i // 2), "section_idx": i, "section_title": "Section %d" % i,
             "text": "## Section %d\n\n%s" % (i, ("word%d " % i) * (30 + 10 * (i % 5)))} for i in range(N)]
    ex = {c: {"size": per, "n_papers": 3, "top_paper": "arxiv/p%d" % (c * per // 2), "top_share": 0.5,
              "exemplars": [{"role": "medoid" if j == 0 else "z=%d/2" % j, "row": c * per + j, "z": j / 2} for j in range(1 + c % 3)]} for c in range(n_comm)}
    st = {"lab": lab, "XY": rng.normal(size=(N, 2)), "chosen": 3.0, "cons_ari": 0.9}
    return recs, st, ex


def test_card_plan_has_a_title_a_line_per_exemplar_and_a_total_height_A11():
    recs, st, ex = _card_world()
    items, total = m.card_plan(0, 5, ex[5], None, recs)
    texts = [t for t, *_ in items]
    assert texts[0].startswith("1  community 5  (6 %s)" % m.UNIT)
    assert sum(1 for t in texts if t.startswith(("medoid", "z="))) == len(ex[5]["exemplars"])
    assert total == pytest.approx(items[-1][1] + 0.05, abs=0.3) or total > items[-1][1]
    withs, tot2 = m.card_plan(0, 5, ex[5], {"title": "A topic", "summary": "word " * 60, "model": "x"}, recs)
    assert "A topic" in [t for t, *_ in withs] and tot2 > total                         # a summary is laid out whole and makes the card taller


def test_dunning_terms_are_the_label_until_a_summary_title_exists_and_sit_beside_it_after_A13(tmp_path):
    P = "arxiv/2601_0001"
    allrecs = [_chunk(P, 0, 0, "## A\n\nalpha body"), _chunk(P, 1, 0, "## B\n\nbeta body")]
    ex = {0: {"size": 3, "exemplars": [{"role": "medoid", "row": 0, "z": 0.0}]}, 1: {"size": 2, "exemplars": [{"role": "medoid", "row": 1, "z": 0.0}]}}
    terms = {0: ["kv cache", "eviction", "attention"], 1: ["graph"]}
    path = str(tmp_path / "t.md")
    m.write_sections_md(allrecs, allrecs, ex, {1: {"title": "Graph topic", "summary": "S."}}, path, terms=terms)
    text = Path(path).read_text(encoding="utf-8")
    assert "## community 0 (3 %s): kv cache · eviction · attention" % m.UNIT in text                  # no summary: the terms are the label
    assert "## community 1 (2 %s): Graph topic" % m.UNIT in text and "**Dunning terms:** graph" in text    # a summary title leads, the terms stay beside it
    path2 = str(tmp_path / "t2.md")
    s = {1: {"title": "Graph topic", "title_bold": "**Graph** topic", "summary": "About graph.", "summary_bold": "About **graph**.", "terms_surviving": ["graph"]}}
    m.write_sections_md(allrecs, allrecs, ex, s, path2, terms=terms)
    t2 = Path(path2).read_text(encoding="utf-8")
    assert "## community 1 (2 %s): **Graph** topic" % m.UNIT in t2 and "About **graph**." in t2 and "surviving into the summary, bold there: 1 of 1" in t2
    items, _ = m.card_plan(0, 0, {"size": 3, "n_papers": 1, "top_paper": "arxiv/p", "top_share": 1.0, "exemplars": ex[0]["exemplars"]}, None, allrecs, terms[0])
    assert "kv cache · eviction · attention" in [t for t, *_ in items]


def test_entities_get_their_own_line_beside_the_dunning_terms_in_the_markdown_and_on_the_card_A14(tmp_path):
    P = "arxiv/2601_0001"
    allrecs = [_chunk(P, 0, 0, "## A\n\nalpha body"), _chunk(P, 1, 0, "## B\n\nbeta body")]
    ex = {0: {"size": 3, "exemplars": [{"role": "medoid", "row": 0, "z": 0.0}]}, 1: {"size": 2, "exemplars": [{"role": "medoid", "row": 1, "z": 0.0}]}}
    path = str(tmp_path / "e.md")
    stats = {0: "2 of 3 sections mention an entity", 1: "0 of 2 sections mention an entity"}
    m.write_sections_md(allrecs, allrecs, ex, {}, path, terms={0: ["kv cache"]}, entities={0: ["harness 512", "verifier 169"]}, entity_stats=stats)
    text = Path(path).read_text(encoding="utf-8")
    assert "**Entities:** harness 512 · verifier 169 (2 of 3 sections mention an entity)" in text                      # names with their section counts, then the aggregate
    assert "**Entities:** none characteristic (0 of 2 sections mention an entity)" in text and text.count("**Entities:**") == 2   # a community with stats and no entity says so
    path_b = str(tmp_path / "e2.md")
    m.write_sections_md(allrecs, allrecs, ex, {}, path_b, terms={0: ["kv cache"]}, entities={0: ["harness"]})
    assert "**Entities:** harness\n" in Path(path_b).read_text(encoding="utf-8") and Path(path_b).read_text(encoding="utf-8").count("**Entities:**") == 1      # no stats given: names only; none for community 1
    card = {"size": 3, "n_papers": 1, "top_paper": "arxiv/p", "top_share": 1.0, "exemplars": ex[0]["exemplars"]}
    with_e, tot_e = m.card_plan(0, 0, card, None, allrecs, ["kv cache"], ["harness 512", "verifier 169"], "2 of 3 sections mention an entity")
    without, tot = m.card_plan(0, 0, card, None, allrecs, ["kv cache"])
    texts = [t for t, *_ in with_e]
    assert "entities: harness 512 · verifier 169" in texts and "2 of 3 sections mention an entity" in texts and tot_e > tot
    assert not [t for t, *_ in without if t.startswith("entities:")]


def test_render_cards_draws_the_map_and_the_twelve_largest_as_cards_in_a_figure_that_grows_A11(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(m, "OUT_PNG", str(tmp_path / "cards.png"))
    monkeypatch.setattr(m, "UNIT", "sections")
    recs, st, ex = _card_world(n_comm=14)
    out = m.render_cards(recs, st, ex, {})
    w, h = Image.open(out).size
    assert out == str(tmp_path / "cards.png") and w == m.FIG_W * 80 and h >= m.FIG_H * 80      # the floor height, grown when a card needs it
    with pytest.raises(ValueError):
        monkeypatch.setattr(m, "MAX_FIG_H", 3)                                                  # a cap too small for the longest card is an error, not a clipped card
        m.render_cards(recs, st, ex, {})


@pytest.mark.skipif(sys.platform != "win32", reason="the failure is Windows' refusal to truncate a memory-mapped file")
def test_a_png_held_open_by_a_viewer_goes_to_a_timestamped_sibling_instead_of_crashing(tmp_path, capsys):
    import mmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(1, 1))
    stable = str(tmp_path / "map.png")
    assert m.save_figure(fig, stable) == stable                          # nothing holds it: the stable name is written
    first = Path(stable).read_bytes()
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    with open(stable, "rb") as holder, mmap.mmap(holder.fileno(), 0, access=mmap.ACCESS_READ):    # a viewer showing the file
        with pytest.raises(OSError):
            open(stable, "w+b").close()                                  # the condition itself, Errno 22
        fig.text(0.5, 0.5, "second")
        written = m.save_figure(fig, stable)
    assert written != stable and Path(written).name.startswith("map.") and written.endswith(".png")
    assert Path(stable).read_bytes() == first                            # the held file is untouched
    assert Path(written).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    out = capsys.readouterr().out
    assert "held open by another program" in out and stable in out and written in out
    assert m.save_figure(fig, stable) == stable                          # viewer closed: the stable name works again
    plt.close(fig)
