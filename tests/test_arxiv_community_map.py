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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import arxiv_community_map as m
import hnsw_communities as hc


def _unit(X):
    X = np.asarray(X, np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_mpl_escapes_dollars_so_latex_chunks_do_not_crash_the_render():
    """Regression: '$$\\pi ^ { * } ...' raised ParseException and killed the whole figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = "reward at each step, i.e., $$\\pi ^ { * } ( s..."
    assert m.mpl(s) == "reward at each step, i.e., \\$\\$\\pi ^ { * } ( s..."
    fig = plt.figure()
    fig.text(0.1, 0.5, m.mpl(s))
    fig.canvas.draw()
    plt.close(fig)


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


def test_a_card_whose_text_cannot_fit_raises_instead_of_clipping(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUT_PNG", str(tmp_path / "x.png"))
    rng = np.random.default_rng(0)
    N = 60
    lab = np.repeat([0, 1, 2], 20)
    recs = [{"doc_id": "arxiv/%d" % i, "section_idx": i, "chunk_idx": 0, "section_title": "T", "text": "## T\n\nbody"} for i in range(N)]
    st = {"lab": lab, "XY": rng.normal(size=(N, 2)), "chosen": 3.0, "cons_ari": 0.9}
    ex = {c: {"size": 20, "n": 1, "n_papers": 3, "top_paper": "arxiv/0", "top_share": 0.5,
              "exemplars": [{"role": "medoid", "row": c * 20, "z": 0.0, "key": ["arxiv/%d" % (c * 20), c * 20, 0]}]} for c in range(3)}
    summ = {0: {"title": "t", "model": "x", "summary": " ".join("sentence number %d." % i for i in range(400))}}
    with pytest.raises(ValueError, match="would be clipped"):
        m.render(recs, st, ex, summ)
    m.render(recs, st, ex, {})                                     # without the oversized summary it renders
    assert (tmp_path / "x.png").exists()


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


def _cards(n_sentences):
    rng = np.random.default_rng(0)
    N = 60
    lab = np.repeat([0, 1, 2], 20)
    recs = [{"doc_id": "arxiv/%d" % i, "section_idx": i, "chunk_idx": 0, "section_title": "T", "text": "## T\n\nbody"} for i in range(N)]
    st = {"lab": lab, "XY": rng.normal(size=(N, 2)), "chosen": 3.0, "cons_ari": 0.9}
    ex = {c: {"size": 20, "n": 1, "n_papers": 3, "top_paper": "arxiv/0", "top_share": 0.5,
              "exemplars": [{"role": "medoid", "row": c * 20, "z": 0.0, "key": ["arxiv/%d" % (c * 20), c * 20, 0]}]} for c in range(3)}
    summ = {1: {"title": "t", "model": "x", "summary": " ".join("sentence number %d." % i for i in range(n_sentences))}}
    return recs, st, ex, summ


def test_the_figure_grows_to_the_longest_card_instead_of_clipping_or_failing(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(m, "OUT_PNG", str(tmp_path / "x.png"))
    recs, st, ex, short = _cards(8)
    m.render(recs, st, ex, short)
    floor = Image.open(tmp_path / "x.png").size[1]
    assert floor == int(m.FIG_H * 80)                                           # a short summary: the floor height
    recs, st, ex, longer = _cards(100)                                          # needs more than the floor allows
    items, total = m.card_plan(0, 1, ex[1], longer[1], recs)
    assert total > m.FIG_H * 0.2175 - m.CARD_MARGIN                              # it really would not fit at the floor
    m.render(recs, st, ex, longer)
    assert Image.open(tmp_path / "x.png").size[1] > floor                       # so the figure grew
    items, total = m.card_plan(0, 1, ex[1], longer[1], recs)
    assert [t for t, *_ in items if "sentence number 99." in t]                  # the whole summary is laid out, none cut


def test_card_plan_lays_out_every_line_of_the_summary_and_every_exemplar():
    recs, st, ex, summ = _cards(30)
    items, total = m.card_plan(0, 1, ex[1], summ[1], recs)
    text = " ".join(t for t, *_ in items)
    assert all(("sentence number %d." % i) in text for i in range(30))
    assert total == pytest.approx(items[-1][1] + 0.16 + 0.05, abs=1e-9)         # the last snippet line, then the gap after it
    ys = [y for _, y, _, _ in items]
    assert ys == sorted(ys)                                                    # top to bottom, never overlapping upward
    no_summary = m.card_plan(0, 0, ex[0], None, recs)[0]
    assert any("no summary" in t for t, *_ in no_summary)


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
