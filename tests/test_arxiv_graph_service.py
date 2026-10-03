"""Pins arxiv_graph_service.py: the slot arithmetic that keeps it behind its sibling, the sibling-log reader, the
single-cycle lock, the pipeline state machine, and the incremental ingest under a build's FROZEN statistics.

Real Postgres in a database this suite owns (`graph_sparsevec_test`), a fake embedder, no GPU, no corpus.
Skipped, with the reason, only when no server answers.

Run:  pytest tests/test_arxiv_graph_service.py -v
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import arxiv_graph_service as svc
import sparsevec_store as ss
from ingest_arxiv_sparsevec import B, K1, doc_row, full_views
from salient_grams import bm25_matrix
from stoplist import tokenize

TEST_DB, LABEL = "graph_sparsevec_test", "zz_svc"
FIT = {"lcap": 1992, "m": 10, "hi": 151}
BASE = ["## A\n\nalpha graph node edge community cluster", "## B\n\nbeta sparse vector index query ranking",
        "## C\n\nalpha beta graph sparse network"]
SOUP = "## Soup\n\n" + "\n\n".join("abcdefghij"[i % 10] for i in range(40))
ALPHA_DOC = "## Alpha result\n\nalpha graph community edge network node"
BETA_DOC = "## Beta study\n\nbeta sparse vector index query.\n\n## References\n\n[1] A. Author. Beta methods. 2020."


def fake_embed(texts):
    return np.array([[1.0, 0, 0, 0] if "alpha" in t else [0, 1.0, 0, 0] for t in texts], np.float32)


@pytest.fixture()
def conn():
    try:
        with psycopg.connect(ss.DSN, autocommit=True) as admin:
            if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone():
                admin.execute(f'CREATE DATABASE "{TEST_DB}"')
        c = psycopg.connect(ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB, autocommit=True)
    except Exception as exc:                                            # pragma: no cover
        pytest.skip(f"no Postgres at {ss.DSN}: {exc}")
    yield c
    c.execute("DELETE FROM lex_build WHERE label = %s", (LABEL,))
    c.execute("DELETE FROM lex_chunk_meta WHERE label = %s", (LABEL,))
    ss.reset_label_tables(c, LABEL)
    c.close()


def make_build(c, summaries: bool = True, communities: bool = True) -> int:
    """A complete miniature build: 3 chunks, frozen statistics, 2 communities with unit centroids."""
    tokens = [tokenize(t) for t in BASE]
    BM, TF, terms, df = bm25_matrix(tokens, K1, B)
    N, dim = BM.shape
    sat_t, cos_t, idf, whales = full_views(BM.tocsr(), df, N)
    c.execute("DELETE FROM lex_build WHERE label = %s", (LABEL,))
    ss.reset_label_tables(c, LABEL)
    ss.ensure_schema(c, LABEL, dim)
    ss.ensure_chunk_meta(c)
    ss.ensure_view_schema(c, LABEL, dim, 4)
    ss.ensure_build_schema(c)
    ss.write_vocab(c, LABEL, [str(t) for t in terms], idf=idf)
    recs = [{"doc_id": "arxiv/base%d" % i, "section_idx": 0, "chunk_idx": 0, "section_title": "T", "is_reference": False,
             "is_junk": False, "text": t} for i, t in enumerate(BASE)]
    ss.write_chunk_meta(c, LABEL, recs)

    def rows(M):
        return [(i, recs[i]["doc_id"], "arxiv", dict(zip(M.indices[M.indptr[i]:M.indptr[i + 1]].tolist(),
                                                           M.data[M.indptr[i]:M.indptr[i + 1]].tolist()))) for i in range(N)]
    ss.write_chunks(c, LABEL, dim, rows(sat_t))
    ss.write_chunks(c, LABEL, dim, rows(cos_t), kind="cos")
    E = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.7, 0.7, 0, 0]], np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    ss.write_dense(c, LABEL, [0, 1, 2], E)
    params = {"bm25": {"k1": K1, "b": B, "avgdl": float(np.mean([len(t) for t in tokens])), "n_docs": N, "dim": dim,
                       "whale_cols": whales.tolist()}, "fit": FIT, "dense_dim": 4}
    if communities:
        params["dense"] = {"model": "fake", "dim": 4, "mean": [0.0, 0.0, 0.0, 0.0]}
        params["communities"] = {"n": 2}
    build = ss.new_build(c, LABEL, 3, params)
    if communities:
        ss.write_communities(c, build, [(0, 2, [1.0, 0, 0, 0]), (1, 1, [0, 1.0, 0, 0])])
        ss.write_assignments(c, build, [(0, 0, "consensus", 0.0, 0.0), (1, 1, "consensus", 1.0, 1.0), (2, 0, "consensus", 2.0, 2.0)])
    if summaries:
        ss.write_summaries(c, build, [{"community": k, "status": "draft", "title": "t", "summary": "s", "chunk_keys": []} for k in (0, 1)])
    return build


def fresh_map(tmp_path, monkeypatch, age_s: float = -3600.0) -> Path:
    """A map PNG whose mtime is `age_s` seconds from now (negative = in the past), in a directory the service reads."""
    import os
    import time
    d = tmp_path / "png"
    d.mkdir(exist_ok=True)
    png = d / "arxiv_community_map.png"
    png.write_bytes(b"png")
    t = time.time() + age_s
    os.utime(png, (t, t))
    monkeypatch.setattr(svc, "MAP_DIR", d)
    return png


def vec_of(c, kind, ord_):
    """{col0: weight} of one stored sparsevec row, or None."""
    r = c.execute(f"SELECT vec::text FROM {ss._table(LABEL, kind)} WHERE ord = %s", (ord_,)).fetchone()
    if r is None:
        return None
    body = r[0].split("/")[0].strip("{}")
    return {int(a) - 1: float(b) for a, b in (kv.split(":") for kv in body.split(",") if kv)}


# ------------------------------------------------------------------- schedule ----
def test_slots_fall_two_minutes_after_the_sibling_and_are_strictly_later_than_now():
    at = lambda h, m, s=0, d=3: dt.datetime(2026, 10, d, h, m, s)
    assert svc.next_slot(at(13, 17, 30), 46) == at(13, 18)            # sibling fires :01 :16 :31 :46 -> slots :18 and :48
    assert svc.next_slot(at(13, 18), 46) == at(13, 48)                # not "now": strictly after
    assert svc.next_slot(at(13, 48, 1), 46) == at(14, 18)
    assert svc.next_slot(at(23, 50), 46) == at(0, 18, d=4)            # across midnight
    assert svc.next_slot(at(13, 0), None) == at(13, 17)               # sibling unknown: phase 17, documented fallback


def test_every_cycle_of_the_sibling_gives_the_same_slots_and_a_drifted_sibling_moves_them():
    now = dt.datetime(2026, 10, 3, 13, 0)
    assert {svc.next_slot(now, m) for m in (1, 16, 31, 46)} == {dt.datetime(2026, 10, 3, 13, 18)}
    assert svc.next_slot(now, 14).minute == 1                          # the sibling slid to :14 -> (14 + 17) % 30
    assert svc.next_slot(now, 14) > now


def test_the_sibling_log_is_read_as_utf16_or_utf8_from_its_tail_and_garbage_yields_none(tmp_path):
    lines = ["[watch ] 2026-10-03 13:31 VRAM 53% used\n", "[watch ] 2026-10-03 13:46 VRAM 13% - running cycle\n", "Pending papers: 0\n"]
    u16 = tmp_path / "u16.log"
    u16.write_bytes(("﻿" + "".join(lines)).encode("utf-16-le"))
    assert svc.sibling_minute(u16) == 46
    u8 = tmp_path / "u8.log"
    u8.write_text("".join(lines), encoding="utf-8")
    assert svc.sibling_minute(u8) == 46
    big = tmp_path / "big.log"                                          # far larger than the tail that is read
    big.write_bytes(("﻿" + "filler line that is not a stamp\n" * 20000 + lines[0]).encode("utf-16-le"))
    assert svc.sibling_minute(big) == 31
    junk = tmp_path / "junk.log"
    junk.write_text("no stamps here\n", encoding="utf-8")
    assert svc.sibling_minute(junk) is None and svc.sibling_minute(tmp_path / "missing.log") is None


def test_one_cycle_at_a_time_a_live_holder_blocks_and_a_dead_one_is_taken_over(tmp_path):
    lock = tmp_path / "x.lock"
    assert svc.acquire_lock(lock) is True and lock.read_text().strip() == str(__import__("os").getpid())
    assert svc.acquire_lock(lock) is True                               # re-entrant for the same process
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        lock.write_text(str(other.pid))
        assert svc.acquire_lock(lock) is False                          # a live other process holds it
        assert lock.read_text().strip() == str(other.pid)               # and the file is left alone
    finally:
        other.kill()
        other.wait()
    assert svc.acquire_lock(lock) is True                               # the holder is dead: stale, taken over
    lock.write_text("not a pid")
    assert svc.acquire_lock(lock) is True                               # an unreadable lock is stale too
    svc.release_lock(lock)
    assert not lock.exists()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        lock.write_text(str(other.pid))
        svc.release_lock(lock)                                          # never removes a lock it does not own
        assert lock.exists()
    finally:
        other.kill()
        other.wait()


# ------------------------------------------------------------------- pipeline ----
def test_pending_steps_names_what_the_live_build_still_lacks(conn, tmp_path, monkeypatch):
    ss.ensure_build_schema(conn)
    conn.execute("DELETE FROM lex_build WHERE label = %s", (LABEL,))
    assert svc.pending_steps(conn, LABEL, tmp_path) == ["chunk", "ingest", "map", "summarize", "render"]
    make_build(conn, summaries=False, communities=False)
    assert svc.pending_steps(conn, LABEL, tmp_path) == ["map", "summarize", "render"]
    make_build(conn, summaries=False, communities=True)
    assert svc.pending_steps(conn, LABEL, tmp_path) == ["summarize", "render"]
    make_build(conn, summaries=True, communities=True)
    assert svc.pending_steps(conn, LABEL, tmp_path) == ["render"]                    # everything but the picture


def test_a_finished_build_whose_render_failed_is_resumed_at_render_and_a_viewer_fallback_picture_counts(conn, tmp_path, monkeypatch):
    make_build(conn)
    png = fresh_map(tmp_path, monkeypatch, age_s=-3600)                                # a picture from before the build
    assert svc.pending_steps(conn, LABEL, png.parent) == ["render"]
    calls = []
    out = svc.cycle(LABEL, root=str(tmp_path), conn=conn, rebuild_fn=calls.append)
    assert calls == [["render"]] and out == {"resumed": ["render"]}
    fresh_map(tmp_path, monkeypatch, age_s=+3600)                                       # a picture newer than the build
    assert svc.pending_steps(conn, LABEL, png.parent) == []
    png.unlink()
    sibling = png.parent / "arxiv_community_map.20261003-145900.png"                    # what save_figure writes when a viewer holds the stable name
    sibling.write_bytes(b"png")
    import os
    import time
    os.utime(sibling, (time.time() + 3600, time.time() + 3600))
    assert svc.pending_steps(conn, LABEL, png.parent) == []


def test_frozen_reads_the_statistics_vocabulary_and_unit_centroids_of_the_live_build(conn):
    make_build(conn)
    st = svc.frozen(conn, LABEL)
    tokens = [tokenize(t) for t in BASE]
    BM, TF, terms, df = bm25_matrix(tokens, K1, B)
    assert st["col"] == {str(t): j for j, t in enumerate(terms)} and st["dim"] == len(terms)
    assert np.allclose(st["idf"], full_views(BM.tocsr(), df, 3)[2], rtol=1e-5)
    assert st["centroids"].shape == (2, 4) and st["cids"] == [0, 1] and st["fit"] == FIT and st["n_chunks"] == 3
    conn.execute("DELETE FROM lex_build WHERE label = %s", (LABEL,))
    make_build(conn, communities=False)
    with pytest.raises(AssertionError, match="map stage has not run"):
        svc.frozen(conn, LABEL)


# --------------------------------------------------------------------- ingest ----
def test_new_documents_are_stored_flagged_vectorised_with_frozen_statistics_and_placed(conn):
    make_build(conn)
    st = svc.frozen(conn, LABEL)
    ids = ["arxiv/new_alpha", "arxiv/new_soup", "arxiv/new_beta", "arxiv/new_alpha_copy"]
    out = svc.ingest_new(conn, LABEL, ids, [ALPHA_DOC, SOUP, BETA_DOC, ALPHA_DOC], st, fake_embed)
    assert out == {"docs": 4, "chunks": 4, "duplicates": 1, "junk": 1, "references": 1, "placed": 2, "replaced_chunks": 0}
    meta = conn.execute("SELECT ord, doc_id, is_junk, is_reference FROM lex_chunk_meta WHERE label = %s AND ord >= 3 ORDER BY ord",
                        (LABEL,)).fetchall()
    assert meta == [(3, "arxiv/new_alpha", False, False), (4, "arxiv/new_soup", True, False),
                    (5, "arxiv/new_beta", False, False), (6, "arxiv/new_beta", False, True)]
    text3 = conn.execute("SELECT text FROM lex_chunk_meta WHERE label = %s AND ord = 3", (LABEL,)).fetchone()[0]
    sat, cos = doc_row(tokenize(text3), st["col"], st["idf"], st["avgdl"], st["whales"])
    got_sat, got_cos = vec_of(conn, "chunk", 3), vec_of(conn, "cos", 3)
    assert got_sat.keys() == sat.keys() and all(abs(got_sat[j] - sat[j]) < 1e-4 for j in sat)         # S1: the row a build gives it
    assert got_cos.keys() == cos.keys() and all(abs(got_cos[j] - cos[j]) < 1e-4 for j in cos)
    assert vec_of(conn, "chunk", 4) is None and vec_of(conn, "cos", 4) is None                       # S3: junk has no vectors
    assert conn.execute(f"SELECT count(*) FROM {ss._table(LABEL, 'dense')} WHERE ord = 4").fetchone()[0] == 0
    assert vec_of(conn, "chunk", 6) is not None                                                       # a reference keeps sparse rows...
    assert conn.execute(f"SELECT count(*) FROM {ss._table(LABEL, 'dense')} WHERE ord = 6").fetchone()[0] == 0   # ...no dense vector
    assert conn.execute(f"SELECT ord FROM {ss._table(LABEL, 'dense')} WHERE ord >= 3 ORDER BY ord").fetchall() == [(3,), (5,)]
    placed = conn.execute("SELECT ord, cid, how, x, y FROM lex_assign WHERE build_id = %s AND how = 'nearest-centroid' ORDER BY ord",
                          (st["build"],)).fetchall()
    assert placed == [(3, 0, "nearest-centroid", None, None), (5, 1, "nearest-centroid", None, None)]   # S4; alpha -> 0, beta -> 1


def test_ingesting_the_same_documents_again_adds_nothing_and_a_known_text_is_skipped(conn, tmp_path, monkeypatch):
    make_build(conn)
    fresh_map(tmp_path, monkeypatch, age_s=+3600)
    st = svc.frozen(conn, LABEL)
    svc.ingest_new(conn, LABEL, ["arxiv/a"], [ALPHA_DOC], st, fake_embed)
    again = svc.ingest_new(conn, LABEL, ["arxiv/b"], [ALPHA_DOC], st, fake_embed)               # new doc id, text already stored
    assert again["chunks"] == 0 and again["duplicates"] == 1 and again["placed"] == 0
    assert conn.execute("SELECT count(*) FROM lex_chunk_meta WHERE label = %s", (LABEL,)).fetchone()[0] == 4
    (tmp_path / "x1.md").write_text(BETA_DOC, encoding="utf-8")
    calls = []
    first = svc.cycle(LABEL, root=str(tmp_path), embed_fn=fake_embed, conn=conn, rebuild_fn=calls.append)
    assert first["new_docs"] == 1 and first["chunks"] == 2
    second = svc.cycle(LABEL, root=str(tmp_path), embed_fn=fake_embed, conn=conn, rebuild_fn=calls.append)
    assert second["new_docs"] == 0 and "chunks" not in second                                        # S2: nothing new, nothing touched
    assert conn.execute("SELECT count(*) FROM lex_chunk_meta WHERE label = %s", (LABEL,)).fetchone()[0] == 6


# ----------------------------------------------------------------- the trigger ----
def test_a_rebuild_starts_exactly_when_added_chunks_reach_the_fraction_of_the_build(conn, tmp_path, monkeypatch):
    make_build(conn)
    fresh_map(tmp_path, monkeypatch, age_s=+3600)
    conn.execute("UPDATE lex_build SET n_chunks = 20 WHERE label = %s", (LABEL,))
    filler = [{"doc_id": "arxiv/fill%d" % i, "section_idx": 0, "chunk_idx": 0, "section_title": "T", "is_reference": False,
               "is_junk": True, "text": "## F\n\nfiller %d" % i} for i in range(18)]
    ss.append_chunk_meta(conn, LABEL, filler)                          # 3 + 18 = 21 rows against a build of 20
    calls = []
    out = svc.cycle(LABEL, root=str(tmp_path), conn=conn, rebuild_fn=calls.append)
    assert out["added_fraction"] == 0.05 and out["rebuild"] is True and calls == [list(svc.STEPS)]   # 0.05 is inclusive
    conn.execute("UPDATE lex_build SET n_chunks = 21 WHERE label = %s", (LABEL,))
    calls.clear()
    out = svc.cycle(LABEL, root=str(tmp_path), conn=conn, rebuild_fn=calls.append)
    assert out["added_fraction"] == 0.0 and out["rebuild"] is False and calls == []
    conn.execute("UPDATE lex_build SET n_chunks = 22 WHERE label = %s", (LABEL,))                    # one of the build's own ords is gone
    out = svc.cycle(LABEL, root=str(tmp_path), conn=conn, rebuild_fn=calls.append)
    assert out["added_fraction"] == round(1 / 22, 4) and out["rebuild"] is False                      # 0.0455: overwritten counts as change


def test_an_incomplete_build_is_resumed_at_its_first_missing_step_before_anything_is_ingested(conn, tmp_path):
    make_build(conn, summaries=False, communities=True)
    (tmp_path / "x1.md").write_text(ALPHA_DOC, encoding="utf-8")
    calls = []
    out = svc.cycle(LABEL, root=str(tmp_path), embed_fn=fake_embed, conn=conn, rebuild_fn=calls.append)
    assert calls == [["summarize", "render"]] and out == {"resumed": ["summarize", "render"]}
    assert conn.execute("SELECT count(*) FROM lex_chunk_meta WHERE label = %s", (LABEL,)).fetchone()[0] == 3    # nothing ingested yet


# ------------------------------------------------------------------- versions ----
def _section(title: str, tag: str, paragraphs: int = 12) -> str:
    """A section big enough to close its own chunk (the chunker merges anything under 10 newline units)."""
    return "## %s\n\n" % title + "\n\n".join("alpha graph %s paragraph %d community edge network" % (tag, i) for i in range(paragraphs))


SHARED = _section("Shared introduction", "shared")
PAPER_V1 = SHARED + "\n\n" + _section("Results", "firstversion")
PAPER_V2 = SHARED + "\n\n" + _section("Results", "secondversion")


def _rows(conn, doc_id):
    return conn.execute("SELECT ord, version, text FROM lex_chunk_meta WHERE label = %s AND doc_id = %s ORDER BY ord",
                        (LABEL, doc_id)).fetchall()


def _cycle(conn, root, calls):
    return svc.cycle(LABEL, root=str(root), embed_fn=fake_embed, conn=conn, rebuild_fn=calls.append)


def test_a_higher_version_overwrites_the_paper_and_the_shared_chunks_survive_the_swap(conn, tmp_path, monkeypatch):
    make_build(conn)
    fresh_map(tmp_path, monkeypatch, age_s=+3600)
    calls = []
    _touch_md(tmp_path, **{"2601_0001": PAPER_V1})                                       # no version in the name: v1
    first = _cycle(conn, tmp_path, calls)
    assert first["new_docs"] == 1 and first["upgraded_docs"] == 0 and first["chunks"] == 2
    held = _rows(conn, "arxiv/2601_0001")
    assert [v for _, v, _ in held] == [1, 1] and any("firstversion" in t for _, _, t in held)
    old_ords = [o for o, _, _ in held]

    _touch_md(tmp_path, **{"2601_0001v2": PAPER_V2})
    second = _cycle(conn, tmp_path, calls)
    assert second["new_docs"] == 0 and second["upgraded_docs"] == 1
    assert second["replaced_chunks"] == 2 and second["chunks"] == 2                      # the shared chunk was NOT lost as a duplicate
    now = _rows(conn, "arxiv/2601_0001")
    assert [v for _, v, _ in now] == [2, 2]
    texts = [t for _, _, t in now]
    assert sum("shared paragraph" in t for t in texts) == 1 and any("secondversion" in t for t in texts)
    assert not any("firstversion" in t for t in texts)                                   # the old results section is gone
    assert ss.stored_versions(conn, LABEL)["arxiv/2601_0001"] == 2
    live = ss.live_build(conn, LABEL)[0]
    ords = [o for o, _, _ in now]
    for kind in ("chunk", "cos", "dense"):
        assert [r[0] for r in conn.execute(f"SELECT ord FROM {ss._table(LABEL, kind)} WHERE ord >= 3 ORDER BY ord").fetchall()] == ords, kind
    assert [r[0] for r in conn.execute("SELECT ord FROM lex_assign WHERE build_id = %s AND ord >= 3 ORDER BY ord", (live,)).fetchall()] == ords
    assert all(o >= 3 for o in ords) and old_ords                                         # never into the build's own ords (0-2)


def test_the_same_or_a_lower_version_is_never_read_and_the_highest_file_present_wins(conn, tmp_path, monkeypatch):
    make_build(conn)
    fresh_map(tmp_path, monkeypatch, age_s=+3600)
    calls = []
    _touch_md(tmp_path, **{"2601_0002v3": PAPER_V2})
    assert _cycle(conn, tmp_path, calls)["new_docs"] == 1
    before = _rows(conn, "arxiv/2601_0002")
    assert [v for _, v, _ in before] == [3, 3]
    _touch_md(tmp_path, **{"2601_0002v3": PAPER_V1})                                     # the same version, different text
    again = _cycle(conn, tmp_path, calls)
    assert again["new_docs"] == 0 and again["upgraded_docs"] == 0 and "chunks" not in again
    (tmp_path / "2601_0002v3.md").unlink()
    _touch_md(tmp_path, **{"2601_0002v2": PAPER_V1, "2601_0002": PAPER_V1})              # only older ones on disk now
    older = _cycle(conn, tmp_path, calls)
    assert older["new_docs"] == 0 and older["upgraded_docs"] == 0 and "chunks" not in older
    assert _rows(conn, "arxiv/2601_0002") == before                                      # untouched, text and all


def test_a_methods_extract_is_a_part_of_the_paper_not_a_version_of_it(conn, tmp_path, monkeypatch):
    make_build(conn)
    fresh_map(tmp_path, monkeypatch, age_s=+3600)
    calls = []
    _touch_md(tmp_path, **{"2601_0003": PAPER_V1})
    _cycle(conn, tmp_path, calls)
    _touch_md(tmp_path, **{"2601_0003_methods": _section("Methods", "methodsonly")})
    out = _cycle(conn, tmp_path, calls)
    assert out["new_docs"] == 1 and out["upgraded_docs"] == 0                            # a new paper key, not an overwrite
    held = ss.stored_versions(conn, LABEL)
    assert held["arxiv/2601_0003"] == 1 and held["arxiv/2601_0003_methods"] == 1
    assert any("firstversion" in t for _, _, t in _rows(conn, "arxiv/2601_0003"))        # the main paper is intact


def test_a_paper_stored_with_its_version_in_the_id_is_recognised_and_not_ingested_a_second_time(conn, tmp_path, monkeypatch):
    make_build(conn)
    fresh_map(tmp_path, monkeypatch, age_s=+3600)
    ss.append_chunk_meta(conn, LABEL, [{"doc_id": "arxiv/2601_0009v1", "section_idx": 0, "chunk_idx": 0, "section_title": "T",
                                        "is_reference": False, "is_junk": False, "text": "stored before versions were a key"}], min_ord=3)
    _touch_md(tmp_path, **{"2601_0009v1": PAPER_V1})                                       # the file the id came from
    out = _cycle(conn, tmp_path, [])
    assert out["new_docs"] == 0 and out["upgraded_docs"] == 0 and "chunks" not in out
    held = ss.stored_versions(conn, LABEL)
    assert held["arxiv/2601_0009"] == 1 and "arxiv/2601_0009v1" not in held
    conn.execute("ALTER TABLE lex_chunk_meta DROP COLUMN version")                         # a database from before the column existed
    _touch_md(tmp_path, **{"2601_0010": PAPER_V2})
    assert _cycle(conn, tmp_path, [])["new_docs"] == 1                                       # the cycle adds the column itself
    assert ss.stored_versions(conn, LABEL)["arxiv/2601_0010"] == 1


def test_overwritten_chunks_count_toward_the_rebuild_trigger(conn):
    make_build(conn)
    conn.execute("UPDATE lex_chunk_meta SET doc_id = 'arxiv/2601_0100' WHERE label = %s AND ord = 0", (LABEL,))
    assert svc.changed_fraction(conn, LABEL, 3) == 0.0
    assert ss.delete_papers(conn, LABEL, ["arxiv/2601_0100"]) == 1                       # one of the build's three chunks overwritten away
    assert svc.changed_fraction(conn, LABEL, 3) == round(1 / 3, 4)
    ss.append_chunk_meta(conn, LABEL, [{"doc_id": "arxiv/2601_0100", "version": 2, "section_idx": 0, "chunk_idx": 0,
                                        "section_title": "T", "is_reference": False, "is_junk": False, "text": "replacement"}], min_ord=3)
    assert svc.changed_fraction(conn, LABEL, 3) == round(2 / 3, 4)                       # the replacement is an addition on top


def _touch_md(root, **files):
    for name, body in files.items():
        (root / (name + ".md")).write_text(body, encoding="utf-8")
