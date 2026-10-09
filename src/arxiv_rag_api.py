"""arxiv_rag_api.py -- the arxiv graph as a retrieval service: hybrid search that returns WHOLE SECTIONS.

NO GOVERNING SPEC beyond operator request 2026-10-03: "how do I use this rag myself? ... create a function for
openwebui to use this backend to reconstruct whole sections for use with rag". Tasks: playbook.md T125, T126.
Skill: sparsevec-lexsem-graph (R6 query-side idf, R9 centered dense).

    STAGE     MECHANISM                                                                            GUARD
    SPARSE    inner product over saturated tf, query weights idf*qtf from the BUILD's vocabulary    Q1
    DENSE     MiniLM on the CPU, the build's frozen mean, cosine over lex_dense                      Q1
    FUSE      reciprocal rank fusion, k = 60: ranks, not scores                                      Q2
    FILTER    references and junk out unless asked                                                   Q4
    EXPAND    each hit -> its whole section, rebuilt from the chunks that share (doc_id, section)   Q3
    ANNOTATE  community id and its draft title; arXiv link; version

Q1  A query is weighted with the live build's own vocabulary and idf. A term the build never saw carries no
    weight and is reported, not silently dropped; with no known term the sparse arm is empty and says so.
Q2  The arms' scores are not comparable (an inner product against a cosine), so they are fused by rank.
Q3  A section is rebuilt, not stored, by domain_corpora.reaggregate_section (C12): its chunks in chunk_idx order,
    the header the chunker prepended to each continuation chunk dropped, NO overlap trimming (none is stored;
    a trim would delete repeats the source contains). Junk chunks (OCR debris) are left out and counted in
    `junk_chunks_omitted`. A chunk is keyed by its FIRST section, so a small section merged into a chunk after it
    comes along with it. The whole section is returned: `max_chars` defaults to 200,000 (the 99.9th percentile
    of section size is 79,000) and a cut is NEVER silent: `truncated` is set and `chars_total` says what was left out.
Q4  Reference and junk chunks are stored and flagged (C9, C10); search leaves them out by default.
Q5  Degradation is visible: when the dense arm cannot run, hybrid answers from the sparse arm and the response
    carries a warning; a dense-only query that cannot run returns nothing and says why.
Q6  The service reads the live build on every request and reloads its frozen statistics when the build changes,
    so a rebuild needs no restart. It binds to 127.0.0.1 unless told otherwise and has no authentication.

Run:  python -u src\\arxiv_rag_api.py --serve [--port 8780]
      python -u src\\arxiv_rag_api.py --ask "how does GRPO remove the value function" [-k 3]
OpenWebUI: Settings -> Tools -> add an OpenAPI tool server at http://127.0.0.1:8780 (it reads /openapi.json).
"""
from __future__ import annotations

import argparse
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import sparsevec_store as ss
from arxiv_graph_service import frozen
from domain_corpora import SECTION_DISPLAY_CAP, arxiv_url, reaggregate_section
from ingest_arxiv_sparsevec import query_terms

LABEL = "arxiv_sect"
PORT = 8780
RRF_K, POOL_PER_K, MAX_K = 60, 10, 20
DEFAULT_K = 3
DEFAULT_MAX_CHARS, CEILING_MAX_CHARS = SECTION_DISPLAY_CAP, 20_000_000     # sections: p50 2.7k chars, p99.9 79k, one degenerate one 16.7M
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------------------------------------------- pure parts ----
def rrf(rank_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Q2. Guarantee: [(item, score)] by descending reciprocal-rank-fusion score, ties by item, over the ranked
    lists given (best first). An item in several lists outranks one in a single list at the same ranks."""
    score: dict[int, float] = defaultdict(float)
    for lst in rank_lists:
        for r, item in enumerate(lst):
            score[item] += 1.0 / (k + r + 1)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def _cut(text: str, max_chars: int | None) -> tuple[str, bool, int]:
    """Q3. Guarantee: (text cut to max_chars, whether it was cut, the full length)."""
    if max_chars and len(text) > max_chars:
        return text[:max_chars], True, len(text)
    return text, False, len(text)


# ------------------------------------------------------------------ retrieval ----
class QueryEmbedder:
    """The query side of the dense arm: MiniLM on the CPU (no VRAM contention with the extraction pipeline),
    loaded on first use. __call__(text) -> raw L2-normalised embedding, the space `hc.center` takes its mean from."""

    def __init__(self):
        self._model, self._lock = None, threading.Lock()

    def __call__(self, text: str) -> np.ndarray:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(MINILM, device="cpu")
                self._model.max_seq_length = 256
            return self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)


def sparse_ranked(conn, label: str, st: dict, query: str, pool: int) -> list[int]:
    """Q1. Guarantee: ords of up to `pool` inner-product matches, best first, each with a score above zero: the
    index returns `pool` rows however few contain a query term, and a chunk scoring 0 matched nothing. [] when no
    query term is in the vocabulary."""
    cols, weights = query_terms(query, st["col"], st["idf"])
    if not cols:
        return []
    return [o for o, ip in ss.query(conn, label, st["dim"], cols, pool, ef_search=max(pool, 100), weights=weights) if ip > 0]


def dense_ranked(conn, label: str, st: dict, query: str, pool: int, embed_query) -> list[int]:
    """Q1. Guarantee: ords of the `pool` nearest dense vectors by cosine, best first."""
    import hnsw_communities as hc
    q = hc.center(np.asarray(embed_query(query), np.float32)[None, :], st["mean"])[0]
    lit = "[" + ",".join("%.6g" % x for x in q) + "]"
    conn.execute("SET hnsw.ef_search = %d" % max(pool, 100))                      # V5: SET, never SET LOCAL
    rows = conn.execute(f"SELECT ord FROM {ss._table(label, 'dense')} ORDER BY emb <=> %s::vector LIMIT %s", (lit, pool)).fetchall()
    return [r[0] for r in rows]


def section_chunks(conn, label: str, doc_id: str, section_idx: int) -> tuple[list[tuple[int, int, str]], int]:
    """Q3. Guarantee: ([(ord, chunk_idx, text)] of one section in chunk order, how many junk chunks were left out).
    A junk chunk (C10) is extraction debris -- OCR letter soup, `, of-` repeats -- and is not part of what the
    section says; the count is returned so the omission is never silent."""
    rows = conn.execute("SELECT ord, chunk_idx, text, is_junk FROM lex_chunk_meta WHERE label = %s AND doc_id = %s "
                        "AND section_idx = %s ORDER BY chunk_idx", (label, doc_id, section_idx)).fetchall()
    return [(o, c, t) for o, c, t, junk in rows if not junk], sum(r[3] for r in rows)


def communities_of(conn, build: int, ords: list[int]) -> dict[int, dict]:
    """Guarantee: {ord: {cid, how, title, status}} for the ords the live build placed; title is a model-authored DRAFT."""
    rows = conn.execute("SELECT a.ord, a.cid, a.how, s.title, s.status FROM lex_assign a LEFT JOIN lex_summary s "
                        "ON s.build_id = a.build_id AND s.cid = a.cid WHERE a.build_id = %s AND a.ord = ANY(%s)",
                        (build, ords)).fetchall()
    return {o: {"cid": c, "how": how, "title": t, "status": s} for o, c, how, t, s in rows}


def search(conn, label: str, st: dict, query: str, k: int = DEFAULT_K, mode: str = "hybrid",
           include_references: bool = False, max_chars: int | None = DEFAULT_MAX_CHARS, embed_query=None) -> dict:
    """Q1-Q5. Guarantee: {query, mode, build, results, warnings}; at most k results, one per section, best first,
    each carrying its whole section (Q3) and the chunks that matched. Hits in the same section fold into one result."""
    pool = max(POOL_PER_K * k, 50)
    warnings, arms = [], {}
    if mode in ("hybrid", "sparse"):
        arms["sparse"] = sparse_ranked(conn, label, st, query, pool)
        if not arms["sparse"]:
            warnings.append("no query term is in the build's vocabulary: the sparse arm is empty")
    if mode in ("hybrid", "dense"):
        try:
            arms["dense"] = dense_ranked(conn, label, st, query, pool, embed_query)
        except Exception as exc:
            arms["dense"] = []
            warnings.append("dense arm unavailable (%s: %s)%s" % (type(exc).__name__, exc, "; answering from the sparse arm" if mode == "hybrid" else ""))
    fused = rrf(list(arms.values()))
    meta = {r[0]: r for r in conn.execute(
        "SELECT ord, doc_id, version, section_idx, chunk_idx, section_title, is_reference, is_junk FROM lex_chunk_meta "
        "WHERE label = %s AND ord = ANY(%s)", (label, [o for o, _ in fused])).fetchall()}
    results, where = [], {}
    for ord_, score in fused:
        m = meta.get(ord_)
        if m is None or m[7] or (m[6] and not include_references):                  # Q4
            continue
        key = (m[1], m[3])
        if key in where:
            results[where[key]]["matched_ords"].append(ord_)
            continue
        if len(results) == k:
            continue
        where[key] = len(results)
        results.append({"rank": len(results) + 1, "score": round(score, 6), "doc_id": m[1], "version": m[2], "url": arxiv_url(m[1]),
                        "section_idx": m[3], "section_title": m[5], "is_reference": m[6], "matched_ords": [ord_],
                        "matched_by": [a for a, lst in arms.items() if ord_ in lst]})
    comm = communities_of(conn, st["build"], [r["matched_ords"][0] for r in results])
    for r in results:
        chunks, omitted = section_chunks(conn, label, r["doc_id"], r["section_idx"])
        r["text"], r["truncated"], r["chars_total"] = _cut(reaggregate_section([(c, t) for _, c, t in chunks]), max_chars)
        r["junk_chunks_omitted"] = omitted
        r["community"] = comm.get(r["matched_ords"][0])
    return {"query": query, "mode": mode, "build": st["build"], "results": results, "warnings": warnings}


def get_section(conn, label: str, doc_id: str, section_idx: int, max_chars: int | None = DEFAULT_MAX_CHARS) -> dict | None:
    """Q3. Guarantee: the section as {doc_id, version, url, section_idx, section_title, text, truncated, chars_total, chunk_ords}, None if no chunk has that key."""
    chunks, omitted = section_chunks(conn, label, doc_id, section_idx)
    if not chunks:
        return None
    meta = conn.execute("SELECT version, section_title FROM lex_chunk_meta WHERE label = %s AND ord = %s", (label, chunks[0][0])).fetchone()
    text, cut, total = _cut(reaggregate_section([(c, t) for _, c, t in chunks]), max_chars)
    return {"doc_id": doc_id, "version": meta[0], "url": arxiv_url(doc_id), "section_idx": section_idx,
            "section_title": meta[1], "text": text, "truncated": cut, "chars_total": total,
            "chunk_ords": [o for o, _, _ in chunks], "junk_chunks_omitted": omitted}


# ------------------------------------------------------------------- the API ----
class Community(BaseModel):
    cid: int
    how: str = Field(description="'consensus' for a community derived at the last full build, 'nearest-centroid' for a chunk placed provisionally since")
    title: str | None = Field(None, description="a model-written DRAFT title, not a confirmed label")
    status: str | None = None


class Hit(BaseModel):
    rank: int
    score: float = Field(description="reciprocal-rank-fusion score; only comparable within one response")
    doc_id: str = Field(description="'arxiv/<arXiv id>' (version removed); '_methods' marks the methods extract")
    version: int
    url: str | None = Field(None, description="arXiv abstract page, null for a book")
    section_idx: int
    section_title: str
    is_reference: bool
    matched_by: list[str] = Field(description="which arms ranked the best-matching chunk: sparse, dense")
    matched_ords: list[int] = Field(description="chunk ids inside this section that matched")
    text: str = Field(description="the WHOLE section, rebuilt from its chunks")
    truncated: bool = Field(description="true when `text` was cut to max_chars; `chars_total` is the full length")
    chars_total: int
    junk_chunks_omitted: int = Field(0, description="OCR-debris chunks inside this section that were left out of `text`")
    community: Community | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    build: int
    results: list[Hit]
    warnings: list[str]


class SectionResponse(BaseModel):
    doc_id: str
    version: int
    url: str | None
    section_idx: int
    section_title: str
    text: str
    truncated: bool
    chars_total: int
    chunk_ords: list[int]
    junk_chunks_omitted: int = 0


class CommunityRow(BaseModel):
    cid: int
    size: int
    title: str | None
    summary: str | None
    status: str | None = Field(None, description="'draft': model-written, not confirmed")


def create_app(label: str = LABEL, dsn: str | None = None, embed_query=None) -> FastAPI:
    """Q6. Guarantee: a FastAPI app whose every request reads the live build; `embed_query` replaces the MiniLM query encoder (tests)."""
    app = FastAPI(title="arXiv paper library", version="1.0", description=(
        "Search the user's local library of arXiv papers. Each result is a WHOLE SECTION of a paper (not a snippet), found by "
        "BM25 and dense retrieval fused by rank. Use `search_arxiv` to find sections that answer a question, quote them with "
        "their `url` and `section_title`, and call `get_section` to re-read one. References are left out unless asked."))
    embed = embed_query or QueryEmbedder()
    state, lock = {"build": None, "st": None}, threading.Lock()

    def connect():
        return ss.connect(dsn) if dsn else ss.connect()

    def current(conn) -> dict:
        live = ss.live_build(conn, label)
        if live is None:
            raise HTTPException(503, "no live build for %r: run src/arxiv_graph_service.py --rebuild" % label)
        with lock:
            if state["build"] != live[0]:                                             # Q6: reload when a rebuild lands
                try:
                    state["st"], state["build"] = frozen(conn, label), live[0]
                except AssertionError as exc:
                    raise HTTPException(503, str(exc))
            return state["st"]

    @app.get("/search", operation_id="search_arxiv", response_model=SearchResponse,
             summary="Find whole sections of arXiv papers that answer a question")
    def search_ep(query: str = Query(..., min_length=1, description="a question or key phrases"),
                  k: int = Query(DEFAULT_K, ge=1, le=MAX_K, description="how many sections to return"),
                  mode: Literal["hybrid", "sparse", "dense"] = Query("hybrid", description="hybrid fuses BM25 and dense retrieval"),
                  include_references: bool = Query(False, description="also return bibliography sections"),
                  max_chars: int = Query(DEFAULT_MAX_CHARS, ge=500, le=CEILING_MAX_CHARS, description="longest text returned per section (default 200,000 returns nearly every section whole); a cut sets `truncated`")):
        conn = connect()
        try:
            return search(conn, label, current(conn), query, k, mode, include_references, max_chars, embed)
        finally:
            conn.close()

    @app.get("/section", operation_id="get_section", response_model=SectionResponse,
             summary="Read one whole section of a paper again")
    def section_ep(doc_id: str = Query(..., description="e.g. arxiv/2404_08634"), section_idx: int = Query(..., ge=0),
                   max_chars: int = Query(DEFAULT_MAX_CHARS, ge=500, le=CEILING_MAX_CHARS)):
        conn = connect()
        try:
            got = get_section(conn, label, doc_id, section_idx, max_chars)
        finally:
            conn.close()
        if got is None:
            raise HTTPException(404, "no section %s/%d" % (doc_id, section_idx))
        return got

    @app.get("/communities", operation_id="list_communities", response_model=list[CommunityRow],
             summary="List the topic communities of the library, largest first, with their draft titles")
    def communities_ep(limit: int = Query(30, ge=1, le=300)):
        conn = connect()
        try:
            st = current(conn)
            rows = conn.execute("SELECT c.cid, c.size, s.title, s.summary, s.status FROM lex_community c LEFT JOIN lex_summary s "
                                "ON s.build_id = c.build_id AND s.cid = c.cid WHERE c.build_id = %s ORDER BY c.size DESC LIMIT %s",
                                (st["build"], limit)).fetchall()
        finally:
            conn.close()
        return [dict(zip(("cid", "size", "title", "summary", "status"), r)) for r in rows]

    @app.get("/health", operation_id="health", summary="Which build is being served")
    def health_ep():
        conn = connect()
        try:
            st = current(conn)
            n = conn.execute("SELECT count(*), count(DISTINCT doc_id) FILTER (WHERE NOT is_reference AND NOT is_junk) "
                             "FROM lex_chunk_meta WHERE label = %s", (label,)).fetchone()
        finally:
            conn.close()
        return {"build": st["build"], "label": label, "chunks": n[0], "papers": n[1], "vocabulary": len(st["col"])}

    return app


# ------------------------------------------------------------------------ CLI ----
def say(text: str) -> None:
    """Guarantee: `text` is printed whatever the console's code page: a character it cannot encode (a Greek letter in
    a paper) is written as a \\u escape instead of killing the run."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(enc, "backslashreplace").decode(enc))


def ask(question: str, k: int, mode: str, cut: int | None = None) -> None:
    """Prints each hit's WHOLE section; `cut` shortens the display for skimming and says so."""
    conn = ss.connect()
    out = search(conn, LABEL, frozen(conn, LABEL), question, k, mode, False, DEFAULT_MAX_CHARS, QueryEmbedder())
    for w in out["warnings"]:
        say("WARNING: " + w)
    for r in out["results"]:
        c = r["community"] or {}
        say("\n#%d  %s  v%d  |  %s  |  %s  |  community %s %s" % (
            r["rank"], r["doc_id"], r["version"], r["section_title"], r["url"] or "-", c.get("cid", "-"), c.get("title") or ""))
        say("    matched by %s; %d chars%s%s" % ("+".join(r["matched_by"]), r["chars_total"], " (cut at the API limit)" if r["truncated"] else "",
                                               "; %d junk chunk(s) left out" % r["junk_chunks_omitted"] if r["junk_chunks_omitted"] else ""))
        text = r["text"] if not cut or len(r["text"]) <= cut else r["text"][:cut] + "\n[display cut at %d of %d chars: omit --cut for the whole section]" % (cut, len(r["text"]))
        say("    " + text.replace("\n", "\n    "))
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", action="store_true")
    g.add_argument("--ask", metavar="QUESTION")
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    ap.add_argument("--mode", choices=("hybrid", "sparse", "dense"), default="hybrid")
    ap.add_argument("--cut", type=int, default=None, help="--ask: show only the first N characters of each section")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    if args.ask:
        ask(args.ask, args.k, args.mode, args.cut)
        return
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
