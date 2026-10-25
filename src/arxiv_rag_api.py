"""| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| WARM | Validate matching graph/BM25 builds and warm the section encoder and cross-encoder before ready. | [mandatory] | Q7 |
| SEARCH | Read section vectors and BM25 chunk vectors from their owning databases using exact operators. | [mandatory] | Q10 |
| SPARSE | Build a BPE query from the live tokenizer and IDF; scan exact sparsevecs. | [mandatory] | Q1, Q10 |
| DENSE | Embed the query in the frozen section space; scan exact section vectors. | [optional] | Q1, Q10 |
| FUSE | Combine sparse and dense ranks with reciprocal rank fusion. | [mandatory] | Q2 |
| COMMUNITY REWEIGHT | For `/agent`, apply selected Dunning-term community affinity to fused candidates before top-k seeds. | [optional] | A3 |
| FILTER | Exclude references and junk unless requested. | [mandatory] | Q4 |
| EXPAND | Rebuild each hit from eligible BM25 chunks and preserve its whole-section response. | [mandatory] | Q3, Q10 |
| ANNOTATE | Add community, URL, and version metadata. | [mandatory] | Q3 |
| AGENT | `POST /agent` returns the agent answer separately from text-bearing evidence records. | [contract] | A1, A2 |
| AGGREGATE JOB | Submit, inspect, and resume bounded global semantic tallies. | [opt] | A5 |
| BOUND | Share a configurable deadline across internal model calls and retries. | [contract] | A4 |

Q1 A query is weighted with the live build's own vocabulary and idf. A term the build never saw carries no
    weight and is reported, not silently dropped; with no known term the sparse arm is empty and says so.
Q2 The arms' scores are not comparable (an inner product against a cosine), so they are fused by rank.
Q3 A section is rebuilt, not stored, by domain_corpora.reaggregate_section (C12): its chunks in chunk_idx order,
    the header the chunker prepended to each continuation chunk dropped, NO overlap trimming (none is stored;
    a trim would delete repeats the source contains). Junk chunks (OCR debris) are left out and counted in
    `junk_chunks_omitted`. A chunk is keyed by its FIRST section, so a small section merged into a chunk after it
    comes along with it. The whole section is returned: `max_chars` defaults to 200,000 (the 99.9th percentile
    of section size is 79,000) and a cut is NEVER silent: `truncated` is set and `chars_total` says what was left out.
Q4 Reference and junk chunks are stored and flagged (C9, C10); search leaves them out by default.
Q5 Degradation is visible: when the dense arm cannot run, hybrid answers from the sparse arm and the response
    carries a warning; a dense-only query that cannot run returns nothing and says why.
Q6 Each request pins the current graph and BM25 builds; an absent or mismatched pair fails before retrieval. The service
    binds to 127.0.0.1 unless told otherwise and has no authentication.
A1 `/agent` calls section_graphrag.agent_answer against the live xpa section map; the chosen Open WebUI model
     is passed through after removing its `openrouter/` connection prefix. The response omits internal vectors
     and includes bounded evidence passages and metadata needed by the Filter renderer.
A2 Every accepted evidence row carries its key, ordinal, role, passage text, and truncation flag; `sections`
     carries section metadata only. The final agent answer remains a separate `final` field. The relevance gate
     is not an entailment check, and rejected questions continue to return no accepted evidence.
A3 `/community-terms` exposes the live section build's stored Dunning term/G² records for the user dropdown. `/agent` accepts `selected_terms`; it leaves the original question intact, forwards terms to lexical/community ranking, and returns the selected terms plus adjusted seed scores. Unknown terms are ignored by the retriever and reported.
A4 `/agent` establishes one monotonic deadline (timeout_seconds, default 180, range 5–480) before resolving history. Timeouts return HTTP 504; upstream model errors and empty/unfinished replies return HTTP 502. They never become successful evidence reports.
Q7 API startup must verify a live build, load and validate the BPE chunk sparsevec build, vocabulary/IDF, and frozen tokenizer metadata, load the dense query embedder, and load the cross-encoder with a warm
    inference before it reports ready; text-only model imports must ignore unused torchvision/torchaudio packages;
    missing database or model prerequisites fail startup, not the first query.
Q8 `/agent` reads graph sections and community terms from `CHUNKGRAPH_DSN` and BM25 vectors/tokenizer/metadata from `ARXIV_BM25_DSN`; startup verifies complete section-key coverage before serving.
Q9 Startup migrates legacy section embeddings and BM25 vectors into their narrow tables before retrieval validation; the generation's column numbering is checked against the zero-based query representation.
Q10 `/search` preserves its whole-section response shape and sparse/dense/hybrid modes while using the exact split-store retrieval path; no active API route queries legacy HNSW vectors.
Q11 `/agent` accepts `graph_backend=sql|age`; SQL is the default, AGE requires a matching build graph, and backend identity is returned in the report.
Q12 `POST /aggregate` pins the live xpa source build, model/prompt identity, random seed, draw order, and fixed call budget before work begins.
Q13 `GET /aggregate/{job_id}` reports queued/running/incomplete/completed/failed state, call counts, result, and errors.
Q14 `POST /aggregate/{job_id}/resume` reuses committed responses and unchanged budgets; uncertain calls remain visible failures.
Q15 Aggregate work runs with a bounded worker pool and uses the existing tally SQLite database for durable job state.
Q16 `/agent` keeps retrieval as its default; `answer_kind=auto` routes only recognizable corpus-wide count questions to a bounded semantic tally job.
Q17 The optional aggregate job SQLite path configures both job state and semantic label checkpoints; default storage is the shared `.tmp/global_aggregate_tally.sqlite` file.

CLOSED: `/search` and `/agent` share the exact split-store retrieval path; `/agent` adds graph-agent iteration and selected-term controls.
Lazy first-request loading of local retrieval models is replaced by startup warmup. No Open WebUI chat bridge or
Pipelines server is required.

Preconditions: service startup requires both graph and BM25 database connections, the live `xpa` section build,
the section encoder and cross-encoder model assets. `/agent` additionally requires OpenRouter credentials. Failure modes: missing startup
prerequisites prevent readiness; request-time upstream model failures return HTTP 502 and model-budget exhaustion returns HTTP 504.

Run:  python -u src\\arxiv_rag_api.py --serve [--port 8780]
      python -u src\\arxiv_rag_api.py --ask "how does GRPO remove the value function" [-k 3]
OpenWebUI: Admin Settings -> Functions -> import src/owui_filter.py as a user-toggleable Filter Function.
The OpenWebUI host must reach this service at port 8780.
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
import hashlib
import logging
import random
import re
import sys
import threading
import time
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

import config
from aggregate_jobs import AggregateJobs
import sparsevec_store as ss
import section_store as store
from domain_corpora import SECTION_DISPLAY_CAP, arxiv_url, reaggregate_section
from ingest_arxiv_sparsevec import query_terms

LABEL = "arxiv_sect"
BM25_LABEL = "arxiv_sect_bpe"
PORT = 8780
LOG = logging.getLogger(__name__)
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
    loaded on first use. __call__(text) -> raw L2-normalised embedding, the space `hc.center` takes its mean from.

    Require: sentence-transformers and MiniLM assets are available.
    Guarantee: load the text encoder without importing unused vision/audio backends, then return its normalized vector.
    Maintain: one lazily loaded model protected by the instance lock.
    Assert: encoder output is the same MiniLM embedding regardless of installed torchvision/torchaudio packages."""

    def __init__(self):
        self._model, self._lock = None, threading.Lock()

    def __call__(self, text: str) -> np.ndarray:
        with self._lock:
            if self._model is None:
                for name in ("torchvision", "torchaudio"):
                    sys.modules[name] = None
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


def search_section_graph(conn, bm25_conn, query: str, k: int, mode: str,
                         include_references: bool, max_chars: int,
                         chunk_label: str = BM25_LABEL) -> dict:
    """Search the active section and BM25 stores without the legacy ANN path.

    Require:
        `conn` and `bm25_conn` address the active graph and BM25 databases, and the BM25 generation matches `xpa`.
    Guarantee:
        The existing `/search` response shape is returned from exact dense, sparse, or fused rankings.
    Maintain:
        Chunk ordinals are read only from BM25 metadata; section ordinals are read only from the active graph build.
    Assert:
        Every returned section has graph text and matching BM25 section metadata.

    Args:
        conn: Graph database connection.
        bm25_conn: BM25 database connection.
        query: User's search text.
        k: Maximum unique sections to return.
        mode: `hybrid`, `sparse`, or `dense` retrieval mode.
        include_references: Whether reference sections may be returned.
        max_chars: Maximum text characters returned per section.

    Returns:
        The documented search response dictionary.
    """
    import section_genre as genre
    import section_graphrag as graph

    graph_live = store.live_build(conn, "xpa")
    bm25_live = ss.live_build(bm25_conn, chunk_label)
    if graph_live is None or bm25_live is None:
        raise HTTPException(503, "active graph or BM25 build is missing")
    if bm25_live[2].get("source_graph_build") != graph_live[0]:
        raise HTTPException(503, "BM25 generation does not match the active graph build")
    sg = graph.SectionGraph(conn, "xpa", chunk_label=chunk_label, bm25_conn=bm25_conn)
    warnings: list[str] = []
    sparse_detail: dict[int, dict] = {}
    if mode == "hybrid":
        try:
            ranked = sg.ranked(query, include_references=include_references)
            arms = ranked["arms"]
            order = ranked["ranked"]
            sparse_detail = {row["ord"]: row for row in ranked["rank_details"]}
        except Exception as exc:
            sparse_order, sparse_detail = sg.reranked_chunk_sections(
                query, [], include_references=include_references
            )
            arms = {"sparse": sparse_order}
            order = [ord_ for ord_, _ in rrf([sparse_order])]
            warnings.append("dense arm unavailable (%s: %s); answering from sparse retrieval" % (type(exc).__name__, exc))
    elif mode == "sparse":
        sparse_order, sparse_detail = sg.reranked_chunk_sections(
            query, [], include_references=include_references
        )
        arms = {"sparse": sparse_order}
        order = [ord_ for ord_, _ in rrf([sparse_order])]
    else:
        try:
            q = sg.embed(query)
            if sg.V is None:
                qd, col = q, "emb"
            else:
                qd, col = genre.remove_axes(q[None, :], sg.V)[0], "emb_b"
            dense_order = store.seeds_dense(conn, sg.build, qd, max(POOL_PER_K * k, 50), col=col)
            arms = {"dense": dense_order}
            order = [ord_ for ord_, _ in rrf([dense_order])]
        except Exception as exc:
            if mode == "dense":
                return {"query": query, "mode": mode, "build": sg.build, "results": [],
                        "warnings": ["dense arm unavailable (%s: %s)" % (type(exc).__name__, exc)]}
            raise

    if mode in ("hybrid", "sparse") and not arms.get("sparse"):
        warnings.append("no query term in the live BPE vocabulary: the sparse arm is empty")
    if not order:
        return {"query": query, "mode": mode, "build": sg.build, "results": [], "warnings": warnings}

    section_rows = store.sections(conn, sg.build, order)
    keys = {(row["doc_id"], row["section_idx"]) for row in section_rows.values()}
    doc_ids = sorted({doc_id for doc_id, _ in keys})
    section_ids = sorted({section_idx for _, section_idx in keys})
    metadata_rows = bm25_conn.execute(
        "SELECT ord, doc_id, version, section_idx, chunk_idx, is_reference, is_junk, text "
        "FROM lex_chunk_meta WHERE label = %s AND build_id = %s "
        "AND doc_id = ANY(%s) AND section_idx = ANY(%s) ORDER BY ord",
        (sg.chunk_label, sg.chunk_build, doc_ids, section_ids),
    ).fetchall()
    by_key: dict[tuple[str, int], list[tuple[int, int, int, bool, bool, str]]] = defaultdict(list)
    for chunk_ord, doc_id, version, section_idx, chunk_idx, is_reference, is_junk, chunk_text in metadata_rows:
        key = (doc_id, section_idx)
        if key in keys:
            by_key[key].append((chunk_ord, version, chunk_idx, is_reference, is_junk, chunk_text))

    if keys - set(by_key):
        raise AssertionError("active graph search result is missing matching BM25 section metadata")
    score_by_ord = {ord_: 1.0 / (RRF_K + rank + 1) for rank, ord_ in enumerate(order)}
    if mode == "hybrid":
        score_by_ord = {row["ord"]: float(row["adjusted_score"]) for row in ranked["rank_details"]}
    community_ids = sorted({section_rows[ord_]["community"] for ord_ in order if ord_ in section_rows})
    summaries = store.summaries(conn, sg.build, community_ids)
    results = []
    for ord_ in order:
        section = section_rows.get(ord_)
        if section is None:
            continue
        key = (section["doc_id"], section["section_idx"])
        chunk_rows = by_key[key]
        is_reference = all(row[3] for row in chunk_rows)
        if (is_reference and not include_references) or all(row[4] for row in chunk_rows):
            continue
        matched_chunk = sparse_detail.get(ord_, {}).get("matched_chunk_ord")
        cid = section["community"]
        summary = summaries.get(cid)
        full_text = reaggregate_section([(row[2], row[5]) for row in chunk_rows if not row[4]])
        text, truncated, chars_total = _cut(full_text, max_chars)
        results.append({
            "rank": len(results) + 1,
            "score": round(score_by_ord[ord_], 6),
            "doc_id": section["doc_id"],
            "version": max(row[1] for row in chunk_rows),
            "url": arxiv_url(section["doc_id"]),
            "section_idx": section["section_idx"],
            "section_title": section["section_title"] or "",
            "is_reference": is_reference,
            "matched_ords": [matched_chunk] if matched_chunk is not None else [],
            "matched_by": [name for name, values in arms.items() if ord_ in values],
            "text": text,
            "truncated": truncated,
            "chars_total": chars_total,
            "junk_chunks_omitted": sum(row[4] for row in chunk_rows),
            "community": ({"cid": cid, "how": "consensus", "title": summary["title"],
                           "status": "draft"} if summary else None),
        })
        if len(results) == k:
            break
    return {"query": query, "mode": mode, "build": sg.build, "results": results, "warnings": warnings}


def get_section_from_graph(conn, bm25_conn, doc_id: str, section_idx: int,
                           max_chars: int, chunk_label: str = BM25_LABEL) -> dict | None:
    """Read a whole section from the active graph and its matching BM25 chunks.

    Require:
        Both database connections are open and the BM25 generation matches the live `xpa` graph.
    Guarantee:
        Return the section response assembled from non-junk chunk rows, or `None` when the key is absent.
    Maintain:
        Graph and chunk record IDs remain separate; `(doc_id, section_idx)` is the cross-store key.
    Assert:
        Every graph section key read has BM25 metadata in the pinned generation.

    Args:
        conn: Graph database connection.
        bm25_conn: BM25 database connection.
        doc_id: Stable paper key.
        section_idx: Stable section key within the paper.
        max_chars: Maximum response text length.

    Returns:
        The documented section response dictionary, or `None` when absent.
    """
    graph_live = store.live_build(conn, "xpa")
    bm25_live = ss.live_build(bm25_conn, chunk_label)
    if graph_live is None or bm25_live is None:
        raise RuntimeError("active graph or BM25 build is missing")
    build_id = graph_live[0]
    if bm25_live[2].get("source_graph_build") != build_id:
        raise RuntimeError("BM25 build does not match the active section graph")
    section = conn.execute(
        "SELECT section_title FROM sect_node WHERE build_id = %s AND doc_id = %s AND section_idx = %s",
        (build_id, doc_id, section_idx),
    ).fetchone()
    if section is None:
        return None
    rows = bm25_conn.execute(
        "SELECT ord, version, chunk_idx, is_junk, text FROM lex_chunk_meta "
        "WHERE label = %s AND build_id = %s AND doc_id = %s AND section_idx = %s ORDER BY chunk_idx",
        (chunk_label, bm25_live[0], doc_id, section_idx),
    ).fetchall()
    if not rows:
        raise AssertionError("active graph section is missing from BM25 metadata")
    usable = [row for row in rows if not row[3]]
    if not usable:
        return None
    full_text = reaggregate_section([(row[2], row[4]) for row in usable])
    text, truncated, chars_total = _cut(full_text, max_chars)
    return {
        "doc_id": doc_id,
        "version": max(row[1] for row in rows),
        "url": arxiv_url(doc_id),
        "section_idx": section_idx,
        "section_title": section[0] or "",
        "text": text,
        "truncated": truncated,
        "chars_total": chars_total,
        "chunk_ords": [row[0] for row in usable],
        "junk_chunks_omitted": len(rows) - len(usable),
    }


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


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    model: str | None = Field(None, description="Selected Open WebUI model id; `openrouter/` is stripped for the OpenRouter API.")
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)
    k: int = Field(default=3, ge=1, le=5)
    hops: int = Field(default=3, ge=0, le=5)
    queries: int = Field(default=3, ge=1, le=5)
    per_query: int = Field(default=18, ge=1, le=50)
    max_chars: int = Field(default=6000, ge=500, le=20000)
    selected_terms: list[str] = Field(default_factory=list, max_length=32, description="Community Dunning terms to add to lexical retrieval and use for post-fusion community reweighting.")
    timeout_seconds: float = Field(default=180.0, ge=5.0, le=480.0, description="Shared wall-time budget for history resolution and graph-agent model calls.")
    graph_backend: Literal["sql", "age"] = Field(default="sql", description="Section-neighbor traversal backend; AGE never falls back to SQL.")
    answer_kind: Literal["retrieval", "auto", "lexical_term", "semantic", "summary"] = Field(
        default="retrieval", description="Answer path; retrieval remains the compatibility default.")


class AggregateRequest(BaseModel):
    """Validated request contract for one bounded global semantic tally.

    Require: predicate describes a binary corpus property.
    Guarantee: budgets and source scope are bounded and explicit.
    Maintain: model defaults to the configured classifier when omitted.
    Assert: pilot budget is no larger than the total call budget.
    """

    predicate: str = Field(min_length=1, max_length=2000)
    model: str | None = Field(None, max_length=200)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    budget: int = Field(default=400, ge=1, le=2000)
    pilot_budget: int = Field(default=200, ge=1, le=2000)
    target_halfwidth: float = Field(default=0.02, gt=0, le=1)


def aggregate_population(conn, source_build_id: int) -> dict[str, int]:
    """Load one canonical paper and primary community per source paper.

    Require: source build exists in sect_node.
    Guarantee: modal section community with lowest-ID tie break matches global_aggregate grouping.
    Maintain: only stable IDs and assignments are loaded; source text stays out of the job manifest.
    Assert: each canonical paper has exactly one selected primary community.

    Args:
        conn: PostgreSQL connection to the graph corpus.
        source_build_id: Immutable section build identifier.
    Returns:
        Canonical paper ID to primary community mapping.
    Raises:
        RuntimeError: If the selected build has no canonical papers.
    """
    rows = conn.execute(
        "WITH counts AS (SELECT CASE WHEN right(doc_id,8)='_methods' THEN left(doc_id,length(doc_id)-8) "
        "ELSE doc_id END AS paper_id, community AS community_id, count(*) AS n FROM sect_node "
        "WHERE build_id=%s GROUP BY 1,2), ranked AS (SELECT paper_id, community_id, "
        "row_number() OVER (PARTITION BY paper_id ORDER BY n DESC, community_id ASC) AS rn FROM counts) "
        "SELECT paper_id, community_id FROM ranked WHERE rn=1 ORDER BY paper_id", (source_build_id,)
    ).fetchall()
    if not rows:
        raise RuntimeError(f"source build {source_build_id} contains no canonical papers")
    return {paper_id: community_id for paper_id, community_id in rows}


def infer_aggregate_predicate(question: str) -> str | None:
    """Recognize explicit global count questions and extract their binary predicate.

    Require: a natural-language question.
    Guarantee: returns a predicate only for explicit corpus count/prevalence wording.
    Maintain: ambiguous questions fall through to section retrieval.
    Assert: returned text excludes the count lead-in and is non-empty.

    Args:
        question: User's natural-language question.
    Returns:
        Predicate text, or None when count intent is not explicit.
    """
    match = re.search(
        r"\b(?:how many|what (?:fraction|proportion|percentage)|what percent(?:age)? of)\s+papers?\s+"
        r"(?:discuss|mention|study|studied|use|using|contain|include|are about|focus on|address)\s+(.+?)\s*[?!.]*$",
        question.strip(), re.IGNORECASE,
    )
    return match.group(1).strip() if match and match.group(1).strip() else None


def json_safe(value: Any) -> Any:
    """Require: nested graph result. Guarantee: NumPy and tuple values become JSON-safe; maintain scalar meaning."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value)]
    return value


def create_app(label: str = BM25_LABEL, dsn: str | None = None, *, warm_start: bool = True,
               bm25_dsn: str | None = None,
               aggregate_jobs_path: str | Path | None = None) -> FastAPI:
    """Require: database and local model assets for `label`; guarantee: warm search and agent routes over live builds.

    Maintain: `/search` retains its documented response contract and modes; `warm_start=False` is test-only.
    Assert: startup validates both stores and warms the section encoder and cross-encoder before serving.

    Args:
        label: Live BM25 generation label.
        dsn: Optional graph database connection string.
        warm_start: Whether startup validates and warms retrieval dependencies.
        bm25_dsn: Optional separate BM25 database connection string.
        aggregate_jobs_path: SQLite location shared by durable jobs and semantic labels.
    Returns:
        Configured FastAPI application.
    """
    state = {"build": None}
    jobs_path = Path(aggregate_jobs_path) if aggregate_jobs_path else ROOT / ".tmp" / "global_aggregate_tally.sqlite"
    aggregate_jobs = AggregateJobs(jobs_path, max_workers=2)

    def connect():
        return ss.connect(dsn) if dsn else ss.connect()

    def connect_bm25():
        return ss.connect(bm25_dsn or config.ARXIV_BM25_DSN)

    def warm_retrieval() -> None:
        LOG.info("Startup warmup: validating live %s build", label)
        conn = connect()
        bm25_conn = connect_bm25()
        try:
            ss.ensure_generation_schema(bm25_conn, label)
            import section_store as store
            store.ensure_schema(conn)
            import section_graphrag as graph
            section_graph = graph.SectionGraph(conn, "xpa", chunk_label=label, bm25_conn=bm25_conn)
            state["build"] = section_graph.build
            section_graph.embed("Open WebUI startup warmup")
        finally:
            conn.close()
            bm25_conn.close()
        warm_question = "Open WebUI startup warmup"
        LOG.info("Startup warmup: loading cross-encoder")
        graph.cross_scores(warm_question, ["Warmup evidence section."])
        LOG.info("Startup warmup complete for build %s", state["build"])

    @asynccontextmanager
    async def lifespan(_app):
        if warm_start:
            warm_retrieval()
        try:
            yield
        finally:
            aggregate_jobs.close(wait=True)

    app = FastAPI(
        title="arXiv paper library",
        version="1.1.0",
        description=(
            "Search the user's local library of arXiv papers. Each result is a WHOLE SECTION of a paper (not a snippet), found by "
            "BM25 and dense retrieval fused by rank. Use `search_arxiv` to find sections that answer a question, quote them with "
            "their `url` and `section_title`, and call `get_section` to re-read one. Use `answer_with_section_graph` for a graph-agent "
            "answer with trace, entities, gate scores, and citations. References are left out unless asked."
        ),
        lifespan=lifespan,
    )

    @app.get("/search", operation_id="search_arxiv", response_model=SearchResponse,
             summary="Find whole sections of arXiv papers that answer a question")
    def search_ep(query: str = Query(..., min_length=1, description="a question or key phrases"),
                  k: int = Query(DEFAULT_K, ge=1, le=MAX_K, description="how many sections to return"),
                  mode: Literal["hybrid", "sparse", "dense"] = Query("hybrid", description="hybrid fuses BM25 and dense retrieval"),
                  include_references: bool = Query(False, description="also return bibliography sections"),
                  max_chars: int = Query(DEFAULT_MAX_CHARS, ge=500, le=CEILING_MAX_CHARS, description="longest text returned per section (default 200,000 returns nearly every section whole); a cut sets `truncated`")):
        conn = connect()
        bm25_conn = connect_bm25()
        try:
            return search_section_graph(conn, bm25_conn, query, k, mode, include_references, max_chars, label)
        finally:
            conn.close()
            bm25_conn.close()

    def run_aggregate(job_id: str) -> dict[str, Any]:
        """Run a pinned semantic tally from the durable job manifest.

        Require: job was created with a source build, complete paper population, and frozen draw order.
        Guarantee: each classifier response passes through the durable job-call ledger.
        Maintain: model calls, source text, and database connections remain worker-local.
        Assert: the population digest still matches the submitted manifest.

        Args:
            job_id: Durable identifier of a queued or resumed job.
        Returns:
            JSON-safe tally result.
        Raises:
            KeyError: If the job disappeared before execution.
            RuntimeError: If its pinned population or draw order changed.
        """
        import global_aggregate as ga

        snapshot = aggregate_jobs.get(job_id)
        if snapshot is None:
            raise KeyError(job_id)
        request_data = snapshot.manifest["request"]
        source_build_id = int(snapshot.manifest["source_build_id"])
        conn = connect()
        try:
            population = aggregate_population(conn, source_build_id)
            digest = hashlib.sha256("\n".join(f"{pid}\t{population[pid]}" for pid in sorted(population)).encode()).hexdigest()
            if digest != snapshot.manifest["population_sha256"]:
                raise RuntimeError("pinned source population changed after job submission")
            expected_order = snapshot.manifest["draw_order"]
            for community_id in sorted(set(population.values())):
                paper_ids = sorted(pid for pid, cid in population.items() if cid == community_id)
                random.Random(f"{snapshot.manifest['seed']}:{community_id}").shuffle(paper_ids)
                if expected_order.get(str(community_id)) != paper_ids:
                    raise RuntimeError("pinned sample draw order no longer matches the source population")
            text_loader = lambda ids, cap: ga.load_paper_texts(conn, source_build_id, ids, cap)
            selected_model = (request_data.get("model") or ga.interpret.OPENROUTER_MODEL).removeprefix("openrouter/")
            config_obj = ga.SemanticTallyConfig(
                budget=int(snapshot.manifest["budget"]),
                sample_budget=int(snapshot.manifest["budget"]),
                pilot_budget=min(int(snapshot.manifest["pilot_budget"]), int(snapshot.manifest["budget"])),
                target_halfwidth=float(snapshot.manifest["target_halfwidth"]),
                seed=int(snapshot.manifest["seed"]),
                classifier_version=str(snapshot.manifest["classifier_version"]),
                cache_path=jobs_path,
            )
            provider_attempts = 0

            def durable_call(key: str, callback: Callable[[], Mapping]) -> Mapping:
                """Track whether this worker invokes the provider or reuses a committed response."""
                nonlocal provider_attempts

                def counted_callback():
                    nonlocal provider_attempts
                    provider_attempts += 1
                    return callback()

                return aggregate_jobs.call(job_id, key, counted_callback)

            result = ga.semantic_tally(
                population, str(request_data["predicate"]), text_loader,
                lambda predicate, text, truncated: ga.classify_predicate(
                    predicate, text, truncated, model=selected_model),
                config_obj, source_build_id,
                call_executor=durable_call,
            )
            result_data = asdict(result)
            result_data["provider_attempts_this_run"] = provider_attempts
            result_data["durable_responses_reused"] = result.calls - provider_attempts
            return result_data
        finally:
            conn.close()

    @app.post("/aggregate", status_code=202, operation_id="submit_global_aggregate",
              summary="Submit a resumable corpus-wide semantic tally")
    def submit_aggregate(request: AggregateRequest, response: Response):
        """Pin a source population and enqueue one bounded semantic count.

        Args:
            request: Predicate, model, seed, sample budget, and stopping target.
            response: FastAPI response used to return HTTP 202 and the job location.
        Returns:
            Submission receipt containing job ID and immutable manifest.
        Raises:
            HTTPException: If source build, strata count, or worker pool is unavailable.
        Require: The live xpa build is readable and budgets cover every community.
        Guarantee: Job inputs are persisted before the receipt is returned.
        Maintain: The submitted budget and sample order do not change on resume.
        Assert: Every submitted paper ID belongs to exactly one primary community.
        """
        import global_aggregate as ga

        if request.pilot_budget > request.budget:
            raise HTTPException(status_code=422, detail="pilot_budget cannot exceed budget")
        conn = connect()
        try:
            live = store.live_build(conn, "xpa")
            if live is None:
                raise HTTPException(status_code=503, detail="no live xpa source build")
            source_build_id = int(live[0])
            population = aggregate_population(conn, source_build_id)
        finally:
            conn.close()
        if request.budget < len(set(population.values())) or request.pilot_budget < len(set(population.values())):
            raise HTTPException(status_code=422, detail="budgets must cover every primary community")
        draw_order = {}
        for community_id in sorted(set(population.values())):
            paper_ids = sorted(pid for pid, cid in population.items() if cid == community_id)
            random.Random(f"{request.seed}:{community_id}").shuffle(paper_ids)
            draw_order[str(community_id)] = paper_ids
        population_digest = hashlib.sha256("\n".join(
            f"{pid}\t{population[pid]}" for pid in sorted(population)).encode()).hexdigest()
        classifier_model = (request.model or ga.interpret.OPENROUTER_MODEL).removeprefix("openrouter/")
        classifier_version = classifier_model + ":" + ga.PREDICATE_PROMPT_VERSION
        manifest = {
            "source_build_id": source_build_id,
            "population_sha256": population_digest,
            "population_size": len(population),
            "community_count": len(draw_order),
            "seed": request.seed,
            "draw_order": draw_order,
            "budget": request.budget,
            "pilot_budget": request.pilot_budget,
            "target_halfwidth": request.target_halfwidth,
            "classifier_version": classifier_version,
            "prompt_version": ga.PREDICATE_PROMPT_VERSION,
        }
        job_id = aggregate_jobs.create(request.model_dump(), manifest)
        if not aggregate_jobs.start(job_id, run_aggregate):
            raise HTTPException(status_code=503, detail="aggregate worker pool is unavailable")
        response.status_code = 202
        response.headers["Location"] = f"/aggregate/{job_id}"
        return {"job_id": job_id, "status": "queued", "manifest": manifest}

    @app.get("/aggregate/{job_id}", operation_id="get_global_aggregate")
    def get_aggregate(job_id: str):
        """Return durable progress, uncertainty, errors, and completed results.

        Args:
            job_id: Durable job identifier.
        Returns:
            Persisted job snapshot.
        Raises:
            HTTPException: If the job does not exist.
        Require: job_id is URL-decoded by FastAPI.
        Guarantee: Response includes persisted status and call counts.
        Maintain: Read operations do not mutate the job.
        Assert: Snapshot data is JSON serializable.
        """
        snapshot = aggregate_jobs.get(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="aggregate job not found")
        return asdict(snapshot)

    @app.post("/aggregate/{job_id}/resume", status_code=202, operation_id="resume_global_aggregate")
    def resume_aggregate(job_id: str):
        """Resume a pinned job without increasing its original call budget.

        Args:
            job_id: Durable job identifier.
        Returns:
            HTTP 202 receipt for accepted work.
        Raises:
            HTTPException: For absent, active, completed, or uncertain jobs.
        Require: Job has no unconfirmed provider call.
        Guarantee: Accepted work reuses the persisted manifest and completed responses.
        Maintain: The total call budget remains unchanged.
        Assert: A runner is scheduled only for incomplete or failed jobs.
        """
        snapshot = aggregate_jobs.get(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="aggregate job not found")
        if snapshot.calls_uncertain:
            raise HTTPException(status_code=409, detail="job has an unconfirmed provider call; reconcile before resume")
        if not aggregate_jobs.resume(job_id, run_aggregate):
            raise HTTPException(status_code=409, detail=f"job cannot resume from {snapshot.status}")
        return {"job_id": job_id, "status": "queued"}

    @app.post("/agent", operation_id="answer_with_section_graph",
              summary="Answer with the section-graph agent and return its trace and evidence")
    def agent_ep(request: AgentRequest, response: Response):
        """Require: validated question and live xpa graph. Guarantee: JSON-safe answer, trace, gate, and bounded evidence.

        Maintain: one request-local connection and only public report fields. Assert: close the connection on every exit.
        """
        import section_graphrag as graph
        import section_store as store

        if request.answer_kind in {"lexical_term", "summary"}:
            raise HTTPException(status_code=501,
                                detail=f"answer_kind={request.answer_kind} is not implemented; use auto, semantic, or retrieval")
        predicate = (infer_aggregate_predicate(request.question) if request.answer_kind == "auto" else
                     request.question if request.answer_kind == "semantic" else None)
        if predicate:
            receipt = submit_aggregate(AggregateRequest(
                predicate=predicate,
                model=request.model,
            ), response)
            return {"aggregate_job": receipt, "answer_kind": "semantic", "question": request.question}

        deadline = time.monotonic() + request.timeout_seconds
        conn = connect()
        bm25_conn = connect_bm25()
        try:
            sg = graph.SectionGraph(conn, "xpa", bm25_conn=bm25_conn, graph_backend=request.graph_backend)
            selected_model = request.model.removeprefix("openrouter/") if request.model else None
            question = request.question
            if request.history:
                import summarize_clusters as clusters
                history = "\n".join(
                    f"{item['role']}: {item['content']}" for item in request.history
                    if item.get("role") in {"user", "assistant"} and item.get("content")
                )
                rewritten = clusters.chat(
                    selected_model or clusters.MODEL,
                    "Rewrite the latest user request as one standalone research question using the earlier turns only to resolve references. Return only that question.\n\n"
                    f"Earlier turns:\n{history}\n\nLatest user request:\n{request.question}", deadline=deadline,
                )["content"].strip()
                if rewritten:
                    question = rewritten
            result = graph.agent_answer(
                sg,
                question,
                k=request.k,
                model=selected_model,
                agent_model=selected_model,
                hops=request.hops,
                queries=request.queries,
                per_query=request.per_query,
                selected_terms=request.selected_terms,
                deadline=deadline,
            )
            included = {
                "question", "seeds", "subgraph", "partials", "partials_rejected",
                "global_answer", "final", "evidence", "traversal", "section_scores",
                "section_threshold", "question_rejected", "agent", "bm25_coverage",
                "selected_terms", "ignored_terms", "seed_scores",
            }
            report = {key: value for key, value in result.items() if key in included}
            report["original_question"] = request.question
            report["question"] = question
            report["partials_rejected"] = [
                {key: item[key] for key in ("community", "verdict") if key in item}
                for item in result.get("partials_rejected", [])
                if isinstance(item, dict)
            ]
            report["agent"] = {
                key: value for key, value in result.get("agent", {}).items()
                if key in {"nodes", "trace", "stop"}
            }

            ords = set(result.get("seeds", []))
            ords.update(row["ord"] for row in result.get("subgraph", []))
            ords.update(row["ord"] for row in result.get("evidence", []))
            ords.update(node["ord"] for node in result.get("agent", {}).get("nodes", {}).values())
            sections = store.sections(conn, sg.build, sorted(ords)) if ords else {}
            doc_ids = sorted({section["doc_id"] for section in sections.values()})
            titles = store.paper_titles(conn, doc_ids)
            report["sections"] = {
                str(ord_): {
                    "doc_id": section["doc_id"],
                    "section_idx": section["section_idx"],
                    "section_title": section["section_title"],
                    "community": section["community"],
                    "paper_title": titles.get(section["doc_id"], section["doc_id"]),
                    "url": arxiv_url(section["doc_id"]),
                }
                for ord_, section in sections.items()
            }
            report["evidence"] = [
                {
                    **row,
                    "text": sections[row["ord"]]["text"][:request.max_chars],
                    "truncated": len(sections[row["ord"]]["text"]) > request.max_chars,
                }
                for row in result.get("evidence", [])
            ]
            community_ids = {cid for cid, _ in result.get("partials", [])}
            community_ids.update(
                item["community"] for item in result.get("partials_rejected", [])
                if isinstance(item, dict) and item.get("community") is not None
            )
            community_ids.update(
                item["cid"] for item in (result.get("traversal") or {}).get("by_community", [])
            )
            community_ids.update(
                row["community"] for row in result.get("section_scores", [])
                if row.get("community") is not None
            )
            summaries = store.summaries(conn, sg.build, sorted(community_ids)) if community_ids else {}
            report["communities"] = summaries
            report["build"] = sg.build
            report["graph_backend"] = request.graph_backend
            return json_safe(report)
        except TimeoutError as exc:
            LOG.warning("Graph agent timed out: %s", exc)
            raise HTTPException(504, f"Graph agent exceeded its {request.timeout_seconds:g}-second request budget: {exc}") from exc
        except RuntimeError as exc:
            LOG.warning("Graph agent backend unavailable: %s", exc)
            raise HTTPException(503, f"Graph retrieval backend unavailable: {exc}") from exc
        except urllib.error.HTTPError as exc:
            LOG.warning("Graph agent upstream HTTP error: %s", exc.code)
            raise HTTPException(502, f"OpenRouter model request failed with HTTP {exc.code}: {exc.reason}") from exc
        except (ValueError, urllib.error.URLError) as exc:
            LOG.warning("Graph agent model response failed: %s", exc)
            raise HTTPException(502, f"Graph agent model request failed: {exc}") from exc
        finally:
            conn.close()
            bm25_conn.close()

    @app.get("/section", operation_id="get_section", response_model=SectionResponse,
             summary="Read one whole section of a paper again")
    def section_ep(doc_id: str = Query(..., description="e.g. arxiv/2404_08634"), section_idx: int = Query(..., ge=0),
                   max_chars: int = Query(DEFAULT_MAX_CHARS, ge=500, le=CEILING_MAX_CHARS)):
        conn = connect()
        bm25_conn = connect_bm25()
        try:
            got = get_section_from_graph(conn, bm25_conn, doc_id, section_idx, max_chars, label)
        finally:
            conn.close()
            bm25_conn.close()
        if got is None:
            raise HTTPException(404, "no section %s/%d" % (doc_id, section_idx))
        return got

    @app.get("/community-terms", operation_id="list_community_terms",
             summary="List the live section build's Dunning terms and their community scores")
    def community_terms_ep():
        conn = connect()
        try:
            live = store.live_build(conn, "xpa")
            if live is None:
                raise HTTPException(503, "no live xpa section graph build")
            build_id, _ = live
            terms_by_community = store.community_dunning_terms(conn, build_id)
            cids = sorted(terms_by_community)
            summaries = store.summaries(conn, build_id, cids)
        finally:
            conn.close()
        by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cid, records in terms_by_community.items():
            summary = summaries.get(cid) or {}
            for record in records:
                by_term[record["term"]].append({
                    "cid": cid,
                    "title": summary.get("title") or f"Community {cid}",
                    "g2": record["g2"],
                })
        result = []
        for term, communities in by_term.items():
            communities.sort(key=lambda row: (-row["g2"], row["cid"]))
            labels = "; ".join(row["title"] for row in communities[:3])
            result.append({"term": term, "label": f"{term} — {labels}", "communities": communities})
        result.sort(key=lambda row: (row["term"].casefold()))
        return {"build": build_id, "terms": result}

    @app.get("/communities", operation_id="list_communities", response_model=list[CommunityRow],
             summary="List the topic communities of the library, largest first, with their draft titles")
    def communities_ep(limit: int = Query(30, ge=1, le=300)):
        conn = connect()
        try:
            live = store.live_build(conn, "xpa")
            if live is None:
                raise HTTPException(503, "no live xpa section graph build")
            rows = conn.execute(
                "SELECT cid, size, title, summary, status FROM sect_community "
                "WHERE build_id = %s ORDER BY size DESC, cid LIMIT %s",
                (live[0], limit),
            ).fetchall()
        finally:
            conn.close()
        return [dict(zip(("cid", "size", "title", "summary", "status"), r)) for r in rows]

    @app.get("/health", operation_id="health", summary="Which build is being served")
    def health_ep():
        conn = connect()
        bm25_conn = connect_bm25()
        try:
            import section_store as store
            graph_live = store.live_build(conn, "xpa")
            bm25_live = ss.live_build(bm25_conn, label)
            bm25_n = bm25_conn.execute("SELECT count(*), count(DISTINCT doc_id) FILTER (WHERE NOT is_reference AND NOT is_junk) "
                                        "FROM lex_chunk_meta WHERE label = %s AND build_id = %s",
                                        (label, bm25_live[0] if bm25_live else -1)).fetchone()
        finally:
            conn.close()
            bm25_conn.close()
        if graph_live is None or bm25_live is None:
            raise HTTPException(503, "active graph or BM25 build is missing")
        return {"build": bm25_live[0], "graph_build": graph_live[0], "bm25_build": bm25_live[0], "label": label,
                "chunks": bm25_n[0], "papers": bm25_n[1], "vocabulary": bm25_live[2].get("bm25", {}).get("dim"),
                "source_graph_build": bm25_live[2].get("source_graph_build")}

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
    bm25_conn = ss.connect(config.ARXIV_BM25_DSN)
    try:
        out = search_section_graph(conn, bm25_conn, question, k, mode, False, DEFAULT_MAX_CHARS)
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
    finally:
        conn.close()
        bm25_conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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
