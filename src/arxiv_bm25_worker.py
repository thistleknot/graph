"""Watch the live section build and rebuild its BM25 companion index when it changes.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| CHECK | Compare the live graph build ID with the BM25 build's recorded source ID. | [mandatory] | R1 |
| BUILD | Validate vector/metadata coverage and run the live-source BM25 ingester when stale. | [mandatory] | R2-R3 |
| WATCH | Repeat the check at a bounded interval under a scheduled task. | [opt] | R4 |

R1 A missing graph or BM25 endpoint is an error; an existing BM25 build is current only when its source build matches `xpa` and its scoped metadata, vocabulary, and vectors are complete.
R2 Sparse vectors are counted by joining their record_id to metadata for the requested BM25 build.
R3 The worker delegates actual construction to `ingest_arxiv_sparsevec.py --from-live-graph`; it never reads the legacy cache.
R4 Polling defaults to 60 seconds. Each check either exits or sleeps, so the process does not spin while idle.

CLOSED: NSSM service hosting is not used; Task Scheduler launches the same persistent polling worker at user logon.

Preconditions: both database DSNs are configured and the graph build is complete before its ID changes. Failure modes: a failed index build is logged and retried on the next poll.

Run: `python -u src\\arxiv_bm25_worker.py --watch`; register the user-logon task with `scripts/register_arxiv_bm25_task.ps1`.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config
import sparsevec_store as ss
import section_store

LOG = logging.getLogger("arxiv_bm25_worker")
LABEL = "arxiv_sect_bpe"


def index_needed(graph_conn, bm25_conn, tag: str = "xpa", label: str = LABEL) -> tuple[bool, int]:
    """Return whether the lexical index source build differs from the graph's live build.

    Args:
        graph_conn: Connection to the section graph database.
        bm25_conn: Connection to the BM25 index database.
        tag: Live graph build tag.
        label: BM25 build label.

    Returns:
        A pair containing the rebuild decision and current graph build ID.

    Raises:
        RuntimeError: If no graph build is available to index.
    """
    graph_live = section_store.live_build(graph_conn, tag)
    if graph_live is None:
        raise RuntimeError("no live section graph build for %r" % tag)
    bm25_live = ss.live_build(bm25_conn, label)
    source_build = bm25_live[2].get("source_graph_build") if bm25_live else None
    complete = False
    if bm25_live:
        build_id, chunk_count, params = bm25_live
        dim = int(params.get("bm25", {}).get("dim", 0))
        table = ss._table(label, "sparse")
        counts = bm25_conn.execute(
            f"SELECT (SELECT count(*) FROM lex_chunk_meta WHERE label=%s AND build_id=%s), "
            f"(SELECT count(*) FROM {table} v JOIN lex_chunk_meta m ON m.record_id=v.record_id WHERE m.label=%s AND m.build_id=%s), "
            "(SELECT count(*) FROM lex_build_vocab WHERE build_id=%s)",
            (label, build_id, label, build_id, build_id),
        ).fetchone()
        complete = counts == (chunk_count, chunk_count, dim)
    return source_build != graph_live[0] or not complete, graph_live[0]


def rebuild(graph_build: int, python: str = sys.executable) -> None:
    """Run a full BM25 build and assert it recorded the requested source build.

    Args:
        graph_build: Active graph build ID observed by the caller.
        python: Interpreter that owns the graph/BM25 dependencies.

    Raises:
        RuntimeError: If the builder exits unsuccessfully or publishes another source build.
    """
    cmd = [python, "-u", str(ROOT / "src" / "ingest_arxiv_sparsevec.py"), "--from-live-graph",
           "--label", LABEL, "--bpe", "10000", "--no-battery"]
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError("BM25 ingest failed with exit %d" % result.returncode)
    with ss.connect(config.ARXIV_BM25_DSN) as bm25_conn:
        live = ss.live_build(bm25_conn, LABEL)
    if live is None or live[2].get("source_graph_build") != graph_build:
        raise RuntimeError("BM25 build did not publish source graph build %d" % graph_build)


def check_once() -> bool:
    """Check the database pair and build BM25 when the graph build is not indexed.

    Returns:
        True when a build ran, otherwise False.
    """
    with ss.connect(config.DSN) as graph_conn, ss.connect(config.ARXIV_BM25_DSN) as bm25_conn:
        ss.ensure_generation_schema(bm25_conn, LABEL)
        needed, graph_build = index_needed(graph_conn, bm25_conn)
    if not needed:
        LOG.info("BM25 index already matches graph build %d", graph_build)
        return False
    LOG.info("BM25 index is stale; building source graph ID %d", graph_build)
    rebuild(graph_build)
    LOG.info("BM25 index now matches graph build %d", graph_build)
    return True


def main() -> None:
    """Run one synchronization check or poll until stopped."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="check and build once")
    mode.add_argument("--watch", action="store_true", help="poll until stopped")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if args.once:
        check_once()
        return
    while True:
        try:
            check_once()
        except Exception:
            LOG.exception("BM25 synchronization failed; retrying in %d seconds", args.poll_seconds)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
