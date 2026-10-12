"""Batched community-summary relevance scores from a local Jev-Style service.

One request carries one query and a batch of community summaries as shared state,
plus one independent NOUL question per summary. A single consumer keeps only one
GPU request active; a bounded CPU queue holds at most one prepared successor.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request


DEFAULT_BATCH_SIZE = 1                  # measured 2026-10-12: a score depends on the other summaries in its shared state; only a solo request is a pair score
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 128
PAGE_SIZE = 64                          # summaries read from Postgres at a time; independent of how many share one scoring request
MAX_ESTIMATED_INPUT_CHARS = 65000       # conservative guard below Jev-Style's 25.6k-token input limit
QUEUE_DEPTH = 1                         # one active request + one prefetched CPU batch = two in flight
DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_THRESHOLD = 0.5
REQUEST_TIMEOUT = 600
_END = object()


def _batch_size() -> int:
    raw = int(os.environ.get("COMMUNITY_JEV_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    if not MIN_BATCH_SIZE <= raw <= MAX_BATCH_SIZE:
        raise ValueError("COMMUNITY_JEV_BATCH_SIZE must be between %d and %d" % (MIN_BATCH_SIZE, MAX_BATCH_SIZE))
    return raw


def configured_batch_size() -> int:
    """Summaries per scoring request (COMMUNITY_JEV_BATCH_SIZE, default 1).

    Measured on Jev-Style 0.8B v3, query "core speculative-decoding methods", batch 5 of the candidate list (64 communities): the three materials-science
    communities 755/754/745 scored 0.025/0.028/0.029 alone, 0.06/0.04/0.02 at 8 per request, 0.20/0.05/0.04 at 16, 0.20/0.23/0.65 at 32 and
    0.86/0.80/0.74 at 64, while the speculative-decoding community c94 scored 0.769 alone and 0.246 inside its natural batch of 64. Anything above 1 makes the score
    candidate-relative, so a size above 1 is an explicit, auditable choice."""
    return _batch_size()


def relevance_threshold() -> float:
    value = float(os.environ.get("COMMUNITY_JEV_THRESHOLD", DEFAULT_THRESHOLD))
    if not 0.0 <= value <= 1.0:
        raise ValueError("COMMUNITY_JEV_THRESHOLD must be between 0 and 1")
    return value


def _endpoint() -> str:
    base = os.environ.get("JEV_STYLE_URL", DEFAULT_URL).rstrip("/")
    if base.endswith("/v1/systemone"):
        return base
    if base.endswith("/v1"):
        return base + "/systemone"
    return base + "/v1/systemone"


def _payload(question: str, rows: list[dict]) -> bytes:
    candidates = []
    questions = {}
    for i, row in enumerate(rows):
        key = "r%d" % i
        label = "c%d" % int(row["community"])
        candidates.append({"id": label, "title": row["title"], "summary": row["summary"]})
        # Keep each question short: JEV shares the query/state across questions,
        # but the question/readout budget is separate from the state budget.
        questions[key] = {"type": "noul", "instructions": "Is summary %s relevant to answering the query?" % label}
    state = {"query": question, "community_summaries": candidates}
    return json.dumps({"state": state, "questions": questions}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _batches(question: str, rows: list[dict], size: int):
    current = []
    current_chars = 0
    for row in rows:
        # Count the actual text plus a small allowance for JSON keys, ids and
        # per-summary NOUL question. Oversized individual summaries are never
        # silently truncated; the service will report its input-limit error.
        row_chars = len(row["title"]) + len(row["summary"]) + 160
        if current and (len(current) >= size or current_chars + row_chars > MAX_ESTIMATED_INPUT_CHARS):
            yield current, _payload(question, current)
            current, current_chars = [], 0
        current.append(row)
        current_chars += row_chars
    if current:
        yield current, _payload(question, current)


def _post(body: bytes) -> tuple[dict, float]:
    req = urllib.request.Request(_endpoint(), data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
            if result.get("error"):
                raise RuntimeError("Jev-Style API error: %s" % result["error"])
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("Jev-Style API error %d: %s" % (exc.code, detail)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Jev-Style scoring service is unavailable at %s. Start it with "
            "`jev-style serve --backend torch --device cuda` (or set JEV_STYLE_URL). "
            "The Ollama chat endpoint does not expose Jev-Style's typed NOUL scores." % _endpoint()
        ) from exc
    return result, time.perf_counter() - started


def _score_one_batch(question: str, rows: list[dict], body: bytes) -> tuple[list[dict], dict]:
    try:
        result, elapsed = _post(body)
    except RuntimeError as exc:
        # A server-side context or memory error may be resolved by splitting a
        # large batch. Other connectivity/configuration failures should surface.
        msg = str(exc).lower()
        if len(rows) > 1 and any(term in msg for term in ("token", "context", "memory", "cuda", "oom")):
            middle = len(rows) // 2
            left, right = rows[:middle], rows[middle:]
            a, sa = _score_one_batch(question, left, _payload(question, left))
            b, sb = _score_one_batch(question, right, _payload(question, right))
            return a + b, {"batches": sa["batches"] + sb["batches"], "elapsed_s": sa["elapsed_s"] + sb["elapsed_s"]}
        raise

    answers = result.get("answers") or {}
    scores = []
    for i, row in enumerate(rows):
        key = "r%d" % i
        answer = answers.get(key) or {}
        if answer.get("type") != "noul" or "noul" not in answer:
            raise RuntimeError("Jev-Style response omitted NOUL score %r (community %s)" % (key, row["community"]))
        score = float(answer["noul"])
        if not 0.0 <= score <= 1.0:
            raise RuntimeError("Jev-Style returned out-of-range relevance score %r for community %s" % (score, row["community"]))
        scores.append({"community": int(row["community"]), "score": score})
    usage = result.get("usage") or {}
    stats = {"candidates": len(rows), "input_tokens": usage.get("input_tokens"), "elapsed_s": round(elapsed, 3)}
    return scores, {"batches": [stats], "elapsed_s": elapsed}


def score_communities(question: str, summary_pages) -> tuple[list[dict], list[dict]]:
    """Score every supplied summary, returning all scores and per-request stats.

    The producer serializes at most one next batch in CPU memory while the main
    thread submits the current batch. Thus the in-flight bound is two batches:
    one at the service/GPU and one prepared in the bounded queue. The server sees
    one request at a time, so no overlapping GPU inference is introduced.
    """
    size = _batch_size()
    pending: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    stop = threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                pending.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def prepare() -> None:
        try:
            pages = [summary_pages] if isinstance(summary_pages, dict) else summary_pages
            for page in pages:
                rows = [{"community": int(cid), "title": str(rec.get("title") or ""), "summary": str(rec.get("summary") or "")}
                        for cid, rec in page.items()]
                for batch_rows, body in _batches(question, rows, size):
                    if not put((batch_rows, body)):
                        return
        except BaseException as exc:
            put(exc)
        finally:
            put(_END)

    producer = threading.Thread(target=prepare, name="community-jev-prefetch", daemon=True)
    producer.start()
    scores, batch_stats = [], []
    try:
        while True:
            item = pending.get()
            if item is _END:
                break
            if isinstance(item, BaseException):
                raise item
            batch_rows, body = item
            batch_scores, stats = _score_one_batch(question, batch_rows, body)
            scores.extend(batch_scores)
            batch_stats.extend(stats["batches"])
    finally:
        stop.set()
        while producer.is_alive():
            try:
                pending.get(timeout=0.1)
            except queue.Empty:
                pass
        producer.join()
    return scores, batch_stats
