"""
label_communities.py — draft human-readable names for a run's Louvain communities.

THE DETERMINISM BOUNDARY
Membership is computed; labels are text. Louvain assigned every `cid` at ingest
with a fixed seed and no model in the loop, and nothing here changes that. The
model only *names* an already-frozen cluster, which is the one thing steering
permits it to do:

    "Louvain membership is never LLM-assigned. Partition is deterministic;
     models may only name an already-frozen cluster."

Every label produced here is therefore a DRAFT: it is written to a JSON artifact,
never used as a join key, and always carries the model that authored it so a
confirmed label can never be mistaken for a generated one.

INPUT   the run's stored cid, size, tf*idf keywords, and medoid text
OUTPUT  community_labels.json  — {cid, label, gloss, model, drafted_at, ...}

CONCURRENCY
At most MAX_WORKERS in flight; a local GPU serialises them anyway and the
operator may be contending for it. Each call retries with exponential backoff
(BASE_TIMEOUT * GROWTH**attempt), because a timeout here means "the box is busy",
not "the request is malformed".

GUARDS (EARS)
L1  The system SHALL NOT modify community membership, only produce label text.
L2  Every label SHALL record the model that authored it and the time, so drafts
    stay distinguishable from human-confirmed labels.
L3  WHEN a request times out, the system SHALL retry with a longer timeout rather
    than dropping the community.
L4  IF a community cannot be labelled after MAX_ATTEMPTS, THEN it SHALL be
    recorded with label=None and its error, never silently omitted.
L5  At most MAX_WORKERS requests SHALL be in flight at once.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import httpx
import ollama

import graph_tools as gt

MODEL = os.environ.get("LABEL_MODEL", "qwen3.5-oc:4b")
LABEL_OUT = os.environ.get("LABEL_OUT", "community_labels.json")

# Ollama serialises requests per model unless OLLAMA_NUM_PARALLEL says otherwise,
# so "more workers" does NOT mean more throughput — it means each request waits
# behind the others and is far likelier to hit its own timeout. A first run at 4
# workers timed out on every single community without labeling one. 2 is the
# honest default; raise it only if OLLAMA_NUM_PARALLEL is set.
MAX_WORKERS = int(os.environ.get("LABEL_WORKERS", "2"))       # L5
BASE_TIMEOUT = float(os.environ.get("LABEL_TIMEOUT", "180"))  # seconds
GROWTH = 1.7              # L3 exponential backoff
MAX_ATTEMPTS = 5

_print_lock = threading.Lock()


def _host() -> str:
    h = os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
    h = h.replace("0.0.0.0", "127.0.0.1")
    return h if "://" in h else f"http://{h}"


PROMPT = """These text chunks were grouped together by a clustering algorithm. \
Your job is to name the group.

Distinctive terms (highest tf*idf): {keywords}

Most central chunk in the group:
\"\"\"{medoid}\"\"\"

Group size: {size} chunks.

Reply with exactly two lines and nothing else:
LABEL: <2-5 words naming what this group is about>
GLOSS: <one sentence, max 20 words, describing the group>"""


# Thinking models (qwen3.5-*, and others) put their trace in `thinking` and
# leave `content` EMPTY until the trace finishes. Measured on qwen3.5-oc:4b:
# num_predict=120 -> done_reason=length, content=''; num_predict=800 -> STILL
# length, content='' after 62s. Raising the budget does not fix it. Naming a
# cluster needs no reasoning trace, so we turn thinking off: same model answered
# in 3.6s with content populated. _THINK_OFF flips to False if a model rejects
# the flag, so non-thinking models still work.
_THINK_OFF = True


def _chat(client, prompt: str, timeout: float):
    """One completion, thinking disabled where supported."""
    global _THINK_OFF
    kw = {"think": False} if _THINK_OFF else {}
    client._client.timeout = timeout
    try:
        resp = client.chat(model=MODEL,
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.2, "num_predict": 200},
                           **kw)
    except (ollama.ResponseError, TypeError) as e:
        if not _THINK_OFF or "think" not in str(e).lower():
            raise
        _THINK_OFF = False                      # model has no thinking mode
        return _chat(client, prompt, timeout)
    msg = resp.message
    # Fall back to the trace if content is empty but the model reasoned aloud.
    return msg.content or (getattr(msg, "thinking", None) or "")


def _parse(text: str):
    """Pull LABEL/GLOSS out of the reply. Models add preamble; be forgiving
    about that but strict about what counts as a label."""
    label = gloss = None
    for line in (text or "").splitlines():
        line = line.strip().lstrip("*# ").strip()
        m = re.match(r"^LABEL\s*[:\-]\s*(.+)$", line, re.I)
        if m and not label:
            label = m.group(1).strip().strip('"').strip("*")
        m = re.match(r"^GLOSS\s*[:\-]\s*(.+)$", line, re.I)
        if m and not gloss:
            gloss = m.group(1).strip().strip('"').strip("*")
    return label, gloss


def label_one(client, c: dict) -> dict:
    """L3/L4: retry with growing timeout; record failure rather than dropping."""
    prompt = PROMPT.format(
        keywords=", ".join(c["keywords"][:8]),
        medoid=(c["medoid_text"] or "")[:900],
        size=c["size"])

    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        timeout = BASE_TIMEOUT * (GROWTH ** attempt)
        try:
            t0 = time.time()
            text = _chat(client, prompt, timeout)
            label, gloss = _parse(text)
            if not label:                              # unusable reply -> retry
                last_err = f"no LABEL line in reply: {text[:120]!r}"
                continue
            with _print_lock:
                print(f"  c{c['cid']:<3} [{time.time()-t0:5.1f}s] {label}",
                      flush=True)
            return {"cid": c["cid"], "size": c["size"],
                    "keywords": list(c["keywords"]),
                    "label": label, "gloss": gloss,
                    "status": "draft",                 # L2 — never 'confirmed'
                    "model": MODEL,
                    "drafted_at": datetime.now(timezone.utc).isoformat(),
                    "attempts": attempt + 1}
        except (httpx.TimeoutException, httpx.HTTPError, ollama.ResponseError) as e:
            last_err = f"{type(e).__name__}: {e}"
            with _print_lock:
                print(f"  c{c['cid']:<3} attempt {attempt+1} failed "
                      f"({timeout:.0f}s): {type(e).__name__} — backing off",
                      flush=True)
            time.sleep(min(2 ** attempt, 20))

    with _print_lock:
        print(f"  c{c['cid']:<3} GAVE UP: {last_err}", flush=True)
    return {"cid": c["cid"], "size": c["size"], "keywords": list(c["keywords"]),
            "label": None, "gloss": None, "status": "failed",
            "model": MODEL, "error": last_err,
            "drafted_at": datetime.now(timezone.utc).isoformat(),
            "attempts": MAX_ATTEMPTS}


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "brown-50"
    conn = gt.connect()                                # L1: read-only
    run = gt.get_run(conn, label)

    with conn.cursor() as cur:
        cur.execute("""SELECT cid, size, keywords, medoid, medoid_text
                         FROM community WHERE run_id = %s ORDER BY size DESC""",
                    (run.run_id,))
        comms = cur.fetchall()

    print(f"run    : {run.label}  ({run.run_id})")
    print(f"model  : {MODEL}   workers: {MAX_WORKERS}   "
          f"timeout: {BASE_TIMEOUT:.0f}s x{GROWTH}^n")
    print(f"labeling {len(comms)} communities\n", flush=True)

    client = ollama.Client(host=_host())
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:   # L5
        futs = {ex.submit(label_one, client, c): c for c in comms}
        for f in as_completed(futs):
            results.append(f.result())

    results.sort(key=lambda r: -r["size"])
    ok = [r for r in results if r["label"]]
    out = {
        "run_id": str(run.run_id), "label": run.label,   # psycopg returns UUID
        "model": MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "DRAFT labels over a frozen Louvain partition. Membership was "
                "computed deterministically at ingest; the model only named "
                "each cluster. Never use `label` as a join key — `cid` is the "
                "identity.",
        "communities": results,
    }
    with open(LABEL_OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"\n{len(ok)}/{len(results)} labeled in {time.time()-t0:.0f}s "
          f"-> {LABEL_OUT}")
    if len(ok) < len(results):
        print("failed:", [r["cid"] for r in results if not r["label"]])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
