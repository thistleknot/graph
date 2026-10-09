"""summarize_clusters.py -- an LLM summary per community, from its exemplar chunks, never truncated.

NO GOVERNING SPEC. Basis: operator instructions 2026-10-02 ("you're going to need an llm to
provide a summary for each cluster using the top n chunks represented for the cluster and not
truncate it. This is the key final piece" ... "just use openrouter"). Exemplars: read from
.tmp/arxiv_exemplars.json, written by arxiv_community_map.derive (1 to 3 per community by size:
medoid, then picks at z = k/n inside the centroid band, per sparsevec-lexsem-graph R26), so the
summary is written from exactly the chunks the map shows. The same file carries the measured
provenance (papers spanned, top paper's share), which goes into the prompt.

    STAGE    MECHANISM
    PROMPT   the FULL text of each exemplar chunk plus the measured provenance line;
             runs of spaces/tabs collapsed to one (S1)
    FIT      refuse, never cut, a prompt that cannot fit the model's context window (S2)
    CALL     OpenRouter chat completions, temperature 0, fixed seed, transforms OFF (S3)
    CHECK    finish_reason == stop AND prompt tokens <= 90% of the context window (S4)
    STORE    .tmp/arxiv_cluster_summaries.json, one record per community, checkpointed (S5)

S1  Collapsing space runs drops no word: markdown tables here are padded with thousands of
    spaces (community 76: 266,540 chars for 175 words). Non-whitespace text is asserted
    identical before and after. Nothing else is shortened, ever.
S2  Tokens are bounded as chars/2 (conservative for glyph soup and punctuation). A prompt that
    needs more than 90% of the model's context is NOT summarized and is recorded as such.
S3  OpenRouter applies "middle-out" prompt compression to some endpoints unless told not to;
    `transforms: []` switches it off so nothing is removed from the prompt behind our back.
S4  finish_reason == "length" means the ANSWER was cut. prompt_tokens near the window means the
    PROMPT was cut. Either is a failure recorded with its reason, not a summary.
S5  A summary is a model-authored DRAFT over a frozen community, never a join key and never
    indistinguishable from a confirmed label (structure.md, determinism boundary). status is
    always "draft"; model, provider, token counts and cost are stored with it. The exemplar
    chunk text goes to a third-party API: these are public arXiv papers, sent on the
    operator's instruction.

Run:  python -u tools\\summarize_clusters.py --ids 76,32,70,2     (pilot)
      python -u tools\\summarize_clusters.py --all                 (resumes from the JSON)
Needs OPENROUTER_API_KEY in the environment (never printed or stored).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_corpora import retrievable

CACHE, EXEMPLARS = ".tmp/arxiv_section_chunks.pkl", ".tmp/arxiv_exemplars.json"
OUT = ".tmp/arxiv_cluster_summaries.json"
API = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-flash-lite"
MAX_TOKENS, SEED, MAX_FILL, WORKERS, RETRIES = 1024, 7, 0.90, 8, 4

PROMPT = """You label clusters of text chunks taken from a corpus of research papers (mostly arXiv) and a few books.

Below are {n} chunk(s) from ONE cluster (cluster {cid}; {size} chunks in all). They are the cluster's medoid and representative neighbours at increasing distance from it. Read all of them in full.

Provenance, measured over ALL {size} chunks of the cluster: {provenance}
{terms}
Reply in exactly this form and nothing else:
TITLE: <at most 8 words naming the topic>
SUMMARY: <2 to 4 sentences: what the cluster is about, and what kind of text it holds (for example related-work discussion, proofs, prompt templates, result tables, method description)>

Rules: use only what the chunks and the provenance say. If one document supplies most of the cluster, say so. If the chunks are unreadable extraction artifacts, garbled text or mostly numbers, say that plainly instead of inventing a topic.

{chunks}"""


def provenance(e: dict) -> str:
    """Guarantee: the measured paper spread of a community, as a sentence the model can use."""
    return "%d distinct document%s; the largest, %s, supplies %.0f%% of the chunks." % (
        e["n_papers"], "" if e["n_papers"] == 1 else "s", e["top_paper"].replace("arxiv/", ""), 100 * e["top_share"])


def collapse_ws(text: str) -> str:
    """S1. Guarantee: text with space/tab runs collapsed and line-end spaces dropped; the
    non-whitespace characters are identical (asserted)."""
    out = re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", " ", text))
    assert re.sub(r"\s+", "", out) == re.sub(r"\s+", "", text), "collapse_ws changed non-whitespace text"
    return out


TERMS_LINE = ("\nDunning terms, measured over the whole cluster (G2 against every other section): the words and phrases most over-represented here: {terms}.\n"
              "They are measurements, not guesses. Where they fit what the chunks say, use them in the TITLE and SUMMARY exactly as written; never use one the chunks do not support.\n")


def build_prompt(cid: int, size: int, exemplars: list[dict], prov: str, terms: list[str] | None = None) -> str:
    """Guarantee: the prompt holding every exemplar in full; when `terms` is given (T164, section mode) the cluster's Dunning terms are in it."""
    parts = []
    for k, r in enumerate(exemplars, 1):
        parts.append('=== chunk %d of %d | paper %s | section "%s" ===\n%s'
                     % (k, len(exemplars), r["doc_id"].replace("arxiv/", ""), r["section_title"], collapse_ws(r["text"])))
    return PROMPT.format(n=len(exemplars), cid=cid, size=size, provenance=prov, chunks="\n\n".join(parts),
                         terms=TERMS_LINE.format(terms=", ".join(terms)) if terms else "")


def fits(prompt_chars: int, context_length: int) -> bool:
    """S2. Guarantee: True iff chars/2 prompt tokens + MAX_TOKENS stay within MAX_FILL of the window."""
    return prompt_chars / 2 + MAX_TOKENS <= MAX_FILL * context_length


def assert_not_truncated(resp: dict, context_length: int) -> None:
    """S4. Raises unless neither the answer nor the prompt was cut."""
    assert resp.get("finish_reason") == "stop", "answer cut: finish_reason=%r" % resp.get("finish_reason")
    assert resp["prompt_tokens"] <= MAX_FILL * context_length, (
        "prompt is %d tokens in a %d window: a provider may truncate silently here" % (resp["prompt_tokens"], context_length))


def parse(content: str) -> tuple[str, str]:
    """Guarantee: (title, summary) from the LAST TITLE/SUMMARY block, so a draft earlier in the
    reply can never be taken for the answer."""
    blocks = re.findall(r"TITLE:\s*(.+?)\s*\n+\s*SUMMARY:\s*(.+?)(?=\n\s*TITLE:|\Z)", content.strip(), re.S)
    assert blocks, "reply not in TITLE/SUMMARY form: %r" % content[:200]
    title, summary = blocks[-1]
    return title.strip(), re.sub(r"\s+", " ", summary).strip()


def _request(path: str, body: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json", "X-Title": "arxiv cluster summaries"}
    if body is not None:
        headers["Authorization"] = "Bearer " + os.environ["OPENROUTER_API_KEY"]
    req = urllib.request.Request(API + path, json.dumps(body).encode() if body is not None else None, headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def context_length_of(model: str) -> int:
    for m in _request("/models")["data"]:
        if m["id"] == model:
            return int(m["context_length"])
    raise SystemExit("model %r is not on OpenRouter" % model)


def chat(model: str, prompt: str) -> dict:
    """S3. Guarantee: the reply plus usage. Retries 429 and 5xx with backoff; anything else raises."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": MAX_TOKENS, "seed": SEED, "transforms": []}
    for attempt in range(RETRIES):
        try:
            d = _request("/chat/completions", body)
            if "error" in d:
                raise urllib.error.HTTPError(API, d["error"].get("code", 500), str(d["error"].get("message")), {}, None)
            ch, u = d["choices"][0], d.get("usage", {})
            return {"content": ch["message"]["content"] or "", "finish_reason": ch.get("finish_reason"),
                    "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                    "cost": u.get("cost"), "provider": d.get("provider")}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise
    raise AssertionError("unreachable")


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)     # a Greek letter in a title must not kill a run


def load_inputs():
    """Guarantee: (records, exemplars) -- the SAME exemplar chunks the map shows, from arxiv_community_map.derive."""
    recs = retrievable(pickle.load(open(CACHE, "rb"))["records"])      # the list an exemplar's `row` indexes (map.load)
    ex = {int(c): v for c, v in json.load(open(EXEMPLARS, encoding="utf-8")).items()}
    return recs, ex


def persist(done: dict, exem: dict, label: str = "arxiv_sect", conn=None) -> int:
    """V7. Guarantee: the label's live build holds exactly the summary records in `done` that were written
    for the exemplars the build holds NOW (same chunk keys), replacing any earlier rows. A record left over
    from an earlier build -- another community numbering, other exemplars -- is not carried over. No live
    build is an error: the map stage opens none, the ingest does. Returns rows written."""
    import sparsevec_store as ss
    conn = conn or ss.connect()
    ss.ensure_build_schema(conn)
    live = ss.live_build(conn, label)
    assert live is not None, "no live build for %r: run tools/ingest_arxiv_sparsevec.py first" % label
    current = [{**d, "completion_tokens": d.get("output_tokens")} for c, d in sorted(done.items())
               if c in exem and d["chunk_keys"] == [e["key"] for e in exem[c]["exemplars"]]]
    n = ss.write_summaries(conn, live[0], current)
    log("persisted %d of %d summary records to build %d" % (n, len(done), live[0]))
    return n


def summarize_one(model: str, context_length: int, c: int, size: int, keys: list, prompt: str, terms: list[str] | None = None) -> dict:
    rec = {"community": c, "size": size, "model": model, "chunk_keys": keys, "prompt_chars": len(prompt)}
    if not fits(len(prompt), context_length):
        return {**rec, "status": "not_summarized",
                "reason": "prompt could need %d+ tokens of a %d window; never truncated" % (len(prompt) // 2, context_length)}
    t0 = time.time()
    try:
        resp = chat(model, prompt)
        assert_not_truncated(resp, context_length)
        title, summary = parse(resp["content"])
    except Exception as e:                                   # one bad community must not stop the other 134
        return {**rec, "status": "failed", "reason": "%s: %s" % (type(e).__name__, str(e)[:300])}
    if terms:                                                # T164: the Dunning terms that SURVIVED into the model's own words, bolded for display (the raw text stays beside)
        from term_salience import bold_terms
        (title_b, used_t), (summary_b, used_s) = bold_terms(title, terms), bold_terms(summary, terms)
        rec = {**rec, "terms": terms, "terms_surviving": [t for t in terms if t in used_t or t in used_s], "title_bold": title_b, "summary_bold": summary_b}
    return {**rec, "status": "draft", "title": title, "summary": summary, "provider": resp["provider"],
            "prompt_tokens": resp["prompt_tokens"], "output_tokens": resp["completion_tokens"], "cost": resp["cost"],
            "chars_per_token": round(len(prompt) / max(resp["prompt_tokens"], 1), 2), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--tag", default="", help="section map (.tmp/sections_<tag>_*): its exemplars and Dunning terms go in the prompt, summaries go to "
                    ".tmp/sections_<tag>_summaries.json, and NOTHING is persisted to Postgres (T165 waits for the operator's go)")
    args = ap.parse_args()
    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is not set"
    out, terms_of = OUT, {}
    if args.tag:
        import section_corpus
        recs = section_corpus.load()
        exem = {int(c): v for c, v in json.load(open(".tmp/sections_%s_exemplars.json" % args.tag, encoding="utf-8")).items()}
        terms_of = {int(c): [t for t, _ in v] for c, v in json.load(open(".tmp/sections_%s_dunning_terms.json" % args.tag, encoding="utf-8")).items()}
        out = ".tmp/sections_%s_summaries.json" % args.tag
    else:
        recs, exem = load_inputs()
    ids = sorted(exem) if args.all else [int(x) for x in args.ids.split(",") if x]
    done = {}
    if os.path.exists(out):
        done = {d["community"]: d for d in json.load(open(out, encoding="utf-8"))}
    ctx = context_length_of(args.model)
    work = []
    for c in ids:
        keys = [e["key"] for e in exem[c]["exemplars"]]
        d0 = done.get(c)
        if (d0 and d0.get("model") == args.model and d0.get("status") in ("draft", "not_summarized")
                and d0.get("chunk_keys") == keys and d0.get("provenance") is not None and d0.get("terms") == (terms_of.get(c) or None)):
            continue                                   # done for THESE exemplars and THESE terms; a stale record is redone
        ex = [recs[e["row"]] for e in exem[c]["exemplars"]]
        prov = provenance(exem[c])
        work.append((c, exem[c]["size"], keys, build_prompt(c, exem[c]["size"], ex, prov, terms_of.get(c)), prov))
    log("%d communities to do, %d already done; model %s (context %d); %d workers"
        % (len(work), len(ids) - len(work), args.model, ctx, args.workers))
    lock, tally = threading.Lock(), {"draft": 0, "failed": 0, "not_summarized": 0}
    t_all = time.time()
    with ThreadPoolExecutor(args.workers) as ex_pool:
        futs = {ex_pool.submit(summarize_one, args.model, ctx, c, s, keys, p, terms_of.get(c)): prov for c, s, keys, p, prov in work}
        for f in as_completed(futs):
            rec = {**f.result(), "provenance": futs[f]}
            with lock:
                done[rec["community"]] = rec
                tally[rec["status"]] += 1
                json.dump(sorted(done.values(), key=lambda d: d["community"]),
                          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            if rec["status"] == "draft":
                log("community %3d (size %4d) %6d tok  %4.1fs | %s" % (rec["community"], rec["size"], rec["prompt_tokens"], rec["seconds"], rec["title"]))
            else:
                log("community %3d %s: %s" % (rec["community"], rec["status"].upper(), rec["reason"]))
    drafts = [d for d in done.values() if d["status"] == "draft"]
    log("done in %.0fs: %s | prompt tokens %d | cost $%.4f" % (
        time.time() - t_all, tally, sum(d["prompt_tokens"] for d in drafts), sum(d["cost"] or 0 for d in drafts)))
    if not args.tag:
        persist(done, exem)


if __name__ == "__main__":
    main()
