"""
interpret.py -- a model classifies a frozen walk's chunks as evidence, then
answers from the entailed ones. SERVE side only.

Nothing upstream changes: chunking, edges, communities, terms, medoids and the
walk are computed before this module is called (sampler S6, the bundle is
frozen first). The output is a DRAFT layer with the same status as the draft
labels -- model-authored, checkable chunk by chunk, never a key, never an
input to anything upstream.

GUARDS (EARS)
I1  Every ord the model names -- as a verdict or as a citation -- SHALL be
    checked against the bundle. Foreign ords are surfaced. A citation in the
    answer that is not in the ENTAILED set is surfaced as self-contradiction.
I2  The request SHALL bound context explicitly (num_ctx on Ollama; the bundle
    is clipped per community so it fits either backend).
I3  render_bundle() SHALL be deterministic: same bundle, same text.
I4  WHERE no backend answers, answer() SHALL return ok=False with the error
    and SHALL NOT raise; the walk is unchanged (R5 posture).
I5  OpenRouter (OPENROUTER_API_KEY, model qwen/qwen3.5-9b) SHALL be the default
    backend; local Ollama the fallback. BOTH requests disable reasoning.
    Measured on both: qwen3.5-oc:4b spent the whole num_predict in <think>
    and returned empty content; qwen/qwen3.5-9b via OpenRouter spent 1699 of
    1800 completion tokens reasoning, finish_reason=length, content empty --
    a 200 OK that answers nothing. A length-truncated reply SHALL be reported
    as truncation, not as bad JSON. The result SHALL name which backend
    answered.
I6  The job is ENTAILMENT, not prose: one verdict per retrieved chunk in
    {entails, contradicts, neutral}, then an answer citing only entailed
    chunks. ENTAILS means 'contains information that answers or partly
    answers' -- measured 2026-08-29: phrased as 'entails an answer' with a
    'most are neutral, do not inflate' bias, the model read the paragraph
    'electoral planning in Morocco ... the first elections' and returned
    'discusses elections but not Morocco's first'. Claim-strict NLI is the
    wrong standard for a how-question. Duplicate verdicts for one id are
    collapsed to the first (measured: 73 verdicts for 64 ids). Output is JSON; a reply that is not parseable JSON, or that
    judges NONE of the shown chunks (measured: qwen3.5-oc:4b invented ords
    0..12, all foreign), is not an answer and falls through to the next
    backend. WHERE a fallback answered, the primary's failure is reported as
    fallback_reason rather than discarded.
I8  Anchors -- the BM25 top-k the walk started from -- SHALL get ANCHOR_MULT
    times the evidence budget of walk-reached chunks: they are where the
    answer most likely is, and at document level a 1,500-char excerpt of a
    13-paragraph essay is one paragraph. Each kept paragraph is capped at a
    third of the budget so breadth beats depth.
    WHERE a chunk is longer than the per-chunk budget, the bundle SHALL show
    the paragraphs that carry the prompt's terms (source order, within the
    budget), falling back to the opening only when no paragraph carries any.
    Measured at document level (2,300-word nodes, 420-char clip): 63 of 64
    retrieved documents were judged neutral on 'how were Morocco's first
    elections organized' -- a question cj37 answers -- because the judge was
    reading openings. Excerpting by the prompt is what makes a document-level
    node judgeable at all.
I7  WHERE RERANK_MODEL names a cached ColBERT checkpoint and pylate imports,
    chunks SHALL be reranked by MaxSim against the prompt and clipped to
    RERANK_TOP before rendering. Otherwise the stage is a no-op and says so.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import graph_tools as gt
from label_communities import _host   # one definition of the Ollama connect address

OPENROUTER_MODEL = os.environ.get("INTERPRET_MODEL", "qwen/qwen3.5-9b")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_MODEL = os.environ.get("LABEL_MODEL", "qwen3.5-oc:4b")
NUM_CTX = int(os.environ.get("INTERPRET_NUM_CTX", "16384"))          # I2
MAX_TOKENS = 4096        # 64 verdicts x ~30 tokens + answer; 1800 truncated (measured)
MAX_CHUNK_CHARS = 1500      # per-chunk evidence budget; excerpted by the prompt (I8)
ANCHOR_MULT = 4             # anchors (BM25 top-k) get 4x: they are where the answer most likely is
RERANK_MODEL = os.environ.get("RERANK_MODEL")                          # I7
RERANK_TOP = int(os.environ.get("RERANK_TOP", "24"))
VERDICTS = ("entails", "contradicts", "neutral")

_CITE = re.compile(r"#(\d+)")
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

SYSTEM = 'You are an evidence classifier. You will be shown a PROMPT, a list of VALID IDS, and chunks of text each tagged [id=<n>]. For EVERY valid id decide, on that chunk\'s content alone: ENTAILS = the chunk contains information that answers or partly answers the prompt; CONTRADICTS = it contains information that contradicts an answer; NEUTRAL = it carries no information bearing on the prompt. A partial answer is ENTAILS. Then write a short answer built only from entailing chunks, citing #<id> after each claim. If nothing entails, leave the answer empty.\n\nRules: exactly one verdict per VALID ID, no more, no fewer. Copy each id exactly from the list; ids you invent or renumber are discarded. Reply with ONE JSON object and nothing else:\n{"verdicts": [{"id": <valid id>, "verdict": "entails"|"contradicts"|"neutral", "why": "<=8 words"}, ...], "answer": "<text with #id citations>"}'

ANSWER_SYSTEM = 'You are answering a PROMPT using ONLY the ENTAILED EXCERPTS shown, each tagged [id=<n>]. Write a short, concrete answer. Cite #<id> after every claim; every claim must trace to an excerpt. Do not use anything not shown. Reply with ONE JSON object and nothing else: {"answer": "<text with #id citations>"}'



def _clip(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ")
    if len(text) <= n:
        return text
    head = text[:n]
    return (head.rsplit(" ", 1)[0] if " " in head else head) + " ..."


# ------------------------------------------------------------------ I7 rerank

def rerank(conn, run, query: str, ords: list[int]) -> tuple[list[int], str]:
    """ColBERT MaxSim over the retrieved chunks, top-RERANK_TOP in rank order.
    Guarantee: (ords, note). No-op with a note when the stage is not armed."""
    if not RERANK_MODEL:
        return ords, "rerank: off (RERANK_MODEL unset)"
    try:
        from pylate import models, rank
    except Exception as e:                                # pragma: no cover
        return ords, f"rerank: off (pylate unavailable: {type(e).__name__})"
    try:
        model = models.ColBERT(model_name_or_path=RERANK_MODEL)
        docs = [gt.node(conn, run, o)["body"] for o in ords]
        qe = model.encode([query], is_query=True)
        de = model.encode([docs], is_query=False)
        scored = rank.rerank(documents_ids=[list(range(len(ords)))],
                             queries_embeddings=qe, documents_embeddings=de)[0]
        keep = [ords[int(s["id"])] for s in scored[:RERANK_TOP]]
        return keep, f"rerank: {RERANK_MODEL}, kept {len(keep)} of {len(ords)}"
    except Exception as e:                                # pragma: no cover
        return ords, f"rerank: failed ({type(e).__name__}), passed through"


def excerpt(body: str, query: str, n_chars: int = MAX_CHUNK_CHARS, embed=None) -> str:
    """I8. The paragraphs of `body` most relevant to the prompt, in SOURCE order,
    within n_chars. Relevance = lexical (distinct prompt terms, then hits)
    blended with dense cosine(embed(prompt), embed(paragraph)) when `embed` is
    given. The dense half is not optional in practice: measured on cj37, the
    prompt terms (morocco / first / elections) are densest in the essay's
    introduction, while the paragraphs that answer HOW say registration,
    districts, voting -- zero lexical overlap with "organized". Lexical-only
    excerpting served the introduction and the judge said neutral; the full
    document judged ENTAILS. With no hit and no embed, the opening is
    returned -- the case a reader should distrust. Deterministic."""
    qt = set(gt.tokenize(query))
    paras = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p.strip()]
    if not paras:
        return ""
    lex = []
    for p in paras:
        toks = gt.tokenize(p)
        hits = sum(1 for t in toks if t in qt)
        distinct = len({t for t in toks if t in qt})
        lex.append(distinct * 10 + hits)
    mx = max(lex) or 1
    score = [x / mx for x in lex]
    if embed is not None:
        try:
            import numpy as _np
            E = embed([query] + [p.replace("\n", " ") for p in paras])
            cos = (E[1:] @ E[0]).tolist()
            lo, hi = min(cos), max(cos)
            cosn = [(c - lo) / (hi - lo) if hi > lo else 0.0 for c in cos]
            score = [0.5 * a + 0.5 * b for a, b in zip(score, cosn)]
        except Exception:
            pass                                          # lexical only
    if max(score) <= 0:
        return _clip(body, n_chars)
    order = sorted(range(len(paras)), key=lambda k: (-score[k], k))
    # Breadth over depth: cap each paragraph at a third of the budget so at
    # least three ranked paragraphs reach the reader. Measured on cj37:
    # paragraphs run ~1,100 chars against a 1,500 budget, so uncapped the
    # excerpt was ONE paragraph -- the introduction -- and the paragraphs that
    # answered HOW (ranks 2, 4, 5, 6) never got in, whatever the scoring.
    cap = max(n_chars // 3 - 1, 300)            # three capped pieces + separators fit exactly
    keep, spent = [], 0
    for k in order:
        if score[k] <= 0:
            break
        cost = min(len(paras[k]), cap) + 1
        if spent + cost > n_chars and keep:
            continue
        keep.append(k); spent += cost
        if spent >= n_chars:
            break
    keep.sort()
    text = " ".join(_clip(paras[k].replace(chr(10), " "), cap) for k in keep)
    return _clip(text, n_chars)


# ------------------------------------------------------------------ I3 render

def render_bundle(conn, run, bundle, terms: dict, concept: dict,
                  ords: list[int] | None = None,
                  max_chunks_per_community: int = 1000, embed=None) -> str:
    """Deterministic evidence text (I3). Same bundle -> same string.

    Every chunk in `ords` is rendered: the VALID IDS header must match the
    bodies shown, or the model is asked to judge chunks it cannot see
    (measured: 64 listed, 56 shown, coverage 0.88). Cutting is the rerank
    stage's job (I7), not this function's."""
    ords = list(ords) if ords is not None else list(bundle.sampled)
    keep = set(ords)
    anchors = set(getattr(bundle, "anchors", []) or [])
    touched = [t for t in gt.communities_touched(conn, run, ords)]
    cid_of = {o: gt.node(conn, run, o)["cid"] for o in ords}
    lines = [f"PROMPT: {bundle.query}", "",
             f"VALID IDS ({len(ords)} chunks -- return exactly {len(ords)} verdicts, "
             f"copy these numbers exactly): " + ", ".join(str(o) for o in ords), "",
             f"The chunks are grouped by community for context; judge each chunk "
             f"by its own [id=...] tag.", ""]
    for t in touched:
        c = t["cid"]
        mine = [o for o in ords if cid_of[o] == c]
        lm = gt.local_medoid(conn, run, mine, weights=bundle.scores)
        gm = gt.community(conn, run, c)["medoid"]
        lines += [f"== COMMUNITY c{c} · {t['hits']} of {t['size']} members retrieved",
                  f"   as the prompt sees it: {' / '.join(terms.get(c, []))}",
                  f"   unsupervised concept:  {' / '.join(concept.get(c, []))}",
                  f"   local medoid #{lm} · global medoid #{gm}"]
        for o in mine[:max_chunks_per_community]:
            nd = gt.node(conn, run, o)
            budget = MAX_CHUNK_CHARS * (ANCHOR_MULT if o in anchors else 1)
            lines.append(f"   [id={o}] ({nd['doc_id']}) {excerpt(nd['body'], bundle.query, budget, embed)}")
        if len(mine) > max_chunks_per_community:
            lines.append(f"   (+{len(mine) - max_chunks_per_community} more retrieved in c{c})")
        lines.append("")
    xc = [x for x in gt.cross_community(conn, run, ords) if x["ord"] in keep]
    if xc:
        lines.append("== IN BETWEEN (retrieved chunks with edges into other retrieved communities)")
        for x in xc[:8]:
            nd = gt.node(conn, run, x["ord"])
            reach = ", ".join(f"c{k}" for k in x["foreign_cids"])
            lines.append(f"   [id={x['ord']}] c{x['cid']} -> {reach} · {_clip(nd['body'], 200)}")
    return "\n".join(lines)


# ------------------------------------------------------------ I6 parse, I1 check

def citations(text: str) -> list[int]:
    seen, out = set(), []
    for m in _CITE.finditer(text or ""):
        o = int(m.group(1))
        if o not in seen:
            seen.add(o); out.append(o)
    return out


def parse_reply(text: str) -> dict | None:
    """I6. {'verdicts': [{ord, verdict, why}], 'answer': str} or None."""
    text = _THINK.sub("", text or "").strip()
    cand = None
    m = _FENCE.search(text)
    if m:
        cand = m.group(1)
    else:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            cand = text[i:j + 1]
    if not cand:
        return None
    try:
        obj = json.loads(cand)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("verdicts"), list):
        return None
    verdicts, seen = [], set()
    for v in obj["verdicts"]:
        try:
            o = int(v["id"] if "id" in v else v["ord"]); vd = str(v.get("verdict", "")).lower().strip()
        except (KeyError, TypeError, ValueError):
            continue
        if vd in VERDICTS and o not in seen:               # first verdict per id wins
            seen.add(o)
            verdicts.append({"ord": o, "verdict": vd, "why": str(v.get("why", ""))[:200]})
    return {"verdicts": verdicts, "answer": str(obj.get("answer", "")).strip()}


def check(parsed: dict, shown: list[int]) -> dict:
    """I1 on both channels."""
    allowed = set(shown)
    verdict_of = {v["ord"]: v["verdict"] for v in parsed["verdicts"]}
    foreign = sorted(o for o in verdict_of if o not in allowed)
    entailed = [v["ord"] for v in parsed["verdicts"]
                if v["verdict"] == "entails" and v["ord"] in allowed]
    cited = citations(parsed["answer"])
    foreign += [o for o in cited if o not in allowed and o not in foreign]
    self_contra = [o for o in cited if o in allowed and o not in entailed]
    judged = [o for o in verdict_of if o in allowed]
    return {"entailed": entailed,
            "contradicts": [v["ord"] for v in parsed["verdicts"]
                            if v["verdict"] == "contradicts" and v["ord"] in allowed],
            "cited": [o for o in cited if o in allowed],
            "foreign": foreign, "self_contradicting": self_contra,
            "coverage": (len(judged) / len(allowed)) if allowed else 0.0}


# --------------------------------------------------------------- I5 backends

def _via_openrouter(system: str, user: str, timeout: float,
                    retries: int = 3) -> tuple[str, str]:
    """Same shape as rl_V2/src/build_pools_r3.py::_openrouter_chat, which runs
    against this endpoint without incident: stdlib urllib, reasoning disabled,
    three attempts with exponential backoff on ANY failure except 400/401/403,
    and empty content treated as a failure worth retrying. httpx with a
    connect-only retry was measured to die on a TLS handshake timeout that the
    endpoint did not reproduce a minute later."""
    import json as _json
    import random
    import time
    import urllib.error
    import urllib.request
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY unset")
    payload = {"model": OPENROUTER_MODEL, "temperature": 0.1,
               "max_tokens": MAX_TOKENS,
               "reasoning": {"enabled": False},                        # I5
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OPENROUTER_URL, data=_json.dumps(payload).encode("utf-8"),
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json",
                         "HTTP-Referer": "http://localhost/chunkgraph",
                         "X-Title": "chunkgraph walker"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = _json.load(r)
            ch = (d.get("choices") or [{}])[0]
            content = (ch.get("message") or {}).get("content") or ""
            if ch.get("finish_reason") == "length" and not content.rstrip().endswith("}"):
                raise RuntimeError(f"truncated at max_tokens={MAX_TOKENS} "
                                   f"(finish_reason=length, {len(content)} chars)")
            if content.strip():
                return content, f"openrouter:{OPENROUTER_MODEL}"
            last = RuntimeError("openrouter returned empty content")
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403):
                raise
            last = e
        except Exception as e:                                          # noqa: BLE001
            last = e
        if attempt < retries - 1:
            time.sleep(1.5 * (2 ** attempt) + random.uniform(0, 1.0))
    raise RuntimeError(f"openrouter exhausted {retries} attempts: "
                       f"{type(last).__name__}: {str(last)[:200]}")


def _via_ollama(system: str, user: str, timeout: float) -> tuple[str, str]:
    import ollama
    client = ollama.Client(host=_host(), timeout=timeout)
    kw = {"model": OLLAMA_MODEL, "format": "json",
          "messages": [{"role": "system", "content": system},
                       {"role": "user", "content": user}],
          "options": {"temperature": 0.1, "num_predict": MAX_TOKENS,
                      "num_ctx": NUM_CTX}}                                # I2
    try:
        resp = client.chat(think=False, **kw)                            # I5
    except TypeError:                                                    # older client
        resp = client.chat(**kw)
    content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
    return (content or ""), f"ollama:{OLLAMA_MODEL}"


def _followup_answer(conn, run, bundle, entailed, embed, timeout, backend) -> dict:
    """Stage two: answer from the ENTAILED excerpts only (I6). Citations are
    checked against the entailed set; anything else is self-contradiction."""
    lines = [f"PROMPT: {bundle.query}", "", "ENTAILED EXCERPTS:"]
    for o in entailed:
        nd = gt.node(conn, run, o)
        lines.append(f"   [id={o}] ({nd['doc_id']}) "
                     f"{excerpt(nd['body'], bundle.query, MAX_CHUNK_CHARS * ANCHOR_MULT, embed)}")
    user = chr(10).join(lines)
    res = {"answer_stage": "followup", "answer_error": None}
    try:
        text, _ = backend(ANSWER_SYSTEM, user, timeout)
    except Exception as e:                                          # I4
        res["answer_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return res
    obj = None
    try:
        i, j = text.find("{"), text.rfind("}")
        obj = json.loads(text[i:j + 1]) if i != -1 and j > i else None
    except json.JSONDecodeError:
        obj = None
    ans = str((obj or {}).get("answer", "")).strip()
    if not ans:
        res["answer_error"] = "follow-up returned no answer"
        return res
    cited = citations(ans)
    res["answer"] = ans
    res["cited"] = [o for o in cited if o in set(entailed)]
    res["self_contradicting"] = [o for o in cited if o not in set(entailed)]
    return res


def answer(conn, run, bundle, terms: dict, concept: dict,
           timeout: float = 240.0, use_rerank: bool = True, embed=None) -> dict:
    """Classify then answer. Never raises on transport failure (I4).

    Returns {ok, backend, text, answer, verdicts, entailed, contradicts, cited,
    foreign, self_contradicting, coverage, shown, rerank_note, evidence,
    drafted_at, error}.
    """
    ords, note = (rerank(conn, run, bundle.query, list(bundle.sampled))
                  if use_rerank else (list(bundle.sampled), "rerank: skipped"))
    evidence = render_bundle(conn, run, bundle, terms, concept, ords=ords, embed=embed)
    out = {"ok": False, "backend": None, "text": "", "answer": "", "verdicts": [],
           "entailed": [], "contradicts": [], "cited": [], "foreign": [],
           "self_contradicting": [], "coverage": 0.0, "shown": ords,
           "rerank_note": note, "evidence": evidence,
           "drafted_at": datetime.now(timezone.utc).isoformat(),
           "error": None, "fallback_reason": None}
    errors = []
    for backend in (_via_openrouter, _via_ollama):                       # I5 order
        try:
            text, name = backend(SYSTEM, evidence, timeout)
        except Exception as e:                                           # I4
            errors.append(f"{backend.__name__}: {type(e).__name__}: {str(e)[:300]}")
            continue
        out["text"] = text                       # last raw reply, even on failure
        parsed = parse_reply(text)
        if parsed is None:                                               # I6
            errors.append(f"{name}: reply was not the JSON object asked for "
                          f"({len(text)} chars)")
            continue
        verdict = check(parsed, ords)                                    # I1
        if verdict["coverage"] == 0.0:
            # every ord it named is foreign, or it named none: not an answer.
            errors.append(f"{name}: judged 0 of {len(ords)} shown chunks "
                          f"({len(verdict['foreign'])} foreign ids)")
            continue
        out.update(parsed); out.update(verdict)
        out["backend"], out["text"], out["ok"] = name, text, True
        out["answer_stage"] = "single"
        if errors:                              # a fallback answered: say why
            out["fallback_reason"] = " | ".join(errors)
        if verdict["entailed"] and not parsed["answer"].strip():
            # Measured: 64 verdicts then an empty answer with one chunk judged
            # entailing. A second, smaller call over the entailed excerpts only.
            out.update(_followup_answer(conn, run, bundle, verdict["entailed"],
                                        embed, timeout, backend))
        return out
    out["error"] = " | ".join(errors) or "no backend available"
    return out
