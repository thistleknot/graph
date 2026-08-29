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
    chunks. Output is JSON; a reply that is not parseable JSON, or that
    judges NONE of the shown chunks (measured: qwen3.5-oc:4b invented ords
    0..12, all foreign), is not an answer and falls through to the next
    backend. WHERE a fallback answered, the primary's failure is reported as
    fallback_reason rather than discarded.
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
MAX_TOKENS = 1800
MAX_CHUNK_CHARS = 420
RERANK_MODEL = os.environ.get("RERANK_MODEL")                          # I7
RERANK_TOP = int(os.environ.get("RERANK_TOP", "24"))
VERDICTS = ("entails", "contradicts", "neutral")

_CITE = re.compile(r"#(\d+)")
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

SYSTEM = 'You are an evidence classifier. You will be shown a PROMPT, a list of VALID IDS, and chunks of text each tagged [id=<n>]. For EVERY valid id, decide whether that chunk ENTAILS an answer to the prompt, CONTRADICTS one, or is NEUTRAL (retrieved but not evidence). Most are neutral; do not inflate. Then write a short answer using ONLY entailing chunks, citing #<id> after each claim. If nothing entails, leave the answer empty.\n\nRules: return exactly one verdict per VALID ID, no more, no fewer. Copy each id exactly from the list; ids you invent or renumber are discarded. Reply with ONE JSON object and nothing else:\n{"verdicts": [{"id": <valid id>, "verdict": "entails"|"contradicts"|"neutral", "why": "<=15 words"}, ...], "answer": "<text with #id citations>"}'


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


# ------------------------------------------------------------------ I3 render

def render_bundle(conn, run, bundle, terms: dict, concept: dict,
                  ords: list[int] | None = None,
                  max_chunks_per_community: int = 1000) -> str:
    """Deterministic evidence text (I3). Same bundle -> same string.

    Every chunk in `ords` is rendered: the VALID IDS header must match the
    bodies shown, or the model is asked to judge chunks it cannot see
    (measured: 64 listed, 56 shown, coverage 0.88). Cutting is the rerank
    stage's job (I7), not this function's."""
    ords = list(ords) if ords is not None else list(bundle.sampled)
    keep = set(ords)
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
            lines.append(f"   [id={o}] ({nd['doc_id']}) {_clip(nd['body'], MAX_CHUNK_CHARS)}")
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
    verdicts = []
    for v in obj["verdicts"]:
        try:
            o = int(v["id"] if "id" in v else v["ord"]); vd = str(v.get("verdict", "")).lower().strip()
        except (KeyError, TypeError, ValueError):
            continue
        if vd in VERDICTS:
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

def _via_openrouter(system: str, user: str, timeout: float) -> tuple[str, str]:
    import httpx
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY unset")
    r = httpx.post(OPENROUTER_URL, timeout=timeout,
                   headers={"Authorization": f"Bearer {key}",
                            "HTTP-Referer": "http://localhost/chunkgraph",
                            "X-Title": "chunkgraph walker"},
                   json={"model": OPENROUTER_MODEL, "temperature": 0.1,
                         "max_tokens": MAX_TOKENS,
                         "reasoning": {"enabled": False},            # I5
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": user}]})
    r.raise_for_status()
    ch = r.json()["choices"][0]
    content = ch["message"].get("content") or ""
    if ch.get("finish_reason") == "length" and not content.rstrip().endswith("}"):
        raise RuntimeError(f"truncated at max_tokens={MAX_TOKENS} "
                           f"(finish_reason=length, content {len(content)} chars)")
    return content, f"openrouter:{OPENROUTER_MODEL}"


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


def answer(conn, run, bundle, terms: dict, concept: dict,
           timeout: float = 240.0, use_rerank: bool = True) -> dict:
    """Classify then answer. Never raises on transport failure (I4).

    Returns {ok, backend, text, answer, verdicts, entailed, contradicts, cited,
    foreign, self_contradicting, coverage, shown, rerank_note, evidence,
    drafted_at, error}.
    """
    ords, note = (rerank(conn, run, bundle.query, list(bundle.sampled))
                  if use_rerank else (list(bundle.sampled), "rerank: skipped"))
    evidence = render_bundle(conn, run, bundle, terms, concept, ords=ords)
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
        if errors:                              # a fallback answered: say why
            out["fallback_reason"] = " | ".join(errors)
        return out
    out["error"] = " | ".join(errors) or "no backend available"
    return out
