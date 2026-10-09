"""
text2cypher.py -- ask the neo4j mirror a question in English; a model writes the
Cypher, this module runs it read-only and shows both. SERVE side only.

NO INCUMBENT. Searched before writing: export_neo4j.py writes the mirror and owns
_tx but generates no queries; interpret.py calls models but only over rendered
evidence text, never over the graph; graph_tools.py is postgres-side; the cookbook
is static text with no reader. Nothing here generates or executes Cypher from a
question, so this is a new module -- the one new file of the campaign.

Nothing upstream changes. The construction path (chunking, edges, communities,
walks, the export) is complete before this module is called; it only reads the
mirror those stages built, so the determinism commitment is untouched.

Spec: .spec/specs/graph-explorer/design.md 6.18 (Q1-Q7)
Task: playbook.md T20

Require:   a mirror imported into neo4j (export_neo4j), and one working model
           backend (interpret I5: OPENROUTER_API_KEY, or a local Ollama).
Guarantee: ask() returns {ok, query, rows, attempts, error, ...} -- the query that
           produced the rows, or every attempt that failed. Never a write. Never
           raises on transport or model failure.
Maintain:  read-only against the mirror; SCHEMA is a hand-written constant
           mirroring export_neo4j's writers; the few-shots are parsed from the
           cookbook, never copied into this file.

GUARDS (EARS)
Q1 A generated query SHALL be checked for write clauses and rejected BEFORE it is
   posted. The check SHALL run on the query with string literals blanked and
   comments stripped, so a write word inside a text literal is not a false
   rejection and a write clause hidden behind a comment is not a false pass.
   This module SHALL never construct a write statement of its own.
Q2 Generation SHALL be capped at MAX_ATTEMPTS attempts. A failed attempt -- bad
   JSON, a read-only rejection, or a neo4j error -- SHALL be recorded with its
   query and its error and fed into the next attempt. Exhaustion SHALL return
   ok=False with every attempt kept, and SHALL NOT raise (I4's posture).
Q3 The schema SHALL be a hand-written constant mirroring export_neo4j's writers,
   never apoc.meta introspection: the prompt must not be a function of database
   state. WHERE a writer's node or relationship shape changes, the constant SHALL
   be amended in the same task.
Q4 The few-shot bank SHALL be parsed from docs/cookbook/evidence_queries.cypher at call
   time, never inlined as a copy. The cookbook is the one source of truth for what
   a good query against this mirror looks like (T15).
Q5 The model SHALL produce Cypher only. Rows SHALL be returned as the database
   gave them; narration is opt-in, marked draft, and never edits a row.
Q6 The schema text SHALL state that Chunk.id is a STRING. The CSV import runs
   --id-type=STRING (X9), so a generated query comparing c.id to an integer
   matches nothing and returns zero rows with no error -- a silent wrong answer,
   which is the worst failure this module can have.
Q7 Exactly ONE statement SHALL be executed per attempt. A generated string
   carrying a second statement SHALL be rejected, not split.

Usage:
    python text2cypher.py "which chunks bridge two sources on a highlighted path?" \
        --param prompt="how were Morocco's first elections organized"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import interpret
from export_neo4j import NEO4J_AUTH, NEO4J_DB, NEO4J_HTTP, _tx

FEWSHOT_PATH = Path(__file__).resolve().parent.parent / "docs" / "cookbook" / "evidence_queries.cypher"
MAX_ATTEMPTS = 3          # Q2
ROW_CAP = 200             # rows carried back into the result dict; LIMIT is the query's job
TIMEOUT = 60.0            # per generation call, interactive bound (I5's posture)

SCHEMA = """
NODES
  (:Chunk {id, doc_id, source, text, cid, embedding, salient})
      id       STRING  -- ALWAYS a string. `{id: "3439"}`, never `{id: 3439}`.
      doc_id   STRING  -- the source document this chunk was cut from
      source   STRING  -- corpus name, e.g. "brown", "wikitext"; compare for cross-source hops
      text     STRING  -- the chunk body
      cid      INTEGER -- Louvain community; the node ALSO carries label `C<cid>`
      salient  LIST<STRING> -- top terms, set only for chunks a walk reached
      embedding LIST<FLOAT> -- vector index `chunk_embedding`, cosine
  (:Term {name})           -- name is the id; terms are corpus-scoped
  (:Walk {prompt, run_id, n, edges, wcc, wcc_sizes, largest_component_frac,
          density, conductance})   -- prompt is UNIQUE; one node per walked prompt
  (:CommunitySummary {cid, keywords, size})   -- cid is UNIQUE

RELATIONSHIPS
  (:Chunk)-[:CONTAINS {tf}]->(:Term)
  (:Chunk)-[:SIMILAR {strength, prov, sim_sparse, sim_dense}]->(:Chunk)
  (:Walk)-[:ANCHORS]->(:Chunk)
  (:Chunk)-[:PATHWAY {of, dwpc, n_paths, len}]->(:Chunk)      -- `of` = the walk's prompt
  (:Chunk)-[:NEXT_IN_CHAIN {of, chain, pos}]->(:Chunk)        -- `of` = the walk's prompt
  (:Walk)-[:TOUCHED {hits}]->(:CommunitySummary)

NOTES
  - Walk-scoped relationships (PATHWAY, NEXT_IN_CHAIN) are filtered by `of: $prompt`.
    Omitting `of` mixes every walk ever written.
  - Two chunks sharing a term:
    (a:Chunk)-[:CONTAINS]->(t:Term)<-[:CONTAINS]-(b:Chunk) -- the term-mediated hop.
  - This database is READ-ONLY. Never write MERGE, CREATE, SET, DELETE, REMOVE or DROP.
""".strip()

# Q1 -- matched word-boundary, case-insensitive, on the normalized (literal- and
# comment-stripped) text. START/STOP cover `START DATABASE`; USING covers
# `USING PERIODIC COMMIT`.
_WRITE_CLAUSES = ("CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP",
                  "FOREACH", "LOAD", "GRANT", "REVOKE", "ALTER", "RENAME",
                  "TERMINATE", "START", "STOP", "USING")

_READ_KEYWORDS = ("MATCH", "RETURN", "WITH", "UNWIND", "CALL")

_STRING_LITERAL_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`(?:[^`\\]|\\.)*`")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_SPLIT_RE = re.compile(r"(?m)^//\s*-{3,}\s*$")
_QUOTED_QUESTION_RE = re.compile(r'"([^"]+)"')
_LEADING_INDEX_RE = re.compile(r"(\d+)\.")


# ------------------------------------------------------------- cookbook parsing (Q4)

def load_fewshots(path: Path = FEWSHOT_PATH) -> list[dict]:
    """Guarantee: one dict per cookbook block -- {n, question, note, query} -- in file
    order. Blocks with no quoted question (the file preamble) are skipped."""
    text = path.read_text(encoding="utf-8")
    blocks = _BLOCK_SPLIT_RE.split(text)
    shots = []
    for block in blocks:
        lines = block.splitlines()
        header_lines = [ln for ln in lines if ln.lstrip().startswith("//")]
        body_lines = [ln for ln in lines if not ln.lstrip().startswith("//")]
        joined = " ".join(ln.lstrip()[2:].strip() for ln in header_lines)
        m = _QUOTED_QUESTION_RE.search(joined)
        if not m:
            continue                                            # the preamble drops out
        question = m.group(1)
        idx_m = _LEADING_INDEX_RE.match(joined)
        n = int(idx_m.group(1)) if idx_m else len(shots) + 1
        note = joined
        note = note[:m.start()] + note[m.end():]
        if idx_m and idx_m.start() == 0:
            note = _LEADING_INDEX_RE.sub("", note, count=1)
        note = note.strip()
        body = "\n".join(body_lines).strip()
        if body.endswith(";"):
            body = body[:-1].rstrip()
        if not body:
            continue
        shots.append({"n": n, "question": question, "note": note, "query": body})
    if not shots:
        raise ValueError(f"no few-shots parsed from cookbook at {path}")
    return shots


def render_fewshots(shots: list[dict]) -> str:
    """Q: <question>\nCypher:\n<query>\n\n ... -- the block that goes in the prompt."""
    parts = []
    for s in shots:
        parts.append(f"Q: {s['question']}\nCypher:\n{s['query']}")
    return "\n\n".join(parts)


# --------------------------------------------------------- read-only enforcement (Q1, Q7)

def _normalize(query: str) -> str:
    """Blank string literals FIRST (so `//` inside 'http://x' does not eat the line and
    a write word inside a literal cannot trip the scan), then strip // line comments
    and /* */ blocks. Returns the scannable text."""
    text = _STRING_LITERAL_RE.sub("''", query)
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _LINE_COMMENT_RE.sub("", text)
    return text


def check_read_only(query: str) -> str | None:
    """Guarantee: None when the query is safe to post; otherwise a one-line reason
    naming what was rejected -- that string is what the retry loop shows the model."""
    normalized = _normalize(query or "")
    stripped = normalized.strip()
    if not stripped or not any(
            re.search(rf"\b{kw}\b", stripped, re.IGNORECASE) for kw in _READ_KEYWORDS):
        return "not a Cypher query"
    semi = stripped.find(";")
    if semi != -1 and stripped[semi + 1:].strip():                     # Q7
        return "rejected: more than one statement; send exactly one"
    for word in _WRITE_CLAUSES:                                        # Q1
        if re.search(rf"\b{word}\b", stripped, re.IGNORECASE):
            return f"rejected: write clause {word}; this database is read-only"
    m = re.search(r"\bCALL\s+[A-Za-z_]", stripped, re.IGNORECASE)
    if m:
        name_m = re.search(r"\bCALL\s+([A-Za-z_][A-Za-z0-9_.]*)", stripped, re.IGNORECASE)
        name = name_m.group(1) if name_m else "?"
        return f"rejected: procedure call CALL {name}; not permitted"
    return None


# ------------------------------------------------------------------------ generation

def build_prompt(question: str, shots: list[dict], params: dict | None = None,
                  failures: tuple = ()) -> tuple[str, str]:
    """Guarantee: (system, user). Deterministic for the same inputs (I3's posture)."""
    system = (
        "You write read-only Cypher against a fixed neo4j graph. You never invent "
        "labels, properties or relationship types that are not in the schema.\n\n"
        f"SCHEMA:\n{SCHEMA}\n\n"
        f"EXAMPLES (question -> query):\n{render_fewshots(shots)}\n\n"
        "Reply with ONE JSON object, keys in this order:\n"
        '  "relationships": [ ... ]   names of the labels and relationship types you will\n'
        "                             traverse, from the schema only -- name them BEFORE\n"
        "                             writing the query\n"
        '  "reasoning": "..."         one or two sentences: how those pieces answer the question\n'
        '  "query": "..."             ONE Cypher statement, read-only, no trailing semicolon\n'
        # The order is load-bearing: `relationships` first forces the model to commit
        # to schema elements before it writes syntax.
    )
    user_parts = [f"QUESTION: {question}"]
    if params:
        lines = []
        for k, v in params.items():
            rep = repr(v)
            if len(rep) > 120:
                rep = rep[:120] + "...(clipped)"
            lines.append(f"${k} = {rep}")
        user_parts.append("PARAMETERS AVAILABLE:\n" + "\n".join(lines) +
                           "\nuse these as parameters, do not inline their values")
    if failures:
        lines = []
        for a in failures:
            lines.append(f"query: {a.get('query')}\nerror: {a.get('error')}")
        user_parts.append("PREVIOUS ATTEMPTS FAILED -- fix them:\n" + "\n\n".join(lines))
    return system, "\n\n".join(user_parts)


def generate(question: str, *, shots: list[dict], params: dict | None = None,
             failures: tuple = (), call=None, timeout: float = TIMEOUT) -> dict:
    """Guarantee: {relationships, reasoning, query, backend, raw, error}. A model or
    parse failure fills `error` and leaves `query` empty; it does not raise (Q2)."""
    call = call or interpret._call
    system, user = build_prompt(question, shots, params=params, failures=failures)
    result = {"relationships": [], "reasoning": "", "query": "", "backend": None,
              "raw": "", "error": None}
    try:
        text, backend = call(system, user, timeout)
    except Exception as e:                                              # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return result
    result["raw"] = text
    result["backend"] = backend
    obj = interpret._json_obj(text)
    if not obj or not obj.get("query"):
        result["error"] = "reply was not the JSON object asked for"
        return result
    query = str(obj["query"]).strip()
    query = re.sub(r"^```(?:cypher)?\s*|\s*```$", "", query, flags=re.IGNORECASE).strip()
    if query.endswith(";"):
        query = query[:-1].rstrip()
    result["query"] = query
    result["relationships"] = obj.get("relationships", [])
    result["reasoning"] = obj.get("reasoning", "")
    return result


# ------------------------------------------------------------------------- execution

def execute(query: str, params: dict | None = None, *, post=None,
            url: str = NEO4J_HTTP, auth: tuple = NEO4J_AUTH, db: str = NEO4J_DB) -> dict:
    """Require: check_read_only(query) is None -- assert it here too, so the guard
    cannot be bypassed by calling execute directly (Q1).
    Guarantee: {columns, rows, n, truncated}. rows are dicts, column -> value."""
    reason = check_read_only(query)
    assert reason is None, f"execute() called with a rejected query: {reason}"
    post = post or _tx
    results = post([{"statement": query, "parameters": params or {}}],
                    url=url, auth=auth, db=db)
    payload = results[0]
    columns = payload.get("columns", [])
    data = payload.get("data", [])
    all_rows = [dict(zip(columns, d["row"])) for d in data]
    rows = all_rows[:ROW_CAP]
    return {"columns": columns, "rows": rows, "n": len(all_rows),
            "truncated": len(all_rows) > ROW_CAP}


# ------------------------------------------------------------------------------ loop

def narrate_rows(question: str, columns: list, rows: list, *, call=None,
                  timeout: float = TIMEOUT) -> str | None:
    """Optional (Q5): a draft sentence over the rows the database returned. The rows
    are never edited; this is a caption, not a source."""
    call = call or interpret._call
    system = ('You write one short draft sentence summarizing query results. '
               'Reply with ONE JSON object: {"narration": "..."}. Never invent values '
               'not present in the rows.')
    user = f"QUESTION: {question}\nCOLUMNS: {columns}\nROWS: {json.dumps(rows)[:4000]}"
    try:
        text, _backend = call(system, user, timeout)
    except Exception:                                                    # noqa: BLE001
        return None
    obj = interpret._json_obj(text)
    if not obj:
        return None
    narration = obj.get("narration")
    return str(narration) if narration else None


def ask(question: str, *, params: dict | None = None, max_attempts: int = MAX_ATTEMPTS,
        call=None, post=None, shots: list[dict] | None = None, narrate: bool = False,
        timeout: float = TIMEOUT, url: str = NEO4J_HTTP, auth: tuple = NEO4J_AUTH,
        db: str = NEO4J_DB) -> dict:
    """Guarantee: the result dict below, always -- never raises (Q2)."""
    shots = shots if shots is not None else load_fewshots()
    result = {"question": question, "ok": False, "query": "", "columns": [], "rows": [],
              "n": 0, "truncated": False, "relationships": [], "reasoning": "",
              "backend": None, "attempts": [], "n_attempts": 0, "error": None,
              "narration": None}
    attempts: list[dict] = []
    for i in range(1, max_attempts + 1):
        g = generate(question, shots=shots, params=params, failures=tuple(attempts),
                     call=call, timeout=timeout)
        if g["error"] or not g["query"]:
            attempts.append({"i": i, "stage": "generate", "query": g["query"],
                              "error": g["error"] or "empty query"})
            continue
        reason = check_read_only(g["query"])
        if reason is not None:
            attempts.append({"i": i, "stage": "reject", "query": g["query"],
                              "error": reason})
            continue                                          # never posted (Q1)
        try:
            out = execute(g["query"], params=params, post=post, url=url, auth=auth, db=db)
        except Exception as e:                                          # noqa: BLE001
            attempts.append({"i": i, "stage": "execute", "query": g["query"],
                              "error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue
        result.update(query=g["query"], columns=out["columns"], rows=out["rows"],
                       n=out["n"], truncated=out["truncated"],
                       relationships=g["relationships"], reasoning=g["reasoning"],
                       backend=g["backend"], ok=True)
        break
    result["attempts"] = attempts
    result["n_attempts"] = len(attempts) + (1 if result["ok"] else 0)
    if not result["ok"]:
        result["error"] = " | ".join(
            f"attempt {a['i']} ({a['stage']}): {a['error']}" for a in attempts)
    if narrate and result["ok"]:
        try:
            result["narration"] = narrate_rows(question, result["columns"],
                                               result["rows"], call=call, timeout=timeout)
        except Exception:                                                # noqa: BLE001
            result["narration"] = None
    return result


# -------------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ask the neo4j mirror a question in English.")
    parser.add_argument("question")
    parser.add_argument("--param", action="append", default=[],
                         help="name=value, may repeat")
    parser.add_argument("--attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--narrate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    params = {}
    for p in args.param:
        name, _, value = p.partition("=")
        params[name] = value

    res = ask(args.question, params=params or None, max_attempts=args.attempts,
              narrate=args.narrate)

    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return 0 if res["ok"] else 1

    if not res["ok"]:
        print(f"FAILED after {res['n_attempts']} attempt(s):")
        print(res["error"])
        return 1

    print(res["query"])
    print()
    for row in res["rows"][:20]:
        print(" ".join(f"{k}={v}" for k, v in row.items()))
    print(f"\n{res['n']} rows, {res['n_attempts']} attempt(s), backend={res['backend']}")
    if res.get("narration"):
        print(f"\n{res['narration']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
