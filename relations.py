"""relations.py -- Relations v0: unsupervised, directional, typed relation
detection over sentence windows, read from node.body verbatim.

Spec: .spec/specs/graph-explorer/design.md §6.20, guards E10-E14
Task: playbook.md T27 (spec), T28 (this module)

E10 Relations v0 SHALL be built from node.body verbatim -- the one store that
    preserves case, punctuation, and order -- via a per-source deterministic
    sentence splitter (line-per-sentence for brown/quotes; regex boundary scan
    with an abbreviation guard list, a MIN_SENT_TOKENS fragment floor, and
    wikitext @-@/@,@/@.@ normalization for wiki and unlabelled sources) and a
    NEW order-, case-, and clitic-preserving tokenizer rel_tokenize.
    gt.tokenize and chunkgraph._tok SHALL NOT change: W21 depends on the query
    tokenizer and the entity vocabulary being the same vocabulary. Relations
    SHALL be a NEW run-scoped table; the E1 tables' shape survives untouched --
    no ALTER.

E11 Candidates SHALL be ordered pairs of entity occurrences within ONE
    sentence whose intervening surface span is at most MAX_CONNECTOR_TOKENS
    (= 4) tokens -- one constant, serving as both connector bound and
    pair-distance cap. Entity occurrences SHALL be anchored by greedy
    longest-match of the run's stored tf vocabulary over the sentence's
    gt-eligible projection (losslessly reconstructing fit-time Phraser bigram
    merges, which merged over the stopworded stream), and SHALL aggregate over
    canonical_id, falling back to entity_id where resolution has not run --
    never an error.

E12 X's Y and Y of X SHALL normalize to ONE direction-normalized genitive row
    (src=X, dst=Y, template='GEN'). For every other template, direction SHALL
    be surface order: src precedes dst. The relation label is the corpus's own
    connector string -- no ontology, no label inventory. The template tier
    collapses non-closed-class connector tokens to 'w' (e.g. 'was w in'); the
    exact surface connector rides beside it (most frequent form, ties by
    ascending string -- deterministic).

E13 A relations row SHALL be written only where BOTH floors pass: n >=
    MIN_REL_SUPPORT (= 3) supporting sentences AND Dunning G2 >= 10.83
    (gt.llr, W17's gate, one ruler) on the pair x template 2x2 over the run's
    candidate events. Sentence-level NPMI (via entities.npmi_ppmi, sentence
    dfs, N = run sentence count) SHALL ride on the SAME row, so ranking is a
    column choice, never a rebuild (E5's law restated).

E14 The builder SHALL print per-stage wall-clock timings (fetch / split /
    tokenize+match / candidates / score / write) on EVERY build, and NO
    cardinality bound (an E9-style cap) SHALL be added without a measured
    stage exceeding Article VII's budget on a live run. E9's scar cuts both
    ways: chunk windows needed the bound only after 2.47e9 measured slots;
    sentence windows (~10-20 content tokens, gap-capped pairs) are expected
    orders smaller, and bounding an unmeasured pass is as much a sin as not
    bounding a measured one.

Two places in the spec text invite a hand-authored inventory; the operator's
mid-planning ruling ("remember what I said about unsupervised, that's a
supervised smell") forecloses both, and T28's subplan binds the resolution:
"closed-class connectors" (E12) is implemented as NOT is_content -- the ONE
gt-eligibility predicate, reusing stoplist._STOP (chunkgraph R18, "the ONE
stoplist") -- rather than a new CONNECTOR_CLOSED literal; and the "abbreviation
guard list" (E10) is derive_abbreviations(), a census over the run's own
bodies (a type is an abbreviation where it never appears bare in the run),
rather than a typed _ABBREV set. Zero new word lists anywhere in this file.

Require:  a live run exists for the given label (graph_tools.get_run), and
          entities.py has already been built over that run (relations' FKs
          reference entities; an empty entity map raises rather than writing
          zero rows).
Guarantee: build_relations() is read-only against node/entities and
          idempotent per run -- a rebuild replaces exactly that run's
          relations rows, leaving every other run's rows untouched (E3's
          pattern, restated here since relations carries the same FK shape).

DDL lives HERE, not in sql/, for the same reason as entities.py (§6.15 block
C): 001_schema.sql runs once on an empty volume and cannot create a table on
an existing populated database.
"""
from __future__ import annotations

import itertools
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

import entities as ent
import graph_tools as gt
from stoplist import _STOP
import config

DSN = config.DSN
MAX_CONNECTOR_TOKENS = 4   # E11: connector bound AND pair-distance cap, one knob
MIN_REL_SUPPORT = 3        # E13
G2_GATE = 10.83            # E13: W17's gate, one significance ruler
MIN_SENT_TOKENS = 3        # E10: fragment floor
MAX_PHRASE_TOKENS = 4      # E11: longest-match window; see anchor()
POSS = "'s"                # E10: the clitic marker's surface form


@dataclass(frozen=True)
class Tok:
    low: str
    cap: bool
    poss: bool


@dataclass(frozen=True)
class Occ:
    eid: int
    start: int
    end: int      # token index span, end inclusive


@dataclass(frozen=True)
class Cand:
    src: int
    dst: int
    template: str
    connector: str
    ord: int
    sent_i: int


def is_content(low: str) -> bool:
    """gt-eligibility, replicated exactly (gt.tokenize is W21-frozen; this
    mirrors it, it does not call it -- gt.tokenize takes a string and destroys
    order, we need the test per token). The closed class of E12 is exactly
    `not is_content` -- no CONNECTOR_CLOSED constant exists in this file."""
    return low.isalpha() and len(low) > 2 and low not in _STOP


# ------------------------------------------------------------- wikitext (E10)

_WIKI_AT = re.compile(r"\s@([-,.])@\s")


def normalize_wikitext(body: str) -> str:
    return _WIKI_AT.sub(r"\1", body)


# --------------------------------------------- abbreviation census (E10, §0)

_TOK_PERIOD = re.compile(r"([A-Za-z][A-Za-z.]*)\.")   # token immediately before a period
_TOK_BARE = re.compile(r"[A-Za-z]+")


def derive_abbreviations(bodies) -> frozenset:
    """Which period-terminated tokens are part of the token, not a sentence
    boundary -- decided by the corpus, never by a typed list (operator,
    2026-09-05).

    A type is an abbreviation WHERE it never appears bare in the run: 'mr'
    occurs only as 'Mr.', while 'million' occurs bare thousands of times.
    Knob-free -- the test is existence, not a tuned ratio.
    """
    with_period: Counter = Counter()
    bare: set = set()
    for body in bodies:
        for m in _TOK_PERIOD.finditer(body):
            with_period[m.group(1).rstrip(".").lower()] += 1
        end = len(body)
        for m in _TOK_BARE.finditer(body):
            if m.end() < end and body[m.end()] == ".":
                continue   # immediately followed by a period -- not a bare use
            bare.add(m.group(0).lower())
    return frozenset(t for t in with_period if t not in bare)


# ------------------------------------------------------------- splitter (E10)

_BOUNDARY = re.compile(r'[.!?]+["\')\]]*\s+')


def rel_tokenize(sentence: str) -> list:
    """Order-, case- and clitic-preserving tokenizer (E10). NO stopwording and
    NO length floor here -- order, case and clitics are the whole point; the
    filtering happens later in project()."""
    toks = []
    for m in _REL_TOK.finditer(sentence):
        w = m.group(0)
        if w[0].isdigit():
            toks.append(Tok(w, False, False))
            continue
        apos_idx = None
        for i, ch in enumerate(w):
            if ch in ("'", "’"):
                apos_idx = i
                break
        if apos_idx is None:
            toks.append(Tok(w.lower(), w[:1].isupper(), False))
            continue
        base, suffix = w[:apos_idx], w[apos_idx + 1:]
        toks.append(Tok(base.lower(), base[:1].isupper(), False))
        if suffix.lower() in ("s", ""):
            toks.append(Tok(POSS, False, True))
        else:
            toks.append(Tok(suffix.lower(), False, False))
    return toks


_REL_TOK = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]*)?|[0-9]+(?:[.,][0-9]+)*")


def split_sentences(body: str, source, abbrev: frozenset = frozenset()) -> list:
    if source in ("brown", "quotes"):
        sents = [ln.strip() for ln in body.split("\n") if ln.strip()]
    else:
        body = normalize_wikitext(body)
        sents, prev = [], 0
        for m in _BOUNDARY.finditer(body):
            head = body[prev:m.start()]
            last = head.split()[-1] if head.split() else ""
            last_l = last.rstrip(".").lower()
            if len(last_l) <= 1 or "." in last.rstrip(".") or last_l in abbrev:
                continue                       # not a boundary; keep scanning
            sents.append(body[prev:m.end()].strip())
            prev = m.end()
        tail = body[prev:].strip()
        if tail:
            sents.append(tail)
    return [s for s in sents if len(rel_tokenize(s)) >= MIN_SENT_TOKENS]


# --------------------------------------------------- anchoring etc. (E11/E12)


def project(toks) -> list:
    """gt-eligible projection: indices of content tokens, in order (E11)."""
    return [i for i, t in enumerate(toks) if is_content(t.low)]


def anchor(toks, vocab, canon_by_name, max_phrase: int = MAX_PHRASE_TOKENS) -> list:
    """Greedy longest-match over the content projection (E11). Joining
    consecutive PROJECTED tokens with '_' reconstructs a fit-time Phraser
    bigram losslessly, since the Phraser merged over exactly this stopworded
    stream."""
    proj = project(toks)
    occs = []
    p = 0
    while p < len(proj):
        n = min(max_phrase, len(proj) - p)
        matched = False
        for k in range(n, 0, -1):
            idxs = proj[p:p + k]
            phrase = "_".join(toks[i].low for i in idxs)
            if phrase in vocab:
                eid = canon_by_name.get(phrase)
                if eid is not None:
                    occs.append(Occ(eid, idxs[0], idxs[-1]))
                p += k
                matched = True
                break
        if not matched:
            p += 1
    return occs


def template_of(conn_toks) -> str:
    """E12: non-closed-class connector tokens collapse to 'w'; the closed
    class is exactly `not is_content` -- no CONNECTOR_CLOSED constant exists
    in this file. Empty connector (adjacency) yields ''."""
    return " ".join(t.low if not is_content(t.low) else "w" for t in conn_toks)


def candidates(occs, toks, ord: int = 0, sent_i: int = 0,
              max_gap: int = MAX_CONNECTOR_TOKENS) -> list:
    """Ordered pairs of entity occurrences within one sentence, gap-capped
    (E11), direction-normalized (E12) -- the ONLY place GEN reversal happens,
    so nothing downstream reasons about direction twice."""
    out = []
    for i in range(len(occs)):
        oi = occs[i]
        for j in range(i + 1, len(occs)):
            oj = occs[j]
            if oi.eid == oj.eid:
                continue
            gap = oj.start - oi.end - 1
            if not (0 <= gap <= max_gap):
                continue
            c = toks[oi.end + 1:oj.start]
            connector = " ".join(t.low for t in c)
            if len(c) == 1 and c[0].poss:
                out.append(Cand(oi.eid, oj.eid, "GEN", connector, ord, sent_i))
            elif len(c) > 0 and c[0].low == "of" and not any(is_content(t.low) for t in c):
                out.append(Cand(oj.eid, oi.eid, "GEN", connector, ord, sent_i))
            else:
                out.append(Cand(oi.eid, oj.eid, template_of(c), connector, ord, sent_i))
    return out


# --------------------------------------------------------------- scoring (E13)


def score(events, sent_df, pair_sent_df, n_sent) -> list:
    by_pair = Counter((e.src, e.dst) for e in events)
    by_tmpl = Counter(e.template for e in events)
    events_by_key: dict = defaultdict(list)
    for e in events:
        events_by_key[(e.src, e.dst, e.template)].append(e)
    total = len(events)

    rows = []
    for (src, dst, tmpl), evs in events_by_key.items():
        k11 = len(evs)
        k12 = by_pair[(src, dst)] - k11
        k21 = by_tmpl[tmpl] - k11
        k22 = total - k11 - k12 - k21
        llr = gt.llr(k11, k12, k21, k22)

        n = len({(e.ord, e.sent_i) for e in evs})
        if n < MIN_REL_SUPPORT or llr < G2_GATE:
            continue

        npmi, _ppmi = ent.npmi_ppmi(
            sent_df[src], sent_df[dst],
            pair_sent_df[frozenset((src, dst))], n_sent)

        example_ord = min(e.ord for e in evs)
        conns = [e.connector for e in evs]
        connector = min(Counter(conns).items(), key=lambda kv: (-kv[1], kv[0]))[0]

        rows.append({"src": src, "dst": dst, "template": tmpl, "connector": connector,
                     "n": n, "llr": llr, "npmi": npmi, "example_ord": example_ord})
    return rows


# ----------------------------------------------------------- schema/build/CLI

_DDL = (
    """CREATE TABLE IF NOT EXISTS relations (
        run_id      uuid    NOT NULL REFERENCES graph_run ON DELETE CASCADE,
        src         integer NOT NULL,
        dst         integer NOT NULL,
        template    text    NOT NULL,
        connector   text    NOT NULL,
        n           integer NOT NULL,
        llr         real    NOT NULL,
        npmi        real    NOT NULL,
        example_ord integer NOT NULL,
        PRIMARY KEY (run_id, src, dst, template),
        FOREIGN KEY (run_id, src) REFERENCES entities (run_id, entity_id) ON DELETE CASCADE,
        FOREIGN KEY (run_id, dst) REFERENCES entities (run_id, entity_id) ON DELETE CASCADE
    )""",
    """CREATE INDEX IF NOT EXISTS relations_llr ON relations (run_id, llr DESC)""",
)


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        for stmt in _DDL:
            cur.execute(stmt)


def build_relations(conn, run) -> dict:
    ensure_schema(conn)
    rid = run.run_id
    timings = {}

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("""SELECT ord, body, attrs->>'source' AS source
                         FROM node WHERE run_id=%s ORDER BY ord""", (rid,))
        chunk_rows = cur.fetchall()
        cur.execute("SELECT name, COALESCE(canonical_id, entity_id) AS cid "
                   "FROM entities WHERE run_id = %s", (rid,))
        canon_by_name = {r["name"]: r["cid"] for r in cur.fetchall()}
    if not canon_by_name:
        raise LookupError(f"no entities for run {run.label}; run entities.py first")
    vocab = set(gt.corpus_index(conn, run)["df"])
    timings["fetch"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bodies = [r["body"] for r in chunk_rows]
    abbrev = derive_abbreviations(bodies)
    chunks_sents = [(r["ord"], split_sentences(r["body"], r["source"], abbrev))
                    for r in chunk_rows]
    timings["split"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tokenized = []
    sent_df: Counter = Counter()
    pair_sent_df: Counter = Counter()
    n_sent = 0
    for ord_, sents in chunks_sents:
        for sent_i, sent in enumerate(sents):
            toks = rel_tokenize(sent)
            occs = anchor(toks, vocab, canon_by_name)
            tokenized.append((ord_, sent_i, toks, occs))
            n_sent += 1
            eids = sorted({o.eid for o in occs})
            for e in eids:
                sent_df[e] += 1
            for a, b in itertools.combinations(eids, 2):
                pair_sent_df[frozenset((a, b))] += 1
    timings["tokenize+match"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    events = []
    for ord_, sent_i, toks, occs in tokenized:
        events.extend(candidates(occs, toks, ord_, sent_i))
    n_pairs = len({(e.src, e.dst) for e in events})
    timings["candidates"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows = score(events, sent_df, pair_sent_df, n_sent)
    timings["score"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM relations WHERE run_id = %s", (rid,))
        with cur.copy("COPY relations (run_id, src, dst, template, connector, "
                     "n, llr, npmi, example_ord) FROM STDIN") as cp:
            for row in rows:
                cp.write_row((rid, row["src"], row["dst"], row["template"],
                             row["connector"], row["n"], row["llr"], row["npmi"],
                             row["example_ord"]))
    conn.commit()
    timings["write"] = time.perf_counter() - t0

    for stage, dt in timings.items():
        print(f"  {stage:<18} {dt:7.2f}s")

    return {"n_sentences": n_sent, "n_events": len(events), "n_pairs": n_pairs,
            "relations": len(rows), "timings": timings}


def main(argv: list) -> int:
    if len(argv) < 2:
        print("usage: python relations.py <label>", file=sys.stderr)
        return 2
    label = argv[1]
    conn = psycopg.connect(DSN, row_factory=dict_row)
    try:
        try:
            run = gt.get_run(conn, label)
        except LookupError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        result = build_relations(conn, run)

        with conn.cursor() as cur:
            cur.execute("""
                SELECT sa.name AS src_name, sb.name AS dst_name, r.template,
                       r.connector, r.n, r.llr, r.npmi
                  FROM relations r
                  JOIN entities sa ON sa.run_id = r.run_id AND sa.entity_id = r.src
                  JOIN entities sb ON sb.run_id = r.run_id AND sb.entity_id = r.dst
                 WHERE r.run_id = %s
                 ORDER BY r.llr DESC, r.src, r.dst, r.template
                 LIMIT 15""", (run.run_id,))
            top = cur.fetchall()
    finally:
        conn.close()

    print(f"{label} {run.run_id} sentences={result['n_sentences']} "
          f"events={result['n_events']} pairs={result['n_pairs']} "
          f"relations={result['relations']}")
    for row in top:
        print(f"  {row['src_name']!r} -[{row['template']} / {row['connector']!r}]-> "
              f"{row['dst_name']!r}  n={row['n']} llr={row['llr']:.2f} npmi={row['npmi']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
