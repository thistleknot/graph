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
    (= 5, amended 6.26 T87 -- was 4) tokens -- one constant, serving as both
    connector bound and pair-distance cap. Entity occurrences SHALL be
    anchored by greedy longest-match of the run's stored tf vocabulary over
    the sentence's gt-eligible projection (losslessly reconstructing fit-time
    Phraser bigram merges, which merged over the stopworded stream), and
    SHALL aggregate over canonical_id, falling back to entity_id where
    resolution has not run -- never an error.

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
an existing populated database. The relation_classes DDL added below (E16)
lives here for the identical reason.

Spec: .spec/specs/graph-explorer/design.md §6.24, guards E16/E17/E19
Task: playbook.md T71 (relation classes)

E16 Templates SHALL be grouped into rel_classes by set-cosine similarity
    (|A intersect B| / sqrt(|A|*|B|)) over each template's pair-set, gated by
    BOTH REL_CLASS_MIN_SHARED (shared-pair floor, same ruler as E13/E15) and
    REL_CLASS_SIM (cosine threshold). Partition is connected components (not
    Louvain) so it is order-independent and needs no seed. rel_class SHALL be
    stored in a NEW run-scoped mapping table `relation_classes`
    (run_id, template) -> rel_class -- never an ALTER on `relations`, whose
    PK-heavy, FK-referenced shape makes a functionally-dependent-on-template-
    alone column a normalization defect and an ACCESS EXCLUSIVE lock hazard
    (entities.py:229-251's law, restated here).

E17 The per-class ranking SHALL be a query (top classes by summed n, joining
    relations to relation_classes), not a rebuilt or duplicated column --
    E5's "ranking is a column choice, never a rebuild" restated at class
    grain. This task delivers the ranking query; the factbook render is out
    of scope.

E19 assign_rel_classes SHALL print per-stage wall-clock timings on EVERY
    build (read / similarity / partition / write), and NO cardinality bound
    SHALL be added without a measured stage exceeding Article VII's budget --
    E14's law, restated for the class pass.

E11 amendment (6.26 T87, 2026-09-08): MAX_CONNECTOR_TOKENS moved 4 -> 5.
    Two measurements were taken and both are recorded here as the basis.
    (1) The stored relations' own span distribution is CENSORED by the very
    constant under test -- span==5 and span==6 are exactly 0 because the old
    MAX_CONNECTOR_TOKENS=4 truncated at build time -- so the stored table
    cannot be used to derive its own window. (2) The UNCENSORED inter-entity
    gap distribution, measured over 1,568,471 pairs from 400 chunks with the
    cap raised to 12, is FLAT: gap 0..8 hold 8.9/10.0/10.4/9.4/8.8/8.3/7.8/
    7.3/6.8 percent, mean 5.22, sdev 3.61, median 5, estimator-pair band
    9.45. A flat distribution has no elbow -- it measures sentence geometry,
    not relation quality -- so a band derived from it would be meaningless
    and would only admit noise. Given that derivation failed, the constant
    is tagged CONVENTION rather than DERIVED, citing the +-5 collocation
    context window (Church & Hanks 1990, Word Association Norms, Mutual
    Information, and Lexicography, Computational Linguistics 16(1)), the
    same window word2vec inherited. ReVerb's POS-pattern constraint on the
    relation phrase (Fader et al. 2011) is the alternative best practice but
    remains unavailable since 6.24 bars a tagger. The gold lane
    (src/diag_agentic.py) is the validation for this move, per 6.26 T3(b).
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
MAX_CONNECTOR_TOKENS = 5   # CONVENTION: best practice is the +-5 collocation context
                           # window (Church & Hanks 1990, Word Association Norms,
                           # Mutual Information, and Lexicography, Computational
                           # Linguistics 16(1)), the same window word2vec inherited.
                           # ReVerb's POS-pattern constraint on the relation phrase
                           # (Fader et al. 2011) is the alternative best practice but
                           # is unavailable here since 6.24 bars a tagger. E11:
                           # connector bound AND pair-distance cap, one knob. Moved
                           # 4 -> 5 (6.26 T87, E11 amendment) -- see the module
                           # docstring's E11 amendment note for the measurements
                           # behind the move; the gold lane is the validation.
MIN_REL_SUPPORT = 3        # ARBITRARY: chosen, not derived. Best practice is a
                           # support floor calibrated on held-out data -- ReVerb
                           # (Fader et al. 2011) requires >= 20 distinct argument
                           # pairs. Replace via 6.26 T3(b), gold-lane calibration.
                           # E13
G2_GATE = 10.83            # DERIVED: chi-square critical value, 1 df, p<0.001
                           # (Dunning 1993, gt.llr). E13: W17's gate, one
                           # significance ruler.
MIN_SENT_TOKENS = 3        # ARBITRARY: chosen, not derived. Best practice would be
                           # a floor set by parse validity (e.g. a minimum verb-
                           # bearing span) -- unavailable parser-free (6.24). Replace
                           # via 6.26 T3(b), gold-lane calibration. E10: fragment
                           # floor.
MAX_PHRASE_TOKENS = 4      # ARBITRARY: chosen, not derived. Best practice is
                           # ReVerb's POS-pattern constraint (Fader et al. 2011) on
                           # the phrase's longest match, unavailable parser-free
                           # (6.24). Replace via 6.26 T3(b), gold-lane calibration.
                           # E11: longest-match window; see anchor()
POSS = "'s"                # E10: the clitic marker's surface form
REL_CLASS_MIN_SHARED = 3   # ARBITRARY: chosen, not derived. Best practice is a
                           # support floor calibrated on held-out data (as above).
                           # Replace via 6.26 T3(b), gold-lane calibration. E16
                           # shared-pair floor. Same ruler as E15's CLASS_MIN_JOINT
                           # / E13's MIN_REL_SUPPORT -- three co-occurrences is this
                           # project's one support floor; no new knob invented for
                           # a new pass.
REL_CLASS_SIM = 0.10       # ARBITRARY: chosen, not derived. Best practice is the
                           # same estimator-pair band the CHUNK stage already uses,
                           # computed on the observed cosine distribution (6.26
                           # T3(a)); that census is the named replacement. E16
                           # similarity threshold on set cosine. Basis: pair-set
                           # sizes span ~4 orders of magnitude ('' ~9.7e4 pairs vs a
                           # long template's handful), so any threshold above ~0.2
                           # can only ever join same-size templates. Provisional
                           # until the live census (main() --rel-classes-only) prints
                           # the distribution; the census print IS the evidence
                           # trail (Article XI).


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


# ---------------------------------------------------------- relation classes
# Spec: design.md §6.24 E16/E17/E19 · Task: playbook.md T71


def template_pairsets(rows, split_gen: bool = False) -> dict:
    """{template_key: {(src, dst), ...}}. DB-free -- rows is any iterable of
    mappings/tuples exposing template/connector/src/dst (E16).

    split_gen=True keys GEN rows as f"GEN/{connector}" instead of "GEN"; every
    other template is unchanged. This is the §5-subplan genitive acceptance
    instrument and is OFF on the production path (main() never passes it)."""
    out: dict = defaultdict(set)
    for r in rows:
        template = r["template"] if isinstance(r, dict) else r[0]
        connector = r["connector"] if isinstance(r, dict) else r[1]
        src = r["src"] if isinstance(r, dict) else r[2]
        dst = r["dst"] if isinstance(r, dict) else r[3]
        key = f"GEN/{connector}" if (split_gen and template == "GEN") else template
        out[key].add((src, dst))
    return dict(out)


def _shared_pair_scores(pairsets: dict, min_shared: int) -> dict:
    """{(a, b): (cosine, shared)} for every C(T,2) template pair clearing ONLY
    the shared-pair floor (threshold not applied) -- the shared helper behind
    template_similarity and the below-threshold census, so the T^2/2
    intersection pass runs exactly once."""
    out: dict = {}
    keys = sorted(pairsets)
    for a, b in itertools.combinations(keys, 2):
        sa, sb = pairsets[a], pairsets[b]
        if len(sb) < len(sa):
            sa, sb = sb, sa
        shared = sum(1 for pair in sa if pair in sb)
        if shared < min_shared:
            continue
        cosine = shared / ((len(pairsets[a]) * len(pairsets[b])) ** 0.5)
        out[(a, b)] = (cosine, shared)
    return out


def template_similarity(pairsets: dict, min_shared: int = REL_CLASS_MIN_SHARED,
                        threshold: float = REL_CLASS_SIM) -> dict:
    """{(a, b): (cosine, shared)} for a < b, over all C(T,2) template pairs
    (E16). Emitted only where shared >= min_shared AND cosine >= threshold.
    Python intersects the smaller side; cost is O(min(|A|,|B|)) per pair, so
    the whole pass is ~T^2/2 intersections bounded by the smaller set each
    time -- T is the distinct-template count, measured (not "dozens" as
    first assumed) at T=11,328 on mixed-full-dual, since `template` includes
    the free-text 'w'-collapsed connector shape, not a small closed set.
    similarity stage measured 65.21s there -- seconds-to-a-minute, well
    inside Article VII's 15 min budget, so NO cardinality bound is added
    because nothing has been measured to overrun (E14's law, restated for
    the class pass)."""
    return {pair: sc for pair, sc in _shared_pair_scores(pairsets, min_shared).items()
            if sc[0] >= threshold}


def rel_class_partition(edges: dict, all_templates) -> dict:
    """Connected components over `edges` (E16) via iterative union-find, path
    compression, union by min string so the representative is always the
    component's smallest template -- deterministic without a seed, unlike
    Louvain's random_state (entities.class_partition). Every template in
    `all_templates` with no edge maps to itself, the singleton fallback exactly
    parallel to class_partition's (entities.py:537); non-None for every
    template."""
    parent = {t: t for t in all_templates}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for (a, b) in edges:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    return {t: find(t) for t in parent}


def assign_rel_classes(conn, run, min_shared: int = REL_CLASS_MIN_SHARED,
                       threshold: float = REL_CLASS_SIM, split_gen: bool = False) -> dict:
    """The only writer of relation_classes (E16/E19). Structure mirrors
    entities.assign_classes line-for-line: read, DB-free compute stages, then
    a single write pass, timings throughout.

    split_gen re-splits GEN by connector before computing -- the §5-subplan
    genitive acceptance instrument (main()'s --gen-check); it never writes
    (callers that pass split_gen=True must not commit its result as the
    production mapping)."""
    ensure_schema(conn)
    rid = run.run_id
    timings: dict = {}

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("SELECT template, connector, src, dst FROM relations WHERE run_id=%s",
                    (rid,))
        n_rows = 0
        pairsets: dict = defaultdict(set)
        for r in cur:
            n_rows += 1
            key = f"GEN/{r['connector']}" if (split_gen and r["template"] == "GEN") else r["template"]
            pairsets[key].add((r["src"], r["dst"]))
        pairsets = dict(pairsets)
        print(f"  templates={len(pairsets)} rows={n_rows}")
    timings["read"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = _shared_pair_scores(pairsets, min_shared)
    sim = {pair: sc for pair, sc in scores.items() if sc[0] >= threshold}
    below = len(scores) - len(sim)
    print(f"  edges={len(sim)} below_threshold_above_floor={below}")
    for (a, b), (cos, shared) in sorted(sim.items(), key=lambda kv: -kv[1][0])[:20]:
        print(f"    {a!r} ~ {b!r} cos={cos:.3f} shared={shared}")
    edges = {pair: cos for pair, (cos, _shared) in sim.items()}
    timings["similarity"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    classes = rel_class_partition(edges, sorted(pairsets))
    timings["partition"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if not split_gen:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM relation_classes WHERE run_id = %s", (rid,))
            cur.executemany(
                "INSERT INTO relation_classes (run_id, template, rel_class) VALUES (%s, %s, %s)",
                [(rid, tmpl, cls) for tmpl, cls in classes.items()])
        conn.commit()
    timings["write"] = time.perf_counter() - t0

    for stage, dt in timings.items():
        print(f"  {stage:<18} {dt:7.2f}s")

    return {"templates": len(pairsets), "rows": n_rows, "edges": len(edges),
            "classes": len(set(classes.values())),
            "classed": sum(1 for t, c in classes.items() if c != t),
            "assignments": classes, "timings": timings}


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
    """CREATE TABLE IF NOT EXISTS relation_classes (
        run_id    uuid NOT NULL REFERENCES graph_run ON DELETE CASCADE,
        template  text NOT NULL,
        rel_class text NOT NULL,
        PRIMARY KEY (run_id, template)
    )""",
    """CREATE INDEX IF NOT EXISTS relation_classes_class ON relation_classes (run_id, rel_class)""",
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


def _print_class_ranking(conn, rid) -> None:
    """E17: the per-class ranking QUERY -- top 10 classes by summed n. A
    query, not a rebuilt column (E5's law at class grain); the factbook
    render is out of scope for T71."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rc.rel_class, count(*) AS templates, sum(r.n) AS mass,
                   string_agg(DISTINCT r.template, ', ' ORDER BY r.template) AS members
              FROM relations r JOIN relation_classes rc
                ON rc.run_id = r.run_id AND rc.template = r.template
             WHERE r.run_id = %s GROUP BY rc.rel_class ORDER BY mass DESC LIMIT 10""",
            (rid,))
        for row in cur.fetchall():
            print(f"  class={row['rel_class']!r} templates={row['templates']} "
                  f"mass={row['mass']}  members=[{row['members']}]")


def main(argv: list) -> int:
    """usage: python relations.py <label> [--rel-classes-only] [--gen-check]

    --rel-classes-only skips build_relations and runs assign_rel_classes
    alone against an already-built run, then prints the E17 ranking query --
    mirrors entities.py's --classes-only (one flag on the incumbent CLI beats
    a new module, Article III).

    --gen-check is the §6.24-subplan genitive acceptance instrument: re-reads
    relations with GEN re-split by connector, runs the identical similarity +
    partition, and prints whether the surviving GEN/<connector> keys land in
    one class. Read-only -- writes nothing (assign_rel_classes(split_gen=True)
    never commits)."""
    if len(argv) < 2:
        print("usage: python relations.py <label> [--rel-classes-only] [--gen-check]",
              file=sys.stderr)
        return 2
    label = argv[1]
    rel_classes_only = "--rel-classes-only" in argv[2:]
    gen_check = "--gen-check" in argv[2:]
    conn = psycopg.connect(DSN, row_factory=dict_row)
    try:
        try:
            run = gt.get_run(conn, label)
        except LookupError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if gen_check:
            gc = assign_rel_classes(conn, run, split_gen=True)
            gen_classes = {t: c for t, c in gc["assignments"].items() if t.startswith("GEN/")}
            all_one_class = len(set(gen_classes.values())) <= 1
            print(f"{label} {run.run_id} gen-check templates={gc['templates']}")
            for key, cls in sorted(gen_classes.items()):
                print(f"  {key!r} -> class={cls!r}")
            print(f"  all GEN/* co-classed: {all_one_class}")
            return 0

        if rel_classes_only:
            cls = assign_rel_classes(conn, run)
            print(f"{label} {run.run_id} templates={cls['templates']} "
                  f"rows={cls['rows']} edges={cls['edges']} "
                  f"classes={cls['classes']} classed={cls['classed']}")
            _print_class_ranking(conn, run.run_id)
            return 0

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
