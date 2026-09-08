"""walker_core.py -- the walker's UI-free logic: clip, labels, partitions, assess.

Spec: .spec/specs/graph-explorer/design.md 6.21(c) (walker split, UI wiring only)
Task: playbook.md T36

Imports NOTHING from streamlit, plotly or networkx and opens NO connection at
import time: `python -c "import walker_core"` must exit 0 with the database
down. Every function takes conn/run/state as parameters -- there is no module
state here. Spring layout deliberately stays in walker_app (design 6.21(a)).
"""
from __future__ import annotations

import html
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import evidence
import graph_tools as gt
import interpret

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#FF7F0E",
           "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
           "#BCBD22", "#17BECF"]


def cid_color(cid):
    return "#DDDDDD" if cid is None else PALETTE[int(cid) % len(PALETTE)]


def group_color(gid):
    """Alias so the intent reads at every call site: `cid_color` IS the one
    palette, shared by stored cids and ephemeral louvain gids alike (P6)."""
    return cid_color(gid)


def rgba(hex_color, a):
    """Plotly rejects 8-digit hex (#RRGGBBAA); alpha must be rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


# ---------- UI chrome palette (P15(f)): one home for the non-group hues ----------
# Spec: .spec/specs/graph-explorer/design.md 6.22 P15(f)
# Task: playbook.md T50
PRIMARY = "#7c5cff"     # violet: accents, citation pills, stat bars
GOOD    = "#2ea86a"     # entails / supports
BAD     = "#e5484d"     # contradicts
WARN    = "#e8a33d"     # splits/merges divergence, Partitions accent
MUTED   = "#8b8f9e"     # neutral verdicts, captions
BG_CARD = "#171a23"     # card ground on the near-black canvas
BG_ROW  = "#141722"     # evidence-row ground
BORDER  = "#262a35"     # hairline


def pill(text: str, color: str, *, outline: bool = False) -> str:
    """One status/badge pill (P15(c)/(d)/(e)). Solid = filled tint of `color`
    with a full-hue border; outline = transparent ground, hue text + border.
    Content is html.escape'd; the pill is a single inline <span> and carries no
    newline, so it never disturbs a card's <pre> byte contract.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P15(c)/(d)/(e)
    Task: playbook.md T50
    """
    base = ("display:inline-block;padding:.05rem .45rem;border-radius:999px;"
            "font-size:.72rem;font-weight:600;white-space:nowrap;"
            f"border:1px solid {color};")
    fill = "background:transparent;color:{0}".format(color) if outline else \
        f"background:{rgba(color, 0.18)};color:{color}"
    return f'<span style="{base}{fill}">{html.escape(text)}</span>'


def stat_card(icon: str, label: str, value, caption, color: str = PRIMARY) -> str:
    """One KPI tile of the P15(b) stat row: icon+label line, big number, muted
    one-line caption, and a thin accent bar at the bottom in `color`. Pure
    string builder -- every field is html.escape'd, and the tile is one <div>
    with NO raw newlines (streamlit re-flows them; T48).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P15(b)
    Task: playbook.md T50
    """
    value = html.escape(str(value))
    caption = str(caption or "")
    cap_html = (f'<div style="font-size:.72rem;opacity:.6">{html.escape(caption)}</div>'
                if caption else "")
    return (
        f'<div style="background:linear-gradient(160deg,{rgba(color, 0.16)},{BG_CARD});'
        f'border:1px solid {BORDER};border-radius:8px;padding:.55rem .7rem;margin:.15rem 0">'
        f'<div style="font-size:.75rem;opacity:.75">{html.escape(icon)} {html.escape(label)}</div>'
        f'<div style="font-size:1.5rem;font-weight:700;line-height:1.2">{value}</div>'
        f'{cap_html}'
        f'<div style="height:3px;border-radius:2px;background:{color};margin-top:.4rem"></div>'
        f'</div>'
    )


def evidence_row(head: str, snippet: str, badge: str, color: str) -> str:
    """One judged-evidence row (P15(d)): dark rounded row, id/source/community
    text left, snippet beneath, and the pre-built `badge` pill (from `pill`)
    floated right. `head` and `snippet` are escaped here; `badge` is trusted
    HTML because `pill` already escaped its own text.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P15(d)
    Task: playbook.md T50
    """
    head_html = html.escape(head).replace("\n", "<br>")
    snippet_html = html.escape(snippet).replace("\n", "<br>")
    snippet_div = (f'<div style="font-size:.84rem;opacity:.78;margin-top:.25rem">'
                   f'{snippet_html}</div>' if snippet else "")
    return (
        f'<div style="background:{BG_ROW};border:1px solid {BORDER};'
        f'border-left:3px solid {color};border-radius:6px;padding:.4rem .6rem;'
        f'margin:.25rem 0">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'gap:.5rem"><span>{head_html}</span>{badge}</div>'
        f'{snippet_div}</div>'
    )


def hero_answer(answer: str, cites, model_line: str, color: str = PRIMARY) -> str:
    """The P15(e) answer hero: large type, citation ords as small violet pills,
    model/brief counts as a muted caption. `cites` is an iterable of ords.

    NOTE (accepted cost): the answer body is ESCAPED, so model-authored markdown
    (**bold**, links) renders literally here where the old st.markdown call
    rendered it. P15(e) asks for one hero card, and escaping is the P14(e) rule
    for every card; a markdown-rendered answer cannot live inside a styled div in
    streamlit. If the operator misses formatted answers, the fallback is to drop
    the hero wrapper for the body and keep only the pill/caption rows.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P15(e)
    Task: playbook.md T50
    """
    body = html.escape(answer).replace("\n", "<br>")
    cites = list(cites or [])
    pills_div = ('<div style="margin-top:.5rem">'
                 + " ".join(pill(f"#{o}", color) for o in cites) + '</div>') if cites else ""
    caption_div = (f'<div style="font-size:.76rem;opacity:.6;margin-top:.4rem">'
                   f'{html.escape(model_line)}</div>') if model_line else ""
    return (
        f'<div style="background:linear-gradient(160deg,{rgba(color, 0.14)},{BG_CARD});'
        f'border:1px solid {BORDER};border-left:4px solid {color};border-radius:8px;'
        f'padding:.8rem 1rem;margin:.2rem 0">'
        f'<div style="font-size:1.05rem;line-height:1.55">{body}</div>'
        f'{pills_div}{caption_div}</div>'
    )


_SUPERLATIVE_MARKERS = {
    "most", "best", "greatest", "largest", "biggest", "first", "top", "worst",
    "longest", "highest", "fastest", "leading",
}
_SUPERLATIVE_AGGREGATE_PHRASES = ("how many", "total number", "in total")
# "-est" morphology false positives: ordinary words that happen to end in
# "-est" but carry no comparative/superlative sense. "latest" is the
# borderline case -- it reads as RECENCY ("the latest album"), not a ranking
# over a population, so a prompt asking for "the latest X" is answerable from
# a single dated chunk and should not be routed through the unrankable-by-
# chunk gate the way "the largest X" must be.
_SUPERLATIVE_EST_EXCLUSIONS = {
    "interest", "forest", "honest", "earnest", "west", "test", "request",
    "protest", "harvest", "modest", "latest",
}


def is_superlative(prompt: str) -> bool:
    """A14(a): pure predicate over the PROMPT text -- a superlative/aggregate
    marker makes the prompt unrankable by any single chunk unless some
    entailing chunk carries the same superlative claim (A14(b)).

    Token-based and case-insensitive: an exact-match marker set ("most",
    "best", "first", "top", ...) plus "-est" morphology on a word longer than
    4 chars (comparative adjectives: "largest", "highest", "fastest") that is
    not in the known false-positive set above, plus aggregate-count phrases
    ("how many", "total number", "in total"). No deps -- string ops only.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A14(a)
    Task: playbook.md T76
    """
    text = (prompt or "").lower()
    for phrase in _SUPERLATIVE_AGGREGATE_PHRASES:
        if phrase in text:
            return True
    for tok in text.split():
        tok = tok.strip(string.punctuation)
        if not tok:
            continue
        # A16: split internal hyphens too ("best-selling") so a compound
        # carrying a marker word is caught the same as the bare word would be.
        for piece in tok.split("-"):
            if not piece:
                continue
            if piece in _SUPERLATIVE_MARKERS:
                return True
            if (len(piece) > 4 and piece.endswith("est")
                    and piece not in _SUPERLATIVE_EST_EXCLUSIONS):
                return True
    return False


# A16: possessive determiners/'s that scope a superlative to the SUBJECT's own
# body of work or timeline, rather than to a population. "peak"/"height"/
# "prime"/"high" are included as head nouns because they are inherently
# self-scoped even without an accompanying -est/marker word ("her peak
# years"), whereas a bare superlative marker after a non-possessive
# determiner ("the biggest band") says nothing about scope by itself.
_REFLEXIVE_POSSESSIVES = {"his", "her", "its", "their"}
_REFLEXIVE_HEAD_NOUNS = {"peak", "height", "prime", "high"}
_REFLEXIVE_WINDOW = 4   # ARBITRARY: chosen, not derived. Best practice is a
                        # window calibrated on held-out data (6.26 T3(b),
                        # gold-lane); ReVerb's POS-pattern constraint (Fader et
                        # al. 2011) is unavailable parser-free (6.24).
_REFLEXIVE_IDIOMS = (
    "height of", "peak of", "career high", "personal best", "to date",
    "of his career", "of her career", "of their career", "of its career",
)


def is_reflexive_superlative(text: str) -> bool:
    """A16(a): pure predicate over a TEXT SPAN (a chunk, a premise, a judge
    'why' string) -- True when a superlative/peak claim is SELF-SCOPED: its
    comparison class is the subject's own career or timeline rather than a
    population. Two lexical signals, either one sufficient:

    1. A self-scoped idiom ("height of", "peak of", "career high", "personal
       best", "to date", "of his/her/their/its career") -- these name a
       maximum over the subject's own history by construction.
    2. A possessive pronoun or possessive-'s (his/her/its/their/<name>'s)
       within a short word-window of a superlative marker, -est morphology,
       or a self-scoped head noun (peak/height/prime/high) -- "his biggest
       hit", "Gallagher's greatest record", "her peak years".

    HONESTY (this is lexical, not syntactic): it does not parse dependency
    structure, so it will misfire on constructions where a possessive sits
    near a superlative without governing it ("his review called it the best
    album of the decade" would false-positive on "his ... best"), and it will
    miss reflexive scope expressed without a possessive or listed idiom
    ("a high point for the band" -- no possessive, "high point" not in the
    idiom list). It is a demotion heuristic for A14(b)'s exemption, not a
    parser; false negatives leave the population gate's existing behaviour
    (require a matching population superlative) intact, and false positives
    only widen what counts as reflexive, which is the safer failure direction
    for a "don't crown a winner" gate.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A16(a)
    Task: playbook.md T81
    """
    text = (text or "").lower()
    for idiom in _REFLEXIVE_IDIOMS:
        if idiom in text:
            return True
    tokens = re.findall(r"[a-z']+", text)
    for i, tok in enumerate(tokens):
        if tok in _REFLEXIVE_POSSESSIVES or tok.endswith("'s"):
            window = tokens[i + 1:i + 1 + _REFLEXIVE_WINDOW]
            for w in window:
                if (w in _SUPERLATIVE_MARKERS or w in _REFLEXIVE_HEAD_NOUNS
                        or (len(w) > 4 and w.endswith("est")
                            and w not in _SUPERLATIVE_EST_EXCLUSIONS)):
                    return True
    return False


def count_population_superlatives(texts) -> int:
    """A16(b): count of `texts` that carry a superlative/aggregate marker
    (`is_superlative`) AND are NOT reflexive (`is_reflexive_superlative`).
    Reflexive superlatives establish a maximum over the subject's own
    history and must not count as population-ranking evidence -- this is
    what A14(b)'s `superlative_entails` should have been counting all along.

    A18(a): HEDGED superlatives are not maxima and never count. "one of the
    most famous Malagasy artists" ranks nothing -- it places its subject in an
    unbounded set. Live receipt (2026-09-07): that exact premise exempted the
    gate and let the answer crown a DIFFERENT musician.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A16(b), A18(a)
    Task: playbook.md T81, T83
    """
    return sum(1 for t in texts
               if is_superlative(t)
               and not is_reflexive_superlative(t)
               and not is_hedged_superlative(t))


HEDGES = ("one of the", "among the", "some of the", "one of its",
          "one of his", "one of her", "one of their", "amongst the")


def is_hedged_superlative(text: str) -> bool:
    """A18(a): True when the superlative is hedged into set membership rather
    than a maximum -- "one of the most famous X", "among the greatest Y".
    Such a claim is compatible with any number of other members and so can
    never establish that its subject leads a population.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A18(a)
    Task: playbook.md T83
    """
    low = " ".join(str(text or "").lower().split())
    return any(h in low for h in HEDGES)


def cited_ords(text: str) -> list[int]:
    """Ords the ANSWER TEXT itself cites, as "#123" markers -- order
    preserved, de-duplicated. Pure string/regex, no deps. Mirrors
    interpret.citations()'s "#(\\d+)" convention but lives here so
    answer_gate's callers (and its tests) don't need a judge-shaped record.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A14(c)
    Task: playbook.md T76
    """
    seen, out = set(), []
    for m in re.finditer(r"#(\d+)", text or ""):
        o = int(m.group(1))
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def answer_gate(answer: str, entails: int, *, n_iters: int = 0,
                n_chunks: int = 0, found_entails: int = 0,
                prompt: str | None = None, entail_ords=(), answer_ords=(),
                superlative_entails: int = 0,
                entail_texts: dict | None = None) -> tuple[bool, str]:
    """A6/A14: a confident claim over zero entailing chunks is a defect
    (A6, unchanged, takes precedence -- A14(d)); a confident claim that cites
    a chunk its own judge did not entail is a defect (A14(c)); a confident
    superlative/aggregate claim over entailing chunks that never themselves
    rank the population is a defect (A14(b)). Returns (gated, text). PURE --
    no escaping here; hero_answer escapes what it renders.

    BACKWARD COMPATIBLE: every new parameter is keyword-only with a default
    that reproduces today's behaviour exactly -- a caller that passes only
    (answer, entails, n_iters=, n_chunks=, found_entails=) is unaffected,
    because with prompt=None and empty ords the A14(b)/(c) checks below can
    never fire (is_superlative(None) is False; an empty answer_ords has
    nothing to fall outside entail_ords).

    entails == 0 -> gated on the A6 text: the corpus does not answer this,
    with the walk size and how many agentic iterations were tried, pointing
    at Agentic retrieval when the loop separately surfaced entails.

    entails > 0 -> A14(c) is checked before A14(b): an answer that cites an
    ord outside entail_ords is gated regardless of the superlative question,
    because citing unjudged/non-entailing evidence is a defect on its own.
    Then A14(b): a superlative/aggregate prompt with superlative_entails == 0
    gates even though entails > 0 -- entailing chunks about one candidate
    never establish a maximum over a population; the caller renders the
    candidate set separately (this function only says the two channels
    disagree, it does not build that render).

    Spec: .spec/specs/graph-explorer/design.md 6.23 A6, A14(a)-(d)
    Task: playbook.md T62, T76
    """
    if entails <= 0:
        tried = (f" {n_iters} agentic iteration{'s' if n_iters != 1 else ''} were tried"
                 if n_iters else " No further iterations were tried")
        text = (f"The corpus, as walked, does not answer this. {n_chunks} chunks were "
                f"walked and none of them entails the prompt.{tried}. "
                f"A superlative or aggregate prompt (\"most famous\", \"best\", \"first\") "
                f"is not entailed by any single chunk -- the graph can show what it "
                f"holds, it cannot crown a candidate.")
        if found_entails:
            text += (f" Agentic retrieval did surface {found_entails} entailing "
                     f"chunk{'s' if found_entails != 1 else ''} -- see Agentic "
                     f"retrieval, below.")
        return True, text

    entail_ords = list(entail_ords)
    answer_ords = list(answer_ords)
    uncited = [o for o in answer_ords if o not in entail_ords]
    if uncited:
        ids = ", ".join(f"#{o}" for o in uncited)
        text = (f"The answer cites chunk{'s' if len(uncited) != 1 else ''} its own "
                f"judge did not mark entailing: {ids}. {n_chunks} chunks were walked "
                f"and {entails} entailed the prompt; the answer above is withheld "
                f"until it is grounded only in what the judge accepted.")
        return True, text

    # A18(b): when the caller supplies the entailing chunks' own text, the
    # ranking evidence is recounted over ONLY the chunks this answer CITES.
    # Evidence about a subject the answer does not crown cannot license the
    # crown -- live receipt: "Rakoto Frah was one of the most famous Malagasy
    # artists" (#8680, uncited) exempted an answer crowning Gallagher (#3594).
    if entail_texts:
        cited = set(answer_ords) & set(entail_ords)
        superlative_entails = count_population_superlatives(
            [entail_texts[o] for o in cited if o in entail_texts])

    if prompt is not None and is_superlative(prompt) and superlative_entails == 0:
        text = (f"The corpus can show what it holds about the candidate{'s' if entails != 1 else ''} "
                f"named in the entailing evidence, but it cannot rank or crown a winner across "
                f"the wider population this prompt asks about. {n_chunks} chunks were walked "
                f"and {entails} entailed the prompt, but none of the entailing chunks itself "
                f"carries a superlative or ranking claim -- entailing evidence about one "
                f"candidate never establishes a maximum. The candidates the corpus holds are "
                f"below.")
        return True, text

    return False, answer


def needs_more_evidence(answer: str, entails: int, *, prompt: str | None = None,
                         entail_ords=(), answer_ords=(),
                         superlative_entails: int = 0,
                         entail_texts: dict | None = None) -> bool:
    """A15: the loop's trigger and the gate's verdict are the SAME question --
    "is this answer insufficient" -- so they must consult the SAME logic
    rather than two definitions that can drift apart. Delegates to
    answer_gate() and returns only its `gated` flag; the branch logic (A6
    zero-entails, A14(c) uncited-ord, A14(b) unranked-superlative) lives in
    exactly one place.

    Live defect this closes: run A had 0 entails, the old `_entails == 0`
    trigger fired, the loop found Nirvana. Run B had 3 entails, each merely
    "X was famous in the 1990s" -- the old trigger never fired because it
    only checked entails == 0, the loop never ran, and the candidate set held
    only Frah/Gallagher/Gilmour. answer_gate already gates run B (A14(b));
    this predicate makes that gate the loop's trigger too.

    Spec: .spec/specs/graph-explorer/design.md A15
    Task: playbook.md T80
    """
    gated, _ = answer_gate(answer, entails, prompt=prompt, entail_ords=entail_ords,
                           entail_texts=entail_texts,
                            answer_ords=answer_ords,
                            superlative_entails=superlative_entails)
    return gated


def loop_answer_caption(n_iters: int) -> str:
    """A13: the hero caption when the answer is drawn from the loop's
    entails-first bundle rather than the base walk. PURE, no escaping --
    hero_answer escapes what it renders.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A13
    Task: playbook.md T67
    """
    return f"answered after {n_iters} agentic iteration{'s' if n_iters != 1 else ''}"


def stop_reason_label(stop_reason: str) -> str:
    """A17(d): honest stop semantics. "budget" means the loop ran out of
    iterations -- it is NOT a claim that the evidence was judged sufficient,
    and every surface that prints a stop_reason must not read it that way.
    Every other stop_reason (sufficient/no-movement/fixed-point/no-op
    action/...) already names itself honestly and passes through unchanged.

    Spec: .spec/specs/graph-explorer/design.md 6.23 A17(d)
    Task: playbook.md T81
    """
    if stop_reason == "budget":
        return "budget (iterations exhausted, not sufficiency)"
    return stop_reason


def clip(text: str, n: int) -> str:
    """Never cut inside a word (design 6.3). Clip at the last whitespace
    before n and mark the cut; short text is returned untouched."""
    text = text or ""
    if len(text) <= n:
        return text
    head = text[:n]
    cut = head.rsplit(None, 1)[0] if " " in head else head
    return cut + " …"


def load_labels(path: Path, run_id: str) -> dict:
    """Draft labels for THIS run, or {}. Absent file, unreadable file and
    malformed JSON are all the normal empty state.

    cid is RUN-LOCAL: a labels file drafted against another run is not stale,
    it is wrong -- it put 'early electrical science history' on the
    Moroccan elections. A foreign file returns {"__stale_run__": <its run_id>} so the
    caller can say so; it never returns that file's labels.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if str(data.get("run_id")) != str(run_id):
        return {"__stale_run__": data.get("run_id")}
    return {c["cid"]: c for c in data.get("communities", []) if c.get("label")}


def src_of_map(conn, run, ords) -> dict:
    """{ord: source-or-None} for `ords`. One line, two call sites in the UI
    (the 3D tab and the Map partitions table) that each had their own copy."""
    return {o: gt.source_of(gt.node(conn, run, o)) for o in ords}


def partition_rows(ds: dict, src_of: dict, *, counts=None) -> list[dict]:
    """One row per dendrite CHUNK chain: size, source mix, and the chain
    members' own BM25-salient vocabulary ranked by how many members carry it
    (top 12; 'term(n)' when n > 1). Pure: `ds` is dendrite_state's output,
    `src_of` comes from src_of_map. No DB, no st.

    `counts`, when given, is `group_counts` output keyed by the SAME 1-based
    chain number this function already assigns (E17 amendment, grouping 3):
    each row gains "entities"/"relations" columns right after "chunks". When
    `counts` is None the output is byte-identical to before T72.

    Spec: .spec/specs/graph-explorer/design.md 6.22, 6.24 E17 amendment
    Task: playbook.md T39, T72
    """
    sal = ds["sal"]
    rows = []
    for ci, chain in enumerate(ds["chunks"]["chains"]):
        tc = Counter()
        mix = Counter()
        for o in chain:
            for t in (sal.get(o, {}).get("top") or []):
                tc[t] += 1
            mix[src_of.get(o) or "?"] += 1
        row = {
            "chain": ci + 1,
            "chunks": len(chain),
        }
        if counts is not None:
            cc = counts.get(ci + 1, {})
            row["entities"] = cc.get("entities", 0)
            row["relations"] = cc.get("relations", 0)
        row["sources"] = " ".join(f"{k}:{v}" for k, v in mix.most_common())
        row["salient terms (carried by N members)"] = \
            ", ".join(f"{t}({n})" if n > 1 else t for t, n in tc.most_common(12))
        rows.append(row)
    return rows


# ---------- analysis view (T39, design 6.22) ----------

LOUVAIN_SEED = 7   # CONVENTION: reproducibility only; any fixed value works.
                    # repo convention: chunkgraph.py:732, graph3d.py:168


def node_dwpc(pathways: dict) -> dict:
    """P3 numerator: node_dwpc(o) = sum of pair['dwpc'] over every pathway pair
    whose BEST path contains chunk o. Chunks on no best path are absent (== 0.0).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P3
    Task: playbook.md T39
    """
    out: dict = {}
    for p in (pathways.get("pairs") or []):
        dwpc = float(p["dwpc"])
        for o in (p.get("path") or []):
            out[o] = out.get(o, 0.0) + dwpc
    return out


def term_bm25_rank(members, sal) -> dict:
    """BM25 salience proxy for the P3 tie-break. gt.chunk_salient returns terms
    already ORDERED by BM25 descending but not the scores themselves, so
    salience is the summed normalised rank over the member chunks that carry
    the term: sum over o of (len(kept_o) - i) / len(kept_o) for the term at
    index i. Monotone in the per-chunk BM25 order, which is the only signal
    the gate exposes.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P3
    Task: playbook.md T39
    """
    out: dict = {}
    for o in members:
        kept = sal.get(o, {}).get("kept") or []
        n = len(kept)
        if not n:
            continue
        for i, t in enumerate(kept):
            out[t] = out.get(t, 0.0) + (n - i) / n
    return out


def rank_group_terms(members, sal, ndw, k=12) -> list:
    """P3: term_dwpc(t) = sum of node_dwpc(o) over member chunks o whose
    salient vocab carries t. Sort key = (-term_dwpc, -bm25_rank, term): ties
    break by BM25 salience then lexicographic, and a zero-dwpc term therefore
    ranks below every positive term while keeping its BM25 order inside the
    zero class.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P3
    Task: playbook.md T39
    """
    bm25 = term_bm25_rank(members, sal)
    term_dwpc: dict = {}
    term_chunks: dict = {}
    for o in members:
        kept = sal.get(o, {}).get("kept") or []
        dw = ndw.get(o, 0.0)
        for t in kept:
            term_dwpc[t] = term_dwpc.get(t, 0.0) + dw
            term_chunks.setdefault(t, []).append(o)
    rows = [{"term": t, "dwpc": term_dwpc[t], "bm25": bm25.get(t, 0.0),
             "chunks": term_chunks[t]} for t in term_dwpc]
    rows.sort(key=lambda r: (-r["dwpc"], -r["bm25"], r["term"]))
    return rows if k is None else rows[:k]


def subgraph_louvain(ords, edges, seed=LOUVAIN_SEED) -> dict:
    """P4 LEFT panel: louvain re-run on the walked subgraph only. `edges` is
    evidence.subgraph_edge_weights output (weight = strength). EPHEMERAL VIEW
    ONLY -- these ids are never persisted, never joined to community.cid,
    never carried across reruns (design 6.22 P4). Isolated chunks each get
    their own group.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P4
    Task: playbook.md T39
    """
    ords = list(ords)
    if len(ords) < 2 or not edges:
        return {o: i for i, o in enumerate(sorted(ords))}

    import networkx as nx
    import community as community_louvain

    G = nx.Graph()
    G.add_nodes_from(ords)
    for (a, b), w in edges.items():
        G.add_edge(a, b, weight=w)
    part = community_louvain.best_partition(G, weight="weight", random_state=seed)

    groups: dict = {}
    for o, g in part.items():
        groups.setdefault(g, []).append(o)
    order = sorted(groups, key=lambda g: (-len(groups[g]), min(groups[g])))
    relabel = {g: i for i, g in enumerate(order)}
    return {o: relabel[g] for o, g in part.items()}


def group_entities(members, ents, floor=2, k=12) -> list:
    """P5: entities ranked by lift = (mention share in group) / (mention share
    in corpus), with a mention floor. group_share = sum(cnt over member
    chunks) / sum(cnt over member chunks, all entities); corpus_share =
    corpus[eid] / corpus_total. Entities with group_cnt < floor are dropped (a
    single mention gives an unbounded, meaningless lift). Sort
    (-lift, -cnt, name).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P5
    Task: playbook.md T39
    """
    mentions = ents["mentions"]
    names = ents["names"]
    corpus = ents["corpus"]
    corpus_total = ents["corpus_total"]

    group_cnt: dict = {}
    for o in members:
        for eid, cnt in mentions.get(o, {}).items():
            group_cnt[eid] = group_cnt.get(eid, 0) + cnt
    group_total = sum(group_cnt.values())

    rows = []
    for eid, cnt in group_cnt.items():
        if cnt < floor:
            continue
        if group_total <= 0:
            continue
        corpus_cnt = corpus.get(eid, 0)
        if corpus_total <= 0 or corpus_cnt <= 0:
            continue
        group_share = cnt / group_total
        corpus_share = corpus_cnt / corpus_total
        lift = group_share / corpus_share
        rows.append({"entity_id": eid, "name": names.get(eid), "cnt": cnt,
                     "lift": lift, "group_share": group_share,
                     "corpus_share": corpus_share})
    rows.sort(key=lambda r: (-r["lift"], -r["cnt"], r["name"] or ""))
    return rows if k is None else rows[:k]


def group_relations(members, ents, rels, k=12) -> list:
    """P5: relation templates counted over pairs whose src AND dst entities
    are BOTH mentioned in this group's chunks. Counting is per
    (template, connector): `n` sums the corpus-level co-occurrence count,
    `pairs` counts distinct entity pairs that qualified inside the group.
    E18 ranking half: rows lead with walk-local mass (`pairs`), corpus-wide
    `n` demoted to the tie-break -- the scoping half of E18 is already this
    function's job (only rows whose src AND dst are both mentioned in the
    group qualify, and `top` is drawn from exactly those rows).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P5, 6.24 E18
    Task: playbook.md T39, T72
    """
    mentions = ents["mentions"]
    present = {eid for o in members for eid in mentions.get(o, {})}

    agg: dict = {}
    for r in rels:
        if r["src"] not in present or r["dst"] not in present:
            continue
        key = (r["template"], r["connector"])
        a = agg.setdefault(key, {"n": 0, "pairs": set(), "top": []})
        a["n"] += r["n"]
        a["pairs"].add((r["src"], r["dst"]))
        a["top"].append((r["src_name"], r["dst_name"], r["llr"]))

    rows = []
    for (template, connector), a in agg.items():
        top = sorted(a["top"], key=lambda t: -t[2])[:3]
        rows.append({"template": template, "connector": connector, "n": a["n"],
                     "pairs": len(a["pairs"]), "top": top})
    rows.sort(key=lambda r: (-r["pairs"], -r["n"], r["template"], r["connector"]))
    return rows if k is None else rows[:k]


def group_counts(groups, ents, rels) -> dict:
    """E17 amendment: the ONE counts function. `groups` is {group_id: [ords]}
    -- global cids, relative louvain groups, and the correlation-sorted
    dendrite chains all reduce to that shape, so the three numbers are
    commensurable across every grouping the walker renders. Pure, DB-free.

    - "chunks" = len(members) (member list as given; caller owns dedup).
    - "entities" = count of DISTINCT entity ids mentioned in the group's chunks.
    - "relations" = count of DISTINCT (src, dst, template) relation ROWS whose
      src AND dst are both in that entity set. A different grain from
      `group_relations`, which aggregates by (template, connector); this
      counts rows.

    Guarantee: {gid: {"chunks": int, "entities": int, "relations": int}} with
    a key for EVERY gid in `groups` -- an empty member list gives three zeros.

    Spec: .spec/specs/graph-explorer/design.md 6.24 E17 amendment
    Task: playbook.md T72
    """
    mentions = ents.get("mentions", {})
    out: dict = {}
    for gid, members in groups.items():
        present = {eid for o in members for eid in mentions.get(o, {})}
        n_rel = sum(1 for r in rels if r["src"] in present and r["dst"] in present)
        out[gid] = {"chunks": len(members), "entities": len(present),
                    "relations": n_rel}
    return out


def group_classes(groups, ents, rels, sal, ndw, *, floor=2, k=12) -> list:
    """The one function walker_app (T40) calls per panel. `groups` is
    {ord: gid} (from subgraph_louvain for the relative panel, or
    Evidence.cid_of restricted to the walked chunks for the global panel --
    same shape, so ONE function serves both panels of P4). `sal` degrades to
    {} when ev.dendrite is None -- terms come back empty, entities/relations
    still populate, never a raise. Each row carries "counts" from
    `group_counts` (E17 amendment) -- the three commensurable numbers, never
    recomputed a second, divergent way.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P4/P5, 6.24 E17 amendment
    Task: playbook.md T39, T72
    """
    by_group: dict = {}
    for o, gid in groups.items():
        by_group.setdefault(gid, []).append(o)

    counts = group_counts(by_group, ents, rels)

    rows = []
    for gid, members in by_group.items():
        members = sorted(members)
        rows.append({
            "gid": gid,
            "members": members,
            "size": len(members),
            "terms": rank_group_terms(members, sal, ndw, k=k),
            "entities": group_entities(members, ents, floor=floor, k=k),
            "relations": group_relations(members, ents, rels, k=k),
            "counts": counts[gid],
        })
    rows.sort(key=lambda r: (-r["size"], r["gid"]))
    return rows


def assess(conn, run, bundle, ev, ds, q, *, embed=None) -> tuple[dict, str | None]:
    """ONE model call (I13) over the walk, then the best-effort neo4j mirror.
    Returns (reason-result, mirror_error-or-None); the caller owns the spinner,
    the session_state write and how the error is shown.

    NEO4J_MIRROR=0 skips the mirror entirely (T17 behaviour, unchanged).
    """
    rr = interpret.reason(conn, run, bundle, ev.terms, ev.concept, embed=embed,
                          judge=True, pw=ev.pathways, digest=ev.digest)
    err = None
    if os.environ.get("NEO4J_MIRROR", "1") != "0":
        err = evidence.mirror_walk(bundle, ev, ds, prompt=q)
    return rr, err


# ---------- factbook digests (T43, design 6.22 P10-P12) ----------

_SEP = " . "


def _pack(items, width, *, sep=_SEP, prefix="", empty="-", max_items=None) -> str:
    """Greedy width packer, the single truncation rule for every rendered line
    (P11(f)). Keeps items, in order, only while the line so far PLUS a
    projected " + N more" tail still fits `width`; the rest collapse into
    that trailing count. `max_items` caps the list before packing even
    considers width (P11(d)'s per-panel cap) -- items past the cap fold into
    the same "+ N more" count as anything width drops. A single item that
    alone exceeds `width` is kept whole rather than cut mid-token (`clip`'s
    posture).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P11
    Task: playbook.md T43
    """
    items = list(items)
    total = len(items)
    if max_items is not None and total > max_items:
        capped, overflow = items[:max_items], total - max_items
    else:
        capped, overflow = items, 0

    if not capped:
        return prefix + (f"+ {overflow} more" if overflow else empty)

    kept = []
    for item in capped:
        trial = kept + [item]
        remaining = overflow + (len(capped) - len(trial))
        line = prefix + sep.join(trial)
        probe = line + sep + f"+ {remaining} more" if remaining else line
        if len(probe) <= width or not kept:
            kept = trial
        else:
            break

    dropped = len(capped) - len(kept)
    remaining = overflow + dropped
    line = prefix + sep.join(kept)
    if remaining:
        line += sep + f"+ {remaining} more"
    return line


def group_digest(gc, ents, src_of=None, *, prefix="g", marker=None, width=110,
                  max_pairs=8) -> str:
    """Render ONE `group_classes` row as 4 TOON-style lines joined by "\\n":
    header, terms(dwpc), entities(mentions), relations. Pure/deterministic --
    same inputs produce a byte-identical string, so the UI can diff runs.

    Header (E17 amendment): when `gc.get("counts")` is present, the two extra
    counts land right after chunks, before the source mix -- old callers with
    no "counts" key keep their exact string.

    Entity items (E15): a classed, non-singleton entity gets a trailing
    " ~<class label>" -- `class_id == entity_id` is the singleton rule, not a
    real class, and is never annotated.

    Relations (E18): walk-local `pairs` leads, corpus-wide `n` demotes to a
    `corpus_n=` suffix -- the scoping half of E18 already lives in
    `group_relations` (only rows whose src AND dst are both mentioned in the
    group qualify, and `top` is drawn from exactly those rows).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P10/P11, 6.24 E15/E17/E18
    Task: playbook.md T43, T72
    """
    members = gc.get("members") or []
    header = f"{prefix}{gc['gid']} . {gc['size']} chunks"
    counts = gc.get("counts")
    if counts is not None:
        header += f" . {counts.get('entities', 0)} entities . {counts.get('relations', 0)} relations"
    if src_of is not None:
        mix = Counter(src_of.get(o) or "?" for o in members)
        mix_str = " ".join(f"{name}:{cnt}" for name, cnt in
                            sorted(mix.items(), key=lambda kv: (-kv[1], kv[0])))
        header += " . " + mix_str
    if marker is not None:
        header += " . " + marker

    terms = gc.get("terms") or []
    terms_line = _pack([f"{t['term']} {t['dwpc']:.1f}" for t in terms], width,
                        prefix="terms(dwpc): ")

    def _corpus_cnt(eid):
        c = ents.get("corpus", {}).get(eid)
        if c is not None:
            return c
        total = ents.get("corpus_total") or 0
        return round((row.get("group_share") or 0) * total)

    names = ents.get("names", {})
    class_of = ents.get("class_of") or {}
    class_names = ents.get("class_names") or {}
    ent_items = []
    for row in sorted(gc.get("entities") or [],
                       key=lambda e: (-e["cnt"], e.get("name") or e["entity_id"])):
        eid = row["entity_id"]
        name = row.get("name") or names.get(eid, eid)
        corpus_cnt = _corpus_cnt(eid)
        item = f"{name} {row['cnt']}"
        if corpus_cnt > row["cnt"]:
            item += f" x{row['lift']:.1f}"
        cls = class_of.get(eid)
        if cls is not None and cls != eid:                     # E15 singleton rule
            item += " ~" + str(class_names.get(cls, cls))
        ent_items.append(item)
    ent_line = _pack(ent_items, width, prefix="entities(mentions): ")

    rel_items = []
    for r in gc.get("relations") or []:
        top = r.get("top") or []
        pairs = r.get("pairs", 0)
        if top:
            a, b, _llr = top[0]
            rel_items.append(f"{a} -[{r['template']}]-> {b} pairs={pairs} corpus_n={r['n']}")
        else:
            rel_items.append(f"-[{r['template']}]-> pairs={pairs} corpus_n={r['n']}")
    rel_line = _pack(rel_items, width, prefix="relations: ", max_items=max_pairs)

    return "\n".join([header, terms_line, ent_line, rel_line])


def class_digest(cref, *, width=110, k_ent=None, k_rel=None) -> str:
    """Render `evidence.class_reference` output as a TOON-style digest for the
    P18 REFERENCE Classes panel: one header line, one line per entity class,
    one line per relation class. Pure, deterministic, byte-identical for
    identical input -- rendered through `digest_card`, which owns escaping
    and `<br>` (P11).

    Both lists empty -> header plus one dash line per side, never "".

    Spec: .spec/specs/graph-explorer/design.md 6.24 E15-E19 (E17 amendment), P18
    Task: playbook.md T72
    """
    entity_classes = cref.get("entity_classes") or []
    relation_classes = cref.get("relation_classes") or []

    lines = [f"classes . {len(entity_classes)} entity . {len(relation_classes)} relation"]

    if entity_classes:
        for c in entity_classes:
            prefix = f"E{c['class_id']} {c['label']} . {c['members']} members . mass {c['mass']}: "
            items = [f"{name} {cnt}" for name, cnt in (c.get("top") or [])]
            lines.append(_pack(items, width, prefix=prefix, max_items=k_ent))
    else:
        lines.append(_pack([], width))

    if relation_classes:
        for c in relation_classes:
            prefix = f"R {c['rel_class']} . {c['templates']} templates . n {c['mass']}: "
            items = [f"{template} {n}" for template, n in (c.get("top") or [])]
            lines.append(_pack(items, width, prefix=prefix, max_items=k_rel))
    else:
        lines.append(_pack([], width))

    return "\n".join(lines)


def digest_card(text: str, color: str, *, alpha: float = 0.10, badge=None,
                 badge_outline: bool = False) -> str:
    """Wrap one `group_digest` string as a colored HTML card (P14(b)/(e)).

    The card is background = `color` at `alpha`, a 4px solid left border in the
    full hue, a header row carrying a color chip + the digest's own id token, and
    a monospace body. The body text is `html.escape`d and otherwise UNTOUCHED:
    `strip_tags(card) == text` must hold byte for byte, because the same string is
    the artifact a downstream LLM pass consumes (P11(g) -- the human view and the
    machine view never fork).

    `badge`, when given, renders as a `pill` in the header row, right of the
    ident chip (P15(c)) -- solid in `color` unless `badge_outline`, which uses
    WARN to flag splits/merges divergence.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P14(b)/(e), P15(c)
    Task: playbook.md T47, T50
    """
    ident = (text.splitlines() or [""])[0].split(" ")[0]
    # Streamlit's markdown pass re-flows raw newlines even inside <pre>
    # (observed live, T48): explicit <br> is the only break it preserves.
    # strip_tags folds <br> back to \n so the byte-identity contract holds.
    body = html.escape(text).replace("\n", "<br>")
    badge_html = (pill(badge, WARN if badge_outline else color, outline=badge_outline)
                  if badge is not None else "")
    return (
        f'<div style="border:1px solid {BORDER};background:{rgba(color, alpha)};'
        f'border-left:4px solid {color};border-radius:4px;'
        f'padding:.5rem .7rem;margin:.35rem 0">'
        f'<div style="height:3px;border-radius:2px;background:{color};'
        f'margin:-.1rem 0 .4rem"></div>'
        f'<div style="font-weight:600;font-size:.86rem;margin-bottom:.3rem;'
        f'display:flex;justify-content:space-between;align-items:center">'
        f'<span><span style="display:inline-block;width:.7rem;height:.7rem;'
        f'border-radius:2px;background:{color};margin-right:.45rem;'
        f'vertical-align:middle"></span>{html.escape(ident)}</span>{badge_html}</div>'
        f'<pre style="margin:0;font-size:.78rem;line-height:1.35;'
        f'white-space:pre-wrap;overflow-x:auto">{body}</pre></div>'
    )


def dedup_groups(rel_rows, glob_rows) -> list:
    """Merge the relative and global `group_classes` panels into ONE
    annotated list the UI iterates once, instead of rendering the same
    community twice (P11(c)). A relative group whose member set equals a
    global one is a straight rename; one that partially overlaps several
    globals gets a decomposition marker naming which globals it merges or
    splits; any global left untouched still renders on its own.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P11(c)/(e)
    Task: playbook.md T43
    """
    gsets = {r["gid"]: set(r["members"]) for r in glob_rows}
    consumed = set()
    out = []

    for rel_row in rel_rows:
        S = set(rel_row["members"])
        equal_cid = None
        for cid, g in sorted(gsets.items()):
            if g == S:
                equal_cid = cid
                break
        if equal_cid is not None:
            consumed.add(equal_cid)
            out.append({"row": rel_row, "prefix": "g", "gid": rel_row["gid"],
                        "size": rel_row["size"], "kind": "merged",
                        "marker": f"= c{equal_cid} (global)",
                        "cids": [(equal_cid, len(S))]})
            continue

        inter = [(cid, len(S & g)) for cid, g in gsets.items() if S & g]
        inter.sort(key=lambda t: (-t[1], t[0]))
        partial = [cid for cid, n in inter if n < len(gsets[cid])]

        if partial:
            parts = " + ".join(f"c{cid}:{n}" for cid, n in inter)
            splits = ", ".join(f"c{c}" for c in partial)
            marker = f"= {parts} (splits {splits})"
        elif len(inter) > 1:
            marker = "= " + "+".join(f"c{cid}" for cid, _n in inter) + " (merges)"
        elif len(inter) == 1:
            marker = f"= c{inter[0][0]} (global)"
        else:
            marker = None

        out.append({"row": rel_row, "prefix": "g", "gid": rel_row["gid"],
                    "size": rel_row["size"], "kind": "local", "marker": marker,
                    "cids": inter})

    for glob_row in glob_rows:
        if glob_row["gid"] in consumed:
            continue
        out.append({"row": glob_row, "prefix": "c", "gid": glob_row["gid"],
                    "size": glob_row["size"], "kind": "global", "marker": None,
                    "cids": []})

    out.sort(key=lambda e: (-e["size"], e["prefix"], e["gid"]))
    return out


def card_color(entry) -> str:
    """The ONE hue for a dedup_groups row: a merged row (member set == a global
    cid) and a global row use their own cid; a local-only row borrows its
    DOMINANT overlapping cid's hue -- `cids` is already ordered by overlap
    descending, cid ascending. A local group overlapping nothing falls back to
    its own ephemeral gid, which is the figure's color for it anyway.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P14(b)
    Task: playbook.md T47
    """
    if entry["kind"] == "global":
        return cid_color(entry["gid"])
    cids = entry.get("cids") or []
    return cid_color(cids[0][0]) if cids else cid_color(entry["gid"])


def chain_communities(chains, cid_of, *, width=110) -> list:
    """One line per dendrite chunk chain, parallel to `partition_rows`' rows:
    `"c6:31 c0:4 c12:1"`, counts descending then cid ascending. An ord absent
    from `cid_of` (or mapped to None) counts under the `"c?"` bucket, sorted
    last regardless of its count.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P12
    Task: playbook.md T43
    """
    lines = []
    for chain in chains:
        counts = Counter()
        for o in chain:
            cid = cid_of.get(o)
            counts["c?" if cid is None else f"c{cid}"] += 1
        ordered = sorted(counts.items(),
                          key=lambda kv: (kv[0] == "c?", -kv[1], kv[0]))
        items = [f"{label}:{n}" for label, n in ordered]
        lines.append(_pack(items, width, sep=" "))
    return lines


# ---------- 3D scene builder (T53, design 6.22 P16(a)-(h) + (i) umap) ----------

WALK3D_FG_URL = "https://cdn.jsdelivr.net/npm/3d-force-graph@1.73.4/dist/3d-force-graph.min.js"
# three-spritetext's UMD build reads the global THREE, which 3d-force-graph
# bundles privately -- so plain three UMD loads FIRST (<= 0.159: later
# releases dropped the UMD build). Console receipt for this ordering:
# "Cannot read properties of undefined (reading 'LinearFilter')" (T54 live).
WALK3D_THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"
WALK3D_ST_URL = "https://cdn.jsdelivr.net/npm/three-spritetext@1.8.2/dist/three-spritetext.min.js"


def _walk3d_body_doc(entry):
    """Normalise a `bodies` value: either a bare string or a dict carrying
    'body'/'doc_id'. Missing body -> "", missing doc_id -> None."""
    if isinstance(entry, dict):
        return entry.get("body") or "", entry.get("doc_id")
    return entry or "", None


def walk3d_payload(ords, *, cid_of, src_of, scores, bodies, kw=None, salient=None,
                    pathways=None, edges=None, umap=None, tip_chars=200) -> dict:
    """Pure data builder for the 3D walk scene (P16(a)-(h), P16(i) umap
    addendum). Every human string that lands in the payload is escaped HERE,
    at build time -- `walk3d_html`'s JS only writes it back out (via
    innerHTML, because the tip already carries our own `<b>`/`<br>` chrome),
    so this function is the ONLY line of defence against injected HTML.

    Require: plain dicts/lists, no DB handle. Guarantee: pure, deterministic,
    JSON-serialisable, every string HTML-escaped. Maintain: positions are a
    VIEW (P16(g)) -- nothing here is persisted or used as a join key.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P16(a)-(h), P16(i)
    Task: playbook.md T53
    """
    ords = list(ords)
    if not ords:
        return {"nodes": [], "links": [], "sprites": [], "paths": [], "has_umap": False}

    kw = kw or {}
    salient = salient or {}
    pathways = pathways or {}
    edges = edges or []
    umap = umap or {}
    node_set = set(ords)

    all_scores = [scores.get(o, 0.0) for o in ords]
    lo, hi = min(all_scores), max(all_scores)

    has_umap = bool(umap) and all(o in umap for o in ords)
    umap_xyz = {}
    if has_umap:
        xs = [float(umap[o][0]) for o in ords]
        ys = [float(umap[o][1]) for o in ords]
        zs = [float(umap[o][2]) for o in ords]
        cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
        span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        scale = 200.0 / span if span > 1e-9 else 1.0
        for o, x, y, z in zip(ords, xs, ys, zs):
            umap_xyz[o] = [round((x - cx) * scale, 3), round((y - cy) * scale, 3),
                           round((z - cz) * scale, 3)]

    nodes = []
    for o in ords:
        cid = cid_of.get(o)
        s = scores.get(o, 0.0)
        size = 3.0 if hi == lo else 1.0 + 5.0 * (s - lo) / (hi - lo)
        body, doc_id = _walk3d_body_doc(bodies.get(o))
        source = src_of.get(o)

        line1 = f"<b>#{html.escape(str(o))}</b> · {html.escape(str(source) if source else 'unlabelled')}"
        if doc_id:
            line1 += f" · {html.escape(str(doc_id))}"
        cid_str = f"c{cid}" if cid is not None else "c?"
        line2 = f"{cid_str} · walk {s:.3f}"
        body_html = html.escape(clip(body, tip_chars)) if body else ""
        tip = line1 + "<br>" + line2 + "<br>" + body_html

        node = {"id": o, "cid": cid, "color": cid_color(cid), "size": round(size, 3),
                "tip": tip}
        if has_umap:
            node["umap"] = umap_xyz[o]
        nodes.append(node)

    seen_links = {}
    for e in edges:
        if isinstance(e, dict):
            a, b, w = e.get("src"), e.get("dst"), e.get("strength")
        else:
            a, b, w = e
        if a is None or b is None or a == b:
            continue
        if a not in node_set or b not in node_set:
            continue
        key = (a, b) if a < b else (b, a)
        seen_links.setdefault(key, float(w))
    links = [{"source": a, "target": b, "w": w}
             for (a, b), w in sorted(seen_links.items())]

    cid_members = {}
    for o in ords:
        c = cid_of.get(o)
        if c is not None:
            cid_members.setdefault(c, []).append(o)

    sprites = []
    for cid in sorted(cid_members):
        members = cid_members[cid]
        terms = list(kw.get(cid) or [])[:3]
        if not terms:
            seen_terms = set()
            for o in members:
                if len(terms) >= 3:
                    break
                for t in (salient.get(o, {}).get("top") or []):
                    if t not in seen_terms:
                        seen_terms.add(t)
                        terms.append(t)
                    if len(terms) >= 3:
                        break
        text = f"c{cid}: " + " ".join(terms) if terms else f"c{cid}"
        sprites.append({"cid": cid, "color": cid_color(cid),
                         "text": html.escape(text), "members": members})

    paths = []
    for p in (pathways.get("pairs") or []):
        path = [o for o in (p.get("path") or []) if o in node_set]
        if len(path) >= 2:
            paths.append(path)

    return {"nodes": nodes, "links": links, "sprites": sprites, "paths": paths,
            "has_umap": has_umap}


_WALK3D_TEMPLATE = string.Template("""<!doctype html>
<meta charset="utf-8">
<style>
html,body{margin:0;background:#0f1117;color:#e6e8ef;font:12px/1.4 system-ui}
#g{width:100%;height:${height}px}
#tip{position:absolute;display:none;pointer-events:none;background:${bg_card};
  border:1px solid ${border};border-radius:4px;max-width:320px;padding:.5rem;
  font-size:.78rem;z-index:10}
#ctl{position:absolute;top:.5rem;left:.5rem;z-index:10;display:flex;gap:.4rem}
#ctl button{background:${bg_card};border:1px solid ${border};color:${primary};
  border-radius:4px;padding:.25rem .55rem;font-size:.72rem;cursor:pointer}
#err{position:absolute;top:.5rem;left:.5rem;display:none;color:${muted};
  font-size:.8rem;z-index:10}
</style>
<div id="g"></div>
<div id="tip"></div>
<div id="ctl">
<button id="pathbtn" style="${hide_path}">paths: all</button>
<button id="umapbtn" style="${hide_umap}">layout: force</button>
</div>
<div id="err"></div>
<script src="${three_url}" onerror="window.__w3dfail=1"></script>
<script src="${fg_url}" onerror="window.__w3dfail=1"></script>
<script src="${st_url}" onerror="window.__w3dfail=1"></script>
<script id="w3d-data" type="application/json">${data_json}</script>
<script>
try {
function fail(m){var e=document.getElementById('err');e.style.display='block';
  e.textContent='3D scene unavailable: '+m;document.getElementById('g').style.display='none';}
if (window.__w3dfail || typeof ForceGraph3D==='undefined' || typeof SpriteText==='undefined'){
  fail('could not load the 3d-force-graph libraries (offline?)');
} else {
var D = JSON.parse(document.getElementById('w3d-data').textContent);
var nodeById = {};
D.nodes.forEach(function(n){ nodeById[n.id] = n; });

var pathSet = {};
var pathIdx = {};
D.paths.forEach(function(path, i){
  for (var k = 0; k < path.length - 1; k++){
    var a = path[k], b = path[k+1];
    var key = a < b ? (a + '|' + b) : (b + '|' + a);
    pathSet[key] = true;
    pathIdx[key] = i;
  }
});
function linkKey(l){
  var a = typeof l.source === 'object' ? l.source.id : l.source;
  var b = typeof l.target === 'object' ? l.target.id : l.target;
  return a < b ? (a + '|' + b) : (b + '|' + a);
}

var hotCid = null;
function rgba(hex, a){
  var h = hex.replace('#','');
  var r = parseInt(h.substring(0,2),16), g = parseInt(h.substring(2,4),16), b = parseInt(h.substring(4,6),16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
}
function col(n){ return (hotCid===null || n.cid===hotCid) ? n.color : rgba(n.color, 0.15); }
function lcol(l){
  var an = typeof l.source==='object' ? l.source : nodeById[l.source];
  var bn = typeof l.target==='object' ? l.target : nodeById[l.target];
  var base = pathSet[linkKey(l)] ? (an ? an.color : '#8b8f9e') : '#8b8f9e';
  var dim = hotCid!==null && !(an && an.cid===hotCid) && !(bn && bn.cid===hotCid);
  return dim ? rgba(base, 0.15) : base;
}
function lwidth(l){ return pathSet[linkKey(l)] ? 2.5 : 0.5; }

var pathMode = 'all';
var pathOne = 0;
var pathBtn = document.getElementById('pathbtn');
if (D.paths.length === 0){ pathBtn.style.display = 'none'; }
pathBtn.addEventListener('click', function(){
  if (pathMode === 'all'){ pathMode = D.paths.length ? 'one' : 'off'; pathOne = 0; }
  else if (pathMode === 'one'){ pathOne += 1; if (pathOne >= D.paths.length){ pathMode = 'off'; } }
  else { pathMode = 'all'; }
  pathSet = {}; pathIdx = {};
  var use = pathMode === 'all' ? D.paths : (pathMode === 'one' ? [D.paths[pathOne]] : []);
  use.forEach(function(path, i){
    for (var k = 0; k < path.length - 1; k++){
      var a = path[k], b = path[k+1];
      var key = a < b ? (a + '|' + b) : (b + '|' + a);
      pathSet[key] = true; pathIdx[key] = i;
    }
  });
  pathBtn.textContent = pathMode === 'all' ? 'paths: all' :
    (pathMode === 'one' ? ('paths: one ' + (pathOne+1) + '/' + D.paths.length) : 'paths: off');
  G.linkWidth(lwidth).linkColor(lcol);
});

var G = ForceGraph3D()(document.getElementById('g'))
  .backgroundColor('#0f1117')
  .graphData({
    nodes: D.nodes.map(function(n){ return Object.assign({}, n); }),
    links: D.links.map(function(l){ return {source: l.source, target: l.target, w: l.w}; })
  })
  .nodeVal(function(n){ return n.size; })
  .nodeColor(col)
  .nodeLabel(null)
  .nodeOpacity(0.95)
  .linkColor(lcol)
  .linkOpacity(0.12)
  .linkWidth(lwidth);

var tip = document.getElementById('tip');
document.addEventListener('mousemove', function(ev){
  tip.style.left = (ev.pageX + 12) + 'px';
  tip.style.top = (ev.pageY + 12) + 'px';
});
G.onNodeHover(function(n){
  hotCid = n ? n.cid : null;
  if (n){ tip.style.display = 'block'; tip.innerHTML = n.tip; }
  else { tip.style.display = 'none'; }
  G.nodeColor(col).linkColor(lcol);
  sprites.forEach(function(s){
    s.sprite.material.opacity = (hotCid===null || s.cid===hotCid) ? 1 : 0.15;
    s.sprite.material.transparent = true;
  });
});

var sprites = D.sprites.map(function(s){
  var sp = new SpriteText(s.text);
  sp.color = s.color;
  sp.textHeight = 27;
  sp.material.depthWrite = false;
  G.scene().add(sp);
  return {sprite: sp, cid: s.cid, members: s.members};
});
// The sim mutates the COPIES handed to graphData, never D.nodes -- so the
// centroid tick must read the live objects (found stuck at origin, T54 live).
var liveById = {};
G.graphData().nodes.forEach(function(n){ liveById[n.id] = n; });
G.onEngineTick(function(){
  sprites.forEach(function(s){
    var mx = 0, my = 0, mz = 0, n = 0;
    s.members.forEach(function(id){
      var nd = liveById[id];
      if (nd && typeof nd.x === 'number'){ mx += nd.x; my += nd.y; mz += nd.z; n += 1; }
    });
    if (n > 0){ s.sprite.position.set(mx/n, my/n, mz/n); }
  });
});

var umapBtn = document.getElementById('umapbtn');
if (!D.has_umap){ umapBtn.style.display = 'none'; }
var layout = 'force';
var savedForces = {};
umapBtn.addEventListener('click', function(){
  if (layout === 'force'){
    layout = 'umap';
    ['charge','link','center'].forEach(function(name){
      savedForces[name] = G.d3Force(name);
      G.d3Force(name, null);
    });
    var t0 = Date.now();
    (function animate(){
      var t = Math.min(1, (Date.now() - t0) / 600);
      var e = t*t*(3-2*t);
      D.nodes.forEach(function(n){
        var nd = liveById[n.id];
        if (!nd || !n.umap) return;
        var sx = nd.__sx===undefined ? nd.x : nd.__sx;
        var sy = nd.__sy===undefined ? nd.y : nd.__sy;
        var sz = nd.__sz===undefined ? nd.z : nd.__sz;
        nd.__sx = sx; nd.__sy = sy; nd.__sz = sz;
        nd.fx = sx + (n.umap[0]-sx)*e;
        nd.fy = sy + (n.umap[1]-sy)*e;
        nd.fz = sz + (n.umap[2]-sz)*e;
      });
      G.refresh();
      if (t < 1){ requestAnimationFrame(animate); }
    })();
    umapBtn.textContent = 'layout: umap';
  } else {
    layout = 'force';
    D.nodes.forEach(function(n){
      var nd = liveById[n.id];
      if (!nd) return;
      nd.fx = undefined; nd.fy = undefined; nd.fz = undefined;
      nd.__sx = undefined; nd.__sy = undefined; nd.__sz = undefined;
    });
    Object.keys(savedForces).forEach(function(name){ G.d3Force(name, savedForces[name]); });
    G.d3ReheatSimulation();
    umapBtn.textContent = 'layout: force';
  }
});

G.width(document.body.clientWidth).height(${height});
window.addEventListener('resize', function(){ G.width(document.body.clientWidth); });
}
} catch(e) { fail(e.message); }
</script>
""")


def walk3d_html(payload: dict, *, height: int = 700) -> str:
    """Render `walk3d_payload`'s dict as a standalone HTML document for
    `st.components.v1.html(...)` -- an iframe that renders the string
    VERBATIM as a real HTML document, so (unlike `digest_card`) newlines are
    legal and wanted here; do not cargo-cult `<br>` into this scene.

    The payload rides inside a `<script type="application/json">` block, not
    inline JS text, with the `</script>` guard applied to the dump -- that
    keeps a tip's literal "</script>" from ever closing the data tag early.
    CDN versions are pinned exactly (P16: "exact versions ... pinned by
    test"); both tags carry `onerror` so a blocked/offline CDN degrades to a
    one-line message instead of a blank iframe (P16(h)).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P16(a)-(h), P16(i)
    Task: playbook.md T53
    """
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return _WALK3D_TEMPLATE.substitute(
        height=height,
        bg_card=BG_CARD,
        border=BORDER,
        primary=PRIMARY,
        muted=MUTED,
        hide_path="display:none" if not payload.get("paths") else "",
        hide_umap="display:none" if not payload.get("has_umap") else "",
        three_url=WALK3D_THREE_URL,
        fg_url=WALK3D_FG_URL,
        st_url=WALK3D_ST_URL,
        data_json=data_json,
    )
