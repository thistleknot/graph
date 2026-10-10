"""section_graphrag.py -- hybrid GraphRAG over the section map in Postgres: a SUBGRAPH answer from the retrieved sections and a GLOBAL answer from community summaries, then one synthesis.

Spec: operator 2026-10-07 (Agentic GraphRAG ch.1: "pregenerate community summaries ... map-reduce"; "Hybrid GraphRAG: vector search, graph traversal, context
synthesis"; "leverage pgvector in docker"; approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md). Task: playbook.md T172. No other governing spec.

    STAGE     MECHANISM                                                                                                   GUARD
    SEED      pgvector exact cosine over the centred section vectors + Postgres full text over the body, fused by RRF k=60   G1
    EXPAND    one hop over sect_edge_sym from each seed, strongest edges first; neighbours ranked by RRF over seed lists     G2
    CANDIDATE communities of the seeds and of the subgraph, plus those whose summary vector is nearest the question          G3
    MAP       one LLM call per candidate community, over that community's summary alone; first line RELEVANT: YES|NO         G4
    REDUCE    one query-focused summary of the partial answers                                                               G4
    ANSWER    one call over (global answer, evidence): the typical section of each of the top 3 communities, then the 6          G5, G7
              subgraph sections nearest the question; every claim cites a section key
    TRAVERSE  the question's subgraph is HOPS hops out (PER_QUERY = 18 sections); a ReAct agent sees it and a Box-Cox-sampled map of the communities, and   G8, G9, G10, G11, G13
              makes up to AGENT_QUERIES more searches, each its own subgraph kept apart from the earlier ones (top-up or mask); the final interpretation reads all of them   G14
    STORE     src/section_store.py: sect_node / sect_edge / sect_community (python -u src\\section_store.py --tag xpa)

G1  The seed arms are the live service's (arxiv_rag_api.search) moved onto SECTIONS: the live build is chunks, this map is sections.
G6  WHEN the build holds the genre axes (section_store D7) the dense arm seeds from the vectors with them removed and projects the question the same way: hybrid recall@50 on the known-item
    battery .690 -> .748 (paired 30 against 7). The community-by-summary ranking stays in the original space (not measured with the axes removed).
G2  A neighbour is ranked by the weight of its edge (section_store D5): the significance the communities were built from, not a raw nearest-neighbour rank.
G3  A community of a single section is never a candidate, nor offered to the agent (operator 2026-10-08: "I thought we were going to drop communities that didn't have more than 1"; 408 of the 823
    communities of build 32, 408 of 80,642 sections, and none has a community holding more than half of its edge weight, so none can be folded into a neighbour). A genre-flagged community is a candidate unless `exclude_genre` (default OFF): the flag put "Long-term memory for AI agents" (score .537, bar .533) out and left the
    global answer empty on a question it answers. The map step's own RELEVANT: NO is the filter.
G4  The map step answers in a fixed form: first line `RELEVANT: YES|NO`, then the answer after a YES. A reply without that line is counted as `unparsed` and not used.
    A summary is the model's DRAFT (status 'draft'); a community without one is never a candidate. Nothing is truncated: a prompt that would not fit stops the call.
G7  The evidence is a CENTREPOINT (section_store D8) of each of the top CENTRE_K communities, ranked by how many retrieved + neighbour sections they hold, then the CLOSEST_K subgraph
    sections nearest the question: 8-9 sections instead of 14-18 (15,000 / 38,812 / 27,580 characters against 27,362 / 76,440 / 55,359 on three questions). Not measured: answer quality.
G8  (operator 2026-10-08: "it's not enough to simply retrieve a subgraph; show the subgraph to an llm react agent that can choose to iterate more"; 2026-10-09: "by default branch out 3 hops from
    whatever nodes our initial sub graph lands on ... a tunable parm, same as number of searches the react agent can do ... no more than 3 for either"; "this avoids letting the user specify more
    hops out") The agent never extends the graph by hops. HOPS (default 3) is a parameter of every query's walk (G13); AGENT_QUERIES (default 3) is how many searches the agent may make. Each of
    AGENT_QUERIES search rounds gives the agent up to MAX_READS reads and one SEARCH words (or ANSWER); a closing round lets it READ and then ANSWER; ANSWER at any point stops. The loop is pure:
    the model call and the database actions are injected.
G9  A reply that is not an action, a key outside the working set, a read already done or out of reads, a SEARCH with no words, or a round that never searches costs that attempt or chance and is told
    to the agent in its history; it is never corrected or retried for the agent. The final answer is G5's synthesis over the global view and the evidence: every section read first, then the standard evidence.
G11 (operator 2026-10-08: "the react agent decides its searches after seeing its query informed by the community AND seeing the subgraph pulled from its first query ... a few-shot example of what
    input -> query produced in terms of subgraph ... an informed second attempt ... followed by a potential third, each time a fresh proportional sample of communities") A SEARCH returns the whole
    subgraph of its query (G13), every section is tagged with its query, and the prompt groups the sections seen by query, headed with the words and the count, so each earlier query is a worked example
    of words -> subgraph. The community sample is redrawn each search round (G10). The question is query 1; the agent's searches are queries 2 and up.
G13 (operator 2026-10-09: "n+ queries, 1 per mutually exclusive subgraph ... if we do [re-select], we just mask/ignore those and continue to count until we get our magic number (I prefer top 13
    chunks)") A query's subgraph is its unmasked hybrid seeds (SEARCH_N) plus HOPS hops of vector neighbours, each hop at most HOP_CAP sections ranked by edge weight (`hop_pool`); the PER_QUERY (13)
    sections shown are the pool's nearest to the query by dense similarity. Observed 2026-10-09: the survey section arxiv/2605_12357#19 (a hop-1 neighbour) is 18th of the 48-section pool by similarity
    (the 13th is at .379, it is .357) and 16th by edge order, so at 13 sections it is not shown and the answer lost the three categories (externalised text, latent, parametric) the agent had drawn
    from it when the old subgraph showed 28 sections; at 18 sections either order shows it, so PER_QUERY is 18 (operator 2026-10-09: yesterday's answer "was sufficient"; its evidence held that section).
    18 is set from this one case and sits at its edge: not tuned. Two ways to keep a query's subgraph apart from earlier ones (Explorer.mode): 'topup' (default; operator 2026-10-09:
    "count how many nodes and vectors overlap with existing query, and simply gain that many more to 'hop' from ... Repeat for the 3rd query"): the query's own natural subgraph, minus what an
    earlier query showed, plus k more, k = the overlap, hopped out along vector edges from the whole natural subgraph (`top_up`); and 'mask': a section any earlier query showed is skipped at the
    seeds and at every hop, and the walk goes on past it. Neither is measured against the other. A pool that runs out gives fewer than PER_QUERY and says so; it is never padded. Not measured: whether hops beat the same-length hybrid list (the old edge-neighbour test, 255 against 197 of 400, could
    not see them: its truth was the source paper and edges exclude same-paper pairs).
G10 (operator 2026-10-08: "the global communities help inform the react agent ... randomly sample a few examples based on their box-cox relative proportions as ratios (weighted sum to 1) ... resample
    in between each opportunity ... and they have access to prior communities") Each round shows CATALOGUE_K community titles with sizes, drawn without replacement by weight Box-Cox(size) / sum
    (lambda fitted once on every size; a singleton weighs 0 and is never shown), or the largest in order when mode is 'top'. The earlier rounds' communities stay in the prompt, the new ones marked.
    The seed is the question's checksum and the round number, so a run is reproducible. Measured on build 32: the 12 largest communities hold 19.5% of the sections and 4.1% of the weight.
G14 (operator 2026-10-09: "the global summarized answer (over partials) is input to the subgraph interpretation for the final response") WHEN the agent ran, the final response interprets the WHOLE
    subgraph the traversal saw, in the light of the global answer: every section the agent read, the centrepoint of each top community, then every section of every query's subgraph nearest the
    question first (`subgraph_order`). A section any query showed is never left out; the characters per section shrink to fit ANSWER_BUDGET_CHARS (`answer_chars`) and every cut says so (G5). Before this
    the evidence came from the question's first retrieval alone, so a search of the agent's could not change the answer unless it read what it found.
G5  The final prompt carries each section whole up to `section_chars` and says so when it cuts; a cut is never silent. Sections are keyed doc_id#section_idx and the
    section header carries NO community number: with "community 12" in the header the model cited [c12] for sections (observed 2026-10-07, three times in three questions).
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import section_store as st

SEARCH_N, SNIPPET_CHARS, READ_CHARS, MAX_READS = 3, 160, 6000, 2                                    # G8, G13: seeds per query
HOPS, AGENT_QUERIES, PER_QUERY, HOP_CAP, HOP_NEIGHBOURS = 3, 3, 18, 15, 10                           # G8, G13: the two tunable depths (tried at 1, 2, 3), the sections shown per query (18: 13 dropped the survey section, see G13), the cap per hop
CATALOGUE_K = 12                                                                                     # G10
ANSWER_BUDGET_CHARS, MIN_SECTION_CHARS = 140000, 1200                                                # G14: the final prompt's evidence budget, about 35,000 tokens (the old evidence ran 15,000 to 38,812 characters; 60,000 cut the survey section at 1,200 of its 3,644 characters and lost two of its three categories)
ACTIONS, KEY_ACTIONS = ("READ", "SEARCH", "ANSWER"), ("READ",)
RRF_K, POOL, PER_SEED, MAX_CANDIDATES = 60, 100, 5, 12
CENTRE_K, CLOSEST_K, MIN_MEMBERS = 3, 6, 2      # G7: typical sections of the top 3 communities holding >= 2 subgraph sections, then the 6 subgraph sections nearest the question
MAP_PROMPT = """You answer ONE question using ONLY the summary of one cluster of sections from research papers.
Question: {question}

Cluster {cid} ({size} sections) - {title}
{summary}

Reply in exactly this form. The first line is RELEVANT: YES or RELEVANT: NO.
YES only when the summary itself states something that answers the question. NO when the summary is about something else, or says it does not cover the question.
After a YES, write 2-4 sentences answering the question from this summary only, with no outside knowledge. After a NO, write nothing more."""
REDUCE_PROMPT = """Question: {question}

Below are partial answers, each drawn from one cluster of sections of research papers. Write one answer to the question that combines them.
Keep the cluster tag, e.g. [c12], after each claim it supports. Say what the clusters leave unanswered. Use nothing but the partial answers.

{partials}"""
ANSWER_PROMPT = """Answer the question from two kinds of context. The GLOBAL view is a synthesis across communities of related sections and gives the framing.
The SUBGRAPH view is the evidence. It opens with the TYPICAL section of each of the communities that hold most of the retrieved subgraph (what that community is generally like; use it
as evidence only where it speaks to the question), then the sections of the subgraph NEAREST the question, nearest first.

Question: {question}

GLOBAL view:
{global_answer}

SUBGRAPH view (each section has a key in square brackets; after every claim you take from it, cite that key exactly as printed, the document id AND the # and section number, in square brackets, in the form [<document id>#<section number>]):
{sections}

Write ONE answer to the question, in at most four short paragraphs, for a reader who has not seen the context. Cite section keys only: the [c12]-style tags in the GLOBAL view are cluster
tags, never cite them and never invent new ones. Do not describe the views ("the GLOBAL view says") and print no heading or label; begin with the answer. The GLOBAL view frames the answer and can miss things: cover the distinct approaches or findings the sections report, including those it does not mention. Where the evidence does not support part of the question, say so in one
closing sentence. Use nothing but the two kinds of context."""
AGENT_PROMPT = """You search a graph of sections from research papers to answer a question. You choose the words of each new search, and you may stop as soon as you have enough.
Sections are nodes. A VECTOR edge joins two sections that read alike. Every search returns a subgraph: the sections best matching its words plus their neighbours out to hop {hops}, {per_query} in all. A section an earlier search already showed is never shown again, so a new search always returns new sections.

Question: {question}

Framing, a synthesis over summaries of related sections (it can miss the answer):
{global_answer}

Topics the corpus holds, a sample of its communities of sections: size, title, and the entities most characteristic of it with the number of its sections that mention each. Each round adds a new sample and the earlier ones stay. Use them to choose where to look next:
{communities}

Sections seen so far, grouped by the query that found them (query 1 is the question; a SEARCH of yours is the next query), one per line: [key] hop | topic of its community | similarity to that query | how it was reached | the first words. A * marks a section you have read in full. Each query shows what its words returned: use what an earlier query returned to word a better next one, and use the entities in the topics to choose words.
{working}

Sections you have read in full:
{read}

What you have done so far:
{history}

{round_text}
Reply in one of these forms.
  ACTION: READ <key>
  WHY: one sentence.
or
  ACTION: SEARCH <the words of a new search>
  WHY: one sentence.
or
  ACTION: ANSWER
  WHY: one sentence.
The first words of a section are not evidence: read a section before you rely on it. Replies left this round: {left}."""
ROUND_SEARCH = "You are in search round {round} of {queries}. You may READ up to {reads} sections first, then SEARCH once; the search runs and you see its sections next round. If what you have is already enough, ANSWER."
ROUND_CLOSE = "No more searches. You may READ up to {reads} more sections, then ANSWER."


def rrf(rank_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """G1. Guarantee: [(item, score)] by descending reciprocal-rank-fusion score, ties by item (arxiv_rag_api.rrf, kept here so the pure parts need no service import)."""
    score: dict[int, float] = defaultdict(float)
    for lst in rank_lists:
        for r, item in enumerate(lst):
            score[item] += 1.0 / (k + r + 1)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def expand(seeds: list[int], neigh: dict[int, list[dict]]) -> list[dict]:
    """G2. Require: neigh = section_store.neighbours' output, each seed's list strongest first and free of seeds. Guarantee: [{ord, score, weight, via}], best first:
    the neighbours ranked by RRF over the per-seed lists, `weight` the strongest edge to any seed, `via` the seeds that reach it in seed order."""
    weight, via = {}, defaultdict(list)
    for s in seeds:
        for n in neigh.get(s, []):
            weight[n["ord"]] = max(weight.get(n["ord"], -1.0), n["weight"])
            via[n["ord"]].append(s)
    return [{"ord": o, "score": sc, "weight": weight[o], "via": via[o]} for o, sc in rrf([[n["ord"] for n in neigh.get(s, [])] for s in seeds])]


def candidate_communities(lab_of: dict[int, int], genre: set, seeds: list[int], sub: list[int], by_summary: list[int], cap: int = MAX_CANDIDATES) -> list[int]:
    """G3. Guarantee: up to `cap` distinct communities, never one in `genre` (empty unless the caller excludes them): those of the seeds in seed order, then of the subgraph
    in rank order, then `by_summary`."""
    out = []
    for c in [lab_of[s] for s in seeds] + [lab_of[s] for s in sub] + by_summary:
        c = int(c)
        if c not in genre and c not in out:
            out.append(c)
    return out[:cap]


def top_communities(lab_of: dict[int, int], seeds: list[int], sub: list[int], k: int = CENTRE_K, min_members: int = MIN_MEMBERS) -> list[tuple[int, int]]:
    """G7. Guarantee: up to k (community, members in the subgraph) pairs, the communities holding the most of the retrieved sections plus their neighbours first; a tie goes to the community
    seen first in seed order, then in neighbour order. A community holding fewer than `min_members` of them is left out: its typical section is framing for a topic the question barely
    touched (observed 2026-10-07: a third slot went to a community with 1 of 14 sections), and when nothing reaches the minimum the single best community is kept."""
    count, first = defaultdict(int), {}
    for o in list(seeds) + list(sub):
        c = int(lab_of[o])
        count[c] += 1
        first.setdefault(c, len(first))
    ranked = sorted(count.items(), key=lambda kv: (-kv[1], first[kv[0]]))
    return ([kv for kv in ranked if kv[1] >= min_members] or ranked[:1])[:k]


def choose_sections(got_seeds: list[int], sub: list[dict], centre: dict[int, int], top: list[tuple[int, int]], sim: dict[int, float], closest_k: int = CLOSEST_K) -> list[dict]:
    """G7. Require: centre = {community: its centrepoint ord}, top = top_communities' output, sim = {ord: similarity to the question} for every subgraph section. Guarantee: the evidence list:
    first the centrepoint of each top community (role 'centre', with how many subgraph sections its community holds), then the `closest_k` subgraph sections nearest the question, nearest first,
    each marked 'retrieved' (a seed) or 'neighbour' (via the seeds that reach it), a centrepoint never repeated."""
    out, used = [], set()
    n_sub = len(got_seeds) + len(sub)
    for c, n in top:
        if c in centre and centre[c] not in used:
            used.add(centre[c])
            out.append({"ord": centre[c], "role": "centre", "holds": n, "of": n_sub})
    via = {s["ord"]: s["via"] for s in sub}
    pool = [(o, "retrieved") for o in got_seeds] + [(s["ord"], "neighbour") for s in sub]
    for o, role in sorted((p for p in pool if p[0] not in used), key=lambda p: -sim[p[0]])[:closest_k]:
        out.append({"ord": o, "role": role, "via": via.get(o, []), "sim": sim[o]})
    return out


def map_prompt(question: str, rec: dict) -> str:
    """G4. Guarantee: the map-step prompt for one community's summary record."""
    return MAP_PROMPT.format(question=question, cid=rec["community"], size=rec["size"], title=rec["title"], summary=rec["summary"])


def parse_map(reply: str) -> tuple[bool | None, str]:
    """G4. Guarantee: (relevant, answer text). The reply's FIRST line is read as `RELEVANT: YES` or `RELEVANT: NO` (any case); the answer is what follows a YES, '' after a NO.
    A reply with no such first line is (None, reply): the caller counts it and does not use it. Measured 2026-10-07: when the exit was a reply that STARTED with a phrase, a community
    whose reply opened with content and ended "this summary does not contain information about..." was counted as relevant."""
    first, _, rest = reply.strip().partition("\n")
    m = re.match(r"\s*RELEVANT\s*:\s*(YES|NO)\b", first, re.I)
    if not m:
        return None, reply
    return (True, rest.strip()) if m.group(1).upper() == "YES" else (False, "")


def reduce_prompt(question: str, partials: list[tuple[int, str]]) -> str:
    """G4. Guarantee: the reduce-step prompt; every partial answer carries its cluster tag [c<id>]."""
    return REDUCE_PROMPT.format(question=question, partials="\n\n".join("[c%d] %s" % (c, t.strip()) for c, t in partials))


def sections_block(items: list[dict], section_chars: int) -> str:
    """G5. Guarantee: the subgraph view as text, one block per section headed by its key; a cut section ends with a visible marker naming how much was left out."""
    blocks = []
    for it in items:
        t = it["text"]
        if len(t) > section_chars:
            t = t[:section_chars] + "\n[... cut: %d of %d characters shown]" % (section_chars, len(t))
        blocks.append("[%s] (%s)\n%s" % (it["key"], it["role"], t))
    return "\n\n".join(blocks)


def strip_cluster_tags(text: str) -> str:
    """G5. Guarantee: `text` without its cluster tags ([c12], [c12, c40]). The reduce step keeps them so a partial answer is traceable; the answer prompt must not show them, because a model
    shown a tag cites it (observed 2026-10-08: [c643], [c21], [c44] in 3 of 4 final answers, after telling it not to)."""
    return re.sub(r"[ \t]*\[c\d+(?:\s*,\s*c\d+)*\]", "", text)


def answer_prompt(question: str, global_answer: str, items: list[dict], section_chars: int = 6000) -> str:
    """G5. Guarantee: the final synthesis prompt over the global answer (cluster tags removed) and the subgraph sections."""
    return ANSWER_PROMPT.format(question=question, global_answer=strip_cluster_tags(global_answer).strip() or "(no community summary bore on the question)",
                                sections=sections_block(items, section_chars))


def parse_action(reply: str) -> tuple[str | None, str, str]:
    """G8, G9. Guarantee: (action, argument, why). The FIRST line must be `ACTION: <one of ACTIONS> [argument]` (any case); a key argument loses the brackets and quotes it was given in.
    `WHY:` is read from any later line, '' when absent. A reply whose first line is not an action is (None, reply, ''): the loop counts it as a lost step."""
    lines = [ln for ln in reply.strip().splitlines() if ln.strip()]
    m = re.match(r"\s*ACTION\s*:\s*([A-Za-z]+)\s*(.*)$", lines[0], re.I) if lines else None
    if not m or m.group(1).upper() not in ACTIONS:
        return None, reply, ""
    action, arg = m.group(1).upper(), m.group(2).strip()
    if action in KEY_ACTIONS:
        arg = arg.strip("[]<>`'\" ")
    why = next((re.sub(r"^\s*WHY\s*:\s*", "", ln, flags=re.I).strip() for ln in lines[1:] if re.match(r"\s*WHY\s*:", ln, re.I)), "")
    return action, arg, why


def working_block(nodes: dict[str, dict], read: dict[str, str]) -> str:
    """G8, G11. Guarantee: the sections seen, grouped by the query that found them (query 1 is the question), each group headed `QUERY n: "words" -> k sections` and each section one line in the
    order seen: `*` when read, [key], hop, community topic, similarity, how it was reached, the first words. A node without a `query` belongs to query 1."""
    groups: dict[int, list[tuple[str, dict]]] = {}
    for k, n in nodes.items():
        groups.setdefault(n.get("query", 1), []).append((k, n))
    out = []
    for q in sorted(groups):
        out.append('QUERY %d: "%s" -> %d sections' % (q, groups[q][0][1].get("query_text", ""), len(groups[q])))
        out += ["%s[%s]%s hop %d | %s | similarity %.2f | %s | %s" % ("*" if k in read else " ", k, ' "%s"' % n["paper"] if n.get("paper") else "", n["hop"], n["topic"], n["sim"], n["via"],
                                                                    n["snippet"]) for k, n in groups[q]]
    return "\n".join(out)


def read_block(read: dict[str, str]) -> str:
    """G8. Guarantee: each section read, whole up to READ_CHARS, a longer one ending with a visible marker naming how much was left out; '(none yet)' when nothing was read."""
    out = []
    for k, t in read.items():
        if len(t) > READ_CHARS:
            t = t[:READ_CHARS] + "\n[... cut: %d of %d characters shown]" % (READ_CHARS, len(t))
        out.append("[%s]\n%s" % (k, t))
    return "\n\n".join(out) or "(none yet)"


def hop_pool(seeds: list[int], neigh, hops: int, cap: int, masked: set[int]) -> tuple[list[dict], int]:
    """G13. Require: neigh(ords) -> {ord: [{ord, weight}]} strongest first (section_store.neighbours with room for the masked ones). Guarantee: ([{ord, hop, weight}], skipped): the sections
    within `hops` hops of `seeds` that are not seeds and not in `masked`, hop by hop, each hop at most `cap` of them ranked by `expand` (RRF over the per-section edge lists, strongest edge as the
    weight); the next hop spreads from the sections the last hop kept; a section is in the pool once, at the first hop that reached it. `skipped` counts the masked sections the walk met."""
    seen, frontier, pool, skipped = set(seeds), list(seeds), [], 0
    for hop in range(1, hops + 1):
        if not frontier:
            break
        neighbours = neigh(frontier)
        skipped += len({n["ord"] for lst in neighbours.values() for n in lst if n["ord"] in masked and n["ord"] not in seen})
        live = {s: [n for n in lst if n["ord"] not in masked and n["ord"] not in seen] for s, lst in neighbours.items()}
        kept = [e for e in expand(frontier, live) if e["ord"] not in seen][:cap]
        pool += [{"ord": e["ord"], "hop": hop, "weight": e["weight"]} for e in kept]
        seen |= {e["ord"] for e in kept}
        frontier = [e["ord"] for e in kept]
    return pool, skipped


def top_up(natural: list[int], occupied: set[int], neigh, size: int) -> tuple[list[int], dict[int, int], int]:
    """G13. Require: natural = a query's own unmasked subgraph, nearest first; neigh(ords) -> {ord: [{ord, weight}]} strongest first. Guarantee: (chosen, depth, k): `chosen` the sections of
    `natural` not in `occupied`, in order, then enough more to reach `size`, hopping out from the WHOLE natural subgraph (occupied sections included, so a query that overlaps entirely still
    has somewhere to hop from) along vector edges, `expand`'s order (RRF over the per-section edge lists), going deeper only while more are needed; `depth` = {added ord: hops it took}; `k` =
    how many of `natural` were occupied, the only number the rule needs (operator 2026-10-09: "count how many ... overlap ... and simply gain that many more to hop from")."""
    chosen = [o for o in natural if o not in occupied]
    k = len(natural) - len(chosen)
    taken, depth, frontier, d = set(natural) | set(occupied) | set(chosen), {}, list(natural), 0
    while len(chosen) < size and frontier:
        d += 1
        live = {s: [n for n in lst if n["ord"] not in taken] for s, lst in neigh(frontier).items()}
        got = [e["ord"] for e in expand(frontier, live) if e["ord"] not in taken][:size - len(chosen)]
        chosen += got
        depth |= {o: d for o in got}
        taken |= set(got)
        frontier = got
    return chosen, depth, k


def boxcox_weights(sizes: list[int]) -> np.ndarray:
    """G10. Guarantee: one weight per size, Box-Cox(size) / sum, summing to 1 (lambda fitted by maximum likelihood over all of `sizes`); a size of 1 weighs 0. All sizes equal: uniform."""
    from scipy.stats import boxcox
    n = np.asarray(sizes, float)
    if np.ptp(n) == 0:
        return np.full(len(n), 1.0 / len(n))
    y = boxcox(n)[0]
    return y / y.sum()


def sample_communities(catalogue: list[tuple[int, int, str]], k: int, seed: int, shown: set[int], mode: str = "sample") -> list[tuple[int, int, str]]:
    """G10. Require: catalogue = [(community, size, title)] largest first. Guarantee: up to k communities not in `shown`: mode 'sample' draws without replacement by Box-Cox weight (the weights
    are fitted on the WHOLE catalogue and renormalised over what is left; a community of weight 0 is never drawn; fewer than k left means all of them), seeded by `seed`; mode 'top' takes the
    largest. Shown in descending size."""
    w = boxcox_weights([c[1] for c in catalogue])
    left = [i for i, c in enumerate(catalogue) if c[0] not in shown and (mode == "top" or w[i] > 0)]
    if mode == "top":
        pick = left[:k]
    else:
        p = w[left] / w[left].sum() if left else w[left]
        pick = list(np.random.default_rng(seed).choice(left, size=min(k, len(left)), replace=False, p=p)) if left else []
    return sorted((catalogue[i] for i in pick), key=lambda c: -c[1])


def communities_block(batches: list[list[tuple[int, int, str]]], facts: dict[int, str] | None = None) -> str:
    """G10. Guarantee: the communities shown so far, earlier rounds first and the last round's marked (new); no community number is printed (the model cites what it is shown). A community
    with an entry in `facts` {community: 'entity sections . entity sections'} ends its line with it: the entities most characteristic of the community and how many of its sections mention each."""
    facts = facts or {}
    return "\n".join("%s%d sections | %s%s" % ("(new) " if b is batches[-1] and len(batches) > 1 else "      ", size, title, " | " + facts[cid] if cid in facts else "")
                     for b in batches for cid, size, title in b) or "(none)"


def agent_prompt(question: str, global_answer: str, nodes: dict[str, dict], read: dict[str, str], history: list[str], round_no: int, searching: bool, left: int,
                 batches: list | None = None, facts: dict[int, str] | None = None, queries: int = AGENT_QUERIES, hops: int = HOPS, per_query: int = PER_QUERY) -> str:
    """G8, G10, G13. Guarantee: the prompt of one agent turn. A search round (`searching`) lets the agent READ and then SEARCH; the closing round only READ and ANSWER."""
    round_text = ROUND_SEARCH.format(round=round_no, queries=queries, reads=MAX_READS) if searching else ROUND_CLOSE.format(reads=MAX_READS)
    if searching and left == 1:                                             # observed 2026-10-08: a third READ used up the round and the search was lost
        round_text += " THIS IS YOUR LAST REPLY THIS ROUND: reply SEARCH or ANSWER now; anything else loses the search for good."
    return AGENT_PROMPT.format(question=question, global_answer=strip_cluster_tags(global_answer).strip() or "(no community summary bore on the question)",
                               communities=communities_block(batches or [], facts), working=working_block(nodes, read), read=read_block(read), history="\n".join(history) or "(nothing yet)",
                               round_text=round_text, hops=hops, per_query=per_query, left=left)


def run_agent(question: str, global_answer: str, initial: list[dict], tools, ask, queries: int = AGENT_QUERIES, catalogue: list | None = None, k: int = CATALOGUE_K, seed: int = 0,
              mode: str = "sample", facts: dict[int, str] | None = None, hops: int = HOPS, per_query: int = PER_QUERY) -> dict:
    """G8, G9, G10, G11. Require: `initial` = node dicts {key, ord, topic, sim, via, snippet, hop, query, query_text} of the question's subgraph; `tools` has read(key) -> text and
    search(words, seen) -> node dicts for sections not in `seen` (a set of keys); `ask(prompt) -> reply text`; `catalogue` = [(community, size, title)] largest first. Guarantee: {nodes, read, trace,
    stop, shown}: every section seen, the text of each section read, one trace record per reply {round, action, arg, why, result, new}, the community batches shown, and stop 'answered' (the agent
    chose ANSWER) or 'done' (every round ended). There are `queries` search rounds and then one closing round; in each the agent has MAX_READS + 1 replies; a search round ends at its SEARCH, a new
    batch of communities is drawn before it (G10), and a round that never searches loses that search."""
    nodes = {n["key"]: dict(n) for n in initial}
    read: dict[str, str] = {}
    history: list[str] = []
    trace: list[dict] = []
    batches: list[list] = []
    shown: set[int] = set()
    for rnd in range(1, queries + 2):
        searching = rnd <= queries
        if catalogue and searching:
            batches.append(sample_communities(catalogue, k, seed + rnd, shown, mode))
            shown |= {c[0] for c in batches[-1]}
        searched, reads = False, 0
        for left in range(MAX_READS + 1, 0, -1):
            reply = ask(agent_prompt(question, global_answer, nodes, read, history, rnd, searching, left, batches, facts, queries, hops, per_query))
            action, arg, why = parse_action(reply)
            new: list[str] = []
            if action == "ANSWER":
                trace.append({"round": rnd, "action": action, "arg": "", "why": why, "result": "stopped", "new": []})
                return {"nodes": nodes, "read": read, "trace": trace, "stop": "answered", "shown": batches}
            if action is None:
                result = "your reply was not in one of the ACTION forms; nothing was done"
            elif action == "READ" and arg not in nodes:
                result = "[%s] is not one of the sections seen; nothing was done" % arg
            elif action == "READ" and arg in read:
                result = "you have already read [%s]; nothing was done" % arg
            elif action == "READ" and reads >= MAX_READS:
                result = "no reads left this round; nothing was done"
            elif action == "READ":
                read[arg] = tools.read(arg)
                reads += 1
                result = "read in full: %d characters" % len(read[arg])
            elif not searching:
                result = "there is no search left; nothing was done"
            elif not arg:
                result = "SEARCH had no words; nothing was done"
            else:
                for n in tools.search(arg, set(nodes)):
                    if n["key"] not in nodes:
                        nodes[n["key"]] = dict(n)
                        new.append(n["key"])
                result = ("search %d: added %d: %s" % (rnd + 1, len(new), ", ".join("[%s]" % x for x in new))) if new else "search %d: nothing new" % (rnd + 1)
                searched = True
            trace.append({"round": rnd, "action": action, "arg": arg if action in ("READ", "SEARCH") else "", "why": why, "result": result, "new": new})
            history.append("round %d: %s%s -> %s" % (rnd, action or "(no action)", " " + arg if action in ("READ", "SEARCH") else "", result))
            if searched:
                break
        if searching and not searched:
            history.append("round %d: no search was made" % rnd)
            trace.append({"round": rnd, "action": None, "arg": "", "why": "", "result": "no search: this round's search was lost", "new": []})
    return {"nodes": nodes, "read": read, "trace": trace, "stop": "done", "shown": batches}


class Explorer:
    """G8, G13. The agent's database actions over one retrieval `got` of a SectionGraph: it turns section ords into node dicts, remembers which key is which ord, and runs each query's masked walk."""

    def __init__(self, sg: "SectionGraph", got: dict, hops: int = HOPS, per_query: int = PER_QUERY, mode: str = "topup"):
        self.sg, self.got, self.ord, self.queries, self.hops, self.per_query, self.mode, self.stats = sg, got, {}, 1, hops, per_query, mode, {}

    def tag(self, nodes: list[dict], query: int, text: str) -> list[dict]:
        """G11. Guarantee: `nodes` each tagged with the query number and words that led to it."""
        return [{**n, "query": query, "query_text": text} for n in nodes]

    def nodes(self, ords: list[int], via: dict[int, str], qd=None, col: str | None = None) -> list[dict]:
        """Guarantee: a node dict per ord, in order: {key, ord, topic, sim, via, snippet}, `sim` the similarity to the query vector `qd` in `col` (the question's, unless given); the first words are cut at SNIPPET_CHARS with an ellipsis."""
        conn, build = self.sg.conn, self.sg.build
        rows = st.sections(conn, build, ords)
        sim = st.similarity(conn, build, ords, self.got["qd"] if qd is None else qd, col or self.got["col"])
        topics = st.summaries(conn, build, sorted({r["community"] for r in rows.values()}))
        papers = st.paper_titles(conn, sorted({r["doc_id"] for r in rows.values()}))                   # G12: the paper's title, so 'Conclusion' is a conclusion OF something
        out = []
        for o in ords:
            r = rows[o]
            key = "%s#%s" % (r["doc_id"], r["section_idx"])
            self.ord[key] = o
            body = " ".join(r["text"].split())
            out.append({"key": key, "ord": o, "topic": topics[r["community"]]["title"] if r["community"] in topics else "(no summary)", "sim": sim[o],
                        "via": via[o], "snippet": body[:SNIPPET_CHARS] + ("..." if len(body) > SNIPPET_CHARS else "")} | ({"paper": papers[r["doc_id"]]} if r["doc_id"] in papers else {}))
        return out

    def pick(self, words: str, masked: set[int], mode: str | None = None) -> tuple[list[int], dict[int, int], dict, dict]:
        """G13. Guarantee: (chosen, hop, stats, arms): up to `per_query` section ords not in `masked`; {ord: hop} (0 = a hybrid seed); {held, pool, masked}; SectionGraph.ranked's output. Mode
        'mask': the nearest to `words` by dense similarity among the SEARCH_N unmasked hybrid seeds and their neighbours out to `hops` hops (`hop_pool`), nearest first; `masked` in the stats counts
        the shown sections the walk met. Mode 'topup' (the default, operator 2026-10-09): the query's own natural subgraph (the 'mask' walk with nothing masked), minus what is shown, plus as many
        more as were shown, hopped out along vector edges (`top_up`); `masked` in the stats is that count, k."""
        conn, build = self.sg.conn, self.sg.build
        arms = self.sg.ranked(words)
        room = HOP_NEIGHBOURS + len(masked)
        if (mode or self.mode) == "topup":
            chosen, hop, stats, _ = self.pick(words, set(), "mask")
            chosen, depth, k = top_up(chosen, masked, lambda ords: st.neighbours(conn, build, ords, room), self.per_query)
            return chosen, hop | depth, {"held": len(chosen), "pool": stats["pool"], "masked": k}, arms
        seeds = [o for o in arms["ranked"] if o not in masked][:SEARCH_N]
        pool, skipped = hop_pool(seeds, lambda ords: st.neighbours(conn, build, ords, room), self.hops, HOP_CAP, masked)
        hop = {o: 0 for o in seeds} | {p["ord"]: p["hop"] for p in pool}
        sim = st.similarity(conn, build, list(hop), arms["qd"], arms["col"]) if hop else {}
        chosen = sorted(hop, key=lambda o: -sim[o])[:self.per_query]
        return chosen, hop, {"held": len(chosen), "pool": len(hop), "masked": skipped}, arms

    def walk(self, words: str, query: int, masked: set[int]) -> list[dict]:
        """G13. Guarantee: the subgraph of query number `query` as nodes (`pick`'s ords, each tagged with the query and its hop). self.stats[query] = {words, held, pool, masked}."""
        chosen, hop, stats, arms = self.pick(words, masked)
        via = {o: "found by searching '%s'" % words if hop[o] == 0 else "vector edge, hop %d from a section found by '%s'" % (hop[o], words) for o in chosen}
        self.stats[query] = {"words": words} | stats
        return self.tag([{**n, "hop": hop[n["ord"]]} for n in self.nodes(chosen, via, arms["qd"], arms["col"])], query, words)

    def initial(self) -> list[dict]:
        """G13. Guarantee: the question's own subgraph, query 1, nothing masked."""
        return self.walk(self.got["question"], 1, set())

    def read(self, key: str) -> str:
        return st.sections(self.sg.conn, self.sg.build, [self.ord[key]])[self.ord[key]]["text"]

    def search(self, words: str, seen: set[str]) -> list[dict]:
        """G11, G13. Guarantee: the subgraph of the agent's next query (the question is query 1), every section in `seen` masked."""
        self.queries += 1
        return self.walk(words, self.queries, {self.ord[k] for k in seen})


class SectionGraph:
    """The persisted section map made queryable. Reads sect_* rows only; embeds the question with the same model and centring mean the build used."""

    def __init__(self, conn, tag: str = "xpa", model_dir: str | None = None, exclude_genre: bool = False):
        live = st.live_build(conn, tag)
        assert live is not None, "no live sect_build for tag %r: run python -u src\\section_store.py --tag %s" % (tag, tag)
        self.conn, (self.build, self.params), self.exclude_genre = conn, live, exclude_genre
        self.mu = np.asarray(self.params["mu"], np.float32)
        from model2vec import StaticModel
        self.model = StaticModel.from_pretrained(model_dir or os.path.expanduser("~/models/m2v-jina-v5-nano-256"))
        self.genre = {r[0] for r in conn.execute("SELECT cid FROM sect_community WHERE build_id = %s AND genre_flag", (self.build,)).fetchall()} if exclude_genre else set()
        self.V = np.asarray(self.params["genre_axes"], np.float64) if self.params.get("genre_axes") else None       # G6: the genre axes the dense arm projects out, when the build holds them
        self.small = {r[0] for r in conn.execute("SELECT cid FROM sect_community WHERE build_id = %s AND size < 2", (self.build,)).fetchall()}      # G3: a community of one section is never a candidate (408 of 823 on the xpa map)

    def embed(self, text: str) -> np.ndarray:
        import section_embed as se
        return se.center(se.pool(self.model, [text]), self.mu)[0][0]

    def ranked(self, question: str) -> dict:
        """G1, G6, G13. Guarantee: {q, qd, col, arms, ranked}: the question's vector, the vector and column the dense arm uses, the two arms' POOL-long lists, and their RRF order (all ords, best first)."""
        q = self.embed(question)
        if self.V is None:
            qd, self.col = q, "emb"
        else:
            import section_genre as sgn
            qd, self.col = sgn.remove_axes(q[None, :], self.V)[0], "emb_b"
        arms = {"dense": st.seeds_dense(self.conn, self.build, qd, POOL, col=self.col), "lexical": st.seeds_lexical(self.conn, self.build, question, POOL)}
        return {"q": q, "qd": qd, "col": self.col, "arms": arms, "ranked": [o for o, _ in rrf(list(arms.values()))]}

    def retrieve(self, question: str, k: int = 3, per_seed: int = PER_SEED, m: int = MAX_CANDIDATES) -> dict:
        """Guarantee: {question, seeds, arms, subgraph, candidates} -- everything before the first LLM call."""
        r = self.ranked(question)
        q, qd, arms = r["q"], r["qd"], r["arms"]
        seeds = r["ranked"][:k]
        sub = expand(seeds, st.neighbours(self.conn, self.build, seeds, per_seed))
        lab_of = st.community_of(self.conn, self.build, seeds + [s["ord"] for s in sub])
        by_summary = st.communities_by_summary(self.conn, self.build, q, m, self.exclude_genre)
        return {"question": question, "seeds": seeds, "arms": arms, "subgraph": sub, "lab_of": lab_of, "qd": qd, "col": self.col,
                "candidates": candidate_communities(lab_of, self.genre | self.small, seeds, [s["ord"] for s in sub], by_summary, m)}

    def items(self, got: dict, centre_k: int = CENTRE_K, closest_k: int = CLOSEST_K) -> list[dict]:
        """G5, G7. Guarantee: the evidence list for the final step: the centrepoint of each of the top `centre_k` communities (by how many subgraph sections they hold), then the `closest_k`
        subgraph sections nearest the question, each with its key, community and a role line (the community is kept for display, never printed into the prompt)."""
        sub_ords = [s["ord"] for s in got["subgraph"]]
        top = top_communities(got["lab_of"], got["seeds"], sub_ords, centre_k)
        centre = st.centrepoints(self.conn, self.build, [c for c, _ in top], got["col"])
        sim = st.similarity(self.conn, self.build, got["seeds"] + sub_ords, got["qd"], got["col"])
        chosen = choose_sections(got["seeds"], got["subgraph"], centre, top, sim, closest_k)
        rows = st.sections(self.conn, self.build, [c["ord"] for c in chosen])
        key = lambda o: "%s#%s" % (rows[o]["doc_id"], rows[o]["section_idx"])
        out = []
        for c in chosen:
            if c["role"] == "centre":
                role = "typical section of a community that holds %d of the %d sections retrieved" % (c["holds"], c["of"])
            elif c["role"] == "retrieved":
                role = "retrieved, similarity %.2f" % c["sim"]
            else:
                role = "graph neighbour of a retrieved section, similarity %.2f" % c["sim"]
            out.append({"key": key(c["ord"]), "ord": c["ord"], "community": rows[c["ord"]]["community"], "role": role, "text": rows[c["ord"]]["text"]})
        return out


def global_view(sg: SectionGraph, got: dict, model: str) -> dict:
    """G4. Guarantee: {partials, not_relevant, unparsed, global_answer} -- the map step over every candidate community of `got`, then the reduce step. LLM calls: one per candidate, one reduce."""
    import summarize_clusters as scz
    summ = st.summaries(sg.conn, sg.build, got["candidates"])
    partials, skipped, unparsed = [], 0, 0
    for c in got["candidates"]:
        relevant, text = parse_map(scz.chat(model, map_prompt(got["question"], summ[c]))["content"])
        if relevant:
            partials.append((c, text))
        else:
            skipped += 1
            unparsed += relevant is None
    return {"partials": partials, "not_relevant": skipped, "unparsed": unparsed,
            "global_answer": scz.chat(model, reduce_prompt(got["question"], partials))["content"] if partials else ""}


def answer(sg: SectionGraph, question: str, k: int = 3, model: str | None = None, per_seed: int = PER_SEED, m: int = MAX_CANDIDATES) -> dict:
    """Guarantee: {..retrieve.., partials, not_relevant, global_answer, final}. LLM calls: one per candidate community, one reduce, one final (summarize_clusters.chat: never truncates)."""
    import summarize_clusters as scz
    model = model or scz.MODEL
    got = sg.retrieve(question, k, per_seed, m)
    glob = global_view(sg, got, model)
    items = sg.items(got)
    final = scz.chat(model, answer_prompt(question, glob["global_answer"], items))["content"]
    return {**got, **glob, "final": final, "evidence": [{"key": i["key"], "ord": i["ord"], "role": i["role"]} for i in items]}


def traversal_summary(run: dict, entities: list[tuple[str, int]], by_community: list[dict] | None = None, walks: dict[int, dict] | None = None, hops: int = HOPS,
                      per_query: int = PER_QUERY) -> dict:
    """G8, G13. Guarantee: {hops, size, per_hop, read, stop, entities, by_community, queries, walks, per_query}: the hops each query was asked to walk, the number of sections seen, how many each hop
    holds (index = hop; padded to `hops`), the number read, why the traversal ended, the entity composition of the whole subgraph (surface, sections mentioning it) and the same broken out by community
    (section_store.subgraph_entities_by_community), the number of queries run, each query's walk {query, words, held, pool, masked} (Explorer.stats, in query order) and the size asked of each.
    This is what the query strip prints."""
    per_hop = np.bincount([n["hop"] for n in run["nodes"].values()], minlength=hops + 1).tolist()
    return {"hops": hops, "size": len(run["nodes"]), "per_hop": per_hop, "read": len(run["read"]), "stop": run["stop"], "entities": entities, "by_community": by_community or [],
            "queries": len({n.get("query", 1) for n in run["nodes"].values()}), "walks": [{"query": q} | w for q, w in sorted((walks or {}).items())], "per_query": per_query}


def subgraph_order(nodes: dict[str, dict], read: dict[str, str], centre: dict[int, int], top: list[tuple[int, int]], sim: dict[int, float], centre_keys: dict[int, str] | None = None) -> list[dict]:
    """G14. Require: nodes = every section the traversal saw, {key: {ord, query, hop, ...}}; read = {key: text}; centre = {community: centrepoint ord}; top = top_communities' output over all of
    them; sim = {ord: similarity to the QUESTION}; centre_keys = {ord: key} for a centrepoint the traversal did not see. Guarantee: [{key, ord, role}] -- the evidence of the final interpretation, in
    this order: every section the agent read (it chose them), the centrepoint of each top community not already in the list (seen or not: a community's typical section is framing, and
    a centrepoint outside the subgraph was dropped once, 2026-10-09, for 2 of 3 top communities), then every other section of every query's subgraph, nearest the question first (ties by key).
    Nothing the traversal saw is left out."""
    key_of = {n["ord"]: k for k, n in nodes.items()} | (centre_keys or {})
    out = [{"key": k, "ord": nodes[k]["ord"], "role": "read by the agent"} for k in read]
    used = {o["ord"] for o in out}
    for c, n in top:
        if c in centre and centre[c] not in used and centre[c] in key_of:
            used.add(centre[c])
            out.append({"key": key_of[centre[c]], "ord": centre[c], "role": "typical section of a community that holds %d of the %d sections seen" % (n, len(nodes))})
    rest = sorted((k for k, n in nodes.items() if n["ord"] not in used), key=lambda k: (-sim[nodes[k]["ord"]], k))
    return out + [{"key": k, "ord": nodes[k]["ord"], "role": "query %d, hop %d, similarity %.2f" % (nodes[k].get("query", 1), nodes[k]["hop"], sim[nodes[k]["ord"]])} for k in rest]


def answer_chars(n_sections: int, budget: int = ANSWER_BUDGET_CHARS) -> int:
    """G14. Guarantee: the characters each section may take in the final prompt: READ_CHARS (6000) while the whole fits `budget`, else the budget shared evenly, never below MIN_SECTION_CHARS; the
    prompt says so on every section it cuts (G5), so the cut is never silent."""
    return max(MIN_SECTION_CHARS, min(READ_CHARS, budget // max(n_sections, 1)))


def agent_answer(sg: SectionGraph, question: str, k: int = 3, model: str | None = None, per_seed: int = PER_SEED, m: int = MAX_CANDIDATES, agent_model: str | None = None,
                 catalogue_mode: str = "sample", hops: int = HOPS, queries: int = AGENT_QUERIES, per_query: int = PER_QUERY) -> dict:
    """G8, G9, G10, G13. Guarantee: answer()'s fields plus `agent` = run_agent's {nodes, read, trace, stop, shown} and `traversal` = traversal_summary. The question's own subgraph is `hops` hops out and
    `per_query` sections; the agent (`agent_model`, default `model`) sees it and the global view, sees a Box-Cox sample of communities each round (`catalogue_mode` 'top' shows the largest instead,
    'none' shows none), may make up to `queries` more searches, each its own masked subgraph, and may stop at any reply; the final answer is G5's synthesis over the global view and the evidence:
    FIRST every section the agent read (role 'read by the agent': it chose them), then the standard evidence it did not read (observed 2026-10-08 with the reads last: two speculative-decoding
    sections the agent found were cited by neither of the answer's two paragraphs on that topic)."""
    import zlib
    import summarize_clusters as scz
    model = model or scz.MODEL
    agent_model = agent_model or model
    got = sg.retrieve(question, k, per_seed, m)
    glob = global_view(sg, got, model)
    ex = Explorer(sg, got, hops, per_query)
    run = run_agent(question, glob["global_answer"], ex.initial(), ex, lambda p: scz.chat(agent_model, p)["content"], queries,
                    catalogue=None if catalogue_mode == "none" else st.community_catalogue(sg.conn, sg.build), seed=zlib.crc32(question.encode()), mode=catalogue_mode,
                    facts=st.community_entity_lines(sg.conn, sg.build), hops=hops, per_query=per_query)
    seen = [n["ord"] for n in run["nodes"].values()]
    traversal = traversal_summary(run, st.subgraph_entities(sg.conn, sg.build, seen), st.subgraph_entities_by_community(sg.conn, sg.build, seen), ex.stats, hops, per_query)
    lab = st.community_of(sg.conn, sg.build, seen)
    top = top_communities(lab, [], seen, CENTRE_K)
    centre = st.centrepoints(sg.conn, sg.build, [c for c, _ in top], got["col"])
    sim = st.similarity(sg.conn, sg.build, seen, got["qd"], got["col"])
    out_of = [o for o in centre.values() if o not in set(seen)]
    away = st.sections(sg.conn, sg.build, out_of) if out_of else {}
    order = subgraph_order(run["nodes"], run["read"], centre, top, sim | {o: 0.0 for o in out_of}, {o: "%s#%s" % (r["doc_id"], r["section_idx"]) for o, r in away.items()})
    text = st.sections(sg.conn, sg.build, [o["ord"] for o in order if o["role"] != "read by the agent"])
    lab |= {o: r["community"] for o, r in away.items()}
    items = [{"key": o["key"], "ord": o["ord"], "community": lab[o["ord"]], "role": o["role"],
              "text": run["read"][o["key"]] if o["role"] == "read by the agent" else text[o["ord"]]["text"]} for o in order]
    final = scz.chat(model, answer_prompt(question, glob["global_answer"], items, answer_chars(len(items))))["content"]
    return {**got, **glob, "final": final, "agent": run, "traversal": traversal, "evidence": [{"key": i["key"], "ord": i["ord"], "role": i["role"]} for i in items]}
