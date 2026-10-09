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
    TRAVERSE  a ReAct agent sees the subgraph (hops 0-2) and a Box-Cox-sampled map of the communities, plans hop 3, then hops 4-5, and can stop   G8, G9, G10
    STORE     tools/section_store.py: sect_node / sect_edge / sect_community (python -u tools\\section_store.py --tag xpa)

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
G8  (operator 2026-10-08: "it's not enough to simply retrieve a subgraph; show the subgraph to an llm react agent that can choose to iterate more"; "up to 3 to 5 hops, but at hop 2 the agent has
    to plan its next hop and at hop 3 its next 2 hops, so it only gets two chances to plan"; "and can stop early") Hops count outward from the retrieved sections: hop 0 the seeds, hop 1 their
    vector neighbours, hop 2 an entity hop from those (both automatic). PLAN_ROUNDS = ((3,), (4, 5)): round 1 plans hop 3, round 2 plans hops 4 and 5 TOGETHER, so hop 5 runs without the agent
    having seen hop 4. A hop is up to MAX_MOVES moves: EXPAND key (vector edges), ENTITY key (entity bridges, shared entities named), SEARCH words (a new hybrid retrieval); a key-less EXPAND or
    ENTITY spreads from the PLAN_FRONTIER sections nearest the question among those the previous hop added. A move adds at most EXPAND_N sections and a hop at most MAX_NEW_PER_HOP. Before a plan
    the agent may READ up to MAX_READS sections; a closing round lets it READ and then ANSWER; ANSWER at any point stops the hops. The loop is pure: the model call and the database actions are injected.
G9  A reply that is not an action, a key outside the working set, a read already done or out of reads, a move with no valid source, or a round that never makes its plan costs that attempt, move or
    chance and is told to the agent in its history; it is never corrected or retried for the agent. The final answer is G5's synthesis over the global view and the evidence: every section read first,
    then the standard evidence.
G11 (operator 2026-10-08: "the react agent decides its searches after seeing its query informed by the community AND seeing the subgraph pulled from its first query ... a few-shot example of what
    input -> query produced in terms of subgraph ... an informed second attempt ... followed by a potential third, each time a fresh proportional sample of communities") A SEARCH move returns
    the whole subgraph of its query (top sections plus their strongest vector neighbours), every section is tagged with its query (a section reached by EXPAND or ENTITY inherits the query of the
    section it spread from), and the prompt groups the sections seen by query, headed with the words and the count, so each earlier query is a worked example of words -> subgraph. The
    community sample is redrawn each planning round (G10).
G10 (operator 2026-10-08: "the global communities help inform the react agent ... randomly sample a few examples based on their box-cox relative proportions as ratios (weighted sum to 1) ... resample
    in between each opportunity ... and they have access to prior communities") Each round shows CATALOGUE_K community titles with sizes, drawn without replacement by weight Box-Cox(size) / sum
    (lambda fitted once on every size; a singleton weighs 0 and is never shown), or the largest in order when mode is 'top'. The earlier rounds' communities stay in the prompt, the new ones marked.
    The seed is the question's checksum and the round number, so a run is reproducible. Measured on build 32: the 12 largest communities hold 19.5% of the sections and 4.1% of the weight.
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
sys.path.insert(0, os.path.join(ROOT, "tools"))

import section_store as st

EXPAND_N, SEARCH_N, SNIPPET_CHARS, READ_CHARS = 5, 3, 160, 6000                                     # G8
PLAN_ROUNDS = ((3,), (4, 5))                                                                         # G8: round 1 plans hop 3, round 2 plans hops 4 and 5 together
MAX_MOVES, MAX_READS, PLAN_FRONTIER, AUTO_HOP_N, MAX_NEW_PER_HOP = 3, 2, 3, 10, 15                   # G8
CATALOGUE_K = 12                                                                                     # G10
ACTIONS, KEY_ACTIONS, MOVES = ("READ", "PLAN", "ANSWER"), ("READ",), ("EXPAND", "ENTITY", "SEARCH")
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

SUBGRAPH view (each section has a key in square brackets; after every claim you take from it, cite that key exactly, e.g. [arxiv/2401_12345#7]):
{sections}

Write ONE answer to the question, in at most four short paragraphs, for a reader who has not seen the context. Cite section keys only: the [c12]-style tags in the GLOBAL view are cluster
tags, never cite them and never invent new ones. Do not describe the views ("the GLOBAL view says"); just answer. Where the evidence does not support part of the question, say so in one
closing sentence. Use nothing but the two kinds of context."""
AGENT_PROMPT = """You explore a graph of sections from research papers to answer a question. You decide how the search spreads, and you may stop as soon as you have enough.
Sections are nodes. A VECTOR edge joins two sections that read alike. An ENTITY bridge joins two sections that mention the same named things.
Hops count outward from the retrieved sections: hop 0 is what the search retrieved, hop 1 their strongest vector neighbours, hop 2 sections that share entities with those, then the hops you plan.

Question: {question}

Framing, a synthesis over summaries of related sections (it can miss the answer):
{global_answer}

Topics the corpus holds, a sample of its communities of sections: size, title, and the entities most characteristic of it with the number of its sections that mention each. Each round adds a new sample and the earlier ones stay. Use them to choose where to look next:
{communities}

Sections seen so far, grouped by the query that found them (query 1 is the question; a SEARCH of yours is the next query), one per line: [key] hop | topic of its community | similarity to the question | how you reached it | the first words. A * marks a section you have read in full. Each query shows what those words returned: use what an earlier query returned to word a better next one.
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
  ACTION: PLAN
  {hop_lines}
  WHY: one sentence.
or
  ACTION: ANSWER
  WHY: one sentence.
A move is EXPAND <key> (vector edges of that section), ENTITY <key> (sections sharing entities with it) or SEARCH <words> (a new search); put up to {max_moves} moves on a hop line, separated by |.
EXPAND or ENTITY with no key spreads from the {frontier} sections nearest the question among those the previous hop added: use that for a hop that comes after a hop you cannot see yet.
Each move adds at most {expand_n} sections. The first words of a section are not evidence: read a section before you rely on it. Replies left this round: {left}."""
ROUND_PLAN = "You are in planning round {round}. {unseen}Plan {hops} now. You get only two chances to plan, and a hop you plan runs without you; you may READ up to {reads} sections first. If what you have read is already enough, ANSWER."
ROUND_CLOSE = "No more hops. You may READ up to {reads} more sections, then ANSWER."


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


def parse_plan(reply: str) -> dict[int, list[tuple[str, str]]]:
    """G8. Guarantee: {hop: [(verb, argument)]} from the reply's `HOP <n>: EXPAND <key> | ENTITY <key> | SEARCH <words>` lines (any case). A key loses the brackets and quotes it was given in;
    EXPAND and ENTITY may have no key; a part that is not a move is dropped; a hop line with no move is an empty list."""
    plan: dict[int, list[tuple[str, str]]] = {}
    for ln in reply.splitlines():
        m = re.match(r"\s*HOP\s+(\d+)\s*:\s*(.*)$", ln, re.I)
        if not m:
            continue
        moves = []
        for part in m.group(2).split("|"):
            mv = re.match(r"\s*(EXPAND|ENTITY|SEARCH)\b\s*(.*)$", part.strip(), re.I)
            if mv:
                verb, arg = mv.group(1).upper(), mv.group(2).strip()
                moves.append((verb, arg if verb == "SEARCH" else arg.strip("[]<>`'\" ")))
        plan[int(m.group(1))] = moves
    return plan


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


def agent_prompt(question: str, global_answer: str, nodes: dict[str, dict], read: dict[str, str], history: list[str], round_no: int, hops: tuple, left: int,
                 batches: list | None = None, facts: dict[int, str] | None = None) -> str:
    """G8, G10. Guarantee: the prompt of one agent turn. A round with `hops` is a planning round, one without is the closing round."""
    unseen = "Hop %d will run without you seeing hop %d. " % (hops[1], hops[0]) if len(hops) > 1 else ""
    round_text = ROUND_PLAN.format(round=round_no, unseen=unseen, hops=" and ".join("hop %d" % h for h in hops), reads=MAX_READS) if hops else ROUND_CLOSE.format(reads=MAX_READS)
    if hops and left == 1:                                                  # observed 2026-10-08: the third reply was a READ and the planning chance for hop 3 was lost
        round_text += " THIS IS YOUR LAST REPLY THIS ROUND: reply PLAN or ANSWER now; anything else loses the plan for good."
    return AGENT_PROMPT.format(question=question, global_answer=strip_cluster_tags(global_answer).strip() or "(no community summary bore on the question)",
                               communities=communities_block(batches or [], facts), working=working_block(nodes, read), read=read_block(read), history="\n".join(history) or "(nothing yet)",
                               round_text=round_text, hop_lines="\n  ".join("HOP %d: <up to %d moves separated by |>" % (h, MAX_MOVES) for h in hops) or "(no hop to plan)",
                               max_moves=MAX_MOVES, frontier=PLAN_FRONTIER, expand_n=EXPAND_N, left=left)


def apply_hop(hop: int, moves: list[tuple[str, str]], nodes: dict[str, dict], tools) -> tuple[str, list[str]]:
    """G8, G9. Guarantee: (what happened, the keys added). Each of the first MAX_MOVES moves runs against the sections seen so far; every section it adds is tagged with `hop`, at most
    MAX_NEW_PER_HOP in all. A keyed EXPAND or ENTITY needs a key already seen; a key-less one spreads from the PLAN_FRONTIER sections nearest the question among those the previous hop added.
    A move that cannot run is named in the result and costs nothing else."""
    frontier = sorted((n for n in nodes.values() if n["hop"] == hop - 1), key=lambda n: -n["sim"])[:PLAN_FRONTIER]
    new, notes = [], []
    for verb, arg in moves[:MAX_MOVES]:
        if verb == "SEARCH":
            runs = [(verb, arg)] if arg else []
        elif arg:
            runs = [(verb, arg)] if arg in nodes else []
        else:
            runs = [(verb, n["key"]) for n in frontier]
        if not runs:
            notes.append("%s %s: no source (%s)" % (verb, arg, "no words" if verb == "SEARCH" else "[%s] is not a section seen" % arg if arg else "the previous hop added nothing"))
        for v, a in runs:
            for n in {"EXPAND": tools.expand, "ENTITY": tools.entity, "SEARCH": tools.search}[v](a, set(nodes)):
                if n["key"] not in nodes and len(new) < MAX_NEW_PER_HOP:
                    nodes[n["key"]] = {**n, "hop": hop}
                    new.append(n["key"])
    head = "added %d: %s" % (len(new), ", ".join("[%s]" % k for k in new)) if new else ("nothing new" if moves else "no move given")
    return head + ("".join(" (%s)" % x for x in notes)), new


def run_agent(question: str, global_answer: str, initial: list[dict], tools, ask, rounds: tuple = PLAN_ROUNDS, catalogue: list | None = None, k: int = CATALOGUE_K, seed: int = 0,
              mode: str = "sample", facts: dict[int, str] | None = None) -> dict:
    """G8, G9, G10. Require: `initial` = node dicts {key, ord, topic, sim, via, snippet, hop} for hops 0-2; `tools` has read(key) -> text and expand(key, seen), entity(key, seen), search(words, seen)
    -> node dicts for sections not in `seen` (a set of keys); `ask(prompt) -> reply text`; `catalogue` = [(community, size, title)] largest first. Guarantee: {nodes, read, trace, stop, shown}:
    every section seen (each with its hop), the text of each section read, one trace record per reply {round, action, arg, why, result, new}, the community batches shown, and stop 'answered'
    (the agent chose ANSWER) or 'done' (every round ended). Each round in `rounds` is a planning round, then one closing round with no hop; in each round the agent has MAX_READS + 1 replies, a
    new batch of communities is drawn before it (G10), and a round that never makes its plan loses that chance."""
    nodes = {n["key"]: dict(n) for n in initial}
    read: dict[str, str] = {}
    history: list[str] = []
    trace: list[dict] = []
    batches: list[list] = []
    shown: set[int] = set()
    for rnd, hops in enumerate(list(rounds) + [()], 1):
        if catalogue and hops:
            batches.append(sample_communities(catalogue, k, seed + rnd, shown, mode))
            shown |= {c[0] for c in batches[-1]}
        planned, reads = False, 0
        for left in range(MAX_READS + 1, 0, -1):
            reply = ask(agent_prompt(question, global_answer, nodes, read, history, rnd, hops, left, batches, facts))
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
            elif action == "PLAN" and not hops:
                result = "there is no hop to plan now; nothing was done"
            else:
                plan, parts = parse_plan(reply), []
                for h in hops:
                    res, got = apply_hop(h, plan.get(h, []), nodes, tools)
                    new += got
                    parts.append("hop %d: %s" % (h, res))
                result, planned = "; ".join(parts), True
            trace.append({"round": rnd, "action": action, "arg": arg if action == "READ" else "", "why": why, "result": result, "new": new})
            history.append("round %d: %s%s -> %s" % (rnd, action or "(no action)", " " + arg if action == "READ" else "", result))
            if planned:
                break
        if hops and not planned:
            history.append("round %d: no plan was made, so hop %s did not run" % (rnd, " and ".join(str(h) for h in hops)))
            trace.append({"round": rnd, "action": None, "arg": "", "why": "", "result": "no plan: hop %s did not run" % " and ".join(str(h) for h in hops), "new": []})
    return {"nodes": nodes, "read": read, "trace": trace, "stop": "done", "shown": batches}


class Explorer:
    """G8. The agent's database actions over one retrieval `got` of a SectionGraph: it turns section ords into node dicts and remembers which key is which ord."""

    def __init__(self, sg: "SectionGraph", got: dict):
        self.sg, self.got, self.ord, self.queries, self.qof = sg, got, {}, 1, {}

    def tag(self, nodes: list[dict], query: int, text: str) -> list[dict]:
        """G11. Guarantee: `nodes` each tagged with the query number and words that led to it, and remembered so a section reached from it later inherits the same query."""
        out = [{**n, "query": query, "query_text": text} for n in nodes]
        self.qof.update({n["key"]: (query, text) for n in out})
        return out

    def nodes(self, ords: list[int], via: dict[int, str]) -> list[dict]:
        """Guarantee: a node dict per ord, in order: {key, ord, topic, sim, via, snippet}; the first words are cut at SNIPPET_CHARS with an ellipsis."""
        conn, build = self.sg.conn, self.sg.build
        rows = st.sections(conn, build, ords)
        sim = st.similarity(conn, build, ords, self.got["qd"], self.got["col"])
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

    def initial(self) -> list[dict]:
        """G8. Guarantee: hops 0 to 2 as nodes: the seeds (hop 0), their vector neighbours (hop 1), then up to AUTO_HOP_N sections that share the rarest entities with any of those (hop 2)."""
        ords = self.got["seeds"] + [s["ord"] for s in self.got["subgraph"]]
        via = {o: "retrieved" for o in self.got["seeds"]} | {s["ord"]: "vector edge from a retrieved section" for s in self.got["subgraph"]}
        out = [{**n, "hop": 0 if n["ord"] in self.got["seeds"] else 1} for n in self.nodes(ords, via)]
        hop = st.entity_hop(self.sg.conn, self.sg.build, ords, ords, 100, AUTO_HOP_N)
        out += [{**n, "hop": 2} for n in self.nodes([h["ord"] for h in hop], {h["ord"]: "entity bridge from the retrieved subgraph, %d shared entities" % h["shared"] for h in hop})]
        return self.tag(out, 1, self.got["question"])

    def read(self, key: str) -> str:
        return st.sections(self.sg.conn, self.sg.build, [self.ord[key]])[self.ord[key]]["text"]

    def expand(self, key: str, seen: set[str]) -> list[dict]:
        """Guarantee: up to EXPAND_N sections not in `seen` along the strongest vector edges of `key`."""
        o, taken = self.ord[key], {self.ord[k] for k in seen}
        nb = [n for n in st.neighbours(self.sg.conn, self.sg.build, [o], 6 * EXPAND_N)[o] if n["ord"] not in taken][:EXPAND_N]
        return self.tag(self.nodes([n["ord"] for n in nb], {n["ord"]: "vector edge from [%s], weight %.1f" % (key, n["weight"]) for n in nb}), *self.qof[key])

    def entity(self, key: str, seen: set[str]) -> list[dict]:
        """Guarantee: up to EXPAND_N sections not in `seen` that share the rarest entities with `key`, the shared entities named in `via`."""
        o, taken = self.ord[key], [self.ord[k] for k in seen]
        hop = st.entity_hop(self.sg.conn, self.sg.build, [o], taken, 100, EXPAND_N)
        shared = st.shared_entities(self.sg.conn, self.sg.build, o, [h["ord"] for h in hop])
        return self.tag(self.nodes([h["ord"] for h in hop], {h["ord"]: "entity bridge from [%s] via %s" % (key, ", ".join(shared.get(h["ord"], ["?"]))) for h in hop}), *self.qof[key])

    def search(self, words: str, seen: set[str]) -> list[dict]:
        """G11. Guarantee: the SUBGRAPH of a new query: up to SEARCH_N top hybrid-search sections for `words` and up to 2 x EXPAND_N of their strongest vector neighbours, none in `seen`, each
        tagged with the number and the words of the query (the question is query 1)."""
        self.queries += 1
        got = self.sg.retrieve(words, SEARCH_N)
        taken = {self.ord[k] for k in seen}
        seeds = [o for o in got["seeds"] if o not in taken]
        nb = [s["ord"] for s in got["subgraph"] if s["ord"] not in taken and s["ord"] not in seeds][:2 * EXPAND_N]
        via = {o: "found by searching '%s'" % words for o in seeds} | {o: "vector edge from a section found by '%s'" % words for o in nb}
        return self.tag(self.nodes(seeds + nb, via), self.queries, words)


class SectionGraph:
    """The persisted section map made queryable. Reads sect_* rows only; embeds the question with the same model and centring mean the build used."""

    def __init__(self, conn, tag: str = "xpa", model_dir: str | None = None, exclude_genre: bool = False):
        live = st.live_build(conn, tag)
        assert live is not None, "no live sect_build for tag %r: run python -u tools\\section_store.py --tag %s" % (tag, tag)
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

    def retrieve(self, question: str, k: int = 3, per_seed: int = PER_SEED, m: int = MAX_CANDIDATES) -> dict:
        """Guarantee: {question, seeds, arms, subgraph, candidates} -- everything before the first LLM call."""
        q = self.embed(question)
        if self.V is None:
            qd, self.col = q, "emb"
        else:
            import section_genre as sgn
            qd, self.col = sgn.remove_axes(q[None, :], self.V)[0], "emb_b"
        dense = st.seeds_dense(self.conn, self.build, qd, POOL, col=self.col)
        arms = {"dense": dense, "lexical": st.seeds_lexical(self.conn, self.build, question, POOL)}
        seeds = [o for o, _ in rrf(list(arms.values()))[:k]]
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


def traversal_summary(run: dict, entities: list[tuple[str, int]], by_community: list[dict] | None = None) -> dict:
    """G8. Guarantee: {hops, size, per_hop, read, stop, entities, by_community}: the deepest hop that holds a section, the number of sections seen, how many each hop holds (index = hop), the
    number read, why the traversal ended, the entity composition of the whole subgraph (surface, sections mentioning it) and the same broken out by community
    (section_store.subgraph_entities_by_community). This is what the query strip prints."""
    per_hop = np.bincount([n["hop"] for n in run["nodes"].values()]).tolist()
    return {"hops": len(per_hop) - 1, "size": len(run["nodes"]), "per_hop": per_hop, "read": len(run["read"]), "stop": run["stop"], "entities": entities, "by_community": by_community or [],
            "queries": len({n.get("query", 1) for n in run["nodes"].values()})}


def agent_answer(sg: SectionGraph, question: str, k: int = 3, model: str | None = None, per_seed: int = PER_SEED, m: int = MAX_CANDIDATES, agent_model: str | None = None,
                 catalogue_mode: str = "sample") -> dict:
    """G8, G9, G10. Guarantee: answer()'s fields plus `agent` = run_agent's {nodes, read, trace, stop, shown} and `traversal` = traversal_summary. The agent (`agent_model`, default `model`) starts
    from hops 0-2 and the global view, sees a Box-Cox sample of communities each round (`catalogue_mode` 'top' shows the largest instead, 'none' shows none), and may stop at any reply; the final answer is G5's
    synthesis over the global view and the evidence: FIRST every section the agent read (role 'read by the agent': it chose them), then the standard evidence it did not read (observed
    2026-10-08 with the reads last: two speculative-decoding sections the agent found were cited by neither of the answer's two paragraphs on that topic)."""
    import zlib
    import summarize_clusters as scz
    model = model or scz.MODEL
    agent_model = agent_model or model
    got = sg.retrieve(question, k, per_seed, m)
    glob = global_view(sg, got, model)
    ex = Explorer(sg, got)
    run = run_agent(question, glob["global_answer"], ex.initial(), ex, lambda p: scz.chat(agent_model, p)["content"],
                    catalogue=None if catalogue_mode == "none" else st.community_catalogue(sg.conn, sg.build), seed=zlib.crc32(question.encode()), mode=catalogue_mode,
                    facts=st.community_entity_lines(sg.conn, sg.build))
    seen = [n["ord"] for n in run["nodes"].values()]
    traversal = traversal_summary(run, st.subgraph_entities(sg.conn, sg.build, seen), st.subgraph_entities_by_community(sg.conn, sg.build, seen))
    comm = st.community_of(sg.conn, sg.build, [ex.ord[key] for key in run["read"]])
    items = [{"key": key, "ord": ex.ord[key], "community": comm[ex.ord[key]], "role": "read by the agent", "text": text} for key, text in run["read"].items()]
    items += [i for i in sg.items(got) if i["key"] not in run["read"]]
    final = scz.chat(model, answer_prompt(question, glob["global_answer"], items))["content"]
    return {**got, **glob, "final": final, "agent": run, "traversal": traversal, "evidence": [{"key": i["key"], "ord": i["ord"], "role": i["role"]} for i in items]}
