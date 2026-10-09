"""Pins src/section_graphrag.py guards G1-G5 on hand-built neighbour lists and prompts. No model, no network, no database (the SQL side is pinned in test_section_store.py).

Run:  pytest tests/test_section_graphrag.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import section_graphrag as g


def nb(*pairs):
    return [{"ord": o, "weight": w} for o, w in pairs]


def test_rrf_item_in_two_lists_beats_item_in_one():
    assert g.rrf([[7, 8], [9, 7]])[0][0] == 7


def test_expand_ranks_by_rrf_over_seed_lists_and_keeps_the_strongest_weight_and_the_seeds_that_reach_it():
    neigh = {0: nb((1, 3.0), (2, 1.0)), 4: nb((2, 2.0), (5, 1.5))}
    got = g.expand([0, 4], neigh)
    assert [e["ord"] for e in got] == [2, 1, 5]                           # 2 is in both lists, then 1 (rank 1 of seed 0), then 5 (rank 2 of seed 4)
    two = got[0]
    assert two["via"] == [0, 4] and two["weight"] == 2.0                  # the stronger of its two edges
    assert got[1]["via"] == [0] and got[2]["via"] == [4]


def test_expand_with_no_edges_is_empty():
    assert g.expand([3], {3: []}) == [] and g.expand([3], {}) == []


def test_candidates_drop_genre_keep_seed_order_then_subgraph_then_summary_and_cap():
    lab_of = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    got = g.candidate_communities(lab_of, {1}, seeds=[0, 2], sub=[3, 4], by_summary=[4, 5, 3], cap=10)
    assert got == [0, 2, 3, 4, 5]                  # community 1 is out, and 4 and 3 are not repeated
    assert g.candidate_communities(lab_of, set(), [0], [2], [5], cap=2) == [0, 1]
    assert g.candidate_communities(lab_of, {1}, [2], [], [], cap=5) == []      # a genre-only world gives no candidate when the caller excludes them


def test_top_communities_rank_by_subgraph_members_then_first_seen():
    lab_of = {0: 5, 1: 5, 2: 7, 3: 7, 4: 7, 5: 9, 6: 3}
    assert g.top_communities(lab_of, [0, 2], [1, 3, 4, 5, 6], k=3, min_members=1) == [(7, 3), (5, 2), (9, 1)]       # 7 holds three, 5 two, then 9 (seen before 3)
    assert g.top_communities(lab_of, [0, 2], [1, 3, 4, 5, 6], k=3) == [(7, 3), (5, 2)]                                # 9 and 3 hold one each: below the default minimum
    assert g.top_communities(lab_of, [5, 6], [], k=3, min_members=1) == [(9, 1), (3, 1)]              # equal counts: seed order decides
    assert g.top_communities(lab_of, [0], [], k=3) == [(5, 1)]                                         # nothing reaches the minimum: the single best community stays


def test_top_communities_leave_out_a_community_holding_fewer_than_the_minimum_of_the_subgraph():
    lab_of = {0: 5, 1: 5, 2: 5, 3: 7, 4: 9}
    assert g.top_communities(lab_of, [0, 3], [1, 2, 4], k=3) == [(5, 3)]                               # 7 and 9 hold 1 each: out
    assert g.top_communities(lab_of, [0, 3], [1, 2, 4], k=3, min_members=1) == [(5, 3), (7, 1), (9, 1)]


def test_choose_sections_puts_centrepoints_first_then_the_closest_by_similarity_never_repeating_a_centrepoint():
    sub = [{"ord": 10, "via": [1]}, {"ord": 11, "via": [2]}, {"ord": 12, "via": [1, 2]}]
    sim = {1: 0.50, 2: 0.80, 10: 0.70, 11: 0.20, 12: 0.60}
    got = g.choose_sections([1, 2], sub, centre={5: 99, 7: 2}, top=[(5, 3), (7, 2)], sim=sim, closest_k=3)
    assert [(c["ord"], c["role"]) for c in got] == [(99, "centre"), (2, "centre"), (10, "neighbour"), (12, "neighbour"), (1, "retrieved")]
    assert got[0]["holds"] == 3 and got[0]["of"] == 5
    assert [c["ord"] for c in got[2:]] == [10, 12, 1]                    # 0.70, 0.60, 0.50; section 2 (0.80) is a centrepoint already, section 11 (0.20) is cut by closest_k


def test_choose_sections_skips_a_community_without_a_centrepoint_and_handles_no_subgraph():
    got = g.choose_sections([1], [], centre={}, top=[(5, 1)], sim={1: 0.4}, closest_k=2)
    assert [(c["ord"], c["role"]) for c in got] == [(1, "retrieved")]


def test_parse_map_reads_the_first_line_and_only_a_yes_carries_an_answer():
    assert g.parse_map("RELEVANT: YES\nIt drafts tokens.\nThen verifies.") == (True, "It drafts tokens.\nThen verifies.")
    assert g.parse_map("  relevant : no\nThis summary is about something else.") == (False, "")
    assert g.parse_map("Relevant: Yes - extra words\nBody") == (True, "Body")


def test_parse_map_does_not_take_a_disclaimer_buried_in_content_for_relevance():
    leak = "The cluster covers RLHF methods.\n\nThis summary does not contain information about long-term memory."
    assert g.parse_map(leak) == (None, leak)                             # no header: unparsed, never counted relevant
    assert g.parse_map("RELEVANT: NO\nThis summary does not contain information about memory.")[0] is False
    assert g.parse_map("")[0] is None


def test_prompts_carry_the_cluster_tags_and_the_question():
    rec = {"community": 12, "size": 40, "title": "Speculative decoding", "summary": "Drafts tokens with a small model."}
    p = g.map_prompt("how to speed up decoding", rec)
    assert "how to speed up decoding" in p and "Cluster 12 (40 sections)" in p and "RELEVANT: YES or RELEVANT: NO" in p
    r = g.reduce_prompt("q", [(12, "a one"), (7, "a two ")])
    assert "[c12] a one" in r and "[c7] a two" in r


def test_sections_block_cuts_visibly_never_silently_and_prints_no_community_number():
    items = [{"key": "2401.1#3", "community": 5, "role": "retrieved", "text": "x" * 50},
             {"key": "2401.2#1", "community": 6, "role": "neighbour", "text": "short"}]
    out = g.sections_block(items, section_chars=20)
    assert "[2401.1#3] (retrieved)" in out and "community" not in out
    assert "[... cut: 20 of 50 characters shown]" in out
    assert "short" in out and out.count("[... cut") == 1


def test_answer_prompt_says_so_when_no_summary_bore_on_the_question():
    p = g.answer_prompt("q", "  ", [{"key": "k#1", "community": 1, "role": "retrieved", "text": "body"}])
    assert "(no community summary bore on the question)" in p and "[k#1]" in p


def test_cluster_tags_never_reach_the_answer_or_agent_prompt_but_section_keys_do():
    glob = "Memory is stored as episodes [c21]. Retrieval can be active [c21, c44] or passive [arxiv/2308_11432#3]."
    assert g.strip_cluster_tags(glob) == "Memory is stored as episodes. Retrieval can be active or passive [arxiv/2308_11432#3]."
    item = [{"key": "k#1", "community": 1, "role": "retrieved", "text": "body"}]
    node = {"key": "k#1", "topic": "t", "sim": 0.1, "via": "retrieved", "snippet": "s", "hop": 0}
    for p in (g.answer_prompt("q", glob, item), g.agent_prompt("q", glob, {"k#1": node}, {}, [], 1, (3,), 3)):
        assert "[c21" not in p and "[c44" not in p and "Memory is stored as episodes." in p and "[arxiv/2308_11432#3]" in p


def test_parse_action_reads_the_first_line_strips_key_brackets_and_finds_the_why():
    assert g.parse_action("ACTION: READ [arxiv/2401_1#7]\nWHY: the method is described here") == ("READ", "arxiv/2401_1#7", "the method is described here")
    assert g.parse_action("action: read `a#1`\n\nwhy: more like it") == ("READ", "a#1", "more like it")
    assert g.parse_action("ACTION: PLAN\nHOP 3: EXPAND a#1\nWHY: spread") == ("PLAN", "", "spread")
    assert g.parse_action("ACTION: ANSWER") == ("ANSWER", "", "")


def test_parse_action_rejects_anything_not_an_action_on_the_first_line():
    for bad in ["I will read the first one.\nACTION: READ a#1", "ACTION: DELETE a#1", "ACTION: EXPAND a#1", "READ a#1", ""]:      # a move is only ever part of a plan
        assert g.parse_action(bad) == (None, bad, "")


def test_parse_plan_reads_hop_lines_moves_and_keyless_spreads():
    plan = g.parse_plan("ACTION: PLAN\nhop 4: EXPAND [a#1] | ENTITY `b#2` | search kv cache words | nonsense\nHOP 5: EXPAND | ENTITY\nHOP 6:\nWHY: x")
    assert plan == {4: [("EXPAND", "a#1"), ("ENTITY", "b#2"), ("SEARCH", "kv cache words")], 5: [("EXPAND", ""), ("ENTITY", "")], 6: []}
    assert g.parse_plan("ACTION: READ a#1") == {}


def test_boxcox_weights_sum_to_one_give_singletons_nothing_and_flatten_the_giants():
    sizes = [1630, 1369, 336, 63, 50, 30, 15, 11, 4, 2, 2, 1, 1]
    w = g.boxcox_weights(sizes)
    assert abs(w.sum() - 1) < 1e-9 and w[-1] == 0 and w[-2] == 0 and (w[:-2] > 0).all()
    assert w[0] / w[3] < sizes[0] / sizes[3] / 3                                                  # the largest outweighs a 63-section community by far less than its raw 26x
    assert np.allclose(g.boxcox_weights([7, 7, 7]), 1 / 3)


CAT = [(10, 1630, "Agent runtimes"), (11, 1369, "Experimental details"), (12, 336, "Uncertainty"), (13, 63, "Tool schemas"), (14, 50, "Training details"), (15, 30, "Math"),
       (16, 15, "Oxygen vacancy"), (17, 11, "Medicine"), (18, 2, "Tiny a"), (19, 1, "Singleton a"), (20, 1, "Singleton b")]


def test_sample_communities_draws_without_replacement_by_weight_never_a_singleton_and_is_seeded():
    a = g.sample_communities(CAT, 4, seed=7, shown=set())
    assert a == g.sample_communities(CAT, 4, seed=7, shown=set()) and len({c[0] for c in a}) == 4 and [c[1] for c in a] == sorted((c[1] for c in a), reverse=True)
    assert not {19, 20} & {c[0] for c in a}
    b = g.sample_communities(CAT, 4, seed=8, shown={c[0] for c in a})                              # the next round: none of the first round's again
    assert not {c[0] for c in a} & {c[0] for c in b}
    assert len(g.sample_communities(CAT, 50, seed=1, shown=set())) == 9                           # 11 communities, 2 singletons never drawn
    assert g.sample_communities(CAT, 3, seed=1, shown=set(), mode="top") == CAT[:3]
    assert g.sample_communities(CAT, 3, seed=1, shown={10, 11}, mode="top") == CAT[2:5]


def test_communities_block_keeps_earlier_rounds_and_marks_the_newest():
    assert g.communities_block([]) == "(none)"
    one = [(10, 1630, "Agent runtimes")]
    two = [(12, 336, "Uncertainty")]
    assert g.communities_block([one]) == "      1630 sections | Agent runtimes"
    assert g.communities_block([one, two]).splitlines() == ["      1630 sections | Agent runtimes", "(new) 336 sections | Uncertainty"]
    facts = {10: "harness 512 · verifier 169", 99: "unused 1"}
    assert g.communities_block([one, two], facts).splitlines() == ["      1630 sections | Agent runtimes | harness 512 · verifier 169", "(new) 336 sections | Uncertainty"]      # a community with no facts has none

    node = {"key": "a#1", "hop": 0, "topic": "t", "sim": 0.5, "via": "retrieved", "snippet": "s"}
    p = g.agent_prompt("q", "", {"a#1": node}, {}, [], 1, (3,), 3, [one], facts)
    assert "1630 sections | Agent runtimes | harness 512 · verifier 169" in p and "the entities most characteristic of it" in p


def test_working_and_read_blocks_mark_what_was_read_and_cut_a_long_read_visibly():
    nodes = {"a#1": {"key": "a#1", "hop": 0, "topic": "Topic A", "sim": 0.5, "via": "retrieved", "snippet": "first words"},
             "b#2": {"key": "b#2", "hop": 2, "topic": "(no summary)", "sim": 0.25, "via": "vector edge from [a#1], weight 3.0", "snippet": "other"}}
    w = g.working_block(nodes, {"a#1": "x"}).splitlines()
    assert w[0] == 'QUERY 1: "" -> 2 sections' and w[1] == "*[a#1] hop 0 | Topic A | similarity 0.50 | retrieved | first words" and w[2].startswith(" [b#2] hop 2 | (no summary) | similarity 0.25 | vector edge from [a#1]")


def test_explorer_tags_nodes_with_their_query_and_remembers_it_for_sections_that_spread_from_them():
    ex = g.Explorer(None, {"question": "q"})
    out = ex.tag([{"key": "a#1", "ord": 1}, {"key": "b#2", "ord": 2}], 2, "kv cache")
    assert out == [{"key": "a#1", "ord": 1, "query": 2, "query_text": "kv cache"}, {"key": "b#2", "ord": 2, "query": 2, "query_text": "kv cache"}]
    assert ex.qof == {"a#1": (2, "kv cache"), "b#2": (2, "kv cache")}
    assert ex.tag([{"key": "c#3"}], *ex.qof["a#1"])[0]["query"] == 2 and ex.queries == 1             # a section reached from a#1 joins query 2; the counter is only advanced by a SEARCH


def test_a_section_line_names_its_paper_when_the_title_is_known_G12():
    nodes = {"a#1": {"key": "a#1", "hop": 0, "topic": "t", "sim": 0.5, "via": "retrieved", "snippet": "s", "paper": "A Survey of Speculative Decoding"},
             "b#2": {"key": "b#2", "hop": 1, "topic": "t", "sim": 0.4, "via": "v", "snippet": "s"}}
    lines = g.working_block(nodes, {"a#1": "x"}).splitlines()
    assert lines[1] == '*[a#1] "A Survey of Speculative Decoding" hop 0 | t | similarity 0.50 | retrieved | s' and lines[2] == " [b#2] hop 1 | t | similarity 0.40 | v | s"


def test_working_block_groups_sections_by_the_query_that_found_them_in_query_order():
    nodes = {"a#1": {"key": "a#1", "hop": 0, "topic": "t", "sim": 0.5, "via": "retrieved", "snippet": "s", "query": 1, "query_text": "how does it work"},
             "e#5": {"key": "e#5", "hop": 3, "topic": "t", "sim": 0.2, "via": "found by searching 'kv cache'", "snippet": "s", "query": 2, "query_text": "kv cache"},
             "b#2": {"key": "b#2", "hop": 1, "topic": "t", "sim": 0.6, "via": "v", "snippet": "s", "query": 1, "query_text": "how does it work"},
             "f#6": {"key": "f#6", "hop": 3, "topic": "t", "sim": 0.1, "via": "vector edge from a section found by 'kv cache'", "snippet": "s", "query": 2, "query_text": "kv cache"}}
    heads = [ln for ln in g.working_block(nodes, {}).splitlines() if ln.startswith("QUERY")]
    assert heads == ['QUERY 1: "how does it work" -> 2 sections', 'QUERY 2: "kv cache" -> 2 sections']
    lines = g.working_block(nodes, {}).splitlines()
    assert [ln.split("]")[0].strip(" *[") for ln in lines if not ln.startswith("QUERY")] == ["a#1", "b#2", "e#5", "f#6"]      # the second query's sections follow the first's, each group in the order seen
    assert g.read_block({}) == "(none yet)"
    r = g.read_block({"a#1": "y" * (g.READ_CHARS + 5)})
    assert r.startswith("[a#1]\n") and "[... cut: %d of %d characters shown]" % (g.READ_CHARS, g.READ_CHARS + 5) in r


class FakeTools:
    """Four database actions over a four-section world: a#1 (seed) links to b#2 and c#3, b#2 to d#4."""
    EDGES = {"a#1": ["b#2", "c#3"], "b#2": ["d#4", "a#1"]}

    def __init__(self):
        self.calls = []

    SIM = {"a#1": 0.9, "b#2": 0.6, "c#3": 0.3, "d#4": 0.5, "e#5": 0.2}

    def node(self, key, via, hop=None):
        n = {"key": key, "ord": ord(key[0]), "topic": "t", "sim": self.SIM[key], "via": via, "snippet": "s"}
        return n if hop is None else n | {"hop": hop}

    def read(self, key):
        self.calls.append(("read", key))
        return "full text of " + key

    def expand(self, key, seen):
        self.calls.append(("expand", key))
        return [self.node(k, "edge from " + key) for k in self.EDGES.get(key, []) if k not in seen]

    def entity(self, key, seen):
        self.calls.append(("entity", key))
        return []

    def search(self, words, seen):
        self.calls.append(("search", words))
        return [self.node("e#5", "found by searching '%s'" % words) | {"query": 2, "query_text": words}] if "e#5" not in seen else []


def scripted(*replies):
    it = iter(replies)
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return next(it)
    ask.prompts = prompts
    return ask


def start(tools):
    return [tools.node("a#1", "retrieved", 0), tools.node("b#2", "vector edge from a retrieved section", 1)]


def test_agent_plans_hop_3_then_hops_4_and_5_together_reads_between_and_stops_in_the_closing_round():
    tools = FakeTools()
    ask = scripted("ACTION: READ a#1\nWHY: the seed", "ACTION: PLAN\nHOP 3: EXPAND a#1\nWHY: spread",
                   "ACTION: PLAN\nHOP 4: EXPAND b#2\nHOP 5: EXPAND | SEARCH new words\nWHY: more",
                   "ACTION: READ d#4\nWHY: the new one", "ACTION: ANSWER\nWHY: enough")
    run = g.run_agent("q?", "global", start(tools), tools, ask, catalogue=CAT, seed=3)
    assert run["stop"] == "answered" and {k: n["hop"] for k, n in run["nodes"].items()} == {"a#1": 0, "b#2": 1, "c#3": 3, "d#4": 4, "e#5": 5}
    assert list(run["read"]) == ["a#1", "d#4"]
    assert [t["result"] for t in run["trace"]] == ["read in full: 16 characters", "hop 3: added 1: [c#3]", "hop 4: added 1: [d#4]; hop 5: added 1: [e#5]", "read in full: 16 characters", "stopped"]
    assert tools.calls == [("read", "a#1"), ("expand", "a#1"), ("expand", "b#2"), ("expand", "d#4"), ("search", "new words"), ("read", "d#4")]      # the key-less EXPAND of hop 5 spread from hop 4's d#4
    assert len(run["shown"]) == 2 and not {c[0] for c in run["shown"][0]} & {c[0] for c in run["shown"][1]}                                    # a fresh batch per planning round, none repeated
    assert "Replies left this round: 3" in ask.prompts[0] and "Plan hop 3 now" in ask.prompts[0] and "[c#3]" not in ask.prompts[0]          # hop 3 does not exist before it is planned
    assert "LAST REPLY THIS ROUND" not in ask.prompts[0] and "LAST REPLY THIS ROUND" not in ask.prompts[1]                                    # replies left 3 and 2
    assert g.agent_prompt("q", "", {}, {}, [], 1, (3,), 1).count("THIS IS YOUR LAST REPLY THIS ROUND") == 1 and "LAST REPLY" not in g.agent_prompt("q", "", {}, {}, [], 3, (), 1)      # only in a planning round, only at the last reply
    assert "Plan hop 4 and hop 5 now" in ask.prompts[2] and "Hop 5 will run without you seeing hop 4." in ask.prompts[2] and "[c#3] hop 3" in ask.prompts[2]
    first, second = run["shown"]
    assert ask.prompts[0].count("(new)") == 0 and ask.prompts[2].count("(new) ") == len(second)                                           # round 2 marks only its own batch
    assert all(("      %d sections | %s" % (size, title)) in ask.prompts[2] for _, size, title in first)                                    # and keeps the first round's, unmarked
    assert "No more hops." in ask.prompts[3] and "round 2: PLAN" in ask.prompts[3] and "*[a#1] hop 0" in ask.prompts[3] and "full text of a#1" in ask.prompts[3]
    summary = g.traversal_summary(run, [("harness", 3)])
    assert summary == {"hops": 5, "size": 5, "per_hop": [1, 1, 0, 1, 1, 1], "read": 2, "stop": "answered", "entities": [("harness", 3)], "by_community": [], "queries": 2}      # the question and one SEARCH
    assert 'QUERY 2: "new words" -> 1 sections' in ask.prompts[3] and "QUERY 1:" in ask.prompts[3]                                  # the closing round sees the search as its own query
    by = [{"cid": 5, "n": 3, "entities": [("harness", 3)]}]
    assert g.traversal_summary(run, [("harness", 3)], by)["by_community"] == by


def test_agent_loses_attempts_moves_and_a_whole_planning_chance_and_is_told_so():
    tools = FakeTools()
    ask = scripted("let me think", "ACTION: READ z#9\nWHY: x", "ACTION: PLAN\nHOP 3: EXPAND a#1 | EXPAND z#9\nWHY: x",
                   "ACTION: READ a#1\nWHY: x", "ACTION: READ a#1\nWHY: again", "ACTION: EXPAND a#1\nWHY: not a plan",
                   "ACTION: PLAN\nHOP 4: EXPAND a#1\nWHY: closing has no hops", "ACTION: ANSWER\nWHY: done")
    run = g.run_agent("q?", "", start(tools), tools, ask)
    assert [t["result"] for t in run["trace"]] == [
        "your reply was not in one of the ACTION forms; nothing was done", "[z#9] is not one of the sections seen; nothing was done",
        "hop 3: added 1: [c#3] (EXPAND z#9: no source ([z#9] is not a section seen))",
        "read in full: 16 characters", "you have already read [a#1]; nothing was done", "your reply was not in one of the ACTION forms; nothing was done",
        "no plan: hop 4 and 5 did not run", "there is no hop to plan now; nothing was done", "stopped"]
    assert tools.calls == [("expand", "a#1"), ("read", "a#1")] and run["stop"] == "answered"       # nothing the agent got wrong reached the database
    assert "round 2: no plan was made, so hop 4 and 5 did not run" in ask.prompts[6]
    assert run["shown"] == []                                                                       # no catalogue given: nothing sampled, nothing shown


def test_agent_may_stop_at_its_first_reply():
    tools = FakeTools()
    run = g.run_agent("q?", "", start(tools), tools, scripted("ACTION: ANSWER\nWHY: the retrieved sections already answer it"), catalogue=CAT)
    assert run["stop"] == "answered" and tools.calls == [] and len(run["trace"]) == 1 and run["nodes"].keys() == {"a#1", "b#2"} and len(run["shown"]) == 1


class Wide:
    """Every expand and search returns 8 new sections, so the caps are what bind."""
    def __init__(self):
        self.n = 0

    def batch(self):
        self.n += 8
        return [{"key": "w%d#1" % i, "ord": i, "topic": "t", "sim": 1 / (1 + i), "via": "v", "snippet": "s"} for i in range(self.n - 8, self.n)]

    expand = entity = lambda self, key, seen: self.batch()
    search = lambda self, words, seen: self.batch()


def test_a_hop_adds_at_most_15_sections_and_runs_at_most_3_moves():
    nodes = {"a#1": {"key": "a#1", "hop": 2, "sim": 0.9}}
    text, new = g.apply_hop(3, [("SEARCH", "a"), ("SEARCH", "b"), ("SEARCH", "c"), ("SEARCH", "d")], nodes, Wide())
    assert len(new) == 15 and text.startswith("added 15:") and all(nodes[k]["hop"] == 3 for k in new)      # 3 moves x 8 = 24 offered, 15 kept; the 4th move never ran
    w = Wide()
    g.apply_hop(3, [("SEARCH", "a"), ("SEARCH", "b"), ("SEARCH", "c"), ("SEARCH", "d")], {"a#1": {"key": "a#1", "hop": 2, "sim": 0.9}}, w)
    assert w.n == 24
    assert g.apply_hop(3, [], {"a#1": {"key": "a#1", "hop": 2, "sim": 0.9}}, Wide())[0] == "no move given"
    assert g.apply_hop(4, [("EXPAND", "")], {"a#1": {"key": "a#1", "hop": 2, "sim": 0.9}}, Wide())[0] == "nothing new (EXPAND : no source (the previous hop added nothing))"
