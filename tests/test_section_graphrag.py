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
    assert g.parse_action("ACTION: SEARCH  kv cache eviction  \nWHY: spread") == ("SEARCH", "kv cache eviction", "spread")        # a search keeps its words whole, quotes and brackets included
    assert g.parse_action("ACTION: ANSWER") == ("ANSWER", "", "")


def test_parse_action_rejects_anything_not_an_action_on_the_first_line():
    for bad in ["I will read the first one.\nACTION: READ a#1", "ACTION: DELETE a#1", "ACTION: EXPAND a#1", "ACTION: PLAN\nHOP 3: EXPAND a#1", "READ a#1", ""]:      # hops are not the agent's to plan any more
        assert g.parse_action(bad) == (None, bad, "")


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
    p = g.agent_prompt("q", "", {"a#1": node}, {}, [], 1, True, 3, [one], facts)
    assert "1630 sections | Agent runtimes | harness 512 · verifier 169" in p and "the entities most characteristic of it" in p


def test_working_and_read_blocks_mark_what_was_read_and_cut_a_long_read_visibly():
    nodes = {"a#1": {"key": "a#1", "hop": 0, "topic": "Topic A", "sim": 0.5, "via": "retrieved", "snippet": "first words"},
             "b#2": {"key": "b#2", "hop": 2, "topic": "(no summary)", "sim": 0.25, "via": "vector edge from [a#1], weight 3.0", "snippet": "other"}}
    w = g.working_block(nodes, {"a#1": "x"}).splitlines()
    assert w[0] == 'QUERY 1: "" -> 2 sections' and w[1] == "*[a#1] hop 0 | Topic A | similarity 0.50 | retrieved | first words" and w[2].startswith(" [b#2] hop 2 | (no summary) | similarity 0.25 | vector edge from [a#1]")


def test_explorer_tags_nodes_with_their_query():
    ex = g.Explorer(None, {"question": "q"})
    out = ex.tag([{"key": "a#1", "ord": 1}, {"key": "b#2", "ord": 2}], 2, "kv cache")
    assert out == [{"key": "a#1", "ord": 1, "query": 2, "query_text": "kv cache"}, {"key": "b#2", "ord": 2, "query": 2, "query_text": "kv cache"}]
    assert ex.queries == 1 and (ex.hops, ex.per_query) == (3, 18) and "2401_12345" not in g.ANSWER_PROMPT                                      # the counter is only advanced by a SEARCH; the defaults are 3 hops and 13 sections


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
    """The agent's two database actions over a five-section world; a search finds e#5 (then f#6) and none already seen."""
    SIM = {"a#1": 0.9, "b#2": 0.6, "c#3": 0.3, "d#4": 0.5, "e#5": 0.2, "f#6": 0.1}
    FOUND = ["e#5", "f#6"]

    def __init__(self):
        self.calls = []

    def node(self, key, via, hop=None):
        n = {"key": key, "ord": ord(key[0]), "topic": "t", "sim": self.SIM[key], "via": via, "snippet": "s"}
        return n if hop is None else n | {"hop": hop}

    def read(self, key):
        self.calls.append(("read", key))
        return "full text of " + key

    def search(self, words, seen):
        self.calls.append(("search", words))
        left = [k for k in self.FOUND if k not in seen]
        return [self.node(left[0], "found by searching '%s'" % words, 0) | {"query": 2 + len(self.FOUND) - len(left), "query_text": words}] if left else []


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


def test_agent_reads_then_searches_each_round_and_reads_and_stops_in_the_closing_round():
    tools = FakeTools()
    ask = scripted("ACTION: READ a#1\nWHY: the seed", "ACTION: SEARCH kv cache words\nWHY: new angle",
                   "ACTION: SEARCH second words\nWHY: another", "ACTION: ANSWER\nWHY: enough")
    run = g.run_agent("q?", "global", start(tools), tools, ask, queries=2, catalogue=CAT, seed=3)
    assert run["stop"] == "answered" and list(run["nodes"]) == ["a#1", "b#2", "e#5", "f#6"] and list(run["read"]) == ["a#1"]
    assert [t["result"] for t in run["trace"]] == ["read in full: 16 characters", "search 2: added 1: [e#5]", "search 3: added 1: [f#6]", "stopped"]
    assert tools.calls == [("read", "a#1"), ("search", "kv cache words"), ("search", "second words")]
    assert len(run["shown"]) == 2 and not {c[0] for c in run["shown"][0]} & {c[0] for c in run["shown"][1]}      # a fresh batch per search round, none repeated
    assert "search round 1 of 2" in ask.prompts[0] and "Replies left this round: 3" in ask.prompts[0] and "[e#5]" not in ask.prompts[0]      # the search does not exist before it is made
    assert "LAST REPLY THIS ROUND" not in ask.prompts[0] and "LAST REPLY THIS ROUND" not in ask.prompts[1]                                   # replies left 3 and 2
    assert g.agent_prompt("q", "", {}, {}, [], 1, True, 1).count("THIS IS YOUR LAST REPLY THIS ROUND") == 1 and "LAST REPLY" not in g.agent_prompt("q", "", {}, {}, [], 3, False, 1)
    assert ask.prompts[0].count("(new)") == 0 and ask.prompts[2].count("(new) ") == len(run["shown"][1])                                      # round 2 marks only its own batch
    assert 'QUERY 2: "kv cache words" -> 1 sections' in ask.prompts[2] and "round 1: SEARCH kv cache words" in ask.prompts[2]               # the next round sees the search as its own query
    assert "ACTION: PLAN" not in ask.prompts[0] and "EXPAND" not in ask.prompts[0] and "hop 3, %d in all" % g.PER_QUERY in ask.prompts[0]                  # no plan, no move: only READ, SEARCH, ANSWER
    walks = {2: {"words": "kv cache words", "held": 1, "pool": 4, "masked": 2}}
    summary = g.traversal_summary(run, [("harness", 3)], None, walks)
    assert summary == {"hops": 3, "size": 4, "per_hop": [3, 1, 0, 0], "read": 1, "stop": "answered", "entities": [("harness", 3)], "by_community": [], "queries": 3,
                       "walks": [{"query": 2, "words": "kv cache words", "held": 1, "pool": 4, "masked": 2}], "per_query": g.PER_QUERY}
    by = [{"cid": 5, "n": 3, "entities": [("harness", 3)]}]
    assert g.traversal_summary(run, [("harness", 3)], by)["by_community"] == by


def test_agent_loses_replies_and_a_whole_search_and_is_told_so():
    tools = FakeTools()
    ask = scripted("let me think", "ACTION: READ a#1\nWHY: x", "ACTION: SEARCH\nWHY: no words",
                   "ACTION: READ a#1\nWHY: again", "ACTION: SEARCH words in the closing round\nWHY: none left", "ACTION: ANSWER\nWHY: done")
    run = g.run_agent("q?", "", start(tools), tools, ask, queries=1)
    assert [t["result"] for t in run["trace"]] == [
        "your reply was not in one of the ACTION forms; nothing was done", "read in full: 16 characters", "SEARCH had no words; nothing was done", "no search: this round's search was lost",
        "you have already read [a#1]; nothing was done", "there is no search left; nothing was done", "stopped"]
    assert tools.calls == [("read", "a#1")] and run["stop"] == "answered"                            # nothing the agent got wrong reached the database
    assert "round 1: no search was made" in ask.prompts[3] and "No more searches." in ask.prompts[3]
    assert run["shown"] == []                                                                       # no catalogue given: nothing sampled, nothing shown
    other = g.run_agent("q?", "", start(FakeTools()), FakeTools(), scripted("ACTION: READ z#9\nWHY: x", "ACTION: PLAN\nWHY: gone", "ACTION: READ a#1\nWHY: x", "ACTION: ANSWER\nWHY: y"), queries=0)
    assert [t["result"] for t in other["trace"]][:3] == ["[z#9] is not one of the sections seen; nothing was done", "your reply was not in one of the ACTION forms; nothing was done",
                                                         "read in full: 16 characters"] and other["stop"] == "done"      # queries=0: only the closing round, three replies, no ANSWER reached


def test_agent_may_stop_at_its_first_reply():
    tools = FakeTools()
    run = g.run_agent("q?", "", start(tools), tools, scripted("ACTION: ANSWER\nWHY: the retrieved sections already answer it"), catalogue=CAT)
    assert run["stop"] == "answered" and tools.calls == [] and len(run["trace"]) == 1 and run["nodes"].keys() == {"a#1", "b#2"} and len(run["shown"]) == 1


def test_a_search_that_finds_nothing_new_still_spends_the_round():
    tools = FakeTools()
    tools.FOUND = []
    run = g.run_agent("q?", "", start(tools), tools, scripted("ACTION: SEARCH x\nWHY: try", "ACTION: ANSWER\nWHY: done"), queries=1)
    assert [t["result"] for t in run["trace"]] == ["search 2: nothing new", "stopped"] and run["nodes"].keys() == {"a#1", "b#2"}


# the masked walk: a graph of 12 sections in a line-and-fan, so hops are countable
def ring_neigh(ords):
    """Section n is joined to n+1..n+4 (weight falling with the gap) and n-1..n-4: the strongest first."""
    return {s: [{"ord": s + d, "weight": 10.0 - d} for d in (1, 2, 3, 4)] + [{"ord": s - d, "weight": 5.0 - d} for d in (1, 2, 3, 4) if s - d >= 0] for s in ords}


def test_hop_pool_walks_outward_one_hop_at_a_time_and_caps_each_hop():
    pool, skipped = g.hop_pool([10], ring_neigh, 1, 3, set())
    assert [p["ord"] for p in pool] == [11, 12, 13] and {p["hop"] for p in pool} == {1} and skipped == 0                  # four neighbours above, cap 3: the strongest three
    pool, _ = g.hop_pool([10], ring_neigh, 2, 3, set())
    assert [p["hop"] for p in pool] == [1, 1, 1, 2, 2, 2] and not {p["ord"] for p in pool if p["hop"] == 2} & {10, 11, 12, 13}   # hop 2 spreads from hop 1's three and repeats none
    assert len(g.hop_pool([10], ring_neigh, 3, 3, set())[0]) == 9 and g.hop_pool([10], ring_neigh, 0, 3, set()) == ([], 0)


def test_hop_pool_never_returns_a_masked_section_walks_past_it_and_counts_it():
    pool, skipped = g.hop_pool([10], ring_neigh, 2, 4, {11, 12})
    assert not {11, 12} & {p["ord"] for p in pool} and skipped >= 2                                                     # masked ones are skipped, and the walk still fills its cap
    assert [p["ord"] for p in pool if p["hop"] == 1] == [13, 14, 9, 8]                                                  # the next strongest after the two masked
    assert g.hop_pool([10], lambda ords: {s: [] for s in ords}, 3, 5, set()) == ([], 0)                                 # a seed with no edges: an empty pool, not an error


def test_top_up_keeps_the_new_part_of_the_natural_subgraph_and_adds_as_many_as_overlapped_hopping_from_all_of_it():
    natural = [10, 11, 12, 13]
    chosen, depth, k = g.top_up(natural, {11, 13}, ring_neigh, 4)
    assert k == 2 and chosen[:2] == [10, 12] and len(chosen) == 4                                   # the new two stay, in order; two more are added
    assert not {11, 13} & set(chosen) and not set(natural) & set(depth) and set(depth) == set(chosen[2:])      # nothing occupied or natural is added; the added ones are the ones with a depth
    assert set(depth.values()) == {1} and all(abs(o - 11.5) < 6 for o in chosen[2:])               # one hop was enough, and it is near the natural subgraph
    assert g.top_up(natural, set(), ring_neigh, 4) == (natural, {}, 0)                              # no overlap: nothing added, k = 0
    chosen, depth, k = g.top_up(natural, set(natural), ring_neigh, 4)
    assert k == 4 and len(chosen) == 4 and not set(natural) & set(chosen)                          # everything overlapped: the whole result is hopped to, from the natural subgraph
    deeper, depth, _ = g.top_up([10], {10}, ring_neigh, 9)
    assert len(deeper) == 9 and max(depth.values()) >= 2                                            # more needed than one hop holds: it goes deeper only as far as needed
    assert g.top_up([5], {5}, lambda ords: {s: [] for s in ords}, 3) == ([], {}, 1)                 # nowhere to hop: fewer than asked, never padded


def test_the_final_evidence_is_the_whole_traversal_read_first_then_centrepoints_then_nearest_the_question_G14():
    nodes = {"a#1": {"ord": 1, "query": 1, "hop": 0}, "b#2": {"ord": 2, "query": 1, "hop": 1}, "c#3": {"ord": 3, "query": 2, "hop": 0}, "d#4": {"ord": 4, "query": 2, "hop": 2},
             "e#5": {"ord": 5, "query": 3, "hop": 1}}
    sim = {1: 0.4, 2: 0.9, 3: 0.7, 4: 0.2, 5: 0.7}
    out = g.subgraph_order(nodes, {"d#4": "full text"}, {7: 3, 8: 99}, [(7, 2), (8, 1)], sim)
    assert [o["key"] for o in out] == ["d#4", "c#3", "b#2", "e#5", "a#1"]                            # read first, the centrepoint c#3 second, the rest by similarity (c#3 and e#5 tie: key order)
    assert out[0]["role"] == "read by the agent" and out[1]["role"].startswith("typical section of a community that holds 2 of the 5 sections seen")      # centre 8 is not in the traversal: skipped
    assert out[2]["role"] == "query 1, hop 1, similarity 0.90" and out[4]["role"] == "query 1, hop 0, similarity 0.40"
    assert {o["key"] for o in out} == set(nodes) and len(out) == 5                                    # nothing the traversal saw is left out, nothing twice
    away = g.subgraph_order(nodes, {"d#4": "full text"}, {7: 3, 8: 99}, [(7, 2), (8, 1)], sim | {99: 0.0}, {99: "z#9"})
    assert [o["key"] for o in away][:3] == ["d#4", "c#3", "z#9"] and len(away) == 6                   # a centrepoint the traversal never saw is still evidence (2 of 3 were dropped once)
    assert [o["key"] for o in g.subgraph_order(nodes, {}, {}, [], sim)] == ["b#2", "c#3", "e#5", "a#1", "d#4"]


def test_each_section_gets_the_whole_6000_while_the_budget_holds_then_an_even_share_never_under_the_floor():
    assert g.answer_chars(10) == g.READ_CHARS and g.answer_chars(g.ANSWER_BUDGET_CHARS // g.READ_CHARS) == g.READ_CHARS
    assert g.answer_chars(70) == 2000 and g.answer_chars(54) == 2592                                  # an even share of the 140,000: the memory question's 54 sections
    assert g.answer_chars(200) == g.MIN_SECTION_CHARS == 1200                                         # the even share (700) is under the floor, so the floor binds
    assert g.answer_chars(500) == g.MIN_SECTION_CHARS and g.answer_chars(0) == g.READ_CHARS


def test_two_walks_with_the_first_one_masked_share_no_section():
    first, _ = g.hop_pool([10], ring_neigh, 2, 5, set())
    taken = {10} | {p["ord"] for p in first}
    second, _ = g.hop_pool([30], ring_neigh, 2, 5, taken)
    assert not taken & {p["ord"] for p in second}
