# Plan: fixed-depth hops and masked, mutually exclusive queries for the section agent

Written 2026-10-09. This file is the goal: a session can process it top to bottom. Status of every step: OPEN.

## Goal, in the operator's words

"by default, I think we should branch out 3 hops from whatever nodes our initial sub graph lands on for the given query, however this should be a tunable parm, same as number of searches react agent can do."

"we tune for recall and precision using n hops out and n react iterations, no more than 3 for either (so we can test 1,2,3 and 1,2,3)."

"Instead of n+ queries, 1 per separate non mutually exclusive subgraphs, we will have n+ queries, 1 per mutually exclusive subgraph. This way we naturally extend the subgraph but don't re-select prior selected nodes/vectors, and if we do, we just mask/ignore those and continue to count until we get our magic number (I prefer top 13 chunks). This avoids letting the user specify more hops out ... So the react agent isn't trying to figure out what to ask. Also the communities entities (Dunning/NPMI/DIRT entities) are very valuable for the react agent to search by."

Query shape: query 1 is chosen from a sample of the global communities. Query 2 sees the prior community, query and output as a worked example, plus a fresh sample of global communities. A possible query 3 is one more query, not a two-step plan.

## What changes

The agent today plans hops 3 to 5 itself, through EXPAND, ENTITY and SEARCH moves in two planning rounds (`PLAN_ROUNDS`, `apply_hop`, `MOVES` in `src/section_graphrag.py`). That is replaced by:

- Hops are a parameter, `HOPS = 3` by default, tried at 1, 2, 3. The agent never extends the graph.
- Agent searches are a parameter, `AGENT_QUERIES = 3` by default, tried at 1, 2, 3.
- Each query returns `PER_QUERY = 13` sections. Its subgraph is the unmasked hybrid seeds plus `HOPS` hops of vector neighbours (each hop capped, strongest edge weight first). Sections already shown by any earlier query are masked, and the walk keeps going (further hybrid ranks, then the next hop) until 13 unmasked sections are held or the pool is exhausted. A short result is reported, not padded.
- Agent actions shrink to SEARCH words, READ key, ANSWER. PLAN, EXPAND and ENTITY go.
- Each agent turn shows the question, the global answer, a fresh Box-Cox community sample (existing `sample_communities`, seed = crc32(question) + round, newest batch marked, community entities in each line), and for every earlier query its words, the communities it landed in and its 13 sections with paper titles and entity composition. That is the worked example the operator described.

## Amendment 2026-10-09 (operator, mid-run): overlap-count top-up instead of masking

"do the regular non mutually exclusive subgraph query, count how many nodes and vectors overlap with existing query, and simply gain that many more to 'hop' from ... Repeat for the 3rd query. Simple as that, no need to keep track of numbers, numbers are simply based on what would have been eligible from the initial subgraph and what is already occupied."

- Query 2 and 3: run the ordinary walk with nothing masked (the natural 13). `k` = how many of those 13 are already shown by an earlier query. Keep the 13 - k new ones, then add `k` more by hopping out from the whole natural subgraph along vector edges, strongest edge weight first, deeper hops only as far as `k` needs. Deterministic; the only count is `k`.
- Two modes of `Explorer.pick`, compared in stage B: `mask` (built: masked seeds and hops, refill from the ranked pool) and `topup` (this amendment). `HOPS` stays a parameter for the question's own query; later queries' depth comes from `k`.
- Entity (node) top-up of `k` more is a third arm, not the default: entity hops lost the known-item test (`+ entity hop 1  47.1 | 206 / 400 = 0.515 | 307 / 400 = 0.767 | 5 / 106`).
- Stage A (one query, nothing shown yet) cannot tell `mask` from `topup`; they differ only at query 2 and 3.

## Grounded numbers behind the design

- Graph fan-out on build 32 (80,642 sections): mean degree 39.4, median 32, p90 68, max 1,668. Three seeds reach about 118 sections at hop 1, about 4,600 at hop 2, and the whole corpus by hop 3 before overlap. The "five or six per node, fully connected by six hops" rule does not hold here; saturation is near hop 3. A hop count therefore needs a per-hop cap to mean anything.
- Old hop test, `.tmp/entity_hop_recall.log`, 400 known-item queries: `+ edge neighbours  17.1 | 197 / 400 = 0.492 | 255 / 400 = 0.637 | 5 / 63` (expansion against same-length hybrid retrieval, paired counts). Entity hops lost the same way. This instrument cannot judge hops fairly: truth was any section of the source paper, and edges exclude same-paper pairs (`src/section_graph.py:114`). It is not a verdict on hops either way. The default of 3 is a hypothesis, to be re-opened by the tests below.
- Masking trades precision for coverage: a masked second query returns the next unseen sections, not the most similar ones. The same-length control below measures that trade.
- Hop kind is vector edges only. Entity hops lost the known-item test and stay out of the default; they are an optional extra arm.

## Steps

1. Spec first. Amend the guards in the `src/section_graphrag.py` docstring (G8 to G11 replaced; new guards for the masking rule and for hops and query count as parameters) and add a task to `playbook.md`. No governing spec exists beyond the operator instructions above.
2. `src/section_store.py`: add `hop_pool(conn, build, seeds, hops, cap, masked)` over the existing `neighbours`. Extend, no new module. Guard plus a database-backed test in a rolled-back transaction.
3. `src/section_graphrag.py`: remove planning rounds, moves and `apply_hop`; make `Explorer.search` the masked walk; write the new prompt with the prior-query worked examples; give `agent_answer` `hops` and `queries` arguments. Update `tests/test_section_graphrag.py` with planted worlds: a masked section never reappears, a query returns exactly 13 or fewer-and-reported, hops=1 and hops=3 give different pool sizes, the query count is respected.
4. `src/section_query_panel.py`: the traversal block shows hops, queries run, 13 per query, how many were masked out, and entity composition per query. Update `tests/test_section_query_panel.py`.
5. Evaluation, scratch script `.tmp/hop_grid.py`. One number per arm, with n and a Wilson interval. Decided before the run:
   - Truth: the 200 body-sentence queries from `ingest_arxiv_sparsevec.gold`, scored on the exact source section, with paper-level truth beside it.
   - Precision: share of shown sections a MiniLM cross-encoder rates relevant to the question, threshold calibrated on the gold sections' own scores and stated before the run.
   - Stage A, no model call, 400 queries: hops 1, 2, 3 at zero agent queries, each against a same-length hybrid-retrieval control.
   - Stage B, agent, 60 questions, search texts logged so reruns are free: agent queries 1, 2, 3 crossed with hops 1, 2, 3, each against a control of the same total length (13 x (queries + 1)).
   - A winner is demoted, never frozen: record corpus, truth and ranker beside it.
6. Update `README.md` and `playbook.md` with the receipt lines.

## Out of scope here (separate spec needed)

The conclusion and entailment idea: judge which extracted facts are entailed by a conclusion, run one cheap-model call for missing entailed phrases with the prior ones as few-shot, and index conclusions to sentence or character positions in the source. It compresses the answer-side evidence after retrieval. It does not change hops or queries, so it composes later. The small decoder classifier (JEV) is optional; a MiniLM cross-encoder is the lighter judge.

## Files

Edit: `src/section_graphrag.py`, `src/section_store.py`, `src/section_query_panel.py`, `tests/test_section_graphrag.py`, `tests/test_section_store.py`, `tests/test_section_query_panel.py`, `playbook.md`, `README.md`.
Reuse: `sample_communities`, `communities_block`, `Explorer.nodes`, `st.neighbours`, `st.seeds_dense`, `st.seeds_lexical`, `rrf`, `.tmp/entity_hop_recall.py`, `ingest_arxiv_sparsevec.gold`.
Scratch: `.tmp/hop_grid.py`.

## Verification

- Full suite before and after. Baseline: 1196 passed, 1 failed (the panels render test, failing before this work), 4 skipped.
- Stage A and stage B receipt lines pasted with n.
- Strip re-rendered with the new traversal block; the PNG and markdown opened and checked.

## Risks

- The cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` is not cached locally (only the all-MiniLM-L6-v2 bi-encoder is). The download is about 90 MB and needs approval. Without it, precision falls back to dense cosine, which is circular with the retrieval ranker.
- Edges exclude same-paper pairs, so a hop cannot add a sibling section of a seed's paper. Exact-section truth understates any gain from siblings.
- Whether masked later queries beat plainly taking more hybrid ranks is unmeasured. The same-length control answers it.
- Stage B cost is roughly 60 questions x up to 4 model calls x 9 cells. Cached search texts make reruns cheap.
- The two scheduled tasks (`arxiv-graph-ingest`, `arxiv-rag-api`) still launch `tools\...` and need their actions changed to `src\...`. Unrelated to this plan, still open.

## Outcome 2026-10-09 (what was built and measured)

- Built: `HOPS` and `AGENT_QUERIES` as parameters (default 3, tunable 1 to 3, `--hops`, `--queries`), `PER_QUERY` 18 (13 dropped the survey section, see guard G13), masked walk and overlap-count top-up (`top_up`), agent actions READ / SEARCH / ANSWER only, final interpretation over every section the traversal saw (`subgraph_order`, guard G14), strip lists all of them (guard P5).
- Measured, stage A (200 queries): walk 0.445 to 0.450 against hybrid 0.555 on the exact section; hops 1, 2, 3 identical. Query 2 (200 pairs): mask and top-up tie at 0.41, plain next-13 hybrid 0.505. The graph walk does not beat plain retrieval on this instrument, which cannot see answer quality.
- NOT TESTED: walk from the hybrid top 13 instead of top 3 (the one change that could make the hop knob live); the agent grid (queries 1 to 3 x hops 1 to 3); answer quality (blind 20-question comparison); the entity (node) top-up arm.
- Open defect: the long-term-memory answer does not list the three memory types (externalised text, latent, parametric) that yesterday's run did, although the survey section holding them is now in the evidence whole.
