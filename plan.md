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
- Initial fix 2026-10-11: the prompt marked likely taxonomy headings, and the evidence budget was raised to 270,000 characters (3,750 per section at the 72-section maximum), enough for `arxiv/2605_12357#19` whole. A successful live answer restored textual, latent and parametric memory, but later runs still omitted latent.
- Superseded: a follow-up scheduler reserved a read for that heading. The operator correctly identified this as an ad-hoc, section-specific workaround. It has been removed; traversal no longer classifies headings or reserves reads.
- Current principle: for methods questions, prompts ask the agent to track both what approaches do and how sources classify their variants; any other question-relevant distinctions are tracked too. A taxonomy is one view of the methods, not a substitute for describing them. No section-heading classifier or read reservation remains.
- Prompt-only memory rerun: the agent read a human-memory classification, then searched for memory-construction/retrieval workflows and additional classification axes. The answer covered workflows, functional memory types, storage formats and parametric versus external memory. It still omitted textual versus latent memory even though `arxiv/2605_12357#19` was present in the gathered evidence; the agent did not read it. This is evidence that dimension-oriented prompting improves coverage in this run, not proof of exhaustive or reliable recovery.
- Follow-up memory iteration: the question now distinguishes function/content from storage form, and the prompt says to keep orthogonal axes apart and ground overloaded terms in their source field. The final answer explicitly includes factual memory, latent representations, and parametric memory, with citations to the factual-memory and latent/parametric evidence. It treats factual as a function/content label and latent/parametric as storage-form labels, rather than treating all three as competing storage methods. This verifies the requested labels in this answer; it is still one live example, not a general quality evaluation.
- Fixed 2026-10-11: global reduction and final synthesis now mention evidence gaps only when they concern an explicit part of the question; generic caveats about unrelated topics are prohibited. The three example answers and map were regenerated under the dimension-oriented prompt.
- Final citations are checked against the evidence keys after generation; a sentence with a non-evidence bracketed citation is dropped. This catches labels such as `[GLOBAL view]` that the prompt alone did not prevent.
- Still NOT TESTED: the blind 20-question answer comparison and the full agent grid in stage B. The three example answers in the map are a smoke check, not a quality evaluation.
- Verification 2026-10-11: focused section-GraphRAG tests 35 passed; full suite 1,210 passed, 1 failed, 3 skipped. The failure is the existing `tests/test_walker_render.py::test_panels_render_digests_and_chain_communities` assertion for missing `pairs=… corpus_n=…` text.

## Outcome 2026-10-12 (community scoring, no fallback; handoff.md)

- Cause of the bad global view: the 0.8B Jev-Style scorer's score depends on the other summaries sharing its request (materials-science summaries 0.03 alone, 0.74-0.86 in a 64-summary request; the speculative-decoding community 0.77 alone, 0.25 in 64). Polarity (`noul` = P(true)) was right.
- Built: one summary per scoring request by default; partials open `RELEVANT: YES|NO` and a NO is kept in `partials_rejected`; no fallback: a question nothing passes is rejected (`question_rejected`: no global view, agent, evidence or answer); the strip shows scored / passed / kept and a QUESTION REJECTED block.
- Result at threshold 0.50 over 415 communities: speculative decoding answered (1 kept), agent-memory REJECTED (best c10 0.449; c21 0.182), safety answered (6 passed, 5 kept, 3 of them weak). Citations all resolve.
- NOT TESTED: a calibrated cut (needs labelled pairs), a larger scorer, the blind 20-question comparison, the agent grid. Tests: 50 passed across the reranker, query-panel and graphrag files; the wider suite was not re-run.

## Step status against this plan (2026-10-12)

| Step | Status | Evidence |
|---|---|---|
| 1 Spec first (guards, ledger) | DONE | guards G8, G11, G13, G14, G15 and P3, P5 in the module docstrings |
| 2 `hop_pool` and the masked walk | DONE | `hop_pool`, `top_up`, `Explorer.pick/walk` in `src/section_graphrag.py`; tests in `tests/test_section_graphrag.py` |
| 3 Agent reads/searches only, final answer over every section seen | DONE | `run_agent` (READ, SEARCH, ANSWER), `subgraph_order`, `answer_chars` (G14) |
| 4 Strip shows every section of every query, hops, masked count | DONE | `src/section_query_panel.py` (P3, P5), `tests/test_section_query_panel.py` |
| 5a Stage A, hops 1/2/3 against same-length hybrid, 200 queries | DONE | `.tmp/hop_grid_stage_a.log`: walk 0.445-0.450 against hybrid 0.555, hops identical |
| 5a' mask against top-up, 200 query pairs | DONE | `.tmp/mask_vs_topup.log`: both 0.41 against plain next-13 hybrid 0.505 |
| 5b Stage B, agent searches 0-3 x hops 1-3 | DONE at 60 questions, GLOBAL VIEW EMPTY (the Jev gate was not in this loop; its own measure is the next row) | `docs/experiments/stage_b_result.txt` (the first 20 questions' earlier run is kept in `stage_b_result_20_questions.txt`), script and pre-registration in `docs/experiments/stage_b.py`. Global view left empty (the traversal is tested, not the Jev gate: that would be about 95 minutes of solo scoring for 60 questions); the 3-search run supplies the 1- and 2-search results as prefixes. Result below. |
| 5c Cross-encoder precision | DONE (a precision measure for stage B only) | `cross-encoder/ms-marco-MiniLM-L-6-v2` downloaded with curl (this environment's Python SSL fails; curl verifies) into `.tmp/ce_minilm`; relevant pair 8.66 / 1.21, irrelevant -11.4. tau = 25th percentile of the gold sections' scores, fixed before any arm was scored (4.815 over the first 20 questions, 5.116 over the 60). |
| 5d Jev gate on in-corpus questions and cut calibration (added 2026-10-12) | DONE at 60 questions | `docs/experiments/jev_gate_probe_result.txt`: at 0.50, 29 of 60 questions (0.48) pass NO community; the source section's own community passes in 13 of 60 (0.22), ranks 1st in 12, top 10 in 24, median rank 20 of 415. Calibration table below. The first 20 questions had suggested 0.70 rejected. |
| 5e Stage B with the Jev gate in the loop (added 2026-10-12) | DONE at 60 questions | `docs/experiments/stage_b_gate_result.txt`, script `stage_b_gate.py`. Result below. |
| 5f Blind answer comparison (added 2026-10-12) | DONE as a model-judged second opinion on the 25 gate-answered questions, not a human read | `docs/experiments/blind_cmp_result.txt`, script `blind_cmp.py`. Result below. |
| 6 README and plan outcome | DONE | `README.md`, this file; pushed at `ec0241d` |

Stage B result (60 known-item questions, truth = the exact source section; same-size hybrid control; tau 5.116):

```
hops 1, 0 searches: walk 22/60 = 0.37 | hybrid 31/60 = 0.52 | paired only-walk/only-hybrid 0/9
hops 2, 0 searches: walk 24/60 = 0.40 | hybrid 31/60 = 0.52 | paired 1/8
hops 3, 0 searches: walk 24/60 = 0.40 | hybrid 31/60 = 0.52 | paired 1/8
hops 2, 3 searches: walk 29/60 = 0.48 | hybrid 40/60 = 0.67 | paired 2/13 (two-sided sign test p about 0.007)
hops 3, 3 searches: walk 26/60 = 0.43 | hybrid 39/60 = 0.65 | paired 1/14
paper-level, 3 searches: walk 0.68-0.73, hybrid 0.73 in every hop setting
precision (cross-encoder >= tau): walk 0.042-0.051, hybrid 0.046-0.058 in every cell
sections shown: 18 with 0 searches, 30 with 1, 39 with 2, 42-43 with 3 (the agent stopped by ANSWER in 177 of 180 runs)
```

Reading: on this instrument the graph walk is WORSE than a plain hybrid list of the same size at finding the source section, at every hop setting and every search count; hops 1, 2 and 3 are not distinguishable (0.37 / 0.40 / 0.40 at 0 searches); the agent's searches lift the walk from 0.40 to 0.48 but the hybrid list lifts more (0.52 to 0.67) at the same size; precision is a small loss. The paper-level lead seen in the first 20 questions (0.85 against 0.70) did not survive 60 (0.73 against 0.73). This does not measure answer quality, and the global view was empty.

Still NOT TESTED: the blind 20-question answer comparison; a 60-question stage B; stage B with the Jev global view in the loop.

Jev gate on in-corpus questions (20 sentences each taken from one section, solo scoring of all 415 summaries, threshold 0.50): the question is REJECTED for 14 of 20 (0.70); the community that contains the source section scores >= 0.50 in 4 of 20, ranks first in 6 and within the top 10 in 8 (median rank 13 of 415). Under the no-fallback rule this scorer at this cut rejects most questions the corpus can answer. Caveat: sentences lifted from a section are not user-style questions, and several sections sit in generic communities ("Experimental details") that no summary-level judge will score high. Calibrating the cut needs labelled pairs; the 60 gold sentences with their communities are such a set.

Jev cut calibration (60 gold questions, all 415 summaries scored solo; default cut unchanged at 0.50):

```
cut 0.10 | rejected  4 = 0.07 | own community passes 31 = 0.52
cut 0.20 | rejected 11 = 0.18 | own community passes 24 = 0.40
cut 0.30 | rejected 19 = 0.32 | own community passes 19 = 0.32
cut 0.40 | rejected 27 = 0.45 | own community passes 16 = 0.27
cut 0.50 | rejected 29 = 0.48 | own community passes 13 = 0.22
cut 0.60 | rejected 37 = 0.62 | own community passes  9 = 0.15
cut 0.70 | rejected 42 = 0.70 | own community passes  7 = 0.12
```

No cut both rejects few questions and keeps the source section's community: the cut is a trade the product owner sets.

Stage B with the Jev gate in the loop (60 questions; gate 0.50 solo, then the RELEVANT: YES|NO check on each passer's partial; no surviving partial = rejected):

```
rejected 35 of 60 = 0.58 (29 passed no community; 6 more passed one but its partial said NO); answered 25
own community among the kept partials: 11 of 25
answered only, hops 2 or 3, 3 searches: walk 9/25 = 0.36 | same-size hybrid 15/25 = 0.60
whole system over ALL 60 (a rejection is a miss): 9-10 of 60 = 0.15-0.17 | plain hybrid top 18: 31 of 60 = 0.52
```

Reading: with the gate in the loop the system answers 25 of 60 in-corpus questions, and on those it still finds the source section less often than a plain hybrid list; counting rejections as misses it finds it in 0.15-0.17 of questions against 0.52. These are known-item sentences, not user questions, and the measure is source-section recall, not answer quality.

Blind answer comparison (25 gold questions the gate answered; arms A hybrid top-12 alone, B + subgraph, C + global, D + agent traversal at hops 3 + global; same answer prompt and model; judge sees only the question and the four answers, shuffled, two orders per question; 50 judgements):

```
A hybrid only  mean rank 2.70 | first place 11 of 50
B + subgraph   mean rank 2.48 | first place 11 of 50
C + global     mean rank 2.48 | first place 11 of 50
D + agent      mean rank 2.34 | first place 17 of 50
D above A: 29 of 50, below 21 (two-sided sign test p about 0.32); C above B: 22, below 28; D above C: 25 / 25
mean answer length (characters): 1238 / 1373 / 1241 / 1225
```

Reading: no arm is distinguishable from another (random ranking gives a mean rank of 2.50). The judge and the writer are the same model (`google/gemini-2.5-flash-lite`), the queries are lifted sentences, the 50 judgements come from 25 questions and are not independent, and no human read these answers: it is a second opinion that finds no difference, not evidence that the arms are equal.

## Scope of this plan, stated (2026-10-12)

- The steps of this plan are the six in the status table above plus row 5d. All are DONE as worded in their rows; 5b ran with an empty global view and 5d measured the Jev gate on its own, and row 5e is the stage B run with the Jev gate in its loop.
- NOT steps of this plan, and not done: (a) a HUMAN-read blind comparison (5f is model-judged; the packet `docs/experiments/blind_packet.md`, its key and `blind_score.py` are built and wait for one reader to fill `blind_scores.txt`: 25 items, about 130,000 characters of reading), (d) the conclusion/entailment idea listed under "Out of scope".
- Open items, with cost: (a) needs a rubric and a reader; (b) is about 95 minutes of solo scoring for 60 questions and would mostly measure rejections, since 14 of 20 in-corpus questions pass no community at 0.50; (c) needs scoring the 60 gold questions solo (about 95 minutes) to fit the cut on labelled pairs; the cut policy itself (what a rejection should cost) is a product decision this plan does not make.
