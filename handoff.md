# Handoff prompt: repair community-summary reranking and regenerate the map

Continue the in-progress GraphRAG community-summary selection work in this repository. Read the current code and artifacts first, then diagnose the observed bad result, make the smallest principled correction, regenerate the PNG, and report the full absolute paths of both this handoff and the updated PNG.

## User's goal and design

The user wants every eligible global community summary evaluated for relevance to each query, then partial summaries generated from all summaries classified as relevant. The point is to avoid selecting a small sample of global communities that can omit evidence (for example, the earlier missing latent-memory evidence), without putting all 800+ communities into one huge prompt and risking lost-in-the-middle/partial-summary behavior.

Use the installed Jev-Style 0.8B model for pair decisions. Batch 64–128 community summaries, with one GPU batch active and at most one next batch prepared in a bounded CPU queue (semaphore/queue depth 2). Score the complete eligible set before generating any partial summaries. The user had previously shelved training a smaller model and explicitly chose the 0.8B model for now.

## Important current finding: do not call this “zero communities”

The latest image and saved JSON show a **selection/semantic failure**, not zero selected items:

- The latest memory-query image header says 415 community summaries were scored and 3 passed the 0.50 threshold.
- `.tmp/sections_xpa_queries.json` has 3 queries. For query 1 (speculative decoding), **7** communities passed; for query 2 (LLM-agent memory taxonomy), **3** passed; for query 3 (LLM safety evaluation), **6** passed.
- Despite those scores, the generated partials for nearly every selected community say that the community is irrelevant and cannot contribute. The aggregate/global answer then repeats those rejection sentences. Query 1 has a possible exception: community 754's partial describes engineering organisms/self-healing agents, which is also off-topic. The query 2 selected communities 745, 754, and 755 all explicitly reject relevance. Query 3 selected communities likewise reject relevance.
- Therefore, do not describe the state as “no communities passed.” The actual defect is that high-scoring communities are being routed as relevant even though the partial-answer model judges their summaries irrelevant (or the score interpretation/response mapping is inverted/misparsed). The generated “partial summary” prompt currently permits a refusal-style irrelevance statement to count as a partial, and the reducer treats it as evidence.

Inspect raw score records and runtime responses, including what score means (which typed NOUL outcome is being measured), the exact system prompt and labels, and the mapping from response fields to `score`. Determine whether the cutoff uses P(relevant) versus P(irrelevant), whether these are calibrated as independent pair scores, and whether summary IDs/scores/partial results remain aligned through batching. Do not assume the model or the threshold is the only cause. Use the actual underlying summaries for the selected community IDs to verify the judgment.

## Relevant files and artifacts

- Main orchestration: `src/section_graphrag.py`, especially the global community-summary scoring/partial-answer path near `answer_global` (search for `score_communities`, `community_threshold`, and `PARTIAL_PROMPT`).
- Scoring integration: `src/community_reranker.py`.
- Rendering and query JSON: `src/section_query_panel.py`.
- Community data access: `src/section_store.py`.
- Existing design/training playbook: `docs/playbooks/community-reranker.md`. Training is shelved; use the 0.8B scorer.
- Current score/result artifact: `.tmp/sections_xpa_queries.json`.
- Latest generated PNG (the one the user says is wrong): `docs/section_map/community_map.png`.
- Previous pre-JEV PNG: `.tmp/sections_xpa_queries_prejev_20261012.png`.
- Query panel historical outputs and logs are under `.tmp/` (see `sections_xpa_queries_community_map*.png`, `panel_rerun*.log`, and `summ_xpa_all.log`).
- Latest screenshot from user shows query 2, the memory taxonomy query. It includes the 415-scored / 3-passed banner but selected communities' partials say they are not relevant.
- This is the Windows workspace `C:\Users\user\Documents\dev\graph`. A previous Claude transcript refers to the same project under its former `F:\Documents\dev\graph` path; do not blindly use stale drive paths.

## Conversation-history context

The relevant Claude transcript is:

`C:\Users\user\.claude\projects\F--Documents-dev-graph\10bbda5a-0b15-4f78-afff-a440294a33e9.jsonl`

It contains earlier GraphRAG design context: global communities should inform querying; concern that selecting examples/samples instead of using all communities can lose evidence; Microsoft GraphRAG's all-community map-reduce also risks partial summaries; and the desired alternative is score all eligible community summaries and create partials only for the selected relevant ones. The conversation emphasizes holistic principles rather than ad-hoc extra reads for known missing taxonomies. Relevant recent interactions in this current session further specify the 0.8B JEV all-community scoring, bounded 64–128 batches with queue/semaphore capacity 2, partial summaries, and delivery of the revised PNG.

The user has also explicitly corrected the prior handoff style: they expect the **full absolute paths** in the final response (they have repeatedly asked for the updated PNG and say they always ask for full paths). Include both the generated PNG and this `handoff.md` path.

## Workspace state at handoff

Uncommitted changes already exist; preserve them and inspect diffs before editing:

- Modified: `README.md`, `docs/section_map/community_map.png`, `src/section_graphrag.py`, `src/section_query_panel.py`, `src/section_store.py`.
- Untracked: `docs/playbooks/`, `src/community_reranker.py`, `termgraph_disposition_2026-10-08.md`, `termgraph_plan_2026-10-08.md`.
- No commit was made for this current reranking work.
- Do not overwrite the previous PNG backup.

## Work to do

1. Inspect the complete code path, current git diff, scoring artifact, and runtime configuration. Identify exactly which 415 summaries are eligible and how the 0.8B model's typed decisions become scores.
2. Diagnose why high-scoring pairs produce irrelevant partials. Check prompt shape, class/outcome polarity, probability extraction, batch alignment, candidate input formatting/truncation, model identity/version, and whether the score is candidate-relative rather than independent. Compare known relevant and irrelevant community pairs manually using the same runtime if useful.
3. Make a principled fix. Preserve all-community scoring and bounded batching. A selected community must not create a “partial summary” that merely says it is irrelevant. Ensure zero selected communities is represented honestly, and ensure the reducer only sees genuine content-bearing partials. Keep score provenance and all per-community scores in the JSON for audit.
4. Regenerate the three query answers and the PNG. Check that the speculative-decoding and memory queries recover the relevant evidence (including latent memory if it is in an eligible community), that safety has distinct evaluation protocols/metrics versus interventions, that citations/section anchors remain valid, and that the diagram distinguishes the score from the generated partial. If scorer quality remains inadequate, do not hide it with a threshold tweak; report the failure and use a transparent fallback/ranking strategy that still makes the candidate set auditable.
5. Run only the verification needed by the user's requested deliverable; do not launch an unnecessarily broad suite. Do not commit or push unless the user asks.
6. Final response: briefly state the cause and correction, summarize verification, and end with full absolute Windows paths to the final PNG and `handoff.md`.

## Important constraints

- Preserve user changes; do not reset or clean the workspace.
- Do not claim the reranker works merely because calls completed or scores exceed a numeric threshold. Inspect the selected summaries and resulting partials.
- Do not turn “3 passed threshold” into “3 relevant”; current artifacts demonstrate these concepts disagree.
- Prefer recovering the full evidence set over maximizing precision if those objectives trade off. Any fallback or score threshold must be stated and traceable.
