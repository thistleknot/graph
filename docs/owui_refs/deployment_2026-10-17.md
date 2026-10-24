# Open WebUI deployment record

## Superseded second interpretation pass — 2026-10-23

Version 0.8.0 restored canonical source groups and direct Mermaid rendering. Its hosted query completed in 223.9 seconds with 19 passages/eight canonical papers, but the second model falsely asserted that parameter changes/distillation were out of scope despite the HARNESS-ZERO experiment passage. Assistant branch `57da115e-f54e-45ce-867c-91390435fbeb` remains preserved. Frozen replay also omitted that lone paper until the final prompt added a canonical-paper inventory and required additional-approaches coverage. Version 0.9.0 displays API synthesis directly and requests only an acknowledgement in the framework generation step; the final hosted/browser validation is in the current deployment record. The prior false output is preserved locally at `.tmp/critique_comparison_content.md`; prompt fixtures/replays are `.tmp/verify_comparison*.json` and `.tmp/critique_pipeline_replay*.json`.

## Superseded inline-only display and document counts — 2026-10-23

The installed client's disclosure tokenizer requires opening tags on separate lines; the one-line tag serialization caused the apparent lack of support. Eight canonical arXiv URLs occur in the prior report's nine document IDs. Function 0.8.0 restores disclosures and canonical grouping; source records remain distinct. Prior deployment observations follow.

- Actual-user-chat recovery: `120e55ce-d5d8-4e76-9674-1536401c9379` originally returned a Python harness skeleton with no retrieval status or source view. Regenerated the same user turn with explicit `filter_ids=["raxiv_graphrag"]`; the selected assistant branch is now `f21f61d1-0578-4306-8a24-69efab74b962`. It completed in 104.1s with 19 source passages from nine papers, synchronized client output, and research commentary with 11 paper links and no Python skeleton. The original answer remains as a sibling branch. Full prior chat backup: `/root/.cache/arxiv-filter-backups/20261024T021202Z-chat-before-retrieval.json`. [Open the recovered chat](http://192.168.3.18:8080/c/120e55ce-d5d8-4e76-9674-1536401c9379). Model attachment alone does not establish that a particular request invoked the opt-in Function.
- The hosted `raxiv_graphrag` Function uses `src/owui_filter.py` version 0.7.2: exact API passages precede **Model commentary** in visible Markdown text blocks, with links and truncation notices. It synchronizes legacy `content` and client-rendered structured `output`, preserving reasoning/tool records. Passage text is inserted by Python; the model is asked to interpret it without reprinting it.
- The previous Function/valves are backed up at `/root/.cache/arxiv-filter-backups/20261024T013531Z.json` (UTC). Authenticated deployment confirmed active=true, global=false, toggle=true, API URL and 600-second Filter timeout preserved. The Function is already updated on `.18`; no manual re-upload is required.
- Focused rendering validation passed 35 tests (three skipped), covering exact text, empty/multipart responses, repeated rendering, rejection, ordering, and structured-output synchronization. Browser fixture `02b3d638-99ec-4715-a01e-1f6a98ca0111` displayed three source headings and three code editors, preserving literal citations, HTML and embedded code fences, with commentary afterward and no literal wrapper tags. The two duplicate string/list render branches were consolidated into one path. Superseded content-only and HTML-wrapper attempts are preserved in deployment history.
- Final live/browser gate passed: [context-first verification chat](http://192.168.3.18:8080/c/eca1ab62-5e34-46e5-8e5f-4757655f9c7e) completed in 90.0 seconds (`done=true`, `finish_reason=stop`), with 19 source blocks from nine papers and 6,130 model-text characters. The Filter reported 49.6 seconds for its graph response. Persisted structured output equals rendered content; the actual browser showed all 19 source headings before **Model commentary**, 23 native code editors including metadata/diagnostics, visible passage text, and no literal HTML passage wrappers. Screenshot: `.tmp/owui_context_visible.png`.

Prior API, completion and Function dispositions: [deployment history](deployment_2026-10-17.md).

## Superseded HTML passage wrappers — 2026-10-23

Function 0.7.1 synchronized `content` and `output`; the hosted query completed in 98.0 seconds with 19 passages in both fields. Browser inspection confirmed the source text reached the page, but the client displayed `<details open>` and `<pre>` literally (19 passage labels, zero native details/pre elements). Version 0.7.2 switches only the source view to standard Markdown headings and adaptively sized text fences, whose rendered values were verified on three varied browser fixtures. Backup before that change: `/root/.cache/arxiv-filter-backups/20261024T013531Z.json`. The earlier output-synchronization-only backup was `/root/.cache/arxiv-filter-backups/20261024T013031Z.json`.

## Superseded content-only display — 2026-10-23

Function 0.7.0 prepended 19 expanded passage cards to `message.content`; a saved chat completed in 97.4 seconds, but a browser inspection found zero visible passage cards because this OWUI client renders `message.output`. Installed middleware forwards and persists that field separately. This invalidated the saved-content-only acceptance test. The previous Function backup was `/root/.cache/arxiv-filter-backups/20261024T012448Z.json`; source/output synchronization is now required by R11 and the current deployment record. Reproduction chat: `5a795311-a4a5-4809-aa77-2bb326d0ab7f`.

## Prior API and Function disposition — 2026-10-22

- Graph API is running from `F:\Python\mcp\graph\Scripts\python.exe` on `192.168.3.122:8780`, serving PID 461272 (recorded in `.tmp/arxiv_rag_api_timeout.pid`); `/health` reports graph build 1163, BM25 build 3, and source graph build 1163.
- The user's 600-second harness-design timeout is repaired: internal Qwen task calls disable optional reasoning, reject empty/unfinished text, and share a configurable 180-second request deadline. A matched action probe took 31.62 seconds and returned empty `finish_reason=length` output with default reasoning, versus 2.52 seconds and a completed action with reasoning disabled. Final Open WebUI generation retains its model settings. No Function source change or re-upload is required for this repair.
- The former Function deployment backup was `/root/.cache/arxiv-filter-backups/20261022T171955Z.json`; it is superseded by the 2026-10-23 context-display update above.
- The Function's dynamic `selected_terms` multiselect received 1,719 live choices from `/community-terms`; the selected-term retrieval path accepted a valid term, ignored an unknown term, and applied nonzero community affinity to 23 fused candidates.
- The exact failing query, `recursive self improvement agentic harness design`, completed through the hosted Function in 122.8 seconds: 77.1 seconds for retrieval plus agent synthesis, 4,792 assistant-text characters, `finish_reason=stop`, persisted `done=true`, one graph summary card, 19 passage cards, and arXiv links from nine papers. [Open the verified chat](http://192.168.3.18:8080/c/70e15c7e-c6a3-431a-bf04-a951a0b0c1b5). API stages: retrieval 1.92s, global view 18.33s cumulative, traversal 62.73s cumulative, complete 75.30s cumulative; connection/setup and response formatting account for the difference from the Filter's 77.1s.
- Focused checks: 111 passed, 3 skipped, 5 deselected; a subsequent deadline-propagation check passed 13 tests. Three live structured Qwen task calls completed in 1.33–3.33 seconds; results are `.tmp/agent_timeout_model_battery.json`. API stage/model logs are `.tmp/arxiv_rag_api_timeout.stderr.log` and access logs are `.tmp/arxiv_rag_api_timeout.stdout.log`. The hosted verification checks visible completed text as well as cards and links.
- A separate direct Gemini `/agent` probe returned HTTP 429; the Qwen request from the hosted Function returned HTTP 200.
- BM25 synchronization uses the current-user logon task `arxiv-bm25-watch`, registered by `scripts/register_arxiv_bm25_task.ps1`; the task is running under `F:\Python\mcp\graph\Scripts\python.exe`, and its initial check confirmed BM25 matches graph build 1163. It polls every 60 seconds and logs to `.tmp/arxiv_bm25_worker.log`.

## Superseded completion disposition — 2026-10-22

The memory-topic Filter verification completed in 213.6 seconds: 18 evidence sections from 17 papers retrieved in 182.4 seconds, one graph summary card, 18 passage cards, and arXiv links persisted with `done=true`. The explicit final-generation token cap in the first headless attempt was removed. This success did not cover the later harness-design query, which timed out after 600 seconds. Matched probes then identified a separate internal Qwen action failure: all 1024 completion tokens were consumed with no visible action. The repaired request path and the exact 122.8-second hosted regression are documented in [deployment_current.md](deployment_current.md).

## Prior deployment — 2026-10-21

The hosted `raxiv_graphrag` Function existed and was active, but its 29,118-character source predated the passage-forwarding contract. The model `qwen/qwen3.6-27b` already referenced its filter ID. Updated the Function through Open WebUI's authenticated `/api/v1/functions/id/raxiv_graphrag/update` route with the current `src/owui_filter.py`; post-update inspection confirmed active=true, global=false, and toggle=true. The previous source is backed up at `/root/.cache/arxiv-filter-backups/20261021T-filter-before-evidence-update.json`; Open WebUI also retained its function version history.

The API at `192.168.3.122:8780` returned passage-bearing evidence rows. A saved authenticated Open WebUI chat with `filter_ids: ["raxiv_graphrag"]` completed in 475.3 seconds. Its persisted assistant message has 74 passage cards, `entities[6]`, `communities[6]`, and How this was answered, Subgraph, Agent trace, and Gate scores cards; retrieval reported 74 sections from 39 papers in 411.9 seconds. [Open the verification chat](http://192.168.3.18:8080/c/dee2641b-2822-408b-8747-87b439338ab2). The graph API health endpoint returned HTTP 200 from the Open WebUI host. The current filter passes graph evidence to the model and appends the retrieval report in its outlet.

## Prior deployment — 2026-10-17

## Prior deployment status — 2026-10-17

The `raxiv_graphrag` function on `192.168.3.18:8080` was updated through its authenticated API;
its graph endpoint is `http://192.168.3.122:8780` and its timeout is 600 seconds.
The previous function and valves are backed up on `.18` at
`/root/.cache/arxiv-filter-backups/20261018T003947Z.json`.
Focused verification: 22 offline checks passed, including HTTP failures, timeouts, invalid reports,
status emission, and duplicate/bypassed calls.
Live result: [arXiv graph verification](http://192.168.3.18:8080/c/666a1c3a-112d-45b2-bdb8-30dfdd95ab26)
completed through `.18`'s authenticated chat API in 116.7 seconds: retrieval took 99.8 seconds and
returned 18 evidence sections from 13 arXiv papers; the saved answer contains paper citations and
one set of graph cards. Both pending and terminal statuses are persisted in the chat.
The selected model remained `qwen/qwen3.6-27b`; retrieval parameters were unchanged.

## Superseded deployment notes

The earlier “port refused connections / hosted deployment unverified” disposition is superseded by
HTTP 200 from `.18` to `.122`, authenticated inspection of the installed function, and the API deployment.
The assumption that the graph API belongs on the Open WebUI host is superseded by the operator's
confirmed `.122` graph host. Earlier text is retained below for provenance.

### Previous README section

## Open WebUI arXiv Filter Function

The in-process Filter calls the arXiv API's `POST /agent` route from its `inlet`; its `outlet` adds the agent trace,
section subgraph, entity/source TOON tables, and gate scores in expandable cards. The selected Open WebUI model in
`body["model"]` is passed to the API and remains selected for the final response. Background prompts beginning
`### Task:` bypass retrieval.

Run the API on the Open WebUI host with the graph database, `OPENROUTER_API_KEY`, and section-model assets available:

```powershell
python -u src/arxiv_rag_api.py --serve --host 127.0.0.1 --port 8780
```

The API exposes `POST /agent` alongside whole-section `GET /search`. At startup it checks the live build and warms the
dense query encoder and cross-encoder before reporting ready; restart it after updating `src/arxiv_rag_api.py`.
In Open WebUI, import `src/owui_filter.py` under **Admin Settings → Functions** and leave it active but non-global.
Attach it to the intended model under **Workspace → Models → Edit → Filters**. The Filter sets `self.toggle = True`,
so you can turn it on for a chat from the Integrations menu or its input-bar chip; the model's **Start enabled in new
chats** setting controls its initial state. Global filters are forced on, so leave the Global switch off for opt-in use.
The default `rag_api_url` is `http://192.168.3.122:8780`, where the graph service runs; make sure that address is
reachable from the Open WebUI server at `192.168.3.18`. `api_key` forwards a bearer token for an API deployment
protected by a reverse proxy.

Open WebUI `0.11.4` was reachable at `192.168.3.18:8080` during this integration check. The graph host's port `8780`
refused connections, so the new `/agent` route has not been verified on that deployment. The authenticated
`/api/v1/tasks/queries/completions` endpoint is not used by the Filter.


### Previous playbook entry

## Superseded handoff — T176 — Open WebUI global arXiv Filter Function (2026-10-17)

- [IN PROGRESS] Deliver an importable global Filter for the hosted Open WebUI server, with graph-agent answer and expandable Mermaid/TOON evidence cards; no chat bridge or Pipelines server.
- _Files:_ `src/owui_filter.py`, `src/arxiv_rag_api.py`, `tests/test_owui_filter.py`, `tests/test_arxiv_rag_api.py`, `README.md`
- _Verify:_ focused tests; headless Mermaid parse; deploy `/agent` to `192.168.3.18:8780`; import the Filter in Open WebUI `0.11.4`; run a real query and inspect all cards.
- _Status:_ code and local checks are complete (33 tests; three TOON decoder round trips; 15 fixture Mermaid diagrams parsed; real build-32 `/agent` query and report rendering passed). The hosted API port `8780` refused connections, so deployment, function import, and hosted UI rendering remain open.


