# Superseded context display — 2026-10-23

## Superseded interpretation pass and prompt attempts

## Current applied critique — 2026-10-23

Behavioral change: use canonical API source URLs to group/count evidence in the self-contained OWUI Function; fall back to document IDs with `_methods` removed. Grouping changes presentation and counts only, never source ordinals, retrieval ranking, or passage content. Preserve source order within each paper group and show the methods-source variant explicitly. Extend `_details(title, language, content)` so `language=None` carries Markdown rather than a fenced diagnostic. Emit a newline after `<details>` and after `<summary>...</summary>` to satisfy the installed tokenizer. Use the incumbent adaptive-fence logic inside source groups.

The overview and subgraph are direct Mermaid sections so their diagrams survive normal Markdown copying; agent trace and gate scores remain expandable. Show seed reweighting only when selected terms exist. Extend existing pipeline/agent instructions to carry explicit source-defined criteria into comparisons, distinguish executor/harness/improver and parameter-changing boundaries, and separate demonstrated findings from proposals. Outlet displays pipeline synthesis directly; the mandatory final generation only acknowledges completion. No per-query hardcoded paper names, axes, or new LLM stages.

Root gates: verify three disclosure fixtures in the real browser before publishing code; verify canonical counts and exact-source preservation offline; test three comparison mechanisms with small synthetic sources; then deploy, restart the API for its prompt changes, and run the exact harness query. Preserve the old user answer as a sibling branch when refreshing their chat. Record current behavior and archive superseded display conclusions in deployment history.

The first comparison gate exposed axis labels mistaken for method names and an invented online timing cell; preserve explicit criterion names as table columns, actual alternatives as rows, and mark missing values not reported. Re-run the same three fixtures after this prompt repair.
The second run preserved method distinctions but renamed axes; replace the accumulated prose prompt with concise rules and a final explicit table contract, retaining the incumbent citation/taxonomy safeguards. No extra model stage or model retry is introduced.
Fixture disposition: terse prose saying "we compare Artifact, Supervision, Persistence" remained ambiguous and did not model the real source's explicit comparison table. Use fully specified synthetic source tables for the retention gate, preserve prior failures, and explicitly check that missing timing is not invented. These checks establish table retention on clear inputs, not universal factual correctness of model interpretation.

Live disposition: the hosted final model asserted that distillation was excluded despite the source passage. A frozen-evidence replay also omitted that boundary, so this is synthesis scope selection, not missing retrieval. The answer contract must explicitly name relevant boundary methods outside a source's comparison table before making universal claims; validate against the frozen 19 passages before repeating live retrieval.
Boundary-prompt replay still omitted the lone distillation paper. Add a canonical-paper coverage inventory to incumbent answer_prompt and require a cited additional-approaches row for relevant papers absent from the main table. Display report.final directly in outlet, as the user requested the pipeline's LLM response; inlet requests only a short acknowledgement for the framework's mandatory generation call. This removes the failed second interpretation pass without changing Function type, selected pipeline model, API interface, or retrieval. Test report.final precedence against string/multipart/empty downstream content before deployment.

Superseded context-display design and browser conclusions: [2026-10-23 history](superseded_context_2026-10-23.md); the current design uses canonical source groups and the verified disclosure grammar.

## Prior interpretation decisions

- 2026-10-23: Use native Markdown source blocks rather than arbitrary HTML wrappers: the actual OWUI browser printed `<details open>`/`<pre>` literally. Three rendered fixtures preserved citations, HTML text and code fences; the final hosted query displayed 19 source blocks before commentary with `content` and `output` synchronized. Basis: browser DOM/code-editor inspection and `.tmp/owui_context_visible.png`; superseded attempts are in deployment history.
- 2026-10-23: Interpret direct retrieved-context display as showing the API's exact passage text before model commentary, expanded by default; preserve the selected model's ability to interpret the pipeline draft and evidence. Basis: the user explicitly wants to read source text without relying on model transcription; the incumbent outlet already appends those passages but collapses them below the answer.

Superseded by the current applied-critique contract: the client supports disclosures when the opening tags occupy separate lines, and `_methods` variants share canonical paper identity. The former nine-paper count represented document IDs; the same saved evidence contains eight canonical papers. Preserve the prior observations and contracts below as history.

## Prior requirements

Current context-display extension 2026-10-23:

Current acceptance closure: 35 focused tests passed (three skipped). Function 0.7.2 is deployed and AST-equal to the repo with activation/toggle/valves preserved. The final hosted retrieval completed in 90.0 seconds with 19 source blocks from nine papers; structured output equals content. Browser inspection verified all 19 passage headings before commentary, source text in native code editors, and no literal HTML wrappers. Evidence: `docs/owui_refs/deployment_current.md`, chat `eca1ab62-5e34-46e5-8e5f-4757655f9c7e`, screenshot `.tmp/owui_context_visible.png`.

Browser evidence revised the first implementation: the saved response contained 19 expanded cards, but the actual page displayed zero details elements because Open WebUI rendered `message.output` instead of modified `message.content`. The installed outlet middleware forwards and persists both fields, so both must be synchronized. The 97.4-second saved-content check is insufficient as UI acceptance.

Browser evidence after output synchronization: the client displays arbitrary HTML `details open` and `pre` as literal markup. Replace passage HTML cards with ordinary Markdown headings and literal `text` fences; the client already renders its TOON/code blocks correctly. This supersedes HTML-based expansion as the display contract.

- When the Function renders an accepted response, it SHALL insert every API evidence passage before the model commentary.
- When a passage is rendered, it SHALL use the API's exact returned text inside a visible Markdown `text` fence under its source heading.
- When a source contains backticks, the Function SHALL use a fence longer than any backtick run in the source.
- When the model receives retrieved context, the Function SHALL instruct it to discuss the evidence rather than reproduce the displayed passages.
- If the model produces empty text or a multipart response without a text part, the Function SHALL still render the retrieved context.
- When outlet runs again on an already rendered message, it SHALL preserve the passage bytes and produce no duplicate context or commentary headers.
- When the relevance gate rejects a question, the Function SHALL preserve the rejection notice and return no invented passages.
- The Function SHALL preserve source links, truncation notices, entity/community metadata, and diagnostic cards.
- When an assistant has structured `output`, the Function SHALL synchronize its final assistant message's `output_text` with the rendered `content` while preserving reasoning and tool records.
- If structured `output` lacks an assistant message, the Function SHALL add one carrying the rendered source and commentary.
- Given the hosted chat page, When it loads in the browser, Then every source passage SHALL appear in a rendered code block before model commentary, with no literal HTML wrapper tags.
- Given literal citations, HTML, and code fences in source passages, When outlet runs twice, Then the fenced passage text SHALL equal the original source text in all three cases.
- Given string, text-list, and image-only model content, When outlet runs, Then retrieved context SHALL precede separately labeled model commentary for all three shapes.

Spec validation: Behavioral rendering extends the incumbent renderer/outlet; no retrieval, storage, model-selection, or API change is required. The module's rendered R4/R7/R10 guards own this display contract; acceptance covers ordering, source preservation, absent model text, idempotence, and rejection.

Saved-content gate 2026-10-23: 32 focused tests passed (three skipped); the deployed AST matches the repo and activation/toggle/valves were preserved. The 97.4-second saved completion proved content insertion but failed the browser gate; structured-output synchronization and browser verification are required for closure.

## Prior design

## Current context-display extension — 2026-10-23

Acceptance closed: 35 tests passed (three skipped); all three native-Markdown browser fixtures retained source values. The final 90.0-second hosted completion synchronized `content` and `output`, and the browser displayed all 19 source headings/code blocks before commentary with no HTML wrapper artifacts. Current deployment record and screenshot document the client-visible result. No retrieval/API behavior changed.

Behavioral layer: extend `_render_visible_retrieval` with one Markdown source heading and adaptive text fence per passage; `outlet` prepends that source view to separately labeled model commentary, with diagnostics afterward. The raw source is preserved directly; citation rewriting applies only to model commentary. A generated-block marker makes repeated outlet calls idempotent. Consolidate the existing duplicate string/list render branches; create a text part when a multipart response has none. The inlet tells the selected model that source passages are displayed independently and requests interpretation without copying them. The API/pipeline answer remains available to the model as its grounded draft.

Browser-discovered handoff: Open WebUI renders the structured `output` field; its installed `outlet_filter_handler` forwards a deep copy of that field and persists changes. Synchronize the final `type=message` assistant item's `content` with the fully rendered text using `type=output_text`, preserving other reasoning/tool items and nontext message parts; add an assistant message if absent. Perform this synchronization even when legacy `content` already has the generated marker. Browser DOM counts/open state and source-before-commentary order are the final gate.

Display pivot after browser evidence: arbitrary HTML passage wrappers are displayed literally by this client. Use the incumbent Markdown/code-block path instead: one `#### Passage:` heading and a `text` fence per source, sized above the source's longest backtick run. Preserve source bytes directly, without HTML escaping. Count rendered source headings and code editors in the browser, confirm their values equal stored API passages, and confirm they precede commentary. Graph diagnostics retain their existing renderer.

Verification: three varied source strings and three model content shapes, repeated rendering, truncation and rejection, then deploy the existing Function via its authenticated API and inspect a saved hosted completion for open passage cards before the commentary. Preserve activation, toggle, model attachments and deployment valves.

Saved-content check 2026-10-23: the initial 32 passing tests and 97.4s chat proved content insertion, but browser DOM inspection found zero visible cards because structured output was unchanged. Added R11 synchronization and three varied output-record cases; 35 tests now pass (three skipped). Sprawl review: collapsed the two string/list outlet branches into one rendering path.
