# Playbook: Sqrt anchors, neo4j mirror, entities v0
Source: operator-approved queue, 2026-09-03 (previous campaign fully DONE, in git history)

Settled design (do not relitigate): (1) per-source BM25 anchor allocation
k_anchor_s ∝ n_s^0.5 over gt.run_sources, anchors selected per-source then
merged, R1 validation unchanged, single-source runs byte-identical to today —
measured basis in sampler.py's DEFAULT_K_ANCHOR comment (global k=9 lifted B
partway but broke C3's brown floor 30%→19%; a global count cannot fix it).
Tuning target is the FROZEN diagnostic set (.spec/specs/graph-explorer/
diagnostic-prompts.md): B-class quotes ≥ 40% of the ef24 walk on
mixed-full-dual (sparse run proves attainable at 40–53%) without breaking any
passing A/C/D/E row. (2) neo4j export moves to the modern 4-file layout
(chunks/terms/contains/similar.csv — dual-ID single-file header is rejected by
current neo4j-admin; the scratch splitter imported 286,158 nodes / 6.46M rels
in 42s with --multiline-fields=true). (3) Entities v0 is schema + builder over
the EXISTING salient/phrase vocabulary — deterministic NLP only, no LLM, no
NER dependency (real NER is LATER). NPMI floored at joint count ≥ 5 chunks.
(4) Titles are persisted node metadata (R20's law extended: parse once at the
boundary), degrade-safe on runs that predate it. Article IX: every spec
amendment lands before its code task opens.

Amendment 2026-09-03 (fable re-scope on T7's measured evidence): item (1)'s
sqrt reallocation is SUPERSEDED — it capped the majority source and broke
A1/E3 while moving B the wrong way. Amended design in design.md §6.15
(S15/S16 amended, S17 new): anchors are strictly additive (global top-k
untouched, minority sources get at most one EXTRA competitive anchor), and
the source-aware ring share is promoted from LATER as the actual B lever.
Everything else in the settled design stands.

## Layer 1 — sequential
- [DONE] T1 Amend the spec: sqrt-anchor S-guard, title payload contract, entities v0 schema
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "sqrt\|title\|entities" .spec/specs/graph-explorer/design.md
  _Notes:_ Three amendments in ONE pass, drafted verbatim so Layer 2 tasks
  paste them into docstrings: (a) new sampler S-guard — k_anchor_s ∝ n_s^0.5
  per source in gt.run_sources, per-source BM25 selection then merge, R1
  validation unchanged, single-source ("default") degenerate path byte-identical
  to today, tuned ONLY against the frozen diagnostic set's do-no-harm rule;
  (b) §6.14 payload contract gains `title` (from ingest_mixed.wiki_title(),
  parse once at ingest, absent key = degrade to doc_id display, never None);
  (c) entities v0 tables beside the existing store: entities(entity_id, name,
  type) / mentions(chunk_id, entity_id, cnt) / entity_edges(a, b, npmi, ppmi,
  bm25), NPMI floored at joint count ≥ 5 chunks (rare-event bias), entities v0
  = existing salient/phrase vocabulary, type = "term_v0". State each amendment's
  acceptance as a testable line.
  _Lessons:_ Guards minted S15/S16 (sampler, sqrt-allocation + starvation/forfeit
  rules), R22 (chunkgraph, extends R20 with `title`, absent-not-null contract),
  E1-E5 (new entities.py store: entities/mentions/entity_edges, run-scoped,
  MIN_JOINT_CHUNKS=5 floor, npmi/ppmi/bm25 co-resident). §6.14 body and R19-R21
  left byte-untouched — Layer 2 pastes guard bodies verbatim (T2/T6 -> sampler.py
  after S14, T6 -> chunkgraph.py after R21, T4 -> new entities.py) with no
  re-derivation needed.

## Layer 2 — parallel
- [DONE] T2 Per-source sqrt anchor allocation in the sampler (THE priority)
  _Files:_ sampler.py, tests/test_sampler.py
  _Verify:_ pytest tests/test_sampler.py -q
  _Notes:_ Implement T1's S-guard text exactly; paste it into sampler.py's
  docstring. Sources from gt.run_sources; k_anchor_s ∝ n_s^0.5 normalized to
  total k_anchor, per-source gt.search then merge, dedupe, R1 validation
  unchanged. Single-source runs MUST be byte-identical to today (regression
  test, the "default" degenerate path). Keep scope minimal: NO source-aware
  ring share in this task — quotes chunks arriving via ring top-offs is a
  measured dilution path, but the diagnostic (T7) decides whether ring work is
  needed; if T7 fails on anchors alone, the ring share is T7's escalation
  evidence, not a preemptive build here. Tests follow test_sampler.py's
  conventions (pure-selection tests need no DB; pipeline tests skip cleanly
  without one).
  _Lessons:_ T7's diagnostic re-run should know anchors are now allocated per
  source (sqrt share, largest-remainder + zero-fix, S15/S16) via
  `sampler.anchor_hits`/`allocate_anchors`/`select_anchors` in sampler.py, wired
  into `ef_search` at k_anchor=3 (unchanged) — DEFAULT_K_ANCHOR stays 3, only
  WHERE the 3 anchors come from changed; single-source and pre-R20 runs are
  byte-identical to before (regression-tested against the live DB, not just
  mocked), so any prompt-level shift T7 sees is attributable to the allocation
  itself, not to a default change. Live pipeline tests 15-17 ran against the
  actual DB (not skipped) and passed, confirming behavior on real data, not
  just the pure-function unit tests.

- [DONE] T3 export_neo4j: 4-file layout, embeddings column, vector index, C<cid> labels
  _Files:_ export_neo4j.py, tests/test_export_neo4j.py
  _Verify:_ pytest tests/test_export_neo4j.py -q
  _Notes:_ Fold the proven scratch splitter into the exporter: chunks.csv,
  terms.csv, contains.csv, similar.csv; update import.sh it emits (add
  --multiline-fields=true); pin with a header-shape test (the dual-ID
  single-file header is the measured rejection). chunks.csv gains a 256-dim
  embedding column from node_embedding plus an emitted cypher file with the
  CREATE VECTOR INDEX statement; C<cid> labels carried into import via a
  :LABEL column (preferred) or the post-import cypher file — subplanner's
  call, but the label must reproduce the live-demonstrated state (10,820
  labeled). Existing X1–X5 guards stay; keep the source column and R20
  degrade clause.
  _Lessons:_ 4-file layout (chunks/terms/contains/similar.csv) landed with
  :LABEL-column C<cid> labels and a `;`-separated embedding column read via
  `embedding::text` (graph_tools.subgraph_embeddings pattern, no numpy); dim
  measured from data, vector_index.cypher emitted iff any embedding written —
  T8 will extend this same module (PATHWAY/MERGE writers), sequential, same
  file, so land those edits after chunks.csv/import.sh are stable, not beside them.

- [DONE] T4 Entities v0: relational bipartite tables built from the salient/phrase vocabulary
  _Files:_ entities.py, tests/test_entities.py
  _Verify:_ pytest tests/test_entities.py -q
  _Notes:_ Implements T1's schema amendment verbatim. Builder reads the
  existing store (terms/tf attrs) — READ-ONLY against pg_store's tables, no
  pg_store.py edit (that file belongs to T6 this layer). Deterministic NLP
  only, no LLM. NPMI floor: pairs with joint count < 5 chunks get no
  entity_edges row. Check for an incumbent module (salient_grams/graph_tools)
  before minting entities.py — extend if the builder naturally lives there BUT
  graph_tools.py belongs to T5 this layer; if the incumbent is graph_tools,
  the task is [BLOCKED] for re-layering, not a quiet file grab. Tests on a
  small planted corpus with known co-occurrence counts.
  _Lessons:_ No incumbent (grep for entit|mentions|npmi|ppmi hit only gensim
  Phraser scoring); entities.py is new, DDL created idempotently inside the
  builder per design.md block C. Used `integer` not `smallint` for entity_id
  (32k cap risked overflow, undocumented in the spec) and implemented bm25
  (rejected "NULL bm25 for v0" — E5 fixes the formula). All 13 tests, DB-free
  and DB-backed (planted 12-chunk corpus, 3 edges above the floor, 3 below),
  passed against a live Postgres :5433 — not skipped.

- [DONE] T5 Promote the second-order term ladder into graph_tools
  _Files:_ graph_tools.py, tests/test_graph_tools.py
  _Verify:_ pytest tests/test_graph_tools.py -q
  _Notes:_ Promote .tmp/second_order_probe.py's three-rung ladder: (1)
  Dunning-LLR gate on chunk-level co-occurrence, (2) Schütze context-centroid
  cosine on survivors, (3) Mann-Whitney AUC re-rank ONLY when the skew
  diagnostic trips (R2/R6 gate-then-fallback pattern at term level). Nomen
  pool = the operator's keyness floor: high-BM25/low-PPMI quadrant. Pin with
  planted-data tests (constructed co-occurrence where each rung's ranking is
  known); never a single case.
  _Lessons:_ 9 planted-data tests + 1 live-run shape test, 92/92 pytest green
  (incl. the DB-backed one, running against :5433 not skipped); the skew-trip
  reranking test needed a deliberate "hub direction" embedding construction
  (broad shared component inflates decoy's cosine without carrying row-level
  separating signal) to reproduce a genuine cosine-inversion, found by
  scripted parameter search rather than hand algebra.

- [DONE] T6 Title as persisted node metadata, displayed in walker and digest
  _Files:_ ingest_mixed.py, chunkgraph.py, pg_store.py, walker_app.py, interpret.py, tests/test_ingest_mixed.py, tests/test_pg_store.py, tests/test_interpret.py, tests/test_walker_render.py
  _Verify:_ pytest tests/test_ingest_mixed.py tests/test_pg_store.py tests/test_interpret.py tests/test_walker_render.py -q
  _Notes:_ Implements T1's payload amendment: carry ingest_mixed.wiki_title()
  through fit into node payload jsonb (R20 pattern — parse once, no downstream
  scanning). Walker and digest display title where present, degrade to
  wiki/NNNN doc_id where absent — old runs must not crash or change behavior
  (that IS the campaign's verification; no re-ingest in this task). Display
  reads the payload key directly; if a graph_tools.py change proves
  unavoidable, [BLOCKED] for re-layering (graph_tools belongs to T5).
  _Lessons:_ gt.node() already selects n.ord first, so chunk_label(row, titles) reads it for free — no graph_tools.py touch needed; 106 passed/2 skipped (live OPENROUTER-only), one pre-existing flaky timestamp-collision test in test_resave_supersedes_without_deleting passed on rerun, unrelated to T6.

## Layer 3 — sequential
- [DONE] T7 Re-run the frozen diagnostic and record sqrt-anchor before/after on B
  _Rescoped: fable 2026-09-03_ — the diagnostic ran and its evidence is recorded
  (that part is done); the acceptance it failed is re-scoped into T7a/T7b/T7c
  below on the root cause it surfaced: the sqrt allocation is a CAP on the
  majority (broke A1/E3), and the ring lane, not anchors, is where B's quotes
  chunks actually arrive. design.md §6.15 amended first (S15/S16 amended, S17
  new) per Article IX. History below preserved verbatim.
  _Files:_ .spec/specs/graph-explorer/diagnostic-prompts.md, .tmp/diag_rerun.py
  _Verify:_ PYTHONPATH=. python .tmp/diag_rerun.py mixed-full-dual
  _Notes:_ Needs live Postgres :5433 with run mixed-full-dual. Run all 20
  frozen rows at the calibration knobs (walker default + ef24 where the row
  was measured at ef24), assert per the measures section (mix/anchors/xprov).
  Acceptance: B1–B5 quotes ≥ 40% of the ef24 walk AND every previously
  passing A/C/D/E row still passes (A2 stays KNOWN-FAIL, it is not this fix's
  target). Update diagnostic-prompts.md's B verdicts with measured
  before/after numbers and the run id. The set is FROZEN — verdict columns
  update, prompts and expectations never. If B stays under 40%, record the
  measured shares and the ring-share evidence as _Blocked: — do not tune
  further inline.
  _Lessons:_ Live rerun (run bfa594df, mixed-full-dual, PYTHONPATH=. python
  .tmp/diag_rerun.py mixed-full-dual, exit 0, 20/20 rows): B quotes 6–31% ->
  2–25% (before -> after), still under the 40% gate.
  _Blocked: sonnet_ — sqrt anchors alone did not clear the B gate AND broke
  do-no-harm on two previously-passing rows, so this is blocked on both
  grounds. Run bfa594df-1238-448f-8e2b-d05136e6307a, ef24/T0/bridge0:
  B1 19% B2 17% B3 2% B4 25% B5 10% (all-origin); walk-only 4%/17%/4%/17%/4%.
  Ring origin carries 8/24, 4/24, 0/24, 8/24, 4/24 quotes chunks per walk
  respectively — ring shares track close to all-origin shares (e.g. B1
  quotes_all 19% vs quotes_walk only 4%, ring 8/24=33% quotes), so the ring
  top-off IS a residual dilution path on top of anchor allocation; trigger
  for the LATER "source-aware ring share" item stands independent of the
  regression below.
  Do-no-harm regression (the blocking finding): A1 flipped PASS -> FAIL —
  wiki is 80/81 (98.8%) of the mix, comfortably clearing the 70% floor, but
  the sqrt allocation caps wiki at 1-of-3 anchors on this 3-source run
  (brown, quotes, wiki all present), so wiki/28410 is no longer among
  bundle.anchors (anchor is now wiki/16278). E3 flipped PASS -> FAIL — brown
  + political-wiki combined share fell to 2.8% (was ≥50%), same root cause:
  fewer wiki/brown anchors seed fewer on-topic chunks. C1–C4, D1–D3, E1–E2
  all still pass; A2 stays KNOWN-FAIL as expected (not this fix's target).
  Root cause candidate: allocate_anchors' sqrt-share formula gives every
  present source at least 1 anchor slot before wiki's share dominance is
  reflected, so a 3-source run always drops wiki to k_anchor - (n_sources-1)
  anchors regardless of wiki's true mix share — that per-source floor is
  what the S-guard's spec text (design.md S15/S16) should be checked
  against; not tuned here (out of T7's boundary). Diagnostic-prompts.md
  updated per §6 regardless (B verdict cell + one dated line, sections A–E
  byte-identical, `git diff` confirmed). No source file touched.

- [DONE] T7a Anchors do-no-harm: keep the global top-k, add minority extras only when competitive
  _Files:_ sampler.py, tests/test_sampler.py
  _Verify:_ pytest tests/test_sampler.py -q
  _Notes:_ Implements amended S15/S16 (design.md §6.15, dated 2026-09-03) —
  paste the amended guard bodies into the docstrings, replacing the sqrt text.
  Rework allocate_anchors/select_anchors/anchor_hits: the global gt.search
  top-k_anchor stands untouched; each source with n_s > 0 absent from it gets
  at most ONE extra anchor beyond k_anchor, iff its best hit's BM25 score >=
  COMPETITIVE_FRAC (0.5, new module constant) of the global k-th hit's score;
  merged list deduped by ord, ordered (-score, ord); extras never displace a
  base anchor. Byte-identity regression tests: single-source, pre-R20, and any
  multi-source query where no minority clears the threshold return EXACTLY
  gt.search(k=k_anchor). Pin A1's shape as a planted-score unit test: a
  99%-one-source query keeps all k of that source's anchors. Delete now-dead
  sqrt code paths (Article VIII: remove dead code first).
  _Lessons:_ 53/53 tests green (live DB present, all live tests ran, not
  skipped) — anchor_hits now returns the untouched base plus at most one
  competitive extra per absent source, so T7b's ring allocation must expect an
  anchor source mix of size 1 as the common case (S17's skip-allocation branch
  will fire often); allocate_anchors and its 8 pinned tests are gone, plus the
  3 forfeit/each-source tests whose subject (quota partitioning) no longer
  exists.

- [DONE] T7b Source-aware ring share: ring budget allocated by the anchor source mix (S17)
  _Files:_ sampler.py, tests/test_sampler.py
  _Verify:_ pytest tests/test_sampler.py -q
  _Notes:_ THE B lever, promoted from LATER on T7's ring evidence (B1–B5 ring
  origin carries 8/24, 4/24, 0/24, 8/24, 4/24 quotes chunks while walk-only
  quotes is 4–17%). Implements S17 verbatim: S13's candidate pool and strength
  ordering unchanged; only the allocation of the RING_TOP x RING_PER slots
  across sources changes — largest-remainder over the anchor list's source
  composition, per-source fill in strength order, shortfall forfeits to global
  strength order. Skip allocation entirely (byte-identical fill, regression
  test) when every anchor shares one source or the run reports one label.
  Sequential after T7a because the anchor source mix S17 allocates by is T7a's
  output.
  _Lessons:_ ef_search publishes anchor_mix via gt.source_mix(hits); ring gains
  mix=None (early-return, byte-identical) plus a pure select_ring (largest-
  remainder quotas, strength-order fill, forfeit to global order); 63/63 live,
  0 skipped, including the live steering invariant against a real multi-source
  run.

- [BLOCKED] T7c Re-run the frozen diagnostic on amended anchors + ring share
  _Files:_ .spec/specs/graph-explorer/diagnostic-prompts.md, .tmp/diag_rerun.py
  _Verify:_ PYTHONPATH=. python .tmp/diag_rerun.py mixed-full-dual
  _Notes:_ Runner exists — reuse it, same knobs as T7 (walker default + ef24
  where measured at ef24). Acceptance: A1 and E3 restored to PASS (do-no-harm
  by construction — still verify), every other previously passing A/C/D/E row
  passes, A2 stays KNOWN-FAIL, and B1–B5 quotes >= 40% of the ef24 walk.
  Record before/after in the B verdict cell plus one dated line; the set stays
  FROZEN, verdict columns only. LADDER LAW: this is the second fable-tier
  scope of the B fix — if B is still under 40% here, the campaign STOPS and
  hands to the operator with the measured shares as evidence. No third
  re-scope, no inline tuning, no new structural change invented on the spot.
  _Blocked: sonnet_ — Live rerun (run bfa594df-1238-448f-8e2b-d05136e6307a,
  mixed-full-dual, PYTHONPATH=. python .tmp/diag_rerun.py mixed-full-dual,
  exit 0, 20/20 rows, T7a additive anchors + T7b S17 ring share): 13/20 PASS,
  A2 KNOWN-FAIL as expected. A1 do-no-harm RESTORED to PASS (n=77, wiki
  76/77, wiki/28410 anchored). E3 NOT restored — still FAIL (n=72, wiki
  68/72, brown 4/72, game 0; brown+political-wiki combined share far under
  the ≥50% target — a different failure mode than the sqrt-era cap, since
  anchors no longer cap the majority; the miss now sits in ring/PPR
  steering, outside this fix's scope). B1–B5 quotes_all: 15% 19% 8% 19% 29%
  (min 8% max 29%); quotes_walk: 4% 12% 8% 12% 8%; ring fill (of 24 ring
  slots): B1 6/24, B2 6/24, B3 2/24, B4 6/24, B5 12/24. S17 moved quotes_all
  from the sqrt-era 2–25% to 8–29% but did not clear the 40% gate. Two
  acceptance clauses fail (E3 not restored, B under 40%) — per LADDER LAW
  this is the second fable-tier scope on the B fix and the campaign STOPS;
  no further inline tuning or structural change. Full detail recorded in
  .spec/specs/graph-explorer/diagnostic-prompts.md's 2026-09-03 T7c
  calibration-run entry.

- [DONE] T8 Walk annotation writer: MERGE Walk and PATHWAY into live neo4j
  _Files:_ export_neo4j.py, tests/test_export_neo4j.py
  _Verify:_ pytest tests/test_export_neo4j.py -q
  _Notes:_ Needs live neo4j over bolt/HTTP for the integration case (skip
  cleanly without one; shape/cypher-text tests run offline). Small function
  taking a walk (Bundle + gt.pathways output) and MERGEing
  (:Walk {prompt,n,edges,wcc,density,conductance})-[:ANCHORS]->(:Chunk) and
  (:Chunk)-[:PATHWAY {dwpc,of}]->(:Chunk) — reproduce the 2026-09-03 live
  demo (midway walk: density 0.67, conductance 0.80, top DWPC 0.138
  3439<->3552). Sequential after T3 because both own export_neo4j.py; extend
  the module, no sibling script.
  _Lessons:_ stdlib urllib matched interpret.py's house HTTP style, no new
  dep (grep for requests/neo4j driver -> 0). One bug the subplan called out
  in advance and it hit for real on first live run: neo4j rejects a schema
  statement (CREATE CONSTRAINT) sharing a tx with writes ("Write query after
  executing Schema modification"), so the constraint now goes over the wire
  in its own `post()` call before the write tx — offline tests updated to
  read the write statements from `post.captured[-1]`. Chunk ids matched as
  `id: STRING` per X9, confirmed by the live round-trip actually finding
  Chunk nodes (not skipping). All 33 tests pass (24 existing + 9 new,
  including the live integration — neo4j was reachable at :7474 so it ran
  for real, wrote a throwaway `__t8_selftest__` walk, asserted idempotent
  counts on a second write, and cleaned up).

- [DONE] T9 Reconcile spec and queue against what shipped
  _Files:_ .spec/specs/graph-explorer/design.md, .spec/specs/graph-explorer/playbook.md, .spec/PIPELINE.md
  _Verify:_ grep -c "sqrt" .spec/specs/graph-explorer/design.md sampler.py
  _Notes:_ Article IX closing pass: T7's measured B before/after lands in the
  spec beside the S-guard; drift between minted amendment text and shipped
  docstrings resolved in the spec's favor or amended with rationale; the spec
  playbook's [TODO] rows this campaign closed (4-file export, neo4j mirror
  analysis layer, title metadata, second-order lane) flip to [DONE] with
  one-line evidence.
  _Lessons:_ design.md append-only edits at 1a-1f (sqrt closure sentences at
  §6.14 x2, new §6.15 A outcome block dated T7c, acceptance (a)/(c) measured
  verdicts, new §6.16 recording X6-X10/W17 with docstring-verbatim guard
  bodies); spec playbook.md 4 rows flipped [TODO]->[DONE] with guard-id+test-count
  evidence, entities v0 [DONE] row and B-lane [BLOCKED] row added;
  PIPELINE.md 5 clause edits (E in header, R22 in persist row, S15/S16/S17 in
  Phase B bullets 1/3, W17 row in Phase C table, X6-X10/E1-E5 in periphery +
  letter list). Verify: grep -c sqrt -> design.md 7->15 (rose, as required —
  0 would mean history deleted), sampler.py steady at 4 (all inside S15
  [SUPERSEDED] docstring block); allocate_anchors grep empty; pytest
  tests/test_sampler.py -q -> 63/63, no source touched. diagnostic-prompts.md
  untouched (its diff predates T9, from T7c).

## LATER (queued, not in this campaign)
- ~~Source-aware ring share in the sampler~~ — PROMOTED into Layer 3 as T7b
  (2026-09-03): T7's _Blocked:_ ring evidence was the trigger, exactly as queued.
- Re-ingest to populate titles on a live run (T6 ships degrade-safe; titles
  appear on the next natural re-ingest, e.g. the queued Leiden swap).
- Real NER entities (v1) replacing the salient/phrase vocabulary as the
  entities table population; extraction lane decided then.
- AGE extension install (Postgres-native graph queries).
- Dense-dilution structural fixes beyond sqrt anchors (source-aware expansion,
  anchor keyness thresholds) — same diagnostic-set gate.
- Correlation-sorting terminal-negative cross-links.
- Chunked-memory pass for dense NORMAL past ~15k chunks.
- Leiden in place of Louvain at ingest (next re-ingest, coarse-7 problem).
- Dual-run probe battery as process (cross-register dense-bridge assertions,
  per-cid source-mix audit, both-provenance exhibit).
