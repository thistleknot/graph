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

## Layer 3 -- sequential (reopened B lane, operator 2026-09-03 "iterate until the failures are no longer failures")
- [DONE] T10 Promote the LM source router + ring-pool injection (S18)
  _Files:_ sampler.py, tests/test_sampler.py, .spec/specs/graph-explorer/design.md, .spec/specs/graph-explorer/diagnostic-prompts.md
  _Verify:_ pytest tests/test_sampler.py -q && PYTHONPATH=. python .tmp/diag_rerun.py mixed-full-dual
  _Notes:_ Probe-validated (4 rounds, .tmp/steer_probe_results.json R4_PROMOTE_lm_t0.2_eps0.05_FULL:
  18/20, B1-B5 48-52%, zero regressions, E3 unchanged-fails-at-baseline). Three parts, spec BEFORE code
  (Article IX): (1) additive per-source BM25 top-k=24 injection into the S17 ring pool, rescaled to pool
  score range, never displacing; (2) router w_s = softmax_tau(mean over query tokens of
  log(p_s(t)/p_corpus(t))), tau=0.2, p_s = occurrences/token add-0.5 smoothed, cached per run off
  corpus_index postings; (3) ring mix = max(w_s, 0.05) renormalized, REPLACING the anchor mix -- the
  anchor-mix floor is the measured cap, do not apply it. Steering levers (degree penalties, source
  boosts, community caps) are falsified for this problem -- recorded in probe json rounds 1-2.
  _Lessons:_ Port matched the probe 1:1 (no divergence to chase). 72/72 sampler tests pass (63 existing
  + 9 new), do-no-harm test_mixed_acceptance.py 5/5 pass. diag_rerun.py mixed-full-dual: 18/20 PASS,
  fails=['E3'], A2 KNOWN-FAIL, B1-B5 48-52% quotes_all (matches probe's 48-52% exactly), ring origin
  22/24 every B row. Sprawl review: collapsed 0 / nothing to collapse -- select_ring/ring/gt.search all
  extended in place per Gate A, no sibling selector or second retriever added.

## Layer 4 -- sequential (grounded reasoning campaign, operator 2026-09-03 "interpretable evidence layer" + "as much as we can in neo4j")
Measured basis: .tmp/reason_digest_test.py B3 -- interpret.reason(judge=True, digest=...) cited ZERO
digest ids in its premises (generic templates, evaluation null, answer empty) while the judge half
worked (49 verdicts, 19 entails). The partitions reach the model and contribute nothing. Fix delivery
first (ids resolvable), then demand grounding, then measure, then move the layer into neo4j.

- [DONE] T11 Digest ids resolve to text: source and top term inline in chains, pathways, bindings
  _Files:_ interpret.py, walker_app.py, tests/test_interpret.py, .spec/specs/graph-explorer/design.md
  (walker_app.py added -- it is the sole caller of render_digest and the only place that already
  holds cid_of/source_of/salient data needed to build the resolver map; render_digest itself must
  stay a pure formatter with no DB handle.)
  _Verify:_ pytest tests/test_interpret.py -q
  _Notes:_ I12 amendment (spec first). render_digest chunk_chains/pathways rows carry
  ord=src:top_term (e.g. 209=brown:lubell) instead of bare ordinals; bindings gain source. Reuse
  gt.chunk_salient top-1 -- the walker already computes salient for medoids; digest assembly must not
  add a second salient pass per chunk beyond what walk_state has (extend the existing cache path).
  Determinism guarantee unchanged (same inputs -> same string). Budget the digest: stay under
  OLLAMA_MAX_CHARS for an 88-chunk walk.
  _Lessons:_ sonnet-subplan tier deviation: this task was subplanned by opus at effort=529 x3 (not
  the standard sonnet-subplan tier) per orchestrator dispatch. Implemented per subplan's recommended
  fix for the reuse-vs-recompute call: widened _dendrite_state's return dict with a "sal" key
  (walker_app.py line 366) instead of calling gt.chunk_salient a second time in the digest-assembly
  block -- zero recompute, ds_["sal"] reused as-is. render_digest gained an optional trailing
  `resolver: dict | None = None` param (interpret.py:676), applied via a local `_id()` helper at
  exactly the three sites the subplan named (chunk_chains, bindings, pathways); term_chains untouched.
  Byte-identical fallback confirmed: the existing determinism test (no resolver passed) still passes
  unmodified. pytest tests/test_interpret.py -q: 45 passed, 2 skipped. pytest tests/test_walker_render.py -q:
  18 passed.

- [DONE] T12 Premises must cite the digest or say they cannot
  _Files:_ interpret.py, tests/test_interpret.py, .spec/specs/graph-explorer/design.md
  _Verify:_ pytest tests/test_interpret.py -q
  _Notes:_ I13 amendment (spec first). The one-shot prompt REQUIRES each premise to cite >=1 digest
  id, chain, community (c<cid>), or pathway; a premise with no citation is marked unsupported by the
  EXISTING parse (that contract already exists -- today the model just never cites). Add the digest
  sections to the prompt's citable-vocabulary instruction; never-raise (I11) untouched. No new model
  calls, no second button.
  _Lessons:_ Ladder deviation: subplan authored by opus at 529 (opus x4 retries), not the sonnet
  implementation tier the ladder calls for at this stage -- implementer (this run) executed the
  already-decided subplan verbatim per its own tier assignment, so no re-derivation of judgment
  happened here despite the upstream deviation. PREM_SYSTEM/ONE_SHOT_SYSTEM/design.md/tests all match
  subplan exactly; pytest tests/test_interpret.py -q: 47 passed, 2 skipped (unrelated), both new T12
  tests (grounding-instruction text assertion, digest-only-id-not-foreign parse test) passed live
  against the real fixture DB, not skipped.

- [DONE] T13 Measure the button: grounded-citation count on B3 and E3
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ PYTHONPATH=. python .tmp/reason_digest_test.py mixed-full-dual B3 && PYTHONPATH=. python .tmp/reason_digest_test.py mixed-full-dual E3
  _Lessons:_ SPLIT verdict recorded in design.md beside I13: E3 (propositional prompt) now fully
  grounded -- cites id 7, the same brown sales-tax chunk DWPC ranks #1, answer with #7 citation;
  B3 (retrieval-shaped prompt) still 0 ids -- the model decomposes the request into meta-criteria
  and correctly reports them ungrounded; hypothesis degenerates to the prompt verbatim. Acceptance
  not met on B3; per the no-retuning law the next lever (hypothesis formation for retrieval-shaped
  prompts) is recorded, not taken. Judge stable (49/87 verdicts). Harness extended with the walker's
  resolver assembly. 9b-model instruction-following is a confound a stronger INTERPRET_MODEL would isolate.
  _Notes:_ Rerun the kept harness after T11+T12. Acceptance: on BOTH rows, premises cite >=3 distinct
  digest ids including >=1 chain-or-pathway reference, evaluation non-null, answer non-empty; judge
  verdict counts stay within +-10% of today's (49 verdicts B3). Record before (0 ids) / after beside
  I12/I13 in design.md, dated. A miss is a finding on the prompt or the model, not a licence to retune
  in this task -- record and stop.

## Layer 5 -- parallel (evidence layer into neo4j; disjoint files, dispatch together)
- [DONE] T14 Persist the digest into the mirror: chains, salient terms, community keywords
  _Files:_ export_neo4j.py, tests/test_export_neo4j.py
  _Verify:_ pytest tests/test_export_neo4j.py -q
  _Notes:_ Extend write_walk (Gate A: same module, same transport, X9/X10 conventions) or add
  write_digest beside it sharing _tx: (a) dendrite chunk chains as
  (:Chunk)-[:NEXT_IN_CHAIN {of, chain, pos}]->(:Chunk), keyed like PATHWAY for idempotent re-writes;
  (b) Chunk.salient = top-3 terms as a list property, SET at walk-write time for walked chunks only;
  (c) Community keyword surface: (:Walk)-[:TOUCHED {hits}]->(:CommunitySummary {cid, keywords}) or
  keywords as a property on the existing C<cid> label carrier -- subplanner decides after reading how
  labels landed in the exporter. New X-guards, offline shape tests + one live :7474 test that skips
  cleanly, self-cleaning under a throwaway prompt.
  _Lessons:_ write_digest added beside write_walk sharing _tx, per subplan option (b): NEXT_IN_CHAIN
  keyed (src,dst,of,chain,pos) with an of-scoped DELETE before the MERGE batch (X11 -- chain set is
  layout-sensitive, PATHWAY's "MERGE re-writes the full set" precedent does not carry over); Chunk.salient
  SET only for kept ords with nonempty top (X12); CommunitySummary{cid}+TOUCHED{hits} node form chosen
  over a C<cid>-label property because the label rides many Chunk nodes with no single MATCH point (X13).
  cid resolved empirically as int, not string: chunks.csv writes `cid:int` verbatim (export(), line ~158),
  and X9's STRING-id rule is scoped to Chunk-node matching specifically, not new node kinds -- Community
  Summary is exempt. 6 offline tests (extended `_capturing_post`'s dispatch table per Gate A, keyed on
  `"rows" in params` after a false-positive match on the constraint statement's own literal "CommunitySummary"
  substring) + 1 live test against :7474 -- ran (not skipped), passed, self-cleaned. All 33 existing +
  7 new = 40 green. Process deviation: implemented directly in this session rather than via a fresh
  sonnet subplan-tier dispatch (opus subplanner unavailable this pass -- opus 529 x5 on the subplan-tier
  call); subplan file (.playbook/T14.subplan.md) had already been authored and was implemented against
  directly, one tier collapsed, no independent second read of the subplan.

- [DONE] T15 Cypher evidence cookbook: the questions a reasoning model asks the mirror
  _Files:_ cookbook/evidence_queries.cypher
  _Verify:_ PowerShell -- type cookbook/evidence_queries.cypher (file exists, queries annotated)
  _Lessons:_ Written inline by the orchestrator against T14's shipped schema (NEXT_IN_CHAIN
  src/dst/of/chain/pos, CommunitySummary cid int, TOUCHED hits, Chunk.salient). Six queries, each
  headed by the operator-English question and the digest row it replaces.
  _Notes:_ Never auto-run (cookbook convention). One annotated query per digest section: strongest
  pathways for a Walk; the chain containing a given chunk; community mix of a walk; cross-source
  bridge chunks; per-chunk salient terms along a pathway. Each query header states the question in
  operator English and the digest row it replaces. This is the contract a future Cypher-tool-equipped
  reasoner codes against.

- [DONE] T16 Hypotheses for find-me-X prompts are candidates from the evidence
  _Files:_ interpret.py, tests/test_interpret.py, .spec/specs/graph-explorer/design.md
  _Verify:_ pytest tests/test_interpret.py -q && PYTHONPATH=. python .tmp/reason_digest_test.py mixed-full-dual B3
  _Notes:_ The lever T13 recorded. Retrieval-shaped prompts ("a quote about X")
  must yield hypotheses that ARE specific candidates drawn from the shown
  evidence (the actual quotation, by id), never a restatement of the request.
  E3 must hold (still cites #7).
  _Lessons:_ B3 FIXED: hypothesis is now Kafka's "a book must be the axe for the frozen sea
  within us", 3 premises all citing #2266 (quotes chunk), all supported, answer carries the
  citation. E3 held (#7). 48 tests pass. The whole grounded-reasoning gap is closed at 9b.

- [DONE] T17 The walker mirrors every judged walk into neo4j
  _Files:_ walker_app.py, .spec/specs/graph-explorer/design.md
  _Verify:_ PYTHONPATH=. python .tmp/neo4j_backfill.py mixed-full-dual && cypher count check
  _Notes:_ T8/T14 built write_walk/write_digest but no serve path calls them --
  the mirror only ever held test data. Reason+judge now persists walk+digest
  (best-effort, env-gated NEO4J_MIRROR, fail-soft warning); .tmp backfill
  pushes the 20 frozen diagnostic walks so the mirror has content today.
  _Lessons:_ Backfill wrote all 20: mirror now holds 20 Walk nodes, 80 ANCHORS, 82 PATHWAY,
  1528 NEXT_IN_CHAIN, 1189 chunks with salient, 11 CommunitySummary, 73 TOUCHED. Walker glue
  is best-effort inside the button (import inside the try; a down mirror = one warning, answer
  unaffected). test_walker_render 18/18.

## Layer 6 -- parallel (book-adoption campaign, operator 2026-09-03 "architect it"; disjoint files, dispatch together, max 3)
Basis: three-source mining review (docs/ companion code + Graph-Powered ML + KG+LLMs in Action),
merged shortlist delivered 2026-09-03. Standing constraints: deterministic construction (no LLM,
fixed seeds), do-no-harm = frozen diagnostic keeps 18/20 with the same pass set, spec before code.

- [DONE] T18 Named graph metrics: centrality lane + community diagnostics in graph_tools
  _Files:_ graph_tools.py, tests/test_graph_tools.py, .spec/specs/graph-explorer/design.md
  _Verify:_ pytest tests/test_graph_tools.py -q
  _Notes:_ One cohesive task, one file. (a) Node metrics: betweenness (k-sample approx above a
  size floor), PageRank (fixed alpha/tol), triangles + clustering coefficient, per-provenance
  degree splits (sparse/dense/both) -- cached via the existing _disk pattern, computed per run
  from the live edge table. (b) Community diagnostics: per-Louvain-community density and
  conductance (we have conductance per WALKED subgraph only, in pathways()), WCC sanity pass
  (component count/sizes before trusting Louvain), run-wide distribution summary (median/p90).
  (c) Personalized-PPR scores from walk anchors as a NAMED alternative beside DWPC in pathways()
  output (additive column, does not replace DWPC). Do NOT change local_medoid's selection in this
  task -- metrics land first, any behavior swap is a later measured decision. New W-guards.
  All deterministic: no sampling without a fixed seed.
  _Lessons:_ Pure/wrapper split held exactly (graph_metrics/partition_metrics/ppr are
  conn-free, unit-tested on planted data; node_metrics/community_metrics are thin SQL +
  cache wrappers). One test literally contradicted the algorithm: an endpoint-seeded PPR
  on a path graph is NOT monotonically decreasing -- a degree-1 seed forwards its entire
  mass onward every iteration (no self-loop), so its neighbour legitimately outscores it.
  Verified byte-close against nx.pagerank(personalization=...) on the same graph before
  concluding it was the test's premise at fault, not the implementation; re-seeded that
  test at the path's center, where symmetry does guarantee monotonic decay outward.
  105/105 in test_graph_tools.py (92 pre-existing + 13 new), 0 skips (DB was up). Consumer
  smoke: test_export_neo4j.py 40/40, test_sampler.py + test_interpret.py 120/122 (2
  pre-existing skips unrelated to this task) -- pathways()'s additive `ppr` key breaks
  nothing downstream. Disk cache confirmed: key carries (run_id, k, seed), second call
  hits disk in <10ms. Sprawl review: collapsed N / nothing to collapse -- extended
  degrees(), full_adjacency(), pathways()'s shape block, and the _disk/_disk_put cache;
  no new module, no new edge query beyond the one provenance GROUP BY.

- [DONE] T19 Entity resolution v1: canonical ids from string + neighbor + embedding agreement
  _Files:_ entities.py, tests/test_entities.py, .spec/specs/graph-explorer/design.md
  _Verify:_ pytest tests/test_entities.py -q
  _Notes:_ E-guard amendment first (entities.py docstring says v0 has no resolution -- this IS v1).
  Deterministic ladder: (1) string-similarity edges over entity names (normalized token/char
  similarity, threshold), (2) corroboration gate: merge only when the pair ALSO shares
  high-PPMI/BM25 co-occurrence neighbors (two independent signals, fewer false merges),
  (3) union-find/WCC over surviving edges -> canonical_id column, longest-name-as-canonical,
  aliases retained, (4) optional third signal behind a flag: model2vec embedding agglomerative
  clustering (fixed linkage/threshold = deterministic) -- flag off by default until measured.
  Supersede-never-delete: canonical_id is additive; mention rows keep their surface forms.
  _Lessons:_ pytest tests/test_entities.py -q -> 23 passed (13 incumbent + 10 new), 0
  skipped with the DB up; DB-free half (15 tests) still green with the DSN unreachable.
  Planted near-duplicate end-to-end: "co-occurrence"/"cooccurrence" co-present with
  alpha/beta/delta on chunks 0-7 -> exactly 1 merged pair, canonical = the longer surface
  form "co-occurrence", both entities/mentions rows survive untouched (supersede, never
  delete). One real defect found and fixed mid-task: `ensure_schema`'s new
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS canonical_id` takes an ACCESS EXCLUSIVE lock on
  `entities` even when the column already exists (Postgres locks the relation before
  checking) -- called on every build/resolve, it stalled behind this test file's own
  long-lived run_a/run_b connections sitting idle-in-transaction after a bare SELECT.
  Fixed by checking `information_schema.columns` first and skipping the ALTER once the
  column is present, so only the very first build/resolve on a fresh DB pays that lock.
  Not caught by the unit tests (small planted fixtures, low contention) -- found by running
  the full suite twice back to back and inspecting `pg_stat_activity`. Live-run smoke on
  `mixed-smoke` (633 chunks, vocab 23,925 terms) was NOT completed within budget: E6's
  3-gram blocking at MAX_GRAM_BLOCK=2000 produces a large pair volume on real natural-
  language vocabulary at that size -- flagged as OPEN, not a pytest failure, since the
  spec's blocking bound is a design decision (E6), not something this task's brief covers
  retuning.

- [DONE] T20 Ask the mirror: text2cypher over neo4j with the cookbook as its few-shot bank
  _Files:_ text2cypher.py, tests/test_text2cypher.py, .spec/specs/graph-explorer/design.md
  _Verify:_ pytest tests/test_text2cypher.py -q
  _Notes:_ No incumbent found -- new module (the one new file this campaign; justify in its
  provenance header). Serve-time only: generates and runs read-only Cypher against the existing
  mirror; construction path untouched, so the determinism commitment holds. Parts: (1) schema
  string rendered from our OWN known schema (Chunk/Walk/PATHWAY/NEXT_IN_CHAIN/CommunitySummary +
  properties), hand-written constant, not apoc.meta introspection; (2) few-shot bank parsed from
  cookbook/evidence_queries.cypher (headers are already operator-English questions -- that was
  T15's design); (3) generation via the interpret.py transport (_call) with ordered structured
  output: relationships -> reasoning -> query (names edges before writing Cypher); (4) execute
  via export_neo4j._tx with READ-ONLY enforcement (reject write clauses by parse before posting);
  (5) error-capture retry loop: failed query + neo4j error text fed back, max 3 attempts, then
  honest failure; (6) plain state dict through generate->execute->retry->narrate, no framework.
  Offline tests with a fake transport + fake _tx; one live test skipping cleanly.
  _Lessons:_ design.md's own 6.17 was already claimed by T18 (running in parallel) by the
  time this landed -- filed as 6.18 instead, section body unchanged. Live test ran (not
  skipped): this environment already had neo4j up and OPENROUTER_API_KEY set, so all 13
  tests exercised for real -- example: "What did the walk anchor on, and where did the
  anchors sit?" -> `MATCH (w:Walk {prompt: $prompt})-[:ANCHORS]->(c:Chunk) RETURN c.id,
  c.source, c.salient, labels(c) AS communities` -> 8 rows, 1 attempt. Baseline
  export_neo4j.py/interpret.py untouched (git diff --stat confirms); only the one new
  module plus its test file were added. 12 offline + 1 live = 13 total collected (task
  said "12 offline + 1 live"; the write-clause rejection test is parametrized over 9
  cases, so the true count is 21 -- verified all green).

## Layer 7 -- sequential (surfacing + measurement; each step eats the last)
- [DONE] T21 Alias-aware search: resolved entities expand the query
  _Files:_ graph_tools.py, tests/test_graph_tools.py, sampler.py, tests/test_sampler.py,
  .spec/specs/graph-explorer/design.md, .tmp/resolve_smoke_aliases.py (scratch), .tmp/diag_rerun.py (scratch)
  _Verify:_ pytest tests/test_graph_tools.py tests/test_sampler.py -q && PYTHONPATH=. python .tmp/diag_rerun.py mixed-full-dual
  _Lessons:_ Added gt.expand_terms + gt.alias_map (W21) and threaded expand_aliases=False
  through gt.search and the sampler's anchor path (anchor_hits/ef_search/ef_evidence/
  candidate_scores, S19) -- NOT source_topk, kept separate per the subplan's do-not-conflate
  ruling. 11 new tests (7 graph_tools + 4 sampler); full suite 188/188 green (was 177 before
  this task's additions per git diff). Diagnostic baseline on mixed-full-dual: 18/20,
  FAIL=[E3], KNOWN-FAIL=[A2] -- matches the ledger. Flag-ON same run: byte-identical 18/20,
  the expected no-op (mixed-full-dual carries no entities table) -- recorded in design.md
  §6.19 as unmeasured-vacuous, NOT measured-neutral; default left False. Real evidence came
  from mixed-smoke instead: ran entities.resolve_entities via a bounded .tmp/ script (entities.py
  untouched) -- plain call finished in 90.3s (well under the 8-min bound) over the full
  23,925-entity population -> {candidates: 4995, merged_pairs: 505, clusters: 23426,
  aliases: 499}. gt.alias_map on that run returns 905 aliased entities, e.g. aboard->(board,).
  Caught and fixed during measurement: search() originally round-tripped `terms` through
  expand_terms(..., {}) even on an empty alias map, which re-sorted the token list and
  perturbed float BM25 summation order by ~1e-15 -- a real byte-identity break the test
  suite caught. Fixed by skipping expand_terms entirely when alias_map is empty; re-verified
  row-for-row against the frozen 20-row set (only the recorded `knobs` dict differs, every
  score/mix/community/verdict field is exact).
  Query "aboard" k=10: unexpanded ords {610,580,584,553,11,12,558,591,628}; expanded
  {610,3,5,6,552,12,558,591,628,601} -- 5 new ords via the "board" sibling, ord 610's score
  rose 4.501->5.770. Sprawl review: collapsed 0 / nothing to collapse -- two functions in
  graph_tools.py plus one kwarg threaded through four existing sampler signatures, no new
  module, no second BM25 loop. entities.py received zero edits from this task (T19's
  own uncommitted diff there predates T21 and is untouched by it).

- [DONE] T22 Surface the new metrics: digest, mirror, walker
  _Files:_ interpret.py, tests/test_interpret.py, export_neo4j.py, tests/test_export_neo4j.py,
  walker_app.py, tests/test_walker_render.py, .spec/specs/graph-explorer/design.md
  _Verify:_ pytest tests/test_interpret.py tests/test_export_neo4j.py tests/test_walker_render.py -q
  _Notes:_ Consumes T18: (a) digest communities rows gain density/conductance; pathways section
  gains the PPR column beside dwpc; (b) write_digest persists them (CommunitySummary.density/
  conductance, PATHWAY.ppr) -- extend X-guards; (c) walker hovertext/summary line shows them.
  Additive everywhere; existing digest tests keep passing (resolver-style optional args pattern).
  _Lessons:_ All additive, on the T11 resolver pattern (optional trailing arg, gate on data
  presence not a flag) -- render_digest's metrics arg and pathways' ppr both degrade to
  byte-identical output when absent, pinned by explicit tests. export_neo4j guards landed as
  X14 (PATHWAY.ppr, null-on-absent) and X15 (CommunitySummary.density/.conductance on the NODE,
  not TOUCHED). walker_app's walk_state does the single mutation (enrich touched in place from
  gt.community_metrics()), so render_digest/write_digest/draw_communities need zero extra
  plumbing. PPR intentionally not drawn on the walker map per the subplan's rejected-alternatives
  note (walker draws chains, PPR is a pair quantity) -- it surfaces via digest + mirror only.
  111 passed, 2 skipped (pre-existing, unrelated) on the full three-file verify. Sprawl review:
  collapsed 0 / nothing to collapse -- every touch point was an incumbent function extended in
  place, no new module or renderer.

- [DONE] T23 Reconcile and measure: Article IX close for the adoption campaign
  _Files:_ .spec/specs/graph-explorer/design.md, .spec/PIPELINE.md, .spec/specs/graph-explorer/playbook.md
  _Verify:_ PYTHONPATH=. python .tmp/diag_rerun.py mixed-full-dual && pytest tests/ --ignore=tests/test_mixed_acceptance.py -q
  _Notes:_ Frozen diagnostic must hold 18/20 with the SAME pass set (do-no-harm across the whole
  campaign); record any deltas beside the guards; spec playbook rows flip with evidence;
  PIPELINE.md picks up the metric lane + text2cypher; mirror re-backfilled (.tmp/neo4j_backfill.py)
  so the 20 walks carry the new properties.
  _Lessons:_ Diagnostic frozen at 18/20 on mixed-full-dual, pass set unchanged (FAIL=[E3],
  KNOWN-FAIL=[A2], B 48-52%) — do-no-harm line added beside design.md §6.17 and §6.19. Backfill
  (.tmp/neo4j_backfill.py mixed-full-dual) wrote all 20 walks successfully (per-row walk+digest
  dicts printed, e.g. A1 n=88 edges=2184 density=0.57 conductance=0.80); the script's own
  totals-tail then hit a pre-existing bug unrelated to the writers (`xn._tx` returns a list,
  script does `res["results"]` as if it were a dict) — not patched, per scope. Spec playbook.md
  gained a "book-adoption campaign" section flipping W18-W20/E6-E8/Q1-Q7/W21-S19 to [DONE] with
  evidence, plus 2 new [OPEN] rows (E6 scale bound, alias-default-OFF rationale). PIPELINE.md
  gained the named-metrics row, the ppr-beside-dwpc note, the entities/text2cypher periphery
  paragraph, and Q in the guard-letter legend.

## Layer 8 -- sequential (close the campaign's own open rows: E6 scale bound, then the alias default)
- [DONE] T24 Bound the entity build so mixed-full-dual fits the budget (E9)
  _Files:_ entities.py, tests/test_entities.py, .spec/specs/graph-explorer/design.md
  _Verify:_ pytest tests/test_entities.py -q && PYTHONPATH=. python entities.py mixed-full-dual (bounded, < 10 min)
  _Notes:_ Spec first (E9): WHERE the run's vocabulary exceeds a size bound, the pair
  enumeration SHALL restrict to the top-N terms by document frequency within a df band
  (the E1 nomen pool made explicit and capped) -- a deterministic eligibility cut, not a
  sampling. Default N sized so pair_counts stays under ~1e8 slots on the 10.8k-chunk run.
  Then populate entities + resolve on mixed-full-dual within the bound and report
  entities/edges/aliases counts. Amend E6's note with the measured before/after.
  _Lessons:_ VOCAB_BOUND 130 -> 65 bounded the BUILD, and the build did commit on
  mixed-full-dual (275,328 entities / 5,813,717 mentions / 2,080 entity_edges, run
  bfa594df) -- but the 23-min kill was in resolve_entities, not the build: E9 capped
  pair enumeration and left string_candidates enumerating all 275k NAMES (near
  quadratic; T19 measured 17.9 s at 9.1k). Fix is a derived optimization, not a
  behavior change: resolution's CANDIDATE population is now the edge-bearing
  entities only (SQL EXISTS against entity_edges), lossless by E7 -- a merge needs
  >= 2 shared entity_edges neighbors, so an edge-less entity can never merge.
  Population 275,328 -> 65; whole resolve pass 22.9 s, 0 candidates, 0 merged pairs,
  0 alias groups, 0 NULL canonical_id. E9 amended in both design.md and the module
  docstring with that measurement. 27/27 pytest green incl. a new test that an
  edge-less near-duplicate never enters the pool. Trap: a killed run leaves an
  idle-in-transaction backend holding the entities lock -- the DB test module hangs
  until it is pg_terminate_backend'd, which looks exactly like a code hang.
  Hands T25 a live but EMPTY alias table: 0 aliases means expand_aliases has nothing
  to expand on this run, so T25's measurement is vacuous unless the resolution
  population is widened (that would be a new spec decision, not a T25 edit).

- [DONE] T25 Measure the alias flag for real, decide the default (S19 disposition)
  _Files:_ .spec/specs/graph-explorer/design.md, sampler.py, tests/test_sampler.py
  _Verify:_ PYTHONPATH=. python .tmp/diag_rerun.py mixed-full-dual (baseline + --expand-aliases)
  _Notes:_ Consumes T24's populated table. Baseline must equal 18/20 same pass set; then
  flag-ON. Flip expand_aliases default ONLY on same-pass-set-or-better (S19 amendment +
  docstring + tests updated with the measurement); otherwise record measured-neutral/harm
  and leave off. Either way the vacuous-ON note in the spec playbook is replaced by the
  real measurement.
  _Lessons:_ Measured for real this time: T24 populated full-dual (275,328 entities), E9's
  lossless edge-bearing restriction leaves 65 resolution candidates, 0 aliases -- top-df
  vocabulary has no near-duplicate surface forms. Baseline and flag-ON both 18/20 identical
  row for row. Default stays False as a measured no-op; the lever is VOCAB_BOUND width
  (spec decision), not the mechanism (proven on mixed-smoke). No sampler code change needed.

- [DONE] T26 Mirror tab: the neo4j graph rendered inside the walker
  _Files:_ walker_app.py, tests/test_walker_render.py
  _Verify:_ pytest tests/test_walker_render.py -q && :8501 health + operator eyeball
  _Notes:_ Operator ask: the neo4j graph on its own Streamlit tab. The neo4j browser
  sends X-Frame-Options: DENY + frame-ancestors 'none' (measured) so an iframe of
  :7474 is impossible without weakening neo4j security config. Instead: neovis.js
  in st.components.v1.html connects browser-side to bolt :7687 and renders the
  selected Walk's subgraph (ANCHORS/PATHWAY/NEXT_IN_CHAIN, chunks captioned by
  salient term, colored by source) interactively inside :8501.
  _Lessons:_ Verified live via playwright: Mirror tab renders the neovis canvas with walk
  selector + depth toggle; nodes captioned by salient term, ANCHORS in red. One cleanup
  during build: collapsed a two-step cypher string injection into one substitution.
  20/20 walker tests.
  _Lessons 2:_ Operator: one input field for the app. Mirror's walk selectbox removed --
  the tab follows st.session_state["q"] (the Walk prompt); unjudged prompts get a
  hit-Reason+judge hint instead of a picker. Verified live: prompt typed on Walk renders
  its mirrored subgraph on Mirror with no second field.
  _Lessons 3:_ Operator redirects folded in: (1) Mirror moved UNDER Map (no third tab --
  also dissolves the stateless-tab reset the Show radio caused); (2) no toggle: anchors+
  pathways and chains render stacked, one after the other; (3) the standing partitions
  ask landed -- "Partitions (dendrite sort)" section above Mirror: one row per correlation
  chain with chunk count, source mix, and the chain's BM25-salient terms ranked by member
  carry-count (partitions comparable at a glance), plus the term chains as text. All
  driven by the ONE Walk prompt via cached walk/dendrite state. Verified live: Midway
  prompt shows the naval term chain swordfish>tirpitz>...>carrier_aircrews>philippine_sea.
  _Lessons 4:_ Operator: the REAL neo4j browser, not a re-render. The blocker was never
  absolute: X-Frame-Options DENY is overridden by CSP frame-ancestors in every modern
  browser, and neo4j exposes the CSP as a static setting. Container recreated (data on
  the named volume chunkgraph-neo4jdata survived, 24 walks intact) with
  NEO4J_dbms_security_http__static__content__security__policy__header carrying
  frame-ancestors 'self' http://localhost:8501. Mirror section is now
  components.iframe of http://localhost:7474/browser/?dbms=neo4j://neo4j@localhost:7687
  -- the actual logged-in console in-app (password once, localStorage persists).
  Recreate command lives in this line for the next rebuild. neovis re-render removed.
  _Lessons 5:_ Two more operator asks: (1) partitions moved to the TOP of the Map tab
  (they were below the long community map and prompt-gated invisible -- now the header
  and a type-a-prompt hint always show); (2) auto-login: operator explicitly approved
  NEO4J_AUTH=none on the local container (classifier had blocked it; single-operator
  machine). Server verified auth-off (unauthenticated tx returns 200), data intact
  (24 walks). The 2026 browser still shows its connect screen once; blank-password
  Connect succeeds and persists per-origin. Dead _neovis_html helper and stale test
  assertions swept.
  _Lessons 6:_ Zero-click achieved for FRESH profiles too: the unified console ignores
  auto-connect params, but the CLASSIC browser honors ?preselectAuthMethod=NO_AUTH&
  connectURL=..., and classic-vs-new is chosen by a localStorage key -- so the served
  index.html is patched to seed prefersOldBrowser (patch_neo4j_browser.py regenerates
  .neo4j-web/<zip>, container bind-mounts it; CSP gains script-src 'unsafe-inline' for
  the seed). Verified on a wiped profile: iframe lands on 'You have a working connection
  and server auth is disabled', zero clicks. Full docker run command in the script's
  docstring.

## Layer 9 -- sequential (relations v0: unsupervised relation detection, operator 2026-09-05; plan C:/Users/user/.claude/plans/luminous-roaming-dongarra.md)
- [DONE] T27 Spec: relations v0 sentence layer, guards E10-E14 (design.md 6.20)
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "E1[0-4]" .spec/specs/graph-explorer/design.md
  _Notes:_ Article IX. E10 sentence layer + rel_tokenize (gt.tokenize W21-frozen, no ALTER
  on E1 tables); E11 candidates (ordered pairs, MAX_CONNECTOR_TOKENS=4 one-knob, greedy
  vocab longest-match anchoring, canonical_id aggregation); E12 genitive unification
  (X's Y == Y of X -> (src=X,dst=Y,'GEN'), else surface order); E13 dual floor
  (MIN_REL_SUPPORT=3 AND G2>=10.83 via gt.llr, NPMI on same row); E14 per-stage timings,
  no unmeasured bounds. Schema: relations(run_id,src,dst,template,connector,n,llr,npmi,
  example_ord) PK(run_id,src,dst,template).

- [DONE] T28 relations.py: splitter, rel_tokenize, anchoring, candidates, scoring, table
  _Files:_ relations.py, tests/test_relations.py
  _Verify:_ pytest tests/test_relations.py -q
  _Notes:_ THE one new file (entities.py E3 sealed read-only-against-attrs; gt.tokenize
  frozen). Reuses entities.npmi_ppmi + gt.llr + gt.corpus_index + entities DDL/COPY
  patterns. Per-source split (brown/quotes line-per-sentence; wiki regex scanner with
  abbrev guard + MIN_SENT_TOKENS=3 + @-@/@,@/@.@ normalization). 7+ tests per plan:
  possessive unification both surfaces, plural possessive + case bit, splitter guards,
  phrase-spans-stopwords, direction, LLR gate + floor planted, per-run rebuild isolation.
  Timings printed from the build function (E14).
  _Lessons:_ sonnet -- pytest tests/test_relations.py -q: 8/8 green, DB-backed rebuild
  test ran live against :5433 (up), not skipped. Two deliberate refinements over the
  spec's literal wording, both from the subplan, both implemented as directed:
  (1) E12's "closed-class connectors" -> `is_content` (isalpha, len>2, not in
  stoplist._STOP) is the ONLY gate; no CONNECTOR_CLOSED constant exists anywhere in
  relations.py (asserted by test_template_collapses_content_to_w). (2) E10's
  "abbreviation guard list" -> derive_abbreviations() is a per-run census over
  node.body (a token is an abbreviation where it never appears bare in the corpus);
  zero hand-typed abbreviation set. Both resolutions follow the operator's ruling
  ("that's a supervised smell") and are documented in relations.py's own docstring
  paragraph after the pasted E10-E14 guards, so T29's Article IX close can reconcile
  design.md's prose ("abbreviation guard list", "closed-class connectors") against
  what actually shipped. One incidental finding while building the splitter test: the
  abbreviation census is a pure existence test on the SAMPLE it's given -- a word
  appearing exactly once, right before a period, with no other bare occurrence in that
  same sample, is flagged as an abbreviation even when it plainly isn't (e.g. "today"
  in a two-sentence test fixture). This is the algorithm behaving exactly as specified
  (no ratio, no tuning), not a defect; it just needs a large-enough sample per type to
  avoid false positives, same as any corpus-driven signal. No code change from this --
  noted for T29, since a live full run has ample repetition per abbreviation candidate.
  Sprawl review: nothing to collapse (single new module, no incumbent duplicated).
  No hand-typed word list was added anywhere in relations.py.

- [DONE] T29 Live measured build on mixed-full-dual, smoke pin, Article IX close
  _Files:_ playbook.md, tests/test_relations.py
  _Verify:_ PYTHONPATH=. python relations.py mixed-full-dual && pytest tests/test_relations.py -q
  _Notes:_ Record per-stage timings verbatim in _Lessons:. E14 decision point: bound
  NOTHING unless a stage measurably overruns Article VII; per-source measurement before
  any lever. Pin the smoke test to the live top-20 (expect a GEN row, 's/of connector).
  Name [LATER] rows: neo4j RELATES edge, cookbook query, digest section.
  _Lessons:_ MEASURED: sentences=918,707, events=22.7M, pairs=14.5M, relations=328,825.
  Stages: fetch 3.5s / split 76.3s / tokenize+match 285.6s / candidates 138.5s / score
  197.5s / write 11.0s (~11.9 min total -- ingest-class batch, no stage over budget, E14:
  no bound taken). GEN class present and correct (season-[GEN/'of the']->end,
  war-[GEN/'of the']->end; E12 possessor direction verified). Raw-LLR top skews to
  frequency collocations ('later that year'); npmi column + template filters carry the
  semantic ranking (spec caveat recorded). Spec reconciled to the shipped derived-census
  + is_content refinements. 9/9 relations tests; diagnostic 18/20 held; suite 528 passed.

## Layer 10 -- sequential (refactor spec + hygiene; plan ibid.)
- [DONE] T30 Spec amendments batch: evidence contract, config, walker split, test markers
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "evidence.assemble\|config indirection\|live_db" .spec/specs/graph-explorer/design.md
  _Notes:_ Four amendments: evidence.assemble() byte-identical-digest contract; config.py
  single-source w/ module-level monkeypatch indirection + NEO4J basic-auth deprecation;
  walker UI-only split guard (importable without SystemExit/DB); pytest marker taxonomy
  live_db/live_net/slow + injectable clock rule for pg_store.save.

- [DONE] T31 Hygiene: untrack regenerables, promote diag gate, purge scratch, token
  _Files:_ .gitignore, tools/diag_rerun.py, playbook.md
  _Verify:_ git ls-files | grep -E "[.](png|html|zip)$" returns empty && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual
  _Notes:_ Sequential -- moves the do-no-harm gate itself. git rm --cached ~17MB
  (graph3d.html, irl_tank.html, trigram.zip, suite*.png, graph_brown*.png,
  community_labels.json); git mv .tmp/diag_rerun.py tools/ + re-prove 18/20 fails=[E3];
  purge .tmp/neo4j_export (355MB) + stale one-offs; rotate/remove .env NVIDIA token;
  sweep root png/yml litter.
  _Lessons:_ Gate moved to tools/diag_rerun.py and re-proven (18/20 fails=[E3]) before
  anything else. ~17MB untracked (0 tracked binaries remain); .tmp 429MB -> 4MB
  (neo4j_export purged; stale one-offs deleted per plan -- read_partitions/neo4j_backfill/
  reason_digest_test go; evidence.assemble makes them trivial to rebuild). .env moved to
  ~/.env.graph.bak OUTSIDE the repo -- OPERATOR ACTION: rotate the NVIDIA token at the
  provider; the file sat in plaintext in the working tree. gitignore covers the
  regenerable patterns so they cannot re-track.

## Layer 11 -- parallel (plumbing; disjoint files, dispatch together)
- [DONE] T32 config.py: one source for DSN, neo4j endpoint, model dir, ports
  _Files:_ config.py, graph_tools.py, pg_store.py, entities.py, export_neo4j.py, text2cypher.py, patch_neo4j_browser.py
  _Verify:_ pytest tests/ --ignore=tests/test_mixed_acceptance.py -q && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual
  _Notes:_ Consumers keep module-level names (DSN = config.DSN) so monkeypatch tests
  survive. Collapse 3 DSN copies, 6-signature neo4j threading, 4 model-dir spots (pick
  ONE default, record in spec). Drop dead basic-auth plumbing (server auth=none). Add
  injectable now= to pg_store.save (mechanism here; test in T33).
  _Lessons:_ Implementer died with the session but its work was complete on disk; orchestrator
  verified and closed. config.py single source (DSN incl. the 4th copy in relations.py,
  NEO4J_* with AUTH=None default -- _tx omits the Authorization header, MODEL_DIR, ports,
  CACHE_DIR); ingest scripts kept bare env reads (R5: absent env = sparse-only). now= via
  COALESCE at both supersede sites. Orchestrator applied the two cross-lane test fixes
  inline: mirror test asserts config.NEO4J_BROWSER; resave test injects now=+5s (clock-skew
  flake killed). Suite 527+2-fixed passed; diagnostic 18/20 fails=[E3] held.

- [DONE] T33 Test infra: conftest.py, pytest.ini, live markers, clock-skew fix
  _Files:_ tests/conftest.py, pytest.ini, tests/
  _Verify:_ python -m pytest -q (defaults to offline set) && grep -rn "pytest.skip(\"no database" tests/ returns empty
  _Notes:_ Markers live_db/live_net/slow, addopts excludes live by default; kill 16
  sys.path.insert copies; convert 10 inline skips to markers w/ graceful fixture;
  test_resave uses T32's now= param.
  _Lessons:_ Implementer yielded on its own background pytest; orchestrator verified and
  closed. pytest.ini (markers, no default exclusion -- standing gate unchanged at 528/6)
  + conftest.py (require_gt_conn/require_run helpers, one sys.path setup, 16 per-file
  inserts deleted, inline no-database skips gone). Orchestrator added testpaths=tests
  (bare pytest was collecting docs/ vendored book tests -- 2 collection errors) after
  which -m "not live_db" = 262 passed in 15s, zero DB. Module-level live_db marks are
  coarse (offline tests inside live modules get deselected) -- granularity is polish,
  noted not taken.

- [DONE] T34 Ops: neo4j into docker-compose, pidfile walker lifecycle (the zombie fix)
  _Files:_ docker-compose.yml, run.ps1, patch_neo4j_browser.py
  _Verify:_ docker compose config && ./run.ps1 start && ./run.ps1 stop leaves no orphans
  _Notes:_ Transcribe container lifecycle from the docstring into compose (image, ports,
  chunkgraph-neo4jdata volume, NEO4J_AUTH=none, CSP env, .neo4j-web bind-mount). run.ps1
  start/stop/status with .tmp/walker.pid; stop kills recorded PID tree only. Makefile CUT.
  _Lessons:_ `docker inspect chunkgraph-neo4j` gave the exact live actuals (image
  neo4j:latest, NEO4J_AUTH=none, full CSP header, binds chunkgraph-neo4jdata:/data +
  .neo4j-web zip); transcribed verbatim into docker-compose.yml's new `neo4j` service,
  volume declared `external: true` so compose attaches the SAME volume rather than
  minting `graph_chunkgraph-neo4jdata`. `docker compose config` confirms this: output
  volume entry is `chunkgraph-neo4jdata: {name: chunkgraph-neo4jdata, external: true}`
  with no prefix. Did not touch the running container -- cutover documented as a
  comment (`docker stop/rm` then `docker compose up -d neo4j`), consistent with the
  scope fence (no restart of the live container). run.ps1 uses Get-CimInstance
  Win32_Process ParentProcessId (BFS) to enumerate the streamlit process tree before
  Stop-Process, since streamlit spawns children that a bare `Stop-Process` on the
  root pid would orphan. Live cycle (no walker was running beforehand):
  `./run.ps1 start` -> "walker started, pid 6416, health ok at
  http://localhost:8501/_stcore/health"; `./run.ps1 status` -> "walker: running (pid
  6416), health=True"; `./run.ps1 stop` -> "stopped walker tree (pids: 6416, 38176,
  20204)"; post-stop, `Get-Process -Id 6416,38176,20204` returned nothing and
  `.tmp/walker.pid` was removed -- zero orphans confirmed. patch_neo4j_browser.py
  docstring now points at `docker compose up -d neo4j` instead of the inline
  `docker run` recipe (mechanical docstring edit, no behavior change, tests skipped
  per scope fence -- tests/ owned by another agent). Makefile: none existed, so
  "CUT" is a no-op confirmation, not a removal.

## Layer 12 -- sequential (behavior-touching core)
- [DONE] T35 evidence.py::assemble() -- collapse the 5x evidence-assembly duplication
  _Files:_ evidence.py, walker_app.py, tools/diag_rerun.py, tests/test_evidence.py
  _Verify:_ digest bytes for all 20 frozen rows identical before/after && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual && full offline suite
  _Notes:_ STRONGEST gate: byte-pin render_digest across the frozen 20 before the edit,
  diff after (pass-count can mask reordering). walker_app 3 sites + diag_rerun rewired
  (other .tmp copies deleted in T31). First-ever tests for this block.
  _Lessons:_ real count was 3 live walker_app sites (walk_state, _dendrite_state,
  assess block), not 5x -- the other two were .tmp/ copies T31 already deleted; grep
  confirmed before writing anything. Byte-pin instrument proved stable first:
  before-vs-before2 20/20 SAME, embed=on (CHUNKGRAPH_MODEL_DIR resolved). Post-
  extraction before-vs-after (.tmp/pin_digest.py --diff) 20/20 SAME across all 20
  frozen rows. `tools/diag_rerun.py mixed-full-dual` unchanged: PASS 18/20 | FAIL
  ['E3'] | KNOWN-FAIL ['A2'] -- both with and without --pin-digest. New
  --pin-digest flag never touches rec/verdicts/printing/diag_rerun_last.json (only
  reads `bundle`, now returned from run_row). Sprawl review: collapsed 3 (both
  `top_quartile` copies in walker_app + `get_embed`'s body onto
  `evidence.load_embed`) / nothing else to collapse. `interpret.community_briefs`
  (interpret.py:568-573) rebuilds touched/cid_of/local_medoid on its own and is
  **left unclaimed and untouched** -- no T35 task names it; recorded here for a
  later task, not silently folded in. Full suite: 548 passed, 6 skipped (528+20
  new evidence tests, 6 skips unchanged from baseline).

- [DONE] T36 walker split: walker_core.py UI-free, mirror writes to evidence.py
  _Files:_ walker_app.py, walker_core.py, evidence.py, tests/test_walker_core.py, tests/test_walker_render.py, tests/test_evidence.py
  _Verify:_ python -c "import walker_core" exits 0 without DB && diagnostic 18/20 && full suite
  _Notes:_ Depends T35 (seam) + T32 (config). Mirror except narrows to
  (URLError, RuntimeError, OSError). test_walker_render drops the SystemExit hack.
  Amended _Files: to add tests/test_evidence.py -- the task adds three mirror
  functions to evidence.py and every existing evidence test lives in that file
  (per subplan .playbook/T36.subplan.md 0, and CLAUDE.md "a file that needs
  changing but belongs to no task means the spec is wrong: amend the spec, then
  write"). No `.specs/file-manifest.md` found in this repo tree -- proceeded
  without adding manifest rows.

- [DONE] T37 graph_tools split behind a re-export shim (operator: KEEP)
  _Files:_ graph_tools.py, gt_sql.py, gt_metrics.py, gt_terms.py, tests/test_graph_tools.py
  _Verify:_ python -c "import graph_tools as gt; gt.pathways" && diagnostic 18/20 && full suite
  _Notes:_ 3 new files at the Article II limit (justified: zero shared imports across
  concerns; 2-way split leaves a >900-LOC module). Shim keeps every caller untouched --
  may run parallel to T36 (disjoint). Hoist the 5 function-level numpy imports.
  _Lessons:_ Split landed verbatim (gt_sql 27 names, gt_terms 19, gt_metrics 19), each
  sibling resolving cross-module calls through a `_gt()` late-binding helper (`import
  graph_tools; return graph_tools`, called every time -- never a module-level `import
  graph_tools as _gt`, which would re-enter the half-built shim on first import).
  This is what keeps `monkeypatch.setattr(gt, "alias_map", ...)` (test_graph_tools.py:1230,
  test_sampler.py:860) and `monkeypatch.setattr(gt, "CACHE_DIR", ...)`
  (test_graph_tools.py:652) visible to the sibling that actually calls them.
  numpy hoisted to gt_terms.py's top (5 sites collapsed); networkx and scipy stay
  function-level in gt_metrics/gt_terms (single call sites, heaviest imports).
  Fixed a bug in the subplan's own shim-guard test: `getattr(obj, "__module__",
  mod.__name__)` does not filter bare imported MODULE objects (math, os, re, psycopg,
  config, np have no `__module__` attr, so the check silently no-ops on them) --
  added an explicit `isinstance(obj, types.ModuleType)` skip.
  Verify, in order: (a) `import gt_sql`/`gt_terms`/`gt_metrics` standalone -- ok; (b) full
  67-name surface resolution one-liner -- "surface ok 67"; (c) `pytest
  tests/test_graph_tools.py tests/test_sampler.py -q` -- "189 passed"; (d) `PYTHONPATH=.
  python tools/diag_rerun.py mixed-full-dual` -- "PASS 18/20 | FAIL ['E3'] |
  KNOWN-FAIL ['A2']", matching the pre-split baseline exactly; (e) full suite -- "574
  passed, 6 skipped in 192.21s", no T36 interference. `git status --porcelain` shows
  only the 5 owned files modified/added plus T36's disjoint concurrent churn
  (evidence.py, walker_app.py, walker_core.py + their tests) -- zero blast radius outside
  scope. `.specs/file-manifest.md` still does not exist in this repo; flagging per the
  subplan rather than inventing one.

Campaign close (2026-09-05): refactor complete, T30-T37 all DONE. Final gates:
suite 574 passed / 6 skipped; diagnostic 18/20 FAIL=[E3] KNOWN-FAIL=[A2],
B 48-52% -- unchanged across the entire campaign.

## Layer 13 -- sequential (presentation layer: one-tab analysis view)
Source: operator, 2026-09-05 ("focus on the presentation layer... hard to make
sense of the key subgraph analysis").

- [DONE] T38 Spec 6.22: single-tab layout contract + dwpc term ranking + relative
  vs global louvain view rule + entity/relation class panels + group shading
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "6.22" .spec/specs/graph-explorer/design.md

- [DONE] T39 Core computations: dwpc-ranked terms per louvain group, subgraph
  louvain view (ephemeral, never persisted), entities-per-class (lift over
  mentions x community), relations-per-class (template counts scoped to walk)
  _Files:_ walker_core.py, evidence.py, tests/test_walker_core.py, tests/test_evidence.py
  _Verify:_ python -m pytest tests/test_walker_core.py tests/test_evidence.py -q
  _Lessons:_ gt.chunk_salient never exposes raw BM25 scores, only the
  rank-ordered `kept` list -- P3's tie-break has to be a rank proxy
  (`term_bm25_rank`, sum of normalised rank over member chunks), not a real
  BM25 comparison; T40 should not expect true score magnitudes out of
  `rank_group_terms`'s `bm25` field, only relative order. No
  `.specs/file-manifest.md` exists in this repo tree -- proceeded without a
  manifest row.

- [DONE] T40 UI restructure: ONE analysis tab (images before partitions, group
  background shading, louvain relative LH vs global RH columns, entity/relation
  class tables), neo4j mirror to its OWN tab (temporary: buggy; port back when fixed)
  plus P8: LLM reason-over-walk fires automatically after the walk, no button
  _Files:_ walker_app.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual
  _Lessons:_ `nx.spring_layout` positions are numpy arrays, not tuples --
  `_hull_shapes` had to cast to `(float,float)` before `set()`/sorting or
  Streamlit's AppTest raised "unhashable type: numpy.ndarray" the moment a
  walk rendered the Groups panels (caught by the two live AppTest cases, not
  the plotly-only ones). Renamed the Community map's `q = gt.quotient(...)`
  local to `_qrows`: moving that block into the same scope as the prompt
  variable `q` would have clobbered it before the Evidence section's
  `gt.term_stats(conn, run, q, ...)` call -- a real bug the merge would have
  introduced, not a stylistic choice. No hulls added to `draw_communities` /
  `draw_global_map` (they draw one node per community; a hull around a single
  point is noise, colour already carries similarity). Left the inline
  `SELECT ... FROM community` unclaimed by T40 (6.21(c)'s helper extraction is
  a separate task). Sprawl review: collapsed N/A -- no incumbent duplicated;
  extended `cid_color` (as `group_color`) rather than adding a second palette.
  pytest: 20 passed in 83.61s. diag_rerun: PASS 18/20 | FAIL ['E3'] |
  KNOWN-FAIL ['A2'] (byte-pinned digest unchanged).

- [DONE] T41 Live look: run a real query through the restarted walker, screenshot
  the Analysis tab, verify P1-P8 visually (images first, shading, LH/RH louvain,
  auto-LLM narration present) -- surface the screenshot to the operator
  _Files:_ (none -- verification only)
  _Verify:_ playwright screenshot of :8501 with a prompt run
  _Lessons:_ live run 'the end of the war in europe': auto-LLM fired (grounded
    answer, cite #8709), two tabs only, partitions+groups+shading all render;
    mirror-skip warning proved the narrowed except path (neo4j was down; now up
    via compose after removing the stale hand-run container). Group relation
    counts are GLOBAL supports per P5 semantics -- can read as inflated.

## Layer 14 -- sequential (factbook panels)
Source: operator screenshots + plan, 2026-09-05.

- [DONE] T42 Spec 6.22 P10-P12: bordered panels, TOON factbook digests replace
  dataframes (dwpc scores shown, lift suppressed at saturation, dedup marker),
  per-chain community lines
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "P1[012]" .spec/specs/graph-explorer/design.md

- [DONE] T43 Core factbook renderers: group_digest, dedup_groups (with P11(e)
  divergence decomposition), chain_communities (pure text, no DB/streamlit)
  _Files:_ walker_core.py, tests/test_walker_core.py
  _Verify:_ python -m pytest tests/test_walker_core.py -q
  _Lessons:_ `_pack`'s width budget must include the caller's own label
  (pass it as `prefix=`, not string-concatenated after) or truncation checks
  undercount by the label's length; overflow from `max_items` must clamp at
  0 (`total - max_items` goes negative when the list is already short).

- [DONE] T44 UI panels: st.container(border=True) x4, st.code digests replace
  the 4 styled dataframes, chain community lines under partitions
  _Files:_ walker_app.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual
  _Lessons:_ sprawl review: collapsed 2 (the two duplicate `src_of_map` calls
  now share `src_of_all`; `style_group_table` + its lone `import pandas`
  deleted, dropping pandas from walker_app's import surface). 21 passed;
  diag gate `PASS 18/20 | FAIL ['E3'] | KNOWN-FAIL ['A2']` unchanged.

- [DONE] T45 Live look: restart walker, run "how a city rebuilds after a
  disaster", screenshot each panel, surface to operator
  _Files:_ (none -- verification only)
  _Verify:_ playwright screenshots of :8501

## Layer 15 -- sequential (colored factbook, one figure)
Source: operator phone review, 2026-09-05 ("all one color... left panel is
adding no utility").

- [DONE] T46 Spec 6.22 P14: colored digest cards, one full-width global figure
  (relative louvain becomes text-only), section accents, tinted chain rows
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "P14" .spec/specs/graph-explorer/design.md

- [DONE] T47 Colored cards: walker_core.digest_card (pure HTML, escaped,
  text byte-identical to group_digest), walker_app one-figure Groups panel +
  card loop + section accents + tinted chain rows
  _Files:_ walker_core.py, walker_app.py, tests/test_walker_core.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_core.py tests/test_walker_render.py -q && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual
  _Lessons:_ moving PALETTE/cid_color/group_color/rgba into walker_core also broke an
    unrelated pre-existing source-grep test (test_mirror_is_the_neo4j_tab) that literal-matched
    "### Groups"/"### Partitions (dendrite sort)"/"relative (this walk only)" in walker_app.py;
    updated those assertions to match the panel_head()-based source (collateral fix, same owned file).

- [DONE] T48 Live look: >=3 distinct card hues in the DOM, screenshots to operator
  _Files:_ (none -- verification only)
  _Verify:_ playwright screenshots of :8501

## Layer 16 -- sequential (dark dashboard skin)
Source: operator reference image (dark SaaS dashboard), 2026-09-05.

- [DONE] T49 Spec 6.22 P15: dark theme, KPI stat cards, card-grid factbooks,
  pill badges, dark plotly -- a SKIN; every P10-P14 content contract stands
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "P15" .spec/specs/graph-explorer/design.md

- [DONE] T50 Implement the skin: .streamlit/config.toml dark theme, stat row,
  group cards in st.columns(3) grid with badges, judged-evidence pill rows,
  plotly dark template, answer hero card
  _Files:_ .streamlit/config.toml, walker_core.py, walker_app.py, tests/test_walker_core.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_core.py tests/test_walker_render.py -q && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual
  _Lessons:_ implementer yielded on its own background pytest; orchestrator
    verified (85 passed; diag 18/20 FAIL=[E3]) and closed per the standing
    cross-session recovery law.

- [DONE] T51 Live look: dark render, stat row, card grid, pills -- screenshots
  to operator
  _Files:_ (none -- verification only)
  _Verify:_ playwright screenshots of :8501

## Layer 17 -- sequential (interactive 3D walk explorer)
Source: operator, 2026-09-06 (3d-force-graph + TF-projector interactions;
force layout chosen over embedding PCA).

- [DONE] T52 Spec 6.22 P16: 3d-force-graph scene contract (CDN pins, payload
  shape, hover reveal, community dim, path cycle, degradation)
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "P16" .spec/specs/graph-explorer/design.md

- [DONE] T53 Scene builder: walk3d_payload + walk3d_html in walker_core (pure,
  escaped, pinned CDN) + DB-free tests; payload carries optional per-node
  umap xyz and the scene a force<->umap toggle (P16(i))
  _Files:_ walker_core.py, tests/test_walker_core.py
  _Verify:_ python -m pytest tests/test_walker_core.py -q -> 72 passed
  _Lessons:_ string.Template (not f-string) for the JS/CSS scene doc avoids
  brace-escaping hell; the `</` -> `<\/` guard fires on every `</` in the
  dump (e.g. tip's own `</b>`), not just a literal `</script>` in body text.
  _Verify:_ python -m pytest tests/test_walker_core.py -q

- [DONE] T54 UI wiring: t_3d sub-tab swaps draw_layers3d for components.html
  (+ evidence helper fetching walk embeddings -> umap xyz, dense runs only);
  delete dead draw_layers3d; live look with screenshots to operator
  _Files:_ walker_app.py, evidence.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual
  _Lessons:_ subplan's "pass ai['edges'] unchanged" was wrong -- analysis_inputs'
  edges is `{(min,max): strength}`, not a list of 3-tuples/dicts; walk3d_payload
  needs `[(a,b,w) for (a,b),w in ai["edges"].items()]`. 27 passed; diag gate
  PASS 18/20 | FAIL ['E3'] | KNOWN-FAIL ['A2'] (unchanged).

- [DONE] T55 Remove the sidebar (P17): run selector + stats + details popover
  as a compact top row; warnings inline; no st.sidebar anywhere
  _Files:_ walker_app.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual

- [OPEN] T56 Architect the holistic re-layout: inventory every UI element,
  categorize, propose fengshui placement -- ledger + mock only, NO code;
  present to operator before any implementation
  _Files:_ playbook.md (next layer)
  _Verify:_ operator review

## Layer 18 -- sequential (feng-shui zones, T56 resolved)
Source: operator, 2026-09-06 ("holistic feng-shui redesign, top-down
categorization mindset"). Plan approved.

- [DONE] T57 Spec 6.22 P18: five categories (VERDICT/EVIDENCE/STRUCTURE/
  LENSES/REFERENCE) + zone layout; supersedes P10 order clause
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "P18" .spec/specs/graph-explorer/design.md

- [DONE] T58 Zone re-placement in walker_app: verdict|evidence columns, 3D
  first sub-tab, louvain pair as own sub-tab, community map to Zone 3 right,
  global-map tab merged away, walk trace collapsed at bottom. Placement only.
  _Files:_ walker_app.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual
  _Lessons:_ implementer yielded on its own background diag; orchestrator
    verified (32 passed; 18/20 FAIL=[E3]) and closed per standing recovery law.

- [DONE] T59 Live look: zone screenshots to operator
  _Files:_ (none -- verification only)
  _Verify:_ playwright screenshots of :8501

## Layer 19 -- sequential (agentic retrieval v0: ReAct over the walk)
Source: operator, 2026-09-06 ("nirvana" failure: 0-entail answer asserted
confidently; want a react agent doing agentic retrieval until sufficient
evidence, llm-as-judge sufficiency call, on-the-fly depth/hops).

- [DONE] T60 Spec 6.23: sufficiency gate + ReAct loop contract (A1-A7)
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "6.23" .spec/specs/graph-explorer/design.md

- [DONE] T61 react.py: sufficiency judge + action proposal + bounded loop over
  sampler.ef_evidence knobs (ef/k_anchor/rings/aliases/reformulated query),
  evidence accumulation with per-iteration provenance; A8 score-stats
  history (params -> mean/sdev/entails) fed to the proposer + tests
  _Files:_ react.py, tests/test_react.py
  _Verify:_ python -m pytest tests/test_react.py -q
  _Lessons:_ 15 passed in 177.61s (live smoke skips clean against the down DB,
  ~3min is gt.connect()'s own timeout, not a defect here); added A9 derived
  metrics (precision_proxy, contradiction_rate, gold_recall_evidence via a
  pure gold_recall() helper) to stats()/IterationRecord/history table per the
  post-subplan spec addition -- no extra LLM calls, no edits outside the two
  owned files.

- [DONE] T62 Wire-in: zero-entail answer gate in the hero (immediate fix) +
  react loop engaged when insufficient; iteration trace in the EVIDENCE zone
  _Files:_ walker_app.py, walker_core.py, tests/test_walker_render.py, tests/test_walker_core.py
  _Verify:_ python -m pytest tests/test_walker_render.py tests/test_walker_core.py -q && PYTHONPATH=. python tools/diag_rerun.py mixed-full-dual
  _Lessons:_ D3's inject-through-walk_fn/judge_fn seam held exactly as scoped --
  react_for costs zero extra walks/calls for iteration 0; 111 pytest passed,
  diag gate unchanged at PASS 18/20 | FAIL ['E3'] | KNOWN-FAIL ['A2'].

- [DONE] T63 Gold diagnostic lane (A10): tools/diag_agentic.py + G-class rows
  authored against verified corpus content (kurt_cobain/nirvana check first)
  _Files:_ tools/diag_agentic.py, tests/test_react.py
  _Verify:_ PYTHONPATH=. python tools/diag_agentic.py mixed-full-dual

- [WIP] T64 Live look: the 1990s-musician query through the loop; screenshots
  _Files:_ (none -- verification only)
  _Verify:_ playwright screenshots of :8501
