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
  _Files:_ .gitignore, src/diag_rerun.py, playbook.md
  _Verify:_ git ls-files | grep -E "[.](png|html|zip)$" returns empty && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
  _Notes:_ Sequential -- moves the do-no-harm gate itself. git rm --cached ~17MB
  (graph3d.html, irl_tank.html, trigram.zip, suite*.png, graph_brown*.png,
  community_labels.json); git mv .tmp/diag_rerun.py tools/ + re-prove 18/20 fails=[E3];
  purge .tmp/neo4j_export (355MB) + stale one-offs; rotate/remove .env NVIDIA token;
  sweep root png/yml litter.
  _Lessons:_ Gate moved to src/diag_rerun.py and re-proven (18/20 fails=[E3]) before
  anything else. ~17MB untracked (0 tracked binaries remain); .tmp 429MB -> 4MB
  (neo4j_export purged; stale one-offs deleted per plan -- read_partitions/neo4j_backfill/
  reason_digest_test go; evidence.assemble makes them trivial to rebuild). .env moved to
  ~/.env.graph.bak OUTSIDE the repo -- OPERATOR ACTION: rotate the NVIDIA token at the
  provider; the file sat in plaintext in the working tree. gitignore covers the
  regenerable patterns so they cannot re-track.

## Layer 11 -- parallel (plumbing; disjoint files, dispatch together)
- [DONE] T32 config.py: one source for DSN, neo4j endpoint, model dir, ports
  _Files:_ config.py, graph_tools.py, pg_store.py, entities.py, export_neo4j.py, text2cypher.py, patch_neo4j_browser.py
  _Verify:_ pytest tests/ --ignore=tests/test_mixed_acceptance.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
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
  _Files:_ evidence.py, walker_app.py, src/diag_rerun.py, tests/test_evidence.py
  _Verify:_ digest bytes for all 20 frozen rows identical before/after && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual && full offline suite
  _Notes:_ STRONGEST gate: byte-pin render_digest across the frozen 20 before the edit,
  diff after (pass-count can mask reordering). walker_app 3 sites + diag_rerun rewired
  (other .tmp copies deleted in T31). First-ever tests for this block.
  _Lessons:_ real count was 3 live walker_app sites (walk_state, _dendrite_state,
  assess block), not 5x -- the other two were .tmp/ copies T31 already deleted; grep
  confirmed before writing anything. Byte-pin instrument proved stable first:
  before-vs-before2 20/20 SAME, embed=on (CHUNKGRAPH_MODEL_DIR resolved). Post-
  extraction before-vs-after (.tmp/pin_digest.py --diff) 20/20 SAME across all 20
  frozen rows. `src/diag_rerun.py mixed-full-dual` unchanged: PASS 18/20 | FAIL
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
  python src/diag_rerun.py mixed-full-dual` -- "PASS 18/20 | FAIL ['E3'] |
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
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
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
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
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
  _Verify:_ python -m pytest tests/test_walker_core.py tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
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
  _Verify:_ python -m pytest tests/test_walker_core.py tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
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
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
  _Lessons:_ subplan's "pass ai['edges'] unchanged" was wrong -- analysis_inputs'
  edges is `{(min,max): strength}`, not a list of 3-tuples/dicts; walk3d_payload
  needs `[(a,b,w) for (a,b),w in ai["edges"].items()]`. 27 passed; diag gate
  PASS 18/20 | FAIL ['E3'] | KNOWN-FAIL ['A2'] (unchanged).

- [DONE] T55 Remove the sidebar (P17): run selector + stats + details popover
  as a compact top row; warnings inline; no st.sidebar anywhere
  _Files:_ walker_app.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual

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
  _Verify:_ python -m pytest tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
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
  _Verify:_ python -m pytest tests/test_walker_render.py tests/test_walker_core.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
  _Lessons:_ D3's inject-through-walk_fn/judge_fn seam held exactly as scoped --
  react_for costs zero extra walks/calls for iteration 0; 111 pytest passed,
  diag gate unchanged at PASS 18/20 | FAIL ['E3'] | KNOWN-FAIL ['A2'].

- [DONE] T63 Gold diagnostic lane (A10): src/diag_agentic.py + G-class rows
  authored against verified corpus content (kurt_cobain/nirvana check first)
  _Files:_ src/diag_agentic.py, tests/test_react.py
  _Verify:_ PYTHONPATH=. python src/diag_agentic.py mixed-full-dual

- [DONE] T64 Live look: the 1990s-musician query through the loop; screenshots
  _Files:_ (none -- verification only)
  _Verify:_ playwright screenshots of :8501

Campaign close (2026-09-06): agentic retrieval v0 complete, T60-T64 DONE.
Gold lane AGENTIC 5/5 (evid=1.0 all rows) after the A3 fixed-point guard and
A9 bodies-scored recall; live UI shows the gated hero + the loop finding
nirvana/cobain chunk #6323 at iteration 1. Frozen diagnostic untouched:
PASS 18/20 | FAIL [E3] | KNOWN-FAIL [A2] (A7 held).

## Layer 20 -- sequential (SNR-guided proposer + loop answer)
Source: operator SNR frame + log2 deviation band + estimator-pair bound,
2026-09-06. Plan approved.

- [DONE] T65 Spec 6.23 A12 (log2 estimator-pair dilution band, WIDEN
  exclusion, REANCHOR-first ladder) + A13 (entails-first loop answer)
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "A1[23]" .spec/specs/graph-explorer/design.md

- [DONE] T66 react.py: dilution_detected (log2 band, min(mean-sdev,
  median-1.4826*MAD)), WIDEN exclusion in propose, ladder reorder,
  entails_first_bundle -> result["answer_bundle"]
  _Files:_ react.py, tests/test_react.py
  _Verify:_ python -m pytest tests/test_react.py -q
  _Lessons:_ "bundle" in run()'s return was only ever the LAST walk's bundle,
    not a union (A5 says evidence unions) -- built the real accumulated-union
    bundle (scores_all/origin_all) so answer_bundle draws from every
    iteration's entails, not just the final walk's. 26 passed, 1 deselected
    (live_net); full DB-free suite 386 passed, 1 skipped.

- [DONE] T67 Loop answer wired: hero answers from answer_bundle when the loop
  found entails ("answered after N agentic iterations"); diag_agentic swaps
  to answer_bundle
  _Files:_ walker_app.py, walker_core.py, src/diag_agentic.py, tests/test_walker_render.py, tests/test_walker_core.py
  _Verify:_ python -m pytest tests/test_walker_render.py tests/test_walker_core.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
  _Lessons:_ base-walk-sufficient path (_entails>0) stays byte-identical to
    pre-T67 (A7); the new branch only engages inside the existing
    `_entails == 0` react_for gate. Added a cached loop_answer_for wrapper
    (distinct key from assess_for/react_for) that calls interpret.answer on
    react.run()'s answer_bundle, keyed (run_id, q); citation pills come
    straight from rx["entails"], not the model's own cited list, per spec's
    "pills from the entailing set." When the loop-answer call itself fails
    (not ok / empty answer) despite found entails, falls back to
    answer_gate's found_entails branch so the superlative caution still
    shows (A13's "cannot crown a single candidate" case). diag_agentic.py:
    dropped the react.cap_bundle(bundle, 50) re-derivation entirely --
    result["answer_bundle"] is already the A5-capped, entails-first bundle
    (Article VI: no re-deriving what react.run() already computed).
    142/142 pytest (test_walker_render + test_walker_core + test_react,
    -m "not live_net"); diag_rerun gate held byte-identical: PASS 18/20 |
    FAIL ['E3'] | KNOWN-FAIL ['A2'] (A7 confirmed unmoved).

- [DONE] T68 Gates + gold rerun (ans delta vs 5/5 baseline) + live look
  _Files:_ (none -- verification only)
  _Verify:_ PYTHONPATH=. python src/diag_agentic.py mixed-full-dual

Campaign close (2026-09-07): SNR-guided proposer + loop answer, T65-T68 DONE.
Gold ans-recall vs baseline: G4 0.0->0.5, G5 0.0->0.5 (single-row re-probe;
its suite-run evid=0.5 was proposer variance -- re-probe evid=1.0 PASS),
G1/G2 hold 1.0, G3 holds 0.5. evid stays 1.0 on every verified row. Live UI:
the Gallagher query now answers "Nirvana s Nevermind marked grunge
phenomenon #6323..." after 3 agentic iterations. Frozen diagnostic
PASS 18/20 | FAIL [E3] | KNOWN-FAIL [A2] (A7 held). NOTE: the loop is
LLM-proposer-stochastic run to run; per-row single runs are noise-prone --
n>=3 per row before any future pass/fail claim on a single G-row delta.

## Layer 21 -- sequential (unsupervised classes: entities and relations)
Source: operator, 2026-09-06 ("top entities by class... unsupervised classes
created however that would look like, my guess was co-occurrence analysis";
"graph-based entity resolution but for relations"); executed on "make it so",
2026-09-07.

- [DONE] T69 Spec 6.24: E15-E19 -- entity co-mention graph (NPMI-weighted,
  from mentions) -> Louvain entity classes, batch + fixed seed, stored
  additively (E8 pattern); DIRT-style relation classes (template similarity
  over shared (src,dst) pair sets -> components); per-class top entities and
  top relations; walk-local relation scoping for group digests
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "E1[5-9]" .spec/specs/graph-explorer/design.md

- [DONE] T70 Entity classes build: entities.py additive pass (or classes.py
  if entities is sealed -- check E3 first) computing class_id over the
  co-mention graph; CLI + per-stage timings (E14 discipline); live build on
  mixed-full-dual + smoke pin
  _Files:_ entities.py or classes.py, tests/test_entities.py or tests/test_classes.py
  _Verify:_ live build completes inside Article VII + pytest -q on its tests
  _Lessons:_ extended entities.py (E3 check: it seals the read set, not the
    file -- canonical_id was already additive there, same shape). Pair-bound
    receipt (E19): 5,813,717 mentions / 10,830 chunks ~ 537 entities/chunk
    mean -> unbounded sum C(k,2) ~ 1.5e9 pair slots, the E9 wall (killed at
    2.47e9). A PER-CHUNK top-32 cut (not E9's corpus-wide VOCAB_BOUND, which
    would class only 65 entities total) bounds it to
    10,830 * C(32,2) ~ 5.4e6 slots. Live `python entities.py mixed-full-dual
    --classes-only`: entities=275328 n_chunks=10816 pairs=2653691 edges=16073
    classes=270620 classed=4708 (largest class 338 members); stages
    read+pairs=17.76s graph=0.49s louvain=1.47s write=6.84s, total ~27s --
    well inside Article VII's 15 min, no CLASS_CHUNK_TOPK tightening needed.
    Full-suite regression: 690 passed, 3 skipped, 0 failed.

- [DONE] T71 Relation classes build: template pair-set similarity -> class
  ids stored beside relations (additive table/column); live build + smoke
  _Files:_ relations.py, tests/test_relations.py
  _Verify:_ python -m pytest tests/test_relations.py -q + live build
  _Lessons:_ new `relation_classes` mapping table (run_id,template)->rel_class,
  never an ALTER on the PK-heavy `relations`; live `--rel-classes-only` on
  mixed-full-dual measured T=11,328 distinct templates (not the "dozens"
  planned) so similarity took 65.21s (still well inside Article VII) --
  genitive acceptance verdict (a) FAILED live (GEN/'s, GEN/of, GEN/of the
  land in separate singleton classes; 864 distinct GEN/<connector> keys,
  virtually all singletons) because `connector` rides free surface text, not
  a canonical few forms; verdict (b) PASSED-partial (the/w/in/a groups
  co-class sensibly, 301/11328 templates classed). Full suite: 15/15 passed.

- [DONE] T72 Surface: REFERENCE-zone panel "Classes" (top entities per class,
  top relations per class, factbook style); group digests annotate entities
  with class; relations lines in digests become walk-local-scoped; ONE
  counts fn (chunks/entities/relations) applied to global cids, relative
  louvain AND the correlation-sorted dendrite chains (E17 amendment)
  _Files:_ evidence.py, walker_core.py, walker_app.py, tests/test_walker_core.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_core.py tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
  _Lessons:_ group_counts is the one new pure fn (E17 amendment); class_reference
  is the one new evidence.py query, called only through classes_for(run_id) so it
  never joins analysis_inputs' per-(run,prompt,ords) cache. 132 pure+live_db tests
  green; diag gate held exactly at PASS 18/20 | FAIL ['E3'] | KNOWN-FAIL ['A2'].

- [DONE] T73 Live look: classes panel screenshots to operator
  _Files:_ (none -- verification only)
  _Verify:_ playwright screenshots of :8501

Campaign close (2026-09-07): unsupervised classes T69-T73 DONE. Entity
classes 2653691 pairs -> 16073 edges -> 4708 classed (27s); relation classes
11328 templates -> 396 edges -> 301 classed (68s); Classes panel live in the
REFERENCE zone; E17 counts on all three groupings (live: chain 1 . 82 chunks
. 10582 entities . 163769 relations . c6:79 c12:3). Frozen diagnostic
PASS 18/20 | FAIL [E3] | KNOWN-FAIL [A2]. OPEN defect: E16 gen-check
negative -- genitive connector variants do not co-class because connector
holds raw surface spans; fix is canonicalizing connectors (run the check
over template, not connector), NOT lowering the threshold.
Wiki updated: "C:/Users/user/Documents/wiki/data science/llm/unsupervised
entity and relation extraction.md" (renamed 2026-09-07) -- implementation notes appended (3 deviations + the failed
acceptance test), operator text untouched.

## Layer 22 -- sequential (superlative gate; the Gallagher regression)
Source: operator, 2026-09-07 -- Gallagher crowned again with 2 entails / 2
contradicts; A6's superlative clause was never implemented off the
zero-entail path, and no diagnostic covered the base-answer path.

- [DONE] T74 Spec 6.23 A14 (a-d): superlative predicate, candidate-set
  rendering, citation gate, coverage admission
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "A14" .spec/specs/graph-explorer/design.md

- [DONE] T75 KNOWN-BAD FIRST: a failing check before any fix -- pure
  is_superlative + gate predicate, and a test that feeds THIS answer
  ("Noel Gallagher is the most famous musician...", entails=[3594,7666],
  cites 7666 not entailing) and asserts it is REJECTED. Test must fail
  against today's answer_gate, then pass after T76.
  _Files:_ tests/test_walker_core.py
  _Verify:_ python -m pytest tests/test_walker_core.py -q (expect the new
  tests RED first -- record the red output verbatim)

- [DONE] T76 Implement: answer_gate gains the superlative + citation
  branches; walker_app renders the candidate set when gated
  _Files:_ walker_core.py, walker_app.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_core.py tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual
  _Lessons:_ the entails>0 branch never called answer_gate at all (A7's
  "byte-identical" comment hid the gap) -- routed it through the same gate
  with prompt/entail_ords/answer_ords/superlative_entails, all new kwargs
  keyword-only with no-op defaults so every old caller stays byte-identical;
  137/137 green (incl. the 4 KNOWN-BAD-FIRST + a new cited_ords test) and
  diag_rerun unchanged at 18/20 FAIL=[E3] KNOWN-FAIL=[A2].

- [OPEN] T77 Coverage: G-rows for the BASE path (entails>0) in
  src/diag_agentic.py, asserting the gate; live rerun
  _Files:_ src/diag_agentic.py, tests/test_react.py
  _Verify:_ PYTHONPATH=. python src/diag_agentic.py mixed-full-dual

- [DONE] T78 Live look: the musician query must NOT crown; screenshot
  _Files:_ (none -- verification only)
  _Verify:_ playwright screenshot of :8501

- [DONE] T79 Class naming: name = argmax(mass x distinctiveness) with the
  PPMI/df demotion band; class_id stays min(members) as the join key
  _Files:_ evidence.py, tests/test_evidence.py
  _Verify:_ python -m pytest tests/test_evidence.py -q + live spot-check that
  the song/album class no longer reads "later"
  _Lessons:_ class_labels() SQL already existed unused (prior commit); wired it
  into walk_entities/class_reference and extracted the ranking arithmetic into
  a pure pick_class_label() for a DB-free test. Live: class 3502 (song/album/
  became/later) scored album=35172 > song=32383 > later=13521 > became=13432,
  so label -> "album", not "later".

- [DONE] T80 A15: the gate becomes the loop trigger -- react fires whenever
  answer_gate would gate the base answer (0 entails OR superlative-unranked
  OR bad citation), gate re-runs over accumulated evidence afterward
  _Files:_ walker_app.py, walker_core.py, tests/test_walker_core.py, tests/test_walker_render.py
  _Verify:_ python -m pytest tests/test_walker_core.py tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual

- [DONE] T81 A16 reflexive superlative guard + A17 (yield column, per-token
  dedup, no-movement stop, honest budget semantics): is_reflexive_superlative()
  pure predicate; discount reflexive spans from superlative_entails; Reason
  prompt states the population-vs-self scope distinction
  _Files:_ walker_core.py, walker_app.py, interpret.py, tests/test_walker_core.py
  _Verify:_ python -m pytest tests/test_walker_core.py tests/test_walker_render.py -q && PYTHONPATH=. python src/diag_rerun.py mixed-full-dual

## Layer 23 -- parallel (operator: "everything you suggested", 2026-09-07)
Disjoint files, dispatched together.

- [DONE] T82 A17(c) loosened: no-movement stops on ONE condition -- an
  iteration added no NEW entailing chunks (drop mean-flat/cid-flat); verify
  against the gold lane (keep if 5/5 holds with fewer iterations, revert if a
  row drops -- the extra iterations were then load-bearing)
  _Files:_ react.py, tests/test_react.py
  _Verify:_ python -m pytest tests/test_react.py -q && PYTHONPATH=. python src/diag_agentic.py mixed-full-dual

- [DONE] T77 Base-path G-rows: G6-G8 whose base walk has entails>0 so the
  UI's non-loop path is covered; assert the gate fires (A14/A15)
  _Files:_ src/diag_agentic.py, tests/test_react.py
  _Verify:_ PYTHONPATH=. python src/diag_agentic.py mixed-full-dual

- [DONE] T79 Class naming: argmax(mass x ln(chunks/df)) -- the PPMI demotion
  band -- replaces min(entity_id); class_id stays the join key
  _Files:_ evidence.py, tests/test_evidence.py
  _Verify:_ python -m pytest tests/test_evidence.py -q + live: the song/album
  class no longer reads "later"
  _Lessons:_ see T79 above (same task listed twice in the ledger) -- 32/32
  pytest, live label for class 3502 is "album".

- [DONE] T83 A18: hedged superlatives never count; superlative evidence
  recounted over CITED chunks only (evidence about Frah cannot license a
  crown on Gallagher). Known-bad tests RED first, then green; 104 core +
  81 render/react pass; diag 18/20 FAIL=[E3].
  _Files:_ walker_core.py, walker_app.py, tests/test_walker_core.py
  _Lessons:_ I verified the trace table screenshot and never read the hero in
    the same shot -- the crown was visible in a screenshot I took and passed
    over. Read the ANSWER, not just the instrument, on every live look.

Layer 23 close (2026-09-07): T82 KEPT per its own decision rule -- gold lane
AGENTIC 8/9, all five original rows PASS, 7 of 9 now stop at 2 iterations
(no-movement) instead of 3. T77 base-path rows G6-G8 all PASS. T79 committed.
NOTE (process defect, mine): T82/T77 were swept into commit fb30808 by a
git add -A during the A18 fix, BEFORE this verification ran and before the
operator answered keep/revert. Verified after the fact; verdict happens to be
keep. The rule stands: verify, then ask, then commit.

- [DONE] T84 G9 gate hole: the superlative pin ("what was the deadliest
  hurricane on record") returns gate=ok -- the gate let a superlative
  through. Same class as the Gallagher defect, caught by the diagnostic this
  time. Diagnose whether is_superlative misses "-est" here or a cited chunk
  supplies a ranking claim that should not count.
  _Files:_ walker_core.py, tests/test_walker_core.py
  _Verify:_ PYTHONPATH=. python src/diag_agentic.py mixed-full-dual -> 9/9
  _Lessons:_ NOT a gate hole -- an INSTRUMENT hole. gate_reason() was a FOURTH
    answer_gate call site and never received entail_texts, so the diagnostic
    reproduced the exact Gallagher defect it exists to catch. Re-probe after
    the fix: gate=superlative, passed=True. The regression pin now scans
    src/diag_agentic.py too, not just walker_app.py -- a pin that only
    covers the app cannot see a defect living in the measuring device.

## Layer 24 -- sequential (correlation sorting actually partitions)
- [DONE] T85 W22/W23: dendrite_sort hops require the estimator-pair band
  (r >= max(mean+sdev, median+1.4826*MAD) over positive r), significance
  retained as necessary-not-sufficient; known-bad test first (a dense matrix
  must NOT yield one snake)
  _Files:_ gt_terms.py, tests/test_graph_tools.py or tests/test_gt_terms.py
  _Verify:_ pytest the gt tests -q && live: the musician walk yields >1
  non-trivial chunk chain

  _Lessons:_ the partitions were never missing -- they rendered as ONE chain
    of 87/88, which reads as nothing. Root cause measured, not guessed:
    p<0.05 at n=88 means r>=0.21 while the median positive r was 0.59, so
    57%% of pairs were eligible hops. Band fix -> live chunk chains
    [10,5,4,4,2,2,2,2] + singletons, 8 threads. Significance saturates at
    scale; magnitude relative to the observed distribution does not.

## Layer 25 -- threshold provenance (operator, 2026-09-08)
- [DONE] T86 Tag every constant DERIVED/CONVENTION/ARBITRARY per 6.26 T1-T2,
  best practice cited first; no value changes in this task -- labelling only
  _Files:_ relations.py, entities.py, react.py, gt_terms.py, walker_core.py
  _Verify:_ grep shows a class tag on every module-level numeric constant
  _Lessons:_ gt_terms.py has no module-level numeric constants (confirmed by
    grep, zero lines to tag); K1/B (entities.py, BM25 defaults) and
    MIN_SHARED_NEIGHBORS/GRAM_N (not in 6.26 T4's audit) needed fresh
    classification -- extended the audit rather than skipping them.
- [DONE] T87 Replace the highest-leverage ARBITRARY ones per T3: connector
  window from the observed inter-entity span distribution (band), support
  floors calibrated on the gold lane
  _Files:_ relations.py, entities.py, tests
  _Verify:_ PYTHONPATH=. python src/diag_rerun.py mixed-full-dual ->
    PASS 18/20 | FAIL ['E3'] | KNOWN-FAIL ['A2']
  _Lessons:_ the uncensored gap distribution is flat (no elbow), so T3(a)'s
    band derivation fails on this constant -- MAX_CONNECTOR_TOKENS moved
    4 -> 5 as CONVENTION (Church & Hanks 1990 +-5 window), not DERIVED; no
    test pinned the old value 4, tests/test_relations.py + test_react.py
    52 passed unchanged.

## Layer 26 -- sequential (three-tier chunking, small-to-big retrieval)
Source: operator, 2026-09-16. Root cause of the judge's 3/163 entail rate:
96% of documents were never chunked (measured).

- [DONE] T89 Spec 6.27 C1-C5 with the measured per-source Box-Cox fits
  _Files:_ .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "C[1-5]" .spec/specs/graph-explorer/design.md

- [DONE] T90 chunkgraph.py R23: per-source paragraph fit (headers excluded), L0/L1/L2
  with parent ids, recursive 537/215 splitter, reduce_overlaps de-overlap;
  DB-free tests incl. reconstruct-the-source property test
  _Files:_ chunkgraph.py, tests/test_chunk.py
  _Verify:_ python -m pytest tests/test_chunk.py -q
  _Lessons:_ landed INSIDE chunkgraph.py (extends _paras/_bc_center/_chunk), not a new
    chunker.py -- a parallel chunking implementation would be the sprawl the rules forbid.
    Two defects the synthetic fixtures could not catch, both found on the live corpus:
    (1) wikitext paragraphs are separated by a SINGLE newline, so the incumbent's
    re.split on blank lines reported one block per section and a CONSTANT paragraph count
    (lam=-96066, m=hi=1) that falsely licensed a character ruler; (2) an existence test
    for body-blanks flipped all 7,822 wiki docs to blank-separated on a handful of
    strays -- it must be the MEDIAN over documents. Both are now pinned as known-bads.
    Also widened a stale test_react pin ('no-movement' stop, from this session's A17).

- [OPEN] T91 Schema + ingest: node gains tier + parent_ord; ingest writes all
  three tiers; new run label mixed-full-3tier (C5: never supersede)
  _Files:_ sql/, pg_store.py, chunkgraph.py, ingest_mixed.py, tests
  _Verify:_ live ingest completes; per-tier counts reported

- [OPEN] T92 Small-to-big serve: anchor at L2, expand L2->L1->L0, cap at L0
  _Files:_ sampler.py, evidence.py, tests
  _Verify:_ python -m pytest tests/test_sampler.py -q

- [OPEN] T93 A/B on the gold lane: old run vs 3-tier, G1-G9, n>=3 per row
  _Files:_ (none -- measurement only)
  _Verify:_ PYTHONPATH=. python src/diag_agentic.py <both labels>

## Layer 27 — sequential (query path, 2026-09-17)
- [DONE] T96 R25 ONE tokenizer, digits kept, years emit their decade
  _Files:_ stoplist.py, chunkgraph.py, gt_terms.py, tests/test_chunk.py
  _Verify:_ python -m pytest tests/test_chunk.py -q
  _Lessons:_ the root cause of the 1990s-musician failure. Both tokenizers used
    re.findall(r'[a-z]+'), so every year was invisible to BM25 -- 5,061 qterms
    sampled, zero with a digit. Index and query agreed, so nothing looked broken.
    G1 evidence 0/2 -> 2/2, gold survival 0.778 -> 0.889. Took two attempts: the
    first emitted year AND decade adjacently, a 100%-collocated bigram the PHRASE
    stage merged into 2008_2000s, destroying both halves.

- [DONE] T97 A19 answer cap 50 -> ANSWER_CAP=100
  _Files:_ react.py, tests/test_react.py
  _Verify:_ python -m pytest tests/test_react.py -q
  _Lessons:_ the call site hardcoded 50 against cap_bundle's documented 100 and
    discarded 7 of the 10 gold chunks the walk found. Did NOT change the outcome;
    per-chunk chars halve as the count doubles (60k total is what binds).

- [DONE] T98 S20 dense entry point (gt_sql.dense_search), opt-in
  _Files:_ gt_sql.py, graph_tools.py, sampler.py
  _Verify:_ python -m pytest tests/test_sampler.py tests/test_graph_tools.py -q
  _Lessons:_ the HNSW index was built at ingest and never queried by anything.
    Three stacked bugs: (1) the index is global while queries filter by run_id, so
    LIMIT 200 returned 8 rows; (2) SET LOCAL is a no-op under autocommit=True;
    (3) my own wiring fetched k=min(k_anchor,ef)=3. After all three, k=100 returns
    100 rows with gold at ranks 20 and 45.

- [OPEN] T99 Dense anchors must survive the walk
  _Files:_ sampler.py
  _Verify:_ PYTHONPATH=. python src/diag_evidence.py ab-section
  _Notes:_ anchors enter (12 with dense vs 4 lexical) and the score-ranked
    expansion evicts them -- gold in bundle unchanged at 0/92. Design work.

- [DONE] T100 A20: the loop retrieves until sufficient or budget, never until bored
  _Files:_ react.py, tests/test_react.py
  _Verify:_ python -m pytest tests/test_react.py -q
  _Lessons:_ the operator's question ("how can it not do retrievals until sufficient")
    was literally true: no-movement and budget broke BEFORE the proposer was called.
    Measured: base walk gives 11 Selena / 0 Nirvana, judge entails 9, next iteration
    adds none, loop quits, UI answers "Hypothesis: Selena". A20(b) makes stagnation a
    HINT, never a halt. First attempt was a NO-OP -- `i >= max_iters - 1` is `i >= 2`
    at MAX_ITERS=3, byte-identical to what it replaced; the n=4 run caught it in one
    iteration. stop=budget iters=3 now on every run (was no-movement iters=2).
    NAMING RATE NOT TESTED: 3/4 vs baseline 3/4, inside the comparator's own noise.

- [SUPERSEDED] T100 The loop checks sufficiency LAST
  _Files:_ react.py
  _Notes:_ react.py:542-562 breaks on no-movement and budget BEFORE calling the
    proposer, so the sufficiency judge is consulted only when the loop both grew
    its entail count and has iterations left. stop=sufficient is essentially never
    observed. Operator asked why it does not retrieve until sufficient; this is why.

- [OPEN] T101 Every phrase the index merges is unreachable from a query
  _Notes:_ the PHRASE stage (R9) merges index tokens; gt_terms.tokenize never
    merges. So 17th_century, 1840s_1850s et al. can never be matched. Found while
    diagnosing T96.

## Layer 28 — sequential (domain term analysis sidecar, 2026-09-26)

Objective, operator's words: **"the intent behind the two datasets is domain
specific term analysis"** — neoplatonic texts and arxiv papers, per-domain BM25,
log2 score bands, df only for the >50% mask, BPE over survivors. Sidecar only:
no graph edges, no pg_store, so none of the measured n^2 pair-RAM wall.
Plan: C:\Users\user\.claude\plans\i-ve-been-thinking-about-quiet-cray.md

- [DONE] T104 Load both corpora and chunk them on their own statistics
  _Files:_ domain_corpora.py, tests/test_domain_corpora.py
  _Verify:_ pytest tests/test_domain_corpora.py -q
  _Lessons:_ font-size heading detection FALSIFIED before it was built — of 20
    PDFs, 8 have ZERO bold heading candidates, body-size-by-char-mass lands on
    FOOTNOTE text in the Lewy book (8.2pt => 20.1 "headings"/page), and italic is
    body text in 3 files. So the planned '##' emitter was dropped, not shipped.
    pymupdf blocks are LINES in half the PDFs and PARAGRAPHS in the other half
    (p90/p50: 100/93 vs 3597/2485), so chunks pack whole blocks to a measured
    char target and never split one. Ruler fitted on the only neoplatonic file
    with real sections (Lewy repaired md, 53 headings / 30 with paragraphs):
    target=7310 chars, lam=-0.080; arxiv fits its own at 1448. Conservation 0/19
    and 0/200 documents failed. sparsevec 1000-nonzero: neop 0%, arxiv 0.2%.
    A unit test caught md_sections returning the whole document as one section
    when a document had no headings — the exact R23(a) defect.

- [DONE] T105 Per-domain term selection: log2 BM25, equal-count bands, dual sigma
  _Files:_ domain_terms.py
  _Verify:_ PYTHONPATH=. python src/diag_domain_terms.py --arxiv 200
  _Lessons:_ banding BEATS global top-N at matched vocabulary size on probe
    recall, both domains: neop 0.67 vs 0.52, arxiv 0.88 vs 0.50. Three measured
    defects. (1) The stage is nearly INERT as a filter — admitted fraction is
    flat at ~87% per band (neop 79.8-100, arxiv 80.7-98.8), matching the ~87%
    predicted from a uniform-slice argument before the run. (2) The 50% df mask
    removed ZERO terms in both domains; no term reaches half the chunks.
    (3) Dropping salient_grams' `* sqrt(df)` (operator's design excludes df from
    the score) makes max-BM25 rank single-occurrence noise first: top neop-only
    terms are binghamton/gymnasia/flute, top arxiv-only are downarrow/jmath/plt,
    while plotinus/porphyry/proclus/soul are MISSED. Mechanism: max-over-chunks
    is maximised by a hapax in a short chunk, and idf crushes high-df central
    vocabulary into the bottom band where the ~12% cut lands — which is R2.5's
    idf-suppression claim, observed.

- [DONE] T106 The report, deterministic and model-free
  _Files:_ src/diag_domain_terms.py
  _Verify:_ PYTHONPATH=. python src/diag_domain_terms.py --arxiv 200
  _Lessons:_ carries the band table, admitted-fraction spread, which branch binds
    (parametric 1 / robust 9 in both domains), the outlier mask's measured bound
    and the df of every term it flags, probe recall vs global top-N at matched
    size, the BPE sweep, and the cross-domain diff.

- [OPEN] T107 BPE at scale_factor 2/3/5 is a no-op — the sweep axis is dead
  _Files:_ domain_terms.py
  _Notes:_ measured, all three arms identical: actual_vocab 18718 (neop) /
    38638 (arxiv), single-token 100%, mean_pieces=1.00. When the BPE budget
    EXCEEDS the term count, every term merges to one token and no shared subword
    is ever found; n_terms * 2 guarantees it. trigram/trigram.md:58 has the
    intended scale — "up to 300 merges, min pair freq 2". The factor must SHRINK
    the vocabulary below n_terms, or the sweep must be on a merge budget.
    Needs the operator's call on which.

- [OPEN] T108 The score ranks hapax noise above central domain terms
  _Files:_ domain_terms.py
  _Notes:_ see T105 defect (3). Three candidate fixes, none measured, none
    chosen: restore `max_bm * sqrt(df)` (the incumbent, but re-admits df to the
    score); collapse by mean-over-occurrences instead of max; or band on df as
    spec R2.1 specifies, which holds idf near-constant within a band and is the
    thing R2.5 was written to test. Operator's design decision, not mine.

- [OPEN] T109 LaTeX command names and one 152k-char unit leak through docling
  _Notes:_ arxiv-only top terms include downarrow, jmath, dashrightarrow,
    betainc, lessorequalslant — LaTeX macros surviving docling's markdown. And
    one docling "paragraph" is 152,264 chars (a reference list or table dump),
    which D4 never splits, giving p99=9183 against a 1448 target and driving all
    25 over-ceiling chunks.

- [DONE] T107 BPE: both parameterisations, and they are the same curve
  _Files:_ domain_terms.py, src/diag_domain_terms.py
  _Verify:_ PYTHONPATH=. python src/diag_domain_terms.py --arxiv 200
  _Lessons:_ fractional scale_factor and a merge budget trace ONE trade-off
    (smaller vocab -> more pieces per term), so they are reparameterisations of
    each other: f=0.2 -> 5,840 vocab / 9.1% single-token / 2.22 mean pieces;
    merges=10000 -> 8,479 / 14.0% / 2.04. f>=2 is inert at every factor
    (38,895 vocab, 100% single-token) because a budget above the term count never
    needs a shared subword.

- [DONE] T108 The score ranks hapax noise above central domain terms
  _Files:_ domain_terms.py
  _Verify:_ PYTHONPATH=. python src/diag_domain_terms.py --arxiv 200
  _Lessons:_ `max * sqrt(df)` -- the salient_grams incumbent -- is the ONLY arm
    that ranks domain terms at all. arxiv top-1000: 29 of 35 against ZERO for
    bare max, mean-over-occurrences, and df-banding alike. df here is a RANK
    WEIGHT that removes nothing, so the operator's "df only masks >=50%"
    constraint is intact; the two are different jobs. Two of my own positions
    were falsified: df-banding (spec R2.1), which I argued for twice and which
    came LAST on both domains, and mean-collapse, which I proposed as the fix.
    Also: the diagnostic list must be READ OFF the corpus -- the invented list
    named terms the corpora barely use and could not separate the arms.

- [OPEN] T110 With the score fixed, the band gate no longer earns its place
  _Files:_ domain_terms.py
  _Notes:_ neop banded 27/33 vs global top-N 33/33 at matched size; arxiv 35/35
    both. The earlier "banding wins 0.67 vs 0.52" was an ARTIFACT of the broken
    score -- banding was compensating for a bad ranking by keeping terms spread
    across bands, and that compensation is worthless once the ranking is right.
    The stage is also ~88% flat per band (inert as a filter) and the 50% df mask
    removes 8 terms on neop, 0 on arxiv. So three of the four stages in the
    operator's selection design now measure as near-no-ops. Operator's call:
    keep banding for a property not yet measured, or cut to rank + cardinality.

- [DONE] T111 Tune the sigma factor against retrieval recall
  _Files:_ src/diag_domain_recall.py, domain_terms.py, tests/test_domain_corpora.py
  _Verify:_ PYTHONPATH=. python src/diag_domain_recall.py --arxiv 200
  _Lessons:_ NULL RESULT, and the operator's framing was right to demand it --
    diagnostic term coverage could never have tuned this knob (every factor gave
    an identical top-1000). Built deterministic gold with no judge and no model:
    arxiv paper title -> its own chunks, arxiv body sentence -> its own paper,
    Lewy section heading -> that section, Lewy body sentence -> its document.
    Across the full factor range 0.00-1.00, recall@10 moves <=0.007 on all four
    evals while the vocabulary changes 4x; the per-eval winner varies at the third
    decimal. The instrument WORKS -- it separates full vocabulary from selected
    (+0.110 neop headings, -0.006 arxiv titles) -- so the FACTOR axis is inert,
    and must be: BM25 sums weights over query terms and the band only trims the
    bottom of the score distribution, where query terms never live. Factor set to
    0.00 on index size, the one criterion that responds. Two bugs of the same
    class in my own eval, both caught: headings paired against an independently
    filtered section list (hit@1 = 0.000, reported as MY defect not as a finding),
    then unstripped paragraphs that only substring-matched by accident. Also
    rewrapping sys.stdout at import time killed pytest capture for any test that
    imported the tool -- moved into main().

- [OPEN] T112 Does a lexical lane earn its place here at all?
  _Notes:_ the vocabulary question is now answered; the space question is not.
    Selected-vocabulary BM25 versus dense embeddings versus fusion, on the same
    four gold sets. That is a DIFFERENT experiment from the vocabulary arms --
    dense changes the scorer and the space, so it cannot serve as the baseline for
    a vocabulary change (it would attribute the effect to the wrong thing).

- [DONE] T113 Does the BPE subword index approximate full-vocabulary BM25?
  _Files:_ src/diag_domain_recall.py, domain_terms.py
  _Verify:_ PYTHONPATH=. python src/diag_domain_recall.py --arxiv 200
  _Lessons:_ YES, at 4-10% of the vocabulary -- this was the whole objective, in
    the operator's words "a tokenizer that provides reasonable lexical
    performance ... a way to approximate a smaller corpus (vocab)". A BM25 index
    over 5,309 BPE pieces (24x smaller than the 128,053-term full vocabulary, no
    OOV possible) beats full BM25 on arxiv title MRR (0.886 vs 0.879) and ties it
    exactly on neop body sentences (0.947 / 0.377 / 0.895, identical). Two costs
    stated bare: arxiv body-sentence recall@10 falls 0.106 -> 0.096, 9% relative,
    OVER the 5% shippable bar; and on neop headings whole-term selection (0.474)
    beats pieces (0.391), ~2 queries of 23. Mechanism both ways -- pieces share
    weight across embedding/embeddings (helps short queries) and blur covariance
    into co+variance (hurts specific ones). Every prior "recall" number in this
    layer was term-list COVERAGE, not retrieval; this is the first retrieval
    measurement of the tokenizer itself. Merge budget saturates at 2,900 on neop.

- [DONE] T114 Derive the vocabulary on a 100k-chunk subset and prove it transfers
  _Files:_ src/diag_subset_sparsevec.py, tests/test_diag_subset_sparsevec.py
  _Verify:_ PYTHONPATH=. python src/diag_subset_sparsevec.py --no-psql
  _Notes:_ operator's log-ratio allocation asks neop for 34,631 of its 588 chunks
    (59x infeasible); feasible rule = min(size, quota) -> all neop + seeded 58.9%
    of arxiv. Transfer test = subset-derived pieces vs full-derived pieces, both
    scored on the FULL arxiv gold. Full-corpus load measured at 585 s for 2,227
    docs vs 10 s for 200 -- 5x superlinear, a perf defect; cached to .tmp/.

- [DONE] T115 Index the sample as pgvector sparsevec and measure recall in psql
  _Files:_ sparsevec_store.py, tests/test_sparsevec_store.py, src/diag_subset_sparsevec.py
  _Verify:_ PYTHONPATH=. python src/diag_subset_sparsevec.py
  _Notes:_ <d,q> with d = BM25 weights over pieces and q binary IS the numpy score,
    so exact scan must match numpy to the digit (asserted on top-10 sets, same
    top-1000 truncation both sides) before any HNSW number is read. Then HNSW
    sparsevec_ip_ops at ef_search 40/100/400: recall + ms/query. 293 chunks
    (0.17%) exceed the 1000-nonzero cap; their vectors keep the heaviest 1000.
    Docker Desktop was down; started it (pid 35480, ledger row in .tmp/processes.md).
  _Lessons (T114):_ RAN AT 30k, NOT 100k -- the 100k run was reaped by the harness for
    system memory beside the operator's 14 GB GRPO job; 100k is UNTESTED. Vocabulary
    from 30,000 chunks (17% of arxiv) transfers to the full 168,794-chunk corpus with
    no measurable loss: 2,203 title queries, MRR 0.770 subset-derived vs 0.768
    full-derived vs 0.773 full-vocab BM25 (1,486,001 words); hit@10 0.897/0.895/0.899.
    8,763 pieces = 0.6% of the word vocabulary. The two piece vocabularies share only
    HALF their pieces (Jaccard 0.504) and score the same -- coverage matters, not the
    specific pieces. rec@10 -8% relative on titles, over the 5% bar, same pattern as
    every piece result. First rerun died on my .tolist() on a list; the cache build
    cost 767 s and is the only reason the rerun was 5 min.
  _Lessons (T115):_ PARITY PROVEN: psql exact scan reproduces numpy on 100.0% of
    2,143 title and 199 body queries (identical hit/recall/MRR to three decimals).
    HNSW sparsevec_ip_ops: ef=400 within ~1.5% of exact at 12-17 ms/q vs 49-73 exact;
    ef=40 loses 11% hit@10 at 4-6 ms/q. 29,964 rows, 77 MB, build 29 s, 62 rows
    (0.2%) truncated to their 1000 heaviest pieces. DEFECT, mine: chunks were sampled
    uniformly, which SPLITS every paper, so the psql gold sets are fragments and the
    absolute psql recall (hit@10 0.776) is NOT comparable to the transfer numbers
    (0.897). One neop heading query (1 of 23) has a different exact-scan top-10 SET
    with identical metrics -- hypothesis: score tie at rank 10 broken differently by
    Postgres and numpy's stable sort. Not checked.

- [OPEN] T116 Sample by DOCUMENT for the psql lane, not by chunk
  _Files:_ src/diag_subset_sparsevec.py
  _Notes:_ whole papers in or out, so title -> own paper gold is complete inside the
    index and psql recall becomes comparable to the full-corpus transfer numbers.
    Chunk-level sampling stays correct for VOCABULARY derivation (that was the
    operator's design and it transferred); only the retrieval eval needs documents.

- [OPEN] T117 The 1-of-23 exact-scan parity mismatch on neop headings
  _Files:_ src/diag_subset_sparsevec.py
  _Notes:_ print that query's rank-9..11 scores from both numpy and psql. If they tie,
    the "defect" flag threshold should compare score-sets, not ord-sets; if they do
    not tie, it is a real scoring divergence and the parity claim is wrong for it.

- [OPEN] T118 Full-corpus load is 5x superlinear (585-767 s for 2,227 docs vs 10 s for 200)
  _Files:_ domain_corpora.py
  _Notes:_ profile per stage before touching anything; one 3,395,651-char docling
    unit is the leading suspect. Cached for now, so it costs nothing per iteration.

- [DONE] T119 The 2x2: {raw words, BPE pieces} x {exact, HNSW} on the full population
  _Files:_ src/diag_sparsevec_arms.py
  _Verify:_ PYTHONPATH=. python -u src/diag_sparsevec_arms.py
  _Notes:_ operator asked for three arms at ef_search=40; measured -11% hit@10 from HNSW
    at ef=40 vs -0.5% MRR from the vocabulary, so a single-ef design is dominated by the
    index knob -- run the square, sweep ef {40,400}, index the FULL 169,382-row population
    (chunk sampling fragments gold: 0.776 vs 0.897). Verified 2026-09-27: sparsevec dim
    ceiling 1e9 (1,486,001 parses), SPARSEVEC_MAX_NNZ 16000 in the header (1,000 is the
    HNSW cap per docs, unverified on disk), HNSW returns at most ef_search rows (ef=40,
    LIMIT 100/50/40 -> 40) so every rec@50 at ef=40 reported 2026-09-26 was over 40 rows
    and is INVALID; chunkgraph._sparse_sim L2-normalises rows (692-693) -> raw arm uses
    salient_grams.bm25_matrix. Index bytes is a first-class column.
  _Lessons:_ PREMISE REVERSED -- the BPE sparsevec index is 40% BIGGER than raw
    (408.6 MB vs 292.9 MB, hnsw build 161 s vs 113 s). sparsevec is coordinate-list
    storage: bytes scale with NONZEROS PER ROW, not dimension, and a word splits into
    ~2 pieces. A compact vocabulary shrinks vocabulary-bounded structures (term table,
    tsvector dictionary, embedding matrix) and ENLARGES a sparsevec index. Recall,
    169,382 rows, 500 titles / 200 body: raw exact 0.896/0.945 hit@10; raw hnsw ef=40
    0.798/0.715 (-11%/-24%); bpe hnsw ef=40 0.822/0.690 (-8%/-27%); ef=400 raw
    0.886/0.890, bpe 0.888/0.860. So at ef=40 HNSW is the loss, not the vocabulary,
    and the vocabulary at matched HNSW is a wash (+3%/-3.5%, under the 5% bar). Short
    queries collapse under HNSW on BOTH arms: neop headings 0.23-0.26 exact ->
    0.04 at ef=40, 0.09-0.17 at ef=400 (n=22-23; hypothesis: 2-3-term queries give
    the IP graph no gradient). Truncation to 1,000 nnz cost nothing measurable.
    Parity: 100% titles both arms, 99.5% body both arms, raw neop 86.4% (19/22) with
    identical metrics -- rank-10 ties, three instances now (T117). Calibration: hnsw
    ef=400 posted MRR 0.766 vs exact 0.758 on raw titles, impossible for the same
    scores except via tie-breaking -> MRR deltas under ~0.01 here are tie noise.
    Visibility defect fixed for future runs: the UTF-8 stdout wrapper block-buffered
    regardless of `python -u`; line_buffering=True in all three tools.
  _Lessons (rec@50 rerun, operator: "drop non differentiating comparisons (rec@10) and
    only focus on rec@50"):_ rec@10 is capped at 10/76 = 0.13 by chunks per paper and
    every arxiv cell sat within 0.005 of each other -- non-differentiating, as the
    operator said. Under rec@50 (ceiling 0.66, cells at ~0.23) TWO conclusions change.
    (1) HNSW at ef=50 costs NOTHING on arxiv: raw 0.228 vs exact 0.232 titles, 0.118 vs
    0.115 body; BPE 0.210 vs 0.206, 0.102 vs 0.096 -- at 5-11 ms/q vs 94-147 exact.
    The "-24% at production settings" was a hit@k story: HNSW misses the single best
    chunk but fills the top 50 with other chunks of the same paper (hypothesis for the
    mechanism; the numbers stand). (2) The vocabulary IS the loss at every ef: BPE vs
    raw -8% titles / -14% body at ef=50, -11% / -17% at exact, over the 5% bar, and
    40% more disk. Under rec@50 there is no case for BPE-as-sparsevec on this corpus.
    Unchanged: short queries collapse under HNSW on both arms (0.364 -> 0.091 raw,
    0.522 -> 0.087 BPE at ef=50; ef=400 recovers half). rec@50 does not exist below
    ef_search=50 (pgvector caps rows at ef), so the sweep is {50,100,400}. Parity
    byte-identical to the first pass: deterministic rank-10 ties, not flakiness.
    --reuse worked once its equality check tolerated the 182 empty-vector rows.

## Layer 29 -- sequential (arxiv graph: junk out, everything in Postgres, a 30-minute service; operator 2026-10-03)
Source: this session, 2026-10-03. Operator: "can we fix everything you and I just brought up" and "can we setup
a service just like arxiv-llmtxt-ingest to automagically process remaining extracted markdowns into this graph
representation for us, checking every 30 minutes (15 minutes after arxiv-llmtxt-ingest)".
NO GOVERNING SPEC beyond those two instructions and ~/.skills/sparsevec-lexsem-graph. Executive decisions, basis
logged: (1) junk = the `latexi` image-OCR marker OR >=90% of lettered lines holding <=2 alphanumerics with >=8
such lines, or no letter or digit at all (259 of 56,310 chunks; every statistic tried first -- in-vocabulary share, single-token share,
short-line share -- overlapped legitimate math or pseudocode and was rejected; the live count is 259: 258 by
marker or line profile plus one chunk with no letter or digit); (2) between full rebuilds the
vocabulary, idf, avgdl and the embedding mean are FROZEN, new chunks get a nearest-centroid community marked as
such; (3) a full rebuild runs when new chunks reach 5% of the build (an arbitrary starting constant, logged
every cycle); (4) the service is a logon-started Scheduled Task running one watch loop, like its sibling.
- [DONE] T120 Flag extraction-junk chunks, never drop them, and exclude them wherever references are excluded
  _Files:_ domain_corpora.py, tests/test_domain_corpora.py
  _Verify:_ pytest tests/test_domain_corpora.py -q
  _Lessons:_ junk was its own communities AND the exemplars; four character statistics all overlapped legitimate math, so the rule is the exact image-OCR marker plus an extreme lone-letter-line profile (259 chunks). The synthetic chunker test needs the real fit: the toy corpus fits hi=13, too small to hold 8 lines.
- [DONE] T121 Keep the chunk text, vectors (sparse ip, sparse cosine, dense), communities, exemplars and draft summaries in Postgres
  _Files:_ sparsevec_store.py, tests/test_sparsevec_store.py
  _Verify:_ pytest tests/test_sparsevec_store.py -q
  _Lessons:_ CREATE TABLE IF NOT EXISTS bakes the vocabulary size into the column type, so a full build must drop the label's tables (reset_label_tables). Versions overwrite (delete_papers) and ords below the build's size are never reused (min_ord).
- [DONE] T122 Make the full build write what it derives to Postgres and leave nothing only in .tmp
  _Files:_ src/ingest_arxiv_sparsevec.py, src/arxiv_community_map.py, src/summarize_clusters.py, tests/test_arxiv_community_map.py, tests/test_summarize_clusters.py, tests/test_ingest_arxiv_sparsevec.py
  _Verify:_ pytest tests/test_arxiv_community_map.py tests/test_summarize_clusters.py tests/test_ingest_arxiv_sparsevec.py -q
  _Lessons:_ a cache valid by row count is not an identity: purge by content fingerprint. The summarizer indexed exemplar rows into the wrong list after junk exclusion until load_inputs used retrievable(). The PNG write died twice because a viewer held the file memory-mapped (Errno 22); save_figure falls back to a timestamped sibling. The figure now grows to its longest card.
- [DONE] T123 Process newly extracted markdowns into the graph between rebuilds and rebuild when enough have arrived
  _Files:_ src/arxiv_graph_service.py, tests/test_arxiv_graph_service.py
  _Verify:_ pytest tests/test_arxiv_graph_service.py -q
  _Lessons:_ doc_row must break weight ties by column index like keep_heaviest or an incremental row differs from the full build's; delete an overwritten paper BEFORE the duplicate-text check; a cycle must add missing columns and normalise ids itself, the live database predated both; a failed render is invisible to the build state, so pending_steps reads the PNG's age.
- [DONE] T124 Register the 30-minute watcher as a Scheduled Task offset 15 minutes from arxiv-llmtxt-ingest
  _Files:_ src/arxiv_graph_service.py
  _Verify:_ Get-ScheduledTask arxiv-graph-ingest; one live cycle in .tmp/arxiv_graph_watch.log
  _Lessons:_ registered arxiv-graph-ingest mirroring arxiv-llmtxt-ingest's settings; the slot is (sibling minute mod 15) + 17 re-read each cycle; first unattended cycle ran 15:19:01 on its slot.
- [DONE] T125 Serve hybrid search over the graph that returns each hit's WHOLE section (operator: "how do I use this rag myself ... a function for openwebui to use this backend to reconstruct whole sections for use with rag")
  _Files:_ src/arxiv_rag_api.py, tests/test_arxiv_rag_api.py
  _Verify:_ pytest tests/test_arxiv_rag_api.py -q; python src/arxiv_rag_api.py --ask "<question>" against the live build
  _Lessons:_ the sparse arm returned zero-score chunks padded to the pool until only ip > 0 counted; the default max_chars cuts a ~22k-character section and says so. Whole section = chunks sharing (doc_id, section_idx) with continuation headers stripped.
- [DONE] T126 Register the retrieval API as a logon Scheduled Task so OpenWebUI can load it as an OpenAPI tool server
  _Files:_ src/arxiv_rag_api.py
  _Verify:_ Get-ScheduledTask arxiv-rag-api; GET /health and /openapi.json answer on its port
  _Lessons:_ registered arxiv-rag-api (port 8780, 127.0.0.1); OpenAPI at /openapi.json; live HTTP search verified. Not verified against an OpenWebUI instance: none runs here.
- [DONE] T127 Show the entire section wherever a chunk is selected, re-aggregated from its chunks (operator: "the entire section re-aggregated (using reduce_overlaps)")
  _Files:_ domain_corpora.py, src/arxiv_rag_api.py, src/arxiv_community_map.py, tests/test_domain_corpora.py, tests/test_arxiv_rag_api.py, tests/test_arxiv_community_map.py
  _Verify:_ pytest tests/test_domain_corpora.py tests/test_arxiv_rag_api.py tests/test_arxiv_community_map.py -q; .tmp/arxiv_communities.md
  _Lessons:_ reduce_overlaps was NOT applied, on measured grounds: the chunks are stored disjoint (C5), 72 of 1,245 neighbouring pairs share an exact span and every sample is a repeat the SOURCE contains, so a trim would delete real text. 779 of 800 live documents reassemble exactly; the other 21 are omitted junk (16) and the keep-identical-chunks-once rule C8 (5, confirmed). The old rebuild recognised a continuation header by comparing text; chunk_idx > 0 says it exactly. The default cut moved from 20,000 to 200,000 characters because 2% of sections are longer than 20,000; the companion file cuts at the same cap, since one degenerate book section is 16.7M characters and was 95% of it. The FIRST delivery put the sections in arxiv_communities.md and left a 210-character snippet on the PNG cards, which was not what was asked ("I asked for the sections in the png"); the cards now draw each selected chunk's whole section (34 of the 36 on the 12 cards whole, 2 cut at 12,000 characters with a red note), the card grid grows by row and the map panel keeps its size (figure 2400x6338).

## Layer 30 -- sequential (arXiv entities by information theory, no spaCy; operator 2026-10-04)
Source: `C:\Users\user\.claude\plans\derive-arxiv-entities-with-information-theory.md` (approved 2026-10-04). Operator: "the one thing
we're really missing is actual entities ... I don't want to rely on spacy ... autonomous ways to derive entities using information theory
such as DIRT, NPMI, PPMI, PDMI" and "port this skill over to ~\.skills". Decisions: PDMI is a typo for PPMI (dropped); scope = port the
skill, then build entities; no entity kinds fixed in advance -- a hand-reviewed DIAGNOSTIC SET of sentences plus a hyperparameter sweep decides;
port to ~/.skills only. Guards AE1-AE13 live in the entity_derive.py docstring (entities.py owns E1-E19). Relations are the NEXT work item,
NOT TESTED here (they need the open connector-canonicalisation fix first).
- [DONE] T128 Port the entity skill to ~/.skills with frontmatter, expansions, chatbot tail removed; reload the retrieval index
  _Files:_ C:\Users\user\.skills\unsupervised-entity-relations\SKILL.md
  _Verify:_ frontmatter parses; body diff vs skills\unsupervised-entity-relations\SKILL.md shows only the additions and the 4 removed lines; /reindex then retrieve_skills finds it
  _Lessons:_ port = frontmatter + terms line + 4 chatbot lines dropped; index 178 -> 179, retrieve_skills ranks it 1. The repo copy and the port now differ by design.
- [DONE] T129 Draft the diagnostic set: 60 random sentences stratified by community, spans and types labelled, operator review gate before any sweep
  _Files:_ tests/fixtures/arxiv_entity_diagnostic.json
  _Verify:_ operator approves the labelled set; mentions hash-split into tuning and held-out halves
  _Lessons:_ 120 sentences, 120 communities, 84 draft mentions (41 tuning / 43 held-out), 64 sentences with no entity; labels written by the assistant from the sentences alone, never from a derived inventory. Types: METHOD 24, MODEL 20, TERM 18, DATASET 6, METRIC 7, TOOL 5, ORG 4; TERM is the generic-vocabulary type, so dropping it is one filter. Held-out interval is about +-0.15, not the planned +-0.09. GATE WAIVED BY THE OPERATOR'S DIRECTION 2026-10-04 ("you're supposed to make the call on T132, T133, and T135, not me"): the labels were NEVER reviewed by the operator, so every number measured against them (T132 sweep and held-out, T133 type check) measures agreement with the assistant's own draft. Correcting a label is cheap for the tuning half; the held-out half was read once and is spent, so a corrected set needs a freshly drawn held-out sample before it can be used the same way.
- [DONE] T130 Core derivation module: case-preserving tokenizer, apriori numpy n-gram counter with paper-level df, cohesion, boundary entropy, form signature, burstiness, C-value demotion, acronym pairs, cipher detector, scoring
  _Files:_ entity_derive.py, tests/test_entity_derive.py
  _Verify:_ pytest tests/test_entity_derive.py -q
  _Lessons:_ 64 tests pass. entities.npmi_ppmi is scalar-only, so entity_derive carries a vectorised npmi_vec pinned equal to it element by element. Real-corpus smoke run still to do.
- [DONE] T131 Persist entities and mentions in Postgres; add the entities step to the service; delete_papers cleans mentions
  _Files:_ sparsevec_store.py, src/arxiv_graph_service.py, entity_derive.py, tests/test_sparsevec_store.py, tests/test_arxiv_graph_service.py, tests/test_entity_derive.py
  _Verify:_ pytest tests/test_sparsevec_store.py tests/test_arxiv_graph_service.py tests/test_entity_derive.py -q
  _Lessons:_ lex_entity + lex_mention per build (store guard V10); service guard S9: entities are OPT-IN per build via `--freeze-entities`, a cycle matches new retrievable chunks against the frozen inventory and changes no statistic, a build without an inventory does nothing, a tokenizer-version mismatch is never matched, a full rebuild re-freezes only if the build it replaced had an inventory, and delete_papers removes mentions. 117 tests pass across store, service, derive, salience; full suite 1,010 passed, 6 skipped, 1 failed (tests/test_walker_render.py E18 'pairs=' lead; outside these files, not investigated). NOT run on the live database: no inventory has been frozen, so the live service is unchanged in behaviour.
  _Lessons:_ LIVE PATH RUN 2026-10-04 (`--freeze-entities` on build 1, 56,051 retrievable chunks, 2,521 papers), four freezes, each exposing a defect the synthetic data could not: (1) k=5000 gave tuning recall 7/41 on the full corpus against 16-19/41 on 421-841 papers, because the candidate pool tripled: a pre-registered k sweep (5k..80k, smallest k within 90% of the best recall) chose k=40,000 (recall 26/41 at 40k and 80k, matches per no-entity sentence 0.92 vs 2.69), now ENTITY_K; k does not carry over to another corpus size. (2) the corpus-wide cipher z flagged 62 papers, 59 of them ordinary English and 55 of them the short _methods extracts; comparing a paper only with papers of similar length gives 13 (guard AE3b, mutation-checked). (3) 79 inventory entries were single-character math residue ('b c b d', 'z z z z'): guard AE15, min_tok_len=2. (4) 'the current', 'the next', 'a small' ranked top by paper count because 'the' is capitalised 2.4% of the time and 'a' 8.9% (a fixed 2% cut cannot separate them from 'model' 2.8%); tokens in more than 97% of papers now close a gram (AE6b, universal_df). (5) docling glyph placeholders ('glyph' 59,646 mentions, 'glyph lt c' 58,318): AE2b, tokenizer t3 including the HTML-escaped form. Third freeze, tuning recall 26/41 at 0.97 matches per no-entity sentence against the pre-registered 0.92: NOT met, by 2 match rows of 35; the assistant KEPT AE6b anyway (deviation, logged): of the 35 matches inside the no-entity sentences about 27 are real terms the draft labels did not mark (Friston, UK, English, Census, graphene, amyloid, fibrils, electrical conductivity, resource allocation, cumulative reward, synthetic dataset, taxonomy) and about 8 are junk (Fig, see Appendix, our framework, D train, don't know, Q Lee, ctx), so the metric measures label conservatism more than error and a 2-row gap is inside that noise. Remaining junk seen live: 'an example', 'an LLM', 'Here we', 'our method' (determiner-like first tokens below 97% of papers). Later layers use tokenizer t3; an inventory frozen under an older tokenizer is never matched.
  _Lessons:_ FOURTH FREEZE (tokenizer t3, current live state of build 1): 40,000 entities, 1,058,970 mention rows over 55,767 of 56,051 retrievable chunks, 14 cipher papers left out. Against the diagnostic set: tuning recall 26/41 at 0.94 matches per no-entity sentence, held-out 26/43 at 0.79 (held-out is spent: information only, nothing is tuned on it). 'glyph lt c' (58,318 mentions) is gone; the most-mentioned entries are now PG, Fig, GPT-4, training data, CoT, GPS, math, RL, RAG, LoRA. 0 determiner-led ('the X', 'a X') entries and 0 single-character entries remain. STILL THERE: three glyph-like entries from a spelling the pattern does not catch (GLYPH 8 mentions, GLYPH cmap 3, GLYPH cmap d835 36; 47 in all), and junk such as 'an example', 'an LLM', 'Here we', 'our method', 'Fig' (7,072 mentions) among the top. The 14 flagged cipher papers were not checked one by one. The service step that matches newly ingested chunks is tested against synthetic builds only; no real ingest cycle has run against this inventory.
- [DONE] T132 Sweep rounds 1-3 against the diagnostic set on 600-paper samples, then the full corpus; reviewer package
  _Files:_ entity_derive.py
  _Verify:_ each axis proven live before sweeping; paired comparison; held-out half run once; results in lex_entity_sweep
  _Lessons:_ PROVISIONAL round 1 (.tmp/sweep_round1.py), tuning half of the DRAFT diagnostic set only (41 mentions, 36 no-entity sentences; the held-out half was never read), one-factor-at-a-time on features computed once from 421 papers. Every axis is live (each changes the selected set). Recall is 16/41 at baseline (Wilson95 [0.26, 0.54]) and every setting lands at 13-17 of 41, so NO axis separates on recall. What does move is matches in no-entity sentences (baseline 2.53 per sentence): rule=cohesion 0.42 at recall 17/41, drop=bh 0.81, uni_quota=0.3 1.50, df_floor=2 1.83, drop=ridf 4.00, k=2000 0.97 (recall 15) vs k=5000 2.53 (16). That number counts agreement with the assistant's own draft labels: an unlabelled but real domain term (e.g. neural network) counts as spurious, so it overstates error. 25 of 41 mentions are missed at every k up to 10,000: rare names (ProGraph, MILP, GraphFC, PyGAD, ArmoRM, Cre-lox) below the 3-paper floor in a 421-paper subset, plus plural/variant forms (LLMs, System 2, Llama 2). Not yet tested: whether more papers raise recall (next: 600+ papers), and anything on the held-out half, which waits on the operator's review of the labels.
  _Lessons:_ PROVISIONAL round 1 again on 841 papers (every third paper; the 421-paper set is a SUBSET of it, so agreement between the two is weaker evidence than two independent samples). Baseline recall 19/41 (Wilson95 [0.32, 0.61]) and 1.78 matches per no-entity sentence, against 16/41 and 2.53 at 421 papers: the recall gain is 3 mentions, inside the interval. Cipher detector flagged 18 papers at 841 vs 4 at 421, not checked for false positives. rule=cohesion did NOT hold: recall 17/41 at 421 papers, 14/41 at 841 (baseline 16 then 19); that first-round advantage was a property of the sample. Same direction in both: drop=bh matches 0.81 -> 0.28 at recall 18/41; df_floor=2 fewer matches (1.83 / 1.28) at unchanged recall; dropping ridf or cap raises matches and lowers recall; k=5000 is the knee (k=10000: recall 20/41, matches 3.58). Next: an independent paper sample (not a superset), then the held-out half once the operator has reviewed the labels.
  _Lessons:_ Inventory defect fixed ahead of the sweep (guard AE14): 'does not' (7,048 chunks) and 'do not' (6,258) passed because 'not' is capitalised 2.2% of the time, over the 2% cut. A residual-IDF floor of 0.5 (measured gap: function-word grams 0.19-0.33, group 1.09, true 0.91, attention 0.83) now scores them 0; multiword entries in >20% of papers fell 46 -> 33. Remaining residue in those 33: about 7 junk (x y, t t, x i, x x from math fragments outside $...$; a unified; see appendix; did not at ridf 0.80), about 26 real domain terms. Ordinary single words (free, true, group, name) still enter through the unigram quota; whether they count as entities is the operator's TERM ruling. ridf_floor is a sweep axis.
  _Lessons:_ DECIDED BY THE ASSISTANT (operator: the call on T132/T133/T135 is the assistant's), on the assistant's own DRAFT labels. Stage 1: two DISJOINT paper samples (421 + 420), tuning half only, 54 settings, pre-registered rule (admissible = recall within 1 of baseline in both samples; winner = fewest matches per no-entity sentence, must beat baseline by 0.20). Winner: drop boundary entropy, df_floor=2, uni_quota=None, k=5000; tuning matches 2.58 -> 0.33 at recall 16/41 and 16/41 (baseline 16, 16). Stage 2, the held-out half read ONCE on 841 papers: baseline recall 14/43 (Wilson95 [0.20, 0.47]), 1.36 matches per no-entity sentence (n=28); winner recall 11/43 ([0.15, 0.40]), 0.14 per sentence; paired winner-only 2, baseline-only 5, exact McNemar p=0.453. The precision gain held out; the recall loss is not significant but its direction is down and it sits in MODEL (5 -> 2 of 12). Adopted as ENTITY_PARAMS in src/arxiv_graph_service.py. Defects: labels are the assistant's own; held-out n=43 is too small to rule out a real recall cost; the cipher detector flagged 18 papers at 841 vs 4 at 421, unchecked.
  _Lessons:_ CAVEAT ON EVERY 'matches in no-entity sentences' NUMBER ABOVE (found 2026-10-04 on the live inventory): most matches inside those sentences are real terms the assistant's draft labels did not mark (about 27 of 35 in the tuning half), so that figure overstates false matches; the direction of the sweep results may stand, their magnitudes should not be read as error rates.
- [DONE] T133 Entity types: PPMI context vectors plus a spelling view, kNN, leiden with the 0.8 seed-ARI gate
  _Files:_ entity_derive.py, tests/test_entity_derive.py
  _Verify:_ pytest tests/test_entity_derive.py -q; operator reads top contexts and 10 members per class
  _Lessons:_ type_entities, context_vectors, spelling_vectors, match_spans in entity_derive.py (guard AE10; 72 entity tests pass). Live run on 841 papers with the T132 winner: 5,000 entities; seed-ARI gate passes ONLY at resolution 0.5 (4 classes, ARI 0.905); 9 classes 0.765, 31 classes 0.695, 144 classes 0.622 all fail. The four stable classes are FORM classes: capitalised names (CoT, ChatGPT, Claude, LoRA), lowercase concept phrases (reinforcement learning, neural network), all-caps acronyms (SFT, RL, F1, GSM8K, MMLU), and math-fragment residue (x y, t t, i j; 275 entries). Against the 13 labelled tuning mentions that are in the inventory and typed: ARI -0.136 (shuffled-label 95th percentile 0.214), NMI 0.164, i.e. semantic types (MODEL / METHOD / DATASET) are NOT recovered; GSM8K and MMLU sit with SFT and RL. Every class's top contexts are function words (the, and, of). Useful result: class 3 lists inventory entries that are math residue. NEXT WORK ITEM (hypothesis, untested): down-weight contexts that are ubiquitous across entities and widen the window so 'evaluated on _' style contexts carry the type signal.

## Layer 31 -- parallel (term salience in the sparse space; operator 2026-10-04; plan C:\Users\user\.claude\plans\jolly-soaring-moler.md)
- [DONE] T134 Term-by-community keyness (Dunning log-likelihood) and a term's spread entropy over the communities, from the stored assignments
  _Files:_ src/term_salience.py, tests/test_term_salience.py
  _Verify:_ pytest tests/test_term_salience.py -q; one pooled number per term with its chunk count beside it
  _Lessons:_ 4,719 entities with >=20 chunks: median share of a term's chunks in its best community 0.154, median spread entropy 0.603. PTQ/QAT/weight-only quantization sit in community 42 (share 0.77-0.83); generative recommendation in 55 (0.94). Defect found: the inventory holds 'does not' (7,048 chunks) and 'do not' (6,258), plus common words (free, true, group, name); the closed-class gate let them in at k=5000. Module src/term_salience.py written with tests/test_term_salience.py (16 tests pass, guards TS1-TS6); first measured as scratch in .tmp/term_probe*.py.
- [DONE] T135 Retokenise the sparse build with the reviewed entity inventory merged as single tokens (a new build; every later artifact is NOT TESTED until redone). BLOCKED on the operator's review of the diagnostic set.
  _Files:_ src/ingest_arxiv_sparsevec.py
  _Verify:_ recall at 50 on the 200 + 200 battery no worse than the current build; BM25 summand sum equals the stored inner product
  _Lessons:_ DECIDED BY THE ASSISTANT: tested, NOT applied; the live build and its vector tables were never touched. Run in memory (.tmp/t135_battery.py) on the ingest's own 200 title + 200 body-sentence battery, exact numpy inner product, parity of the saturated product against the idf-baked score OK on every arm. Entity tokens were ADDED beside the words, never replacing one (the repo's record: additive 3 wins, re-ranking 0), so the task's 'merged as single tokens' became 'added'. Inventory: the T132 winner on 841 papers, 5,000 entries, 2,659 multiword; 54,249 of 58,870 chunks hold one. Results, base -> arm, recall@50 with paired bootstrap 95% CI: multiword-additive title 0.472 -> 0.457 (-0.0150 [-0.0251, -0.0056], better 15 / worse 41), body 0.202 -> 0.205 (+0.0032 [-0.0015, +0.0090]); all-entities-additive title 0.472 -> 0.465 (-0.0073 [-0.0197, +0.0042]), body 0.202 -> 0.208 (+0.0058 [+0.0002, +0.0120], better 25 / worse 11). The task's own check, recall no worse than the current build, FAILS for multiword on titles. Mechanism for the title loss is a HYPOTHESIS, untested: a high-idf phrase token concentrates the score on the chunks holding the phrase and displaces the paper's other chunks, which are also relevant. Not tested: replacing instead of adding, a lower weight for the entity column, the query side without the document side.

## Layer 32 -- sequential (needs Layer 31)
- [DONE] T136 Sparse ablation graph: drop one term column, report the share of that term's chunk edges that disappear
  _Files:_ src/term_salience.py, tests/test_term_salience.py
  _Verify:_ pytest tests/test_term_salience.py -q
  _Lessons:_ 15 single-word terms among the 38: a term carries a median 6% of a lexical edge's cosine (per-term 2-10%) and >= half in 0-5% of its edges, so dropping one column rarely removes an edge. Sparse median edge share against dense |delta| across terms: Spearman 0.614 (p=0.015, n=15). Module src/term_salience.py written with tests/test_term_salience.py (16 tests pass, guards TS1-TS6); first measured as scratch in .tmp/term_probe*.py.
- [DONE] T137 Dense occlusion sample and the stability gate: delta_t(d) = e(d) - e(d without t); cosine across chunks for one term against a random-deletion null
  _Files:_ src/term_salience.py, tests/test_term_salience.py
  _Verify:_ pytest tests/test_term_salience.py -q; the gate is a pooled median per term against the null, n beside each
  _Lessons:_ 38 terms, 710 pairs, MiniLM 256-token window. delta_t same term across chunks: median cosine 0.322 (n=6,399 pairs) against -0.001 mismatched terms and 0.000 random-word deletions; all 38 terms above the mismatched median, so the gate passes. |delta| term 0.1449 vs random word 0.0905 (term larger in 73.4%). cos(delta_t, e(term alone)) median 0.445, so the additive reading is rejected. Spearman(idf, |delta|) -0.013 (p=0.94, n=38). Module src/term_salience.py written with tests/test_term_salience.py (16 tests pass, guards TS1-TS6); first measured as scratch in .tmp/term_probe*.py.

## Layer 33 -- sequential (needs Layer 32)
- [DONE] T138 Surrogate: each token's share of the mean-pooled vector, calibrated to occlusion; adopt only at rank correlation 0.8 on held-out pairs
  _Files:_ src/term_salience.py
  _Verify:_ rank correlation on held-out (chunk, term) pairs
  _Lessons:_ Spearman(|token share of the pooled vector|, |delta|) 0.816 on the held-out half (n=353), gate 0.80, so it passes narrowly; cosine of the share vector with delta is only 0.421, so it predicts how much a term moves a chunk, not in which direction. Module src/term_salience.py written with tests/test_term_salience.py (16 tests pass, guards TS1-TS6); first measured as scratch in .tmp/term_probe*.py.
- [DONE] T139 Term direction e*(t) = mean of delta_t across chunks, used for subtraction; only if the stability gate passes
  _Files:_ src/term_salience.py
  _Verify:_ subtracting e*(t) moves a chunk toward e(chunk without t) more than a random direction does
  _Lessons:_ e*(t) from half the chunks, applied to the other half (n=353): cos to e(chunk without t) 0.9896 -> 0.9915 (+0.0018 median, improved in 65.4% of pairs); a different term's e*(t') worsens it (-0.0058, improved in 2.3%). Right direction, tiny effect: deleting one term barely moves a chunk (cos 0.9896 already). Module src/term_salience.py written with tests/test_term_salience.py (16 tests pass, guards TS1-TS6); first measured as scratch in .tmp/term_probe*.py.

## Layer 34 -- sequential (operator 2026-10-04: "i don't want to use bm25, I want to use sparsevec inner product")
- [DONE] T140 Term salience as a command, in the SPARSEVEC INNER-PRODUCT terms: a term's share of the inner product between a chunk and its sparsevec neighbours (stored cosine view, neighbours from the HNSW index), spread over communities, optional embedding effect
  _Files:_ src/term_salience.py, tests/test_term_salience.py
  _Verify:_ pytest tests/test_term_salience.py -q; `python src/term_salience.py <term> --dense` prints the three views from the live database; shares of all terms of one inner product sum to 1
  _Notes:_ replaces the first cut, which ranked chunks by idf x saturated tf (the BM25 framing). A phrase has no sparsevec dimension in this build (T135 was tested and not adopted), so a multiword entity is shown through its words.

  _Lessons:_ `python src/term_salience.py <term> [--k] [--top] [--dense]` reads the live Postgres build and prints, for a term spelled as vocabulary columns (BPE '##' pieces supported), its share of the sparsevec inner product between its most salient chunks and their k HNSW-cosine neighbours, its spread over communities, and with --dense its effect on the MiniLM embedding. Shares add exactly over columns, so a phrase's share is the sum of its pieces' (pinned by a test). 21 tests pass. Live, WORD vocabulary: grpo in 769 chunks of 204 papers, 37 of 45 neighbours hold it, median 15.7% of the inner product, 21% of its chunks in one community (G2 612), embedding delta 0.235 (random word ~0.09), delta-to-delta cosine 0.735. 'kv cache' cannot be spelled from the word vocabulary ('kv' is below its tokenizer's floor). A bug the tests caught: an evenly spread term listed all communities with G2 0. Skill updated: bpe-bm25 (master Documents\dev\skills, mirror ~\.skills, router reindexed, retrieve_skills ranks it first) now carries 'Term salience' as the default option; the operator's report that BPE-as-input 'works much better' is recorded as a report, NOT as a measurement, and the retrieval measurements (reduced BPE vocabulary lost recall; raw words beat BPE by 8-17% rec@50 on this repo's arXiv build) are kept beside it as a different job. NOT done: the same terms spelled both ways on the same chunks, which is the test that would settle word versus BPE for salience; no BPE-vocabulary build of the arXiv corpus exists to run it on. The master skills repo has a large number of tracked files deleted in its working tree (not by this work); nothing there was committed and sync_skills.ps1 was NOT run, one file was copied. The scheduled watcher was restarted 2026-10-04 18:10 (it had been running pre-entity code since 10/3 15:16); the old pair was stopped by PID.

## Layer 35 -- sequential (operator 2026-10-04: "build a BPE-vocabulary version of the arXiv build so the word-versus-BPE comparison can be measured")
- [DONE] T141 BPE-vocabulary build of the arXiv corpus on a SIDE label (arxiv_sect_bpe; the live label is untouched), same chunks, same ords, hosted in Postgres; then the word-versus-BPE comparison for term salience and for retrieval recall
  _Files:_ src/ingest_arxiv_sparsevec.py, tests/test_ingest_arxiv_sparsevec.py
  _Verify:_ pytest tests/test_ingest_arxiv_sparsevec.py -q; the side build's battery prints recall@50 and psql-vs-numpy parity; the salience comparison reports, per term and paired, the share of a term's sparsevec neighbours that really contain it under each vocabulary, plus the share of terms each vocabulary can spell
  _Notes:_ BPE trained on the surviving term strings only with a merge budget (bpe-bm25 procedure, steps 4-6 omitted: the band is a size dial that did not move recall). Ground truth for "the neighbour really holds the term" is the neighbour's TEXT, independent of either tokenizer.
  _Lessons:_ Word vocabulary beat BPE on term salience, paired over 213 terms spellable both ways: recall 0.993 vs 0.471, precision 0.410 vs 0.102, F1 0.515 vs 0.146, neighbour-truth 0.244 vs 0.106, all intervals exclude 0. BPE spells more terms (85% vs 71%). Rebuilt BPE label recall@50 (exact): title 0.424 vs 0.472, body 0.176 vs 0.202 (word run predates junk-flagging, populations differ slightly). A bug the first battery exposed (recall@50 = 0.001): the query was spelled from characters, pinned by a new test, 12 tests pass. NOT resolved: 71 of 213 terms had zero BPE hits (cause unconfirmed: whale-piece removal or nnz trim suspected); re-scoring without them is the next item. The operator's "BPE works much better" is not reproduced here; bpe-bm25 skill master updated (mirror NOT synced).

## Layer 36 -- sequential (community map shows every community; sections versus chunks; operator 2026-10-05)
Operator: "we should show all the communities ... a markdown document to pair with the chart. The chart should just have the star and adjacent top n chunks we selected for" and "it's sections that matter ... two passes: chunks cluster, then aggregate to sections". Measured first (.tmp/section_vs_chunk.py, 56,051 chunks): 55,392 sections, 99.3% of them ONE chunk, only 1.8% of chunks sit in a multi-chunk section; 98.5% of the 470 exemplar chunks are whole sections already. Of the 369 multi-chunk sections, 55.3% have every chunk in one community, 44.7% span two or more. CORRECTED 2026-10-05: those counts group chunks by (paper, first-unit section index), and the chunker MERGES short consecutive sections into one chunk up to the median, so they are not true sections. True header sections (raw markdown, .tmp/section_lengths.csv): 98,264 now (89,211 at the live fit); 47.0% of chunks hold 1 header block, 36.1% two, 16.9% three to five (96,701 header lines in 56,051 chunks). The split bound is median + 2*1.4826*MAD of log size: 151 newline-units (m = 10, lcap = 1992), about 16,600 characters (the operator's Excel without the 1.4826 factor gives 7,264 characters, 62.5 newlines); at the operator's 23/107.25 tokens per character that is ~3,560 tokens, the median section 1,311 characters ~ 281 tokens. MiniLM: max_seq_length 256 set in arxiv_community_map.py, 512 position embeddings, 384 is the vector size.
- [DONE] T142 Draw every community on the map (253): a star at the medoid and a marker for each other selected exemplar, labelled with the community number; no cards; the paired markdown (already every community, largest first) is the reading side
  _Files:_ src/arxiv_community_map.py, tests/test_arxiv_community_map.py
  _Verify:_ pytest tests/test_arxiv_community_map.py -q; .tmp/arxiv_community_map.png shows 253 stars and 217 exemplar diamonds (470 exemplars = 253 medoids + 217 others); every label in the PNG has a heading in .tmp/arxiv_communities.md
  _Lessons:_ render(st, ex) now draws only map_marks(XY, ex): a numbered star per community and a diamond per other exemplar joined to its star; the card layout (card_plan, section_lines, mpl, the 12-largest grid) and its 8 tests are removed, 2 new tests, 15 pass. Rendered from the saved state (.tmp/arxiv_community_map.20261005-160317.png: the stable name is held open by a viewer, so save_figure wrote a sibling). NOT done: the full main() run, which would re-persist and may need the GPU the dense test holds. DEFECT seen in the picture: a diamond at z=k/n of the band is far from its star on the 2D layout for many communities (the layout is UMAP, the band is embedding distance), so some joining lines cross the whole map; the operator asked for "adjacent" chunks and these are not adjacent in the picture.
- [OPEN] T143 Decide the section unit before any second pass: "section" today is an ATX-header block (22 per paper), not a numbered top-level section. Either keep it (then the second pass touches 1.8% of chunks) or group by top-level heading (a different, coarser unit that does not exist yet). NOT TESTED: section-level embedding (mean of chunk embeddings, since the 256-token window sees only a chunk's opening) and section-level BPE (BPE lost to words at chunk level on 2026-10-04)
  _Files:_ (none until the unit is chosen)
  _Verify:_ operator picks the unit; then a plurality-vote section assignment is compared with the chunk assignment on the 369 multi-chunk sections
- [DONE] T144 Does a longer window find what MiniLM-256 cannot? Dense arms on the existing 20,000-chunk, 200-sentence battery (rec@50, exact cosine): MiniLM at 256 (baseline 0.590), 512, windows mean-pooled, jina-embeddings-v5-text-nano at 8,192 (239M parameters, 768 dimensions, cc-by-nc-4.0, MTEB English v2 71.0 per its model card)
  _Files:_ src/diag_dense_arms.py
  _Verify:_ python -u src/diag_dense_arms.py --arms minilm,minilm512,minilmwin,jina (log .tmp/dense_arms2.log); each arm prints rec@10, rec@50 with a bootstrap interval and the paired difference. Caveat built into this battery: 101 of 193 queries end beyond MiniLM's 256 tokens, so a longer window is favoured by construction; it measures the truncation defect, not general quality.
  _Results so far (20,000 chunks, 200 sentence queries, exact cosine):_ MiniLM-256 rec@50 0.590 [0.525, 0.660] (reproduced twice); MiniLM-512 0.685 [0.620, 0.745]; jina-v5-nano THROUGH model2vec at 256 dimensions 0.240 [0.180, 0.300], paired -0.350 [-0.430, -0.265] against MiniLM-256 (the operator's proposal, run 2026-10-05 after first running jina as a full transformer by mistake). NOT TESTED: jina as a transformer (its retrieval adapter and prompts were never in the distilled model), jina-model2vec at 768 dimensions without PCA, MiniLM windows pooled. Pad fix for model2vec batch encode (jina's tokenizer pads with id 128004, past the 128,001 rows model2vec keeps): zero rows and zero weights, batched equals one-at-a-time at cosine 1.0.
  _Lessons:_ MiniLM with 250-token windows mean-pooled (at most 32 per chunk, so the whole chunk is read) scores rec@50 0.720 [0.655, 0.780], paired +0.130 [+0.070, +0.195] against MiniLM-256, the setting the map uses today; the interval excludes 0. It is not separated from MiniLM-512 (0.685; the intervals overlap, no paired test run). Cost: 942 s for the windowed arm on 20,000 chunks (tokenising and ~4x the encodes) against 175 s. Jina through model2vec lost (0.240). NOT TESTED: jina as a transformer, jina-model2vec at 768 dimensions, windowed pooling on whole sections, whether windows pooled changes the 253 communities. Built-in bias to say aloud: 101 of 193 queries end past token 256, so any longer window is favoured by this battery; it measures the truncation defect, not general quality.
- [DONE] T145 Adopt window-pooled MiniLM embeddings in the community map (arxiv_community_map.embed: 250-token windows, at most 32, mean-pooled, L2). CONSEQUENCE: new embeddings re-derive the kNN graph, the 253 communities, the exemplars and the persisted dense vectors, so the LLM summaries tied to the old exemplars go stale and every downstream artifact is NOT TESTED until redone. Needs the operator's go (it supersedes the live map) and the section-unit decision (T143)
  _Files:_ src/arxiv_community_map.py, tests/test_arxiv_community_map.py
  _Verify:_ pytest tests/test_arxiv_community_map.py -q; the 200+200 battery on the dense view; community count and seed-ARI printed beside the old 253 / 0.876
  _Lessons:_ Rebuilt 2026-10-05 18:29-18:52 (embedding 839 s for 56,051 chunks in 233,485 windows (4.2 per chunk; the ledger first said ~160,000, which was a guess), then kNN built twice identical, sweep, consensus, UMAP, persist, markdown, PNG): 245 communities, consensus seed-ARI 0.896 (gate 0.8 PASS; old 253 / 0.876), same chosen resolution 6.162. The change is real, not seed noise: ARI(old partition, new partition) = 0.330; 185 of 253 old communities (73%) keep at least half their chunks together in one new community, 91 keep 80% or more, 13 keep under 30%; chunk-weighted 53.6%. 253 LLM summaries are stale (written for other exemplars) and are NOT shown; regenerating them costs API calls and was not done. Old files kept in .tmp/map_before_windows_2026-10-05/ (old caches also keep their old names). A trap closed first: the embedding, state and kNN caches were keyed by row count only, so new embeddings would have loaded the old communities; they now have _win names and a test pins it. NOT TESTED: the dense recall battery on the new map's own vectors, whether the new communities are better than the old ones (only that they differ and are as seed-stable), the 200+200 retrieval API on the new build. My monitor missed the whole run because `tr` buffers when piped: do not pipe a watch through tr.
  _Notes:_ OPERATOR INTENT (2026-10-05, carried forward): topics should be derived over SECTIONS, chunks are for retrieval precision; embed whole sections (MiniLM sees 256 tokens, jina 8,192); BPE tokenizer on sections for sparsevec inner product (measured 2026-10-04: raw words beat BPE on this corpus, so BPE is not adopted); chart shows every community, star plus selected exemplars, a markdown document beside it (done, T142). Section unit and the cap for giant sections are still the operator's call (T143).

## Layer 37 -- sequential (a few salient words per community, drawn on the UMAP; operator 2026-10-05)
Operator: "each summary reduced to a single phrase and drawn centered within the top n 'sections' plotted on the umap ... salient against the other terms ... bm25 over all the summary terms and then mmr ... I'm told that's asymmetric ... cosine over bpe tokenizer ... mmr 'words' per text to inform the llm ... maybe a few-shot mmr set". Measured first (.tmp/label_words_probe.py, current 245-community map): Dunning keyness over ALL 56,051 chunks (term_salience.keyness_by_community) surfaces the concept words (community 0: RAG, retriever, retrieved documents; 3: RL, reward function, GRPO; 4: code generation, HumanEval; 5: caption, LLaVA, CLIP; 8: long-term memory, episodic memory) but also junk and benchmark names (1: ARC-C, PIQA, HellaSwag, "1.00 0.00"; 9: "Your task", "your answer"; 11: "You must", JSON) and 47 of 245 communities own no entity at df>=20 (10 is one). All 245 titles are ~3,065 tokens and all titles+summaries ~29,293, so one LLM call CAN see every topic. The summaries rest on 488 of 56,051 chunks (0.87%). Three LLM titles written in isolation overlap (1, 2, 6: "language model evaluation and training details" / "LLM performance evaluation ..." / "training and evaluating LLMs for reasoning").
- [OPEN] T146 Candidate words per community: Dunning keyness over all member chunks (incumbent keyness_by_community), junk filtered, then MMR with SYMMETRIC cosine between term vectors (relevance = normalised G2, redundancy = cosine of the terms' chunk-presence rows), k = 8, whole words and the entity inventory (not BPE pieces)
  _Files:_ src/community_labels.py, tests/test_community_labels.py
  _Verify:_ pytest tests/test_community_labels.py -q; the 8 words for communities 0, 3, 4, 5, 8 match the probe; redundancy is symmetric (cos(a,b) == cos(b,a)); junk terms (numbers, prompt boilerplate) absent from a sample of 12
- [OPEN] T147 Label each community in ONE phrase of at most 3 words, sequentially largest first, each call seeing the community's MMR words, its summary and EVERY label already assigned (<= ~3,100 tokens), then a deterministic uniqueness check (no equal label, no shared head word among the 8 nearest communities by centroid cosine) with one retry on a collision
  _Files:_ src/community_labels.py, tests/test_community_labels.py
  _Verify:_ 245 labels, 0 duplicates, 0 head-word collisions among nearest neighbours; share of label words inside the community's top-20 G2 words reported; cost printed (expected under $0.10)
- [OPEN] T148 Draw the labels on the UMAP at the median of each community's points, font by sqrt(size), greedy collision avoidance largest first, only labels that fit are drawn, the rest listed; the markdown carries all 245
  _Files:_ src/arxiv_community_map.py, tests/test_arxiv_community_map.py
  _Verify:_ pytest tests/test_arxiv_community_map.py -q; the PNG shows N non-overlapping labels (N printed); labels sit on their community's own points (share of its points within a radius of the label reported)
  _Notes:_ "centered within the top n sections" read as: the median of the community's points, labelling the n largest communities that fit. The centroid of the exemplar marks was rejected: exemplars are picked by embedding distance and sit far apart on the UMAP (the long lines of 2026-10-05).
  _Superseded 2026-10-05:_ T146-T148 (entity-based keyness over communities) are replaced by T157 below, per the operator ("Dunning across sections ... no need for entities").

## Layer 38 -- sequential (the section map, exactly as asked; plan C:\Users\user\.claude\plans\jolly-soaring-moler.md, approved 2026-10-05)
Operator: nodes are SECTIONS; dense = a model2vec version of jina applied to sections; sparse = BPE tokenizer derived from the texts, reranked by adjacent tokens; NO HNSW: exact normalised inner products, a correlation matrix per representation, significant pairs as edges of one graph, Louvain-family communities over it, their own LLM summaries; labels = Dunning-salient words per exemplar section (3, 5 or 8 words). Side build, label `arxiv_sections`; the live `arxiv_sect` map is not touched.
- [DONE] T149 Section corpus: header blocks of the raw markdown, flagged reference / junk, filtered to non-empty bodies of at least 106 characters (exp(median - 2*1.4826*MAD) of log length)
  _Files:_ domain_corpora.py, src/section_corpus.py, tests/test_domain_corpora.py
  _Verify:_ pytest tests/test_domain_corpora.py -q; the count of usable sections printed (expected ~82,600 of 98,264)
  _Lessons:_ 98,264 sections in 2,521 papers: empty 9,050, reference 2,204, short (< 106 chars) 4,420, junk 48, KEPT 82,542 (148 s), cached .tmp/arxiv_sections.pkl; matches the CSV counts. 42 domain_corpora tests pass.
- [DONE] T150 Fast static pooling of a model2vec model over sections (batched tokenise, np.add.reduceat, zero rows and weights for pad ids past the kept rows, max_length None)
  _Files:_ src/section_embed.py, tests/test_section_embed.py
  _Verify:_ pytest tests/test_section_embed.py -q; pooled == StaticModel.encode at cosine 1.0 on 200 sections incl. empty, one-token and over-512-token
  _Lessons:_ 8 tests pass (batched pooling equals StaticModel.encode at cosine 1.0, with and without weights, across gather-block seams, empty text, ids with no row, the saved jina model on real text). ROOT CAUSE of the earlier pad-id IndexError: the jina tokenizer has PADDING ENABLED, so every text in a batch is padded to the longest with id 128004; 2,000 sections of 5.0M characters came out as 28.9M tokens and one 16.7M-character section made a 24 GB gather. pool() now calls tokenizer.no_padding() and no_truncation(): 2,000 sections in 8.2 s (was 81.7 s). Centring by the sample mean vector is NOT np.corrcoef (which removes each row's own mean); the plan's "exactly Pearson" was wrong and the docstring and test say so.
- [DONE] T151 Dense battery on sections and tuning of jina-through-model2vec (PCA 256 / 512 / 768 none, SIF on / off); MiniLM windows pooled as the reference
  _Files:_ src/diag_dense_arms.py
  _Verify:_ python -u src/diag_dense_arms.py --unit section ...; one pooled rec@10 / rec@50 with a bootstrap interval per arm, paired against jina-model2vec-256
  _Lessons:_ 200 queries on 20,000 sections. rec@50: pca256 0.435, pca512 0.445 (+0.010 [+0.000, +0.025], a tie), 768 with SIF 0.330 (-0.105), 768 without SIF 0.270 (-0.165). 256 kept. NOT MEASURED: the MiniLM-windows reference arm on sections.
- [DONE] T152 Sparse matrix on sections: BPE pieces (word arm beside it), saturated tf x idf, L2, adjacent-pair columns scaled by sqrt(lambda); lambda grid {0, 0.5, 1, 2} on the exact numpy battery
  _Files:_ src/section_sparse.py, tests/test_section_sparse.py
  _Verify:_ pytest tests/test_section_sparse.py -q; paired recall@50 per lambda; the salient-term precision probe rerun
  _Lessons:_ built and run at scale: 80,642 sections x 1,223,767 columns, 380 non-zeros per row, 121 s. DEFECT: lambda = 1 was USED, but the battery saturated and never showed lambda 1 beating lambda 0, so whether the adjacency rerank helps is NOT TESTED. Layer 41 tests it on the map scorecard.
- [DONE] T153 Streaming correlation edges for both representations (blockwise, in-loop floor, sample-estimated null with Box-Cox robust z, budget cap, exact top-k backbone, both tails counted), union with provenance
  _Files:_ src/section_graph.py, tests/test_section_graph.py
  _Verify:_ pytest tests/test_section_graph.py -q; edge counts per tail and representation logged; same-paper edge share beside its random base rate
  _Lessons:_ a global budget cap gave 91% same-paper near-duplicates, so edges are per-section significant members of the exact top-15. Same-paper share was 10.4% dense and 45.2% sparse (base rate 0.085%), and half the communities were one paper's outline, so same-paper pairs are now excluded inside the pass (group argument): 613,092 dense and 969,023 sparse cross-paper edges, 2 of 159 communities of size >= 10 at least half one paper. The negative tail is counted (dense 10.3M pairs) and NOT built: cross-domain polarity is the repo's closed result, not antonyms. 8 + 1 tests.
- [WIP] T154 Communities over the fused edge matrix (Leiden sweep, plateau pick, consensus, seed-ARI gate), UMAP from the exact top-15, exemplars, persist to arxiv_sections without HNSW, markdown, PNG
  _Files:_ src/arxiv_community_map.py, src/section_map.py, tests/test_arxiv_community_map.py, tests/test_section_map.py, tests/test_section_map_split.py
  _Verify:_ community count and consensus seed-ARI logged beside 245 / 0.896; ARI against the chunk map; arxiv_sect untouched
  _Lessons:_ cross-paper map: 548 communities, consensus seed-ARI 0.846 (gate 0.8 PASS), 357 singletons, five communities of 1,525-2,324 sections that are GENRES (prompt templates, generic intro/conclusion prose), not topics: median top-heading share of a size>=10 community is 0.05, so it is genre not heading text. Hubness is real (in-degree skew 4.1 dense, 10.7 sparse) but the top 1% of hubs spread over many communities, so it does not build the grab-bags (hypothesis: genre direction, not separated). A markdown-path default bound at definition overwrote the chunk map's file twice; fixed, regression-tested, file regenerated. NOT DONE: PERSIST to Postgres (deliberately withheld), ARI against the chunk map, PNG labels. The oversize split (factor 2 x serving-size mean) is an explicit choice, not a derived bound.
- [OPEN] T155 LLM summaries for the section communities, markdown with the full path reported (carried by T164 in Layer 41; one task, not two)
  _Files:_ src/summarize_clusters.py
  _Verify:_ N drafts, 0 failed, cost printed
- [OPEN] T157 (carried by T163 in Layer 41; one task, not two) Few-word labels: Dunning G2 per exemplar section against all other sections, MMR, 3 / 5 / 8 words by exemplar count, triangulated anchor with a nearer-own-community gate, collision-free drawing
  _Files:_ src/section_labels.py, tests/test_section_labels.py, src/arxiv_community_map.py
  _Verify:_ pytest tests/test_section_labels.py -q; 0 duplicate phrases; drawn count printed

## Layer 39 -- sequential (measure the fix before making it; plan C:\Users\user\.claude\plans\jolly-soaring-moler.md, approved 2026-10-06)
Operator: "plan a proper fix" (the section map's grab-bag communities, 357 singletons, untested adjacency rerank, unreadable PNG labels). Root-first: the grab-bags are GENRE (median top-heading share 0.05), hubness is real but spread, the same-paper binding is already fixed.
- [DONE] T158 Scorecard: label-free numbers per map (communities, size>=10, singletons, oversize against sqrt(m) = 1,260, share in oversize, top-paper and top-heading share, seed-ARI) and a read rubric frozen before any community is read
  _Files:_ src/section_scorecard.py, tests/test_section_scorecard.py
  _Verify:_ pytest tests/test_section_scorecard.py -q; python src/section_scorecard.py .tmp/sections_map_state.npz .tmp/sections_xp_map_state.npz .tmp/sections_xps_map_state.npz
  _Lessons:_ 7 tests pass. Validated against two known results before ranking anything: paper-mixed map 131 of 258 communities at least half one paper, cross-paper map 2 of 159, 357 singletons. The first test expectation was wrong (2/12 not 1/10), the module was right. Seed-ARI printed for the split map is the PRE-split consensus (0.846), not a measurement of the split. Rubric frozen at .tmp/section_rubric.md.

## Layer 40 -- parallel (two independent arms; disjoint files)
- [DONE] T160 Arm B, genre-residualised dense view: remove the top-k directions of the pooled within-paper deviation, k in {0, 2, 8, 32}; prove the axis live first (k=0 vs k=8 must change the edges)
  _Files:_ src/section_genre.py, tests/test_section_genre.py
  _Verify:_ pytest tests/test_section_genre.py -q; edge diff between k values; the scorecard row per k
  _Lessons:_ 6 tests pass (planted genre axes recovered, genre removed and subject kept, k=0 is the identity, axis live, one-section papers, zero rows). Real dense vectors: the axis is LIVE (neighbour lists keep 4.9 / 2.4 / 1.3 of 15 at k = 2 / 8 / 32 against k = 0) and top-8 axes carry 53% of within-paper variance. Cohesion of the four grab-bag communities falls 61-79% at k = 8 against a median of ~27% for eight mid-size communities (proofs, a genre-like topic, 58%): supports the genre hypothesis, does not prove it. LIMIT: a paper that spans subjects puts subject variation into the within-paper axes too; the sparse view has no such axis. The scorecard row per k is in T161.
- [DONE] T159 Arm A, hierarchy with a derived bound: replace the factor-2 choice in split_oversize with sqrt(m), recurse until nothing exceeds it or a split fails the 0.8 seed-ARI gate, keep two levels (genre, topic)
  _Files:_ src/section_map.py, tests/test_section_map_split.py
  _Verify:_ pytest tests/test_section_map_split.py -q; the scorecard row beside the baseline
  _Lessons:_ 5 split tests + 49 section tests pass. Full run: 548 communities at the genre level, 823 after splitting, 408 singletons, largest 1,630, 6 oversize, 11% of sections in oversize communities, 20 communities at least half one paper. EVERY scorecard number is identical to the earlier factor-2 patch: the sqrt(m) = 1,260 bound selects the same communities as 2 x the serving mean (1,194) here, and no depth-1 split cleared the gate, so the recursion added nothing. Five oversize communities still do not split (generic intro/conclusion, proofs, and others return one part at seed-ARI 1.000). Arm A is NOT a fix for them. The seed-ARI column is the pre-split consensus.

## Layer 41 -- sequential (choose by evidence, then label and summarise)
- [WIP] T161 Run the matrix through the scorecard under identical conditions (baseline cross-paper, arm A, arm B per k, sparse lambda 0 vs 1), read 10 communities per finalist under the frozen rubric, choose by scorecard AND read, record the context with the winner
  _Files:_ playbook.md, src/section_screen.py, tests/test_section_screen.py
  _Verify:_ one scorecard table, one rubric tally per finalist with quoted lines
  _Lessons:_ SCREEN, 20,031 sections of 623 whole papers (seed 0), every arm at the baseline's resolution 3.42 (each arm's own plateau pick had made the first run incomparable: 22.6% vs 84.8% in oversize was a resolution effect). communities / size>=10 / singletons / largest / oversize / share of sections in oversize / paper>=.5 / seed-ARI / same-heading edges / fused edges: baseline 138/57/62/1319/6/24.7%/0/.858/2.46%/329,122; C_heading 131/60/55/850/5/17.8%/1/.795/2.03%/330,683; B_k2 128/51/73/962/9/36.2%/0/.883/2.09%/406,660; B_k8 160/61/87/823/5/18.5%/1/.870/1.87%/424,088; lam0 139/57/70/917/7/26.4%/1/.835/2.24%/344,961. NO arm clearly beats the baseline; every arm leaves 18-36% of sections in oversize communities. Adjacency rerank (lam0 vs baseline): not shown to matter. Arm C's seed-ARI .795 is under the 0.8 gate. B_k8 adds 29% more fused edges (a different graph, not a cleaned one) and 25 more singletons. NOISE FLOOR (baseline on 3 samples of whole papers, seeds 0/1/2): seed-ARI .858/.854/.856 and size>=10 57/55/55 are steady; communities 138/139/116, singletons 62/65/45, oversize 6/12/11 and share in oversize 24.7%/48.2%/40.7% swing with the sample, so only PAIRED differences (arm minus baseline on the same sample) are read. PAIRED, 3 samples: C_heading: same-heading edges -0.43/-0.48/-0.44 pts (-16..17%, the ONLY effect that holds); share in oversize -6.9/-7.4/+4.1 pts, size>=10 +3/+3/-3, singletons -7/+5/+27, seed-ARI -.063/+.014/-.008 (NOT replicated: arm C does not reliably change the community structure). B_k8: oversize -1/-8/-7, share in oversize -6.2/-33.3/-26.3 pts, size>=10 +4/+9/+5, seed-ARI +.012/+.042/+.038, same-heading -24%/-24%/-24%, ALL on the better side on all three; COST: singletons +25/+3/+25 and communities +22/+2/+20 (about 20-25 extra small communities on two of three samples). Two earlier readings were withdrawn on later samples (arm C's grab-bag lead, B_k8's singleton rise as noise). Arm B at k=8 is the first arm to beat the baseline on every sample. FULL SCALE (80,642 sections, resolution 6.16 both): cross-paper pre-split 548 communities / 159 size>=10 / 357 singletons / largest 2324 / 12 oversize / 23.0% in oversize / seed-ARI .846; arm B k=8 pre-split 500 / 146 / 332 / 1883 / 12 / 22.0% / .910; so arm B gives a steadier partition (seed-ARI +.064) and a 19% smaller largest community, NOT a smaller grab-bag share (-1 pt); dense same-heading edges 14.8x -> 9.7x chance (-35%). SCREEN AT 6.16 (3 samples): oversize is 0-1 in the baseline, so the screen cannot see the grab-bag effect at the full maps' resolution (the measure depends on graph size and resolution together); B_k8 minus baseline seed-ARI +.024/+.047/+.121, same-heading -24/-23/-24%, communities and singletons mixed. RUBRIC READ (frozen rubric, 10 per map at the same ranks, graded by hand): arm A 4 topic / 5 genre / 1 mixed; arm B 2 / 3 / 5 (arm B's samples are larger: not size-matched); NEITHER map is mostly topics, the read does not show arm B better. GENRE FRACTION probe: genre-graded communities all score above topic-graded ones (P = 1.00 over 48 pairs, margin 0.004), threshold 0.533 pre-registered in T168. NOT DONE: held-out read for the genre flag, arms D and E, combinations (C + B_k8). FLAW: concurrent runs share .tmp/screen_results.json and can overwrite each other's rows; the logs are the record.
- [DONE] T162 Singletons: assign to the community holding more than half of the section's significant-edge weight, else a reported unassigned bucket
  _Lessons (2026-10-08, operator: "I thought we were going to drop communities that didn't have more than 1 ... I see a lot of lone stars"):_ THE DROP HAD NEVER BEEN BUILT: 408 of the 823 communities held one section and every one drew a star. MEASURED before choosing: `singletons_with_edges 408 | top_community_over_half 0 | no_majority 408`, so the assign-to-a-neighbour rule would have assigned none of them; they are bridge or outlier sections. DONE as a drop: `MIN_COMMUNITY = 2` (arxiv_community_map A15) removes them from the map's stars and count (header now `415 communities of 2+ sections (+ 408 single sections left unassigned)`), from the markdown (415 listed, a note says how many were left out), and, in section_graphrag, from the candidate communities, the summary-vector ranking and the agent's catalogue. The 408 sections stay nodes of the map and of retrieval. The stars still far from the cloud belong to real communities (the 12 farthest hold 161, 122, 46, 40, 314, 41, 38, 16, 39, 45, 516, 31 sections). 43 communities of exactly two sections remain.
  _Files:_ src/section_map.py, tests/test_section_map.py
  _Verify:_ pytest tests/test_section_map.py -q; singleton count before and after
- [WIP] T163 Few-word labels at the triangulated anchors for the largest communities (T157's design). BUILT 2026-10-07: term_salience.community_terms (Dunning G2 per community vs all other sections, over-represented only, df>=3, no term inside a better one) and exemplar_term_count (3/5/8); terms ARE the label in the markdown heading and the card header until a summary title exists, then sit beside it. Applied to the saved cross-paper map (unigrams+bigrams of body text, 70,249 terms): the 12 cards read retrieval/RAG, multimodal, scientific discovery, graph/GNN, diffusion/video, SAEs/circuits, attention, RLHF/DPO as topics; c9 (reproducibility-checklist boilerplate), c13 and c23 (figure/image captions) read as genre. NOT DONE: MMR, triangulated anchor placement on the plot. 61 tests pass.
  _Files:_ src/section_labels.py, tests/test_section_labels.py, src/arxiv_community_map.py
  _Verify:_ pytest tests/test_section_labels.py -q; 0 duplicate phrases
- [WIP] T164 LLM summaries, markdown path reported. BUILT 2026-10-07 (operator: "pass the dunning terms with the sections to an llm to summarize and bold any dunning terms that survive into the llm summary phrase"): summarize_clusters.py --tag <map> puts the community's Dunning terms in the prompt, writes .tmp/sections_<tag>_summaries.json, NEVER persists (T165 waits); term_salience.bold_terms bolds whole-word matches in title and summary; the markdown shows the bolded text and "surviving: n of m". PILOT on 15 communities of the cross-paper map (12 cards + 3 agent): 15 draft, 0 failed, $0.0053 (so 823 is about $0.30). Terms surviving per community: topical blocks 2-8 of 8 (RLHF 8/8, multimodal 6/8, scientific research 6/6...), image-caption block c13 0/8, mixed Wikipedia/math prompts c27 0/8; but figure-caption block c23 7/8 because the model quoted the terms back. HYPOTHESIS, NOT A FINDING (n=15, one counter-case): low survival marks a non-topic block. NOT DONE: the remaining 808 communities, the card does not bold (matplotlib), the survival hypothesis test. 61 tests pass.
  _Files:_ src/summarize_clusters.py
  _Verify:_ N drafts, 0 failed, cost printed
- [DONE] T165 Persist to Postgres only on the operator's go (go given 2026-10-07; written to NEW sect_* tables by src/section_store.py instead of arxiv_community_map.py: build 32, 80,642 sections, 1,587,270 edges, 823 communities; arxiv_sect untouched; see T172)
  _Files:_ src/section_store.py, tests/test_section_store.py
  _Verify:_ pytest tests/test_section_store.py -q (19 with the graphrag tests pass); sect_build has one live build for tag xpa

- [OPEN] T171 Card heights per row: size each row of cards to its own longest card instead of one shared height, so the figure is not mostly empty (2400x5712 for the arm A map)
  _Files:_ src/arxiv_community_map.py, tests/test_arxiv_community_map.py
  _Verify:_ pytest tests/test_arxiv_community_map.py -q; the PNG opened; height before and after

- [WIP] T172 Hybrid GraphRAG over the section map (operator 2026-10-07, Agentic GraphRAG ch.1; plan C:\Users\user\.claude\plans\jolly-soaring-moler.md): a SUBGRAPH answer (seed sections by dense+lexical RRF, one hop along the saved top-15 neighbours) and a GLOBAL answer (map each candidate community's summary to a partial answer, reduce), then one synthesis that cites section keys. Offline over the saved xpa state; reads no Postgres, writes none; arxiv_sect untouched
  _Files:_ src/section_graphrag.py, tests/test_section_graphrag.py
  _Verify:_ pytest tests/test_section_graphrag.py -q; .tmp/graphrag_try.py on 3 questions; then the 20-question blind comparison (RRF alone / + subgraph / + global)
  _Lessons:_ 10 tests pass. SUMMARIES: all 823 drafted, 0 failed, $0.13. FIRST RUN, 3 questions: the whole path runs (31 s, 15 s, 12 s). OBSERVED DEFECT: with genre communities excluded, "main approaches to long-term memory for LLM agents" got 12 candidates and 0 relevant partials, because community 21 "Long-term memory for AI agents" (1,024 sections, genre score 0.537 against the 0.533 bar) is flagged genre; with the flag off it is candidate 1 and returns a relevant partial. So the genre flag is a false positive on at least c21 (and c2, c10, c14 are agent communities flagged at 0.541-0.551): the gate is NOT fit to exclude communities from retrieval. Second defect, fixed: the final answer cited community tags [c94] for sections because the section header said "community c94"; headers now say "community 94". OPEN: the c21 summary rests on a few exemplars of 1,024 sections and names only NapMem and Metis, so the global answer was thinner than the subgraph; the final step noticed and said so. Not yet measured: whether global + subgraph beats RRF alone (the 20-question blind comparison)
  _Lessons (2026-10-07, Postgres):_ PERSISTED with the operator's go: `python -u src/section_store.py --tag xpa` -> build 32 in NEW tables sect_build / sect_node / sect_edge / sect_community: 80,642 sections (text whole + vector(256) + tsvector of the body), 1,587,270 fused edges (weight, dense/sparse similarity and z, significance and backbone bitmasks), 823 communities with summaries and summary vectors; 159 s; `arxiv_sect` and every lex_* table untouched. 19 tests pass (store tests run inside a rolled-back transaction; the first run, before the transaction wrapper, left 7 synthetic test builds in the new tables, deleted by tag). Launch error: I started the persist twice, the first died with a MemoryError (two heavy runs at once). `section_graphrag.py` now reads only Postgres: seeds = pgvector exact `<=>` + full text, expansion = a join on sect_edge_sym ranked by edge weight, candidates by summary vector. FIXED on the way: (1) lexical arm ranked by ts_rank_cd with no idf: "models" (52% of sections) buried "safety" (2.1%); a hard cut on common words then returned MCMC sections for the agent-memory question (every topic word is common in an LLM corpus), so the arm now sums idf x ts_rank_cd per word; (2) the final step cited [c12]-style cluster tags for sections because the section header printed "community 12": the header no longer carries a community number. STILL OPEN, observed not fixed: the DENSE arm for "how do researchers evaluate the safety of language models" returns Conclusion / Appendix / Related Work sections (generic register), not safety sections, which is the genre axis showing up in retrieval; on the agent-memory question the final answer restated the global view and cited no section key. Genre exclusion default is now OFF.
  _Lessons (query strip, 2026-10-07):_ operator asked for the three-question strip back under the graphic (the Sep 3 graphic `.tmp/three_questions.png`: the region that answers lights up) with the GLOBAL community summary beside the SUBGRAPH per query. BUILT: `src/section_query_panel.py` + 4 tests; per question the map with the retrieved sections as red stars, their neighbours as orange dots and the communities that answered tinted, the GLOBAL answer whole with its communities, and the SUBGRAPH list; stacked under the base graphic into `.tmp/sections_xpa_queries_community_map.png` (2400x7459). The three questions are the speculative-decoding, agent-memory and safety-evaluation ones (the old three were Brown/wikitext).

  _Lessons (known-item battery, 2026-10-07):_ the repo's own battery (ingest_arxiv_sparsevec.gold, 200 paper titles + 200 body sentences, relevant = any section of the source paper) run on build 32, recall@50 pooled over the 400: dense 171 (.427, Wilson [.380, .476]), lexical 267 (.667), hybrid 276 (.690), dense with the 8 within-paper genre axes removed 231 (.578), hybrid with that dense arm 299 (.748, [.703, .788]). Paired on the same queries: dense_B finds 79 that dense misses, dense finds 19 that dense_B misses; hybrid_B 30 against hybrid 7; lexical against hybrid 30 against 39 (not distinguishable). So the genre-register hypothesis about the dense arm holds on this battery and the dense arm is now seeded from `sect_node.emb_b` (genre axes stored in sect_build.params, `python -u src/section_store.py --tag xpa --genre-vectors`, 80,642 rows, 35 s; 24 tests pass). The safety question now returns safety sections (community 44 defending against jailbreaks, "Safety Fine-Tuning"). OPEN, observed: the map step let a community through whose reply began with content and ended "This summary does not contain information about long-term memory" (community 39, RLHF), because is_relevant only checks the reply's start; the known-item battery says nothing about the global stage
  _Lessons (map step, community ranking, evidence set; 2026-10-07):_ MAP STEP FIXED: the reply's first line is now `RELEVANT: YES|NO` (an unparsed reply is counted, not used). 40 body-sentence queries x up to 12 candidates = 480 communities asked twice: old prompt + start-of-reply rule called 35 relevant and 6 of those (17.1%) contained a disclaimer phrase ("does not directly address", "does not contain information about"), new line called 29 relevant, 0 with a disclaimer, 0 unparsed (the disclaimer test is my own regex; the 3 old leaks were read verbatim and were real). Only 29 of 480 candidates (6%) answered YES. COMMUNITY RANKING by summary vector, recall@12 of the source paper's communities, 400 battery queries: original space 137 (.343, Wilson [.298, .390]), genre axes removed 158 (.395, [.348, .444]), paired 71 against 50: suggestive, not significant (sign test about p=.07), and both low: a community's summary alone finds the source paper's community in the top 12 only about 4 times in 10. EVIDENCE SET (operator 2026-10-07: typical sections of the top 3 communities by subgraph membership, then the sections nearest the question): built as `top_communities`, `choose_sections`, `section_store.centrepoints` (member nearest its community mean, genre axes removed) and `similarity`; the final step now sees 3 centrepoints + the 6 subgraph sections nearest the question: 8-9 sections and 15,000 / 38,812 / 27,580 characters against 14-18 sections and 27,362 / 76,440 / 55,359 on the three strip questions; answers still cite section keys. NOT measured: whether the answers are better (no key for a synthesis); on the speculative-decoding question the third centrepoint belongs to a community holding 1 of 14 sections (framing noise: a minimum of 2 members is the obvious fix)
- [OPEN] T173 Entity hops from the initial subgraph (operator 2026-10-07: "psql is capped at 4, so 3 hops out from the initial subgraph using the entities we identified; SPO would need synsets"). MEASURED before building (chunk level, lex_mention, 55,767 chunks with 19.0 entities each, 40 random seeds, one hop = chunk -> its entities -> every chunk mentioning any of them): bridge entity df cap none: 5,550 / 55,320 / 55,766 chunks reached at hop 1 / 2 / 3 (10.0% / 99.2% / 100%); cap 380 (p99): 1,410 / 52,909 / 55,342; cap 56 (p90): 162 / 14,634 / 51,326 (0.3% / 26.2% / 92.0%); cap 20: 45 / 1,600 / 22,744 (0.1% / 2.9% / 40.8%). So depth 3 is not a bound, the fan-out is: it needs a per-hop frontier cap ranked by weight (idf sum of shared entities), and iterative queries (3 SQL steps), not WITH RECURSIVE (a recursive term cannot prune per level and enumerates paths). The entities are CHUNK-keyed (lex_mention on the live chunk build): a section-level sect_mention is needed, from the same frozen inventory (arxiv_graph_service.frozen_inventory + entity_derive.match_frozen over the 80,642 section bodies). Evaluate on the known-item battery: recall of the source paper against list length after hop 0 (RRF), +edge hop, +1, +2, +3 entity hops, same list budget
  _Files:_ src/section_store.py, src/section_graphrag.py, tests/test_section_store.py
  _Verify:_ pytest tests/test_section_store.py -q; battery recall vs list length per hop; fan-out per hop after the cap
  _Lessons (2026-10-07, in progress):_ BUILT: sect_entity / sect_mention tables, `section_store.persist_mentions`, `entity_hop`, `entity_walk` (a hop is cut to n sections ranked by idf-weighted shared entities; bridges over max_df sections ignored; the frontier is only what the last hop added), `--entities` / `--reuse-mentions` in the store's main; 16 store tests pass (rolled back). MATCHING: the frozen 40,000-entity inventory of the live chunk build (tokenizer t3 matches the code) matched onto all 80,642 section bodies: 1,064,738 mentions, about 6 minutes; the first attempt died with a MemoryError at 22,000-32,000 sections (free virtual memory was 22 GB of commit against 28 GB free RAM, with about 45 python MCP processes resident), the second finished the matching and then lost the write: "server closed the connection unexpectedly". DOCKER: the engine is down (`docker ps` -> "open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified"; wsl lists docker-desktop Stopped; `docker desktop start` says "already running"), so every pgvector query is down until the engine is restarted (`docker desktop restart`, which stops all five containers: not done without the operator). The matches were not saved before the write; the main now saves them to .tmp/sections_xpa_mentions.npy BEFORE writing so a database outage costs no rematch. NOT DONE: write sect_mention, run `.tmp/entity_hop_recall.py` (expansion against a same-length RRF control at each stage, bridges at most 100 and 30 sections). MAP STEP: of 480 candidate communities on 40 queries, YES came 10 from communities holding a retrieved section (9.9% of those 101), 8 from neighbour-only communities (5.8% of 139), 11 from communities ranked by summary vector only (4.6% of 240): restricting the map step to communities that hold subgraph sections would halve the calls (240 against 480) and lose 11 of the 29 YES (38%), so it is TESTED AND REJECTED as proposed (caveat: the 40 queries are body sentences, not questions, and 26 of 40 got no YES at all). The 2-member minimum for typical sections is built (`top_communities`, default 2; 13 graphrag tests pass)

  _Lessons (2026-10-08, entity layer measured; the RL_V2 termgraph probe was a separate repo, files only, never in Postgres):_ WRITTEN to build 32: `1064738 mentions of 39263 entities in 78856 sections (217s)` (sect_entity / sect_mention did not exist until today: the build predates them). BUILT: `section_store.entity_edges` (sect_entity_edge, kind = co_mention, NPMI > 0, min 5 shared sections, bridges over 2000 sections ignored) -> `349110 edges`, 21673 entities with an edge, NPMI quantiles 10/50/90/99 = .094/.279/.640/.905; `section_store.community_entities` (Dunning G2 over-representation, top 8, at least 3 sections) -> lists for 362 of 823 communities, all 216 of 50+ sections. TWO LAYERS KEPT: `[overlap] n=8404 entities  mean top-5 Jaccard(co_mention, vector) = 0.2591  chance (shuffled partner) = 0.0006` (bar < 0.5; RL_V2 on 119-816 textbook chunks measured 0.036 vs 0.008). ENTITY HOPS LOSE the known-item test: `+ entity hop 3  107.1 | 209 / 400 = 0.522 | 330 / 400 = 0.825 | 5 / 126` (max_df 100; max_df 30: 207 vs 329, 6 / 128): at every stage the same-length RRF list holds the source paper more often, and so does the edge-neighbour step (`+ edge neighbours 17.1 | 197 vs 255 | 5 / 63`). NOT ADOPTED as retrieval expansion. NOT TESTED: whether hops add answer breadth (the known-item metric rewards the list nearest the query's own words and cannot see breadth); relation classes (rel:<class>, DIRT templates) were not run on sections. Display: `**Entities:**` line in the labelled md and `entities:` line (5) on each card (arxiv_community_map guard A14).

- [WIP] T174 A ReAct agent that traverses the section graph (operator 2026-10-08: "allowing a react agent to traverse the graph; it's not enough to simply retrieve a subgraph; we should be showing the subgraph to an llm react agent that can choose to iterate more"). INCUMBENT SEARCH: `react.py` (T61/T66) is a ReAct loop over WALK PARAMETERS (ef, anchors, ring) on the old chunk graph; its agent never sees nodes, so it is not extended; the agent loop lands INSIDE `src/section_graphrag.py` beside `answer()`. DESIGN: the agent sees the working set (the retrieved subgraph: key, community topic, similarity, how reached, a 160-character snippet) plus the global view, and each turn replies `ACTION: READ|EXPAND|ENTITY|SEARCH|ANSWER <arg>` and `WHY:`; READ shows a section in full (cut visibly at 6000 characters), EXPAND adds the 5 unseen sections along the strongest VECTOR edges of a section, ENTITY adds the 5 unseen sections that share named entities with it (the shared entities are named), SEARCH re-runs the hybrid retrieval on new words and adds its 3 seeds; at most 6 steps, then a forced stop; a reply that is not in the form, or names a key not in the working set, costs the step and tells the agent so (never silently corrected). The loop is pure (the model call and the four database actions are injected), so it is tested with scripted replies; the final answer is the existing synthesis over the global view and the standard evidence plus every section the agent read. NOT DECIDED by this task: whether the agent's answer is better than the fixed pipeline's (needs the blind comparison); the known-item metric cannot see it.
  AMENDED 2026-10-08 (operator: "up to 3 to 5 hops, but at hop 2 the agent has to plan out its next hop, and at hop 3 it has to plan out its next 2 hops, so it only gets two chances to plan"): the 6-step budget is REPLACED by hops counted outward from the retrieved sections. Hop 0 = the retrieved seeds, hop 1 = their vector neighbours, hop 2 = an entity hop from those (both automatic). PLANNING ROUND 1 (agent sees hop 2) plans hop 3; PLANNING ROUND 2 (sees hop 3) plans hops 4 and 5 TOGETHER, so it cannot see hop 4 before hop 5 runs: a move without a key (`HOP 5: EXPAND`) spreads from the 3 sections nearest the question among those the previous hop added. A hop is up to 3 moves (EXPAND key / ENTITY key / SEARCH words), each adding at most 5 sections, at most 15 per hop. Before each plan the agent may READ up to 2 sections; a closing round lets it READ and then ANSWER; ANSWER at any point stops the hops (so 3 to 5 hops, fewer if it stops). A reply that is not an action or a plan, an unknown key, or a plan with no valid move loses that attempt or move and is reported in the agent's history; a round that never makes its plan loses that chance (never retried for the agent).
  AMENDED AGAIN 2026-10-08 (operator: "and can stop early"; "the global communities help inform the react agent: show the largest top 12 or randomly sample a few examples based on their box-cox relative proportions as ratios (weighted sum to 1)"; "we can resample in between each opportunity the agent gets ... and they have access to prior communities? I'm spitballing"): (1) the agent can stop at any point (ANSWER). (2) COMMUNITY CATALOGUE: every round shows the agent a map of what else the corpus holds, 12 community titles with sizes, drawn WITHOUT replacement from the draft communities by weight = Box-Cox(size) / sum (lambda fitted once on all sizes; a singleton has weight 0 and is never shown); mode `top` shows the largest instead. MEASURED on build 32 (823 draft communities, 408 singletons, lambda -0.243): the 12 largest hold 19.5% of the sections but 4.1% of the Box-Cox weight; the 607 communities under 50 sections hold 5.8% of the sections and 36.3% of the weight; largest-to-median weight 815x raw, 5.4x Box-Cox. A fresh sample is drawn each round and the EARLIER ones stay visible (cumulative, new ones marked), so coverage grows 12, 24, 36. The seed is the question's checksum plus the round, so a run is reproducible. (3) THE QUERY STRIP shows, per question, the hops reached, the size of the whole subgraph, its sections per hop, and a mini composition of its entities (the entities mentioned by most of its sections, rarity-weighted).
  CORRECTED 2026-10-08 (operator: "wtf did I not ask for aggregate entity stats per community within the 12 shown as well as within the subqueries?"): the entity display was under-delivered three ways. (a) The 12 community cards and the md showed entity NAMES with no numbers: each now shows the entity names with their section counts AND an aggregate line per community (sections, share of them that mention an entity, distinct entities, mentions). (b) The strip gave ONE flat entity list for the whole subgraph: it now also breaks the subgraph down BY COMMUNITY (the communities holding most of its sections: sections of the subgraph in each, and each one's characteristic entities with counts). (c) The 12 communities shown to the agent carried titles only: each catalogue line now carries its top entities with counts.
  QUERY CHAINS 2026-10-08 (operator: the agent should see its first query's subgraph and the community sample, then word an informed second query, then a possible third, each time with a fresh sample): a SEARCH now returns its whole subgraph (up to 3 top sections + up to 10 strongest vector neighbours), every section carries the query that led to it (EXPAND and ENTITY inherit it), and the prompt groups sections by query, each group headed with the words and its count. MEASURED: all 823 communities through the map step would be about 203k input tokens per question (99k for the 415 of 2+ sections; estimate at 4 chars per token from the real prompt lengths, mean 989 characters), so presenting every community is feasible and the Box-Cox sample is a choice. Live, 4 questions: queries run [1, 1, 2, 3] with hops [2, 2, 3, 3] in the final run, and [1, 3, 4, 2] with hops [2, 5, 5, 3] in the run before the attribution fix; the second query's words reused terms from a section the agent had read (for the memory question, "taxonomy flat planar hierarchical"). Whether the second query's subgraph is BETTER than the first is NOT measured; the runs differ because the prompt changed between them, so do not read the difference as an effect.
  _Lessons (2026-10-08, planned-hop agent built and run live on 4 questions, claude-haiku-5.5 as the agent; answer quality still NOT measured):_ `hops reached [2, 2, 5, 3] | stops answered x4 | replies READ 8, PLAN 3, ANSWER 4`. Two of four questions the agent answered after READing in round 1 (hops 0-2 were enough), which "can stop early" allows. A SAFETY RUN LOST ITS FIRST PLAN: replies 1 and 2 were READs, reply 3 a READ refused (`no reads left this round`), so hop 3 never ran; the prompt now says THIS IS YOUR LAST REPLY at the last reply of a planning round and the 4-question rerun lost no plan. COMMUNITY SAMPLE, ABLATED (same 4 questions, with the Box-Cox sample vs none): identical hops `[2, 2, 5, 3]` both ways, identical stops, replies READ 8 / 9; `WHY quoting a shown community title: 0` of 15 replies with the sample. NO EVIDENCE the random sample steers the agent; a sample drawn by size-weight is mostly off-topic by construction (shown for the safety question: Psychological Models, Prompt generation for text review, Description of figures). NOT TESTED: a catalogue that adds the communities nearest the question (communities_by_summary) to the Box-Cox sample. ENTITY COMPOSITION DEDUPED (`speculative decoding 20 . speculative 11 . speculative decoding methods 7` was one entity three times). The answer model cites papers as `[arxiv/2306_07174]` without `#section` in some answers, so a count of read sections cited by exact key undercounts. Strip: `sections_xpa_queries_community_map.20261008-174026.png` (a viewer held the plain name).
  _Files:_ src/section_graphrag.py, src/section_store.py, src/section_query_panel.py, tests/test_section_graphrag.py, tests/test_section_store.py, tests/test_section_query_panel.py
  _Verify:_ pytest tests/test_section_graphrag.py tests/test_section_store.py tests/test_section_query_panel.py -q; four questions run live with each hop's size, the plans, the communities shown and the stop printed
  _Lessons (2026-10-08, built and run live on 4 questions; answer quality NOT measured):_ BUILT in section_graphrag.py: `parse_action`, `run_agent` (pure), `Explorer` (READ / EXPAND / ENTITY / SEARCH over Postgres), `agent_answer(agent_model=)`, `global_view` shared with `answer()`; `section_store.shared_entities`. THE AGENT MODEL DECIDES WHETHER THE LOOP WORKS. gemini-2.5-flash-lite: `steps 24 | READ 6, EXPAND 12, ENTITY 0, SEARCH 0, ANSWER 0 | lost to a malformed reply 6 | stops all budget` (it expanded one section 4 times, wrote `READ <key>` without `ACTION:`). claude-haiku-5.5 on the same questions: `steps 21 | READ 14, SEARCH 4, ENTITY 1, ANSWER 2 | lost 0 | stops answered, answered, budget, budget`; on the multi-hop question it searched for the missing half (speculative decoding) and used ENTITY once. Cluster tags leaked into 3 of 4 final answers ([c643], [c21], [c44]) until the answer and agent prompts stripped them (`strip_cluster_tags`): 0 after. Agent-read sections went last in the evidence and one question's answer cited neither of two it had found; reads first: cited 1 of 2, 4 of 4, 4 of 5, 3 of 3. DEFAULT agent model stays the pipeline's model until the blind comparison; use `agent_model="anthropic/claude-haiku-5.5"`. OPEN: the 20-question blind comparison (RRF alone / + subgraph / + global / + agent); 2 of 4 runs still ended on the 6-step budget; ENTITY was chosen once in 21 steps.

- [DONE] T175 Paper titles next to section keys (DONE 2026-10-08: `paper_title` holds 2,216 of 2,217 arXiv papers of build 32, 1,948 from the CSVs `_arxiv_data_with_enriched_bm25.csv`, `_missing_papers.csv` in C:/Users/user/arxiv_id_lists and 268 from the arXiv API; shown on the strip's section lines, the md exemplar headings, the card exemplar lines and the agent's section lines; the one paper without a title is 2602.1932, a truncated id; the CSV `thesis` column is stored, not displayed). ORIGINAL SPEC: Paper titles next to section keys (operator 2026-10-08, on the strip's `2602_06036#24 c94 6. Conclusion`: "we need to get the paper name if we're going to show 'conclusion' 'introduction' ... only showing the arxiv number doesn't do it justice. We should likely pull all those arxiv id titles through the api beforehand"). INCUMBENT SEARCH: no arXiv API client in the repo; `domain_corpora.arxiv_url` already maps a doc id (a `_methods` extract included) to its arXiv id and is reused. The titles held locally are the first section's HEADING (measured: "Modeling Greativity CASE STUDIES IN PYTHON TOM D.", an ALL-CAPS "OUTRAGEOUSLY LARGE NEURAL NETWORKS...", "SearchQA: A New Q&amp;A Dataset"), so they are not used. DESIGN: `src/arxiv_titles.py` fetches the official title of every paper of the build from the arXiv API (https://export.arxiv.org/api/query?id_list=, 100 ids a request, one request every 3.5 seconds, backoff on 429 and 5xx, stops after repeated failure rather than looping, never refetches a title it holds), stored once in a new table `paper_title (arxiv_id, title, fetched_at)` that no build owns; a paper the API does not return keeps no row and is shown by its key alone. DISPLAY: the query strip's subgraph lines, the 12 cards and the md exemplar headings, and the agent's section lines carry the paper title beside the section heading.
  _Files:_ src/arxiv_titles.py, src/section_store.py, src/section_query_panel.py, src/arxiv_community_map.py, src/section_graphrag.py, tests/test_arxiv_titles.py, tests/test_section_store.py, tests/test_section_query_panel.py, tests/test_arxiv_community_map.py, tests/test_section_graphrag.py
  _Verify:_ pytest on those; the titles table row count against the 2,217 + 303 distinct papers of build 32; the strip and a card opened

## Layer 42 -- the operator's inventive moves, built as arms the scorecard scores (operator 2026-10-06; plan C:\Users\user\.claude\plans\jolly-soaring-moler.md)
Operator: headings are about 5% of the text, a stop-phrase: keep the text, drop the heading from what is indexed; subtract an embedding for 'abstract'; exclude connections of this nature (a majority); mask a mostly-'Abstract' community without dropping its content; references masked out entirely (already so). Screen every arm on a seeded 20,000-section sample of WHOLE papers, then run the best two in full.
- [DONE] T166 Arm C, heading as a stop-phrase (built and scored on 3 samples at 2 resolutions: it only trims same-heading edges 16-17%, no grab-bag lead; closed with that context, re-open if the corpus or resolution moves): strip the leading `## heading` line from the text that is pooled and tokenised (both views); the full text stays on the record and in the markdown. Measure the heading share of tokens first
  _Files:_ src/section_corpus.py, src/section_embed.py, src/section_sparse.py, src/section_screen.py, tests/test_section_corpus.py, tests/test_section_screen.py
  _Verify:_ pytest on those; heading share of tokens printed; scorecard row
  _Lessons:_ MEASURED: a heading is 1.51% of all words (median 2.2%) but more than 5% of the words of 23.8% of sections and more than 20% for 2.5%; short sections (< 60 words, 14.4%) have a median heading share of 11.1%. Built: `section_corpus.index_text`, `section_sparse.build(strip_heading=True)`, the dense side re-pooled on stripped text; 12 + 8 tests pass. FIRST SCREEN (20,031 sections, 623 whole papers): same-heading share of fused edges 2.45% (x12 chance) baseline vs 2.03% (x10) arm C, so a heading explains only about a sixth of those edges and the rest is register. DEFECT in my screen: each arm picked its own plateau resolution (3.42 vs 1.9), so community counts, oversize share (22.6% vs 84.8%) and seed-ARI were NOT comparable; re-run at one fixed resolution (communities_from_graph(fixed_res=...), `res=` argument) before any partition verdict.
- [OPEN] T167 Arm D, kind-centroid subtraction: for each frequent normalised heading (coverage rule), subtract the mean dense vector of its sections from them, re-normalise
  _Files:_ src/section_genre.py, tests/test_section_genre.py
  _Verify:_ pytest tests/test_section_genre.py -q; scorecard row
- [OPEN] T170 Arm E, exclude connections between two sections that share a structural heading (frequent normalised heading), as the same-paper rule does: a mask inside the exact pass. Blunt on purpose: it also cuts real abstract-to-abstract links, so it is scored, not assumed. After arm C, recount same-heading edges against the 0.21% base rate to see whether it is needed at all
  _Files:_ src/section_graph.py, src/section_map.py, tests/test_section_graph.py
  _Verify:_ pytest tests/test_section_graph.py -q; same-heading edge share before and after; scorecard row
- [WIP] T168 Structural-community flag (BUILT and validated on 18 held-out; caption/code/list blocks still slip through, next signal pending; REFINED 2026-10-06 after the rubric read: a heading-majority rule would fire on 3 of 159 communities and miss the real problem, genre blocks of prompt templates, proofs, front matter and generic framing prose; the flag is a GENRE flag): a community whose mean GENRE FRACTION (the share of its sections' squared norm inside the top-8 within-paper axes of section_genre.role_axes on the centred dense view) is at least 0.533 is flagged genre: no label, no listing in the PNG or the markdown index, its content stays. PRE-REGISTERED 2026-10-06 on 14 communities hand-graded under the frozen rubric (6 topic max 0.531, 8 genre min 0.535: separation perfect, margin 0.004); HELD-OUT READ DONE 2026-10-06 (18 communities at midpoint ranks, graded blind): 0 of 7 topic/mixed flagged; genre caught 4 of 4 on the cross-paper map but 3 of 7 on the genre-axes-removed map (misses c137, c40, c5, c15). BUILT: section_genre.genre_fraction/genre_flags (size floor 10), flag measured on the ORIGINAL centred view in section_map.main, genre communities greyed and listed last in the markdown, cards = 12 largest topic communities. NEXT SIGNAL, PRE-REGISTERED 2026-10-06 BEFORE ANY SCORE IS COMPUTED (for blocks the genre fraction misses: image-description captions c13, code/list output c27): per community, the mean over its sections of the share of non-alphabetic characters in the text after the heading line. Hypothesis (NOT a finding): captions, code and list blocks sit well above prose. Test: compute it for the 38 communities already graded (20 first read + 18 blind, grades above) plus c13 and c27 graded non-topic, and report P(non-topic above topic) with the margin and n beside it; adopt a threshold only from the 20 first-read grades, then score the 18 blind ones. If it does not separate them, it is dropped and recorded as tested and lost. APPLIED to the saved cross-paper map: 157 communities / 34,402 sections (42.7%) flagged; the 12 cards still include non-topic blocks the flag misses (image-description captions c13, code/list blocks c27), so the flag is necessary, not sufficient. 55 tests pass. Original text: a community in which at least half the members share one structural heading gets no label and no listing in the PNG or the markdown index, its content stays
  _Files:_ src/section_scorecard.py, src/section_map.py, tests/test_section_scorecard.py
  _Verify:_ pytest tests/test_section_scorecard.py -q; count of flagged communities (expected 3 of 159)
- [DONE] T169 Plot: the 12 largest communities as cards BELOW the full UMAP star map (card layout restored from git HEAD: card_plan, the figure that grows to its longest card); markdown for every community
  _Files:_ src/arxiv_community_map.py, tests/test_arxiv_community_map.py
  _Verify:_ pytest tests/test_arxiv_community_map.py -q; the PNG opened and checked; both full paths given
  _Lessons:_ code written (`section_lines`, `card_plan`, `render_cards`; the stars-only `render` untouched so the chunk map is unchanged) and wired into src/section_map.py. Rendered and OPENED on a synthetic map (2400x1880, figure grew from the 21 in floor, cap raises) and on the real arm A map (`.tmp/sections_xpa_cards_community_map.png`, 2400x5712, 823 communities, 12 cards). VERIFIED: once the commit limit freed (it had blocked collection with WinError 1455 for a while: 76 python processes, 47.9 GB, mostly idle MCP servers), tests/test_arxiv_community_map.py passed 21 including the 4 new tests, and all six section suites passed together, 52 tests, after the last code edit. DEFECTS (carried by T171): every card has the same height, so most of each is empty (the figure is sized to its longest card); the 12 largest communities are the ones the split could not break, so the cards show grab-bags first (generic intro/conclusion, proofs, an image-caption '<description>' boilerplate community).
