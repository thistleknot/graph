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
