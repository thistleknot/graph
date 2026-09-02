# Playbook: Multi-source RAG — quotes and wikitext beside Brown
Source: this session, 2026-08-31

Campaign: add HuggingFace `abirate/english_quotes` (~2.5k one-liners) and
`EleutherAI/wikitext_document_level` (~29k long docs) beside NLTK Brown in one
retrieval graph. Settled design (do not relitigate): per-source chunk params
(one Box-Cox fit per source, R17 generalized); source as node metadata; NORMAL/
EDGES normalization per source-pair block (6 blocks per space, each Box-Coxed
with the existing estimator-pair gate, then ONE global k-sigma cut in z space —
the same move FUSE already makes across spaces); NO retrieval quotas (sqrt
anchor allocation is LATER, only on measured starvation). New guards are minted
in the spec as **R19** (per-source chunk params), **R20** (source metadata on
every node, surfaced downstream), **R21** (per-source-pair block normalization,
single global cut). Implementation tasks cite them.

## Layer 1 — sequential
- [DONE] T1 Author the multi-source spec: R19/R20/R21 guards, steering rows, pipeline Phase A
  _Files:_ .spec/specs/graph-explorer/design.md, .spec/steering/product.md, .spec/PIPELINE.md
  _Verify:_ grep -c "R19\|R20\|R21" .spec/specs/graph-explorer/design.md .spec/steering/product.md .spec/PIPELINE.md
  _Notes:_ New design section §6.13 "Multi-source ingest" in the EXISTING
  graph-explorer spec dir — the sources ride the same pipeline, a new spec dir
  would split ownership of chunkgraph.py. Amend product.md CHUNK row (per-source
  hi thresholds) and NORMAL row (per-block Box-Cox, global z cut); PIPELINE.md
  Phase A chunk/normalize/edges rows and the guard index. Fix the node payload
  contract here: `source` key on every node, doc_id prefixed `brown/`, `quotes/`,
  `wiki/`. Guard text drafted here verbatim so T4–T6 paste it into the
  chunkgraph.py docstring. State acceptance criteria 5(a)–(c) as testable lines.
  _Lessons:_ §6.13 was already taken (Bridge discovery, 2026-08-31) so the new section landed at §6.14 instead; §9's Requirements coverage table is numeric (1.1, 6.2…) not guard-shaped, so R19/R20/R21 were not force-fit into it, per the subplan's skip clause.

- [DONE] T1b Document the alignment rationale where a README can find it (operator instruction 2026-08-31)
  _Files:_ .spec/specs/graph-explorer/design.md, README.md
  _Verify:_ grep -c "alignment" README.md .spec/specs/graph-explorer/design.md
  _Lessons:_ §6.14 gained subsection 4b (alignment family, size-imbalance argument, comparable-not-related caveat); new root README.md points to it and the .spec docs.

## Layer 2 — parallel
- [DONE] T2 Add the mixed-corpus driver that fetches both HF datasets and tags source (R20)
  _Files:_ ingest_mixed.py, tests/test_ingest_mixed.py
  _Verify:_ pytest tests/test_ingest_mixed.py -q
  _Notes:_ New file beside ingest_brown.py (incumbent is Brown-specific by name
  and docstring; it invites "swap load_docs()"). Import and reuse
  ingest_brown.load_docs for the Brown arm; `datasets` library for the two HF
  arms; every doc_id source-prefixed. Downscaling knobs (per-source n_docs /
  stride) are CLI args so T7/T8 differ only in arguments. Tests run offline on
  monkeypatched loaders — no network in the suite.
  _Lessons:_ D1 seam (`_fit` inspects `cg.fit`'s signature for `sources`) works against both today's two-arg ChunkGraph.fit and a hypothetical post-T4 three-arg one — verified with two duck-typed fakes — so T4/T5 landing `sources=` on chunkgraph.py needs zero edits here.

- [DONE] T3 Surface source mix in tools, UI, and export (R20 consumers)
  _Files:_ graph_tools.py, walker_app.py, export_neo4j.py, tests/test_graph_tools.py, tests/test_export_neo4j.py
  _Verify:_ pytest tests/test_graph_tools.py tests/test_export_neo4j.py -q
  _Notes:_ Reads the `source` payload key per T1's contract (fixtures fake it,
  so this does not wait on Layer 3). Community summaries, medoid cards, walk
  lists and the Neo4j node export report source mix; walker_app shows it.
  Sources absent (old runs) degrade to today's display, no crash.
  _Lessons:_ `pytest tests/test_graph_tools.py tests/test_export_neo4j.py -q` → 92 passed. The doc_id-prefix fallback in `source_of` never fired against live brown-50/brown-500-dual data (all pre-R20, unprefixed) — it is only exercised by the new synthetic tests — so T10 can safely drop it later if T4/T5's `source` key alone proves sufficient once real prefixed runs exist.

## Layer 3 — sequential
- [DONE] T4 Fit chunk params per source: one Box-Cox and hi threshold per corpus (R19)
  _Files:_ chunkgraph.py, tests/test_chunk.py
  _Verify:_ pytest tests/test_chunk.py -q
  _Notes:_ `fit(docs, doc_ids, sources=None)`; derive_chunk_params runs per
  source group; `chunk_params` becomes {source: {m, hi, unit, lam}} (single-source
  keeps a flat dict or a "default" key — subplanner's call, but pg_store persists
  it unchanged since `chunk_params` is already in `_PARAM_ATTRS`). Chunker applies
  its own source's params. Paste R19 into the module docstring from T1's text.
  _Lessons:_ `chunk_params` is always nested `{source: {...}}` (never a flat dict) — single-source runs use literal key `"default"`, so T5/T6 must index it as `chunk_params[src][...]`, never sniff flat-vs-nested. Brown-500 pin (test 9) skipped — corpus not downloaded locally; test 2's structural regression (derive_chunk_params_by_source == derive_chunk_params under one group, 3 varied corpora) already covers this offline.

- [DONE] T5 Carry source on every node into the store and export rows (R20)
  _Files:_ chunkgraph.py, pg_store.py, tests/test_pg_store.py
  _Verify:_ pytest tests/test_pg_store.py -q
  _Notes:_ `self.source[i]` beside `self.doc_id[i]` (R8's twin); node payload
  jsonb gains `source`; edges() rows already name docs — add source pair.
  Paste R20 into the docstring.
  _Lessons:_ `pytest tests/test_pg_store.py -q` → 16 passed (Postgres was live, test 6 ran as PASSED not skipped); `pytest tests/test_chunk.py tests/test_ingest_mixed.py -q` → 59 passed, 1 skipped (pre-existing, unrelated). Unlabelled runs write no `attrs['source']`/`edge.attrs` key at all (never `None`/`"default"`), matching R20's degrade clause; `edge.attrs jsonb DEFAULT '{}'` absorbed the source pair with no migration.

- [DONE] T6 Normalize similarities per source-pair block, one global k-sigma cut in z (R21)
  _Files:_ chunkgraph.py, tests/test_chunk.py
  _Verify:_ pytest tests/test_chunk.py -q
  _Notes:_ _cut/_edges_and_strength loop the 6 blocks per space (3 intra + 3
  cross); each block Box-Coxed on its nonzero sims with the existing R6
  estimator gate and R2/R3 fallbacks PER BLOCK (a degenerate cross block falls
  back alone, not the whole space); z values reassembled, then today's single
  global k-sigma cut and backbone/FUSE run unchanged. Blocks under ~20 nonzero
  pairs inherit the pooled fit — record which in diagnostics. Single-source runs
  are one block: Brown output stays byte-identical (regression assert). Paste
  R21 into the docstring. Per-block diagnostics record edge rate per block —
  T9's acceptance (a) reads them.
  _Lessons:_ `pytest tests/test_chunk.py -q` → 57 passed, 1 skipped (Brown-500 pin, corpus not downloaded, pre-existing); `pytest tests/test_pg_store.py tests/test_ingest_mixed.py -q` → 34 passed; `.tmp/t6_identity_probe.py` (single-source, fake-embed dense+sparse) → `IDENTICAL` on A/A_sparse/A_dense/provenance/strength/D/blend_mode vs HEAD's chunkgraph.py. `_cut` renamed to `_cut_pooled` as a pure move except the pre-authorized `20`→`MIN_BLOCK_PAIRS` substitution (subplan step 2). Tests exercised all three ladder rungs (own/pooled/rank). T7/T9 read per-block diagnostics at `cg.diagnostics[space]["blocks"]` (a list of dicts, one per realized block, sorted by `block` name; each has `block, n_pairs, n_pairs_all, fit, lam, post_kurt, divergence, edge_rate`) — same shape for single- and multi-source runs, so no branching on run type. Sprawl review: collapsed N / nothing to collapse — extended the incumbent `_cut`/`_edges_and_strength`, no new module.

## Layer 4 — sequential
- [DONE] T7 Smoke-ingest a downscaled mixed run and check per-block edge rates
  _Files:_ ingest_mixed.py, .tmp/mixed_smoke_report.py
  _Verify:_ python ingest_mixed.py mixed-smoke --brown 50 --quotes 500 --wiki 80 --wiki-stride 50 && PYTHONPATH=. python .tmp/mixed_smoke_report.py mixed-smoke
  _Notes:_ Strided sample of all three sources into Postgres :5433 under label
  `mixed-smoke` (new run, bitemporal, nothing superseded). The report script
  prints per-block edge rates from run diagnostics and fails if any two intra
  blocks differ by more than one order of magnitude — acceptance 5(a) on the
  small run. Must finish well under 15 min.
  _Lessons:_ Rung A (`--wiki 200`) was killed mid-fit at ~3.7 GB RSS (n well
  past the 12000-chunk hard stop from step 1's estimate) — downscaled to rung
  B (`--brown 50 --quotes 500 --wiki 80 --wiki-stride 50`), which fit in
  633 chunks (51 brown, 500 quotes, 82 wiki nodes) in under 2 min once HF's
  arrow cache was warm from rung A's aborted run. Sparse-space intra edge
  rates: brown|brown=2.506e-02 (fit=own), quotes|quotes=2.023e-02 (fit=own),
  wiki|wiki=2.289e-02 (fit=rank); ratio max/min=1.239, well inside the
  10x bound — acceptance 5(a) PASS. All three cross blocks fit=own. Report
  script needs `PYTHONPATH=.` (or run from repo root with `.` on path) to
  import `pg_store` — its own directory (`.tmp/`) isn't added to `sys.path`
  automatically; `_Verify:` above amended to include it. Failure paths
  confirmed live: nonexistent label and single-source `brown-50` both exit 2
  (no spurious PASS). T8 should expect wiki full-length docs to dominate
  chunk count fast — pre-size with a stride before committing to `--wiki 200`
  at full scale.

- [DONE] T8 Run the full mixed ingest as a detached, checkpointed job
  _Files:_ ingest_mixed.py, .tmp/mixed_full.log
  _Verify:_ python .tmp/mixed_smoke_report.py mixed-full  (after detached run completes; launch check = first checkpoint line in .tmp/mixed_full.log)
  _Notes:_ RISK, called out for the subplanner: chunkgraph holds dense n×n
  float64 matrices (sim, strength, D); 29k wikitext DOCUMENTS split into far
  more chunks, and n≈30k already means ~7 GB per matrix. First step of this
  task is to measure chunk count on a wikitext stride and pick the largest
  wiki stride that keeps peak RSS inside the machine — a documented stride IS
  the full run if n×n does not fit; widening chunkgraph to sparse z-block
  storage is a separate campaign, not this task. Detached with a staleness
  detector per stall-resume; verification itself stays <15 min.
  _Lessons:_ wiki_stride=4 chosen (probe: stride30->1043 chunks/982 docs, stride8->3903/3681,
  chunks/doc~1.06 stable both points; interpolated chunks(stride)~=31257/stride, stride=4
  confirmed at 7807 chunks/7361 docs, nearest 9000 target without exceeding, stride=3 would
  have hit ~10419 chunks); true wiki corpus is 29,444 non-blank rows (not 29,000), so the
  probe cap must be wiki_n=ceil(29444/stride)+margin or it goes cap-bound at exactly the cap
  (hit this at stride=4 with the subplan's ceil(29000/4)+100=7350 cap vs true 7361 stride-bound
  count — used --wiki 7461 for the real run to clear it). Projected total n~=10,369 docs
  (brown 500 + quotes ~2508 + wiki 7361) -> ~10,900-11,000 chunks, predicted peak RSS ~6.6 GB
  via the 55n^2 rule, comfortably under the 20 GB watchdog cap and the 12,000-chunk n_target.
  Timing anchor (step 2) did not finish inside its 10 min bound on brown+quotes (n~3000) — used
  the pessimistic bound T3k=10min per the abort clause, giving ETA~=10*(10991/3000)^2~=134 min
  (~2.2h), well under the 240 min reject threshold. Launch: 2026-08-31 ~21:39 local
  (2026-09-01T04:39:02Z), CKPT start line confirmed, err clean of tracebacks. Gotcha: `Start-Process
  -FilePath python` on this box resolves to a launcher stub (py310\Scripts\python.exe, PID 34424)
  that re-execs the real worker as a CHILD process (Python310\python.exe, PID 12712) — the
  launcher PID's RSS never moves, so the watchdog was retargeted mid-launch to the child PID
  12712 (confirmed via `Get-CimInstance Win32_Process -Filter "ParentProcessId=..."`), which is
  what `.tmp/mixed_full.pid` now holds. Watchdog PID 36076, log at .tmp/mixed_full.watch.log,
  running detached; first sample rss_gb=2.554 at wall_h=0.017, healthy.
  _Lessons (2):_ first mixed-full attempt spent 100+ CPU-min stuck at O(n^2) in _merge_phrases -- chunkgraph.py:379 rebuilt the full Phrases model inside the per-document listcomp (invisible at Brown's 500 docs, ~100h at 10,369). Killed, hoisted the model out (semantically neutral, pinned by new test_merge_phrases_single_model_matches_per_doc_rebuild, 58 passed), relaunched 23:23 with venv python (system python lacked psycopg); worker PID 29932, watchdog 28812.
  _Lessons (3):_ dense mixed run mixed-full-dual (da0903bb-8d8d-42dd-a0e6-a5c34ee16973): the model2vec artifact was at ~/models/m2v-minilm-l6-256 all along (walker_app.py DEFAULT_MODEL_DIR); CHUNKGRAPH_MODEL_DIR set -> dense=on. First attempt killed by the 20GB watchdog cap mid-NORMAL (dense space = 58.6M all-nonzero pairs, per-pair arrays ~0.5-1GB each); cap raised to 34GB, observed peak ~20.6GB (60s samples), fit 2114s, total 38min. 647,804 edges (sparse 264,099 / dense 274,589 / both 109,116), 16 comms. Report PASS both spaces: sparse intra ratio 1.156, dense intra ratio 2.546, all 12 blocks fit=own. Acceptance battery 5/5 vs mixed-full-dual via new MIXED_ACCEPTANCE_LABEL env override in tests/test_mixed_acceptance.py. A chunked-memory pass on dense per-block NORMAL is a legit LATER if runs grow past ~15k chunks.

- [DONE] T9 Prove the acceptance battery: source-targeted prompts and walker suites
  _Files:_ tests/test_mixed_acceptance.py
  _Verify:_ pytest tests/test_mixed_acceptance.py tests/test_sampler.py tests/test_gist_walk.py tests/test_walker_render.py -q
  _Notes:_ Against the T8 run (falls back to mixed-smoke if T8's label is not
  yet ready, and says so): (b) "a quote about courage" retrieves quotes-source
  chunks, a "how does wikipedia describe ..." prompt reaches wikitext, ≥3
  existing Brown prompts still anchor in Brown (Article V: ≥3 varied inputs per
  claim); (a) per-block edge rates within one order of magnitude, read from run
  params; (c) walker end-to-end suites pass. NO quota logic anywhere — if a
  source never surfaces on its own prompts, record the evidence and leave the
  sqrt-anchor fallback as the queued LATER item; do not build it.
  _Lessons:_ ran against `mixed-smoke` (mixed-full still ingesting, T8), 72/72
  passed; all three sources — brown, quotes, wiki — surfaced on every one of
  their own targeted prompts (~13 prompts, see .tmp/mixed_acceptance_report.txt),
  including both spec-literal prompts ("a quote about courage" -> 26/27 quotes;
  "wikipedia describe the history of the city" -> 22/24 wiki); intra edge-rate
  ratio 1.24 (well under the 10x bound). No source ever failed to surface on its
  own prompts, so the sqrt-anchor LATER item has no supporting evidence yet on
  mixed-smoke — re-check once mixed-full lands (~11k nodes may crowd BM25
  differently at scale).
  _Lessons (2):_ re-ran vs mixed-full (no fallback): 72 passed; ratio 1.156; all 6 blocks fit=own; every source anchored on its own prompts (brown 5-11 chunks on 4 brown prompts) -- LATER sqrt-anchor item has no supporting evidence at full scale either.

- [DONE] T10 Reconcile the spec against what shipped
  _Files:_ .spec/PIPELINE.md, .spec/specs/graph-explorer/design.md
  _Verify:_ grep -c "R19\|R20\|R21" chunkgraph.py .spec/PIPELINE.md
  _Notes:_ Article IX closing pass: measured numbers (per-source m/hi, block
  edge rates, final chunk counts, T8's chosen stride and why) land in the spec;
  any drift between minted guard text and shipped docstring resolved in the
  spec's favor or amended with rationale.
  _Lessons:_ Per-source chunk_params read live from Postgres (run mixed-full,
  86527c4e-e0d0-4701-a554-8be2d7f6f5db): brown m=107/hi=153/lam=-0.4409 (lines),
  quotes m=98/hi=256/lam=-0.2293 (chars), wiki m=30/hi=78/lam=0.1012 (lines).
  Drift verdicts: (1) R19/R20/R21 docstring vs design.md — no semantic drift,
  ASCII transliteration only, verified 2026-09-01; (2) `chunk_params` single-
  source key — spec incomplete, amended R19 to name the literal `"default"`
  key; (3) ≤8 source labels / int8 block codes / `MIN_BLOCK_PAIRS`=20 — code
  constraint absent from spec, amended R21; (4) product.md — no drift, not
  touched; (5) `source_of` doc_id-prefix fallback — kept as specified, noted
  mixed-full is the first live run exercising it end to end. T11 opened to
  re-paste both amended guard bodies into chunkgraph.py's docstring.

## Layer 5 — sequential
- [DONE] T11 Re-paste the amended R19/R21 guard bodies into chunkgraph.py's docstring
  _Files:_ chunkgraph.py
  _Verify:_ pytest tests/test_chunk.py -q
  _Notes:_ T10 amended design.md's R19 (names the `"default"` single-source key)
  and R21 (≤8 source labels, int8 block codes; `MIN_BLOCK_PAIRS`=20 named
  explicitly) but owns no source file. The docstring in chunkgraph.py still has
  the pre-amendment wording — re-paste both bodies verbatim (ASCII
  transliteration only) so spec and docstring stay in sync.
  _Lessons:_ done inline by the orchestrator (mechanical paste): R19 gained the literal "default" key sentence, R21 gained MIN_BLOCK_PAIRS(=20) and the <=8-source int8 ceiling; also fixed an editing artifact in design.md's amended R21 sentence. pytest tests/test_chunk.py -q -> 58 passed.

## LATER (queued, not in this campaign)
- sqrt-allocation of BM25 anchors per source (k_anchor_s ∝ n_s^0.5) — build
  ONLY if T9 shows a source demonstrably never surfacing on prompts that
  target it. Evidence lives in T9's _Lessons:.
