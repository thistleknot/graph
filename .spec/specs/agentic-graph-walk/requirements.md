# Agentic Graph Walk — Requirements

**Spec layer:** Requirements. Structural and behavioral commitments are deferred
to `design.md`; nothing here prescribes a tool signature, a transport, or a
widget layout.

**Provenance of this document — read this first.** This spec is **retroactive**.
`graph_tools.py`, `walker_app.py` and `tests/test_graph_tools.py` were written
first, under an operator-authorized `.spec/BYPASS`, to get a working prototype
in front of the operator quickly. This document records the requirements that
code *actually* satisfies, plus the ones the agentic half has yet to meet. It is
a reconstruction, not a forecast, and it is marked as such so nobody later reads
it as evidence that spec-first was followed here. It was not.

## Context

`chunkgraph.py` builds a dual-space retrieval graph deterministically: chunking,
edge formation, fusion and the Louvain partition all run at ingest with no model
in the loop. `pg_store.py` persists it. `graph-explorer` (separate spec, at
design) covers a fixed-shape query UI over that store.

This spec covers something different: **walking the graph to accumulate
sufficient evidence to answer a question, then answering from the communities
that evidence lands in.**

> **Amended after measurement (see `design.md` §1).** This document originally
> specified an LLM tool-caller choosing the route. That was built and run:
> 14 steps, 26 chunks, `finish` never called, no answer, with community
> saturation flat from step 8 — the evidence had converged six steps before the
> budget expired and the model did not notice. The traversal is now a
> **Boltzmann sample over the induced subgraph**, and the model is confined to
> summarising a frozen evidence bundle. It chooses nothing. Clauses below that
> still assume a router are marked; `design.md` §6 is the full reconciliation.

Four facts govern everything below:

- **The substrate is deterministic; only the route is not.** Communities, edges,
  strengths and keywords are all fixed at ingest. An LLM walking that graph
  chooses which edges to follow — it does not create, weight, or name anything.
  This is what makes an agentic walk auditable rather than a black box.
- **The steering ban is scoped to construction.** `## Banned: No LLM in the
  graph construction path — chunking, edges, fusion, and community membership
  are deterministic.` A retrieval-time walker touches none of those. This spec
  does **not** seek an amendment to that ban, because it does not violate it.
- **A second graph engine is a closed decision at shallow depth.**
  `chunkgraph.py` CLOSED: *"A second graph engine for 1-2 hop traversal:
  rejected. Edge rows in the existing store cover it; an engine earns its place
  at deep traversal."* This spec introduces no engine. It composes SQL over the
  existing Postgres store.
- **`jsonb` is payload-only.** `src`/`dst` are real indexed integer columns and
  are what make expansion an index scan. A tool that let a caller filter
  traversal on `attrs` would be a regression, so the tool surface must make that
  inexpressible rather than merely discouraged.

---

## Requirement 1 — The tool surface is closed

**As an** operator, **I want** the caller to be unable to express an unsafe
query, **so that** autonomy does not depend on the caller's good behaviour.

1.1 The system SHALL expose a fixed set of named traversal primitives and SHALL
NOT expose any primitive that accepts raw SQL, Cypher, or a query fragment.

1.2 The system SHALL NOT interpolate caller-supplied values into SQL
identifiers. All caller input SHALL reach the database as bound parameters.

1.3 Traversal SHALL read the indexed `src`/`dst` integer columns. The system
SHALL NOT offer any primitive that traverses or joins on `attrs`.

1.4 Every primitive returning multiple rows SHALL take an explicit bound, and
SHALL NOT offer an unbounded fetch.

---

## Requirement 2 — Read-only, enforced not promised

**As an** operator, **I want** a walker that cannot mutate an ingest, **so that**
an autonomous caller can be let loose without risking the store.

2.1 The system SHALL open its database connection read-only at the server.

2.2 WHEN any statement attempts a write, the database SHALL reject it, and this
SHALL be verified by test rather than asserted by convention.

---

## Requirement 3 — Communities are the run's own

**As an** operator, **I want** the communities a walk reports to be the ones
committed at ingest, **so that** what the caller reasons about is what a human
reviewer would see.

3.1 The system SHALL report communities using the selected run's stored `cid`,
`keywords`, and `medoid`.

3.2 The system SHALL NOT re-partition, re-cluster, or otherwise recompute
community membership over any walked subgraph.

3.3 WHERE a visited node belongs to no community, the system SHALL say so rather
than omitting the node or inventing a grouping.

3.4 Community-level interconnection SHALL be derived by aggregation over stored
`cid`s only.

---

## Requirement 4 — Every bound is visible

**As an** operator, **I want** to know what a walk did not show me, **so that**
absence of evidence is distinguishable from evidence of absence.

4.1 WHERE a result is truncated by a bound, the system SHALL disclose that
truncation and the bound that caused it.

4.2 The edge-strength floor SHALL be an explicit, caller-visible parameter and
SHALL NOT be a hard-coded literal.

4.3 Hop depth and neighbourhood size SHALL be bounded and operator-controlled.

4.4 WHEN neighbours are truncated, the system SHALL retain the strongest rather
than an arbitrary subset.

---

## Requirement 5 — The walk is auditable

**As an** operator, **I want** to replay what a walk saw, **so that** an answer
can be checked without trusting the caller's narration.

5.1 The system SHALL record what was retrieved and how. **Amended:** with
sampling there is no route to record — the record is the evidence bundle (the
sampled ordinals and the ranked `cid`s) plus the parameters that produced it,
`(run_id, query, anchors, n, T, seed)`.

5.2 Every chunk surfaced SHALL name its source `doc_id` (R8).

5.3 Every edge surfaced SHALL name its `{sparse,dense,both}` provenance.

5.4 A recorded walk SHALL be **exactly reproducible** against the same run.
**Amended — strengthened:** the original clause conceded that a caller's route
could not be reproduced. Sampling is a pure function of stored data and a seed,
so the retrieval is not merely replayable but re-derivable. A disputed answer
can be recomputed rather than taken on trust.

---

## Requirement 6 — Termination is a stated rule

**As an** operator, **I want** a walk to stop on a defined condition, **so that**
"sufficient evidence" is a criterion rather than a vibe.

6.1 The system SHALL bound total work per walk by an explicit budget. The
budget is the sample size `n`; a fixed-`n` draw cannot overrun.

6.2 The system SHALL expose a saturation signal — the rate at which new
communities are being introduced. **Amended:** this is no longer consumed by a
router at query time. It is the offline criterion for choosing `n`: sample
repeatedly at a candidate `n` under different seeds and raise `n` until the
top-community set stabilises across seeds (`design.md` §4).

6.3 **Obsolete.** Superseded by 6.1 — a fixed-`n` sample has no failure mode
where the budget expires before a stopping condition is met, because `n` *is*
the stopping condition. Retained here so the numbering does not shift and the
deletion stays visible.

6.4 The sampled evidence bundle SHALL be frozen before the model is called. The
model SHALL NOT be able to request further retrieval.

---

## Requirement 7 — Honest about the dense space

7.1 IF the selected run has no dense space, THEN the system SHALL operate
sparse-only and SHALL state that it has done so.

7.2 The system SHALL NOT present a single-provenance run as a fused result.

---

## Non-goals

- **No LLM in graph construction.** Chunking, edges, fusion and community
  membership remain deterministic and untouched by this spec.
- **No new graph engine.** No Neo4j, no AGE, no Cypher execution.
- **No writes**, no labelling, no re-ingest, no model selection.
- **Not a replacement for `graph-explorer`.** That spec's fixed-shape UI stays
  LLM-free; this is a separate surface over the same store.

## Open questions deferred to design

1. **What exactly stops a walk?** 6.2 names saturation as *a* signal, not *the*
   rule. Candidates: new-community rate, evidence-coverage against the question,
   or a plain budget. Measurement needed — on `brown-50`, a 2-hop neighbourhood
   from a hub touched 25 of 40 communities raw and 13 with the strength floor
   applied, so the signal is real but its threshold is unset.

2. **How is context budgeted?** A 2-hop hub neighbourhood is ~180 nodes. Design
   must decide what a step returns by default (preview vs. full text) and how
   full text is requested selectively.

3. **Is the walk trace a persisted artifact?** 5.4 requires replayability, which
   R1's read-only constraint means cannot be stored in this database by this
   system. Where a trace lives is a design question.

4. **Does an LLM-selected anchor need R1-style validation?** `chunkgraph.py`'s R1
   gates lexical anchors on the matched term appearing in the chunk's top
   `DISC_M` tf*idf terms. Whether a caller-chosen starting node needs an
   analogous guard is unresolved.

## Implementation status

Updated after the router was measured and replaced. Superseded text is not kept
here; `design.md` §1 holds the evidence that retired it.

**Satisfied and tested** (`tests/test_graph_tools.py`, 22 passing):
R1.1–1.4, R2.1–2.2, R3.1–3.4, R4.2, R4.4, R5.2, R5.3, R7.1–7.2.

**Partially satisfied:** R4.1, R4.3 — bounds exist and are visible in the UI,
but truncation is not yet reported as a distinct signal in the tool return
values.

**Built, measured, and retired:** `walk_agent.py` implements the LLM tool-caller
this document originally specified. It runs, its tool-calling works, and it does
not produce answers — 14 steps, 26 chunks, `finish` never called, saturation
flat from step 8. It stays in the tree only until `explorer/sampler.py` replaces
it, and it should not be read as the current design.

**Not built:** the Boltzmann sampler (`explorer/sampler.py`), the summarisation
step (`explorer/summarize.py`), and the `n`/`T` sweep of §4. Guards S1–S7 in
`design.md` §5 have no code behind them yet. Requirements 5.1, 5.4, 6.1, 6.2 and
6.4 are therefore **specified but unimplemented** — their amended text describes
the sampler, which does not exist.

**Open and unmeasured:** whether the community histogram has a head clean enough
to cut (`design.md` §7.3). The design assumes sampled chunks concentrate in a
few communities; the only measurement to date is that a 2-hop neighbourhood on
`brown-50` touched 25 of 40 communities raw and 13 under the strength floor,
which is a long tail. If the head does not separate, the aggregation step needs
rethinking before `n` is worth tuning.
