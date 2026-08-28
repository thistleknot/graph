# Graph Explorer — Requirements

**Spec layer:** Requirements. Structural and behavioral commitments are deferred
to `design.md`; nothing here prescribes a table shape, a transport, or a widget
layout.

## Context

`chunkgraph.py` SERVE exposes `query()`, but it is in-memory and its anchor
selection is BM25-only: anchors come from `self.bm25` gated by `self.disc` (R1),
then H-hop expansion over the dense `self.A` adjacency, then `_gist_walk`. It
returns no community information. The operator wants to drive retrieval
interactively against **persisted** runs, fusing sparse and dense to locate a
region, then walk the induced neighbourhood and inspect the communities it
touches.

The composition already exists as SQL. `cookbook/queries.sql` holds:

- **#4 hybrid retrieval** — pgvector ANN for anchors plus graph expansion for the
  neighbourhood. `embedding` is L2-normalized, so `<#>` (negative inner product)
  ranks identically to cosine and is cheaper.
- **#1 weighted k-hop expansion** — mirrors `_gist_walk`; hops through
  `edge_sym`, decays by strength, keeps the best path per node, cycle-safe.
- **#3 community lookup** — which community holds a chunk, off the GIN index on
  `members`.

The explorer composes these. It does not invent a retrieval path.

Four facts govern everything below:

- **Subgraph Louvain is not a restriction of global Louvain.** Partitioning an
  induced neighbourhood yields community ids with no stable mapping to the run.
  Steering forbids partitioning at query time; the explorer therefore *reads*
  `community` rows, it does not compute them.
- **`jsonb` is payload-only.** `src`/`dst` are real indexed integer columns and
  are what make k-hop expansion an index scan. Traversal that filters on `attrs`
  is a regression.
- **Runs are superseded, never deleted.** A label can have history behind it, so
  "which run" is an explicit choice, not an assumption.
- **The dense space is optional (R5).** The only run currently persisted,
  `brown-50`, has `embed_dim` NULL, zero `node_embedding` rows, and edge
  provenance that is 100% `sparse` (4568 of 4568). Hybrid retrieval is therefore
  **not exercisable against present data**, and sparse-only is not a hypothetical
  edge case here — it is the current state.

---

## Requirement 1 — Read-only over persisted runs

**As an** operator, **I want** the explorer to read a stored run, **so that**
exploring never mutates or invalidates an ingest.

1.1 The system SHALL read graph state from Postgres and SHALL NOT fit, write,
supersede, or delete any run.

1.2 The system SHALL require an explicit run selection, identified by label.

1.3 WHERE a label has multiple runs, the system SHALL default to the live one and
SHALL make the selected run's identity visible.

---

## Requirement 2 — Hybrid where available, honest when not

**As an** operator, **I want** dense and sparse to be fused when both exist,
**so that** the located region reflects both similarity spaces.

2.1 WHEN the selected run has a dense space, anchor selection SHALL combine BM25
and vector similarity.

2.2 IF the selected run has no dense space — `embed_dim` NULL or no embedding
rows — THEN the system SHALL fall back to sparse-only anchors.

2.3 WHEN the system falls back per 2.2, it SHALL state that it has done so. A
degraded retrieval SHALL NOT be presented as a fused one, and SHALL NOT surface
as an empty result or an error.

2.4 The system SHALL make the effective retrieval mode visible for every query.

---

## Requirement 3 — Communities are the run's own

**As an** operator, **I want** the communities shown to be the ones that were
committed at ingest, **so that** what I inspect is what a reviewer would label.

3.1 The system SHALL display communities using the selected run's stored `cid`,
`keywords`, and `medoid`.

3.2 The system SHALL NOT re-partition, re-cluster, or otherwise recompute
community membership over a retrieved subgraph.

3.3 WHERE retrieved nodes span multiple communities, the system SHALL group them
by their existing `cid` rather than deriving new groupings.

3.4 IF local structure within a subgraph is ever surfaced, THEN it SHALL be
labelled exploratory, SHALL NOT be persisted, and SHALL NOT be presented as or
alongside a `cid`.

---

## Requirement 4 — Traversal stays on the index

**As an** operator, **I want** expansion to remain an index scan, **so that**
the explorer stays usable as runs grow.

4.1 Neighbourhood expansion SHALL traverse via the indexed `src`/`dst` columns.

4.2 The system SHALL NOT use `attrs` as a traversal or join key.

4.3 Hop depth SHALL be bounded, and the bound SHALL be visible and operator-
controlled.

4.4 The system SHALL bound the size of a returned neighbourhood so that a
high-degree anchor cannot return the whole graph.

---

## Requirement 5 — Provenance is first-class

**As an** operator, **I want** to see which space produced each edge, **so that**
a space's contribution can be audited rather than assumed.

5.1 The system SHALL expose each displayed edge's `{sparse,dense,both}`
provenance.

5.2 WHERE every edge in the selected run carries a single provenance value, the
system SHALL state this rather than implying a fused result.

---

## Requirement 6 — Results stay attributable

**As an** operator, **I want** every chunk traced to its source, **so that** a
retrieval can be checked without trusting the ranking.

6.1 Every displayed chunk SHALL name its source `doc_id` (R8).

6.2 The system SHALL make the ranking signal that selected a chunk inspectable,
so that a result can be distinguished from its neighbours.

---

## Non-goals

- No labelling, renaming, confirming, or rejecting of communities. That is the
  `community-labelling` spec; this one is read-only by R1.
- No LLM anywhere in retrieval, expansion, or grouping.
- No re-ingest or model selection from the UI.

## Open question deferred to design

Exercising R2.1 and R5 requires a run with a dense space. `brown-50` has none.
Whether the explorer ships against sparse-only data first, or a dense re-ingest
precedes it, is a sequencing decision for `design.md`. Note that re-ingesting
mints a **new run** under supersede-never-delete, so stamping `embed_dim` on a
fresh run does not engage the immutability constraint on existing rows.
