# Community Labelling — Requirements

**Spec layer:** Requirements. Structural and behavioral commitments are deferred
to `design.md`; nothing here prescribes a table shape.

## Context

`chunkgraph.py` COMMUNITY assigns every chunk a Louvain partition id and derives
top-5 tf\*idf `keywords` plus a `medoid`. Product intends an **SME reviewer** who
confirms, rejects, or renames those communities. That reviewer does not exist.

Two facts from the existing schema govern everything below:

- `community` is keyed `PRIMARY KEY (run_id, cid)`. `cid` is a Louvain partition
  id and is **run-local** — a re-ingest mints entirely new communities.
- `community.members` holds node `ord` values, and `node` is keyed
  `(run_id, ord)`. **`ord` is also run-local.** The declared stable identity is
  `node.chunk_hash` (`bytea`, "stable cross-run identity", indexed `node_hash`
  as "same chunk across runs").

Therefore a label attached to `(run_id, cid)` is destroyed by the next ingest,
and any carry-forward that compares raw `members` arrays is comparing run-local
ordinals and is wrong whenever chunking shifts. Carry-forward must resolve
through `chunk_hash`.

---

## Requirement 1 — Labels survive re-ingest

**As an** SME reviewer, **I want** the names I confirm to persist across
re-ingests, **so that** review effort is not destroyed on every run.

1.1 The system SHALL persist a community label under an identity that is
independent of `run_id` and of `cid`.

1.2 WHEN a new run is persisted under a label that already has a live run, the
system SHALL attempt to carry each existing community label forward to a
community of the new run.

1.3 The system SHALL compute carry-forward candidacy from **`chunk_hash` member
sets**, not from `members` ordinals and not from `cid`.

1.4 IF a prior community's chunks are wholly absent from the new run, THEN the
system SHALL retain its label as unbound history rather than deleting it.

## Requirement 2 — The determinism boundary is preserved

**As** the operator, **I want** labels to stay inert text, **so that** retrieval
never depends on model prose.

2.1 The system SHALL NOT use a label, or any label-derived value, as a join key,
a traversal key, or an input to edge formation, fusion, or partitioning.

2.2 The system SHALL NOT alter Louvain membership on the basis of a label.

2.3 WHERE a label was authored by a model, the system SHALL record it as a
**draft** and SHALL make its status retrievable in the same read that returns
the label text.

2.4 IF a caller requests labels without specifying a status filter, THEN the
system SHALL NOT return model drafts and SME-confirmed labels in a form that
makes the two indistinguishable.

## Requirement 3 — Ambiguous carry-forward is surfaced, never guessed

**As an** SME reviewer, **I want** to be told when a community has split or
merged, **so that** a stale name is not silently reattached to a changed cluster.

3.1 The system SHALL compute an explicit overlap measure between each prior
labelled community and each candidate community of the new run.

3.2 WHEN exactly one candidate exceeds the configured overlap threshold and no
other candidate exceeds a configured ambiguity margin, the system SHALL bind the
label automatically and SHALL record the measure that justified the bind.

3.3 IF two or more candidates exceed the threshold — a split — THEN the system
SHALL NOT bind automatically, and SHALL retain the label as **pending review**
against all tied candidates.

3.4 IF one candidate is the best match for two or more prior labelled
communities — a merge — THEN the system SHALL NOT bind automatically, and SHALL
retain each competing label as pending review.

3.5 IF no candidate exceeds the threshold, THEN the system SHALL leave the new
community unlabelled rather than binding the nearest match.

3.6 The system SHALL make carry-forward deterministic: identical inputs and
thresholds SHALL yield identical bindings, with ties broken by a stated total
order rather than by iteration order.

## Requirement 4 — Review provenance is recorded

**As** the operator, **I want** to know who named a cluster and when, **so that**
a categorization is auditable.

4.1 The system SHALL record, for every label: its status, its author, and the
time it reached that status.

4.2 The system SHALL distinguish at minimum: model draft, SME-confirmed,
SME-rejected, and pending review.

4.3 WHEN an SME confirms, renames, or rejects a label, the system SHALL preserve
the prior state rather than overwriting it.

4.4 WHEN a label is bound by carry-forward, the system SHALL record the overlap
measure and the prior community it descended from.

## Requirement 5 — Confirmation is invalidated by drift

**As an** SME reviewer, **I want** a confirmation to lapse when the cluster it
described has materially changed, **so that** a confirmed name cannot quietly
come to mean something else.

5.1 The system SHALL record the `chunk_hash` member set that was in force at the
moment of SME confirmation.

5.2 WHEN a label carries forward to a community whose membership has drifted
beyond a configured tolerance from the confirmed set, the system SHALL demote
the label to pending review and SHALL NOT present it as confirmed.

5.3 The system SHALL make the drift measure retrievable alongside the label.

## Requirement 6 — Bitemporal consistency with the existing store

**As** the operator, **I want** label history to behave like edge history,
**so that** one as-of pattern serves the whole store.

6.1 The system SHALL supersede label states rather than deleting them,
consistent with `supersede_label` and the `valid_to IS NULL` live-set
convention.

6.2 WHEN a run is superseded, the system SHALL leave labels queryable as of the
superseded run.

6.3 The system SHALL persist label changes within the same atomic commit as the
run they attach to, when they arise from ingest.

## Requirement 7 — The review surface is evidence-first

**As an** SME reviewer, **I want** corpus-derived evidence next to any proposed
name, **so that** I can falsify it without trusting prose.

7.1 WHEN presenting a community for review, the system SHALL include its
`keywords`, its `medoid_text`, and its `size`.

7.2 The system SHALL present a sample of member chunk bodies sufficient to judge
the cluster, bounded to a configured maximum.

7.3 WHERE a model draft is presented, the system SHALL present the corpus-derived
evidence in the same view, and SHALL NOT present the draft alone.

7.4 The system SHALL order the review queue so that communities with no live
label and the largest `size` are presented first.

## Requirement 8 — Labelling is optional and degrades gracefully

**As** the operator, **I want** ingest and query to work with no labels present,
**so that** the feature is additive.

8.1 WHERE no label exists for a community, `query()` and export SHALL behave
exactly as they do today.

8.2 IF the labelling store is empty or absent, THEN ingest SHALL complete
without error.

8.3 The system SHALL NOT make an LLM call a precondition of ingest completing.

---

## Open questions for the operator

1. **Threshold values.** 3.2, 3.3, and 5.2 are stated as configured. I have not
   proposed numbers — the right ones depend on how much your chunking actually
   shifts between ingests of a near-identical corpus, which is measurable and is
   not yet measured.
2. **Overlap measure.** Jaccard penalises a community that legitimately grew;
   containment does not, but binds a stale narrow label to a broadened cluster.
   This is a real trade-off and I do not think it should be settled by default.
3. **Who is the SME in practice.** Product says single operator. If reviewer and
   operator are the same person, Requirement 4's author field is nearly vacuous
   and could be dropped for now.
4. **Scope of drafting.** Requirement 2.3 permits model-authored drafts but this
   spec does not require them. Whether draft generation is in scope at all, or
   whether v1 is confirm/rename/reject over tf\*idf keywords only, is undecided.
