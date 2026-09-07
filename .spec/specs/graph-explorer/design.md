# Graph Explorer — Design

**Spec layer:** Design. This document commits to structure and behavior for the
requirements in `requirements.md`. Task breakdown is deferred to `tasks.md`.

## Decisions taken at this layer

| # | Question from requirements | Decision |
|---|---|---|
| 1 | Sequencing: sparse-only first, or dense re-ingest first? | **Dense re-ingest first**, using `model2vec` distilled from `all-MiniLM-L6-v2`. §2 |
| 2 | Does SQL k-hop (#1) reproduce `_gist_walk`? | **No — settled by inspection.** Different algorithms; the cookbook's "Mirrors `_gist_walk`" comment is false. Open question is only the magnitude of divergence. §5 |
| 3 | R4.4 neighbourhood cap value | **120**, measured: just above the 2-hop reach p99 of 111. §6 |
| 4 | Cookbook #1's undisclosed `score > 0.05` floor | **Keep the value, disclose it.** It discards 67% of the 2-hop neighbourhood and was silent. §4.2 |

All four are backed by measurements against the live `brown-50` run
(`4b5bd4a7-c229-4c93-8151-ca5324d99c8e`, 1789 nodes / 4568 edges / provenance
4568 `sparse`), not by inference from the source alone.

---

## 1. Component structure

Three modules, one direction of dependency, no cycles:

```
explorer/store.py      Postgres access. Read-only. Owns RunHandle + Capability.
explorer/retrieve.py   Anchor selection, expansion, community grouping.
explorer/app.py        Streamlit UI. No SQL, no scoring.
```

`app.py` holds no SQL and no ranking arithmetic; `retrieve.py` holds no widgets.
This keeps R2.4/R5.2/R6.2 — all "make X visible" requirements — testable without
driving a browser, because the values they surface are returned by `retrieve.py`
as data.

`chunkgraph.py` is **not** imported by the explorer at runtime. It is imported by
the equivalence test (§5) only. The explorer's retrieval path is the SQL in
`cookbook/queries.sql`, per the requirements' "It does not invent a retrieval
path."

### 1.1 Read-only is structural, not conventional (R1.1)

`store.py` opens its connection with `read_only=True` and
`default_transaction_read_only` set on the session. Any `INSERT`/`UPDATE`/
`DELETE`/`CREATE` — whether written deliberately, reached by a future edit, or
injected — fails at the server with `ReadOnlySqlTransaction`.

R1.1 is thereby enforced by Postgres rather than by reviewer vigilance. This
matters more than the usual read-only-connection hygiene, because the explorer
sits in the same repo as an ingest path that legitimately writes; a shared helper
being reused later is a realistic way to violate R1.1 by accident.

The explorer never issues DDL, so read-only costs nothing.

### 1.2 Run identity (R1.2, R1.3)

`RunHandle` is the only way to name a run downstream. It carries `run_id`,
`label`, `ingested_at`, `superseded_at`, `embed_dim`, and the model identity
fields of §2.3. Constructing one requires an explicit label — there is no
"current run" default (R1.2).

**Verified against the live schema.** `graph_run` has no `valid_to` column; run
supersession is recorded in `superseded_at`, and `valid_to` belongs to `edge`.
Resolution therefore reads the `live_run` view (`superseded_at IS NULL`), not a
`valid_to` predicate.

Because runs are superseded and never deleted, one label accumulates history.
The **zero-live-run case is real** and is raised rather than papered over: a
label whose only runs are superseded is a legitimate state, and silently taking
the newest would violate "which run is an explicit choice, not an assumption."

The more-than-one case is **not** defended against in application code, because
the database already forecloses it:

```
graph_run_live_label UNIQUE, btree (label) WHERE superseded_at IS NULL
```

An application-side "fails loudly if more than one row" check would be dead code
by construction. The invariant is enforced where it belongs.

The UI displays `run_id` and `ingested_at` alongside the label at all times
(R1.3), not only when several runs exist — an identity that appears only in the
ambiguous case is one operators learn to ignore.

---

## 2. Dense space: model2vec re-ingest

### 2.1 Why re-ingest at all

R2.1 and R5 are not exercisable against `brown-50` (`embed_dim` NULL, zero
`node_embedding` rows, provenance 4568/4568 `sparse`). Shipping sparse-only first
would mean the fusion path (R2.1), the provenance mixing (R5.1), and the "state
the degradation" path (R2.3) all ship with the degraded branch as the only branch
ever executed. The honest-fallback requirements are precisely the ones that need
a non-degraded case to be distinguishable from.

Re-ingest mints a **new run** under supersede-never-delete, so stamping
`embed_dim` does not engage the immutability constraint on existing rows.
`brown-50`'s existing run is superseded, not deleted, and remains selectable —
which conveniently makes it the fixture for R2.2/R2.3 and R5.2.

### 2.2 Why model2vec rather than MiniLM itself

`model2vec` distills a sentence-transformer into a static token embedding matrix.
Encoding becomes a table lookup plus a weighted mean — no transformer forward
pass. For a corpus the size of `brown-50` (1789 chunks) this is not about
throughput; it is that the dense space becomes reproducible from a committed
artifact on CPU, with no GPU, no network, and no inference-time variance.

Verified available in this environment:

- `model2vec` 0.7.0 installed.
- `sentence-transformers/all-MiniLM-L6-v2` already in the local HF cache, so
  distillation runs **fully offline**.
- `CHUNKGRAPH_MODEL_DIR` is currently unset.

### 2.3 Distillation parameters must be pinned, not defaulted

The installed signature is:

```
distill(model_name, vocabulary=None, device=None, pca_dims=256,
        sif_coefficient=0.0001, token_remove_pattern='\[unused\d+\]',
        trust_remote_code=False, quantize_to=DType.Float16,
        vocabulary_quantization=None, pooling=PoolingMode.MEAN)
```

Three defaults are load-bearing and are therefore **passed explicitly** rather
than inherited:

1. **`pca_dims=256`.** The distilled space is 256-dimensional, *not* MiniLM's
   384. Whatever value is used is stamped into the run's `embed_dim` and is
   immutable for that run's lifetime. Inheriting a library default into a
   permanent schema field is how a future `model2vec` release silently changes
   the meaning of an existing column. Pinned explicitly at **256**.

2. **`quantize_to=Float16`.** pgvector's `vector` type is float32. Vectors are
   cast to float32 on the way to the database.

3. **`pooling=MEAN`** with `sif_coefficient=0.0001`. Recorded because the
   similarity distribution (§2.5) depends on it.

The run records `model_name`, `pca_dims`, `sif_coefficient`, and the distilled
artifact's content hash. `CHUNKGRAPH_MODEL_DIR` / `--model-dir` points at the
saved artifact.

### 2.4 The L2 normalization invariant (R2.1, and correctness of #4)

The requirements state: "`embedding` is L2-normalized, so `<#>` (negative inner
product) ranks identically to cosine and is cheaper." That equivalence is an
**ingest-side obligation**, and the installed API does not provide it:

```
StaticModel.encode(sentences, show_progress_bar=False, max_length=512,
                   batch_size=1024, use_multiprocessing=True,
                   multiprocessing_threshold=10000, **kwargs)
```

There is **no `normalize_embeddings` parameter**. Unlike `sentence-transformers`,
`model2vec` returns un-normalized vectors. Ingest therefore normalizes
explicitly, as `E / np.linalg.norm(E, axis=1, keepdims=True)`.

Order matters: **cast float16 → float32 first, then normalize.** Normalizing in
float16 and casting afterward leaves norms off unity by roughly the float16
epsilon, which is small but *systematic* and biased by vector magnitude.

This is a silent-failure risk, not a crash risk. If normalization is skipped,
`<#>` still returns rows, still ranks plausibly, and is simply *not* cosine — it
degenerates toward ranking by vector magnitude, which for SIF-weighted means
correlates with chunk length. The failure mode is "short chunks quietly never
retrieve," which no smoke test catches. It is therefore an explicit invariant:

> **INV-1.** Every `node_embedding` row satisfies `abs(norm(v) - 1.0) <= 1e-6`.

Checked at ingest across all rows before commit, and asserted by a test that
samples the persisted table. A run failing INV-1 is not written.

### 2.5 The distribution risk, stated rather than assumed

Static embeddings have no context: a chunk vector is a SIF-weighted mean of its
token vectors. Its similarity distribution is materially different from
contextual MiniLM's — pairwise cosine typically has a **higher baseline and a
compressed upper tail**, because two chunks sharing common vocabulary are pulled
together with no contextual signal to separate them.

This lands directly on stages the explorer does not own but depends on:

- **NORMAL** — Box-Cox per similarity distribution, with an estimator-pair gate.
- **EDGES** — k-sigma tail cut where `post_kurt <= KURT_OK`, else budget-match.

`SIM_FLOOR`, `KURT_OK`, `DISC_M`, `LAM`, `EPS`, and `DIV_WARN` have **never been
exercised against a dense space** — `brown-50` is sparse-only, so the dense
branch of these stages has no empirical history in this repo at all.

Consequently the re-ingest is a **measurement, not a step**. It records
`lam`, `post_kurt`, `divergence`, and `nonzero_frac` per space, and the run is
inspected before it is used as the explorer's backing data. Two named outcomes:

- **Box-Cox gate rejects, or `post_kurt > KURT_OK` forcing budget-match on the
  dense space.** Acceptable, but must be *observed and recorded* — it changes
  what R5.1 provenance means, since budget-matched dense edges are selected by a
  different rule than k-sigma sparse edges.
- **`nonzero_frac` collapses toward 1.0** (near-complete dense graph before
  thresholding). This is the specific failure the compressed tail predicts. It
  would mean the dense space adds no discrimination, and the correct response is
  to revisit `pca_dims`/`sif_coefficient` — **not** to tighten `SIM_FLOOR` until
  the graph looks right, which would be fitting the threshold to the artifact.

No design decision downstream of this is allowed to assume the dense space is
well-conditioned. R2.2/R2.3 (sparse-only fallback, stated honestly) remain live
paths after re-ingest, not legacy ones.

---

## 3. Capability detection and honest degradation (R2.2, R2.3, R2.4)

A single `Capability` value is computed once per run selection, in `store.py`:

```
dense_available = (embed_dim IS NOT NULL) AND (node_embedding row count > 0)
```

Both conditions are checked. `embed_dim` set with zero rows is exactly the
half-finished-ingest state, and the requirement names both disjuncts explicitly.

`retrieve.py` derives the retrieval mode from `Capability` — it does not accept a
mode argument. The UI *displays* the mode; it cannot *select* one. This makes
R2.3 and R2.4 structurally consistent: there is no code path that performs
sparse-only retrieval while reporting fusion, because the same value decides the
behavior and labels it.

Every result set carries its `mode` (`"fused"` / `"sparse-only"`) and, when
degraded, the reason (`embed_dim` NULL vs. zero embedding rows). Per R2.3 the
degraded case is a **normal result with a stated mode** — never an empty result
and never an error.

---

## 4. Retrieval path

### 4.1 Anchors

- **Fused (R2.1):** BM25 candidates and pgvector ANN candidates (`<#>`,
  cookbook #4) are each taken to depth `k_anchor`, then combined by rank
  fusion over the union. Rank fusion rather than score fusion, because BM25
  scores and inner products have no shared scale, and the Box-Cox parameters
  that would put them on one are per-run properties of the *edge* distribution,
  not the query-to-node distribution. Fusing raw scores here would invent a
  calibration the engine never established.
- **Sparse-only (R2.2):** BM25 candidates alone, `mode="sparse-only"`.

Each anchor retains its per-space contributions: BM25 score, dense similarity,
each space's rank, and the fused rank (R6.2).

### 4.2 Expansion

Cookbook #1: hop over `edge_sym` via the indexed `src`/`dst` integer columns
(R4.1), decaying by strength, keeping the best path per node, cycle-safe.

`attrs` is read for display payload only and never appears in a `JOIN` or
`WHERE` used for traversal (R4.2). `jsonb` is payload-only.

Hop depth is an explicit operator-controlled parameter with a bounded range,
displayed with the result (R4.3).

**The strength floor is a second bound, and it is the dominant one.** Cookbook #1
carries `w.score * s.strength > 0.05`. Measured on `brown-50`, seed 1476
(degree 28, the joint maximum):

| | nodes |
|---|---|
| true 2-hop reach | 180 |
| after the `> 0.05` floor | **59** |
| after the query's `LIMIT 25` | **25** |

The floor alone discards **121 of 180 nodes — 67%** — and it does so before the
R4.4 cap is ever consulted. Because score decays multiplicatively, a hop-2 node
needs roughly `strength >= 0.22` on both legs to survive; typical hop-1 strengths
here sit around 0.27–0.51, so the floor bites hardest exactly at hop 2. It is not
a safety valve, it is the traversal's actual reach limit.

**Decision: keep 0.05, disclose it.** The value is retained — it is what the
existing retrieval path does, and this spec composes rather than reinvents that
path — but it is promoted from a buried literal to an operator-visible parameter,
reported with every result alongside hop depth and the R4.4 cap. Three stacked
truncations (floor, `LIMIT`, cap) are acceptable only if all three are legible;
an undisclosed floor is precisely the silent truncation §6 forbids.

### 4.3 Communities (R3)

Retrieved nodes are joined to the run's stored `community` rows, off the GIN
index on `members` (cookbook #3), and grouped by stored `cid` (R3.3). Displayed
with the run's own `cid`, `keywords`, and `medoid` (R3.1).

No partitioning, clustering, or modularity computation occurs anywhere in the
explorer (R3.2) — subgraph Louvain is not a restriction of global Louvain, and
ids derived from an induced neighbourhood would not map to the run's.

**The explorer surfaces no local structure at all**, so R3.4's exploratory
labelling obligation is discharged by not incurring it. If that changes, R3.4
applies in full and this section must be revised rather than quietly extended.

### 4.4 Attribution (R6)

Every displayed chunk names its source `doc_id` (R6.1). The ranking signal
(R6.2) is shown decomposed, and distinguishes how a node entered the result:

- **anchor** — BM25 score, dense similarity, per-space ranks, fused rank.
- **expanded** — hop count, path strength, and the anchor it descends from.

A node reachable both ways is shown as an anchor with its expansion path noted.
Without this split, a high-ranking anchor and a strength-decayed distant
neighbour are indistinguishable in the result, which is the exact confusion R6.2
exists to prevent.

### 4.5 Provenance (R5)

Each displayed edge exposes its `{sparse,dense,both}` provenance (R5.1).

Per R5.2, the *run's* provenance distribution is computed once and stated. For
`brown-50` (4568/4568 `sparse`) the UI states that every edge carries a single
provenance value — it does not render a provenance breakdown that implies fusion
occurred. This is a run-level statement, not a per-result-set one, so it cannot
be made accidentally true by a query that happens to return only sparse edges.

---

## 5. `_gist_walk` equivalence — measured, not assumed

Cookbook #1 carries the comment `-- Mirrors _gist_walk`. **That comment is
false**, and this is settled by reading the two implementations, not by waiting
on a test:

| | `_gist_walk` (`chunkgraph.py:319`) | cookbook #1 |
|---|---|---|
| input | a candidate set `cand` already supplied, plus `seeds` and BM25 `scores` | a single seed `ord` |
| selection | threshold ladder `t *= (1+EPS)` over `D`, greedy pick by `scores` | recursive CTE, multiplicative strength decay |
| diversity | `f = sum(scores) + LAM*dv`, `dv` = min pairwise distance | none |
| feasibility | skips a threshold when `len(sel) < min(MINK, len(cand))` (R4) | none |
| result size | capped at `self.K` | `LIMIT 25` |

They are different algorithms with different objectives. `_gist_walk` is a
facility-dispersion selector that *deliberately spreads* its picks apart;
cookbook #1 is a best-path ranker that returns whatever decays least. Their only
shared property is that both restrict to nodes reachable within `H` hops.

The comment should be corrected in `cookbook/queries.sql`, which is in scope
(§7). It is load-bearing: it is the reason the requirements describe #1 as
mirroring the engine.

What remains genuinely open is not *whether* they differ but *how much* the
returned sets diverge in practice, which decides whether the difference is
tolerable for the explorer's purpose. That is what the test below measures.

**Differential test.** Over `brown-50` (sparse-only, 1789 nodes / 4568 edges),
for a fixed seed set spanning low-, median-, and high-degree nodes, at each hop
depth in range: run `chunkgraph.py`'s in-memory `_gist_walk` and the SQL k-hop,
then compare.

Comparison is specified carefully, because naive equality will produce false
failures:

- **Node sets** compared exactly as sets.
- **Per-node best-path strength** compared within tolerance (`1e-9` relative).
  numpy float64 products and Postgres `double precision` accumulate in different
  orders; exact equality is the wrong assertion.
- **Ties** compared as sets, never as ordered lists. Equal-strength paths have no
  defined order in either implementation, and asserting an order would lock in an
  accident of iteration.

**Outcomes are pre-committed**, so the test cannot be quietly weakened when it
goes red:

- **Agreement** → cookbook #1 is confirmed as the traversal, and the test becomes
  a regression guard against divergence.
- **Divergence** → this is a finding, not a bug to paper over. The explorer's
  stated contract is to compose the existing retrieval path, so the SQL is
  brought into line with `_gist_walk`, *or* the difference is documented as a
  deliberate, named behavioral difference. Silently shipping a UI whose traversal
  disagrees with the engine is the failure this test exists to prevent.

The engine is not modified under this spec without that being an explicit,
approved task with its own scope entry.

---

## 6. Neighbourhood cap (R4.4)

R4.4's bound is deliberately unspecified in requirements because it depends on
the real degree distribution, and backbone KNN (R7) *guarantees* hubs exist —
every node receives its top-KNN neighbours unioned in, so degree has a hard floor
and a long right tail by construction.

Picking a constant now would be picking it blind. The design specifies the
**rule**; the value is measured.

**Rule.** The cap is applied **per hop frontier**, not to the final result. A
global cap truncates whichever nodes happen to be enumerated last, which biases
toward the first anchor expanded. A per-frontier cap keeps expansion balanced
across anchors and keeps the bound meaningful at every depth.

When a frontier exceeds the cap, it is **ordered by edge strength and truncated**
— retaining the strongest neighbours, which matches the strength-decay semantics
of the traversal itself rather than cutting arbitrarily.

**Value: 120.** Measured, not estimated. Two distributions were computed over
`brown-50`'s live edges (4568 edges, 1789 nodes):

| distribution | min | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| node degree (1-hop) | 3 | 4 | 9 | 20 | **28** |
| 2-hop reach | 5 | 20 | 45 | **111** | 191 |

**120 sits just above the 2-hop p99 of 111**, so it bounds the hub tail while
leaving the typical case (p50 = 20, p90 = 45) untouched.

An earlier draft of this section floated "100–150" against the *degree*
distribution. That was the wrong distribution: **maximum degree is 28**, so any
cap in that range applied to a 1-hop frontier could never bind at all. The rule
is per-frontier, and it is the hop-2 frontier that grows — sizing the cap
requires the reach distribution, not the degree distribution. Recorded because
the near-miss is instructive: the number was accidentally reasonable while the
reasoning behind it was wrong.

Both figures are properties of *this* corpus at this scale and are re-measured,
not assumed, when the cap is applied to a materially different run — in
particular after the §2 dense re-ingest, which adds a second edge space and will
shift both distributions.

**Truncation is never silent.** When a frontier is capped, the result states that
the neighbourhood was bounded and how many nodes were withheld. A silently
truncated neighbourhood is indistinguishable from a sparse one, which would
undermine R6.2's premise that a result can be checked without trusting the
ranking. This is the same honesty obligation R2.3 imposes on degraded retrieval.

The cap is operator-visible alongside hop depth (R4.3).

---

## 7. Scope

Files this spec authorizes:

```
explorer/store.py       explorer/retrieve.py       explorer/app.py
ingest_brown.py         sql/001_schema.sql         cookbook/queries.sql
tests/*.py
```

`chunkgraph.py` and `pg_store.py` are **read but not modified** under this
design. If §5 finds divergence requiring an engine change, that is a separate
approved task, not an implicit licence granted here.

## 8. Prerequisites before UI work

Both are gates on trusting the explorer, not preliminaries to rush:

1. **Dense re-ingest** (§2) — distill, pin parameters, verify INV-1, record the
   distribution diagnostics of §2.5 and inspect them.
2. **Equivalence test** (§5) — establish whether cookbook #1 reproduces
   `_gist_walk` before building a UI on it.

Both run against persisted data and neither requires the UI to exist.

## 9. Requirements coverage

| Req | Addressed |
|---|---|
| 1.1 | §1.1 read-only session, `ReadOnlySqlTransaction` |
| 1.2 | §1.2 explicit label, no default run |
| 1.3 | §1.2 live-run resolution, identity always displayed |
| 2.1 | §4.1 rank fusion of BM25 + pgvector ANN |
| 2.2 | §3 both-disjunct capability check |
| 2.3 | §3 mode + reason on every result; normal result, not error |
| 2.4 | §3 mode derived from capability, not selectable |
| 3.1 | §4.3 stored `cid`/`keywords`/`medoid` |
| 3.2 | §4.3 no partitioning anywhere |
| 3.3 | §4.3 grouped by stored `cid` |
| 3.4 | §4.3 no local structure surfaced |
| 4.1 | §4.2 indexed `src`/`dst` traversal |
| 4.2 | §4.2 `attrs` display-only |
| 4.3 | §4.2 bounded, visible, operator-controlled depth + disclosed `0.05` strength floor |
| 4.4 | §6 per-frontier strength-ordered cap = 120, announced; all three truncations legible |
| 5.1 | §4.5 per-edge provenance |
| 5.2 | §4.5 run-level single-provenance statement |
| 6.1 | §4.4 `doc_id` on every chunk |
| 6.2 | §4.4 decomposed signal, anchor vs expanded |

Non-goals hold: no labelling (R1 read-only), no LLM in retrieval, no model
selection from the UI — §2's re-ingest is an offline operator action with pinned
parameters, not a UI affordance.

## 6. Walk tab: prompt in, communities out (2026-08-29)

**Requirements layer, stated plainly.** A human types a prompt and gets back
the Louvain communities the walk landed in, ranked by how much of the walk
each one holds. They do NOT get a hairball of chunk ids. A chunk graph is the
evidence layer; the community layer is what a person can read.

### What is shown, top to bottom

1. **Community graph.** Nodes are the communities present in the walk, sized
   by presence, labelled with their top-3 terms, spring-laid-out. Edges are the
   walked edges that cross communities, width = count. Chunks are never drawn
   here. The graph is read by its terms.
2. **Community list, most present first.** Per community: presence
   (retrieved chunks / community size), top-3 terms, two medoids.
3. **Expand a community** to see the explicit evidence: the retrieved chunks
   that fall in it, with provenance and doc id.
4. **In-between exemplars.** Retrieved chunks with walked edges into a
   different retrieved community, ranked by how many.

### The two term layers, and why the top-3 is scored globally

The top-3 terms are scored over **every chunk in the community**, not the
retrieved ones. That is the implied evidence: the concept the community holds
as a whole, which the walk has touched. The retrieved chunks under the expander
are the explicit evidence. Showing both is the point; showing only the second
is a search result, not an explanation.

Scoring is BM25 with the community as the document -- term frequency summed
over members, community token length as the document length, idf over the set
of communities. The stored `keywords` column is tf*idf over the same scope and
stays as the fallback; BM25 corrects for the size skew (communities run 8 to
189 members) that tf*idf does not.

### Two medoids

- **Local**: among the retrieved chunks in this community, the one with the
  largest summed walked-edge strength to the others. Central to what THIS walk
  found here.
- **Global**: the stored `community.medoid` -- central to the whole community.
  Already computed at ingest.

Both are argmax-of-centrality picks. If more than one exemplar per community is
wanted later, the extension is MMR over the same similarity, not a different
centre.

### A constraint stated so it is not rediscovered

In this schema every chunk belongs to exactly one community (`community.members`
partitions the node set). "Chunks are not mutually exclusive across communities"
becomes true only when the term-node layer lands (graph-term-selection spec,
approved, unbuilt) and communities are read through terms. Until then,
"in-between" is defined by EDGES: a retrieved chunk whose walked edges reach a
different retrieved community. That is measurable today and is the honest
substitute.

### Out of scope here

Re-partitioning (never at query time), model-authored labels (draft only,
italic, never a key), and the term-node layer itself.

### 6.1 Query-conditioned terms (amendment, 2026-08-29)

The unsupervised top-3 names the community; it does not name the community
*as the prompt sees it*. For "what feelings are associated with betrayal" the
walk's largest community reads `taliesin / olgivanna / wright` -- the Wright
divorce scandal -- which is right and useless at once.

**Rule.** Communities stay unsupervised. The three words shown for each are
chosen from that community's OWN vocabulary (its BM25 top-40), re-ranked by
relevance to the prompt. Query conditioning re-ranks; it never imports a term
the community does not carry (W10).

**Two signals, in priority order.**
1. Lexical: a candidate that is a query term (or a phrase part of one) is
   forced to the top. Exact, cheap, always available.
2. Dense: cosine between the model2vec embedding of the prompt and of each
   candidate term. Static token table, so single-term embeddings are exact and
   deterministic. Requires the run's model directory (`CHUNKGRAPH_MODEL_DIR`);
   absent, the ranking falls back to lexical-then-unsupervised (R5 posture).

Measured on c4 for the betrayal prompt, top-40 re-ranked by cosine:
`charm .32, fortunately .25, joy .24, attachments .23, horrible .21, angel .16`
against `wright -.04, constable -.05, reporters -.06, mrs_wright -.09`.

**What is shown.** The query-conditioned three label the graph node and the
list row. The unsupervised three stay inside the expander as "community
concept", so a reader can see both what the community IS and which part of it
the prompt touched.

**Deferred.** `model_dir` is not stamped into `graph_run.params`; steering says
a run should be reproducible from stored params and the embedder is a param.
Playbook TODO.

### 6.2 The local medoid is query-weighted (amendment, 2026-08-29)

On the betrayal prompt the local and global medoid of the top community were
the same chunk (#797). Not a bug in the arithmetic: 32 of 38 members were
retrieved, so structural centrality over the retrieved set IS the global
centrality. But it defeats the purpose -- the local medoid is supposed to be
central to what the prompt activated, and a prompt does not activate a
community uniformly.

**Rule (W11).** Local centrality weights each retrieved neighbour by its walk
score: `centrality(o) = sum_j score(j) * strength(o, j)`. The walk score is
already query-conditioned (BM25 anchor times decayed path), so the medoid it
produces is the supervised one. The global medoid stays the stored
`community.medoid`. They may still coincide when the prompt genuinely lands on
the community's centre; they no longer coincide by construction.

### 6.3 Chunks never split inside a word (R16, chunkgraph.py)

Two sources, both fixed. The chunker jumped from blank-line paragraphs
straight to a word window over the whole paragraph, so chunks began on stray
punctuation tokens and ignored the sentence lines `ingest_brown` already
writes; it is now recursive (blank lines, then lines packed to `target`, then
a word window only for a single overlong line). And the UI truncated previews
at a character count; it now clips at the last word boundary and marks the cut.
The chunker change re-cuts the corpus, so both live runs are re-ingested and
the tests pinned to `brown-50`'s shape are re-pinned from the new run.

### 6.4 Interpretation layer: a local model reads the walk (2026-08-29)

**Where it sits.** SERVE side, after the walk. Nothing in the construction
path changes: chunking, edges, communities, terms, medoids and the walk itself
are computed before any model is called (sampler S6: the bundle is frozen
first). The model reads a rendered bundle and writes prose. It is the third
model-authored layer after draft labels and it carries the same status --
a draft, marked as such, never a key, never an input to anything upstream.

**What the model sees**, rendered deterministically by `interpret.render_bundle`:
the prompt; each retrieved community ranked by presence with its
query-conditioned terms, its unsupervised concept, and both medoids; the
retrieved chunks under each, as `#ord · doc` plus text; the in-between
exemplars. Same bundle, same text, every time.

**What it is asked to do.** Answer the prompt from the evidence and nothing
else, citing `#ord` for every claim, naming which communities carry the
answer and which are noise, and saying so when the evidence does not answer.

**Citation contract (I1).** Every `#ord` in the output is checked against the
bundle. A citation outside the bundle is reported as foreign and shown to the
reader; the answer is not silently trusted. This is the same falsifiability
the labels layer has -- a human can check every claim against a chunk they
can open.

**Context (I2).** The request sets `num_ctx` explicitly. Ollama's default
(2048-4096) truncates a 64-chunk bundle before the model sees the end of it and
answers from the fragment with finish_reason "stop", which looks like a bad
answer rather than a truncated one. Measured elsewhere on this machine; not
re-measured here because it is the known failure mode.

**Model.** `qwen3.5-oc:4b` via the local Ollama, reusing `label_communities`'s
host resolution. `<think>` blocks are stripped from the reply. Absent Ollama,
the pane says so and the walk is unchanged (R5 posture).

### 6.4a Interpretation: OpenRouter, entailment, optional rerank (amendment)

**Backend (I5).** `qwen/qwen3.5-9b` via OpenRouter (OpenAI-compatible chat
completions, `OPENROUTER_API_KEY`). Local Ollama is the fallback when the key
is absent, with thinking disabled: measured on `qwen3.5-oc:4b`, the model spent
the whole generation budget in `<think>` and returned empty content with no
error -- a success that answers nothing. Which backend answered is recorded on
the result.

**Job (I6).** Not free prose. The model classifies each retrieved chunk against
the prompt -- ENTAILS / CONTRADICTS / NEUTRAL -- and then answers using only
the entailed set, citing `#ord`. Output is JSON. This is the falsifiable
version of "interpret": every verdict is a claim about one chunk a reader can
open. Neutral is the expected majority; a walk that retrieves 64 chunks does
not have 64 answers.

**Citation contract (I1, strengthened).** Verdict ords and answer citations
are both checked against the bundle. Foreign ords are surfaced; a citation in
the answer that is not among the ENTAILED set is surfaced too, because that is
the model contradicting itself.

**Rerank stage (I7, optional, OFF).** ColBERTv2 MaxSim over the retrieved
chunks against the prompt, keeping the top-N in rank order, BEFORE the model
sees them. Cuts the bundle to what late interaction says is relevant and cuts
context cost with it. `pylate` is installed; no ColBERT checkpoint is cached,
so the stage is a hook that engages only when one is (`RERANK_MODEL`).
Downloading a checkpoint is a deliberate act, not a side effect of a query.


### 6.5 Chunks sized by the corpus, not by a constant (R17, 2026-08-29)

R16 fixed the boundary (never inside a word, never mid-line) but kept the size
as `target=120, max_len=200` -- constants nothing measured. The
`reduce_overlaps` skill's method replaces them:

    unit   = lines per blank-line paragraph (chars per line if no blank lines)
    Box-Cox the unit counts; m = median, d = MAD in transformed space
    hi     = m + 2d, inverted to natural scale
    merge  paragraphs < m forward until they reach m
    split  paragraphs > hi into DISJOINT windows of m (stride m)
    assert every source line lands in exactly one chunk

Measured over all 500 Brown documents, 15,667 paragraphs:

    lines/paragraph   p50 3   p95 9   max 124   lambda -0.115   m 3   hi 7.2
    words/paragraph   p50 55  p95 202 max 1820  lambda  0.136   m 55  hi 158
    chars/line        p50 92  p95 242 max 1026  lambda  0.370   m 92  hi 219
    single-line paragraphs 3,401 (22%) -> merge up;  above hi 1,295 (8%) -> split

Two deliberate departures from the recipe as first stated:

- **No overlap, no de-overlap.** The skill's own catch-22 section shows that
  windowing at stride m-d and then de-overlapping collapses back to disjoint
  atoms. For a retrieval graph the intermediate state is worse than useless:
  an overlapping chunk is a near-duplicate node and its edges are artefacts.
  Atoms are built disjoint (stride m) and verified by conservation.
- **Not difflib.** The skill is explicit: overlap a known stride introduces is
  character-identical, so exact suffix/prefix matching is correct and O(n);
  fuzzy matching silently trims text that merely resembles its neighbour. With
  no overlap introduced there is nothing to trim at all.

Stats are computed over the corpus handed to `fit()`, not per document -- a
20-paragraph document does not have a stable median. The derived parameters
are stored with the run so a re-ingest is reproducible from params alone.

**Scaling note.** All 500 Brown documents is the stated goal. Under R16, 50
documents gave 1,668 chunks, so 500 is on the order of 15k -- inside R10's
18k memory wall for the sparse product but against dense n x n `strength`
and `D` matrices that R10 does not cover. The chunk count under R17 is
measured before any 500-document ingest is attempted.


### 6.5a Correction: the unit is the document (2026-08-29)

§6.5 applied the recipe at paragraph level (lines per paragraph, m=3, hi=7)
and produced 15,240 atoms at 500 documents. That was a misreading. The
operator's rule counts newlines per DOCUMENT: a document is a node unless it
is an outlier by the corpus's own Box-Cox threshold, in which case it splits
at paragraph boundaries into windows of hi lines and a short tail merges back.

Measured on all 500 Brown documents: lines/document m=107, hi=153, lambda
-0.441; 12.6% of documents exceed hi. Every one of them still ends as a single
chunk, because the longest document is 240 lines and a split survives the
tail-merge only from hi + m = 260 lines. So 500 documents -> 500 nodes, each
~2,300 words. Conservation 57,340 / 57,340.

Consequences downstream, stated so they are not rediscovered:

- Memory ceases to be a question: four dense n x n matrices at n=500 are
  8 MB, not 7.4 GB. The full corpus ingests in seconds.
- ef=64 now retrieves 64 of 500 documents (13% of the corpus) per walk. The
  ef sweep was measured on 1,789 chunks; it will be re-measured on this graph.
- The entailment judge sees the first 420 characters of each retrieved
  document, not the document. That bound is what keeps 64 x 2,300 words
  inside a context window; it also means the judge is reading openings. The
  rerank stage (I7) and per-document evidence selection are the answer, not a
  bigger MAX_CHUNK_CHARS.
- Louvain now partitions documents; communities are groups of documents and
  the query-conditioned terms are scored over whole documents' tf maps.

The paragraph-level chunker is gone, not kept behind a flag: two chunkers is
sprawl, and R17 names the unit.


### 6.4b Interpretation at document level: what it took (2026-08-29)

Four things, each measured, each now a guard in `interpret.py`:

1. **Transport (I5).** The httpx client died on a TLS handshake timeout that
   the endpoint did not reproduce a minute later. This machine runs Avast's
   web shield (`SSLKEYLOGFILE=\.swMonFltProxy`, `NODE_EXTRA_CA_CERTS`
   pointing at its cert); httpx carries its own CA bundle and stalls in that
   proxy, stdlib `urllib` uses the Windows store and does not. `rl_V2`'s
   client already had this shape -- urllib, three attempts, exponential
   backoff, retry on anything but 400/401/403, empty content retried -- and
   is now the shape here too. Measured from the walker's own environment:
   4.2 s round trip.
2. **The judge was reading openings (I8).** With 2,300-word document nodes
   clipped at 420 chars, 63 of 64 retrieved documents were judged neutral on
   "how were Morocco's first elections organized" -- a question cj37 is an
   essay about. Given the whole document the same model answered ENTAILS
   with the registration / nomination / voting / scrutin uninominal facts.
   Excerpting by the prompt fixed the wrong half first: the prompt terms
   (morocco, first, elections) are densest in the introduction, and the
   paragraphs that answer HOW say registration, districts, voting -- zero
   lexical overlap with "organized"; the static embedder ranks the
   introduction first on cosine too. Two levers that did work: cap each
   paragraph at a third of the budget (uncapped, one 1,100-char paragraph
   filled the 1,500 budget alone), and give ANCHORS -- the BM25 top-k the
   walk started from -- four times the budget, since that is where the
   answer most likely is. cj37 then judged ENTAILS.
3. **Verdict semantics (I6).** "ENTAILS an answer" plus "most are neutral,
   do not inflate" produced "discusses elections but not Morocco's first"
   on a paragraph that says "electoral planning in Morocco ... the first
   elections". ENTAILS now means contains information that answers or partly
   answers; the bias line is gone; duplicate verdicts for one id (73 for 64)
   collapse to the first.
4. **Two stages.** After 60 verdicts the model returned an empty answer with
   one chunk judged entailing. A second, smaller call over the entailed
   excerpts only produces the answer, and its citations are checked against
   the entailed set. Result on the Morocco question: one entailed document,
   five cited sentences, all #330, zero foreign, zero self-contradiction.

Also this round: `brown-500-dual` (500 nodes, 5,112 edges, 7 communities) is
the fixture for every dual-run test -- ef=64 cannot return 64 results on a
51-node run. The sampler's top-community stability claim is pinned on it and
on the documented window n <= 24; on 51 documents in 5 communities the top-1
flips even at n=8..24, which is granularity, not a defect. Render fixtures
derive their hub from the loaded run instead of a constant that broke on the
next ingest.

### 6.6 Reason over community evidence (2026-08-29)

**Why this and not more chunks.** The judge reads retrieved chunks one by one
and says which entail. That answers "is there evidence" but not "what is the
claim, and does the corpus support it". GraphRAG's key piece is a
representative unit per community -- not a node, EVIDENCE: the medoid chunk.
This stage reasons over those units.

**The community brief**, one per community the walk landed in, ranked by
presence, built deterministically before any model call (S6):

    cid, presence (retrieved / size)
    terms as the prompt sees them      (query_terms)      supervised
    terms the community holds          (community_terms)  unsupervised
    local medoid  -- chunk most central to what the WALK found here,
                     weighted by walk score              supervised
    global medoid -- chunk most central to the WHOLE community
                                                          unsupervised
    each medoid rendered as a prompt-conditioned excerpt (I8) tagged [id=n]

Two medoids because they answer different questions: the global one says what
the community is about regardless of the prompt; the local one says which face
of it the prompt activated. When they coincide the prompt landed on the
community's centre; when they differ, the difference is informative.

**Four stages, each a JSON call, each checked (I9-I11):**

1. HYPOTHESIS -- from the prompt and the briefs, propose up to three candidate
   answers as falsifiable statements; pick one to pursue and say why.
2. PREMISES -- the salient premises the chosen hypothesis needs, each naming
   the brief ids ([id=n]) that would support it. A premise citing no id is
   kept but marked unsupported-by-construction.
3. EVALUATE -- for each premise, read the cited excerpts (anchor budget) and
   return supports / contradicts / insufficient with a one-line reason.
4. ANSWER -- the final response to the prompt, built only from premises
   judged supports, citing #id. Contradicted premises are reported, not hidden.

**Guards.**
- I9  Every id the model names, at every stage, is checked against the brief
      ids. Foreign ids are surfaced and discarded.
- I10 The answer cites only ids attached to premises judged supports; any
      other citation is self-contradiction and is surfaced.
- I11 Stage outputs are kept verbatim so the whole chain -- hypothesis,
      premises, verdicts, answer -- is inspectable and falsifiable chunk by
      chunk. A reader can disagree with a verdict by opening the chunk.

**What the model never does.** Change membership, terms or medoids. Those are
computed; it reads them. This is the determinism boundary applied to reasoning:
the substrate is fixed, the model chooses only what to claim about it.

**Cost.** Four calls; briefs are compact (~10 communities x 2 excerpts). Cheaper
than the per-chunk judge, and complementary: the judge asks "which chunks",
this asks "what claim, and does the evidence hold it".

### 6.7 Answer first; the evidence is one collapsed expander (2026-08-29)

The page order is: prompt; the model's answer (Reason, and/or the Judge's
entailed answer) directly beneath it once it arrives; then ONE expander,
"Evidence", holding the community graph, the query-term panel, the ranked
communities with both medoids, and the in-between exemplars. The expander is
open until an answer exists and collapsed after -- the answer pushes the
evidence down, it does not bury it. Streamlit does not nest expanders, so the
per-community expanders became sections inside the one; the model-transparency
expanders (briefs, raw stage replies, what the judge was shown) sit under the
answer they explain, because that is what they are evidence FOR.

### 6.8 A medoid is titled by its own salient terms (W14)

A medoid card reads as the chunk's top salient terms -- BM25 of the chunk's
own terms against the corpus, the chunk as document -- with `#ord · doc`
demoted to small print and the kept salient list plus a prompt-conditioned
excerpt beneath. Corpus df is computed once per run and cached (one GIN probe
per term was measured at >120 s for three documents).

Which terms count as salient is a GATE, not a top-k: log-normalise the BM25
scores and keep all at or above `min(median - 1.4826*MAD, mean - sd)`, the
more permissive of the robust and parametric one-sigma cuts (trigram.md's dual
measure). Measured on brown-500-dual: 440/523, 578/688, 489/588 kept -- about
84% -- and 95% on a synthetic lognormal; threshold lands in decile 3. That is
more than "the upper half and a little more"; the formula is implemented as
stated and the number recorded so it can be tuned against a probe set rather
than by feel.

Stopwords: one list, NLTK English plus the local extras, applied at ingest
(R18) and imported for queries. `didn't` -> `didn` had surfaced as a community
term. Register words (got, knew, looked, eyes) are not stopwords; they are the
keyness prior's job.

### 6.9 Latency: one call's worth (2026-08-29)

"Should only be as long as a single OpenRouter call." Measured before, on the
desegregation prompt against brown-500-dual: walk 14.4 s, terms 9.2 s, four
sequential reasoning calls ~25-62 s -- 50-86 s total. Four causes, each fixed
at its root, none of them the model:

| stage        | before  | after  | what it was                                         |
|--------------|---------|--------|-----------------------------------------------------|
| search()     | 7.3 s x2| 4 ms   | BM25 as SQL over jsonb; now Python over cached postings, one search per walk |
| terms        | 9.2 s   | 0.4 s  | community pool recomputed per prompt; it is prompt-independent -- cached per run |
| import       | 11.6 s  | 0.5 s  | `from nltk.corpus import stopwords` pulls the whole NLTK tree; stoplist.py reads the file |
| reason       | 4 calls | 1 call | one JSON with all four sections; I9/I10/I11 apply unchanged; OpenRouter routed by throughput |

Total 9.0 s: walk 1.0, terms 0.4, one call 7.7. Runs are immutable, so the
per-run index and pool are pickled under ~/.cache/chunkgraph keyed by run_id;
a fresh process reloads in 0.4 s instead of rebuilding in 26 s. The staged
four-call path remains reachable (`one_shot=False`) for strict stage isolation
and is still pinned by tests.

### 6.10 One degree out from the top chunks; no trailing disclaimer (2026-08-30)

Two operator observations on the same run. The answer ended "the evidence does
not settle the specific demographic breakdown ...", which reads as if a kind
of evidence had been wanted and missed. It had not: the walk stops by the
tuned HNSW rule (6.9, S10), which is the sufficiency decision, so the model has
no standing to add one. The clause "say what the evidence does not settle" is
removed from both answer prompts; what a chunk does and does not say remains
visible per premise and per verdict.

The second: extend the evidence one degree out without re-walking. The walk
already ranks W by score; the cheapest extension is the strongest edges of the
top-`ring_top` chunks, added at `score = parent score x edge strength`, capped
at `ring_per` per parent, only if not already in W. Deterministic, edge-table
only (W2), no model. The Bundle grows; `params["ring"]` records how many were
added and the Evidence expander shows it. Defaults `ring_top=3, ring_per=8`
(<= 24 extra chunks, well inside the Judge's evidence cap). The ring is
evidence, not anchors: it never seeds a further expansion.

Guards: S13 -- every ring member is a direct edge_sym neighbour of a top-`ring_top`
member of W and was not in W; ring score never exceeds its parent's.


### 6.11 Evidence pathways: critical connectedness between ideas (2026-08-31)

Operator intent: measure how strongly the IDEAS a walk touched are connected,
not just which chunks scored. The earlier form of the idea was significant
correlations in a score-to-score correlation matrix over chunks; this is the
graph-native version: the matrix rows/columns are ANCHOR nodes ("ideas" --
Louvain global medoids, walk-local medoids, top-scored chunks, or any ordinal
the caller names), and the entries are degree-weighted path counts (DWPC,
damping 0.4) over the subgraph induced by the walk plus its one-degree ring.
Hub correction is the point: a pair connected by several paths through
specific vocabulary outranks a pair connected once through a hub.

Whole-subgraph shape comes with it, because a pathway only means something
inside a shape: number of components and the largest component's share,
density, and conductance of the retrieved set against the rest of the run
(how leaky the evidence neighbourhood is). All computed, no model (S6).

Path score = product of edge strengths x product of deg(v)^-0.4 over every
node on the path, global degree, simple paths up to 3 edges. Each pair keeps
its best path for display: a chain of chunk ids the operator can read as
"how idea A reaches idea B". Pairs are ordered by DWPC.

Guards: W15 in graph_tools. UI: a "Pathways between ideas" block in the
Evidence expander, each pair titled by the anchors' salient terms.


### 6.12 The model sees the structure; reason+judge is one call (2026-08-31)

Operator: "this information is very informative for the model ... seeing is
believing" and "reasoning and judging the walk should be a serialized call ...
a single llm call with a follow on judge key".

Structural evidence (I12). The pathways/shape computed for the operator now
reaches the model, two renderings of the SAME pathways object: an image
(default) -- left the walked subgraph, nodes coloured by community, strongest
DWPC chains in red, lexical anchors ringed; right the global community map
with walked communities filled, hits/size annotated -- or a numbers block
(`structure="text"`). Image build or transport failure falls back to the
numbers block, recorded in `structure_note`. The Evidence expander gained the
same split as tabs: "This walk" vs "Global map" (the identical PNG the model
receives). Scale note: the global panel is the community quotient (k nodes,
not n), so it survives corpus growth; the subgraph panel is bounded by ef.

One call (I13). `reason(judge=True)` extends the one-shot JSON with a
"verdicts" key: the same reply carries hypotheses/premises/evaluations/answer
over the briefs AND one entailment verdict per retrieved chunk over the full
rendered evidence. Each channel is checked exactly as before (I1/I9/I10);
max_tokens raised to 8192 for the combined reply. The walker's two buttons
collapsed into one "Reason + judge this walk"; the disagreement note now
compares the two channels of a single result.

Digest ids resolve to text (I12 amendment, 2026-09-03). Measured basis:
interpret.reason cited 0 digest ids on prompt B3 ("a quote about books and
reading") when chunk_chains/pathways/bindings rows carried bare ordinals --
the model had no text to anchor a citation to. render_digest now accepts an
optional `resolver: dict[int, str] | None` (ord -> "source:top_term") from the
caller; when supplied, every place a bare chunk ordinal appears in
chunk_chains, pathways, and bindings rows is rendered as "id=src:term" instead
of the bare int. term_chains is unaffected (already textual). Absent a
resolver, output is byte-identical to pre-amendment (bare ordinals) -- old
callers do not break. walker_app._dendrite_state already computes
chunk_salient(kept, k=3) for every kept chunk (chunk plane term pool, line
~333); the walk_state cid_of/source_of lookups already exist at the call site
(walker_app.py ~line 545-560). The resolver is built there, once, from data
already in memory -- no second salient pass, no new DB round-trip.

Premises must cite the digest or say so (I13 amendment, 2026-09-03). Measured
basis: reason(judge=True, digest=...) returned every premise with ids:[] ("no
evidence cited") on prompt B3 even though the digest was appended to the same
prompt text -- the grounding contract (I9: cite ids, empty list if none apply)
already existed, the model was simply never told the digest counts as citable
evidence. ONE_SHOT_SYSTEM/PREM_SYSTEM now name the digest explicitly: a
citable id is any brief [id=<n>], any chunk id appearing in the digest's
chunk_chains/pathways/bindings rows (cite the NUMBER before "=" when the
resolver form "n=source:term" is shown), or a judge-shown chunk id when
judge=True. A premise that cannot be grounded in any of those must return an
empty ids list -- the existing parse already renders that as verdict
"unsupported" / why "no evidence cited" (I9), so no parse change is needed for
that half of the contract. No new model call, no new JSON field, no second
button.

Post-fix measurement (T13, 2026-09-03, run mixed-full-dual, backend
openrouter:qwen/qwen3.5-9b, .tmp/reason_digest_test.py): SPLIT verdict, and
the split is the finding. E3 ("the senator fought a losing battle over the tax
bill" -- a propositional prompt) went from nothing to fully grounded: one
premise citing id 7 (brown, pfaff/barnett/sales_tax -- the same chunk the DWPC
pathway layer ranks #1 for this walk), verdict supports, non-empty answer with
a #7 citation, 87 judge verdicts. B3 ("a quote about books and reading" -- a
retrieval request, not a proposition) still cites 0 ids: the model decomposes
the REQUEST into meta-criteria ("there exists a text containing a direct
quotation") which it then correctly reports as ungrounded; evaluation null,
answer empty, judge unaffected (49 verdicts, within +-10% of the pre-fix 49).
Acceptance (>=3 distinct ids on BOTH rows) NOT met. The failure moved: before
the amendments the digest was ignored wholesale; now grounding works exactly
when the hypothesis stage produces a falsifiable proposition, and B-class
prompts defeat the hypothesis stage itself (the "hypothesis" degenerates to
the prompt verbatim). Next lever, if opened: hypothesis formation for
retrieval-shaped prompts (e.g. instruct that for "find me X" prompts the
hypotheses are candidate X's drawn from the evidence, cited by id) -- an I-lane
prompt change, not a digest or parse change. Recorded per T13's
no-retuning law; the 9b model's instruction-following is a confound a stronger
INTERPRET_MODEL would isolate.

Hypotheses for find-me-X prompts are candidates from the evidence (I14
amendment, 2026-09-03, T16 -- the lever T13 recorded, now taken). WHEN the
PROMPT is a retrieval request ("a quote about X", "an example of Y") rather
than a factual assertion, HYP_SYSTEM/ONE_SHOT_SYSTEM SHALL instruct that each
hypothesis is a SPECIFIC candidate drawn from the shown evidence -- the actual
quotation or passage, named by its [id] -- never a restatement or
decomposition of the request itself. Premises then cite the ids holding the
candidate. Propositional prompts are unaffected (E3's grounded behavior is
the do-no-harm check).

Every judged walk lands in the mirror (2026-09-03, T17). WHEN the walker's
"Reason + judge" runs, the walk and its digest SHALL be persisted to the neo4j
mirror via export_neo4j.write_walk + write_digest (Walk/ANCHORS/PATHWAY,
NEXT_IN_CHAIN, Chunk.salient, CommunitySummary/TOUCHED) -- best-effort:
gated by NEO4J_MIRROR (default on), a transport failure surfaces as a visible
warning and never blocks the answer. Rationale: T8/T14 shipped the writers
with no caller on the serve path, so the mirror held only test data; the
evidence layer is aligned with neo4j only if every walk the operator judges
is queryable there (cookbook/evidence_queries.cypher is the read side).

Community references (c<cid>) and pathway pairs named in the digest describe
structure, not a single citable excerpt, and stay outside the numeric `ids`
schema -- widening `ids` to accept them would need a second id namespace with
no consumer (I11's foreign-id bookkeeping, the answer's #<id> citation check,
and the UI's [id=<n>] rendering all assume one flat integer ord space).
Digest-cited CHUNK ids, by contrast, are already ordinals in that same space,
so they need no new schema -- only the valid-id set has to include them
(below).

Valid-id decision: no widening required in the judge=True path -- digest chunk
ids are always drawn from ds_["kept"], a subset of bundle.sampled (walker_app
builds the digest from dendrite_state(_bnd=bnd) over bnd.sampled), and
judge=True already unions `valid` with the full judge_shown = bundle.sampled
(existing behaviour, pinned by test_i13_combined_call_is_one_call_with_both_
channels). The one gap is digest passed WITHOUT judge (judge=False, digest
set): reason() now also unions `valid` with bundle.sampled whenever digest is
non-empty, regardless of judge, so a premise citing a legitimate digest-only
id is never silently dropped as foreign in that path either. Cost: one set
union, no extra call.


### 6.13 Bridge discovery: the whole graph may complete a pathway (2026-08-31)

Scoring stays on the subgraph (6.11); discovery may search everything. For
each pair among the walk's top-3 chunks, gt.best_path (W16) runs Dijkstra over
the entire live edge set maximising prod(strength) x prod(deg^-0.4 over
interior nodes) -- the DWPC weight of a single path. Chunks the winning path
crosses that the walk never retrieved enter the Bundle as origin="bridge" at
score = min(endpoint scores) x path weight, capped at 8 per walk (S14).

Presentation contract (operator): **bold #id** marks a bridge chunk wherever
evidence ids are listed; its *salient terms are italicised* in the Discovered
bridges block; a key line above the evidence explains the markup. The Bundle
carries origin (walk | ring | bridge) so any renderer can honour this.

6.12 addendum (2026-08-31, first live run): with judge=True the model is shown
every retrieved chunk, so the Reason channel's citable set widens from the
brief ids to ALL shown ids -- the first run discarded two on-screen citations
(#98, #163) as "foreign" and then flagged the answer for citing them. The
judge addendum also regained the standalone Judge's calibration ("a partial
answer is entails"; judge against the PROMPT, not the chosen hypothesis) after
the combined run returned 0 entails on evidence the split runs had entailed;
and premises must be stated in the model's own words, not pasted quotes.

### 6.14 Multi-source ingest: three corpora, one graph (2026-08-31)

Operator intent: Brown alone is one register. Add HuggingFace
`abirate/english_quotes` (~2.5k one-line quotations) and
`EleutherAI/wikitext_document_level` (~29k long articles) so one retrieval graph
spans aphorism, encyclopedia, and 1960s American prose. The sources ride the
SAME pipeline — no second spec dir, no second chunker, no retrieval quotas.

**1. The problem is scale mismatch, in two places.** A quote is one line; a
wikitext article is hundreds. R17 derives ONE `hi` from ONE Box-Cox over the
corpus; pooled across these three, the fit is dominated by whichever source
brings the most documents, so quotes never split (correct, but by accident) and
wikitext splits at a threshold set by Brown. Same mismatch downstream: pooled
similarity distributions are dominated by the largest source-pair block, so a
single k-sigma cut in a single fit keeps intra-wiki edges and drops every
cross-source pair. The fix in both places is to fit per group and cut once.

**2. Chunking is fitted per source (R19).** `fit(docs, doc_ids, sources=None)`
groups documents by source and runs `derive_chunk_params` once per group;
`chunk_params` becomes `{source: {unit, m, hi, lam}}`. Each document is chunked
against its OWN source's params. `chunk_params` is already in `_PARAM_ATTRS`, so
persistence and reproducibility are unchanged. A single-source run is one group
and reproduces today's numbers exactly (Brown: m=107, hi=153, lambda −0.441).

**3. Source is node metadata, not a retrieval dial (R20).** Every chunk carries
`source` beside `doc_id`. Node payload contract, fixed here so T2/T3 and T5 agree
without a second conversation:

- `source` is a short lowercase token, one of `brown`, `quotes`, `wiki`.
- `doc_id` is source-prefixed: `brown/<fileid>`, `quotes/<index>`, `wiki/<index>`.
  The prefix is part of the id, not a display convenience; `doc_id.split("/", 1)[0]`
  reproduces `source`, and `source` is nevertheless stored explicitly so no
  consumer has to parse ids. `mixed-full` (2026-09-01) is the first live run with
  prefixed ids; the explicit `source` key is now exercised end to end and the
  prefix fallback is belt-and-braces, kept as specified rather than dropped.
- The persisted node payload (jsonb, `pg_store`) carries `source` on EVERY node;
  edge export rows name the source pair alongside the documents they came from.
- Consumers (graph_tools, walker_app, export_neo4j) surface source mix where they
  already surface documents. A run written before R20 has no `source` key;
  consumers degrade to today's unlabelled display and MUST NOT raise.

Source steers nothing at query time. Anchors stay pure BM25 over the whole run.
sqrt-allocation of anchors per source is queued LATER and built only on measured
starvation (T9 records the evidence either way).

**Closed 2026-09-03:** it was built, measured, and superseded — see §6.15 A. No
sqrt code path exists in `sampler.py`; anchors are additive-only (S15/S16) and the
ring share (S17) is the source lever.

**4. Normalization is per source-pair block, the cut is global (R21).** With
three sources each similarity space has six blocks — 3 intra (brown×brown,
quotes×quotes, wiki×wiki) and 3 cross (brown×quotes, brown×wiki, quotes×wiki).
Each block's nonzero similarities get their own Box-Cox with the existing R6
estimator-pair gate, and R2/R3 fall back PER BLOCK (a degenerate cross block
falls back alone, never dragging the space with it). The per-block z values are
reassembled into one array and then cut ONCE with today's global k-sigma rule;
backbone (R7) and FUSE (R3) run unchanged. This is the move FUSE already makes
across spaces, applied across source pairs. Blocks with fewer than ~20 nonzero
pairs inherit the pooled fit rather than fitting on noise, and diagnostics record
which blocks were pooled plus the edge rate per block. A single-source run is one
block and is byte-identical to today.

**4b. Why this works — one alignment trick, applied twice.** The general move:
whenever two score populations come off different rulers, align their
distributions first, compare after. The full nonparametric version is quantile
matching (ogive alignment); Box-Cox-to-z is the smooth two-parameter version of
the same idea, and the rank/quantile fallback (R2/R3) is the full version taken
literally when the parametric fit fails its estimator gate. FUSE already makes
this move once, across the two spaces — raw sparse cosine is never compared with
raw dense cosine. R21 makes it again, one level down, across source-pair blocks:
same math, different axis of incomparability (spaces before, text registers now).
After alignment, "significantly similar for what these two kinds of text are" is
one comparable statement everywhere, and a quote–quote edge earns its place
against its own null, not against wikitext's.

Nothing extra is needed for the ~29k:2.5k:0.5k size imbalance, and that is the
point. Size hurt exactly one thing: it let the largest source's distribution
impersonate the global distribution, so the one threshold was really wikitext's
threshold. Per-block fits remove that; block size stops mattering for fairness,
and the per-node KNN backbone (R7) is size-blind. The imbalance that remains is
legitimate — wikitext simply offers more candidates, so a wiki-flavored question
gets more wiki results. That is relevance, not bias. The one real size effect
left is statistical: a tiny block gives a noisy fit, handled by the <~20-pair
pooled-fit inheritance recorded in diagnostics.

Standing caveat: alignment makes the blocks comparable, not related. It says how
strong a quote–quote resemblance is *for quotes*; whether a quote and a wiki
passage are actually about the same thing is still cosine's job, in whichever
space.

**5. Acceptance.**
(a) Per-block edge rate (edges kept / candidate pairs, read from run diagnostics)
    for the three intra blocks lies within one order of magnitude of each other on
    a mixed run.
(b) "a quote about courage" returns at least one `source=quotes` chunk in the
    Bundle; a "how does wikipedia describe ..." prompt returns at least one
    `source=wiki` chunk; and at least 3 existing Brown prompts still anchor in
    `source=brown` chunks.
(c) `tests/test_sampler.py`, `tests/test_gist_walk.py`, `tests/test_walker_render.py`
    pass unchanged against a mixed run — multi-source changes nothing observable
    for a single-source consumer.

Guards: R19, R20, R21 in `chunkgraph.py` (bodies below, pinned by
`tests/test_chunk.py`, `tests/test_pg_store.py`, `tests/test_mixed_acceptance.py`).
Spec and docstring verified identical modulo ASCII transliteration (`--` for
`—`, `*` for `·`, curly quotes flattened), 2026-09-01 — no semantic drift; do
not re-diff.

R19 WHERE documents carry a source label, chunk params SHALL be derived once per
   source — Box-Cox, m = median, hi = m + 2·MAD in transformed space, inverted —
   and each document SHALL be chunked against its own source's `hi`. Generalizes
   R17, which fitted the corpus as one population: pooled across corpora of
   different registers the fit follows the most numerous source, so a 1-line
   quote and a 400-line article are judged by the same threshold. `chunk_params`
   SHALL be `{source: {unit, m, hi, lam}}` and SHALL be recorded in diagnostics
   and run params. A single-source run is one group, keyed by the literal source
   label `"default"`, and reproduces R17's numbers exactly.

R20 WHEN chunks are produced, each SHALL carry its `source` beside its `doc_id`
   (R8's twin), `doc_id` SHALL be source-prefixed (`brown/`, `quotes/`, `wiki/`),
   and `source` SHALL be persisted on every node payload so retrieval results,
   community summaries, and exports can name the corpus a chunk came from.
   Source is metadata only: it SHALL NOT weight, quota, or filter retrieval.
   WHERE a run predates this guard and carries no source, consumers SHALL degrade
   to unlabelled display rather than fail.

R21 WHEN similarities are normalized in a multi-source run, the Box-Cox fit SHALL
   run per SOURCE-PAIR BLOCK (3 intra + 3 cross for three sources), each block
   gated by R6 and falling back per R2/R3 independently, and the reassembled z
   values SHALL then be cut ONCE by the global k-sigma rule. One pooled fit is
   dominated by the largest block, which keeps that block's edges and starves
   every cross-source pair; per-block fits make the blocks commensurable, exactly
   as FUSE makes the two spaces commensurable, and a single global cut keeps ONE
   significance standard for the run. Blocks with fewer than `MIN_BLOCK_PAIRS`
   (= 20) nonzero pairs SHALL inherit the pooled fit, and per-block edge rates
   SHALL be recorded in diagnostics. A run SHALL
   carry at most 8 source labels — block codes are packed int8, so a ninth label
   overflows the code space. A single-source run is one block and is
   byte-identical to today.

**Measured (2026-09-01, run mixed-full).**

1. **Run identity and size.** Run `86527c4e-e0d0-4701-a554-8be2d7f6f5db`, label
   `mixed-full`: 10,369 documents → 10,826 chunks (wiki 7,818 / quotes 2,508 /
   brown 500), 373,215 edges, 27 communities. Wall 669 s, fit 549 s, peak RSS
   7.1 GB against 6.6 GB predicted by the 55n² rule — the rule held.

2. **Per-source chunk params (R19 evidence), read from `graph_run.params ->
   'chunk_params'` in Postgres for this run:**

   | source | unit  | m   | hi  | lam      |
   |---|---|---|---|---|
   | brown  | lines | 107 | 153 | -0.4409  |
   | quotes | chars | 98  | 256 | -0.2293  |
   | wiki   | lines | 30  | 78  | 0.1012   |

   quotes' `hi` (256, char-scale) runs over 3x wiki's `hi` (78, line-scale) and
   brown's `hi` (153, line-scale) sits between them — three different fitted
   ceilings from three different registers, exactly what pooling a single
   corpus-wide `hi` would have erased. brown reproduces R17's single-source
   numbers exactly (m=107, hi=153, lam=-0.441), confirming the "single-source
   run is one group" clause above still holds inside a multi-source run.

3. **The chosen wiki stride, and why (R19/scale evidence).** `--wiki 7461
   --wiki-stride 4`. Stride 4 is the largest stride keeping projected n under
   the 12,000-chunk target derived from the dense n×n float64 memory budget;
   stride 3 projected ~10,419 chunks with no headroom, and the earlier
   unstrided `--wiki 200` attempt blew a 3.7 GB RSS cap mid-fit. The target is
   memory-derived, not arbitrary.

4. **Per-block sparse edge rates (acceptance 5(a) evidence).** All six:
   brown|brown 2.292e-02, quotes|quotes 1.984e-02, wiki|wiki 2.002e-02,
   brown|quotes 1.540e-02, brown|wiki 1.793e-02, quotes|wiki 1.553e-02. Intra
   max/min ratio **1.156** against the 10x bound — 5(a) PASS. All six blocks
   `fit=own` at this scale. Contrast with the smoke run (mixed-smoke, 633
   chunks): ratio 1.239, wiki|wiki `fit=rank` — the per-block fallback firing
   for one block alone is R21 working as specified, not a defect.

5. **Acceptance battery result (5(b)/(c) evidence).** 72 tests green against
   mixed-full; every source anchored on its own targeted prompts (brown: 5-11
   chunks across 4 Brown prompts). Therefore the LATER sqrt-anchor allocation
   has **no** supporting evidence at full scale and **stays parked** — this
   verdict is explicit so the queued item is not mistaken for pending work.

   **Superseded 2026-09-03:** the item was un-parked on B-class evidence, built (T2),
   and MEASURED harmful — it capped the majority source rather than flooring
   minorities. §6.15 A carries the amendment and the numbers.

6. **R5 degradation, stated as a limitation.** Both mixed runs were
   **sparse-only**: no embed model in the run environment, dense arm disabled
   at ingest. Every number in this subsection is therefore the sparse space;
   the per-block normalization above is proven on sparse and **untested on
   dense**, and a dense mixed run is queued. This is the honest boundary on the
   whole subsection.

7. **Campaign defect worth recording.** `_merge_phrases` rebuilt the full
   gensim `Phrases` model once per document (O(n²)); invisible at Brown's 500
   docs, ~100 h projected at 10,369. Diagnosed with py-spy after 100+ CPU-min
   burned; the model is now hoisted to one construction, semantically neutral,
   pinned by `test_merge_phrases_single_model_matches_per_doc_rebuild`. The law
   it implies: a per-document loop that constructs a corpus-wide model is a
   scale bomb no small-corpus test can see.

### 6.15 Per-source anchors, titles, entities v0 (2026-09-03)

Three amendments, one pass, so the code tasks that follow paste rather than re-decide.
(A) anchors are allocated per source instead of taken globally; (B) the R20 payload
contract gains `title`; (C) entities get a bipartite store of their own beside the chunk
graph. Nothing here changes what a single-source run does.

**A. Anchors are allocated per source, by sqrt(n_s) (S15, S16).**

The measured finding is that a GLOBAL anchor count cannot fix register starvation.
On run `bfa594df` (mixed-full-dual, 2026-09-02) raising `k_anchor` 3 -> 9 lifted the
quotes share on register-explicit prompts from 14-18% to 23-29% but dropped C3's brown
floor from 30% to 19%: extra global slots are spent in whichever source has the most
candidates, so buying one register's recall sells another's. The frozen diagnostic set's
rule is do-no-harm, so k=3 stood and the structural fix was queued. This is that fix.

The allocation: `n_s` is the run's per-source chunk count from `gt.run_sources`, each
source's share is `sqrt(n_s) / sum_t sqrt(n_t)`, and the total anchor budget `k_anchor`
is split by largest-remainder rounding so the parts sum to the budget exactly. Sqrt and
not linear: linear proportionality reproduces the global behaviour (wiki holds 72% of
mixed-full's chunks and would hold 72% of the anchors), while equal split ignores that a
larger source really does offer more candidates. Sqrt is the standard compromise between
"size counts" and "every register gets a seat" -- the same shape as sqrt-allocation in
stratified sampling, where it minimises variance across strata of unequal size.
[empirical:cited -- k=3/k=9 numbers from sampler.py's DEFAULT_K_ANCHOR comment and
diagnostic-prompts.md; the stratified-sampling rationale is convention, not a measurement
on this corpus.]

Selection is per source and merging is by score: each source runs its own BM25 search for
its own `k_s`, the results are concatenated, deduplicated and ordered by score. R1's
anchor validation (the matched query term must appear in the chunk's top DISC_M tf*idf
terms) is applied exactly as today, per anchor, unchanged -- allocation decides how many
anchors a source may contribute, never whether a weak one is admitted.

A run with one source label -- including the literal `"default"` label R19 mints for a
single-source run, and a pre-R20 run whose nodes carry no source at all -- takes the
degenerate path: one group, the whole budget, one search, and an anchor list byte-identical
to today's single global `gt.search(conn, run, query, k=k_anchor)`. That identity is a
regression test, not a hope.

`DEFAULT_K_ANCHOR` stays 3. Allocation changes WHERE the three anchors come from, not how
many there are. If the frozen-set re-run shows B still under 40%, the recorded escalation
lever is the run-level `k_anchor` knob plus the queued source-aware ring share -- with the
measured per-source shares as the evidence -- and not a second structural change invented
inline.

**Amended 2026-09-03 (T7 measurement, run bfa594df-1238-448f-8e2b-d05136e6307a).**
The sqrt split above is SUPERSEDED. Measured on the frozen set, it acts as a CAP on the
majority source rather than a floor for minorities: on the 3-source dual run wiki is held
to 1-of-3 anchors even on a 99%-wiki query, flipping A1 and E3 PASS -> FAIL, and B moved
the wrong way (quotes 6-31% -> 2-25%, B3 at 2%). The amended law: an anchor earned
competitively is never taken away -- the global BM25 top-k stands untouched, and minority
sources receive at most one EXTRA anchor each, beyond `k_anchor`, only when their best hit
is competitive (>= COMPETITIVE_FRAC of the global k-th score). Anchors are therefore
do-no-harm only; they are not the B lever. The measured B lever is the ring lane: on
B1-B5 the ring origin carries 8/24, 4/24, 0/24, 8/24, 4/24 quotes chunks per walk while
walk-only quotes sits at 4-17%, so the queued source-aware ring share is promoted (S17):
ring top-off slots are allocated across sources by the anchor list's source mix.
[empirical:cited -- all numbers from T7's re-run, playbook.md T7 _Blocked:_ and
diagnostic-prompts.md's 2026-09-03 calibration line.]

**Outcome 2026-09-03 (T7c measurement, run bfa594df-1238-448f-8e2b-d05136e6307a).**
The three-point B trajectory in one line: pre-campaign (global k=3) **6-31%** -> sqrt
allocation (T2) **2-25%** -> additive anchors + S17 ring share (T7a/T7b) **8-29%**
quotes_all; quotes_walk 4-12%; the gate is **>=40%** and none of the three cleared it.
Ring fill of the 24 ring slots: B1 6, B2 6, B3 2, B4 6, B5 12.

Do-no-harm ledger: A1 broke under sqrt and is **RESTORED** to PASS under additive
anchors (n=77, wiki 76/77, `wiki/28410` anchored) -- the amended law works exactly as
specified. **E3 is NOT restored** (n=72, wiki 68/72, brown 4/72): this is a *different*
failure mode from the sqrt cap, since anchors no longer cap the majority -- the miss
now sits upstream in ring/PPR steering. 13/20 rows PASS, matching the original
calibration count; A2 stays KNOWN-FAIL.

The disposition, stated as a stop, not a defect: per the 2-round ladder law the B lane
is **STOPPED and handed to the operator**. Two fable-tier scopes were spent (sqrt ->
additive+ring). No third re-scope, no inline tuning. What is pinned and green is the
*mechanism* (S15/S16/S17, 63/63 sampler tests); what is open is the *B share target and
E3*.

**The sqrt closure sentence -- this one is load-bearing for T9's `_Verify:`:** sqrt
allocation is DEAD. `allocate_anchors` and its per-source split are deleted from
`sampler.py`; every remaining occurrence of the word "sqrt" in this spec and in
`sampler.py`'s S15 docstring is superseded-history prose, retained deliberately so the
measurement that killed it is not re-derived. A grep for "sqrt" returning hits is
expected and means history preserved, not code shipped.
[empirical:cited -- playbook.md T7/T7c _Blocked:_ and diagnostic-prompts.md's two
2026-09-03 entries.]

**Amended 2026-09-03 (T10, S18): the ring lane gets a query-side router and an
additive injection.** T7c stopped the B lane at 8-29% with S17's anchor-mix ring
share; a fourth probe round (`.tmp/steer_probe.py`, 4 rounds, 51 configs, run
`bfa594df-1238-448f-8e2b-d05136e6307a`) found the missing piece is that the ring
pool never *contains* the minority source's on-topic chunks, and that the
allocation signal must come from the **query**, not from the corpus scores.

The mechanism, three clauses, in this order:

1. *Additive per-source injection.* Per-source BM25 top-k (k=24) over the same
   `gt.corpus_index` postings and the same K1=1.5 / B=0.75 formula `gt.search` uses,
   bucketed by source **before** the cut, UNIONed into the S17 ring pool. Scores are
   rescaled to the walk pool's own range (`(score / global_max_injected) *
   max_pool_score`) so an injected row cannot outrank the pool by BM25 magnitude; a row
   already in the pool keeps `max(existing, injected)`; a row already in `W` is skipped.
   Nothing is removed -- the walk is untouched, and `select_ring` still does the cutting.
2. *LM log-odds router.* `w_s = softmax_tau( mean over query tokens of
   log( p_s(t) / p_corpus(t) ) )`, `tau=0.2`, `p_s(t) = (occurrences of t in source s +
   0.5) / (tokens in s + 0.5)`, `p_corpus(t) = (occurrences of t everywhere) / (tokens
   everywhere)`, tokens counted from `corpus_index["dl"]`. Occurrences, not document
   frequency: quotes chunks are ~15 tokens against wiki's ~200, so df makes a token look
   rare in quotes purely because quotes documents are short.
3. *eps floor, and the anchor mix is REPLACED.* Ring mix = `max(w_s, 0.05)`
   renormalized, then scaled to integer counts for `select_ring`. The floor is a small
   **constant**, never the anchor mix: the anchor-mix floor is the measured cap.

Falsified alternatives -- one line each, with the killing number. All from the same
probe json, same run, same frozen 20-row set (`quick` = 10 rows, `full` = 20). Every
line is `[empirical:cited -- .tmp/steer_probe_results.json key <K>]`.

| lever | config | killing number |
|---|---|---|
| Degree penalty on walk candidates (H1) | `alpha=1.0`, `alpha=2.0` | B moved 15-29% -> 21-52% but broke C1 **and** C3 (brown floors), 12/20 vs baseline 13/20 -- buys quotes by selling brown. Same at `alpha=2.0`; it is not a tuning miss. |
| Relative degree penalty per source (R2 C2) | `rel_alpha=1.0` | B **4/58/27/2/2%** -- moves B1/B4/B5 the wrong way; 3/10 quick pass. |
| Minority source boost (H2) | `beta=1.5`, `beta=2.5` | B **0/35/31/0/0%** at 1.5; at 2.5, 10/20 with A3, D3, E1, E3 all flipped to FAIL. A multiplier on minority edges starves the query's own register. |
| Community-share cap (H3) | `cap_x=0.3`, `0.4` | B 17-38%, indistinguishable from baseline's 15-29%; `cap_x=0.3` additionally flips C1. Not a lever, a coin flip. |
| Ring quotas from the walked set instead of the anchors (R2 C1) | `ring_walk_mix` | B1 54% but B2/B3 at 21%; C1 and C3 both FAIL. Same trade as H1. |
| Brown-only floor (R2 C4) | `brown_boost=2.0` | Identical to C1's numbers -- the boost changes nothing the ring mix had not already decided. |
| Injection with **no** re-quota (R3a) | `inject_k=24` alone | B **15/19/15/19/29%** -- baseline, to within one chunk. Injection alone is inert: the rows enter the pool and `select_ring` cuts them straight back out. This is the load-bearing negative result -- injection and router only work as a pair. |
| Corpus-score quota, linear (R3b) | `inject_quota` | B 19-25%. Mean BM25 barely separates the sources. |
| Corpus-score quota, sharpened (R3e) | `quota_power=8`, `16` | B1 40% -> 48% while B2-B5 **fall** to 4-8%, and C1 flips. Sharpening amplifies a signal that is flat. |
| Lift quota -- per-source top-k mean over that source's own baseline (R3f) | `quota_lift` | B **6/8/6/6/6%**, worse than baseline at every power. The sign is wrong: wiki chunks are long and title-rich, so lift ranks the corpus the query is *not* about. |
| Dense centroid router, global-centroid-centered (R4 S1) | `router=dense_c, tau=0.1` | B **10-19%**, below baseline. Cosine to a corpus centroid ranks generic-ness, not topic. |
| Per-source **df** router (R4 S2) | `router=terms` | Length-confounded by construction -- superseded by the LM variant before a full row; the LM router is the same log-odds shape with occurrences in place of df. |
| LM router **floored at the anchor mix** (R4) | `router_floor=True, tau=0.1` | B **31-38%**, still under the 40% gate -- with the same router, dropping the anchor floor gives 52-56%. This is the direct measurement that **the anchor-mix floor is the cap**, and the reason S18's floor is a constant. |
| tau sweep | 0.1 / 0.2 / 0.5 / 1.0 | 0.1, 0.2 and 0.5 all reach 48-56%; 1.0 flattens to 44-46%. tau=0.2 chosen mid-plateau, not at an edge. |
| eps sweep | 0.0 / 0.05 / 0.1 | 0.0 gives 52-56%, 0.05 gives 48-52%, 0.1 gives 44-48%. 0.05 chosen: the ~3-point cost buys a hard "no source is ever starved to zero" guarantee. |

The measured result. `R4_PROMOTE_lm_t0.2_eps0.05_FULL`, full 20-row set, run
`bfa594df-1238-448f-8e2b-d05136e6307a`, 20.5 s:
- **18/20 PASS** (baseline 13/20). A2 stays KNOWN-FAIL (expected, not this fix's target).
- **B1-B5 48-52%** quotes_all (B1 47.9, B2 52.1, B3 50.0, B4 52.1, B5 50.0), against the
  >=40% gate -- first time the gate is cleared. quotes_walk 4-13%; ring origin carries
  **22/24** slots on every B row. The B lane is won entirely in the ring, exactly where
  T7's evidence said it would be.
- **Zero regressions:** A1, A3, A4, A5, C1-C4, D1-D3, E1, E2 all PASS. The oracle row
  (`R3_oracle_quotes_ring_full`, whole ring budget handed to quotes) scores 52-56% and
  18/20 -- S18 is within ~4 points of the ceiling the ring lane can ever reach, and hits
  the same pass count.
- **E3 is OUT OF SCOPE and stays FAIL** (combined brown+political-wiki share 12.5% vs the
  >=50% expectation). It fails at baseline too, under every one of the 51 probe configs,
  and its miss is brown-side: E3 wants brown prose promoted on a query whose tokens are
  wiki-shaped, which is a different mechanism from the register routing S18 ships.
  Recorded as an open finding, not a regression, and not a reason to tune S18.

Acceptance (a) is extended with the S18 clause: on a run reporting >= 2 source labels,
`ef_evidence` ring members are chosen from the S17 pool UNIONed with the per-source BM25
top-24, allocated by the eps-floored router mix; on a run reporting one source label the
ring fill is byte-identical to today, element for element and in order. Pinned by
`pytest tests/test_sampler.py -q` and by the frozen-set re-run.

"Guards:" is extended below to "S15, S16, S17, S18".

S18 (new 2026-09-03; promotes the probe-validated ring router. Trigger: T7c stopped
    the B lane at 8-29% with S17's anchor-mix share, and a 4-round steering probe
    measured that the ring POOL never contains the minority register's on-topic
    chunks, while every corpus-score allocation signal is flat or sign-wrong.)
    WHERE the run reports more than one source label, the S13/S17 ring candidate
    pool SHALL be UNIONed with a per-source BM25 top-INJECT_K, scored by the same
    formula and postings gt.search uses and bucketed by source BEFORE the cut,
    rescaled into the walk pool's own score range so an injected row never enters
    above the pool's strongest member. Injection is ADDITIVE: no pooled candidate is
    removed, no member of W is re-entered, and select_ring still performs the cut.
    The ring allocation SHALL come from the QUERY, not from the corpus scores:
    w_s = softmax_ROUTER_TAU( mean over query tokens t of
    log( p_s(t) / p_corpus(t) ) ), with p_s(t) = (occ_s(t) + 0.5) / (tokens_s + 0.5)
    counted in OCCURRENCES per token of corpus, never in document frequency -- df is
    length-confounded across registers. The mix passed to select_ring SHALL be
    max(w_s, ROUTER_EPS) renormalized, REPLACING the anchor mix: the anchor-mix
    floor is MEASURED as the cap (same router, anchor floor on -> B 31-38%, floor
    off -> B 52-56%), so it SHALL NOT be applied. WHERE the query shares no token
    with the run's postings the router SHALL return no weights and the ring SHALL
    fall back to S17's anchor mix unchanged. WHERE the run reports one source label
    -- including R19's literal "default" and a pre-R20 run carrying no labels -- the
    router degenerates to a single weight of 1.0, injection is SKIPPED, and the ring
    fill SHALL be identical to today's, element for element and in order, reached by
    the same early return.

**B. `title` joins the node payload (R22, extending R20).**

R20 fixed the law: parse at the boundary, store explicitly, never make a consumer scan
text. `source` obeys it; the article title did not, and every consumer that wanted to show
"Battle of Midway" instead of `wiki/28410` had to re-parse the leading `= Title =` line
out of chunk text -- or, more often, showed the opaque id. `ingest_mixed.wiki_title()`
already parses it once at the boundary; R22 carries that value through `fit()` into the
persisted payload.

Payload contract, extending the §6.14 R20 list:

- `title` is the document's own title as a NON-EMPTY string. It is per-DOCUMENT: every
  chunk of a document carries the same value.
- Where a document has no title -- a source with no such notion (brown, quotes), a parse
  returning None, or a run written before this guard -- the `title` key is **ABSENT** from
  the payload. It is never present-and-null and never the empty string, so
  `attrs.get("title")` being falsy has exactly one meaning.
- Consumers display `title` where present and fall back to `doc_id` where absent, and a
  pre-R22 run renders exactly as it does today. No re-ingest is required by this
  amendment; titles appear on the next natural re-ingest.
- Titles reach `fit()` as a list aligned with `docs` (the same shape as `sources`), whose
  entries may be None.
- Like `source`, `title` is metadata: it does not weight, quota, or filter retrieval.

**C. Entities v0: a bipartite entity store beside the chunk graph (E1-E5).**

v0 fixes the SHAPE, not the extraction. The population is the vocabulary the run already
has -- the salient/phrase terms persisted in `node.attrs->'tf'` -- typed `"term_v0"`, built
with deterministic NLP only: no LLM, no NER dependency. Real NER is v1 and replaces the
population without changing the tables. Three tables, run-scoped like everything else
(W1), created idempotently by the builder itself:

    entities(run_id, entity_id, name, type)          -- PK (run_id, entity_id)
                                                     -- UNIQUE (run_id, name)
    mentions(run_id, ord, entity_id, cnt)            -- PK (run_id, ord, entity_id)
                                                     -- FK (run_id, ord) -> node
                                                     -- FK (run_id, entity_id) -> entities
    entity_edges(run_id, a, b, npmi, ppmi, bm25)     -- PK (run_id, a, b), CHECK (a < b)
                                                     -- a, b are entity_ids

`mentions.ord` IS the chunk id: chunk identity in this schema is the run-local node
ordinal, the same key `node`, `community.members` and the walker already use, so no new
id space is minted (X1's lesson, one level down).

Counting definitions, fixed here so the builder and its tests agree:

    N          = number of chunks in the run
    df(x)      = number of chunks with cnt(x) >= 1
    joint(a,b) = number of chunks containing both a and b
    p(x)       = df(x) / N ;  p(a,b) = joint(a,b) / N
    pmi(a,b)   = log( p(a,b) / (p(a) * p(b)) )
    npmi(a,b)  = pmi(a,b) / -log( p(a,b) )
    ppmi(a,b)  = max(pmi(a,b), 0)
    bm25(a,b)  = bm25_dir(a -> b) + bm25_dir(b -> a)        -- symmetric by construction
    bm25_dir(x -> y) = sum over chunks c containing x of
        idf(y) * tf(y,c) * (K1 + 1) / (tf(y,c) + K1 * (1 - B + B * dl(c) / avgdl))
      with idf(y) = log(1 + (N - df(y) + 0.5) / (df(y) + 0.5)), tf from mentions.cnt,
      dl(c) from node.attrs->'n_tok', and K1 = 1.5, B = 0.75 -- graph_tools.search's
      constants, so the entity weighting and the retrieval weighting are the same ruler.

Three weights on one row because the queued A/B between them (BM25 vs length-normalized
PMI for chunk-term weighting) must be a column choice at read time, never a rebuild.

**Acceptance.**

(a) **Anchors + ring (amended 2026-09-03).** On any query where no minority source's
    best hit clears 0.5x the global k-th score -- and on every single-source or pre-R20
    run -- the anchor list is EQUAL, element for element and in order, to
    `gt.search(conn, run, query, k=k_anchor)`. Where a minority source's best hit does
    clear it, the anchor list is the global top-k PLUS that hit, ordered (-score, ord):
    a 99%-wiki query keeps all its wiki anchors by construction (A1's shape, pinned as
    a planted-score unit test). Ring: on an anchor list spanning >= 2 sources, ring
    slots split by the anchor source mix (anchors {wiki 2, quotes 1}, budget 24 ->
    {wiki 16, quotes 8}); on a one-source anchor list the ring fill is byte-identical
    to today. Pinned by `pytest tests/test_sampler.py -q`.

    **MEASURED 2026-09-03.** The byte-identity and planted-score clauses PASS --
    `pytest tests/test_sampler.py -q`, 63/63, live DB, 0 skipped. The B-share target
    (>=40%) and E3's >=50% combined share are **OPEN**, not met, and are no longer this
    section's acceptance: they are handed to the operator per the ladder law.

(b) **Title.** A `fit()` over two documents whose titles are `["Battle of Midway", None]`
    persists `title == "Battle of Midway"` on every chunk of the first document and NO
    `title` key on any chunk of the second; walker and digest render the title for the
    first and `doc_id` for the second; and a payload with no `title` key at all (a pre-R22
    run) renders exactly as it does today and raises nothing. Pinned by
    `pytest tests/test_ingest_mixed.py tests/test_pg_store.py tests/test_interpret.py
    tests/test_walker_render.py -q`.

(c) **Entities v0.** On a planted corpus with hand-counted co-occurrence, `entities`,
    `mentions` and `entity_edges` hold exactly the expected rows; every pair whose joint
    count is below 5 has NO `entity_edges` row while every pair at or above 5 has one;
    `npmi` matches the closed form above to 1e-9 on at least 3 planted pairs; and a second
    build over the same run leaves that run's row counts identical and another run's rows
    untouched. Pinned by `pytest tests/test_entities.py -q`.

    **Shipped deviation, accepted 2026-09-03:** `entity_id` is `integer`, not
    `smallint` -- 32k entities is inside reach on a full wiki run and the spec named no
    width. Rationale recorded rather than reverted.

    (d) **Entity resolution v1.** On planted names, a near-duplicate pair sharing >= 2
        co-occurrence neighbors merges and a near-duplicate pair sharing none does not;
        the canonical of every component is its longest name (ties by ascending name);
        the same input in a different order yields an identical mapping; a second
        `resolve_entities` over the same run leaves that run's rows identical and
        another run's rows untouched; and every `entities` row of the run carries a
        non-NULL `canonical_id`. The embedding signal is off by default and contributes
        nothing when unavailable. Pinned by `pytest tests/test_entities.py -q`.

**D. Entity resolution v1: canonical ids from string + neighbor agreement
(E6-E8, amended 2026-09-03).** E1 reserved the phrase "v1" for NER replacing the
v0 population. This is a different v1 and does not claim that one: the population
stays `term_v0`, extraction is untouched, and what is added is a *resolution*
pass that decides which of those rows name the same thing. NER remains future
work and E1's population clause stands unamended.

The ladder is deterministic and runs on two independent signals, in this order:
(1) string similarity over normalized entity names proposes candidate pairs;
(2) a corroboration gate keeps a candidate only where the pair ALSO shares
co-occurrence neighbors in `entity_edges`; (3) union-find over the survivors
yields connected components, and each component's longest name (ties by ascending
name) becomes its canonical entity. A third signal -- model2vec embedding
agreement -- exists behind a flag that is OFF by default and stays off until
measured. Surface forms are never rewritten: `entities.name` and `mentions` are
untouched, and `canonical_id` is an added column, so the aliases are exactly the
rows whose `canonical_id != entity_id`. Supersede, never delete.

    entities(run_id, entity_id, name, type, canonical_id)   -- ADDed column

Every row gets a non-NULL `canonical_id`: a singleton's canonical is itself, so a
consumer joins on `canonical_id` unconditionally and never has to branch on NULL.

Guards: S15, S16, S17, S18 in `sampler.py`; R22 in `chunkgraph.py` (with `ingest_mixed.py`,
`pg_store.py` and the display consumers honouring it); E1-E9 in `entities.py`. Bodies below
are ASCII and are pasted into those docstrings unchanged -- spec text and docstring are the
same bytes by construction, so there is nothing to transliterate and nothing to re-diff.

S15 (amended 2026-09-03; supersedes the sqrt reallocation, which MEASURED on run
    bfa594df flipped A1 and E3 PASS -> FAIL by capping wiki to 1-of-3 anchors on a
    3-source run, and moved B 6-31% -> 2-25%.) BM25 anchors SHALL start from the
    plain global search top-k_anchor, unchanged from today: a source that earned an
    anchor competitively SHALL never lose it to allocation. WHERE the run reports
    more than one source label, each source with n_s > 0 holding no anchor in that
    global top-k MAY receive at most ONE extra anchor slot BEYOND k_anchor, filled
    by that source's best hit ONLY IF that hit's BM25 score >= COMPETITIVE_FRAC
    (0.5) of the global k_anchor-th hit's score. Extra anchors, never reallocated
    ones. Every anchor, base or extra, SHALL still pass R1 validation unchanged:
    the floor decides how many anchors a source MAY add, never whether a weak one
    is admitted.

S16 (amended 2026-09-03.) WHERE no minority hit clears S15's competitive threshold
    -- and on every run reporting one source label, including R19's literal
    "default" and a pre-R20 run carrying no labels -- the anchor list SHALL be
    identical, element for element and in order, to today's single
    gt.search(k=k_anchor): byte-identity whenever the allocation would change
    nothing. The merged list (base top-k plus extras) SHALL be deduplicated by
    ordinal and ordered by BM25 score descending, ties by ordinal ascending. Extra
    anchors draw against ef as all anchors do (S10) and SHALL never displace a
    base anchor.

S17 (new 2026-09-03; promotes the queued source-aware ring share. Trigger: T7's
    ring evidence, run bfa594df -- B1-B5 ring origin carries 8/24, 4/24, 0/24,
    8/24, 4/24 quotes chunks per walk while walk-only quotes sits at 4-17%, so the
    ring top-off, not the walk, is where register share is won or lost.) WHERE the
    run reports more than one source label AND the anchor list spans more than one
    source, the ring top-off budget (RING_TOP x RING_PER, S13) SHALL be allocated
    across sources in proportion to the anchor list's source composition by
    largest-remainder rounding, each source's slots filled by its strongest ring
    candidates -- S13's candidate pool and strength ordering unchanged, only which
    slots go to which source changes. A source with fewer candidates than slots
    SHALL forfeit the shortfall to the remaining candidates in global strength
    order, so the walk never shrinks. WHERE every anchor shares one source -- or
    the run reports one label -- allocation SHALL be skipped and the ring fill
    SHALL be identical to today's, element for element and in order.

S18 (new 2026-09-03; promotes the probe-validated ring router. Trigger: T7c stopped
    the B lane at 8-29% with S17's anchor-mix share, and a 4-round steering probe
    measured that the ring POOL never contains the minority register's on-topic
    chunks, while every corpus-score allocation signal is flat or sign-wrong.)
    WHERE the run reports more than one source label, the S13/S17 ring candidate
    pool SHALL be UNIONed with a per-source BM25 top-INJECT_K, scored by the same
    formula and postings gt.search uses and bucketed by source BEFORE the cut,
    rescaled into the walk pool's own score range so an injected row never enters
    above the pool's strongest member. Injection is ADDITIVE: no pooled candidate is
    removed, no member of W is re-entered, and select_ring still performs the cut.
    The ring allocation SHALL come from the QUERY, not from the corpus scores:
    w_s = softmax_ROUTER_TAU( mean over query tokens t of
    log( p_s(t) / p_corpus(t) ) ), with p_s(t) = (occ_s(t) + 0.5) / (tokens_s + 0.5)
    counted in OCCURRENCES per token of corpus, never in document frequency -- df is
    length-confounded across registers. The mix passed to select_ring SHALL be
    max(w_s, ROUTER_EPS) renormalized, REPLACING the anchor mix: the anchor-mix
    floor is MEASURED as the cap (same router, anchor floor on -> B 31-38%, floor
    off -> B 52-56%), so it SHALL NOT be applied. WHERE the query shares no token
    with the run's postings the router SHALL return no weights and the ring SHALL
    fall back to S17's anchor mix unchanged. WHERE the run reports one source label
    -- including R19's literal "default" and a pre-R20 run carrying no labels -- the
    router degenerates to a single weight of 1.0, injection is SKIPPED, and the ring
    fill SHALL be identical to today's, element for element and in order, reached by
    the same early return.

R22 WHEN a document carries a parsed title, every chunk of that document SHALL carry
    `title` in its persisted node payload beside `doc_id` and `source`, and the title
    SHALL be parsed ONCE at the ingest boundary (ingest_mixed.wiki_title() for
    wikitext) and passed into fit() as a list aligned with `docs`. Extends R20's
    contract and obeys R20's law: parse at the boundary, select by equality, never
    scan chunk text downstream. WHERE a document has no title -- a source with no
    such notion, a parse returning None, or a run written before this guard -- the
    `title` key SHALL BE ABSENT rather than present and null or empty, and consumers
    SHALL degrade to displaying `doc_id` rather than fail. Title is display and
    selection metadata only: it SHALL NOT weight, quota, or filter retrieval.

E1  Entities v0 SHALL be populated from the run's EXISTING salient/phrase vocabulary
    (the terms already persisted in node.attrs->'tf'), typed "term_v0", by
    deterministic NLP only -- no LLM call, no NER dependency. v0 exists to fix the
    SHAPE, a bipartite entity-mention store beside the chunk graph, not to improve
    extraction; real NER is v1 and SHALL replace the population without changing
    these tables.

E2  Every entities table SHALL be run-scoped with run_id as the first column and the
    leading key predicate (W1), and rows SHALL cascade with graph_run. Chunk identity
    SHALL be the run-local node ordinal `ord`, the key node, community.members and
    the walker already use; no new chunk id space SHALL be minted (X1's lesson one
    level down).

E3  The builder SHALL be READ-ONLY against node, edge and community -- it reads
    node.attrs and writes only entities, mentions and entity_edges -- and SHALL be
    re-runnable: a second build over the same run REPLACES that run's rows and leaves
    every other run's rows untouched.

E4  WHERE a pair of entities co-occurs in fewer than MIN_JOINT_CHUNKS (= 5) chunks,
    NO entity_edges row SHALL be written. NPMI on a joint count of one or two is
    dominated by rare-event bias -- a pair occurring exactly once, together, scores
    1.0 by construction -- so an unfloored ranking is a list of hapax coincidences
    rather than of associations.

E5  entity_edges SHALL carry npmi, ppmi and bm25 on the SAME row, computed over the
    same chunk-level co-occurrence counts, so choosing between weightings is a column
    choice at read time and never a rebuild. bm25 SHALL be the symmetric sum of both
    directions and SHALL use graph_tools.search's constants (K1 = 1.5, B = 0.75), so
    entity weighting and retrieval weighting share one ruler. a < b SHALL hold on
    every row: one row per undirected pair, the same law `edge` carries.

E6  Candidate pairs SHALL come from a NORMALIZED string similarity over
    entities.name -- casefold, separators collapsed to single spaces, surrounding
    punctuation stripped -- scored by difflib.SequenceMatcher.ratio(), stdlib only,
    NO new dependency. A pair scoring below SIM_THRESHOLD (= 0.90) SHALL NOT be a
    candidate. Candidate generation SHALL be blocked by a character-3-gram
    inverted index plus the lossless length bound 2*min(la,lb)/(la+lb) >=
    SIM_THRESHOLD: an unblocked pass is O(V^2) over a vocabulary that reaches six
    figures on a full wiki run, which is not a tuning choice but the difference
    between a pass that terminates and one that does not.

E7  A candidate pair SHALL be merged ONLY WHERE it also shares at least
    MIN_SHARED_NEIGHBORS (= 2) co-occurrence neighbors, taken as each entity's top
    NEIGHBOR_TOP_K (= 25) entity_edges partners by bm25 among rows with ppmi > 0.
    String similarity alone merges "united" into "untied"; requiring a SECOND,
    independent signal -- that the two names keep the same company in the corpus --
    is what makes a false merge cost two coincidences rather than one. An entity
    with no qualifying edges SHALL merge with nothing.

E8  Resolution SHALL be additive and idempotent. canonical_id SHALL be added by
    idempotent DDL (ALTER TABLE ... ADD COLUMN IF NOT EXISTS), SHALL be non-NULL
    on every row of the run -- an unmerged entity is its own canonical -- and SHALL
    be chosen per connected component as the LONGEST name, ties broken by ascending
    name, so the value does not depend on row order, dict order, or the order pairs
    were discovered. No row and no surface form SHALL be deleted or rewritten: the
    alias set of a canonical is exactly the rows carrying its canonical_id. The
    embedding signal (model2vec, fixed threshold, deterministic) SHALL be reachable
    only through an explicit flag defaulting to OFF, and where the model or the
    dependency is absent it SHALL contribute no candidates rather than raise.

E9  (new 2026-09-03; closes the scale-bound row §6.19 MEASURED left open.) WHERE
    the run's df-eligible pool (E4's `df >= MIN_JOINT_CHUNKS` floor, E1's nomen
    pool) exceeds VOCAB_BOUND terms, pair enumeration SHALL restrict further to
    the top-VOCAB_BOUND terms of that pool BY DOCUMENT FREQUENCY, ties broken by
    ascending term name -- a deterministic cut on the pool's cardinality, never a
    sample. Measured motivation: unbounded `pair_counts` on `mixed-full-dual`
    (10,830 chunks) enumerated 2.47e9 pair slots at the default floor, and §6.19
    already measured that RAISING the floor barely helps (1.75e9 at df>=50, 1.09e9
    at df>=200) because the explosion is concentrated in the highest-df terms
    themselves, which a floor cannot remove -- only a cap on how MANY of them are
    eligible bounds the enumeration. VOCAB_BOUND (= 130 initially, ~9.1e7 worst
    case) still overran the 10-minute Article VII budget live -- the pure-Python
    per-chunk enumeration, not the DB write, is the bottleneck -- so it was halved
    once per plan to 65, keeping the worst case at n_chunks * C(VOCAB_BOUND, 2) ~
    2.3e7 slots on this corpus size, regardless of how many chunks any one term
    appears in. The cut is scoped to `entity_edges`
    candidate generation only: `entities` and `mentions` still cover the FULL
    vocabulary (E1, E2 unchanged) -- an entity outside the bound simply carries no
    entity_edges rows, and so cannot reach resolution (E7) via the
    neighbor-corroboration path. WHERE the pool is at or below VOCAB_BOUND, the
    cut SHALL be a no-op and enumeration SHALL be byte-identical to today's.

    (amended 2026-09-03, second live measurement.) The bound above covers pair
    enumeration but NOT `resolve_entities`, whose `string_candidates` pass
    enumerates over every entity NAME in the run. Measured: the first bounded
    live `mixed-full-dual` build committed correctly (275,328 entities,
    5,813,717 mentions, 2,080 entity_edges) and then hung in resolution, killed
    at 23 min -- the blocked-gram pass is near-quadratic in the name count and
    T19 already measured 17.9 s at 9,100 names. Therefore: `resolve_entities`
    SHALL restrict its CANDIDATE population to the entities appearing in
    `entity_edges` for the run (either endpoint), leaving `canonical_id`
    assignment over the FULL population unchanged (E8: non-NULL on every row,
    an unmerged entity is its own canonical). This is lossless by E7, not a
    behavior change: merging requires >= MIN_SHARED_NEIGHBORS shared
    `entity_edges` neighbors, so an entity with no `entity_edges` row can never
    be merged and its candidate pairs can only ever be discarded. On this run
    the restriction takes the resolution population from 275,328 names to 65
    (= VOCAB_BOUND), and the whole resolve pass then runs in 22.9 s.

### 6.16 Export, live annotation, and the second-order term lane (2026-09-03)

Article IX repair: these guards shipped in `export_neo4j.py` and `graph_tools.py`
docstrings with no spec text anywhere in this file or PIPELINE.md. This subsection
records what exists; it does not redesign anything.

**A. The 4-file neo4j layout (X6-X8).** Measured rejection that forced it: the
dual-ID single-file header (`id:ID(Chunk)` + `id:ID(Term)`) is rejected by current
`neo4j-admin` as a duplicate property; the split into
`chunks/terms/contains/similar.csv` imported 286,158 nodes / 6,462,576 rels in 42 s
with `--multiline-fields=true`. Plus: a `;`-separated 256-dim embedding column read
via `embedding::text` (dim measured from data, `vector_index.cypher` emitted iff any
embedding was written), and `C<cid>` carried as a `:LABEL` column (10,820 labeled
live). X1-X5 stand unchanged.

**B. Walks are first-class in the mirror (X9, X10).**
`(:Walk {prompt,n,edges,wcc,density,conductance})-[:ANCHORS]->(:Chunk)` and
`(:Chunk)-[:PATHWAY {dwpc,of}]->(:Chunk)`, written from `gt.pathways` at retrieval
time over stdlib `urllib` (interpret.py's house transport, no new dep). Two facts
worth spec text because they cost a live failure: chunk ids are matched as
`id: STRING` (X9) and Chunk nodes are **MATCHed, never MERGEd**, so the writer cannot
mint phantom chunks; and neo4j refuses a schema statement sharing a transaction with
writes, so `CREATE CONSTRAINT` goes over the wire in its own call before the write
tx. Live demo: midway walk, density 0.67, conductance 0.80, top DWPC 0.138
3439<->3552.

**C. Second-order term ladder (W17).** The three rungs as shipped: Dunning-LLR gate
on chunk-level co-occurrence, Schutze context-centroid cosine on survivors,
Mann-Whitney AUC re-rank fired ONLY when the skew diagnostic trips -- named as the
R2/R6 gate-then-fallback pattern applied at term level. Pool = the operator's
high-BM25/low-PPMI keyness quadrant. Pinned by 9 planted-data tests + 1 live shape
test.

Guard bodies below are copied verbatim from the shipped docstrings -- spec text and
docstring are the same bytes, the §6.15 convention.

X6 Nodes SHALL be emitted one id-space per file. A single file carrying two
   `:ID(...)` columns is rejected by current neo4j-admin; the 4-file layout is
   what makes X1 structural rather than a per-row invariant.
X7 The embedding column SHALL be `;`-separated tokens copied verbatim from
   storage, of uniform length across the run, and `vector_index.cypher` SHALL
   be emitted if and only if at least one vector was written.
X8 A chunk with a community SHALL carry `C<cid>` as a second label in the
   `:LABEL` column, so the imported database is partitionable without a
   post-import pass.
X9  The live writer SHALL MATCH Chunk nodes, never MERGE them, and SHALL address
    them by `id` as a STRING -- the CSV import runs --id-type=STRING, so an integer
    parameter matches nothing and the write silently succeeds having written nothing.
X10 A walk SHALL be re-writable: (:Walk) is keyed on `prompt`, (:PATHWAY) on
    (src, dst, of) with the ordinal pair normalized low->high, so a second write of
    the same walk updates in place instead of duplicating. Any transport or MATCH
    shortfall SHALL raise; there is no partial-write fallback.

W17 second_order_terms() SHALL apply the ladder as a gate-then-fallback chain,
    in order: (a) Dunning-LLR co-occurrence gate (rung "llr") admits only
    candidates whose 2x2 association with the target clears g2_gate, and
    min_df floors BOTH the target and every candidate before anything else
    runs; (b) Schutze context-centroid cosine (rung "centroid") ranks the
    LLR survivors by shared company, not raw co-occurrence; (c) a skew
    diagnostic over the survivors' cosine scores decides whether the
    centroid ranking is trustworthy -- ONLY when it trips (|skew| >
    skew_trip, and only past min_skew_n survivors) does Mann-Whitney AUC
    re-rank the top_k (rung "auc"); otherwise the centroid ordering from (b)
    stands untouched. WHERE a run has no dense space (W5), the ladder
    degrades to the LLR ordering (rung "llr", cos/auc None) rather than
    failing -- rungs (b)/(c) are a dense-only refinement, never a
    requirement. The v0 nomen pool is a df-band + stoplist floor (df in
    [min_df, max_df_frac*n], len > 2, not stoplisted); true PPMI demotion of
    the high-frequency band is LATER, once entities v0's ppmi table exists.

### 6.17 Named graph metrics: the centrality lane (2026-09-03)

Operator intent (book-adoption campaign): the graph has carried degree and DWPC and
nothing else named. This adds the standard measures, each one named and pinned, so a
later "which chunk actually matters" question is answered by a metric with parameters
on the record rather than by whatever was convenient.

**Node metrics (W18), whole-run, cached per run_id.** Betweenness — exact at or below
2000 nodes, and the k-sample estimator with k=512 and seed=20260903 above it (mixed-
full-dual is 10.8k nodes / 649k edges; exact is not affordable and an unseeded sample
is not a measurement). PageRank — alpha 0.85, tol 1e-08, max_iter 200, weighted by
edge strength. Triangles and the clustering coefficient — unweighted. Per-provenance
degree split — sparse / dense / both, read from edge_sym's provenance column, summing
to the node's degree. Betweenness and clustering are deliberately unweighted:
networkx reads `weight` as a distance, and our strengths are similarities.

**Community diagnostics (W19).** Per stored Louvain cid: density and conductance,
same algebra as the walk-subgraph shape in §6.11 so the two are comparable — until
now conductance existed only for the WALKED subgraph. Whole-run WCC count and sizes
come with them, because a partition of a shattered graph is a number without a shape,
and the run-wide summary is median and p90 (community sizes are heavy-tailed; a mean
would report the tail).

**Personalized PPR beside DWPC (W20).** pathways() gains a `ppr` column: personalized
PageRank restarting on the walk's anchors (restart mass 0.15, tol 1e-10), per node and
symmetrized per pair. It is ADDITIVE — DWPC still orders the pairs, every existing key
keeps its name, and local_medoid's selection (W11) is untouched. Nothing consumes the
new numbers yet; surfacing them in the digest, the mirror and the walker is T22, and
any behaviour swap is a later measured decision, not a side effect of adding the
measure.

Guards: W18, W19, W20 in graph_tools. Guard bodies below are copied verbatim from the
shipped docstrings -- spec text and docstring are the same bytes, the §6.15 convention.

W18 node_metrics() SHALL compute the centrality lane over the WHOLE run and cache
    it by (run_id, k, seed); `ords` filters the returned view, never the
    computation, so two callers asking about different chunks get numbers off the
    same graph. Betweenness is exact at or below BETWEENNESS_EXACT_MAX nodes and
    the k-sample estimator above it, with k and seed PINNED as constants -- an
    unseeded sample would make the number a coin flip and the cache a lie.
    Betweenness and clustering are computed unweighted: networkx reads `weight`
    as a DISTANCE and our strengths are similarities, so passing them would make
    the strongest edges the longest. PageRank, where higher weight genuinely is
    closer, uses strength, with alpha and tol pinned. Per-provenance degree
    (sparse/dense/both) is read from edge_sym's provenance column and SHALL sum
    to the node's degree.
W19 community_metrics() SHALL score the STORED partition -- density and
    conductance per cid read from community.members, never a re-partitioning of
    an induced subgraph -- and SHALL report the whole-run WCC (component count
    and sizes) beside it, because a modularity partition of a shattered graph is
    a number without a shape. The run-wide summary is median and p90, not a mean:
    community sizes are heavy-tailed and a mean reports the tail.
W20 Personalized PageRank from the walk anchors is an ADDITIVE named column in
    pathways(): every pair keeps its dwpc, the pair ordering stays DWPC's, and
    `ppr` rides beside it. Two measures of connectedness, named separately, so a
    later swap is a measured decision rather than a silent one. Nothing consumes
    these metrics yet -- surfacing is T22's job.

**Surfaced (T22, 2026-09-03).** The three consumers named above now read the metrics.
render_digest() takes an optional `metrics` map ({cid: community row}) on the T11 resolver
pattern -- absent, the digest is byte-identical; present, each communities row gains
`|d=<density>|c=<conductance>` and the header says so. The pathways section switches to
`dwpc/ppr` when the pairs carry W20's column, gated on the data rather than on a flag, so
DWPC still orders the pairs and a pre-W20 caller renders unchanged. The mirror persists both:
PATHWAY.ppr (X14) and CommunitySummary.density/.conductance (X15) -- on the community NODE,
not on TOUCHED, since hits belongs to a walk and the partition metrics belong to the run. The
walker enriches walk_state's `touched` rows once per run from community_metrics() and shows
d/c in the community-map hovertext; PPR is not drawn there, because the walker draws chains
and PPR is a pair quantity -- it reaches the operator through the digest and the mirror. No
ordering, selection or weighting changed anywhere; the "behaviour swap is a later measured
decision" clause above still stands unexercised.

Campaign do-no-harm re-run (T23, 2026-09-03): frozen diagnostic 18/20 on mixed-full-dual, pass set unchanged (FAIL=[E3], KNOWN-FAIL=[A2]), B 48-52% intact.

### 6.18 Ask the mirror: text2cypher at serve time (T20, 2026-09-03)

The mirror (§6.16) is now rich enough that the interesting questions are not the ones
render_digest was built to answer. T15 wrote those questions down as a cookbook --
`cookbook/evidence_queries.cypher`, headers in operator English, bodies in Cypher --
which is exactly the shape of a few-shot bank. So the cookbook becomes the prompt, and
a model writes the seventh question's query.

**The model generates queries, never graph content.** Rows come back from neo4j
unedited; the query is shown beside them so any answer can be refuted by re-running it.
Same posture as the draft labels (R5): model-authored, checkable, never an input to
anything upstream. Narration over the rows is opt-in and marked draft; the module's
deliverable is question -> (query, rows).

**Read-only is enforced by parse, not by trust.** The generated string is normalized
(string literals blanked, then comments stripped) and scanned for write clauses before
it goes over the wire; a hit is a rejection, and the rejection text is fed back as a
retry. A read-only *credential* would be the structural version of this and is the
right eventual answer -- the compose file ships a single superuser, so the parse gate
is what exists today, and it is stated here so the weaker guarantee is not mistaken
for the stronger one.

**The schema is a hand-written constant, not `apoc.meta.schema`.** Introspection
describes whatever happens to be in the database -- a half-finished import, a stale
run, a walk someone wrote by hand -- and makes the prompt a function of database
state. The constant describes what export_neo4j's writers *guarantee*, which is the
thing the query should be written against. The cost is that it must be amended when a
writer's shape changes; that cost is named in Q3 so it lands in the same task that
changes the writer (T22 adds CommunitySummary.density/conductance and PATHWAY.ppr --
the constant moves with it).

**Errors are the loop.** A failed query plus neo4j's own message go back into the next
generation, at most three attempts, then an honest structured failure with every
attempt kept. State is a plain dict through generate -> check -> execute -> retry; no
framework, no agent loop, no new dependency. Transport is interpret._call (OpenRouter
then Ollama, I5) and export_neo4j._tx (X9's wire), both injectable so the whole loop
tests offline.

GUARDS (EARS) -- copied verbatim from text2cypher.py's docstring, the §6.15 convention.

Q1 A generated query SHALL be checked for write clauses and rejected BEFORE it is
   posted. The check SHALL run on the query with string literals blanked and
   comments stripped, so a write word inside a text literal is not a false
   rejection and a write clause hidden behind a comment is not a false pass.
   This module SHALL never construct a write statement of its own.
Q2 Generation SHALL be capped at MAX_ATTEMPTS attempts. A failed attempt -- bad
   JSON, a read-only rejection, or a neo4j error -- SHALL be recorded with its
   query and its error and fed into the next attempt. Exhaustion SHALL return
   ok=False with every attempt kept, and SHALL NOT raise (I4's posture).
Q3 The schema SHALL be a hand-written constant mirroring export_neo4j's writers,
   never apoc.meta introspection: the prompt must not be a function of database
   state. WHERE a writer's node or relationship shape changes, the constant SHALL
   be amended in the same task.
Q4 The few-shot bank SHALL be parsed from cookbook/evidence_queries.cypher at call
   time, never inlined as a copy. The cookbook is the one source of truth for what
   a good query against this mirror looks like (T15).
Q5 The model SHALL produce Cypher only. Rows SHALL be returned as the database
   gave them; narration is opt-in, marked draft, and never edits a row.
Q6 The schema text SHALL state that Chunk.id is a STRING. The CSV import runs
   --id-type=STRING (X9), so a generated query comparing c.id to an integer
   matches nothing and returns zero rows with no error -- a silent wrong answer,
   which is the worst failure this module can have.
Q7 Exactly ONE statement SHALL be executed per attempt. A generated string
   carrying a second statement SHALL be rejected, not split.

### 6.19 Alias-aware search: resolved entities expand the query (T21, 2026-09-03)

T19 built entity resolution (E6-E8): a candidate pair passes a blocked string-similarity
gate, then survives only where it also shares co-occurrence neighbors, and the merged
component's canonical_id is the run's alias table. Nothing downstream read that table
until now -- `search()` still tokenized the query and matched it against `entities.name`'s
own vocabulary with no bridge between a surface form and its siblings.

Expansion is opt-in (`expand_aliases`, default OFF) because it changes what a query
matches, and a change to matching semantics is exactly the kind of thing that must be a
measured decision, not a silent default. It is additive-never-displace: an alias-bearing
token contributes its siblings as EXTRA OR-terms at full BM25 weight, and no original
query term is ever dropped or discounted. The full-weight choice is deliberate --
BM25 already prices a rarer alias down through idf and saturates its tf, so a
hand-picked expansion discount would stack a second, unmeasured knob on top of the one
the scorer already applies, and this campaign has three measured wins for
additive-never-displace and none for re-weighting (see additive ring injection, S17/S18).

GUARDS (EARS) -- copied verbatim from graph_tools.py's and sampler.py's docstrings,
the §6.15 convention.

W21 Alias expansion SHALL be opt-in (`expand_aliases`, default OFF) and ADDITIVE:
    a query token that is itself an entity surface form SHALL contribute the other
    names sharing its `canonical_id` as extra OR-terms at full BM25 weight, and NO
    original query term SHALL be dropped or reweighted. Matching SHALL be exact
    against `entities.name` -- `tokenize` and the entity vocabulary are the same
    vocabulary, and a looser match reopens the false merges E7 exists to close.
    WHERE resolution has not run for a run -- no `entities` table, no rows, or
    `canonical_id` NULL -- the map SHALL read back empty and search SHALL be
    byte-identical to the unexpanded call, never an error.

S19 The sampler SHALL pass `expand_aliases` through the ANCHOR path only
    (`ef_evidence` -> `ef_search` -> `anchor_hits` -> `gt.search`, and
    `candidate_scores`'s legacy call), defaulting OFF, and SHALL record it in
    `Bundle.params`. The S18 ring injection (`source_topk`) SHALL NOT expand: two
    term sets changing in one measurement makes a diagnostic delta unattributable.

MEASURED (2026-09-03) -- one no-op result and one real one, and why both are honest

`entities` is populated for exactly one live run, `mixed-smoke` (23,925 entities,
87,707 mentions, 596,283 entity_edges); `canonical_id` was NULL on every row there
before this task (T19 resolved only planted fixtures). `mixed-full-dual` (10,830
chunks) has no entities rows at all and cannot get them inside Article VII's bound:
`entity_edges.pair_counts` enumerates every co-occurring term pair per chunk, which is
2.47e9 pair slots at the default MIN_JOINT_CHUNKS=5 on this corpus, and raising the df
floor barely helps -- 1.75e9 at df>=50, 1.45e9 at df>=100, 1.09e9 at df>=200. There is
no floor that makes `build_entities` terminate here in minutes; a pair-enumeration path
that scales to a 10.8k-chunk corpus is a follow-up for `entities.py`, on its own ledger
row, not T21's.

Diagnostic baseline (flag OFF, `mixed-full-dual`, frozen 20-row set): 18/20, FAIL=[E3],
KNOWN-FAIL=[A2] -- the same pass set the ledger already records. Flag ON, same run: 18/20,
byte-identical results row for row. This is the expected outcome from step 0's finding
above (no entities table means the alias map reads back `{}` and W21's degrade-to-empty
guard makes it a no-op) -- **it is recorded as unmeasured-vacuous, not measured-neutral,
and the default stays False.** A no-op cannot be evidence for a flip.

MEASURED (T25, 2026-09-03, after T24/E9 populated mixed-full-dual): with the entities
table present (275,328 entities, resolution restricted losslessly to the 65 edge-bearing
terms per E9's amendment), resolution finds 0 merged pairs and 0 alias groups on this run
-- the E9-bounded pool is the highest-df common vocabulary, which carries no
near-duplicate surface forms. Diagnostic baseline and flag-ON are both 18/20, FAIL=[E3],
identical row for row. This is now a REAL measurement of the shipped configuration:
alias expansion is a measured no-op on mixed-full-dual at the current E9 bound, and the
default stays False. The lever that would change this is widening E9's VOCAB_BOUND (a
spec decision with a build-cost trade-off), not the expansion mechanism, whose behavior
is proven on mixed-smoke (905 alias groups; "aboard"->"board" surfaced 5 new chunks).

`resolve_entities` was run once against `mixed-smoke` (bounded invocation from `.tmp/`,
`entities.py` untouched) to give the alias table real rows: the plain call completed in
90.3s, well inside the 8-minute bound, over the full 23,925-entity population --
{entities: 23925, candidates: 4995, merged_pairs: 505, clusters: 23426, aliases: 499}.
`gt.alias_map` on that run now returns 905 entities carrying >= 1 alias, e.g. `aboard ->
(board,)`, `according -> (accordingly,)`. The real behavioural evidence: query "aboard"
(k=10) returns ords {610, 580, 584, 553, 11, 12, 558, 591, 628} unexpanded; with
`expand_aliases=True` the same query returns {610, 3, 5, 6, 552, 12, 558, 591, 628, 601}
-- 5 new ords (3, 5, 6, 552, 601) surfaced via the "board" sibling, and ord 610's own
score rose from 4.501 to 5.770 as the alias term's postings stacked onto it. Expansion
does something real where the alias table exists; the default stays False because the
20-row diagnostic corpus is not that run.

Campaign do-no-harm re-run (T23, 2026-09-03): frozen diagnostic 18/20 on mixed-full-dual, pass set unchanged (FAIL=[E3], KNOWN-FAIL=[A2]), B 48-52% intact.

### 6.20 Relations v0: unsupervised relation detection over sentence windows (T27, 2026-09-05)

The gap this closes: entity_edges (E5) is symmetric, typeless, CHUNK-window
co-occurrence -- there is no directional or typed association anywhere in the
store, and gt.tokenize destroys the very tokens a relation needs (possessive
's clitics, closed-class connectors, case, order). The raw material exists:
node.body stores every chunk verbatim, so this layer runs over EXISTING runs
with no re-ingest. Approach per the operator's brief: Dunning + co-occurrence
around connector phrases, fully unsupervised, no pretrained models, no LLM --
the relation label is the corpus's own connector string, not an ontology.

Module: relations.py (new; justification: entities.py's E3 seals its builder
as read-only-against-attrs, and gt.tokenize is W21-frozen -- both firewalls
are structural only if relations lives beside them, not inside). Table:

    relations(run_id uuid CASCADE, src int, dst int, template text,
              connector text, n int, llr real, npmi real, example_ord int,
              PRIMARY KEY (run_id, src, dst, template),
              FKs (run_id, src)/(run_id, dst) -> entities)

DDL idempotent in-module (entities pattern); build = DELETE-by-run + COPY (E3
pattern). Constants: MAX_CONNECTOR_TOKENS=4 (relation phrases fit; doubles as
the pair-distance cap -- one knob), MIN_REL_SUPPORT=3 (below 3 the 2x2 is
rare-event dominated; lower than E4's 5 because the G2 gate is a second
independent floor chunk edges never had), G2_GATE=10.83 (chi^2 p<.001,
byte-identical to W17's gate -- one significance ruler), MIN_SENT_TOKENS=3
(fragment floor catching abbreviation splits the guard list misses).

Guards (ASCII bodies pasted into relations.py's docstring unchanged -- spec
text and docstring are the same bytes, the 6.15 convention):

E10 Relations v0 SHALL be built from node.body verbatim -- the one store that
    preserves case, punctuation, and order -- via a per-source deterministic
    sentence splitter (line-per-sentence for brown/quotes; regex boundary scan
    with an abbreviation guard list, a MIN_SENT_TOKENS fragment floor, and
    wikitext @-@/@,@/@.@ normalization for wiki and unlabelled sources) and a
    NEW order-, case-, and clitic-preserving tokenizer rel_tokenize.
    gt.tokenize and chunkgraph._tok SHALL NOT change: W21 depends on the query
    tokenizer and the entity vocabulary being the same vocabulary. Relations
    SHALL be a NEW run-scoped table; the E1 tables' shape survives untouched --
    no ALTER.

E11 Candidates SHALL be ordered pairs of entity occurrences within ONE
    sentence whose intervening surface span is at most MAX_CONNECTOR_TOKENS
    (= 4) tokens -- one constant, serving as both connector bound and
    pair-distance cap. Entity occurrences SHALL be anchored by greedy
    longest-match of the run's stored tf vocabulary over the sentence's
    gt-eligible projection (losslessly reconstructing fit-time Phraser bigram
    merges, which merged over the stopworded stream), and SHALL aggregate over
    canonical_id, falling back to entity_id where resolution has not run --
    never an error.

E12 X's Y and Y of X SHALL normalize to ONE direction-normalized genitive row
    (src=X, dst=Y, template='GEN'). For every other template, direction SHALL
    be surface order: src precedes dst. The relation label is the corpus's own
    connector string -- no ontology, no label inventory. The template tier
    collapses non-closed-class connector tokens to 'w' (e.g. 'was w in'); the
    exact surface connector rides beside it (most frequent form, ties by
    ascending string -- deterministic).

E13 A relations row SHALL be written only where BOTH floors pass: n >=
    MIN_REL_SUPPORT (= 3) supporting sentences AND Dunning G2 >= 10.83
    (gt.llr, W17's gate, one ruler) on the pair x template 2x2 over the run's
    candidate events. Sentence-level NPMI (via entities.npmi_ppmi, sentence
    dfs, N = run sentence count) SHALL ride on the SAME row, so ranking is a
    column choice, never a rebuild (E5's law restated).

E14 The builder SHALL print per-stage wall-clock timings (fetch / split /
    tokenize+match / candidates / score / write) on EVERY build, and NO
    cardinality bound (an E9-style cap) SHALL be added without a measured
    stage exceeding Article VII's budget on a live run. E9's scar cuts both
    ways: chunk windows needed the bound only after 2.47e9 measured slots;
    sentence windows (~10-20 content tokens, gap-capped pairs) are expected
    orders smaller, and bounding an unmeasured pass is as much a sin as not
    bounding a measured one.

Deferred, named [LATER]: neo4j RELATES {template, connector, llr} mirror edge;
cookbook evidence query; digest relations section. v0 surfaces via the CLI
(top-15 by llr with resolved names) only.

Reconciliation (T29, 2026-09-05, spec follows shipped -- Article IX). Two E10/E12
clauses shipped MORE unsupervised than drafted, per the operator's brief: (a) the
abbreviation "guard list" is not a hand-typed set but derive_abbreviations(), a
census over the run's own text (a type is an abbreviation where it never appears
bare -- existence, not a tuned ratio; single-letter and internal-dot forms fall
out structurally); (b) there is no CONNECTOR_CLOSED word set -- closed-class is
defined as NOT gt-eligible via one is_content predicate (alpha, len>2, not in
the R18 stoplist), used identically for the vocabulary projection, the template
'w' collapse, and the genitive test ("starts with of, no content token").

MEASURED (first live build, mixed-full-dual): sentences=918,707,
candidate events=22,707,085, distinct pairs=14,547,626, relations rows=328,825.
Per-stage: fetch 3.5s / split 76.3s / tokenize+match 285.6s / candidates 138.5s
/ score 197.5s / write 11.0s -- total ~11.9 min. No stage exceeds Article VII's
per-test bound; the total slightly exceeds 10 min but this is a per-run batch
build (ingest-class, not test-class), run once per run label. E14 disposition:
NO bound taken; if a future corpus pushes tokenize+match past budget, measure
per-source first. Known v0 ranking caveat: raw LLR favors high-frequency
collocations ("later that year"); npmi rides on every row so semantic ranking
is a column choice (E13), and template-level filtering (e.g. GEN-only views)
already isolates the possessive/genitive class the layer was built for.

### 6.21 Refactor contracts: evidence assembly, config, walker split, test taxonomy (T30, 2026-09-05)

Four contracts governing the refactor campaign (plan:
C:/Users/user/.claude/plans/luminous-roaming-dongarra.md). Spec lands first;
each contract is the acceptance criterion for its implementing task.

(a) evidence.assemble() -- ONE assembly for walker, diagnostics, backfill.
    The ~40-line evidence-assembly block (communities_touched -> kw/concept ->
    cid_of/src_of/src_counts -> subgraph_embeddings -> dendrite planes ->
    salient -> resolver -> pathways -> render_digest inputs) exists 5x today.
    evidence.py SHALL own it as a pure function of (conn, run, bundle, embed);
    walker caching wraps it, never re-implements it. ACCEPTANCE: render_digest
    output byte-identical across the frozen 20-row diagnostic before vs after
    the extraction -- the pass-count alone can mask reordering, so the gate is
    bytes, not verdicts. Layout-only code (spring positions) stays in the UI.

(b) config.py -- one source for environment facts (config indirection). DSN (3 copies today),
    neo4j endpoint/db (threaded through 6 signatures), model dir (4 sites,
    2 divergent defaults -- the winner is ~/models/m2v-minilm-l6-256 expanded
    per-user, matching walker_app's DEFAULT_MODEL_DIR), ports, cache dir.
    Consuming modules SHALL keep module-level names (DSN = config.DSN) so
    existing monkeypatch tests keep their patch points. The neo4j BASIC-AUTH
    plumbing is DEPRECATED: the local container runs NEO4J_AUTH=none; _tx
    SHALL omit the Authorization header when auth is None (default), and the
    hardcoded credential pair leaves the tree.

(c) walker split -- walker_app.py becomes UI wiring only. Extracted logic
    (digest/resolver/mirror block) SHALL be importable with no DB, no
    Streamlit session, and no SystemExit; mirror writes live beside
    assemble() in evidence.py; the mirror catch narrows from Exception to
    (URLError, RuntimeError, OSError) so programming errors surface instead
    of reading as an unreachable server.

(d) test taxonomy -- markers live_db / live_net / slow registered in
    pytest.ini; default addopts excludes live; the 10 inline
    pytest.skip("no database") sites become markers with a graceful fixture;
    conftest.py kills the 16 sys.path.insert copies. pg_store.save gains an
    injectable now= (deterministic clock) so the resave supersede test stops
    racing the container clock (measured flake: host-vs-docker drift).

### 6.22 Presentation layer: the one-tab analysis view (T38, 2026-09-05)

Operator directive: "hard to make sense of the key subgraph analysis" -- the
walk's partition evidence is scattered across two tabs and four sub-tabs.
This section is the layout and computation contract for the consolidation.

**P1 (single tab).** WHERE the walker renders, there SHALL be exactly TWO
top-level tabs: **Analysis** (everything) and **Neo4j** (the mirror iframe,
alone). The Neo4j tab is a declared temporary exception to the one-tab rule:
the embedded browser is buggy; WHEN it is fixed, the mirror section returns to
Analysis and the tab is removed. One prompt input drives the whole app (T26
rule stands).

**P2 (order).** WITHIN Analysis, images PRECEDE partitions: (1) the graph
figures (walk communities, term graph, 3D, global map), then (2) the partition
text blocks (correlation-sorted chains, louvain panels, entity/relation
classes), then (3) evidence/digest. Rationale: the operator scans shape first,
reads text second.

**P3 (dwpc term ranking).** WHERE a louvain group's terms are listed, they
SHALL be ranked by DWPC mass: node_dwpc(o) = sum of pairs[i].dwpc over every
pathway pair whose best path (gt.pathways W15 output) contains chunk o;
term_dwpc(t) = sum of node_dwpc(o) over member chunks o whose salient vocab
carries t. Ties break by BM25 salience then lexicographic. Chunks outside
every best path contribute 0 -- a term with term_dwpc 0 ranks below any
positive term but keeps its BM25 order within the zero class.

**P4 (relative vs global louvain).** The Analysis tab SHALL show two louvain
panels side by side: LEFT = louvain re-run on the walked subgraph only
(python-louvain, fixed seed, edge weight = strength); RIGHT = the global
ingest-time cids restricted to the walked chunks. The left panel is a VIEW
ONLY: its partition ids are ephemeral, never persisted, never joined, never
carried across reruns (steering: subgraph louvain is not a restriction of
global louvain). The panel header SHALL say "relative (this walk only)" so the
non-canonical status is visible on screen.

**P5 (classes).** Per louvain group (both panels), the view SHALL list:
member chunks (ord, source, doc_id), terms ranked per P3, entities ranked by
lift = (mention share in group) / (mention share in corpus) with a mention
floor, and relation templates with counts scoped to pairs whose src AND dst
entities are mentioned in the group's chunks. Queries live in evidence.py /
walker_core.py, NEVER inline in walker_app (6.21(c) walker split holds).

**P6 (shading).** Groupings SHALL be visually separated by background color:
in figures, a translucent convex-hull or rect shape behind each community's
nodes (plotly layout.shapes, alpha <= 0.15, same hue as the group's node
color); in tables, per-row background by group id (styled dataframe), same
palette. One palette, defined once, shared by figures and tables so a group
is the same color everywhere it appears.

**P7 (no recompute).** Every panel is a rendering of already-computed state
(walk_state / dendrite_state caches). The subgraph louvain of P4 runs at most
once per (run, prompt) and is cached with the same key. No panel issues its
own DB queries at render time beyond the evidence.py helpers.

**P8 (auto LLM, operator 2026-09-05).** WHEN a walk completes, the LLM
reason-over-walk call SHALL fire automatically -- no manual button. The call
remains cached per (run, prompt) like every other panel (P7), so a rerun on
the same prompt replays the cached narration rather than re-invoking the
model. A visible spinner/status line covers the in-flight window; failure
renders as a warning line, never a blank panel.

**P9 (gestalt, operator 2026-09-05: "seeing is believing").** The Analysis
tab's layout SHALL lean on gestalt principles: ENCLOSURE (P6 shading bounds a
group), PROXIMITY (a group's figure, terms, entities and relations sit
adjacent, never split across distant sections), SIMILARITY (one palette, one
hue per group everywhere -- P6), CONTINUITY (spring layouts for relational
figures so connected things read as connected; tables only where items are
enumerable facts). Where a figure and its table describe the same grouping,
the figure comes first and the table sits directly beneath it.

**P10 (panels, T42 2026-09-05).** The Analysis tab SHALL render as bordered
panels (st.container(border=True)), in order: (1) Answer, (2) Judged evidence,
(3) Groups -- the figure sub-tab strip (This walk / Term graph / 3D / Global
map) at the top of the panel, then the side-by-side LH/RH spring figures with
hulls (P4/P6 unchanged -- operator-approved as shipped), then the per-group
factbook digests in the SAME panel, largest group first, (4) Partitions
(correlation sorting) followed by each chain's own community make-up. Only
the tabular detail beneath the figures changes; the figures do not.

**P11 (factbook digests, T42).** Per-group detail SHALL be compact pregrouped
text (st.code, TOON-style key: value lines), NEVER dataframes. Format:

    g0 . 36 chunks . wiki:30 brown:5 quotes:1 . = c6 (global)
    terms(dwpc): hurricane 233.9 . gulf_mexico 187.2 . ...
    entities(mentions): wakulla 8 . hurricane_humberto 5 . ...
    relations: declared -[disaster]-> area corpus_n=12 . ...

Rules: (a) every dwpc-ranked term shows its score to 1 decimal (operator's
explicit ask); (b) entities rank by group mention count, count shown; lift
renders as x{lift:.1f} ONLY when the entity also appears outside the group
(corpus_cnt > group cnt) -- at the saturation ceiling (walk-exclusive entity,
lift == corpus_total/group_total for all such rows) it is omitted because it
discriminates nothing; (c) when a local (relative) group's member set equals a
global cid's restriction to the walk, ONE digest renders, carrying the
"= c<cid> (global)" marker -- never two duplicates; (d) relation lines render
example pairs among the group's own mentioned entities, capped ~8, with the
corpus-wide support labelled corpus_n= so a global count is never read as
walk-local.

**P12 (partition communities, T42).** Each dendrite chain row SHALL be
followed by its own community line: the global cids its member chunks belong
to, with counts, compact digest style (e.g. "c6:31 c0:4 c12:1").

P11 addendum (operator: "think triz + bono's six hats"): (e) DIVERGENCE is
first-class -- when a local group's member set is NOT equal to any single
global cid restriction, its header line decomposes it over the global cids it
intersects, naming the relationship: "g1 = c0:28 + c12:8 (splits c0)" or
"g2 = c3+c7 (merges)". The disagreement between relative and global louvain
is the signal the side-by-side exists to surface; it SHALL be readable in the
text digest, not only inferable from the figures. (f) digests wrap at a fixed
width (~110 cols) and long tails truncate with a count ("+ 14 more"); at most
8 group digests render inline, the rest behind an expander. (g) the digest
text is the same artifact a future LLM reasoning pass consumes -- pregrouped
text over pixels; no information may live only in the figures.

**P13 (shareable prompt URL, operator 2026-09-05).** WHEN the page loads with a
?q= query parameter and the session has no prompt yet, the prompt input SHALL
initialize from it; WHEN a walk runs, the URL's ?q= SHALL be updated to match,
so any session's current view is a link another browser session can open and
see the same populated results (walk/LLM state is cached process-wide per
(run, prompt), so the second session renders from cache).

**P14 (colored cards, one figure -- T46, 2026-09-05; supersedes P4's
side-by-side FIGURES and P11's st.code rendering; every other P4/P11 rule
stands).** (a) The Groups panel SHALL show ONE spring figure, full width: the
walked subgraph in GLOBAL cid colors with hulls. The relative-louvain figure
is removed; its computation stays, feeding the splits/merges divergence text
in the digests -- the text is where the comparison earns its keep. (b) Every
group digest renders as a COLORED CARD: background = the group's figure color
at alpha ~0.10, left border 4px solid full hue, color chip + id header,
monospace body -- same group_color palette as the figure, one hue per group
everywhere. Local-only groups take the hue of their dominant overlapping cid.
(c) The four section panels carry distinct accents (Answer green, Judged
evidence blue, Groups violet, Partitions amber). (d) Each partition chain row
is tinted by its dominant community's color. (e) Cards are HTML via
st.markdown(unsafe_allow_html=True) with ESCAPED content; the digest text
stays byte-identical to group_digest's output (P11(g): the LLM artifact and
the human view never fork).

**P15 (dark dashboard skin -- T49, 2026-09-05; a SKIN over P10-P14, whose
content contracts all stand: digest text byte-identical, one figure, dedup,
dwpc scores, chain lines).** Reference: operator-supplied dark SaaS dashboard.
(a) THEME: .streamlit/config.toml sets base="dark", near-black canvas
(~#0f1117), darker sidebar, violet primary (~#7c5cff); plotly figures use
template "plotly_dark" with transparent paper/plot backgrounds so cards show
through. (b) KPI STAT ROW: directly under the prompt, four gradient stat
cards (icon, big number, one-line caption, thin accent bar): chunks walked,
communities touched, entail/contradict counts, model + briefs/judged counts.
Numbers come from already-computed state (P7: no new queries). (c) CARD GRID:
group factbook cards render in a responsive st.columns(3) grid (largest
groups first, row-major), each card: header row (color chip + id + size),
BADGE PILL for the global relationship (= c6 -> solid pill in the group hue;
splits/merges -> outlined amber pill), monospace digest body, hue accent
bar at the card top. (d) JUDGED EVIDENCE ROWS: each judged chunk renders as a
dark rounded row -- id + source + community chip left, snippet middle, a
colored pill right (entails green, contradicts red, neutral gray) with the
walk score. (e) ANSWER HERO: the answer renders in one hero card: large
type, citation ids as small violet pills, model line as muted caption.
(f) All colors from ONE palette module (walker_core): the group hues stay
P6-consistent; UI chrome hues (violet primary, green/red/amber status) are
named constants beside them, not scattered literals.

P14(a) REVERSED (operator, 2026-09-05: "the way they were left to right was
very informative"). The Groups panel returns to TWO side-by-side spring
figures -- relative louvain LH, global cids RH, per the original P4 layout --
AND keeps the splits/merges divergence text in the digests. Both views ship;
neither substitutes for the other. Every other P14 rule (colored cards,
section accents, tinted chain rows, escaped HTML) stands.

**P16 (interactive 3D walk explorer -- T52, 2026-09-06).** The "3D layers"
sub-tab renders an embedded 3d-force-graph scene (streamlit components.html
iframe; the BROWSER loads pinned CDN scripts: 3d-force-graph and
three-spritetext, exact versions written into walk3d_html and pinned by
test). Contract: (a) nodes = walked chunks, color = stored cid via the one
palette, size ~ walk score, NO labels by default; (b) hovering a node shows a
tooltip card -- ord, source/doc, cid, walk score, first ~200 chars of body --
content hidden until hover; (c) one always-visible SpriteText per community
in the walk at its member centroid: "cN: t1 t2 t3" from community keywords,
falling back to the walk's salient terms; text ~3x the old 9px labels;
(d) hovering a node or sprite dims every other community's nodes and links
to ~15% opacity; (e) DWPC best paths (ws.pathways pairs' path lists) render
as thick colored links with an in-iframe control cycling all/one/off -- the
control is scene JS, never a streamlit rerun; (f) plain links default to
~0.12 alpha; (g) POSITIONS ARE A VIEW: the in-browser force sim is not
deterministic and is never exported or persisted -- deterministic layouts
remain the 2D figures and graph3d.py; (h) degradation: no pathways -> path
control hidden; CDN unreachable -> the iframe body shows a one-line failure
message, never an unexplained blank; every string in the payload destined
for HTML is escaped at build time.

P16 addendum (operator, 2026-09-06: "I also like umap projection").
(i) WHEN the run is dense, the scene carries a second position set per node:
UMAP(n_components=3, fixed random_state) over the walked chunks' stored
embeddings (umap-learn 0.5.x, already installed), computed once per
(run, prompt) beside the other cached state; an in-scene toggle animates
between force-sim positions and the UMAP projection (TF-projector view).
Sparse-only runs hide the toggle. The projection, like the force sim, is a
VIEW (P16(g)) -- never persisted, never a join key.

**P17 (no sidebar -- operator, 2026-09-06, repeated ask).** The walker SHALL
render WITHOUT a Streamlit sidebar. The run selector, run stats line and
details expander move to a single compact row at the top of the main area
(selector left, stats muted beside it, details in a popover); degraded-mode
warnings render inline under that row. Nothing else may reintroduce
st.sidebar.

**P18 (feng-shui zones -- T57, 2026-09-06; supersedes P10's ORDER clause;
P10's bordered-panel, accent, and factbook contracts stand).** Every element
of the Analysis tab belongs to exactly ONE category, and placement follows
category: VERDICT (what the corpus said: hero answer, hypothesis+premises,
KPI stat row), EVIDENCE (why believe it: judged pill rows, neutral expander,
reason-vs-judge, raw model I/O), STRUCTURE (shape of the walked knowledge:
3D explorer, 2D walk spring, LH/RH louvain pair, term graph, factbook grid),
LENSES (alternative readings: dendrite partitions + chain communities + term
chains), REFERENCE (run-level, prompt-independent: community map + quotient
links + run stats/details). The legacy medoids/anchors/one-degree-out block
is a WALK TRACE -- diagnostic, not evidence. Layout: Zone 1 = VERDICT left |
EVIDENCE right (st.columns ~[1.1,1]); Zone 2 = STRUCTURE full width, figure
sub-tabs with 3D FIRST and default, louvain pair as its own sub-tab,
factbook grid beneath in the same panel; Zone 3 = LENSES left | REFERENCE
right (~[1.4,1]), the community map sitting beside the chain lines that cite
its cids; the redundant global-map image tab merges into the community-map
slot; the walk trace renders LAST as a collapsed expander. RULE: any future
element names its category BEFORE it lands; an element with no category has
no place on the tab.

### 6.23 Agentic retrieval v0: ReAct over the walk (T60, 2026-09-06)

Operator directive after the "most famous musician of the 1990s" failure: the
system asserted a confident answer from a walk with 0 entailing chunks (85/88
neutral). Single-shot retrieval tuned one walk; nothing re-tried.

**A1 (determinism boundary preserved).** Every INDIVIDUAL walk stays
deterministic and attributable (sampler.ef_evidence with explicit params,
logged). The agent chooses parameters BETWEEN walks; it never edits chunks,
edges or communities. A transcript of (params, walk stats, judgment) per
iteration is part of the result.

**A2 (sufficiency judge).** WHEN a walk's evidence is assembled, an LLM call
SHALL judge sufficiency: input = the prompt + the digest + entail/contradict
counts; output = {sufficient: bool, missing: [what info is absent],
proposal: next action}. Zero entails SHALL force sufficient=false without
spending the call unless the judge is needed for the proposal.

**A3 (action space).** The proposal picks ONE action per iteration from:
WIDEN (raise ef and ring budgets), REANCHOR (judge-proposed replacement or
additional query terms -- e.g. missing "grunge nirvana" for a 1990s-music
prompt), PIVOT (walk from the strongest adjacent community via quotient
edges), DEEPEN (raise hop/expansion depth). Params map to existing
ef_evidence knobs (ef, k_anchor, ring_top, ring_per, expand_aliases) plus
query text; no new retrieval machinery.

**A4 (budget -- Article VII).** Max 3 iterations beyond the base walk; each
iteration's wall time is bounded by the same walk cost as the base; the loop
stops early on sufficient=true. Iteration count and stop reason are surfaced.

**A5 (accumulation).** Evidence UNIONS across iterations; every chunk carries
the iteration that found it (provenance per iteration). Judged verdicts are
never recomputed for already-judged chunks.

**A6 (answer gate).** WHEN the final evidence has 0 entailing chunks, the
answer SHALL render as insufficient evidence -- "the corpus, as walked, does
not answer this" -- with the iterations tried; a confident claim over zero
entails is a defect. Superlative/aggregate prompts that no single chunk can
entail SHALL say so rather than crown an arbitrary candidate.

**A7 (do-no-harm).** The loop ENGAGES only when the base walk is judged
insufficient (or has 0 entails). A prompt whose base walk is sufficient
behaves byte-identically to today -- the frozen diagnostic (18/20,
FAIL=[E3]) must not move.

**A8 (score-distribution feedback -- operator, 2026-09-06: "the key
information feedback").** Each iteration record SHALL carry the walk's score
statistics: n_chunks, mean and sdev of walk scores, entail/contradict/neutral
counts, and per-community score means. The proposer receives the FULL history
of (params -> stats) pairs, so it infers which knob to perturb from how the
distribution moved under prior settings (e.g. widening ef raised n but
dropped mean walk score -> the frontier is diluting: reanchor rather than
widen again). The history table renders in the iteration trace (A1) --
numbers, not narrative.

**A9 (derived retrieval metrics -- operator, 2026-09-06: ragas-type signal
without overhead).** Per-iteration metrics are DERIVED from verdicts already
produced, never from extra LLM calls: precision_proxy = entails / judged
(clean entailment as the precision signal), contradiction_rate =
contradicts / judged, coverage = communities touched / communities the
proposal targeted, and -- WHEN a gold answer is known -- gold_recall_evidence
(gold terms present in the final evidence bodies) and gold_recall_answer
(gold terms present in the rendered answer). These ride in the A8 history
table the proposer sees.

**A10 (gold diagnostic lane).** A NEW agentic diagnostic suite (G-class rows:
{prompt, gold_terms}) evaluates the graph + agent end to end against
operator-authored gold -- e.g. "most famous musician of the 1990s" ->
gold_terms [kurt_cobain, nirvana]. Scored by A9 gold_recall, both evidence
and answer. Gold rows SHALL be authored against what the corpus verifiably
contains (checked at authoring time); the FROZEN 20-row do-no-harm set is
untouched -- this is a second lane, additive (A7).

**A3 amendment (fixed-point guard -- 2026-09-06, from the first live gold
run: G1 re-ran a byte-identical query 3x because the judge's "missing" echoed
the prompt's own words).** (a) The proposer's `missing` SHALL name candidate
evidence entities/topics ABSENT from the digest, never words of the prompt.
(b) REANCHOR terms are algorithmically filtered against the prompt's own
tokens; WHEN the filter empties them, the action escalates to the next
ladder move. (c) A (query, params) pair identical to ANY prior iteration
SHALL never re-run: the loop forces the next ladder action, and if the pair
is still identical, stops with stop_reason="fixed-point". Receipt:
"i=1..3 action=REANCHOR q=<identical> n=88" / "missing: ['1990s','famous',
'musician']".

**A5 amendment (payload cap -- from the same run: accumulated evidence hit
121K chars and truncated the answer call at max_tokens).** Accumulated
bundles fed to ANY downstream model call SHALL cap at the base-walk scale:
top ~100 chunks by walk score, lowest dropped first. Receipt: "G4 ...
truncated at max_tokens=4096 (finish_reason=length, 11439 chars) | ollama
fallback skipped: 121087 chars exceeds 40000".

**A11 (SNR framing of the feedback signal -- operator, 2026-09-06).** The A8
history is read as a signal-to-noise problem: signal is what repeats
(entailing chunks), noise is what doesn't (the neutral mass);
precision_proxy = entails/judged IS the SNR. Consequences the proposer's
guidance encodes: (a) prefer window-NARROWING actions (REANCHOR with
absent-from-prompt terms = matched filter) over amplification (WIDEN) when a
prior WIDEN raised n but dropped mean score -- dilution admits noise;
(b) iterating the SAME walk is correlated noise and averages out nothing --
the A3(c) no-repeat guard is this law enforced mechanically; (c) an answer
over near-zero SNR is a guess -- the A6 gate is the SNR floor. Receipts:
gold run 1 (WIDEN-free REANCHOR found #6323 at iteration 1; three identical
walks bought zero recall).

**A12 (SNR-guided proposal -- T65, 2026-09-06).** Dilution is detected with
the house ruler: log2 of each iteration's mean_score; the band's lower bound
is the LOWER of the two estimators over the history -- mean - sdev and
median - 1.4826*MAD (the estimator-pair gate, same idiom as the NORMAL
stage and the CHUNK stage's MAD sizing; operator: "the lower of the two").
An action whose resulting log2 mean falls below that bound while n_chunks
ROSE is dilution: it is EXCLUDED from the allowed action set for the rest of
the loop -- enforced algorithmically, and stated to the proposer in
numbers-first form. The log2 scale is load-bearing: equal multiplicative
drops are equal band distances anywhere on the score scale (ratios become
differences); signal ADDITION stays linear-domain (the averaging law) --
log space is for thresholds, linear space for accumulation. The fallback
ladder reorders to REANCHOR -> DEEPEN -> WIDEN: narrowing first, the
matched filter beat amplification in every gold run.

**A13 (answer from loop evidence -- T65).** WHEN the loop terminates with
entailing chunks from ANY iteration, the answer SHALL be synthesized from an
ENTAILS-FIRST bundle: all entailing chunks first (by walk score), remaining
capacity filled by score, total capped per A5. The hero renders this loop
answer marked "answered after N agentic iterations" with citation pills from
the entailing set; A6's insufficient-evidence text renders only when even
the loop ends at zero entails; the superlative caution stays whenever
entails exist but cannot crown a single candidate -- the answer model
answers only what the entailing chunks support. Receipt driving this guard:
gold G4/G5 evid=1.0 ans=0.0 -- the evidence held the gold and the answer,
drawn from a score-capped bundle where neutrals crowd out entails, never
named it.

### 6.24 Unsupervised classes: entities and relations (T69, 2026-09-07)

Operator asks, gathered verbatim: "top entities by class (idk if you would
use npmi, I didn't have defined classes and would prefer unsupervised
classes created however that would look like, my guess was co-occurrence
analysis)"; "we can construct the graph we already have setup, and add
graph based entity resolution but for relations and let whatever
unsupervised relationships that survived be cast to others that measure
some type of co-occurrence threshold between similar terms".

**E15 (entity classes from co-mention).** A batch pass SHALL build an entity
co-mention graph straight from the mentions table: an edge between two
entities that share a chunk, weighted by NPMI over co-mention counts
(reuse entities.npmi_ppmi; N = chunks with >= 1 mention), with a support
floor (>= 3 shared chunks). Louvain over this graph (fixed seed, python-
louvain -- the ingest-time idiom) partitions entities into CLASSES. Stored
additively per the E8/canonical_id pattern: class_id column (or sibling
table), assigned over the FULL entity population -- an unclassed entity is
its own singleton class. Deterministic: same rows, same classes. This
sidesteps the E9-throttled entity_edges entirely -- mentions are already
materialized.

**E16 (relation classes, DIRT).** Two relation templates belong to the same
class WHEN they connect similar (src, dst) populations: similarity = NPMI
(or cosine) over the sets of canonical pairs each template joins, gated by a
threshold and a shared-pair floor; connected components (or Louvain) over
the template-similarity graph mint rel_class ids [empirical:cited -- DIRT,
Lin & Pantel 2001, unsupervised discovery of inference rules from text].
Stored additively beside relations (mapping table template -> rel_class,
run-scoped). The hand-built GEN unification stays; E16 must REDISCOVER it
(of / of the / 's landing in one class is the acceptance check).

**E17 (per-class rankings).** Top entities per class = mention mass within
class; top relations per class = summed n over the class's templates.
Rendered factbook-style (P11 idiom): one line per class, members with
counts, capped with "+ N more".

**E18 (walk-local relation scoping).** Group digests' relations lines SHALL
be scoped to pairs BOTH of whose endpoints are mentioned in the group's own
chunks AND whose template's example pairs intersect the group's entities --
labelled with the walk-local pair count first and corpus_n only as a
suffix. Kills the "second -[]-> world_war corpus_n=527581 in a railway
group" misread (live receipt, factbook v1).

**E19 (cost discipline).** Both passes print per-stage wall clock (E14
idiom); no cardinality bound is added unless a measured stage overruns
Article VII. Estimated: co-mention pairs ~ chunks x C(mentions-per-chunk, 2)
~ low millions; template similarity over ~20 templates x pair sets --
minutes, not hours. The timings are the authority.

**E17 amendment (partition counts -- operator, 2026-09-07: "we need to add
entity and relation counts not just for louvain, but also my correlation
sorted partitions").** Per-class/per-group counts are NOT a Louvain-only
view. Every grouping the walker renders SHALL carry the same three counts --
chunks, distinct entities mentioned, distinct relations whose src AND dst
are mentioned in the group -- and they SHALL be computed by ONE function
over a {group_id -> [ords]} mapping, so the numbers are commensurable
across groupings. The three groupings that get them: (1) global Louvain
cids, (2) relative (walk-local) louvain groups, (3) the CORRELATION-SORTED
dendrite chains (correlation sorting.md) -- which today carry only chunk
counts and salient terms. A chain's row reads e.g.
"chain 1 . 36 chunks . 214 entities . 87 relations" beside its
community make-up line (P12).


**E16/E15 attributions (operator asked for the matching papers, 2026-09-07).**
NPMI as used for entity co-mention edges and relation plausibility:
Bouma 2009, Normalized (Pointwise) Mutual Information in Collocation
Extraction (GSCL); ancestor Church & Hanks 1990. G2 gate: Dunning 1993.
Louvain: Blondel et al. 2008. Pattern-similarity bet: Lin & Pantel 2001
(DIRT). The PARSER-FREE method we actually implement is closer to Hasegawa,
Sekine & Grishman 2004, Discovering Relations among Named Entities from
Large Corpora (ACL) -- surface context between entity pairs, clustered by
cosine, no dependency parse and no POS assumption. Operator constraint
driving this: domain corpora carry part numbers/SKUs/codes that no tagger
resolves as nouns, and spaCy is not called on the deterministic path.
Term-inventory lineage for the same reason: Frantzi & Ananiadou C-value/
NC-value; TopMine (arXiv:1406.6312); AutoPhrase (arXiv:1702.04457). Verified
identifiers: Hasegawa = aclanthology.org/P04-1053 (NOT P04-1052, which is a
different ACL-04 paper -- checked 2026-09-07); Church & Hanks = J90-1003;
Dunning = J93-1003; Blondel = arXiv:0803.0476; Mikolov = arXiv:1310.4546.
Full write-up with the bibliography:
"C:/Users/user/Documents/wiki/data science/llm/DIRT relation extraction.md".
