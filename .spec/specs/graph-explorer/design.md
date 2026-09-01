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
