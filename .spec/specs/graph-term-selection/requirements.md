# Graph Term Selection — Requirements

**Spec layer:** Requirements. Structural and behavioral commitments are deferred
to `design.md`; nothing here prescribes a function signature, a matrix layout, or
a storage shape.

**Supersedes:** the first draft of this file (term-term similarity feeding df
pooling). That draft described the wrong structure — terms are graph nodes here,
not a similarity table consumed by eligibility.

## Context

Two term-selection pipelines exist in this repo and they do not meet.

`salient_grams.py` is a fully specified vocabulary selector (R1–R10): alnum-aware
tokenization that preserves digit-bearing identifiers, optional stemming, BM25 +
optional keyness utility, a df band, a box-cox length anomaly mask, cubic-spline
eligibility refinement, MMR diversity, and a cost-density waterline.

`chunkgraph.py` does not use it. The graph's term identification is
`chunkgraph.py:90`:

```python
def _tok(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2]
```

A ~150-word hardcoded stoplist, a length-3 floor, and a regex that **silently
discards every digit-bearing token before BM25 ever sees it**. R12 admits a
caller-supplied `vocab`, but it defaults to `None` and no driver in the repo
passes one, so the restriction path is dead code in practice.

### The structural problem

The sparse space connects chunks by **shared term identity**. BM25 cosine is
bag-of-words overlap, so relatedness between two *distinct* terms contributes
exactly zero — a "fire" chunk and a "heat" chunk connect only through other words
they happen to share. On the only persisted run this is the entire graph:
`embed_dim` NULL, 4568/4568 edges `sparse`.

The dense space exists to supply that missing relation and is currently off. This
spec builds the **deterministic, attributable** alternative. That is its
justification — not capability. Setting `model_dir` would deliver relatedness
today; it would not deliver an SME-checkable reason for each link.

### Two relations, not one

```
syntagmatic   fire~heat    co-occur together     "the fire gave off heat"
paradigmatic  fire~blaze   occupy the same slot  "the ___ spread"
```

They are separated by one dial — context width and ordering:

| context | relation surfaced | correct use |
|---|---|---|
| whole chunk, unordered | topical / associative | **edge** — connect, keep distinct |
| ordered ±1–2 words | substitutable | **collapse** — merge the node |

Collapsing on the wrong relation is a lossy, irreversible defect: merging
`fire`+`heat` means a query for `fire` can never again be distinguished from
`heat`. `doctor`/`hospital` correlate strongly and are not interchangeable.

### The sparse relation is already bipartite

`_sparse_sim` builds `X` (chunks × terms, BM25 values) and immediately collapses
it. But the projection is an identity over 2-hop paths:

```
(X @ X.T)[a,b] = SUM over terms t of X[a,t] * X[b,t]
                 = every path a --t--> b, weighted by the product of its edges
```

A direct sparse chunk-chunk edge is therefore **not independent evidence** about
`a` and `b`. It is a summary statistic over the term paths between them. Storing
both, in the same objective, counts one fact twice — inflating the sparse vote
against the dense one and breaking the dual-space premise that the two are
independent.

**This spec's central rule follows:**

> The sparse relation SHALL enter any single stage in exactly one representation.

| stage | sparse representation |
|---|---|
| storage / traversal / provenance | bipartite chunk↔term **only** |
| partitioning | projection, computed transiently, **never stored** |
| dense | direct chunk-chunk, unchanged |

The partitioner therefore never sees a bipartite graph. Its unipartite
configuration-model null stays valid, Barber's bipartite modularity is not
required, and no new dependency is introduced on that account. The
`computed-but-never-materialized` discipline is already house style (R10).

### The projection must be hub-corrected

The naive projection `X @ X.T` weights every 2-hop path equally, so a term with
df=182 contributes as much to each of its 16,471 chunk pairs as a term with df=3
contributes to its 3. That is the hub problem, and a df ceiling is a blunt
instrument against it.

`graph_analysis.compute_dwpc` in the `consolidation` skill solves it properly:
degree-weighted path count divides by the degree of the intermediate node. A
chunk→term→chunk hop is exactly a k=2 metapath with the term as intermediate.
[empirical:cited — Himmelstein & Baranzini hetnet DWPC; 0.4 damping exponent]

That toolkit's implementation is O(N³) explicit loops, documented as *"fine for
N≤200"*, against ~2,300 nodes here. The bipartite k=2 case has a closed form that
costs no more than the product already computed:

```
DWPC_2[a,b] = SUM over terms t of  X[a,t] * X[t,b] / deg(t)^0.4
            = X @ diag(deg^-0.4) @ X.T
```

One diagonal scaling inserted into the existing line. Note the toolkit's loop
computes `pc2 / (SUM of intermediate degrees)^0.4` while its docstring specifies
the *product* of intermediate degrees; for a single intermediate those differ, and
the per-path form above is the one matching the docstring and admitting the closed
form.

### Sizing, derived

The binding constraint on term-node count is what `_gist_walk` can traverse. With
`T` terms per chunk, `V` collapsed term nodes, `N = 1789` chunks [observed:
`graph_run.n_chunks` on the live `brown-50` run]:

```
mean df of a term node        d = N*T / V
2-hop frontier from one chunk   = T * d = N * T^2 / V
```

**Measured, before choosing a target.** Walking the live run over 41 seeds at
H=2 with the cookbook's own recursive CTE:

| measure | observed |
|---|---|
| mean 2-hop frontier | **7.7 nodes** |
| max 2-hop frontier | 30 |
| mean node degree | 5.11 |
| p50 / p95 degree | 4 / 12 |
| mean edge strength | 0.180 |

**The incumbent walk is score-limited, not structure-limited.** At mean strength
0.180, a 2-hop path scores 0.032 — below the cookbook's `score > 0.05` cutoff. Most
structurally available 2-hop paths are pruned by score before structure matters.

This corrects an error in the prior draft, which set a frontier target of ~500 as
"meaningful against `k_expand=6`". Against a measured frontier of 7.7 that is a
**65x expansion**, not a healthy operating point. The structural bound below is a
worst case the score cut never reaches today:

| T (terms/chunk) | V for structural frontier ≈ 500 | V for structural frontier ≈ 50 |
|---|---|---|
| 10 | ~360 | ~3,580 |
| 15 | ~810 | ~8,050 |
| 20 | ~1,430 | ~14,300 |

Holding the *structural* frontier near the measured 7.7 would demand more term
nodes than there are chunks — i.e. near-singleton terms, which defeats the purpose.
That is the honest tension: **the bipartite layer necessarily widens the structural
frontier, and the score cut is what must contain it.**

So the terms/chunk figure cannot be derived from structure alone. Roughly **10–15
terms/chunk over 400–800 term nodes** remains the working target — against 7,256
currently eligible — but it is a starting point for measurement, not a derived
bound. What R10 anticipated still holds: all 7,256 eligible terms cost only 228.1
chars/row, so the budget is inert and `max_terms` is the row that binds.

It also closes the df-ceiling question. At V≈500, mean df ≈ 44 (2.4% of N). The
10% ceiling fires only on genuine hubs; 50% would put 910 chunks behind one node
and destroy the walk. **10%, and for the walk's sake, not the vocabulary's.**

**Scope boundary.** IN: term identification feeding SPARSE, its eligibility gate,
term collapse, the bipartite sparse layer, relocated significance, and structural
community labels. OUT: query-side expansion (CLOSED already rejects the
WordNet form for observed drift; corpus-derived synsets share the failure mode and
need their own falsification); BPE or subword decomposition; any change to dense
similarity, fusion arithmetic, or the Louvain algorithm itself.

**Observed, out of scope, worth recording:** `self.strength` and `self.D` are
dense `n x n` numpy arrays built in `_edges_and_strength`, which sits in tension
with R10's no-materialized-`n x n` constraint at ~18k chunks. This spec does not
change that and does not fix it.

---

## Requirement 1 — Selected vocabulary is the default, and the tokenizers agree

**As an** operator, **I want** graph terms chosen by the measured selector rather
than a hardcoded stoplist, **so that** term identification is one specified
pipeline instead of two divergent ones.

1.1 The system SHALL select the SPARSE-stage vocabulary with
`salient_grams.select_vocab_unified` by default; the current `[a-z]+`/stoplist
behaviour SHALL be reachable only by explicit opt-out.

1.2 The system SHALL apply ONE tokenizer to both vocabulary selection and chunk
tokenization. WHERE the two disagree on any token, that is a defect, not a tuning
parameter.

1.3 The system SHALL preserve digit-bearing tokens through graph tokenization,
per `salient_grams` R1. The `[a-z]+` filter SHALL NOT survive this change.

1.4 WHEN phrase merging is enabled, vocabulary selection SHALL observe the same
merged token stream the restriction is later applied to, so no merged phrase is
eliminated by a restriction selected over its unmerged parts.

1.5 Vocabulary size SHALL be bounded by `max_terms`, and that bound SHALL be
derived from the walk's tolerance per the sizing model above — not inherited from
the chars/row budget, which R10 records as inert at this corpus size.

1.6 The df ceiling SHALL be 10% of N. Its governing job is hub control in the
bipartite layer (R5), not vocabulary compactness.

1.7 Stopword handling SHALL have exactly one owner. WHERE the selector's stopword
parameter governs, `_STOP` SHALL be removed rather than left as a dormant second
list.

1.8 IF selection would leave too few terms for stable distributions, THEN the
system SHALL raise rather than emit a degenerate graph — per `salient_grams` R7
and the existing `n >= 10` precondition.

1.9 The system SHALL record vocabulary size, binding constraint, and
`cost_if_all_eligible` in `self.diagnostics`, per R10.

### Acceptance

- A fit over brown-50 with defaults yields a non-empty vocabulary and a
  `binding_constraint` naming eligibility, cardinality, or weight.
- A corpus containing digit-bearing identifiers yields at least one such token in
  the fitted vocabulary. Under current code that count is zero.
- With phrases enabled, surviving `_`-bearing tokens after restriction is > 0.
- Round-trip: every selected term is producible by the graph's tokenizer over the
  corpus it was selected from. Zero orphans.
- Selected term count lands in 400–800 on brown-50, or the sizing model is
  reported as falsified with the observed number.

---

## Requirement 2 — Permissive dual-measure band eligibility

**As an** operator, **I want** an admission gate tuned for recall, **so that**
borderline domain terms do not flip out of the vocabulary between corpus draws.

2.1 The system SHALL offer a band-gated eligibility mode partitioning eligible
terms into df bands on a log2 scale — one edge per doubling of df.

2.2 WITHIN a band, the system SHALL admit a term whose utility clears EITHER the
band's robust centre less one scaled-MAD sigma OR its parametric centre less one
standard deviation. The disjunction SHALL NOT be reduced to either branch alone.

2.3 The robust branch SHALL scale MAD by 1.4826 so both branches express "centre
minus one sigma" on a common scale. An unscaled MAD makes them incomparable.

2.4 The band gate ADMITS; `spline_eligibility` DEMOTES. WHERE both are enabled,
the system SHALL state which acts last, and the composition SHALL be measured —
a recall gate followed by a precision gate can be a net no-op, and shipping both
without evidence of joint benefit is not permitted.

2.5 Banding SHALL be justified or abandoned on evidence. The claimed mechanism is
that BM25's IDF term mechanically suppresses high-df domain vocabulary, so a
global threshold discards what a within-band comparison recovers. IF that
suppression is not observed on this corpus, THEN the band gate SHALL NOT ship.

### Acceptance

- On brown-50, admitted-term count for: spline only, band only, both, neither.
- Across three disjoint samples, symmetric difference of the admitted set for
  spline-only vs band-only. The band gate's claim is lower churn; if it is not
  lower, the claim is falsified.
- A probe set of domain terms fixed BEFORE measurement, with recall reported per
  configuration. The probe set SHALL NOT be drawn from what the gate admits.

---

## Requirement 3 — Term association, measured on both axes

**As an** operator, **I want** term relatedness measured from the corpus, **so
that** the graph can express relations BM25 overlap cannot.

3.1 The system SHALL compute term-term association from co-occurrence counts, NOT
from BM25 weights. BM25 is a ranking weight whose `k1` saturation and `b` length
normalization would be conflated with association.

3.2 The system SHALL compute association at two settings from one counting pass:
chunk-wide unordered (**syntagmatic**) and ordered ±1–2 words (**paradigmatic**).

3.3 Association SHALL be significance-gated, not magnitude-gated. At `df=5` over
1821 chunks a correlation estimate is noise. The system SHALL use Dunning G² on
the 2×2 co-occurrence table — correct at low counts where χ² is not, and already
importable via the `BigramAssocMeasures` dependency `_merge_phrases` uses.

3.4 WHERE PMI-family measures are used, the system SHALL apply context
distribution smoothing or a count floor. Raw PMI is maximal for rare
perfectly-co-occurring events, which would make the graph's strongest edges its
least reliable ones.

3.5 The syntagmatic axis SHALL produce edges and SHALL NOT drive collapse. The
paradigmatic axis SHALL drive collapse (R4) and SHALL NOT be reported as topical
relatedness.

3.6 The stage SHALL be deterministic with no model-authored output, per the
steering ban.

3.7 WHERE the corpus is too small for counts to be estimable, the stage SHALL
degrade to a no-op with a recorded diagnostic, in the manner of R5 and R3. It
SHALL NOT emit low-confidence groupings silently.

### Acceptance

- Top-ranked pairs on brown-50 listed per axis and manually inspectable.
- The 3.5 separation is demonstrated: pairs ranked high on BOTH axes are named,
  since those are the ones a single-measure design would collapse wrongly.
- Stage runtime on brown-50 reported, inside the existing fit budget.

---

## Requirement 4 — Collapse: NOT ADOPTED, on measurement

**As an** operator, **I want** substitutable terms merged into one node, **so
that** a concept is not split across its expressions.

**Disposition: measured on brown-50 and rejected.** The paradigmatic axis does not
find synonymy. It finds co-hyponymy — semantic *class* membership — which is
precisely what must never be merged.

### The structural gap

Substitutability is not synonymy. Terms in the same semantic class occupy the same
slots and rarely co-occur, so they score maximally on context similarity AND pass
the low-co-occurrence conjunct. `hot`/`cold`, `monday`/`tuesday`, `four`/`six` are
indistinguishable from `fire`/`blaze` by construction of the step-4 gate.

### Measured (2026-08-27, live `brown-50`)

**Method.** Ordered ±2 context profiles over 846 targets at df≥15, 40,031
position-tagged context features, PPMI-weighted, cosine between profiles; signed
Dunning G² on the 2×2 chunk co-occurrence table as the second conjunct.

**One: the four-pair battery, by class mean context similarity.**

| class | pairs | mean ctx_sim |
|---|---|---|
| SYNONYM (should merge) | big/large, home/house, said/says | **0.022** |
| ANTONYM (must not) | good/bad, high/low, young/old | **0.039** |
| CO-HYPONYM (must not) | men/women, school/church, state/city | 0.030 |
| COLLOCATE (must not) | united/states, world/war, high/school | 0.019 |

**Antonyms outscore synonyms by ~1.8x.** The signal is not weak, it is *inverted*.
No threshold on this axis separates the class that should merge from the class
that must not.

**Two: the co-occurrence conjunct works, and cannot help here.** G² correctly
flags the collocates (united/states 318.3, men/women 49.4, world/war 26.8) but
antonyms do not co-occur (good/bad 2.1, high/low 3.4), so they sail through. The
conjunct is doing its job; antonymy is simply outside what it can see.

**Three: what the axis actually ranks highest.** Top pairs of 357,435, by context
similarity:

```
months/years  minutes/years  days/months  days/years  hours/minutes  afternoon/morning
four/six      hundred/million            few/two
tried/trying  want/wanted
stood/walked  know/think     tell/want
```

Time units, numbers, verb classes — **co-hyponyms**, uniformly. The only genuinely
mergeable class present is inflectional (`tried`/`trying`, `want`/`wanted`), and
`salient_grams` already merges those with snowball stemming at a measured +4pts
recall and ~18% of slots freed.

**So the axis returns nothing safely usable that the incumbent does not already
provide.**

**Four: there is no separable population to threshold.** Across all 357,435 pairs
the maximum context similarity is 0.146, p99 is 0.044, median 0.0091. A thin tail,
not a mode.

### Requirements

4.1 Distributional collapse SHALL NOT ship. The measurement above is the evidence;
it is not a tuning problem, because the discriminating signal points the wrong way.

4.2 No further conjunct SHALL be added to the step-4 gate on this corpus. A third
condition can only help if some measurable signal ranks synonyms above
co-hyponyms; none was found, and adding conjuncts against an inverted signal is
fitting noise.

4.3 **Stemming remains the collapse mechanism**, and keeps its full job. It is the
incumbent, it is measured, and it covers the one class the distributional axis
found that is safe to merge.

4.4 IF collapse is revisited, THEN it SHALL be on a corpus where the battery
above shows synonyms outscoring antonyms — that is the entry condition, and it is
cheap to re-run. Corpus size is a plausible cause (1,789 chunks, df≥15 floor) but
is unproven; the inversion may be intrinsic to distributional context.

4.5 The syntagmatic axis (R3.5) is unaffected. It feeds edges, not merges, and its
disposition is independent of this one.

## Requirement 5 — The bipartite sparse layer

**As an** operator, **I want** term nodes in the graph, **so that** a chunk
connection can name the term that produced it.

5.1 Term nodes SHALL be first-class graph nodes with chunk↔term edges.

5.2 Chunk↔term edge weight SHALL be BM25 — the value `_sparse_sim` already
computes. `X` IS the bipartite adjacency matrix; this requirement persists the
intermediate the pipeline currently discards rather than introducing a new one.

5.3 BM25 SHALL be used rather than PPMI for this edge because chunk length varies
by design (`_chunk` targets 120 words, permits 200, with overlapping windows).
PPMI has no length correction and would make long chunks hubs on a chunking
artifact; BM25's `b=0.75` is that correction, and its bounded idf avoids PMI's
rare-term blowup.

5.4 **Sparse chunk-chunk edges SHALL NOT be persisted.** The bipartite layer is
the stored sparse relation.

5.5 The projection MAY be computed transiently as partitioner input and SHALL NOT
be materialized as edges, entered into the walk, or persisted. No stage SHALL see
both representations.

5.5.1 The projection SHALL be hub-corrected by degree-weighting the intermediate
term node — `X @ diag(deg^-0.4) @ X.T`, the closed form of k=2 DWPC over the
bipartite graph. The naive `X @ X.T` SHALL NOT be used, because it weights a
df=182 term's paths equally with a df=3 term's.

5.5.2 The O(N³) loop implementation in `consolidation/graph_analysis.compute_dwpc`
SHALL NOT be used at this scale; it is documented as fit for N≤200 against ~2,300
nodes here. The closed form is required, not an optimization.

5.5.3 The damping exponent SHALL be treated as a parameter and its default (0.4)
recorded as inherited from hetnet convention, not as a value measured on this
corpus.

5.5.4 **A term hop SHALL NOT consume a scoring hop.** A term-mediated connection
SHALL be scored as ONE edge carrying the DWPC value, never as two traversals whose
strengths multiply.

Without this rule the bipartite layer silently deletes the sparse space. The walk
multiplies edge strengths and cuts at 0.05; mean live strength is 0.180, so:

```
1 hop:  0.180                  passes
2 hops: 0.180 x 0.180 = 0.032  FAILS
```

Every chunk→term→chunk connection is structurally 2 hops. Moving sparse edges into
the bipartite layer therefore drops the average sparse connection below the floor.
The measured 2-hop frontier is already only 7.7 nodes — the graph is barely walked
past hop 1 as it stands.

Term nodes remain traversable for attribution and explanation; they are excluded
from the hop budget that scoring uses. This keeps the projected score scale
comparable to today's direct edges, which is what stops the frontier collapsing.

5.6 Dense chunk-chunk edges SHALL remain direct and unchanged. Embedding
similarity is not a function of shared terms, so it is independent evidence — the
premise the provenance design rests on.

5.7 Node kind (chunk vs term) is a filter key and SHALL be a real column. Hard
constraint 3 forbids it in `attrs`.

5.8 Edge kind SHALL be its own column and SHALL NOT be a value added to the
`edge_provenance` enum. Measured against the live `brown-50` run:

- **`edge_sym` carries no kind predicate.** Its definition is a bare
  `UNION ALL` over `edge` in both directions. Cookbook #1, #2 and #4 all traverse
  it unfiltered, so admitting chunk↔term rows makes every existing walk return term
  nodes as results — a wrong answer, not an error.
- **`provenance` is an enum type and enum values cannot be removed.** Adding
  `member` is a one-way door, and `ALTER TYPE ... ADD VALUE` cannot be used in the
  same transaction that inserts it, which conflicts with `pg_store`'s single atomic
  commit.
- **The two concepts are not the same question.** Provenance answers *which
  similarity space voted*; kind answers *what relation this edge is*. Cookbook #8
  (`GROUP BY provenance` to audit the blend) becomes meaningless once a non-vote
  value enters the column.
- **Provenance currently carries zero information** — 4,568 of 4,568 live edges are
  `sparse` and `sim_dense` is NULL on every row. So the overload would go unnoticed
  today and surface only when embeddings are enabled. That is an argument against
  it, not for it.

5.11 The `edge_canonical CHECK (src < dst)` constraint SHALL be reconciled. It
imposes a canonical ordering that carries no information about which endpoint is
which; on a bipartite edge the term would sometimes be `src` and sometimes `dst`
depending on `ord` allocation, so no query could tell the endpoints apart without a
join. Either bipartite edges get their own relation, or kind plus an explicit
ordering convention replaces the bare inequality.

5.12 The `node` table's NOT NULL columns SHALL be reconciled. `doc_id`,
`chunk_hash` and `body` are all NOT NULL, and a term node has none of them.
Sentinel values SHALL NOT be used to satisfy them — a sentinel `doc_id` would
corrupt cookbook #6, which finds cross-document bridges by `src_doc <> dst_doc`.

5.9 Hub control has two layers. DWPC degree-weighting (5.5.1) is the primary
mechanism and is graded — a hub's contribution decays with its degree. The df
ceiling (1.6) is the backstop for the traversal path, where DWPC does not apply:
a term node with df=182 still dumps 182 chunks into a frontier sized for
`k_expand=6` regardless of how the projection weights it.

5.10 R7's backbone SHALL cover the bipartite layer. At 10–15 terms/chunk some
chunks lose all salient terms and go isolate; "significance ranks, backbone
connects" applies here or the 23%-isolate failure is rebuilt in a new place.

### Acceptance

- Stored edge count reported by kind. Bipartite postings on brown-50 are expected
  near 36,000 against the current 4,568 chunk-chunk — 8x larger than what is
  stored today, 45x smaller than the 1.6M complete relation it encodes. This is
  compression against the complete relation, not against the thresholded one.
- Zero persisted sparse chunk-chunk edges after the change.
- Isolate count in the bipartite layer, before and after backbone union.
- Max term-node degree is at or under the ceiling.

---

## Requirement 6 — Significance, relocated

**As an** operator, **I want** the significance discipline preserved, **so that**
dropping stored chunk-chunk edges does not silently drop the tail cut with them.

6.1 The k-sigma cut SHALL be relocated, not removed. It moves from "is this chunk
PAIR similar enough" to "is this chunk-TERM link strong enough."

6.2 The relocated cut SHALL retain R2's shape: Box-Cox normalize, k-sigma tail
cut, budget-match fallback when kurtosis is high. The terms-per-chunk bound (1.5)
is that fallback.

6.3 The Box-Cox lambda SHALL be refitted against the chunk-term BM25 distribution
and SHALL NOT be assumed to carry over from the chunk-pair similarity
distribution. Different distribution, different skew.

6.4 R6's estimator-pair divergence warning SHALL apply to the new distribution.

6.5 The cut SHALL carry a multiple-testing correction. The incumbent `_cut` applies
a fixed `k_sigma=2.0` across ~1.66M candidate pairs with no family-wise control, so
`k=2` is a magic constant rather than a derived threshold.
`graph_analysis.significance_tau` already solves this: model the null as the
*empirical background distribution* of pairwise similarity, test one-tailed against
it, and apply Bonferroni or Benjamini-Hochberg across all `N(N-1)/2` pairs.

6.6 The empirical null SHALL be used rather than a zero-centred one. A dense corpus
has a non-zero background similarity floor, so "is this pair correlated at all" is
the wrong question and "is this pair more similar than typical" is the right one.
This is the same reasoning the incumbent Box-Cox step already encodes; 6.5 adds the
error control it lacks.

6.7 Bonferroni vs BH SHALL be a recorded choice with its edge-count consequence,
not a silent default. Bonferroni is conservative and will deepen the isolate
problem R7/5.10 exist to manage.

### Acceptance

- Fitted lambda, kurtosis, and whether the tail cut or the budget fallback bound,
  reported for the chunk-term distribution on brown-50.
- Terms-per-chunk resulting from the derived cut, against the 10–15 model.
- Edge counts under: fixed k-sigma (incumbent), Bonferroni, and BH. Three numbers.
- Isolate count under each, since the correction's cost is paid in isolates.

---

## Requirement 7 — Community labels from structure

**As an** operator, **I want** community labels to be graph structure, **so that**
SME effort survives re-ingest.

7.1 Each term node SHALL be assigned to the community holding the majority of its
BM25 edge weight, deterministically.

7.2 Those assigned terms SHALL be the community's label, replacing post-hoc
tf\*idf `keywords`.

7.2.1 **The incumbent baseline is strong, and the acceptance bar is set against it,
not against a strawman.** Observed on the brown-10 render, tf\*idf keywords produce
`congo / belgians / independence / lumumba`, `comedie / moliere / seigner /
tartuffe`, `catholic / england / churches / church`. Structural labels SHALL be
shown to beat that quality, not merely to exist. IF they do not, R7 ships only its
stability claim (7.4 acceptance) and keeps tf\*idf for label text.

7.3 Term nodes SHALL NOT participate in the partition itself — 5.5 keeps the
partitioner unipartite so its null model stays valid. Labelling is assignment after
partitioning, not membership during it.

7.5 The partitioner SHALL remain Louvain unless the defect that motivates
replacing it is observed. Leiden's guarantee is that communities are internally
connected, which Louvain does not ensure. [empirical:cited — Traag, Waltman & van
Eck 2019, *From Louvain to Leiden*]

**Measured: the defect is absent here.** All 40 Louvain communities on brown-50
have `lcc_ratio = 1.0` — zero internally disconnected communities. Leiden would
therefore buy a dependency and change every `cid` in exchange for a guarantee
about something already true. It SHALL NOT ship on this evidence.

7.5.1 IF a future run shows `lcc_ratio < 1.0` on any community, THEN Leiden is
justified and 7.5 is superseded. `lcc_ratio` (8.6) is the trigger, and it is
already being measured, so this needs no new instrumentation.

7.6 WHERE Leiden requires a dependency the environment lacks, the system SHALL fall
back to Louvain and record the substitution, in the manner of R5/R9's graceful
degradation. `graph_analysis.louvain_partition` already models this pattern by
falling back to networkx greedy modularity.

7.4 The determinism boundary is unchanged: membership computed, labels corpus-
derived, model-authored names remain drafts and never join keys.

### Acceptance

- Structural labels compared side by side against current tf\*idf `keywords` on
  brown-50.
- Label stability across two ingests of the same corpus, reported as overlap, for
  both methods. The claim is that collapsed terms are more stable than re-cut
  chunk ids; if overlap is not higher, the claim is falsified.

---

## Requirement 8 — The graph must not regress

**As an** operator, **I want** proof this improved retrieval, **so that** a
better-specified vocabulary is not mistaken for a better graph.

8.1 Changing the vocabulary changes the BM25 space, which changes edge formation.
The system SHALL report the downstream effect, not only the vocabulary diff.

8.2 Reported before and after on the same corpus: community homophily, isolate
count, component count, edge count by kind and provenance. Steering's baseline:
homophily .95, and the significance-only configuration's 23% isolates / 655
components at 1821 nodes as the known failure shape.

8.6 Per-community quality SHALL be reported using the established subgraph
measures, not homophily alone — **density**, **conductance**, and **LCC ratio**,
as computed by `graph_analysis.subgraph_metrics`. Homophily says a community is
internally similar; it does not say the community is well-cut (conductance) or
even connected (LCC ratio). Given this repo's 23%-isolate history, LCC ratio is
the measure most likely to catch a regression the others miss.

8.7 A community whose LCC ratio is below 1.0 is internally disconnected and SHALL
be reported as a defect, not averaged away into a corpus-level score.

8.3 IF homophily falls or isolate count rises, THEN the change SHALL NOT ship on
vocabulary metrics alone.

8.4 Every stage introduced here SHALL be individually disableable, so a regression
can be attributed to one stage rather than to the set.

8.5 A walk-level check SHALL accompany the graph-level one: 2-hop frontier size
from a fixed seed set, before and after. The measured baseline is **mean 7.7, max
30 over 41 seeds** on the live run. Any target SHALL be stated as a multiple of
that baseline rather than as an absolute, and a frontier orders larger means hub
control failed.

8.8 Provenance and source-document lineage SHALL survive the walk. This is
**demonstrated on the live run**, not aspirational: carrying `s.provenance` and
`n.doc_id` as arrays through the cookbook's recursive CTE returns a full
`prov_path` and `doc_path` per result at no additional join, because `edge_sym`
already exposes provenance and both endpoint docs. Any change to edge storage
SHALL preserve that property, and the acceptance test SHALL be the array-carrying
walk, not merely the presence of a provenance column.

8.10 The renderer SHALL be able to display the partition it is asked to verify.
This brings `draw()` into scope as the visual instrument for 8.2, and only for that
— no other DRAW behaviour is in scope.

Measured justification: brown-50 is **76.4% intra-community** (3,488 of 4,568 live
edges, 40 communities covering 1,787 of 1,789 chunks). A partition that clean must
be visible, and under the incumbent renderer it is not.

8.11 Community layout SHALL take the partition as **input**, not hope a global
force simulation rediscovers it. Two passes: spring the community meta-graph for
blob centres, spring each community's induced subgraph for member placement, then
scale centres apart until no two blobs overlap. Approach per
`graph_analysis.community_aware_layout`, adapted to place related communities near
each other rather than on a fixed circle.

8.11.1 **The cause is the single-pass architecture, not a parameter value.**
Measured on a planted-partition fixture matched to the live graph's density (~3.4
mean degree, ~80% intra), scoring nodes by whether they sit nearest their own
community's centroid:

| layout | misplaced |
|---|---|
| single-pass spring, `k=0.9` (incumbent) | 39% |
| single-pass spring, networkx default `k` | **58%** |
| two-pass community layout | **0%** |

Lowering `k` toward the networkx default makes separation *worse*. An earlier draft
of this requirement attributed the failure to `k=0.9` being 19x the default; that
is falsified. At mean degree ~3–4 a node has too few edges to be pulled decisively
toward its community, so global repulsion dominates at any `k`. Tuning is not a fix
available here.

8.11.2 The layout test fixture SHALL match the measured graph's sparsity. A fixture
of dense cliques is separable by any layout and would let the suite pass against the
very renderer it replaced; a guard test SHALL assert the fixture defeats single-pass
spring.

8.9 Cross-document reach SHALL be reported. Measured baseline: the graph is
**32.1% cross-document** (1,466 of 4,568 live edges), and the 2-hop frontier is
**31.9% cross-document** — the walk is doc-neutral, neither favouring nor avoiding
cross-document paths. A change that drops frontier cross-doc share materially below
the graph's base rate has made the walk parochial, which defeats the synthesis use
case cookbook #6 exists to serve.

### Acceptance

- One before/after table over brown-50 covering 8.2.
- Each new stage toggled off independently reproduces the prior numbers.
- Frontier sizes for a fixed anchor set, before and after.

---

## Requirement 9 — Not every community is worth traversing

**As an** operator, **I want** communities gated on measured quality, **so that** a
partition artifact is not treated as a retrievable region.

9.1 Communities SHALL be gated for traversal and retrieval by cohesion, not
cardinality. Three findings against the incumbent `communities(min_size=5)`:

- **It is inert.** The smallest live community is 8 members, so the threshold
  rejects nothing observable on this run.
- **It filters the wrong direction.** Conductance correlates with SIZE — the tight
  communities are the small ones (sizes 15–45 at conductance 0.02–0.08) and the
  loose ones are the large ones (56–108 at 0.29–0.36). A cardinality floor removes
  precisely the tightest candidates.
- **It destroys its own evidence.** `communities()` drops sub-threshold groups
  BEFORE persisting, so what it removed cannot be recovered or audited from the
  store. Whatever replaces it SHALL record rather than discard (9.4).

9.2 The gate has measured precedent: in the `parallel-rrf-graph-rag-...` skill a
density/conductance-gated community lane delivered **+25% relative recall and
precision on thematic queries**. That is evidence the gate matters, not evidence
for specific thresholds.

9.2.1 **The threshold cannot be read off a valley, because there is none.**
Conductance across the 40 live communities is a smooth continuum:

```
0.02 0.03 0.03 0.04 0.05 0.05 0.06 0.07 0.07 0.08 0.11 0.11 0.11 0.12 0.13
0.15 0.16 0.19 0.20 0.22 0.22 0.23 0.23 0.25 0.25 0.25 0.26 0.29 0.29 0.29
0.30 0.31 0.32 0.33 0.34 0.34 0.36 0.36 0.38 0.40
median 0.219   mean 0.200   stdev 0.116   largest gap 0.033
```

A largest gap of 0.033 against a standard deviation of 0.116 is noise. Any gate
SHALL therefore be a **relative bound** — a percentile of this run's own
distribution — and SHALL NOT be a fixed constant presented as discovered.

9.2.2 Conductance is only interpretable against this network's own baseline, per
`subgraph-cohesion-metrics`. Whole-graph density is 0.00286; community density
runs 0.032–0.464. A reference PPI network's Louvain clusters sat at conductance
0.2–0.7 peaking near 0.6; brown-50's median of 0.219 is at the low end of that
band, which is normal-to-good, **not** pathological. The gate exists to rank, not
to condemn.

9.2.3 **The domain control SHALL be run and reported.** `doc_id` is a
domain-derived partition already present, so the corpus's prescribed comparison —
algorithmic partition versus domain partition — costs nothing:

| partition | n | conductance (min/med/max) | density (med) | lcc (min/med) |
|---|---|---|---|---|
| Louvain | 40 | 0.021 / **0.219** / 0.404 | 0.090 | **1.000** / 1.000 |
| `doc_id` | 50 | 0.025 / **0.265** / 0.688 | 0.107 | 0.175 / 0.909 |

Read honestly: Louvain is **barely more separated than the documents themselves**
(0.219 vs 0.265) while being far more internally coherent (lcc 1.000 vs 0.909,
with one document as low as 0.175). **The community layer is largely recovering
document boundaries.** Since `doc_id` is already a stored column, that bounds what
the community layer currently adds — and it is the strongest argument for the term
layer, whose whole purpose is to connect across documents rather than within them.

9.2.4 A regression check SHALL use this control: IF communities become *more*
document-aligned after the term layer lands, the term layer has failed at the one
thing it exists to do.

9.3 Singletons and isolates SHALL have an explicit disposition rather than being
dropped. `graph_analysis.compute_singleton_affinities` assigns each singleton to
its nearest multi-member community by average similarity. This repo has the same
failure in a more severe form (23% isolates, 655 components at 1821 nodes), and R7
backbone union is the current answer; 9.3 asks whether affinity assignment is a
better one for the *community* layer specifically, where backbone union is not
available.

9.4 Gating SHALL be reported, never silent. A community excluded from traversal
SHALL still be persisted and still be inspectable — steering's supersede-never-
delete posture applies to judgements about communities, not only to runs.

9.5 The determinism boundary holds. Gates are computed from graph structure. The
`consolidation` skill's own `graph_disposition.txt` is the cautionary case: an
LLM observer emitted self-contradicting verdicts across communities (the identical
sentence *"the member names do not represent genuinely overlapping skill domains,
as they overlap significantly with each other"* appears as justification for both
KEEP and INSPECT). That is direct evidence for steering's rule that models may
name a frozen cluster but never judge membership.

### Acceptance

- Density, conductance, and LCC ratio reported per community on brown-50, with the
  gate's admit/reject decision beside each.
- Count of communities admitted vs rejected, and the retrieval effect of excluding
  the rejected ones.
- Singleton/isolate count before and after whichever disposition 9.3 selects.

---

## Open questions

- **[NEEDS CLARIFICATION]** R2.4 ordering: band-then-spline or spline-then-band.
  Not resolvable from either module's evidence; needs R2's measurement first.
*(R5.8 was the second open question. Closed by measurement — see Resolved below.)*

## Resolved during drafting

- **df ceiling** — 10% of N, set by walk tolerance (Context, sizing). Was open in
  the prior draft.
- **Chunk-chunk correlation** — closed. Pearson is cosine on mean-centered rows;
  at ~50 nonzeros over 7,256 terms the centering shifts similarity by ~0.0068,
  absorbed by the tail cut, while densifying the CSR from ~50 to 7,256 nonzeros
  per row. Full price of the R10 constraint for a numerically inert change.
- **BM25 vs PPMI for chunk↔term** — BM25 (5.3). PPMI keeps the term-term
  association job (3.1), which is a different edge.
- **Bipartite modularity** — does not arise. 5.5 keeps the partitioner unipartite.
- **Hub control** — DWPC degree-weighting is primary (5.5.1), df ceiling is the
  traversal backstop (5.9). Previously the ceiling carried the whole job.
- **Distributional collapse** — NOT adopted (R4). Measured: antonyms outscore
  synonyms 0.039 vs 0.022 on context similarity; the top-ranked pairs corpus-wide
  are time units and numerals. The axis finds co-hyponymy, not synonymy.
- **Leiden** — NOT adopted (7.5). All 40 communities measure `lcc_ratio` 1.000, so
  the defect it guards against is absent. Revisit if that ever drops below 1.
- **Why WCC *and* a modularity partitioner** — they answer different questions.
  WCC asks *is this fragmented* (answer: no, 1 component) and is the cheap
  post-backbone assertion. Modularity asks *where is the structure inside that one
  component* (answer: 40 communities). Neither substitutes for the other.

## Corrections made during drafting

Two numbers reported earlier in this spec's development were wrong and are
superseded by the table above:

- **Conductance was understated.** An earlier SQL formulation joined community
  membership with `e.src = m.ord OR e.dst = m.ord`, double-counting internal edges
  and halving the conductance numerator. It reported 0.011–0.253 (median 0.123);
  the correct values are 0.021–0.404 (median 0.219).
- **The distribution is not bimodal.** The apparent "3x gap separating tight from
  loose communities" was an artifact of that same bug. Corrected, the largest gap
  is 0.033 against a standard deviation of 0.116 — a continuum with no valley, so
  no threshold is discoverable from its shape (9.2.1).
- **Edge kind: own column, not a provenance value** (5.8). Closed by querying the
  live run: `edge_sym` has no kind predicate, the enum is append-only and conflicts
  with `pg_store`'s atomic commit, and provenance answers a different question.
- **Does the walk reach provenance?** Yes, demonstrated (8.8). Provenance and doc
  lineage carry through the recursive CTE as arrays at no extra join cost.

## Measurements taken against the live run

All from `brown-50` (`n_chunks` 1789, `embed_dim` NULL, single live run), 2026-08-27.

| measure | value |
|---|---|
| live edges | 4,568, all `provenance='sparse'`, `sim_dense` NULL on every row |
| mean / max node degree | 5.11 / 28 (p50 4, p95 12) |
| mean / max 2-hop frontier (41 seeds) | 7.7 / 30 |
| mean edge strength | 0.180 |
| cross-document edges | 1,466 / 4,568 = 32.1% |
| cross-document share of frontier | 31.9% |
| WCC components | **1** — all 1,789 nodes, zero singletons |
| Louvain communities | 40, sizes 8–108, all `lcc_ratio` = 1.000 |
| Louvain conductance | 0.021 / 0.219 / 0.404 (min/median/max) |
| `doc_id` conductance (control) | 0.025 / 0.265 / 0.688 |
| `doc_id` lcc (control) | 0.175 / 0.909 (min/median) |
| whole-graph density | 0.00286 |

**WCC returns one component covering every node.** Read two ways, both useful:
the graph is not fragmented, so R7's backbone union is doing its job — and
steering's CLOSED note (*23% isolates, 655 components*) describes the
**pre-backbone** significance-only configuration, not the shipped one. WCC is
therefore a cheap assertion here rather than an analysis; it answers one question
and that question's answer is "no problem".

Two of these overturn statements in the prior draft: `n_chunks` is 1789 (not the
1821 carried over from steering's CLOSED note), and the ~500 frontier target was
65x the measured value. Both corrected in place.

Provenance carrying zero information on the only persisted run means the
`edge_prov` index and cookbook #8 are currently **inert**. Worth stating plainly:
this spec's changes cannot regress a provenance signal that does not yet exist, and
equally, provenance-based acceptance criteria cannot be validated until the dense
space is enabled.

## Incumbents reused

Per anti-sprawl Gate A, these exist and are extended rather than rebuilt. All live
in `C:\Users\user\.skills\consolidation\graph_analysis.py`:

| Function | Used for | Caveat |
|---|---|---|
| `compute_dwpc` | hub-corrected projection (5.5.1) | O(N³) loops, N≤200; use the closed form |
| `significance_tau` | empirical-null cut + FDR control (6.5–6.7) | written for cosine; refit for BM25 |
| `subgraph_metrics` | density / conductance / LCC (8.6) | direct reuse |
| `compute_singleton_affinities` | isolate disposition (9.3) | direct reuse |
| `louvain_partition` | fallback pattern (7.6) | pattern, not the code |
| `community_aware_layout` | partition-visible render (8.11) | approach, adapted to int node ids |

Not adopted, and why: `kmeans_elbow` (a different clustering paradigm with no role
here), `render_spring_layout` (`draw()` already owns the drawing; only the layout
approach is borrowed).

`community_aware_layout` was declined in an earlier draft on the grounds that
visualization is out of scope. The brown-10 render falsified that: the renderer is
the instrument R8 depends on, and it could not show a 76%-intra partition.


## Amendment 2026-08-31 — the collapse intent, restated by the operator

The rejected collapse pass (5.x measurements) tested SUBSTITUTABILITY via
position-tagged context profiles. The operator's intent was narrower and is
recorded here so it is not lost: **collapse INDIVIDUAL WORDS (not phrases)
into synset-like nodes** using the same machinery already trusted for chunks
-- a correlation matrix, this time over word columns of the doc-term matrix
(second-order co-occurrence: two words are similar when they keep the same
company, "lh"/"left handed" style).

Proposed mechanics, unmeasured, OPEN:
1. Columns of the BM25 doc-term matrix -> word-word correlation (cosine).
2. Merge threshold found from the DISTRIBUTION, not hand-set: a correlation
   knee, and/or pairs flagged anomalous against the pair population
   (isolation forest or density clustering over the correlation values).
3. Merged words form one term node carrying its member surface forms --
   mimicking a WordNet synset without WordNet.

Standing guard that still applies: association is not substitutability -- the
antonym confound was MEASURED (synonyms did not outscore antonyms on context
similarity). Any merge pass must therefore pass the dual-measure band gate
(5.x) or an equivalent discriminating check before two words become one node;
the knee/anomaly detector chooses CANDIDATES, it does not certify them.

Status: intent recorded; implementation OPEN, gated on the same measurement
battery that rejected the first attempt (4+ varied pairs incl. an antonym
pair that must NOT merge).
